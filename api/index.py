from flask import Flask, request, jsonify
import numpy as np
import tempfile
import os
import urllib.request
import subprocess
import json as jsonlib
import re
import struct

app = Flask(__name__)

FRAME_WIDTH = 80
FRAME_HEIGHT = 20
FFMPEG_PATH = "/tmp/ffmpeg"


def ensure_ffmpeg():
    """Download a static ffmpeg binary if not present."""
    if os.path.exists(FFMPEG_PATH) and os.access(FFMPEG_PATH, os.X_OK):
        return FFMPEG_PATH

    url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
    tar_path = "/tmp/ffmpeg.tar.xz"

    # Download
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0"
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        with open(tar_path, "wb") as f:
            f.write(resp.read())

    # Extract just the ffmpeg binary
    subprocess.run(
        ["tar", "-xf", tar_path, "--wildcards", "*/ffmpeg", "--strip-components=1", "-C", "/tmp"],
        check=True, timeout=30
    )
    os.chmod(FFMPEG_PATH, 0o755)

    # Cleanup
    if os.path.exists(tar_path):
        os.unlink(tar_path)

    return FFMPEG_PATH


def download_video(url):
    ext = ".mp4"
    for e in [".webm", ".mp4", ".avi", ".mkv", ".mov"]:
        if e in url.lower():
            ext = e
            break

    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })
    with urllib.request.urlopen(req, timeout=30) as response:
        tmp.write(response.read())
        tmp.close()

    file_size = os.path.getsize(tmp.name)
    if file_size < 1000:
        with open(tmp.name, "r", errors="ignore") as f:
            content = f.read(500)
        if "<html" in content.lower() or "<!doctype" in content.lower():
            os.unlink(tmp.name)
            raise ValueError(f"URL returned HTML instead of video ({file_size} bytes)")

    return tmp.name


def get_video_info(ffmpeg, path):
    """Get video info using ffprobe (bundled with ffmpeg)."""
    ffprobe = ffmpeg.replace("ffmpeg", "ffprobe")
    if not os.path.exists(ffprobe):
        ffprobe = ffmpeg  # fallback

    try:
        r = subprocess.run(
            [ffprobe, "-v", "quiet", "-print_format", "json",
             "-show_streams", "-show_format", path],
            capture_output=True, text=True, timeout=10
        )
        info = jsonlib.loads(r.stdout)
        for s in info.get("streams", []):
            if s.get("codec_type") == "video":
                fps_parts = s.get("r_frame_rate", "30/1").split("/")
                fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30.0
                return {"fps": fps}
    except Exception:
        pass
    return {"fps": 30}


def extract_frames(ffmpeg, path, target_fps, max_frames):
    """Extract frames as raw RGB using ffmpeg."""
    cmd = [
        ffmpeg, "-i", path,
        "-vf", f"fps={target_fps},scale={FRAME_WIDTH}:{FRAME_HEIGHT}",
        "-pix_fmt", "rgb24",
        "-f", "rawvideo",
        "-frames:v", str(max_frames),
        "-v", "quiet",
        "-y",
        "pipe:1"
    ]

    r = subprocess.run(cmd, capture_output=True, timeout=60)
    if r.returncode != 0 or not r.stdout:
        return []

    raw = r.stdout
    frame_size = FRAME_WIDTH * FRAME_HEIGHT * 3
    count = len(raw) // frame_size

    frames = []
    for i in range(count):
        offset = i * frame_size
        frame = np.frombuffer(raw[offset:offset + frame_size], dtype=np.uint8)
        frame = frame.reshape(FRAME_HEIGHT, FRAME_WIDTH, 3)
        frames.append(frame)

    return frames


def rgb_to_hex(r, g, b):
    return f"#{r:02X}{g:02X}{b:02X}"


def is_bw(frame, threshold=15.0):
    diffs = np.max(frame.astype(int), axis=2) - np.min(frame.astype(int), axis=2)
    return float(np.mean(diffs)) < threshold


def frame_to_ascii(frame, bw_only=False):
    lines = []
    for y in range(FRAME_HEIGHT):
        line = ""
        for x in range(FRAME_WIDTH):
            r, g, b = int(frame[y, x, 0]), int(frame[y, x, 1]), int(frame[y, x, 2])
            if bw_only:
                gray = int(0.299 * r + 0.587 * g + 0.114 * b)
                line += "<#FFFFFF>\u23f9" if gray > 127 else "<#000000>\u23f9"
            else:
                line += f"<{rgb_to_hex(r, g, b)}>\u23f9"
        lines.append(line)
    return "\n".join(lines)


def parse_request():
    raw = request.get_data(as_text=True).strip()
    data = None

    try:
        data = jsonlib.loads(raw)
    except Exception:
        pass

    if not data and raw.startswith("{"):
        try:
            url_match = re.search(r'"url"\s*:\s*"(.*?)",\s*"fps"', raw)
            fps_match = re.search(r'"fps"\s*:\s*(\d+)', raw)
            max_match = re.search(r'"max_frames"\s*:\s*(\d+)', raw)
            if url_match:
                data = {
                    "url": url_match.group(1),
                    "fps": int(fps_match.group(1)) if fps_match else 10,
                    "max_frames": int(max_match.group(1)) if max_match else 100
                }
        except Exception:
            pass

    if not data and raw.startswith("http"):
        data = {
            "url": raw,
            "fps": int(request.args.get("fps", 10)),
            "max_frames": int(request.args.get("max_frames", 100))
        }

    if not data and request.args.get("url"):
        data = dict(request.args)

    return data


@app.route("/", methods=["GET"])
def home():
    if request.args.get("url"):
        return convert_video(dict(request.args))
    return jsonify({"status": "ok", "usage": "POST with {url, fps, max_frames}"})


@app.route("/", methods=["POST"])
def convert():
    data = parse_request()
    if not data or not isinstance(data, dict) or "url" not in data:
        return jsonify({"error": "Could not parse request"}), 400
    return convert_video(data)


def convert_video(data):
    video_url = data["url"]
    target_fps = int(data.get("fps", 10))
    max_frames = int(data.get("max_frames", 100))

    tmp_path = None
    try:
        # Ensure ffmpeg is available
        ffmpeg = ensure_ffmpeg()

        # Download video
        tmp_path = download_video(video_url)
        file_size = os.path.getsize(tmp_path)

        # Extract frames
        frames = extract_frames(ffmpeg, tmp_path, target_fps, max_frames)

        if not frames:
            info = get_video_info(ffmpeg, tmp_path)
            return jsonify({
                "error": "No frames extracted",
                "file_size": file_size,
                "video_info": info
            }), 400

        # Check BW
        sample = frames[::max(1, len(frames) // 10)]
        bw_only = sum(1 for f in sample if is_bw(f)) > len(sample) * 0.8

        # Convert to ASCII
        ascii_frames = [frame_to_ascii(f, bw_only=bw_only) for f in frames]

        return jsonify({
            "frames": ascii_frames,
            "frame_count": len(ascii_frames),
            "dimensions": f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
            "black_and_white": bw_only,
            "fps": target_fps
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
