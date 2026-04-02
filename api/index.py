from flask import Flask, request, jsonify
import numpy as np
import tempfile
import os
import urllib.request
import subprocess
import json as jsonlib
import re

app = Flask(__name__)

FRAME_WIDTH = 80
FRAME_HEIGHT = 20


def find_ffmpeg():
    """Find ffmpeg binary - try multiple methods."""
    # Method 1: Try imageio-ffmpeg (has bundled binary)
    try:
        import imageio_ffmpeg
        path = imageio_ffmpeg.get_ffmpeg_exe()
        if os.path.exists(path):
            return path
    except Exception:
        pass

    # Method 2: Check if ffmpeg is on PATH
    try:
        r = subprocess.run(["which", "ffmpeg"], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass

    # Method 3: Common locations
    for p in ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/tmp/ffmpeg"]:
        if os.path.exists(p):
            return p

    # Method 4: Download static binary as last resort
    return download_ffmpeg()


def download_ffmpeg():
    """Download a small static ffmpeg binary."""
    dest = "/tmp/ffmpeg"
    if os.path.exists(dest) and os.access(dest, os.X_OK):
        return dest

    # Single binary from eugeneware/ffmpeg-static GitHub releases - no extraction needed
    url = "https://github.com/eugeneware/ffmpeg-static/releases/download/b6.1.1/ffmpeg-linux-x64"

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        with open(dest, "wb") as f:
            f.write(resp.read())

    os.chmod(dest, 0o755)
    return dest


_ffmpeg_path = None

def get_ffmpeg():
    global _ffmpeg_path
    if _ffmpeg_path is None:
        _ffmpeg_path = find_ffmpeg()
    return _ffmpeg_path


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


def extract_frames(path, target_fps, max_frames):
    ffmpeg = get_ffmpeg()
    cmd = [
        ffmpeg, "-i", path,
        "-vf", f"fps={target_fps},scale={FRAME_WIDTH}:{FRAME_HEIGHT}",
        "-pix_fmt", "rgb24",
        "-f", "rawvideo",
        "-frames:v", str(max_frames),
        "-v", "quiet",
        "-y", "pipe:1"
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
        frames.append(frame.reshape(FRAME_HEIGHT, FRAME_WIDTH, 3))

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
        tmp_path = download_video(video_url)
        file_size = os.path.getsize(tmp_path)

        frames = extract_frames(tmp_path, target_fps, max_frames)

        if not frames:
            return jsonify({
                "error": "No frames extracted",
                "file_size": file_size,
                "ffmpeg": get_ffmpeg()
            }), 400

        sample = frames[::max(1, len(frames) // 10)]
        bw_only = sum(1 for f in sample if is_bw(f)) > len(sample) * 0.8

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
