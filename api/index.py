from flask import Flask, request, jsonify
import numpy as np
import tempfile
import os
import urllib.request
import json as jsonlib
import re

app = Flask(__name__)

FRAME_WIDTH = 80
FRAME_HEIGHT = 20


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


def rgb_to_hex(r, g, b):
    return f"#{r:02X}{g:02X}{b:02X}"


def is_bw(frame, threshold=15.0):
    diffs = np.max(frame.astype(int), axis=2) - np.min(frame.astype(int), axis=2)
    return float(np.mean(diffs)) < threshold


def resize_frame(frame, w, h):
    """Simple nearest-neighbor resize using numpy (no PIL needed)."""
    old_h, old_w = frame.shape[:2]
    row_idx = (np.arange(h) * old_h // h).astype(int)
    col_idx = (np.arange(w) * old_w // w).astype(int)
    return frame[np.ix_(row_idx, col_idx)]


def frame_to_ascii(frame, bw_only=False):
    small = resize_frame(frame, FRAME_WIDTH, FRAME_HEIGHT)
    lines = []
    for y in range(FRAME_HEIGHT):
        line = ""
        for x in range(FRAME_WIDTH):
            r, g, b = int(small[y, x, 0]), int(small[y, x, 1]), int(small[y, x, 2])
            if bw_only:
                gray = int(0.299 * r + 0.587 * g + 0.114 * b)
                line += "<#FFFFFF>\u23f9" if gray > 127 else "<#000000>\u23f9"
            else:
                line += f"<{rgb_to_hex(r, g, b)}>\u23f9"
        lines.append(line)
    return "\n".join(lines)


def parse_request():
    """Parse the incoming request from DiamondFire or any client."""
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

        # Use imageio with ffmpeg to read frames
        import imageio.v3 as iio
        import imageio_ffmpeg

        # Get ffmpeg binary path from imageio-ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

        # Read video properties
        gen = imageio_ffmpeg.read_frames(tmp_path)
        meta = next(gen)  # First yield is metadata dict

        original_fps = meta.get("fps", 30)
        frame_size = meta.get("size", (640, 480))  # (width, height)
        nframes = meta.get("nframes", 0)

        frame_interval = max(1, round(original_fps / target_fps))

        # Extract frames
        raw_frames = []
        frame_idx = 0
        w, h = frame_size

        for raw_frame in gen:
            if frame_idx % frame_interval == 0:
                # Convert raw bytes to numpy array
                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(h, w, 3)
                raw_frames.append(frame)
                if len(raw_frames) >= max_frames:
                    break
            frame_idx += 1

        if not raw_frames:
            return jsonify({
                "error": "No frames extracted",
                "file_size": file_size,
                "fps": original_fps,
                "nframes": nframes,
                "frame_size": frame_size
            }), 400

        # Check BW
        sample = raw_frames[::max(1, len(raw_frames) // 10)]
        bw_only = sum(1 for f in sample if is_bw(f)) > len(sample) * 0.8

        # Convert to ASCII
        ascii_frames = [frame_to_ascii(f, bw_only=bw_only) for f in raw_frames]

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
