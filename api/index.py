from flask import Flask, request, jsonify
import cv2
import numpy as np
import tempfile
import os
import urllib.request

app = Flask(__name__)

# Minecraft tab list max dimensions
FRAME_WIDTH = 80
FRAME_HEIGHT = 20


def download_video(url: str) -> str:
    """Download video from URL to a temp file."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    urllib.request.urlretrieve(url, tmp.name)
    return tmp.name


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB to hex string."""
    return f"#{r:02X}{g:02X}{b:02X}"


def is_black_and_white(frame: np.ndarray, threshold: float = 10.0) -> bool:
    """Check if a frame is essentially black and white."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    return float(np.mean(saturation)) < threshold


def frame_to_ascii(frame: np.ndarray, bw_only: bool = False) -> str:
    """Convert a single frame to colored square ASCII art."""
    # Resize to tab list dimensions
    resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
    # OpenCV uses BGR, convert to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    lines = []
    for y in range(FRAME_HEIGHT):
        line = ""
        for x in range(FRAME_WIDTH):
            r, g, b = int(rgb[y, x, 0]), int(rgb[y, x, 1]), int(rgb[y, x, 2])

            if bw_only:
                # Convert to grayscale and snap to black or white
                gray = int(0.299 * r + 0.587 * g + 0.114 * b)
                if gray > 127:
                    line += "<#FFFFFF>\u23f9"
                else:
                    line += "<#000000>\u23f9"
            else:
                hex_color = rgb_to_hex(r, g, b)
                line += f"<{hex_color}>\u23f9"

        lines.append(line)

    return "\n".join(lines)


@app.route("/", methods=["GET"])
def home():
    # If query params include a url, treat as a convert request
    if request.args.get("url"):
        return convert_video(dict(request.args))

    return jsonify({
        "status": "ok",
        "usage": {
            "method": "POST",
            "url": "/",
            "body": {
                "url": "https://example.com/video.mp4",
                "fps": 10,
                "max_frames": 100
            },
            "alt_method": "GET /?url=https://example.com/video.mp4&fps=10&max_frames=100",
            "description": "Send a video URL and get back a list of ASCII frames with hex color codes, sized for Minecraft tab list (80x20)."
        }
    })


@app.route("/", methods=["POST"])
def convert():
    import json as jsonlib

    # Try multiple ways to parse the body (DiamondFire may not send Content-Type header)
    data = None

    # 1. Try normal JSON parsing
    data = request.get_json(silent=True)

    # 2. Try force-parsing the raw body as JSON
    if not data:
        try:
            raw = request.get_data(as_text=True)
            data = jsonlib.loads(raw)
        except Exception:
            pass

    # 3. Try form data
    if not data and request.form:
        data = dict(request.form)

    # 4. Try query params as fallback
    if not data and request.args:
        data = dict(request.args)

    if not data or "url" not in data:
        return jsonify({
            "error": "Missing 'url' in request body",
            "hint": "Send JSON like: {\"url\": \"https://example.com/video.mp4\"}",
            "received_content_type": request.content_type,
            "received_body": request.get_data(as_text=True)[:500]
        }), 400

    return convert_video(data)


def convert_video(data):
    """Shared video conversion logic for both GET and POST."""
    video_url = data["url"]
    target_fps = int(data.get("fps", 10))
    max_frames = int(data.get("max_frames", 100))

    tmp_path = None
    try:
        # Download the video
        tmp_path = download_video(video_url)

        # Open with OpenCV
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return jsonify({"error": "Could not open video"}), 400

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps <= 0:
            original_fps = 30.0

        frame_interval = max(1, int(original_fps / target_fps))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # First pass: check if video is black and white (sample a few frames)
        sample_indices = np.linspace(0, total_frames - 1, min(10, total_frames), dtype=int)
        bw_votes = 0
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, sample_frame = cap.read()
            if ret and is_black_and_white(sample_frame):
                bw_votes += 1

        bw_only = bw_votes > len(sample_indices) * 0.8  # 80%+ frames are BW

        # Second pass: extract frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                ascii_frame = frame_to_ascii(frame, bw_only=bw_only)
                frames.append(ascii_frame)

                if len(frames) >= max_frames:
                    break

            frame_count += 1

        cap.release()

        return jsonify({
            "frames": frames,
            "frame_count": len(frames),
            "dimensions": f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
            "black_and_white": bw_only,
            "fps": target_fps
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
