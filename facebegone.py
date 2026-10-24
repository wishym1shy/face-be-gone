#!/usr/bin/env python3
"""
facebegone.py

================================================================================
  Script by:      wishym1shy (GitHub)
  Coded with:     Claude Code
  Version:        1.1.0 (keep in sync with __version__ and CHANGELOG.md)
  License:        MIT (see LICENSE). Free to use, modify and share,
                  commercially or not; keep the copyright notice.
                  Dependencies keep their own licenses: ultralytics is
                  AGPL-3.0 (see README, "Third-party licenses").
================================================================================

Automatic per-frame face blurring for video, using YOLOv8-Face (GPU-accelerated)
as the primary detector, with MediaPipe and OpenCV Haar cascade available as
fallback backends. Designed for precut footage: since detection runs
independently on every single frame, cuts don't break anything -- there's no
"track" to lose the way a traditional NLE tracker would.

Usage:
    python facebegone.py input.mp4 output.mp4
    python facebegone.py input.mp4 output.mp4 --strength 80 --margin 0.45 --confidence 0.25 --smoothing 0.5 --yolo-size x --imgsz 1280
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

import cv2
import numpy as np

__version__ = "1.1.0"

MODEL_URLS = {
    "short": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
    "full": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/1/blaze_face_full_range.tflite",
}
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))


def model_path_for(variant):
    fname = "blaze_face_short_range.tflite" if variant == "short" else "blaze_face_full_range.tflite"
    return os.path.join(MODEL_DIR, fname)


def ensure_model(variant="full"):
    """Download the requested MediaPipe face detector model on first run if not present.

    variant="short": tuned for close-up, mostly front-facing faces (e.g. one
        person centered in frame). Lighter, but drops off-angle/profile faces
        more readily.
    variant="full": tuned for a wider range of distances and angles -- the
        better default for footage with movement, multiple people, or
        turned/profile faces, which is why it's the default here.
    """
    path = model_path_for(variant)
    if os.path.exists(path) and os.path.getsize(path) > 100_000:
        return path
    print(f"Downloading face detection model ({variant}-range, one-time)...", file=sys.stderr)
    urllib.request.urlretrieve(MODEL_URLS[variant], path)
    print("Model downloaded.", file=sys.stderr)
    return path


YOLO_FACE_URLS = {
    "n": "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.pt",
    "s": "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8s-face-lindevs.pt",
    "m": "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8m-face-lindevs.pt",
    "l": "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8l-face-lindevs.pt",
    "x": "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8x-face-lindevs.pt",
}


def ensure_yolo_model(size="m"):
    """Download the requested YOLOv8-face model (lindevs/yolov8-face, WIDERFace-trained)
    on first run if not present. Sizes n/s/m/l/x trade speed for accuracy; m is a good
    default balance for GPU-equipped machines, matching the wider distance/angle
    accuracy improvement over MediaPipe's BlazeFace, at meaningfully higher compute cost.
    """
    fname = f"yolov8{size}-face-lindevs.pt"
    path = os.path.join(MODEL_DIR, fname)
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path
    print(f"Downloading YOLOv8{size}-face model (one-time, may take a moment)...", file=sys.stderr)
    urllib.request.urlretrieve(YOLO_FACE_URLS[size], path)
    print("Model downloaded.", file=sys.stderr)
    return path


def expand_box(x, y, w, h, frame_w, frame_h, margin=0.25):
    """Grow the detected box by `margin` fraction on each side, clamped to frame."""
    dx = int(w * margin)
    dy = int(h * margin)
    x0 = max(0, x - dx)
    y0 = max(0, y - dy)
    x1 = min(frame_w, x + w + dx)
    y1 = min(frame_h, y + h + dy)
    return x0, y0, x1, y1


def blur_region(frame, x0, y0, x1, y1, strength, pixelate=False, shape="rect"):
    """Apply blur/pixelate to the (x0,y0,x1,y1) region of frame, in-place.

    shape="rect" (default): fills the entire rectangular box, matching all
        prior versions of this script.
    shape="circle": blurs the same region, but only blends the blurred
        pixels back in within an ellipse inscribed in the box -- the four
        corners of the box stay untouched (original pixels), producing a
        soft circular/oval blur instead of a hard-edged square one.
    """
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return
    if pixelate:
        h, w = roi.shape[:2]
        # downscale then upscale (nearest) for a mosaic look
        small_w = max(1, w // 12)
        small_h = max(1, h // 12)
        small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        effect = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        k = strength | 1  # kernel size must be odd
        effect = cv2.GaussianBlur(roi, (k, k), 0)

    if shape == "circle":
        h, w = roi.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        axes = (max(1, w // 2), max(1, h // 2))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, thickness=-1)
        # feather the mask edge slightly so the circle blends softly rather
        # than leaving a hard-edged cutout
        feather = max(3, min(w, h) // 20) | 1  # odd kernel size
        mask = cv2.GaussianBlur(mask, (feather, feather), 0)
        mask_f = (mask.astype(np.float32) / 255.0)[:, :, None]
        blended = (effect.astype(np.float32) * mask_f +
                   roi.astype(np.float32) * (1 - mask_f))
        frame[y0:y1, x0:x1] = blended.astype(np.uint8)
    else:
        frame[y0:y1, x0:x1] = effect


class SmoothedBox:
    """
    Very small temporal smoother: keeps a short history of a box's position/size
    and returns an exponential-moving-average box, so blur regions don't jitter
    frame to frame even though detection is fully independent per frame.
    """

    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.state = None  # (cx, cy, w, h)

    def update(self, x0, y0, x1, y1):
        cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
        if self.state is None:
            self.state = (cx, cy, w, h)
        else:
            a = self.alpha
            self.state = tuple(
                a * new + (1 - a) * old for new, old in zip((cx, cy, w, h), self.state)
            )
        cx, cy, w, h = self.state
        return int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)


def match_detections_to_tracks(detections, tracks, max_dist):
    """
    Greedy nearest-centroid matching between this frame's detections and
    existing smoothed tracks, so each face keeps its own smoothing state
    across frames instead of jittering independently.
    Returns: dict detection_index -> track_key (existing or new)
    """
    assignments = {}
    used_tracks = set()
    track_items = list(tracks.items())

    for i, (x0, y0, x1, y1) in enumerate(detections):
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        best_key, best_dist = None, max_dist
        for key, smoother in track_items:
            if key in used_tracks or smoother.state is None:
                continue
            tcx, tcy = smoother.state[0], smoother.state[1]
            dist = ((cx - tcx) ** 2 + (cy - tcy) ** 2) ** 0.5
            if dist < best_dist:
                best_key, best_dist = key, dist
        if best_key is not None:
            assignments[i] = best_key
            used_tracks.add(best_key)
        else:
            assignments[i] = None  # signal: needs a new track key
    return assignments


def make_detector(confidence, backend="yolo", model_variant="full", yolo_size="m", device=None):
    """
    Build a face detector.

    backend="yolo" (default): YOLOv8-face (lindevs/yolov8-face), a proper
        object-detection network fine-tuned on faces via WIDERFace. Much more
        discriminating than MediaPipe's BlazeFace -- rejects background
        textures (grass, fabric patterns) that BlazeFace sometimes flags as
        faces, and holds onto off-angle/profile faces better. Meaningfully
        heavier to run; benefits a lot from a GPU (auto-detected via CUDA if
        available, falls back to CPU otherwise).
    backend="mediapipe": MediaPipe's current Tasks API
        (mediapipe.tasks.python.vision), which replaced the old
        mp.solutions.face_detection interface used in older tutorials/examples.
        Fast and fine on CPU, but more prone to false positives and dropping
        off-angle faces than YOLOv8-face.
    backend="haar": OpenCV's bundled Haar cascade -- no model download
        required, useful for a quick sanity test of the rest of the
        pipeline (matching/smoothing/blur/mux) but noticeably less accurate
        than either of the above.
    """
    if backend == "haar":
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        return ("haar", cv2.CascadeClassifier(cascade_path))

    if backend == "yolo":
        from ultralytics import YOLO
        import torch
        model_path = ensure_yolo_model(yolo_size)
        model = YOLO(model_path)
        resolved_device = device
        if resolved_device is None:
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"YOLOv8-face running on: {resolved_device}", file=sys.stderr)
        return ("yolo", (model, resolved_device))

    model_path = ensure_model(model_variant)
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    base_options = BaseOptions(model_asset_path=model_path)
    options = vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=confidence,
    )
    return ("mediapipe", vision.FaceDetector.create_from_options(options))


def detect_faces(detector, frame_rgb, frame_bgr, w, h, confidence=0.5, imgsz=640):
    """Run detection on one frame, return list of (x0, y0, x1, y1) in pixel coords."""
    backend, impl = detector
    boxes = []
    if backend == "haar":
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = impl.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, bw, bh) in faces:
            boxes.append((x, y, x + bw, y + bh))
    elif backend == "yolo":
        model, device = impl
        results = model.predict(frame_bgr, conf=confidence, device=device, imgsz=int(imgsz), verbose=False)
        r = results[0]
        for box in r.boxes:
            x0, y0, x1, y1 = box.xyxy[0].tolist()
            boxes.append((int(x0), int(y0), int(x1), int(y1)))
    else:
        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = impl.detect(mp_image)
        for det in result.detections:
            bbox = det.bounding_box
            boxes.append((bbox.origin_x, bbox.origin_y,
                          bbox.origin_x + bbox.width, bbox.origin_y + bbox.height))
    return boxes


def format_elapsed(seconds):
    """Format elapsed seconds as e.g. '1m 5s' or '42s' for a clean summary line."""
    seconds = int(round(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


class FaceBeGoneError(Exception):
    """A problem the user can fix (bad input, missing ffmpeg, wrong container).
    main() prints the message and exits with status 1."""


# Output codecs for --codec. `quality` is the encoder's quality flag (None means
# the codec is fixed-quality and --crf is ignored); `containers` limits which
# output extensions the codec can go in (None means any).
CODECS = {
    "h264": {
        "encoder": "libx264", "pix_fmt": "yuv420p", "args": ["-preset", "slow"],
        "quality": "-crf", "default_crf": 16, "containers": None,
    },
    "hevc": {
        "encoder": "libx265", "pix_fmt": "yuv420p", "args": ["-preset", "slow"],
        "quality": "-crf", "default_crf": 18, "containers": None,
    },
    "h264_nvenc": {
        "encoder": "h264_nvenc", "pix_fmt": "yuv420p", "args": ["-preset", "p5", "-rc", "vbr", "-b:v", "0"],
        "quality": "-cq", "default_crf": 19, "containers": None,
    },
    "hevc_nvenc": {
        "encoder": "hevc_nvenc", "pix_fmt": "yuv420p", "args": ["-preset", "p5", "-rc", "vbr", "-b:v", "0"],
        "quality": "-cq", "default_crf": 21, "containers": None,
    },
    "prores": {
        # ProRes 422 HQ, the usual intermediate for NLE round-trips
        "encoder": "prores_ks", "pix_fmt": "yuv422p10le", "args": ["-profile:v", "3", "-vendor", "apl0"],
        "quality": None, "default_crf": None, "containers": (".mov", ".mkv"),
    },
    "dnxhr": {
        "encoder": "dnxhd", "pix_fmt": "yuv422p", "args": ["-profile:v", "dnxhr_hq"],
        "quality": None, "default_crf": None, "containers": (".mov", ".mkv"),
    },
}

# Audio codecs that can be stream-copied into each container without
# re-encoding. Anything else is re-encoded to AAC. Matroska takes everything.
_MP4_AUDIO = {"aac", "mp3", "ac3", "eac3", "alac"}
AUDIO_COPY_OK = {
    ".mp4": _MP4_AUDIO,
    ".m4v": _MP4_AUDIO,
    ".mov": _MP4_AUDIO | {"pcm_s16le", "pcm_s16be", "pcm_s24le", "pcm_s24be",
                          "pcm_s32le", "pcm_s32be", "pcm_f32le", "pcm_f32be"},
    ".mkv": "any",
}

# ffprobe color_space -> the matching matrix name for ffmpeg's scale filter
SCALE_MATRIX = {
    "bt709": "bt709", "smpte170m": "smpte170m", "bt470bg": "bt470",
    "bt2020nc": "bt2020", "bt2020c": "bt2020", "smpte240m": "smpte240m", "fcc": "fcc",
}


def _known(value):
    """ffprobe reports missing color tags as absent, 'unknown' or 'unspecified'."""
    return value if value and value not in ("unknown", "unspecified", "reserved") else None


def probe_video(path):
    """Read what we need from the source with ffprobe: upright frame size (after
    rotation), frame rate, frame count estimate, color tags, audio streams and
    timecode."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise FaceBeGoneError(f"could not read {path}:\n{result.stderr.strip()}")
    info = json.loads(result.stdout)
    streams = info.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and not s.get("disposition", {}).get("attached_pic")), None)
    if video is None:
        raise FaceBeGoneError(f"no video stream found in {path}")

    rotation = 0
    for side_data in video.get("side_data_list", []):
        if "rotation" in side_data:
            rotation = int(round(float(side_data["rotation"])))
    if not rotation and "rotate" in video.get("tags", {}):
        rotation = int(video["tags"]["rotate"])
    w, h = int(video["width"]), int(video["height"])
    if rotation % 180 != 0:
        w, h = h, w  # ffmpeg auto-rotates on decode, so frames arrive upright

    avg_rate = video.get("avg_frame_rate", "0/0")
    real_rate = video.get("r_frame_rate", "0/0")
    rate = avg_rate if _rate_value(avg_rate) > 0 else real_rate
    if _rate_value(rate) <= 0:
        rate = "30"

    duration = video.get("duration") or info.get("format", {}).get("duration")
    if duration:
        total_frames = int(round(float(duration) * _rate_value(rate)))
    else:
        total_frames = int(video.get("nb_frames") or 0)

    pix_fmt = video.get("pix_fmt", "")
    color_range = _known(video.get("color_range"))
    if color_range is None and pix_fmt.startswith("yuvj"):
        color_range = "pc"

    sar = video.get("sample_aspect_ratio", "1:1")
    if sar in ("1:1", "0:1", "N/A") or ":" not in sar:
        sar = None
    elif rotation % 180 != 0:
        num, den = sar.split(":")
        sar = f"{den}:{num}"  # pixels are rotated too

    timecode = None
    for tags in [info.get("format", {}).get("tags", {})] + [s.get("tags", {}) for s in streams]:
        if tags.get("timecode"):
            timecode = tags["timecode"]
            break

    return {
        "width": w, "height": h, "rotation": rotation,
        "rate": rate, "fps": _rate_value(rate),
        "vfr": _rate_value(avg_rate) > 0 and _rate_value(avg_rate) != _rate_value(real_rate),
        "total_frames": total_frames,
        "color_range": color_range,
        "color_space": _known(video.get("color_space")),
        "color_primaries": _known(video.get("color_primaries")),
        "color_transfer": _known(video.get("color_transfer")),
        "audio": [s for s in streams if s.get("codec_type") == "audio"],
        "timecode": timecode,
        "sar": sar,
    }


def _rate_value(rate):
    """'30000/1001' -> 29.97; returns 0 for '0/0' or garbage."""
    try:
        num, _, den = str(rate).partition("/")
        return float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return 0.0


def _color_conversion(info):
    """Matrix and range used for both YUV->BGR on decode and BGR->YUV on encode.
    Using the same pair both ways is what keeps unblurred pixels looking
    identical to the source; for untagged sources, fall back to the usual
    convention (BT.709 for HD, BT.601 for SD) and leave the output untagged too."""
    matrix = SCALE_MATRIX.get(info["color_space"]) or ("bt709" if info["height"] >= 720 else "smpte170m")
    rng = info["color_range"] or "tv"
    return matrix, rng


def _stderr_tail(f, lines=20):
    f.seek(0)
    text = f.read().decode("utf-8", errors="replace").strip().splitlines()
    return "\n".join(text[-lines:])


def available_encoders():
    result = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
    names = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 6:
            names.add(parts[1])
    return names


class FfmpegReader:
    """Decode the source with ffmpeg into upright BGR frames on a pipe.

    Rotation is applied by ffmpeg, color is converted with an explicit matrix
    (see _color_conversion), and the `fps` filter resamples variable frame rate
    sources to a constant rate by timestamp, so frame N is always at N / fps
    seconds and stays in sync with the audio."""

    def __init__(self, path, info):
        self.w, self.h = info["width"], info["height"]
        self.frame_bytes = self.w * self.h * 3
        matrix, rng = _color_conversion(info)
        vf = f"fps={info['rate']},scale=in_color_matrix={matrix}:in_range={rng},format=bgr24"
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-map", "0:v:0",
               "-an", "-sn", "-dn", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        self.stderr = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=self.stderr)

    def read(self):
        data = self.proc.stdout.read(self.frame_bytes)
        if len(data) < self.frame_bytes:
            return None
        return np.frombuffer(data, dtype=np.uint8).reshape((self.h, self.w, 3)).copy()

    def close(self):
        self.proc.stdout.close()
        returncode = self.proc.wait()
        if returncode != 0:
            raise FaceBeGoneError(f"ffmpeg failed while decoding the input:\n{_stderr_tail(self.stderr)}")

    def abort(self):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()


class FfmpegWriter:
    """Encode BGR frames from a pipe straight into the final file, muxing the
    source's audio, metadata and timecode in the same ffmpeg command."""

    def __init__(self, out_path, in_path, info, codec="h264", crf=None):
        spec = CODECS[codec]
        ext = os.path.splitext(out_path)[1].lower()
        matrix, rng = _color_conversion(info)

        vf = []
        if info["width"] % 2 or info["height"] % 2:
            vf.append("pad=ceil(iw/2)*2:ceil(ih/2)*2")  # 4:2:0 / 4:2:2 need even dimensions
        vf.append(f"scale=out_color_matrix={matrix}:out_range={rng},format={spec['pix_fmt']}")
        if info["sar"]:
            vf.append(f"setsar={info['sar'].replace(':', '/')}")  # anamorphic sources keep their display shape

        cmd = ["ffmpeg", "-y", "-v", "error", "-nostdin",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{info['width']}x{info['height']}",
               "-r", info["rate"], "-i", "-",
               "-i", in_path,
               "-map", "0:v:0", "-map", "1:a?", "-map_metadata", "1",
               "-vf", ",".join(vf),
               "-c:v", spec["encoder"], *spec["args"]]
        if spec["quality"]:
            cmd += [spec["quality"], str(crf if crf is not None else spec["default_crf"])]
        if codec in ("hevc", "hevc_nvenc") and ext in (".mp4", ".m4v", ".mov"):
            cmd += ["-tag:v", "hvc1"]  # lets QuickTime and Apple devices play HEVC
        for flag, key in (("-color_range", "color_range"), ("-color_primaries", "color_primaries"),
                          ("-color_trc", "color_transfer"), ("-colorspace", "color_space")):
            if info[key]:
                cmd += [flag, info[key]]
        if info["timecode"] and ext in (".mov", ".mp4", ".m4v"):
            cmd += ["-timecode", info["timecode"]]

        self.audio_summary = []
        copy_ok = AUDIO_COPY_OK.get(ext, set())
        for i, stream in enumerate(info["audio"]):
            name = stream.get("codec_name", "?")
            if copy_ok == "any" or name in copy_ok:
                cmd += [f"-c:a:{i}", "copy"]
                self.audio_summary.append(f"{name} (copied)")
            else:
                cmd += [f"-c:a:{i}", "aac", f"-b:a:{i}", "320k"]
                self.audio_summary.append(f"{name} -> aac (can't be copied into {ext or 'this container'})")

        cmd.append(out_path)
        self.out_path = out_path
        self.encoder = spec["encoder"]
        self.stderr = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self.stderr)

    def write(self, frame):
        try:
            self.proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError):
            self.proc.wait()
            hint = ""
            if "nvenc" in self.encoder:
                hint = "\nNVENC needs an NVIDIA GPU with a current driver; use --codec h264 or hevc otherwise."
            raise FaceBeGoneError(
                f"ffmpeg stopped while encoding {self.out_path}:\n{_stderr_tail(self.stderr)}{hint}")

    def close(self):
        try:
            self.proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        returncode = self.proc.wait()
        if returncode != 0:
            raise FaceBeGoneError(f"ffmpeg failed while encoding {self.out_path}:\n{_stderr_tail(self.stderr)}")

    def abort(self):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()
        _remove_partial(self.out_path)


class OpenCVReader:
    """Fallback decoder used only with --allow-no-audio when ffmpeg is missing."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FaceBeGoneError(f"could not open {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        self.cap.release()

    abort = close


class OpenCVWriter:
    """Fallback encoder (mp4v, no audio) used only with --allow-no-audio."""

    def __init__(self, out_path, fps, w, h):
        self.out_path = out_path
        self.writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def write(self, frame):
        self.writer.write(frame)

    def close(self):
        self.writer.release()

    def abort(self):
        self.writer.release()
        _remove_partial(self.out_path)


def _remove_partial(path):
    """Delete an unfinished output so a truncated render can't be mistaken for a finished one."""
    try:
        os.remove(path)
        print(f"Removed incomplete output: {path}", file=sys.stderr)
    except OSError:
        pass


def check_ffmpeg(codec, out_path):
    """Fail fast, before any model loading or detection, on problems that would
    otherwise only show up after minutes of processing."""
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            return f"{tool} not found on PATH"
    spec = CODECS[codec]
    ext = os.path.splitext(out_path)[1].lower()
    if spec["containers"] and ext not in spec["containers"]:
        raise FaceBeGoneError(
            f"--codec {codec} needs a {' or '.join(spec['containers'])} output file, not '{ext or out_path}'.")
    if spec["encoder"] not in available_encoders():
        raise FaceBeGoneError(
            f"your ffmpeg build has no '{spec['encoder']}' encoder (needed for --codec {codec}). "
            "Install a full ffmpeg build or pick another --codec.")
    return None


def process_video(in_path, out_path, strength=35, confidence=0.5, pixelate=False,
                   margin=0.25, smoothing=0.5, model_variant="full", backend="yolo",
                   yolo_size="m", device=None, imgsz=640, shape="rect",
                   codec="h264", crf=None, allow_no_audio=False):
    start_time = time.time()
    print(f"face-be-gone {__version__}", file=sys.stderr)

    if not os.path.exists(in_path):
        raise FaceBeGoneError(f"could not open {in_path}")
    if os.path.abspath(in_path) == os.path.abspath(out_path):
        raise FaceBeGoneError("output path is the same as the input; pick a different output file name.")
    if crf is not None and CODECS[codec]["quality"] is None:
        print(f"Note: --crf is ignored for --codec {codec} (fixed-quality codec).", file=sys.stderr)

    ffmpeg_problem = check_ffmpeg(codec, out_path)
    if ffmpeg_problem and not allow_no_audio:
        raise FaceBeGoneError(
            f"{ffmpeg_problem}. ffmpeg is needed to encode the output and keep the audio.\n"
            "Install ffmpeg (https://ffmpeg.org/download.html) and make sure ffmpeg and ffprobe are on PATH.\n"
            "To render anyway with no audio and lower quality, re-run with --allow-no-audio.")

    if ffmpeg_problem:
        print("=" * 72, file=sys.stderr)
        print(f"WARNING: {ffmpeg_problem}. Rendering with OpenCV because of --allow-no-audio:", file=sys.stderr)
        print("  the output will have NO AUDIO and lower quality (mp4v codec).", file=sys.stderr)
        print("=" * 72, file=sys.stderr)
        reader = OpenCVReader(in_path)
        w, h, fps, total_frames = reader.w, reader.h, reader.fps, reader.total_frames
    else:
        info = probe_video(in_path)
        w, h, total_frames = info["width"], info["height"], info["total_frames"]
        if info["vfr"]:
            print(f"Variable frame rate source: output will be a constant {info['fps']:.3f} fps "
                  "to keep audio in sync.", file=sys.stderr)

    detector = make_detector(confidence, backend=backend, model_variant=model_variant,
                              yolo_size=yolo_size, device=device)

    if ffmpeg_problem:
        writer = OpenCVWriter(out_path, fps, w, h)
    else:
        reader = FfmpegReader(in_path, info)
        writer = FfmpegWriter(out_path, in_path, info, codec=codec, crf=crf)
        spec = CODECS[codec]
        quality = f", {spec['quality'].lstrip('-').upper()} {crf if crf is not None else spec['default_crf']}" \
            if spec["quality"] else ""
        print(f"Encoding: {codec} ({spec['encoder']}{quality}), {w}x{h} @ {info['fps']:.3f} fps", file=sys.stderr)
        print(f"Audio: {', '.join(writer.audio_summary) or 'none in source'}", file=sys.stderr)

    tracks = {}          # key -> SmoothedBox
    next_key = 0
    max_match_dist = max(w, h) * 0.08  # how far a face can move between frames and still match

    frame_idx = 0
    try:
        while True:
            frame = reader.read()
            if frame is None:
                break
            frame_idx += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = detect_faces(detector, rgb, frame, w, h, confidence=confidence, imgsz=imgsz)

            raw_boxes = []
            for (x, y, x1_raw, y1_raw) in detections:
                bw = x1_raw - x
                bh = y1_raw - y
                x0, y0, x1, y1 = expand_box(x, y, bw, bh, w, h, margin=margin)
                raw_boxes.append((x0, y0, x1, y1))

            assignments = match_detections_to_tracks(raw_boxes, tracks, max_match_dist)

            active_keys = set()
            for i, box in enumerate(raw_boxes):
                key = assignments[i]
                if key is None:
                    key = next_key
                    next_key += 1
                    tracks[key] = SmoothedBox(alpha=smoothing)
                sx0, sy0, sx1, sy1 = tracks[key].update(*box)
                blur_region(frame, sx0, sy0, sx1, sy1, strength, pixelate=pixelate, shape=shape)
                active_keys.add(key)

            # drop tracks that weren't seen this frame (face left / cut happened)
            for key in list(tracks.keys()):
                if key not in active_keys:
                    del tracks[key]

            writer.write(frame)

            if frame_idx % 100 == 0 or frame_idx == total_frames:
                pct = (frame_idx / total_frames * 100) if total_frames else 0
                elapsed = format_elapsed(time.time() - start_time)
                print(f"  {frame_idx}/{total_frames} frames ({pct:.1f}%) -- {elapsed} elapsed", file=sys.stderr)

        reader.close()
        if frame_idx == 0:
            raise FaceBeGoneError(f"no frames could be read from {in_path}")
        writer.close()
    except BaseException:
        reader.abort()
        writer.abort()
        raise
    finally:
        if detector[0] == "mediapipe":
            detector[1].close()

    print(f"Done{' (no audio)' if ffmpeg_problem else ''}: {out_path}")
    print(f"Total time: {format_elapsed(time.time() - start_time)}")


def main():
    parser = argparse.ArgumentParser(description="Auto-blur faces in a video, cut-proof.")
    parser.add_argument("--version", action="version", version=f"face-be-gone {__version__}")
    parser.add_argument("input", help="input video path")
    parser.add_argument("output", help="output video path")
    parser.add_argument("--strength", type=int, default=35,
                         help="Gaussian blur kernel size (odd number, default 35)")
    parser.add_argument("--confidence", type=float, default=0.5,
                         help="min detection confidence 0-1 (default 0.5; lower = more sensitive)")
    parser.add_argument("--pixelate", action="store_true",
                         help="use mosaic/pixelate instead of Gaussian blur")
    parser.add_argument("--shape", default="rect", choices=["rect", "circle"],
                         help="blur region shape (default rect): 'circle' blurs within an "
                              "ellipse inscribed in the detected box, leaving the box corners "
                              "unblurred, for a softer/rounder look instead of a hard-edged square")
    parser.add_argument("--margin", type=float, default=0.25,
                         help="expand detected box by this fraction (default 0.25)")
    parser.add_argument("--smoothing", type=float, default=0.5,
                         help="0-1, lower = smoother/laggier, higher = snappier/jitterier (default 0.5)")
    parser.add_argument("--model", default="full", choices=["full", "short"],
                         help="MediaPipe model variant (only used with --backend mediapipe): "
                              "full=wider range of angles/distances (default), "
                              "short=close-up front-facing faces only, lighter but drops profiles more readily")
    parser.add_argument("--backend", default="yolo", choices=["yolo", "mediapipe", "haar"],
                         help="detector backend (default yolo: most accurate, GPU-accelerated if available; "
                              "mediapipe: fast on CPU, more false positives; "
                              "haar: no download needed, least accurate, for pipeline testing only)")
    parser.add_argument("--yolo-size", default="m", choices=["n", "s", "m", "l", "x"],
                         help="YOLOv8-face model size, n(ano) to x(large) -- bigger is more accurate but "
                              "slower (default m: good accuracy/speed balance for GPU use)")
    parser.add_argument("--device", default=None,
                         help="compute device for YOLO backend, e.g. 'cuda', 'cuda:0', or 'cpu'. "
                              "Default: auto-detect (uses GPU if available).")
    parser.add_argument("--imgsz", type=int, default=640,
                         help="YOLO detection resolution (default 640). Ultralytics downsizes every "
                              "frame to this before detecting, regardless of source resolution -- "
                              "raising it (e.g. 960, 1280) helps catch small/distant faces at the cost "
                              "of slower inference and more VRAM. Only affects --backend yolo.")
    parser.add_argument("--codec", default="h264", choices=list(CODECS),
                         help="output video codec (default h264). h264/hevc: high-quality software "
                              "encode; h264_nvenc/hevc_nvenc: fast NVIDIA GPU encode; prores: ProRes 422 HQ "
                              "and dnxhr: DNxHR HQ, both for NLE round-trips (need a .mov or .mkv output)")
    parser.add_argument("--crf", type=int, default=None,
                         help="output quality, lower = better and bigger (defaults: h264 16, hevc 18, "
                              "h264_nvenc 19, hevc_nvenc 21). Ignored for prores and dnxhr.")
    parser.add_argument("--allow-no-audio", action="store_true",
                         help="if ffmpeg is missing, render anyway with OpenCV instead of stopping: "
                              "no audio and lower quality (mp4v)")
    args = parser.parse_args()

    try:
        process_video(
            args.input, args.output,
            strength=args.strength,
            confidence=args.confidence,
            pixelate=args.pixelate,
            margin=args.margin,
            smoothing=args.smoothing,
            model_variant=args.model,
            backend=args.backend,
            yolo_size=args.yolo_size,
            device=args.device,
            imgsz=args.imgsz,
            shape=args.shape,
            codec=args.codec,
            crf=args.crf,
            allow_no_audio=args.allow_no_audio,
        )
    except FaceBeGoneError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
