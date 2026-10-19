#!/usr/bin/env python3
"""
blur_faces_v1.1.py

================================================================================
  Script by:      wishym1shy (GitHub)
  Coded with:     Claude Code
  Version:        1.1
  License:        CC BY-NC 4.0 (Attribution-NonCommercial 4.0 International)
                  https://creativecommons.org/licenses/by-nc/4.0/

                  You are free to share and adapt this script for any
                  non-commercial purpose, provided you give appropriate
                  credit to wishym1shy and indicate if changes were
                  made. Commercial use is not permitted under this license
                  without separate written permission from the author.
================================================================================

Automatic per-frame face blurring for video, using YOLOv8-Face (GPU-accelerated)
as the primary detector, with MediaPipe and OpenCV Haar cascade available as
fallback backends. Designed for precut footage: since detection runs
independently on every single frame, cuts don't break anything -- there's no
"track" to lose the way a traditional NLE tracker would.

Usage:
    python blur_faces_v1.1.py input.mp4 output.mp4
    python blur_faces_v1.1.py input.mp4 output.mp4 --strength 80 --margin 0.45 --confidence 0.25 --smoothing 0.5 --yolo-size x --imgsz 1280
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from collections import deque

import cv2
import numpy as np

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


def process_video(in_path, out_path, strength=35, confidence=0.5, pixelate=False,
                   margin=0.25, smoothing=0.5, model_variant="full", backend="yolo",
                   yolo_size="m", device=None, imgsz=640, shape="rect"):
    start_time = time.time()

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        print(f"ERROR: could not open {in_path}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Write to a temp silent video, then mux original audio back in with ffmpeg.
    tmp_video = out_path + ".noaudio.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (w, h))

    detector = make_detector(confidence, backend=backend, model_variant=model_variant,
                              yolo_size=yolo_size, device=device)

    tracks = {}          # key -> SmoothedBox
    next_key = 0
    max_match_dist = max(w, h) * 0.08  # how far a face can move between frames and still match

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
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

    cap.release()
    writer.release()
    if detector[0] == "mediapipe":
        detector[1].close()

    # Mux original audio back onto the processed (silent) video.
    if shutil.which("ffmpeg") is None:
        print("WARNING: ffmpeg not found on PATH. Output will have no audio.", file=sys.stderr)
        print("Install ffmpeg (e.g. https://ffmpeg.org/download.html) and ensure", file=sys.stderr)
        print("it's on PATH, then re-run to get audio in the output.", file=sys.stderr)
        shutil.copyfile(tmp_video, out_path)
        try:
            os.remove(tmp_video)
        except OSError:
            pass
        print(f"Done (no audio): {out_path}")
        print(f"Total time: {format_elapsed(time.time() - start_time)}")
        return

    cmd = [
        "ffmpeg", "-y",
        "-i", tmp_video,
        "-i", in_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-shortest",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ffmpeg mux failed, output has no audio. stderr:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        # fall back: at least deliver the silent video under the requested name
        shutil.copyfile(tmp_video, out_path)
    else:
        try:
            os.remove(tmp_video)
        except OSError:
            pass  # not critical if cleanup fails; output is already written

    print(f"Done: {out_path}")
    print(f"Total time: {format_elapsed(time.time() - start_time)}")


def main():
    parser = argparse.ArgumentParser(description="Auto-blur faces in a video, cut-proof.")
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
    args = parser.parse_args()

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
    )


if __name__ == "__main__":
    main()
