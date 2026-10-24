"""Shared helpers for the test suite: running the script, making synthetic
clips with ffmpeg, and measuring outputs with ffprobe.

Every test clip is generated on the fly from ffmpeg's built-in test sources.
Never add real footage to this repo (see CLAUDE.md)."""

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "facebegone.py"
STUB_RUNNER = Path(__file__).resolve().parent / "stub_run.py"


def haar_available():
    """False on OpenCV 5, which no longer ships the Haar face model."""
    import cv2
    data_dir = getattr(getattr(cv2, "data", None), "haarcascades", "")
    return hasattr(cv2, "CascadeClassifier") and os.path.exists(
        os.path.join(data_dir, "haarcascade_frontalface_alt2.xml"))


def load_script():
    """Import the script as a module, for unit-testing its functions."""
    spec = importlib.util.spec_from_file_location("face_be_gone_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_script(*args, stub=None, env=None, timeout=600):
    """Run the script in a subprocess. With stub="none" or stub="box", the
    detector is replaced (see stub_run.py) so only the I/O path is exercised."""
    if stub:
        cmd = [sys.executable, str(STUB_RUNNER), str(SCRIPT), stub, *map(str, args)]
    else:
        cmd = [sys.executable, str(SCRIPT), *map(str, args)]
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *map(str, args)], check=True)


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)


def video_stream(path):
    return next(s for s in probe(path)["streams"] if s["codec_type"] == "video")


def audio_streams(path):
    return [s for s in probe(path)["streams"] if s["codec_type"] == "audio"]


def rotation(stream):
    for side_data in stream.get("side_data_list", []):
        if "rotation" in side_data:
            return int(round(float(side_data["rotation"])))
    return int(stream.get("tags", {}).get("rotate", 0))


def frame_count(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return int(out.strip())


def opencv_read_info(path):
    """Which OpenCV backend opens `path`, and how many frames it reads.
    Used in failure messages, since OpenCV picks different backends per OS."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    backend = cap.getBackendName() if cap.isOpened() else "none"
    count = 0
    while cap.read()[0]:
        count += 1
    cap.release()
    return f"OpenCV {cv2.__version__} via {backend} reads {count} frames"


def psnr(reference, distorted):
    """Average PSNR in dB of `distorted` against `reference`, frame by frame."""
    err = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(reference), "-i", str(distorted),
         "-lavfi", "[0:v][1:v]psnr", "-f", "null", "-"],
        capture_output=True, text=True,
    ).stderr
    match = re.search(r"PSNR .*average:([\d.]+|inf)", err)
    assert match, f"no PSNR in ffmpeg output:\n{err[-2000:]}"
    return float(match.group(1))


def flash_times(path):
    """Timestamps where the picture turns from dark to bright."""
    path = Path(path)
    # movie= takes a filter-graph string, so pass a bare filename (no drive
    # letters or separators to escape) and run from the clip's folder.
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-f", "lavfi", "-i", f"movie={path.name},signalstats",
         "-show_entries", "frame=pts_time:frame_tags=lavfi.signalstats.YAVG", "-of", "csv=p=0"],
        capture_output=True, text=True, check=True, cwd=path.parent,
    ).stdout
    times, was_bright = [], False
    for line in out.splitlines():
        fields = line.split(",")
        if len(fields) < 2:
            continue
        bright = float(fields[1]) > 128
        if bright and not was_bright:
            times.append(float(fields[0]))
        was_bright = bright
    return times


def beep_times(path):
    """Timestamps where the audio turns from silence to tone."""
    err = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path), "-af", "silencedetect=n=-30dB:d=0.02", "-f", "null", "-"],
        capture_output=True, text=True,
    ).stderr
    return [0.0] + [float(t) for t in re.findall(r"silence_end: ([\d.]+)", err)]


def env_without_ffmpeg(empty_dir):
    """The current environment with PATH set to an empty folder, so ffmpeg
    can't be found. (Not Python's folder: on Linux that's often /usr/bin,
    where ffmpeg lives too.) The script is started by absolute path, so it
    doesn't need PATH itself."""
    env = dict(os.environ)
    env["PATH"] = str(empty_dir)
    return env
