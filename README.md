# face-be-gone

GPU-accelerated face blurring for video, built for precut footage. Every frame is detected independently, so hard cuts never break tracking the way they do in an NLE's face tracker.

- **Detector:** YOLOv8-face (default), with MediaPipe and OpenCV Haar as lighter alternatives
- **Output:** blurred video with the original audio muxed back in (needs `ffmpeg`)
- **Accuracy-first:** tuned so small, distant and off-angle faces, children's included, still get covered

## Quick start

```powershell
py -3.12 blur_faces_v1.1.py input.mp4 output.mp4 --strength 80 --margin 0.45 --confidence 0.25 --smoothing 0.5 --yolo-size x --imgsz 1280
```

That's the max-accuracy combination: slower, but it catches faces the defaults miss. For everyday clips, the defaults are fine:

```powershell
py -3.12 blur_faces_v1.1.py input.mp4 output.mp4 --strength 60
```

Use `--shape circle` for a feathered round blur, or `--pixelate` for a mosaic.

## Setup

Use **Python 3.12** and a **CUDA build of PyTorch**. Newer Pythons may not get GPU wheels, and pip silently falls back to a CPU build. The full walkthrough, including how to verify the GPU is actually in use, is in [docs/tutorial.md](docs/tutorial.md).

## License

[CC BY-NC 4.0](LICENSE) — attribution required, non-commercial use only.
Script by wishym1shy, coded with Claude Code.
