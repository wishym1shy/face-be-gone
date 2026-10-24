# face-be-gone

GPU-accelerated face blurring for video, built for precut footage. Every frame is detected independently, so hard cuts never break tracking the way they do in an NLE's face tracker.

- **Detector:** YOLOv8-face (default), with MediaPipe and OpenCV Haar as lighter alternatives
- **Output:** high-quality H.264 by default, or HEVC, NVENC, ProRes or DNxHR (`--codec`), with the original audio, rotation, color tags and timecode carried over (needs `ffmpeg`)
- **Accuracy-first:** tuned so small, distant and off-angle faces, children's included, still get covered

## Quick start

```powershell
py -3.12 facebegone.py input.mp4 output.mp4 --strength 80 --margin 0.45 --confidence 0.25 --smoothing 0.5 --yolo-size x --imgsz 1280
```

That's the max-accuracy combination: slower, but it catches faces the defaults miss. For everyday clips, the defaults are fine:

```powershell
py -3.12 facebegone.py input.mp4 output.mp4 --strength 60
```

Use `--shape circle` for a feathered round blur, or `--pixelate` for a mosaic. For footage going back into an NLE, write ProRes: `py -3.12 facebegone.py input.mp4 output.mov --codec prores`.

See [CHANGELOG.md](CHANGELOG.md) for what changed between versions; `--version` prints the one you have.

## Setup

Use **Python 3.12**, a **CUDA build of PyTorch** and **ffmpeg** (with `ffprobe`) on PATH. Newer Pythons may not get GPU wheels, and pip silently falls back to a CPU build. The full walkthrough, including how to verify the GPU is actually in use, is in [docs/tutorial.md](docs/tutorial.md).

## License

[MIT](LICENSE). Use it, change it, ship it, commercially or not; just keep the copyright notice.
Script by wishym1shy, coded with Claude Code.

### Third-party licenses

face-be-gone's own code is MIT, but its dependencies keep their own licenses. Most are permissive. The exception is [Ultralytics](https://github.com/ultralytics/ultralytics) (used by the default YOLO backend), which is **AGPL-3.0**:

- Running the script, and using the videos it produces, is unaffected, including for paid work.
- If you distribute a bundle that includes Ultralytics (an installer, an app, a Docker image), or run a modified version as a network service, the AGPL's terms apply to that bundle. Ultralytics also sells a commercial license.
- The MediaPipe and Haar backends don't use Ultralytics.
