# Changelog

All notable changes to face-be-gone are listed here, newest first. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow `MAJOR.MINOR.PATCH` (see [CLAUDE.md](CLAUDE.md#versioning-and-changelog) for what bumps which).

## [Unreleased]

### Changed
- The script is now `facebegone.py` (was `blur_faces_v1.1.py`). The filename no longer changes between versions, so saved commands keep working after an update. Rename your local copy, or update any saved commands and batch files.
- Relicensed from CC BY-NC 4.0 to MIT. Commercial use is allowed; keep the copyright notice. Third-party licenses are listed in the README.

### Added
- `--version` prints the script version and exits. Runs also print the version at startup, so logs and bug reports show which version produced an output.
- This changelog.

## [1.1.0] - 2026-10-19

First public release, as `blur_faces_v1.1.py`.

- Per-frame face detection, so hard cuts in precut footage never break coverage.
- Detector backends: YOLOv8-face (default, GPU-accelerated through PyTorch/CUDA when available), MediaPipe and OpenCV Haar.
- YOLO model sizes `n`/`s`/`m`/`l`/`x` (`--yolo-size`) and detection resolution (`--imgsz`). Models download on first use.
- Gaussian blur (`--strength`) or mosaic (`--pixelate`), rectangular or feathered-ellipse regions (`--shape`), and box expansion (`--margin`).
- Per-face EMA smoothing of blur boxes (`--smoothing`).
- Original audio muxed back in with ffmpeg.

[Unreleased]: https://github.com/wishym1shy/face-be-gone/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/wishym1shy/face-be-gone/releases/tag/v1.1.0
