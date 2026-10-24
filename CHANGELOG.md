# Changelog

All notable changes to face-be-gone are listed here, newest first. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow `MAJOR.MINOR.PATCH` (see [CLAUDE.md](CLAUDE.md#versioning-and-changelog) for what bumps which).

## [Unreleased]

### Changed
- The script is now `facebegone.py` (was `blur_faces_v1.1.py`). The filename no longer changes between versions, so saved commands keep working after an update. Rename your local copy, or update any saved commands and batch files.
- Relicensed from CC BY-NC 4.0 to MIT. Commercial use is allowed; keep the copyright notice. Third-party licenses are listed in the README.

### Fixed
- Output quality ([#5](https://github.com/wishym1shy/face-be-gone/issues/5)). Frames used to go through OpenCV's old `mp4v` codec at a low bitrate, which visibly softened every output. They're now piped straight into ffmpeg and encoded once, as H.264 at CRF 16 by default. On test footage, quality outside blurred areas went from 25.8 dB to 44.8 dB PSNR, with a smaller file.
- Audio sync on variable frame rate footage. Frames are now resampled to a constant rate by timestamp, so audio stays in sync; on an 11-minute VFR test clip, v1.1 drifted by up to two minutes.
- Colors, full-range footage and rotation now round-trip correctly: decoding and encoding use the same color matrix, color tags are copied to the output, and rotated phone footage comes out upright.
- All audio tracks are kept (previously only the first). Audio is copied without re-encoding when the output container supports the codec, and re-encoded to AAC otherwise. Global metadata and timecode are carried over.
- The last frame is no longer dropped when the audio ends slightly before the video.

### Changed
- ffmpeg and ffprobe are now required. A missing ffmpeg stops the run with install instructions instead of silently writing a video with no audio. `--allow-no-audio` restores the old OpenCV fallback.
- A failed or cancelled run deletes the incomplete output file instead of leaving a truncated video behind.
- Errors are printed as one `ERROR:` message with exit status 1 instead of a traceback; Ctrl+C exits cleanly with status 130.

### Added
- `--codec {h264,hevc,h264_nvenc,hevc_nvenc,prores,dnxhr}` to choose the output codec. ProRes 422 HQ and DNxHR HQ are for NLE round-trips and need a `.mov` or `.mkv` output.
- `--crf` to set output quality for h264, hevc and the NVENC codecs.
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
