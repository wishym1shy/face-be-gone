# CLAUDE.md

Guidance for Claude Code (and other contributors) working in this repository.

## What this project is

A GPU-accelerated command-line tool that detects and blurs faces in video. Detection runs independently on every frame, so hard cuts in precut footage never break it. The default detector is YOLOv8-face, with MediaPipe and OpenCV Haar available as lighter alternatives. Audio is muxed back in with `ffmpeg`.

- `blur_faces_v1.1.py`: the script (single file; the version is in the filename).
- `docs/tutorial.md`: end-user setup and usage guide.
- `README.md`: short overview and quick start.
- `LICENSE`: CC BY-NC 4.0 (attribution required, non-commercial only).

## Workflow: always use pull requests

- Never commit or push directly to `main`. Every change to the script, the docs, or any future web app goes through a branch and a pull request, however small.
- Branch from an up-to-date `main`, using a short descriptive name (`fix-edge-detection`, `docs-flag-reference`).
- Keep each PR focused on one change, and describe what changed and how it was tested.
- Do not merge your own PR unless the repo owner says to.

## Privacy: keep the repo anonymous and generic

This is a public repo. Do not add, anywhere in code, docs, comments, commit messages, PR text, or sample output:

- Real personal names, email addresses, job titles, or employer/department names.
- Client, company, or project names, or anything that identifies who the tool was built for.
- Machine names, usernames, local file paths, or hardware that identifies a person (generic specs such as "an 8GB NVIDIA GPU" are fine).
- Real footage, screenshots of real people, or test clips. Never commit video files.

Use the owner's GitHub handle for attribution and generic wording ("users", "editors", "your footage") elsewhere. When writing git commits, use the configured GitHub identity, not a personal email.

## Versioning

- The script's version lives in its filename and header (`blur_faces_vX.Y.py`).
- For a behavior change, add a new versioned file rather than overwriting the old one, and keep the previous version available until the new one is validated on real footage.
- Update every command example in the README and tutorial to the current filename when the version changes.
- Keep the header's license and attribution block intact.

## Accuracy comes first

Missing a face is the worst failure this tool can have, children's faces especially. Speed is secondary.

- Don't trade detection recall for speed or smoothness by default.
- Don't add smoothing, prediction, or tiling logic without testing it on real, varied footage first. Faces at frame edges, small or distant faces, and off-angle faces are the cases that regress. Synthetic tests are not enough.
- `--imgsz` is not monotonic: a higher value can miss faces a lower one catches. Validate changes against both.
- Keep opt-in accuracy flags opt-in; the defaults should stay fast and predictable.

## Environment notes

- Target Python 3.12. CUDA builds of PyTorch are not reliably available for newer Python versions, and pip silently installs a CPU-only build instead of failing.
- Never hardcode a CUDA index tag in docs as if it were permanent. Point readers to https://pytorch.org/get-started/locally/ and tell them to verify `torch.cuda.is_available()` is `True` and the version has a `+cuXXX` suffix.
- Windows users hit `pip` PATH problems often; documented commands use `python -m pip` and the `py -3.12` launcher.
- Scripts must run on Windows, macOS and Linux: use `os`/`shutil` rather than shell commands for file operations, and check that `ffmpeg` exists before calling it.

## Style

- Keep the script a single dependency-light file with a clear `--help`; document every flag in the README or tutorial.
- Print what the user needs to confirm the run is healthy: the device in use (`cuda` or `cpu`), periodic progress, and total elapsed time.
- Don't commit model weights (`*.pt`, `*.tflite`); the script downloads them on first use.
