# Automated Face Blurring for Video (YOLOv8-Face)

A guide to setting up and running a Python script that automatically detects and blurs faces in video, frame-by-frame — including across hard cuts, where traditional NLE trackers (DaVinci Resolve, Premiere) fall apart. Uses YOLOv8-Face, a GPU-accelerated face detection model, with tuning guidance for tricky footage (off-angle faces, small/distant faces, multiple people in frame).

**Estimated time:** 5–15 minutes setup (one-time), 1–4 minutes per clip after setup.

---

## Why this exists

If you're editing precut footage that needs faces blurred for privacy, the standard NLE workflow is painful: you manually window-and-track each face, and every single cut resets the tracker — because Resolve/Premiere's tracker has no way to know "this is the same face" across a scene change. Off-angle and distant faces also tend to slip through unblurred with default tracker settings.

**The alternative:** run one script command. It detects every face independently on every frame — so cuts don't break anything, since there's no persistent "track" to lose in the first place. For difficult clips, a documented set of flags pushes accuracy higher at the cost of longer render time.

| | |
|---|---|
| ❌ **Manual tracking** | Reset the mask at every cut. Off-angle and distant faces slip through. Full manual re-review needed before shipping. |
| ✅ **This script** | One command. Every face, every frame, cuts included. Tune flags up for the rare difficult clip instead of re-doing everything by hand. |

---

## Step-by-Step Process

### Step 1 — Install Python 3.12

Use Python 3.12 specifically. Do not install 3.14 or newer — as of writing this, PyTorch's GPU-accelerated (CUDA) builds are not reliably available for Python versions newer than 3.12, and installing a newer version will silently fall back to CPU-only processing with no GPU speed benefit.

- Go to the official Python 3.12 download page (search "python 3.12.x download" and confirm you're on python.org) and download the Windows installer.
- **Critical step:** On the first installer screen, check the box labeled "Add python.exe to PATH" before clicking Install Now. This is the single most commonly missed step and causes `'python' is not recognized` errors later.
- Close and reopen any open PowerShell or terminal windows after installing — PATH changes do not apply to windows that were already open.
- Verify the install by running:
  ```powershell
  python --version
  ```
  It should print something like `Python 3.12.x`.
- ⚠ If you already have a newer Python version installed (e.g. 3.14) for other purposes, don't uninstall it — Windows can run multiple Python versions side by side. Use the `py -3.12` launcher prefix to specifically target 3.12 for every command below if `python` on its own resolves to a different version.

### Step 2 — Verify pip is working

- Run (use `py -3.12 -m pip --version` instead if you have multiple Python versions installed):
  ```powershell
  python -m pip --version
  ```
- This should print a pip version number. If it says pip is not recognized, always call pip through Python using `python -m pip` rather than a bare `pip` command — this avoids a common Windows PATH issue where pip's own shortcut isn't linked even though it's installed.

### Step 3 — Install required Python packages

This installs the AI/computer-vision libraries the script depends on, including PyTorch with GPU (CUDA) support.

- Run:
  ```powershell
  python -m pip install ultralytics mediapipe opencv-python
  ```
- **Separately**, install the GPU-enabled version of PyTorch. ⚠ **Don't hardcode a CUDA index tag from an old doc (including this one) without checking first** — PyTorch periodically retires older CUDA index versions (e.g. `cu124` was supported when this was first written, then later dropped in favor of newer tags), and pip does NOT error loudly when a tag no longer matches — it can silently fall back to installing a CPU-only build instead. Before running the install command:
  1. Visit https://pytorch.org/get-started/locally/
  2. Under "Your OS" pick Windows, "Package" pick Pip, "Language" pick Python, and for "Compute Platform" pick the newest CUDA version offered (not CPU).
  3. Copy the exact command shown there — it'll look like the one below but with a possibly different CUDA tag (e.g. `cu126`, `cu128`, etc. instead of `cu124`):
  ```powershell
  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX   # replace cuXXX with the tag pytorch.org shows
  ```
  - **If you're on Python 3.12:** the command should succeed and pull in CUDA-enabled PyTorch directly.
  - **If you're on a newer Python version and it fails** with `No matching distribution found for torch`: this confirms the version incompatibility from Step 1. Install Python 3.12 and repeat this step using `py -3.12 -m pip install ...` instead of plain `python -m pip install ...`.
- This is a large download (roughly 2–3GB) since CUDA-enabled PyTorch bundles GPU runtime libraries. Expect it to take a few minutes depending on connection speed.
- **Install ffmpeg.** The script uses it to read your footage and write the finished video with its audio, and stops with an error if it's missing.
  - Windows: `winget install Gyan.FFmpeg` (or download a build from https://ffmpeg.org/download.html and add its `bin` folder to PATH)
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg` (or your distro's package manager)
  - Close and reopen the terminal, then confirm both `ffmpeg -version` and `ffprobe -version` print a version.

### Step 4 — Verify GPU (CUDA) is actually being used

This is the step that confirms the install actually worked — it's possible for the install to succeed but silently produce a CPU-only build, which works correctly but runs far slower.

- Run (prefix with `py -3.12` if applicable):
  ```powershell
  python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
  ```
- **You want to see:** a version string containing a `+cuXXX` suffix (for example `2.6.0+cu124` or `2.9.0+cu128` — the exact number will vary depending on what's current when you install), and `True` on the second line.
- ⚠ If the version string ends in `+cpu` instead, or `cuda.is_available()` prints `False`, the GPU build didn't install correctly. This can happen even when you followed every step correctly, if the CUDA index tag used in Step 3 is no longer current — pip installs `torch` with no visible error either way, so a successful-looking install is not proof it worked. Go back to https://pytorch.org/get-started/locally/, grab the current command, uninstall (`python -m pip uninstall torch torchvision -y`) and reinstall.
- Run `nvidia-smi` in a terminal to confirm your NVIDIA driver is installed and recognized. This also shows current GPU memory usage — background apps (browsers, Discord, game clients) can reserve GPU memory, reducing the speed benefit available to this script.

### Step 5 — Get the script and a test clip ready

- Place `facebegone.py` in a working folder (e.g. a folder on your Desktop or in Documents).
- Place a short test video clip (5–15 seconds is plenty for testing) in the same folder.
- Open a terminal/PowerShell window and navigate to that folder:
  ```powershell
  cd path\to\your\folder
  ```

### Step 6 — Run the script with default (fast) settings

This is the standard command for routine volume — fast, and accurate for the large majority of clips.

- Run (use `py -3.12` prefix if applicable):
  ```powershell
  python facebegone.py test_clip.mp4 test_output.mp4 --strength 60
  ```
- The first time you use a given detection backend or model size, the script automatically downloads a small model file (a few hundred KB to ~50MB depending on model size) — expected, only happens once per model.
- Watch the terminal output for the line `YOLOv8-face running on: cuda`. If it says `cpu` instead, GPU acceleration isn't active — revisit Step 4.
- The `Encoding:` and `Audio:` lines show how the output is being written, for example `h264 (libx264, CRF 16)` and `aac (copied)`.
- Open the resulting output video and confirm faces are blurred correctly with no obvious gaps.

### Step 7 — Know when to reach for the higher-accuracy settings

Default settings work well for most footage. Use the settings below only on clips where review catches specific problems — not as a blanket default, since they meaningfully increase render time.

- **Off-angle or turned faces being missed:** lower `--confidence` (try `0.25` instead of the default `0.5`) and increase `--yolo-size` from `m` to `l` or `x` for a more capable detection model.
- **Distant/small background faces being missed:** add `--imgsz 1280` (or up to `--imgsz 1920` for a truly difficult clip). This tells the detector to examine each frame at a higher resolution before deciding whether a face is present — necessary because the detector normally shrinks every frame to a small fixed size internally regardless of your video's actual resolution.
- ⚠ Don't default `--imgsz` above 640 for routine batches. It significantly increases render time on every single frame of every video, whether or not a distant face ever actually appears in that footage — reserve it for specific flagged clips only.
- **Blur itself looks too light or too small:** increase `--strength` (try 75–85) and/or `--margin` (try 0.45) for a bigger, heavier blur region.
- **Tested maximum-accuracy combination** (confirmed to catch off-angle and distant faces that default settings miss, at the cost of significantly longer render time):
  ```powershell
  python facebegone.py clip.mp4 output.mp4 --strength 80 --margin 0.45 --confidence 0.25 --smoothing 0.5 --yolo-size x --imgsz 1280
  ```

### Step 8 — Pick an output format

The default output is high-quality H.264 that plays anywhere. Audio is copied from the source untouched when the output container supports it, and re-encoded to AAC otherwise.

- **Bringing the result back into Resolve or Premiere for more editing:** use an intermediate codec with a `.mov` output, so the footage doesn't lose quality to another lossy generation:
  ```powershell
  python facebegone.py clip.mp4 output.mov --codec prores
  ```
  `--codec dnxhr` works the same way. Both make large files.
- **Faster encoding on an NVIDIA GPU:** `--codec h264_nvenc` or `--codec hevc_nvenc`. Encoding is rarely the slow part, so this mostly helps with the fastest detection settings.
- **Smaller files:** `--codec hevc`, or raise `--crf` (lower is better quality and bigger; the h264 default is 16, which is visually close to the source).

### Step 9 — Know when it's faster to just do it by hand

- If a clip has one persistent, isolated face the script keeps missing even at max settings (e.g. one background extra at extreme distance), it's faster to let the automated pass handle everything else and manually window-and-track just that one face, rather than re-rendering the whole clip at ever-higher settings.
- **Rule of thumb:** escalate settings when the problem is a recurring, recognizable pattern (e.g. this footage always has a moving interviewer in frame). A one-off background miss is cheaper to fix by hand.

---

## Cautionary Notes

- ⚠ Don't install Python 3.14 or newer expecting GPU support to work — CUDA-enabled PyTorch builds aren't reliably available for Python versions newer than 3.12 as of writing this, and the install will silently succeed with a CPU-only build instead of failing loudly.
- ⚠ Always verify `torch.cuda.is_available()` prints `True` before assuming a run will use the GPU. A successful pip install doesn't guarantee GPU support was actually installed.
- ⚠ A hardcoded CUDA index tag (like `cu124`) in an install command can go stale over time as PyTorch retires old CUDA versions in favor of newer ones — pip will not error when this happens, it silently installs a CPU-only build instead. Always cross-check https://pytorch.org/get-started/locally/ for the current tag rather than trusting an old copy-pasted command, including the one in this doc.
- ⚠ Don't default `--imgsz` above 640 for routine batches — it's a per-clip accuracy tool for flagged problem footage, not a blanket setting, and meaningfully increases render time across all footage if left on by default.
- ⚠ Background applications (browsers, chat apps, game clients) can reserve significant GPU memory and reduce the speed benefit available to this script — check `nvidia-smi` if GPU runs feel slower than expected.
- ⚠ Variable frame rate footage (common from phones and screen recordings) is converted to a constant frame rate so the audio stays in sync. The script prints a note when this happens.
- ⚠ This script processes video locally on your machine. No footage is uploaded anywhere; only a small one-time AI model file (a few hundred KB to ~50MB) is downloaded from the model provider's servers.

---

## Tips for Efficiency

- Keep a short (5–10 second) standard test clip on hand for quickly verifying new settings before committing to a full-length render.
- Test new flag combinations on a short clip first — render time scales with both video length and chosen settings, so a 3-minute settings test can become a 20+ minute mistake on a full episode.
- If running many clips with the same settings, save the full command in a text file for quick copy-paste rather than re-typing flags each time.
- Check GPU memory usage with `nvidia-smi` before starting a batch of heavy renders (e.g. `--yolo-size x --imgsz 1280`) to confirm other applications aren't already using most of the available VRAM.
- When you find a good flag combination for a specific type of footage, jot it down so you skip the trial-and-error next time.
