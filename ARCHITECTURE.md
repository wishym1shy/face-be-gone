# Architecture

How face-be-gone works today, why it works that way, and where planned features plug in. Read this before changing detection, tracking, rendering or output code. For contribution rules (PR workflow, privacy, versioning), see [CLAUDE.md](CLAUDE.md).

Line references point to `blur_faces_v1.1.py`.

---

## 1. Design goals, in priority order

1. **Never miss a face.** Recall beats everything: speed, smoothness, and a few false positives. Children's faces are the highest-stakes case. When two designs trade recall against anything else, pick recall.
2. **Cut-proof on precut footage.** No state that has to survive a hard cut. Each frame is detected on its own, so edits in the source never break coverage.
3. **Predictable for non-technical users.** One command, sensible defaults, clear progress output and a clear sign of whether the GPU is in use.
4. **Speed last.** A slower run that catches more faces is the right trade. Expensive options (`--yolo-size x`, `--imgsz 1280`) are opt-in, not defaults.

## 2. Pipeline overview

Everything is one streaming pass over the video, followed by an audio mux:

```
input.mp4
   │
   ▼
cv2.VideoCapture ──► frame (BGR, uint8, W×H)
   │
   ├─► detect_faces()        per-frame boxes, pixel coords (x0,y0,x1,y1)
   │
   ├─► expand_box()          grow each box by --margin on every side, clamp to frame
   │
   ├─► match_detections_to_tracks()   greedy nearest-centroid → track key per box
   │
   ├─► SmoothedBox.update()  EMA on (cx, cy, w, h) per track → smoothed box
   │
   ├─► blur_region()         Gaussian blur or pixelate, rect or feathered ellipse, in place
   │
   └─► cv2.VideoWriter (mp4v) ──► <output>.noaudio.mp4
                                        │
                                        ▼
                      ffmpeg: copy video, AAC audio from input ──► output.mp4
```

The script runs top to bottom in `process_video()` (line 291). `main()` (line 406) only parses arguments and calls it.

## 3. Components

### 3.1 Model management (lines 42–95)

- **MediaPipe BlazeFace** (`.tflite`): `short` or `full` range, downloaded from Google's model storage by `ensure_model()`.
- **YOLOv8-face** (`.pt`): sizes `n/s/m/l/x` from the [lindevs/yolov8-face](https://github.com/lindevs/yolov8-face) GitHub releases (trained on WIDER FACE), downloaded by `ensure_yolo_model()`.
- Weights are saved **next to the script** (`MODEL_DIR`), not in a cache folder. They're gitignored.
- "Already downloaded" is checked only by file size (>100 KB for tflite, >1 MB for pt). There's no checksum, and a partial download larger than the threshold is treated as valid.
- URLs point at `releases/latest`, so the model can change underneath us. lindevs publishes SHA-256 checksums per asset. Pinning a release and verifying the checksum would make runs reproducible.
- lindevs also publishes **ONNX** versions of every size, which matters for issue #11.

### 3.2 Detector backends (`make_detector`, line 204; `detect_faces`, line 252)

A detector is a tuple `(backend_name, impl)`. `detect_faces()` branches on the name and always returns a list of `(x0, y0, x1, y1)` integer pixel boxes.

| Backend | impl | Confidence | `--imgsz` | Notes |
|---|---|---|---|---|
| `yolo` (default) | `(ultralytics.YOLO, device)` | per call via `predict(conf=...)` | yes | Best recall and fewest false positives. Device auto-selects `cuda` if `torch.cuda.is_available()`, else `cpu`, and is printed at startup. |
| `mediapipe` | `vision.FaceDetector` (Tasks API) | set once at construction | no | Fast on CPU but produces false positives on textures (grass, fabric) and drops off-angle faces. Uses the Tasks API; the old `mp.solutions` API is gone in current releases. |
| `haar` | `cv2.CascadeClassifier` | ignored | no | No download. Only for testing the rest of the pipeline. |

Details that matter:

- **Confidence scores are discarded.** Only box coordinates leave `detect_faces()`. The review report (#8) and the JSON pipeline (#4) need the score kept.
- **`--imgsz`:** Ultralytics resizes the frame's longer side to `imgsz` before inference (letterboxed, multiple of 32), whatever the source resolution. At the default 640, a 1080×1920 frame is shrunk about 3×, so distant faces become a few pixels and are lost. Raising it to 1280 recovers them at roughly 3–4× the compute and much more VRAM.
- **`--imgsz` is not monotonic.** On real footage, 640 caught some faces at frame edges that 1280 missed, and vice versa. Never assume a higher value is strictly better; benchmark both.
- `cvtColor` to RGB runs on every frame, but only MediaPipe uses the RGB copy (Ultralytics takes BGR numpy arrays directly). It's cheap but wasted for YOLO.
- Inference runs one frame at a time in FP32. No batching and no `half=True`.

### 3.3 Box expansion (`expand_box`, line 98)

Grows each detection by `margin × width` horizontally and `margin × height` vertically **on each side**, then clamps to the frame. A margin of 0.45 makes the box about 1.9× the detected size in each dimension.

YOLO face boxes are tight (roughly brow to chin). Without margin, hairlines, ears and the edges of profile faces stay visible. 0.45 is the validated value for max-accuracy runs.

Expansion happens **before** tracking and smoothing, so the tracker and smoother work on expanded boxes.

### 3.4 Tracking (`match_detections_to_tracks`, line 175)

There's no real tracker, by design. The "tracks" exist only to give each face its own smoothing state.

- Each frame, every detection is matched greedily, in detection order, to the nearest unused existing track whose smoothed centroid is within `max_dist = 0.08 × max(W, H)` pixels. That's about 154 px on a 1920-pixel side.
- Unmatched detections start a new track.
- **Any track not matched in the current frame is deleted immediately** (lines 348–351). There's no hold, no coasting and no re-identification.

Consequences:

- Hard cuts are handled automatically: everything unmatched disappears, and new faces start fresh.
- **A single missed detection means that face is unblurred for that frame**, and its smoothing restarts from scratch on the next detection. This is the main recall gap. Gap filling (#9) is the planned fix.
- Greedy matching in detection order can swap tracks when two faces cross. That only affects smoothing, not coverage.

### 3.5 Smoothing (`SmoothedBox`, line 151)

An exponential moving average over `(cx, cy, w, h)`:

```
state = alpha × new + (1 − alpha) × state
```

`--smoothing` is `alpha`. **Higher means more responsive and more jitter; lower means smoother and more lag.** This is easy to get backwards, and the flag name doesn't help.

The blur is drawn at the **smoothed** box, not the raw detection. On fast motion the smoothed box trails the face and can leave part of it uncovered, which the margin partly hides. Any future smoother must keep the blur at least as large as the raw detection, for example by using the union of the raw and smoothed boxes.

**Tried and reverted:** a velocity-aware (predictive) smoother. It extrapolated box motion forward and, on real footage, pulled boxes off faces at frame edges, leaving them unblurred. It looked fine on synthetic tests. Lesson: causal prediction is risky. Prefer non-causal, two-pass approaches (#4, #9) and validate on real footage.

### 3.6 Blur rendering (`blur_region`, line 109)

Runs in place on the frame, once per smoothed box.

- **Gaussian (default):** kernel size `strength | 1` (forced odd), sigma derived by OpenCV. The kernel is a **fixed pixel size regardless of face size**, so a strength that hides a large face can look lighter on a 2000 px close-up. Strength 80 is the validated heavy setting.
- **Pixelate (`--pixelate`):** downscale the region to about 1/12 size, then upscale with nearest-neighbour. **`--strength` is ignored** in this mode, and the block size scales with the box.
- **Shape `rect`:** replace the whole box.
- **Shape `circle`:** draw a filled ellipse inscribed in the box, feather its edge with a Gaussian (kernel ≈ min(w,h)/20), and alpha-blend the effect over the original. The corners of the box stay unblurred, so the ellipse covers less area than `rect` at the same margin. Pair it with a larger margin. Not yet validated on real footage.
- Overlapping boxes are processed one after another, so overlaps get blurred twice. That's harmless.
- The blur only samples pixels inside the box (OpenCV reflects at the box border), so the blur can't pick up detail from outside the face region.

### 3.7 Output and audio mux (lines 306–403)

1. Frames go to `cv2.VideoWriter` with fourcc **`mp4v`** (MPEG-4 Part 2) at the source fps, to `<output>.noaudio.mp4`.
2. If `ffmpeg` is on PATH, it runs:
   `ffmpeg -y -i tmp -i input -c:v copy -c:a aac -map 0:v:0 -map 1:a:0? -shortest output`
   - The video stream is **copied**, so the final file keeps the lossy `mp4v` encode. This is a known quality bug (#5).
   - Only the **first audio track** is kept and always re-encoded to AAC. Other audio tracks, subtitles, timecode, rotation and color metadata are dropped.
   - `1:a:0?` makes audio optional, so inputs without audio still work.
3. If ffmpeg is missing, the silent temp file is copied to the output path with only a warning. If the mux fails, the same fallback happens and the temp file is left behind.

Other I/O assumptions:

- `fps` falls back to 30 if OpenCV reports 0. Variable-frame-rate sources are treated as constant, so audio can drift on long VFR clips.
- `total_frames` comes from container metadata and can be approximate, so the progress percentage can be slightly off.
- Rotation metadata on phone footage isn't explicitly handled; it depends on OpenCV's backend.

### 3.8 Logging and timing

- Startup: model download notices and `YOLOv8-face running on: cuda|cpu` on stderr.
- Progress line every 100 frames on stderr: `N/total frames (pct) -- Xm Ys elapsed`.
- End: `Done: <path>` and `Total time: ...` on stdout.
- `process_video()` calls `sys.exit(1)` if the input can't be opened, which is unfriendly for library use. Raise instead when refactoring.

## 4. CLI reference (current defaults)

| Flag | Default | Affects | Notes |
|---|---|---|---|
| `input`, `output` | required | | Output container is inferred by ffmpeg from the extension. |
| `--backend` | `yolo` | detection | `yolo`, `mediapipe`, `haar` |
| `--yolo-size` | `m` | detection | `n s m l x`; `x` ≈ 130 MB weights, best recall |
| `--imgsz` | `640` | detection (YOLO) | Use multiples of 32. 1280 is the validated high-recall value. |
| `--confidence` | `0.5` | detection | Lower catches more off-angle faces. 0.25 is the validated high-recall value. |
| `--device` | auto | detection (YOLO) | `cuda`, `cuda:0`, `cpu` |
| `--model` | `full` | detection (MediaPipe only) | `full`, `short` |
| `--margin` | `0.25` | box | Fraction added per side. 0.45 validated. |
| `--smoothing` | `0.5` | box | EMA alpha. Higher is snappier. |
| `--strength` | `35` | render | Gaussian kernel px. 80 validated. Ignored with `--pixelate`. |
| `--pixelate` | off | render | Mosaic instead of blur |
| `--shape` | `rect` | render | `rect`, `circle` |

**Validated max-accuracy combination:**

```
--strength 80 --margin 0.45 --confidence 0.25 --smoothing 0.5 --yolo-size x --imgsz 1280
```

## 5. Performance reference

Relative timings for the same short 1080×1920 clip on an 8 GB consumer NVIDIA GPU (RTX 30-series class):

| Configuration | Time |
|---|---|
| YOLO `m`, 640, CPU | ~3:00 |
| YOLO `m`, 640, GPU | ~1:05 |
| YOLO `x`, 1280, GPU | ~4:00 |
| MediaPipe, CPU | fastest, but too many false positives |

VRAM: `x` at 1280 fits on 8 GB, but other apps (browsers, chat, game launchers) can hold several GB. There's no out-of-memory handling yet; a CUDA OOM crashes the run.

Where the time goes: YOLO inference dominates, then decode/encode. Batching frames and FP16 are the obvious speedups, but both must be checked against the recall benchmark (#2).

## 6. Environment and dependencies

| Dependency | Role | Notes |
|---|---|---|
| Python **3.12** | runtime | CUDA builds of PyTorch weren't available for 3.14 when this was written. Newer Pythons silently get CPU-only torch. |
| `torch` (CUDA build) | YOLO inference | Install from the index URL shown at pytorch.org/get-started/locally. The version string must end in `+cuXXX`, and `torch.cuda.is_available()` must be `True`. Retired CUDA index tags make pip install a `+cpu` build **with no error**. |
| `ultralytics` | YOLO wrapper | `YOLO(path).predict(...)` |
| `opencv-python` | decode, image ops, encode | |
| `numpy` | arrays | |
| `mediapipe` | optional backend | Imported lazily |
| `ffmpeg` binary | audio mux | Must be on PATH |

There's no `requirements.txt` or `pyproject.toml` yet; install steps live in `docs/tutorial.md`. Imports for optional backends are lazy, so `haar` runs with only OpenCV and numpy.

**Dependency licensing.** face-be-gone's own code is MIT. It imports `ultralytics`, which is **AGPL-3.0**. Running the script and using its output is unaffected. A distributed bundle that includes Ultralytics (installer, packaged GUI, Docker image), or a modified version run as a network service, must meet the AGPL's terms for that bundle: the corresponding source is available, which a public repo already covers. MIT code can be combined with AGPL code, so nothing here blocks the roadmap. The ONNX Runtime path (#11) would remove the Ultralytics dependency entirely if a fully permissive bundle is ever wanted; check the face weights' own license before distributing them.

## 7. Known limitations

| Limitation | Effect | Tracked in |
|---|---|---|
| Tracks deleted on first missed frame | Single-frame exposures; smoothing restarts | #9 |
| Backward-only EMA | Box trails fast-moving faces | #4, #9 |
| Blur at the smoothed box, not the union with the raw box | Fast motion can uncover part of a face | #9 |
| Confidence scores discarded | No way to flag near misses | #4, #8 |
| `mp4v` intermediate copied into the output | Quality loss and larger files | #5 |
| Metadata, extra audio tracks and subtitles dropped | Rotation, color and VFR issues | #5 |
| Missing ffmpeg only warns | Silent deliverable | #5 |
| `-shortest` stream-copy mux drops the last frame on ffmpeg 9+ | One frame lost per clip with current Windows/macOS ffmpeg builds (6.1 and 7.0 unaffected) | #5 |
| No OOM handling | Crash at high `--imgsz` / `x` with low free VRAM | |
| Unpinned model URLs, size-only check | Non-reproducible or corrupt weights | |
| No recall benchmark; CI can't test detection accuracy | Detection regressions judged by eye | #2 |
| Single-file script, `sys.exit` inside logic | Hard to reuse from a GUI | #4 |

## 8. Target architecture

Where the roadmap issues lead. The direction: split the script into a small package where the CLI, the GUI and the benchmark all call the same functions.

```
face_be_gone/
  __init__.py        __version__
  cli.py             argparse, presets (#6), batch mode (#7), --doctor (#3)
  doctor.py          environment checks (#3)
  models.py          download, checksum, cache dir
  detect/
    base.py          Detector interface: detect(frame_bgr) -> list[Detection]
    yolo.py          ultralytics backend
    onnx.py          onnxruntime backend (#11)
    mediapipe.py
    haar.py
  detections.py      Detection dataclass, JSON read/write (#4)
  track.py           linking, gap filling, cut detection (#9)
  smooth.py          EMA (current) and forward-backward smoothing (#9)
  review.py          flagged-moment analysis, CSV/EDL/FCPXML export (#8)
  render.py          blur_region, shapes, matte output
  io.py              decode, ffmpeg encode pipe and mux (#5)
  gui/               FastAPI app + static frontend (#10)
bench/
  run.py             recall benchmark (#2); data/ is gitignored
tests/
```

### Two-pass data flow (#4)

```
detect:  video ──► Detector ──► faces.json   (raw boxes + confidences, no margin, no smoothing)
review:  faces.json ──► review.csv / markers  (#8)
edit:    GUI adds/removes boxes ──► faces.json (manual entries flagged as such)
render:  faces.json ──► track/gap-fill/smooth ──► expand ──► blur ──► ffmpeg encode + mux
```

Rules:

- `faces.json` stores **raw** detections. Margin, smoothing, gap filling and blur are all render-time settings, so changing them never needs re-detection.
- The final blur region for each face in each frame must **contain the raw detection** (after margin). Smoothing and interpolation may only add coverage.
- Never interpolate across a detected hard cut.
- Detection settings (backend, model and its checksum, `imgsz`, confidence) are stored in the JSON. Render re-detects if they don't match the request.

### Draft `faces.json` schema

```json
{
  "schema": 1,
  "video": { "path": "input.mp4", "width": 1080, "height": 1920, "fps": 29.97, "frames": 1800 },
  "detector": { "backend": "yolo", "model": "yolov8x-face-lindevs", "sha256": "…", "imgsz": 1280, "confidence": 0.25 },
  "cuts": [0, 412, 977],
  "frames": [
    { "i": 0, "faces": [ { "box": [512, 300, 640, 452], "conf": 0.91, "src": "det" } ] },
    { "i": 1, "faces": [ { "box": [515, 302, 643, 455], "conf": 0.42, "src": "det" },
                         { "box": [80, 1200, 130, 1262], "conf": 1.0, "src": "manual" } ] }
  ]
}
```

`src` distinguishes detector output from manual GUI edits (`manual`) and interpolated boxes (`interp`). Manual removals can be stored as `"src": "suppress"` regions.

## 9. Testing strategy

**What exists** (`tests/`, run by `.github/workflows/ci.yml` on Linux, Windows and macOS with Python 3.12):

- `test_units.py`: `expand_box` clamping, `blur_region` never touches pixels outside its box (every shape and mode), circle corners stay untouched, `SmoothedBox`, matching and `format_elapsed`.
- `test_pipeline.py`: end-to-end runs on clips generated by ffmpeg in `conftest.py`. One test runs the real Haar detector; the rest use `stub_run.py`, which swaps in a detector that finds nothing or a fixed box, so only I/O and rendering are exercised. It checks frame counts, rotation, blur placement, quality against the source (PSNR), A/V sync on variable frame rate (flash and beep markers every 6 s), color range, audio tracks and the missing-ffmpeg path. Known bugs are `xfail(strict=True)` and name their issue.
- CI pins `opencv-python-headless<5`, because OpenCV 5 dropped `cv2.CascadeClassifier` and with it the Haar backend.
- CI has no GPU and doesn't install PyTorch, so the YOLO and MediaPipe backends aren't exercised there.

**Still to add:**

- **Recall benchmark (#2)** is the gate for any change to detection, tracking, smoothing, margin or rendering geometry. PRs report before/after recall per category: edge, small/distant, off-angle, child, occluded.
- **Real footage over synthetic.** Synthetic tests missed the velocity-smoother regression. Use them for unit tests of geometry, not as evidence of recall.
- **Unit tests** as features land: JSON round-trip (#4), gap filling never crosses cuts (#9), the smoothed region always contains the raw box.
- Benchmark and review data stay local and gitignored. Never commit real footage (see CLAUDE.md).

## 10. Decision log

| Decision | Why |
|---|---|
| Per-frame detection, not an NLE tracker | NLE trackers reset at every cut in precut footage; per-frame detection has nothing to reset. |
| YOLOv8-face over MediaPipe as default | MediaPipe produced frequent false positives (textures, duplicate boxes) and dropped off-angle faces. |
| `--imgsz` opt-in, default 640 | 1280 recovers distant faces but costs about 3–4× per frame on every clip; not every clip needs it. |
| Plain EMA, velocity smoother reverted | Predictive smoothing dropped faces at frame edges on real footage. |
| Multi-scale / tiled detection reverted | Merging detections from several scales degraded edge coverage on real footage. Revisit only behind the benchmark. |
| Python 3.12 | Reliable CUDA PyTorch wheels; 3.14 silently fell back to CPU. |
| Install command taken from pytorch.org, never hardcoded | A hardcoded `cu124` index went stale and silently installed CPU builds. |
| MIT license (replaced CC BY-NC 4.0) | The goal is to share the tool with anyone, commercial use included. CC licenses aren't designed for software, and the non-commercial clause conflicted with AGPL dependencies for packaged releases. |
