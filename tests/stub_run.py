"""Run the script with its detector replaced, so tests can check the video
I/O path without a model or real faces.

    python stub_run.py <script> none <script args...>   # never finds a face
    python stub_run.py <script> box  <script args...>   # one fixed box per frame
"""

import importlib.util
import sys

script, mode, *args = sys.argv[1:]
spec = importlib.util.spec_from_file_location("face_be_gone_script", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

module.make_detector = lambda *a, **k: ("stub", None)
if mode == "none":
    module.detect_faces = lambda *a, **k: []
elif mode == "box":
    # Upper-left area: x from 1/8 to 3/8 of the width, y from 1/8 to 3/8 of the height
    module.detect_faces = lambda det, rgb, bgr, w, h, **k: [(w // 8, h // 8, 3 * w // 8, 3 * h // 8)]
else:
    sys.exit(f"unknown stub mode: {mode}")

sys.argv = [script, *args]
module.main()
