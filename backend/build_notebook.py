"""Regenerates colab/Sketch2Shape_Hunyuan3D.ipynb from backend/server.py.

Run:  python3 backend/build_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
server_src = (ROOT / "backend" / "server.py").read_text()


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


cells = [
    md("""
# Sketch2Shape AI — Hunyuan3D-2 GPU backend (Google Colab)

TechZephyr 4.0 · IIT Bhubaneswar · *From a 2D Sketch to a 3D Asset*

This notebook turns Colab into the GPU backend for the Sketch2Shape web app:

`sketch → SD1.5 + ControlNet Scribble (concept render) → background removal → Hunyuan3D-DiT (shape) → Hunyuan3D-Paint (texture) → trimesh validation → .glb / .obj`

**Before you start:** `Runtime → Change runtime type → GPU`.

| GPU | VRAM | Recommended shape model | Texture |
|---|---|---|---|
| T4 (free) | 15 GB | `Hunyuan3D-2mini` turbo (default) | yes, low-VRAM offload (slow) |
| L4 | 22 GB | `Hunyuan3D-2` turbo | yes |
| A100 | 40 GB | `Hunyuan3D-2` (full, 30 steps) | yes |

Run the cells top to bottom. The first run takes roughly 15–25 minutes (installs, CUDA compilation, about 15 GB of model downloads).
"""),
    md("## 0 · Check the GPU\nRuns in a subprocess on purpose: importing torch/numpy in this notebook before the installs leaves old modules in memory and causes mixed-version errors later."),
    code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
!python -c "import sys, torch; print('python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.version.cuda, '| gpu', torch.cuda.is_available())"
"""),
    md("## 1 · Clone and install Hunyuan3D-2\nColab already ships PyTorch with CUDA, so we keep its build and only install the repo requirements."),
    code("""
%cd /content
!git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git
%cd /content/Hunyuan3D-2
!pip install -q -r requirements.txt
!pip install -q -e .
"""),
    md("""
**Expected warning:** pip may print `dependency conflicts` for preinstalled Colab packages (`pytensor`, `cudf`, `cuml`, `jax`…). This project does not use them, so it is safe to ignore.

**Do not pin `numpy<2`, `numba<0.65` or Pillow.** On Colab's Python 3.13, rembg needs numpy>=2.3 and OpenCV 5 is built against numpy 2.
"""),
    md("## 2 · Extra dependencies (ControlNet, API server, mesh tools)\n`transformers` is pinned below 5: v5 renamed the DINOv2 layers that the Hunyuan3D-2 checkpoints expect."),
    code("""
!pip install -q "diffusers>=0.30.0" "transformers>=4.46,<5" "accelerate>=0.33.0" \\
    fastapi uvicorn python-multipart "trimesh>=4.0" rembg opencv-python-headless \\
    pygltflib xatlas pymeshlab ninja pybind11
"""),
    md("## 3 · Compile the texture rasterizer (CUDA, ~5–10 min)\nRequired for Hunyuan3D-Paint. If this fails, shape generation still works and the app exports untextured meshes."),
    code("""
%cd /content/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer
!pip install -q --no-build-isolation . || python3 setup.py install
%cd /content/Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer
!pip install -q --no-build-isolation . || python3 setup.py install
%cd /content/Hunyuan3D-2
# Newer diffusers require trust_remote_code=True for Hunyuan's local paint pipeline
!sed -i 's/custom_pipeline=custom_pipeline_path, torch_dtype=torch.float16)/custom_pipeline=custom_pipeline_path, torch_dtype=torch.float16, trust_remote_code=True)/' hy3dgen/texgen/utils/multiview_utils.py
!grep -c "trust_remote_code=True" hy3dgen/texgen/utils/multiview_utils.py
"""),
    md("""
## 3b · Restart the session (required)
The installs above upgraded numpy, Pillow and others on disk. This cell restarts the Python process so every later cell loads one consistent set of versions.
Colab will show *"Your session crashed"*. **That is expected.** Continue with step 3c. Installed packages are kept.
"""),
    code("""
import os
os.kill(os.getpid(), 9)
"""),
    md("## 3c · Verify the environment (after the restart)"),
    code("""
import numpy, numba, scipy.linalg, PIL, cv2, torch, diffusers, transformers
assert int(transformers.__version__.split(".")[0]) < 5, 'transformers>=5 breaks Hunyuan3D-2: pip install "transformers>=4.46,<5" and restart'
from diffusers import StableDiffusionControlNetPipeline  # noqa: F401
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # noqa: F401
from hy3dgen.rembg import BackgroundRemover  # noqa: F401
print("OK | numpy", numpy.__version__, "| numba", numba.__version__, "| pillow", PIL.__version__,
      "| cv2", cv2.__version__, "| torch", torch.__version__, "| diffusers", diffusers.__version__, "| transformers", transformers.__version__)
try:
    import custom_rasterizer  # noqa: F401
    print("custom_rasterizer OK - texture painting enabled")
except Exception as e:
    print("custom_rasterizer NOT available - shapes will be untextured:", e)
"""),
    md("## 4 · Write the Sketch2Shape API server\nSame file as `backend/server.py` in the project repo."),
    code("%%writefile /content/s2s_server.py\n" + server_src),
    md("""
## 5 · Choose models and warm up
Downloads every model once so the first request from the judges is fast.

**Free Colab (12.7 GB RAM):** the server auto-enables *sequential mode*. Each model is loaded straight onto the GPU,
used for its stage, then freed. This keeps peak RAM to one model at a time. Each request reloads models from the disk
cache, which adds roughly 20–60 s. With Colab Pro High-RAM (more than 20 GB), models stay resident and requests are faster.

If an earlier attempt crashed, use `Runtime → Restart session` first, then run from step 4.
On an L4 or A100, switch to the full `Hunyuan3D-2` turbo model for better geometry.
"""),
    code((ROOT / "backend" / "colab_warmup_cell.py").read_text()),
    md("## 6 · Start the API and open a public tunnel\nUses a Cloudflare quick tunnel (no account needed). Copy the printed URL into the web app under **Connect Colab GPU**."),
    code("""
import re, subprocess, sys, threading, time
for p in ("/content", "/content/Hunyuan3D-2"):
    if p not in sys.path:
        sys.path.insert(0, p)
import s2s_server  # same module instance step 5 warmed up
s2s_server.start_in_background(8000)

!wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /content/cloudflared
!chmod +x /content/cloudflared

tunnel = subprocess.Popen(
    ["/content/cloudflared", "tunnel", "--url", "http://localhost:8000", "--no-autoupdate"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)
PUBLIC_URL, t0 = None, time.time()
while time.time() - t0 < 90 and PUBLIC_URL is None:
    m = re.search(r"https://[a-z0-9-]+\\.trycloudflare\\.com", tunnel.stdout.readline())
    if m:
        PUBLIC_URL = m.group(0)
# keep draining logs so the tunnel never blocks on a full pipe
threading.Thread(target=lambda: [None for _ in tunnel.stdout], daemon=True).start()
print("\\n=========================================")
print(" Paste this into Sketch2Shape:", PUBLIC_URL)
print("=========================================")
"""),
    md("## 7 · Smoke test and benchmark\nRuns a synthetic sketch through the full pipeline and prints per-stage timings for your README."),
    code("""
import io, json, requests
from PIL import Image, ImageDraw

img = Image.new("RGB", (512, 512), "white")
d = ImageDraw.Draw(img)
d.rounded_rectangle([170, 150, 340, 400], radius=20, outline="black", width=6)  # mug body
d.arc([300, 210, 400, 330], start=-90, end=90, fill="black", width=6)            # handle
d.ellipse([170, 130, 340, 170], outline="black", width=6)                        # rim
buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)

base = "http://localhost:8000"
print(requests.get(f"{base}/health").json())
job = requests.post(f"{base}/jobs", files={"image": ("sketch.png", buf, "image/png")},
                    data={"prompt": "a glossy blue ceramic coffee mug", "options": json.dumps({"texture": True})}).json()
while True:
    s = requests.get(f"{base}/jobs/{job['job_id']}").json()
    print(f"\\r{s['status']:8s} {s['stage'] or '':10s} {s['message'][:60]:60s}", end="")
    if s["status"] in ("done", "error"):
        break
    time.sleep(2)
print("\\n", json.dumps({k: s[k] for k in ("status", "error", "timings", "total_seconds", "stats")}, indent=2))
"""),
    md("## 8 · (Optional) Download all generated assets"),
    code("""
!cd /content && zip -qr s2s_outputs.zip s2s_outputs
from google.colab import files
files.download("/content/s2s_outputs.zip")
"""),
    md("""
## Troubleshooting
- **Session crashes / "used all available RAM"**: make sure the warm-up log shows `sequential=True`. Restart the session before re-running, and do not run step 7 while another cell is loading models.
- **`numpy.core.multiarray failed to import` / OpenCV ABI errors**: numpy was downgraded. Run `!pip install -q --upgrade "numpy>=2.3,<2.6" "numba>=0.62"` and restart the session.
- **`cannot import name '_Ink' from 'PIL._typing'`** (Pillow files from mixed versions): `!pip uninstall -y pillow`, `!rm -rf /usr/local/lib/python3.13/dist-packages/PIL`, `!pip install --no-cache-dir "pillow==11.3.0"`, then restart the session.
- **`_multiarray_umath has no attribute '_blas_supports_fpe'`** or other numpy internals errors (numpy files from mixed versions): `!pip uninstall -y numpy` (twice), `!rm -rf /usr/local/lib/python3.13/dist-packages/numpy*`, `!pip install --no-cache-dir "numpy>=2.3,<2.6" "numba>=0.62"`, then restart the session. If more packages break, use `Runtime → Disconnect and delete runtime` and run the notebook from the top with no extra pins.
- **`Missing key(s) in state_dict ... main_image_encoder`**: transformers 5.x is installed. `!pip install -q "transformers>=4.46,<5"`, then restart the session.
- **`contains custom code ... Pass trust_remote_code=True`**: re-run step 4. The server patches Hunyuan's paint loader automatically.
- **Always restart the session after any `pip install` that changes versions**. Otherwise old modules stay loaded in memory.
- **CUDA out of memory on T4**: turn off *Paint texture* in the app, lower *Octree* to 196, or switch to an L4 runtime.
- **Tunnel URL changed**: quick tunnels are temporary. Re-run step 6 and paste the new URL.
- **`custom_rasterizer` import error after `setup.py install`**: `Runtime → Restart session`, then re-run from step 4 (installs persist).
- **Colab disconnects when idle**: keep the tab open during judging; models must reload after a disconnect (re-run steps 4–6).
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
out = ROOT / "colab" / "Sketch2Shape_Hunyuan3D.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
