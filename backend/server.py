"""Sketch2Shape AI - Google Colab GPU backend.

Pipeline (one job at a time on the GPU):
  preprocess -> concept (SD1.5 + ControlNet Scribble) -> segment (rembg)
  -> shape (Hunyuan3D-DiT) -> texture (Hunyuan3D-Paint) -> export (trimesh)

Run inside Colab after installing Hunyuan3D-2 (see colab/Sketch2Shape_Hunyuan3D.ipynb):
    python server.py            # or import and call start_in_background()

Environment overrides:
    S2S_SHAPE_MODEL      default tencent/Hunyuan3D-2mini
    S2S_SHAPE_SUBFOLDER  default hunyuan3d-dit-v2-mini-turbo
    S2S_TEX_MODEL        default tencent/Hunyuan3D-2
    S2S_SD_MODEL         default Lykon/dreamshaper-8
    S2S_CN_MODEL         default lllyasviel/control_v11p_sd15_scribble
    S2S_ENABLE_TEXTURE   1/0 (default 1)
    S2S_SEQUENTIAL       1/0 force load-run-unload per stage (default: auto when system RAM < 20 GB)
    S2S_OUTPUT_DIR       default /content/s2s_outputs
    S2S_PORT             default 8000
"""

from __future__ import annotations

import gc
import json
import os
import queue
import threading
import time
import traceback
import uuid
import zipfile
from pathlib import Path

import numpy as np
import torch
import trimesh
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, ImageFilter, ImageOps

VERSION = "1.0.0"
SHAPE_MODEL = os.environ.get("S2S_SHAPE_MODEL", "tencent/Hunyuan3D-2mini")
SHAPE_SUBFOLDER = os.environ.get("S2S_SHAPE_SUBFOLDER", "hunyuan3d-dit-v2-mini-turbo")
TEX_MODEL = os.environ.get("S2S_TEX_MODEL", "tencent/Hunyuan3D-2")
SD_MODEL = os.environ.get("S2S_SD_MODEL", "Lykon/dreamshaper-8")
CN_MODEL = os.environ.get("S2S_CN_MODEL", "lllyasviel/control_v11p_sd15_scribble")
ENABLE_TEXTURE = os.environ.get("S2S_ENABLE_TEXTURE", "1") == "1"
OUT_DIR = Path(os.environ.get("S2S_OUTPUT_DIR", "/content/s2s_outputs"))
PORT = int(os.environ.get("S2S_PORT", "8000"))

OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1024**3 if DEVICE == "cuda" else None
GPU_NAME = torch.cuda.get_device_name(0) if DEVICE == "cuda" else None
# Below ~20 GB (T4 / L4-16) keep only the active model on the GPU.
LOW_VRAM = VRAM_GB is None or VRAM_GB < 20


def _ram_gb() -> float | None:
    try:
        import psutil

        return psutil.virtual_memory().total / 1024**3
    except Exception:
        return None


def _ram_used_gb() -> float | None:
    try:
        import psutil

        return psutil.virtual_memory().used / 1024**3
    except Exception:
        return None


RAM_GB = _ram_gb()
# Free Colab has ~12.7 GB system RAM: CPU offload would park every model in RAM
# and crash the runtime. Sequential mode loads each model straight onto the GPU,
# runs its stage and frees it before the next stage.
_seq_env = os.environ.get("S2S_SEQUENTIAL")
SEQUENTIAL = (_seq_env == "1") if _seq_env in ("0", "1") else (RAM_GB is None or RAM_GB < 20)

NEGATIVE = (
    "multiple objects, cropped, cut off, text, watermark, busy background, scene, floor, "
    "harsh shadow, low quality, blurry, deformed, extra parts"
)
STAGES = ["preprocess", "concept", "segment", "shape", "texture", "export"]
ARTIFACT_FILES = {
    "control": "control.png",
    "concept": "concept.png",
    "segmented": "segmented.png",
    "glb": "model.glb",
    "obj_zip": "model_obj.zip",
}


# --------------------------------------------------------------------------- models
class Models:
    """Lazily loaded model registry with VRAM-aware placement."""

    def __init__(self) -> None:
        self.sd = None
        self.rembg = None
        self.shape = None
        self.paint = None
        self.lock = threading.Lock()

    def flush(self) -> None:
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    def get_sd(self):
        if self.sd is None:
            from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

            dtype = torch.float16 if DEVICE == "cuda" else torch.float32
            cn = ControlNetModel.from_pretrained(CN_MODEL, torch_dtype=dtype)
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                SD_MODEL, controlnet=cn, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
            )
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            if DEVICE == "cuda" and LOW_VRAM and not SEQUENTIAL:
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(DEVICE)
            self.sd = pipe
        return self.sd

    def get_rembg(self):
        if self.rembg is None:
            from hy3dgen.rembg import BackgroundRemover

            self.rembg = BackgroundRemover()
        return self.rembg

    def get_shape(self):
        if self.shape is None:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

            check_transformers()
            try:
                if DEVICE == "cuda" and SEQUENTIAL:
                    pipe = load_shape_low_ram()
                else:
                    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                        SHAPE_MODEL, subfolder=SHAPE_SUBFOLDER, use_safetensors=True, device=DEVICE
                    )
            except FileNotFoundError as exc:
                # Only fall back when the .safetensors file is missing. A key
                # mismatch would fail identically with .ckpt, which is also
                # loaded fully into RAM.
                print("[s2s] safetensors weights missing, falling back to ckpt:", exc)
                pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                    SHAPE_MODEL, subfolder=SHAPE_SUBFOLDER, use_safetensors=False, device=DEVICE
                )
            except RuntimeError as exc:
                if "Missing key(s)" in str(exc):
                    raise RuntimeError(
                        "Hunyuan3D image-encoder weights do not match the installed transformers "
                        'version. Run: pip install "transformers>=4.46,<5", then restart the session.'
                    ) from None
                raise
            try:
                pipe.enable_flashvdm()  # faster volume decoding (turbo models)
            except Exception as exc:  # older checkouts
                print("[s2s] FlashVDM unavailable:", exc)
            self.shape = pipe
        return self.shape

    def get_paint(self):
        if not ENABLE_TEXTURE:
            return None
        if self.paint is None:
            patch_hunyuan_paint()
            from hy3dgen.texgen import Hunyuan3DPaintPipeline

            try:
                pipe = Hunyuan3DPaintPipeline.from_pretrained(TEX_MODEL, subfolder="hunyuan3d-paint-v2-0-turbo")
            except TypeError:
                pipe = Hunyuan3DPaintPipeline.from_pretrained(TEX_MODEL)
            if LOW_VRAM and not SEQUENTIAL and hasattr(pipe, "enable_model_cpu_offload"):
                pipe.enable_model_cpu_offload()
            self.paint = pipe
        return self.paint

    def release(self, name: str) -> None:
        """Drop a model completely (frees both GPU memory and system RAM)."""
        if getattr(self, name, None) is not None:
            setattr(self, name, None)
            self.flush()
            print(f"[s2s] released {name} | RAM used {_ram_used_gb() or 0:.1f} GB")

    def release_if_sequential(self, name: str) -> None:
        if SEQUENTIAL:
            self.release(name)

    def loaded(self) -> dict:
        return {
            "controlnet": self.sd is not None,
            "rembg": self.rembg is not None,
            "shape": self.shape is not None,
            "texture": self.paint is not None,
        }


MODELS = Models()


def load_shape_low_ram():
    """RAM-frugal replacement for Hunyuan3DDiTPipeline.from_single_file.

    The stock loader reads the whole checkpoint into CPU RAM and builds every
    sub-model in fp32 on the CPU (~14 GB peak for 2mini). Here each tensor is
    streamed from disk straight to the GPU, and each sub-model is built on the
    GPU and cast to fp16 before the next one, so system RAM stays near baseline.
    """
    import yaml
    from safetensors import safe_open
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dgen.shapegen.pipelines import instantiate_from_config
    from hy3dgen.shapegen.utils import smart_load_model

    dtype = torch.float16
    config_path, ckpt_path = smart_load_model(
        SHAPE_MODEL, subfolder=SHAPE_SUBFOLDER, use_safetensors=True, variant="fp16"
    )
    ckpt_path = ckpt_path.replace(".ckpt", ".safetensors")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model file {ckpt_path} not found")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    gpu = torch.device("cuda")
    parts = {}
    with safe_open(ckpt_path, framework="pt", device="cuda") as sf:
        keys = list(sf.keys())
        for name, strict in (("model", True), ("vae", False), ("conditioner", True)):
            t0 = time.time()
            with gpu:  # parameters are allocated on the GPU, not in system RAM
                module = instantiate_from_config(config[name])
            prefix = name + "."
            state = {k[len(prefix):]: sf.get_tensor(k) for k in keys if k.startswith(prefix)}
            if state:
                module.load_state_dict(state, strict=strict)
            del state
            module.to(device=gpu, dtype=dtype)
            parts[name] = module
            gc.collect()
            torch.cuda.empty_cache()
            print(
                f"[s2s] shape/{name} on GPU in {time.time() - t0:.0f}s | "
                f"VRAM {torch.cuda.memory_allocated() / 1024**3:.1f} GB | RAM used {_ram_used_gb() or 0:.1f} GB"
            )

    return Hunyuan3DDiTFlowMatchingPipeline(
        vae=parts["vae"],
        model=parts["model"],
        conditioner=parts["conditioner"],
        scheduler=instantiate_from_config(config["scheduler"]),
        image_processor=instantiate_from_config(config["image_processor"]),
        device="cuda",
        dtype=dtype,
        # enable_flashvdm() reads these to locate the turbo VAE
        from_pretrained_kwargs=dict(
            model_path=SHAPE_MODEL,
            subfolder=SHAPE_SUBFOLDER,
            use_safetensors=True,
            variant="fp16",
            dtype=dtype,
            device="cuda",
        ),
    )


def check_transformers() -> None:
    """Hunyuan3D-2 checkpoints use transformers 4.x DINOv2 module names."""
    import transformers

    major = int(transformers.__version__.split(".")[0])
    if major >= 5:
        raise RuntimeError(
            f"transformers {transformers.__version__} renamed DINOv2 layers that Hunyuan3D-2 expects. "
            'Run: pip install "transformers>=4.46,<5", then restart the session.'
        )


def patch_hunyuan_paint() -> None:
    """Newer diffusers require trust_remote_code=True for Hunyuan's local custom
    paint pipeline. Patch the (editable-installed) source once, idempotently."""
    import importlib.util

    spec = importlib.util.find_spec("hy3dgen")
    if not spec or not spec.submodule_search_locations:
        return
    path = Path(list(spec.submodule_search_locations)[0]) / "texgen" / "utils" / "multiview_utils.py"
    if not path.exists():
        return
    src = path.read_text()
    old = "custom_pipeline=custom_pipeline_path, torch_dtype=torch.float16)"
    if "trust_remote_code" not in src and old in src:
        path.write_text(src.replace(old, old[:-1] + ", trust_remote_code=True)"))
        print("[s2s] patched", path, "with trust_remote_code=True")


patch_hunyuan_paint()


def place(pipe, device: str) -> None:
    """Move a Hunyuan pipeline between CPU/GPU when running in low-VRAM mode."""
    if SEQUENTIAL or not (LOW_VRAM and DEVICE == "cuda"):
        return
    try:
        pipe.to(device)
    except Exception as exc:
        print(f"[s2s] could not move pipeline to {device}:", exc)


def texture_available() -> bool:
    if not ENABLE_TEXTURE:
        return False
    try:
        import custom_rasterizer  # noqa: F401  compiled in the notebook

        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- pipeline
def make_control(sketch: Image.Image) -> Image.Image:
    """ControlNet Scribble expects white strokes on black."""
    gray = ImageOps.grayscale(sketch).resize((512, 512), Image.LANCZOS)
    inv = ImageOps.invert(gray).point(lambda v: 255 if v > 60 else 0)
    return inv.filter(ImageFilter.MaxFilter(3)).convert("RGB")


def build_prompt(user_prompt: str) -> str:
    subject = user_prompt.strip() or "a single object"
    return (
        f"{subject}, a single complete object, centered, isolated on a plain white background, "
        "product photography, soft studio lighting, three-quarter view, high detail, sharp focus"
    )


def mesh_stats(mesh: trimesh.Trimesh) -> dict:
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "extents": [float(x) for x in mesh.extents],
        "textured": getattr(mesh.visual, "kind", None) == "texture",
    }


def export_obj_zip(mesh: trimesh.Trimesh, job_dir: Path) -> Path:
    zip_path = job_dir / ARTIFACT_FILES["obj_zip"]
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        try:
            text, files = trimesh.exchange.obj.export_obj(
                mesh, include_texture=True, return_texture=True, mtl_name="model.mtl"
            )
            zf.writestr("model.obj", text)
            for name, data in files.items():
                zf.writestr(name, data)
        except Exception:
            zf.writestr("model.obj", trimesh.exchange.obj.export_obj(mesh, include_texture=False))
    return zip_path


def run_job(job: dict, sketch: Image.Image, prompt: str, opts: dict) -> None:
    job_dir = OUT_DIR / job["id"]
    t_start = time.time()

    def stage(name: str, message: str):
        job["stage"], job["message"] = name, message
        return time.time()

    def done(name: str, t0: float):
        job["timings"][name] = round(time.time() - t0, 2)

    # 1. preprocess
    t = stage("preprocess", "Normalising sketch")
    sketch = ImageOps.exif_transpose(sketch).convert("RGB")
    sketch.save(job_dir / "input.png")
    control = make_control(sketch)
    control.save(job_dir / ARTIFACT_FILES["control"])
    job["artifacts"]["control"] = f"/jobs/{job['id']}/files/control"
    done("preprocess", t)

    # 2. concept render (sketch-guided image generation)
    if opts.get("use_controlnet", True):
        t = stage("concept", "Rendering concept with ControlNet")
        sd = MODELS.get_sd()
        gen = torch.Generator(device="cpu").manual_seed(int(opts.get("seed", 1234)))
        concept = sd(
            build_prompt(prompt),
            image=control,
            negative_prompt=NEGATIVE,
            num_inference_steps=int(opts.get("concept_steps", 25)),
            guidance_scale=7.5,
            controlnet_conditioning_scale=float(opts.get("controlnet_scale", 0.9)),
            generator=gen,
        ).images[0]
        concept.save(job_dir / ARTIFACT_FILES["concept"])
        job["artifacts"]["concept"] = f"/jobs/{job['id']}/files/concept"
        done("concept", t)
        del sd
        MODELS.release_if_sequential("sd")
        MODELS.flush()
    else:
        job["stages_skipped"].append("concept")
        concept = sketch

    # 3. foreground isolation
    t = stage("segment", "Removing background")
    segmented = MODELS.get_rembg()(concept.convert("RGB"))
    segmented.save(job_dir / ARTIFACT_FILES["segmented"])
    job["artifacts"]["segmented"] = f"/jobs/{job['id']}/files/segmented"
    done("segment", t)

    # 4. shape generation
    t = stage("shape", "Generating geometry with Hunyuan3D-DiT")
    shape = MODELS.get_shape()
    place(shape, "cuda")
    steps = 5 if opts.get("turbo", True) else 30
    mesh = shape(
        image=segmented,
        num_inference_steps=steps,
        octree_resolution=int(opts.get("octree_resolution", 256)),
        num_chunks=20000,
        generator=torch.manual_seed(int(opts.get("seed", 1234))),
        output_type="trimesh",
    )[0]
    from hy3dgen.shapegen import DegenerateFaceRemover, FaceReducer, FloaterRemover

    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh, max_facenum=int(opts.get("max_faces", 40000)))
    if len(mesh.faces) == 0:
        raise RuntimeError("Shape model produced empty geometry - try a clearer sketch or a prompt")
    done("shape", t)
    place(shape, "cpu")
    del shape
    MODELS.release_if_sequential("shape")
    MODELS.flush()

    # 5. texture
    if opts.get("texture", True) and texture_available():
        t = stage("texture", "Painting texture with Hunyuan3D-Paint")
        try:
            mesh = MODELS.get_paint()(mesh, image=segmented)
            done("texture", t)
        except torch.cuda.OutOfMemoryError:
            MODELS.flush()
            job["stages_skipped"].append("texture")
            job["message"] = "Texture skipped: out of GPU memory"
        MODELS.release_if_sequential("paint")
        MODELS.flush()
    else:
        job["stages_skipped"].append("texture")

    # 6. validate + export
    t = stage("export", "Validating and exporting")
    glb_path = job_dir / ARTIFACT_FILES["glb"]
    mesh.export(glb_path)
    reloaded = trimesh.load(glb_path, force="mesh")
    if len(reloaded.faces) == 0:
        raise RuntimeError("Exported GLB has no faces")
    export_obj_zip(mesh, job_dir)
    job["artifacts"]["glb"] = f"/jobs/{job['id']}/files/glb"
    job["artifacts"]["obj_zip"] = f"/jobs/{job['id']}/files/obj_zip"
    job["stats"] = mesh_stats(mesh)
    done("export", t)
    job["total_seconds"] = round(time.time() - t_start, 2)


# --------------------------------------------------------------------------- job queue
JOBS: dict[str, dict] = {}
WORK: "queue.Queue[tuple[str, Image.Image, str, dict]]" = queue.Queue()


def worker() -> None:
    while True:
        job_id, img, prompt, opts = WORK.get()
        job = JOBS[job_id]
        job["status"] = "running"
        try:
            with MODELS.lock:
                run_job(job, img, prompt, opts)
            job["status"], job["message"] = "done", "Finished"
        except Exception as exc:
            traceback.print_exc()
            job["status"], job["error"] = "error", f"{type(exc).__name__}: {exc}"
            MODELS.flush()
        finally:
            (OUT_DIR / job_id / "job.json").write_text(json.dumps(job, indent=2))


threading.Thread(target=worker, daemon=True).start()

# --------------------------------------------------------------------------- API
app = FastAPI(title="Sketch2Shape AI backend", version=VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {
        "status": "ok",
        "gpu": GPU_NAME,
        "vram_gb": VRAM_GB,
        "low_vram": LOW_VRAM,
        "ram_gb": RAM_GB,
        "sequential": SEQUENTIAL,
        "shape_model": f"{SHAPE_MODEL}/{SHAPE_SUBFOLDER}",
        "texture_available": texture_available(),
        "controlnet_model": CN_MODEL,
        "loaded": MODELS.loaded(),
        "queue": WORK.qsize(),
        "version": VERSION,
    }


@app.post("/jobs")
async def create_job(image: UploadFile = File(...), prompt: str = Form(""), options: str = Form("{}")):
    try:
        opts = json.loads(options or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(400, f"options must be JSON: {exc}")
    data = await image.read()
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(413, "Image larger than 15 MB")
    try:
        from io import BytesIO

        img = Image.open(BytesIO(data))
        img.load()
    except Exception:
        raise HTTPException(400, "Could not decode image")
    if img.mode in ("RGBA", "LA"):  # flatten transparency onto white paper
        bg = Image.new("RGB", img.size, "white")
        bg.paste(img, mask=img.getchannel("A"))
        img = bg
    job_id = uuid.uuid4().hex[:12]
    (OUT_DIR / job_id).mkdir(parents=True, exist_ok=True)
    JOBS[job_id] = {
        "id": job_id,
        "status": "queued",
        "stage": None,
        "message": f"Queued (position {WORK.qsize() + 1})",
        "error": None,
        "timings": {},
        "stages_skipped": [],
        "artifacts": {},
        "stats": None,
        "total_seconds": None,
        "prompt": prompt,
        "options": opts,
    }
    WORK.put((job_id, img, prompt, opts))
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job")
    return job


@app.get("/jobs/{job_id}/files/{name}")
def get_file(job_id: str, name: str):
    if job_id not in JOBS or name not in ARTIFACT_FILES:
        raise HTTPException(404, "Not found")
    path = OUT_DIR / job_id / ARTIFACT_FILES[name]
    if not path.exists():
        raise HTTPException(404, "File not ready")
    return FileResponse(path, filename=path.name)


# --------------------------------------------------------------------------- helpers
def warmup(texture: bool = True) -> None:
    """Download + load every model once so the first judge request is fast.

    In sequential mode each model is loaded and released in turn, so weights end
    up in the disk cache without ever holding more than one model in RAM.
    """
    t = time.time()
    print(f"[s2s] RAM {RAM_GB or 0:.1f} GB | VRAM {VRAM_GB or 0:.1f} GB | sequential={SEQUENTIAL}")
    steps = [("sd", MODELS.get_sd), ("rembg", MODELS.get_rembg), ("shape", MODELS.get_shape)]
    if texture and texture_available():
        steps.append(("paint", MODELS.get_paint))
    for name, load in steps:
        t0 = time.time()
        model = load()
        if name == "shape":
            place(model, "cpu")
        del model
        print(f"[s2s] {name} ready in {time.time() - t0:.0f}s | RAM used {_ram_used_gb() or 0:.1f} GB")
        if name != "rembg":
            MODELS.release_if_sequential(name)
    MODELS.flush()
    print(f"[s2s] models ready in {time.time() - t:.0f}s  loaded={MODELS.loaded()}")


def start_in_background(port: int = PORT) -> threading.Thread:
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    return th


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
