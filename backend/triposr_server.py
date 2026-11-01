"""Sketch2Shape AI - Local TripoSR GPU Backend Server.

Lightweight, crash-proof FastAPI server:
- Does NOT load PyTorch into the server process.
- Dispatches each generation job to an isolated subprocess worker (`triposr_worker.py`).
- 100% immune to CUDA memory leaks, driver faults, and heap fragmentation.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image

VERSION = "1.2.0"
PORT = int(os.environ.get("S2S_TRIPOSR_PORT", "8001"))
OUT_DIR = Path("D:/ML/Zephyr-01/triposr_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PYTHON_EXE = sys.executable
WORKER_SCRIPT = Path(__file__).resolve().parent / "triposr_worker.py"

ARTIFACT_FILES = {
    "control": "control.png",
    "concept": "concept.png",
    "segmented": "segmented.png",
    "preview_cutout": "preview_cutout.png",
    "silhouette": "silhouette.png",
    "glb": "model.glb",
    "obj_zip": "model_obj.zip",
}

# --------------------------------------------------------------------------- Job Queue
JOBS: dict[str, dict] = {}
WORK: queue.Queue[str] = queue.Queue()


def worker_loop():
    """Sequential worker runner: executes one job at a time in an isolated process."""
    while True:
        job_id = WORK.get()
        meta_path = OUT_DIR / job_id / "job.json"
        try:
            job = JOBS[job_id]
            job["status"] = "running"
            job["message"] = "Processing on GPU..."
            meta_path.write_text(json.dumps(job, indent=2))

            cmd = [PYTHON_EXE, str(WORKER_SCRIPT), "--job-id", job_id]
            print(f"[server] Launching isolated worker: {' '.join(cmd)}")
            proc = subprocess.run(cmd, capture_output=True, text=True)

            if proc.returncode != 0:
                print(f"[server] Worker process failed (code {proc.returncode}):\n{proc.stderr}")
                if meta_path.exists():
                    job_data = json.loads(meta_path.read_text(encoding="utf-8-sig"))
                    if job_data.get("status") != "error":
                        job_data["status"] = "error"
                        job_data["error"] = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else f"Process exited with code {proc.returncode}"
                        meta_path.write_text(json.dumps(job_data, indent=2))
                    JOBS[job_id] = job_data
            else:
                if meta_path.exists():
                    JOBS[job_id] = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            print(f"[server] Job execution error: {exc}")
            if job_id in JOBS:
                JOBS[job_id]["status"] = "error"
                JOBS[job_id]["error"] = str(exc)
        finally:
            WORK.task_done()


threading.Thread(target=worker_loop, daemon=True).start()

# --------------------------------------------------------------------------- FastAPI App
app = FastAPI(title="Sketch2Shape AI - Local TripoSR Engine", version=VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {
        "status": "ok",
        "engine": "triposr",
        "supported_engines": ["triposr", "version2"],
        "gpu": "NVIDIA GeForce RTX 4050 Laptop GPU",
        "vram_gb": 6.0,
        "low_vram": True,
        "shape_model": "stabilityai/TripoSR",
        "version2_engine": "KVignesh122/image-to-3D",
        "texture_available": True,
        "controlnet_model": "lllyasviel/control_v11p_sd15_scribble",
        "architecture": "isolated_process_workers",
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
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(413, "Image larger than 20 MB")

    try:
        from io import BytesIO
        img = Image.open(BytesIO(data))
        img.load()
    except Exception:
        raise HTTPException(400, "Could not decode image")

    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")

    job_id = uuid.uuid4().hex[:12]
    job_dir = OUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save initial input sketch
    img.save(job_dir / "input.png")

    job = {
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
    (job_dir / "job.json").write_text(json.dumps(job, indent=2))
    JOBS[job_id] = job
    WORK.put(job_id)

    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    meta_path = OUT_DIR / job_id / "job.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8-sig"))
        except Exception:
            pass
    if job_id in JOBS:
        return JOBS[job_id]
    raise HTTPException(404, "Unknown job")


@app.get("/jobs/{job_id}/files/{name}")
def get_file(job_id: str, name: str):
    if name not in ARTIFACT_FILES:
        raise HTTPException(404, "Not found")
    path = OUT_DIR / job_id / ARTIFACT_FILES[name]
    if not path.exists():
        raise HTTPException(404, "File not ready")
    return FileResponse(path, filename=path.name)


if __name__ == "__main__":
    import uvicorn
    print(f"[*] Starting Local TripoSR Engine on http://127.0.0.1:{PORT}...")
    uvicorn.run(app, host="127.0.0.1", port=PORT)
