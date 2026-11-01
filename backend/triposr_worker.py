"""Sketch2Shape AI - Isolated Generation Worker.

Executes a single 3D generation job in its own process.
When this script finishes, the OS reclaims 100% of GPU VRAM and CPU RAM,
eliminating CUDA memory fragmentation on Windows.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
import zipfile
from pathlib import Path

# Lock caches to D: drive
IITBBSML_DIR = Path("D:/IITBBSML")
HF_CACHE = IITBBSML_DIR / ".cache" / "huggingface"
TORCH_CACHE = IITBBSML_DIR / ".cache" / "torch"
U2NET_CACHE = IITBBSML_DIR / ".cache" / "u2net"

os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE)
os.environ["TORCH_HOME"] = str(TORCH_CACHE)
os.environ["U2NET_HOME"] = str(U2NET_CACHE)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

TSR_REPO = IITBBSML_DIR / "triposr_repo"
if TSR_REPO.exists() and str(TSR_REPO) not in sys.path:
    sys.path.insert(0, str(TSR_REPO))
if str(IITBBSML_DIR) not in sys.path:
    sys.path.insert(0, str(IITBBSML_DIR))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import trimesh
from PIL import Image, ImageFilter, ImageOps

OUT_DIR = Path("D:/ML/Zephyr-01/triposr_outputs")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

DIFFUSION_BASE_MODEL = "runwayml/stable-diffusion-v1-5"
CONTROLNET_SCRIBBLE_MODEL = "lllyasviel/control_v11p_sd15_scribble"
TRIPOSR_MODEL_ID = "stabilityai/TripoSR"

NEGATIVE = (
    "flat, 2d, illustration, drawing, sketch, cartoon, anime, orthographic, billboard, flat slab, box, "
    "solid block, rectangular block, background wall, backdrop, studio backdrop, cyclorama, curved wall, "
    "wall, floor, ground, table, pedestal, surface, shadow on wall, hard shadows, frame, border, canvas border, "
    "multiple objects, cropped, cut off, text, watermark, low quality, blurry, deformed, extra parts"
)
ARTIFACT_FILES = {
    "control": "control.png",
    "concept": "concept.png",
    "segmented": "segmented.png",
    "preview_cutout": "preview_cutout.png",
    "silhouette": "silhouette.png",
    "glb": "model.glb",
    "obj_zip": "model_obj.zip",
}


def clean_and_repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)])

    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        components = sorted(components, key=lambda c: len(c.vertices), reverse=True)
        mesh = components[0]

    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()

    mesh.vertices -= mesh.bounding_box.centroid
    max_extent = np.max(mesh.bounding_box.extents)
    if max_extent > 0:
        mesh.vertices /= (max_extent / 1.8)

    # Attach PBR dielectric material with low metalness and moderate roughness.
    # Without an explicit material, GLTF defaults to metallicFactor=1.0 which obliterates vertex colors.
    from trimesh.visual.material import PBRMaterial
    mesh.visual.material = PBRMaterial(
        metallicFactor=0.05,
        roughnessFactor=0.6,
        doubleSided=True,
    )

    return mesh


def mesh_stats(mesh: trimesh.Trimesh) -> dict:
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "extents": [float(x) for x in mesh.extents],
        "textured": hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None,
    }


def export_obj_zip(mesh: trimesh.Trimesh, job_dir: Path) -> Path:
    zip_path = job_dir / ARTIFACT_FILES["obj_zip"]
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        try:
            tex_mesh = mesh
            if hasattr(mesh.visual, "to_texture"):
                tex_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, visual=mesh.visual.to_texture())
            text, files = trimesh.exchange.obj.export_obj(
                tex_mesh, include_texture=True, return_texture=True, mtl_name="model.mtl"
            )
            zf.writestr("model.obj", text)
            for name, data in files.items():
                zf.writestr(name, data)
        except Exception:
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


def project_texture_to_mesh(mesh: trimesh.Trimesh, concept_img: Image.Image, seg_img: Image.Image) -> trimesh.Trimesh:
    """
    Forward-projects the high-resolution 2D reference image onto the 3D surface.
    Samples from the centered foreground cutout with margin dilation so edge vertices
    never sample transparent black or background canvas.
    """
    from scipy.ndimage import distance_transform_edt

    seg_img = seg_img.convert("RGBA").resize((512, 512), Image.LANCZOS)
    seg_arr = np.array(seg_img).astype(np.float32)

    raw_rgb = seg_arr[:, :, :3]
    raw_alpha = seg_arr[:, :, 3] / 255.0

    # Margin bleed: propagate border colors outward into empty space
    mask = raw_alpha > 0.05
    if mask.any() and not mask.all():
        indices = distance_transform_edt(~mask, return_distances=False, return_indices=True)
        img_rgb = raw_rgb[indices[0], indices[1]]
    else:
        img_rgb = raw_rgb
    img_alpha = raw_alpha

    # 2D Bounding box of foreground object in segmented image
    y_idx, x_idx = np.where(img_alpha > 0.1)
    if len(x_idx) > 0 and len(y_idx) > 0:
        x_min, x_max = x_idx.min(), x_idx.max()
        y_min, y_max = y_idx.min(), y_idx.max()
    else:
        x_min, x_max, y_min, y_max = 0, 511, 0, 511

    verts = mesh.vertices
    normals = mesh.vertex_normals

    # 3D bounding extents in upright coordinates (+X right, +Y up, +Z front)
    vx_min, vx_max = verts[:, 0].min(), verts[:, 0].max()
    vy_min, vy_max = verts[:, 1].min(), verts[:, 1].max()
    x_span = max(vx_max - vx_min, 1e-5)
    y_span = max(vy_max - vy_min, 1e-5)

    u = (verts[:, 0] - vx_min) / x_span
    v = (verts[:, 1] - vy_min) / y_span

    px = np.clip(np.round(x_min + u * (x_max - x_min)).astype(int), 0, 511)
    py = np.clip(np.round(y_max - v * (y_max - y_min)).astype(int), 0, 511)

    sampled_rgb = img_rgb[py, px]

    # Normal alignment to front camera: normals[:, 2] in upright space (+Z is front)
    nz = normals[:, 2]

    # Front-facing weight with smooth cosine falloff
    front_w = np.clip((nz + 0.15) / 0.50, 0.0, 1.0) ** 1.2
    blend_w = front_w[:, None]

    # Backside: consistent matching color with gentle ambient darkening (0.85)
    back_rgb = sampled_rgb * 0.85

    target_rgb = blend_w * sampled_rgb + (1.0 - blend_w) * back_rgb
    final_rgba = np.zeros((len(verts), 4), dtype=np.uint8)
    final_rgba[:, :3] = np.clip(target_rgb, 0, 255).astype(np.uint8)
    final_rgba[:, 3] = 255

    from trimesh.visual.material import PBRMaterial
    mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=final_rgba)
    mesh.visual.material = PBRMaterial(metallicFactor=0.05, roughnessFactor=0.6, doubleSided=True)
    return mesh


def make_control(sketch: Image.Image) -> Image.Image:
    im_rgb = sketch.convert("RGB").resize((512, 512), Image.LANCZOS)
    arr = np.array(im_rgb)
    is_bw_sketch = float(np.mean(np.std(arr, axis=2) < 10)) > 0.85 and float(np.mean(np.mean(arr, axis=2) > 220)) > 0.50
    gray = ImageOps.grayscale(im_rgb)
    if is_bw_sketch:
        inv = ImageOps.invert(gray).point(lambda v: 255 if v > 50 else 0)
        return inv.filter(ImageFilter.MaxFilter(3)).convert("RGB")
    else:
        # Edge-aware scribble extraction for colored art/photos: prevents solid-white blowouts
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edges = edges.point(lambda v: 255 if v > 35 else 0)
        return edges.filter(ImageFilter.MaxFilter(3)).convert("RGB")


def run(job_id: str):
    job_dir = OUT_DIR / job_id
    meta_path = job_dir / "job.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing job metadata at {meta_path}")

    job = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    input_path = job_dir / "input.png"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input sketch at {input_path}")

    sketch = Image.open(input_path)
    if sketch.mode not in ("RGB", "RGBA"):
        sketch = sketch.convert("RGB")
    prompt = job.get("prompt", "")
    opts = job.get("options", {})
    t_start = time.time()

    def sync_stage(name: str, msg: str):
        job["status"] = "running"
        job["stage"] = name
        job["message"] = msg
        meta_path.write_text(json.dumps(job, indent=2))
        return time.time()

    def sync_done(name: str, t0: float):
        job["timings"][name] = round(time.time() - t0, 2)
        meta_path.write_text(json.dumps(job, indent=2))

    try:
        # 1. Preprocess
        t = sync_stage("preprocess", "Normalizing sketch")
        control = make_control(sketch)
        control.save(job_dir / ARTIFACT_FILES["control"])
        job["artifacts"]["control"] = f"/jobs/{job_id}/files/control"
        sync_done("preprocess", t)

        # 2. Concept render (SD 1.5 + ControlNet)
        use_cn = opts.get("use_controlnet", True) and (bool(prompt.strip()) or opts.get("force_concept", False))
        if use_cn:
            t = sync_stage("concept", "Rendering concept with SD 1.5 + ControlNet")
            from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

            controlnet = ControlNetModel.from_pretrained(
                CONTROLNET_SCRIBBLE_MODEL,
                torch_dtype=DTYPE,
                variant="fp16" if DTYPE == torch.float16 else None,
                cache_dir=str(HF_CACHE),
            )
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                DIFFUSION_BASE_MODEL,
                controlnet=controlnet,
                torch_dtype=DTYPE,
                variant="fp16" if DTYPE == torch.float16 else None,
                safety_checker=None,
                cache_dir=str(HF_CACHE),
            )
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

            if DEVICE == "cuda":
                pipe.enable_model_cpu_offload()
                pipe.enable_attention_slicing()
                try:
                    pipe.enable_vae_slicing()
                except Exception:
                    pass

            user_p = prompt.strip()
            if not user_p:
                user_p = "a ceramic coffee mug, 3d round cylindrical asset"

            full_prompt = (
                f"{user_p}, 3d asset, blender 3d model render, volumetric shading, ambient occlusion, "
                "smooth curved cylindrical geometry, dramatic specular highlights, "
                "single complete isolated object floating in empty space on pure seamless solid white background, "
                "sharp focus, 8k uhd, photorealistic, octane render"
            )
            seed = int(opts.get("seed", 1234))
            steps = int(opts.get("concept_steps", 15))
            ctrl_scale = float(opts.get("control_scale", 0.72))
            ctrl_end = float(opts.get("control_end", 0.80))
            gen = torch.Generator(device="cpu").manual_seed(seed)
            concept = pipe(
                full_prompt,
                image=control,
                negative_prompt=NEGATIVE,
                num_inference_steps=steps,
                guidance_scale=7.0,
                controlnet_conditioning_scale=ctrl_scale,
                control_guidance_end=ctrl_end,
                generator=gen,
            ).images[0]

            concept.save(job_dir / ARTIFACT_FILES["concept"])
            job["artifacts"]["concept"] = f"/jobs/{job_id}/files/concept"
            del pipe, controlnet
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            sync_done("concept", t)
        else:
            job["stages_skipped"].append("concept")
            concept = sketch

        # 3. Foreground isolation & normalization (ISNet + speckle filter + edge erosion)
        t = sync_stage("segment", "Isolating foreground asset (ISNet clean cutout)")
        from backend.clean_cutout import clean_foreground_cutout

        final_rgb, segmented, sil_img, preview_sheet, cutout_stats = clean_foreground_cutout(
            concept,
            model="isnet-general-use",
            out_size=512,
            fill=0.85,
        )
        segmented.save(job_dir / ARTIFACT_FILES["segmented"])
        preview_sheet.save(job_dir / ARTIFACT_FILES["preview_cutout"])
        sil_img.save(job_dir / ARTIFACT_FILES["silhouette"])

        job["artifacts"]["segmented"] = f"/jobs/{job_id}/files/segmented"
        job["artifacts"]["preview_cutout"] = f"/jobs/{job_id}/files/preview_cutout"
        job["artifacts"]["silhouette"] = f"/jobs/{job_id}/files/silhouette"
        job["cutout_stats"] = cutout_stats
        sync_done("segment", t)

        # 4. Shape generation (Image-to-3D neural reconstruction)
        engine_choice = opts.get("engine", "triposr")
        mc_res = int(opts.get("octree_resolution", 192))
        mc_res = min(256, max(128, mc_res))

        if engine_choice == "version2":
            t = sync_stage("shape", "Reconstructing 3D neural surface with Version 2 Engine (KVignesh122)")
            from engines.version2_triposr import reconstruct_3d_version2

            raw_mesh = reconstruct_3d_version2(
                final_rgb,
                resolution=mc_res,
                device=DEVICE,
                use_rotation=True,
            )
        else:
            t = sync_stage("shape", "Reconstructing 3D neural surface with TripoSR")
            from tsr.system import TSR

            snaps_dir = HF_CACHE / "models--stabilityai--TripoSR" / "snapshots"
            model_source = TRIPOSR_MODEL_ID
            if snaps_dir.exists():
                snaps = [p for p in snaps_dir.iterdir() if p.is_dir()]
                if snaps:
                    model_source = str(snaps[0])

            tsr_model = TSR.from_pretrained(
                model_source,
                config_name="config.yaml",
                weight_name="model.ckpt",
            )
            tsr_model.renderer.set_chunk_size(8192)
            tsr_model.to(DEVICE)

            with torch.no_grad():
                scene_codes = tsr_model([final_rgb], device=DEVICE)
                raw_meshes = tsr_model.extract_mesh(
                    scene_codes,
                    has_vertex_color=True,
                    resolution=mc_res,
                )
                from tsr.utils import to_gradio_3d_orientation
                raw_mesh = to_gradio_3d_orientation(raw_meshes[0])

            del tsr_model

        mesh = clean_and_repair_mesh(raw_mesh)
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        sync_done("shape", t)

        # 5. Texture
        t = sync_stage("texture", "Forward camera texture projection & color baking")
        mesh = project_texture_to_mesh(mesh, concept, segmented)
        sync_done("texture", t)

        # 6. Export
        t = sync_stage("export", "Validating & exporting GLB and OBJ")
        glb_path = job_dir / ARTIFACT_FILES["glb"]
        mesh.export(glb_path)
        export_obj_zip(mesh, job_dir)

        job["artifacts"]["glb"] = f"/jobs/{job_id}/files/glb"
        job["artifacts"]["obj_zip"] = f"/jobs/{job_id}/files/obj_zip"
        job["stats"] = mesh_stats(mesh)
        sync_done("export", t)

        job["status"] = "done"
        job["message"] = "Finished"
        job["total_seconds"] = round(time.time() - t_start, 2)
        meta_path.write_text(json.dumps(job, indent=2))
        print(f"[worker] Job {job_id} completed successfully in {job['total_seconds']}s")

    except Exception as exc:
        traceback.print_exc()
        job["status"] = "error"
        job["error"] = f"{type(exc).__name__}: {exc}"
        meta_path.write_text(json.dumps(job, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True, help="Job identifier")
    args = parser.parse_args()
    run(args.job_id)
