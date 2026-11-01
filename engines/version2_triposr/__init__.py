"""Version 2 Image-to-3D Reconstruction Engine (KVignesh122/image-to-3D TripoSR).

Strictly performs Image-to-3D reconstruction:
Takes a preprocessed/background-removed RGBA PIL image, feeds it to the
TSR neural model, and returns an extracted trimesh.Trimesh.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# Ensure version2_triposr directory is on sys.path for flat module resolution
CURRENT_DIR = str(Path(__file__).resolve().parent)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import numpy as np
import torch
import trimesh
from PIL import Image

from .system import TSR
from .rotate import rotate
from .utils import find_class

__all__ = ["TSR", "rotate", "find_class", "reconstruct_3d_version2"]


def reconstruct_3d_version2(
    image: Image.Image,
    resolution: int = 192,
    device: str = "cuda",
    use_rotation: bool = True,
) -> trimesh.Trimesh:
    """Reconstruct a 3D mesh from an RGBA foreground image using Version 2 engine."""
    hf_cache = Path("D:/IITBBSML/.cache/huggingface")
    snaps_dir = hf_cache / "models--stabilityai--TripoSR" / "snapshots"
    src = "stabilityai/TripoSR"
    if snaps_dir.exists():
        snaps = [p for p in snaps_dir.iterdir() if p.is_dir()]
        if snaps:
            src = str(snaps[0])

    p = Path(src)
    if p.is_dir():
        cfg_path = str(p / "config.yaml")
        weight_path = str(p / "model.ckpt")
    else:
        from huggingface_hub import hf_hub_download
        cfg_path = hf_hub_download(src, "config.yaml")
        weight_path = hf_hub_download(src, "model.ckpt")

    model = TSR.from_pretrained(cfg_path, weight_path)
    model.renderer.set_chunk_size(8192)
    model.to(device)

    # Prepare input image with neutral gray background for alpha channel
    np_img = np.array(image.convert("RGBA")).astype(np.float32) / 255.0
    rgb = np_img[:, :, :3] * np_img[:, :, 3:4] + (1.0 - np_img[:, :, 3:4]) * 0.5
    prepared_pil = Image.fromarray((rgb * 255.0).astype(np.uint8))

    with torch.no_grad():
        scene_codes = model([prepared_pil], device=device)
        raw_meshes = model.extract_mesh(
            scene_codes,
            resolution=resolution,
        )
        mesh = raw_meshes[0]

    if use_rotation:
        # Apply canonical TripoSR camera alignment so mesh faces viewer and aligns with texture projection
        mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return mesh
