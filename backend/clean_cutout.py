"""Sketch2Shape AI - Foreground Cutout & Normalization Preprocessing.

Adapted from user pipeline for robust, centered foreground cutouts:
- Uses ISNet ('isnet-general-use') for crisp object boundaries and thin structures
- Connected components filtering to eliminate background dust and specks
- Selective small hole filling
- Edge erosion defringing to remove background color halos
- Aspect-preserving tight bounding box crop + 85% canvas fill centering
- 127-gray background blending for TripoSR / Version 2 NeRF reconstruction
- Diagnostic 3-panel preview sheet (Original | Preprocessed | Silhouette)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
import rembg
from scipy import ndimage as ndi

# Lock cache location
IITBBSML_DIR = Path("D:/IITBBSML")
U2NET_CACHE = IITBBSML_DIR / ".cache" / "u2net"
os.environ["U2NET_HOME"] = str(U2NET_CACHE)

# Shared session cache so we don't reload the ONNX model from disk on every invocation
_SESSION_CACHE: dict[str, rembg.Session] = {}


def get_rembg_session(model_name: str = "isnet-general-use") -> rembg.Session:
    global _SESSION_CACHE
    if model_name not in _SESSION_CACHE:
        print(f"[clean_cutout] Initializing rembg session with model '{model_name}'...")
        _SESSION_CACHE[model_name] = rembg.new_session(model_name)
    return _SESSION_CACHE[model_name]


def clean_foreground_cutout(
    orig: Image.Image,
    model: str = "isnet-general-use",
    out_size: int = 512,
    fill: float = 0.85,
    session: Optional[rembg.Session] = None,
) -> Tuple[Image.Image, Image.Image, Image.Image, Image.Image, dict]:
    """
    Transforms an input image into a clean, centered foreground cutout.

    Returns:
        final_rgb: (out_size, out_size) RGB image composited on 127-gray (for TripoSR/Version 2)
        segmented_rgba: (out_size, out_size) RGBA cutout with alpha channel
        sil_mask: (out_size, out_size) RGB silhouette mask image
        preview_sheet: (T*3+40, T+40) diagnostic preview sheet
        stats: Dictionary containing centering, fill ratio, and component metrics
    """
    if session is None:
        session = get_rembg_session(model)

    orig_rgb = orig.convert("RGB")
    ow, oh = orig.size

    # Check if image already has pre-existing transparent alpha
    has_clean_alpha = False
    if orig.mode == "RGBA":
        alpha_arr = np.array(orig)[..., 3]
        if np.mean(alpha_arr < 200) > 0.05:
            has_clean_alpha = True

    if has_clean_alpha:
        arr = np.array(orig)
    else:
        # Check if the image is a black-and-white line drawing on white paper
        np_orig = np.array(orig_rgb)
        is_gray = float(np.mean(np.std(np_orig, axis=2) < 10)) > 0.90
        is_paper_white = float(np.mean(np.mean(np_orig, axis=2) > 230)) > 0.60

        if is_gray and is_paper_white:
            # For line sketches on white paper: threshold ink and fill enclosed silhouette
            lum = np_orig[..., 0] * 0.299 + np_orig[..., 1] * 0.587 + np_orig[..., 2] * 0.114
            stroke_mask = lum < 185
            enclosed_mask = ndi.binary_fill_holes(stroke_mask)
            arr = np.zeros((oh, ow, 4), dtype=np.uint8)
            arr[..., :3] = np_orig
            arr[..., 3] = (enclosed_mask * 255).astype(np.uint8)
        else:
            # Natural photo or rendered concept art: use ISNet segmenter
            arr = np.array(rembg.remove(orig_rgb, session=session))

    # 1. Hard mask
    m = arr[..., 3] > 127
    if m.sum() == 0:
        m = arr[..., 3] > 20

    # 2. Keep real parts, drop specks / background dust
    lab, n = ndi.label(m)
    if n > 1:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = np.isin(lab, 1 + np.where(sizes >= 0.01 * sizes.max())[0])

    # 3. Fill tiny holes only (preserving intentional loops/openings)
    holes = ndi.binary_fill_holes(m) & ~m
    hl, hn = ndi.label(holes)
    if hn > 0:
        hs = ndi.sum(holes, hl, range(1, hn + 1))
        m = m | np.isin(hl, 1 + np.where(hs < 0.002 * max(m.sum(), 1))[0])

    # 4. Remove edge fringe with 1-iteration erosion
    m_eroded = ndi.binary_erosion(m, iterations=1)
    if m_eroded.sum() > 50:  # Only use eroded mask if it didn't obliterate thin structures
        m = m_eroded
    arr[..., 3] = (m * 255).astype(np.uint8)

    # 5. Crop tight bounding box
    ys, xs = np.where(m)
    if len(xs) == 0 or len(ys) == 0:
        xs = np.array([0, arr.shape[1] - 1])
        ys = np.array([0, arr.shape[0] - 1])

    crop = Image.fromarray(arr).crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))

    # 6. Aspect-preserving resize
    scale = (fill * out_size) / max(crop.size)
    cw = max(1, round(crop.width * scale))
    ch = max(1, round(crop.height * scale))
    crop = crop.resize((cw, ch), Image.LANCZOS)

    ox = (out_size - crop.width) // 2
    oy = (out_size - crop.height) // 2

    # 7. Alpha composite on 127 neutral gray for TripoSR / Version 2
    canvas_rgb = Image.new("RGBA", (out_size, out_size), (127, 127, 127, 255))
    canvas_rgb.alpha_composite(crop, (ox, oy))
    final_rgb = canvas_rgb.convert("RGB")

    # 8. Clean RGBA cutout for texture projection and GLB baking
    canvas_rgba = Image.new("RGBA", (out_size, out_size), (0, 0, 0, 0))
    canvas_rgba.alpha_composite(crop, (ox, oy))

    # 9. Silhouette mask & self-check metrics
    fm = np.array(crop.split()[3]) > 127
    full_mask = np.zeros((out_size, out_size), bool)
    full_mask[oy : oy + crop.height, ox : ox + crop.width] = fm

    yy, xx = np.where(full_mask)
    if len(xx) > 0 and len(yy) > 0:
        l = int(xx.min())
        r = int(out_size - 1 - xx.max())
        t = int(yy.min())
        b = int(out_size - 1 - yy.max())
        longest = float(max(xx.max() - xx.min() + 1, yy.max() - yy.min() + 1) / out_size)
    else:
        l, r, t, b = 0, 0, 0, 0
        longest = 0.0

    ncomp = int(ndi.label(full_mask)[1])

    # Background fringe check
    ring = full_mask & ~ndi.binary_erosion(full_mask, iterations=2)
    px = np.array(final_rgb).astype(int)[ring]
    o = np.array(orig_rgb).astype(int)
    bg = np.median(np.concatenate([o[0], o[-1], o[:, 0], o[:, -1]]), axis=0)
    if len(px) > 0:
        fringe_pct = float(np.mean(np.linalg.norm(px - bg, axis=1) < 25)) * 100.0
    else:
        fringe_pct = 0.0

    stats = {
        "fill_percent": round(longest * 100.0, 1),
        "target_fill": round(fill * 100.0, 1),
        "centered": abs(l - r) <= 2 and abs(t - b) <= 2,
        "margins": {"left": l, "right": r, "top": t, "bottom": b},
        "components": ncomp,
        "fringe_pct": round(fringe_pct, 1),
        "status": "OK" if fringe_pct < 5.0 and ncomp <= 3 else "WARN",
    }

    # 10. Generate 3-panel preview sheet (Original | Clean Cutout | Silhouette)
    T = 420
    prev_a = orig_rgb.copy()
    prev_a.thumbnail((T, T), Image.LANCZOS)
    prev_f = final_rgb.resize((T, T), Image.LANCZOS)
    sil_img = Image.fromarray((full_mask * 255).astype(np.uint8)).convert("RGB").resize((T, T), Image.NEAREST)

    sheet = Image.new("RGB", (T * 3 + 40, T + 40), (30, 30, 30))
    panels = [(prev_a, "Original Input"), (prev_f, "Preprocessed Cutout (ISNet)"), (sil_img, "Silhouette Mask")]
    for i, (im, name) in enumerate(panels):
        pos_x = 10 + i * (T + 10)
        pos_y = 30 + (T - im.height) // 2
        sheet.paste(im, (pos_x, pos_y))
        ImageDraw.Draw(sheet).text((pos_x + 2, 8), name, fill=(255, 255, 255))

    return final_rgb, canvas_rgba, sil_img, sheet, stats
