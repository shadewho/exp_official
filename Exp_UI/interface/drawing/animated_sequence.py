# Exploratory/Exp_UI/interface/drawing/animated_sequence.py
"""
Draw a PNG frame sequence as an animated emblem (no disk caching).
"""

import os, bpy, gpu
from gpu_extras.batch import batch_for_shader
from .config import LOADING_FRAMES_DIR, LOADING_FPS

# Tiny in-memory caches (session-only)
_LOADED_IMAGES = {}     # path -> bpy.types.Image
_LOADED_TEX    = {}     # image.name -> gpu.types.GPUTexture

def _get_image(path):
    if not path:
        return None
    img = _LOADED_IMAGES.get(path)
    try:
        _ = img.name
    except Exception:
        img = None

    if img is None:
        try:
            img = bpy.data.images.load(path, check_existing=True)
        except Exception:
            return None

    # Ensure UI-safe settings even for cached images
    try:
        cs = getattr(img, "colorspace_settings", None)
        if cs and getattr(cs, "name", None) != "Non-Color":
            cs.name = "Non-Color"   # critical: bypass color management for UI art
    except Exception:
        pass
    try:
        img.alpha_mode = 'STRAIGHT'  # harmless for JPG (no alpha), prevents fringes for PNGs
    except Exception:
        pass

    _LOADED_IMAGES[path] = img
    return img

def _get_texture(img):
    if not img:
        return None
    key = img.name
    tex = _LOADED_TEX.get(key)
    if tex:
        return tex
    try:
        tex = gpu.texture.from_image(img)
        _LOADED_TEX[key] = tex
        return tex
    except Exception:
        return None

# Discover frames once
_frames = sorted(f for f in os.listdir(LOADING_FRAMES_DIR) if f.lower().endswith(".png"))
if not _frames:
    print(f"[WARN] No PNG frames in {LOADING_FRAMES_DIR}")

def _texture_for(index: int):
    fname = _frames[index]
    img   = _get_image(os.path.join(LOADING_FRAMES_DIR, fname))
    return _get_texture(img)

def build_loading_frames(center_x, center_y, side_px):
    """Return a dict consumed by the draw loop to animate frames."""
    half   = side_px / 2
    verts  = [(center_x-half, center_y-half),
              (center_x+half, center_y-half),
              (center_x+half, center_y+half),
              (center_x-half, center_y+half)]
    coords = [(0,0), (1,0), (1,1), (0,1)]
    shader = gpu.shader.from_builtin('IMAGE')
    batch  = batch_for_shader(shader, 'TRI_FAN', {"pos": verts, "texCoord": coords})
    return {
        "type": "frame_seq",
        "shader": shader,
        "batch": batch,
        "size": side_px,
        "name": "Loading_Frames",
        "pos": (center_x-half, center_y-half, center_x+half, center_y+half),
        "_n_frames": max(1, len(_frames)),
        "_fps": LOADING_FPS,
    }
