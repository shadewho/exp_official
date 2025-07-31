"""
Utility for drawing a sequence of opaque PNG frames as an in‑viewport animation.
Keeps GPU memory low by sharing one shader and lazy‑loading each frame once.
"""

import os, time, gpu, bpy
from gpu_extras.batch import batch_for_shader
from .config import LOADING_FRAMES_DIR, LOADING_FPS
from ...cache_system.persistence import get_or_load_image, get_or_create_texture

# Discover all *.png files once (natural sort keeps “frame_0001” … in order)
_frames = sorted(
    f for f in os.listdir(LOADING_FRAMES_DIR)
    if f.lower().endswith(".png")
)
if not _frames:
    print(f"[WARN] No PNG frames found in {LOADING_FRAMES_DIR}")

# Lazy‑loaded textures cache  {filename → gpu.types.GPUTexture}
_tex_cache = {}

def _texture_for(index: int):
    """Return (and cache) the GPU texture for the N‑th frame."""
    fname = _frames[index]
    if fname not in _tex_cache:
        img_path = os.path.join(LOADING_FRAMES_DIR, fname)
        img      = get_or_load_image(img_path)
        _tex_cache[fname] = get_or_create_texture(img)
    return _tex_cache[fname]

def build_loading_frames(center_x, center_y, side_px):
    """
    Returns a dict that the master draw‑loop displays each redraw.
    """
    half    = side_px / 2
    verts   = [(center_x-half, center_y-half),
               (center_x+half, center_y-half),
               (center_x+half, center_y+half),
               (center_x-half, center_y+half)]
    coords  = [(0,0), (1,0), (1,1), (0,1)]
    shader  = gpu.shader.from_builtin('IMAGE')
    batch   = batch_for_shader(shader, 'TRI_FAN', {"pos": verts, "texCoord": coords})

    return {
        "type"      : "frame_seq",           # NEW type handled in draw‑loop
        "shader"    : shader,
        "batch"     : batch,
        "size"      : side_px,
        "name"      : "Loading_Frames",
        "pos"       : (center_x-half, center_y-half,
                       center_x+half, center_y+half),
        # runtime data
        "_n_frames" : len(_frames),
        "_fps"      : LOADING_FPS,
    }
