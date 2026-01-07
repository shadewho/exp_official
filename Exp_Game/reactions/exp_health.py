# Exp_Game/reactions/exp_health.py
"""
Health System - Track health for any object in the scene.

Consolidated module containing:
- Core health cache and operations
- Reaction executors (ENABLE_HEALTH, DISPLAY_HEALTH_UI)
- GPU-rendered health bar UI overlay

Architecture:
    1. ENABLE_HEALTH reaction fires -> enable_health() adds object to cache
    2. Runtime: modify_health() / set_health() change values (future damage reactions)
    3. Game Reset: reset_all_health() restores start values
    4. Game End: clear_all_health() empties cache

The cache uses object names (strings) as keys for worker-serializable data.
"""

import bpy
import gpu
import os
import math
from gpu_extras.batch import batch_for_shader
from ..developer.dev_logger import log_game


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH CACHE
# ══════════════════════════════════════════════════════════════════════════════
# Structure: {object_name: {"current": float, "start": float, "min": float, "max": float}}

_health_cache: dict = {}


# ══════════════════════════════════════════════════════════════════════════════
# ENABLE / DISABLE HEALTH
# ══════════════════════════════════════════════════════════════════════════════

def enable_health(obj_name: str, start: float, min_val: float, max_val: float):
    """
    Enable health tracking on an object.
    Called by ENABLE_HEALTH reaction.

    Args:
        obj_name: Name of the object (for serializable lookup)
        start: Initial health value (also used on reset)
        min_val: Minimum health (usually 0)
        max_val: Maximum health
    """
    _health_cache[obj_name] = {
        "current": start,
        "start": start,
        "min": min_val,
        "max": max_val,
    }
    log_game("HEALTH", f"ENABLE obj={obj_name} start={start} range=[{min_val}, {max_val}]")


def disable_health(obj_name: str) -> bool:
    """
    Remove health tracking from an object.
    Returns True if object was being tracked.
    """
    if obj_name in _health_cache:
        del _health_cache[obj_name]
        log_game("HEALTH", f"DISABLE obj={obj_name}")
        return True
    return False


def is_health_enabled(obj_name: str) -> bool:
    """Check if an object has health tracking enabled."""
    return obj_name in _health_cache


# ══════════════════════════════════════════════════════════════════════════════
# GET / SET / MODIFY HEALTH
# ══════════════════════════════════════════════════════════════════════════════

def get_health(obj_name: str) -> dict | None:
    """
    Get health data for an object.
    Returns dict with current, start, min, max - or None if not enabled.
    """
    return _health_cache.get(obj_name)


def get_current_health(obj_name: str) -> float | None:
    """Get just the current health value. Returns None if not enabled."""
    data = _health_cache.get(obj_name)
    return data["current"] if data else None


def set_health(obj_name: str, value: float) -> float | None:
    """
    Set health to a specific value (clamped to min/max).
    Returns the new health value, or None if object not tracked.
    """
    data = _health_cache.get(obj_name)
    if not data:
        return None

    old = data["current"]
    data["current"] = max(data["min"], min(data["max"], value))
    # Note: batch invalidation happens automatically in _draw() via cache_state check
    log_game("HEALTH", f"SET obj={obj_name} {old:.1f} -> {data['current']:.1f}")
    return data["current"]


def modify_health(obj_name: str, delta: float) -> float | None:
    """
    Add or subtract from health (clamped to min/max).
    Use positive delta for healing, negative for damage.
    Returns the new health value, or None if object not tracked.
    """
    data = _health_cache.get(obj_name)
    if not data:
        return None

    old = data["current"]
    data["current"] = max(data["min"], min(data["max"], old + delta))
    # Note: batch invalidation happens automatically in _draw() via cache_state check
    log_game("HEALTH", f"MODIFY obj={obj_name} {old:.1f} -> {data['current']:.1f} (delta={delta:+.1f})")
    return data["current"]


# ══════════════════════════════════════════════════════════════════════════════
# RESET / CLEAR (Lifecycle)
# ══════════════════════════════════════════════════════════════════════════════

def reset_all_health():
    """
    Reset all health values to their start values.
    Called on game reset (R key). Objects remain tracked.
    """
    count = len(_health_cache)
    for data in _health_cache.values():
        data["current"] = data["start"]
    _invalidate_batches()  # Trigger batch rebuild on next draw
    log_game("HEALTH", f"RESET {count} objects restored to start values")


def clear_all_health():
    """
    Clear the entire health cache.
    Called on game start (invoke) and game end (cancel).
    """
    count = len(_health_cache)
    _health_cache.clear()
    _invalidate_batches()  # Clear batch cache
    log_game("HEALTH", f"CLEAR cache emptied (was tracking {count} objects)")


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY / STATS
# ══════════════════════════════════════════════════════════════════════════════

def get_health_stats() -> dict:
    """Get summary statistics for debugging."""
    return {
        "count": len(_health_cache),
        "objects": list(_health_cache.keys()),
    }


def get_all_health() -> dict:
    """Get a copy of the entire health cache (for debugging/serialization)."""
    return dict(_health_cache)


def serialize_for_worker() -> dict:
    """
    Serialize health data for worker process.
    Future: Used when offloading damage calculations to worker.
    """
    return {
        name: {
            "current": data["current"],
            "start": data["start"],
            "min": data["min"],
            "max": data["max"],
        }
        for name, data in _health_cache.items()
    }


# ══════════════════════════════════════════════════════════════════════════════
# REACTION EXECUTORS
# ══════════════════════════════════════════════════════════════════════════════

def execute_enable_health_reaction(r):
    """
    Execute ENABLE_HEALTH reaction.
    Attaches health tracking to the target object.
    Optionally enables health UI overlay if health_show_ui is True.
    """
    from .exp_bindings import resolve_object, resolve_float

    # Resolve object from binding (node connection) or fallback to direct property
    obj = resolve_object(r, "health_target_object", r.health_target_object)
    if not obj:
        log_game("HEALTH", "ENABLE_SKIP no target object specified")
        return

    # Resolve float values from bindings or use direct properties
    start_val = resolve_float(r, "health_start_value", r.health_start_value)
    min_val = resolve_float(r, "health_min_value", r.health_min_value)
    max_val = resolve_float(r, "health_max_value", r.health_max_value)

    enable_health(obj.name, start_val, min_val, max_val)


def execute_display_health_ui_reaction(r):
    """
    Execute DISPLAY_HEALTH_UI reaction.
    Shows health bar overlay on screen for target object.
    """
    from .exp_bindings import resolve_object, resolve_int

    # Resolve object from binding (node connection) or fallback to direct property
    obj = resolve_object(r, "health_ui_target_object", r.health_ui_target_object)
    if not obj:
        log_game("HEALTH", "UI_SKIP no target object specified")
        return

    # Resolve integer values from bindings or use direct properties
    scale = resolve_int(r, "health_ui_scale", r.health_ui_scale)
    offset_x = resolve_int(r, "health_ui_offset_x", r.health_ui_offset_x)
    offset_y = resolve_int(r, "health_ui_offset_y", r.health_ui_offset_y)

    enable_health_ui(
        obj.name,
        position=r.health_ui_position,
        scale=scale,
        offset_x=offset_x,
        offset_y=offset_y
    )


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH UI - STATE
# ══════════════════════════════════════════════════════════════════════════════

_draw_handle = None
_ui_config = None
_images = {}  # {"container": bpy.types.Image, "full": Image, "half": Image, "empty": Image, "cross": Image}

# ══════════════════════════════════════════════════════════════════════════════
# HEALTH UI - CACHED GPU RESOURCES (Performance Optimization)
# ══════════════════════════════════════════════════════════════════════════════
# These are cached to avoid recreating every frame:
# - Shader: fetched once from builtin
# - Textures: created once from loaded images
# - Batches: rebuilt only when health/config/screen changes

_cached_shader = None
_cached_textures = {}  # {"container": gpu.types.GPUTexture, ...}
_cached_batches = {}   # {"container": batch, "full": batch, "half": batch, "empty": batch, "cross": batch}
_cache_state = None    # {"filled_units": int, "config_hash": str, "screen": (w,h)} - for invalidation

# Reference resolution for scaling (matches exp_custom_ui.py)
REF_WIDTH = 1920.0
REF_HEIGHT = 1080.0

# Layout constants (base sizes at reference resolution)
CONTAINER_WIDTH = 1080
CONTAINER_HEIGHT = 160
PIP_SIZE = 240  # Original pip image size
PIP_COUNT = 10  # Health pips (not including cross)

# Calculated pip display size to fit in container with some overlap allowed
PIP_DISPLAY_SIZE = 90  # Scaled pip diameter at reference resolution
PIP_SCALE = PIP_DISPLAY_SIZE / PIP_SIZE  # ~0.375

# Pip positioning within container (11 slots total)
# To center 11 pips: total span = 10*90 + 90 = 990px, margin = (1080-990)/2 = 45px
# First pip center = margin + pip_radius = 45 + 45 = 90
FIRST_PIP_X = 90  # First pip center X from container left
PIP_SPACING = 90  # Center-to-center spacing between pips

# Base offset at reference resolution (user offset multiplies this)
BASE_OFFSET = 50


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH UI - IMAGE LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _get_assets_path():
    """Get path to health sprites folder."""
    return os.path.join(
        os.path.dirname(__file__), "..", "exp_assets", "Health Sprites"
    )


def _load_health_images():
    """Load all health bar images. Called once when UI is enabled."""
    global _images
    _images.clear()

    assets_dir = _get_assets_path()

    files = {
        "container": "health_container.png",
        "full": "health_full.png",
        "half": "health_half.png",
        "empty": "health_empty.png",
        "cross": "health_circle_and_cross.png",
    }

    loaded = 0
    for key, filename in files.items():
        img_path = os.path.join(assets_dir, filename)
        if os.path.exists(img_path):
            # Load image, reuse if already loaded
            img = bpy.data.images.load(img_path, check_existing=True)
            img.gl_load()  # Upload to GPU
            _images[key] = img
            loaded += 1
        else:
            log_game("HEALTH", f"UI_MISSING image: {img_path}")

    log_game("HEALTH", f"UI_LOADED {loaded}/{len(files)} health bar images")
    return loaded == len(files)


def _unload_health_images():
    """Unload images from GPU (not from bpy.data)."""
    for img in _images.values():
        if img:
            try:
                img.gl_free()
            except Exception:
                pass
    _images.clear()


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH UI - GPU CACHE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def _get_shader():
    """Get cached shader (created once)."""
    global _cached_shader
    if _cached_shader is None:
        _cached_shader = gpu.shader.from_builtin('IMAGE')
    return _cached_shader


def _get_texture(key):
    """Get cached GPU texture for an image key."""
    global _cached_textures
    if key not in _cached_textures:
        img = _images.get(key)
        if img:
            _cached_textures[key] = gpu.texture.from_image(img)
    return _cached_textures.get(key)


def _clear_gpu_cache():
    """Clear all cached GPU resources. Call on game end or config change."""
    global _cached_shader, _cached_textures, _cached_batches, _cache_state
    _cached_shader = None
    _cached_textures.clear()
    _cached_batches.clear()
    _cache_state = None


def _invalidate_batches():
    """Invalidate batch cache (keeps shader/textures). Call on health change."""
    global _cached_batches, _cache_state
    _cached_batches.clear()
    _cache_state = None


def _get_config_hash():
    """Generate hash of current config for cache invalidation."""
    if not _ui_config:
        return ""
    return f"{_ui_config['position']}_{_ui_config['scale']}_{_ui_config['offset_x']}_{_ui_config['offset_y']}"


def _build_batches(region, filled_units: int):
    """
    Build cached batches for all health bar elements.
    Groups quads by texture type for efficient drawing.
    Returns dict of {texture_key: (batch, texture)}.
    """
    global _cached_batches, _cache_state

    position = _ui_config["position"]
    user_scale = _ui_config["scale"]
    offset_x = _ui_config["offset_x"]
    offset_y = _ui_config["offset_y"]

    # Compute scaling
    screen_scale_w = region.width / REF_WIDTH
    screen_scale_h = region.height / REF_HEIGHT
    screen_scale = min(screen_scale_w, screen_scale_h)
    user_scale_mult = 0.25 + (user_scale - 1) * (0.75 / 19.0)
    total_scale = screen_scale * user_scale_mult

    # Scaled dimensions
    container_w = CONTAINER_WIDTH * total_scale
    container_h = CONTAINER_HEIGHT * total_scale
    pip_size = PIP_DISPLAY_SIZE * total_scale
    pip_spacing = PIP_SPACING * total_scale
    first_pip_x = FIRST_PIP_X * total_scale

    # Grid offsets
    grid_columns = 40
    grid_unit_x = (REF_WIDTH / grid_columns) * screen_scale
    grid_unit_y = (REF_HEIGHT / grid_columns) * screen_scale
    offset_x_px = offset_x * grid_unit_x
    offset_y_px = offset_y * grid_unit_y

    is_vertical = position in ("LEFT", "RIGHT")

    # Calculate bar position
    if position == "BOTTOM":
        bar_x = (region.width - container_w) / 2 + offset_x_px
        bar_y = offset_y_px
        rotation = 0
    elif position == "TOP":
        bar_x = (region.width - container_w) / 2 + offset_x_px
        bar_y = region.height - container_h + offset_y_px
        rotation = 0
    elif position == "LEFT":
        visual_center_x = container_h / 2 + offset_x_px
        visual_center_y = region.height / 2 + offset_y_px
        bar_x = visual_center_x - container_w / 2
        bar_y = visual_center_y - container_h / 2
        rotation = 90
    elif position == "RIGHT":
        visual_center_x = region.width - container_h / 2 + offset_x_px
        visual_center_y = region.height / 2 + offset_y_px
        bar_x = visual_center_x - container_w / 2
        bar_y = visual_center_y - container_h / 2
        rotation = -90
    else:
        return {}

    shader = _get_shader()
    if not shader:
        return {}

    # Collect quads grouped by texture type
    # Format: {texture_key: [(verts, uvs), ...]}
    quad_groups = {"container": [], "full": [], "half": [], "empty": [], "cross": []}

    # Container quad
    container_verts, container_uvs = _make_quad_data(bar_x, bar_y, container_w, container_h, rotation)
    quad_groups["container"].append((container_verts, container_uvs))

    # Calculate visual bounds for vertical bars
    if is_vertical:
        visual_center_x = bar_x + container_w / 2
        visual_center_y = bar_y + container_h / 2

    # Pip quads
    for i in range(11):
        if is_vertical:
            screen_x = visual_center_x - pip_size / 2
            pip_offset_along_bar = first_pip_x + (i * pip_spacing)
            screen_y = (visual_center_y - container_w / 2) + pip_offset_along_bar - pip_size / 2
        else:
            pip_center_x = first_pip_x + (i * pip_spacing)
            pip_center_y = container_h / 2
            screen_x = bar_x + pip_center_x - pip_size / 2
            screen_y = bar_y + pip_center_y - pip_size / 2

        # Determine texture key
        if i == 10:
            tex_key = "cross"
        else:
            full_threshold = (i + 1) * 2
            half_threshold = i * 2 + 1
            if filled_units >= full_threshold:
                tex_key = "full"
            elif filled_units >= half_threshold:
                tex_key = "half"
            else:
                tex_key = "empty"

        verts, uvs = _make_quad_data(screen_x, screen_y, pip_size, pip_size, 0)
        quad_groups[tex_key].append((verts, uvs))

    # Build batches for each texture type
    # Each texture type gets a list of (batch, texture) for its quads
    batches = {}
    for tex_key, quads in quad_groups.items():
        if not quads:
            continue
        texture = _get_texture(tex_key)
        if not texture:
            continue

        # Build individual batches for each quad (TRI_FAN is simpler and reliable)
        quad_batches = []
        for verts, uvs in quads:
            batch = batch_for_shader(shader, 'TRI_FAN', {"pos": verts, "texCoord": uvs})
            quad_batches.append(batch)
        batches[tex_key] = (quad_batches, texture)

    # Update cache state
    _cache_state = {
        "filled_units": filled_units,
        "config_hash": _get_config_hash(),
        "screen": (region.width, region.height)
    }
    _cached_batches = batches

    return batches


def _make_quad_data(x, y, width, height, rotation):
    """Generate vertex and UV data for a quad."""
    if rotation == 0:
        verts = [
            (x, y),
            (x + width, y),
            (x + width, y + height),
            (x, y + height),
        ]
    else:
        cx = x + width / 2
        cy = y + height / 2
        angle = math.radians(rotation)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        corners = [
            (-width / 2, -height / 2),
            (width / 2, -height / 2),
            (width / 2, height / 2),
            (-width / 2, height / 2),
        ]
        verts = []
        for dx, dy in corners:
            rx = dx * cos_a - dy * sin_a
            ry = dx * sin_a + dy * cos_a
            verts.append((cx + rx, cy + ry))

    uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
    return verts, uvs


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH UI - ENABLE / DISABLE
# ══════════════════════════════════════════════════════════════════════════════

def enable_health_ui(target_obj_name: str, position: str = "BOTTOM",
                     scale: int = 10, offset_x: int = 0, offset_y: int = 1):
    """
    Enable health bar display for an object with health.

    Args:
        target_obj_name: Name of object to track health for
        position: BOTTOM, TOP, LEFT, or RIGHT
        scale: Size (1-20, where 10 is default/normal size)
        offset_x: Horizontal grid offset (-20 to 20)
        offset_y: Vertical grid offset (-20 to 20)
    """
    global _ui_config, _draw_handle

    # Load images if not already loaded
    if not _images:
        if not _load_health_images():
            log_game("HEALTH", "UI_ENABLE_FAILED missing images")
            return

    _ui_config = {
        "target": target_obj_name,
        "position": position.upper(),
        "scale": max(1, min(20, scale)),
        "offset_x": max(-20, min(20, offset_x)),
        "offset_y": max(-20, min(20, offset_y)),
    }

    if _draw_handle is None:
        _draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            _draw, (), 'WINDOW', 'POST_PIXEL'
        )

    log_game("HEALTH", f"UI_ENABLE target={target_obj_name} pos={position} scale={scale} offset=({offset_x},{offset_y})")


def disable_health_ui():
    """Remove health bar overlay and cleanup all GPU resources."""
    global _draw_handle, _ui_config

    if _draw_handle is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, 'WINDOW')
        except Exception:
            pass
        _draw_handle = None

    _ui_config = None
    _clear_gpu_cache()  # Clear cached shader, textures, batches
    _unload_health_images()
    log_game("HEALTH", "UI_DISABLE")


def is_health_ui_enabled() -> bool:
    """Check if health UI is currently enabled."""
    return _draw_handle is not None


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH UI - DRAWING
# ══════════════════════════════════════════════════════════════════════════════

def _find_window_region():
    """Return (region, area) for the first VIEW_3D WINDOW region."""
    try:
        win = bpy.context.window
        if not win or not win.screen:
            return None, None
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                for reg in area.regions:
                    if reg.type == 'WINDOW':
                        return reg, area
    except Exception:
        # Context access can fail during interpreter state transitions
        pass
    return None, None


def _draw():
    """GPU draw callback - render health bar using cached batches."""
    try:
        if not _ui_config or not _images:
            return

        # Get health data
        health_data = get_health(_ui_config["target"])
        if not health_data:
            return

        # Normalize health to 0-20 units
        current = health_data["current"]
        min_val = health_data["min"]
        max_val = health_data["max"]
        range_val = max_val - min_val

        if range_val <= 0:
            filled_units = 20
        else:
            filled_units = round((current - min_val) / range_val * 20)
        filled_units = max(0, min(20, filled_units))

        # Get screen region
        region, _ = _find_window_region()
        if not region:
            return

        # Check if cache is valid
        needs_rebuild = (
            _cache_state is None or
            _cache_state["filled_units"] != filled_units or
            _cache_state["config_hash"] != _get_config_hash() or
            _cache_state["screen"] != (region.width, region.height)
        )

        # Get or rebuild batches
        if needs_rebuild:
            batches = _build_batches(region, filled_units)
        else:
            batches = _cached_batches

        if not batches:
            return

        # Draw all batches (grouped by texture for fewer texture binds)
        shader = _get_shader()
        if not shader:
            return

        gpu.state.blend_set('ALPHA')
        try:
            shader.bind()
            for tex_key, (quad_batches, texture) in batches.items():
                shader.uniform_sampler("image", texture)
                for batch in quad_batches:
                    batch.draw(shader)
        finally:
            gpu.state.blend_set('NONE')

    except Exception:
        # Silently skip draw if context is unavailable (editor transitions)
        pass


