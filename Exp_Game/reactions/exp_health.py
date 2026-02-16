# Exp_Game/reactions/exp_health.py
"""
Health System - Track health for any object in the scene.

Consolidated module containing:
- Core health cache and operations
- Reaction executors (ENABLE_HEALTH, DISPLAY_HEALTH_UI)
- GPU-rendered health bar UI overlay (HUD sprite bar for character)
- World-space health display (floating bar/numbers for non-character objects)

Architecture:
    1. ENABLE_HEALTH reaction fires -> enable_health() adds object to cache
    2. Runtime: modify_health() / set_health() change values (future damage reactions)
    3. Game Reset: reset_all_health() restores start values
    4. Game End: clear_all_health() empties cache

The cache uses object names (strings) as keys for worker-serializable data.
"""

import bpy
import blf
import gpu
import os
import math
from gpu_extras.batch import batch_for_shader
from bpy_extras.view3d_utils import location_3d_to_region_2d
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
    _world_configs.clear()
    _invalidate_batches()  # Clear batch cache
    log_game("HEALTH", f"CLEAR cache emptied (was tracking {count} objects)")


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


def execute_adjust_health_reaction(r):
    """
    Execute ADJUST_HEALTH reaction.
    Adds or subtracts from an object's current health.
    Silently does nothing if the object has no health enabled.
    """
    from .exp_bindings import resolve_object, resolve_float

    obj = resolve_object(r, "adjust_health_target_object", r.adjust_health_target_object)
    if not obj:
        return

    if not is_health_enabled(obj.name):
        return

    amount = resolve_float(r, "adjust_health_amount", r.adjust_health_amount)
    modify_health(obj.name, amount)


def execute_display_health_ui_reaction(r):
    """
    Execute DISPLAY_HEALTH_UI reaction.
    Routes to HUD (character) or world-space (non-character) display.
    """
    from .exp_bindings import resolve_object, resolve_int, resolve_raw_binding

    # Resolve object from binding (node connection) or fallback to direct property
    obj = resolve_object(r, "health_ui_target_object", r.health_ui_target_object)
    if not obj:
        log_game("HEALTH", "UI_SKIP no target object specified")
        return

    # Check if target is the character via raw binding value
    raw_value = resolve_raw_binding(r, "health_ui_target_object")

    if raw_value == "__CHARACTER__":
        # HUD path — character target
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
    else:
        # World-space path — non-character target
        world_scale = resolve_int(r, "health_ui_world_scale", r.health_ui_world_scale)
        world_offset_h = resolve_int(r, "health_ui_world_offset_h", r.health_ui_world_offset_h)
        world_offset_v = resolve_int(r, "health_ui_world_offset_v", r.health_ui_world_offset_v)

        enable_world_health_ui(
            obj.name,
            style=r.health_ui_world_style,
            scale=world_scale,
            offset_h=world_offset_h,
            offset_v=world_offset_v,
            show_through=r.health_ui_world_show_through
        )


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH UI - STATE
# ══════════════════════════════════════════════════════════════════════════════

_draw_handle = None
_hud_config = None
_images = {}  # {"container": bpy.types.Image, "full": Image, "half": Image, "empty": Image, "cross": Image}

# World-space configs: {obj_name: {"style": str, "scale": float, "offset_x": float, "offset_y": float}}
_world_configs: dict = {}

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
    if not _hud_config:
        return ""
    return f"{_hud_config['position']}_{_hud_config['scale']}_{_hud_config['offset_x']}_{_hud_config['offset_y']}"


def _build_batches(region, filled_units: int):
    """
    Build cached batches for all health bar elements.
    Groups quads by texture type for efficient drawing.
    Returns dict of {texture_key: (batch, texture)}.
    """
    global _cached_batches, _cache_state

    position = _hud_config["position"]
    user_scale = _hud_config["scale"]
    offset_x = _hud_config["offset_x"]
    offset_y = _hud_config["offset_y"]

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
# HEALTH UI - ENABLE / DISABLE (HUD)
# ══════════════════════════════════════════════════════════════════════════════

def enable_health_ui(target_obj_name: str, position: str = "BOTTOM",
                     scale: int = 10, offset_x: int = 0, offset_y: int = 1):
    """
    Enable HUD health bar display for an object with health (character).

    Args:
        target_obj_name: Name of object to track health for
        position: BOTTOM, TOP, LEFT, or RIGHT
        scale: Size (1-20, where 10 is default/normal size)
        offset_x: Horizontal grid offset (-20 to 20)
        offset_y: Vertical grid offset (-20 to 20)
    """
    global _hud_config, _draw_handle

    # Load images if not already loaded
    if not _images:
        if not _load_health_images():
            log_game("HEALTH", "UI_ENABLE_FAILED missing images")
            return

    _hud_config = {
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
    """Remove HUD health bar overlay and cleanup all GPU resources."""
    global _draw_handle, _hud_config

    _hud_config = None
    _world_configs.clear()

    if _draw_handle is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, 'WINDOW')
        except Exception:
            pass
        _draw_handle = None

    _clear_gpu_cache()  # Clear cached shader, textures, batches
    _unload_health_images()
    log_game("HEALTH", "UI_DISABLE")


# ══════════════════════════════════════════════════════════════════════════════
# WORLD-SPACE HEALTH UI - ENABLE / DISABLE
# ══════════════════════════════════════════════════════════════════════════════

def enable_world_health_ui(obj_name: str, style: str = "BAR",
                           scale: int = 10, offset_h: int = 0,
                           offset_v: int = 2, show_through: bool = False):
    """
    Enable world-space health display for a non-character object.

    Args:
        obj_name: Name of the object to display health for
        style: "BAR" or "NUMBERS"
        scale: Size (1-20, where 10 is default/normal size)
        offset_h: Horizontal offset in grid units (-20 to 20)
        offset_v: Vertical offset in grid units (-20 to 20)
        show_through: If True, display through meshes (ignore occlusion)
    """
    global _draw_handle

    _world_configs[obj_name] = {
        "style": style.upper(),
        "scale": max(1, min(20, scale)),
        "offset_h": max(-20, min(20, offset_h)),
        "offset_v": max(-20, min(20, offset_v)),
        "show_through": bool(show_through),
    }

    # Ensure draw handler is registered
    if _draw_handle is None:
        _draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            _draw, (), 'WINDOW', 'POST_PIXEL'
        )

    log_game("HEALTH", f"WORLD_UI_ENABLE obj={obj_name} style={style} scale={scale} offset=({offset_h},{offset_v})")


def disable_world_health_ui(obj_name: str):
    """Remove world-space health display for one object."""
    global _draw_handle

    if obj_name in _world_configs:
        del _world_configs[obj_name]
        log_game("HEALTH", f"WORLD_UI_DISABLE obj={obj_name}")

    # Remove draw handler if nothing left to draw
    if not _hud_config and not _world_configs and _draw_handle is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, 'WINDOW')
        except Exception:
            pass
        _draw_handle = None


def disable_all_world_health_ui():
    """Remove all world-space health displays."""
    global _draw_handle

    count = len(_world_configs)
    _world_configs.clear()

    # Remove draw handler if nothing left to draw
    if not _hud_config and _draw_handle is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, 'WINDOW')
        except Exception:
            pass
        _draw_handle = None

    if count > 0:
        log_game("HEALTH", f"WORLD_UI_DISABLE_ALL cleared {count} entries")


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH UI - DRAWING (Dispatch)
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
    """GPU draw callback - dispatches to HUD and world-space drawing."""
    try:
        _draw_hud()
        _draw_all_world_health()
    except Exception:
        # Silently skip draw if context is unavailable (editor transitions)
        pass


def _draw_hud():
    """Render HUD sprite-based health bar (character)."""
    if not _hud_config or not _images:
        return

    # Get health data
    health_data = get_health(_hud_config["target"])
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


# ══════════════════════════════════════════════════════════════════════════════
# WORLD-SPACE HEALTH - DRAWING
# ══════════════════════════════════════════════════════════════════════════════

def _health_fraction(obj_name: str) -> float | None:
    """Get health as 0.0-1.0 fraction. Returns None if not tracked."""
    data = _health_cache.get(obj_name)
    if not data:
        return None
    range_val = data["max"] - data["min"]
    if range_val <= 0:
        return 1.0
    return max(0.0, min(1.0, (data["current"] - data["min"]) / range_val))


def _health_color(fraction: float) -> tuple:
    """Interpolate green (1.0) -> red (0.0) based on health fraction."""
    r = 0.2 + (1.0 - fraction) * 0.6
    g = 0.2 + fraction * 0.6
    return (r, g, 0.2, 1.0)


def _draw_all_world_health():
    """Iterate world configs and draw each object's health display."""
    if not _world_configs:
        return

    region, area = _find_window_region()
    if not region or not area:
        return

    # Get the 3D region data for projection
    rv3d = None
    for space in area.spaces:
        if space.type == 'VIEW_3D':
            rv3d = space.region_3d
            break
    if not rv3d:
        return

    for obj_name, config in _world_configs.items():
        if config["style"] == "BAR":
            _draw_world_health_bar(region, rv3d, obj_name, config)
        else:
            _draw_world_health_numbers(region, rv3d, obj_name, config)


def _is_point_occluded(region, rv3d, world_pos) -> bool:
    """Check if a world-space point is hidden behind scene geometry using the depth buffer."""
    from mathutils import Vector

    screen_pos = location_3d_to_region_2d(region, rv3d, world_pos)
    if not screen_pos:
        return True

    sx = int(max(0, min(screen_pos.x, region.width - 1)))
    sy = int(max(0, min(screen_pos.y, region.height - 1)))

    try:
        fb = gpu.state.active_framebuffer_get()
        depth_buf = fb.read_depth(sx, sy, 1, 1)
        scene_depth = depth_buf.to_list()[0][0]
    except Exception:
        return False  # Can't read depth — assume visible

    # Calculate the point's depth in normalized device coordinates
    p = rv3d.perspective_matrix @ Vector((world_pos.x, world_pos.y, world_pos.z, 1.0))
    if abs(p.w) < 1e-6:
        return True
    point_depth = (p.z / p.w) * 0.5 + 0.5

    # Scene depth < point depth means something closer is in front
    return scene_depth < (point_depth - 0.0001)


def _world_offset_from_grid(offset_units: int) -> float:
    """Convert grid units to Blender world units (0.1 BU per grid unit)."""
    return offset_units * 0.1


def _world_scale_mult(scale: int) -> float:
    """Convert scale (1-20) to a multiplier. 1 = 0.5x, 10 = 2.5x, 20 = 5.0x."""
    return 0.5 + (scale - 1) * (4.5 / 19.0)


def _draw_world_health_bar(region, rv3d, obj_name: str, config: dict):
    """Draw a floating health bar above an object in world space."""
    obj = bpy.data.objects.get(obj_name)
    if not obj:
        return

    fraction = _health_fraction(obj_name)
    if fraction is None:
        return

    # Calculate world position with grid-unit offset
    world_pos = obj.matrix_world.translation.copy()
    world_pos.x += _world_offset_from_grid(config["offset_h"])
    world_pos.z += _world_offset_from_grid(config["offset_v"])

    # Occlusion check — skip if behind geometry
    if not config.get("show_through", False):
        if _is_point_occluded(region, rv3d, world_pos):
            return

    # Project to 2D screen space
    screen_pos = location_3d_to_region_2d(region, rv3d, world_pos)
    if not screen_pos:
        return

    sx, sy = screen_pos
    scale_mult = _world_scale_mult(config["scale"])

    # Bar dimensions (pixels)
    bar_w = 80.0 * scale_mult
    bar_h = 10.0 * scale_mult

    # Background rect (dark gray)
    bg_x = sx - bar_w / 2
    bg_y = sy - bar_h / 2

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')

    try:
        # Background
        bg_verts = [
            (bg_x, bg_y),
            (bg_x + bar_w, bg_y),
            (bg_x + bar_w, bg_y + bar_h),
            (bg_x, bg_y + bar_h),
        ]
        bg_batch = batch_for_shader(shader, 'TRI_FAN', {"pos": bg_verts})
        shader.bind()
        shader.uniform_float('color', (0.2, 0.2, 0.2, 0.7))
        bg_batch.draw(shader)

        # Fill rect (green->red based on health)
        if fraction > 0:
            fill_w = bar_w * fraction
            fill_verts = [
                (bg_x, bg_y),
                (bg_x + fill_w, bg_y),
                (bg_x + fill_w, bg_y + bar_h),
                (bg_x, bg_y + bar_h),
            ]
            fill_batch = batch_for_shader(shader, 'TRI_FAN', {"pos": fill_verts})
            shader.bind()
            shader.uniform_float('color', _health_color(fraction))
            fill_batch.draw(shader)
    finally:
        gpu.state.blend_set('NONE')


def _draw_world_health_numbers(region, rv3d, obj_name: str, config: dict):
    """Draw floating current health number above an object in world space."""
    obj = bpy.data.objects.get(obj_name)
    if not obj:
        return

    data = _health_cache.get(obj_name)
    if not data:
        return

    fraction = _health_fraction(obj_name)
    if fraction is None:
        return

    # Calculate world position with grid-unit offset
    world_pos = obj.matrix_world.translation.copy()
    world_pos.x += _world_offset_from_grid(config["offset_h"])
    world_pos.z += _world_offset_from_grid(config["offset_v"])

    # Occlusion check — skip if behind geometry
    if not config.get("show_through", False):
        if _is_point_occluded(region, rv3d, world_pos):
            return

    # Project to 2D screen space
    screen_pos = location_3d_to_region_2d(region, rv3d, world_pos)
    if not screen_pos:
        return

    sx, sy = screen_pos
    scale_mult = _world_scale_mult(config["scale"])

    # Draw text using BLF
    font_id = 0
    font_size = int(16 * scale_mult)
    text = f"{int(data['current'])}"

    color = _health_color(fraction)
    blf.size(font_id, font_size)
    blf.color(font_id, color[0], color[1], color[2], color[3])

    # Center text horizontally
    text_w, text_h = blf.dimensions(font_id, text)
    blf.position(font_id, sx - text_w / 2, sy - text_h / 2, 0)

    gpu.state.blend_set('ALPHA')
    try:
        blf.draw(font_id, text)
    finally:
        gpu.state.blend_set('NONE')
