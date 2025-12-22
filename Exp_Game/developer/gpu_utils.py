# Exp_Game/developer/gpu_utils.py
"""
Optimized GPU utilities for debug visualizers.

PERFORMANCE OPTIMIZATIONS:
1. Shader cached once, reused forever
2. Pre-computed circle lookup tables (no trig in draw loops)
3. Batch caching with dirty flags (only rebuild when data changes)

Usage:
    from ..developer.gpu_utils import (
        get_cached_shader,
        CIRCLE_8, CIRCLE_12,
        circle_verts_xy, circle_verts_xz,
    )
"""

import math
import gpu
from gpu_extras.batch import batch_for_shader


# ═══════════════════════════════════════════════════════════════════════════════
# CACHED SHADER (Avoid gpu.shader.from_builtin() every frame)
# ═══════════════════════════════════════════════════════════════════════════════

_cached_flat_color_shader = None


def get_cached_shader():
    """
    Get the cached FLAT_COLOR shader.

    First call: ~50-100µs (shader lookup)
    Subsequent calls: ~0.1µs (cached reference)
    """
    global _cached_flat_color_shader
    if _cached_flat_color_shader is None:
        _cached_flat_color_shader = gpu.shader.from_builtin('FLAT_COLOR')
    return _cached_flat_color_shader


def invalidate_shader_cache():
    """Call this if Blender reports shader issues (rare)."""
    global _cached_flat_color_shader
    _cached_flat_color_shader = None


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-COMPUTED CIRCLE LOOKUP TABLES
# ═══════════════════════════════════════════════════════════════════════════════
#
# Instead of computing cos/sin for each segment every frame,
# we pre-compute them once at module load time.
#
# Structure: List of (cos, sin) tuples for angles 0 to 2π
# Includes segment+1 entries so we can do LUT[i] to LUT[i+1] without wrapping

def _build_circle_lut(segments: int) -> tuple:
    """Build lookup table for circle with N segments."""
    return tuple(
        (math.cos(i * 2.0 * math.pi / segments),
         math.sin(i * 2.0 * math.pi / segments))
        for i in range(segments + 1)  # +1 for wraparound
    )


# Pre-computed at module load (zero cost during gameplay)
CIRCLE_8 = _build_circle_lut(8)    # Low quality (fast)
CIRCLE_12 = _build_circle_lut(12)  # Medium quality
CIRCLE_16 = _build_circle_lut(16)  # High quality (slower)


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMIZED CIRCLE VERTEX GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def circle_verts_xy(center: tuple, radius: float, lut: tuple = CIRCLE_8) -> list:
    """
    Generate line vertices for a circle in the XY plane (horizontal).

    Returns list of vertex pairs: [(p1, p2), (p2, p3), ...]
    Uses pre-computed lookup table - NO trig calls.

    Args:
        center: (x, y, z) center position
        radius: Circle radius
        lut: Circle lookup table (CIRCLE_8, CIRCLE_12, or CIRCLE_16)

    Returns:
        List of (start_point, end_point) tuples for LINES drawing
    """
    cx, cy, cz = center
    segments = len(lut) - 1
    verts = []

    for i in range(segments):
        cos1, sin1 = lut[i]
        cos2, sin2 = lut[i + 1]
        verts.append((
            (cx + radius * cos1, cy + radius * sin1, cz),
            (cx + radius * cos2, cy + radius * sin2, cz)
        ))

    return verts


def circle_verts_xz(center: tuple, radius: float, lut: tuple = CIRCLE_8) -> list:
    """Generate circle in XZ plane (vertical, front-facing)."""
    cx, cy, cz = center
    segments = len(lut) - 1
    verts = []

    for i in range(segments):
        cos1, sin1 = lut[i]
        cos2, sin2 = lut[i + 1]
        verts.append((
            (cx + radius * cos1, cy, cz + radius * sin1),
            (cx + radius * cos2, cy, cz + radius * sin2)
        ))

    return verts


def circle_verts_yz(center: tuple, radius: float, lut: tuple = CIRCLE_8) -> list:
    """Generate circle in YZ plane (vertical, side-facing)."""
    cx, cy, cz = center
    segments = len(lut) - 1
    verts = []

    for i in range(segments):
        cos1, sin1 = lut[i]
        cos2, sin2 = lut[i + 1]
        verts.append((
            (cx, cy + radius * cos1, cz + radius * sin1),
            (cx, cy + radius * cos2, cz + radius * sin2)
        ))

    return verts


def sphere_wire_verts(center: tuple, radius: float, lut: tuple = CIRCLE_8) -> list:
    """
    Generate wireframe sphere (3 circles: XY, XZ, YZ planes).

    Returns list of vertex pairs for LINES drawing.
    """
    verts = []
    verts.extend(circle_verts_xy(center, radius, lut))
    verts.extend(circle_verts_xz(center, radius, lut))
    verts.extend(circle_verts_yz(center, radius, lut))
    return verts


def layered_sphere_verts(
    center: tuple,
    radius: float,
    height_ratios: tuple = (-0.5, 0.0, 0.5),
    lut: tuple = CIRCLE_8
) -> list:
    """
    Generate layered sphere visualization (circles at different heights).

    Useful for showing reach limits, collision volumes, etc.

    Args:
        center: Sphere center
        radius: Sphere radius
        height_ratios: Z positions as ratios of radius (-1 to 1)
        lut: Circle lookup table

    Returns:
        List of vertex pairs for LINES drawing
    """
    cx, cy, cz = center
    verts = []

    for h in height_ratios:
        z_offset = radius * h
        # Circle radius at this height (using sphere equation)
        circle_r = radius * math.sqrt(max(0.0, 1.0 - h * h))

        if circle_r < 0.01:
            continue

        layer_center = (cx, cy, cz + z_offset)
        verts.extend(circle_verts_xy(layer_center, circle_r, lut))

    return verts


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH BUILDING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def extend_batch_data(
    all_verts: list,
    all_colors: list,
    vert_pairs: list,
    color: tuple
):
    """
    Add vertex pairs to batch data with a single color.

    Optimized to avoid multiple extend() calls.

    Args:
        all_verts: Master vertex list (modified in place)
        all_colors: Master color list (modified in place)
        vert_pairs: List of (start, end) vertex tuples
        color: RGBA color tuple
    """
    for p1, p2 in vert_pairs:
        all_verts.append(p1)
        all_verts.append(p2)
        all_colors.append(color)
        all_colors.append(color)


def build_and_draw_batch(shader, all_verts: list, all_colors: list):
    """
    Build batch and draw in one call.

    Args:
        shader: The cached shader from get_cached_shader()
        all_verts: List of vertex positions
        all_colors: List of RGBA colors (same length as all_verts)
    """
    if not all_verts:
        return

    batch = batch_for_shader(shader, 'LINES', {
        "pos": all_verts,
        "color": all_colors
    })
    shader.bind()
    batch.draw(shader)


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH CACHING (For static or slowly-changing visualizations)
# ═══════════════════════════════════════════════════════════════════════════════

class CachedBatch:
    """
    Cached GPU batch that only rebuilds when data changes.

    Usage:
        _batch_cache = CachedBatch()

        def draw():
            if data_changed:
                verts, colors = build_geometry()
                _batch_cache.update(verts, colors)

            _batch_cache.draw()
    """

    __slots__ = ('_batch', '_shader', '_version')

    def __init__(self):
        self._batch = None
        self._shader = None
        self._version = 0

    def update(self, verts: list, colors: list):
        """Rebuild the batch with new geometry."""
        if not verts:
            self._batch = None
            return

        self._shader = get_cached_shader()
        self._batch = batch_for_shader(
            self._shader, 'LINES',
            {"pos": verts, "color": colors}
        )
        self._version += 1

    def draw(self):
        """Draw the cached batch (fast - no geometry rebuild)."""
        if self._batch is None:
            return

        self._shader.bind()
        self._batch.draw(self._shader)

    def invalidate(self):
        """Force rebuild on next update."""
        self._batch = None

    @property
    def is_valid(self) -> bool:
        return self._batch is not None


# ═══════════════════════════════════════════════════════════════════════════════
# CROSSHAIR / MARKER HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def crosshair_verts(center: tuple, size: float = 0.03) -> list:
    """Generate 3D crosshair vertex pairs (6 vertices, 3 lines)."""
    cx, cy, cz = center
    return [
        ((cx - size, cy, cz), (cx + size, cy, cz)),  # X axis
        ((cx, cy - size, cz), (cx, cy + size, cz)),  # Y axis
        ((cx, cy, cz - size), (cx, cy, cz + size)),  # Z axis
    ]


def arrow_head_verts(tip: tuple, direction: tuple, size: float = 0.08) -> list:
    """
    Generate arrow head vertex pairs.

    Args:
        tip: Arrow tip position
        direction: Normalized direction vector (pointing toward tip)
        size: Arrow head size

    Returns:
        List of 2 vertex pairs for the arrow head lines
    """
    dx, dy, dz = direction

    # Find perpendicular vector
    if abs(dz) < 0.9:
        perp = (-dy, dx, 0.0)
    else:
        perp = (1.0, 0.0, 0.0)

    # Normalize perpendicular
    perp_len = math.sqrt(perp[0]**2 + perp[1]**2 + perp[2]**2)
    if perp_len < 0.001:
        return []

    px, py, pz = perp[0]/perp_len, perp[1]/perp_len, perp[2]/perp_len

    # Back point
    back = (
        tip[0] - dx * size,
        tip[1] - dy * size,
        tip[2] - dz * size
    )

    # Wing points
    wing_size = size * 0.5
    head1 = (back[0] + px * wing_size, back[1] + py * wing_size, back[2] + pz * wing_size)
    head2 = (back[0] - px * wing_size, back[1] - py * wing_size, back[2] - pz * wing_size)

    return [(tip, head1), (tip, head2)]
