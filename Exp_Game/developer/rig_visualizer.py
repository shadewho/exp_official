# Exp_Game/developer/rig_visualizer.py
"""
Comprehensive Rig Visualizer - Shows animation system state in 3D viewport.

FEATURES:
- Bone group visualization (color bones by group)
- IK chain visualization (targets, poles, reach spheres)
- Animation layer stack (active layers, weights, masks)
- Billboard text overlay (state info above character)
- Independent of N-panel selection - shows when enabled

PERFORMANCE:
- Single batched draw call per category
- Cached shader and pre-computed LUTs
- blf text rendering (Blender's native, very fast)
- Only draws enabled categories
- Minimal per-frame allocations

Toggle via: scene.dev_rig_visualizer_enabled
"""

import bpy
import blf
import gpu
import math
import numpy as np
from gpu_extras.batch import batch_for_shader
from bpy_extras.view3d_utils import location_3d_to_region_2d
from typing import Dict, List, Optional, Tuple
from mathutils import Vector, Matrix

from .gpu_utils import (
    get_cached_shader,
    CIRCLE_8, CIRCLE_12,
    sphere_wire_verts,
    layered_sphere_verts,
    extend_batch_data,
    crosshair_verts,
    arrow_head_verts,
    circle_verts_xy,
)

from ..animations.bone_groups import (
    BONE_INDEX, INDEX_TO_BONE, TOTAL_BONES,
    BONE_GROUPS, BlendMasks,
)


# =============================================================================
# DRAW HANDLER STATE
# =============================================================================

_draw_handler = None
_text_draw_handler = None  # Separate handler for POST_PIXEL (2D text)
_cached_armature_name: str = None

# Cached font ID (avoid lookup every frame)
_font_id = 0


# =============================================================================
# COLOR PALETTE (Distinct colors for visualization)
# =============================================================================

# Bone group colors (high saturation, good contrast)
GROUP_COLORS = {
    # Major regions
    "UPPER_BODY": (0.2, 0.6, 1.0, 0.8),    # Blue
    "LOWER_BODY": (0.2, 1.0, 0.4, 0.8),    # Green
    "SPINE": (1.0, 0.8, 0.2, 0.8),          # Yellow
    "HEAD_NECK": (1.0, 0.4, 0.8, 0.8),      # Pink

    # Arms
    "ARM_L": (0.0, 0.8, 1.0, 0.8),          # Cyan
    "ARM_R": (0.0, 0.8, 1.0, 0.8),          # Cyan
    "ARM_L_IK": (0.0, 1.0, 1.0, 0.9),       # Bright cyan
    "ARM_R_IK": (0.0, 1.0, 1.0, 0.9),       # Bright cyan

    # Legs
    "LEG_L": (0.6, 1.0, 0.2, 0.8),          # Lime
    "LEG_R": (0.6, 1.0, 0.2, 0.8),          # Lime
    "LEG_L_IK": (0.8, 1.0, 0.0, 0.9),       # Bright lime
    "LEG_R_IK": (0.8, 1.0, 0.0, 0.9),       # Bright lime

    # Extremities
    "HAND_L": (1.0, 0.5, 0.0, 0.8),         # Orange
    "HAND_R": (1.0, 0.5, 0.0, 0.8),         # Orange
    "FINGERS": (1.0, 0.3, 0.3, 0.7),        # Red-orange
    "FOOT_L": (0.5, 0.0, 1.0, 0.8),         # Purple
    "FOOT_R": (0.5, 0.0, 1.0, 0.8),         # Purple

    # Root
    "ROOT": (1.0, 1.0, 1.0, 0.9),           # White
    "HIPS": (1.0, 1.0, 1.0, 0.9),           # White
}

# Layer type colors
LAYER_COLORS = {
    "BASE": (0.2, 0.8, 0.2, 0.9),           # Green
    "ADDITIVE": (0.2, 0.6, 1.0, 0.9),       # Blue
    "OVERRIDE": (1.0, 0.4, 0.2, 0.9),       # Orange-red
}

# IK state colors
IK_COLORS = {
    "reachable": (0.2, 1.0, 0.2, 0.9),      # Green
    "at_limit": (1.0, 1.0, 0.0, 0.9),       # Yellow
    "out_of_reach": (1.0, 0.2, 0.2, 0.9),   # Red
    "pole": (1.0, 0.5, 0.0, 0.9),           # Orange
    "chain_upper": (0.0, 1.0, 1.0, 0.9),    # Cyan
    "chain_lower": (1.0, 0.0, 1.0, 0.9),    # Magenta
}

# Full-Body IK colors
FBIK_COLORS = {
    "hips": (1.0, 1.0, 0.0, 0.9),           # Yellow - hips control
    "hips_drop": (1.0, 0.5, 0.0, 0.7),      # Orange - hips drop indicator
    "foot_target": (0.2, 1.0, 0.2, 0.9),    # Green - foot grounded
    "hand_target": (0.2, 0.8, 1.0, 0.9),    # Cyan - hand reach
    "hand_limit": (1.0, 0.6, 0.0, 0.9),     # Orange - hand at limit
    "look_at": (1.0, 0.0, 1.0, 0.9),        # Magenta - look-at target
    "spine_lean": (0.8, 0.8, 0.2, 0.7),     # Pale yellow - spine direction
}


# =============================================================================
# MAIN DRAW CALLBACK
# =============================================================================

def _draw_rig_visualizer():
    """Main GPU draw callback - dispatches to sub-visualizers."""
    scene = bpy.context.scene

    # Master toggle
    if not getattr(scene, 'dev_rig_visualizer_enabled', False):
        return

    # Get target armature
    armature = _get_target_armature(scene)
    if not armature:
        return

    # Set GPU state
    gpu.state.depth_test_set('NONE')  # Always on top
    gpu.state.blend_set('ALPHA')
    line_width = getattr(scene, 'dev_rig_vis_line_width', 2.0)
    gpu.state.line_width_set(line_width)

    shader = get_cached_shader()

    all_verts = []
    all_colors = []

    # Dispatch to enabled visualizers
    if getattr(scene, 'dev_rig_vis_bone_groups', False):
        _draw_bone_groups(armature, all_verts, all_colors, scene)

    if getattr(scene, 'dev_rig_vis_ik_chains', False):
        _draw_ik_chains(armature, all_verts, all_colors, scene)

    if getattr(scene, 'dev_rig_vis_ik_targets', False):
        _draw_ik_targets(armature, all_verts, all_colors, scene)

    if getattr(scene, 'dev_rig_vis_ik_poles', False):
        _draw_ik_poles(armature, all_verts, all_colors, scene)

    if getattr(scene, 'dev_rig_vis_ik_reach', False):
        _draw_ik_reach(armature, all_verts, all_colors, scene)

    if getattr(scene, 'dev_rig_vis_bone_axes', False):
        _draw_bone_axes(armature, all_verts, all_colors, scene)

    if getattr(scene, 'dev_rig_vis_active_mask', False):
        _draw_active_mask(armature, all_verts, all_colors, scene)

    if getattr(scene, 'dev_rig_vis_full_body_ik', False):
        _draw_full_body_ik(armature, all_verts, all_colors, scene)

    # Single batched draw
    if all_verts:
        batch = batch_for_shader(shader, 'LINES', {"pos": all_verts, "color": all_colors})
        shader.bind()
        batch.draw(shader)

    # Reset GPU state
    gpu.state.line_width_set(1.0)
    gpu.state.depth_test_set('NONE')
    gpu.state.blend_set('NONE')


def _get_target_armature(scene) -> Optional[bpy.types.Object]:
    """Get the armature to visualize."""
    # Priority 1: Explicit target in scene
    target = getattr(scene, 'target_armature', None)
    if target and target.type == 'ARMATURE':
        return target

    # Priority 2: Active object if armature
    obj = bpy.context.active_object
    if obj and obj.type == 'ARMATURE':
        return obj

    # Priority 3: First armature in scene
    for obj in scene.objects:
        if obj.type == 'ARMATURE':
            return obj

    return None


# =============================================================================
# BONE GROUP VISUALIZATION
# =============================================================================

def _draw_bone_groups(armature, all_verts, all_colors, scene):
    """Draw bones colored by their group membership."""
    selected_group = getattr(scene, 'dev_rig_vis_selected_group', 'ALL')

    pose_bones = armature.pose.bones
    arm_matrix = armature.matrix_world

    # Determine which bones to highlight
    if selected_group == 'ALL':
        # Show all bones with group-based colors
        bone_colors = _compute_bone_group_colors()
    else:
        # Highlight only the selected group
        bone_colors = {}
        group_bones = BONE_GROUPS.get(selected_group, [])
        color = GROUP_COLORS.get(selected_group, (1.0, 1.0, 1.0, 0.8))
        for bone_name in group_bones:
            bone_colors[bone_name] = color

    # Draw each bone as a line from head to tail
    for bone_name, color in bone_colors.items():
        pose_bone = pose_bones.get(bone_name)
        if not pose_bone:
            continue

        # World positions
        head = tuple((arm_matrix @ pose_bone.head)[:])
        tail = tuple((arm_matrix @ pose_bone.tail)[:])

        all_verts.extend([head, tail])
        all_colors.extend([color, color])

        # Add a small crosshair at head for visibility
        size = 0.02
        extend_batch_data(all_verts, all_colors, crosshair_verts(head, size), color)


def _compute_bone_group_colors() -> Dict[str, Tuple]:
    """Compute a color for each bone based on its primary group."""
    # Priority order for group assignment (most specific first)
    group_priority = [
        "ROOT", "FINGERS", "HAND_L", "HAND_R", "FOOT_L", "FOOT_R",
        "ARM_L_IK", "ARM_R_IK", "LEG_L_IK", "LEG_R_IK",
        "HEAD_NECK", "SPINE", "UPPER_BODY", "LOWER_BODY", "ALL"
    ]

    bone_colors = {}
    default_color = (0.5, 0.5, 0.5, 0.6)  # Gray for unassigned

    for bone_name in BONE_INDEX.keys():
        bone_colors[bone_name] = default_color

        # Find highest priority group containing this bone
        for group_name in group_priority:
            if group_name in BONE_GROUPS:
                if bone_name in BONE_GROUPS[group_name]:
                    color = GROUP_COLORS.get(group_name, default_color)
                    bone_colors[bone_name] = color
                    break

    return bone_colors


# =============================================================================
# IK VISUALIZATION
# =============================================================================

def _draw_ik_chains(armature, all_verts, all_colors, scene):
    """Draw IK chains (upper and lower bones)."""
    from ..animations.runtime_ik import get_ik_state, is_ik_active
    from ..engine.animations.ik import LEG_IK, ARM_IK

    state = get_ik_state()
    chains_data = state.get("chains", {})

    if not chains_data and not is_ik_active():
        # No runtime IK - draw default chain visualization
        _draw_default_ik_chains(armature, all_verts, all_colors)
        return

    # Draw active IK chains
    for chain_name, chain_state in chains_data.items():
        root_pos = chain_state.get("root_pos")
        mid_pos = chain_state.get("last_mid_pos")
        target = chain_state.get("last_target")

        if root_pos is None or mid_pos is None or target is None:
            continue

        root = tuple(root_pos)
        mid = tuple(mid_pos)
        tip = tuple(target)

        # Upper bone (root to mid) - cyan
        upper_color = IK_COLORS["chain_upper"]
        all_verts.extend([root, mid])
        all_colors.extend([upper_color, upper_color])

        # Lower bone (mid to tip) - magenta
        lower_color = IK_COLORS["chain_lower"]
        all_verts.extend([mid, tip])
        all_colors.extend([lower_color, lower_color])


def _draw_default_ik_chains(armature, all_verts, all_colors):
    """Draw IK chain bones when no IK is active (show structure)."""
    from ..engine.animations.ik import LEG_IK, ARM_IK

    pose_bones = armature.pose.bones
    arm_matrix = armature.matrix_world

    # Draw all defined IK chains
    all_chains = {**LEG_IK, **ARM_IK}

    for chain_name, chain_def in all_chains.items():
        root_bone = pose_bones.get(chain_def["root"])
        mid_bone = pose_bones.get(chain_def["mid"])
        tip_bone = pose_bones.get(chain_def["tip"])

        if not all([root_bone, mid_bone, tip_bone]):
            continue

        # World positions
        root = tuple((arm_matrix @ root_bone.head)[:])
        mid = tuple((arm_matrix @ mid_bone.head)[:])
        tip = tuple((arm_matrix @ tip_bone.head)[:])

        # Dimmer colors for inactive chains
        upper_color = (0.0, 0.5, 0.5, 0.5)  # Dim cyan
        lower_color = (0.5, 0.0, 0.5, 0.5)  # Dim magenta

        all_verts.extend([root, mid])
        all_colors.extend([upper_color, upper_color])
        all_verts.extend([mid, tip])
        all_colors.extend([lower_color, lower_color])


def _draw_ik_targets(armature, all_verts, all_colors, scene):
    """Draw IK target spheres."""
    from ..animations.runtime_ik import get_ik_state

    state = get_ik_state()
    chains_data = state.get("chains", {})

    for chain_name, chain_state in chains_data.items():
        target = chain_state.get("last_target")
        reachable = chain_state.get("reachable", True)

        if target is None:
            continue

        pos = tuple(target)

        # Color based on reachability
        if reachable:
            color = IK_COLORS["reachable"]
        else:
            color = IK_COLORS["out_of_reach"]

        # Wireframe sphere
        extend_batch_data(
            all_verts, all_colors,
            sphere_wire_verts(pos, 0.05, CIRCLE_8),
            color
        )


def _draw_ik_poles(armature, all_verts, all_colors, scene):
    """Draw IK pole vectors as arrows."""
    from ..animations.runtime_ik import get_ik_state

    state = get_ik_state()
    chains_data = state.get("chains", {})

    pole_color = IK_COLORS["pole"]

    for chain_name, chain_state in chains_data.items():
        mid_pos = chain_state.get("last_mid_pos")
        pole_pos = chain_state.get("pole_pos")

        if mid_pos is None or pole_pos is None:
            continue

        origin = tuple(mid_pos)
        target = tuple(pole_pos)

        # Main line
        all_verts.extend([origin, target])
        all_colors.extend([pole_color, pole_color])

        # Arrow head
        dx = target[0] - origin[0]
        dy = target[1] - origin[1]
        dz = target[2] - origin[2]
        length = math.sqrt(dx*dx + dy*dy + dz*dz)

        if length > 0.01:
            direction = (dx/length, dy/length, dz/length)
            arrow_size = min(0.08, length * 0.3)
            extend_batch_data(
                all_verts, all_colors,
                arrow_head_verts(target, direction, arrow_size),
                pole_color
            )


def _draw_ik_reach(armature, all_verts, all_colors, scene):
    """Draw IK reach spheres from root joints."""
    from ..animations.runtime_ik import get_ik_state
    from ..engine.animations.ik import LEG_IK, ARM_IK

    state = get_ik_state()
    chains_data = state.get("chains", {})

    reach_color = (1.0, 0.8, 0.0, 0.2)  # Transparent yellow

    for chain_name, chain_state in chains_data.items():
        root_pos = chain_state.get("root_pos")
        if root_pos is None:
            continue

        # Get max reach from chain definition
        if chain_name.startswith("leg"):
            chain_def = LEG_IK.get(chain_name, {})
        else:
            chain_def = ARM_IK.get(chain_name, {})

        max_reach = chain_def.get("reach", 0.5)

        center = tuple(root_pos)

        # Layered sphere (3 layers)
        extend_batch_data(
            all_verts, all_colors,
            layered_sphere_verts(center, max_reach, (-0.5, 0.0, 0.5), CIRCLE_8),
            reach_color
        )


# =============================================================================
# BONE AXES VISUALIZATION
# =============================================================================

def _draw_bone_axes(armature, all_verts, all_colors, scene):
    """Draw local axes for each bone (shows orientation)."""
    pose_bones = armature.pose.bones
    arm_matrix = armature.matrix_world
    axis_length = getattr(scene, 'dev_rig_vis_axis_length', 0.05)

    # Axis colors
    x_color = (1.0, 0.2, 0.2, 0.8)  # Red
    y_color = (0.2, 1.0, 0.2, 0.8)  # Green
    z_color = (0.2, 0.2, 1.0, 0.8)  # Blue

    for bone_name in BONE_INDEX.keys():
        pose_bone = pose_bones.get(bone_name)
        if not pose_bone:
            continue

        # Bone world matrix
        bone_matrix = arm_matrix @ pose_bone.matrix
        origin = tuple(bone_matrix.translation)

        # Extract axes
        x_axis = bone_matrix.to_3x3() @ Vector((1, 0, 0))
        y_axis = bone_matrix.to_3x3() @ Vector((0, 1, 0))
        z_axis = bone_matrix.to_3x3() @ Vector((0, 0, 1))

        # Draw axes
        x_end = tuple((Vector(origin) + x_axis * axis_length)[:])
        y_end = tuple((Vector(origin) + y_axis * axis_length)[:])
        z_end = tuple((Vector(origin) + z_axis * axis_length)[:])

        all_verts.extend([origin, x_end])
        all_colors.extend([x_color, x_color])

        all_verts.extend([origin, y_end])
        all_colors.extend([y_color, y_color])

        all_verts.extend([origin, z_end])
        all_colors.extend([z_color, z_color])


# =============================================================================
# ACTIVE MASK VISUALIZATION
# =============================================================================

def _draw_active_mask(armature, all_verts, all_colors, scene):
    """Draw bones colored by active blend mask weight."""
    from ..animations.blend_system import get_blend_system

    blend_sys = get_blend_system()
    if not blend_sys:
        return

    pose_bones = armature.pose.bones
    arm_matrix = armature.matrix_world

    # Collect mask weights from all active layers
    combined_mask = np.zeros(TOTAL_BONES, dtype=np.float32)

    # Check additive layers
    for layer in blend_sys._additive_layers:
        if layer.active and not layer.finished and layer.weight > 0.001:
            if layer.mask_weights is not None:
                combined_mask = np.maximum(combined_mask, layer.mask_weights * layer.weight)

    # Check override layers
    for layer in blend_sys._override_layers:
        if layer.active and not layer.finished and layer.weight > 0.001:
            if layer.mask_weights is not None:
                combined_mask = np.maximum(combined_mask, layer.mask_weights * layer.weight)

    # Draw bones with weight-based coloring
    for bone_idx in range(TOTAL_BONES):
        bone_name = INDEX_TO_BONE.get(bone_idx)
        if not bone_name:
            continue

        pose_bone = pose_bones.get(bone_name)
        if not pose_bone:
            continue

        weight = combined_mask[bone_idx]
        if weight < 0.001:
            continue  # Skip unaffected bones

        # Color: lerp from blue (low) to red (high)
        r = weight
        g = 0.2
        b = 1.0 - weight
        color = (r, g, b, 0.8)

        # World positions
        head = tuple((arm_matrix @ pose_bone.head)[:])
        tail = tuple((arm_matrix @ pose_bone.tail)[:])

        all_verts.extend([head, tail])
        all_colors.extend([color, color])


# =============================================================================
# FULL-BODY IK VISUALIZATION
# =============================================================================

def _draw_full_body_ik(armature, all_verts, all_colors, scene):
    """
    Draw full-body IK constraints and state.

    Shows:
    - Hips position and drop indicator
    - Foot grounding targets
    - Hand reach targets
    - Spine lean direction
    - Look-at target
    """
    from ..animations.full_body_ik import get_fbik_state

    fbik_state = get_fbik_state()
    if not fbik_state.get("active", False):
        return

    pose_bones = armature.pose.bones
    arm_matrix = armature.matrix_world

    constraints = fbik_state.get("constraints", {})
    last_result = fbik_state.get("last_result")

    # Get Root and Hips positions
    root_bone = pose_bones.get("Root")
    hips_bone = pose_bones.get("Hips")

    if not root_bone or not hips_bone:
        return

    root_world = tuple((arm_matrix @ root_bone.head)[:])
    hips_world = tuple((arm_matrix @ hips_bone.head)[:])

    # =========================================================================
    # HIPS DROP INDICATOR
    # =========================================================================
    hips_drop = constraints.get("hips_drop", 0.0)
    if abs(hips_drop) > 0.001:
        # Draw vertical line showing drop amount
        hips_color = FBIK_COLORS["hips_drop"]

        # Original hips position (before drop)
        orig_hips = (hips_world[0], hips_world[1], hips_world[2] + hips_drop)

        # Line from original to current
        all_verts.extend([orig_hips, hips_world])
        all_colors.extend([hips_color, hips_color])

        # Crosshair at original position
        extend_batch_data(all_verts, all_colors, crosshair_verts(orig_hips, 0.03), hips_color)

        # Box at current hips position
        extend_batch_data(all_verts, all_colors, crosshair_verts(hips_world, 0.04), FBIK_COLORS["hips"])

    # =========================================================================
    # FOOT TARGETS
    # =========================================================================
    for foot_key in ["left_foot", "right_foot"]:
        foot_data = constraints.get(foot_key)
        if not foot_data or not foot_data.get("enabled", True):
            continue

        foot_pos = foot_data.get("position")
        if not foot_pos:
            continue

        # Convert Root-relative to world
        target_world = (
            root_world[0] + foot_pos[0],
            root_world[1] + foot_pos[1],
            root_world[2] + foot_pos[2],
        )

        # Get error from last result
        error_cm = 0.0
        if last_result:
            error_key = f"{foot_key}_error_cm".replace("_foot", "_foot")
            error_cm = last_result.get(error_key, 0.0)

        # Color based on error
        if error_cm < 1.0:
            color = FBIK_COLORS["foot_target"]  # Green - good
        elif error_cm < 5.0:
            color = IK_COLORS["at_limit"]  # Yellow - warning
        else:
            color = IK_COLORS["out_of_reach"]  # Red - bad

        # Wireframe sphere at target
        extend_batch_data(
            all_verts, all_colors,
            sphere_wire_verts(target_world, 0.04, CIRCLE_8),
            color
        )

        # Line from foot target to ground (Z=0 relative to root)
        ground_pos = (target_world[0], target_world[1], root_world[2])
        all_verts.extend([target_world, ground_pos])
        all_colors.extend([color, (color[0], color[1], color[2], 0.3)])

    # =========================================================================
    # HAND TARGETS
    # =========================================================================
    for hand_key in ["left_hand", "right_hand"]:
        hand_data = constraints.get(hand_key)
        if not hand_data or not hand_data.get("enabled", True):
            continue

        hand_pos = hand_data.get("position")
        if not hand_pos:
            continue

        # Convert Root-relative to world
        target_world = (
            root_world[0] + hand_pos[0],
            root_world[1] + hand_pos[1],
            root_world[2] + hand_pos[2],
        )

        # Get error from last result
        error_cm = 0.0
        if last_result:
            side = "left" if "left" in hand_key else "right"
            error_key = f"{side}_hand_error_cm"
            error_cm = last_result.get(error_key, 0.0)

        # Color based on error
        if error_cm < 3.0:
            color = FBIK_COLORS["hand_target"]  # Cyan - good
        elif error_cm < 10.0:
            color = FBIK_COLORS["hand_limit"]  # Orange - at limit
        else:
            color = IK_COLORS["out_of_reach"]  # Red - out of reach

        # Wireframe sphere at target
        extend_batch_data(
            all_verts, all_colors,
            sphere_wire_verts(target_world, 0.05, CIRCLE_8),
            color
        )

        # Line from shoulder area to target (visual aid)
        # Estimate shoulder position
        shoulder_offset = 0.15 if "left" in hand_key else -0.15
        shoulder_z = hips_world[2] + 0.5
        shoulder_world = (hips_world[0] + shoulder_offset, hips_world[1], shoulder_z)

        all_verts.extend([shoulder_world, target_world])
        all_colors.extend([(color[0], color[1], color[2], 0.4), color])

    # =========================================================================
    # LOOK-AT TARGET
    # =========================================================================
    look_at_data = constraints.get("look_at")
    if look_at_data and look_at_data.get("enabled", True):
        look_pos = look_at_data.get("position")
        if look_pos:
            # Convert Root-relative to world
            target_world = (
                root_world[0] + look_pos[0],
                root_world[1] + look_pos[1],
                root_world[2] + look_pos[2],
            )

            color = FBIK_COLORS["look_at"]

            # Crosshair at look-at target
            extend_batch_data(all_verts, all_colors, crosshair_verts(target_world, 0.06), color)

            # Line from head to target
            head_bone = pose_bones.get("Head")
            if head_bone:
                head_world = tuple((arm_matrix @ head_bone.head)[:])
                all_verts.extend([head_world, target_world])
                all_colors.extend([(color[0], color[1], color[2], 0.5), color])


# =============================================================================
# TEXT OVERLAY (Billboard text above character)
# =============================================================================

def _draw_text_overlay():
    """
    Draw text overlay above character showing animation state.

    PERFORMANCE:
    - blf is Blender's native font system (very fast)
    - Single 3D→2D projection per frame
    - Minimal string building
    - Only draws if enabled
    """
    scene = bpy.context.scene

    # Check toggles
    if not getattr(scene, 'dev_rig_visualizer_enabled', False):
        return
    if not getattr(scene, 'dev_rig_vis_text_overlay', False):
        return

    # Get armature
    armature = _get_target_armature(scene)
    if not armature:
        return

    # Get 3D region and view
    region = None
    rv3d = None
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for r in area.regions:
                if r.type == 'WINDOW':
                    region = r
                    rv3d = area.spaces[0].region_3d
                    break
            break

    if not region or not rv3d:
        return

    # Position above character head
    head_bone = armature.pose.bones.get("Head")
    if head_bone:
        head_pos = armature.matrix_world @ head_bone.head
        text_pos_3d = head_pos + Vector((0, 0, 0.3))  # 30cm above head
    else:
        text_pos_3d = armature.location + Vector((0, 0, 2.0))

    # Project to 2D screen coordinates
    text_pos_2d = location_3d_to_region_2d(region, rv3d, text_pos_3d)
    if not text_pos_2d:
        return

    # Build status lines (minimal string operations)
    lines = _build_status_lines()
    if not lines:
        return

    # Draw text
    font_size = getattr(scene, 'dev_rig_vis_text_size', 14)
    blf.size(_font_id, font_size)

    # Calculate line height
    line_height = font_size + 4

    # Draw each line (bottom to top so first line is at top)
    x = text_pos_2d[0]
    y = text_pos_2d[1]

    # Center text horizontally
    max_width = 0
    for line, color in lines:
        w, h = blf.dimensions(_font_id, line)
        max_width = max(max_width, w)

    x -= max_width / 2

    # Draw background box for readability (optional - very lightweight)
    if getattr(scene, 'dev_rig_vis_text_background', True):
        _draw_text_background(x - 4, y - 4, max_width + 8, len(lines) * line_height + 8)

    # Draw lines
    for i, (line, color) in enumerate(lines):
        blf.position(_font_id, x, y - (i * line_height), 0)
        blf.color(_font_id, *color)
        blf.draw(_font_id, line)


def _build_status_lines() -> List[Tuple[str, Tuple]]:
    """
    Build status text lines. Returns list of (text, rgba_color) tuples.

    PERFORMANCE: Minimal work - just reads cached state.
    """
    from ..animations.blend_system import get_blend_system
    from ..animations.runtime_ik import get_ik_state, is_ik_active
    from ..animations.full_body_ik import get_fbik_state

    lines = []

    white = (1.0, 1.0, 1.0, 1.0)
    green = (0.4, 1.0, 0.4, 1.0)
    cyan = (0.4, 1.0, 1.0, 1.0)
    yellow = (1.0, 1.0, 0.4, 1.0)
    orange = (1.0, 0.6, 0.2, 1.0)
    magenta = (1.0, 0.4, 1.0, 1.0)
    gray = (0.6, 0.6, 0.6, 1.0)

    # Full-Body IK Status
    fbik_state = get_fbik_state()
    if fbik_state.get("active", False):
        constraints = fbik_state.get("constraints", {})
        last_result = fbik_state.get("last_result")

        # Build constraint summary
        parts = []
        if constraints.get("hips_drop", 0) != 0:
            parts.append(f"hips:{constraints['hips_drop']:.2f}m")
        if constraints.get("left_foot") and constraints["left_foot"].get("enabled"):
            parts.append("L_foot")
        if constraints.get("right_foot") and constraints["right_foot"].get("enabled"):
            parts.append("R_foot")
        if constraints.get("left_hand") and constraints["left_hand"].get("enabled"):
            parts.append("L_hand")
        if constraints.get("right_hand") and constraints["right_hand"].get("enabled"):
            parts.append("R_hand")
        if constraints.get("look_at") and constraints["look_at"].get("enabled"):
            parts.append("look")

        if parts:
            lines.append((f"FBIK: {' '.join(parts)}", magenta))

        # Show result status
        if last_result:
            satisfied = last_result.get("constraints_satisfied", 0)
            total = last_result.get("constraints_total", 0)
            if total > 0:
                status_color = green if satisfied == total else yellow
                lines.append((f"  {satisfied}/{total} satisfied", status_color))

    # IK Status (limb-only)
    elif is_ik_active():
        ik_state = get_ik_state()
        chains = list(ik_state.get("chains", {}).keys())
        if chains:
            lines.append((f"IK: {' '.join(chains)}", cyan))

    # Blend System State
    blend_sys = get_blend_system()
    if blend_sys:
        # Base locomotion
        if blend_sys._base_layer and not blend_sys._base_layer.finished:
            base_name = blend_sys._base_layer.animation_name or "?"
            # Truncate long names
            if len(base_name) > 20:
                base_name = base_name[:17] + "..."
            lines.append((f"Base: {base_name}", green))

        # Additive layers (just count if multiple)
        active_add = [l for l in blend_sys._additive_layers if l.active and not l.finished]
        if active_add:
            if len(active_add) == 1:
                name = active_add[0].animation_name or "?"
                if len(name) > 15:
                    name = name[:12] + "..."
                w = active_add[0].weight
                lines.append((f"Add: {name} ({w:.0%})", yellow))
            else:
                lines.append((f"Add: {len(active_add)} layers", yellow))

        # Override layers
        active_ovr = [l for l in blend_sys._override_layers if l.active and not l.finished]
        if active_ovr:
            if len(active_ovr) == 1:
                name = active_ovr[0].animation_name or "?"
                if len(name) > 15:
                    name = name[:12] + "..."
                w = active_ovr[0].weight
                lines.append((f"Ovr: {name} ({w:.0%})", orange))
            else:
                lines.append((f"Ovr: {len(active_ovr)} layers", orange))

        # Locomotion lock
        if blend_sys.is_locomotion_locked():
            lines.append(("LOCKED", orange))

    # If nothing active, show idle
    if not lines:
        lines.append(("Idle", gray))

    return lines


def _draw_text_background(x, y, width, height):
    """Draw a semi-transparent background behind text."""
    import gpu
    from gpu_extras.batch import batch_for_shader

    # Simple quad
    vertices = [
        (x, y),
        (x + width, y),
        (x + width, y + height),
        (x, y + height),
    ]
    indices = [(0, 1, 2), (0, 2, 3)]

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    batch = batch_for_shader(shader, 'TRIS', {"pos": vertices}, indices=indices)

    gpu.state.blend_set('ALPHA')
    shader.bind()
    shader.uniform_float("color", (0.0, 0.0, 0.0, 0.6))
    batch.draw(shader)


# =============================================================================
# HANDLER MANAGEMENT
# =============================================================================

def enable_rig_visualizer():
    """Register the rig visualizer draw handlers (3D + text)."""
    global _draw_handler, _text_draw_handler

    # 3D geometry handler (POST_VIEW)
    if _draw_handler is None:
        _draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            _draw_rig_visualizer, (), 'WINDOW', 'POST_VIEW'
        )

    # 2D text handler (POST_PIXEL - for screen-space text)
    if _text_draw_handler is None:
        _text_draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            _draw_text_overlay, (), 'WINDOW', 'POST_PIXEL'
        )

    _tag_redraw()


def disable_rig_visualizer():
    """Unregister the rig visualizer draw handlers."""
    global _draw_handler, _text_draw_handler

    if _draw_handler is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_draw_handler, 'WINDOW')
        except Exception:
            pass
        _draw_handler = None

    if _text_draw_handler is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_text_draw_handler, 'WINDOW')
        except Exception:
            pass
        _text_draw_handler = None

    _tag_redraw()


def is_visualizer_active() -> bool:
    """Check if visualizer is currently active."""
    return _draw_handler is not None


def refresh_rig_visualizer() -> bool:
    """
    Refresh the rig visualizer - ensures it's running if enabled.

    Call this from Play buttons to ensure the visualizer appears.
    Fixes the issue where visualizer doesn't show after addon reload
    or when property is already True but handlers weren't registered.

    Returns:
        True if visualizer is now active, False otherwise
    """
    scene = bpy.context.scene

    # Check if visualizer should be enabled
    if not getattr(scene, 'dev_rig_visualizer_enabled', False):
        return False

    # If handlers aren't registered but should be, register them
    if _draw_handler is None or _text_draw_handler is None:
        enable_rig_visualizer()

    # Tag all views for redraw
    _tag_redraw()

    return True


def _tag_redraw():
    """Tag all VIEW_3D areas for redraw."""
    wm = getattr(bpy.context, "window_manager", None)
    if not wm:
        return
    for win in wm.windows:
        scr = win.screen
        if not scr:
            continue
        for area in scr.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


# =============================================================================
# STATE QUERY (for panel display)
# =============================================================================

def get_visualizer_state() -> Dict:
    """Get current visualizer state for UI display."""
    from ..animations.blend_system import get_blend_system
    from ..animations.runtime_ik import get_ik_state, is_ik_active

    state = {
        "active": is_visualizer_active(),
        "ik_active": is_ik_active(),
        "ik_chains": [],
        "layers": {"base": None, "additive": [], "override": []},
    }

    # IK chains
    ik_state = get_ik_state()
    for chain_name in ik_state.get("chains", {}).keys():
        state["ik_chains"].append(chain_name)

    # Blend layers
    blend_sys = get_blend_system()
    if blend_sys:
        if blend_sys._base_layer:
            state["layers"]["base"] = {
                "name": blend_sys._base_layer.animation_name,
                "weight": blend_sys._base_layer.weight,
            }

        for layer in blend_sys._additive_layers:
            if not layer.finished:
                state["layers"]["additive"].append({
                    "name": layer.animation_name,
                    "mask": layer.mask_name,
                    "weight": layer.weight,
                })

        for layer in blend_sys._override_layers:
            if not layer.finished:
                state["layers"]["override"].append({
                    "name": layer.animation_name,
                    "mask": layer.mask_name,
                    "weight": layer.weight,
                })

    return state


def register():
    """Register the visualizer (handler is added via property callback)."""
    pass


def unregister():
    """Unregister and cleanup."""
    disable_rig_visualizer()
