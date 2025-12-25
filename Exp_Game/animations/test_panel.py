# Exp_Game/animations/test_panel.py
"""
Animation 2.0 Test Operators & Properties.

Uses the SAME worker-based animation system as the game.
No duplicate logic - just UI driving the existing system.

UI is in Developer Tools panel (dev_panel.py).
"""

import bpy
import time
from bpy.types import Operator, PropertyGroup
from bpy.props import FloatProperty, BoolProperty, EnumProperty, PointerProperty

from ..engine.animations.baker import bake_action
from ..engine.animations.ik import (
    solve_leg_ik,
    solve_arm_ik,
    LEG_IK,
    ARM_IK,
)
from ..engine import EngineCore
from .controller import AnimationController
from ..developer.dev_logger import start_session, log_worker_messages, export_game_log, clear_log
import numpy as np
import mathutils
import math
import gpu
from gpu_extras.batch import batch_for_shader

# Optimized GPU utilities
from ..developer.gpu_utils import (
    get_cached_shader,
    CIRCLE_8,
    sphere_wire_verts,
    layered_sphere_verts,
    extend_batch_data,
    crosshair_verts,
    arrow_head_verts,
)


# ═══════════════════════════════════════════════════════════════════════════════
# IK VISUALIZER (GPU Draw Handler - Optimized)
#
# PERFORMANCE OPTIMIZATIONS:
# 1. Cached shader (no gpu.shader.from_builtin() every frame)
# 2. Pre-computed circle lookup tables (no trig in draw loops)
# 3. Reduced segment counts (8 instead of 16)
# 4. Reduced reach sphere layers (3 instead of 5)
# ═══════════════════════════════════════════════════════════════════════════════

_ik_draw_handler = None
_ik_vis_data = None  # Shared visualization data


def _build_runtime_ik_vis_data(state: dict) -> dict:
    """
    Build visualization data from runtime IK state.

    Args:
        state: Runtime IK state from runtime_ik.get_ik_state()

    Returns:
        Visualization data dict compatible with _draw_ik_visual()
    """
    from ..engine.animations.ik import LEG_IK, ARM_IK

    vis_data = {
        'targets': [],
        'chains': [],
        'reach_spheres': [],
        'poles': [],
        'joints': [],
    }

    # Get state values
    target = state.get('last_target')
    mid_pos = state.get('last_mid_pos')
    root_pos = state.get('root_pos')
    pole_pos = state.get('pole_pos')
    chain_name = state.get('chain', 'arm_R')
    reachable = state.get('reachable', True)

    if target is None or root_pos is None:
        return vis_data

    # Get chain definition for reach
    if chain_name.startswith("leg"):
        chain_def = LEG_IK.get(chain_name, {})
    else:
        chain_def = ARM_IK.get(chain_name, {})

    max_reach = chain_def.get('reach', 0.5)

    # Target sphere
    vis_data['targets'].append({
        'pos': tuple(target),
        'reachable': reachable,
        'at_limit': not reachable,
    })

    # Chain lines (if we have mid position)
    if mid_pos is not None:
        # We need the tip position - estimate from target
        vis_data['chains'].append({
            'root': tuple(root_pos),
            'mid': tuple(mid_pos),
            'tip': tuple(target),
        })

    # Reach sphere
    vis_data['reach_spheres'].append({
        'center': tuple(root_pos),
        'radius': max_reach,
    })

    # Pole vector
    if pole_pos is not None and mid_pos is not None:
        vis_data['poles'].append({
            'origin': tuple(mid_pos),
            'target': tuple(pole_pos),
        })

    # Joint markers
    vis_data['joints'].append({'pos': tuple(root_pos), 'type': 'root'})
    if mid_pos is not None:
        vis_data['joints'].append({'pos': tuple(mid_pos), 'type': 'mid'})
    vis_data['joints'].append({'pos': tuple(target), 'type': 'tip'})

    return vis_data


def _draw_ik_visual():
    """GPU draw callback for IK visualization (optimized)."""
    global _ik_vis_data

    scene = bpy.context.scene
    if not getattr(scene, 'dev_debug_ik_visual', False):
        return

    # Check for runtime IK state first (during gameplay)
    from .runtime_ik import get_ik_state, is_ik_active
    runtime_state = get_ik_state()

    # Use runtime IK state if active, otherwise fall back to test data
    if is_ik_active() and runtime_state.get("active"):
        vis_data = _build_runtime_ik_vis_data(runtime_state)
    elif _ik_vis_data is not None:
        vis_data = _ik_vis_data
    else:
        return  # Nothing to draw

    # Read toggles once (with safe defaults)
    show_targets = getattr(scene, 'dev_debug_ik_visual_targets', True)
    show_chains = getattr(scene, 'dev_debug_ik_visual_chains', True)
    show_reach = getattr(scene, 'dev_debug_ik_visual_reach', True)
    show_poles = getattr(scene, 'dev_debug_ik_visual_poles', True)
    show_joints = getattr(scene, 'dev_debug_ik_visual_joints', True)

    # Set GPU state - ALWAYS ON TOP (no depth test)
    gpu.state.depth_test_set('NONE')  # Render on top of everything
    gpu.state.blend_set('ALPHA')
    gpu.state.line_width_set(getattr(scene, 'dev_debug_ik_line_width', 2.5))

    # CACHED shader (major optimization)
    shader = get_cached_shader()

    all_verts = []
    all_colors = []

    # ─────────────────────────────────────────────────────────────────────
    # IK TARGETS (Wireframe spheres at goal positions)
    # Green = reachable, Yellow = at limit, Red = out of reach
    # Uses pre-computed circle LUT (no trig calls)
    # ─────────────────────────────────────────────────────────────────────
    if show_targets and 'targets' in vis_data:
        for target_info in vis_data['targets']:
            pos = target_info['pos']
            reachable = target_info.get('reachable', True)
            at_limit = target_info.get('at_limit', False)

            if not reachable:
                color = (1.0, 0.2, 0.2, 0.9)  # Red
            elif at_limit:
                color = (1.0, 1.0, 0.0, 0.9)  # Yellow
            else:
                color = (0.2, 1.0, 0.2, 0.9)  # Green

            # Wireframe sphere (3 circles in XY, XZ, YZ planes)
            extend_batch_data(
                all_verts, all_colors,
                sphere_wire_verts(pos, 0.05, CIRCLE_8),
                color
            )

    # ─────────────────────────────────────────────────────────────────────
    # BONE CHAINS (Lines from root to mid to tip)
    # Cyan = upper bone, Magenta = lower bone
    # ─────────────────────────────────────────────────────────────────────
    if show_chains and 'chains' in vis_data:
        upper_color = (0.0, 1.0, 1.0, 0.9)  # Cyan
        lower_color = (1.0, 0.0, 1.0, 0.9)  # Magenta

        for chain_info in vis_data['chains']:
            root = chain_info['root']
            mid = chain_info['mid']
            tip = chain_info['tip']

            all_verts.extend([root, mid])
            all_colors.extend([upper_color, upper_color])
            all_verts.extend([mid, tip])
            all_colors.extend([lower_color, lower_color])

    # ─────────────────────────────────────────────────────────────────────
    # REACH LIMITS (Layered sphere from root)
    # Transparent yellow, 3 layers instead of 5 for performance
    # ─────────────────────────────────────────────────────────────────────
    if show_reach and 'reach_spheres' in vis_data:
        reach_color = (1.0, 0.8, 0.0, 0.25)

        for reach_info in vis_data['reach_spheres']:
            # 3 layers: bottom, middle, top (reduced from 5)
            extend_batch_data(
                all_verts, all_colors,
                layered_sphere_verts(
                    reach_info['center'],
                    reach_info['radius'],
                    height_ratios=(-0.5, 0.0, 0.5),
                    lut=CIRCLE_8
                ),
                reach_color
            )

    # ─────────────────────────────────────────────────────────────────────
    # POLE VECTORS (Arrows showing bend direction)
    # Orange arrows from mid-point toward pole target
    # ─────────────────────────────────────────────────────────────────────
    if show_poles and 'poles' in vis_data:
        pole_color = (1.0, 0.5, 0.0, 0.9)

        for pole_info in vis_data['poles']:
            origin = pole_info['origin']
            target = pole_info['target']

            # Main line
            all_verts.extend([origin, target])
            all_colors.extend([pole_color, pole_color])

            # Arrow head (uses helper function)
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

    # ─────────────────────────────────────────────────────────────────────
    # JOINT MARKERS (Crosshairs at root, mid, tip)
    # White = root, Cyan = mid (knee/elbow), Green = tip
    # ─────────────────────────────────────────────────────────────────────
    if show_joints and 'joints' in vis_data:
        for joint_info in vis_data['joints']:
            pos = joint_info['pos']
            joint_type = joint_info.get('type', 'mid')

            if joint_type == 'root':
                color = (1.0, 1.0, 1.0, 0.9)
            elif joint_type == 'tip':
                color = (0.2, 1.0, 0.2, 0.9)
            else:
                color = (0.0, 1.0, 1.0, 0.9)

            extend_batch_data(
                all_verts, all_colors,
                crosshair_verts(pos, 0.03),
                color
            )

    # ═════════════════════════════════════════════════════════════════════
    # SINGLE BATCHED DRAW CALL
    # ═════════════════════════════════════════════════════════════════════
    if all_verts:
        batch = batch_for_shader(shader, 'LINES', {"pos": all_verts, "color": all_colors})
        shader.bind()
        batch.draw(shader)

    # Reset GPU state
    gpu.state.line_width_set(1.0)
    gpu.state.depth_test_set('NONE')
    gpu.state.blend_set('NONE')


def enable_ik_visualizer():
    """Register the IK visualization draw handler."""
    global _ik_draw_handler

    if _ik_draw_handler is None:
        _ik_draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            _draw_ik_visual, (), 'WINDOW', 'POST_VIEW'
        )
        _tag_all_view3d_for_redraw()


def disable_ik_visualizer():
    """Unregister the IK visualization draw handler."""
    global _ik_draw_handler, _ik_vis_data

    if _ik_draw_handler is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_ik_draw_handler, 'WINDOW')
        except Exception:
            pass
        _ik_draw_handler = None

    _ik_vis_data = None
    _tag_all_view3d_for_redraw()


def _tag_all_view3d_for_redraw():
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


def update_ik_visualization(
    obj,
    chain: str,
    target_pos: tuple,
    pole_pos: tuple,
    joint_world_pos: tuple = None,
    is_reachable: bool = True,
    at_limit: bool = False
):
    """
    Update IK visualization data.

    Args:
        obj: Armature object
        chain: Chain name (e.g., "leg_L", "arm_R")
        target_pos: IK target position (world space)
        pole_pos: Pole target position (world space)
        joint_world_pos: Computed mid-joint position (knee/elbow)
        is_reachable: Whether target is within reach
        at_limit: Whether target is at max extension
    """
    global _ik_vis_data

    scene = bpy.context.scene
    if not getattr(scene, 'dev_debug_ik_visual', False):
        return

    # Enable visualizer if needed
    if _ik_draw_handler is None:
        enable_ik_visualizer()

    # Get chain definition
    is_leg = chain.startswith("leg")
    if is_leg:
        chain_def = LEG_IK[chain]
    else:
        chain_def = ARM_IK[chain]

    # Get bone positions from armature
    pose_bones = obj.pose.bones
    root_bone = pose_bones.get(chain_def["root"])
    mid_bone = pose_bones.get(chain_def["mid"])
    tip_bone = pose_bones.get(chain_def["tip"])

    if not all([root_bone, mid_bone, tip_bone]):
        return

    # World positions
    root_pos = tuple((obj.matrix_world @ root_bone.head)[:])
    mid_pos = tuple((obj.matrix_world @ mid_bone.head)[:])
    tip_pos = tuple((obj.matrix_world @ tip_bone.head)[:])

    # Use computed joint position if provided, otherwise use current bone position
    if joint_world_pos is not None:
        computed_mid = tuple(joint_world_pos[:])
    else:
        computed_mid = mid_pos

    # Build visualization data
    _ik_vis_data = {
        'targets': [
            {
                'pos': target_pos,
                'reachable': is_reachable,
                'at_limit': at_limit
            }
        ],
        'chains': [
            {
                'root': root_pos,
                'mid': computed_mid,
                'tip': target_pos  # Show where IK is trying to reach
            }
        ],
        'reach_spheres': [
            {
                'center': root_pos,
                'radius': chain_def['reach']
            }
        ],
        'poles': [
            {
                'origin': computed_mid,
                'target': pole_pos
            }
        ],
        'joints': [
            {'pos': root_pos, 'type': 'root'},
            {'pos': computed_mid, 'type': 'mid'},
            {'pos': tip_pos, 'type': 'tip'}
        ]
    }

    _tag_all_view3d_for_redraw()


# ═══════════════════════════════════════════════════════════════════════════════
# TEST ENGINE & CONTROLLER (Shared instances for testing outside game)
# ═══════════════════════════════════════════════════════════════════════════════

_test_engine = None
_test_controller = None


def get_test_engine() -> EngineCore:
    """Get or create the test engine."""
    global _test_engine
    if _test_engine is None or not _test_engine.is_alive():
        _test_engine = EngineCore()
        _test_engine.start()
        # Wait for workers to be ready
        _test_engine.wait_for_readiness(timeout=2.0)
        # Start log session for animation testing
        start_session()
    return _test_engine


def get_test_controller() -> AnimationController:
    """Get or create the test controller."""
    global _test_controller
    if _test_controller is None:
        _test_controller = AnimationController()
    return _test_controller


def reset_test_controller():
    """Reset the test controller and engine."""
    global _test_controller, _test_engine

    # Stop timer if running
    if bpy.app.timers.is_registered(playback_update):
        bpy.app.timers.unregister(playback_update)

    # Export logs before shutdown
    if _test_engine is not None:
        import os
        log_path = os.path.join(os.path.expanduser("~"), "Desktop", "engine_output_files", "anim_test_log.txt")
        export_game_log(log_path)
        clear_log()

    # Shutdown test engine
    if _test_engine is not None:
        _test_engine.shutdown()
        _test_engine = None

    # Disable IK visualizer
    disable_ik_visualizer()

    _test_controller = None


# ═══════════════════════════════════════════════════════════════════════════════
# OPERATORS
# ═══════════════════════════════════════════════════════════════════════════════

class ANIM2_OT_BakeAll(Operator):
    """Bake ALL actions and cache in workers"""
    bl_idname = "anim2.bake_all"
    bl_label = "Bake All Actions"
    bl_options = {'REGISTER'}

    def execute(self, context):
        start_time = time.perf_counter()

        # Reset for fresh bake
        reset_test_controller()
        ctrl = get_test_controller()
        engine = get_test_engine()

        # Bake ALL actions - no armature dependency
        # Baker extracts data directly from FCurves
        baked_count = 0
        baked_bones = 0
        baked_objects = 0
        failed = []

        for action in bpy.data.actions:
            try:
                anim = bake_action(action)
                ctrl.add_animation(anim)
                baked_count += 1
                if anim.has_bones:
                    baked_bones += 1
                if anim.has_object:
                    baked_objects += 1
            except Exception as e:
                failed.append(f"{action.name}: {e}")

        # Cache in workers
        if baked_count > 0 and engine.is_alive():
            cache_data = ctrl.get_cache_data_for_workers()
            engine.broadcast_job("CACHE_ANIMATIONS", cache_data)

            # Wait for workers to confirm
            confirmed = 0
            wait_start = time.perf_counter()
            while confirmed < 8 and (time.perf_counter() - wait_start) < 1.0:
                results = list(engine.poll_results(max_results=20))
                for r in results:
                    if r.job_type == "CACHE_ANIMATIONS" and r.success:
                        confirmed += 1
                time.sleep(0.01)

        elapsed = (time.perf_counter() - start_time) * 1000

        # Build detailed report
        parts = [f"{baked_count} actions"]
        if baked_bones > 0:
            parts.append(f"{baked_bones} with bones")
        if baked_objects > 0:
            parts.append(f"{baked_objects} with object transforms")

        if failed:
            self.report({'WARNING'}, f"Baked {', '.join(parts)} ({elapsed:.0f}ms). {len(failed)} failed.")
        else:
            self.report({'INFO'}, f"Baked {', '.join(parts)} in {elapsed:.0f}ms")

        return {'FINISHED'}


class ANIM2_OT_StopAll(Operator):
    """Stop ALL animations on ALL objects"""
    bl_idname = "anim2.stop_all"
    bl_label = "Stop All"
    bl_options = {'REGISTER'}

    def execute(self, context):
        global _last_time, _playback_start_time

        stop_all_animations()

        # Reset timer state
        _last_time = None
        _playback_start_time = None

        # Unregister timer if running
        if bpy.app.timers.is_registered(playback_update):
            bpy.app.timers.unregister(playback_update)

        self.report({'INFO'}, "Stopped all animations")
        return {'FINISHED'}


class ANIM2_OT_ClearCache(Operator):
    """Clear all cached animations and shutdown test engine"""
    bl_idname = "anim2.clear_cache"
    bl_label = "Clear Cache"
    bl_options = {'REGISTER'}

    def execute(self, context):
        reset_test_controller()
        self.report({'INFO'}, "Animation cache cleared, engine stopped")
        return {'FINISHED'}


# ═══════════════════════════════════════════════════════════════════════════════
# IK APPLICATION (PROPER SOLVER-BASED)
# ═══════════════════════════════════════════════════════════════════════════════

def apply_ik_chain(armature, chain: str, target_pos, influence: float = 1.0, log: bool = False):
    """
    Apply IK to a bone chain using the proper two-bone IK solver.

    This is the CORRECT way to apply IK - using solver-computed quaternions
    with proper world-to-local space conversion.

    Args:
        armature: Blender armature object
        chain: Chain name ("leg_L", "leg_R", "arm_L", "arm_R")
        target_pos: World-space target position (Vector or array)
        influence: IK influence 0-1
        log: Enable detailed logging

    Returns:
        dict with debug info or None on failure
    """
    from ..engine.animations.ik import (
        LEG_IK, ARM_IK, solve_leg_ik, solve_arm_ik,
        compute_knee_pole_position, compute_elbow_pole_position
    )
    from ..developer.dev_logger import log_game

    # Get chain definition
    IK_CHAINS = {**LEG_IK, **ARM_IK}
    chain_def = IK_CHAINS.get(chain)
    if not chain_def:
        if log:
            log_game("IK-SOLVE", f"ERROR chain={chain} not found")
        return None

    # Parse chain type
    parts = chain.split('_')
    if len(parts) != 2:
        return None
    limb_type, side = parts[0], parts[1]
    is_leg = (limb_type == "leg")

    # Get bone references
    pose_bones = armature.pose.bones
    root_bone = pose_bones.get(chain_def["root"])
    mid_bone = pose_bones.get(chain_def["mid"])
    tip_bone = pose_bones.get(chain_def["tip"])

    if not all([root_bone, mid_bone, tip_bone]):
        if log:
            log_game("IK-SOLVE", f"ERROR bones not found chain={chain}")
        return None

    # Get world-space positions
    arm_matrix = armature.matrix_world
    root_world = arm_matrix @ root_bone.head
    target_world = mathutils.Vector(target_pos[:3])

    # Convert to numpy
    root_pos = np.array([root_world.x, root_world.y, root_world.z], dtype=np.float32)
    target_np = np.array([target_world.x, target_world.y, target_world.z], dtype=np.float32)

    # Get character orientation for pole calculation
    char_forward = np.array([arm_matrix[0][1], arm_matrix[1][1], arm_matrix[2][1]], dtype=np.float32)
    char_right = np.array([arm_matrix[0][0], arm_matrix[1][0], arm_matrix[2][0]], dtype=np.float32)
    char_up = np.array([arm_matrix[0][2], arm_matrix[1][2], arm_matrix[2][2]], dtype=np.float32)

    # Compute pole position (automatic, anatomically correct)
    if is_leg:
        pole_pos = compute_knee_pole_position(root_pos, target_np, char_forward, char_right, side, 0.5)
        upper_quat, lower_quat, joint_world = solve_leg_ik(root_pos, target_np, pole_pos, side)
    else:
        pole_pos = compute_elbow_pole_position(root_pos, target_np, char_forward, char_up, side, 0.3)
        upper_quat, lower_quat, joint_world = solve_arm_ik(root_pos, target_np, pole_pos, side)

    # Calculate reach info
    reach_dist = float(np.linalg.norm(target_np - root_pos))
    max_reach = chain_def["reach"]
    reach_pct = reach_dist / max_reach * 100

    if log:
        # Log current bone state BEFORE IK
        root_world_pos = arm_matrix @ root_bone.head
        mid_world_pos = arm_matrix @ mid_bone.head
        tip_world_pos = arm_matrix @ tip_bone.head
        log_game("IK-SOLVE", f"CHAIN={chain} root={root_bone.name} mid={mid_bone.name} tip={tip_bone.name}")
        log_game("IK-SOLVE", f"BEFORE root_pos=({root_world_pos.x:.3f},{root_world_pos.y:.3f},{root_world_pos.z:.3f})")
        log_game("IK-SOLVE", f"BEFORE mid_pos=({mid_world_pos.x:.3f},{mid_world_pos.y:.3f},{mid_world_pos.z:.3f})")
        log_game("IK-SOLVE", f"BEFORE tip_pos=({tip_world_pos.x:.3f},{tip_world_pos.y:.3f},{tip_world_pos.z:.3f})")
        log_game("IK-SOLVE", f"TARGET=({target_np[0]:.3f},{target_np[1]:.3f},{target_np[2]:.3f}) reach={reach_dist:.3f}m ({reach_pct:.0f}%)")
        log_game("IK-SOLVE", f"POLE=({pole_pos[0]:.3f},{pole_pos[1]:.3f},{pole_pos[2]:.3f})")
        log_game("IK-SOLVE", f"SOLVER_OUT upper_q=({upper_quat[0]:.3f},{upper_quat[1]:.3f},{upper_quat[2]:.3f},{upper_quat[3]:.3f})")
        log_game("IK-SOLVE", f"SOLVER_OUT lower_q=({lower_quat[0]:.3f},{lower_quat[1]:.3f},{lower_quat[2]:.3f},{lower_quat[3]:.3f})")
        log_game("IK-SOLVE", f"SOLVER_OUT joint=({joint_world[0]:.3f},{joint_world[1]:.3f},{joint_world[2]:.3f})")

    # ═══════════════════════════════════════════════════════════════════════════
    # APPLY QUATERNIONS TO BONES
    # The solver returns world-space quaternions. We need to convert to local.
    # ═══════════════════════════════════════════════════════════════════════════

    # Get parent matrices for local space conversion
    if root_bone.parent:
        parent_matrix = arm_matrix @ root_bone.parent.matrix
        parent_rot_inv = parent_matrix.to_quaternion().inverted()
    else:
        parent_rot_inv = arm_matrix.to_quaternion().inverted()

    # Convert upper quaternion from world to local
    upper_world_q = mathutils.Quaternion((upper_quat[0], upper_quat[1], upper_quat[2], upper_quat[3]))
    upper_local_q = parent_rot_inv @ upper_world_q

    # Store original for blending
    original_upper = root_bone.rotation_quaternion.copy()
    original_lower = mid_bone.rotation_quaternion.copy()

    # Apply upper bone rotation
    root_bone.rotation_mode = 'QUATERNION'
    if influence < 1.0:
        root_bone.rotation_quaternion = original_upper.slerp(upper_local_q, influence)
    else:
        root_bone.rotation_quaternion = upper_local_q

    # Update transforms so mid_bone parent matrix is correct
    bpy.context.view_layer.update()

    # Now compute local rotation for mid bone
    # Mid bone's parent is the root bone (which we just rotated)
    mid_parent_matrix = arm_matrix @ root_bone.matrix
    mid_parent_rot_inv = mid_parent_matrix.to_quaternion().inverted()

    lower_world_q = mathutils.Quaternion((lower_quat[0], lower_quat[1], lower_quat[2], lower_quat[3]))
    lower_local_q = mid_parent_rot_inv @ lower_world_q

    # Apply lower bone rotation
    mid_bone.rotation_mode = 'QUATERNION'
    if influence < 1.0:
        mid_bone.rotation_quaternion = original_lower.slerp(lower_local_q, influence)
    else:
        mid_bone.rotation_quaternion = lower_local_q

    if log:
        log_game("IK-SOLVE", f"APPLIED upper_local=({upper_local_q.w:.3f},{upper_local_q.x:.3f},{upper_local_q.y:.3f},{upper_local_q.z:.3f})")
        log_game("IK-SOLVE", f"APPLIED lower_local=({lower_local_q.w:.3f},{lower_local_q.x:.3f},{lower_local_q.y:.3f},{lower_local_q.z:.3f})")
        # Log AFTER state
        bpy.context.view_layer.update()
        root_world_after = arm_matrix @ root_bone.head
        mid_world_after = arm_matrix @ mid_bone.head
        tip_world_after = arm_matrix @ tip_bone.head
        log_game("IK-SOLVE", f"AFTER root_pos=({root_world_after.x:.3f},{root_world_after.y:.3f},{root_world_after.z:.3f})")
        log_game("IK-SOLVE", f"AFTER mid_pos=({mid_world_after.x:.3f},{mid_world_after.y:.3f},{mid_world_after.z:.3f})")
        log_game("IK-SOLVE", f"AFTER tip_pos=({tip_world_after.x:.3f},{tip_world_after.y:.3f},{tip_world_after.z:.3f})")

    # Update runtime_ik state for rig visualizer
    from .runtime_ik import update_ik_state
    update_ik_state(
        chain=chain,
        target_pos=target_np,
        mid_pos=joint_world,
        root_pos=root_pos,
        pole_pos=pole_pos,
        influence=influence,
        reachable=(reach_pct <= 100)
    )

    return {
        "chain": chain,
        "target": target_np,
        "reach_dist": reach_dist,
        "reach_pct": reach_pct,
        "pole": pole_pos,
        "joint": joint_world,
        "upper_quat": upper_quat,
        "lower_quat": lower_quat,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LEGACY IK (for compatibility)
# ═══════════════════════════════════════════════════════════════════════════════

def point_bone_at_target(obj, pose_bone, target_world_pos):
    """
    Rotate a pose bone so it points toward a world position.

    Uses a simple and robust approach:
    1. Get bone's current direction in world space
    2. Get desired direction in world space
    3. Compute world-space rotation delta
    4. Convert to bone-local rotation

    Args:
        obj: Armature object
        pose_bone: The pose bone to rotate
        target_world_pos: World position to point at (mathutils.Vector or numpy array)

    Returns:
        mathutils.Quaternion in bone local space
    """
    target = mathutils.Vector(target_world_pos[:3])

    # Get bone head/tail in world space (current posed position)
    bone_head_world = obj.matrix_world @ pose_bone.head
    bone_tail_world = obj.matrix_world @ pose_bone.tail

    # Current direction (where bone is pointing now)
    current_dir = (bone_tail_world - bone_head_world).normalized()

    # Desired direction (where we want it to point)
    desired_dir = (target - bone_head_world).normalized()

    # World-space rotation from current to desired
    world_delta = current_dir.rotation_difference(desired_dir)

    # Get bone's current world rotation
    bone_world_matrix = obj.matrix_world @ pose_bone.matrix
    bone_world_rot = bone_world_matrix.to_quaternion()

    # Apply delta in world space, then convert back to local
    new_world_rot = world_delta @ bone_world_rot

    # Convert new world rotation to bone local space
    # bone_local = parent_world_inv @ new_world
    if pose_bone.parent:
        parent_world_matrix = obj.matrix_world @ pose_bone.parent.matrix
        parent_world_rot = parent_world_matrix.to_quaternion()
        # Also need to account for bone's rest orientation relative to parent
        bone_rest = pose_bone.bone.matrix_local
        parent_rest = pose_bone.parent.bone.matrix_local
        rest_relative = parent_rest.inverted() @ bone_rest
        rest_rot = rest_relative.to_quaternion()

        # Local rotation = rest_inv @ parent_world_inv @ new_world
        local_rot = rest_rot.inverted() @ parent_world_rot.inverted() @ new_world_rot
    else:
        # No parent - just undo armature world and rest
        armature_rot = obj.matrix_world.to_quaternion()
        bone_rest_rot = pose_bone.bone.matrix_local.to_quaternion()
        local_rot = bone_rest_rot.inverted() @ armature_rot.inverted() @ new_world_rot

    return local_rot


def get_pole_direction_vector(pole_dir: str, is_leg: bool) -> np.ndarray:
    """Convert pole direction enum to world-space vector."""
    directions = {
        "FORWARD": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "BACK": np.array([0.0, -1.0, 0.0], dtype=np.float32),
        "LEFT": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        "RIGHT": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "UP": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "DOWN": np.array([0.0, 0.0, -1.0], dtype=np.float32),
    }

    if pole_dir == "AUTO":
        # Legs: knee forward, Arms: elbow back
        return directions["FORWARD"] if is_leg else directions["BACK"]

    return directions.get(pole_dir, directions["FORWARD"])


# ═══════════════════════════════════════════════════════════════════════════════
# PLAYBACK TIMER (Uses same worker flow as game)
# ═══════════════════════════════════════════════════════════════════════════════

_last_time = None
_playback_start_time = None

def playback_update():
    """Timer callback - uses worker-based animation system."""
    global _last_time, _playback_start_time

    ctrl = get_test_controller()
    engine = get_test_engine()

    if not engine.is_alive():
        _last_time = None
        _playback_start_time = None
        return None  # Stop timer

    # Calculate delta time
    current = time.perf_counter()
    if _last_time is None:
        _last_time = current
        _playback_start_time = current
        dt = 1/60
    else:
        dt = current - _last_time
        _last_time = current

    # Check timeout
    timeout = 20.0  # Default 20 second timeout
    try:
        timeout = bpy.context.scene.anim2_test.playback_timeout
    except:
        pass

    if timeout > 0 and _playback_start_time is not None:
        elapsed = current - _playback_start_time
        if elapsed >= timeout:
            # Stop all animations
            stop_all_animations()
            _last_time = None
            _playback_start_time = None
            return None  # Stop timer

    # 1. Update state (times, fades) - same as game
    ctrl.update_state(dt)

    # 2. Get job data and submit to engine
    jobs_data = ctrl.get_compute_job_data()
    pending_jobs = {}

    for object_name, job_data in jobs_data.items():
        job_id = engine.submit_job("ANIMATION_COMPUTE", job_data)
        if job_id is not None and job_id >= 0:
            pending_jobs[job_id] = object_name

    # 3. Poll for results with short timeout (same-frame sync)
    if pending_jobs:
        poll_start = time.perf_counter()
        while pending_jobs and (time.perf_counter() - poll_start) < 0.003:
            results = list(engine.poll_results(max_results=20))
            for result in results:
                if result.job_type == "ANIMATION_COMPUTE" and result.job_id in pending_jobs:
                    if result.success:
                        object_name = pending_jobs.pop(result.job_id)
                        bone_transforms = result.result.get("bone_transforms", {})
                        object_transform = result.result.get("object_transform")
                        if bone_transforms or object_transform:
                            ctrl.apply_worker_result(object_name, bone_transforms, object_transform)
                        # Process worker logs
                        worker_logs = result.result.get("logs", [])
                        if worker_logs:
                            log_worker_messages(worker_logs)
            time.sleep(0.0001)

    # Force viewport redraw
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()

    # Check if anything is still playing
    has_playing = ctrl.has_active_animations()

    if has_playing:
        return 1/60  # Continue at 60fps
    else:
        _last_time = None
        _playback_start_time = None
        return None  # Stop timer


def stop_all_animations():
    """Stop animations on ALL objects."""
    ctrl = get_test_controller()
    # Stop each object's animations with fade
    for object_name in list(ctrl._states.keys()):
        ctrl.stop(object_name, fade_out=0.2)


# ═══════════════════════════════════════════════════════════════════════════════
# POSE BLEND AUTO-PLAY TIMER
# ═══════════════════════════════════════════════════════════════════════════════

_pose_blend_state = {
    "start_time": 0.0,
    "direction": 1,  # 1 = forward (A→B), -1 = backward (B→A)
    "ik_targets": {},  # chain_name -> world position from Pose B
}


def pose_blend_auto_update():
    """
    Timer callback for pose blend auto-play.

    Uses WORKER-BASED computation - all blending math happens in engine.
    Main thread only submits jobs and applies results.
    """
    import json
    from .bone_groups import BONE_INDEX
    from ..developer.dev_logger import log_game, log_worker_messages

    scene = bpy.context.scene

    # Check if still in POSE mode
    if scene.test_mode != 'POSE':
        return None  # Stop timer

    armature = getattr(scene, 'target_armature', None)
    if not armature:
        return None

    engine = get_test_engine()
    if not engine.is_alive():
        return None

    # Get timing
    duration = scene.pose_blend_duration
    elapsed = time.perf_counter() - _pose_blend_state["start_time"]

    # Calculate normalized time within current half-cycle
    cycle_time = elapsed % (duration * 2)  # Full back-and-forth cycle

    if cycle_time < duration:
        weight = cycle_time / duration
    else:
        weight = 1.0 - ((cycle_time - duration) / duration)

    weight = max(0.0, min(1.0, weight))
    scene.pose_blend_weight = weight

    # Get pose data (already cached at start)
    pose_a_data = _pose_blend_state.get("pose_a_data", {})
    pose_b_data = _pose_blend_state.get("pose_b_data", {})

    if not pose_a_data:
        return None

    # Build IK chain data for worker
    ik_mode = getattr(scene, 'pose_blend_mode', 'POSE_TO_POSE')
    ik_targets = _pose_blend_state.get("ik_targets", {})
    influence = scene.pose_blend_ik_influence

    ik_chain_props = [
        ('pose_blend_ik_arm_L', 'pose_blend_ik_arm_L_target', 'arm_L'),
        ('pose_blend_ik_arm_R', 'pose_blend_ik_arm_R_target', 'arm_R'),
        ('pose_blend_ik_leg_L', 'pose_blend_ik_leg_L_target', 'leg_L'),
        ('pose_blend_ik_leg_R', 'pose_blend_ik_leg_R_target', 'leg_R'),
    ]

    # Get character orientation for pole calculation
    arm_matrix = armature.matrix_world
    char_forward = [arm_matrix[0][1], arm_matrix[1][1], arm_matrix[2][1]]
    char_right = [arm_matrix[0][0], arm_matrix[1][0], arm_matrix[2][0]]
    char_up = [arm_matrix[0][2], arm_matrix[1][2], arm_matrix[2][2]]

    ik_chains_for_worker = []
    for enabled_prop, target_prop, chain_name in ik_chain_props:
        if getattr(scene, enabled_prop, False):
            target_pos = None
            if ik_mode == 'POSE_TO_TARGET':
                target_obj = getattr(scene, target_prop, None)
                if target_obj:
                    target_pos = list(target_obj.matrix_world.translation)
            else:
                if chain_name in ik_targets:
                    target_pos = list(ik_targets[chain_name])

            if target_pos:
                # Get root bone position
                from ..engine.animations.ik import LEG_IK, ARM_IK
                all_chains = {**LEG_IK, **ARM_IK}
                chain_def = all_chains.get(chain_name)
                if chain_def:
                    root_bone = armature.pose.bones.get(chain_def["root"])
                    if root_bone:
                        root_pos = list(arm_matrix @ root_bone.head)
                        ik_chains_for_worker.append({
                            "chain": chain_name,
                            "target": target_pos,
                            "root_pos": root_pos,
                            "influence": influence,
                            "char_forward": char_forward,
                            "char_right": char_right,
                            "char_up": char_up,
                        })

    # Submit POSE_BLEND_COMPUTE job to worker
    job_data = {
        "pose_a": pose_a_data,
        "pose_b": pose_b_data,
        "weight": weight,
        "bone_names": list(set(pose_a_data.keys()) | set(pose_b_data.keys())),
        "ik_chains": ik_chains_for_worker,
    }

    job_id = engine.submit_job("POSE_BLEND_COMPUTE", job_data)

    # Poll for result with short timeout (same-frame sync)
    if job_id is not None and job_id >= 0:
        poll_start = time.perf_counter()
        while (time.perf_counter() - poll_start) < 0.003:  # 3ms timeout
            results = list(engine.poll_results(max_results=5))
            for result in results:
                if result.job_type == "POSE_BLEND_COMPUTE" and result.job_id == job_id:
                    if result.success:
                        # Apply blended bone transforms
                        bone_transforms = result.result.get("bone_transforms", {})
                        _apply_bone_transforms(armature, bone_transforms)

                        # Apply IK results
                        ik_results = result.result.get("ik_results", {})
                        _apply_ik_results(armature, ik_results)

                        # Process worker logs
                        worker_logs = result.result.get("logs", [])
                        if worker_logs:
                            log_worker_messages(worker_logs)
                    break
            else:
                time.sleep(0.0001)
                continue
            break

    # Force viewport redraw
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()

    return 1/60  # Continue at 60fps


def _apply_bone_transforms(armature, bone_transforms: dict):
    """Apply bone transforms from worker result to armature."""
    pose_bones = armature.pose.bones
    for bone_name, transform in bone_transforms.items():
        pose_bone = pose_bones.get(bone_name)
        if pose_bone:
            pose_bone.rotation_mode = 'QUATERNION'
            pose_bone.rotation_quaternion = mathutils.Quaternion((transform[0], transform[1], transform[2], transform[3]))
            pose_bone.location = mathutils.Vector((transform[4], transform[5], transform[6]))
            pose_bone.scale = mathutils.Vector((transform[7], transform[8], transform[9]))


def _apply_ik_results(armature, ik_results: dict):
    """Apply IK results from worker to armature bones."""
    from ..engine.animations.ik import LEG_IK, ARM_IK

    if not ik_results:
        return

    all_chains = {**LEG_IK, **ARM_IK}
    arm_matrix = armature.matrix_world
    pose_bones = armature.pose.bones

    for chain_name, ik_data in ik_results.items():
        chain_def = all_chains.get(chain_name)
        if not chain_def:
            continue

        root_bone = pose_bones.get(chain_def["root"])
        mid_bone = pose_bones.get(chain_def["mid"])
        if not root_bone or not mid_bone:
            continue

        upper_quat = ik_data.get("upper_quat")
        lower_quat = ik_data.get("lower_quat")
        influence = ik_data.get("influence", 1.0)

        if upper_quat and lower_quat:
            # Convert world quaternions to local
            # Get parent matrices for local space conversion
            if root_bone.parent:
                parent_matrix = arm_matrix @ root_bone.parent.matrix
                parent_rot_inv = parent_matrix.to_quaternion().inverted()
            else:
                parent_rot_inv = arm_matrix.to_quaternion().inverted()

            upper_world_q = mathutils.Quaternion(upper_quat)
            upper_local_q = parent_rot_inv @ upper_world_q

            # Store original for blending
            original_upper = root_bone.rotation_quaternion.copy()
            original_lower = mid_bone.rotation_quaternion.copy()

            # Apply upper bone
            root_bone.rotation_mode = 'QUATERNION'
            if influence < 1.0:
                root_bone.rotation_quaternion = original_upper.slerp(upper_local_q, influence)
            else:
                root_bone.rotation_quaternion = upper_local_q

            # Update for mid bone parent
            bpy.context.view_layer.update()

            # Mid bone local conversion
            mid_parent_matrix = arm_matrix @ root_bone.matrix
            mid_parent_rot_inv = mid_parent_matrix.to_quaternion().inverted()

            lower_world_q = mathutils.Quaternion(lower_quat)
            lower_local_q = mid_parent_rot_inv @ lower_world_q

            # Apply lower bone
            mid_bone.rotation_mode = 'QUATERNION'
            if influence < 1.0:
                mid_bone.rotation_quaternion = original_lower.slerp(lower_local_q, influence)
            else:
                mid_bone.rotation_quaternion = lower_local_q

        # Update runtime_ik state for visualizer
        from .runtime_ik import update_ik_state
        update_ik_state(
            chain=chain_name,
            target_pos=np.array(ik_data.get("target", [0, 0, 0]), dtype=np.float32),
            mid_pos=np.array(ik_data.get("joint_world", [0, 0, 0]), dtype=np.float32),
            root_pos=np.array(ik_data.get("root_pos", [0, 0, 0]), dtype=np.float32) if "root_pos" in ik_data else np.zeros(3, dtype=np.float32),
            pole_pos=np.array(ik_data.get("pole_pos", [0, 0, 0]), dtype=np.float32),
            influence=influence,
            reachable=ik_data.get("reachable", True)
        )


def _apply_ik_overlay_standalone(armature, chain, target_obj, influence):
    """Standalone IK overlay for timer callback (no self reference)."""
    target_pos = target_obj.matrix_world.translation
    _apply_ik_overlay_with_position(armature, chain, target_pos, influence)


def _apply_ik_overlay_with_position(armature, chain, target_pos, influence):
    """Apply IK overlay using a world-space position."""
    log_enabled = getattr(bpy.context.scene, 'dev_debug_ik_solve', False)
    apply_ik_chain(armature, chain, target_pos, influence, log=log_enabled)


def stop_pose_blend_auto():
    """Stop pose blend auto-play timer."""
    if bpy.app.timers.is_registered(pose_blend_auto_update):
        bpy.app.timers.unregister(pose_blend_auto_update)


# ═══════════════════════════════════════════════════════════════════════════════
# PROPERTIES
# ═══════════════════════════════════════════════════════════════════════════════

def get_animation_items(self, context):
    """Get list of cached animations for enum."""
    ctrl = get_test_controller()
    items = [("", "Select Animation", "")]
    for name in ctrl.cache.names:
        items.append((name, name, ""))
    return items


class ANIM2_TestProperties(PropertyGroup):
    """Properties for animation 2.0 testing."""

    selected_animation: EnumProperty(
        name="Animation",
        description="Animation to play",
        items=get_animation_items
    )

    blend_animation: EnumProperty(
        name="Blend",
        description="Animation to blend in",
        items=get_animation_items
    )

    blend_weight: FloatProperty(
        name="Blend Weight",
        description="Weight of blended animation",
        default=0.5,
        min=0.0,
        max=1.0
    )

    play_speed: FloatProperty(
        name="Speed",
        description="Playback speed multiplier",
        default=1.0,
        min=0.1,
        max=3.0
    )

    fade_time: FloatProperty(
        name="Fade",
        description="Fade in/out time in seconds",
        default=0.2,
        min=0.0,
        max=2.0
    )

    loop_playback: BoolProperty(
        name="Loop",
        description="Loop the animation",
        default=True
    )

    playback_timeout: FloatProperty(
        name="Timeout",
        description="Auto-stop playback after this many seconds (0 = no timeout)",
        default=20.0,
        min=0.0,
        max=300.0
    )

    # NOTE: Legacy IK properties removed - now using unified test_ik_* properties
    # from dev_properties.py (test_ik_chain, test_ik_target, test_ik_pole, etc.)


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED TEST SUITE OPERATORS
# ═══════════════════════════════════════════════════════════════════════════════

class ANIM2_OT_TestPlay(Operator):
    """Unified Play - dispatches based on test mode (Animation/Pose/IK)"""
    bl_idname = "anim2.test_play"
    bl_label = "Play"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        armature = getattr(scene, 'target_armature', None)
        if armature is None or armature.type != 'ARMATURE':
            return False

        mode = getattr(scene, 'test_mode', 'ANIMATION')
        if mode == 'ANIMATION':
            # Need baked animations
            ctrl = get_test_controller()
            return ctrl.cache.count > 0
        elif mode == 'POSE':
            # Need poses and at least one IK chain enabled
            if len(scene.pose_library) == 0:
                return False

            ik_mode = getattr(scene, 'pose_blend_mode', 'POSE_TO_POSE')
            ik_chains = [
                ('pose_blend_ik_arm_L', 'pose_blend_ik_arm_L_target'),
                ('pose_blend_ik_arm_R', 'pose_blend_ik_arm_R_target'),
                ('pose_blend_ik_leg_L', 'pose_blend_ik_leg_L_target'),
                ('pose_blend_ik_leg_R', 'pose_blend_ik_leg_R_target'),
            ]

            for enabled_prop, target_prop in ik_chains:
                if getattr(scene, enabled_prop, False):
                    if ik_mode == 'POSE_TO_TARGET':
                        # Need object target
                        if getattr(scene, target_prop, None):
                            return True
                    else:
                        # POSE_TO_POSE: Just need enabled chain
                        return True
            return False
        elif mode == 'IK':
            return True
        return False

    def execute(self, context):
        scene = context.scene
        mode = scene.test_mode
        armature = scene.target_armature

        # Refresh rig visualizer if enabled (ensures it appears in viewport)
        from ..developer.rig_visualizer import refresh_rig_visualizer
        refresh_rig_visualizer()

        # Start rig logging session if enabled
        rig_log_enabled = getattr(scene, 'dev_debug_rig_state', False)
        if rig_log_enabled and armature:
            from .rig_logger import start_rig_test_session
            start_rig_test_session(armature, f"TEST_{mode}")

        if mode == 'ANIMATION':
            result = self._play_animation(context, armature)
        elif mode == 'POSE':
            # POSE mode is always IK pose-to-pose with auto-loop
            result = self._play_ik_pose_blend(context, armature)
        elif mode == 'IK':
            result = self._play_ik(context, armature)
        else:
            result = {'CANCELLED'}

        return result

    def _play_animation(self, context, armature):
        """Play animation on target armature."""
        props = context.scene.anim2_test
        anim_name = props.selected_animation

        if not anim_name:
            self.report({'WARNING'}, "No animation selected")
            return {'CANCELLED'}

        ctrl = get_test_controller()

        if not ctrl.has_animation(anim_name):
            self.report({'WARNING'}, f"Animation '{anim_name}' not in cache. Bake first.")
            return {'CANCELLED'}

        # Play with settings
        success = ctrl.play(
            armature.name,
            anim_name,
            weight=1.0,
            speed=props.play_speed,
            looping=props.loop_playback,
            fade_in=props.fade_time,
            replace=True
        )

        if success:
            self.report({'INFO'}, f"Playing '{anim_name}' on {armature.name}")
            if not bpy.app.timers.is_registered(playback_update):
                bpy.app.timers.register(playback_update, first_interval=1/60)

            # Blend secondary if enabled
            scene = context.scene
            if getattr(scene, 'test_blend_enabled', False):
                blend_anim = props.blend_animation
                if blend_anim and ctrl.has_animation(blend_anim):
                    ctrl.play(
                        armature.name,
                        blend_anim,
                        weight=props.blend_weight,
                        speed=props.play_speed,
                        looping=props.loop_playback,
                        fade_in=props.fade_time,
                        replace=False
                    )
                    self.report({'INFO'}, f"Playing '{anim_name}' + blending '{blend_anim}'")
        else:
            self.report({'WARNING'}, f"Failed to play '{anim_name}'")

        return {'FINISHED'}

    def _play_pose(self, context, armature):
        """Apply pose from library to target armature."""
        import json
        from .bone_groups import BONE_GROUPS, BONE_INDEX

        scene = context.scene
        pose_name = scene.pose_test_name
        bone_group = scene.test_bone_group

        # Find pose in library
        pose_entry = None
        for p in scene.pose_library:
            if p.name == pose_name:
                pose_entry = p
                break

        if not pose_entry:
            self.report({'WARNING'}, f"Pose not found: {pose_name}")
            return {'CANCELLED'}

        try:
            bone_data = json.loads(pose_entry.bone_data_json)
        except json.JSONDecodeError:
            self.report({'ERROR'}, "Invalid pose data")
            return {'CANCELLED'}

        # Get target bones based on bone group
        if bone_group == "ALL":
            target_bones = set(BONE_INDEX.keys())
        elif bone_group in BONE_GROUPS:
            target_bones = set(BONE_GROUPS[bone_group])
        else:
            target_bones = set(BONE_INDEX.keys())

        # Apply pose
        pose_bones = armature.pose.bones
        applied_count = 0

        for bone_name, transform in bone_data.items():
            if bone_name not in target_bones:
                continue

            pose_bone = pose_bones.get(bone_name)
            if pose_bone:
                pose_bone.rotation_mode = 'QUATERNION'
                pose_bone.rotation_quaternion = mathutils.Quaternion((transform[0], transform[1], transform[2], transform[3]))
                pose_bone.location = mathutils.Vector((transform[4], transform[5], transform[6]))
                pose_bone.scale = mathutils.Vector((transform[7], transform[8], transform[9]))
                applied_count += 1

        context.view_layer.update()
        self.report({'INFO'}, f"Applied pose: {pose_name} ({applied_count} bones)")
        return {'FINISHED'}

    def _play_ik_pose_blend(self, context, armature):
        """
        IK Pose mode - loops with IK applied.

        Two modes:
        - POSE_TO_POSE: IK targets from Pose B (blend between two poses)
        - POSE_TO_TARGET: IK targets from external objects (reach toward objects)
        """
        import json
        from .bone_groups import BONE_INDEX
        from ..engine.animations.ik import LEG_IK, ARM_IK

        scene = context.scene
        ik_mode = getattr(scene, 'pose_blend_mode', 'POSE_TO_POSE')
        pose_a_name = scene.pose_blend_a

        # Validate pose A (always needed)
        if pose_a_name == "NONE":
            self.report({'WARNING'}, "Select a Pose")
            return {'CANCELLED'}

        pose_a_entry = None
        for p in scene.pose_library:
            if p.name == pose_a_name:
                pose_a_entry = p
                break

        if not pose_a_entry:
            self.report({'WARNING'}, f"Pose not found: {pose_a_name}")
            return {'CANCELLED'}

        # Get enabled IK chains
        ik_chain_props = [
            ('pose_blend_ik_arm_L', 'pose_blend_ik_arm_L_target', 'arm_L'),
            ('pose_blend_ik_arm_R', 'pose_blend_ik_arm_R_target', 'arm_R'),
            ('pose_blend_ik_leg_L', 'pose_blend_ik_leg_L_target', 'leg_L'),
            ('pose_blend_ik_leg_R', 'pose_blend_ik_leg_R_target', 'leg_R'),
        ]

        active_chains = []
        for enabled_prop, target_prop, chain_name in ik_chain_props:
            if getattr(scene, enabled_prop, False):
                if ik_mode == 'POSE_TO_TARGET':
                    # Need object target
                    if getattr(scene, target_prop, None):
                        active_chains.append(chain_name)
                else:
                    active_chains.append(chain_name)

        if not active_chains:
            if ik_mode == 'POSE_TO_TARGET':
                self.report({'WARNING'}, "Enable at least one IK chain with a target object")
            else:
                self.report({'WARNING'}, "Enable at least one IK chain")
            return {'CANCELLED'}

        # Extract IK targets from Pose B (only in POSE_TO_POSE mode)
        ik_targets = {}
        if ik_mode == 'POSE_TO_POSE':
            pose_b_name = scene.pose_blend_b

            # Get Pose B data
            pose_b_data = None
            if pose_b_name == "REST":
                pose_b_data = {bone: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
                              for bone in BONE_INDEX.keys()}
            else:
                for p in scene.pose_library:
                    if p.name == pose_b_name:
                        try:
                            pose_b_data = json.loads(p.bone_data_json)
                        except:
                            pass
                        break

            # Extract tip positions from Pose B
            if pose_b_data:
                # Store current pose
                original_transforms = {}
                for pbone in armature.pose.bones:
                    original_transforms[pbone.name] = (
                        pbone.rotation_quaternion.copy(),
                        pbone.location.copy(),
                        pbone.scale.copy()
                    )

                # Apply Pose B temporarily
                for bone_name, transform in pose_b_data.items():
                    pbone = armature.pose.bones.get(bone_name)
                    if pbone:
                        pbone.rotation_mode = 'QUATERNION'
                        pbone.rotation_quaternion = mathutils.Quaternion((transform[0], transform[1], transform[2], transform[3]))
                        pbone.location = mathutils.Vector((transform[4], transform[5], transform[6]))
                        pbone.scale = mathutils.Vector((transform[7], transform[8], transform[9]))

                context.view_layer.update()

                # Get tip bone positions
                all_chains = {**LEG_IK, **ARM_IK}
                for chain_name in active_chains:
                    chain_def = all_chains.get(chain_name)
                    if chain_def:
                        tip_bone = armature.pose.bones.get(chain_def["tip"])
                        if tip_bone:
                            tip_world = armature.matrix_world @ tip_bone.head
                            ik_targets[chain_name] = mathutils.Vector(tip_world)

                # Restore original pose
                for bone_name, (rot, loc, scale) in original_transforms.items():
                    pbone = armature.pose.bones.get(bone_name)
                    if pbone:
                        pbone.rotation_quaternion = rot
                        pbone.location = loc
                        pbone.scale = scale

                context.view_layer.update()

        # Store IK targets for timer (empty for POSE_TO_TARGET - uses objects directly)
        _pose_blend_state["ik_targets"] = ik_targets

        # Cache pose data for worker-based timer callback
        try:
            pose_a_data = json.loads(pose_a_entry.bone_data_json)
        except:
            pose_a_data = {}

        pose_b_name = scene.pose_blend_b
        if pose_b_name == "REST":
            pose_b_data_cached = {bone: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
                                  for bone in BONE_INDEX.keys()}
        else:
            pose_b_data_cached = {}
            for p in scene.pose_library:
                if p.name == pose_b_name:
                    try:
                        pose_b_data_cached = json.loads(p.bone_data_json)
                    except:
                        pass
                    break

        _pose_blend_state["pose_a_data"] = pose_a_data
        _pose_blend_state["pose_b_data"] = pose_b_data_cached

        # Log extracted targets for debugging
        if ik_targets:
            from ..developer.dev_logger import log_game
            log_game("IK-SOLVE", f"=== POSE-TO-POSE SETUP ===")
            log_game("IK-SOLVE", f"Pose A: {pose_a_name} ({len(pose_a_data)} bones)")
            log_game("IK-SOLVE", f"Pose B: {pose_b_name} ({len(pose_b_data_cached)} bones)")
            for chain_name, target_pos in ik_targets.items():
                log_game("IK-SOLVE", f"TARGET {chain_name}: ({target_pos.x:.3f}, {target_pos.y:.3f}, {target_pos.z:.3f})")
            log_game("IK-SOLVE", f"=== END SETUP ===")

        # Start the auto-loop timer
        if not bpy.app.timers.is_registered(pose_blend_auto_update):
            _pose_blend_state["start_time"] = time.perf_counter()
            _pose_blend_state["direction"] = 1
            bpy.app.timers.register(pose_blend_auto_update, first_interval=1/60)

        duration = scene.pose_blend_duration
        chains_str = ", ".join(active_chains)
        if ik_mode == 'POSE_TO_POSE':
            pose_b_name = scene.pose_blend_b
            self.report({'INFO'}, f"Pose→Pose: {pose_a_name} ↔ {pose_b_name} ({duration:.1f}s) IK:[{chains_str}]")
        else:
            self.report({'INFO'}, f"Pose→Target: {pose_a_name} ({duration:.1f}s) IK:[{chains_str}]")
        return {'FINISHED'}

    # NOTE: _play_pose_blend removed - POSE mode now always uses _play_ik_pose_blend
    # which automatically loops and always applies IK (that's the point of this mode)

    def _apply_blended_pose(self, armature, pose_a_data, pose_b_data, weight, context):
        """
        Apply a blend of two poses to the armature.

        Uses quaternion slerp for rotations, lerp for location/scale.
        """
        from .bone_groups import BONE_INDEX

        pose_bones = armature.pose.bones
        applied_count = 0

        # Get all bones we have data for
        all_bones = set(pose_a_data.keys()) | set(pose_b_data.keys())

        for bone_name in all_bones:
            if bone_name not in BONE_INDEX:
                continue

            pose_bone = pose_bones.get(bone_name)
            if not pose_bone:
                continue

            # Get transforms (default to identity if missing)
            t_a = pose_a_data.get(bone_name, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            t_b = pose_b_data.get(bone_name, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

            # Quaternion slerp for rotation
            q_a = mathutils.Quaternion((t_a[0], t_a[1], t_a[2], t_a[3]))
            q_b = mathutils.Quaternion((t_b[0], t_b[1], t_b[2], t_b[3]))
            q_blend = q_a.slerp(q_b, weight)

            # Linear interpolation for location and scale
            loc_a = mathutils.Vector((t_a[4], t_a[5], t_a[6]))
            loc_b = mathutils.Vector((t_b[4], t_b[5], t_b[6]))
            loc_blend = loc_a.lerp(loc_b, weight)

            scale_a = mathutils.Vector((t_a[7], t_a[8], t_a[9]))
            scale_b = mathutils.Vector((t_b[7], t_b[8], t_b[9]))
            scale_blend = scale_a.lerp(scale_b, weight)

            # Apply
            pose_bone.rotation_mode = 'QUATERNION'
            pose_bone.rotation_quaternion = q_blend
            pose_bone.location = loc_blend
            pose_bone.scale = scale_blend
            applied_count += 1

        return applied_count

    def _apply_ik_overlay(self, armature, chain, target_obj, influence, context):
        """
        Apply IK to a chain on top of the current pose.

        Uses the proper IK solver with world-to-local quaternion conversion.
        """
        # Check if logging enabled
        log_enabled = getattr(context.scene, 'dev_debug_ik_solve', False)

        # Get target position
        target_pos = target_obj.matrix_world.translation

        # Log IK attempt if rig state logging enabled
        rig_log_enabled = getattr(context.scene, 'dev_debug_rig_state', False)
        if rig_log_enabled:
            from .rig_logger import log_ik_attempt, log_bone_chain
            log_ik_attempt(armature, chain, target_pos, influence)
            log_bone_chain(armature, chain, "IK_BEFORE")

        # Apply IK using proper solver-based function
        result = apply_ik_chain(armature, chain, target_pos, influence, log=log_enabled)

        # Log result if rig state logging enabled
        if rig_log_enabled:
            from .rig_logger import log_ik_result, log_collision_check
            context.view_layer.update()  # Ensure transforms are updated
            log_ik_result(armature, chain, target_pos)
            log_collision_check(armature, chain)

        if result and getattr(context.scene, 'dev_debug_ik_visual', False):
            from ..engine.animations.ik import LEG_IK, ARM_IK
            IK_CHAINS = {**LEG_IK, **ARM_IK}
            chain_def = IK_CHAINS.get(chain)
            if chain_def:
                update_ik_visualization(
                    obj=armature,
                    chain=chain,
                    target_pos=tuple(result["target"]),
                    pole_pos=tuple(result["pole"]),
                    joint_world_pos=result["joint"],
                    is_reachable=result["reach_pct"] <= 100,
                    at_limit=result["reach_pct"] > 95
                )

    def _play_ik(self, context, armature):
        """Apply IK to target armature using unified properties."""
        scene = context.scene

        chain = scene.test_ik_chain
        target_obj = scene.test_ik_target
        influence = scene.test_ik_influence
        pole_dir_name = scene.test_ik_pole
        advanced = scene.test_ik_advanced

        # Parse chain
        parts = chain.split('_')
        if len(parts) != 2:
            self.report({'ERROR'}, f"Invalid chain: {chain}")
            return {'CANCELLED'}

        limb_type, side = parts[0], parts[1]
        is_leg = (limb_type == "leg")

        from ..engine.animations.ik import LEG_IK, ARM_IK, solve_leg_ik, solve_arm_ik
        # Combine into IK_CHAINS for convenience
        IK_CHAINS = {**LEG_IK, **ARM_IK}

        # Get chain definition
        chain_def = IK_CHAINS.get(chain)
        if not chain_def:
            self.report({'ERROR'}, f"Unknown IK chain: {chain}")
            return {'CANCELLED'}

        # Get bones
        root_bone = armature.pose.bones.get(chain_def['root'])
        mid_bone = armature.pose.bones.get(chain_def['mid'])
        tip_bone = armature.pose.bones.get(chain_def['tip'])

        if not all([root_bone, mid_bone, tip_bone]):
            self.report({'ERROR'}, f"Missing bones for {chain}")
            return {'CANCELLED'}

        # Reset to rest first
        for pbone in [root_bone, mid_bone, tip_bone]:
            pbone.rotation_mode = 'QUATERNION'
            pbone.rotation_quaternion = mathutils.Quaternion((1, 0, 0, 0))
        context.view_layer.update()

        # Get positions
        root_pos = np.array(armature.matrix_world @ root_bone.head, dtype=np.float32)
        tip_pos = np.array(armature.matrix_world @ tip_bone.head, dtype=np.float32)

        # Compute target position
        if target_obj:
            target_pos = np.array(target_obj.matrix_world.translation, dtype=np.float32)
            target_info = target_obj.name
        elif advanced:
            offset = scene.test_ik_target_xyz
            target_pos = tip_pos + np.array([offset[0], offset[1], offset[2]], dtype=np.float32)
            target_info = f"offset ({offset[0]:.2f}, {offset[1]:.2f}, {offset[2]:.2f})"
        else:
            # Default: use tip position (no movement)
            target_pos = tip_pos.copy()
            target_info = "tip position"

        # Compute pole position
        pole_offset = scene.test_ik_pole_offset if advanced else 0.5
        pole_dir = get_pole_direction_vector(pole_dir_name, is_leg)
        pole_pos = (root_pos + target_pos) / 2 + pole_dir * pole_offset

        # Solve IK
        if is_leg:
            _, _, joint_world = solve_leg_ik(root_pos, target_pos, pole_pos, side)
        else:
            _, _, joint_world = solve_arm_ik(root_pos, target_pos, pole_pos, side)

        # Update visualization if enabled
        target_dist = float(np.linalg.norm(target_pos - root_pos))
        max_reach = chain_def['reach']
        is_reachable = target_dist <= max_reach
        at_limit = target_dist > (max_reach * 0.95)

        update_ik_visualization(
            obj=armature,
            chain=chain,
            target_pos=tuple(target_pos),
            pole_pos=tuple(pole_pos),
            joint_world_pos=joint_world,
            is_reachable=is_reachable,
            at_limit=at_limit
        )

        # Point bones
        root_rot = point_bone_at_target(armature, root_bone, joint_world)
        root_bone.rotation_quaternion = root_rot
        context.view_layer.update()

        mid_rot = point_bone_at_target(armature, mid_bone, target_pos)
        mid_bone.rotation_quaternion = mid_rot
        context.view_layer.update()

        chain_name = "leg" if is_leg else "arm"
        self.report({'INFO'}, f"Applied {side} {chain_name} IK, target: {target_info}")
        return {'FINISHED'}


class ANIM2_OT_TestStop(Operator):
    """Unified Stop - stops based on test mode, exports rig logs"""
    bl_idname = "anim2.test_stop"
    bl_label = "Stop"
    bl_options = {'REGISTER'}

    def execute(self, context):
        scene = context.scene
        mode = scene.test_mode
        armature = getattr(scene, 'target_armature', None)

        # Log final rig state if enabled
        rig_log_enabled = getattr(scene, 'dev_debug_rig_state', False)
        if rig_log_enabled and armature:
            from .rig_logger import end_rig_test_session
            end_rig_test_session(armature, f"TEST_{mode}")

        if mode == 'ANIMATION':
            # Stop animations
            if armature:
                props = scene.anim2_test
                ctrl = get_test_controller()
                ctrl.stop(armature.name, fade_out=props.fade_time)
                self.report({'INFO'}, f"Stopped animations on {armature.name}")
            else:
                stop_all_animations()
                self.report({'INFO'}, "Stopped all animations")
        elif mode == 'POSE':
            # Stop the IK pose-to-pose timer if running
            if bpy.app.timers.is_registered(pose_blend_auto_update):
                stop_pose_blend_auto()
                disable_ik_visualizer()
                # Clear IK state so visualizer doesn't show stale data
                from .runtime_ik import clear_ik_state
                clear_ik_state()
                self.report({'INFO'}, "Stopped IK pose-to-pose loop")
            else:
                self.report({'INFO'}, "No active loop to stop")
        elif mode == 'IK':
            # Disable IK visualization
            disable_ik_visualizer()
            self.report({'INFO'}, "IK visualization disabled")

        # Export logs if enabled
        if getattr(scene, 'dev_export_session_log', False):
            from ..developer.dev_logger import export_game_log, clear_log
            export_game_log("C:/Users/spenc/Desktop/engine_output_files/diagnostics_latest.txt")
            clear_log()
            self.report({'INFO'}, "Logs exported to diagnostics_latest.txt")

        return {'FINISHED'}


class ANIM2_OT_TestReset(Operator):
    """Reset target armature to rest pose"""
    bl_idname = "anim2.test_reset"
    bl_label = "Reset"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        armature = getattr(scene, 'target_armature', None)
        return armature is not None and armature.type == 'ARMATURE'

    def execute(self, context):
        scene = context.scene
        armature = scene.target_armature

        # Stop any animations first
        ctrl = get_test_controller()
        ctrl.stop(armature.name, fade_out=0.0)

        # Reset all pose bones to rest
        for pbone in armature.pose.bones:
            pbone.rotation_mode = 'QUATERNION'
            pbone.rotation_quaternion = mathutils.Quaternion((1, 0, 0, 0))
            pbone.location = mathutils.Vector((0, 0, 0))
            pbone.scale = mathutils.Vector((1, 1, 1))

        # Clear IK visualization
        disable_ik_visualizer()

        context.view_layer.update()
        self.report({'INFO'}, f"Reset {armature.name} to rest pose")
        return {'FINISHED'}


# ═══════════════════════════════════════════════════════════════════════════════
# REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

classes = [
    ANIM2_TestProperties,
    ANIM2_OT_BakeAll,
    ANIM2_OT_StopAll,
    ANIM2_OT_ClearCache,
    # Unified test suite
    ANIM2_OT_TestPlay,
    ANIM2_OT_TestStop,
    ANIM2_OT_TestReset,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.anim2_test = bpy.props.PointerProperty(type=ANIM2_TestProperties)


def unregister():
    # Stop timer and shutdown engine
    reset_test_controller()

    if hasattr(bpy.types.Scene, 'anim2_test'):
        del bpy.types.Scene.anim2_test

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
