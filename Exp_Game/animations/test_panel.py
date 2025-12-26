# Exp_Game/animations/test_panel.py
"""
Animation 2.0 Test Operators & Properties.

Uses the SAME worker-based animation system as the game.
No duplicate logic - just UI driving the existing system.

UI is in Developer Tools panel (dev_panel.py).

═══════════════════════════════════════════════════════════════════════════════
PERFORMANCE ARCHITECTURE - DO NOT REGRESS
═══════════════════════════════════════════════════════════════════════════════

The pose-to-pose system achieves smooth 60Hz+ performance through this split:

WORKER (multiprocessing - heavy computation):
  - Pose blending: 50+ bone quaternion slerp/lerp
  - Joint limit clamping: rotation constraints per bone
  - NumPy vectorized operations
  - ~100-300μs per frame

MAIN THREAD (Blender - light coordination):
  - Apply bone transforms from worker result
  - view_layer.update() to refresh bone matrices
  - IK computation (needs current bone positions from Blender)
  - GPU draw callbacks

WHY IK IS ON MAIN THREAD:
  IK requires the POST-BLEND bone positions to compute correct rotations.
  The worker doesn't have access to Blender's bone hierarchy after blending.
  Main thread reads bone.head positions after view_layer.update(), then calls
  the numpy IK solver. This is fast (~50μs) and architecturally correct.

CRITICAL: Keep heavy math in worker, keep bpy reads on main thread.
═══════════════════════════════════════════════════════════════════════════════
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
from ..engine.animations.default_limits import DEFAULT_JOINT_LIMITS
from ..engine import EngineCore
from .controller import AnimationController
from .runtime_ik import update_ik_state
from ..developer.dev_logger import start_session, log_game, log_worker_messages, export_game_log, clear_log
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
        # Cache joint limits in workers for anatomical constraints
        _cache_joint_limits_in_workers(_test_engine)
    return _test_engine


def _cache_joint_limits_in_workers(engine: EngineCore):
    """
    Send joint limits to all workers via CACHE_JOINT_LIMITS job.
    This enables anatomical constraint enforcement during pose blending.
    """
    from ..developer.dev_logger import log_game

    # Submit caching job (workers will store the limits)
    job_id = engine.submit_job("CACHE_JOINT_LIMITS", {"limits": DEFAULT_JOINT_LIMITS})

    if job_id is not None:
        # Poll for result to confirm caching
        poll_start = time.perf_counter()
        while (time.perf_counter() - poll_start) < 1.0:  # 1 second timeout
            results = list(engine.poll_results(max_results=10))
            for result in results:
                if result.job_id == job_id and result.job_type == "CACHE_JOINT_LIMITS":
                    if result.success:
                        bone_count = result.result.get("bone_count", 0)
                        log_game("JOINT-LIMITS", f"Cached {bone_count} bone limits in worker")
                        # Log worker messages
                        worker_logs = result.result.get("logs", [])
                        if worker_logs:
                            log_worker_messages(worker_logs)
                        return True
            time.sleep(0.01)

    log_game("JOINT-LIMITS", "WARNING: Failed to cache joint limits in worker")
    return False


def get_test_controller() -> AnimationController:
    """Get or create the test controller."""
    global _test_controller
    if _test_controller is None:
        _test_controller = AnimationController()
    return _test_controller


def reset_test_controller():
    """Reset the test controller and engine."""
    global _test_controller, _test_engine, _active_test_modal

    # Stop unified test modal if running
    if _active_test_modal:
        _active_test_modal._test_mode = ""  # Forces modal to self-cancel

    # Export logs before shutdown
    if _test_engine is not None:
        export_game_log("C:/Users/spenc/Desktop/engine_output_files/diagnostics_latest.txt")
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
        global _stop_requested

        # Signal unified test modal to stop
        _stop_requested = True

        stop_all_animations()

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
    # The solver returns world-space quaternions. We need to convert to pose-local.
    #
    # CRITICAL: Blender's pose_bone.rotation_quaternion is relative to the bone's
    # REST pose, not just its parent. We must account for the bone's rest orientation.
    #
    # Formula: pose_rotation = rest_world.inverted() @ solver_world
    # Where rest_world = armature_world @ parent_posed @ bone_rest_local
    # ═══════════════════════════════════════════════════════════════════════════

    # Get bone data (for rest pose info)
    root_bone_data = armature.data.bones[root_bone.name]

    # Compute the bone's "rest world" rotation:
    # This is what the bone's world rotation would be if pose_rotation were identity
    if root_bone.parent:
        parent_name = root_bone.parent.name
        # Parent's posed world matrix
        parent_posed_world = arm_matrix @ root_bone.parent.matrix
        # Bone's rest orientation relative to parent (in parent's local space)
        parent_bone_data = armature.data.bones[root_bone.parent.name]
        bone_rest_rel_parent = parent_bone_data.matrix_local.inverted() @ root_bone_data.matrix_local
        # Rest world = parent posed @ bone rest local
        rest_world_matrix = parent_posed_world @ bone_rest_rel_parent
    else:
        parent_name = "ARMATURE"
        # Root bone - rest world is armature @ bone rest
        rest_world_matrix = arm_matrix @ root_bone_data.matrix_local

    rest_world_q = rest_world_matrix.to_quaternion()

    # Convert solver's world quaternion to pose-local
    upper_world_q = mathutils.Quaternion((upper_quat[0], upper_quat[1], upper_quat[2], upper_quat[3]))
    upper_local_q = rest_world_q.inverted() @ upper_world_q

    # Store original for blending
    original_upper = root_bone.rotation_quaternion.copy()
    original_lower = mid_bone.rotation_quaternion.copy()

    if log:
        log_game("IK-SOLVE", f"CONVERT parent={parent_name}")
        log_game("IK-SOLVE", f"CONVERT rest_world=({rest_world_q.w:.3f},{rest_world_q.x:.3f},{rest_world_q.y:.3f},{rest_world_q.z:.3f})")
        log_game("IK-SOLVE", f"CONVERT upper_world=({upper_world_q.w:.3f},{upper_world_q.x:.3f},{upper_world_q.y:.3f},{upper_world_q.z:.3f})")
        log_game("IK-SOLVE", f"CONVERT upper_local=({upper_local_q.w:.3f},{upper_local_q.x:.3f},{upper_local_q.y:.3f},{upper_local_q.z:.3f})")
        log_game("IK-SOLVE", f"ORIGINAL upper_rot=({original_upper.w:.3f},{original_upper.x:.3f},{original_upper.y:.3f},{original_upper.z:.3f})")
        log_game("IK-SOLVE", f"ORIGINAL lower_rot=({original_lower.w:.3f},{original_lower.x:.3f},{original_lower.y:.3f},{original_lower.z:.3f})")

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
    # Same formula: pose_rotation = rest_world.inverted() @ solver_world
    mid_bone_data = armature.data.bones[mid_bone.name]

    # Parent is root_bone (which we just rotated and updated)
    mid_parent_posed_world = arm_matrix @ root_bone.matrix
    # Mid bone's rest orientation relative to its parent (root_bone)
    mid_bone_rest_rel_parent = root_bone_data.matrix_local.inverted() @ mid_bone_data.matrix_local
    # Rest world = parent posed @ bone rest local
    mid_rest_world_matrix = mid_parent_posed_world @ mid_bone_rest_rel_parent
    mid_rest_world_q = mid_rest_world_matrix.to_quaternion()

    lower_world_q = mathutils.Quaternion((lower_quat[0], lower_quat[1], lower_quat[2], lower_quat[3]))
    lower_local_q = mid_rest_world_q.inverted() @ lower_world_q

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
# IK HELPER FUNCTIONS
# Used by ANIM2_OT_PlayIK for direct bone manipulation
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
# UNIFIED TEST MODAL OPERATOR
#
# Mimics the game's ExpModal pattern for smooth, frame-synchronized animation.
# Uses window_manager.event_timer_add() instead of bpy.app.timers for reliable timing.
#
# Handles ALL test modes:
# - ANIMATION: Regular animation playback
# - ANIMATION + Crossfade: A↔B crossfade loop
# - POSE: Pose-to-pose blending with IK
# - IK: Direct IK manipulation
# ═══════════════════════════════════════════════════════════════════════════════

# Global reference to active test modal (for stopping from outside)
_active_test_modal = None
_stop_requested = False  # Flag to signal modal to stop


def is_test_modal_active() -> bool:
    """Check if the test modal is currently running."""
    return _active_test_modal is not None and _active_test_modal._test_mode != ""


class ANIM2_OT_TestModal(bpy.types.Operator):
    """
    Unified test modal operator for Animation 2.0 testing.

    Matches the game's ExpModal pattern:
    - Uses event_timer_add() for reliable frame-synchronized updates
    - Handles ESC to cancel
    - Proper cleanup on stop
    - Fixed timestep option (60Hz by default)
    """
    bl_idname = "anim2.test_modal"
    bl_label = "Animation Test Modal"
    bl_options = {'INTERNAL'}

    # ═══════════════════════════════════════════════════════════════════════════
    # STATE TRACKING (like ExpModal)
    # ═══════════════════════════════════════════════════════════════════════════

    _timer = None
    _last_time: float = 0.0
    _start_time: float = 0.0
    _frame_count: int = 0

    # Test mode being run (copied from scene at invoke time)
    _test_mode: str = ""  # 'ANIMATION', 'POSE', 'IK'
    _crossfade_enabled: bool = False

    # NOTE: Crossfade and pose blend state is stored in module-level dicts
    # (_crossfade_state, _pose_blend_state) because they're set up before invoke

    # ═══════════════════════════════════════════════════════════════════════════
    # MODAL LIFECYCLE (matches ExpModal pattern)
    # ═══════════════════════════════════════════════════════════════════════════

    def invoke(self, context, event):
        global _active_test_modal

        scene = context.scene
        self._test_mode = scene.test_mode
        self._crossfade_enabled = getattr(scene, 'test_blend_enabled', False)

        # Initialize timing
        self._last_time = time.perf_counter()
        self._start_time = time.perf_counter()
        self._frame_count = 0

        # Set up the timer (1/30s = 33.33ms, same as game's 30Hz fixed timestep)
        wm = context.window_manager
        self._timer = wm.event_timer_add(1/30, window=context.window)
        wm.modal_handler_add(self)

        # Mark as active
        _active_test_modal = self

        # Start logging session
        start_session()
        log_game("TEST_MODAL", f"Started: mode={self._test_mode} crossfade={self._crossfade_enabled}")

        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        global _active_test_modal, _stop_requested

        # Check stop flag (set by Stop button)
        if _stop_requested:
            _stop_requested = False
            self.cancel(context)
            return {'CANCELLED'}

        # ESC to cancel (like game modal)
        if event.type == 'ESC' and event.value == 'PRESS':
            self.cancel(context)
            return {'CANCELLED'}

        # TIMER event - do the actual work
        if event.type == 'TIMER':
            scene = context.scene

            # Check if we should stop
            if not self._should_continue(context):
                self.cancel(context)
                return {'CANCELLED'}

            # Calculate delta time
            current = time.perf_counter()
            dt = current - self._last_time
            self._last_time = current
            self._frame_count += 1

            # Route to appropriate handler based on mode
            try:
                if self._test_mode == 'ANIMATION':
                    if self._crossfade_enabled:
                        self._step_crossfade(context, dt)
                    self._step_animation(context, dt)
                elif self._test_mode == 'POSE':
                    self._step_pose_blend(context, dt)
                elif self._test_mode == 'IK':
                    self._step_ik(context, dt)
            except Exception as e:
                log_game("TEST_MODAL", f"Error in step: {e}")
                import traceback
                traceback.print_exc()

            return {'RUNNING_MODAL'}

        # Pass through non-timer events so UI remains responsive
        return {'PASS_THROUGH'}

    def cancel(self, context):
        global _active_test_modal

        # Remove timer
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None

        # Log stats
        elapsed = time.perf_counter() - self._start_time
        if elapsed > 0:
            avg_fps = self._frame_count / elapsed
            log_game("TEST_MODAL", f"Stopped: {self._frame_count} frames in {elapsed:.1f}s ({avg_fps:.1f} fps)")

        # Export logs to file
        if context.scene.dev_export_session_log:
            export_game_log("C:/Users/spenc/Desktop/engine_output_files/diagnostics_latest.txt")
            clear_log()

        # Clear reference
        _active_test_modal = None

    # ═══════════════════════════════════════════════════════════════════════════
    # HELPER METHODS
    # ═══════════════════════════════════════════════════════════════════════════

    def _should_continue(self, context) -> bool:
        """Check if the modal should continue running."""
        scene = context.scene
        engine = get_test_engine()

        if not engine.is_alive():
            return False

        # Check timeout
        timeout = getattr(scene.anim2_test, 'playback_timeout', 20.0)
        if timeout > 0:
            elapsed = time.perf_counter() - self._start_time
            if elapsed >= timeout:
                return False

        # Mode-specific checks
        if self._test_mode == 'ANIMATION':
            ctrl = get_test_controller()
            return ctrl.has_active_animations()
        elif self._test_mode == 'POSE':
            return scene.test_mode == 'POSE'
        elif self._test_mode == 'IK':
            return scene.test_mode == 'IK'

        return True

    # ═══════════════════════════════════════════════════════════════════════════
    # ANIMATION MODE STEP (replaces playback_update timer)
    # ═══════════════════════════════════════════════════════════════════════════

    def _step_animation(self, context, dt: float):
        """
        Single step of animation playback.
        Uses WORKER-BASED computation - same as game.
        """
        ctrl = get_test_controller()
        engine = get_test_engine()

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

    # ═══════════════════════════════════════════════════════════════════════════
    # CROSSFADE LOOP STEP (replaces crossfade_loop_update timer)
    # ═══════════════════════════════════════════════════════════════════════════

    def _step_crossfade(self, context, dt: float):
        """
        Crossfade logic - switches between animation A and B.
        Uses module-level _crossfade_state dict for state.
        """
        anim_a = _crossfade_state.get("anim_a", "")
        anim_b = _crossfade_state.get("anim_b", "")

        if not anim_a or not anim_b:
            return

        current_time = time.perf_counter()
        ctrl = get_test_controller()

        # Check if it's time to switch
        if current_time >= _crossfade_state.get("switch_time", 0):
            # Get the animation we're switching TO
            if _crossfade_state.get("active_anim", "A") == "A":
                next_anim = anim_b
                _crossfade_state["active_anim"] = "B"
            else:
                next_anim = anim_a
                _crossfade_state["active_anim"] = "A"

            # Get duration of the next animation
            speed = _crossfade_state.get("speed", 1.0)
            anim_data = ctrl.cache.get(next_anim)
            if anim_data:
                duration = anim_data.duration / speed
            else:
                duration = 2.0  # Fallback

            # Schedule next switch
            _crossfade_state["switch_time"] = current_time + duration

            # Play the next animation with crossfade
            ctrl.play(
                _crossfade_state.get("armature", ""),
                next_anim,
                weight=1.0,
                speed=speed,
                looping=True,
                fade_in=_crossfade_state.get("fade_time", 0.2),
                replace=True
            )

            log_game("CROSSFADE", f"→ {next_anim} (duration={duration:.1f}s)")

    # ═══════════════════════════════════════════════════════════════════════════
    # POSE BLEND STEP (replaces pose_blend_auto_update timer)
    # ═══════════════════════════════════════════════════════════════════════════

    def _step_pose_blend(self, context, dt: float):
        """
        Single step of pose-to-pose blending.
        Uses WORKER-BASED computation - same as game.
        Uses module-level _pose_blend_state dict for state.
        """
        step_start = time.perf_counter()

        # Log entry timing (shows inter-frame gaps)
        log_game("POSE-BLEND", f"FRAME_START dt={dt*1000:.1f}ms frame={self._frame_count}")

        scene = context.scene
        armature = getattr(scene, 'target_armature', None)
        if not armature:
            return

        engine = get_test_engine()
        if not engine.is_alive():
            return

        setup_time = (time.perf_counter() - step_start) * 1000

        # Get timing
        duration = scene.pose_blend_duration
        elapsed = time.perf_counter() - self._start_time

        # Calculate normalized time within current half-cycle
        cycle_time = elapsed % (duration * 2)  # Full back-and-forth cycle

        if cycle_time < duration:
            weight = cycle_time / duration
        else:
            weight = 1.0 - ((cycle_time - duration) / duration)

        weight = max(0.0, min(1.0, weight))
        scene.pose_blend_weight = weight

        # Get pose data from module-level state
        pose_a_data = _pose_blend_state.get("pose_a_data", {})
        pose_b_data = _pose_blend_state.get("pose_b_data", {})

        if not pose_a_data:
            return

        # IK is computed LOCALLY after bone transforms are applied (not in worker)
        # This ensures IK uses the correct blended skeleton positions
        influence = scene.pose_blend_ik_influence

        # Submit POSE_BLEND_COMPUTE job to worker (pose blending only, IK done locally)
        build_start = time.perf_counter()
        apply_limits = getattr(scene, 'test_apply_joint_limits', True)
        job_data = {
            "pose_a": pose_a_data,
            "pose_b": pose_b_data,
            "weight": weight,
            "bone_names": list(set(pose_a_data.keys()) | set(pose_b_data.keys())),
            "ik_chains": [],  # IK computed locally after blend is applied
            "apply_limits": apply_limits,
        }
        build_time = (time.perf_counter() - build_start) * 1000

        submit_start = time.perf_counter()
        job_id = engine.submit_job("POSE_BLEND_COMPUTE", job_data)
        submit_time = (time.perf_counter() - submit_start) * 1000

        # Poll for result with timeout (same-frame sync, matches game's KCC pattern)
        if job_id is not None and job_id >= 0:
            poll_start = time.perf_counter()
            poll_timeout = 0.005  # 5ms max wait (matches game's KCC)
            result_received = False
            poll_count = 0
            sleep_us = 25  # Start at 25µs, exponential backoff
            max_sleep_us = 500  # Cap at 500µs

            while True:
                elapsed = time.perf_counter() - poll_start
                if elapsed >= poll_timeout:
                    break

                poll_count += 1
                results = list(engine.poll_results(max_results=5))
                for result in results:
                    if result.job_type == "POSE_BLEND_COMPUTE" and result.job_id == job_id:
                        if result.success:
                            result_received = True
                            poll_time = (time.perf_counter() - poll_start) * 1000

                            # Apply blended bone transforms
                            apply_start = time.perf_counter()
                            bone_transforms = result.result.get("bone_transforms", {})
                            _apply_bone_transforms(armature, bone_transforms)
                            bone_time = (time.perf_counter() - apply_start) * 1000

                            # Apply IK LOCALLY after bone transforms are applied
                            # (Worker IK is broken - computed with pre-blend positions)
                            ik_start = time.perf_counter()

                            # Update armature to reflect new bone transforms
                            context.view_layer.update()

                            # Compute IK locally using current (blended) skeleton positions
                            ik_chain_props = [
                                ('pose_blend_ik_arm_L', 'arm_L'),
                                ('pose_blend_ik_arm_R', 'arm_R'),
                                ('pose_blend_ik_leg_L', 'leg_L'),
                                ('pose_blend_ik_leg_R', 'leg_R'),
                            ]
                            ik_mode = getattr(scene, 'pose_blend_mode', 'POSE_TO_POSE')

                            # Get both target sets for interpolation
                            ik_targets_a = _pose_blend_state.get("ik_targets_a", {})
                            ik_targets_b = _pose_blend_state.get("ik_targets_b", {})

                            # DIAGNOSTIC: Log on first IK frame to diagnose issues
                            # Use _frame_count == 1 because count is incremented BEFORE this code runs
                            log_ik = (self._frame_count == 1)

                            if log_ik:
                                enabled_chains = [cn for prop, cn in ik_chain_props if getattr(scene, prop, False)]
                                log_game("IK", f"=== IK DIAGNOSTIC START ===")
                                log_game("IK", f"mode={ik_mode} weight={weight:.3f} influence={influence:.2f}")
                                log_game("IK", f"enabled_chains={enabled_chains}")
                                log_game("IK", f"targets_a={list(ik_targets_a.keys())} targets_b={list(ik_targets_b.keys())}")
                                # Log armature world matrix
                                arm_mat = armature.matrix_world
                                log_game("IK", f"armature_pos=({arm_mat[0][3]:.3f},{arm_mat[1][3]:.3f},{arm_mat[2][3]:.3f})")

                            ik_applied_count = 0
                            for enabled_prop, chain_name in ik_chain_props:
                                if getattr(scene, enabled_prop, False):
                                    target_pos = None
                                    if ik_mode == 'POSE_TO_TARGET':
                                        target_prop = enabled_prop + '_target'
                                        target_obj = getattr(scene, target_prop, None)
                                        if target_obj:
                                            target_pos = target_obj.matrix_world.translation
                                    else:
                                        # INTERPOLATE between Pose A and Pose B targets based on weight
                                        if chain_name in ik_targets_a and chain_name in ik_targets_b:
                                            local_a = ik_targets_a[chain_name]
                                            local_b = ik_targets_b[chain_name]
                                            # Linear interpolation in local space
                                            local_target = local_a.lerp(local_b, weight)
                                            # Transform to world space
                                            target_pos = armature.matrix_world @ local_target
                                            if log_ik:
                                                log_game("IK", f"CHAIN {chain_name}:")
                                                log_game("IK", f"  local_A=({local_a.x:.3f},{local_a.y:.3f},{local_a.z:.3f})")
                                                log_game("IK", f"  local_B=({local_b.x:.3f},{local_b.y:.3f},{local_b.z:.3f})")
                                                log_game("IK", f"  local_interp=({local_target.x:.3f},{local_target.y:.3f},{local_target.z:.3f})")
                                                log_game("IK", f"  world_target=({target_pos.x:.3f},{target_pos.y:.3f},{target_pos.z:.3f})")

                                    if target_pos:
                                        # Always log on first frame for diagnostics
                                        apply_ik_chain(armature, chain_name, target_pos, influence, log=log_ik)
                                        ik_applied_count += 1
                                    elif log_ik:
                                        log_game("IK", f"CHAIN {chain_name}: NO_TARGET")

                            if log_ik:
                                log_game("IK", f"Applied IK to {ik_applied_count} chains")
                                log_game("IK", f"=== IK DIAGNOSTIC END ===")

                            ik_time = (time.perf_counter() - ik_start) * 1000

                            # Log timing breakdown
                            total_step = (time.perf_counter() - step_start) * 1000
                            worker_calc = result.result.get("calc_time_us", 0) / 1000  # Convert to ms
                            log_game("POSE-BLEND", f"OK total={total_step:.1f}ms setup={setup_time:.1f}ms build={build_time:.1f}ms submit={submit_time:.1f}ms poll={poll_time:.1f}ms({poll_count}) worker={worker_calc:.1f}ms bones={bone_time:.1f}ms ik={ik_time:.1f}ms")

                            # Process worker logs
                            worker_logs = result.result.get("logs", [])
                            if worker_logs:
                                log_worker_messages(worker_logs)
                        break
                else:
                    # Exponential backoff (matches game's KCC pattern)
                    time.sleep(sleep_us / 1_000_000)
                    sleep_us = min(sleep_us * 2, max_sleep_us)
                    continue
                break

            # Log if we timed out waiting for result
            if not result_received:
                poll_time = (time.perf_counter() - poll_start) * 1000
                log_game("POSE-BLEND", f"TIMEOUT after {poll_time:.1f}ms polls={poll_count} setup={setup_time:.1f}ms build={build_time:.1f}ms submit={submit_time:.1f}ms")

    # ═══════════════════════════════════════════════════════════════════════════
    # IK MODE STEP (placeholder for direct IK manipulation)
    # ═══════════════════════════════════════════════════════════════════════════

    def _step_ik(self, context, dt: float):
        """
        IK mode step - currently just maintains IK visualization.
        Could be extended for continuous IK updates.
        """
        pass  # IK mode is currently instant-apply, no continuous updates needed


# ═══════════════════════════════════════════════════════════════════════════════
# TEST MODAL CONTROL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def is_test_modal_running() -> bool:
    """Check if the test modal is currently running."""
    return _active_test_modal is not None


def stop_test_modal():
    """Signal the test modal to stop (it will self-cancel on next timer event)."""
    global _active_test_modal
    if _active_test_modal:
        # The modal checks _should_continue() which will now return False
        # because we're changing the scene state
        pass  # Modal will detect state change and self-cancel


def get_test_modal():
    """Get the active test modal operator (for setup calls)."""
    return _active_test_modal


# ═══════════════════════════════════════════════════════════════════════════════
# ANIMATION HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def stop_all_animations():
    """Stop animations on ALL objects."""
    ctrl = get_test_controller()
    # Stop each object's animations with fade
    for object_name in list(ctrl._states.keys()):
        ctrl.stop(object_name, fade_out=0.2)


# ═══════════════════════════════════════════════════════════════════════════════
# CROSSFADE STATE (used by unified ANIM2_OT_TestModal)
# ═══════════════════════════════════════════════════════════════════════════════

_crossfade_state = {
    "active_anim": "A",  # Which animation is currently playing ("A" or "B")
    "switch_time": 0.0,  # Time when we should switch to the other animation
    "start_time": 0.0,   # When we started
    "anim_a": "",        # Animation A name
    "anim_b": "",        # Animation B name
    "armature": "",      # Target armature name
    "fade_time": 0.2,    # Crossfade duration
    "speed": 1.0,        # Playback speed
}


# ═══════════════════════════════════════════════════════════════════════════════
# POSE BLEND STATE (used by unified ANIM2_OT_TestModal)
# ═══════════════════════════════════════════════════════════════════════════════

_pose_blend_state = {
    "start_time": 0.0,
    "direction": 1,  # 1 = forward (A→B), -1 = backward (B→A)
    "ik_targets_a": {},  # chain_name -> LOCAL position from Pose A (start)
    "ik_targets_b": {},  # chain_name -> LOCAL position from Pose B (end)
    "pose_a_data": {},  # Cached pose A bone transforms
    "pose_b_data": {},  # Cached pose B bone transforms
}


# ═══════════════════════════════════════════════════════════════════════════════
# BONE TRANSFORM APPLICATION (used by unified ANIM2_OT_TestModal)
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_bone_transforms(armature, bone_transforms: dict):
    """
    Apply bone transforms from worker result to armature.

    PERFORMANCE: Uses direct property access to minimize Blender notifications.
    Each bone property set triggers internal Blender updates, so we minimize writes.
    """
    pose_bones = armature.pose.bones

    # Pre-cache the identity values to skip unchanged transforms
    IDENTITY_LOC = (0.0, 0.0, 0.0)
    IDENTITY_SCALE = (1.0, 1.0, 1.0)

    for bone_name, transform in bone_transforms.items():
        pose_bone = pose_bones.get(bone_name)
        if pose_bone:
            # Only set rotation_mode once (check avoids unnecessary update)
            if pose_bone.rotation_mode != 'QUATERNION':
                pose_bone.rotation_mode = 'QUATERNION'

            # Always set rotation (this is what we're animating)
            pose_bone.rotation_quaternion = (transform[0], transform[1], transform[2], transform[3])

            # Only set location/scale if not identity (skip unnecessary writes)
            loc = (transform[4], transform[5], transform[6])
            if loc != IDENTITY_LOC:
                pose_bone.location = loc

            scale = (transform[7], transform[8], transform[9])
            if scale != IDENTITY_SCALE:
                pose_bone.scale = scale


def _apply_ik_results(armature, ik_results: dict):
    """
    Apply IK results from worker to armature bones.

    OPTIMIZED: No view_layer.update() calls - uses manual matrix computation.
    PERFORMANCE: No imports inside this function - all at module level.
    """
    if not ik_results:
        return

    # LEG_IK, ARM_IK already imported at module level
    all_chains = {**LEG_IK, **ARM_IK}
    arm_matrix = armature.matrix_world
    pose_bones = armature.pose.bones

    # Check if visualizer is enabled (skip updates if not)
    visualizer_enabled = getattr(bpy.context.scene, 'dev_debug_ik_visual', False)
    log_enabled = getattr(bpy.context.scene, 'dev_debug_ik_solve', False)

    # Collect data for two-pass application
    upper_bone_data = []  # (root_bone, upper_local_q, original_upper, influence)
    lower_bone_data = []  # (mid_bone, lower_quat_world, original_lower, influence, upper_posed_world_q, root_bone_data, mid_bone_data)
    visualizer_updates = []  # Batch visualizer updates (only if enabled)

    # PASS 1: Prepare upper bones and apply them
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
            # Convert upper world quaternion to pose-local
            # CRITICAL: Must account for bone's rest orientation, not just parent rotation
            # Formula: pose_rotation = rest_world.inverted() @ solver_world
            root_bone_data = armature.data.bones[root_bone.name]

            if root_bone.parent:
                parent_posed_world = arm_matrix @ root_bone.parent.matrix
                parent_bone_data = armature.data.bones[root_bone.parent.name]
                bone_rest_rel_parent = parent_bone_data.matrix_local.inverted() @ root_bone_data.matrix_local
                rest_world_matrix = parent_posed_world @ bone_rest_rel_parent
            else:
                rest_world_matrix = arm_matrix @ root_bone_data.matrix_local

            rest_world_q = rest_world_matrix.to_quaternion()
            upper_world_q = mathutils.Quaternion(upper_quat)
            upper_local_q = rest_world_q.inverted() @ upper_world_q

            # Store original for blending
            original_upper = root_bone.rotation_quaternion.copy()
            original_lower = mid_bone.rotation_quaternion.copy()

            # Apply upper bone immediately
            if root_bone.rotation_mode != 'QUATERNION':
                root_bone.rotation_mode = 'QUATERNION'
            if influence < 1.0:
                final_upper_q = original_upper.slerp(upper_local_q, influence)
            else:
                final_upper_q = upper_local_q
            root_bone.rotation_quaternion = final_upper_q

            if log_enabled:
                log_game("IK-SOLVE", f"APPLY {chain_name} upper: world_q={upper_quat} -> local_q=({upper_local_q.w:.3f},{upper_local_q.x:.3f},{upper_local_q.y:.3f},{upper_local_q.z:.3f}) bone={root_bone.name}")

            # Compute upper bone's final world rotation (after applying pose)
            # WITHOUT calling view_layer.update() (which is extremely slow)
            # upper_world = rest_world @ pose_rotation
            upper_posed_world_q = rest_world_q @ final_upper_q

            # Queue lower bone for pass 2 with upper bone's world rotation and data bone
            mid_bone_data = armature.data.bones[mid_bone.name]
            lower_bone_data.append((mid_bone, lower_quat, original_lower, influence, upper_posed_world_q, root_bone_data, mid_bone_data))

        # Queue visualizer update (only if visualizer is enabled)
        if visualizer_enabled:
            visualizer_updates.append((
                chain_name,
                ik_data.get("target", [0, 0, 0]),
                ik_data.get("joint_world", [0, 0, 0]),
                ik_data.get("root_pos", [0, 0, 0]),
                ik_data.get("pole_pos", [0, 0, 0]),
                influence,
                ik_data.get("reachable", True)
            ))

    # NO view_layer.update() - we computed parent rotations manually above

    # PASS 2: Apply all lower bones using pre-computed parent rotations
    for mid_bone, lower_quat, original_lower, influence, upper_posed_world_q, root_bone_data, mid_bone_data in lower_bone_data:
        # Convert lower world quaternion to pose-local
        # CRITICAL: Must account for mid bone's rest orientation
        # mid_rest_world = upper_posed_world @ mid_rest_rel_parent
        mid_rest_rel_parent = root_bone_data.matrix_local.inverted() @ mid_bone_data.matrix_local
        mid_rest_world_q = upper_posed_world_q @ mid_rest_rel_parent.to_quaternion()

        lower_world_q = mathutils.Quaternion(lower_quat)
        lower_local_q = mid_rest_world_q.inverted() @ lower_world_q

        # Apply lower bone
        if mid_bone.rotation_mode != 'QUATERNION':
            mid_bone.rotation_mode = 'QUATERNION'
        if influence < 1.0:
            final_lower_q = original_lower.slerp(lower_local_q, influence)
        else:
            final_lower_q = lower_local_q
        mid_bone.rotation_quaternion = final_lower_q

        if log_enabled:
            log_game("IK-SOLVE", f"APPLY lower: world_q={lower_quat} -> local_q=({lower_local_q.w:.3f},{lower_local_q.x:.3f},{lower_local_q.y:.3f},{lower_local_q.z:.3f}) bone={mid_bone.name}")

    # Batch update visualizer state (only if enabled and there are updates)
    # update_ik_state already imported at module level
    if visualizer_updates:
        for chain_name, target, joint, root, pole, influence, reachable in visualizer_updates:
            update_ik_state(
                chain=chain_name,
                target_pos=np.array(target, dtype=np.float32),
                mid_pos=np.array(joint, dtype=np.float32),
                root_pos=np.array(root, dtype=np.float32),
                pole_pos=np.array(pole, dtype=np.float32),
                influence=influence,
                reachable=reachable
            )


def _apply_ik_overlay_standalone(armature, chain, target_obj, influence):
    """Standalone IK overlay for timer callback (no self reference)."""
    target_pos = target_obj.matrix_world.translation
    _apply_ik_overlay_with_position(armature, chain, target_pos, influence)


def _apply_ik_overlay_with_position(armature, chain, target_pos, influence):
    """Apply IK overlay using a world-space position."""
    log_enabled = getattr(bpy.context.scene, 'dev_debug_ik_solve', False)
    apply_ik_chain(armature, chain, target_pos, influence, log=log_enabled)


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

        scene = context.scene

        # Check if crossfade loop is enabled
        crossfade_enabled = getattr(scene, 'test_blend_enabled', False)
        blend_anim = props.blend_animation if crossfade_enabled else None

        if crossfade_enabled:
            if not blend_anim or not ctrl.has_animation(blend_anim):
                self.report({'WARNING'}, "Select a secondary animation for crossfade")
                return {'CANCELLED'}

        # Start the animation (modal will handle playback)
        success = ctrl.play(
            armature.name,
            anim_name,
            weight=1.0,
            speed=props.play_speed,
            looping=props.loop_playback,
            fade_in=props.fade_time,
            replace=True
        )

        if not success:
            self.report({'WARNING'}, f"Failed to play '{anim_name}'")
            return {'CANCELLED'}

        # Start the unified test modal
        if not is_test_modal_running():
            # Store crossfade state in module-level dict for modal to access
            if crossfade_enabled and blend_anim:
                # Set up crossfade state for the modal
                _crossfade_state["active_anim"] = "A"
                _crossfade_state["anim_a"] = anim_name
                _crossfade_state["anim_b"] = blend_anim
                _crossfade_state["armature"] = armature.name
                _crossfade_state["fade_time"] = props.fade_time
                _crossfade_state["speed"] = props.play_speed
                # Calculate switch time
                anim_data = ctrl.cache.get(anim_name)
                duration = anim_data.duration / props.play_speed if anim_data else 2.0
                _crossfade_state["switch_time"] = time.perf_counter() + duration
                _crossfade_state["start_time"] = time.perf_counter()

            bpy.ops.anim2.test_modal('INVOKE_DEFAULT')

        if crossfade_enabled:
            self.report({'INFO'}, f"Crossfade loop: {anim_name} ↔ {blend_anim}")
        else:
            self.report({'INFO'}, f"Playing '{anim_name}' on {armature.name}")

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

        # Extract IK targets from BOTH Pose A and Pose B (for interpolation)
        ik_targets_a = {}  # Targets from Pose A (start)
        ik_targets_b = {}  # Targets from Pose B (end)

        if ik_mode == 'POSE_TO_POSE':
            all_chains = {**LEG_IK, **ARM_IK}

            # Helper function to extract tip positions from a pose
            def extract_tip_positions(pose_data, target_dict, pose_name):
                # Store current pose
                original_transforms = {}
                for pbone in armature.pose.bones:
                    original_transforms[pbone.name] = (
                        pbone.rotation_quaternion.copy(),
                        pbone.location.copy(),
                        pbone.scale.copy()
                    )

                # Apply pose temporarily
                for bone_name, transform in pose_data.items():
                    pbone = armature.pose.bones.get(bone_name)
                    if pbone:
                        pbone.rotation_mode = 'QUATERNION'
                        pbone.rotation_quaternion = mathutils.Quaternion((transform[0], transform[1], transform[2], transform[3]))
                        pbone.location = mathutils.Vector((transform[4], transform[5], transform[6]))
                        pbone.scale = mathutils.Vector((transform[7], transform[8], transform[9]))

                context.view_layer.update()

                # Get tip bone positions in LOCAL space
                for chain_name in active_chains:
                    chain_def = all_chains.get(chain_name)
                    if chain_def:
                        tip_bone = armature.pose.bones.get(chain_def["tip"])
                        if tip_bone:
                            tip_local = tip_bone.head.copy()
                            target_dict[chain_name] = tip_local
                            log_game("IK", f"CAPTURE_{pose_name} {chain_name} local=({tip_local.x:.3f},{tip_local.y:.3f},{tip_local.z:.3f})")

                # Restore original pose
                for bone_name, (rot, loc, scale) in original_transforms.items():
                    pbone = armature.pose.bones.get(bone_name)
                    if pbone:
                        pbone.rotation_quaternion = rot
                        pbone.location = loc
                        pbone.scale = scale

                context.view_layer.update()

            # Get Pose A data
            pose_a_data_ik = {}
            try:
                pose_a_data_ik = json.loads(pose_a_entry.bone_data_json)
            except:
                pass

            # Get Pose B data
            pose_b_name = scene.pose_blend_b
            pose_b_data_ik = None
            if pose_b_name == "REST":
                pose_b_data_ik = {bone: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
                              for bone in BONE_INDEX.keys()}
            else:
                for p in scene.pose_library:
                    if p.name == pose_b_name:
                        try:
                            pose_b_data_ik = json.loads(p.bone_data_json)
                        except:
                            pass
                        break

            # Extract targets from Pose A
            if pose_a_data_ik:
                extract_tip_positions(pose_a_data_ik, ik_targets_a, "A")

            # Extract targets from Pose B
            if pose_b_data_ik:
                extract_tip_positions(pose_b_data_ik, ik_targets_b, "B")

        # Store BOTH sets of IK targets for interpolation during blend
        _pose_blend_state["ik_targets_a"] = ik_targets_a  # Start positions
        _pose_blend_state["ik_targets_b"] = ik_targets_b  # End positions
        log_game("IK", f"STORED targets_a={list(ik_targets_a.keys())} targets_b={list(ik_targets_b.keys())}")

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

        # Log extracted targets for debugging (now in LOCAL armature space)
        if ik_targets_a or ik_targets_b:
            log_game("IK-SOLVE", f"=== POSE-TO-POSE SETUP ===")
            log_game("IK-SOLVE", f"Pose A: {pose_a_name} ({len(pose_a_data)} bones)")
            log_game("IK-SOLVE", f"Pose B: {pose_b_name} ({len(pose_b_data_cached)} bones)")
            for chain_name in ik_targets_a:
                local_a = ik_targets_a.get(chain_name)
                local_b = ik_targets_b.get(chain_name)
                if local_a and local_b:
                    log_game("IK-SOLVE", f"TARGET {chain_name}: A=({local_a.x:.3f},{local_a.y:.3f},{local_a.z:.3f}) B=({local_b.x:.3f},{local_b.y:.3f},{local_b.z:.3f})")
            log_game("IK-SOLVE", f"=== END SETUP ===")

        # Start the unified test modal for smooth animation
        if not is_test_modal_running():
            _pose_blend_state["direction"] = 1
            bpy.ops.anim2.test_modal('INVOKE_DEFAULT')

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

        # Stop the unified test modal if running (handles all modes)
        was_running = is_test_modal_running()
        if was_running:
            # Signal modal to stop via flag (checked on next event)
            global _stop_requested
            _stop_requested = True

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
            # Disable IK visualization
            disable_ik_visualizer()
            # Clear IK state so visualizer doesn't show stale data
            from .runtime_ik import clear_ik_state
            clear_ik_state()
            if was_running:
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
    # Unified test modal (handles all modes: ANIMATION, POSE, IK)
    ANIM2_OT_TestModal,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.anim2_test = bpy.props.PointerProperty(type=ANIM2_TestProperties)


def unregister():
    # Stop unified test modal and shutdown engine
    global _active_test_modal
    if _active_test_modal:
        _active_test_modal._test_mode = ""  # Forces modal to self-cancel
    reset_test_controller()

    if hasattr(bpy.types.Scene, 'anim2_test'):
        del bpy.types.Scene.anim2_test

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
