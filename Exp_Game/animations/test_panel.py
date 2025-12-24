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
# IK TEST OPERATORS
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
            # Need poses in library
            pose_name = getattr(scene, 'pose_test_name', 'NONE')
            return pose_name and pose_name != 'NONE'
        elif mode == 'IK':
            return True
        return False

    def execute(self, context):
        scene = context.scene
        mode = scene.test_mode
        armature = scene.target_armature

        if mode == 'ANIMATION':
            return self._play_animation(context, armature)
        elif mode == 'POSE':
            return self._play_pose(context, armature)
        elif mode == 'IK':
            return self._play_ik(context, armature)

        return {'CANCELLED'}

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

        from .ik_solver import IK_CHAINS, solve_leg_ik, solve_arm_ik

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
    """Unified Stop - stops based on test mode"""
    bl_idname = "anim2.test_stop"
    bl_label = "Stop"
    bl_options = {'REGISTER'}

    def execute(self, context):
        scene = context.scene
        mode = scene.test_mode
        armature = getattr(scene, 'target_armature', None)

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
            # Nothing to stop for pose (it's instant)
            self.report({'INFO'}, "Pose applied instantly - nothing to stop")
        elif mode == 'IK':
            # Disable IK visualization
            disable_ik_visualizer()
            self.report({'INFO'}, "IK visualization disabled")

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
