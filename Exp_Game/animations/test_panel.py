# Exp_Game/animations/test_panel.py
"""
Animation 2.0 Test Suite - MINIMAL VERSION

Stripped down to essentials:
- Bake animations
- Play animations on armature
- GPU visualization for debugging
- Engine/controller management

IK and pose blending will be rebuilt properly from scratch.
"""

import bpy
import time
from bpy.types import Operator, PropertyGroup
from bpy.props import FloatProperty, BoolProperty, EnumProperty, PointerProperty

from ..engine.animations.baker import bake_action
from ..engine.animations.ik import LEG_IK, ARM_IK
from ..engine import EngineCore
from .controller import AnimationController
from ..developer.dev_logger import start_session, log_game, log_worker_messages, export_game_log, clear_log
import gpu
from gpu_extras.batch import batch_for_shader

from ..developer.gpu_utils import (
    get_cached_shader,
    CIRCLE_8,
    sphere_wire_verts,
    layered_sphere_verts,
    extend_batch_data,
    crosshair_verts,
    arrow_head_verts,
)


# =============================================================================
# GPU IK VISUALIZER
# =============================================================================

_ik_draw_handler = None
_ik_vis_data = None


def _draw_ik_visual():
    """GPU draw callback for IK visualization."""
    global _ik_vis_data

    scene = bpy.context.scene
    if not getattr(scene, 'dev_debug_ik_visual', False):
        return

    # Check for runtime IK state
    from .runtime_ik import get_ik_state, is_ik_active
    runtime_state = get_ik_state()

    if is_ik_active() and runtime_state.get("active"):
        vis_data = _build_ik_vis_data(runtime_state)
    elif _ik_vis_data is not None:
        vis_data = _ik_vis_data
    else:
        return

    # GPU state
    gpu.state.depth_test_set('NONE')
    gpu.state.blend_set('ALPHA')
    gpu.state.line_width_set(getattr(scene, 'dev_debug_ik_line_width', 2.5))

    shader = get_cached_shader()
    all_verts = []
    all_colors = []

    # Draw targets
    if 'targets' in vis_data:
        for target in vis_data['targets']:
            pos = target['pos']
            reachable = target.get('reachable', True)
            color = (0.2, 1.0, 0.2, 0.9) if reachable else (1.0, 0.2, 0.2, 0.9)
            extend_batch_data(all_verts, all_colors, sphere_wire_verts(pos, 0.05, CIRCLE_8), color)

    # Draw chains
    if 'chains' in vis_data:
        for chain in vis_data['chains']:
            root, mid, tip = chain['root'], chain['mid'], chain['tip']
            all_verts.extend([root, mid])
            all_colors.extend([(0.0, 1.0, 1.0, 0.9)] * 2)
            all_verts.extend([mid, tip])
            all_colors.extend([(1.0, 0.0, 1.0, 0.9)] * 2)

    # Draw reach spheres
    if 'reach_spheres' in vis_data:
        for sphere in vis_data['reach_spheres']:
            extend_batch_data(
                all_verts, all_colors,
                layered_sphere_verts(sphere['center'], sphere['radius'], (-0.5, 0.0, 0.5), CIRCLE_8),
                (1.0, 0.8, 0.0, 0.2)
            )

    # Draw joints
    if 'joints' in vis_data:
        for joint in vis_data['joints']:
            pos = joint['pos']
            color = (1.0, 1.0, 1.0, 0.9) if joint.get('type') == 'root' else (0.0, 1.0, 1.0, 0.9)
            extend_batch_data(all_verts, all_colors, crosshair_verts(pos, 0.03), color)

    # Draw
    if all_verts:
        batch = batch_for_shader(shader, 'LINES', {"pos": all_verts, "color": all_colors})
        shader.bind()
        batch.draw(shader)

    # Reset
    gpu.state.line_width_set(1.0)
    gpu.state.depth_test_set('NONE')
    gpu.state.blend_set('NONE')


def _build_ik_vis_data(state: dict) -> dict:
    """Build visualization data from runtime IK state."""
    vis_data = {'targets': [], 'chains': [], 'reach_spheres': [], 'joints': []}

    target = state.get('last_target')
    mid_pos = state.get('last_mid_pos')
    root_pos = state.get('root_pos')
    chain_name = state.get('chain', 'arm_R')
    reachable = state.get('reachable', True)

    if target is None or root_pos is None:
        return vis_data

    # Get chain reach
    chain_def = LEG_IK.get(chain_name) or ARM_IK.get(chain_name, {})
    max_reach = chain_def.get('reach', 0.5)

    vis_data['targets'].append({'pos': tuple(target), 'reachable': reachable})

    if mid_pos is not None:
        vis_data['chains'].append({'root': tuple(root_pos), 'mid': tuple(mid_pos), 'tip': tuple(target)})

    vis_data['reach_spheres'].append({'center': tuple(root_pos), 'radius': max_reach})
    vis_data['joints'].append({'pos': tuple(root_pos), 'type': 'root'})
    if mid_pos is not None:
        vis_data['joints'].append({'pos': tuple(mid_pos), 'type': 'mid'})
    vis_data['joints'].append({'pos': tuple(target), 'type': 'tip'})

    return vis_data


def enable_ik_visualizer():
    """Register IK visualization draw handler."""
    global _ik_draw_handler
    if _ik_draw_handler is None:
        _ik_draw_handler = bpy.types.SpaceView3D.draw_handler_add(_draw_ik_visual, (), 'WINDOW', 'POST_VIEW')


def disable_ik_visualizer():
    """Unregister IK visualization draw handler."""
    global _ik_draw_handler, _ik_vis_data
    if _ik_draw_handler is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_ik_draw_handler, 'WINDOW')
        except:
            pass
        _ik_draw_handler = None
    _ik_vis_data = None


def set_ik_vis_data(data: dict):
    """Set visualization data from external source."""
    global _ik_vis_data
    _ik_vis_data = data
    if _ik_draw_handler is None:
        enable_ik_visualizer()


# =============================================================================
# TEST ENGINE & CONTROLLER
# =============================================================================

_test_engine = None
_test_controller = None


def get_test_engine() -> EngineCore:
    """Get or create test engine."""
    global _test_engine
    if _test_engine is None or not _test_engine.is_alive():
        _test_engine = EngineCore()
        _test_engine.start()
        _test_engine.wait_for_readiness(timeout=2.0)
        start_session()
    return _test_engine


def get_test_controller() -> AnimationController:
    """Get or create test controller."""
    global _test_controller
    if _test_controller is None:
        _test_controller = AnimationController()
    return _test_controller


def reset_test_controller():
    """Reset test controller and engine."""
    global _test_controller, _test_engine

    # Export logs
    if _test_engine is not None:
        export_game_log("C:/Users/spenc/Desktop/engine_output_files/diagnostics_latest.txt")
        clear_log()
        _test_engine.shutdown()
        _test_engine = None

    disable_ik_visualizer()
    _test_controller = None


# =============================================================================
# BASIC OPERATORS
# =============================================================================

class ANIM2_OT_BakeAll(Operator):
    """Bake all actions and cache in workers"""
    bl_idname = "anim2.bake_all"
    bl_label = "Bake All Actions"
    bl_options = {'REGISTER'}

    def execute(self, context):
        start_time = time.perf_counter()

        reset_test_controller()
        ctrl = get_test_controller()
        engine = get_test_engine()

        baked_count = 0
        failed = []

        for action in bpy.data.actions:
            try:
                anim = bake_action(action)
                ctrl.add_animation(anim)
                baked_count += 1
            except Exception as e:
                failed.append(f"{action.name}: {e}")

        # Cache in workers
        if baked_count > 0 and engine.is_alive():
            cache_data = ctrl.get_cache_data_for_workers()
            engine.broadcast_job("CACHE_ANIMATIONS", cache_data)

            # Wait for confirmation
            wait_start = time.perf_counter()
            while (time.perf_counter() - wait_start) < 1.0:
                results = list(engine.poll_results(max_results=20))
                for r in results:
                    if r.job_type == "CACHE_ANIMATIONS" and r.success:
                        break
                time.sleep(0.01)

        elapsed = (time.perf_counter() - start_time) * 1000

        if failed:
            self.report({'WARNING'}, f"Baked {baked_count} actions ({elapsed:.0f}ms). {len(failed)} failed.")
        else:
            self.report({'INFO'}, f"Baked {baked_count} actions in {elapsed:.0f}ms")

        return {'FINISHED'}


class ANIM2_OT_StopAll(Operator):
    """Stop all animations"""
    bl_idname = "anim2.stop_all"
    bl_label = "Stop All"
    bl_options = {'REGISTER'}

    def execute(self, context):
        global _stop_requested
        _stop_requested = True

        ctrl = get_test_controller()
        if ctrl:
            ctrl.clear_all()

        self.report({'INFO'}, "Stopped all animations")
        return {'FINISHED'}


class ANIM2_OT_ClearCache(Operator):
    """Clear animation cache and shutdown test engine"""
    bl_idname = "anim2.clear_cache"
    bl_label = "Clear Cache"
    bl_options = {'REGISTER'}

    def execute(self, context):
        reset_test_controller()
        self.report({'INFO'}, "Animation cache cleared, engine stopped")
        return {'FINISHED'}


# =============================================================================
# TEST MODAL (ANIMATION PLAYBACK ONLY)
# =============================================================================

_active_test_modal = None
_stop_requested = False


def is_test_modal_running() -> bool:
    return _active_test_modal is not None


# Alias for backwards compatibility
def is_test_modal_active() -> bool:
    return is_test_modal_running()


def stop_test_modal():
    global _stop_requested
    _stop_requested = True


class ANIM2_OT_TestModal(Operator):
    """Animation test modal - plays animations at 30Hz"""
    bl_idname = "anim2.test_modal"
    bl_label = "Animation Test Modal"
    bl_options = {'INTERNAL'}

    _timer = None
    _last_time: float = 0.0
    _start_time: float = 0.0
    _frame_count: int = 0

    def invoke(self, context, event):
        global _active_test_modal

        self._last_time = time.perf_counter()
        self._start_time = time.perf_counter()
        self._frame_count = 0

        wm = context.window_manager
        self._timer = wm.event_timer_add(1/30, window=context.window)
        wm.modal_handler_add(self)

        _active_test_modal = self
        start_session()
        log_game("TEST_MODAL", "Started animation playback")

        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        global _active_test_modal, _stop_requested

        if _stop_requested:
            _stop_requested = False
            self.cancel(context)
            return {'CANCELLED'}

        if event.type == 'ESC' and event.value == 'PRESS':
            self.cancel(context)
            return {'CANCELLED'}

        if event.type == 'TIMER':
            scene = context.scene
            props = scene.anim2_test
            armature = getattr(scene, 'target_armature', None)

            if not armature:
                self.cancel(context)
                return {'CANCELLED'}

            # Check timeout
            timeout = getattr(props, 'playback_timeout', 20.0)
            if timeout > 0 and (time.perf_counter() - self._start_time) >= timeout:
                self.cancel(context)
                return {'CANCELLED'}

            # Update timing
            current = time.perf_counter()
            dt = current - self._last_time
            self._last_time = current
            self._frame_count += 1

            # Step animation
            self._step_animation(context, dt, armature)

            return {'RUNNING_MODAL'}

        return {'PASS_THROUGH'}

    def _step_animation(self, context, dt: float, armature):
        """Update and apply animation."""
        ctrl = get_test_controller()
        engine = get_test_engine()

        if not ctrl or not engine or not engine.is_alive():
            return

        # Update controller state
        ctrl.update_state(dt)

        # Get job data and submit to engine
        jobs_data = ctrl.get_compute_job_data()
        if not jobs_data:
            return

        # Submit batch job
        job_id = engine.submit_job("ANIMATION_COMPUTE_BATCH", {"objects": jobs_data})
        if job_id is None or job_id < 0:
            return

        # Poll for result (with short timeout)
        poll_start = time.perf_counter()
        while (time.perf_counter() - poll_start) < 0.005:  # 5ms max
            results = list(engine.poll_results(max_results=10))
            for result in results:
                if result.job_type == "ANIMATION_COMPUTE_BATCH" and result.job_id == job_id:
                    if result.success:
                        self._apply_result(armature, result.result)
                        # Log worker messages
                        worker_logs = result.result.get("logs", [])
                        if worker_logs:
                            log_worker_messages(worker_logs)
                    return
            time.sleep(0.0001)

    def _apply_result(self, armature, result_data: dict):
        """Apply animation result to armature."""
        results_dict = result_data.get("results", {})
        if not results_dict:
            return

        for object_name, obj_result in results_dict.items():
            if object_name != armature.name:
                continue

            bone_transforms = obj_result.get("bone_transforms", {})
            if not bone_transforms:
                continue

            pose_bones = armature.pose.bones
            for bone_name, transform in bone_transforms.items():
                pose_bone = pose_bones.get(bone_name)
                if pose_bone:
                    pose_bone.rotation_mode = 'QUATERNION'
                    pose_bone.rotation_quaternion = (transform[0], transform[1], transform[2], transform[3])
                    pose_bone.location = (transform[4], transform[5], transform[6])
                    pose_bone.scale = (transform[7], transform[8], transform[9])

            bpy.context.view_layer.update()

    def cancel(self, context):
        global _active_test_modal

        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None

        elapsed = time.perf_counter() - self._start_time
        if elapsed > 0:
            fps = self._frame_count / elapsed
            log_game("TEST_MODAL", f"Stopped: {self._frame_count} frames in {elapsed:.1f}s ({fps:.1f} fps)")

        if context.scene.dev_export_session_log:
            export_game_log("C:/Users/spenc/Desktop/engine_output_files/diagnostics_latest.txt")
            clear_log()

        _active_test_modal = None


# =============================================================================
# PROPERTIES (MINIMAL)
# =============================================================================

def get_animation_items(self, context):
    ctrl = get_test_controller()
    items = [("", "Select Animation", "")]
    for name in ctrl.cache.names:
        items.append((name, name, ""))
    return items


class ANIM2_TestProperties(PropertyGroup):
    """Minimal test properties."""

    selected_animation: EnumProperty(
        name="Animation",
        description="Animation to play",
        items=get_animation_items
    )

    play_speed: FloatProperty(
        name="Speed",
        description="Playback speed",
        default=1.0,
        min=0.1,
        max=3.0
    )

    loop_playback: BoolProperty(
        name="Loop",
        description="Loop animation",
        default=True
    )

    playback_timeout: FloatProperty(
        name="Timeout",
        description="Auto-stop after seconds (0 = no timeout)",
        default=20.0,
        min=0.0,
        max=300.0
    )


# IK chain items for dropdown
def get_ik_chain_items(self, context):
    """Return available IK chains for dropdown."""
    return [
        ("arm_R", "Right Arm", "Right arm IK chain (shoulder → elbow → hand)"),
        ("arm_L", "Left Arm", "Left arm IK chain (shoulder → elbow → hand)"),
        ("leg_R", "Right Leg", "Right leg IK chain (hip → knee → foot)"),
        ("leg_L", "Left Leg", "Left leg IK chain (hip → knee → foot)"),
    ]


# =============================================================================
# IK STATE TEST OPERATOR
# =============================================================================

class ANIM2_OT_TestIKState(Operator):
    """Analyze current IK state and log to diagnostics file"""
    bl_idname = "anim2.test_ik_state"
    bl_label = "Analyze IK State"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        armature = getattr(context.scene, 'target_armature', None)
        if not armature or armature.type != 'ARMATURE':
            return False
        target = getattr(context.scene, 'ik_test_target', None)
        return target is not None

    def execute(self, context):
        from .ik_state import compute_ik_state, log_ik_state

        scene = context.scene
        armature = scene.target_armature
        target_obj = scene.ik_test_target
        chain = scene.ik_test_chain

        if not armature or not target_obj:
            self.report({'ERROR'}, "Set armature and target object")
            return {'CANCELLED'}

        # Start log session
        start_session()

        # Get target world position
        target_pos = target_obj.matrix_world.translation

        # Compute IK state
        state = compute_ik_state(
            armature=armature,
            chain=chain,
            target_pos=target_pos,
            target_object_name=target_obj.name,
            frame=context.scene.frame_current
        )

        # Log the state
        log_ik_state(state)

        # Update GPU visualization
        from ..engine.animations.ik import LEG_IK, ARM_IK
        chain_def = ARM_IK.get(chain) or LEG_IK.get(chain, {})
        max_reach = chain_def.get('reach', 0.5)

        vis_data = {
            'targets': [{'pos': state.target_pos, 'reachable': state.reachable}],
            'chains': [{'root': state.root_pos, 'mid': state.mid_pos, 'tip': state.tip_pos}],
            'reach_spheres': [{'center': state.root_pos, 'radius': max_reach}],
            'joints': [
                {'pos': state.root_pos, 'type': 'root'},
                {'pos': state.mid_pos, 'type': 'mid'},
                {'pos': state.tip_pos, 'type': 'tip'},
            ]
        }
        set_ik_vis_data(vis_data)

        # Export log
        export_game_log("C:/Users/spenc/Desktop/engine_output_files/diagnostics_latest.txt")
        clear_log()

        # Report result
        error_cm = state.error_distance * 100
        if state.bend_correct and state.reachable:
            self.report({'INFO'}, f"IK OK: error={error_cm:.1f}cm, {state.bend_direction} bend")
        else:
            problems = []
            if not state.bend_correct:
                problems.append(f"bend={state.bend_direction} (should be {state.bend_expected})")
            if not state.reachable:
                problems.append("unreachable")
            self.report({'WARNING'}, f"IK issues: {', '.join(problems)}")

        return {'FINISHED'}


class ANIM2_OT_ApplyIK(Operator):
    """Solve IK and apply to armature - moves bones toward target"""
    bl_idname = "anim2.apply_ik"
    bl_label = "Apply IK"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        armature = getattr(context.scene, 'target_armature', None)
        if not armature or armature.type != 'ARMATURE':
            return False
        target = getattr(context.scene, 'ik_test_target', None)
        return target is not None

    def execute(self, context):
        from .ik_solver import solve_two_bone_ik, apply_ik_to_chain, get_bend_hint
        from .ik_state import compute_ik_state, log_ik_state

        scene = context.scene
        armature = scene.target_armature
        target_obj = scene.ik_test_target
        chain = scene.ik_test_chain

        if not armature or not target_obj:
            self.report({'ERROR'}, "Set armature and target object")
            return {'CANCELLED'}

        # Start log session
        start_session()

        # Get chain definition
        is_arm = chain.startswith("arm")
        chain_def = ARM_IK.get(chain) if is_arm else LEG_IK.get(chain)

        if not chain_def:
            self.report({'ERROR'}, f"Unknown chain: {chain}")
            return {'CANCELLED'}

        # Log BEFORE state
        log_game("IK_APPLY", "=" * 60)
        log_game("IK_APPLY", "BEFORE IK APPLICATION:")
        target_pos = target_obj.matrix_world.translation
        before_state = compute_ik_state(armature, chain, target_pos, target_obj.name, scene.frame_current)
        log_ik_state(before_state)

        # Get bone positions
        pose_bones = armature.pose.bones
        root_bone = pose_bones.get(chain_def["root"])
        mid_bone = pose_bones.get(chain_def["mid"])

        if not root_bone or not mid_bone:
            self.report({'ERROR'}, "Missing bones in chain")
            return {'CANCELLED'}

        arm_matrix = armature.matrix_world
        root_pos = arm_matrix @ root_bone.head
        upper_length = root_bone.bone.length
        lower_length = mid_bone.bone.length

        # Get anatomically correct bend direction
        bend_hint = get_bend_hint(chain, armature)

        log_game("IK_APPLY", "-" * 60)
        log_game("IK_APPLY", "SOLVING IK:")

        # Solve IK
        solution = solve_two_bone_ik(
            root_pos=root_pos,
            upper_length=upper_length,
            lower_length=lower_length,
            target_pos=target_pos,
            bend_hint=bend_hint,
            debug=True
        )

        if not solution.success:
            self.report({'ERROR'}, "IK solve failed")
            export_game_log("C:/Users/spenc/Desktop/engine_output_files/diagnostics_latest.txt")
            clear_log()
            return {'CANCELLED'}

        log_game("IK_APPLY", "-" * 60)
        log_game("IK_APPLY", "APPLYING TO BONES:")

        # Apply solution to bones
        success = apply_ik_to_chain(armature, chain, solution, debug=True)

        if not success:
            self.report({'ERROR'}, "Failed to apply IK to bones")
            export_game_log("C:/Users/spenc/Desktop/engine_output_files/diagnostics_latest.txt")
            clear_log()
            return {'CANCELLED'}

        # Force scene update
        context.view_layer.update()

        # Log AFTER state
        log_game("IK_APPLY", "-" * 60)
        log_game("IK_APPLY", "AFTER IK APPLICATION:")
        after_state = compute_ik_state(armature, chain, target_pos, target_obj.name, scene.frame_current)
        log_ik_state(after_state)

        # Validate the resulting pose
        from .pose_validator import validate_pose
        validation = validate_pose(armature, chain=chain, target_pos=target_pos, debug=True)

        # Summary
        log_game("IK_APPLY", "=" * 60)
        log_game("IK_APPLY", f"ERROR: {before_state.error_distance*100:.1f}cm -> {after_state.error_distance*100:.1f}cm")
        if after_state.error_distance < before_state.error_distance:
            improvement = (before_state.error_distance - after_state.error_distance) * 100
            log_game("IK_APPLY", f"  Improved by {improvement:.1f}cm")
        else:
            log_game("IK_APPLY", f"  WARNING: Error increased!")

        log_game("IK_APPLY", f"VALIDATION: {validation.summary}")
        log_game("IK_APPLY", f"  Score: {validation.score:.2f}")
        log_game("IK_APPLY", "=" * 60)

        # Update GPU visualization with solution
        max_reach = chain_def.get('reach', 0.5)
        vis_data = {
            'targets': [{'pos': tuple(solution.target_pos), 'reachable': not solution.clamped}],
            'chains': [{'root': tuple(solution.root_pos), 'mid': tuple(solution.mid_pos), 'tip': tuple(solution.tip_pos)}],
            'reach_spheres': [{'center': tuple(solution.root_pos), 'radius': max_reach}],
            'joints': [
                {'pos': tuple(solution.root_pos), 'type': 'root'},
                {'pos': tuple(solution.mid_pos), 'type': 'mid'},
                {'pos': tuple(solution.tip_pos), 'type': 'tip'},
            ]
        }
        set_ik_vis_data(vis_data)

        # Export log
        export_game_log("C:/Users/spenc/Desktop/engine_output_files/diagnostics_latest.txt")
        clear_log()

        # Report result based on validation
        if validation.valid:
            self.report({'INFO'}, f"IK VALID: error={validation.ik_error_cm:.1f}cm, score={validation.score:.2f}")
        else:
            self.report({'WARNING'}, f"IK INVALID: {validation.summary}")

        return {'FINISHED'}


# =============================================================================
# PLAY/STOP OPERATORS
# =============================================================================

class ANIM2_OT_TestPlay(Operator):
    """Play selected animation"""
    bl_idname = "anim2.test_play"
    bl_label = "Play"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        armature = getattr(context.scene, 'target_armature', None)
        if not armature or armature.type != 'ARMATURE':
            return False
        ctrl = get_test_controller()
        return ctrl.cache.count > 0

    def execute(self, context):
        scene = context.scene
        props = scene.anim2_test
        armature = scene.target_armature
        anim_name = props.selected_animation

        if not anim_name:
            self.report({'WARNING'}, "No animation selected")
            return {'CANCELLED'}

        ctrl = get_test_controller()
        if not ctrl.has_animation(anim_name):
            self.report({'WARNING'}, f"Animation '{anim_name}' not in cache")
            return {'CANCELLED'}

        # Play animation
        success = ctrl.play(
            armature.name,
            anim_name,
            weight=1.0,
            speed=props.play_speed,
            looping=props.loop_playback,
            fade_in=0.2,
            replace=True
        )

        if not success:
            self.report({'WARNING'}, f"Failed to play '{anim_name}'")
            return {'CANCELLED'}

        # Start modal
        if not is_test_modal_running():
            bpy.ops.anim2.test_modal('INVOKE_DEFAULT')

        self.report({'INFO'}, f"Playing '{anim_name}'")
        return {'FINISHED'}


class ANIM2_OT_TestStop(Operator):
    """Stop animation playback"""
    bl_idname = "anim2.test_stop"
    bl_label = "Stop"
    bl_options = {'REGISTER'}

    def execute(self, context):
        stop_test_modal()
        ctrl = get_test_controller()
        if ctrl:
            ctrl.clear_all()
        self.report({'INFO'}, "Stopped")
        return {'FINISHED'}


# =============================================================================
# REGISTRATION
# =============================================================================

def register():
    bpy.utils.register_class(ANIM2_OT_BakeAll)
    bpy.utils.register_class(ANIM2_OT_StopAll)
    bpy.utils.register_class(ANIM2_OT_ClearCache)
    bpy.utils.register_class(ANIM2_OT_TestModal)
    bpy.utils.register_class(ANIM2_TestProperties)
    bpy.utils.register_class(ANIM2_OT_TestPlay)
    bpy.utils.register_class(ANIM2_OT_TestStop)
    bpy.utils.register_class(ANIM2_OT_TestIKState)
    bpy.utils.register_class(ANIM2_OT_ApplyIK)

    bpy.types.Scene.anim2_test = bpy.props.PointerProperty(type=ANIM2_TestProperties)

    # IK Test properties
    bpy.types.Scene.ik_test_chain = bpy.props.EnumProperty(
        name="IK Chain",
        description="Which IK chain to analyze",
        items=get_ik_chain_items,
        default=0
    )
    bpy.types.Scene.ik_test_target = bpy.props.PointerProperty(
        name="IK Target",
        description="Object to use as IK target position",
        type=bpy.types.Object
    )


def unregister():
    if hasattr(bpy.types.Scene, 'ik_test_target'):
        del bpy.types.Scene.ik_test_target
    if hasattr(bpy.types.Scene, 'ik_test_chain'):
        del bpy.types.Scene.ik_test_chain
    del bpy.types.Scene.anim2_test

    bpy.utils.unregister_class(ANIM2_OT_ApplyIK)
    bpy.utils.unregister_class(ANIM2_OT_TestIKState)
    bpy.utils.unregister_class(ANIM2_OT_TestStop)
    bpy.utils.unregister_class(ANIM2_OT_TestPlay)
    bpy.utils.unregister_class(ANIM2_TestProperties)
    bpy.utils.unregister_class(ANIM2_OT_TestModal)
    bpy.utils.unregister_class(ANIM2_OT_ClearCache)
    bpy.utils.unregister_class(ANIM2_OT_StopAll)
    bpy.utils.unregister_class(ANIM2_OT_BakeAll)
