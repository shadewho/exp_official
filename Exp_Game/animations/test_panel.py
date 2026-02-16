# Exp_Game/animations/test_panel.py
"""
Animation Test Panel - STRIPPED DOWN VERSION

Only contains:
- Animation playback testing
- Basic GPU visualization for targets
- Reset pose operator
"""

import bpy
import time
from bpy.types import Operator, PropertyGroup
from bpy.props import FloatProperty, BoolProperty, EnumProperty

from ..engine.animations.baker import bake_action
from ..engine import EngineCore
from .controller import AnimationController
from ..developer.dev_logger import start_session, log_game, export_game_log, clear_log
import gpu
from gpu_extras.batch import batch_for_shader

from ..developer.gpu_utils import (
    get_cached_shader,
    CIRCLE_8,
    sphere_wire_verts,
    extend_batch_data,
    crosshair_verts,
)


# =============================================================================
# GPU TARGET VISUALIZER (kept - useful for any system)
# =============================================================================

_target_handler = None
_target_data = None

TARGET_COLORS = {
    "L_HAND": (0.2, 0.6, 1.0, 0.95),
    "R_HAND": (1.0, 0.4, 0.2, 0.95),
    "L_FOOT": (0.2, 1.0, 0.4, 0.95),
    "R_FOOT": (1.0, 1.0, 0.2, 0.95),
    "LOOK_AT": (1.0, 0.2, 1.0, 0.95),
    "HIPS": (0.6, 0.2, 1.0, 0.95),
}


def _draw_targets():
    """GPU draw callback for target visualization."""
    global _target_data
    if _target_data is None:
        return

    gpu.state.depth_test_set('NONE')
    gpu.state.blend_set('ALPHA')
    gpu.state.line_width_set(3.0)

    shader = get_cached_shader()
    all_verts = []
    all_colors = []

    for target in _target_data:
        tag = target.get('tag', '')
        pos = target.get('pos')
        if not pos:
            continue

        color = TARGET_COLORS.get(tag, (1.0, 1.0, 1.0, 0.9))
        extend_batch_data(all_verts, all_colors, crosshair_verts(pos, 0.08), color)
        extend_batch_data(all_verts, all_colors, sphere_wire_verts(pos, 0.06, CIRCLE_8), color)

        origin = target.get('origin')
        if origin:
            all_verts.extend([origin, pos])
            all_colors.extend([(*color[:3], 0.4), color])

    if all_verts:
        batch = batch_for_shader(shader, 'LINES', {"pos": all_verts, "color": all_colors})
        shader.bind()
        batch.draw(shader)

    gpu.state.line_width_set(1.0)
    gpu.state.blend_set('NONE')


def enable_target_visualizer():
    global _target_handler
    if _target_handler is None:
        _target_handler = bpy.types.SpaceView3D.draw_handler_add(
            _draw_targets, (), 'WINDOW', 'POST_VIEW'
        )


def disable_target_visualizer():
    global _target_handler, _target_data
    if _target_handler is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_target_handler, 'WINDOW')
        except Exception:
            pass
        _target_handler = None
    _target_data = None


def set_target_data(targets: list):
    """Set target data for GPU visualization."""
    global _target_data
    _target_data = targets
    if _target_handler is None:
        enable_target_visualizer()


# =============================================================================
# TEST ENGINE & CONTROLLER
# =============================================================================

_test_engine = None
_test_controller = None


def get_test_engine() -> EngineCore:
    global _test_engine
    if _test_engine is None or not _test_engine.is_alive():
        _test_engine = EngineCore()
        _test_engine.start()
        _test_engine.wait_for_readiness(timeout=2.0)
        start_session()
    return _test_engine


def get_test_controller() -> AnimationController:
    global _test_controller
    if _test_controller is None:
        _test_controller = AnimationController()
    return _test_controller


def reset_test_controller():
    global _test_controller, _test_engine
    if _test_engine is not None:
        try:
            import os
            log_dir = os.path.join(os.path.expanduser("~"), "Desktop", "engine_output_files")
            os.makedirs(log_dir, exist_ok=True)
            export_game_log(os.path.join(log_dir, "diagnostics_latest.txt"))
        except Exception:
            pass
        clear_log()
        _test_engine.shutdown()
        _test_engine = None
    disable_target_visualizer()
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
        for action in bpy.data.actions:
            try:
                anim = bake_action(action)
                ctrl.add_animation(anim)
                baked_count += 1
            except Exception as e:
                pass

        if baked_count > 0 and engine.is_alive():
            cache_data = ctrl.get_cache_data_for_workers()
            engine.broadcast_job("CACHE_ANIMATIONS", cache_data)
            time.sleep(0.1)

        elapsed = (time.perf_counter() - start_time) * 1000
        self.report({'INFO'}, f"Baked {baked_count} actions in {elapsed:.0f}ms")
        return {'FINISHED'}


class ANIM2_OT_ResetPose(Operator):
    """Reset armature to rest pose"""
    bl_idname = "anim2.reset_pose"
    bl_label = "Reset Pose"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        armature = getattr(context.scene, 'target_armature', None)
        return armature and armature.type == 'ARMATURE'

    def execute(self, context):
        armature = context.scene.target_armature
        for pose_bone in armature.pose.bones:
            pose_bone.rotation_mode = 'QUATERNION'
            pose_bone.rotation_quaternion = (1, 0, 0, 0)
            pose_bone.location = (0, 0, 0)
            pose_bone.scale = (1, 1, 1)
        context.view_layer.update()
        self.report({'INFO'}, "Pose reset")
        return {'FINISHED'}


# =============================================================================
# TEST MODAL (ANIMATION PLAYBACK)
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

    def invoke(self, context, event):
        global _active_test_modal
        self._last_time = time.perf_counter()
        self._start_time = time.perf_counter()

        wm = context.window_manager
        self._timer = wm.event_timer_add(1/30, window=context.window)
        wm.modal_handler_add(self)
        _active_test_modal = self
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
            props = context.scene.anim2_test
            armature = getattr(context.scene, 'target_armature', None)
            if not armature:
                self.cancel(context)
                return {'CANCELLED'}

            timeout = getattr(props, 'playback_timeout', 20.0)
            if timeout > 0 and (time.perf_counter() - self._start_time) >= timeout:
                self.cancel(context)
                return {'CANCELLED'}

            current = time.perf_counter()
            dt = current - self._last_time
            self._last_time = current
            self._step_animation(context, dt, armature)
            return {'RUNNING_MODAL'}

        return {'PASS_THROUGH'}

    def _step_animation(self, context, dt: float, armature):
        ctrl = get_test_controller()
        engine = get_test_engine()
        if not ctrl or not engine or not engine.is_alive():
            return

        ctrl.update_state(dt)
        jobs_data = ctrl.get_compute_job_data()
        if not jobs_data:
            return

        job_id = engine.submit_job("ANIMATION_COMPUTE_BATCH", {"objects": jobs_data})
        if job_id is None or job_id < 0:
            return

        poll_start = time.perf_counter()
        while (time.perf_counter() - poll_start) < 0.005:
            results = list(engine.poll_results(max_results=10))
            for result in results:
                if result.job_type == "ANIMATION_COMPUTE_BATCH" and result.job_id == job_id:
                    if result.success:
                        self._apply_result(armature, result.result)
                    return
            time.sleep(0.0001)

    def _apply_result(self, armature, result_data: dict):
        results_dict = result_data.get("results", {})
        for object_name, obj_result in results_dict.items():
            if object_name != armature.name:
                continue
            bone_transforms = obj_result.get("bone_transforms", {})
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
        _active_test_modal = None


# =============================================================================
# PROPERTIES
# =============================================================================

def get_animation_items(self, context):
    ctrl = get_test_controller()
    items = [("", "Select Animation", "")]
    for name in ctrl.cache.names:
        items.append((name, name, ""))
    return items


class ANIM2_TestProperties(PropertyGroup):
    selected_animation: EnumProperty(
        name="Animation",
        items=get_animation_items
    )
    play_speed: FloatProperty(
        name="Speed",
        default=1.0,
        min=0.1,
        max=3.0
    )
    loop_playback: BoolProperty(
        name="Loop",
        default=True
    )
    playback_timeout: FloatProperty(
        name="Timeout",
        default=20.0,
        min=0.0,
        max=300.0
    )


# =============================================================================
# PLAY/STOP
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
        props = context.scene.anim2_test
        armature = context.scene.target_armature
        anim_name = props.selected_animation

        if not anim_name:
            self.report({'WARNING'}, "No animation selected")
            return {'CANCELLED'}

        ctrl = get_test_controller()
        ctrl.play(
            armature.name, anim_name,
            weight=1.0, speed=props.play_speed,
            looping=props.loop_playback, fade_in=0.2, replace=True
        )

        if not is_test_modal_running():
            bpy.ops.anim2.test_modal('INVOKE_DEFAULT')

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
        return {'FINISHED'}


class ANIM2_OT_ClearCache(Operator):
    """Clear animation cache"""
    bl_idname = "anim2.clear_cache"
    bl_label = "Clear Cache"
    bl_options = {'REGISTER'}

    def execute(self, context):
        ctrl = get_test_controller()
        if ctrl:
            ctrl.cache.clear()
        return {'FINISHED'}


# =============================================================================
# REGISTRATION
# =============================================================================

def register():
    bpy.utils.register_class(ANIM2_OT_BakeAll)
    bpy.utils.register_class(ANIM2_OT_ResetPose)
    bpy.utils.register_class(ANIM2_OT_TestModal)
    bpy.utils.register_class(ANIM2_TestProperties)
    bpy.utils.register_class(ANIM2_OT_TestPlay)
    bpy.utils.register_class(ANIM2_OT_TestStop)
    bpy.utils.register_class(ANIM2_OT_ClearCache)
    bpy.types.Scene.anim2_test = bpy.props.PointerProperty(type=ANIM2_TestProperties)


def unregister():
    del bpy.types.Scene.anim2_test
    bpy.utils.unregister_class(ANIM2_OT_ClearCache)
    bpy.utils.unregister_class(ANIM2_OT_TestStop)
    bpy.utils.unregister_class(ANIM2_OT_TestPlay)
    bpy.utils.unregister_class(ANIM2_TestProperties)
    bpy.utils.unregister_class(ANIM2_OT_TestModal)
    bpy.utils.unregister_class(ANIM2_OT_ResetPose)
    bpy.utils.unregister_class(ANIM2_OT_BakeAll)
