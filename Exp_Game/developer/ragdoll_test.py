# Exp_Game/developer/ragdoll_test.py
"""
Ragdoll Test - Minimal dev UI wrapper.

Just UI operators to test the real ragdoll system.
All physics logic is in engine/worker/reactions/ragdoll.py
"""

import bpy
import time
from bpy.types import Operator
from mathutils import Euler

from .dev_logger import start_session, export_game_log, clear_log, log_game
from ..engine.worker.reactions.ragdoll import handle_ragdoll_update_batch

LOG_OUTPUT_PATH = "C:/Users/spenc/Desktop/engine_output_files/diagnostics_latest.txt"


class RagdollTestState:
    def __init__(self):
        self.active = False
        self.armature = None
        self.start_time = 0.0
        self.duration = 3.0
        self.bone_data = {}
        self.bone_physics = {}
        self.initial_rotations = {}
        self.initial_z = 0.0

    def reset(self):
        self.__init__()


_state = RagdollTestState()


def is_ragdoll_test_active():
    return _state.active


def get_ragdoll_test_time_remaining():
    if not _state.active:
        return 0.0
    elapsed = time.time() - _state.start_time
    return max(0.0, _state.duration - elapsed)


def _capture_bone_data(armature):
    """Capture bone data in format expected by worker."""
    bone_data = {}
    bone_physics = {}
    initial_rotations = {}

    for bone in armature.pose.bones:
        name = bone.name
        rest_mat = bone.bone.matrix_local.to_3x3()
        bone_data[name] = {
            "rest_matrix": [
                rest_mat[0][0], rest_mat[0][1], rest_mat[0][2],
                rest_mat[1][0], rest_mat[1][1], rest_mat[1][2],
                rest_mat[2][0], rest_mat[2][1], rest_mat[2][2],
            ]
        }
        bone_physics[name] = {"rot": (0.0, 0.0, 0.0), "ang_vel": (0.0, 0.0, 0.0)}

        rot = bone.rotation_quaternion.to_euler('XYZ') if bone.rotation_mode == 'QUATERNION' else bone.rotation_euler
        initial_rotations[name] = (rot.x, rot.y, rot.z)

    return bone_data, bone_physics, initial_rotations


def _apply_physics(armature, bone_physics, initial_rotations):
    """Apply physics results to armature."""
    for name, phys in bone_physics.items():
        pb = armature.pose.bones.get(name)
        if not pb:
            continue
        init = initial_rotations.get(name, (0, 0, 0))
        rot = phys.get("rot", (0, 0, 0))
        pb.rotation_mode = 'XYZ'
        pb.rotation_euler = Euler((init[0] + rot[0], init[1] + rot[1], init[2] + rot[2]), 'XYZ')


class EXP_OT_ragdoll_test_modal(Operator):
    bl_idname = "exp.ragdoll_test_modal"
    bl_label = "Ragdoll Test Modal"
    bl_options = {'INTERNAL'}

    _timer = None
    _last_time = 0.0

    def modal(self, context, event):
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        if not _state.active or not _state.armature:
            self.cancel(context)
            return {'CANCELLED'}

        elapsed = time.time() - _state.start_time
        if elapsed >= _state.duration:
            log_game("RAGDOLL", f"Test completed ({elapsed:.1f}s)")
            export_game_log(LOG_OUTPUT_PATH)
            self._reset_armature()
            _state.reset()
            self.cancel(context)
            return {'CANCELLED'}

        dt = min(0.1, max(0.001, time.time() - self._last_time))
        self._last_time = time.time()

        # Build job data and call worker physics
        wm = _state.armature.matrix_world
        job_data = {
            "dt": dt,
            "ragdolls": [{
                "id": 0,
                "time_remaining": _state.duration - elapsed,
                "bone_data": _state.bone_data,
                "bone_physics": _state.bone_physics,
                "armature_matrix": [wm[i][j] for i in range(4) for j in range(4)],
                "initialized": True,
            }]
        }

        result = handle_ragdoll_update_batch(job_data, None, None, None)

        if result.get("success") and result.get("updated_ragdolls"):
            _state.bone_physics = result["updated_ragdolls"][0].get("bone_physics", {})
            _apply_physics(_state.armature, _state.bone_physics, _state.initial_rotations)

        context.area.tag_redraw()
        return {'PASS_THROUGH'}

    def _reset_armature(self):
        if _state.armature:
            try:
                for name, rot in _state.initial_rotations.items():
                    pb = _state.armature.pose.bones.get(name)
                    if pb:
                        pb.rotation_mode = 'XYZ'
                        pb.rotation_euler = Euler(rot, 'XYZ')
                _state.armature.location.z = _state.initial_z
            except ReferenceError:
                pass

    def execute(self, context):
        wm = context.window_manager
        self._timer = wm.event_timer_add(1/60, window=context.window)
        self._last_time = time.time()
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)


class EXP_OT_ragdoll_test_start(Operator):
    bl_idname = "exp.ragdoll_test_start"
    bl_label = "Start Ragdoll Test"

    @classmethod
    def poll(cls, context):
        arm = getattr(context.scene, 'target_armature', None)
        return arm and arm.type == 'ARMATURE' and not _state.active

    def execute(self, context):
        scene = context.scene
        arm = scene.target_armature

        start_session()
        clear_log()

        _state.reset()
        _state.armature = arm
        _state.duration = getattr(scene, 'dev_ragdoll_test_duration', 3.0)
        _state.start_time = time.time()
        _state.initial_z = arm.location.z
        _state.bone_data, _state.bone_physics, _state.initial_rotations = _capture_bone_data(arm)

        # Lift for drop test
        if getattr(scene, 'dev_ragdoll_test_include_drop', True) and arm.location.z < 1.5:
            arm.location.z = 2.0

        _state.active = True
        log_game("RAGDOLL", f"Test started: {arm.name}, {len(_state.bone_data)} bones")

        bpy.ops.exp.ragdoll_test_modal('INVOKE_DEFAULT')
        return {'FINISHED'}


class EXP_OT_ragdoll_test_stop(Operator):
    bl_idname = "exp.ragdoll_test_stop"
    bl_label = "Stop"

    @classmethod
    def poll(cls, context):
        return _state.active

    def execute(self, context):
        export_game_log(LOG_OUTPUT_PATH)
        _state.reset()
        return {'FINISHED'}


_classes = [EXP_OT_ragdoll_test_modal, EXP_OT_ragdoll_test_start, EXP_OT_ragdoll_test_stop]

def register():
    for cls in _classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
