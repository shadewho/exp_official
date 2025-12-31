# Exp_Game/reactions/exp_ragdoll.py
"""
Ragdoll System - Main Thread (Rig Agnostic)

Captures per-bone static data at start, sends to worker each frame.
Worker handles bone physics, main thread applies results + handles position drop.

Works on ANY armature - no hardcoded bone names or assumptions.
"""
import bpy
import math
from mathutils import Vector, Euler, Matrix
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from ..props_and_utils.exp_time import get_game_time
from ..developer.dev_logger import log_game
from ..animations.layer_manager import get_layer_manager_for, AnimChannel


def _log(msg: str):
    """Log ragdoll messages."""
    log_game("RAGDOLL", msg)


# =============================================================================
# POSITION DROP CONSTANTS
# =============================================================================

DROP_GRAVITY = -20.0         # m/s^2 (faster drop for collapse feel)
DROP_DAMPING = 0.3           # Low bounce - collapse doesn't bounce much
FLOOR_OFFSET = 0.05          # Minimal offset
DROP_DURATION = 1.5          # Let it run longer for full collapse


# =============================================================================
# STATE
# =============================================================================

@dataclass
class RagdollInstance:
    """Active ragdoll state."""
    ragdoll_id: int
    armature: Any
    armature_name: str
    start_time: float
    duration: float

    # Per-bone static data (captured once at start)
    bone_data: Dict[str, dict] = field(default_factory=dict)

    # Per-bone dynamic state (updated each frame)
    bone_physics: Dict[str, dict] = field(default_factory=dict)

    # Initial rotations for blending back
    initial_rotations: Dict[str, tuple] = field(default_factory=dict)

    # Track initialization
    initialized: bool = False

    # Position drop physics (main thread handles this)
    initial_position: tuple = (0.0, 0.0, 0.0)
    drop_velocity: float = 0.0      # Z velocity for drop
    ground_z: float = 0.0           # Floor level
    drop_active: bool = True        # Is the drop phase active


# Module state
_active_ragdolls: Dict[int, RagdollInstance] = {}
_next_id: int = 0


# =============================================================================
# PUBLIC API
# =============================================================================

def has_active_ragdolls() -> bool:
    return len(_active_ragdolls) > 0


def init_ragdoll_system():
    global _active_ragdolls, _next_id
    _active_ragdolls.clear()
    _next_id = 0
    _log("System initialized")


def shutdown_ragdoll_system():
    global _active_ragdolls
    _active_ragdolls.clear()
    _log("System shutdown")


# =============================================================================
# EXECUTE REACTION
# =============================================================================

def execute_ragdoll_reaction(r):
    """Start ragdoll on armature - captures all bone data."""
    global _next_id

    scene = bpy.context.scene

    # Get target armature
    if getattr(r, "ragdoll_target_use_character", True):
        armature = scene.target_armature
    else:
        armature = getattr(r, "ragdoll_target_armature", None)

    if not armature or armature.type != 'ARMATURE':
        _log("ERROR: No valid armature")
        return

    armature_id = id(armature)

    if armature_id in _active_ragdolls:
        _log(f"Already ragdolling {armature.name}")
        return

    duration = getattr(r, "ragdoll_duration", 2.0)
    _log(f"Starting ragdoll on {armature.name}, duration={duration}s")

    # Capture initial position for drop
    initial_pos = tuple(armature.location)

    # Get ground Z - try KCC first, then scene floor
    ground_z = 0.0
    if hasattr(scene, 'kcc_grounded_z'):
        ground_z = scene.kcc_grounded_z
    elif hasattr(scene, 'floor_z'):
        ground_z = scene.floor_z

    # Capture per-bone static data - NO ROLE DETECTION, all bones equal
    bone_data = {}
    bone_physics = {}
    initial_rotations = {}

    for bone in armature.pose.bones:
        bone_name = bone.name

        rest_mat = bone.bone.matrix_local.to_3x3()
        rest_matrix_flat = [
            rest_mat[0][0], rest_mat[0][1], rest_mat[0][2],
            rest_mat[1][0], rest_mat[1][1], rest_mat[1][2],
            rest_mat[2][0], rest_mat[2][1], rest_mat[2][2],
        ]

        bone_data[bone_name] = {
            "rest_matrix": rest_matrix_flat,
        }

        bone_physics[bone_name] = {
            "rot": (0.0, 0.0, 0.0),
            "ang_vel": (0.0, 0.0, 0.0),
        }

        pb = armature.pose.bones.get(bone_name)
        if pb:
            if pb.rotation_mode == 'QUATERNION':
                rot = pb.rotation_quaternion.to_euler('XYZ')
            else:
                rot = pb.rotation_euler
            initial_rotations[bone_name] = (rot.x, rot.y, rot.z)

    _log(f"  Captured {len(bone_data)} bones, ground_z={ground_z:.2f}")

    instance = RagdollInstance(
        ragdoll_id=_next_id,
        armature=armature,
        armature_name=armature.name,
        start_time=get_game_time(),
        duration=duration,
        bone_data=bone_data,
        bone_physics=bone_physics,
        initial_rotations=initial_rotations,
        initialized=False,
        initial_position=initial_pos,
        drop_velocity=0.0,
        ground_z=ground_z + FLOOR_OFFSET,
        drop_active=True,
    )

    _active_ragdolls[armature_id] = instance
    _next_id += 1

    _lock_animations(armature.name, duration)

    layer_manager = get_layer_manager_for(armature)
    if layer_manager:
        layer_manager.activate_channel(
            channel=AnimChannel.PHYSICS,
            influence=1.0,
            fade_in=0.0,
        )
        _log("PHYSICS channel activated")

    _log(f"Ragdoll {instance.ragdoll_id} started")


# =============================================================================
# POSITION DROP (Main Thread - cheap physics)
# =============================================================================

def _update_position_drop(instance: RagdollInstance, dt: float):
    """
    Apply simple gravity drop to armature position.
    Main thread handles this - just gravity + floor collision.
    """
    if not instance.drop_active:
        return

    armature = instance.armature
    if not armature:
        return

    # Check if drop phase should end (time-based)
    elapsed = get_game_time() - instance.start_time
    if elapsed > DROP_DURATION:
        instance.drop_active = False
        _log(f"DROP ended (timeout) z={armature.location.z:.3f}")
        return

    try:
        current_z = armature.location.z
    except ReferenceError:
        return

    # Apply gravity
    instance.drop_velocity += DROP_GRAVITY * dt

    # Integrate position
    new_z = current_z + instance.drop_velocity * dt

    # Ground collision
    if new_z <= instance.ground_z:
        new_z = instance.ground_z
        instance.drop_velocity *= -DROP_DAMPING  # Bounce with damping
        _log(f"DROP hit_ground z={new_z:.3f} vel={instance.drop_velocity:.2f}")

        # Stop bouncing if velocity is small
        if abs(instance.drop_velocity) < 0.5:
            instance.drop_velocity = 0.0
            instance.drop_active = False
            _log(f"DROP stopped (settled)")

    # Apply to armature
    armature.location.z = new_z


# =============================================================================
# SUBMIT JOB
# =============================================================================

def submit_ragdoll_update(engine, dt: float):
    """Submit ragdoll job to worker with full bone data."""
    if not _active_ragdolls:
        return

    # First, update position drop for all ragdolls (main thread physics)
    for armature_id, inst in list(_active_ragdolls.items()):
        _update_position_drop(inst, dt)

    ragdolls_data = []

    for armature_id, inst in list(_active_ragdolls.items()):
        try:
            _ = inst.armature.name
        except (ReferenceError, AttributeError):
            _log(f"Ragdoll {inst.ragdoll_id} armature invalid, removing")
            del _active_ragdolls[armature_id]
            continue

        elapsed = get_game_time() - inst.start_time
        time_left = max(0, inst.duration - elapsed)

        world_mat = inst.armature.matrix_world
        armature_matrix = [
            world_mat[0][0], world_mat[0][1], world_mat[0][2], world_mat[0][3],
            world_mat[1][0], world_mat[1][1], world_mat[1][2], world_mat[1][3],
            world_mat[2][0], world_mat[2][1], world_mat[2][2], world_mat[2][3],
            world_mat[3][0], world_mat[3][1], world_mat[3][2], world_mat[3][3],
        ]

        ragdolls_data.append({
            "id": inst.ragdoll_id,
            "time_remaining": time_left,
            "bone_data": inst.bone_data,
            "bone_physics": inst.bone_physics,
            "armature_matrix": armature_matrix,
            "ground_z": inst.ground_z,
            "initialized": inst.initialized,
        })

    if not ragdolls_data:
        return

    job_data = {
        "dt": dt,
        "ragdolls": ragdolls_data,
    }

    job_id = engine.submit_job("RAGDOLL_UPDATE_BATCH", job_data)
    if job_id is not None and job_id >= 0:
        _log(f"Submitted job {job_id}: {len(ragdolls_data)} ragdolls")


# =============================================================================
# PROCESS RESULTS
# =============================================================================

def process_ragdoll_results(results: List[dict]):
    """Apply worker results to armatures."""
    if not results:
        return

    for result in results:
        ragdoll_id = result.get("id")
        bone_physics = result.get("bone_physics", {})
        finished = result.get("finished", False)
        initialized = result.get("initialized", False)

        instance = None
        armature_id = None
        for aid, inst in _active_ragdolls.items():
            if inst.ragdoll_id == ragdoll_id:
                instance = inst
                armature_id = aid
                break

        if not instance:
            continue

        instance.bone_physics = bone_physics
        instance.initialized = initialized

        _apply_ragdoll(instance)

        if finished:
            _log(f"Ragdoll {ragdoll_id} finished - restoring locomotion")
            _reset_pose(instance)
            _unlock_animations(instance.armature_name)

            layer_manager = get_layer_manager_for(instance.armature)
            if layer_manager:
                layer_manager.deactivate_channel(
                    channel=AnimChannel.PHYSICS,
                    fade_out=0.3,
                )
                _log("PHYSICS channel deactivated")

            del _active_ragdolls[armature_id]


def _apply_ragdoll(instance: RagdollInstance):
    """Apply bone rotations - combines initial pose with physics offset."""
    armature = instance.armature
    if not armature:
        return

    try:
        pose_bones = armature.pose.bones
    except ReferenceError:
        return

    for bone_name, physics in instance.bone_physics.items():
        pb = pose_bones.get(bone_name)
        if not pb:
            continue

        physics_rot = physics.get("rot", (0.0, 0.0, 0.0))
        initial_rot = instance.initial_rotations.get(bone_name, (0.0, 0.0, 0.0))

        final_rot = (
            initial_rot[0] + physics_rot[0],
            initial_rot[1] + physics_rot[1],
            initial_rot[2] + physics_rot[2],
        )

        pb.rotation_mode = 'XYZ'
        pb.rotation_euler = Euler(final_rot, 'XYZ')


def _reset_pose(instance: RagdollInstance):
    """Reset armature pose to initial state."""
    armature = instance.armature
    if not armature:
        return

    try:
        pose_bones = armature.pose.bones
    except ReferenceError:
        return

    for bone_name, initial_rot in instance.initial_rotations.items():
        pb = pose_bones.get(bone_name)
        if pb:
            pb.rotation_mode = 'XYZ'
            pb.rotation_euler = Euler(initial_rot, 'XYZ')


# =============================================================================
# ANIMATION LOCKING
# =============================================================================

def _lock_animations(armature_name: str, duration: float):
    try:
        from ..animations.blend_system import get_blend_system
        bs = get_blend_system()
        if bs:
            if hasattr(bs, 'start_ragdoll_lock'):
                bs.start_ragdoll_lock(armature_name, duration)
            elif hasattr(bs, 'lock_locomotion'):
                bs.lock_locomotion(duration)
    except ImportError:
        pass
    _log(f"Locked animations for {armature_name}")


def _unlock_animations(armature_name: str):
    try:
        from ..animations.blend_system import get_blend_system
        bs = get_blend_system()
        if bs and hasattr(bs, 'end_ragdoll_lock'):
            bs.end_ragdoll_lock(armature_name)
    except ImportError:
        pass
    _log(f"Unlocked animations for {armature_name}")


