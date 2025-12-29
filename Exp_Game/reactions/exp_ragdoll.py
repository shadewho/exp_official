# Exp_Game/reactions/exp_ragdoll.py
"""
SIMPLE Ragdoll System - Main Thread Side

Main thread responsibilities (MINIMAL):
1. Start ragdoll: capture bone positions, create instance
2. Each frame: send bone states to worker, receive new positions
3. Apply: set bone positions/rotations from worker results

ALL physics happens in worker. Main thread just marshals data.
"""
import bpy
from mathutils import Vector, Matrix, Quaternion
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from ..props_and_utils.exp_time import get_game_time
from ..developer.dev_logger import log_game


def _log(msg: str):
    """Log if debug enabled."""
    try:
        scene = bpy.context.scene
        if scene and getattr(scene, "dev_debug_ragdoll", False):
            log_game("RAGDOLL", msg)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────

@dataclass
class RagdollInstance:
    """Active ragdoll."""
    ragdoll_id: int
    armature: Any
    armature_name: str
    start_time: float
    duration: float
    gravity_multiplier: float

    # Bone hierarchy (computed once at start)
    bone_order: List[str] = field(default_factory=list)
    bone_parents: Dict[str, str] = field(default_factory=dict)
    bone_lengths: Dict[str, float] = field(default_factory=dict)

    # Per-frame state (positions and velocities)
    bone_states: Dict[str, dict] = field(default_factory=dict)

    # Impulse (first frame only)
    impulse: Optional[tuple] = None
    impulse_bone: Optional[str] = None
    impulse_applied: bool = False


# Module state
_active_ragdolls: Dict[int, RagdollInstance] = {}
_next_id: int = 0


# ─────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────
# EXECUTE REACTION (Start ragdoll)
# ─────────────────────────────────────────────────────────

def execute_ragdoll_reaction(r):
    """Start ragdoll on armature."""
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

    # Already ragdolling?
    if armature_id in _active_ragdolls:
        _log(f"Already ragdolling {armature.name}")
        return

    # Get properties
    duration = getattr(r, "ragdoll_duration", 2.0)
    gravity_mult = getattr(r, "ragdoll_gravity_multiplier", 1.0)
    impulse_strength = getattr(r, "ragdoll_impulse_strength", 5.0)
    impulse_dir = tuple(getattr(r, "ragdoll_impulse_direction", (0, 0, 1)))

    # Normalize impulse
    imp_vec = Vector(impulse_dir).normalized() * impulse_strength
    impulse = (imp_vec.x, imp_vec.y, imp_vec.z)

    _log(f"Starting ragdoll on {armature.name}")
    _log(f"  Duration: {duration}s, Gravity mult: {gravity_mult}")
    _log(f"  Impulse: {impulse}")

    # Extract bone hierarchy
    bone_order, bone_parents, bone_lengths = _extract_bones(armature)
    _log(f"  Bones: {len(bone_order)} ({bone_order[:3]}...)")

    # Capture current pose as initial state
    bone_states = _capture_pose(armature, bone_order)
    _log(f"  Captured {len(bone_states)} bone positions")

    # Create instance
    instance = RagdollInstance(
        ragdoll_id=_next_id,
        armature=armature,
        armature_name=armature.name,
        start_time=get_game_time(),
        duration=duration,
        gravity_multiplier=gravity_mult,
        bone_order=bone_order,
        bone_parents=bone_parents,
        bone_lengths=bone_lengths,
        bone_states=bone_states,
        impulse=impulse,
        impulse_bone=bone_order[0] if bone_order else None,  # Apply to root
    )

    _active_ragdolls[armature_id] = instance
    _next_id += 1

    # Lock animations
    _lock_animations(armature.name, duration)

    _log(f"Ragdoll {instance.ragdoll_id} started")


def _extract_bones(armature) -> tuple:
    """Extract bone hierarchy in parent-first order."""
    bone_order = []
    bone_parents = {}
    bone_lengths = {}

    arm_data = armature.data

    # BFS from roots
    roots = [b for b in arm_data.bones if b.parent is None]
    queue = list(roots)

    while queue:
        bone = queue.pop(0)
        bone_order.append(bone.name)
        bone_parents[bone.name] = bone.parent.name if bone.parent else None
        bone_lengths[bone.name] = bone.length

        for child in bone.children:
            queue.append(child)

    return bone_order, bone_parents, bone_lengths


def _capture_pose(armature, bone_order: List[str]) -> Dict[str, dict]:
    """Initialize bone states for ragdoll."""
    states = {}

    for i, bone_name in enumerate(bone_order):
        pb = armature.pose.bones.get(bone_name)
        if not pb:
            continue

        is_root = (i == 0)

        if is_root:
            # Root: track position offset (starts at 0)
            states[bone_name] = {
                "pos": (0.0, 0.0, 0.0),  # position OFFSET, not world pos
                "vel": (0.0, 0.0, 0.0),
                "rot": (0.0, 0.0, 0.0),
                "rot_vel": (0.0, 0.0, 0.0),
            }
        else:
            # Non-root: track rotation offset
            states[bone_name] = {
                "pos": (0.0, 0.0, 0.0),
                "vel": (0.0, 0.0, 0.0),
                "rot": (0.0, 0.0, 0.0),
                "rot_vel": (0.0, 0.0, 0.0),
            }

    return states


# ─────────────────────────────────────────────────────────
# SUBMIT JOB (Each frame)
# ─────────────────────────────────────────────────────────

def submit_ragdoll_update(engine, dt: float):
    """Submit ragdoll job to worker."""
    if not _active_ragdolls:
        return

    scene = bpy.context.scene
    base_gravity = -21.0
    if hasattr(scene, "char_physics"):
        base_gravity = getattr(scene.char_physics, "gravity", -21.0)

    ragdolls_data = []

    for armature_id, inst in list(_active_ragdolls.items()):
        # Check armature still valid
        try:
            _ = inst.armature.name
        except (ReferenceError, AttributeError):
            _log(f"Ragdoll {inst.ragdoll_id} armature invalid, removing")
            del _active_ragdolls[armature_id]
            continue

        # Time remaining
        elapsed = get_game_time() - inst.start_time
        time_left = max(0, inst.duration - elapsed)

        # Apply gravity multiplier per-ragdoll
        ragdoll_gravity = (0.0, 0.0, base_gravity * inst.gravity_multiplier)

        # Build data for worker
        data = {
            "id": inst.ragdoll_id,
            "time_remaining": time_left,
            "bone_order": inst.bone_order,
            "bone_states": inst.bone_states,
            "gravity": ragdoll_gravity,
        }

        # Impulse on first frame
        if not inst.impulse_applied and inst.impulse:
            data["impulse"] = inst.impulse
            data["impulse_bone"] = inst.impulse_bone
            inst.impulse_applied = True

        ragdolls_data.append(data)

    if not ragdolls_data:
        return

    job_data = {
        "dt": dt,
        "ragdolls": ragdolls_data,
    }

    job_id = engine.submit_job("RAGDOLL_UPDATE_BATCH", job_data)
    if job_id is not None and job_id >= 0:
        _log(f"Submitted job {job_id}: {len(ragdolls_data)} ragdolls, dt={dt:.4f}")


# ─────────────────────────────────────────────────────────
# PROCESS RESULTS (From worker)
# ─────────────────────────────────────────────────────────

def process_ragdoll_results(results: List[dict]):
    """Apply worker results to armatures."""
    if not results:
        return

    _log(f"Processing {len(results)} results")

    for result in results:
        ragdoll_id = result.get("id")
        bone_rotations = result.get("bone_rotations", {})
        bone_states = result.get("bone_states", {})
        finished = result.get("finished", False)

        # Find instance
        instance = None
        armature_id = None
        for aid, inst in _active_ragdolls.items():
            if inst.ragdoll_id == ragdoll_id:
                instance = inst
                armature_id = aid
                break

        if not instance:
            _log(f"  No instance for ragdoll {ragdoll_id}")
            continue

        _log(f"  Ragdoll {ragdoll_id}: {len(bone_rotations)} bones, finished={finished}")

        # Update state for next frame
        instance.bone_states = bone_states

        # Apply to armature
        _apply_ragdoll(instance, bone_rotations)

        # Handle finish
        if finished:
            _log(f"  Ragdoll {ragdoll_id} finished")
            _reset_pose(instance)
            _unlock_animations(instance.armature_name)
            del _active_ragdolls[armature_id]


def _apply_ragdoll(instance: RagdollInstance, bone_rotations: dict):
    """
    Apply ragdoll transforms to armature.

    Root bone: position offset (drops/moves the character)
    Other bones: rotation offset (makes them droop/sway)
    """
    armature = instance.armature
    if not armature:
        return

    try:
        _ = armature.name
        pose_bones = armature.pose.bones
    except ReferenceError:
        _log("Armature became invalid, skipping")
        return

    root_name = instance.bone_order[0] if instance.bone_order else None

    for i, bone_name in enumerate(instance.bone_order):
        if bone_name not in bone_rotations:
            continue

        pb = pose_bones.get(bone_name)
        if not pb:
            continue

        is_root = (bone_name == root_name)

        if is_root:
            # Root: bone_rotations contains position OFFSET
            pos_offset = bone_rotations[bone_name]
            pb.location = Vector(pos_offset)
        else:
            # Non-root: bone_rotations contains euler rotation offset
            rot = bone_rotations[bone_name]
            from mathutils import Euler
            euler = Euler((rot[0], rot[1], rot[2]), 'XYZ')
            pb.rotation_mode = 'QUATERNION'
            pb.rotation_quaternion = euler.to_quaternion()


def _reset_pose(instance: RagdollInstance):
    """Reset armature to rest pose when ragdoll finishes."""
    armature = instance.armature
    if not armature:
        return

    try:
        pose_bones = armature.pose.bones
    except ReferenceError:
        return

    for bone_name in instance.bone_order:
        pb = pose_bones.get(bone_name)
        if pb:
            pb.location = Vector((0, 0, 0))
            pb.rotation_quaternion = Quaternion()


# ─────────────────────────────────────────────────────────
# ANIMATION LOCKING
# ─────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────
# CLEANUP
# ─────────────────────────────────────────────────────────

def clear():
    global _active_ragdolls
    _active_ragdolls.clear()
