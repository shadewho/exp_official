# Exp_Game/reactions/exp_ragdoll.py
"""
Verlet Particle Ragdoll - Main Thread

Converts armature bones to particles, sends to worker for physics,
applies results back as bone rotations.

Works on ANY armature - no hardcoded bone names.
"""
import bpy
import math
from mathutils import Vector, Matrix, Quaternion, Euler
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

from ..props_and_utils.exp_time import get_game_time
from ..developer.dev_logger import log_game
from ..animations.layer_manager import get_layer_manager_for, AnimChannel


def _log(msg: str):
    """Log ragdoll messages if debug enabled."""
    try:
        scene = bpy.context.scene
        if scene and getattr(scene, "dev_debug_ragdoll", False):
            log_game("RAGDOLL", msg)
    except:
        pass


def _log_always(msg: str):
    """Always log (for important events)."""
    log_game("RAGDOLL", msg)


# =============================================================================
# CONSTANTS
# =============================================================================

# How long the ragdoll continues after it settles
SETTLE_TIME = 0.5

# Root particle behavior
FIX_ROOT = False  # If True, root stays in place. If False, whole body falls.


# =============================================================================
# STATE
# =============================================================================

@dataclass
class VerletRagdoll:
    """Active ragdoll state using Verlet particles."""
    ragdoll_id: int
    armature: Any
    armature_name: str
    start_time: float
    duration: float

    # Particle system
    particles: List[Tuple[float, float, float]] = field(default_factory=list)
    prev_particles: List[Tuple[float, float, float]] = field(default_factory=list)
    constraints: List[Tuple[int, int, float]] = field(default_factory=list)  # (p1_idx, p2_idx, rest_length)
    fixed_mask: List[bool] = field(default_factory=list)

    # Maps bone_name -> (head_particle_idx, tail_particle_idx)
    bone_map: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    # Initial state for restoration
    initial_bone_rotations: Dict[str, Tuple[float, float, float, float]] = field(default_factory=dict)
    initial_location: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Floor level
    floor_z: float = 0.0

    # Armature's initial world matrix (for restoration)
    initial_matrix: List[float] = field(default_factory=list)


# Module state
_active_ragdolls: Dict[int, VerletRagdoll] = {}
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
    _log_always("Verlet ragdoll system initialized")


def shutdown_ragdoll_system():
    global _active_ragdolls
    _active_ragdolls.clear()
    _log_always("Ragdoll system shutdown")


# =============================================================================
# PARTICLE BUILDING
# =============================================================================

def _build_particles_from_armature(armature) -> Tuple[List, List, Dict, List]:
    """
    Build particle system from armature bones.

    Returns:
        particles: list of (x, y, z) world positions
        constraints: list of (p1_idx, p2_idx, rest_length)
        bone_map: dict of bone_name -> (head_idx, tail_idx)
        fixed_mask: list of bool (True = particle is fixed)
    """
    particles = []
    constraints = []
    bone_map = {}
    fixed_mask = []

    # Map position tuple -> particle index (for deduplication)
    pos_to_idx = {}

    world_matrix = armature.matrix_world

    def get_or_create_particle(pos_world: Vector, is_root: bool = False) -> int:
        """Get existing particle index or create new one."""
        # Round position to avoid floating point issues
        key = (round(pos_world.x, 4), round(pos_world.y, 4), round(pos_world.z, 4))

        if key in pos_to_idx:
            return pos_to_idx[key]

        idx = len(particles)
        particles.append(key)
        fixed_mask.append(FIX_ROOT and is_root)
        pos_to_idx[key] = idx
        return idx

    # Find root bones (no parent)
    root_bones = [b for b in armature.pose.bones if b.parent is None]

    # Process bones breadth-first
    bones_to_process = list(root_bones)
    processed = set()

    while bones_to_process:
        pb = bones_to_process.pop(0)
        if pb.name in processed:
            continue
        processed.add(pb.name)

        # Get world positions of bone head and tail
        head_world = world_matrix @ pb.head
        tail_world = world_matrix @ pb.tail

        # Determine if this is a root bone
        is_root = pb.parent is None

        # Get or create particles
        head_idx = get_or_create_particle(head_world, is_root=is_root)
        tail_idx = get_or_create_particle(tail_world, is_root=False)

        # Store bone mapping
        bone_map[pb.name] = (head_idx, tail_idx)

        # Create constraint (bone length)
        bone_length = (tail_world - head_world).length
        if bone_length > 0.001:  # Ignore zero-length bones
            constraints.append((head_idx, tail_idx, bone_length))

        # Queue children
        for child in pb.children:
            bones_to_process.append(child)

    return particles, constraints, bone_map, fixed_mask


def _capture_initial_state(armature, instance: VerletRagdoll):
    """Capture bone rotations for later restoration."""
    for pb in armature.pose.bones:
        if pb.rotation_mode == 'QUATERNION':
            q = pb.rotation_quaternion
            instance.initial_bone_rotations[pb.name] = (q.w, q.x, q.y, q.z)
        else:
            e = pb.rotation_euler
            # Store as quaternion for consistency
            q = e.to_quaternion()
            instance.initial_bone_rotations[pb.name] = (q.w, q.x, q.y, q.z)

    instance.initial_location = tuple(armature.location)

    # Store world matrix
    m = armature.matrix_world
    instance.initial_matrix = [
        m[0][0], m[0][1], m[0][2], m[0][3],
        m[1][0], m[1][1], m[1][2], m[1][3],
        m[2][0], m[2][1], m[2][2], m[2][3],
        m[3][0], m[3][1], m[3][2], m[3][3],
    ]


# =============================================================================
# EXECUTE REACTION
# =============================================================================

def execute_ragdoll_reaction(r):
    """Start Verlet ragdoll on armature."""
    global _next_id

    scene = bpy.context.scene

    # Get target armature
    if getattr(r, "ragdoll_target_use_character", True):
        armature = scene.target_armature
    else:
        armature = getattr(r, "ragdoll_target_armature", None)

    if not armature or armature.type != 'ARMATURE':
        _log_always("ERROR: No valid armature for ragdoll")
        return

    armature_id = id(armature)

    if armature_id in _active_ragdolls:
        _log(f"Already ragdolling {armature.name}")
        return

    duration = getattr(r, "ragdoll_duration", 3.0)
    _log_always(f"START Verlet ragdoll on {armature.name}, duration={duration}s")

    # Get floor Z
    floor_z = 0.0
    if hasattr(scene, 'kcc_grounded_z'):
        floor_z = scene.kcc_grounded_z
    elif hasattr(scene, 'floor_z'):
        floor_z = scene.floor_z

    # Build particle system
    particles, constraints, bone_map, fixed_mask = _build_particles_from_armature(armature)

    _log(f"Built {len(particles)} particles, {len(constraints)} constraints from {len(bone_map)} bones")

    # Create instance
    instance = VerletRagdoll(
        ragdoll_id=_next_id,
        armature=armature,
        armature_name=armature.name,
        start_time=get_game_time(),
        duration=duration,
        particles=list(particles),
        prev_particles=list(particles),  # Start with no velocity
        constraints=constraints,
        fixed_mask=fixed_mask,
        bone_map=bone_map,
        floor_z=floor_z,
    )

    # Capture initial state
    _capture_initial_state(armature, instance)

    _active_ragdolls[armature_id] = instance
    _next_id += 1

    # Lock animations
    _lock_animations(armature.name, duration)

    # Activate PHYSICS channel
    layer_manager = get_layer_manager_for(armature)
    if layer_manager:
        layer_manager.activate_channel(
            channel=AnimChannel.PHYSICS,
            influence=1.0,
            fade_in=0.0,
        )
        _log("PHYSICS channel activated")

    _log_always(f"Ragdoll {instance.ragdoll_id} started with {len(particles)} particles")


# =============================================================================
# SUBMIT JOB
# =============================================================================

def submit_ragdoll_update(engine, dt: float):
    """Submit ragdoll jobs to worker."""
    if not _active_ragdolls:
        return

    ragdolls_data = []

    for armature_id, inst in list(_active_ragdolls.items()):
        # Verify armature still exists
        try:
            _ = inst.armature.name
        except (ReferenceError, AttributeError):
            _log(f"Ragdoll {inst.ragdoll_id} armature invalid, removing")
            del _active_ragdolls[armature_id]
            continue

        elapsed = get_game_time() - inst.start_time
        time_left = max(0, inst.duration - elapsed)

        ragdolls_data.append({
            "id": inst.ragdoll_id,
            "time_remaining": time_left,
            "particles": inst.particles,
            "prev_particles": inst.prev_particles,
            "constraints": inst.constraints,
            "fixed_mask": inst.fixed_mask,
            "bone_map": inst.bone_map,
            "floor_z": inst.floor_z,
        })

    if not ragdolls_data:
        return

    job_data = {
        "dt": dt,
        "ragdolls": ragdolls_data,
    }

    job_id = engine.submit_job("RAGDOLL_UPDATE_BATCH", job_data)
    if job_id is not None and job_id >= 0:
        _log(f"Submitted job {job_id}: {len(ragdolls_data)} ragdolls, {sum(len(r['particles']) for r in ragdolls_data)} particles")


# =============================================================================
# PROCESS RESULTS
# =============================================================================

def process_ragdoll_results(results: List[dict]):
    """Apply worker results to armatures."""
    _log_always(f"RESULTS received: {len(results)} ragdolls")

    if not results:
        return

    for result in results:
        ragdoll_id = result.get("id")
        new_particles = result.get("particles", [])
        new_prev_particles = result.get("prev_particles", [])
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
            continue

        # Update particle state
        instance.particles = [tuple(p) for p in new_particles]
        instance.prev_particles = [tuple(p) for p in new_prev_particles]

        # Apply to armature
        _apply_particles_to_armature(instance)

        if finished:
            _log_always(f"Ragdoll {ragdoll_id} finished - restoring pose")
            _restore_pose(instance)
            _unlock_animations(instance.armature_name)

            layer_manager = get_layer_manager_for(instance.armature)
            if layer_manager:
                layer_manager.deactivate_channel(
                    channel=AnimChannel.PHYSICS,
                    fade_out=0.3,
                )
                _log("PHYSICS channel deactivated")

            del _active_ragdolls[armature_id]


# =============================================================================
# APPLY PARTICLES TO ARMATURE
# =============================================================================

def _apply_particles_to_armature(instance: VerletRagdoll):
    """
    Convert particle positions back to bone rotations.

    Strategy:
    1. Find the "root" particle (usually hips) and move armature to match
    2. For each bone, calculate rotation to align it with its particles
    """
    armature = instance.armature
    if not armature:
        return

    try:
        pose_bones = armature.pose.bones
    except ReferenceError:
        return

    particles = instance.particles
    bone_map = instance.bone_map

    if not particles or not bone_map:
        return

    # Find the center of mass of all particles to position armature
    if particles:
        avg_x = sum(p[0] for p in particles) / len(particles)
        avg_y = sum(p[1] for p in particles) / len(particles)
        min_z = min(p[2] for p in particles)

        # Move armature so its origin follows the ragdoll
        # Use the lowest point as the ground reference
        armature.location.x = avg_x
        armature.location.y = avg_y
        armature.location.z = min_z

    # Get current armature world matrix for transformations
    armature.matrix_world = armature.matrix_world  # Force update
    world_inv = armature.matrix_world.inverted()

    # Process bones in hierarchy order (parents first)
    processed = set()

    def process_bone(pb):
        if pb.name in processed:
            return
        if pb.parent and pb.parent.name not in processed:
            process_bone(pb.parent)

        processed.add(pb.name)

        if pb.name not in bone_map:
            return

        head_idx, tail_idx = bone_map[pb.name]
        if head_idx >= len(particles) or tail_idx >= len(particles):
            return

        # Get particle positions in world space
        head_world = Vector(particles[head_idx])
        tail_world = Vector(particles[tail_idx])

        # Target direction in world space
        target_dir_world = (tail_world - head_world).normalized()
        if target_dir_world.length < 0.001:
            return

        # Get bone's rest direction in armature space
        # bone.vector is the bone's direction in armature local space
        rest_dir_local = pb.bone.vector.normalized()

        # Transform rest direction to world space using armature's current matrix
        rest_dir_world = (armature.matrix_world.to_3x3() @ rest_dir_local).normalized()

        # Calculate rotation from rest to target
        rotation = rest_dir_world.rotation_difference(target_dir_world)

        # Convert world rotation to bone local rotation
        # Need to account for parent's accumulated rotation
        if pb.parent:
            # Get parent's world rotation
            parent_world_rot = (armature.matrix_world @ pb.parent.matrix).to_quaternion()
            # Convert our world rotation to be relative to parent
            local_rot = parent_world_rot.inverted() @ rotation @ parent_world_rot
        else:
            # Root bone - rotation is relative to armature
            arm_rot = armature.matrix_world.to_quaternion()
            local_rot = arm_rot.inverted() @ rotation @ arm_rot

        # Apply to pose bone
        pb.rotation_mode = 'QUATERNION'
        pb.rotation_quaternion = local_rot

    # Process all bones
    for pb in pose_bones:
        process_bone(pb)

    _log(f"Applied particles to {len(processed)} bones")


def _restore_pose(instance: VerletRagdoll):
    """Restore armature to initial pose."""
    armature = instance.armature
    if not armature:
        return

    try:
        pose_bones = armature.pose.bones
    except ReferenceError:
        return

    # Restore bone rotations
    for bone_name, quat in instance.initial_bone_rotations.items():
        pb = pose_bones.get(bone_name)
        if pb:
            pb.rotation_mode = 'QUATERNION'
            pb.rotation_quaternion = Quaternion((quat[0], quat[1], quat[2], quat[3]))

    # Restore location
    armature.location = Vector(instance.initial_location)

    _log(f"Restored pose for {armature.name}")


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
