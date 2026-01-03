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

    # Bone rest data for worker-side rotation calculation
    # Contains: rest_dirs, hierarchy, parents
    bone_rest_data: Dict[str, Any] = field(default_factory=dict)

    # Initial state for restoration
    initial_bone_rotations: Dict[str, Tuple[float, float, float, float]] = field(default_factory=dict)
    initial_location: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Floor level
    floor_z: float = 0.0

    # Armature's initial world matrix (for restoration)
    initial_matrix: List[float] = field(default_factory=list)

    # Sequence tracking to handle out-of-order results
    submit_seq: int = 0      # Increments each submit
    applied_seq: int = -1    # Tracks newest applied result


# Module state
_active_ragdolls: Dict[int, VerletRagdoll] = {}
_next_id: int = 0


# =============================================================================
# PUBLIC API
# =============================================================================

def has_active_ragdolls() -> bool:
    return len(_active_ragdolls) > 0


def is_armature_ragdolling(armature) -> bool:
    """Check if a specific armature is currently ragdolling."""
    if not armature:
        return False
    armature_id = id(armature)
    return armature_id in _active_ragdolls


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

    # Log armature state
    loc = armature.location
    _log(f"BUILD armature={armature.name} loc=({loc.x:.2f},{loc.y:.2f},{loc.z:.2f})")
    _log(f"BUILD world_matrix row0=({world_matrix[0][0]:.2f},{world_matrix[0][1]:.2f},{world_matrix[0][2]:.2f},{world_matrix[0][3]:.2f})")

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
    _log(f"BUILD root_bones={[b.name for b in root_bones]}")

    # Process bones breadth-first
    bones_to_process = list(root_bones)
    processed = set()

    # Track bone hierarchy for logging
    bone_details = []

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

        # Track for logging (first 5 bones only to avoid spam)
        if len(bone_details) < 5:
            parent_name = pb.parent.name if pb.parent else "None"
            bone_details.append(f"{pb.name}(p{head_idx}->p{tail_idx} len={bone_length:.3f} parent={parent_name})")

        # Queue children
        for child in pb.children:
            bones_to_process.append(child)

    # Log bone details
    for detail in bone_details:
        _log(f"BUILD_BONE {detail}")

    # Log particle position ranges
    if particles:
        xs = [p[0] for p in particles]
        ys = [p[1] for p in particles]
        zs = [p[2] for p in particles]
        _log(f"BUILD_RANGE x=[{min(xs):.2f},{max(xs):.2f}] y=[{min(ys):.2f},{max(ys):.2f}] z=[{min(zs):.2f},{max(zs):.2f}]")

    return particles, constraints, bone_map, fixed_mask


def _build_bone_rest_data(armature, bone_map: Dict) -> Dict:
    """
    Build bone rest data for worker-side rotation calculation.

    This is called once at ragdoll start and cached.
    The worker uses this to compute bone rotations without bpy.

    Returns:
        Dict with:
        - rest_dirs: {bone_name: (x, y, z)} normalized rest direction in armature space
        - hierarchy: [bone_names] in hierarchy order (parents first)
        - parents: {bone_name: parent_name or None}
    """
    rest_dirs = {}
    hierarchy = []
    parents = {}

    # Process bones in hierarchy order (parents before children)
    processed = set()
    pose_bones = armature.pose.bones

    def process_bone(pb):
        if pb.name in processed:
            return
        # Process parent first
        if pb.parent and pb.parent.name not in processed:
            process_bone(pb.parent)

        processed.add(pb.name)

        # Only include bones that are in the bone_map (have particles)
        if pb.name not in bone_map:
            return

        hierarchy.append(pb.name)

        # Store parent name
        parents[pb.name] = pb.parent.name if pb.parent else None

        # Store rest direction in armature local space
        # pb.bone.vector is the bone's direction in armature space
        rest_vec = pb.bone.vector.normalized()
        rest_dirs[pb.name] = (rest_vec.x, rest_vec.y, rest_vec.z)

    # Process all bones
    for pb in pose_bones:
        process_bone(pb)

    return {
        "rest_dirs": rest_dirs,
        "hierarchy": hierarchy,
        "parents": parents,
    }


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

    # Get floor Z - log where it comes from
    floor_z = 0.0
    floor_source = "default"
    if hasattr(scene, 'kcc_grounded_z'):
        floor_z = scene.kcc_grounded_z
        floor_source = "kcc_grounded_z"
    elif hasattr(scene, 'floor_z'):
        floor_z = scene.floor_z
        floor_source = "floor_z"

    _log(f"ENV floor_z={floor_z:.3f} (source={floor_source})")
    _log(f"ENV FIX_ROOT={FIX_ROOT}")

    # Build particle system
    particles, constraints, bone_map, fixed_mask = _build_particles_from_armature(armature)

    _log(f"Built {len(particles)} particles, {len(constraints)} constraints from {len(bone_map)} bones")

    # Build bone rest data for worker-side rotation calculation
    bone_rest_data = _build_bone_rest_data(armature, bone_map)
    _log(f"Built bone_rest_data: {len(bone_rest_data.get('hierarchy', []))} bones in hierarchy")

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
        bone_rest_data=bone_rest_data,
        floor_z=floor_z,
    )

    # Capture initial state
    _capture_initial_state(armature, instance)

    _active_ragdolls[armature_id] = instance
    _next_id += 1

    # Lock animations
    _lock_animations(armature.name, duration)

    # Activate PHYSICS channel and disable native action
    layer_manager = get_layer_manager_for(armature)
    if layer_manager:
        layer_manager.activate_channel(
            channel=AnimChannel.PHYSICS,
            influence=1.0,
            fade_in=0.0,
        )
        # CRITICAL: Disable Blender's native action to prevent depsgraph
        # from overwriting our manually set transforms
        layer_manager.disable_native_action(armature)
        _log("PHYSICS channel activated, native action disabled")

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

        # Increment sequence number for this submission
        inst.submit_seq += 1

        # Debug: log what particles we're submitting
        if inst.particles:
            zs = [p[2] for p in inst.particles]
            _log(f"SUBMIT id={inst.ragdoll_id} seq={inst.submit_seq} z_range=[{min(zs):.3f},{max(zs):.3f}] time_left={time_left:.2f}")

        # Get current armature world matrix for bone rotation calculation
        # Flatten COLUMN-MAJOR to match worker math (matrix_to_quat expects column-major)
        m = inst.armature.matrix_world
        armature_matrix = [
            m[0][0], m[1][0], m[2][0], m[3][0],
            m[0][1], m[1][1], m[2][1], m[3][1],
            m[0][2], m[1][2], m[2][2], m[3][2],
            m[0][3], m[1][3], m[2][3], m[3][3],
        ]

        ragdolls_data.append({
            "id": inst.ragdoll_id,
            "seq": inst.submit_seq,  # Include sequence for ordering
            "time_remaining": time_left,
            "particles": inst.particles,
            "prev_particles": inst.prev_particles,
            "constraints": inst.constraints,
            "fixed_mask": inst.fixed_mask,
            "bone_map": inst.bone_map,
            "bone_rest_data": inst.bone_rest_data,  # For worker-side bone calculation
            "armature_matrix": armature_matrix,     # Current transform
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
    _log(f"RESULTS received: {len(results)} ragdolls")

    if not results:
        return

    for result in results:
        ragdoll_id = result.get("id")
        seq = result.get("seq", 0)
        new_particles = result.get("particles", [])
        new_prev_particles = result.get("prev_particles", [])
        bone_rotations = result.get("bone_rotations", {})  # Pre-computed by worker!
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
            _log(f"RESULT_SKIP id={ragdoll_id} - instance not found")
            continue

        # Check sequence - skip stale (out-of-order) results
        if seq <= instance.applied_seq:
            _log(f"RESULT_STALE id={ragdoll_id} seq={seq} applied_seq={instance.applied_seq} - skipping")
            continue

        # Debug: log what we received vs what we had
        if new_particles:
            old_zs = [p[2] for p in instance.particles] if instance.particles else [0]
            new_zs = [p[2] for p in new_particles]
            _log(f"RESULT_APPLY id={ragdoll_id} seq={seq} old_z=[{min(old_zs):.3f},{max(old_zs):.3f}] new_z=[{min(new_zs):.3f},{max(new_zs):.3f}] bones={len(bone_rotations)} finished={finished}")

        # Update sequence tracker
        instance.applied_seq = seq

        # Update particle state
        instance.particles = [tuple(p) for p in new_particles]
        instance.prev_particles = [tuple(p) for p in new_prev_particles]

        # Apply to armature (now just applies pre-computed quaternions - very fast!)
        _apply_ragdoll_to_armature(instance, bone_rotations)

        if finished:
            _log_always(f"Ragdoll {ragdoll_id} finished - restoring pose")
            _restore_pose(instance)
            _unlock_animations(instance.armature_name)

            layer_manager = get_layer_manager_for(instance.armature)
            if layer_manager:
                # Restore native action before deactivating physics
                layer_manager.restore_native_action(instance.armature)
                layer_manager.deactivate_channel(
                    channel=AnimChannel.PHYSICS,
                    fade_out=0.3,
                )
                _log("PHYSICS channel deactivated, native action restored")

            del _active_ragdolls[armature_id]


def handle_ragdoll_job_failure(error: str = "worker failure"):
    """
    Called when a ragdoll worker job fails/times out.
    Restores all active ragdolls so locks/channels are released.
    """
    for armature_id, inst in list(_active_ragdolls.items()):
        _force_finish_instance(inst, reason=error)
        del _active_ragdolls[armature_id]


# =============================================================================
# APPLY RAGDOLL TO ARMATURE (optimized - worker computes rotations!)
# =============================================================================

def _apply_ragdoll_to_armature(instance: VerletRagdoll, bone_rotations: Dict):
    """
    Apply pre-computed ragdoll state to armature.

    OPTIMIZED: Bone rotation calculation moved to worker!
    Main thread now just:
    1. Updates armature position from particles
    2. Applies pre-computed quaternions directly

    Args:
        instance: VerletRagdoll with particle state
        bone_rotations: Dict of {bone_name: (w, x, y, z)} from worker
    """
    armature = instance.armature
    if not armature:
        return

    try:
        pose_bones = armature.pose.bones
    except ReferenceError:
        return

    particles = instance.particles
    if not particles:
        return

    # === STEP 1: Update armature position from particles ===
    # Use a stable anchor (hips/root head if available) to avoid sliding/teleport.
    anchor_idx = None
    for candidate in ("Hips", "hips", "Root", "root", "ROOT"):
        if candidate in instance.bone_map:
            anchor_idx = instance.bone_map[candidate][0]
            break
    if anchor_idx is None:
        # fallback: first root bone that exists in the map
        for pb in armature.pose.bones:
            if pb.parent is None and pb.name in instance.bone_map:
                anchor_idx = instance.bone_map[pb.name][0]
                break
    if anchor_idx is None:
        anchor_idx = 0  # final fallback

    anchor_idx = max(0, min(anchor_idx, len(particles) - 1))
    anchor = particles[anchor_idx]
    armature.location.x = anchor[0]
    armature.location.y = anchor[1]
    armature.location.z = anchor[2]

    # === STEP 2: Apply pre-computed bone rotations ===
    # This is FAST - just direct quaternion assignment!
    bones_applied = 0
    for bone_name, quat in bone_rotations.items():
        pb = pose_bones.get(bone_name)
        if pb:
            pb.rotation_mode = 'QUATERNION'
            pb.rotation_quaternion = Quaternion((quat[0], quat[1], quat[2], quat[3]))
            bones_applied += 1

    _log(f"APPLY_FAST bones={bones_applied} particles={len(particles)}")


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


def _force_finish_instance(instance: VerletRagdoll, reason: str = ""):
    """
    Fallback when worker fails/timeouts: restore pose and unlock channels.
    """
    _log_always(f"FORCE_FINISH ragdoll {instance.ragdoll_id} ({reason})")
    _restore_pose(instance)
    _unlock_animations(instance.armature_name)

    layer_manager = get_layer_manager_for(instance.armature)
    if layer_manager:
        layer_manager.restore_native_action(instance.armature)
        layer_manager.deactivate_channel(
            channel=AnimChannel.PHYSICS,
            fade_out=0.0,
        )


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
                bs.lock_locomotion(duration, armature_name)
    except ImportError:
        pass
    _log(f"Locked animations for {armature_name}")


def _unlock_animations(armature_name: str):
    try:
        from ..animations.blend_system import get_blend_system
        bs = get_blend_system()
        if bs:
            if hasattr(bs, 'end_ragdoll_lock'):
                bs.end_ragdoll_lock(armature_name)
            if hasattr(bs, 'unlock_locomotion'):
                bs.unlock_locomotion(armature_name)
    except ImportError:
        pass
    _log(f"Unlocked animations for {armature_name}")
