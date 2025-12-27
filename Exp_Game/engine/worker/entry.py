# Exp_Game/engine/worker/entry.py
"""
Worker process entry point.
Contains the main worker loop and job dispatcher.
This module is loaded by worker_bootstrap.py and runs in isolated worker processes.
IMPORTANT: Uses sys.path manipulation for imports to work with spec_from_file_location.

NUMPY OPTIMIZATION (2025-12):
  - Animation blending uses numpy vectorized operations
  - 30-100x faster than previous Python loops
  - Processes ALL bones in single vectorized calls
"""

import time
import traceback
import sys
import os
from queue import Empty

import numpy as np

# Add engine folder to path for worker submodule imports
# This is necessary because bootstrap uses spec_from_file_location
_worker_dir = os.path.dirname(os.path.abspath(__file__))
_engine_dir = os.path.dirname(_worker_dir)
if _engine_dir not in sys.path:
    sys.path.insert(0, _engine_dir)

# Import from sibling modules (using absolute imports with sys.path)
from worker.math import (
    compute_aabb,
    build_triangle_grid,
)

from worker.physics import handle_kcc_physics_step
from worker.jobs import handle_camera_occlusion

# Import interaction/reaction handlers from new submodules
from worker.interactions import (
    handle_interaction_check_batch,
    handle_cache_trackers,
    handle_evaluate_trackers,
    reset_tracker_state,
)
from worker.reactions import (
    handle_projectile_update_batch,
    reset_projectile_state,
    handle_hitscan_batch,
    handle_transform_batch,
    handle_tracking_batch,
)

# Import animation math from single source of truth (no duplicates!)
from animations.blend import (
    sample_bone_animation,
    sample_object_animation,
    blend_bone_poses,
    blend_object_transforms,
    slerp_vectorized,
)

# Import IK solver for worker-side IK computation
from animations.ik import (
    solve_two_bone_ik,
    solve_leg_ik,
    solve_arm_ik,
    compute_knee_pole_position,
    compute_elbow_pole_position,
    LEG_IK,
    ARM_IK,
)

# Import joint limits for anatomical constraints during pose blending
from animations.joint_limits import (
    set_worker_limits,
    get_worker_limits,
    has_worker_limits,
    apply_limits_to_pose,
    clamp_rotation,
)

# Import rig cache for worker-based forward kinematics + IK
from animations.rig_cache import (
    RigFK,
    compute_point_at_target_quat,
    _matrix_to_quat,
)

# Import full-body IK solver
from animations.full_body import handle_full_body_ik


# ============================================================================
# DEBUG FLAG
# ============================================================================
# Controlled by scene.dev_debug_engine in the Developer Tools panel (main thread)
DEBUG_ENGINE = False


# ============================================================================
# WORKER-SIDE CACHES
# ============================================================================

# Grid is sent once via CACHE_GRID job and stored here for all subsequent raycasts.
# This avoids 3MB serialization per raycast (20ms overhead eliminated).
_cached_grid = None

# Will hold dynamic mesh data after CACHE_DYNAMIC_MESH jobs
_cached_dynamic_meshes = {}

# Persistent transform cache: {obj_id: (matrix_16_tuple, world_aabb, inv_matrix)}
# Main thread sends transform updates only when meshes MOVE.
# Worker caches last known transform for each mesh.
_cached_dynamic_transforms = {}

# Animation cache: {anim_name: {bone_transforms: np.ndarray, bone_names: list, duration, fps, ...}}
# Sent once via CACHE_ANIMATIONS job. Per-frame, only times/weights are sent.
# NOW USES NUMPY ARRAYS for 30-100x faster blending!
_cached_animations = {}

# Flag to track if numpy arrays have been reconstructed from lists
_animations_numpy_ready = False

# Pose library cache: {pose_name: {bone_transforms: {bone: [10 floats]}, source_armature: str}}
# Sent once via CACHE_POSES job. Used by PLAY_POSE jobs for pose playback.
_cached_poses = {}

# Rig cache: {armature_name: RigFK instance}
# Sent once via CACHE_RIG job. Used for worker-based forward kinematics + IK.
_cached_rigs = {}

# NOTE: Tracker state moved to worker/interactions/trackers.py


# ============================================================================
# NUMPY ANIMATION COMPUTE (Vectorized worker-side blending)
# ============================================================================

def _ensure_numpy_arrays():
    """
    Convert cached animation data from lists to numpy arrays (one-time operation).
    Called lazily on first animation compute after cache is populated.
    Returns list of log messages to report status.
    """
    global _animations_numpy_ready

    logs = []

    if _animations_numpy_ready:
        return logs

    convert_start = time.perf_counter()

    total_bones = 0
    animated_bones = 0

    for anim_name, anim_data in _cached_animations.items():
        # Convert bone_transforms list to numpy array
        bt = anim_data.get("bone_transforms")
        if bt is not None and not isinstance(bt, np.ndarray):
            if len(bt) > 0:
                anim_data["bone_transforms"] = np.array(bt, dtype=np.float32)
                if len(anim_data["bone_transforms"].shape) > 1:
                    total_bones += anim_data["bone_transforms"].shape[1]
            else:
                anim_data["bone_transforms"] = np.empty((0, 0, 10), dtype=np.float32)

        # Convert animated_mask list to numpy array
        am = anim_data.get("animated_mask")
        if am is not None and not isinstance(am, np.ndarray):
            anim_data["animated_mask"] = np.array(am, dtype=bool)
            animated_bones += int(np.sum(anim_data["animated_mask"]))

        # Convert object_transforms if present
        ot = anim_data.get("object_transforms")
        if ot is not None and not isinstance(ot, np.ndarray):
            anim_data["object_transforms"] = np.array(ot, dtype=np.float32)

    convert_ms = (time.perf_counter() - convert_start) * 1000
    _animations_numpy_ready = True

    # Log numpy conversion success - this confirms numpy is working!
    logs.append(("ANIMATIONS", f"[NUMPY] READY {len(_cached_animations)} anims | {total_bones} bones ({animated_bones} animated) | convert={convert_ms:.1f}ms"))

    return logs


def _compute_single_object_pose(object_name: str, playing_list: list, logs: list) -> dict:
    """
    Compute blended pose for a single object using NUMPY VECTORIZED operations.
    Processes ALL bones at once - no Python loops!

    Supports both:
    - Armatures: returns bone_transforms dict
    - Objects: returns object_transform tuple (for mesh, empty, etc.)

    STATIC BONE OPTIMIZATION: Tracks and logs how many bones were skipped.

    Args:
        object_name: Name of the object
        playing_list: List of playing animation dicts
        logs: List to append log messages to

    Returns:
        {
            "bone_transforms": {bone_name: (10-float tuple), ...},
            "bone_names": list,
            "bones_count": int,
            "object_transform": (10-float tuple) or None,  # For non-armature objects
            "anims_blended": int,
            "static_skipped": int,
        }
    """
    if not playing_list:
        return {
            "bone_transforms": {},
            "bone_names": [],
            "bones_count": 0,
            "object_transform": None,
            "anims_blended": 0,
            "static_skipped": 0,
        }

    # Ensure numpy arrays are ready (one-time conversion)
    # Returns logs on first call only
    numpy_logs = _ensure_numpy_arrays()
    logs.extend(numpy_logs)

    # Sample each playing animation using numpy
    # For bones (armatures)
    poses = []
    bone_weights = []
    # For object-level transforms (mesh, empty, etc.)
    object_transforms = []
    object_weights = []

    anim_names = []
    bone_names = None  # Will be set from first valid animation
    total_static_skipped = 0
    total_animated = 0

    for p in playing_list:
        anim_name = p.get("anim_name")
        anim_time = p.get("time", 0.0)
        weight = p.get("weight", 1.0)
        looping = p.get("looping", True)

        if weight < 0.001:
            continue

        # Get cached animation
        anim_data = _cached_animations.get(anim_name)
        if anim_data is None:
            logs.append(("ANIMATIONS", f"CACHE_MISS obj={object_name} anim='{anim_name}' cached={len(_cached_animations)}"))
            continue

        anim_names.append(f"{anim_name}:{weight:.0%}")

        # Sample BONE transforms (for armatures) - uses imported function
        pose, sample_stats = sample_bone_animation(anim_data, anim_time, looping)
        if pose.size > 0:
            poses.append(pose)
            bone_weights.append(weight)

            # Track static bone skipping stats
            if sample_stats.get("skipped"):
                total_static_skipped = max(total_static_skipped, sample_stats.get("static", 0))
                total_animated = max(total_animated, sample_stats.get("animated", 0))

            # Get bone names from first valid animation
            if bone_names is None:
                bone_names = anim_data.get("bone_names", [])

        # Sample OBJECT transforms (for mesh, empty, etc.) - uses imported function
        obj_transform = sample_object_animation(anim_data, anim_time, looping)
        if obj_transform is not None:
            object_transforms.append(obj_transform)
            object_weights.append(weight)

    # Build result
    result = {
        "bone_transforms": {},
        "bone_names": bone_names or [],
        "bones_count": 0,
        "object_transform": None,
        "anims_blended": max(len(poses), len(object_transforms)),
        "anim_names": anim_names,
        "static_skipped": total_static_skipped,
        "animated_count": total_animated,
    }

    # Blend bone poses (for armatures) - uses imported function
    if poses:
        blended = blend_bone_poses(poses, bone_weights)

        # Convert numpy array to dict for compatibility with apply code
        bone_transforms = {}
        if bone_names and blended.size > 0:
            for i, name in enumerate(bone_names):
                if i < len(blended):
                    bone_transforms[name] = tuple(blended[i].tolist())

        result["bone_transforms"] = bone_transforms
        result["bones_count"] = len(bone_transforms)

    # Blend object transforms (for non-armature objects) - uses imported function
    if object_transforms:
        blended_obj = blend_object_transforms(object_transforms, object_weights)
        if blended_obj is not None:
            result["object_transform"] = tuple(blended_obj.tolist())

    return result


def _handle_animation_compute_batch(job_data: dict) -> dict:
    """
    Handle ANIMATION_COMPUTE_BATCH job - compute blended poses for ALL objects in one job.

    This is the optimized path: ONE IPC round-trip for ALL animated objects.

    Input job_data:
        {
            "objects": {
                "Player": {
                    "playing": [{"anim_name": str, "time": float, "weight": float, "looping": bool}, ...]
                },
                "NPC_1": {
                    "playing": [...]
                },
                ...
            }
        }

    Returns:
        {
            "success": bool,
            "results": {
                "Player": {"bone_transforms": {...}, "bones_count": int, "anims_blended": int},
                "NPC_1": {"bone_transforms": {...}, ...},
                ...
            },
            "total_objects": int,
            "total_bones": int,
            "total_anims": int,
            "calc_time_us": float,
            "logs": [(category, message), ...]
        }
    """
    calc_start = time.perf_counter()
    logs = []

    objects_data = job_data.get("objects", {})

    if not objects_data:
        return {
            "success": True,
            "results": {},
            "total_objects": 0,
            "total_bones": 0,
            "total_anims": 0,
            "calc_time_us": 0.0,
            "logs": []
        }

    # Process ALL objects in this single job
    results = {}
    total_bones = 0
    total_anims = 0
    total_static_skipped = 0
    total_animated_interp = 0
    per_object_logs = []

    for object_name, obj_data in objects_data.items():
        playing_list = obj_data.get("playing", [])

        # Compute pose for this object
        obj_result = _compute_single_object_pose(object_name, playing_list, logs)

        # Store result (includes object_transform for non-armature objects)
        results[object_name] = {
            "bone_transforms": obj_result["bone_transforms"],
            "bones_count": obj_result["bones_count"],
            "object_transform": obj_result.get("object_transform"),
            "anims_blended": obj_result["anims_blended"],
        }

        total_bones += obj_result["bones_count"]
        total_anims += obj_result["anims_blended"]
        total_static_skipped += obj_result.get("static_skipped", 0)
        total_animated_interp += obj_result.get("animated_count", obj_result["bones_count"])

        # Per-object log for detailed debugging
        anim_names = obj_result.get("anim_names", [])
        if anim_names:
            anim_str = "+".join(anim_names[:3])
            if len(anim_names) > 3:
                anim_str += f"+{len(anim_names)-3}more"
            per_object_logs.append(f"{object_name}({anim_str})")

    calc_time_us = (time.perf_counter() - calc_start) * 1_000_000

    # Summary log: one line for entire batch
    # [NUMPY] confirms vectorized path, shows static bone skipping stats
    obj_count = len(results)
    if per_object_logs:
        # Compact format: show up to 3 objects, then count
        if len(per_object_logs) <= 3:
            obj_summary = " ".join(per_object_logs)
        else:
            obj_summary = f"{per_object_logs[0]} +{len(per_object_logs)-1}more"

        # Include static bone skip info if any bones were skipped
        if total_static_skipped > 0:
            skip_info = f" [SKIP {total_static_skipped} static, interp {total_animated_interp}]"
        else:
            skip_info = ""

        logs.append(("ANIMATIONS", f"[NUMPY] BATCH {obj_count}obj {total_bones}bones {total_anims}anims {calc_time_us:.0f}µs{skip_info} | {obj_summary}"))

    return {
        "success": True,
        "results": results,
        "total_objects": obj_count,
        "total_bones": total_bones,
        "total_anims": total_anims,
        "calc_time_us": calc_time_us,
        "logs": logs
    }


# ============================================================================
# POSE BLEND COMPUTE (Worker-side pose-to-pose blending)
# ============================================================================

def _handle_pose_blend_compute(job_data: dict) -> dict:
    """
    Handle POSE_BLEND_COMPUTE job - blend between two poses with optional IK.

    This is the WORKER-BASED version of pose-to-pose blending.
    All slerp/lerp math happens here, not on main thread.

    JOINT LIMITS: If limits are cached (via CACHE_JOINT_LIMITS), they are applied
    after blending to ensure anatomically valid poses.

    Input job_data:
        {
            "pose_a": {bone_name: [qw,qx,qy,qz,lx,ly,lz,sx,sy,sz], ...},
            "pose_b": {bone_name: [10 floats], ...},
            "weight": float (0-1),
            "bone_names": [list of bone names to blend],
            "apply_limits": bool (default True - apply joint limits if cached),
            "ik_chains": [
                {
                    "chain": "arm_R",
                    "target": [x, y, z],
                    "influence": float,
                    "char_forward": [x, y, z],
                    "char_right": [x, y, z],
                    "char_up": [x, y, z],
                },
                ...
            ]
        }

    Returns:
        {
            "success": bool,
            "bone_transforms": {bone_name: (10-float tuple), ...},
            "ik_results": {chain_name: {...}, ...},
            "limits_applied": int (number of bones clamped),
            "calc_time_us": float,
            "logs": [(category, message), ...]
        }
    """
    calc_start = time.perf_counter()
    logs = []

    pose_a = job_data.get("pose_a", {})
    pose_b = job_data.get("pose_b", {})
    weight = job_data.get("weight", 0.5)
    bone_names = job_data.get("bone_names", list(set(pose_a.keys()) | set(pose_b.keys())))
    ik_chains = job_data.get("ik_chains", [])
    apply_limits = job_data.get("apply_limits", True)  # Enable by default

    # Identity transform for missing bones
    identity = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

    # Blend poses using numpy
    bone_transforms = {}
    num_bones = len(bone_names)

    if num_bones > 0:
        # Build numpy arrays for vectorized blending
        transforms_a = np.zeros((num_bones, 10), dtype=np.float32)
        transforms_b = np.zeros((num_bones, 10), dtype=np.float32)

        for i, bone_name in enumerate(bone_names):
            t_a = pose_a.get(bone_name, identity)
            t_b = pose_b.get(bone_name, identity)
            transforms_a[i] = t_a
            transforms_b[i] = t_b

        # Quaternion slerp (indices 0-3) - vectorized
        quats_a = transforms_a[:, :4]
        quats_b = transforms_b[:, :4]
        blended_quats = slerp_vectorized(quats_a, quats_b, weight)

        # Location lerp (indices 4-6) - simple vectorized lerp
        locs_a = transforms_a[:, 4:7]
        locs_b = transforms_b[:, 4:7]
        blended_locs = locs_a + (locs_b - locs_a) * weight

        # Scale lerp (indices 7-9)
        scales_a = transforms_a[:, 7:10]
        scales_b = transforms_b[:, 7:10]
        blended_scales = scales_a + (scales_b - scales_a) * weight

        # Combine into result
        blended = np.zeros((num_bones, 10), dtype=np.float32)
        blended[:, :4] = blended_quats
        blended[:, 4:7] = blended_locs
        blended[:, 7:10] = blended_scales

        # Convert to dict
        for i, bone_name in enumerate(bone_names):
            bone_transforms[bone_name] = tuple(blended[i].tolist())

    # Apply joint limits if enabled and limits are cached
    limits_applied = 0
    clamped_bones_list = []
    if apply_limits and has_worker_limits() and bone_transforms:
        # Convert bone_transforms to format expected by apply_limits_to_pose
        # Format: {bone_name: [qw, qx, qy, qz, lx, ly, lz, sx, sy, sz]}
        transforms_for_limits = {
            bone: list(xform) for bone, xform in bone_transforms.items()
        }

        # Apply limits (returns clamped transforms, count, and list of clamped bones)
        clamped_transforms, limits_applied, clamped_bones_list = apply_limits_to_pose(transforms_for_limits)

        # Convert back to tuple format
        bone_transforms = {
            bone: tuple(xform) for bone, xform in clamped_transforms.items()
        }

        if limits_applied > 0:
            logs.append(("JOINT-LIMITS", f"Clamped {limits_applied} bones: {', '.join(clamped_bones_list[:5])}{'...' if len(clamped_bones_list) > 5 else ''}"))

    # Process IK chains using delta-based approach (fully worker-based)
    # This requires: armature_world matrix, armature_name for rig cache lookup
    ik_results = {}
    armature_name = job_data.get("armature_name", "")
    armature_world_flat = job_data.get("armature_world")

    # Get cached rig for FK computation
    rig_fk = _cached_rigs.get(armature_name) if armature_name else None

    if ik_chains and rig_fk and rig_fk.is_ready() and armature_world_flat:
        # Convert armature world matrix from flat list to numpy
        armature_world = np.array(armature_world_flat, dtype=np.float32).reshape(4, 4).T

        # Extract bone quaternions and locations from blended transforms
        bone_quats = {}
        bone_locs = {}
        for bone_name, xform in bone_transforms.items():
            bone_quats[bone_name] = (xform[0], xform[1], xform[2], xform[3])
            bone_locs[bone_name] = (xform[4], xform[5], xform[6])

        # Compute forward kinematics - bone world matrices from FK blend
        world_matrices = rig_fk.compute_world_matrices(armature_world, bone_quats, bone_locs)

        # Process each IK chain
        for ik_data in ik_chains:
            chain = ik_data.get("chain")
            target = np.array(ik_data.get("target", [0, 0, 0]), dtype=np.float32)
            influence = ik_data.get("influence", 1.0)

            # Get character orientation for pole calculation
            char_forward = np.array(ik_data.get("char_forward", [0, 1, 0]), dtype=np.float32)
            char_right = np.array(ik_data.get("char_right", [1, 0, 0]), dtype=np.float32)
            char_up = np.array(ik_data.get("char_up", [0, 0, 1]), dtype=np.float32)

            # Parse chain type
            parts = chain.split('_') if chain else []
            if len(parts) != 2:
                continue
            limb_type, side = parts[0], parts[1]
            is_leg = (limb_type == "leg")

            # Get chain definition
            chain_def = LEG_IK.get(chain) if is_leg else ARM_IK.get(chain)
            if not chain_def:
                continue

            root_bone = chain_def['root']
            mid_bone = chain_def['mid']
            tip_bone = chain_def['tip']

            # Get bone world positions from FK
            root_head, root_tail = rig_fk.get_bone_head_tail(root_bone, world_matrices)
            mid_head, mid_tail = rig_fk.get_bone_head_tail(mid_bone, world_matrices)

            # Compute pole and solve IK to get joint world position
            if is_leg:
                pole_pos = compute_knee_pole_position(root_head, target, char_forward, char_right, side, 0.5)
                _, _, joint_world = solve_leg_ik(root_head, target, pole_pos, side)
            else:
                pole_pos = compute_elbow_pole_position(root_head, target, char_forward, char_up, side, 0.3)
                _, _, joint_world = solve_arm_ik(root_head, target, pole_pos, side)

            # DELTA-BASED IK: Use point_bone_at_target approach
            # Root bone points at joint, mid bone points at target

            # Get bone world matrices and extract rotations
            root_world_mat = world_matrices.get(root_bone, np.eye(4))
            mid_world_mat = world_matrices.get(mid_bone, np.eye(4))

            root_world_rot = _matrix_to_quat(root_world_mat)
            mid_world_rot = _matrix_to_quat(mid_world_mat)

            # Get rest local rotations
            root_rest = rig_fk.get_rest_local(root_bone)
            mid_rest = rig_fk.get_rest_local(mid_bone)
            root_rest_rot = _matrix_to_quat(root_rest)
            mid_rest_rot = _matrix_to_quat(mid_rest)

            # Get parent world rotations
            root_parent = rig_fk.get_parent(root_bone)
            mid_parent = rig_fk.get_parent(mid_bone)  # This should be root_bone

            root_parent_world_rot = None
            if root_parent and root_parent in world_matrices:
                root_parent_world_rot = _matrix_to_quat(world_matrices[root_parent])

            mid_parent_world_rot = _matrix_to_quat(world_matrices.get(root_bone, np.eye(4)))

            # Compute delta-based IK quaternions
            # STEP 1: Compute root bone rotation to point at joint
            upper_local_quat = compute_point_at_target_quat(
                bone_head=root_head,
                bone_tail=root_tail,
                bone_world_rot=root_world_rot,
                target_world=joint_world,
                rest_local_rot=root_rest_rot,
                parent_world_rot=root_parent_world_rot
            )

            # STEP 2: Recompute root's world matrix with new rotation
            # This is needed because mid bone's position depends on root's rotation
            from animations.rig_cache import _quat_loc_to_matrix
            root_new_pose_mat = _quat_loc_to_matrix(
                tuple(upper_local_quat.tolist()),
                bone_locs.get(root_bone, (0, 0, 0))
            )
            root_rest_local = rig_fk.get_rest_local(root_bone)
            if root_parent and root_parent in world_matrices:
                root_parent_world = world_matrices[root_parent]
            else:
                root_parent_world = armature_world
            root_new_world = root_parent_world @ root_rest_local @ root_new_pose_mat

            # STEP 3: Compute mid bone's new world position (depends on root)
            mid_new_pose_mat = _quat_loc_to_matrix(
                bone_quats.get(mid_bone, (1, 0, 0, 0)),
                bone_locs.get(mid_bone, (0, 0, 0))
            )
            mid_rest_local = rig_fk.get_rest_local(mid_bone)
            mid_new_world = root_new_world @ mid_rest_local @ mid_new_pose_mat

            # Get mid's new head/tail positions
            mid_new_head = mid_new_world[:3, 3].copy()
            mid_length = rig_fk.get_length(mid_bone)
            mid_new_tail = mid_new_head + mid_new_world[:3, 1] * mid_length
            mid_new_world_rot = _matrix_to_quat(mid_new_world)

            # Get root's new world rotation for mid's parent
            root_new_world_rot = _matrix_to_quat(root_new_world)

            # STEP 4: Compute mid bone rotation to point at target
            lower_local_quat = compute_point_at_target_quat(
                bone_head=mid_new_head,
                bone_tail=mid_new_tail,
                bone_world_rot=mid_new_world_rot,
                target_world=target,
                rest_local_rot=mid_rest_rot,
                parent_world_rot=root_new_world_rot
            )

            # Apply IK with influence
            if influence < 1.0:
                # Slerp between original FK quat and IK quat
                orig_upper = np.array(bone_quats.get(root_bone, (1,0,0,0)), dtype=np.float32)
                orig_lower = np.array(bone_quats.get(mid_bone, (1,0,0,0)), dtype=np.float32)
                upper_local_quat = slerp_vectorized(
                    orig_upper.reshape(1, 4),
                    upper_local_quat.reshape(1, 4),
                    influence
                )[0]
                lower_local_quat = slerp_vectorized(
                    orig_lower.reshape(1, 4),
                    lower_local_quat.reshape(1, 4),
                    influence
                )[0]

            # Update bone_transforms with IK result
            if root_bone in bone_transforms:
                orig = list(bone_transforms[root_bone])
                orig[0:4] = upper_local_quat.tolist()
                bone_transforms[root_bone] = tuple(orig)

            if mid_bone in bone_transforms:
                orig = list(bone_transforms[mid_bone])
                orig[0:4] = lower_local_quat.tolist()
                bone_transforms[mid_bone] = tuple(orig)

            # Store result info
            ik_results[chain] = {
                "upper_quat": tuple(upper_local_quat.tolist()),
                "lower_quat": tuple(lower_local_quat.tolist()),
                "joint_world": tuple(joint_world.tolist()) if isinstance(joint_world, np.ndarray) else tuple(joint_world),
                "pole_pos": tuple(pole_pos.tolist()) if isinstance(pole_pos, np.ndarray) else tuple(pole_pos),
                "target": tuple(target.tolist()),
                "influence": influence,
                "fk_based": True,  # Flag that this used delta-based IK
            }
    calc_time_us = (time.perf_counter() - calc_start) * 1_000_000

    # Build log message with limit info if any were applied
    limit_info = f" limits={limits_applied}" if limits_applied > 0 else ""
    logs.append(("POSE-BLEND", f"[NUMPY] {len(bone_transforms)} bones weight={weight:.2f} ik={len(ik_results)} chains{limit_info} {calc_time_us:.0f}µs"))

    return {
        "success": True,
        "bone_transforms": bone_transforms,
        "ik_results": ik_results,
        "bones_count": len(bone_transforms),
        "limits_applied": limits_applied,
        "clamped_bones": clamped_bones_list,
        "calc_time_us": calc_time_us,
        "logs": logs
    }


def _handle_ik_solve_batch(job_data: dict) -> dict:
    """
    Handle IK_SOLVE_BATCH job - solve IK for multiple chains in one job.

    Input job_data:
        {
            "chains": [
                {
                    "chain": "arm_R",
                    "root_pos": [x, y, z],
                    "target": [x, y, z],
                    "influence": float,
                    "char_forward": [x, y, z],
                    "char_right": [x, y, z],
                    "char_up": [x, y, z],
                },
                ...
            ]
        }

    Returns:
        {
            "success": bool,
            "results": {
                "arm_R": {
                    "upper_quat": (w, x, y, z),
                    "lower_quat": (w, x, y, z),
                    "joint_world": (x, y, z),
                    "pole_pos": (x, y, z),
                    ...
                },
                ...
            },
            "calc_time_us": float,
            "logs": [(category, message), ...]
        }
    """
    calc_start = time.perf_counter()
    logs = []

    chains_data = job_data.get("chains", [])
    results = {}

    for ik_data in chains_data:
        chain = ik_data.get("chain")
        if not chain:
            continue

        root_pos = np.array(ik_data.get("root_pos", [0, 0, 0]), dtype=np.float32)
        target = np.array(ik_data.get("target", [0, 0, 0]), dtype=np.float32)
        influence = ik_data.get("influence", 1.0)

        char_forward = np.array(ik_data.get("char_forward", [0, 1, 0]), dtype=np.float32)
        char_right = np.array(ik_data.get("char_right", [1, 0, 0]), dtype=np.float32)
        char_up = np.array(ik_data.get("char_up", [0, 0, 1]), dtype=np.float32)

        # Parse chain type
        parts = chain.split('_')
        if len(parts) != 2:
            continue
        limb_type, side = parts[0], parts[1]
        is_leg = (limb_type == "leg")

        # Compute pole and solve
        if is_leg:
            chain_def = LEG_IK.get(chain, {})
            pole_pos = compute_knee_pole_position(root_pos, target, char_forward, char_right, side, 0.5)
            upper_quat, lower_quat, joint_world = solve_leg_ik(root_pos, target, pole_pos, side)
        else:
            chain_def = ARM_IK.get(chain, {})
            pole_pos = compute_elbow_pole_position(root_pos, target, char_forward, char_up, side, 0.3)
            upper_quat, lower_quat, joint_world = solve_arm_ik(root_pos, target, pole_pos, side)

        # Calculate reach info
        reach_dist = float(np.linalg.norm(target - root_pos))
        max_reach = chain_def.get("reach", 1.0)
        reach_pct = (reach_dist / max_reach * 100) if max_reach > 0 else 100

        results[chain] = {
            "upper_quat": tuple(upper_quat.tolist()) if isinstance(upper_quat, np.ndarray) else tuple(upper_quat),
            "lower_quat": tuple(lower_quat.tolist()) if isinstance(lower_quat, np.ndarray) else tuple(lower_quat),
            "joint_world": tuple(joint_world.tolist()) if isinstance(joint_world, np.ndarray) else tuple(joint_world),
            "pole_pos": tuple(pole_pos.tolist()) if isinstance(pole_pos, np.ndarray) else tuple(pole_pos),
            "target": tuple(target.tolist()),
            "root_pos": tuple(root_pos.tolist()),
            "influence": influence,
            "reach_dist": reach_dist,
            "reach_pct": reach_pct,
            "reachable": reach_pct <= 100,
        }

    calc_time_us = (time.perf_counter() - calc_start) * 1_000_000

    logs.append(("IK-SOLVE", f"[WORKER] {len(results)} chains {calc_time_us:.0f}µs"))

    return {
        "success": True,
        "results": results,
        "chains_solved": len(results),
        "calc_time_us": calc_time_us,
        "logs": logs
    }


# ============================================================================
# TRACKER EVALUATION - MOVED TO worker/interactions/trackers.py
# ============================================================================


# ============================================================================
# JOB DISPATCHER
# ============================================================================

def process_job(job) -> dict:
    """
    Process a single job and return result as a plain dict (pickle-safe).
    IMPORTANT: NO bpy access here!
    """
    global _cached_grid, _cached_dynamic_meshes, _cached_dynamic_transforms
    start_time = time.perf_counter()

    try:
        # ===================================================================
        # JOB TYPE DISPATCH
        # ===================================================================

        if job.job_type == "ECHO":
            # Simple echo test
            result_data = {
                "echoed": job.data,
                "worker_msg": "Job processed successfully"
            }

        elif job.job_type == "PING":
            # Worker verification ping - used during startup to confirm worker responsiveness
            result_data = {
                "pong": True,
                "worker_id": job.data.get("worker_check", -1),
                "timestamp": time.time(),
                "worker_msg": "Worker alive and responsive"
            }

        elif job.job_type == "CACHE_GRID":
            # Cache spatial grid for subsequent raycast jobs
            # This is sent ONCE at game start to avoid 3MB serialization per raycast
            grid = job.data.get("grid", None)
            if grid is not None:
                _cached_grid = grid
                tri_count = len(grid.get("triangles", []))
                cell_count = len(grid.get("cells", {}))
                result_data = {
                    "success": True,
                    "triangles": tri_count,
                    "cells": cell_count,
                    "message": "Grid cached successfully"
                }
            else:
                result_data = {
                    "success": False,
                    "error": "No grid data provided"
                }

        elif job.job_type == "CACHE_DYNAMIC_MESH":
            # Cache dynamic mesh triangles in LOCAL space
            # This is sent ONCE per dynamic mesh (or when mesh changes)
            # Per-frame, only transform matrices are sent (64 bytes vs 3MB!)
            #
            # OPTIMIZATION: Pre-compute local AABB once here.
            # Per-frame we only transform 8 AABB corners instead of N*3 vertices.
            obj_id = job.data.get("obj_id")
            triangles = job.data.get("triangles", [])
            radius = job.data.get("radius", 1.0)

            if obj_id is not None and triangles:
                # DIAGNOSTIC: Check if already cached (potential duplicate caching)
                was_cached = obj_id in _cached_dynamic_meshes

                # Compute local-space AABB ONCE (O(N) here, O(8) per frame)
                local_aabb = compute_aabb(triangles)

                # Build spatial grid for O(cells) ray testing instead of O(N)
                # This is the key optimization for high-poly meshes!
                tri_grid = build_triangle_grid(triangles, local_aabb)
                grid_cells = len(tri_grid["cells"]) if tri_grid else 0

                _cached_dynamic_meshes[obj_id] = {
                    "triangles": triangles,   # List of (v0, v1, v2) tuples in local space
                    "local_aabb": local_aabb, # Pre-computed local AABB for fast world transform
                    "radius": radius,         # Bounding sphere radius for quick rejection
                    "grid": tri_grid          # Spatial grid for fast ray-triangle culling
                }
                tri_count = len(triangles)
                grid_res = tri_grid["resolution"] if tri_grid else 0
                if DEBUG_ENGINE:
                    print(f"[Worker] Dynamic mesh cached: obj_id={obj_id} tris={tri_count:,} radius={radius:.2f}m grid={grid_res}³={grid_cells}cells")

                # Log to diagnostics (one-time cache event)
                status = "RE-CACHED" if was_cached else "CACHED"
                total_cached = len(_cached_dynamic_meshes)
                cache_log = ("DYN-CACHE", f"{status} obj_id={obj_id} tris={tri_count} grid={grid_res}³={grid_cells}cells radius={radius:.2f}m total={total_cached}")
                logs = [cache_log]

                result_data = {
                    "success": True,
                    "obj_id": obj_id,
                    "triangle_count": tri_count,
                    "radius": radius,
                    "message": "Dynamic mesh cached successfully",
                    "logs": logs  # Return all logs to main thread
                }
            else:
                result_data = {
                    "success": False,
                    "error": "Missing obj_id or triangles data"
                }

        elif job.job_type == "CLEAR_DYNAMIC_CACHE":
            # Clear dynamic mesh caches - sent on game end/reset to prevent memory leaks
            # Can clear specific mesh (obj_id provided) or all meshes (obj_id=None)
            obj_id = job.data.get("obj_id", None)
            clear_all = job.data.get("clear_all", False)

            cleared_meshes = 0
            cleared_transforms = 0

            if clear_all or obj_id is None:
                # Clear ALL dynamic caches
                cleared_meshes = len(_cached_dynamic_meshes)
                cleared_transforms = len(_cached_dynamic_transforms)
                _cached_dynamic_meshes.clear()
                _cached_dynamic_transforms.clear()
                if DEBUG_ENGINE:
                    print(f"[Worker] Cleared ALL dynamic caches: {cleared_meshes} meshes, {cleared_transforms} transforms")
            else:
                # Clear specific mesh
                if obj_id in _cached_dynamic_meshes:
                    del _cached_dynamic_meshes[obj_id]
                    cleared_meshes = 1
                if obj_id in _cached_dynamic_transforms:
                    del _cached_dynamic_transforms[obj_id]
                    cleared_transforms = 1
                if DEBUG_ENGINE:
                    print(f"[Worker] Cleared dynamic cache for obj_id={obj_id}")

            result_data = {
                "success": True,
                "cleared_meshes": cleared_meshes,
                "cleared_transforms": cleared_transforms,
                "remaining_meshes": len(_cached_dynamic_meshes),
                "remaining_transforms": len(_cached_dynamic_transforms),
                "message": "Dynamic cache cleared"
            }

        elif job.job_type == "COMPUTE_HEAVY":
            # Stress test - simulate realistic game calculation
            # (e.g., pathfinding, physics prediction, AI decision)
            iterations = job.data.get("iterations", 10)
            data = job.data.get("data", [])

            if DEBUG_ENGINE:
                print(f"[Worker] COMPUTE_HEAVY job - iterations={iterations}, data_size={len(data)}")

            # Simulate realistic computation (1-5ms per job)
            total = 0
            for i in range(iterations):
                for val in data:
                    total += val * i
                    total = (total * 31 + val) % 1000000

            result_data = {
                "iterations_completed": iterations,
                "data_size": len(data),
                "result": total,
                "worker_msg": f"Completed {iterations} iterations",
                "scenario": job.data.get("scenario", "UNKNOWN"),
                "frame": job.data.get("frame", -1),
            }

        elif job.job_type == "CULL_BATCH":
            # Performance culling - distance-based object visibility
            entry_ptr = job.data.get("entry_ptr", 0)
            obj_names = job.data.get("obj_names", [])
            obj_positions = job.data.get("obj_positions", [])
            ref_loc = job.data.get("ref_loc", (0, 0, 0))
            thresh = job.data.get("thresh", 10.0)
            start = job.data.get("start", 0)
            max_count = job.data.get("max_count", 100)

            rx, ry, rz = ref_loc
            t2 = float(thresh) * float(thresh)
            n = len(obj_names)

            if n == 0:
                result_data = {"entry_ptr": entry_ptr, "next_idx": start, "changes": []}
            else:
                i = 0
                changes = []
                idx = start % n

                while i < n and len(changes) < max_count:
                    name = obj_names[idx]
                    px, py, pz = obj_positions[idx]
                    dx = px - rx
                    dy = py - ry
                    dz = pz - rz
                    far = (dx*dx + dy*dy + dz*dz) > t2
                    changes.append((name, far))
                    i += 1
                    idx = (idx + 1) % n

                result_data = {"entry_ptr": entry_ptr, "next_idx": idx, "changes": changes}

        elif job.job_type == "INTERACTION_CHECK_BATCH":
            # Interaction proximity & collision checks - delegated to interactions module
            result_data = handle_interaction_check_batch(job.data)

        elif job.job_type == "KCC_PHYSICS_STEP":
            # Full KCC physics step - delegated to worker.physics module
            result_data = handle_kcc_physics_step(
                job.data,
                _cached_grid,
                _cached_dynamic_meshes,
                _cached_dynamic_transforms
            )

        elif job.job_type == "CAMERA_OCCLUSION_FULL":
            # Camera occlusion - delegated to worker.jobs module
            result_data = handle_camera_occlusion(
                job.data,
                _cached_grid,
                _cached_dynamic_meshes,
                _cached_dynamic_transforms
            )

        elif job.job_type == "CACHE_ANIMATIONS":
            # Cache baked animations for subsequent ANIMATION_COMPUTE jobs
            # Sent ONCE at game start. Per-frame, only times/weights are sent.
            # NOW WITH NUMPY: Arrays are lazily converted on first use
            global _animations_numpy_ready
            animations_data = job.data.get("animations", {})

            if animations_data:
                # Clear existing cache and store new animations
                _cached_animations.clear()
                _animations_numpy_ready = False  # Reset - will convert to numpy on first use

                for anim_name, anim_dict in animations_data.items():
                    _cached_animations[anim_name] = anim_dict

                anim_count = len(_cached_animations)
                # Count bones from new numpy format (bone_names list)
                total_bones = sum(
                    len(anim.get("bone_names", []))
                    for anim in _cached_animations.values()
                )

                # Log for diagnostics (no console print - use log system only)
                logs = [("ANIM-CACHE", f"WORKER_CACHED {anim_count} anims, {total_bones} bones (numpy)")]

                result_data = {
                    "success": True,
                    "animation_count": anim_count,
                    "total_bone_channels": total_bones,
                    "message": "Animations cached successfully (numpy optimized)",
                    "logs": logs
                }
            else:
                result_data = {
                    "success": False,
                    "error": "No animation data provided"
                }

        elif job.job_type == "CACHE_POSES":
            # Cache pose library for subsequent PLAY_POSE jobs
            # Sent ONCE at game start. Poses are static snapshots (no animation).
            global _cached_poses
            poses_data = job.data.get("poses", {})

            if poses_data:
                # Clear existing cache and store new poses
                _cached_poses.clear()

                for pose_name, pose_dict in poses_data.items():
                    _cached_poses[pose_name] = pose_dict

                pose_count = len(_cached_poses)
                total_bones = sum(
                    p.get("bone_count", 0)
                    for p in _cached_poses.values()
                )

                # Log for diagnostics (no console print - use log system only)
                logs = [("POSE-CACHE", f"WORKER_CACHED {pose_count} poses, {total_bones} bones")]

                result_data = {
                    "success": True,
                    "pose_count": pose_count,
                    "total_bones": total_bones,
                    "message": "Poses cached successfully",
                    "logs": logs
                }
            else:
                result_data = {
                    "success": False,
                    "error": "No pose data provided"
                }

        elif job.job_type == "CACHE_JOINT_LIMITS":
            # Cache joint limits for anatomical constraints during pose blending
            # Sent ONCE at game start. Used by POSE_BLEND_COMPUTE to clamp rotations.
            limits_data = job.data.get("limits", {})

            if limits_data:
                # Store in worker via joint_limits module
                set_worker_limits(limits_data)

                bone_count = len(limits_data)
                axis_count = sum(
                    len(bone_limits)
                    for bone_limits in limits_data.values()
                )

                logs = [("JOINT-LIMITS", f"WORKER_CACHED {bone_count} bones, {axis_count} axis constraints")]

                result_data = {
                    "success": True,
                    "bone_count": bone_count,
                    "axis_count": axis_count,
                    "message": "Joint limits cached successfully",
                    "logs": logs
                }
            else:
                result_data = {
                    "success": False,
                    "error": "No joint limits data provided"
                }

        elif job.job_type == "CACHE_RIG":
            # Cache rig data for worker-based forward kinematics + IK.
            # Sent ONCE at game start. Used by POSE_BLEND_COMPUTE for full FK/IK.
            global _cached_rigs
            rig_data = job.data.get("rig_data", {})
            armature_name = job.data.get("armature_name", "")

            if rig_data and armature_name:
                # Create RigFK instance and load data
                rig_fk = RigFK()
                success = rig_fk.load(rig_data)

                if success:
                    _cached_rigs[armature_name] = rig_fk

                    bone_count = rig_fk.bone_count
                    logs = [("ANIM-WORKER", f"CACHE_RIG {armature_name} {bone_count} bones")]

                    result_data = {
                        "success": True,
                        "armature_name": armature_name,
                        "bone_count": bone_count,
                        "message": "Rig cached successfully for FK/IK",
                        "logs": logs
                    }
                else:
                    result_data = {
                        "success": False,
                        "error": "Failed to load rig data"
                    }
            else:
                result_data = {
                    "success": False,
                    "error": "Missing rig_data or armature_name"
                }

        elif job.job_type == "ANIMATION_COMPUTE_BATCH":
            # OPTIMIZED: Compute blended poses for ALL objects in ONE job
            # Input: {"objects": {obj_name: {playing: [...]}, ...}}
            # Output: {"results": {obj_name: {bone_transforms: {...}}, ...}}
            # This eliminates O(n) IPC overhead - one round trip regardless of object count
            result_data = _handle_animation_compute_batch(job.data)

        elif job.job_type == "POSE_BLEND_COMPUTE":
            # Pose-to-pose blending with optional IK (worker-based)
            # Input: {"pose_a": {...}, "pose_b": {...}, "weight": float, "ik_chains": [...]}
            # Output: {"bone_transforms": {...}, "ik_results": {...}}
            result_data = _handle_pose_blend_compute(job.data)

        elif job.job_type == "IK_SOLVE_BATCH":
            # Batch IK solving for multiple chains
            # Input: {"chains": [{chain, target, root_pos, ...}, ...]}
            # Output: {"results": {chain: {upper_quat, lower_quat, ...}, ...}}
            result_data = _handle_ik_solve_batch(job.data)

        elif job.job_type == "FULL_BODY_IK":
            # Full-body IK solving (hips, legs, spine, arms, head)
            # Input: {"constraints": {...}, "current_state": {...}, "armature_name": str}
            # Output: {"bone_transforms": {...}, "constraints_satisfied": int, ...}
            result_data = handle_full_body_ik(job.data, _cached_rigs)

        elif job.job_type == "CACHE_TRACKERS":
            # Cache serialized tracker definitions - delegated to interactions module
            result_data = handle_cache_trackers(job.data)

        elif job.job_type == "EVALUATE_TRACKERS":
            # Evaluate all cached trackers - delegated to interactions module
            result_data = handle_evaluate_trackers(job.data)

        # ─────────────────────────────────────────────────────────────────
        # REACTION JOBS - delegated to reactions module
        # ─────────────────────────────────────────────────────────────────

        elif job.job_type == "PROJECTILE_UPDATE_BATCH":
            # Projectile physics simulation - gravity + sweep raycast
            result_data = handle_projectile_update_batch(
                job.data,
                _cached_grid,
                _cached_dynamic_meshes,
                _cached_dynamic_transforms
            )

        elif job.job_type == "HITSCAN_BATCH":
            # Hitscan instant raycasting
            result_data = handle_hitscan_batch(
                job.data,
                _cached_grid,
                _cached_dynamic_meshes,
                _cached_dynamic_transforms
            )

        elif job.job_type == "TRANSFORM_BATCH":
            # Transform interpolation (lerp/slerp) - offloaded from main thread
            result_data = handle_transform_batch(job.data)

        elif job.job_type == "TRACKING_BATCH":
            # Tracking movement (sweep/slide/gravity) - offloaded from main thread
            # Uses unified_raycast for collision detection
            result_data = handle_tracking_batch(
                job.data,
                _cached_grid,
                _cached_dynamic_meshes,
                _cached_dynamic_transforms
            )

        else:
            # Unknown job type - still succeed but note it
            result_data = {
                "message": f"Unknown job type '{job.job_type}' - no handler registered",
                "data": job.data
            }

        processing_time = time.perf_counter() - start_time

        return {
            "job_id": job.job_id,
            "job_type": job.job_type,
            "result": result_data,
            "success": True,
            "error": None,
            "timestamp": time.perf_counter(),
            "processing_time": processing_time
        }

    except Exception as e:
        processing_time = time.perf_counter() - start_time

        return {
            "job_id": job.job_id,
            "job_type": job.job_type,
            "result": None,
            "success": False,
            "error": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
            "timestamp": time.perf_counter(),
            "processing_time": processing_time
        }


# ============================================================================
# WORKER MAIN LOOP
# ============================================================================

def worker_loop(job_queue, result_queue, worker_id, shutdown_event):
    """
    Main loop for a worker process.
    This is the entry point called by multiprocessing.Process.
    """
    if DEBUG_ENGINE:
        print(f"[Engine Worker {worker_id}] Started")

    jobs_processed = 0

    try:
        while not shutdown_event.is_set():
            try:
                # Wait for a job (with timeout so we can check shutdown_event)
                job = job_queue.get(timeout=0.1)

                # Check if this job is targeted at a specific worker
                target = getattr(job, 'target_worker', -1)
                if target >= 0 and target != worker_id:
                    # This job is for a different worker - put it back and try again
                    try:
                        job_queue.put_nowait(job)
                    except Exception:
                        pass  # Queue full, job will be lost (shouldn't happen)
                    continue

                if DEBUG_ENGINE:
                    print(f"[Engine Worker {worker_id}] Processing job {job.job_id} (type: {job.job_type})")

                # Process the job
                result = process_job(job)

                # Add worker_id to result before sending (CRITICAL for grid cache verification)
                result["worker_id"] = worker_id

                # Send result back
                result_queue.put(result)

                jobs_processed += 1

                if DEBUG_ENGINE:
                    print(f"[Engine Worker {worker_id}] Completed job {job.job_id} in {result['processing_time']*1000:.2f}ms")

            except Empty:
                # Queue is empty, just continue
                continue
            except Exception as e:
                # Handle any queue errors or unexpected issues
                if not shutdown_event.is_set():
                    if DEBUG_ENGINE:
                        print(f"[Engine Worker {worker_id}] Error: {e}")
                        traceback.print_exc()
                continue

    finally:
        if DEBUG_ENGINE:
            print(f"[Engine Worker {worker_id}] Shutting down (processed {jobs_processed} jobs)")
            