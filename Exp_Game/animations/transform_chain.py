# Exp_Game/animations/transform_chain.py
"""
Transform Chain Tracker - Solves the "stale parent" problem in IK.

THE PROBLEM:
When computing bone rotations in a chain (e.g., thigh -> shin), the code
queries Blender for parent state via bone.parent.matrix. But Blender hasn't
been updated yet - we computed the parent's rotation but haven't applied it!

This causes the child rotation to be computed against the WRONG parent state,
resulting in 50-130° errors even though verification "passes".

THE SOLUTION:
Track our own transform chain. When we compute a bone's rotation, we store it.
When computing a child's rotation, we use our stored parent rotation instead
of querying Blender's stale data.

USAGE:
    chain = TransformChainTracker(armature)

    # Compute rotations in parent-to-child order
    thigh_quat = chain.compute_rotation_to_direction("LeftThigh", thigh_dir)
    shin_quat = chain.compute_rotation_to_direction("LeftShin", shin_dir)

    # Get all computed transforms
    transforms = chain.get_transforms()

    # Apply to Blender
    chain.apply_to_armature()
"""

import bpy
import math
from mathutils import Vector, Quaternion, Matrix
from typing import Dict, Optional, Tuple, List

from ..developer.dev_logger import log_game


def _force_log(msg: str):
    """Always log - not gated by debug toggle."""
    from ..developer.dev_logger import _log_buffer, _current_frame
    import time
    _log_buffer.append({
        'frame': _current_frame,
        'time': time.perf_counter(),
        'category': 'TRANSFORM-CHAIN',
        'message': msg
    })


class TransformChainTracker:
    """
    Tracks bone transforms without relying on Blender's stale data.

    Caches rest pose data once, then tracks pending rotations as we compute them.
    When computing a child's rotation, uses the tracked parent rotation.
    """

    def __init__(self, armature: bpy.types.Object):
        self.armature = armature
        self.arm_matrix = armature.matrix_world.copy()
        self.arm_quat = self.arm_matrix.to_quaternion()

        # Cache rest pose data (static, safe to read once)
        self._rest_local: Dict[str, Matrix] = {}  # bone_name -> matrix_local
        self._rest_in_parent: Dict[str, Matrix] = {}  # bone_name -> parent_rest^-1 @ bone_rest
        self._parent_name: Dict[str, Optional[str]] = {}  # bone_name -> parent_name

        # Pending local rotations we've computed but not applied to Blender
        self._pending_local_quats: Dict[str, Quaternion] = {}

        # Diagnostic data for debugging
        self._diagnostics: Dict[str, any] = {}

        self._cache_rest_poses()

    def _cache_rest_poses(self):
        """Cache all bone rest poses - these never change during a solve."""
        for bone in self.armature.pose.bones:
            name = bone.name
            self._rest_local[name] = bone.bone.matrix_local.copy()
            self._parent_name[name] = bone.parent.name if bone.parent else None

            if bone.parent:
                parent_rest = bone.parent.bone.matrix_local
                self._rest_in_parent[name] = parent_rest.inverted() @ bone.bone.matrix_local

    def clear_pending(self):
        """Clear all pending rotations (start fresh)."""
        self._pending_local_quats.clear()
        self._diagnostics.clear()

    def get_bone_world_quat(self, bone_name: str) -> Quaternion:
        """
        Get bone's world orientation, accounting for all pending parent rotations.

        This is THE KEY METHOD. It computes what the bone's world orientation
        WILL BE once all pending rotations are applied.

        DOES NOT query Blender's bone.matrix (which is stale).
        """
        if bone_name not in self._rest_local:
            return Quaternion()

        # Build chain from root to this bone
        chain = []
        current = bone_name
        while current:
            chain.append(current)
            current = self._parent_name.get(current)
        chain.reverse()  # Now goes root -> ... -> bone

        # Start with armature world orientation
        world_quat = self.arm_quat.copy()

        # Walk down the chain
        for name in chain:
            # Get bone's rest pose relative to parent
            if name in self._rest_in_parent:
                rest_quat = self._rest_in_parent[name].to_quaternion()
            else:
                # Root bone - rest_local is already in armature space
                rest_quat = self._rest_local[name].to_quaternion()

            # Accumulate: world = world @ rest_relative_to_parent
            world_quat = world_quat @ rest_quat

            # Apply pending local rotation if we have one
            pending = self._pending_local_quats.get(name)
            if pending:
                world_quat = world_quat @ pending

        world_quat.normalize()
        return world_quat

    def get_bone_rest_y_world(self, bone_name: str) -> Vector:
        """
        Get bone's Y axis in world space at REST pose (no pending rotations).
        """
        if bone_name not in self._rest_local:
            return Vector((0, 1, 0))

        rest_mat = self._rest_local[bone_name]
        rest_y_armature = rest_mat.to_3x3() @ Vector((0, 1, 0))
        rest_y_world = self.arm_matrix.to_3x3() @ rest_y_armature
        return rest_y_world.normalized()

    def get_bone_current_y_world(self, bone_name: str) -> Vector:
        """
        Get bone's Y axis in world space AFTER pending rotations.

        This is what the bone's Y axis WILL point to once applied.
        """
        world_quat = self.get_bone_world_quat(bone_name)
        y_world = world_quat @ Vector((0, 1, 0))
        return y_world.normalized()

    def compute_rotation_to_direction(
        self,
        bone_name: str,
        target_dir: Vector,
        clamp_to_limits: bool = False
    ) -> Quaternion:
        """
        Compute local rotation to make bone's Y axis point at target_dir.

        Uses our tracked transform chain (NOT Blender's stale data).
        Automatically stores the result for use by child bones.

        Args:
            bone_name: Name of bone to rotate
            target_dir: World-space direction the bone should point (normalized)
            clamp_to_limits: Whether to apply joint limits

        Returns:
            Local quaternion to apply to bone.rotation_quaternion
        """
        target_dir = target_dir.normalized()

        if bone_name not in self._rest_local:
            _force_log(f"  [ERROR] Bone '{bone_name}' not found in armature")
            return Quaternion()

        parent_name = self._parent_name.get(bone_name)

        # =================================================================
        # STEP 1: Get bone's CURRENT Y direction (after parent rotations)
        # =================================================================
        if parent_name:
            # Get parent's world orientation (including our pending rotation!)
            parent_world_quat = self.get_bone_world_quat(parent_name)

            # Bone's rest Y in parent's local space
            rest_in_parent_mat = self._rest_in_parent[bone_name]
            rest_y_in_parent = rest_in_parent_mat.to_3x3() @ Vector((0, 1, 0))

            # Bone's current Y in world (after parent rotates, before our rotation)
            current_y_world = (parent_world_quat @ rest_y_in_parent).normalized()

            # Bone's current world orientation (for local frame conversion)
            rest_in_parent_quat = rest_in_parent_mat.to_quaternion()
            bone_current_world_quat = parent_world_quat @ rest_in_parent_quat
        else:
            # Root bone - use armature-space rest pose
            rest_mat = self._rest_local[bone_name]
            rest_y_armature = rest_mat.to_3x3() @ Vector((0, 1, 0))
            current_y_world = (self.arm_matrix.to_3x3() @ rest_y_armature).normalized()

            rest_quat = rest_mat.to_quaternion()
            bone_current_world_quat = self.arm_quat @ rest_quat

        # =================================================================
        # STEP 2: Compute world rotation from current to target
        # =================================================================
        world_rotation = current_y_world.rotation_difference(target_dir)

        # Angle for logging
        dot = max(-1.0, min(1.0, current_y_world.dot(target_dir)))
        angle_deg = math.degrees(math.acos(dot))

        # =================================================================
        # STEP 3: Convert to local space
        # =================================================================
        # Local rotation = bone_world^-1 @ world_rotation @ bone_world
        bone_current_inv = bone_current_world_quat.inverted()
        local_quat = bone_current_inv @ world_rotation @ bone_current_world_quat
        local_quat.normalize()

        # =================================================================
        # STEP 4: Verify the math (BEFORE storing)
        # =================================================================
        # Compute what Y will be after this rotation
        verify_world_quat = bone_current_world_quat @ local_quat
        verify_y = verify_world_quat @ Vector((0, 1, 0))
        verify_y.normalize()

        verify_dot = max(-1.0, min(1.0, verify_y.dot(target_dir)))
        verify_error = math.degrees(math.acos(verify_dot))

        # =================================================================
        # STEP 5: Apply joint limits if requested
        # =================================================================
        if clamp_to_limits:
            local_quat = self._clamp_to_limits(local_quat, bone_name)

        # =================================================================
        # STEP 6: Store for use by children
        # =================================================================
        self._pending_local_quats[bone_name] = local_quat.copy()

        # =================================================================
        # STEP 7: Log diagnostics
        # =================================================================
        _force_log(f"  [CHAIN] {bone_name}:")
        _force_log(f"    current_Y=({current_y_world.x:.3f}, {current_y_world.y:.3f}, {current_y_world.z:.3f})")
        _force_log(f"    target_Y=({target_dir.x:.3f}, {target_dir.y:.3f}, {target_dir.z:.3f})")
        _force_log(f"    angle_to_rotate={angle_deg:.1f}°")
        _force_log(f"    local_quat=({local_quat.w:.4f}, {local_quat.x:.4f}, {local_quat.y:.4f}, {local_quat.z:.4f})")
        _force_log(f"    verify_Y=({verify_y.x:.3f}, {verify_y.y:.3f}, {verify_y.z:.3f})")
        _force_log(f"    verify_error={verify_error:.2f}° {'✓' if verify_error < 1.0 else '✗'}")

        # Store diagnostic data
        self._diagnostics[f"{bone_name}_target_dir"] = (target_dir.x, target_dir.y, target_dir.z)
        self._diagnostics[f"{bone_name}_verify_y"] = (verify_y.x, verify_y.y, verify_y.z)
        self._diagnostics[f"{bone_name}_verify_error"] = verify_error
        self._diagnostics[f"{bone_name}_local_quat"] = (local_quat.w, local_quat.x, local_quat.y, local_quat.z)

        return local_quat

    def _clamp_to_limits(self, quat: Quaternion, bone_name: str) -> Quaternion:
        """Clamp quaternion to joint limits (euler-based)."""
        from ..engine.animations.default_limits import get_bone_limit

        limits = get_bone_limit(bone_name)
        if not limits:
            return quat

        euler = quat.to_euler('XYZ')
        x_deg = math.degrees(euler.x)
        y_deg = math.degrees(euler.y)
        z_deg = math.degrees(euler.z)

        x_limits = limits.get("X", [-180, 180])
        y_limits = limits.get("Y", [-180, 180])
        z_limits = limits.get("Z", [-180, 180])

        x_clamped = max(x_limits[0], min(x_limits[1], x_deg))
        y_clamped = max(y_limits[0], min(y_limits[1], y_deg))
        z_clamped = max(z_limits[0], min(z_limits[1], z_deg))

        from mathutils import Euler
        clamped_euler = Euler((math.radians(x_clamped), math.radians(y_clamped), math.radians(z_clamped)), 'XYZ')
        return clamped_euler.to_quaternion()

    def get_transforms(self) -> Dict[str, List[float]]:
        """
        Get all computed transforms in format ready for FullBodyResult.

        Returns:
            Dict mapping bone_name -> [qw, qx, qy, qz, lx, ly, lz]
        """
        result = {}
        for bone_name, quat in self._pending_local_quats.items():
            result[bone_name] = [quat.w, quat.x, quat.y, quat.z, 0, 0, 0]
        return result

    def get_diagnostics(self) -> Dict[str, any]:
        """Get diagnostic data from this solve."""
        return self._diagnostics.copy()

    def apply_to_armature(self):
        """
        Apply all pending rotations to the armature.

        Call this ONCE after computing all rotations.
        Then call bpy.context.view_layer.update() to propagate changes.
        """
        pose_bones = self.armature.pose.bones

        _force_log("=" * 70)
        _force_log("APPLYING TRANSFORMS TO ARMATURE:")

        for bone_name, local_quat in self._pending_local_quats.items():
            bone = pose_bones.get(bone_name)
            if not bone:
                _force_log(f"  [SKIP] {bone_name} - not found")
                continue

            bone.rotation_mode = 'QUATERNION'
            bone.rotation_quaternion = local_quat
            _force_log(f"  [SET] {bone_name}: ({local_quat.w:.4f}, {local_quat.x:.4f}, {local_quat.y:.4f}, {local_quat.z:.4f})")

        _force_log("=" * 70)

    def verify_post_apply(self) -> Dict[str, float]:
        """
        Verify transforms AFTER applying and updating view layer.

        Call this AFTER apply_to_armature() and view_layer.update().
        Compares actual bone Y directions to what we expected.

        Returns:
            Dict mapping bone_name -> error_degrees
        """
        errors = {}
        pose_bones = self.armature.pose.bones
        arm_matrix = self.armature.matrix_world

        _force_log("=" * 70)
        _force_log("POST-APPLY VERIFICATION:")

        for bone_name in self._pending_local_quats.keys():
            bone = pose_bones.get(bone_name)
            if not bone:
                continue

            # Get actual Y direction from Blender
            bone_world_mat = arm_matrix @ bone.matrix
            actual_y = bone_world_mat.to_3x3() @ Vector((0, 1, 0))
            actual_y.normalize()

            # Get expected Y direction from our diagnostics
            expected_key = f"{bone_name}_target_dir"
            if expected_key not in self._diagnostics:
                continue

            expected = Vector(self._diagnostics[expected_key])
            expected.normalize()

            # Compute error
            dot = max(-1.0, min(1.0, actual_y.dot(expected)))
            error_deg = math.degrees(math.acos(dot))
            errors[bone_name] = error_deg

            status = "✓ OK" if error_deg < 5.0 else "✗ WRONG"
            _force_log(f"  {bone_name}:")
            _force_log(f"    expected=({expected.x:.3f}, {expected.y:.3f}, {expected.z:.3f})")
            _force_log(f"    actual=({actual_y.x:.3f}, {actual_y.y:.3f}, {actual_y.z:.3f})")
            _force_log(f"    error={error_deg:.1f}° {status}")

        _force_log("=" * 70)
        return errors


def compute_limb_rotations(
    armature: bpy.types.Object,
    upper_bone_name: str,
    lower_bone_name: str,
    upper_dir: Vector,
    lower_dir: Vector,
    clamp_upper: bool = True,
    clamp_lower: bool = True,
) -> Tuple[Quaternion, Quaternion, TransformChainTracker]:
    """
    Convenience function to compute rotations for a two-bone limb.

    Handles the parent-child dependency correctly.

    Args:
        armature: The armature object
        upper_bone_name: e.g., "LeftThigh" or "LeftArm"
        lower_bone_name: e.g., "LeftShin" or "LeftForeArm"
        upper_dir: World direction for upper bone
        lower_dir: World direction for lower bone
        clamp_upper: Apply joint limits to upper bone
        clamp_lower: Apply joint limits to lower bone

    Returns:
        Tuple of (upper_quat, lower_quat, chain_tracker)
    """
    chain = TransformChainTracker(armature)

    # CRITICAL: Compute upper first, then lower
    # The chain tracker will use upper's pending rotation when computing lower
    upper_quat = chain.compute_rotation_to_direction(upper_bone_name, upper_dir, clamp_upper)
    lower_quat = chain.compute_rotation_to_direction(lower_bone_name, lower_dir, clamp_lower)

    return upper_quat, lower_quat, chain
