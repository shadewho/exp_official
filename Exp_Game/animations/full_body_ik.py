# Exp_Game/animations/full_body_ik.py
"""
Unified IK System - Main Thread Coordinator

Self-diagnosing IK solving for any body region.
Logs are COMPREHENSIVE - you should never need screenshots to debug.

LOGGING FORMAT (always outputs):
  ═══════════════════════════════════════════════════════════════
  IK SOLVE #N - [REGION]
  ───────────────────────────────────────────────────────────────
  VISUALIZER: [status]
  ───────────────────────────────────────────────────────────────
  BEFORE STATE:
    Root:   world=(x, y, z)
    Hips:   rel=(x, y, z)  rot=(w, x, y, z)
    L_foot: rel=(x, y, z)  dist_from_root=Nm
    R_foot: rel=(x, y, z)  dist_from_root=Nm
    ...
  ───────────────────────────────────────────────────────────────
  CONSTRAINTS:
    Hips:   drop=0.20m
    L_foot: target=(x, y, z)  dist_to_move=Ncm
    ...
  ───────────────────────────────────────────────────────────────
  SOLVING:
    [step-by-step solve process]
  ───────────────────────────────────────────────────────────────
  AFTER STATE:
    Hips:   rel=(x, y, z)  MOVED_BY=(dx, dy, dz)
    L_foot: rel=(x, y, z)  ERROR=Ncm [OK/WARN/FAIL]
    ...
  ───────────────────────────────────────────────────────────────
  RESULT: [SUCCESS/PARTIAL/FAILED] N/M constraints, Nus
  ═══════════════════════════════════════════════════════════════
"""

import bpy
import time
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from mathutils import Vector, Quaternion, Matrix, Euler

from ..developer.dev_logger import log_game
from ..engine.animations.ik import LEG_IK, ARM_IK, solve_leg_ik, solve_arm_ik, compute_knee_pole_position, compute_elbow_pole_position


def _force_log(msg: str):
    """Always log - not gated by debug toggle. For diagnostics."""
    # Write directly to buffer without gate check
    from ..developer.dev_logger import _log_buffer, _current_frame
    import time
    _log_buffer.append({
        'frame': _current_frame,
        'time': time.perf_counter(),
        'category': 'IK-DIAG',
        'message': msg
    })


# =============================================================================
# CROSS-BODY ANALYSIS - Anatomical awareness for arm IK
# =============================================================================

class CrossBodyAnalysis:
    """
    Analysis of cross-body arm reach.

    ANATOMY RULES:
    - Left elbow X must be <= body_center_x (left side or center)
    - Right elbow X must be >= body_center_x (right side or center)
    - Elbow Y must be <= shoulder Y (behind or at shoulder level)
    - These are HARD constraints - violations mean impossible pose
    """

    def __init__(self):
        self.is_cross_body = False
        self.side = "L"
        self.shoulder_x = 0.0
        self.target_x = 0.0
        self.elbow_x = 0.0
        self.body_center_x = 0.0

        # Anatomical checks
        self.elbow_on_wrong_side = False
        self.elbow_in_front = False
        self.anatomically_possible = True
        self.problems = []
        self.recommendations = []

    def log(self):
        """Log cross-body analysis results."""
        _force_log("=" * 70)
        _force_log(f"CROSS-BODY ANALYSIS ({self.side}_ARM):")
        _force_log("-" * 70)

        # Position summary
        side_name = "LEFT" if self.side == "L" else "RIGHT"
        _force_log(f"  Shoulder X:    {self.shoulder_x:+.3f}m ({side_name} side)")
        _force_log(f"  Target X:      {self.target_x:+.3f}m ({'RIGHT' if self.target_x > 0 else 'LEFT'} side)")
        _force_log(f"  Body Center:   {self.body_center_x:.3f}m")

        if self.is_cross_body:
            _force_log(f"  >>> CROSS-BODY REACH DETECTED <<<")
        else:
            _force_log(f"  (normal reach - same side)")

        _force_log("-" * 70)
        _force_log(f"  Computed Elbow X: {self.elbow_x:+.3f}m")

        # Anatomical verdict
        if self.side == "L":
            expected_side = "LEFT (X ≤ 0)"
            actual_side = "LEFT" if self.elbow_x <= 0.01 else "RIGHT ❌"
        else:
            expected_side = "RIGHT (X ≥ 0)"
            actual_side = "RIGHT" if self.elbow_x >= -0.01 else "LEFT ❌"

        _force_log(f"  Elbow should be: {expected_side}")
        _force_log(f"  Elbow actually:  {actual_side} (X={self.elbow_x:+.3f})")

        _force_log("-" * 70)

        if self.anatomically_possible:
            _force_log("  ✅ ANATOMICALLY VALID")
        else:
            _force_log("  ❌ ANATOMICALLY IMPOSSIBLE")
            for p in self.problems:
                _force_log(f"     • {p}")

        if self.recommendations:
            _force_log("  REQUIRED FIXES:")
            for r in self.recommendations:
                _force_log(f"     → {r}")

        _force_log("=" * 70)


def analyze_cross_body(
    side: str,
    shoulder_pos: tuple,
    target_pos: tuple,
    elbow_pos: tuple,
    body_center_x: float = 0.0
) -> CrossBodyAnalysis:
    """
    Analyze an arm reach for cross-body issues.

    Args:
        side: "L" or "R"
        shoulder_pos: (x, y, z) shoulder world position
        target_pos: (x, y, z) hand target world position
        elbow_pos: (x, y, z) computed elbow position
        body_center_x: X coordinate of body centerline (usually 0)

    Returns:
        CrossBodyAnalysis with anatomical verdict
    """
    analysis = CrossBodyAnalysis()
    analysis.side = side
    analysis.shoulder_x = shoulder_pos[0]
    analysis.target_x = target_pos[0]
    analysis.elbow_x = elbow_pos[0]
    analysis.body_center_x = body_center_x

    # Detect cross-body reach
    if side == "L":
        # Left arm: shoulder is on left (X < 0), cross-body if target is on right (X > 0)
        analysis.is_cross_body = target_pos[0] > body_center_x + 0.05

        # Left elbow must stay on left side (X <= body_center)
        if elbow_pos[0] > body_center_x + 0.02:
            analysis.elbow_on_wrong_side = True
            analysis.anatomically_possible = False
            analysis.problems.append(f"Left elbow crossed to RIGHT side (X={elbow_pos[0]:.3f})")
            analysis.recommendations.append("Rotate shoulder internally to keep elbow on left")
    else:
        # Right arm: shoulder is on right (X > 0), cross-body if target is on left (X < 0)
        analysis.is_cross_body = target_pos[0] < body_center_x - 0.05

        # Right elbow must stay on right side (X >= body_center)
        if elbow_pos[0] < body_center_x - 0.02:
            analysis.elbow_on_wrong_side = True
            analysis.anatomically_possible = False
            analysis.problems.append(f"Right elbow crossed to LEFT side (X={elbow_pos[0]:.3f})")
            analysis.recommendations.append("Rotate shoulder internally to keep elbow on right")

    # Check if elbow is in front of shoulder (Y check)
    # In character space, +Y is forward
    elbow_forward_of_shoulder = elbow_pos[1] - shoulder_pos[1]
    if elbow_forward_of_shoulder > 0.03:  # 3cm tolerance
        analysis.elbow_in_front = True
        analysis.anatomically_possible = False
        analysis.problems.append(f"Elbow is {elbow_forward_of_shoulder*100:.1f}cm IN FRONT of shoulder")
        analysis.recommendations.append("Elbow must be behind or at shoulder Y level")

    # Cross-body specific recommendations
    if analysis.is_cross_body and not analysis.anatomically_possible:
        analysis.recommendations.append("Consider: torso rotation toward target")
        analysis.recommendations.append("Consider: shoulder protraction (scapula wrapping)")

    return analysis


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class IKTarget:
    """Single IK target constraint."""
    position: Tuple[float, float, float]  # Target position (Root-relative)
    enabled: bool = True
    weight: float = 1.0  # 0-1, how strongly to satisfy this constraint

    def to_dict(self) -> dict:
        return {
            "position": self.position,
            "enabled": self.enabled,
            "weight": self.weight,
        }


@dataclass
class FullBodyConstraints:
    """All constraints for a full-body IK solve."""

    # Foot grounding (required for most solves)
    left_foot: Optional[IKTarget] = None
    right_foot: Optional[IKTarget] = None

    # Hand reaching (optional)
    left_hand: Optional[IKTarget] = None
    right_hand: Optional[IKTarget] = None

    # Head look-at (optional)
    look_at: Optional[IKTarget] = None

    # Hips control
    hips_drop: float = 0.0  # Meters to drop hips (crouch)
    hips_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # XYZ offset

    def to_dict(self) -> dict:
        return {
            "left_foot": self.left_foot.to_dict() if self.left_foot else None,
            "right_foot": self.right_foot.to_dict() if self.right_foot else None,
            "left_hand": self.left_hand.to_dict() if self.left_hand else None,
            "right_hand": self.right_hand.to_dict() if self.right_hand else None,
            "look_at": self.look_at.to_dict() if self.look_at else None,
            "hips_drop": self.hips_drop,
            "hips_offset": self.hips_offset,
        }

    def get_active_count(self) -> int:
        """Count active constraints."""
        count = 0
        if self.left_foot and self.left_foot.enabled:
            count += 1
        if self.right_foot and self.right_foot.enabled:
            count += 1
        if self.left_hand and self.left_hand.enabled:
            count += 1
        if self.right_hand and self.right_hand.enabled:
            count += 1
        if self.look_at and self.look_at.enabled:
            count += 1
        if abs(self.hips_drop) > 0.001:
            count += 1
        return count


@dataclass
class FullBodyState:
    """Current state of the skeleton (for logging/verification)."""

    # Root (world anchor)
    root_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Hips (relative to Root)
    hips_pos: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    hips_rot: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # wxyz

    # Spine lean (degrees from vertical)
    spine_lean: float = 0.0

    # End effector positions (Root-relative)
    left_foot_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    right_foot_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    left_hand_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    right_hand_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    head_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class FullBodyResult:
    """Result of a full-body IK solve."""

    success: bool = False
    solve_time_us: float = 0.0

    # Constraint satisfaction
    constraints_satisfied: int = 0
    constraints_total: int = 0
    constraints_at_limit: int = 0

    # Per-effector errors (cm)
    left_foot_error_cm: float = 0.0
    right_foot_error_cm: float = 0.0
    left_hand_error_cm: float = 0.0
    right_hand_error_cm: float = 0.0

    # Joint limit violations
    joint_violations: List[str] = field(default_factory=list)

    # Resulting bone transforms (bone_name -> [qw, qx, qy, qz, lx, ly, lz])
    bone_transforms: Dict[str, List[float]] = field(default_factory=dict)


# =============================================================================
# FULL-BODY IK CONTROLLER
# =============================================================================

class FullBodyIK:
    """
    Full-Body IK controller for an armature.

    Manages constraints, submits solve jobs to workers, applies results.
    """

    def __init__(self, armature: bpy.types.Object):
        """
        Initialize full-body IK for an armature.

        Args:
            armature: Blender armature object
        """
        if not armature or armature.type != 'ARMATURE':
            raise ValueError("FullBodyIK requires an armature object")

        self.armature = armature
        self.constraints = FullBodyConstraints()
        self._last_state: Optional[FullBodyState] = None
        self._last_result: Optional[FullBodyResult] = None
        self._solve_count = 0

    # =========================================================================
    # CONSTRAINT SETTERS
    # =========================================================================

    def set_foot_targets(
        self,
        left_pos: Optional[Tuple[float, float, float]] = None,
        right_pos: Optional[Tuple[float, float, float]] = None,
        weight: float = 1.0
    ):
        """
        Set foot grounding targets (Root-relative positions).

        Args:
            left_pos: Left foot target position (None to disable)
            right_pos: Right foot target position (None to disable)
            weight: Constraint weight (0-1)
        """
        if left_pos:
            self.constraints.left_foot = IKTarget(left_pos, True, weight)
        else:
            self.constraints.left_foot = None

        if right_pos:
            self.constraints.right_foot = IKTarget(right_pos, True, weight)
        else:
            self.constraints.right_foot = None

    def set_hand_targets(
        self,
        left_pos: Optional[Tuple[float, float, float]] = None,
        right_pos: Optional[Tuple[float, float, float]] = None,
        weight: float = 1.0
    ):
        """
        Set hand reaching targets (Root-relative positions).
        """
        if left_pos:
            self.constraints.left_hand = IKTarget(left_pos, True, weight)
        else:
            self.constraints.left_hand = None

        if right_pos:
            self.constraints.right_hand = IKTarget(right_pos, True, weight)
        else:
            self.constraints.right_hand = None

    def set_look_at(
        self,
        position: Optional[Tuple[float, float, float]] = None,
        weight: float = 1.0
    ):
        """Set head look-at target (Root-relative position)."""
        if position:
            self.constraints.look_at = IKTarget(position, True, weight)
        else:
            self.constraints.look_at = None

    def set_hips_drop(self, drop_meters: float):
        """
        Set hips drop for crouching.

        Args:
            drop_meters: How far to drop hips (positive = down)
        """
        self.constraints.hips_drop = drop_meters

    def set_hips_offset(self, offset: Tuple[float, float, float]):
        """Set hips XYZ offset for weight shift/lean."""
        self.constraints.hips_offset = offset

    def clear_constraints(self):
        """Clear all constraints."""
        self.constraints = FullBodyConstraints()

    # =========================================================================
    # STATE CAPTURE
    # =========================================================================

    def capture_state(self) -> FullBodyState:
        """
        Capture current skeleton state from armature.

        Returns:
            FullBodyState with current positions
        """
        state = FullBodyState()
        pose_bones = self.armature.pose.bones
        arm_matrix = self.armature.matrix_world

        # Root position (armature origin or Root bone if exists)
        root_bone = pose_bones.get("Root")
        if root_bone:
            root_world = arm_matrix @ root_bone.head
            state.root_pos = tuple(root_world)
        else:
            state.root_pos = tuple(self.armature.location)

        # Hips (relative to Root)
        hips_bone = pose_bones.get("Hips")
        if hips_bone:
            hips_world = arm_matrix @ hips_bone.head
            hips_rel = Vector(hips_world) - Vector(state.root_pos)
            state.hips_pos = tuple(hips_rel)
            state.hips_rot = tuple(hips_bone.rotation_quaternion)

        # Spine lean (angle from vertical)
        spine2_bone = pose_bones.get("Spine2")
        if spine2_bone and hips_bone:
            hips_world = arm_matrix @ hips_bone.head
            spine2_world = arm_matrix @ spine2_bone.head
            spine_vec = Vector(spine2_world) - Vector(hips_world)
            spine_vec.normalize()
            up = Vector((0, 0, 1))
            state.spine_lean = spine_vec.angle(up) * 57.2958  # radians to degrees

        # End effectors (Root-relative)
        root_vec = Vector(state.root_pos)

        for bone_name, attr_name in [
            ("LeftFoot", "left_foot_pos"),
            ("RightFoot", "right_foot_pos"),
            ("LeftHand", "left_hand_pos"),
            ("RightHand", "right_hand_pos"),
            ("Head", "head_pos"),
        ]:
            bone = pose_bones.get(bone_name)
            if bone:
                world_pos = arm_matrix @ bone.head
                rel_pos = Vector(world_pos) - root_vec
                setattr(state, attr_name, tuple(rel_pos))

        self._last_state = state
        return state

    # =========================================================================
    # VISUALIZER MANAGEMENT
    # =========================================================================

    def _ensure_visualizer_active(self):
        """Wake up visualizer if enabled but handlers not registered."""
        scene = bpy.context.scene

        # Check if visualizer should be on
        vis_enabled = getattr(scene, 'dev_rig_visualizer_enabled', False)
        if not vis_enabled:
            _force_log( "VISUALIZER: disabled (dev_rig_visualizer_enabled=False)")
            return

        # Check if handlers are registered
        from ..developer.rig_visualizer import is_visualizer_active, refresh_rig_visualizer

        if not is_visualizer_active():
            _force_log( "VISUALIZER: was enabled but handlers dead - WAKING UP")
            refresh_rig_visualizer()
            _force_log( "VISUALIZER: handlers re-registered")
        else:
            _force_log( "VISUALIZER: active and running")

    # =========================================================================
    # COMPREHENSIVE LOGGING (SELF-DIAGNOSING)
    # =========================================================================

    def _log_header(self, region: str = "FULL_BODY"):
        """Log solve header with visualizer status."""
        _force_log("=" * 64)
        _force_log(f"IK SOLVE #{self._solve_count} - {region}")
        _force_log("-" * 64)

    def _log_before_state(self, state: FullBodyState):
        """Log complete skeleton state BEFORE solving."""
        _force_log( "BEFORE STATE:")

        # Root - world position
        _force_log( f"  Root:   world=({state.root_pos[0]:.3f}, {state.root_pos[1]:.3f}, {state.root_pos[2]:.3f})")

        # Hips - relative to root with rotation
        hips_dist = math.sqrt(sum(x*x for x in state.hips_pos))
        _force_log( f"  Hips:   rel=({state.hips_pos[0]:.3f}, {state.hips_pos[1]:.3f}, {state.hips_pos[2]:.3f})  dist={hips_dist:.3f}m")
        _force_log( f"          rot=({state.hips_rot[0]:.3f}, {state.hips_rot[1]:.3f}, {state.hips_rot[2]:.3f}, {state.hips_rot[3]:.3f})")
        _force_log( f"  Spine:  lean={state.spine_lean:.1f}deg from vertical")

        # Feet - with distance to ground (Z=0)
        l_ground = state.left_foot_pos[2]  # Z relative to root
        r_ground = state.right_foot_pos[2]
        _force_log( f"  L_foot: rel=({state.left_foot_pos[0]:.3f}, {state.left_foot_pos[1]:.3f}, {state.left_foot_pos[2]:.3f})  ground_z={l_ground:.3f}")
        _force_log( f"  R_foot: rel=({state.right_foot_pos[0]:.3f}, {state.right_foot_pos[1]:.3f}, {state.right_foot_pos[2]:.3f})  ground_z={r_ground:.3f}")

        # Hands - with reach from shoulder estimate
        _force_log( f"  L_hand: rel=({state.left_hand_pos[0]:.3f}, {state.left_hand_pos[1]:.3f}, {state.left_hand_pos[2]:.3f})")
        _force_log( f"  R_hand: rel=({state.right_hand_pos[0]:.3f}, {state.right_hand_pos[1]:.3f}, {state.right_hand_pos[2]:.3f})")
        _force_log( f"  Head:   rel=({state.head_pos[0]:.3f}, {state.head_pos[1]:.3f}, {state.head_pos[2]:.3f})")

    def _log_constraints(self, state: FullBodyState):
        """Log active constraints with distance-to-move calculations."""
        _force_log( "-" * 64)
        _force_log( "CONSTRAINTS:")

        active = 0

        # Hips drop
        if abs(self.constraints.hips_drop) > 0.001:
            new_z = state.hips_pos[2] - self.constraints.hips_drop
            _force_log( f"  Hips:   DROP {self.constraints.hips_drop:.3f}m  (current_z={state.hips_pos[2]:.3f} -> target_z={new_z:.3f})")
            active += 1

        # Left foot
        if self.constraints.left_foot and self.constraints.left_foot.enabled:
            t = self.constraints.left_foot.position
            c = state.left_foot_pos
            dist = math.sqrt(sum((t[i]-c[i])**2 for i in range(3))) * 100
            _force_log( f"  L_foot: target=({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})  move={dist:.1f}cm  weight={self.constraints.left_foot.weight:.2f}")
            active += 1

        # Right foot
        if self.constraints.right_foot and self.constraints.right_foot.enabled:
            t = self.constraints.right_foot.position
            c = state.right_foot_pos
            dist = math.sqrt(sum((t[i]-c[i])**2 for i in range(3))) * 100
            _force_log( f"  R_foot: target=({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})  move={dist:.1f}cm  weight={self.constraints.right_foot.weight:.2f}")
            active += 1

        # Left hand
        if self.constraints.left_hand and self.constraints.left_hand.enabled:
            t = self.constraints.left_hand.position
            c = state.left_hand_pos
            dist = math.sqrt(sum((t[i]-c[i])**2 for i in range(3))) * 100
            max_reach = ARM_IK["arm_L"]["reach"] * 100
            status = "REACHABLE" if dist < max_reach else f"OUT_OF_REACH (max={max_reach:.0f}cm)"
            _force_log( f"  L_hand: target=({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})  move={dist:.1f}cm  {status}")
            active += 1

        # Right hand
        if self.constraints.right_hand and self.constraints.right_hand.enabled:
            t = self.constraints.right_hand.position
            c = state.right_hand_pos
            dist = math.sqrt(sum((t[i]-c[i])**2 for i in range(3))) * 100
            max_reach = ARM_IK["arm_R"]["reach"] * 100
            status = "REACHABLE" if dist < max_reach else f"OUT_OF_REACH (max={max_reach:.0f}cm)"
            _force_log( f"  R_hand: target=({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})  move={dist:.1f}cm  {status}")
            active += 1

        # Look-at
        if self.constraints.look_at and self.constraints.look_at.enabled:
            t = self.constraints.look_at.position
            _force_log( f"  LookAt: target=({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})")
            active += 1

        if active == 0:
            _force_log( "  (none)")

        _force_log( f"  TOTAL: {active} active constraints")

    def _log_solve_step(self, step: str, detail: str):
        """Log individual solve step - always logged (not gated)."""
        _force_log(f"[{step}] {detail}")

    def _verify_limb_ik(
        self,
        limb_name: str,
        root_pos,  # hip or shoulder (numpy array)
        joint_pos,  # knee or elbow (numpy array)
        target_pos,  # foot or hand target (numpy array)
        upper_len: float,
        lower_len: float,
        char_forward,  # character forward direction
        is_leg: bool = True
    ):
        """
        Verify IK result with comprehensive diagnostics.

        Logs:
        - Expected vs actual end effector position
        - Bend direction (correct/wrong)
        - Joint angle
        - Error classification
        """
        import numpy as np

        # Compute expected end effector from joint position
        joint_to_target = target_pos - joint_pos
        joint_to_target_dist = np.linalg.norm(joint_to_target)

        if joint_to_target_dist > 0.001:
            lower_dir = joint_to_target / joint_to_target_dist
        else:
            lower_dir = np.array([0, 0, -1], dtype=np.float32)

        # Expected end effector = joint + lower_bone_direction * lower_length
        expected_end = joint_pos + lower_dir * lower_len

        # Error from target
        error_vec = expected_end - target_pos
        error_dist = np.linalg.norm(error_vec) * 100  # cm

        # Error classification
        if error_dist < 1.0:
            error_status = "OK"
        elif error_dist < 5.0:
            error_status = "WARN"
        else:
            error_status = "FAIL"

        # Bend direction verification
        # For legs: knee should be IN FRONT of hip-foot line (positive Y in char space)
        # For arms: elbow should be BEHIND shoulder-hand line (negative Y in char space)

        # Vector from root to target (the "reach line")
        reach_vec = target_pos - root_pos
        reach_len = np.linalg.norm(reach_vec)
        if reach_len > 0.001:
            reach_dir = reach_vec / reach_len
        else:
            reach_dir = np.array([0, 0, -1], dtype=np.float32)

        # Vector from root to joint
        root_to_joint = joint_pos - root_pos

        # Project joint onto reach line to find "inline" position
        inline_dist = np.dot(root_to_joint, reach_dir)
        inline_pos = root_pos + reach_dir * inline_dist

        # Offset from inline (perpendicular to reach line)
        offset_vec = joint_pos - inline_pos
        offset_dist = np.linalg.norm(offset_vec)

        # Check if offset is in correct direction relative to character
        offset_forward = np.dot(offset_vec, char_forward)  # positive = forward

        if is_leg:
            # Knees should bend FORWARD (positive offset_forward)
            bend_correct = offset_forward > 0
            bend_dir = "FORWARD" if offset_forward > 0 else "BACKWARD"
            expected_dir = "forward"
        else:
            # Elbows should bend BACKWARD (negative offset_forward)
            bend_correct = offset_forward < 0
            bend_dir = "BACKWARD" if offset_forward < 0 else "FORWARD"
            expected_dir = "backward"

        bend_status = "CORRECT" if bend_correct else "WRONG!"

        # For arms: ADDITIONAL check - elbow must be BEHIND shoulder in ABSOLUTE character space
        # This catches cross-body reaching where elbow can be "backward relative to reach line"
        # but still forward of the shoulder in character space
        abs_position_ok = True
        abs_forward_offset = 0.0
        if not is_leg:
            # How far forward/backward is elbow relative to shoulder (not reach line)?
            shoulder_to_elbow = joint_pos - root_pos
            abs_forward_offset = float(np.dot(shoulder_to_elbow, char_forward))
            abs_position_ok = abs_forward_offset <= 0.01  # Elbow at or behind shoulder

        # Joint angle (how bent is the limb?)
        # angle = 180 - arccos((a² + b² - c²) / 2ab) where c = reach distance
        a, b, c = upper_len, lower_len, min(reach_len, upper_len + lower_len - 0.001)
        cos_angle = (a*a + b*b - c*c) / (2*a*b)
        cos_angle = np.clip(cos_angle, -1, 1)
        joint_angle = 180 - np.degrees(np.arccos(cos_angle))

        # Log everything
        self._log_solve_step(limb_name, f"joint=({joint_pos[0]:.3f}, {joint_pos[1]:.3f}, {joint_pos[2]:.3f})")
        self._log_solve_step(limb_name, f"expected_end=({expected_end[0]:.3f}, {expected_end[1]:.3f}, {expected_end[2]:.3f})")
        self._log_solve_step(limb_name, f"target=({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})")
        self._log_solve_step(limb_name, f"ERROR: {error_dist:.1f}cm [{error_status}]")
        self._log_solve_step(limb_name, f"BEND: {bend_dir} (expected {expected_dir}) [{bend_status}]")
        self._log_solve_step(limb_name, f"ANGLE: {joint_angle:.1f}° (0°=straight, 180°=fully bent)")
        self._log_solve_step(limb_name, f"offset_from_line={offset_dist*100:.1f}cm")

        # For arms: log absolute position check
        if not is_leg:
            abs_status = "✓OK" if abs_position_ok else "✗FORWARD!"
            self._log_solve_step(limb_name, f"ABS_POSITION: elbow {abs_forward_offset*100:.1f}cm {'behind' if abs_forward_offset <= 0 else 'IN FRONT of'} shoulder [{abs_status}]")

    def _log_after_state(self, before: FullBodyState, after: FullBodyState, result: FullBodyResult):
        """Log complete state AFTER solving with delta and error analysis."""
        _force_log( "-" * 64)
        _force_log( "AFTER STATE:")

        # Hips delta
        hips_delta = tuple(after.hips_pos[i] - before.hips_pos[i] for i in range(3))
        _force_log( f"  Hips:   rel=({after.hips_pos[0]:.3f}, {after.hips_pos[1]:.3f}, {after.hips_pos[2]:.3f})")
        _force_log( f"          MOVED_BY=({hips_delta[0]:.3f}, {hips_delta[1]:.3f}, {hips_delta[2]:.3f})")

        # Feet with error status
        if self.constraints.left_foot and self.constraints.left_foot.enabled:
            err = result.left_foot_error_cm
            status = "OK" if err < 1.0 else ("WARN" if err < 5.0 else "FAIL")
            _force_log( f"  L_foot: rel=({after.left_foot_pos[0]:.3f}, {after.left_foot_pos[1]:.3f}, {after.left_foot_pos[2]:.3f})  ERROR={err:.1f}cm [{status}]")

        if self.constraints.right_foot and self.constraints.right_foot.enabled:
            err = result.right_foot_error_cm
            status = "OK" if err < 1.0 else ("WARN" if err < 5.0 else "FAIL")
            _force_log( f"  R_foot: rel=({after.right_foot_pos[0]:.3f}, {after.right_foot_pos[1]:.3f}, {after.right_foot_pos[2]:.3f})  ERROR={err:.1f}cm [{status}]")

        # Hands with error status
        if self.constraints.left_hand and self.constraints.left_hand.enabled:
            err = result.left_hand_error_cm
            status = "OK" if err < 3.0 else ("AT_LIMIT" if err < 10.0 else "OUT_OF_REACH")
            _force_log( f"  L_hand: rel=({after.left_hand_pos[0]:.3f}, {after.left_hand_pos[1]:.3f}, {after.left_hand_pos[2]:.3f})  ERROR={err:.1f}cm [{status}]")

        if self.constraints.right_hand and self.constraints.right_hand.enabled:
            err = result.right_hand_error_cm
            status = "OK" if err < 3.0 else ("AT_LIMIT" if err < 10.0 else "OUT_OF_REACH")
            _force_log( f"  R_hand: rel=({after.right_hand_pos[0]:.3f}, {after.right_hand_pos[1]:.3f}, {after.right_hand_pos[2]:.3f})  ERROR={err:.1f}cm [{status}]")

        # Spine lean change
        lean_delta = after.spine_lean - before.spine_lean
        _force_log( f"  Spine:  lean={after.spine_lean:.1f}deg  CHANGED_BY={lean_delta:+.1f}deg")

    def _log_result(self, result: FullBodyResult):
        """Log final result summary."""
        _force_log( "-" * 64)

        # Collect all errors
        errors = []
        error_threshold = 5.0  # cm - same as FAIL threshold in _verify_limb_ik
        if result.left_foot_error_cm > 0:
            errors.append(("L_FOOT", result.left_foot_error_cm))
        if result.right_foot_error_cm > 0:
            errors.append(("R_FOOT", result.right_foot_error_cm))
        if result.left_hand_error_cm > 0:
            errors.append(("L_HAND", result.left_hand_error_cm))
        if result.right_hand_error_cm > 0:
            errors.append(("R_HAND", result.right_hand_error_cm))

        # Any error >= 5cm is a FAILURE, regardless of "satisfied" count
        max_error = max((e[1] for e in errors), default=0.0)
        has_failed_target = max_error >= error_threshold

        # Determine overall status based on ACTUAL results
        if has_failed_target:
            status = "FAILED"  # Cannot call it success if targets are missed by > 5cm
        elif result.success and result.constraints_satisfied == result.constraints_total:
            status = "SUCCESS"
        elif result.constraints_satisfied > 0:
            status = "PARTIAL"
        else:
            status = "FAILED"

        _force_log( f"RESULT: {status}")
        _force_log( f"  Constraints: {result.constraints_satisfied}/{result.constraints_total} satisfied")
        _force_log( f"  At limit:    {result.constraints_at_limit}")
        _force_log( f"  Violations:  {len(result.joint_violations)} joints")
        _force_log( f"  Solve time:  {result.solve_time_us:.0f}us")
        _force_log( f"  Bones moved: {len(result.bone_transforms)}")

        # Log each target's error
        for name, err in errors:
            if err >= error_threshold:
                _force_log( f"  {name}: {err:.1f}cm <<<MISSED TARGET>>>")
            elif err >= 1.0:
                _force_log( f"  {name}: {err:.1f}cm (warn)")
            else:
                _force_log( f"  {name}: {err:.1f}cm (ok)")

        if result.joint_violations:
            _force_log( f"  Violated:    {', '.join(result.joint_violations[:5])}")

        _force_log( "=" * 64)

    def _diagnose_ik_result(self, before_bones: dict, after_bones: dict, constraints: 'FullBodyConstraints'):
        """
        ALWAYS-ON diagnostic that compares actual Blender bone positions.

        This reads REAL positions from Blender, not computed values.
        Identifies exactly what went wrong.
        """
        import numpy as np

        _force_log("=" * 70)
        _force_log("IK DIAGNOSTIC - ACTUAL BONE POSITIONS (from Blender)")
        _force_log("=" * 70)

        # Get root for relative calculations
        root_pos = np.array(before_bones.get("Root", (0, 0, 0)), dtype=np.float32)

        problems = []

        # Check each constrained effector
        if constraints.left_foot and constraints.left_foot.enabled:
            target = np.array(constraints.left_foot.position, dtype=np.float32)
            before = np.array(before_bones.get("LeftFoot", (0, 0, 0)), dtype=np.float32) - root_pos
            after = np.array(after_bones.get("LeftFoot", (0, 0, 0)), dtype=np.float32) - root_pos

            error_before = np.linalg.norm(before - target) * 100
            error_after = np.linalg.norm(after - target) * 100
            moved = np.linalg.norm(after - before) * 100

            _force_log(f"LEFT FOOT:")
            _force_log(f"  Target:     ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})")
            _force_log(f"  Before:     ({before[0]:.3f}, {before[1]:.3f}, {before[2]:.3f})  error={error_before:.1f}cm")
            _force_log(f"  After:      ({after[0]:.3f}, {after[1]:.3f}, {after[2]:.3f})  error={error_after:.1f}cm")
            _force_log(f"  Movement:   {moved:.1f}cm")

            if error_after > 5:
                problems.append(f"L_FOOT: {error_after:.1f}cm from target")
            if error_after > error_before:
                problems.append(f"L_FOOT: got WORSE ({error_before:.1f}cm -> {error_after:.1f}cm)")

        if constraints.right_foot and constraints.right_foot.enabled:
            target = np.array(constraints.right_foot.position, dtype=np.float32)
            before = np.array(before_bones.get("RightFoot", (0, 0, 0)), dtype=np.float32) - root_pos
            after = np.array(after_bones.get("RightFoot", (0, 0, 0)), dtype=np.float32) - root_pos

            error_before = np.linalg.norm(before - target) * 100
            error_after = np.linalg.norm(after - target) * 100
            moved = np.linalg.norm(after - before) * 100

            _force_log(f"RIGHT FOOT:")
            _force_log(f"  Target:     ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})")
            _force_log(f"  Before:     ({before[0]:.3f}, {before[1]:.3f}, {before[2]:.3f})  error={error_before:.1f}cm")
            _force_log(f"  After:      ({after[0]:.3f}, {after[1]:.3f}, {after[2]:.3f})  error={error_after:.1f}cm")
            _force_log(f"  Movement:   {moved:.1f}cm")

            if error_after > 5:
                problems.append(f"R_FOOT: {error_after:.1f}cm from target")
            if error_after > error_before:
                problems.append(f"R_FOOT: got WORSE ({error_before:.1f}cm -> {error_after:.1f}cm)")

        # Check knee positions (bend direction)
        if "LeftShin" in after_bones and "LeftThigh" in after_bones:
            hip = np.array(after_bones["LeftThigh"], dtype=np.float32)
            knee = np.array(after_bones["LeftShin"], dtype=np.float32)
            foot = np.array(after_bones.get("LeftFoot", knee), dtype=np.float32)

            # Knee should be IN FRONT of hip-foot line (positive Y)
            hip_to_foot = foot - hip
            hip_to_knee = knee - hip

            # Project knee onto hip-foot line
            hip_foot_len = np.linalg.norm(hip_to_foot)
            if hip_foot_len > 0.001:
                hip_foot_dir = hip_to_foot / hip_foot_len
                inline_dist = np.dot(hip_to_knee, hip_foot_dir)
                inline_pos = hip + hip_foot_dir * inline_dist
                knee_offset = knee - inline_pos

                # Check if knee is forward (positive Y in world)
                knee_forward = knee_offset[1]  # Y component

                _force_log(f"LEFT KNEE BEND:")
                _force_log(f"  Knee pos:   ({knee[0]:.3f}, {knee[1]:.3f}, {knee[2]:.3f})")
                _force_log(f"  Offset Y:   {knee_forward:.3f}m ({'FORWARD' if knee_forward > 0 else 'BACKWARD'})")

                if knee_forward < 0:
                    problems.append("L_KNEE: bending BACKWARD (should be forward)")

        if "RightShin" in after_bones and "RightThigh" in after_bones:
            hip = np.array(after_bones["RightThigh"], dtype=np.float32)
            knee = np.array(after_bones["RightShin"], dtype=np.float32)
            foot = np.array(after_bones.get("RightFoot", knee), dtype=np.float32)

            hip_to_foot = foot - hip
            hip_to_knee = knee - hip

            hip_foot_len = np.linalg.norm(hip_to_foot)
            if hip_foot_len > 0.001:
                hip_foot_dir = hip_to_foot / hip_foot_len
                inline_dist = np.dot(hip_to_knee, hip_foot_dir)
                inline_pos = hip + hip_foot_dir * inline_dist
                knee_offset = knee - inline_pos

                knee_forward = knee_offset[1]

                _force_log(f"RIGHT KNEE BEND:")
                _force_log(f"  Knee pos:   ({knee[0]:.3f}, {knee[1]:.3f}, {knee[2]:.3f})")
                _force_log(f"  Offset Y:   {knee_forward:.3f}m ({'FORWARD' if knee_forward > 0 else 'BACKWARD'})")

                if knee_forward < 0:
                    problems.append("R_KNEE: bending BACKWARD (should be forward)")

        # Check hands (similar to feet)
        if constraints.left_hand and constraints.left_hand.enabled:
            target = np.array(constraints.left_hand.position, dtype=np.float32)
            before = np.array(before_bones.get("LeftHand", (0, 0, 0)), dtype=np.float32) - root_pos
            after = np.array(after_bones.get("LeftHand", (0, 0, 0)), dtype=np.float32) - root_pos

            error_before = np.linalg.norm(before - target) * 100
            error_after = np.linalg.norm(after - target) * 100
            moved = np.linalg.norm(after - before) * 100

            _force_log(f"LEFT HAND:")
            _force_log(f"  Target:     ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})")
            _force_log(f"  Before:     ({before[0]:.3f}, {before[1]:.3f}, {before[2]:.3f})  error={error_before:.1f}cm")
            _force_log(f"  After:      ({after[0]:.3f}, {after[1]:.3f}, {after[2]:.3f})  error={error_after:.1f}cm")
            _force_log(f"  Movement:   {moved:.1f}cm")

            if error_after > 5:
                problems.append(f"L_HAND: {error_after:.1f}cm from target")
            if moved < 0.5 and error_before > 5:
                problems.append(f"L_HAND: DIDN'T MOVE (was {error_before:.1f}cm away)")

        if constraints.right_hand and constraints.right_hand.enabled:
            target = np.array(constraints.right_hand.position, dtype=np.float32)
            before = np.array(before_bones.get("RightHand", (0, 0, 0)), dtype=np.float32) - root_pos
            after = np.array(after_bones.get("RightHand", (0, 0, 0)), dtype=np.float32) - root_pos

            error_before = np.linalg.norm(before - target) * 100
            error_after = np.linalg.norm(after - target) * 100
            moved = np.linalg.norm(after - before) * 100

            _force_log(f"RIGHT HAND:")
            _force_log(f"  Target:     ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})")
            _force_log(f"  Before:     ({before[0]:.3f}, {before[1]:.3f}, {before[2]:.3f})  error={error_before:.1f}cm")
            _force_log(f"  After:      ({after[0]:.3f}, {after[1]:.3f}, {after[2]:.3f})  error={error_after:.1f}cm")
            _force_log(f"  Movement:   {moved:.1f}cm")

            if error_after > 5:
                problems.append(f"R_HAND: {error_after:.1f}cm from target")
            if moved < 0.5 and error_before > 5:
                problems.append(f"R_HAND: DIDN'T MOVE (was {error_before:.1f}cm away)")

        # Check elbow positions (bend direction)
        if "LeftForeArm" in after_bones and "LeftArm" in after_bones:
            shoulder = np.array(after_bones["LeftArm"], dtype=np.float32)
            elbow = np.array(after_bones["LeftForeArm"], dtype=np.float32)
            hand = np.array(after_bones.get("LeftHand", elbow), dtype=np.float32)

            # Elbow should be BEHIND shoulder-hand line (negative Y in character space)
            shoulder_to_hand = hand - shoulder
            shoulder_to_elbow = elbow - shoulder

            sh_len = np.linalg.norm(shoulder_to_hand)
            if sh_len > 0.001:
                sh_dir = shoulder_to_hand / sh_len
                inline_dist = np.dot(shoulder_to_elbow, sh_dir)
                inline_pos = shoulder + sh_dir * inline_dist
                elbow_offset = elbow - inline_pos

                # In character space, Y is forward, so elbow should have negative Y offset
                elbow_back = -elbow_offset[1]  # negative Y = backward

                _force_log(f"LEFT ELBOW BEND:")
                _force_log(f"  Elbow pos:  ({elbow[0]:.3f}, {elbow[1]:.3f}, {elbow[2]:.3f})")
                _force_log(f"  Offset Y:   {elbow_offset[1]:.3f}m ({'BACKWARD' if elbow_back > 0 else 'FORWARD'})")

                if elbow_back < 0:
                    problems.append("L_ELBOW: bending FORWARD (should be backward)")

        if "RightForeArm" in after_bones and "RightArm" in after_bones:
            shoulder = np.array(after_bones["RightArm"], dtype=np.float32)
            elbow = np.array(after_bones["RightForeArm"], dtype=np.float32)
            hand = np.array(after_bones.get("RightHand", elbow), dtype=np.float32)

            shoulder_to_hand = hand - shoulder
            shoulder_to_elbow = elbow - shoulder

            sh_len = np.linalg.norm(shoulder_to_hand)
            if sh_len > 0.001:
                sh_dir = shoulder_to_hand / sh_len
                inline_dist = np.dot(shoulder_to_elbow, sh_dir)
                inline_pos = shoulder + sh_dir * inline_dist
                elbow_offset = elbow - inline_pos

                elbow_back = -elbow_offset[1]

                _force_log(f"RIGHT ELBOW BEND:")
                _force_log(f"  Elbow pos:  ({elbow[0]:.3f}, {elbow[1]:.3f}, {elbow[2]:.3f})")
                _force_log(f"  Offset Y:   {elbow_offset[1]:.3f}m ({'BACKWARD' if elbow_back > 0 else 'FORWARD'})")

                if elbow_back < 0:
                    problems.append("R_ELBOW: bending FORWARD (should be backward)")

        # Check hips
        if "Hips" in before_bones and "Hips" in after_bones:
            hips_before = np.array(before_bones["Hips"], dtype=np.float32)
            hips_after = np.array(after_bones["Hips"], dtype=np.float32)
            hips_moved = hips_after - hips_before

            _force_log(f"HIPS:")
            _force_log(f"  Before:     ({hips_before[0]:.3f}, {hips_before[1]:.3f}, {hips_before[2]:.3f})")
            _force_log(f"  After:      ({hips_after[0]:.3f}, {hips_after[1]:.3f}, {hips_after[2]:.3f})")
            _force_log(f"  Movement:   ({hips_moved[0]:.3f}, {hips_moved[1]:.3f}, {hips_moved[2]:.3f})")

        # ASCII arm visualization (top-down view, +Y is forward, +X is right)
        _force_log("-" * 70)
        _force_log("TOP-DOWN VIEW (looking down at character):")
        _force_log("  +Y = forward, +X = right, character faces up (↑)")

        # Create a simple grid
        grid_size = 11
        grid = [[' ' for _ in range(grid_size * 2)] for _ in range(grid_size)]
        center_x = grid_size  # X center
        center_y = grid_size // 2  # Y center

        def plot(x, y, char):
            # Convert world coords to grid (scale: 1 grid = 0.1m)
            gx = int(center_x + x * 10)
            gy = int(center_y - y * 10)  # Y inverted for display
            if 0 <= gx < grid_size * 2 and 0 <= gy < grid_size:
                grid[gy][gx] = char

        # Plot spine/hips as reference
        plot(0, 0, 'O')  # Character center

        # Plot arms
        if "LeftArm" in after_bones:
            shoulder = after_bones["LeftArm"]
            plot(shoulder[0], shoulder[1], 'L')
        if "RightArm" in after_bones:
            shoulder = after_bones["RightArm"]
            plot(shoulder[0], shoulder[1], 'R')
        if "LeftForeArm" in after_bones:
            elbow = after_bones["LeftForeArm"]
            plot(elbow[0], elbow[1], 'l')
        if "RightForeArm" in after_bones:
            elbow = after_bones["RightForeArm"]
            plot(elbow[0], elbow[1], 'r')
        if "LeftHand" in after_bones:
            hand = after_bones["LeftHand"]
            plot(hand[0], hand[1], '◄')
        if "RightHand" in after_bones:
            hand = after_bones["RightHand"]
            plot(hand[0], hand[1], '►')

        # Plot target
        if self.constraints.left_hand and self.constraints.left_hand.enabled:
            target = self.constraints.left_hand.position
            plot(target[0], target[1], '*')

        # Print top-down grid
        _force_log("     " + "".join(['-' for _ in range(grid_size * 2)]))
        for row in grid:
            _force_log("     |" + "".join(row) + "|")
        _force_log("     " + "".join(['-' for _ in range(grid_size * 2)]))
        _force_log("  Legend: O=center L/R=shoulders l/r=elbows ◄/►=hands *=target")

        # =====================================================================
        # SIDE VIEW (Y vs Z) - Shows if elbows bend FORWARD or BACKWARD
        # =====================================================================
        _force_log("-" * 70)
        _force_log("SIDE VIEW (looking from LEFT, character faces RIGHT →):")
        _force_log("  +Y = forward (→), +Z = up (↑)")

        # Create side view grid
        side_h = 12  # Height (Z)
        side_w = 20  # Width (Y)
        side_grid = [[' ' for _ in range(side_w)] for _ in range(side_h)]

        def plot_side(y, z, char):
            # Y: 0 = back, side_w = forward
            # Z: 0 = floor, side_h = head height
            gy = int(side_w / 2 + y * 10)  # Y scaled by 10, centered
            gz = side_h - 1 - int(z * 5)   # Z scaled by 5, inverted
            if 0 <= gy < side_w and 0 <= gz < side_h:
                side_grid[gz][gy] = char

        # Plot body reference (spine at y=0.05ish)
        plot_side(0.05, 1.0, '┃')  # Hips
        plot_side(0.05, 1.3, '┃')  # Spine
        plot_side(0.05, 1.6, 'O')  # Shoulders
        plot_side(0.05, 1.9, '○')  # Head

        # Plot left arm (from side, just show Y and Z)
        if "LeftArm" in after_bones:
            shoulder = after_bones["LeftArm"]
            plot_side(shoulder[1], shoulder[2], 'S')
        if "LeftForeArm" in after_bones:
            elbow = after_bones["LeftForeArm"]
            plot_side(elbow[1], elbow[2], 'E')
            # Mark if elbow is FORWARD or BACK of shoulder
            if "LeftArm" in after_bones:
                shoulder_y = after_bones["LeftArm"][1]
                if elbow[1] > shoulder_y + 0.02:
                    elbow_pos = "FWD"
                elif elbow[1] < shoulder_y - 0.02:
                    elbow_pos = "BCK"
                else:
                    elbow_pos = "---"
        if "LeftHand" in after_bones:
            hand = after_bones["LeftHand"]
            plot_side(hand[1], hand[2], 'H')

        # Plot target
        if self.constraints.left_hand and self.constraints.left_hand.enabled:
            target = self.constraints.left_hand.position
            plot_side(target[1], target[2], '*')

        # Draw floor line
        for i in range(side_w):
            if side_grid[side_h - 1][i] == ' ':
                side_grid[side_h - 1][i] = '_'

        # Print side grid with labels
        _force_log("     ↑Z")
        _force_log("     " + "".join(['-' for _ in range(side_w + 2)]))
        for i, row in enumerate(side_grid):
            if i == 0:
                _force_log("     |" + "".join(row) + "| head")
            elif i == side_h - 1:
                _force_log("  →Y |" + "".join(row) + "| floor")
            else:
                _force_log("     |" + "".join(row) + "|")
        _force_log("     " + "".join(['-' for _ in range(side_w + 2)]))
        _force_log("      ←BACK        FORWARD→")
        _force_log("  Legend: S=shoulder E=elbow H=hand *=target O=chest ○=head")

        # Print elbow position summary
        if "LeftForeArm" in after_bones and "LeftArm" in after_bones:
            l_shoulder_y = after_bones["LeftArm"][1]
            l_elbow_y = after_bones["LeftForeArm"][1]
            l_offset = l_elbow_y - l_shoulder_y
            l_status = "✓BACKWARD" if l_offset < -0.02 else "❌FORWARD" if l_offset > 0.02 else "NEUTRAL"
            _force_log(f"  L_ELBOW: Y_offset={l_offset:.3f}m ({l_status})")
        if "RightForeArm" in after_bones and "RightArm" in after_bones:
            r_shoulder_y = after_bones["RightArm"][1]
            r_elbow_y = after_bones["RightForeArm"][1]
            r_offset = r_elbow_y - r_shoulder_y
            r_status = "✓BACKWARD" if r_offset < -0.02 else "❌FORWARD" if r_offset > 0.02 else "NEUTRAL"
            _force_log(f"  R_ELBOW: Y_offset={r_offset:.3f}m ({r_status})")

        # Summary
        _force_log("-" * 70)
        if problems:
            _force_log("PROBLEMS DETECTED:")
            for p in problems:
                _force_log(f"  ❌ {p}")
        else:
            _force_log("✓ No obvious problems detected")
        _force_log("=" * 70)

    def _capture_bone_positions(self) -> dict:
        """Capture current world positions of key bones from Blender."""
        positions = {}
        pose_bones = self.armature.pose.bones
        arm_matrix = self.armature.matrix_world

        key_bones = [
            "Root", "Hips",
            "LeftThigh", "LeftShin", "LeftFoot",
            "RightThigh", "RightShin", "RightFoot",
            "LeftArm", "LeftForeArm", "LeftHand",
            "RightArm", "RightForeArm", "RightHand",
            "Head"
        ]

        for bone_name in key_bones:
            bone = pose_bones.get(bone_name)
            if bone:
                world_pos = arm_matrix @ bone.head
                positions[bone_name] = (world_pos.x, world_pos.y, world_pos.z)

        return positions

    # =========================================================================
    # BONE ROTATION COMPUTATION
    # =========================================================================

    def _compute_bone_rotation_to_direction(
        self,
        bone: bpy.types.PoseBone,
        target_dir: Vector,
        arm_matrix: Matrix
    ) -> Quaternion:
        """
        Compute the LOCAL rotation quaternion to make a bone point in target_dir.

        IDEMPOTENT: Always computes from REST pose, not current pose.
        Multiple calls with same target_dir produce same result.

        CRITICAL: rotation_quaternion is relative to PARENT, not armature!
        We must compute the rotation in the bone's parent-local frame.

        Args:
            bone: The pose bone to rotate
            target_dir: World-space direction the bone should point (normalized)
            arm_matrix: Armature's world matrix

        Returns:
            Local quaternion to set on bone.rotation_quaternion
        """
        target_dir = target_dir.normalized()

        # =================================================================
        # STEP 1: Get bone's rest Y in world space
        # bone.bone.matrix_local is rest pose in ARMATURE space
        # =================================================================
        rest_matrix = bone.bone.matrix_local
        rest_y_armature = rest_matrix.to_3x3() @ Vector((0, 1, 0))
        rest_y_world = (arm_matrix.to_3x3() @ rest_y_armature).normalized()

        # =================================================================
        # STEP 2: Compute world rotation from rest_y to target
        # =================================================================
        world_rotation = rest_y_world.rotation_difference(target_dir)

        # Angle for logging
        dot = rest_y_world.dot(target_dir)
        angle_deg = math.degrees(math.acos(max(-1, min(1, dot))))

        # =================================================================
        # STEP 3: Convert to PARENT-LOCAL frame
        # rotation_quaternion is applied relative to parent's CURRENT pose!
        # =================================================================
        if bone.parent:
            # Get parent's CURRENT posed world orientation (NOT rest!)
            # bone.parent.matrix is the posed matrix in armature space
            parent_posed_world = arm_matrix @ bone.parent.matrix
            parent_posed_quat = parent_posed_world.to_quaternion()

            # Bone's rest pose relative to parent's rest (for local frame transform)
            parent_rest = bone.parent.bone.matrix_local
            bone_rest_in_parent = parent_rest.inverted() @ rest_matrix
            bone_local_rest_quat = bone_rest_in_parent.to_quaternion()

            # Transform world rotation to parent's CURRENT frame
            parent_inv = parent_posed_quat.inverted()
            rotation_in_parent = parent_inv @ world_rotation @ parent_posed_quat

            # Express in bone's local coordinate system
            bone_inv = bone_local_rest_quat.inverted()
            local_quat = bone_inv @ rotation_in_parent @ bone_local_rest_quat
        else:
            # No parent - just convert to armature frame
            arm_quat = arm_matrix.to_quaternion()
            arm_inv = arm_quat.inverted()
            rotation_in_arm = arm_inv @ world_rotation @ arm_quat

            bone_rest_quat = rest_matrix.to_quaternion()
            bone_inv = bone_rest_quat.inverted()
            local_quat = bone_inv @ rotation_in_arm @ bone_rest_quat

        local_quat.normalize()

        # =================================================================
        # DEBUG: Detailed logging
        # =================================================================
        _force_log(f"  [ROT] {bone.name}:")
        _force_log(f"    rest_Y_world=({rest_y_world.x:.2f}, {rest_y_world.y:.2f}, {rest_y_world.z:.2f})")
        _force_log(f"    target_dir=({target_dir.x:.2f}, {target_dir.y:.2f}, {target_dir.z:.2f})")
        _force_log(f"    angle_between={angle_deg:.1f}°")
        _force_log(f"    computed_local=({local_quat.w:.3f}, {local_quat.x:.3f}, {local_quat.y:.3f}, {local_quat.z:.3f})")

        # Verify: what direction will this rotation produce?
        if bone.parent:
            parent_posed_world = arm_matrix @ bone.parent.matrix
            parent_posed_quat = parent_posed_world.to_quaternion()
            parent_rest = bone.parent.bone.matrix_local
            bone_rest_in_parent = parent_rest.inverted() @ rest_matrix
            bone_local_rest_quat = bone_rest_in_parent.to_quaternion()

            verify_quat = bone_local_rest_quat @ local_quat
            verify_y = verify_quat @ Vector((0, 1, 0))
            verify_y_world = parent_posed_quat @ verify_y
        else:
            verify_quat = rest_matrix.to_quaternion() @ local_quat
            verify_y = verify_quat @ Vector((0, 1, 0))
            verify_y_world = arm_matrix.to_quaternion() @ verify_y
        verify_y_world.normalize()

        result_dot = verify_y_world.dot(target_dir)
        result_angle = math.degrees(math.acos(max(-1, min(1, result_dot))))
        _force_log(f"    verify_Y_world=({verify_y_world.x:.2f}, {verify_y_world.y:.2f}, {verify_y_world.z:.2f})")
        _force_log(f"    verify_error={result_angle:.1f}° {'✓OK' if result_angle < 1.0 else '❌WRONG'}")

        return local_quat

    def _compute_child_rotation_for_ik(
        self,
        child_bone: bpy.types.PoseBone,
        target_dir: Vector,
        parent_local_quat: Quaternion,
        arm_matrix: Matrix
    ) -> Quaternion:
        """
        Compute child bone rotation for IK, accounting for parent's rotation.

        In IK chains, after the parent rotates, the child's starting direction
        changes. We need to compute the child's rotation from its NEW position
        (after parent rotates), not from rest.

        Args:
            child_bone: The child pose bone (e.g., LeftForeArm)
            target_dir: World-space direction the child should point
            parent_local_quat: The LOCAL quaternion we're applying to the parent
            arm_matrix: Armature's world matrix

        Returns:
            Local quaternion for the child bone
        """
        target_dir = target_dir.normalized()
        parent_bone = child_bone.parent

        if not parent_bone:
            # No parent - use regular method
            return self._compute_bone_rotation_to_direction(child_bone, target_dir, arm_matrix)

        # =================================================================
        # Compute parent's NEW world orientation after applying parent_local_quat
        # Must account for grandparent's CURRENT pose!
        # =================================================================
        if parent_bone.parent:
            # Get grandparent's current posed world orientation
            grandparent_world = arm_matrix @ parent_bone.parent.matrix
            grandparent_quat = grandparent_world.to_quaternion()

            # Parent's rest pose relative to grandparent's rest
            grandparent_rest = parent_bone.parent.bone.matrix_local
            parent_rest_in_grandparent = grandparent_rest.inverted() @ parent_bone.bone.matrix_local
            parent_local_rest_quat = parent_rest_in_grandparent.to_quaternion()

            # Parent's new world = grandparent_current @ parent_rest_in_grandparent @ parent_local_quat
            parent_new_quat = grandparent_quat @ parent_local_rest_quat @ parent_local_quat
        else:
            # Parent has no grandparent - use armature
            parent_rest_world = arm_matrix @ parent_bone.bone.matrix_local
            parent_rest_quat = parent_rest_world.to_quaternion()
            parent_new_quat = parent_rest_quat @ parent_local_quat

        # Child's rest in parent space
        child_rest = child_bone.bone.matrix_local
        parent_rest_mat = parent_bone.bone.matrix_local
        child_rest_in_parent = parent_rest_mat.inverted() @ child_rest

        # Child's Y axis in parent's local space
        child_y_in_parent = child_rest_in_parent.to_3x3() @ Vector((0, 1, 0))

        # After parent rotates, child's Y axis in world becomes:
        child_y_world_after = parent_new_quat @ child_y_in_parent
        child_y_world_after.normalize()

        # Rotation from child's new Y to target direction (in world space)
        world_rotation = child_y_world_after.rotation_difference(target_dir)

        # Check angle
        dot = child_y_world_after.dot(target_dir)
        angle_deg = math.degrees(math.acos(max(-1, min(1, dot))))

        # Convert world rotation to child's local space
        # Child's new world orientation (after parent rotates, before child's own rotation)
        child_rest_in_parent_quat = child_rest_in_parent.to_quaternion()
        child_new_world_quat = parent_new_quat @ child_rest_in_parent_quat

        # Local rotation = child_world⁻¹ @ world_rotation @ child_world
        child_new_inv = child_new_world_quat.inverted()
        local_quat = child_new_inv @ world_rotation @ child_new_world_quat

        local_quat.normalize()

        # Verify: what direction will this rotation produce?
        final_quat = child_new_world_quat @ local_quat
        verify_y_world = final_quat @ Vector((0, 1, 0))
        verify_y_world.normalize()
        result_dot = verify_y_world.dot(target_dir)
        result_angle = math.degrees(math.acos(max(-1, min(1, result_dot))))

        # Logging
        _force_log(f"  [ROT-CHILD] {child_bone.name}:")
        _force_log(f"    child_Y_after_parent=({child_y_world_after.x:.2f}, {child_y_world_after.y:.2f}, {child_y_world_after.z:.2f})")
        _force_log(f"    target_dir=({target_dir.x:.2f}, {target_dir.y:.2f}, {target_dir.z:.2f})")
        _force_log(f"    angle_between={angle_deg:.1f}°")
        _force_log(f"    computed_local=({local_quat.w:.3f}, {local_quat.x:.3f}, {local_quat.y:.3f}, {local_quat.z:.3f})")
        _force_log(f"    verify_Y_world=({verify_y_world.x:.2f}, {verify_y_world.y:.2f}, {verify_y_world.z:.2f})")
        _force_log(f"    verify_error={result_angle:.1f}° {'✓OK' if result_angle < 1.0 else '❌WRONG'}")

        return local_quat

    # =========================================================================
    # CROSS-BODY FIX - Find anatomically valid elbow position
    # =========================================================================

    def _fix_cross_body_elbow(
        self,
        shoulder_pos: 'np.ndarray',
        target_pos: 'np.ndarray',
        bad_elbow_pos: 'np.ndarray',
        bad_upper_dir: 'np.ndarray',
        side: str,
        char_forward: 'np.ndarray',
        char_up: 'np.ndarray',
        upper_len: float,
        lower_len: float,
    ) -> tuple:
        """
        Fix an anatomically impossible elbow position.

        CONSTRAINTS:
        - Left elbow X must be <= 0 (left side)
        - Right elbow X must be >= 0 (right side)
        - Elbow Y must be <= shoulder Y (behind or at shoulder)
        - Elbow must satisfy: |elbow - shoulder| = upper_len
        - Elbow must satisfy: |target - elbow| <= lower_len

        STRATEGY:
        The valid elbow positions form a CIRCLE in 3D space (intersection of two spheres).
        Sample points around this circle and find the one that satisfies anatomical constraints.

        Args:
            shoulder_pos: Shoulder world position
            target_pos: Hand target world position
            bad_elbow_pos: The invalid computed elbow position
            bad_upper_dir: The invalid computed upper arm direction
            side: "L" or "R"
            char_forward: Character forward direction
            char_up: Character up direction
            upper_len: Upper arm length
            lower_len: Forearm length

        Returns:
            Tuple of (fixed_upper_dir, fixed_lower_dir, fixed_elbow_pos)
        """
        from ..engine.animations.ik import normalize, quat_from_axis_angle, quat_rotate_vector

        shoulder_pos = np.asarray(shoulder_pos, dtype=np.float32)
        target_pos = np.asarray(target_pos, dtype=np.float32)
        char_forward = np.asarray(char_forward, dtype=np.float32)
        char_up = np.asarray(char_up, dtype=np.float32)
        char_right = np.cross(char_forward, char_up)
        char_back = -char_forward

        # Reach direction
        reach_vec = target_pos - shoulder_pos
        reach_dist = float(np.linalg.norm(reach_vec))
        if reach_dist < 0.001:
            return bad_upper_dir, normalize(target_pos - bad_elbow_pos), bad_elbow_pos

        reach_dir = reach_vec / reach_dist

        # Clamp reach to valid range
        max_reach = upper_len + lower_len - 0.001
        min_reach = abs(upper_len - lower_len) + 0.001
        clamped_dist = max(min_reach, min(reach_dist, max_reach))

        # =====================================================================
        # COMPUTE ELBOW CIRCLE GEOMETRY
        # =====================================================================
        # The elbow must be:
        #   - On sphere of radius upper_len around shoulder
        #   - On sphere of radius lower_len around hand_pos
        # The intersection of these spheres is a circle.
        #
        # Using law of cosines to find angle at shoulder:
        #   cos(angle) = (upper^2 + dist^2 - lower^2) / (2 * upper * dist)

        a, b, c = upper_len, lower_len, clamped_dist

        cos_shoulder = (a*a + c*c - b*b) / (2*a*c)
        cos_shoulder = np.clip(cos_shoulder, -1.0, 1.0)
        shoulder_angle = np.arccos(cos_shoulder)

        # The elbow circle has:
        #   - Center on the line from shoulder toward hand
        #   - Center at distance = upper_len * cos(shoulder_angle) from shoulder
        #   - Radius = upper_len * sin(shoulder_angle)

        circle_center_dist = upper_len * np.cos(shoulder_angle)
        circle_radius = upper_len * np.sin(shoulder_angle)

        circle_center = shoulder_pos + reach_dir * circle_center_dist

        _force_log(f"  [FIX] Elbow circle: center_dist={circle_center_dist:.3f}m, radius={circle_radius:.3f}m")

        # Build orthonormal basis perpendicular to reach_dir
        # We'll use this to sample points around the elbow circle
        if abs(np.dot(reach_dir, char_up)) < 0.9:
            perp1 = normalize(np.cross(reach_dir, char_up))
        else:
            perp1 = normalize(np.cross(reach_dir, char_right))
        perp2 = normalize(np.cross(reach_dir, perp1))

        # =====================================================================
        # SAMPLE ELBOW CIRCLE FOR VALID POSITIONS
        # =====================================================================
        num_samples = 72  # Every 5 degrees
        best_elbow = None
        best_upper_dir = None
        best_score = -float('inf')

        _force_log(f"  [FIX] Searching {num_samples} positions on elbow circle...")

        for i in range(num_samples):
            angle = (2.0 * np.pi * i) / num_samples

            # Point on elbow circle
            test_elbow = (circle_center +
                          circle_radius * np.cos(angle) * perp1 +
                          circle_radius * np.sin(angle) * perp2)

            # Check anatomical constraints
            elbow_x = test_elbow[0]
            elbow_y = test_elbow[1]
            shoulder_y = shoulder_pos[1]

            # Constraint 1: Elbow on correct side (RELAXED for cross-body)
            # Allow elbow to be at body center (X ~= 0) for cross-body reaches
            if side == "L":
                side_valid = elbow_x <= 0.05  # Left elbow can be at center for cross-body
                side_score = -elbow_x  # Prefer more left (more negative X)
            else:
                side_valid = elbow_x >= -0.05  # Right elbow can be at center for cross-body
                side_score = elbow_x  # Prefer more right (more positive X)

            # Constraint 2: Elbow behind shoulder (RELAXED)
            forward_offset = elbow_y - shoulder_y
            behind_valid = forward_offset <= 0.08  # 8cm tolerance for cross-body
            behind_score = -forward_offset  # Prefer more backward

            if not side_valid or not behind_valid:
                continue  # Skip invalid positions

            # Score: prefer anatomically natural positions
            # Higher side_score = elbow more on correct side
            # Higher behind_score = elbow more behind
            score = side_score * 2.0 + behind_score * 1.5

            if score > best_score:
                best_score = score
                best_elbow = test_elbow.copy()
                best_upper_dir = normalize(test_elbow - shoulder_pos)

        # If we found a valid position, use it
        if best_elbow is not None:
            _force_log(f"  [FIX] ✓ Found valid elbow at X={best_elbow[0]:.3f}, Y={best_elbow[1]:.3f}")

            # Compute lower arm direction (from elbow to clamped hand position)
            hand_pos = shoulder_pos + reach_dir * clamped_dist
            lower_dir = normalize(hand_pos - best_elbow)

            return best_upper_dir, lower_dir, best_elbow
        else:
            # No valid position found - use the BEST invalid position (most backward, most on-side)
            _force_log(f"  [FIX] ⚠️ No fully valid position - using best compromise...")

            # Find position that's most backward (least forward offset)
            best_backward = float('inf')
            best_elbow = bad_elbow_pos.copy()

            for i in range(num_samples):
                angle = (2.0 * np.pi * i) / num_samples
                test_elbow = (circle_center +
                              circle_radius * np.cos(angle) * perp1 +
                              circle_radius * np.sin(angle) * perp2)

                forward_offset = test_elbow[1] - shoulder_pos[1]

                # Also penalize being on wrong side
                if side == "L":
                    side_penalty = max(0, test_elbow[0]) * 2  # Penalty for being on right
                else:
                    side_penalty = max(0, -test_elbow[0]) * 2  # Penalty for being on left

                total_badness = forward_offset + side_penalty

                if total_badness < best_backward:
                    best_backward = total_badness
                    best_elbow = test_elbow.copy()

            best_upper_dir = normalize(best_elbow - shoulder_pos)
            hand_pos = shoulder_pos + reach_dir * clamped_dist
            lower_dir = normalize(hand_pos - best_elbow)

            _force_log(f"  [FIX] Compromise elbow at X={best_elbow[0]:.3f}, Y={best_elbow[1]:.3f}")

            return best_upper_dir, lower_dir, best_elbow

    # =========================================================================
    # SOLVING
    # =========================================================================

    def solve(self, use_engine: bool = True, region: str = "FULL_BODY") -> FullBodyResult:
        """
        Solve IK and apply to armature.

        Args:
            use_engine: If True, submit to worker. If False, solve locally.
            region: IK region being solved (for logging)

        Returns:
            FullBodyResult with solve status and metrics
        """
        self._solve_count += 1
        start_time = time.perf_counter()

        # ALWAYS capture bone positions BEFORE (for diagnostic)
        before_bones = self._capture_bone_positions()

        # Wake up visualizer if needed
        self._ensure_visualizer_active()

        # Log header
        self._log_header(region)

        # Capture BEFORE state
        before_state = self.capture_state()
        self._log_before_state(before_state)
        self._log_constraints(before_state)

        # Log solving process
        _force_log("-" * 64)
        _force_log("SOLVING:")

        # Build job data
        job_data = self._build_job_data(before_state)

        if use_engine:
            result = self._solve_via_engine(job_data, before_state)
        else:
            result = self._solve_local(job_data, before_state)

        result.solve_time_us = (time.perf_counter() - start_time) * 1_000_000

        # Apply result to armature
        if result.success and result.bone_transforms:
            self._apply_result(result)
            self._log_solve_step("APPLY", f"Applied {len(result.bone_transforms)} bone transforms")

        # Force Blender update
        bpy.context.view_layer.update()

        # Capture AFTER bone positions (for diagnostic)
        after_bones = self._capture_bone_positions()

        # Capture AFTER state and compute errors
        after_state = self.capture_state()
        result = self._compute_errors(result, after_state)

        # Log results
        self._log_after_state(before_state, after_state, result)
        self._log_result(result)

        # ALWAYS run diagnostic (not gated by debug flag)
        self._diagnose_ik_result(before_bones, after_bones, self.constraints)

        self._last_result = result
        return result

    def _build_job_data(self, state: FullBodyState) -> dict:
        """Build job data for worker."""
        return {
            "armature_name": self.armature.name,
            "constraints": self.constraints.to_dict(),
            "current_state": {
                "root_pos": state.root_pos,
                "hips_pos": state.hips_pos,
                "hips_rot": state.hips_rot,
                "left_foot_pos": state.left_foot_pos,
                "right_foot_pos": state.right_foot_pos,
                "left_hand_pos": state.left_hand_pos,
                "right_hand_pos": state.right_hand_pos,
            },
        }

    def _solve_via_engine(self, job_data: dict, before_state: FullBodyState) -> FullBodyResult:
        """Submit job to engine worker and wait for result."""
        from ..engine import get_engine

        engine = get_engine()
        if not engine or not engine.is_alive():
            self._log_solve_step("ENGINE", "NOT AVAILABLE - falling back to local solve")
            return self._solve_local(job_data, before_state)

        # Submit job
        job_id = engine.submit_job("FULL_BODY_IK", job_data)
        if job_id is None or job_id < 0:
            self._log_solve_step("ENGINE", "JOB REJECTED (queue full) - falling back to local solve")
            return self._solve_local(job_data, before_state)

        # Poll for result (with timeout)
        poll_start = time.perf_counter()
        timeout = 0.05  # 50ms max wait

        while (time.perf_counter() - poll_start) < timeout:
            results = list(engine.poll_results(max_results=10))
            for r in results:
                if r.job_type == "FULL_BODY_IK" and r.job_id == job_id:
                    if r.success:
                        return self._parse_engine_result(r.result)
                    else:
                        log_game("FULL-BODY-IK", f"WORKER FAILED: {r.error}")
                        return FullBodyResult(success=False)
            time.sleep(0.001)

        log_game("FULL-BODY-IK", "TIMEOUT waiting for worker result")
        return FullBodyResult(success=False)

    def _solve_local(self, job_data: dict, before_state: FullBodyState) -> FullBodyResult:
        """
        Solve IK locally with comprehensive step-by-step logging.

        This solver:
        1. Applies hips drop/offset first
        2. Solves leg IK to maintain foot positions
        3. Solves arm IK for hand targets
        4. Logs each step in detail for debugging
        """
        import numpy as np

        result = FullBodyResult()
        result.constraints_total = self.constraints.get_active_count()
        result.constraints_satisfied = 0
        result.constraints_at_limit = 0

        pose_bones = self.armature.pose.bones
        arm_matrix = self.armature.matrix_world

        # Get Root position for coordinate transforms
        root_bone = pose_bones.get("Root")
        if root_bone:
            root_world = arm_matrix @ root_bone.head
        else:
            root_world = Vector(self.armature.location)
        root_pos = np.array([root_world.x, root_world.y, root_world.z], dtype=np.float32)

        # Character orientation (from armature)
        char_forward = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        char_right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        char_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        self._log_solve_step("INIT", f"root_world=({root_world.x:.3f}, {root_world.y:.3f}, {root_world.z:.3f})")

        # =====================================================================
        # STEP 1: HIPS DROP/OFFSET
        # =====================================================================
        if abs(self.constraints.hips_drop) > 0.001 or any(abs(x) > 0.001 for x in self.constraints.hips_offset):
            hips_bone = pose_bones.get("Hips")
            if hips_bone:
                drop = self.constraints.hips_drop
                offset = self.constraints.hips_offset

                # ═══════════════════════════════════════════════════════════════
                # IMPORTANT: Transform world-space drop into bone-local space
                # ═══════════════════════════════════════════════════════════════
                # The drop is specified in WORLD space (down = -Z world)
                # But bone.location is in BONE-LOCAL space
                # We need to convert the world offset to local offset
                #
                # World offset = (offset[0], offset[1], offset[2] - drop)
                # Local offset = inverse(bone_rest_matrix) @ world_offset

                # World-space offset we want
                world_offset = Vector((offset[0], offset[1], -drop))

                # Get the bone's rest matrix (transforms local to armature space)
                # bone.matrix_local is the bone's rest pose in armature space
                bone_rest = hips_bone.bone.matrix_local

                # To convert world offset to local: multiply by inverse of rest matrix's rotation
                # (ignoring translation since we want offset, not position)
                bone_rest_rot = bone_rest.to_3x3()
                local_offset = bone_rest_rot.inverted() @ world_offset

                self._log_solve_step("HIPS", f"drop={drop:.3f}m offset=({offset[0]:.3f}, {offset[1]:.3f}, {offset[2]:.3f})")
                self._log_solve_step("HIPS", f"world_offset=({world_offset.x:.3f}, {world_offset.y:.3f}, {world_offset.z:.3f})")
                self._log_solve_step("HIPS", f"local_offset=({local_offset.x:.3f}, {local_offset.y:.3f}, {local_offset.z:.3f})")

                # Apply location change in LOCAL space
                hips_bone.location = local_offset

                result.bone_transforms["Hips"] = [
                    1.0, 0.0, 0.0, 0.0,  # quat (unchanged)
                    local_offset.x, local_offset.y, local_offset.z,  # location (local)
                ]

                result.constraints_satisfied += 1
            else:
                self._log_solve_step("HIPS", "BONE NOT FOUND - cannot apply hips transform")

        # Force update to get new hip positions for leg IK
        bpy.context.view_layer.update()

        # =====================================================================
        # STEP 2: LEFT LEG IK
        # =====================================================================
        if self.constraints.left_foot and self.constraints.left_foot.enabled:
            self._log_solve_step("L_LEG", "solving...")

            # Get both leg bones
            left_thigh = pose_bones.get("LeftThigh")
            left_shin = pose_bones.get("LeftShin")
            if left_thigh and left_shin:
                hip_world = arm_matrix @ left_thigh.head
                hip_pos = np.array([hip_world.x, hip_world.y, hip_world.z], dtype=np.float32)

                # Target is Root-relative, convert to world
                target_rel = self.constraints.left_foot.position
                target_world = root_pos + np.array(target_rel, dtype=np.float32)

                # Compute knee pole
                knee_pole = compute_knee_pole_position(hip_pos, target_world, char_forward, char_right, "L")

                # Log geometry
                hip_to_target = np.linalg.norm(target_world - hip_pos)
                max_reach = LEG_IK["leg_L"]["reach"]
                self._log_solve_step("L_LEG", f"hip=({hip_pos[0]:.3f}, {hip_pos[1]:.3f}, {hip_pos[2]:.3f})")
                self._log_solve_step("L_LEG", f"target=({target_world[0]:.3f}, {target_world[1]:.3f}, {target_world[2]:.3f})")
                self._log_solve_step("L_LEG", f"dist={hip_to_target:.3f}m reach={max_reach:.3f}m {'REACHABLE' if hip_to_target <= max_reach else 'AT_LIMIT'}")

                # Solve IK - returns DIRECTIONS not quaternions
                # Pass char_forward so knee bends correctly even for high kicks
                thigh_dir, shin_dir, knee_pos = solve_leg_ik(hip_pos, target_world, knee_pole, "L", char_forward)

                # Log computed directions
                self._log_solve_step("L_LEG", f"COMPUTED thigh_dir=({thigh_dir[0]:.3f}, {thigh_dir[1]:.3f}, {thigh_dir[2]:.3f})")
                self._log_solve_step("L_LEG", f"COMPUTED shin_dir=({shin_dir[0]:.3f}, {shin_dir[1]:.3f}, {shin_dir[2]:.3f})")
                self._log_solve_step("L_LEG", f"COMPUTED knee=({knee_pos[0]:.3f}, {knee_pos[1]:.3f}, {knee_pos[2]:.3f})")

                # Verify and log results
                self._verify_limb_ik(
                    "L_LEG", hip_pos, knee_pos, target_world,
                    LEG_IK["leg_L"]["len_upper"], LEG_IK["leg_L"]["len_lower"],
                    char_forward, is_leg=True
                )

                # Compute rotations - PARENT FIRST, then child relative to rotated parent
                # Same pattern as arms: direction -> bone rotation
                self._log_solve_step("L_LEG", "Computing bone rotations...")

                # Thigh: simple rotation from rest to target direction
                thigh_quat = self._compute_bone_rotation_to_direction(left_thigh, Vector(thigh_dir), arm_matrix)

                # Shin: must account for thigh's rotation (child inherits parent rotation)
                shin_quat = self._compute_child_rotation_for_ik(
                    left_shin, Vector(shin_dir), thigh_quat, arm_matrix
                )

                # Store transforms
                result.bone_transforms["LeftThigh"] = [thigh_quat.w, thigh_quat.x, thigh_quat.y, thigh_quat.z, 0, 0, 0]
                result.bone_transforms["LeftShin"] = [shin_quat.w, shin_quat.x, shin_quat.y, shin_quat.z, 0, 0, 0]

                result.constraints_satisfied += 1
                if hip_to_target > max_reach:
                    result.constraints_at_limit += 1
            else:
                self._log_solve_step("L_LEG", "LeftThigh or LeftShin BONE NOT FOUND")

        # =====================================================================
        # STEP 3: RIGHT LEG IK
        # =====================================================================
        if self.constraints.right_foot and self.constraints.right_foot.enabled:
            self._log_solve_step("R_LEG", "solving...")

            # Get both leg bones
            right_thigh = pose_bones.get("RightThigh")
            right_shin = pose_bones.get("RightShin")
            if right_thigh and right_shin:
                hip_world = arm_matrix @ right_thigh.head
                hip_pos = np.array([hip_world.x, hip_world.y, hip_world.z], dtype=np.float32)

                target_rel = self.constraints.right_foot.position
                target_world = root_pos + np.array(target_rel, dtype=np.float32)

                knee_pole = compute_knee_pole_position(hip_pos, target_world, char_forward, char_right, "R")

                hip_to_target = np.linalg.norm(target_world - hip_pos)
                max_reach = LEG_IK["leg_R"]["reach"]
                self._log_solve_step("R_LEG", f"hip=({hip_pos[0]:.3f}, {hip_pos[1]:.3f}, {hip_pos[2]:.3f})")
                self._log_solve_step("R_LEG", f"target=({target_world[0]:.3f}, {target_world[1]:.3f}, {target_world[2]:.3f})")
                self._log_solve_step("R_LEG", f"dist={hip_to_target:.3f}m reach={max_reach:.3f}m {'REACHABLE' if hip_to_target <= max_reach else 'AT_LIMIT'}")

                # Solve IK - returns DIRECTIONS not quaternions
                # Pass char_forward so knee bends correctly even for high kicks
                thigh_dir, shin_dir, knee_pos = solve_leg_ik(hip_pos, target_world, knee_pole, "R", char_forward)

                # Log computed directions
                self._log_solve_step("R_LEG", f"COMPUTED thigh_dir=({thigh_dir[0]:.3f}, {thigh_dir[1]:.3f}, {thigh_dir[2]:.3f})")
                self._log_solve_step("R_LEG", f"COMPUTED shin_dir=({shin_dir[0]:.3f}, {shin_dir[1]:.3f}, {shin_dir[2]:.3f})")
                self._log_solve_step("R_LEG", f"COMPUTED knee=({knee_pos[0]:.3f}, {knee_pos[1]:.3f}, {knee_pos[2]:.3f})")

                # Verify and log results
                self._verify_limb_ik(
                    "R_LEG", hip_pos, knee_pos, target_world,
                    LEG_IK["leg_R"]["len_upper"], LEG_IK["leg_R"]["len_lower"],
                    char_forward, is_leg=True
                )

                # Compute rotations - PARENT FIRST, then child relative to rotated parent
                self._log_solve_step("R_LEG", "Computing bone rotations...")

                # Thigh: simple rotation from rest to target direction
                thigh_quat = self._compute_bone_rotation_to_direction(right_thigh, Vector(thigh_dir), arm_matrix)

                # Shin: must account for thigh's rotation (child inherits parent rotation)
                shin_quat = self._compute_child_rotation_for_ik(
                    right_shin, Vector(shin_dir), thigh_quat, arm_matrix
                )

                result.bone_transforms["RightThigh"] = [thigh_quat.w, thigh_quat.x, thigh_quat.y, thigh_quat.z, 0, 0, 0]
                result.bone_transforms["RightShin"] = [shin_quat.w, shin_quat.x, shin_quat.y, shin_quat.z, 0, 0, 0]

                result.constraints_satisfied += 1
                if hip_to_target > max_reach:
                    result.constraints_at_limit += 1
            else:
                self._log_solve_step("R_LEG", "RightThigh or RightShin BONE NOT FOUND")

        # =====================================================================
        # STEP 4: LEFT ARM IK
        # =====================================================================
        if self.constraints.left_hand and self.constraints.left_hand.enabled:
            self._log_solve_step("L_ARM", "solving...")

            left_arm = pose_bones.get("LeftArm")
            left_forearm = pose_bones.get("LeftForeArm")
            if left_arm and left_forearm:
                shoulder_world = arm_matrix @ left_arm.head
                shoulder_pos = np.array([shoulder_world.x, shoulder_world.y, shoulder_world.z], dtype=np.float32)

                target_rel = self.constraints.left_hand.position
                target_world = root_pos + np.array(target_rel, dtype=np.float32)

                elbow_pole = compute_elbow_pole_position(shoulder_pos, target_world, char_forward, char_up, "L")

                # Log pole direction relative to body (check outward bias)
                pole_from_shoulder = elbow_pole - shoulder_pos
                pole_len = np.linalg.norm(pole_from_shoulder)
                if pole_len > 0.001:
                    pole_dir = pole_from_shoulder / pole_len
                    char_right = np.cross(char_forward, char_up)
                    pole_leftward = -np.dot(pole_dir, char_right)  # + = leftward (correct for L arm)
                    pole_backward = -np.dot(pole_dir, char_forward)  # + = backward
                    pole_upward = np.dot(pole_dir, char_up)  # + = upward
                    self._log_solve_step("L_ARM", f"POLE BIAS: left={pole_leftward:.2f} back={pole_backward:.2f} up={pole_upward:.2f}")

                shoulder_to_target = np.linalg.norm(target_world - shoulder_pos)
                max_reach = ARM_IK["arm_L"]["reach"]
                self._log_solve_step("L_ARM", f"shoulder=({shoulder_pos[0]:.3f}, {shoulder_pos[1]:.3f}, {shoulder_pos[2]:.3f})")
                self._log_solve_step("L_ARM", f"target=({target_world[0]:.3f}, {target_world[1]:.3f}, {target_world[2]:.3f})")
                self._log_solve_step("L_ARM", f"dist={shoulder_to_target:.3f}m reach={max_reach:.3f}m {'REACHABLE' if shoulder_to_target <= max_reach else 'AT_LIMIT'}")

                # Solve for DIRECTIONS (not quaternions)
                # Pass char_forward for cross-body constraint (elbow must stay behind shoulder)
                upper_dir, lower_dir, elbow_pos = solve_arm_ik(shoulder_pos, target_world, elbow_pole, "L", char_forward)

                # ═══════════════════════════════════════════════════════════════════
                # CROSS-BODY ANALYSIS - Check anatomical validity BEFORE proceeding
                # ═══════════════════════════════════════════════════════════════════
                cross_analysis = analyze_cross_body(
                    side="L",
                    shoulder_pos=tuple(shoulder_pos),
                    target_pos=tuple(target_world),
                    elbow_pos=tuple(elbow_pos),
                    body_center_x=0.0
                )
                cross_analysis.log()

                # If anatomically impossible, try to fix it
                if not cross_analysis.anatomically_possible:
                    self._log_solve_step("L_ARM", ">>> APPLYING ANATOMICAL FIX <<<")
                    upper_dir, lower_dir, elbow_pos = self._fix_cross_body_elbow(
                        shoulder_pos, target_world, elbow_pos, upper_dir, "L",
                        char_forward, char_up, ARM_IK["arm_L"]["len_upper"], ARM_IK["arm_L"]["len_lower"]
                    )
                    self._log_solve_step("L_ARM", f"FIXED elbow=({elbow_pos[0]:.3f}, {elbow_pos[1]:.3f}, {elbow_pos[2]:.3f})")

                # Log computed directions
                self._log_solve_step("L_ARM", f"COMPUTED upper_dir=({upper_dir[0]:.3f}, {upper_dir[1]:.3f}, {upper_dir[2]:.3f})")
                self._log_solve_step("L_ARM", f"COMPUTED lower_dir=({lower_dir[0]:.3f}, {lower_dir[1]:.3f}, {lower_dir[2]:.3f})")
                self._log_solve_step("L_ARM", f"COMPUTED elbow=({elbow_pos[0]:.3f}, {elbow_pos[1]:.3f}, {elbow_pos[2]:.3f})")

                # Verify and log results
                self._verify_limb_ik(
                    "L_ARM", shoulder_pos, elbow_pos, target_world,
                    ARM_IK["arm_L"]["len_upper"], ARM_IK["arm_L"]["len_lower"],
                    char_forward, is_leg=False
                )

                # Compute rotations - PARENT FIRST, then child relative to rotated parent
                self._log_solve_step("L_ARM", "Computing bone rotations...")

                # Upper arm: simple rotation from rest to target
                upper_quat = self._compute_bone_rotation_to_direction(left_arm, Vector(upper_dir), arm_matrix)

                # Lower arm: must account for upper arm's rotation!
                # After upper arm rotates, forearm's starting direction changes
                lower_quat = self._compute_child_rotation_for_ik(
                    left_forearm, Vector(lower_dir), upper_quat, arm_matrix
                )

                result.bone_transforms["LeftArm"] = [upper_quat.w, upper_quat.x, upper_quat.y, upper_quat.z, 0, 0, 0]
                result.bone_transforms["LeftForeArm"] = [lower_quat.w, lower_quat.x, lower_quat.y, lower_quat.z, 0, 0, 0]

                result.constraints_satisfied += 1
                if shoulder_to_target > max_reach:
                    result.constraints_at_limit += 1
            else:
                self._log_solve_step("L_ARM", "LeftArm or LeftForeArm BONE NOT FOUND")

        # =====================================================================
        # STEP 5: RIGHT ARM IK
        # =====================================================================
        if self.constraints.right_hand and self.constraints.right_hand.enabled:
            self._log_solve_step("R_ARM", "solving...")

            right_arm = pose_bones.get("RightArm")
            right_forearm = pose_bones.get("RightForeArm")
            if right_arm and right_forearm:
                shoulder_world = arm_matrix @ right_arm.head
                shoulder_pos = np.array([shoulder_world.x, shoulder_world.y, shoulder_world.z], dtype=np.float32)

                target_rel = self.constraints.right_hand.position
                target_world = root_pos + np.array(target_rel, dtype=np.float32)

                elbow_pole = compute_elbow_pole_position(shoulder_pos, target_world, char_forward, char_up, "R")

                # Log pole direction relative to body (check outward bias)
                pole_from_shoulder = elbow_pole - shoulder_pos
                pole_len = np.linalg.norm(pole_from_shoulder)
                if pole_len > 0.001:
                    pole_dir = pole_from_shoulder / pole_len
                    char_right = np.cross(char_forward, char_up)
                    pole_rightward = np.dot(pole_dir, char_right)  # + = rightward (correct for R arm)
                    pole_backward = -np.dot(pole_dir, char_forward)  # + = backward
                    pole_upward = np.dot(pole_dir, char_up)  # + = upward
                    self._log_solve_step("R_ARM", f"POLE BIAS: right={pole_rightward:.2f} back={pole_backward:.2f} up={pole_upward:.2f}")

                shoulder_to_target = np.linalg.norm(target_world - shoulder_pos)
                max_reach = ARM_IK["arm_R"]["reach"]
                self._log_solve_step("R_ARM", f"shoulder=({shoulder_pos[0]:.3f}, {shoulder_pos[1]:.3f}, {shoulder_pos[2]:.3f})")
                self._log_solve_step("R_ARM", f"target=({target_world[0]:.3f}, {target_world[1]:.3f}, {target_world[2]:.3f})")
                self._log_solve_step("R_ARM", f"dist={shoulder_to_target:.3f}m reach={max_reach:.3f}m {'REACHABLE' if shoulder_to_target <= max_reach else 'AT_LIMIT'}")

                # Solve for DIRECTIONS (not quaternions)
                # Pass char_forward for cross-body constraint (elbow must stay behind shoulder)
                upper_dir, lower_dir, elbow_pos = solve_arm_ik(shoulder_pos, target_world, elbow_pole, "R", char_forward)

                # ═══════════════════════════════════════════════════════════════════
                # CROSS-BODY ANALYSIS - Check anatomical validity BEFORE proceeding
                # ═══════════════════════════════════════════════════════════════════
                cross_analysis = analyze_cross_body(
                    side="R",
                    shoulder_pos=tuple(shoulder_pos),
                    target_pos=tuple(target_world),
                    elbow_pos=tuple(elbow_pos),
                    body_center_x=0.0
                )
                cross_analysis.log()

                # If anatomically impossible, try to fix it
                if not cross_analysis.anatomically_possible:
                    self._log_solve_step("R_ARM", ">>> APPLYING ANATOMICAL FIX <<<")
                    upper_dir, lower_dir, elbow_pos = self._fix_cross_body_elbow(
                        shoulder_pos, target_world, elbow_pos, upper_dir, "R",
                        char_forward, char_up, ARM_IK["arm_R"]["len_upper"], ARM_IK["arm_R"]["len_lower"]
                    )
                    self._log_solve_step("R_ARM", f"FIXED elbow=({elbow_pos[0]:.3f}, {elbow_pos[1]:.3f}, {elbow_pos[2]:.3f})")

                # Log computed directions
                self._log_solve_step("R_ARM", f"COMPUTED upper_dir=({upper_dir[0]:.3f}, {upper_dir[1]:.3f}, {upper_dir[2]:.3f})")
                self._log_solve_step("R_ARM", f"COMPUTED lower_dir=({lower_dir[0]:.3f}, {lower_dir[1]:.3f}, {lower_dir[2]:.3f})")
                self._log_solve_step("R_ARM", f"COMPUTED elbow=({elbow_pos[0]:.3f}, {elbow_pos[1]:.3f}, {elbow_pos[2]:.3f})")

                # Verify and log results
                self._verify_limb_ik(
                    "R_ARM", shoulder_pos, elbow_pos, target_world,
                    ARM_IK["arm_R"]["len_upper"], ARM_IK["arm_R"]["len_lower"],
                    char_forward, is_leg=False
                )

                # Compute rotations - PARENT FIRST, then child relative to rotated parent
                self._log_solve_step("R_ARM", "Computing bone rotations...")

                # Upper arm: simple rotation from rest to target
                upper_quat = self._compute_bone_rotation_to_direction(right_arm, Vector(upper_dir), arm_matrix)

                # Lower arm: must account for upper arm's rotation!
                lower_quat = self._compute_child_rotation_for_ik(
                    right_forearm, Vector(lower_dir), upper_quat, arm_matrix
                )

                result.bone_transforms["RightArm"] = [upper_quat.w, upper_quat.x, upper_quat.y, upper_quat.z, 0, 0, 0]
                result.bone_transforms["RightForeArm"] = [lower_quat.w, lower_quat.x, lower_quat.y, lower_quat.z, 0, 0, 0]

                result.constraints_satisfied += 1
                if shoulder_to_target > max_reach:
                    result.constraints_at_limit += 1
            else:
                self._log_solve_step("R_ARM", "RightArm or RightForeArm BONE NOT FOUND")

        # =====================================================================
        # STEP 6: HEAD LOOK-AT
        # =====================================================================
        # From rig.md: Distribute rotation across neck and head bones
        # Weights: NeckLower=15%, NeckUpper=25%, Head=60%
        if self.constraints.look_at and self.constraints.look_at.enabled:
            self._log_solve_step("HEAD", "solving look-at...")

            head = pose_bones.get("Head")

            if head:
                # Target in world space
                look_target_rel = self.constraints.look_at.position
                look_target_world = root_pos + np.array(look_target_rel, dtype=np.float32)

                # Head position
                head_world = arm_matrix @ head.head
                head_pos = np.array([head_world.x, head_world.y, head_world.z], dtype=np.float32)

                # Direction from head to target
                look_dir = look_target_world - head_pos
                look_dist = np.linalg.norm(look_dir)

                if look_dist > 0.01:
                    look_dir = look_dir / look_dist
                    look_dir_vec = Vector(look_dir)

                    # Head's forward is +Z from rig.md ("Head: +Z Points FORWARD")
                    # Compute direction the head should point
                    look_target_dir = look_dir_vec.normalized()

                    # Distribute rotation with weights from rig.md
                    weights = [
                        ("NeckLower", 0.15),
                        ("NeckUpper", 0.25),
                        ("Head", 0.60)
                    ]

                    for bone_name, weight in weights:
                        bone = pose_bones.get(bone_name)
                        if bone:
                            # Get bone's rest forward (Z axis for spine/head bones)
                            rest_matrix = bone.bone.matrix_local
                            rest_forward = rest_matrix.to_3x3() @ Vector((0, 0, 1))
                            rest_forward_world = (arm_matrix.to_3x3() @ rest_forward).normalized()

                            # Rotation from current forward to target
                            full_rotation = rest_forward_world.rotation_difference(look_target_dir)

                            # Scale by weight (slerp from identity)
                            partial_rotation = Quaternion().slerp(full_rotation, weight)

                            # Convert to local bone space
                            bone_rest_world = arm_matrix @ rest_matrix
                            bone_rest_quat = bone_rest_world.to_quaternion()
                            bone_rest_inv = bone_rest_quat.inverted()
                            local_quat = bone_rest_inv @ partial_rotation @ bone_rest_quat

                            result.bone_transforms[bone_name] = [
                                local_quat.w, local_quat.x, local_quat.y, local_quat.z, 0, 0, 0
                            ]
                            self._log_solve_step("HEAD", f"{bone_name}: {weight:.0%}")

                    result.constraints_satisfied += 1
                    self._log_solve_step("HEAD", f"look-at dist={look_dist:.2f}m")
                else:
                    self._log_solve_step("HEAD", "target too close")
            else:
                self._log_solve_step("HEAD", "Head bone NOT FOUND")

        # =====================================================================
        # FINAL STATUS
        # =====================================================================
        result.success = result.constraints_satisfied > 0 or result.constraints_total == 0
        self._log_solve_step("DONE", f"satisfied={result.constraints_satisfied}/{result.constraints_total} at_limit={result.constraints_at_limit}")

        return result

    def _parse_engine_result(self, result_data: dict) -> FullBodyResult:
        """Parse engine result into FullBodyResult."""
        result = FullBodyResult()
        result.success = result_data.get("success", False)
        result.constraints_satisfied = result_data.get("constraints_satisfied", 0)
        result.constraints_total = result_data.get("constraints_total", 0)
        result.constraints_at_limit = result_data.get("constraints_at_limit", 0)
        result.joint_violations = result_data.get("joint_violations", [])
        result.bone_transforms = result_data.get("bone_transforms", {})
        return result

    def _apply_result(self, result: FullBodyResult):
        """
        Apply solved bone transforms to armature.

        The transforms are already LOCAL quaternions (computed using Blender's
        actual bone data in _compute_bone_rotation_to_direction).
        """
        import numpy as np

        pose_bones = self.armature.pose.bones
        arm_matrix = self.armature.matrix_world

        _force_log("-" * 70)
        _force_log("APPLYING TRANSFORMS (already local):")

        for bone_name, transform in result.bone_transforms.items():
            bone = pose_bones.get(bone_name)
            if not bone:
                _force_log(f"  {bone_name}: BONE NOT FOUND!")
                continue

            # Ensure quaternion mode
            bone.rotation_mode = 'QUATERNION'

            # Get BEFORE rotation
            old_quat = tuple(bone.rotation_quaternion)

            # Get bone's current world position for debugging
            bone_world = arm_matrix @ bone.head

            if len(transform) >= 4:
                w, x, y, z = transform[0], transform[1], transform[2], transform[3]

                # Convert quaternion to axis-angle for human readability
                angle_rad = 2 * np.arccos(np.clip(w, -1, 1))
                angle_deg = np.degrees(angle_rad)

                if abs(angle_rad) > 0.001:
                    s = np.sqrt(1 - w*w)
                    if s > 0.001:
                        axis = (x/s, y/s, z/s)
                    else:
                        axis = (1, 0, 0)
                else:
                    axis = (1, 0, 0)

                _force_log(f"  {bone_name}:")
                _force_log(f"    bone_world=({bone_world.x:.3f}, {bone_world.y:.3f}, {bone_world.z:.3f})")
                _force_log(f"    before: local=({old_quat[0]:.3f}, {old_quat[1]:.3f}, {old_quat[2]:.3f}, {old_quat[3]:.3f})")
                _force_log(f"    apply:  local=({w:.3f}, {x:.3f}, {y:.3f}, {z:.3f})")
                _force_log(f"    axis=({axis[0]:.2f}, {axis[1]:.2f}, {axis[2]:.2f}) angle={angle_deg:.1f}°")

                # Check if quaternions differ
                quat_diff = abs(old_quat[0] - w) + abs(old_quat[1] - x) + abs(old_quat[2] - y) + abs(old_quat[3] - z)
                if quat_diff < 0.01:
                    _force_log(f"    ⚠️ WARNING: QUAT UNCHANGED!")
                else:
                    _force_log(f"    ✓ QUAT CHANGED (diff={quat_diff:.3f})")

                # Apply the rotation
                bone.rotation_quaternion = (w, x, y, z)

            if len(transform) >= 7:
                bone.location = (transform[4], transform[5], transform[6])
                _force_log(f"    loc=({transform[4]:.3f}, {transform[5]:.3f}, {transform[6]:.3f})")

            # Verify it changed
            new_quat = tuple(bone.rotation_quaternion)
            _force_log(f"    after:  local=({new_quat[0]:.3f}, {new_quat[1]:.3f}, {new_quat[2]:.3f}, {new_quat[3]:.3f})")

        # Force Blender to update
        bpy.context.view_layer.update()
        _force_log("-" * 70)

    def _compute_errors(self, result: FullBodyResult, after_state: FullBodyState) -> FullBodyResult:
        """Compute position errors between targets and achieved positions."""

        def pos_error_cm(target: Optional[IKTarget], actual: Tuple[float, float, float]) -> float:
            if not target or not target.enabled:
                return 0.0
            t = Vector(target.position)
            a = Vector(actual)
            return (t - a).length * 100  # meters to cm

        result.left_foot_error_cm = pos_error_cm(self.constraints.left_foot, after_state.left_foot_pos)
        result.right_foot_error_cm = pos_error_cm(self.constraints.right_foot, after_state.right_foot_pos)
        result.left_hand_error_cm = pos_error_cm(self.constraints.left_hand, after_state.left_hand_pos)
        result.right_hand_error_cm = pos_error_cm(self.constraints.right_hand, after_state.right_hand_pos)

        return result

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    def ground_feet(self):
        """
        Set foot targets to current foot positions (keep feet where they are).

        Useful as baseline before applying other constraints.
        """
        state = self.capture_state()
        self.set_foot_targets(
            left_pos=state.left_foot_pos,
            right_pos=state.right_foot_pos
        )

    def crouch(self, amount: float):
        """
        Simple crouch by dropping hips while keeping feet grounded.

        Args:
            amount: Crouch amount 0-1 (0=standing, 1=full crouch ~0.4m drop)
        """
        self.ground_feet()
        self.set_hips_drop(amount * 0.4)  # Max 40cm drop

    def reach_for(self, position: Tuple[float, float, float], hand: str = "R"):
        """
        Reach toward a position with specified hand.

        Args:
            position: Target position (Root-relative)
            hand: "L" for left, "R" for right
        """
        if hand == "L":
            self.set_hand_targets(left_pos=position)
        else:
            self.set_hand_targets(right_pos=position)


# =============================================================================
# MODULE-LEVEL STATE
# =============================================================================

_active_fbik: Optional[FullBodyIK] = None


def get_full_body_ik(armature: Optional[bpy.types.Object] = None) -> Optional[FullBodyIK]:
    """
    Get or create FullBodyIK instance for armature.

    Args:
        armature: Armature object. If None, uses scene.target_armature.

    Returns:
        FullBodyIK instance or None if no valid armature
    """
    global _active_fbik

    if armature is None:
        scene = bpy.context.scene
        armature = getattr(scene, 'target_armature', None)

    if not armature or armature.type != 'ARMATURE':
        return None

    # Reuse existing if same armature
    if _active_fbik and _active_fbik.armature == armature:
        return _active_fbik

    # Create new
    _active_fbik = FullBodyIK(armature)
    return _active_fbik


def clear_full_body_ik():
    """Clear the active FullBodyIK instance."""
    global _active_fbik
    _active_fbik = None


def get_fbik_state() -> dict:
    """
    Get current full-body IK state for visualization.

    Returns dict with:
        - active: bool
        - constraints: dict of active constraints
        - last_result: last solve result metrics
    """
    if not _active_fbik:
        return {"active": False}

    return {
        "active": True,
        "constraints": _active_fbik.constraints.to_dict(),
        "last_result": {
            "success": _active_fbik._last_result.success if _active_fbik._last_result else False,
            "left_foot_error_cm": _active_fbik._last_result.left_foot_error_cm if _active_fbik._last_result else 0,
            "right_foot_error_cm": _active_fbik._last_result.right_foot_error_cm if _active_fbik._last_result else 0,
            "left_hand_error_cm": _active_fbik._last_result.left_hand_error_cm if _active_fbik._last_result else 0,
            "right_hand_error_cm": _active_fbik._last_result.right_hand_error_cm if _active_fbik._last_result else 0,
        } if _active_fbik._last_result else None,
    }
