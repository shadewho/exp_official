"""
IK Test Suite v3 - structured tests with slight random variations.

Goal:
- Generate deterministic test coverage with slight randomization for variety
- Auto-evaluate solver results (target error, joint limit hits, bend/balance).
- Persist rich session JSON for offline analysis/tuning.
"""

import json
import math
import os
import random
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import bpy
from mathutils import Vector


# =============================================================================
# VARIATION SYSTEM - Adds slight randomness to keep tests fresh
# =============================================================================

def _vary(value: float, amount: float = 0.03) -> float:
    """Add small random variation to a value (±amount)."""
    return value + random.uniform(-amount, amount)


def _vary_target(target: Tuple[float, float, float], amount: float = 0.03) -> Tuple[float, float, float]:
    """Add small random variation to each coordinate of a target."""
    return (
        round(_vary(target[0], amount), 4),
        round(_vary(target[1], amount), 4),
        round(_vary(target[2], amount), 4),
    )

# NOTE: rig_test_data lives under animations; engine is a sibling of animations.
from ...engine.animations.default_limits import get_default_limits
# NOTE: get_last_solver_diagnostics is imported inside record() to avoid circular import

# =============================================================================
# CRITICAL: Output directory for session data
# =============================================================================
# NEVER use __file__ - it resolves to AppData when Blender runs the addon!
# ALWAYS save to the Desktop source directory so data persists and can be
# version controlled. This is the DEVELOPMENT directory, not the install dir.
# =============================================================================
OUTPUT_DIR = r"C:\Users\spenc\Desktop\Exploratory\addons\Exploratory\Exp_Game\animations\rig_test_data\output_data"


# =============================================================================
# Test Case Definitions
# =============================================================================


@dataclass
class IKTestCase:
    goal_type: str
    description: str
    tags: Dict[str, str]
    hip_drop: float = 0.0
    left_foot_target: Optional[Tuple[float, float, float]] = None
    right_foot_target: Optional[Tuple[float, float, float]] = None
    left_hand_target: Optional[Tuple[float, float, float]] = None
    right_hand_target: Optional[Tuple[float, float, float]] = None
    look_at_target: Optional[Tuple[float, float, float]] = None
    # Spine/lean control - direction the torso should lean toward
    lean_target: Optional[Tuple[float, float, float]] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _height_word(z: float, rig: dict | None = None) -> str:
    # If rig heights supplied, use them to pick a band
    if rig:
        ground = rig.get("ground", 0.0)
        knee = rig.get("knee", ground + 0.4)
        hip = rig.get("hip", ground + 1.0)
        chest = rig.get("chest", ground + 1.3)
        shoulder = rig.get("shoulder", ground + 1.5)
        head = rig.get("head", ground + 1.8)
        if z <= (ground + 0.1):
            return "GROUND"
        if z <= knee:
            return "KNEE"
        if z <= hip:
            return "HIP"
        if z <= chest:
            return "CHEST"
        if z <= shoulder:
            return "SHOULDER"
        if z <= head:
            return "HEAD"
        return "OVERHEAD"
    # Fallback coarse bands
    if z < 0.2:
        return "GROUND"
    if z < 0.6:
        return "KNEE"
    if z < 1.0:
        return "HIP"
    if z < 1.4:
        return "CHEST"
    if z < 1.7:
        return "SHOULDER"
    return "OVERHEAD"


def human_goal_text(test: IKTestCase, rig_heights: dict | None = None) -> str:
    """Generate simple, readable goal text."""
    side = test.tags.get("side", "")
    region = test.tags.get("region", "")

    if region == "arm":
        return f"{side} hand to target"
    if region == "leg":
        return f"{side} foot to target"
    if region == "lean":
        dir_tag = test.tags.get("dir", "forward")
        return f"Lean {dir_tag}"
    if region == "combined":
        return test.tags.get("action", test.description)
    if test.goal_type == "CROUCH":
        depth_cm = test.hip_drop * 100
        return f"Squat {depth_cm:.0f}cm"
    if test.goal_type == "LOOK_AT":
        dir_tag = test.tags.get("dir", "ahead")
        vert = test.tags.get("vert", "")
        if vert:
            return f"Look {vert}"
        return f"Look {dir_tag}"
    return test.description


def human_judge_text(test: IKTestCase) -> str:
    """Generate simple judge-by criteria."""
    region = test.tags.get("region", "")

    if region == "arm":
        return "Hand at target? Elbow back? Shoulder ok?"
    if region == "leg":
        return "Foot at target? Knee forward? Hip ok?"
    if region == "lean":
        return "Spine curved? Shoulders follow? Hips stable?"
    if region == "spine_stress":
        return "Spine bent correctly? No broken joints? Hips compensating?"
    if region == "neck_stress":
        return "Head facing target? Neck not broken? Natural rotation?"
    if region == "shoulder_stress":
        return "Arms positioned? Elbows correct? Shoulders not dislocated?"
    if region == "extreme_arm":
        return "Maximum reach achieved? Elbow not inverted? Shoulder ok?"
    if region == "extreme_leg":
        return "Leg extended? Knee bending correctly? Hip rotating properly?"
    if region == "athletic":
        action = test.tags.get("action", "").lower()
        if "kick" in action:
            return "Kicking leg extended? Standing leg stable? Balance ok?"
        if "punch" in action or "jab" in action or "cross" in action:
            return "Arm extended? Shoulders rotated? Guard hand up?"
        if "throw" in action:
            return "Weight transfer? Arm position? Body rotation?"
        return "Dynamic pose? Weight balanced? Natural motion?"
    if region == "extreme":
        return "Maximum reach achieved? Body compensating? Still stable?"
    if region == "balance":
        action = test.tags.get("action", "").lower()
        if "one leg" in action:
            return "Standing leg stable? Lifted leg positioned? Arms balancing?"
        if "lunge" in action:
            return "Front knee over ankle? Back leg extended? Hips square?"
        if "pose" in action:  # yoga poses
            return "Balance maintained? Limbs positioned? Posture correct?"
        return "Balance stable? Weight distributed? Pose held?"
    if region == "interact":
        return "Hand at interaction point? Body positioned correctly? Natural reach?"
    if region == "asymmetric":
        return "Both arms correct? Positions different but natural? No mirroring errors?"
    if region == "combined":
        # Multi-part check based on what's active
        checks = []
        if test.left_hand_target or test.right_hand_target:
            checks.append("Hand at target?")
        if test.left_foot_target or test.right_foot_target:
            checks.append("Feet placed?")
        if test.hip_drop > 0:
            checks.append("Knees bent?")
        if test.look_at_target:
            checks.append("Looking at target?")
        if test.lean_target:
            checks.append("Torso leaning?")
        return " ".join(checks) if checks else "Pose looks natural?"
    if test.goal_type == "CROUCH":
        return "Hips lower? Knees bent forward? Feet planted?"
    if test.goal_type == "LOOK_AT":
        vert = test.tags.get("vert", "")
        if vert == "up":
            return "Head tilted up? Neck extended? Spine helps?"
        if vert == "down":
            return "Head tilted down? Chin tucked? Spine curves?"
        return "Head facing target? Neck natural?"
    return "Does the pose look correct?"


def _reach_bins(side: str, height_z: float) -> List[IKTestCase]:
    """Generate reach bins - fewer easy, more at challenging heights."""
    is_left = side == "left"
    x_sign = -1 if is_left else 1
    desc_side = "LEFT" if is_left else "RIGHT"

    bins = []

    # Above shoulder = challenging (elbow-up issue) - test all directions
    if height_z >= 1.5:
        for dir_name, xy in [
            ("CENTER", (0.0, 0.35)),
            ("FORWARD", (0.0, 0.5)),
            ("SIDE", (0.25 * x_sign, 0.3)),
        ]:
            target = _vary_target((xy[0], xy[1], height_z))
            tags = {"region": "arm", "side": side, "dir": dir_name.lower(), "height": f"{height_z:.2f}"}
            bins.append(
                IKTestCase(
                    goal_type=f"REACH_{desc_side}_HAND",
                    description=f"{desc_side} HAND: {dir_name} {height_z:.2f}m",
                    tags=tags,
                    left_hand_target=target if is_left else None,
                    right_hand_target=target if not is_left else None,
                )
            )
    else:
        # Below shoulder = easier - just one forward reach as baseline
        target = _vary_target((0.0, 0.5, height_z))
        tags = {"region": "arm", "side": side, "dir": "forward", "height": f"{height_z:.2f}"}
        bins.append(
            IKTestCase(
                goal_type=f"REACH_{desc_side}_HAND",
                description=f"{desc_side} HAND: FORWARD {height_z:.2f}m",
                tags=tags,
                left_hand_target=target if is_left else None,
                right_hand_target=target if not is_left else None,
            )
        )
    return bins


def _foot_bins(side: str, height_z: float) -> List[IKTestCase]:
    """Generate foot targets - one baseline, more challenging high kicks."""
    is_left = side == "left"
    x_sign = -1 if is_left else 1
    desc_side = "LEFT" if is_left else "RIGHT"

    bins = []

    # High kicks = challenging - test forward and side
    if height_z >= 0.8:
        for dir_name, xy in [
            ("FRONT KICK", (0.0, 0.45)),
            ("SIDE KICK", (0.35 * x_sign, 0.15)),
        ]:
            target = _vary_target((xy[0], xy[1], height_z))
            tags = {"region": "leg", "side": side, "dir": dir_name.lower(), "height": f"{height_z:.2f}"}
            bins.append(
                IKTestCase(
                    goal_type=f"REACH_{desc_side}_FOOT",
                    description=f"{desc_side} FOOT: {dir_name} {height_z:.2f}m",
                    tags=tags,
                    left_foot_target=target if is_left else None,
                    right_foot_target=target if not is_left else None,
                )
            )
    else:
        # Ground level = baseline - just one forward step
        target = _vary_target((0.0, 0.35, height_z))
        tags = {"region": "leg", "side": side, "dir": "step", "height": f"{height_z:.2f}"}
        bins.append(
            IKTestCase(
                goal_type=f"REACH_{desc_side}_FOOT",
                description=f"{desc_side} FOOT: STEP {height_z:.2f}m",
                tags=tags,
                left_foot_target=target if is_left else None,
                right_foot_target=target if not is_left else None,
            )
        )
    return bins


def _crouch_bins(ground_z: float = 0.0, foot_spacing: float = 0.1) -> List[IKTestCase]:
    """
    Generate crouch tests - one baseline, one deep.
    """
    bins = []
    left_foot = _vary_target((-foot_spacing, 0.0, ground_z), 0.02)
    right_foot = _vary_target((foot_spacing, 0.0, ground_z), 0.02)

    # Medium crouch (baseline)
    bins.append(
        IKTestCase(
            goal_type="CROUCH",
            description="CROUCH 25cm",
            tags={"region": "crouch", "depth": "0.25"},
            hip_drop=_vary(0.25, 0.02),
            left_foot_target=left_foot,
            right_foot_target=right_foot,
        )
    )
    # Deep crouch (challenging)
    bins.append(
        IKTestCase(
            goal_type="CROUCH",
            description="CROUCH 45cm",
            tags={"region": "crouch", "depth": "0.45"},
            hip_drop=_vary(0.45, 0.02),
            left_foot_target=left_foot,
            right_foot_target=right_foot,
        )
    )
    return bins


def _look_bins() -> List[IKTestCase]:
    """Generate look tests - one baseline, plus challenging extremes."""
    bins = []

    # One baseline horizontal look
    bins.append(
        IKTestCase(
            goal_type="LOOK_AT",
            description="LOOK LEFT",
            tags={"region": "look", "dir": "left"},
            look_at_target=_vary_target((-0.5, 2.0, 1.6), 0.05),
        )
    )

    # Extreme up (ceiling) - requires max neck extension
    bins.append(
        IKTestCase(
            goal_type="LOOK_AT",
            description="LOOK STRAIGHT UP",
            tags={"region": "look", "vert": "up"},
            look_at_target=_vary_target((0.0, 0.3, 3.0), 0.1),
        )
    )

    # Extreme down (ground)
    bins.append(
        IKTestCase(
            goal_type="LOOK_AT",
            description="LOOK AT GROUND",
            tags={"region": "look", "vert": "down"},
            look_at_target=_vary_target((0.0, 0.5, 0.0), 0.1),
        )
    )

    # Over shoulder looks (extreme rotation - challenging)
    bins.append(
        IKTestCase(
            goal_type="LOOK_AT",
            description="LOOK BEHIND LEFT",
            tags={"region": "look", "dir": "behind_left"},
            look_at_target=_vary_target((-1.0, -1.0, 1.6), 0.08),
        )
    )
    bins.append(
        IKTestCase(
            goal_type="LOOK_AT",
            description="LOOK BEHIND RIGHT",
            tags={"region": "look", "dir": "behind_right"},
            look_at_target=_vary_target((1.0, -1.0, 1.6), 0.08),
        )
    )

    return bins


def _lean_bins(ground_z: float = 0.0) -> List[IKTestCase]:
    """
    Generate lean/spine tests (with slight variation).

    Lean tests check if the spine bends correctly to shift weight or reach.
    Feet stay planted, spine curves toward target.
    """
    bins = []
    foot_spacing = 0.1
    left_foot = _vary_target((-foot_spacing, 0.0, ground_z), 0.02)
    right_foot = _vary_target((foot_spacing, 0.0, ground_z), 0.02)

    # Forward lean (reaching for something in front, low)
    bins.append(
        IKTestCase(
            goal_type="LEAN",
            description="LEAN FORWARD",
            tags={"region": "lean", "dir": "forward"},
            lean_target=_vary_target((0.0, 0.8, 1.0), 0.05),  # forward and down
            left_foot_target=left_foot,
            right_foot_target=right_foot,
        )
    )

    # Backward lean
    bins.append(
        IKTestCase(
            goal_type="LEAN",
            description="LEAN BACKWARD",
            tags={"region": "lean", "dir": "backward"},
            lean_target=_vary_target((0.0, -0.3, 1.2), 0.05),  # behind
            left_foot_target=left_foot,
            right_foot_target=right_foot,
        )
    )

    # Side leans
    bins.append(
        IKTestCase(
            goal_type="LEAN",
            description="LEAN LEFT",
            tags={"region": "lean", "dir": "left"},
            lean_target=_vary_target((-0.5, 0.2, 1.2), 0.05),  # left side
            left_foot_target=left_foot,
            right_foot_target=right_foot,
        )
    )
    bins.append(
        IKTestCase(
            goal_type="LEAN",
            description="LEAN RIGHT",
            tags={"region": "lean", "dir": "right"},
            lean_target=_vary_target((0.5, 0.2, 1.2), 0.05),  # right side
            left_foot_target=left_foot,
            right_foot_target=right_foot,
        )
    )

    return bins


def _combined_bins(ground_z: float = 0.0, shoulder_height: float = 1.5) -> List[IKTestCase]:
    """
    Generate combined full-body tests - real-world poses (with slight variation).

    These test multiple systems working together:
    - Arms + Look (reach and watch hand)
    - Crouch + Reach (pick up object)
    - Lean + Arm (far reach)
    - Full body coordination
    """
    bins = []
    foot_spacing = 0.1
    left_foot = _vary_target((-foot_spacing, 0.0, ground_z), 0.02)
    right_foot = _vary_target((foot_spacing, 0.0, ground_z), 0.02)

    # === PICK UP OBJECT (crouch + reach down + look at hand + LEAN FORWARD) ===
    # CRITICAL: lean_target is required for the spine to bend forward when reaching ground!
    pickup_target = _vary_target((0.0, 0.3, ground_z + 0.1), 0.03)  # near ground, in front
    bins.append(
        IKTestCase(
            goal_type="COMBINED",
            description="PICK UP OBJECT",
            tags={"region": "combined", "action": "Pick up from ground"},
            hip_drop=_vary(0.35, 0.03),
            left_foot_target=left_foot,
            right_foot_target=right_foot,
            right_hand_target=pickup_target,
            look_at_target=pickup_target,
            lean_target=_vary_target((0.0, 0.4, ground_z + 0.3), 0.03),  # lean forward toward pickup
        )
    )

    # === REACH HIGH SHELF (arm up + look up + slight lean) ===
    shelf_target = _vary_target((0.0, 0.3, shoulder_height + 0.5), 0.04)  # high shelf
    bins.append(
        IKTestCase(
            goal_type="COMBINED",
            description="REACH HIGH SHELF",
            tags={"region": "combined", "action": "Reach high shelf"},
            right_hand_target=shelf_target,
            look_at_target=shelf_target,
            lean_target=_vary_target((0.0, 0.3, shoulder_height + 0.3), 0.03),
        )
    )

    # === TWO HAND CARRY (both hands together, looking forward) ===
    carry_height = _vary(1.0, 0.05)  # waist height
    bins.append(
        IKTestCase(
            goal_type="COMBINED",
            description="TWO HAND CARRY",
            tags={"region": "combined", "action": "Carry with both hands"},
            left_hand_target=_vary_target((-0.15, 0.35, carry_height), 0.02),
            right_hand_target=_vary_target((0.15, 0.35, carry_height), 0.02),
            look_at_target=_vary_target((0.0, 2.0, 1.6), 0.05),  # looking ahead
        )
    )

    # === PUSH FORWARD (lean + both arms forward) ===
    push_target_y = _vary(0.5, 0.03)
    push_height = _vary(1.2, 0.03)
    bins.append(
        IKTestCase(
            goal_type="COMBINED",
            description="PUSH FORWARD",
            tags={"region": "combined", "action": "Push something forward"},
            left_hand_target=_vary_target((-0.2, push_target_y, push_height), 0.02),
            right_hand_target=_vary_target((0.2, push_target_y, push_height), 0.02),
            lean_target=(0.0, push_target_y, push_height),
            left_foot_target=left_foot,
            right_foot_target=right_foot,
        )
    )

    # === CROUCH AND LOOK UP (defensive/hiding pose) ===
    bins.append(
        IKTestCase(
            goal_type="COMBINED",
            description="CROUCH LOOK UP",
            tags={"region": "combined", "action": "Crouch and look up"},
            hip_drop=_vary(0.3, 0.03),
            left_foot_target=left_foot,
            right_foot_target=right_foot,
            look_at_target=_vary_target((0.0, 1.0, 3.0), 0.1),  # looking up
        )
    )

    # === REACH ACROSS BODY (right hand to left side) ===
    bins.append(
        IKTestCase(
            goal_type="COMBINED",
            description="REACH ACROSS LEFT",
            tags={"region": "combined", "action": "Right hand reaches left"},
            right_hand_target=_vary_target((-0.4, 0.3, 1.2), 0.03),  # across to left side
            lean_target=_vary_target((-0.3, 0.2, 1.3), 0.03),  # lean to help
            look_at_target=_vary_target((-0.4, 0.3, 1.2), 0.03),  # look at target
        )
    )
    bins.append(
        IKTestCase(
            goal_type="COMBINED",
            description="REACH ACROSS RIGHT",
            tags={"region": "combined", "action": "Left hand reaches right"},
            left_hand_target=_vary_target((0.4, 0.3, 1.2), 0.03),  # across to right side
            lean_target=_vary_target((0.3, 0.2, 1.3), 0.03),
            look_at_target=_vary_target((0.4, 0.3, 1.2), 0.03),
        )
    )

    # === STEP AND REACH (one foot forward, reach far) ===
    step_foot = _vary_target((0.0, 0.4, ground_z), 0.02)  # stepped forward
    far_reach = _vary_target((0.0, 0.7, 1.3), 0.04)  # far forward reach
    bins.append(
        IKTestCase(
            goal_type="COMBINED",
            description="STEP AND REACH",
            tags={"region": "combined", "action": "Step forward and reach"},
            left_foot_target=left_foot,  # back foot planted
            right_foot_target=step_foot,  # front foot stepped
            right_hand_target=far_reach,
            lean_target=_vary_target((0.0, 0.5, 1.2), 0.03),
            look_at_target=far_reach,
        )
    )

    return bins


# =============================================================================
# ATHLETIC / DYNAMIC POSES
# =============================================================================

def _athletic_bins(ground_z: float = 0.0, shoulder_height: float = 1.5) -> List[IKTestCase]:
    """
    Athletic action poses - throwing, punching, kicking (with slight variation).
    Tests dynamic weight transfer and extreme limb positions.
    """
    bins = []

    # === THROWING POSE (wind-up) ===
    # Right arm back, left arm forward for balance, weight on back foot
    bins.append(
        IKTestCase(
            goal_type="ATHLETIC",
            description="THROW WIND-UP",
            tags={"region": "athletic", "action": "Throwing wind-up"},
            left_foot_target=_vary_target((-0.15, 0.3, ground_z), 0.02),  # front foot forward
            right_foot_target=_vary_target((0.15, -0.1, ground_z), 0.02),  # back foot planted
            left_hand_target=_vary_target((-0.3, 0.4, shoulder_height), 0.03),  # left arm forward for balance
            right_hand_target=_vary_target((0.4, -0.3, shoulder_height + 0.3), 0.03),  # right arm back and up
            lean_target=_vary_target((0.0, -0.1, shoulder_height), 0.03),  # lean back slightly
            look_at_target=_vary_target((-0.3, 2.0, shoulder_height), 0.05),  # looking at target
        )
    )

    # === THROWING POSE (release) ===
    bins.append(
        IKTestCase(
            goal_type="ATHLETIC",
            description="THROW RELEASE",
            tags={"region": "athletic", "action": "Throwing release"},
            left_foot_target=_vary_target((-0.15, 0.2, ground_z), 0.02),
            right_foot_target=_vary_target((0.15, 0.1, ground_z), 0.02),  # weight transferred forward
            left_hand_target=_vary_target((-0.2, 0.2, shoulder_height - 0.2), 0.03),  # left arm tucked
            right_hand_target=_vary_target((0.1, 0.6, shoulder_height + 0.1), 0.03),  # right arm extended forward
            lean_target=_vary_target((0.0, 0.4, shoulder_height), 0.03),  # lean forward
            look_at_target=_vary_target((0.0, 2.0, shoulder_height), 0.05),
        )
    )

    # === PUNCH (jab) ===
    bins.append(
        IKTestCase(
            goal_type="ATHLETIC",
            description="PUNCH JAB",
            tags={"region": "athletic", "action": "Jab punch"},
            left_foot_target=_vary_target((-0.12, 0.15, ground_z), 0.02),
            right_foot_target=_vary_target((0.12, -0.05, ground_z), 0.02),
            left_hand_target=_vary_target((-0.05, 0.65, shoulder_height), 0.03),  # left jab extended
            right_hand_target=_vary_target((0.15, 0.1, shoulder_height - 0.1), 0.02),  # right guard
            lean_target=_vary_target((0.0, 0.2, shoulder_height), 0.03),
            look_at_target=_vary_target((0.0, 2.0, shoulder_height), 0.05),
        )
    )

    # === PUNCH (cross) ===
    bins.append(
        IKTestCase(
            goal_type="ATHLETIC",
            description="PUNCH CROSS",
            tags={"region": "athletic", "action": "Cross punch"},
            left_foot_target=_vary_target((-0.12, 0.1, ground_z), 0.02),
            right_foot_target=_vary_target((0.12, 0.05, ground_z), 0.02),
            left_hand_target=_vary_target((-0.15, 0.15, shoulder_height), 0.02),  # left guard
            right_hand_target=_vary_target((0.05, 0.7, shoulder_height), 0.03),  # right cross extended
            lean_target=_vary_target((0.0, 0.3, shoulder_height), 0.03),  # rotate into punch
            look_at_target=_vary_target((0.0, 2.0, shoulder_height), 0.05),
        )
    )

    # === KICK (front) ===
    bins.append(
        IKTestCase(
            goal_type="ATHLETIC",
            description="FRONT KICK",
            tags={"region": "athletic", "action": "Front kick"},
            left_foot_target=_vary_target((-0.1, 0.0, ground_z), 0.02),  # planted
            right_foot_target=_vary_target((0.1, 0.5, shoulder_height - 0.5), 0.04),  # kicking forward and up
            left_hand_target=_vary_target((-0.3, 0.1, shoulder_height), 0.03),  # arms for balance
            right_hand_target=_vary_target((0.3, 0.1, shoulder_height), 0.03),
            lean_target=_vary_target((0.0, -0.1, shoulder_height), 0.03),  # lean back for balance
            look_at_target=_vary_target((0.0, 2.0, shoulder_height - 0.3), 0.05),
        )
    )

    # === KICK (side) ===
    bins.append(
        IKTestCase(
            goal_type="ATHLETIC",
            description="SIDE KICK",
            tags={"region": "athletic", "action": "Side kick"},
            left_foot_target=_vary_target((-0.1, 0.0, ground_z), 0.02),  # planted
            right_foot_target=_vary_target((0.6, 0.1, shoulder_height - 0.5), 0.04),  # kicking to side
            left_hand_target=_vary_target((-0.25, 0.2, shoulder_height + 0.1), 0.03),  # arms up for balance
            right_hand_target=_vary_target((0.1, 0.2, shoulder_height), 0.03),
            lean_target=_vary_target((-0.2, 0.0, shoulder_height), 0.03),  # lean away from kick
            look_at_target=_vary_target((0.5, 0.5, shoulder_height), 0.05),  # looking at target
        )
    )

    return bins


def _extreme_reach_bins(ground_z: float = 0.0, shoulder_height: float = 1.5) -> List[IKTestCase]:
    """
    Maximum extension tests - pushing IK to its limits (with slight variation).
    """
    bins = []
    foot_spacing = 0.1
    left_foot = _vary_target((-foot_spacing, 0.0, ground_z), 0.02)
    right_foot = _vary_target((foot_spacing, 0.0, ground_z), 0.02)
    max_arm_reach = 0.55  # approximate full arm extension

    # === REACH MAXIMUM FORWARD ===
    bins.append(
        IKTestCase(
            goal_type="EXTREME",
            description="MAX REACH FORWARD",
            tags={"region": "extreme", "action": "Maximum forward reach"},
            left_foot_target=left_foot,
            right_foot_target=right_foot,
            right_hand_target=_vary_target((0.0, max_arm_reach + 0.1, shoulder_height), 0.03),
            lean_target=_vary_target((0.0, 0.4, shoulder_height - 0.2), 0.03),
            look_at_target=_vary_target((0.0, 1.0, shoulder_height), 0.05),
        )
    )

    # === REACH MAXIMUM UP ===
    bins.append(
        IKTestCase(
            goal_type="EXTREME",
            description="MAX REACH UP",
            tags={"region": "extreme", "action": "Maximum upward reach"},
            left_foot_target=left_foot,
            right_foot_target=right_foot,
            right_hand_target=_vary_target((0.1, 0.2, shoulder_height + max_arm_reach), 0.03),
            left_hand_target=_vary_target((-0.1, 0.2, shoulder_height + max_arm_reach - 0.05), 0.03),
            lean_target=_vary_target((0.0, 0.1, shoulder_height + 0.3), 0.03),
            look_at_target=_vary_target((0.0, 0.5, shoulder_height + 0.5), 0.05),
        )
    )

    # === REACH MAXIMUM DOWN (without crouch) ===
    bins.append(
        IKTestCase(
            goal_type="EXTREME",
            description="MAX REACH DOWN STANDING",
            tags={"region": "extreme", "action": "Reach down while standing"},
            left_foot_target=left_foot,
            right_foot_target=right_foot,
            right_hand_target=_vary_target((0.1, 0.3, ground_z + 0.3), 0.03),  # trying to reach low
            lean_target=_vary_target((0.0, 0.4, shoulder_height - 0.4), 0.03),  # heavy forward lean
            look_at_target=_vary_target((0.0, 0.5, ground_z + 0.3), 0.05),
        )
    )

    # === REACH BEHIND ===
    bins.append(
        IKTestCase(
            goal_type="EXTREME",
            description="REACH BEHIND",
            tags={"region": "extreme", "action": "Reach behind back"},
            left_foot_target=left_foot,
            right_foot_target=right_foot,
            right_hand_target=_vary_target((0.2, -0.3, shoulder_height - 0.3), 0.03),  # behind and low
            lean_target=_vary_target((0.0, -0.1, shoulder_height), 0.03),
            look_at_target=_vary_target((0.3, -0.5, shoulder_height), 0.05),  # looking back
        )
    )

    # === WIDE STANCE REACH ===
    bins.append(
        IKTestCase(
            goal_type="EXTREME",
            description="WIDE STANCE REACH",
            tags={"region": "extreme", "action": "Wide stance side reach"},
            left_foot_target=_vary_target((-0.35, 0.0, ground_z), 0.02),  # wide left
            right_foot_target=_vary_target((0.35, 0.0, ground_z), 0.02),  # wide right
            hip_drop=_vary(0.15, 0.02),  # slight squat from wide stance
            right_hand_target=_vary_target((0.6, 0.2, shoulder_height - 0.2), 0.03),  # far right reach
            lean_target=_vary_target((0.3, 0.1, shoulder_height - 0.1), 0.03),
            look_at_target=_vary_target((0.6, 0.5, shoulder_height), 0.05),
        )
    )

    return bins


def _balance_bins(ground_z: float = 0.0, shoulder_height: float = 1.5) -> List[IKTestCase]:
    """
    Balance tests - one leg, unstable positions (with slight variation).
    """
    bins = []

    # === ONE LEG STAND (right leg up) ===
    bins.append(
        IKTestCase(
            goal_type="BALANCE",
            description="ONE LEG STAND R",
            tags={"region": "balance", "action": "Stand on left leg"},
            left_foot_target=_vary_target((-0.05, 0.0, ground_z), 0.02),  # planted
            right_foot_target=_vary_target((0.15, 0.2, ground_z + 0.3), 0.03),  # lifted forward
            left_hand_target=_vary_target((-0.3, 0.1, shoulder_height), 0.03),  # arms out for balance
            right_hand_target=_vary_target((0.3, 0.1, shoulder_height), 0.03),
            look_at_target=_vary_target((0.0, 2.0, shoulder_height), 0.05),
        )
    )

    # === ONE LEG STAND (left leg up) ===
    bins.append(
        IKTestCase(
            goal_type="BALANCE",
            description="ONE LEG STAND L",
            tags={"region": "balance", "action": "Stand on right leg"},
            left_foot_target=_vary_target((-0.15, 0.2, ground_z + 0.3), 0.03),  # lifted forward
            right_foot_target=_vary_target((0.05, 0.0, ground_z), 0.02),  # planted
            left_hand_target=_vary_target((-0.3, 0.1, shoulder_height), 0.03),
            right_hand_target=_vary_target((0.3, 0.1, shoulder_height), 0.03),
            look_at_target=_vary_target((0.0, 2.0, shoulder_height), 0.05),
        )
    )

    # === TREE POSE (yoga) ===
    bins.append(
        IKTestCase(
            goal_type="BALANCE",
            description="TREE POSE",
            tags={"region": "balance", "action": "Yoga tree pose"},
            left_foot_target=_vary_target((-0.05, 0.0, ground_z), 0.02),  # planted
            right_foot_target=_vary_target((0.0, 0.0, ground_z + 0.4), 0.03),  # foot against inner thigh (approximate)
            left_hand_target=_vary_target((-0.1, 0.15, shoulder_height + 0.4), 0.03),  # hands together above head
            right_hand_target=_vary_target((0.1, 0.15, shoulder_height + 0.4), 0.03),
            look_at_target=_vary_target((0.0, 2.0, shoulder_height), 0.05),
        )
    )

    # === WARRIOR POSE ===
    bins.append(
        IKTestCase(
            goal_type="BALANCE",
            description="WARRIOR POSE",
            tags={"region": "balance", "action": "Warrior yoga pose"},
            left_foot_target=_vary_target((-0.1, 0.5, ground_z), 0.02),  # front foot forward
            right_foot_target=_vary_target((0.1, -0.4, ground_z), 0.02),  # back foot back
            hip_drop=_vary(0.2, 0.02),  # lunge depth
            left_hand_target=_vary_target((-0.5, 0.3, shoulder_height), 0.03),  # arms out to sides
            right_hand_target=_vary_target((0.5, -0.2, shoulder_height), 0.03),
            lean_target=_vary_target((0.0, 0.2, shoulder_height), 0.03),
            look_at_target=_vary_target((0.0, 2.0, shoulder_height), 0.05),
        )
    )

    # === LUNGE DEEP ===
    bins.append(
        IKTestCase(
            goal_type="BALANCE",
            description="DEEP LUNGE",
            tags={"region": "balance", "action": "Deep forward lunge"},
            left_foot_target=_vary_target((-0.1, 0.0, ground_z), 0.02),  # back foot
            right_foot_target=_vary_target((0.1, 0.6, ground_z), 0.02),  # front foot far forward
            hip_drop=_vary(0.35, 0.03),
            left_hand_target=_vary_target((-0.2, 0.3, shoulder_height - 0.3), 0.03),
            right_hand_target=_vary_target((0.2, 0.3, shoulder_height - 0.3), 0.03),
            lean_target=_vary_target((0.0, 0.3, shoulder_height - 0.2), 0.03),
            look_at_target=_vary_target((0.0, 2.0, shoulder_height), 0.05),
        )
    )

    return bins


def _interaction_bins(ground_z: float = 0.0, shoulder_height: float = 1.5) -> List[IKTestCase]:
    """
    Real-world interaction poses - doors, climbing, pulling (with slight variation).
    """
    bins = []
    foot_spacing = 0.1
    left_foot = _vary_target((-foot_spacing, 0.0, ground_z), 0.02)
    right_foot = _vary_target((foot_spacing, 0.0, ground_z), 0.02)

    # === OPEN DOOR (handle height) ===
    bins.append(
        IKTestCase(
            goal_type="INTERACT",
            description="OPEN DOOR",
            tags={"region": "interact", "action": "Grabbing door handle"},
            left_foot_target=left_foot,
            right_foot_target=right_foot,
            right_hand_target=_vary_target((0.3, 0.45, shoulder_height - 0.4), 0.03),  # door handle height
            lean_target=_vary_target((0.1, 0.2, shoulder_height), 0.03),
            look_at_target=_vary_target((0.2, 0.5, shoulder_height - 0.2), 0.05),
        )
    )

    # === PULL ROPE/LEVER ===
    bins.append(
        IKTestCase(
            goal_type="INTERACT",
            description="PULL LEVER",
            tags={"region": "interact", "action": "Pulling lever down"},
            left_foot_target=_vary_target((-0.15, 0.1, ground_z), 0.02),
            right_foot_target=_vary_target((0.15, -0.1, ground_z), 0.02),  # braced stance
            right_hand_target=_vary_target((0.2, 0.35, shoulder_height + 0.2), 0.03),  # lever up high
            lean_target=_vary_target((0.0, 0.1, shoulder_height), 0.03),
            look_at_target=_vary_target((0.2, 0.4, shoulder_height + 0.3), 0.05),
        )
    )

    # === CLIMB UP (reaching for ledge) ===
    bins.append(
        IKTestCase(
            goal_type="INTERACT",
            description="CLIMB REACH UP",
            tags={"region": "interact", "action": "Reaching up for ledge"},
            left_foot_target=_vary_target((-0.1, 0.1, ground_z + 0.3), 0.02),  # foot on wall
            right_foot_target=_vary_target((0.1, 0.0, ground_z), 0.02),  # foot on ground
            left_hand_target=_vary_target((-0.15, 0.25, shoulder_height + 0.5), 0.03),  # reaching up
            right_hand_target=_vary_target((0.15, 0.25, shoulder_height + 0.5), 0.03),
            lean_target=_vary_target((0.0, 0.15, shoulder_height + 0.2), 0.03),
            look_at_target=_vary_target((0.0, 0.3, shoulder_height + 0.6), 0.05),
        )
    )

    # === PUSH HEAVY OBJECT ===
    bins.append(
        IKTestCase(
            goal_type="INTERACT",
            description="PUSH HEAVY",
            tags={"region": "interact", "action": "Pushing heavy object"},
            left_foot_target=_vary_target((-0.15, -0.2, ground_z), 0.02),  # feet back for leverage
            right_foot_target=_vary_target((0.15, -0.1, ground_z), 0.02),
            left_hand_target=_vary_target((-0.2, 0.5, shoulder_height - 0.2), 0.03),
            right_hand_target=_vary_target((0.2, 0.5, shoulder_height - 0.2), 0.03),
            lean_target=_vary_target((0.0, 0.5, shoulder_height - 0.3), 0.03),  # really leaning in
            look_at_target=_vary_target((0.0, 1.0, shoulder_height), 0.05),
        )
    )

    # === PULL HEAVY (tug of war) ===
    bins.append(
        IKTestCase(
            goal_type="INTERACT",
            description="PULL HEAVY",
            tags={"region": "interact", "action": "Pulling/tugging backward"},
            left_foot_target=_vary_target((-0.15, 0.3, ground_z), 0.02),  # feet forward
            right_foot_target=_vary_target((0.15, 0.2, ground_z), 0.02),
            hip_drop=_vary(0.2, 0.02),  # squat for power
            left_hand_target=_vary_target((-0.1, 0.5, shoulder_height - 0.3), 0.03),  # hands forward gripping
            right_hand_target=_vary_target((0.1, 0.5, shoulder_height - 0.3), 0.03),
            lean_target=_vary_target((0.0, -0.2, shoulder_height - 0.2), 0.03),  # leaning back
            look_at_target=_vary_target((0.0, 1.0, shoulder_height), 0.05),
        )
    )

    # === CARRY ON SHOULDER ===
    bins.append(
        IKTestCase(
            goal_type="INTERACT",
            description="SHOULDER CARRY",
            tags={"region": "interact", "action": "Carrying on shoulder"},
            left_foot_target=left_foot,
            right_foot_target=right_foot,
            left_hand_target=_vary_target((-0.1, 0.1, shoulder_height + 0.2), 0.03),  # stabilizing load
            right_hand_target=_vary_target((0.3, 0.0, shoulder_height + 0.3), 0.03),  # on shoulder
            lean_target=_vary_target((-0.15, 0.0, shoulder_height), 0.03),  # leaning to compensate
            look_at_target=_vary_target((0.0, 2.0, shoulder_height), 0.05),
        )
    )

    return bins


def _spine_stress_bins(ground_z: float = 0.0, shoulder_height: float = 1.5) -> List[IKTestCase]:
    """
    Extreme spine/torso tests - pushing spinal flexibility limits.
    """
    bins = []
    foot_spacing = 0.12
    left_foot = _vary_target((-foot_spacing, 0.0, ground_z), 0.02)
    right_foot = _vary_target((foot_spacing, 0.0, ground_z), 0.02)

    # === TOUCH TOES (extreme forward bend) ===
    bins.append(
        IKTestCase(
            goal_type="SPINE_STRESS",
            description="TOUCH TOES",
            tags={"region": "spine_stress", "action": "Touch toes forward bend"},
            left_foot_target=left_foot,
            right_foot_target=right_foot,
            left_hand_target=_vary_target((-0.1, 0.15, ground_z + 0.05), 0.02),
            right_hand_target=_vary_target((0.1, 0.15, ground_z + 0.05), 0.02),
            lean_target=_vary_target((0.0, 0.5, ground_z + 0.3), 0.03),
            look_at_target=_vary_target((0.0, 0.3, ground_z), 0.03),
        )
    )

    # === BACK BEND (looking at ceiling) ===
    bins.append(
        IKTestCase(
            goal_type="SPINE_STRESS",
            description="BACK BEND",
            tags={"region": "spine_stress", "action": "Lean back look up"},
            left_foot_target=_vary_target((-foot_spacing, 0.1, ground_z), 0.02),
            right_foot_target=_vary_target((foot_spacing, 0.1, ground_z), 0.02),
            lean_target=_vary_target((0.0, -0.4, shoulder_height + 0.2), 0.03),
            look_at_target=_vary_target((0.0, -0.5, shoulder_height + 1.0), 0.05),
        )
    )

    # === EXTREME SIDE BEND LEFT ===
    bins.append(
        IKTestCase(
            goal_type="SPINE_STRESS",
            description="SIDE BEND LEFT EXTREME",
            tags={"region": "spine_stress", "action": "Extreme left side bend"},
            left_foot_target=left_foot,
            right_foot_target=right_foot,
            left_hand_target=_vary_target((-0.15, 0.1, ground_z + 0.4), 0.03),  # reaching down left side
            right_hand_target=_vary_target((0.3, 0.1, shoulder_height + 0.5), 0.03),  # reaching over head
            lean_target=_vary_target((-0.5, 0.0, shoulder_height - 0.3), 0.03),
            look_at_target=_vary_target((-0.5, 1.0, shoulder_height), 0.05),
        )
    )

    # === EXTREME SIDE BEND RIGHT ===
    bins.append(
        IKTestCase(
            goal_type="SPINE_STRESS",
            description="SIDE BEND RIGHT EXTREME",
            tags={"region": "spine_stress", "action": "Extreme right side bend"},
            left_foot_target=left_foot,
            right_foot_target=right_foot,
            left_hand_target=_vary_target((-0.3, 0.1, shoulder_height + 0.5), 0.03),
            right_hand_target=_vary_target((0.15, 0.1, ground_z + 0.4), 0.03),
            lean_target=_vary_target((0.5, 0.0, shoulder_height - 0.3), 0.03),
            look_at_target=_vary_target((0.5, 1.0, shoulder_height), 0.05),
        )
    )

    # === TWIST LEFT (rotation) ===
    bins.append(
        IKTestCase(
            goal_type="SPINE_STRESS",
            description="TORSO TWIST LEFT",
            tags={"region": "spine_stress", "action": "Twist torso left"},
            left_foot_target=left_foot,
            right_foot_target=right_foot,
            left_hand_target=_vary_target((-0.4, -0.3, shoulder_height), 0.03),  # reaching back left
            right_hand_target=_vary_target((0.2, 0.4, shoulder_height), 0.03),   # reaching forward right
            lean_target=_vary_target((-0.2, -0.1, shoulder_height), 0.03),
            look_at_target=_vary_target((-0.8, -0.5, shoulder_height), 0.05),  # looking behind
        )
    )

    # === TWIST RIGHT (rotation) ===
    bins.append(
        IKTestCase(
            goal_type="SPINE_STRESS",
            description="TORSO TWIST RIGHT",
            tags={"region": "spine_stress", "action": "Twist torso right"},
            left_foot_target=left_foot,
            right_foot_target=right_foot,
            left_hand_target=_vary_target((-0.2, 0.4, shoulder_height), 0.03),
            right_hand_target=_vary_target((0.4, -0.3, shoulder_height), 0.03),
            lean_target=_vary_target((0.2, -0.1, shoulder_height), 0.03),
            look_at_target=_vary_target((0.8, -0.5, shoulder_height), 0.05),
        )
    )

    return bins


def _neck_stress_bins(ground_z: float = 0.0, shoulder_height: float = 1.5) -> List[IKTestCase]:
    """
    Extreme neck/head tests - pushing neck rotation limits.
    """
    bins = []
    foot_spacing = 0.1
    left_foot = _vary_target((-foot_spacing, 0.0, ground_z), 0.02)
    right_foot = _vary_target((foot_spacing, 0.0, ground_z), 0.02)

    # === LOOK STRAIGHT UP (ceiling) ===
    bins.append(
        IKTestCase(
            goal_type="NECK_STRESS",
            description="LOOK STRAIGHT UP",
            tags={"region": "neck_stress", "action": "Head tilted max up"},
            left_foot_target=left_foot,
            right_foot_target=right_foot,
            look_at_target=_vary_target((0.0, 0.1, shoulder_height + 2.0), 0.05),
        )
    )

    # === LOOK STRAIGHT DOWN (at feet) ===
    bins.append(
        IKTestCase(
            goal_type="NECK_STRESS",
            description="LOOK AT FEET",
            tags={"region": "neck_stress", "action": "Head tilted max down"},
            left_foot_target=left_foot,
            right_foot_target=right_foot,
            look_at_target=_vary_target((0.0, 0.3, ground_z), 0.03),
        )
    )

    # === MAX LEFT TURN ===
    bins.append(
        IKTestCase(
            goal_type="NECK_STRESS",
            description="HEAD MAX LEFT",
            tags={"region": "neck_stress", "action": "Head turned max left"},
            left_foot_target=left_foot,
            right_foot_target=right_foot,
            look_at_target=_vary_target((-2.0, 0.0, shoulder_height + 0.2), 0.05),
        )
    )

    # === MAX RIGHT TURN ===
    bins.append(
        IKTestCase(
            goal_type="NECK_STRESS",
            description="HEAD MAX RIGHT",
            tags={"region": "neck_stress", "action": "Head turned max right"},
            left_foot_target=left_foot,
            right_foot_target=right_foot,
            look_at_target=_vary_target((2.0, 0.0, shoulder_height + 0.2), 0.05),
        )
    )

    # === LOOK BEHIND LEFT (over shoulder extreme) ===
    bins.append(
        IKTestCase(
            goal_type="NECK_STRESS",
            description="LOOK BEHIND LEFT",
            tags={"region": "neck_stress", "action": "Look behind over left shoulder"},
            left_foot_target=left_foot,
            right_foot_target=right_foot,
            look_at_target=_vary_target((-1.0, -1.5, shoulder_height), 0.05),
        )
    )

    # === LOOK BEHIND RIGHT ===
    bins.append(
        IKTestCase(
            goal_type="NECK_STRESS",
            description="LOOK BEHIND RIGHT",
            tags={"region": "neck_stress", "action": "Look behind over right shoulder"},
            left_foot_target=left_foot,
            right_foot_target=right_foot,
            look_at_target=_vary_target((1.0, -1.5, shoulder_height), 0.05),
        )
    )

    # === HEAD TILT + TURN COMBO ===
    bins.append(
        IKTestCase(
            goal_type="NECK_STRESS",
            description="HEAD TILT LEFT LOOK UP",
            tags={"region": "neck_stress", "action": "Tilt head left while looking up"},
            left_foot_target=left_foot,
            right_foot_target=right_foot,
            look_at_target=_vary_target((-0.8, 0.5, shoulder_height + 1.5), 0.05),
        )
    )

    return bins


def _shoulder_stress_bins(ground_z: float = 0.0, shoulder_height: float = 1.5) -> List[IKTestCase]:
    """
    Extreme shoulder/arm position tests - testing shoulder joint limits.
    """
    bins = []
    foot_spacing = 0.1
    left_foot = _vary_target((-foot_spacing, 0.0, ground_z), 0.02)
    right_foot = _vary_target((foot_spacing, 0.0, ground_z), 0.02)

    # === ARMS STRAIGHT UP (surrender) ===
    bins.append(
        IKTestCase(
            goal_type="SHOULDER_STRESS",
            description="ARMS STRAIGHT UP",
            tags={"region": "shoulder_stress", "action": "Both arms straight up"},
            left_foot_target=left_foot,
            right_foot_target=right_foot,
            left_hand_target=_vary_target((-0.15, 0.1, shoulder_height + 0.6), 0.03),
            right_hand_target=_vary_target((0.15, 0.1, shoulder_height + 0.6), 0.03),
            look_at_target=_vary_target((0.0, 2.0, shoulder_height), 0.05),
        )
    )

    # === HANDS BEHIND HEAD ===
    bins.append(
        IKTestCase(
            goal_type="SHOULDER_STRESS",
            description="HANDS BEHIND HEAD",
            tags={"region": "shoulder_stress", "action": "Hands clasped behind head"},
            left_foot_target=left_foot,
            right_foot_target=right_foot,
            left_hand_target=_vary_target((-0.08, -0.15, shoulder_height + 0.25), 0.02),
            right_hand_target=_vary_target((0.08, -0.15, shoulder_height + 0.25), 0.02),
            look_at_target=_vary_target((0.0, 2.0, shoulder_height), 0.05),
        )
    )

    # === HANDS BEHIND BACK (low) ===
    bins.append(
        IKTestCase(
            goal_type="SHOULDER_STRESS",
            description="HANDS BEHIND BACK",
            tags={"region": "shoulder_stress", "action": "Hands clasped behind back"},
            left_foot_target=left_foot,
            right_foot_target=right_foot,
            left_hand_target=_vary_target((0.05, -0.2, shoulder_height - 0.5), 0.02),
            right_hand_target=_vary_target((-0.05, -0.2, shoulder_height - 0.5), 0.02),
            look_at_target=_vary_target((0.0, 2.0, shoulder_height), 0.05),
        )
    )

    # === CHICKEN WING LEFT (elbow up, hand on shoulder) ===
    bins.append(
        IKTestCase(
            goal_type="SHOULDER_STRESS",
            description="CHICKEN WING LEFT",
            tags={"region": "shoulder_stress", "action": "Left elbow up hand on shoulder"},
            left_foot_target=left_foot,
            right_foot_target=right_foot,
            left_hand_target=_vary_target((-0.12, -0.08, shoulder_height + 0.1), 0.02),
            right_hand_target=_vary_target((0.25, 0.15, shoulder_height - 0.2), 0.03),
            look_at_target=_vary_target((0.3, 2.0, shoulder_height), 0.05),
        )
    )

    # === CHICKEN WING RIGHT ===
    bins.append(
        IKTestCase(
            goal_type="SHOULDER_STRESS",
            description="CHICKEN WING RIGHT",
            tags={"region": "shoulder_stress", "action": "Right elbow up hand on shoulder"},
            left_foot_target=left_foot,
            right_foot_target=right_foot,
            left_hand_target=_vary_target((-0.25, 0.15, shoulder_height - 0.2), 0.03),
            right_hand_target=_vary_target((0.12, -0.08, shoulder_height + 0.1), 0.02),
            look_at_target=_vary_target((-0.3, 2.0, shoulder_height), 0.05),
        )
    )

    # === ARMS WIDE T-POSE MAX ===
    bins.append(
        IKTestCase(
            goal_type="SHOULDER_STRESS",
            description="ARMS WIDE T-POSE",
            tags={"region": "shoulder_stress", "action": "Arms extended max sideways"},
            left_foot_target=left_foot,
            right_foot_target=right_foot,
            left_hand_target=_vary_target((-0.7, 0.05, shoulder_height), 0.03),
            right_hand_target=_vary_target((0.7, 0.05, shoulder_height), 0.03),
            look_at_target=_vary_target((0.0, 2.0, shoulder_height), 0.05),
        )
    )

    # === REACH BEHIND FOR ZIPPER ===
    bins.append(
        IKTestCase(
            goal_type="SHOULDER_STRESS",
            description="REACH BEHIND BACK UP",
            tags={"region": "shoulder_stress", "action": "Reach up behind back"},
            left_foot_target=left_foot,
            right_foot_target=right_foot,
            left_hand_target=_vary_target((-0.2, 0.2, shoulder_height), 0.03),
            right_hand_target=_vary_target((0.0, -0.2, shoulder_height - 0.1), 0.02),  # reaching up spine
            look_at_target=_vary_target((0.0, 2.0, shoulder_height), 0.05),
        )
    )

    return bins


def _extreme_arm_bins(ground_z: float = 0.0, shoulder_height: float = 1.5) -> List[IKTestCase]:
    """
    Extreme arm reach tests - pushing arm IK to absolute limits.
    """
    bins = []
    foot_spacing = 0.1
    left_foot = _vary_target((-foot_spacing, 0.0, ground_z), 0.02)
    right_foot = _vary_target((foot_spacing, 0.0, ground_z), 0.02)
    max_reach = 0.56  # arm length

    # === REACH OVERHEAD MAX ===
    bins.append(
        IKTestCase(
            goal_type="EXTREME_ARM",
            description="REACH OVERHEAD MAX",
            tags={"region": "extreme_arm", "action": "Maximum overhead reach"},
            left_foot_target=left_foot,
            right_foot_target=right_foot,
            right_hand_target=_vary_target((0.0, 0.15, shoulder_height + max_reach + 0.1), 0.02),
            lean_target=_vary_target((0.0, 0.1, shoulder_height + 0.3), 0.03),
            look_at_target=_vary_target((0.0, 0.2, shoulder_height + 0.8), 0.05),
        )
    )

    # === REACH BEHIND MAX ===
    bins.append(
        IKTestCase(
            goal_type="EXTREME_ARM",
            description="REACH BEHIND MAX",
            tags={"region": "extreme_arm", "action": "Maximum reach behind"},
            left_foot_target=left_foot,
            right_foot_target=right_foot,
            right_hand_target=_vary_target((0.2, -0.45, shoulder_height - 0.2), 0.03),
            lean_target=_vary_target((0.0, -0.15, shoulder_height), 0.03),
            look_at_target=_vary_target((0.3, -1.0, shoulder_height), 0.05),
        )
    )

    # === CROSS BODY FAR (left hand far right) ===
    bins.append(
        IKTestCase(
            goal_type="EXTREME_ARM",
            description="CROSS BODY FAR LEFT",
            tags={"region": "extreme_arm", "action": "Left arm cross far right"},
            left_foot_target=left_foot,
            right_foot_target=right_foot,
            left_hand_target=_vary_target((0.45, 0.25, shoulder_height - 0.1), 0.03),
            lean_target=_vary_target((0.25, 0.1, shoulder_height), 0.03),
            look_at_target=_vary_target((0.5, 0.5, shoulder_height), 0.05),
        )
    )

    # === CROSS BODY FAR (right hand far left) ===
    bins.append(
        IKTestCase(
            goal_type="EXTREME_ARM",
            description="CROSS BODY FAR RIGHT",
            tags={"region": "extreme_arm", "action": "Right arm cross far left"},
            left_foot_target=left_foot,
            right_foot_target=right_foot,
            right_hand_target=_vary_target((-0.45, 0.25, shoulder_height - 0.1), 0.03),
            lean_target=_vary_target((-0.25, 0.1, shoulder_height), 0.03),
            look_at_target=_vary_target((-0.5, 0.5, shoulder_height), 0.05),
        )
    )

    # === REACH DOWN BETWEEN LEGS ===
    bins.append(
        IKTestCase(
            goal_type="EXTREME_ARM",
            description="REACH BETWEEN LEGS",
            tags={"region": "extreme_arm", "action": "Reach down between legs"},
            left_foot_target=_vary_target((-0.2, 0.0, ground_z), 0.02),
            right_foot_target=_vary_target((0.2, 0.0, ground_z), 0.02),
            hip_drop=_vary(0.25, 0.02),
            right_hand_target=_vary_target((0.0, -0.15, ground_z + 0.15), 0.02),
            lean_target=_vary_target((0.0, 0.3, shoulder_height - 0.5), 0.03),
            look_at_target=_vary_target((0.0, 0.2, ground_z + 0.3), 0.03),
        )
    )

    # === BOTH ARMS BEHIND (hands together) ===
    bins.append(
        IKTestCase(
            goal_type="EXTREME_ARM",
            description="ARMS BACK TOGETHER",
            tags={"region": "extreme_arm", "action": "Both arms reaching back"},
            left_foot_target=left_foot,
            right_foot_target=right_foot,
            left_hand_target=_vary_target((-0.1, -0.35, shoulder_height - 0.25), 0.02),
            right_hand_target=_vary_target((0.1, -0.35, shoulder_height - 0.25), 0.02),
            lean_target=_vary_target((0.0, 0.15, shoulder_height), 0.03),
            look_at_target=_vary_target((0.0, 2.0, shoulder_height), 0.05),
        )
    )

    # === OPPOSITE CORNERS (left high, right low) ===
    bins.append(
        IKTestCase(
            goal_type="EXTREME_ARM",
            description="OPPOSITE CORNERS",
            tags={"region": "extreme_arm", "action": "Arms to opposite corners"},
            left_foot_target=left_foot,
            right_foot_target=right_foot,
            left_hand_target=_vary_target((-0.35, 0.3, shoulder_height + 0.5), 0.03),
            right_hand_target=_vary_target((0.35, 0.3, shoulder_height - 0.7), 0.03),
            lean_target=_vary_target((-0.1, 0.15, shoulder_height), 0.03),
            look_at_target=_vary_target((-0.3, 1.0, shoulder_height + 0.3), 0.05),
        )
    )

    return bins


def _extreme_leg_bins(ground_z: float = 0.0, shoulder_height: float = 1.5) -> List[IKTestCase]:
    """
    Extreme leg tests - high kicks, splits, extreme positions.
    """
    bins = []

    # === HIGH FRONT KICK (head height) ===
    bins.append(
        IKTestCase(
            goal_type="EXTREME_LEG",
            description="HIGH FRONT KICK",
            tags={"region": "extreme_leg", "action": "Front kick to head height"},
            left_foot_target=_vary_target((-0.1, 0.0, ground_z), 0.02),
            right_foot_target=_vary_target((0.1, 0.55, shoulder_height + 0.15), 0.04),
            left_hand_target=_vary_target((-0.3, 0.15, shoulder_height), 0.03),
            right_hand_target=_vary_target((0.25, 0.15, shoulder_height), 0.03),
            lean_target=_vary_target((0.0, -0.2, shoulder_height - 0.1), 0.03),
            look_at_target=_vary_target((0.0, 1.5, shoulder_height), 0.05),
        )
    )

    # === HIGH SIDE KICK (head height) ===
    bins.append(
        IKTestCase(
            goal_type="EXTREME_LEG",
            description="HIGH SIDE KICK",
            tags={"region": "extreme_leg", "action": "Side kick to head height"},
            left_foot_target=_vary_target((-0.1, 0.0, ground_z), 0.02),
            right_foot_target=_vary_target((0.7, 0.1, shoulder_height + 0.1), 0.04),
            left_hand_target=_vary_target((-0.3, 0.2, shoulder_height + 0.2), 0.03),
            right_hand_target=_vary_target((0.15, 0.15, shoulder_height), 0.03),
            lean_target=_vary_target((-0.3, 0.0, shoulder_height), 0.03),
            look_at_target=_vary_target((0.6, 0.8, shoulder_height), 0.05),
        )
    )

    # === BACK KICK ===
    bins.append(
        IKTestCase(
            goal_type="EXTREME_LEG",
            description="BACK KICK",
            tags={"region": "extreme_leg", "action": "Kick straight back"},
            left_foot_target=_vary_target((-0.08, 0.1, ground_z), 0.02),
            right_foot_target=_vary_target((0.15, -0.6, shoulder_height - 0.6), 0.04),
            left_hand_target=_vary_target((-0.25, 0.3, shoulder_height), 0.03),
            right_hand_target=_vary_target((0.2, 0.3, shoulder_height), 0.03),
            lean_target=_vary_target((0.0, 0.35, shoulder_height - 0.15), 0.03),
            look_at_target=_vary_target((0.0, 1.5, shoulder_height), 0.05),
        )
    )

    # === FRONT SPLITS (attempting) ===
    bins.append(
        IKTestCase(
            goal_type="EXTREME_LEG",
            description="FRONT SPLITS",
            tags={"region": "extreme_leg", "action": "Front splits position"},
            left_foot_target=_vary_target((-0.1, -0.5, ground_z), 0.02),
            right_foot_target=_vary_target((0.1, 0.5, ground_z), 0.02),
            hip_drop=_vary(0.7, 0.03),
            left_hand_target=_vary_target((-0.3, 0.0, shoulder_height - 0.5), 0.03),
            right_hand_target=_vary_target((0.3, 0.0, shoulder_height - 0.5), 0.03),
            lean_target=_vary_target((0.0, 0.0, shoulder_height - 0.6), 0.03),
            look_at_target=_vary_target((0.0, 2.0, shoulder_height - 0.3), 0.05),
        )
    )

    # === SIDE SPLITS (attempting) ===
    bins.append(
        IKTestCase(
            goal_type="EXTREME_LEG",
            description="SIDE SPLITS",
            tags={"region": "extreme_leg", "action": "Side splits position"},
            left_foot_target=_vary_target((-0.55, 0.0, ground_z), 0.02),
            right_foot_target=_vary_target((0.55, 0.0, ground_z), 0.02),
            hip_drop=_vary(0.65, 0.03),
            left_hand_target=_vary_target((-0.2, 0.25, shoulder_height - 0.5), 0.03),
            right_hand_target=_vary_target((0.2, 0.25, shoulder_height - 0.5), 0.03),
            lean_target=_vary_target((0.0, 0.1, shoulder_height - 0.5), 0.03),
            look_at_target=_vary_target((0.0, 2.0, shoulder_height - 0.3), 0.05),
        )
    )

    # === DEEP SQUAT (ass to grass) ===
    bins.append(
        IKTestCase(
            goal_type="EXTREME_LEG",
            description="DEEP SQUAT",
            tags={"region": "extreme_leg", "action": "Maximum depth squat"},
            left_foot_target=_vary_target((-0.15, 0.0, ground_z), 0.02),
            right_foot_target=_vary_target((0.15, 0.0, ground_z), 0.02),
            hip_drop=_vary(0.55, 0.03),
            left_hand_target=_vary_target((-0.2, 0.35, shoulder_height - 0.5), 0.03),
            right_hand_target=_vary_target((0.2, 0.35, shoulder_height - 0.5), 0.03),
            lean_target=_vary_target((0.0, 0.2, shoulder_height - 0.4), 0.03),
            look_at_target=_vary_target((0.0, 2.0, shoulder_height - 0.2), 0.05),
        )
    )

    # === PISTOL SQUAT (one leg out front) ===
    bins.append(
        IKTestCase(
            goal_type="EXTREME_LEG",
            description="PISTOL SQUAT",
            tags={"region": "extreme_leg", "action": "One leg squat other extended"},
            left_foot_target=_vary_target((-0.1, 0.0, ground_z), 0.02),
            right_foot_target=_vary_target((0.1, 0.5, ground_z + 0.35), 0.03),  # leg extended forward
            hip_drop=_vary(0.5, 0.03),
            left_hand_target=_vary_target((-0.25, 0.35, shoulder_height - 0.4), 0.03),
            right_hand_target=_vary_target((0.25, 0.35, shoulder_height - 0.4), 0.03),
            lean_target=_vary_target((0.0, 0.25, shoulder_height - 0.4), 0.03),
            look_at_target=_vary_target((0.0, 2.0, shoulder_height - 0.2), 0.05),
        )
    )

    # === LEG RAISED BEHIND (arabesque) ===
    bins.append(
        IKTestCase(
            goal_type="EXTREME_LEG",
            description="ARABESQUE",
            tags={"region": "extreme_leg", "action": "Leg raised high behind"},
            left_foot_target=_vary_target((-0.08, 0.0, ground_z), 0.02),
            right_foot_target=_vary_target((0.15, -0.55, shoulder_height - 0.2), 0.04),
            left_hand_target=_vary_target((-0.35, 0.4, shoulder_height + 0.1), 0.03),
            right_hand_target=_vary_target((0.35, 0.4, shoulder_height + 0.1), 0.03),
            lean_target=_vary_target((0.0, 0.35, shoulder_height), 0.03),
            look_at_target=_vary_target((0.0, 2.0, shoulder_height + 0.2), 0.05),
        )
    )

    # === SCORPION POSE (leg behind and UP toward head) ===
    bins.append(
        IKTestCase(
            goal_type="EXTREME_LEG",
            description="SCORPION",
            tags={"region": "extreme_leg", "action": "Leg behind curving up toward head"},
            left_foot_target=_vary_target((-0.08, 0.0, ground_z), 0.02),
            right_foot_target=_vary_target((0.1, -0.35, shoulder_height + 0.3), 0.04),  # behind AND high
            left_hand_target=_vary_target((-0.25, 0.3, shoulder_height), 0.03),
            right_hand_target=_vary_target((0.25, 0.3, shoulder_height), 0.03),
            lean_target=_vary_target((0.0, 0.25, shoulder_height - 0.1), 0.03),
            look_at_target=_vary_target((0.0, 1.5, shoulder_height), 0.05),
        )
    )

    # === HIGH BACK KICK (higher than normal) ===
    bins.append(
        IKTestCase(
            goal_type="EXTREME_LEG",
            description="HIGH BACK KICK",
            tags={"region": "extreme_leg", "action": "Back kick to shoulder height"},
            left_foot_target=_vary_target((-0.08, 0.05, ground_z), 0.02),
            right_foot_target=_vary_target((0.12, -0.5, shoulder_height), 0.04),  # straight back at shoulder height
            left_hand_target=_vary_target((-0.3, 0.35, shoulder_height + 0.1), 0.03),
            right_hand_target=_vary_target((0.25, 0.35, shoulder_height + 0.1), 0.03),
            lean_target=_vary_target((0.0, 0.4, shoulder_height - 0.2), 0.03),
            look_at_target=_vary_target((0.0, 2.0, shoulder_height), 0.05),
        )
    )

    # === STANDING BOW (grab foot behind) ===
    bins.append(
        IKTestCase(
            goal_type="EXTREME_LEG",
            description="STANDING BOW",
            tags={"region": "extreme_leg", "action": "Grab foot behind while standing"},
            left_foot_target=_vary_target((-0.05, 0.0, ground_z), 0.02),
            right_foot_target=_vary_target((0.15, -0.4, shoulder_height - 0.4), 0.03),  # foot behind at hip height
            left_hand_target=_vary_target((-0.2, 0.45, shoulder_height + 0.2), 0.03),  # one arm forward for balance
            right_hand_target=_vary_target((0.1, -0.25, shoulder_height - 0.3), 0.03),  # grabbing foot behind
            lean_target=_vary_target((0.0, 0.3, shoulder_height), 0.03),
            look_at_target=_vary_target((0.0, 2.0, shoulder_height + 0.1), 0.05),
        )
    )

    # === HELICOPTER KICK WIND-UP (leg behind and to side) ===
    bins.append(
        IKTestCase(
            goal_type="EXTREME_LEG",
            description="SIDE BACK KICK",
            tags={"region": "extreme_leg", "action": "Kick back and to side"},
            left_foot_target=_vary_target((-0.1, 0.0, ground_z), 0.02),
            right_foot_target=_vary_target((0.5, -0.4, shoulder_height - 0.3), 0.04),  # back AND to side
            left_hand_target=_vary_target((-0.35, 0.25, shoulder_height), 0.03),
            right_hand_target=_vary_target((0.2, 0.25, shoulder_height), 0.03),
            lean_target=_vary_target((-0.2, 0.2, shoulder_height - 0.1), 0.03),
            look_at_target=_vary_target((0.4, 1.0, shoulder_height), 0.05),
        )
    )

    # === EXTREME ARABESQUE (horizontal body) ===
    bins.append(
        IKTestCase(
            goal_type="EXTREME_LEG",
            description="EXTREME ARABESQUE",
            tags={"region": "extreme_leg", "action": "Body horizontal leg behind max"},
            left_foot_target=_vary_target((-0.08, 0.0, ground_z), 0.02),
            right_foot_target=_vary_target((0.1, -0.65, shoulder_height + 0.1), 0.04),  # leg as high as possible behind
            left_hand_target=_vary_target((-0.2, 0.55, shoulder_height - 0.3), 0.03),  # arms forward
            right_hand_target=_vary_target((0.2, 0.55, shoulder_height - 0.3), 0.03),
            lean_target=_vary_target((0.0, 0.5, shoulder_height - 0.4), 0.03),  # torso forward
            look_at_target=_vary_target((0.0, 1.5, shoulder_height - 0.2), 0.05),
        )
    )

    return bins


def _asymmetric_bins(ground_z: float = 0.0, shoulder_height: float = 1.5) -> List[IKTestCase]:
    """
    Asymmetric poses - focus on challenging positions only.
    """
    bins = []
    foot_spacing = 0.1

    # === ARM UP / ARM DOWN (tests above-shoulder reach) ===
    bins.append(
        IKTestCase(
            goal_type="ASYMMETRIC",
            description="ARM UP ARM DOWN",
            tags={"region": "asymmetric", "action": "One arm up, one down"},
            left_foot_target=_vary_target((-foot_spacing, 0.0, ground_z), 0.02),
            right_foot_target=_vary_target((foot_spacing, 0.0, ground_z), 0.02),
            left_hand_target=_vary_target((-0.2, 0.2, shoulder_height + 0.5), 0.03),
            right_hand_target=_vary_target((0.2, 0.2, shoulder_height - 0.6), 0.03),
            look_at_target=_vary_target((0.0, 2.0, shoulder_height), 0.05),
        )
    )

    # === SCRATCHING HEAD (tests behind-head reach) ===
    bins.append(
        IKTestCase(
            goal_type="ASYMMETRIC",
            description="SCRATCH HEAD",
            tags={"region": "asymmetric", "action": "Scratching head"},
            left_foot_target=_vary_target((-foot_spacing, 0.0, ground_z), 0.02),
            right_foot_target=_vary_target((foot_spacing, 0.0, ground_z), 0.02),
            left_hand_target=_vary_target((-0.2, 0.1, shoulder_height - 0.2), 0.03),
            right_hand_target=_vary_target((0.1, -0.1, shoulder_height + 0.35), 0.03),
            look_at_target=_vary_target((0.0, 1.5, shoulder_height - 0.2), 0.05),
        )
    )

    # === CROSSED ARMS (tests spine crossover) ===
    bins.append(
        IKTestCase(
            goal_type="ASYMMETRIC",
            description="CROSSED ARMS",
            tags={"region": "asymmetric", "action": "Arms crossed"},
            left_foot_target=_vary_target((-foot_spacing, 0.0, ground_z), 0.02),
            right_foot_target=_vary_target((foot_spacing, 0.0, ground_z), 0.02),
            left_hand_target=_vary_target((0.18, 0.15, shoulder_height - 0.25), 0.03),
            right_hand_target=_vary_target((-0.18, 0.15, shoulder_height - 0.25), 0.03),
            look_at_target=_vary_target((0.0, 2.0, shoulder_height), 0.05),
        )
    )

    return bins


def build_default_plan(rig_heights: dict | None = None) -> List[IKTestCase]:
    """
    Structured test coverage with slight random variations.

    Each call generates slightly different target positions and shuffled order,
    so running tests multiple times won't be exactly the same.

    rig_heights: optional dict with keys ground, knee, hip, chest, shoulder, head for rig-scaled heights.
    """
    plan: List[IKTestCase] = []
    # Resolve heights
    if rig_heights:
        chest = rig_heights.get("chest", 1.2)
        shoulder = rig_heights.get("shoulder", 1.5)
        head = rig_heights.get("head", shoulder + 0.2)
        overhead = head + 0.1
        ground = rig_heights.get("ground", 0.0)
        knee = rig_heights.get("knee", ground + 0.4)
    else:
        chest, shoulder, head, overhead = 1.2, 1.5, 1.7, 1.8
        ground, knee = 0.0, 0.35

    # Hand heights: one baseline (chest), then challenging heights (shoulder, head, overhead)
    hand_heights = [chest, shoulder, head, overhead]
    # Foot heights: one baseline (ground step), then challenging heights (waist, chest)
    foot_heights = [ground + 0.05, shoulder - 0.5, shoulder]

    # === ISOLATED LIMB TESTS ===
    # Hands (arm IK only) - fewer easy, more hard
    for h in hand_heights:
        plan.extend(_reach_bins("left", h))
        plan.extend(_reach_bins("right", h))

    # Feet (leg IK only) - fewer easy, more high kicks
    for h in foot_heights:
        plan.extend(_foot_bins("left", h))
        plan.extend(_foot_bins("right", h))

    # === LOWER BODY TESTS ===
    # Crouch - feet planted on ground while hips drop
    plan.extend(_crouch_bins(ground_z=ground))

    # === UPPER BODY TESTS ===
    # Look - head/neck tracking (includes up/down/over-shoulder)
    plan.extend(_look_bins())

    # Lean - spine bending (forward/back/side)
    plan.extend(_lean_bins(ground_z=ground))

    # === FULL BODY COMBINED TESTS ===
    # Real-world poses that test multiple systems together
    plan.extend(_combined_bins(ground_z=ground, shoulder_height=shoulder))

    # === ATHLETIC / DYNAMIC POSES ===
    # Throwing, punching, kicking - tests weight transfer and dynamic positions
    plan.extend(_athletic_bins(ground_z=ground, shoulder_height=shoulder))

    # === EXTREME REACH TESTS ===
    # Maximum extension - pushing IK to its limits
    plan.extend(_extreme_reach_bins(ground_z=ground, shoulder_height=shoulder))

    # === BALANCE TESTS ===
    # One-leg stands, yoga poses, lunges
    plan.extend(_balance_bins(ground_z=ground, shoulder_height=shoulder))

    # === INTERACTION POSES ===
    # Real-world interactions - doors, climbing, pushing, pulling
    plan.extend(_interaction_bins(ground_z=ground, shoulder_height=shoulder))

    # === ASYMMETRIC POSES ===
    # Different arm/leg positions - tests IK independence
    plan.extend(_asymmetric_bins(ground_z=ground, shoulder_height=shoulder))

    # === SPINE STRESS TESTS ===
    # Extreme torso bends, twists - pushing spinal limits
    plan.extend(_spine_stress_bins(ground_z=ground, shoulder_height=shoulder))

    # === NECK STRESS TESTS ===
    # Extreme head/neck positions - max rotation/tilt
    plan.extend(_neck_stress_bins(ground_z=ground, shoulder_height=shoulder))

    # === SHOULDER STRESS TESTS ===
    # Challenging shoulder positions - behind back, overhead, chicken wing
    plan.extend(_shoulder_stress_bins(ground_z=ground, shoulder_height=shoulder))

    # === EXTREME ARM TESTS ===
    # Maximum arm reaches - behind, overhead, cross-body far
    plan.extend(_extreme_arm_bins(ground_z=ground, shoulder_height=shoulder))

    # === EXTREME LEG TESTS ===
    # High kicks, splits, deep squats - pushing leg IK hard
    plan.extend(_extreme_leg_bins(ground_z=ground, shoulder_height=shoulder))

    # Shuffle the test order so each session is different
    random.shuffle(plan)

    return plan


# =============================================================================
# Auto Evaluation
# =============================================================================

_LIMITS = get_default_limits()


def _bone_world_pos(armature, name: str) -> Optional[Vector]:
    pb = armature.pose.bones.get(name)
    if not pb:
        return None
    return armature.matrix_world @ pb.head


def _bone_rotation_deg(armature, name: str) -> Optional[Dict[str, float]]:
    """Get bone's local rotation in degrees (X, Y, Z Euler)."""
    pb = armature.pose.bones.get(name)
    if not pb:
        return None
    eul = pb.matrix_basis.to_euler('XYZ')
    return {
        "X": round(math.degrees(eul[0]), 1),
        "Y": round(math.degrees(eul[1]), 1),
        "Z": round(math.degrees(eul[2]), 1),
    }


def _vec_to_list(v: Vector) -> List[float]:
    """Convert Vector to rounded list for JSON."""
    return [round(v.x, 4), round(v.y, 4), round(v.z, 4)]


def _compute_bend_angle(p1: Vector, p2: Vector, p3: Vector) -> float:
    """Compute bend angle at p2 between p1-p2-p3 chain. Returns degrees."""
    v1 = (p1 - p2).normalized()
    v2 = (p3 - p2).normalized()
    dot = max(-1.0, min(1.0, v1.dot(v2)))
    return math.degrees(math.acos(dot))


def _compute_pole_direction(shoulder: Vector, elbow: Vector, wrist: Vector) -> List[float]:
    """
    Compute where the elbow is pointing (pole vector direction).
    Returns normalized direction vector from chain plane.
    """
    # Chain direction
    chain_dir = (wrist - shoulder).normalized()
    # Vector to elbow
    to_elbow = elbow - shoulder
    # Project onto chain to find "inline" point
    proj_len = to_elbow.dot(chain_dir)
    inline_point = shoulder + chain_dir * proj_len
    # Pole direction is from inline point to actual elbow
    pole_dir = (elbow - inline_point)
    if pole_dir.length > 0.001:
        pole_dir = pole_dir.normalized()
        return _vec_to_list(pole_dir)
    return [0.0, 0.0, 0.0]


def _extract_arm_chain(armature, side: str) -> Optional[Dict]:
    """
    Extract full arm chain data for analysis with problem detection.

    Captures: Shoulder → Arm → ForeArm → Hand
    Also computes reach analysis and flags potential problems.
    """
    prefix = "Left" if side == "left" else "Right"

    clavicle = _bone_world_pos(armature, f"{prefix}Shoulder")  # Clavicle
    upper_arm = _bone_world_pos(armature, f"{prefix}Arm")      # Shoulder joint
    forearm = _bone_world_pos(armature, f"{prefix}ForeArm")    # Elbow
    hand = _bone_world_pos(armature, f"{prefix}Hand")          # Wrist

    if not all([upper_arm, forearm, hand]):
        return None

    # Chain lengths
    upper_len = (forearm - upper_arm).length
    lower_len = (hand - forearm).length
    clavicle_len = (upper_arm - clavicle).length if clavicle else 0.0
    total_reach = upper_len + lower_len

    # Bend angle at elbow (180 = straight, less = bent)
    elbow_bend = _compute_bend_angle(upper_arm, forearm, hand)

    # Pole direction (where elbow points - should be backward -Y)
    pole_dir = _compute_pole_direction(upper_arm, forearm, hand)

    # Shoulder angle (clavicle to upper arm to elbow)
    shoulder_bend = None
    if clavicle:
        shoulder_bend = round(_compute_bend_angle(clavicle, upper_arm, forearm), 1)

    # =========================================================================
    # REACH ANALYSIS - Compute where hand is relative to shoulder
    # =========================================================================
    reach_vec = hand - upper_arm
    reach_length = reach_vec.length
    relative_reach = reach_length / total_reach if total_reach > 0 else 0

    # Decompose reach direction (assumes character facing +Y, up is +Z)
    reach_up = reach_vec.z / reach_length if reach_length > 0.01 else 0
    reach_forward = reach_vec.y / reach_length if reach_length > 0.01 else 0
    reach_right = reach_vec.x / reach_length if reach_length > 0.01 else 0

    # Classify reach type
    reach_type = "NEUTRAL"
    if reach_up > 0.5:
        reach_type = "UPWARD"
    elif reach_up < -0.5:
        reach_type = "DOWNWARD"
    elif reach_forward > 0.5:
        reach_type = "FORWARD"
    elif abs(reach_right) > 0.5:
        reach_type = "SIDEWAYS"

    if relative_reach < 0.45:
        reach_type = "SHORT/GUARD"

    # =========================================================================
    # PROBLEM DETECTION - Flag obvious issues
    # =========================================================================
    problems = []

    # Problem 1: Pole pointing UP (Z > 0.3) when it should point BACK/DOWN
    if len(pole_dir) >= 3 and pole_dir[2] > 0.3:
        problems.append(f"POLE_UP: Z={pole_dir[2]:.2f} (should be negative for back/down)")

    # Problem 2: Pole pointing FORWARD (Y > 0.3) instead of BACK
    if len(pole_dir) >= 3 and pole_dir[1] > 0.3:
        problems.append(f"POLE_FWD: Y={pole_dir[1]:.2f} (should be negative for backward)")

    # Problem 3: Left elbow on right side of body (or vice versa)
    if side == "left" and len(pole_dir) >= 3 and pole_dir[0] > 0.3:
        problems.append(f"POLE_WRONG_SIDE: X={pole_dir[0]:.2f} (L elbow pointing right)")
    elif side == "right" and len(pole_dir) >= 3 and pole_dir[0] < -0.3:
        problems.append(f"POLE_WRONG_SIDE: X={pole_dir[0]:.2f} (R elbow pointing left)")

    # Problem 4: Elbow nearly straight for a bent pose
    if elbow_bend > 170 and relative_reach < 0.9:
        problems.append(f"ELBOW_STRAIGHT: {elbow_bend:.0f}° but reach={relative_reach:.0%}")

    # Problem 5: Elbow hyperextended
    if elbow_bend < 30:
        problems.append(f"ELBOW_HYPER: {elbow_bend:.0f}° (too bent)")

    return {
        "clavicle_pos": _vec_to_list(clavicle) if clavicle else None,
        "shoulder_pos": _vec_to_list(upper_arm),
        "elbow_pos": _vec_to_list(forearm),
        "wrist_pos": _vec_to_list(hand),
        "clavicle_len": round(clavicle_len, 4),
        "upper_arm_len": round(upper_len, 4),
        "forearm_len": round(lower_len, 4),
        "total_reach": round(total_reach, 4),
        "shoulder_bend_deg": shoulder_bend,
        "elbow_bend_deg": round(elbow_bend, 1),
        "pole_direction": pole_dir,
        # NEW: Reach analysis
        "reach_length": round(reach_length, 4),
        "relative_reach": round(relative_reach, 3),
        "reach_type": reach_type,
        "reach_components": {
            "up": round(reach_up, 3),
            "forward": round(reach_forward, 3),
            "right": round(reach_right, 3),
        },
        # NEW: Problem flags
        "problems": problems,
        "rotations": {
            f"{prefix}Shoulder": _bone_rotation_deg(armature, f"{prefix}Shoulder"),
            f"{prefix}Arm": _bone_rotation_deg(armature, f"{prefix}Arm"),
            f"{prefix}ForeArm": _bone_rotation_deg(armature, f"{prefix}ForeArm"),
            f"{prefix}Hand": _bone_rotation_deg(armature, f"{prefix}Hand"),
        }
    }


def _extract_leg_chain(armature, side: str) -> Optional[Dict]:
    """Extract full leg chain data for analysis with problem detection."""
    prefix = "Left" if side == "left" else "Right"

    thigh = _bone_world_pos(armature, f"{prefix}Thigh")
    shin = _bone_world_pos(armature, f"{prefix}Shin")
    foot = _bone_world_pos(armature, f"{prefix}Foot")
    toe = _bone_world_pos(armature, f"{prefix}ToeBase")

    if not all([thigh, shin, foot]):
        return None

    # Chain lengths
    thigh_len = (shin - thigh).length
    shin_len = (foot - shin).length
    foot_len = (toe - foot).length if toe else 0.0
    total_reach = thigh_len + shin_len

    # Bend angle at knee (180 = straight, less = bent)
    knee_bend = _compute_bend_angle(thigh, shin, foot)

    # Pole direction (where knee points - should be forward +Y)
    pole_dir = _compute_pole_direction(thigh, shin, foot)

    # Ankle angle (foot relative to shin)
    ankle_bend = None
    if toe:
        ankle_bend = round(_compute_bend_angle(shin, foot, toe), 1)

    # =========================================================================
    # PROBLEM DETECTION FOR LEGS
    # =========================================================================
    problems = []

    # Problem 1: Knee pointing BACKWARD (Y < -0.3) instead of FORWARD
    if len(pole_dir) >= 3 and pole_dir[1] < -0.3:
        problems.append(f"KNEE_BACK: Y={pole_dir[1]:.2f} (should be positive for forward)")

    # Problem 2: Knee pointing sideways too much
    if len(pole_dir) >= 3 and abs(pole_dir[0]) > 0.5:
        problems.append(f"KNEE_SIDEWAYS: X={pole_dir[0]:.2f} (too lateral)")

    # Problem 3: Knee hyperextended (leg bent backward)
    if knee_bend > 185:
        problems.append(f"KNEE_HYPEREXT: {knee_bend:.0f}° (bent backward!)")

    # Problem 4: Knee locked straight when it should bend
    foot_height = foot.z if foot else 0
    if knee_bend > 175 and foot_height > 0.3:  # Foot lifted but knee straight
        problems.append(f"KNEE_LOCKED: {knee_bend:.0f}° with foot at z={foot_height:.2f}")

    return {
        "thigh_pos": _vec_to_list(thigh),
        "knee_pos": _vec_to_list(shin),
        "ankle_pos": _vec_to_list(foot),
        "toe_pos": _vec_to_list(toe) if toe else None,
        "thigh_len": round(thigh_len, 4),
        "shin_len": round(shin_len, 4),
        "foot_len": round(foot_len, 4),
        "total_reach": round(total_reach, 4),
        "knee_bend_deg": round(knee_bend, 1),
        "ankle_bend_deg": ankle_bend,
        "pole_direction": pole_dir,
        # NEW: Problem flags
        "problems": problems,
        "rotations": {
            f"{prefix}Thigh": _bone_rotation_deg(armature, f"{prefix}Thigh"),
            f"{prefix}Shin": _bone_rotation_deg(armature, f"{prefix}Shin"),
            f"{prefix}Foot": _bone_rotation_deg(armature, f"{prefix}Foot"),
            f"{prefix}ToeBase": _bone_rotation_deg(armature, f"{prefix}ToeBase"),
        }
    }


def _extract_spine_chain(armature) -> Optional[Dict]:
    """
    Extract spine/torso chain data for lean and posture analysis.

    Captures: Hips → Spine → Spine1 → Spine2 (torso only, not neck/head)
    Includes relative angles between each segment for bend analysis.
    """
    # Core spine bones (Hips through upper chest)
    bone_names = ["Hips", "Spine", "Spine1", "Spine2"]

    positions = {}
    rotations = {}

    for name in bone_names:
        pos = _bone_world_pos(armature, name)
        if pos:
            positions[name] = _vec_to_list(pos)
        rot = _bone_rotation_deg(armature, name)
        if rot:
            rotations[name] = rot

    if len(positions) < 3:
        return None

    # Calculate segment angles (bend between adjacent bones)
    segment_angles = {}
    for i in range(len(bone_names) - 2):
        b1, b2, b3 = bone_names[i], bone_names[i + 1], bone_names[i + 2]
        p1 = _bone_world_pos(armature, b1)
        p2 = _bone_world_pos(armature, b2)
        p3 = _bone_world_pos(armature, b3)
        if p1 and p2 and p3:
            angle = _compute_bend_angle(p1, p2, p3)
            segment_angles[f"{b2}_bend"] = round(angle, 1)

    # Calculate overall lean direction
    hips_pos = _bone_world_pos(armature, "Hips")
    chest_pos = _bone_world_pos(armature, "Spine2")

    lean_data = {}
    if hips_pos and chest_pos:
        # Lean is how much chest is offset from directly above hips
        chest_offset = Vector((chest_pos.x - hips_pos.x, chest_pos.y - hips_pos.y, 0))
        lean_data["chest_offset_from_hips"] = _vec_to_list(chest_offset)
        lean_data["lean_magnitude"] = round(chest_offset.length, 4)

        # Lean direction (normalized XY)
        if chest_offset.length > 0.01:
            lean_dir = chest_offset.normalized()
            lean_data["lean_direction"] = [round(lean_dir.x, 3), round(lean_dir.y, 3)]
        else:
            lean_data["lean_direction"] = [0.0, 0.0]

    # Calculate total spine length
    spine_length = 0.0
    for i in range(len(bone_names) - 1):
        p1 = _bone_world_pos(armature, bone_names[i])
        p2 = _bone_world_pos(armature, bone_names[i + 1])
        if p1 and p2:
            spine_length += (p2 - p1).length

    # Chord length (straight line hips to chest)
    chord_length = 0.0
    if hips_pos and chest_pos:
        chord_length = (chest_pos - hips_pos).length

    # Curvature ratio (1.0 = straight, higher = more curved)
    curvature_ratio = spine_length / chord_length if chord_length > 0.01 else 1.0

    return {
        "positions": positions,
        "rotations": rotations,
        "segment_angles": segment_angles,
        "lean": lean_data,
        "spine_length": round(spine_length, 4),
        "chord_length": round(chord_length, 4),
        "curvature_ratio": round(curvature_ratio, 3),
    }


def _extract_head_chain(armature) -> Optional[Dict]:
    """
    Extract head/neck data for look analysis.

    Captures: NeckLower → NeckUpper → Head (two neck bones!)
    """
    neck_lower = _bone_world_pos(armature, "NeckLower")
    neck_upper = _bone_world_pos(armature, "NeckUpper")
    head_pos = _bone_world_pos(armature, "Head")

    if not all([neck_lower, neck_upper, head_pos]):
        return None

    # All rotations
    rotations = {
        "NeckLower": _bone_rotation_deg(armature, "NeckLower"),
        "NeckUpper": _bone_rotation_deg(armature, "NeckUpper"),
        "Head": _bone_rotation_deg(armature, "Head"),
    }

    # Neck bend angles
    spine2 = _bone_world_pos(armature, "Spine2")
    neck_lower_bend = None
    neck_upper_bend = None
    if spine2 and neck_lower and neck_upper:
        neck_lower_bend = round(_compute_bend_angle(spine2, neck_lower, neck_upper), 1)
    if neck_lower and neck_upper and head_pos:
        neck_upper_bend = round(_compute_bend_angle(neck_lower, neck_upper, head_pos), 1)

    # Head facing direction (from head bone's local Y axis in world space)
    pb = armature.pose.bones.get("Head")
    facing_dir = [0.0, 1.0, 0.0]
    if pb:
        head_matrix = armature.matrix_world @ pb.matrix
        # Z-axis is forward for head based on rig.md
        forward = Vector((head_matrix[0][2], head_matrix[1][2], head_matrix[2][2]))
        facing_dir = _vec_to_list(forward.normalized())

    return {
        "positions": {
            "NeckLower": _vec_to_list(neck_lower),
            "NeckUpper": _vec_to_list(neck_upper),
            "Head": _vec_to_list(head_pos),
        },
        "rotations": rotations,
        "neck_lower_bend_deg": neck_lower_bend,
        "neck_upper_bend_deg": neck_upper_bend,
        "facing_direction": facing_dir,
    }


def _limit_hits(armature) -> List[str]:
    hits = []
    for bone_name, limits in _LIMITS.items():
        pb = armature.pose.bones.get(bone_name)
        if not pb:
            continue
        eul = pb.matrix_basis.to_euler('XYZ')
        for axis, idx in (("X", 0), ("Y", 1), ("Z", 2)):
            min_v, max_v = limits[axis]
            deg = math.degrees(eul[idx])
            if deg < min_v - 1 or deg > max_v + 1:
                hits.append(f"{bone_name}.{axis}:{deg:.1f} not in [{min_v},{max_v}]")
    return hits


def evaluate_pose(armature, test: IKTestCase) -> dict:
    """Compute comprehensive metrics for IK analysis."""
    metrics = {}

    # --- Error vectors (not just magnitude) ---
    def target_error_detailed(target, effector_name):
        if not target:
            return None
        pos = _bone_world_pos(armature, effector_name)
        if not pos:
            return None
        target_vec = Vector(target)
        error_vec = pos - target_vec
        return {
            "error_cm": round(error_vec.length * 100.0, 2),
            "error_xyz_cm": [round(error_vec.x * 100, 2), round(error_vec.y * 100, 2), round(error_vec.z * 100, 2)],
            "target": _vec_to_list(target_vec),
            "actual": _vec_to_list(pos),
        }

    metrics["left_hand"] = target_error_detailed(test.left_hand_target, "LeftHand")
    metrics["right_hand"] = target_error_detailed(test.right_hand_target, "RightHand")
    metrics["left_foot"] = target_error_detailed(test.left_foot_target, "LeftFoot")
    metrics["right_foot"] = target_error_detailed(test.right_foot_target, "RightFoot")

    # --- Chain analysis (THE KEY DATA) ---
    # Extract full chain geometry for arms/legs that have targets
    if test.left_hand_target:
        metrics["left_arm_chain"] = _extract_arm_chain(armature, "left")
    if test.right_hand_target:
        metrics["right_arm_chain"] = _extract_arm_chain(armature, "right")
    if test.left_foot_target:
        metrics["left_leg_chain"] = _extract_leg_chain(armature, "left")
    if test.right_foot_target:
        metrics["right_leg_chain"] = _extract_leg_chain(armature, "right")

    # For crouch, extract BOTH leg chains
    if test.hip_drop > 0:
        metrics["left_leg_chain"] = _extract_leg_chain(armature, "left")
        metrics["right_leg_chain"] = _extract_leg_chain(armature, "right")
        # Also track hip height
        hips = _bone_world_pos(armature, "Hips")
        if hips:
            metrics["hip_height"] = round(hips.z, 4)
            metrics["hip_drop_requested"] = test.hip_drop

    # For look tests, extract head/neck data
    if test.look_at_target:
        metrics["head_chain"] = _extract_head_chain(armature)
        # Also check if we hit the look target
        head_pos = _bone_world_pos(armature, "Head")
        if head_pos:
            target_vec = Vector(test.look_at_target)
            to_target = (target_vec - head_pos).normalized()
            metrics["look_target_direction"] = _vec_to_list(to_target)

    # For lean tests, extract spine data
    if test.lean_target:
        metrics["spine_chain"] = _extract_spine_chain(armature)

    # For combined/athletic/balance/interact/asymmetric tests, always capture full body
    # These dynamic tests need complete data for analysis
    dynamic_regions = {"combined", "athletic", "extreme", "balance", "interact", "asymmetric"}
    if test.tags.get("region") in dynamic_regions:
        # Always capture spine and head for posture analysis
        if "spine_chain" not in metrics:
            metrics["spine_chain"] = _extract_spine_chain(armature)
        if "head_chain" not in metrics:
            metrics["head_chain"] = _extract_head_chain(armature)
        # Always capture BOTH arm chains for asymmetric analysis
        if "left_arm_chain" not in metrics:
            metrics["left_arm_chain"] = _extract_arm_chain(armature, "left")
        if "right_arm_chain" not in metrics:
            metrics["right_arm_chain"] = _extract_arm_chain(armature, "right")
        # Always capture BOTH leg chains for balance/kick analysis
        if "left_leg_chain" not in metrics:
            metrics["left_leg_chain"] = _extract_leg_chain(armature, "left")
        if "right_leg_chain" not in metrics:
            metrics["right_leg_chain"] = _extract_leg_chain(armature, "right")

    # --- Foot ground contact analysis (for kicks and one-leg poses) ---
    left_foot_pos = _bone_world_pos(armature, "LeftFoot")
    right_foot_pos = _bone_world_pos(armature, "RightFoot")
    ground_threshold = 0.15  # 15cm above ground = "lifted"

    if left_foot_pos and right_foot_pos:
        # Determine which feet are on ground vs lifted
        left_on_ground = left_foot_pos.z < ground_threshold
        right_on_ground = right_foot_pos.z < ground_threshold

        metrics["foot_ground_contact"] = {
            "left_foot_z": round(left_foot_pos.z, 4),
            "right_foot_z": round(right_foot_pos.z, 4),
            "left_on_ground": left_on_ground,
            "right_on_ground": right_on_ground,
            "stance": "both" if (left_on_ground and right_on_ground) else
                      "left_only" if left_on_ground else
                      "right_only" if right_on_ground else "airborne"
        }

    # --- Balance check ---
    hips = _bone_world_pos(armature, "Hips")
    left_foot = _bone_world_pos(armature, "LeftFoot")
    right_foot = _bone_world_pos(armature, "RightFoot")

    if hips and left_foot and right_foot:
        com = (hips + left_foot + right_foot) / 3
        feet_center = (left_foot + right_foot) / 2
        com_offset = (Vector((com.x, com.y, 0)) - Vector((feet_center.x, feet_center.y, 0))).length
        metrics["balance_ok"] = com_offset < 0.35
        metrics["com_offset_m"] = round(com_offset, 4)
    else:
        metrics["balance_ok"] = True
        metrics["com_offset_m"] = 0.0

    # --- Joint limit violations ---
    hits = _limit_hits(armature)
    metrics["limit_hits"] = hits

    # =========================================================================
    # AGGREGATE ALL CHAIN PROBLEMS
    # =========================================================================
    all_chain_problems = []

    # Collect problems from arm chains
    if metrics.get("left_arm_chain") and metrics["left_arm_chain"].get("problems"):
        for prob in metrics["left_arm_chain"]["problems"]:
            all_chain_problems.append(f"L_ARM: {prob}")
    if metrics.get("right_arm_chain") and metrics["right_arm_chain"].get("problems"):
        for prob in metrics["right_arm_chain"]["problems"]:
            all_chain_problems.append(f"R_ARM: {prob}")

    # Collect problems from leg chains
    if metrics.get("left_leg_chain") and metrics["left_leg_chain"].get("problems"):
        for prob in metrics["left_leg_chain"]["problems"]:
            all_chain_problems.append(f"L_LEG: {prob}")
    if metrics.get("right_leg_chain") and metrics["right_leg_chain"].get("problems"):
        for prob in metrics["right_leg_chain"]["problems"]:
            all_chain_problems.append(f"R_LEG: {prob}")

    metrics["chain_problems"] = all_chain_problems

    # --- Auto verdict ---
    def get_error(detail):
        if detail is None:
            return None
        return detail.get("error_cm")

    def ok(val, thresh):
        return val is None or val <= thresh

    hand_ok = ok(get_error(metrics.get("left_hand")), 5) and ok(get_error(metrics.get("right_hand")), 5)
    foot_ok = ok(get_error(metrics.get("left_foot")), 5) and ok(get_error(metrics.get("right_foot")), 5)
    # Include chain problems in auto-pass check
    auto_pass = hand_ok and foot_ok and metrics["balance_ok"] and len(hits) == 0 and len(all_chain_problems) == 0

    # Combine all problems for easy viewing
    all_problems = hits + all_chain_problems

    return {
        "metrics": metrics,
        "auto_pass": auto_pass,
        "problems": all_problems,
        "limit_hits": hits,
        "chain_problems": all_chain_problems,
    }


# =============================================================================
# Session Recorder
# =============================================================================

@dataclass
class RecordedTest:
    test: dict
    evaluation: dict
    human_verdict: str
    human_category: str
    note: str = ""
    solver_diagnostics: dict = field(default_factory=dict)  # IK solver internal state


class SessionRecorder:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.tests: List[RecordedTest] = []

    def record(self, test: IKTestCase, evaluation: dict, verdict: str, category: str, note: str = ""):
        # Capture solver diagnostics from the last IK solve
        # Import here to avoid circular import (test_panel imports test_suite)
        from ..test_panel import get_last_solver_diagnostics
        solver_diag = get_last_solver_diagnostics()

        self.tests.append(
            RecordedTest(
                test=test.to_dict(),
                evaluation=evaluation,
                human_verdict=verdict,
                human_category=category,
                note=note,
                solver_diagnostics=solver_diag,
            )
        )

    def save(self) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        # Use timestamp to prevent overwrites
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_{timestamp}_{len(self.tests)}_tests.json"
        path = os.path.join(self.output_dir, filename)
        data = [asdict(t) for t in self.tests]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        # =====================================================================
        # ALSO SAVE COMPACT CSV SUMMARY
        # =====================================================================
        csv_path = path.replace(".json", "_summary.csv")
        self._save_csv_summary(csv_path)

        return path

    def _save_csv_summary(self, csv_path: str):
        """
        Save a detailed CSV summary for analysis.

        Columns include targets, actual positions, errors, joint angles, problems,
        AND solver diagnostics (internal decisions from IK solver).
        """
        lines = []
        header = (
            "idx,description,verdict,auto_pass,"
            "l_hand_target,l_hand_err,l_elbow_pos,l_elbow_angle,"
            "r_hand_target,r_hand_err,r_elbow_pos,r_elbow_angle,"
            "l_foot_target,l_foot_err,l_knee_pos,l_knee_angle,"
            "r_foot_target,r_foot_err,r_knee_pos,r_knee_angle,"
            "hip_drop,problems,note,"
            # SOLVER DIAGNOSTICS - what the solver computed
            "solver_leg_R_is_elevated,solver_leg_R_knee_fwd,solver_leg_R_correction,"
            "solver_leg_L_is_elevated,solver_leg_L_knee_fwd,solver_leg_L_correction,"
            "solver_arm_R_is_up,solver_arm_R_elbow_out,solver_arm_R_elbow_below,"
            "solver_arm_L_is_up,solver_arm_L_elbow_out,solver_arm_L_elbow_below,"
            # CONVERSION DIAGNOSTICS - direction to rotation conversion
            "conv_l_leg_target_dir,conv_l_leg_verify_dir,conv_l_leg_verify_err,conv_l_leg_verify_ok,"
            "conv_r_leg_target_dir,conv_r_leg_verify_dir,conv_r_leg_verify_err,conv_r_leg_verify_ok,"
            "conv_l_arm_target_dir,conv_l_arm_verify_dir,conv_l_arm_verify_err,conv_l_arm_verify_ok,"
            "conv_r_arm_target_dir,conv_r_arm_verify_dir,conv_r_arm_verify_err,conv_r_arm_verify_ok,"
            # POST-APPLY VERIFICATION - actual bone direction vs target
            "postapply_l_leg_shin_y,postapply_l_leg_error,postapply_r_leg_shin_y,postapply_r_leg_error,"
            "postapply_l_arm_forearm_y,postapply_l_arm_error,postapply_r_arm_forearm_y,postapply_r_arm_error"
        )
        lines.append(header)

        for i, t in enumerate(self.tests):
            desc = t.test.get("description", "").replace(",", ";")
            verdict = t.human_verdict
            auto_pass = t.evaluation.get("auto_pass", False)
            metrics = t.evaluation.get("metrics", {})

            # Helper: format target position from test case
            def fmt_target(target):
                if target and len(target) == 3:
                    return f"({target[0]:.2f} {target[1]:.2f} {target[2]:.2f})"
                return "-"

            # Helper: get error value
            def get_err(key):
                d = metrics.get(key)
                if d and "error_cm" in d:
                    return f"{d['error_cm']:.1f}"
                return "-"

            # Helper: get joint position from chain
            def get_joint_pos(chain_key, joint_key):
                chain = metrics.get(chain_key)
                if chain and joint_key in chain:
                    pos = chain[joint_key]
                    if pos and len(pos) == 3:
                        return f"({pos[0]:.2f} {pos[1]:.2f} {pos[2]:.2f})"
                return "-"

            # Helper: get joint angle from chain
            def get_joint_angle(chain_key, angle_key):
                chain = metrics.get(chain_key)
                if chain and angle_key in chain:
                    return f"{chain[angle_key]:.0f}"
                return "-"

            # Get targets from test case
            l_hand_target = fmt_target(t.test.get("left_hand_target"))
            r_hand_target = fmt_target(t.test.get("right_hand_target"))
            l_foot_target = fmt_target(t.test.get("left_foot_target"))
            r_foot_target = fmt_target(t.test.get("right_foot_target"))
            hip_drop = t.test.get("hip_drop", 0.0)

            # Get errors
            l_hand_err = get_err("left_hand")
            r_hand_err = get_err("right_hand")
            l_foot_err = get_err("left_foot")
            r_foot_err = get_err("right_foot")

            # Get joint positions
            l_elbow = get_joint_pos("left_arm_chain", "elbow_pos")
            r_elbow = get_joint_pos("right_arm_chain", "elbow_pos")
            l_knee = get_joint_pos("left_leg_chain", "knee_pos")
            r_knee = get_joint_pos("right_leg_chain", "knee_pos")

            # Get joint angles
            l_elbow_angle = get_joint_angle("left_arm_chain", "elbow_bend_deg")
            r_elbow_angle = get_joint_angle("right_arm_chain", "elbow_bend_deg")
            l_knee_angle = get_joint_angle("left_leg_chain", "knee_bend_deg")
            r_knee_angle = get_joint_angle("right_leg_chain", "knee_bend_deg")

            # Collect all problems
            problems = t.evaluation.get("problems", [])
            prob_str = ";".join(problems).replace(",", " ") if problems else "-"

            # Get user note
            note = t.note.replace(",", ";").replace("\n", " ") if t.note else "-"

            # Get solver diagnostics
            diag = t.solver_diagnostics if hasattr(t, 'solver_diagnostics') else {}

            # Helper to safely get diagnostic value
            def get_diag(key, default="-"):
                val = diag.get(key, default)
                if isinstance(val, bool):
                    return "Y" if val else "N"
                elif isinstance(val, float):
                    return f"{val:.3f}"
                return str(val) if val is not None else default

            # Leg diagnostics
            leg_r_elevated = get_diag("leg_R_is_elevated_kick")
            leg_r_knee_fwd = get_diag("leg_R_knee_forward_final")
            leg_r_correction = get_diag("leg_R_correction_applied")
            leg_l_elevated = get_diag("leg_L_is_elevated_kick")
            leg_l_knee_fwd = get_diag("leg_L_knee_forward_final")
            leg_l_correction = get_diag("leg_L_correction_applied")

            # Arm diagnostics
            arm_r_up = get_diag("arm_R_is_reaching_up")
            arm_r_out = get_diag("arm_R_elbow_outward_final")
            arm_r_below = get_diag("arm_R_elbow_below_hand")
            arm_l_up = get_diag("arm_L_is_reaching_up")
            arm_l_out = get_diag("arm_L_elbow_outward_final")
            arm_l_below = get_diag("arm_L_elbow_below_hand")

            # Conversion diagnostics - direction to rotation conversion for child bones
            conv_l_leg_target = get_diag("conv_l_leg_child_target_dir")
            conv_l_leg_verify = get_diag("conv_l_leg_child_verify_dir")
            conv_l_leg_err = get_diag("conv_l_leg_child_verify_err")
            conv_l_leg_ok = get_diag("conv_l_leg_child_verify_ok")
            conv_r_leg_target = get_diag("conv_r_leg_child_target_dir")
            conv_r_leg_verify = get_diag("conv_r_leg_child_verify_dir")
            conv_r_leg_err = get_diag("conv_r_leg_child_verify_err")
            conv_r_leg_ok = get_diag("conv_r_leg_child_verify_ok")
            conv_l_arm_target = get_diag("conv_l_arm_child_target_dir")
            conv_l_arm_verify = get_diag("conv_l_arm_child_verify_dir")
            conv_l_arm_err = get_diag("conv_l_arm_child_verify_err")
            conv_l_arm_ok = get_diag("conv_l_arm_child_verify_ok")
            conv_r_arm_target = get_diag("conv_r_arm_child_target_dir")
            conv_r_arm_verify = get_diag("conv_r_arm_child_verify_dir")
            conv_r_arm_err = get_diag("conv_r_arm_child_verify_err")
            conv_r_arm_ok = get_diag("conv_r_arm_child_verify_ok")

            # Post-apply verification diagnostics
            postapply_l_leg_shin = get_diag("postapply_l_leg_shin_y")
            postapply_l_leg_err = get_diag("postapply_l_leg_error_deg")
            postapply_r_leg_shin = get_diag("postapply_r_leg_shin_y")
            postapply_r_leg_err = get_diag("postapply_r_leg_error_deg")
            postapply_l_arm_forearm = get_diag("postapply_l_arm_forearm_y")
            postapply_l_arm_err = get_diag("postapply_l_arm_error_deg")
            postapply_r_arm_forearm = get_diag("postapply_r_arm_forearm_y")
            postapply_r_arm_err = get_diag("postapply_r_arm_error_deg")

            line = (
                f"{i+1},{desc},{verdict},{auto_pass},"
                f"{l_hand_target},{l_hand_err},{l_elbow},{l_elbow_angle},"
                f"{r_hand_target},{r_hand_err},{r_elbow},{r_elbow_angle},"
                f"{l_foot_target},{l_foot_err},{l_knee},{l_knee_angle},"
                f"{r_foot_target},{r_foot_err},{r_knee},{r_knee_angle},"
                f"{hip_drop},{prob_str},{note},"
                f"{leg_r_elevated},{leg_r_knee_fwd},{leg_r_correction},"
                f"{leg_l_elevated},{leg_l_knee_fwd},{leg_l_correction},"
                f"{arm_r_up},{arm_r_out},{arm_r_below},"
                f"{arm_l_up},{arm_l_out},{arm_l_below},"
                f"{conv_l_leg_target},{conv_l_leg_verify},{conv_l_leg_err},{conv_l_leg_ok},"
                f"{conv_r_leg_target},{conv_r_leg_verify},{conv_r_leg_err},{conv_r_leg_ok},"
                f"{conv_l_arm_target},{conv_l_arm_verify},{conv_l_arm_err},{conv_l_arm_ok},"
                f"{conv_r_arm_target},{conv_r_arm_verify},{conv_r_arm_err},{conv_r_arm_ok},"
                f"{postapply_l_leg_shin},{postapply_l_leg_err},{postapply_r_leg_shin},{postapply_r_leg_err},"
                f"{postapply_l_arm_forearm},{postapply_l_arm_err},{postapply_r_arm_forearm},{postapply_r_arm_err}"
            )
            lines.append(line)

        with open(csv_path, "w") as f:
            f.write("\n".join(lines))
