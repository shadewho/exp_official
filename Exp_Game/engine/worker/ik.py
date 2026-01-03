# engine/worker/ik.py
"""
Neural IK Worker Module

Pure NumPy implementation for engine workers. NO bpy imports allowed.

This module handles neural IK inference in worker processes.
All functions must be pickle-safe and use only numpy.
"""

import numpy as np
import time
from typing import Dict, Optional, Tuple

# =============================================================================
# WORKER GLOBALS (cached at startup via CACHE_NEURAL_IK job)
# =============================================================================

_ik_weights: Optional[Dict] = None  # Network weights
_ik_log_enabled: bool = False       # Logging toggle


def _log(msg: str):
    """Log if enabled. Worker logging goes to diagnostics."""
    if _ik_log_enabled:
        print(f"[IK] {msg}")


# =============================================================================
# WEIGHT CACHING (called via CACHE_NEURAL_IK job)
# =============================================================================

def cache_weights(weights_dict: Dict) -> Dict:
    """
    Cache network weights in worker memory.

    Called once at game start via CACHE_NEURAL_IK broadcast job.
    Weights stay in memory for entire game session.

    Args:
        weights_dict: Dict with keys 'W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'W4', 'b4'

    Returns:
        Result dict with success status and param count
    """
    global _ik_weights

    # Validate required keys
    required_keys = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'W4', 'b4']
    missing = [k for k in required_keys if k not in weights_dict]

    if missing:
        return {
            "success": False,
            "error": f"Missing weight keys: {missing}"
        }

    # Store weights
    _ik_weights = weights_dict

    # Count parameters
    param_count = sum(weights_dict[k].size for k in required_keys)

    _log(f"Cached {param_count} parameters")

    return {
        "success": True,
        "param_count": param_count,
        "message": "Neural IK weights cached"
    }


# =============================================================================
# CONSTANTS (hardcoded to avoid config.py import)
# =============================================================================

# Effector names in order - must match training data
_EFFECTORS = ["LeftHand", "RightHand", "LeftFoot", "RightFoot", "Head"]
_CONTACT_EFFECTORS = ["LeftFoot", "RightFoot"]

# Input structure: 50 dimensions total
# [0:30]  - 5 effectors × 6 (pos xyz + rot xyz)
# [30:36] - root orientation (forward xyz + up xyz)
# [36:48] - 2 feet × 6 (height, normal xyz, grounded, contact)
# [48:50] - motion state (phase, task_type)

_POSITION_SCALE = 1.0      # Positions already in meters
_ROTATION_SCALE = 3.141592653589793  # np.pi - divide to get ~[-1, 1]
_HEIGHT_SCALE = 2.0        # Character height ~2m
_GROUND_START = 36         # Index where ground context starts


# =============================================================================
# INPUT BUILDING (worker builds input from raw data)
# =============================================================================

def build_input(data: Dict) -> np.ndarray:
    """
    Build 50-dim input vector from raw target data.

    Main thread sends minimal data, worker builds the full input.

    Args:
        data: Dict with:
            - effector_positions: {name: (x,y,z)} for each effector
            - effector_rotations: {name: (x,y,z)} euler angles (optional)
            - root_position: (x,y,z)
            - root_forward: (x,y,z) unit vector
            - root_up: (x,y,z) unit vector
            - ground_height: float
            - ground_normal: (x,y,z) unit vector (optional, default up)
            - contact_left: bool
            - contact_right: bool
            - motion_phase: float 0-1
            - task_type: int

    Returns:
        Shape (50,) input vector
    """
    input_vec = np.zeros(50, dtype=np.float32)

    # Get data with defaults
    effector_positions = data.get("effector_positions", {})
    effector_rotations = data.get("effector_rotations", {})
    root_pos = np.array(data.get("root_position", (0, 0, 0)), dtype=np.float32)
    root_fwd = np.array(data.get("root_forward", (0, 1, 0)), dtype=np.float32)
    root_up = np.array(data.get("root_up", (0, 0, 1)), dtype=np.float32)
    ground_height = data.get("ground_height", 0.0)
    ground_normal = np.array(data.get("ground_normal", (0, 0, 1)), dtype=np.float32)
    contact_left = 1.0 if data.get("contact_left", True) else 0.0
    contact_right = 1.0 if data.get("contact_right", True) else 0.0
    motion_phase = data.get("motion_phase", 0.0)
    task_type = data.get("task_type", 0)

    # Build rotation matrix for world-to-root transform
    root_right = np.cross(root_fwd, root_up)
    root_right /= (np.linalg.norm(root_right) + 1e-8)
    rot_matrix = np.array([root_right, root_fwd, root_up])  # 3x3

    # 1. Effector data (30 values)
    for i, effector_name in enumerate(_EFFECTORS):
        base = i * 6

        # Position (root-relative)
        if effector_name in effector_positions:
            world_pos = np.array(effector_positions[effector_name], dtype=np.float32)
            relative_pos = rot_matrix @ (world_pos - root_pos)
            input_vec[base:base+3] = relative_pos
        # else: zeros (default)

        # Rotation (world euler)
        if effector_name in effector_rotations:
            rot = effector_rotations[effector_name]
            input_vec[base+3:base+6] = rot
        # else: zeros (default)

    # 2. Root orientation (6 values)
    input_vec[30:33] = root_fwd
    input_vec[33:36] = root_up

    # 3. Ground context (12 values = 2 feet × 6)
    for foot_idx, foot_name in enumerate(_CONTACT_EFFECTORS):
        foot_base = _GROUND_START + foot_idx * 6

        # Height offset
        if foot_name in effector_positions:
            foot_pos = np.array(effector_positions[foot_name], dtype=np.float32)
            height_offset = foot_pos[2] - ground_height
        else:
            height_offset = 0.0

        input_vec[foot_base] = height_offset
        input_vec[foot_base+1:foot_base+4] = ground_normal

        # Contact flags
        is_grounded = contact_left if foot_idx == 0 else contact_right
        input_vec[foot_base+4] = is_grounded
        input_vec[foot_base+5] = 1.0  # desired_contact always 1

    # 4. Motion state (2 values)
    input_vec[48] = motion_phase
    input_vec[49] = float(task_type)

    return input_vec


# =============================================================================
# INPUT NORMALIZATION
# =============================================================================

def normalize_input(input_vector: np.ndarray) -> np.ndarray:
    """
    Normalize input vector for network consumption.

    Scales positions and rotations to comparable magnitudes (~[-1, 1]).
    Must match context.py normalize_input() exactly.

    Args:
        input_vector: Shape (50,) or (batch, 50) - raw input

    Returns:
        Normalized input, same shape
    """
    normalized = input_vector.copy()
    single = normalized.ndim == 1
    if single:
        normalized = normalized.reshape(1, -1)

    # Effector data (30 values): alternating pos/rot per effector
    for i in range(5):
        base = i * 6
        # Position: divide by POSITION_SCALE
        normalized[:, base:base+3] /= _POSITION_SCALE
        # Rotation: divide by pi to get ~[-1, 1]
        normalized[:, base+3:base+6] /= _ROTATION_SCALE

    # Root orientation (indices 30-35): already unit vectors, skip

    # Ground context (indices 36-47)
    for foot in range(2):
        foot_base = _GROUND_START + foot * 6
        # Height offset: divide by height scale
        normalized[:, foot_base] /= _HEIGHT_SCALE
        # Normal (3 values): already unit, skip
        # Contact flags (2 values): already 0/1, skip

    # Motion state (indices 48-49): phase is 0-1, task is small int, skip

    if single:
        normalized = normalized[0]

    return normalized


# =============================================================================
# AXIS-ANGLE TO QUATERNION
# =============================================================================

def axis_angle_to_quaternion(axis_angles: np.ndarray) -> np.ndarray:
    """
    Convert axis-angle rotations to quaternions.

    Axis-angle: 3D vector where direction = axis, magnitude = angle (radians)
    Quaternion: (w, x, y, z) where w = cos(θ/2), (x,y,z) = axis * sin(θ/2)

    Args:
        axis_angles: Shape (N, 3) or (3,) - axis-angle rotations

    Returns:
        Shape (N, 4) or (4,) - quaternions as (w, x, y, z)
    """
    single = axis_angles.ndim == 1
    if single:
        axis_angles = axis_angles.reshape(1, -1)

    n = axis_angles.shape[0]
    quats = np.zeros((n, 4), dtype=np.float32)

    # Compute angle (magnitude of axis-angle vector)
    angles = np.linalg.norm(axis_angles, axis=1)  # (N,)

    # Handle near-zero angles (identity rotation)
    small_angle_mask = angles < 1e-8
    large_angle_mask = ~small_angle_mask

    # Small angles: quaternion is (1, 0, 0, 0)
    quats[small_angle_mask, 0] = 1.0

    # Large angles: compute properly
    if np.any(large_angle_mask):
        valid_angles = angles[large_angle_mask]
        valid_aa = axis_angles[large_angle_mask]

        # Normalized axis
        axes = valid_aa / valid_angles[:, np.newaxis]

        # Half angle
        half_angles = valid_angles * 0.5

        # Quaternion components
        quats[large_angle_mask, 0] = np.cos(half_angles)  # w
        sin_half = np.sin(half_angles)
        quats[large_angle_mask, 1] = axes[:, 0] * sin_half  # x
        quats[large_angle_mask, 2] = axes[:, 1] * sin_half  # y
        quats[large_angle_mask, 3] = axes[:, 2] * sin_half  # z

    if single:
        return quats[0]
    return quats


def axis_angle_to_quaternion_batch(rotations_flat: np.ndarray) -> np.ndarray:
    """
    Convert network output (69,) to quaternions (23, 4).

    Args:
        rotations_flat: Shape (69,) - 23 bones × 3 axis-angle

    Returns:
        Shape (23, 4) - quaternions for each bone as (w, x, y, z)
    """
    # Reshape to (23, 3)
    axis_angles = rotations_flat.reshape(-1, 3)
    return axis_angle_to_quaternion(axis_angles)


# =============================================================================
# NETWORK FORWARD PASS
# =============================================================================

def network_forward(
    weights: Dict,
    x: np.ndarray,
) -> np.ndarray:
    """
    Forward pass through the neural IK network.

    Architecture: 50 → 256 → 256 → 128 → 69
    Activation: LeakyReLU (alpha=0.1)

    Args:
        weights: Dict with keys 'W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'W4', 'b4'
                 W1: (50, 256), b1: (256,)
                 W2: (256, 256), b2: (256,)
                 W3: (256, 128), b3: (128,)
                 W4: (128, 69), b4: (69,)
        x: Input array, shape (50,) or (batch, 50)

    Returns:
        Output array, shape (69,) or (batch, 69)
        Values are axis-angle bone rotations for 23 bones.
    """
    # Handle single sample
    single = x.ndim == 1
    if single:
        x = x.reshape(1, -1)

    # Layer 1: input → hidden1
    z1 = x @ weights['W1'] + weights['b1']
    a1 = np.maximum(z1, 0.1 * z1)  # LeakyReLU(0.1)

    # Layer 2: hidden1 → hidden2
    z2 = a1 @ weights['W2'] + weights['b2']
    a2 = np.maximum(z2, 0.1 * z2)  # LeakyReLU(0.1)

    # Layer 3: hidden2 → hidden3
    z3 = a2 @ weights['W3'] + weights['b3']
    a3 = np.maximum(z3, 0.1 * z3)  # LeakyReLU(0.1)

    # Layer 4: hidden3 → output (no activation)
    z4 = a3 @ weights['W4'] + weights['b4']

    if single:
        return z4[0]
    return z4


def network_forward_timed(
    weights: Dict,
    x: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Forward pass with timing.

    Args:
        weights: Network weights dict
        x: Input array

    Returns:
        (output, elapsed_microseconds)
    """
    t0 = time.perf_counter()
    output = network_forward(weights, x)
    elapsed_us = (time.perf_counter() - t0) * 1e6
    return output, elapsed_us


# =============================================================================
# FULL INFERENCE PIPELINE (called via NEURAL_IK_SOLVE job)
# =============================================================================

def solve(data: Dict) -> Dict:
    """
    Full neural IK inference pipeline.

    Worker builds input from raw data, runs inference, returns quaternions.
    Main thread only sends target positions - all computation is here.

    Args:
        data: Raw target data dict (see build_input() for structure)

    Returns:
        Dict with:
            - success: bool
            - bone_rotations: (23, 4) quaternions if successful
            - inference_us: float, inference time in microseconds
            - error: str if failed
    """
    global _ik_weights

    # Check weights cached
    if _ik_weights is None:
        return {
            "success": False,
            "error": "Weights not cached - call CACHE_NEURAL_IK first"
        }

    try:
        t0 = time.perf_counter()

        # 1. Build input from raw data (worker does this, not main thread)
        input_vector = build_input(data)

        # 2. Normalize input
        normalized = normalize_input(input_vector)

        # 3. Network forward pass (50 → 69 axis-angles)
        axis_angles = network_forward(_ik_weights, normalized)

        # 4. Convert to quaternions (69 → 23×4)
        quaternions = axis_angle_to_quaternion_batch(axis_angles)

        elapsed_us = (time.perf_counter() - t0) * 1e6

        # Comprehensive diagnostic stats
        input_min = float(input_vector.min())
        input_max = float(input_vector.max())
        input_mean = float(input_vector.mean())
        norm_min = float(normalized.min())
        norm_max = float(normalized.max())
        norm_mean = float(normalized.mean())
        output_min = float(axis_angles.min())
        output_max = float(axis_angles.max())
        output_mean = float(axis_angles.mean())

        # Per-effector relative positions (first 30 values = 5 effectors × 6)
        effector_rel_pos = {}
        for i, name in enumerate(_EFFECTORS):
            base = i * 6
            pos = input_vector[base:base+3]
            effector_rel_pos[name] = (float(pos[0]), float(pos[1]), float(pos[2]))

        # Axis-angle stats per bone (23 bones × 3)
        aa_reshaped = axis_angles.reshape(23, 3)
        angle_magnitudes = np.linalg.norm(aa_reshaped, axis=1)  # Rotation magnitude per bone
        angle_min_deg = float(np.rad2deg(angle_magnitudes.min()))
        angle_max_deg = float(np.rad2deg(angle_magnitudes.max()))
        angle_mean_deg = float(np.rad2deg(angle_magnitudes.mean()))

        _log(f"Solve: {elapsed_us:.1f}µs in=[{input_min:.2f},{input_max:.2f}] out=[{output_min:.2f},{output_max:.2f}]")

        return {
            "success": True,
            "bone_rotations": quaternions,  # Shape (23, 4)
            "inference_us": elapsed_us,
            # Input stats
            "input_raw": input_vector.tolist(),  # Full 50-dim input for debugging
            "input_range": (input_min, input_max),
            "input_mean": input_mean,
            "normalized_range": (norm_min, norm_max),
            "normalized_mean": norm_mean,
            # Output stats
            "output_range": (output_min, output_max),
            "output_mean": output_mean,
            "axis_angles": axis_angles.tolist(),  # Full 69-dim output
            "angle_magnitudes_deg": (angle_min_deg, angle_max_deg, angle_mean_deg),
            # Effector relative positions (what network sees)
            "effector_rel_pos": effector_rel_pos,
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Inference failed: {str(e)}"
        }


# =============================================================================
# VERIFICATION TEST (run standalone to verify against network.py)
# =============================================================================

if __name__ == "__main__":
    # Verification test: Compare this implementation against network.py
    # Run: python ik.py (from this directory)
    import sys
    import os

    print("=" * 60)
    print(" NEURAL IK WORKER - VERIFICATION TEST")
    print("=" * 60)

    # Add parent paths for imports
    script_dir = os.path.dirname(os.path.abspath(__file__))
    addon_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    sys.path.insert(0, addon_dir)

    # Load weights
    weights_path = os.path.join(
        addon_dir,
        "Exp_Game", "animations", "neural_network",
        "training_data", "weights", "best.npy"
    )

    if not os.path.exists(weights_path):
        print(f"ERROR: Weights not found at {weights_path}")
        sys.exit(1)

    print(f"\nLoading weights from: {weights_path}")
    weights = np.load(weights_path, allow_pickle=True).item()

    print(f"Weights loaded:")
    for key in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'W4', 'b4']:
        print(f"  {key}: {weights[key].shape}")

    # Create random test input
    np.random.seed(42)  # Reproducible
    test_input = np.random.randn(50).astype(np.float32)

    print(f"\nTest input shape: {test_input.shape}")
    print(f"Test input range: [{test_input.min():.3f}, {test_input.max():.3f}]")

    # Run worker version
    print("\n" + "-" * 60)
    print(" Running WORKER forward pass...")
    print("-" * 60)

    worker_output, worker_time = network_forward_timed(weights, test_input)

    print(f"Worker output shape: {worker_output.shape}")
    print(f"Worker output range: [{worker_output.min():.3f}, {worker_output.max():.3f}]")
    print(f"Worker time: {worker_time:.1f} µs")

    # Run original network.py version for comparison
    print("\n" + "-" * 60)
    print(" Running ORIGINAL network.py forward pass...")
    print("-" * 60)

    try:
        from Exp_Game.animations.neural_network.network import FullBodyIKNetwork

        network = FullBodyIKNetwork(load_weights=True)

        t0 = time.perf_counter()
        original_output = network.forward(test_input)
        original_time = (time.perf_counter() - t0) * 1e6

        print(f"Original output shape: {original_output.shape}")
        print(f"Original output range: [{original_output.min():.3f}, {original_output.max():.3f}]")
        print(f"Original time: {original_time:.1f} µs")

        # Compare outputs
        print("\n" + "-" * 60)
        print(" COMPARISON")
        print("-" * 60)

        diff = np.abs(worker_output - original_output)
        max_diff = diff.max()
        mean_diff = diff.mean()

        print(f"Max difference:  {max_diff:.2e}")
        print(f"Mean difference: {mean_diff:.2e}")

        # Verify match
        TOLERANCE = 1e-6
        if max_diff < TOLERANCE:
            print(f"\n[PASS] Outputs match within tolerance ({TOLERANCE})")
        else:
            print(f"\n[FAIL] Outputs differ by {max_diff:.2e} (tolerance: {TOLERANCE})")
            print("\nFirst 10 values comparison:")
            print(f"  Worker:   {worker_output[:10]}")
            print(f"  Original: {original_output[:10]}")
            sys.exit(1)

    except ImportError as e:
        print(f"Could not import network.py for comparison: {e}")
        print("Standalone test only - no comparison performed.")

    # Test normalize_input
    print("\n" + "-" * 60)
    print(" NORMALIZE_INPUT TEST")
    print("-" * 60)

    test_input_raw = np.random.randn(50).astype(np.float32)
    worker_normalized = normalize_input(test_input_raw)

    print(f"Raw input range:        [{test_input_raw.min():.3f}, {test_input_raw.max():.3f}]")
    print(f"Normalized input range: [{worker_normalized.min():.3f}, {worker_normalized.max():.3f}]")

    try:
        from Exp_Game.animations.neural_network.context import normalize_input as original_normalize
        original_normalized = original_normalize(test_input_raw)

        norm_diff = np.abs(worker_normalized - original_normalized).max()
        print(f"Max diff from original: {norm_diff:.2e}")

        if norm_diff < 1e-6:
            print("[PASS] normalize_input matches original")
        else:
            print(f"[FAIL] normalize_input differs by {norm_diff:.2e}")
            sys.exit(1)
    except ImportError as e:
        print(f"Could not import context.py for comparison: {e}")
        print("Verifying normalization logic manually...")
        # Check that rotation values got divided by pi
        if abs(worker_normalized[3] - test_input_raw[3] / np.pi) < 1e-6:
            print("[PASS] Rotation normalization correct (divided by pi)")
        else:
            print("[FAIL] Rotation normalization incorrect")
            sys.exit(1)

    # Test axis_angle_to_quaternion
    print("\n" + "-" * 60)
    print(" AXIS_ANGLE_TO_QUATERNION TEST")
    print("-" * 60)

    # Test cases: various axis-angles
    test_cases = [
        np.array([0.0, 0.0, 0.0]),          # Identity
        np.array([np.pi/2, 0.0, 0.0]),      # 90° around X
        np.array([0.0, np.pi/2, 0.0]),      # 90° around Y
        np.array([0.0, 0.0, np.pi/2]),      # 90° around Z
        np.array([0.5, 0.3, 0.2]),          # Arbitrary
        np.array([1.0, 1.0, 1.0]),          # Arbitrary larger
    ]

    print("Testing individual conversions...")
    all_passed = True

    for i, aa in enumerate(test_cases):
        worker_quat = axis_angle_to_quaternion(aa)
        angle = np.linalg.norm(aa)

        if angle < 1e-8:
            # Identity case
            expected = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            axis = aa / angle
            half = angle / 2
            expected = np.array([
                np.cos(half),
                axis[0] * np.sin(half),
                axis[1] * np.sin(half),
                axis[2] * np.sin(half),
            ])

        diff = np.abs(worker_quat - expected).max()
        status = "OK" if diff < 1e-6 else "FAIL"
        if diff >= 1e-6:
            all_passed = False
        print(f"  Case {i}: aa={aa}, diff={diff:.2e} [{status}]")

    if all_passed:
        print("[PASS] axis_angle_to_quaternion correct")
    else:
        print("[FAIL] axis_angle_to_quaternion has errors")
        sys.exit(1)

    # Test batch conversion (23 bones)
    test_rotations = np.random.randn(69).astype(np.float32) * 0.5  # Reasonable rotation range
    batch_quats = axis_angle_to_quaternion_batch(test_rotations)
    print(f"\nBatch test: (69,) -> {batch_quats.shape} [OK]")

    # Verify quaternions are unit length
    quat_norms = np.linalg.norm(batch_quats, axis=1)
    max_norm_error = np.abs(quat_norms - 1.0).max()
    print(f"Max quaternion norm error: {max_norm_error:.2e} [{'OK' if max_norm_error < 1e-6 else 'FAIL'}]")

    # Batch test
    print("\n" + "-" * 60)
    print(" BATCH TEST (100 samples)")
    print("-" * 60)

    batch_input = np.random.randn(100, 50).astype(np.float32)
    batch_output, batch_time = network_forward_timed(weights, batch_input)

    print(f"Batch input shape:  {batch_input.shape}")
    print(f"Batch output shape: {batch_output.shape}")
    print(f"Batch time: {batch_time:.1f} µs ({batch_time/100:.1f} µs per sample)")

    print("\n" + "=" * 60)
    print(" VERIFICATION COMPLETE")
    print("=" * 60)
