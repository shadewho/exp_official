# Exp_Game/animations/neural_network/__init__.py
"""
Neural Network IK System

A learning-based full-body IK solver that:
- Trains on your animations (OUTSIDE Blender for speed)
- Uses FK loss to actually reach target positions
- Respects environment context (ground, contacts)
- Generalizes to novel poses

ARCHITECTURE SEPARATION:

    BLENDER (requires bpy):
        - data.py         Extract training data from animations
        - test_panel.py   UI for testing/training

    STANDALONE (PyTorch GPU, runs outside Blender):
        - torch_trainer.py        GPU training with autograd (run from terminal)
        - forward_kinematics.py   FK math (NumPy fallback)
        - network.py              Network architecture

    SHARED:
        - config.py       Rig data, hyperparameters, paths
        - context.py      Input normalization

WORKFLOW:
    1. In Blender: Extract data → Save to disk
    2. In Terminal: python torch_trainer.py
    3. In Blender: Reload weights → Run tests
"""

# Config - used by run_test_suite and re-exported
from .config import (
    NUM_BONES,
    END_EFFECTORS,
)

# Network - used by run_test_suite and re-exported
from .network import (
    get_network,
    reset_network,
)

# Forward kinematics - used by run_test_suite
from .forward_kinematics import (
    forward_kinematics,
    get_effector_positions,
)

# Lazy imports for bpy-dependent modules (only load when needed)
def get_extractor(armature=None):
    """Lazy import to avoid bpy dependency at module load."""
    from .data import get_extractor as _get_extractor
    return _get_extractor(armature)

def extract_all(armature):
    """Lazy import to avoid bpy dependency at module load."""
    from .data import extract_all as _extract_all
    return _extract_all(armature)

# Lazy class import - returns the class itself, not an instance
def _get_AnimationDataExtractor_class():
    from .data import AnimationDataExtractor
    return AnimationDataExtractor

# Make AnimationDataExtractor importable but lazy-loaded
class _LazyAnimationDataExtractor:
    """Proxy that lazily loads AnimationDataExtractor on first use."""
    _real_class = None

    def __new__(cls, *args, **kwargs):
        if cls._real_class is None:
            from .data import AnimationDataExtractor
            cls._real_class = AnimationDataExtractor
        return cls._real_class(*args, **kwargs)

AnimationDataExtractor = _LazyAnimationDataExtractor


# =============================================================================
# TEST SUITE
# =============================================================================

class TestReport:
    """Human-readable test results."""
    def __init__(self):
        self.tests = []
        self.passed = 0
        self.failed = 0
        self.total = 0
        self.position_rmse_cm = 0.0
        self.rotation_error_deg = 0.0
        self.grade = "UNKNOWN"

    def add_test(self, name: str, passed: bool, details: dict = None):
        self.tests.append({'name': name, 'passed': passed, 'details': details or {}})
        self.total += 1
        if passed:
            self.passed += 1
        else:
            self.failed += 1


def run_test_suite(
    train_inputs,
    train_outputs,
    test_inputs,
    test_outputs,
    train_effector_targets,
    train_root_positions,
    test_effector_targets,
    test_root_positions,
    train_effector_rotations=None,
    test_effector_rotations=None,
    train_root_forwards=None,
    train_root_ups=None,
    test_root_forwards=None,
    test_root_ups=None,
):
    """
    Run comprehensive test suite on trained neural network.

    Tests position accuracy and rotation accuracy against ground truth.
    Returns human-readable TestReport.
    """
    import numpy as np
    # forward_kinematics and get_effector_positions already imported at module level

    report = TestReport()
    net = get_network()

    print("\n" + "=" * 70)
    print(" NEURAL IK TEST SUITE")
    print("=" * 70)

    n_test = len(test_inputs)
    if n_test == 0:
        print(" ERROR: No test samples")
        report.grade = "NO DATA"
        return report

    # Sample up to 200 test cases
    sample_size = min(200, n_test)
    indices = np.random.choice(n_test, sample_size, replace=False)

    # Per-effector position errors
    effector_names = END_EFFECTORS
    all_pos_errors = {name: [] for name in effector_names}
    all_rot_errors = []

    print(f"\n Testing {sample_size} samples...")

    for idx in indices:
        inp = test_inputs[idx]
        gt_output = test_outputs[idx]
        target_pos = test_effector_targets[idx].reshape(5, 3)
        root_pos = test_root_positions[idx]

        # Build root rotation matrix
        if test_root_forwards is not None and test_root_ups is not None:
            root_fwd = test_root_forwards[idx]
            root_up = test_root_ups[idx]
            root_fwd = root_fwd / (np.linalg.norm(root_fwd) + 1e-8)
            root_up = root_up / (np.linalg.norm(root_up) + 1e-8)
            root_right = np.cross(root_fwd, root_up)
            root_right = root_right / (np.linalg.norm(root_right) + 1e-8)
            root_up = np.cross(root_right, root_fwd)
            root_rot = np.array([
                [root_right[0], root_fwd[0], root_up[0]],
                [root_right[1], root_fwd[1], root_up[1]],
                [root_right[2], root_fwd[2], root_up[2]],
            ], dtype=np.float32)
        else:
            root_rot = np.eye(3, dtype=np.float32)

        # Predict
        pred_output = net.predict(inp)
        pred_rots = pred_output.reshape(NUM_BONES, 3)

        # FK to get effector positions
        fk_positions, _ = forward_kinematics(pred_rots, root_pos, root_rot)
        pred_effector_pos = get_effector_positions(fk_positions)

        # Position errors per effector
        for i, name in enumerate(effector_names):
            error = np.linalg.norm(pred_effector_pos[i] - target_pos[i])
            all_pos_errors[name].append(error)

        # Rotation error
        rot_error = np.mean(np.abs(pred_output - gt_output))
        all_rot_errors.append(rot_error)

    # Compute stats and run tests
    print("\n" + "-" * 70)
    print(" POSITION ACCURACY (per effector)")
    print("-" * 70)

    POSITION_THRESHOLD_CM = 15.0  # Pass if mean < 15cm

    total_errors = []
    for name in effector_names:
        errors = np.array(all_pos_errors[name])
        mean_err = np.mean(errors) * 100  # to cm
        std_err = np.std(errors) * 100
        max_err = np.max(errors) * 100

        passed = mean_err < POSITION_THRESHOLD_CM
        status = "PASS" if passed else "FAIL"

        print(f"   {name:12s}: Mean={mean_err:6.2f}cm  Std={std_err:5.2f}cm  Max={max_err:5.1f}cm  [{status}]")

        report.add_test(
            f"{name} Position",
            passed,
            {'mean_cm': round(mean_err, 2), 'std_cm': round(std_err, 2), 'max_cm': round(max_err, 2)}
        )
        total_errors.extend(errors)

    # Overall position RMSE
    overall_rmse = np.sqrt(np.mean(np.array(total_errors) ** 2)) * 100
    report.position_rmse_cm = round(overall_rmse, 2)

    print("-" * 70)
    print(f"   OVERALL RMSE: {overall_rmse:.2f}cm")

    # Rotation accuracy
    print("\n" + "-" * 70)
    print(" ROTATION ACCURACY")
    print("-" * 70)

    ROTATION_THRESHOLD_DEG = 30.0
    mean_rot_err = np.mean(all_rot_errors)
    rot_err_deg = np.degrees(mean_rot_err)
    rot_passed = rot_err_deg < ROTATION_THRESHOLD_DEG

    status = "PASS" if rot_passed else "FAIL"
    print(f"   Mean Rotation Error: {rot_err_deg:.2f}°  [{status}]")

    report.add_test("Rotation Accuracy", rot_passed, {'mean_deg': round(rot_err_deg, 2)})
    report.rotation_error_deg = round(rot_err_deg, 2)

    # Grade
    pct = (report.passed / report.total * 100) if report.total > 0 else 0
    if pct == 100 and overall_rmse < 5.0:
        report.grade = "EXCELLENT"
    elif pct >= 80:
        report.grade = "GOOD"
    elif pct >= 50:
        report.grade = "FAIR"
    else:
        report.grade = "POOR"

    # Summary
    print("\n" + "=" * 70)
    print(f" SUMMARY: {report.passed}/{report.total} tests passed")
    print(f" Position RMSE: {overall_rmse:.2f}cm")
    print(f" Rotation Error: {rot_err_deg:.2f}°")
    print(f" Grade: {report.grade}")
    print("=" * 70 + "\n")

    return report