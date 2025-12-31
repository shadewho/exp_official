# Exp_Game/animations/neural_network/tests.py
"""
Test Suite for Neural IK

These tests PROVE the network is learning, not memorizing.
If these tests pass, the network can generalize to novel poses.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

from .network import FullBodyIKNetwork, get_network
from .config import INPUT_SIZE, OUTPUT_SIZE, NUM_BONES, LIMITS_MIN, LIMITS_MAX
from .forward_kinematics import (
    compute_fk_loss, clamp_rotations, compute_fk_loss_with_orientation,
)


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    score: float
    threshold: float
    message: str


@dataclass
class TestSuiteReport:
    """Results from running the full test suite."""
    tests: List[TestResult]
    passed: int
    failed: int
    total: int

    def summary(self) -> str:
        lines = [
            "",
            "══════════════════════════════════════════════════════════════",
            " NEURAL IK TEST SUITE",
            "══════════════════════════════════════════════════════════════",
            "",
        ]

        for test in self.tests:
            status = "✓ PASS" if test.passed else "✗ FAIL"
            lines.append(f" [{status}] {test.name}")
            lines.append(f"         Score: {test.score:.4f} (threshold: {test.threshold:.4f})")
            lines.append(f"         {test.message}")
            lines.append("")

        lines.append("──────────────────────────────────────────────────────────────")
        lines.append(f" TOTAL: {self.passed}/{self.total} tests passed")

        if self.failed == 0:
            lines.append(" STATUS: ✓ ALL TESTS PASSED - Network is learning")
        else:
            lines.append(f" STATUS: ✗ {self.failed} TESTS FAILED - Check training")

        lines.append("══════════════════════════════════════════════════════════════")

        return "\n".join(lines)


class NeuralIKTestSuite:
    """
    Test suite to verify the network is learning.

    Tests:
        1. Holdout Test - Can it handle unseen data?
        2. Interpolation Test - Can it blend between known poses?
        3. Consistency Test - Same input = same output?
        4. Noise Robustness - Small input changes = small output changes?
    """

    def __init__(self, network: FullBodyIKNetwork = None):
        self.network = network or get_network()

    def run_all(
        self,
        train_inputs: np.ndarray,
        train_outputs: np.ndarray,
        test_inputs: np.ndarray,
        test_outputs: np.ndarray,
        train_effector_targets: np.ndarray = None,
        train_root_positions: np.ndarray = None,
        test_effector_targets: np.ndarray = None,
        test_root_positions: np.ndarray = None,
        train_effector_rotations: np.ndarray = None,
        test_effector_rotations: np.ndarray = None,
        train_root_forwards: np.ndarray = None,
        train_root_ups: np.ndarray = None,
        test_root_forwards: np.ndarray = None,
        test_root_ups: np.ndarray = None,
    ) -> TestSuiteReport:
        """
        Run all tests.

        Args:
            train_inputs: Training data inputs
            train_outputs: Training data outputs
            test_inputs: Held-out test inputs
            test_outputs: Held-out test outputs
            train_effector_targets: Shape (n, 5, 3) - effector target positions
            train_root_positions: Shape (n, 3) - root positions
            test_effector_targets: Shape (n, 5, 3) - effector target positions
            test_root_positions: Shape (n, 3) - root positions
            train_effector_rotations: Shape (n, 5, 3) - effector target rotations (Euler)
            test_effector_rotations: Shape (n, 5, 3) - effector target rotations (Euler)
            train_root_forwards/ups: Shape (n, 3) - root orientation vectors
            test_root_forwards/ups: Shape (n, 3) - root orientation vectors

        Returns:
            TestSuiteReport with all results
        """
        results = []

        # Test 1: Holdout Position (FK loss)
        results.append(self.test_holdout_fk(
            test_inputs, test_effector_targets, test_root_positions,
            test_root_forwards, test_root_ups
        ))

        # Test 2: Holdout Orientation (geodesic loss)
        results.append(self.test_holdout_orientation(
            test_inputs, test_effector_targets, test_effector_rotations,
            test_root_positions, test_root_forwards, test_root_ups
        ))

        # Test 3: Interpolation (FK loss)
        results.append(self.test_interpolation_fk(
            train_inputs, train_effector_targets, train_root_positions,
            train_root_forwards, train_root_ups
        ))

        # Test 4: Consistency
        results.append(self.test_consistency(test_inputs))

        # Test 5: Noise robustness
        results.append(self.test_noise_robustness(test_inputs))

        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed

        return TestSuiteReport(
            tests=results,
            passed=passed,
            failed=failed,
            total=len(results),
        )

    def test_holdout_fk(
        self,
        test_inputs: np.ndarray,
        test_effector_targets: np.ndarray,
        test_root_positions: np.ndarray,
        test_root_forwards: np.ndarray = None,
        test_root_ups: np.ndarray = None,
        threshold: float = 0.15,
    ) -> TestResult:
        """
        Holdout Test (FK Position): Can the network reach targets it never saw?

        Uses FK loss (actual end effector position error in meters).
        This is the same metric used during training.

        Threshold: 0.15m average error (15cm) is reasonable for animation.
        """
        if test_inputs is None or len(test_inputs) == 0:
            return TestResult(
                name="Holdout Position Test",
                passed=False,
                score=0.0,
                threshold=threshold,
                message="No test data available",
            )

        if test_effector_targets is None or test_root_positions is None:
            return TestResult(
                name="Holdout Position Test",
                passed=False,
                score=0.0,
                threshold=threshold,
                message="Missing effector targets or root positions",
            )

        # Build root rotation matrices if available
        root_rotations = None
        if test_root_forwards is not None and test_root_ups is not None:
            from .forward_kinematics import euler_to_quaternion  # For building rotation
            # Build rotation matrix from forward/up
            batch_size = len(test_root_forwards)
            forwards = test_root_forwards / (np.linalg.norm(test_root_forwards, axis=1, keepdims=True) + 1e-8)
            ups = test_root_ups / (np.linalg.norm(test_root_ups, axis=1, keepdims=True) + 1e-8)
            rights = np.cross(forwards, ups)
            rights = rights / (np.linalg.norm(rights, axis=1, keepdims=True) + 1e-8)
            ups = np.cross(rights, forwards)
            root_rotations = np.zeros((batch_size, 3, 3), dtype=np.float32)
            root_rotations[:, :, 0] = rights
            root_rotations[:, :, 1] = forwards
            root_rotations[:, :, 2] = ups

        # Forward pass
        predicted = self.network.forward(test_inputs)

        # Clamp to joint limits
        pred_r = predicted.reshape(-1, NUM_BONES, 3)
        pred_c, _ = clamp_rotations(pred_r, LIMITS_MIN, LIMITS_MAX)
        pred_cf = pred_c.reshape(-1, OUTPUT_SIZE)

        # Compute FK loss with root rotation
        targets = test_effector_targets.reshape(-1, 5, 3)
        fk_loss, _ = compute_fk_loss(pred_cf, targets, test_root_positions, root_rotations)

        passed = fk_loss < threshold
        rmse_cm = np.sqrt(fk_loss) * 100

        return TestResult(
            name="Holdout Position Test",
            passed=passed,
            score=fk_loss,
            threshold=threshold,
            message=f"Position RMSE: {rmse_cm:.1f}cm (threshold: {np.sqrt(threshold)*100:.0f}cm)",
        )

    def test_holdout_orientation(
        self,
        test_inputs: np.ndarray,
        test_effector_targets: np.ndarray,
        test_effector_rotations: np.ndarray,
        test_root_positions: np.ndarray,
        test_root_forwards: np.ndarray = None,
        test_root_ups: np.ndarray = None,
        threshold: float = 0.5,
    ) -> TestResult:
        """
        Holdout Test (Orientation): Are effectors facing the right direction?

        Uses geodesic distance on SO(3) - proper rotation distance.
        Threshold: 0.5 rad² (~40° RMSE) is reasonable initial target.
        """
        if test_inputs is None or len(test_inputs) == 0:
            return TestResult(
                name="Holdout Orientation Test",
                passed=False,
                score=0.0,
                threshold=threshold,
                message="No test data available",
            )

        if test_effector_rotations is None:
            return TestResult(
                name="Holdout Orientation Test",
                passed=False,
                score=0.0,
                threshold=threshold,
                message="Missing effector rotation targets (re-extract data)",
            )

        # Build root rotation matrices if available
        root_rotations = None
        if test_root_forwards is not None and test_root_ups is not None:
            batch_size = len(test_root_forwards)
            forwards = test_root_forwards / (np.linalg.norm(test_root_forwards, axis=1, keepdims=True) + 1e-8)
            ups = test_root_ups / (np.linalg.norm(test_root_ups, axis=1, keepdims=True) + 1e-8)
            rights = np.cross(forwards, ups)
            rights = rights / (np.linalg.norm(rights, axis=1, keepdims=True) + 1e-8)
            ups = np.cross(rights, forwards)
            root_rotations = np.zeros((batch_size, 3, 3), dtype=np.float32)
            root_rotations[:, :, 0] = rights
            root_rotations[:, :, 1] = forwards
            root_rotations[:, :, 2] = ups

        # Forward pass
        predicted = self.network.forward(test_inputs)

        # Clamp to joint limits
        pred_r = predicted.reshape(-1, NUM_BONES, 3)
        pred_c, _ = clamp_rotations(pred_r, LIMITS_MIN, LIMITS_MAX)
        pred_cf = pred_c.reshape(-1, OUTPUT_SIZE)

        # Compute orientation loss
        targets = test_effector_targets.reshape(-1, 5, 3)
        target_rots = test_effector_rotations.reshape(-1, 5, 3)
        _, orient_loss, _, orient_errors = compute_fk_loss_with_orientation(
            pred_cf, targets, target_rots, test_root_positions, root_rotations
        )

        passed = orient_loss < threshold
        rmse_deg = np.sqrt(orient_loss) * 180 / np.pi

        return TestResult(
            name="Holdout Orientation Test",
            passed=passed,
            score=orient_loss,
            threshold=threshold,
            message=f"Orientation RMSE: {rmse_deg:.1f}° (threshold: {np.sqrt(threshold)*180/np.pi:.0f}°)",
        )

    def test_interpolation_fk(
        self,
        inputs: np.ndarray,
        effector_targets: np.ndarray,
        root_positions: np.ndarray,
        root_forwards: np.ndarray = None,
        root_ups: np.ndarray = None,
        threshold: float = 0.20,
    ) -> TestResult:
        """
        Interpolation Test (FK): Can it handle poses BETWEEN training examples?

        Creates midpoint inputs and targets, checks if predicted pose reaches them.
        Uses FK loss for meaningful spatial error measurement.
        """
        if inputs is None or len(inputs) < 10:
            return TestResult(
                name="Interpolation Test",
                passed=False,
                score=0.0,
                threshold=threshold,
                message="Not enough data for interpolation test",
            )

        if effector_targets is None or root_positions is None:
            return TestResult(
                name="Interpolation Test",
                passed=False,
                score=0.0,
                threshold=threshold,
                message="Missing effector targets or root positions",
            )

        errors = []
        n_tests = min(50, len(inputs) // 2)

        for i in range(n_tests):
            # Pick two random samples
            idx1, idx2 = np.random.choice(len(inputs), 2, replace=False)

            # Interpolate input and targets
            alpha = 0.5
            interp_input = inputs[idx1] * (1 - alpha) + inputs[idx2] * alpha
            interp_targets = effector_targets[idx1] * (1 - alpha) + effector_targets[idx2] * alpha
            interp_root = root_positions[idx1] * (1 - alpha) + root_positions[idx2] * alpha

            # Interpolate root rotation if available
            root_rot = None
            if root_forwards is not None and root_ups is not None:
                interp_fwd = root_forwards[idx1] * (1 - alpha) + root_forwards[idx2] * alpha
                interp_up = root_ups[idx1] * (1 - alpha) + root_ups[idx2] * alpha
                interp_fwd = interp_fwd / (np.linalg.norm(interp_fwd) + 1e-8)
                interp_up = interp_up / (np.linalg.norm(interp_up) + 1e-8)
                interp_right = np.cross(interp_fwd, interp_up)
                interp_right = interp_right / (np.linalg.norm(interp_right) + 1e-8)
                interp_up = np.cross(interp_right, interp_fwd)
                root_rot = np.zeros((1, 3, 3), dtype=np.float32)
                root_rot[0, :, 0] = interp_right
                root_rot[0, :, 1] = interp_fwd
                root_rot[0, :, 2] = interp_up

            # Predict
            predicted = self.network.forward(interp_input.reshape(1, -1))

            # Clamp
            pred_r = predicted.reshape(-1, NUM_BONES, 3)
            pred_c, _ = clamp_rotations(pred_r, LIMITS_MIN, LIMITS_MAX)
            pred_cf = pred_c.reshape(-1, OUTPUT_SIZE)

            # FK error
            fk_loss, _ = compute_fk_loss(pred_cf, interp_targets.reshape(1, 5, 3), interp_root.reshape(1, 3), root_rot)
            errors.append(fk_loss)

        avg_error = float(np.mean(errors))
        passed = avg_error < threshold
        rmse_cm = np.sqrt(avg_error) * 100

        return TestResult(
            name="Interpolation Test",
            passed=passed,
            score=avg_error,
            threshold=threshold,
            message=f"Interpolation RMSE: {rmse_cm:.1f}cm (threshold: {np.sqrt(threshold)*100:.0f}cm)",
        )

    def test_holdout(
        self,
        test_inputs: np.ndarray,
        test_outputs: np.ndarray,
        threshold: float = 0.1,
    ) -> TestResult:
        """
        DEPRECATED: Use test_holdout_fk instead.
        This uses rotation MSE which doesn't match training metric.
        """
        if len(test_inputs) == 0:
            return TestResult(
                name="Holdout Test (Rotation MSE)",
                passed=False,
                score=0.0,
                threshold=threshold,
                message="No test data available",
            )

        predicted = self.network.forward(test_inputs)
        error = predicted - test_outputs
        mse = float(np.mean(error ** 2))

        passed = mse < threshold

        return TestResult(
            name="Holdout Test (Rotation MSE)",
            passed=passed,
            score=mse,
            threshold=threshold,
            message=f"MSE on unseen data: {mse:.6f}",
        )

    def test_interpolation(
        self,
        inputs: np.ndarray,
        outputs: np.ndarray,
        threshold: float = 0.15,
    ) -> TestResult:
        """
        DEPRECATED: Use test_interpolation_fk instead.
        This uses rotation MSE which doesn't match training metric.
        """
        if len(inputs) < 10:
            return TestResult(
                name="Interpolation Test (Rotation MSE)",
                passed=False,
                score=0.0,
                threshold=threshold,
                message="Not enough data for interpolation test",
            )

        errors = []
        n_tests = min(50, len(inputs) // 2)

        for i in range(n_tests):
            # Pick two random samples
            idx1, idx2 = np.random.choice(len(inputs), 2, replace=False)

            # Interpolate input
            alpha = 0.5
            interp_input = inputs[idx1] * (1 - alpha) + inputs[idx2] * alpha
            expected_output = outputs[idx1] * (1 - alpha) + outputs[idx2] * alpha

            # Predict
            predicted = self.network.forward(interp_input)

            # Error
            error = np.mean((predicted - expected_output) ** 2)
            errors.append(error)

        avg_error = float(np.mean(errors))
        passed = avg_error < threshold

        return TestResult(
            name="Interpolation Test (Rotation MSE)",
            passed=passed,
            score=avg_error,
            threshold=threshold,
            message=f"Avg interpolation error: {avg_error:.6f}",
        )

    def test_consistency(
        self,
        inputs: np.ndarray,
        threshold: float = 1e-6,
    ) -> TestResult:
        """
        Consistency Test: Same input = same output?

        The network should be deterministic. Running the same input
        twice should give identical results.
        """
        if len(inputs) == 0:
            return TestResult(
                name="Consistency Test",
                passed=False,
                score=0.0,
                threshold=threshold,
                message="No test data available",
            )

        # Run same input twice
        test_input = inputs[0:1]  # Single sample
        output1 = self.network.forward(test_input)
        output2 = self.network.forward(test_input)

        diff = float(np.max(np.abs(output1 - output2)))
        passed = diff < threshold

        return TestResult(
            name="Consistency Test",
            passed=passed,
            score=diff,
            threshold=threshold,
            message=f"Max difference between runs: {diff:.10f}",
        )

    def test_noise_robustness(
        self,
        inputs: np.ndarray,
        noise_scale: float = 0.01,
        threshold: float = 0.5,
    ) -> TestResult:
        """
        Noise Robustness Test: Small input changes = small output changes?

        A well-trained network should have smooth outputs.
        Tiny input noise shouldn't cause huge output changes.
        """
        if len(inputs) == 0:
            return TestResult(
                name="Noise Robustness Test",
                passed=False,
                score=0.0,
                threshold=threshold,
                message="No test data available",
            )

        test_inputs = inputs[:10]  # Use first 10 samples

        # Original outputs
        original = self.network.forward(test_inputs)

        # Add small noise
        noise = np.random.randn(*test_inputs.shape).astype(np.float32) * noise_scale
        noisy_inputs = test_inputs + noise

        # Noisy outputs
        noisy = self.network.forward(noisy_inputs)

        # Output change should be proportional to input change
        input_change = float(np.mean(np.abs(noise)))
        output_change = float(np.mean(np.abs(noisy - original)))
        ratio = output_change / input_change if input_change > 0 else 0

        passed = ratio < threshold

        return TestResult(
            name="Noise Robustness Test",
            passed=passed,
            score=ratio,
            threshold=threshold,
            message=f"Output/input change ratio: {ratio:.2f}x",
        )


# =============================================================================
# Convenience function
# =============================================================================

def run_test_suite(
    train_inputs: np.ndarray,
    train_outputs: np.ndarray,
    test_inputs: np.ndarray,
    test_outputs: np.ndarray,
    train_effector_targets: np.ndarray = None,
    train_root_positions: np.ndarray = None,
    test_effector_targets: np.ndarray = None,
    test_root_positions: np.ndarray = None,
    train_effector_rotations: np.ndarray = None,
    test_effector_rotations: np.ndarray = None,
    train_root_forwards: np.ndarray = None,
    train_root_ups: np.ndarray = None,
    test_root_forwards: np.ndarray = None,
    test_root_ups: np.ndarray = None,
) -> TestSuiteReport:
    """
    Run the full test suite on the global network.

    Args:
        train_inputs: Training data inputs (normalized)
        train_outputs: Training data outputs (bone rotations)
        test_inputs: Held-out test inputs (normalized)
        test_outputs: Held-out test outputs (bone rotations)
        train_effector_targets: Shape (n, 5, 3) - target positions for FK test
        train_root_positions: Shape (n, 3) - root positions for FK test
        test_effector_targets: Shape (n, 5, 3) - target positions for FK test
        test_root_positions: Shape (n, 3) - root positions for FK test
        train/test_effector_rotations: Shape (n, 5, 3) - target Euler rotations
        train/test_root_forwards/ups: Shape (n, 3) - root orientation vectors

    Returns:
        TestSuiteReport
    """
    suite = NeuralIKTestSuite()
    report = suite.run_all(
        train_inputs, train_outputs,
        test_inputs, test_outputs,
        train_effector_targets, train_root_positions,
        test_effector_targets, test_root_positions,
        train_effector_rotations, test_effector_rotations,
        train_root_forwards, train_root_ups,
        test_root_forwards, test_root_ups,
    )
    print(report.summary())
    return report
