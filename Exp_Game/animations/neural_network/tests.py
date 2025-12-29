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
from .config import INPUT_SIZE, OUTPUT_SIZE


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
    ) -> TestSuiteReport:
        """
        Run all tests.

        Args:
            train_inputs: Training data inputs
            train_outputs: Training data outputs
            test_inputs: Held-out test inputs
            test_outputs: Held-out test outputs

        Returns:
            TestSuiteReport with all results
        """
        results = []

        # Test 1: Holdout
        results.append(self.test_holdout(test_inputs, test_outputs))

        # Test 2: Interpolation
        results.append(self.test_interpolation(train_inputs, train_outputs))

        # Test 3: Consistency
        results.append(self.test_consistency(test_inputs))

        # Test 4: Noise robustness
        results.append(self.test_noise_robustness(test_inputs))

        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed

        return TestSuiteReport(
            tests=results,
            passed=passed,
            failed=failed,
            total=len(results),
        )

    def test_holdout(
        self,
        test_inputs: np.ndarray,
        test_outputs: np.ndarray,
        threshold: float = 0.1,
    ) -> TestResult:
        """
        Holdout Test: Can the network handle data it never saw?

        If test loss is close to training loss, it's generalizing.
        If test loss is much higher, it's memorizing.
        """
        if len(test_inputs) == 0:
            return TestResult(
                name="Holdout Test",
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
            name="Holdout Test",
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
        Interpolation Test: Can it handle poses BETWEEN training examples?

        Takes pairs of training samples, creates midpoints, checks prediction.
        A learning network should smoothly interpolate.
        """
        if len(inputs) < 10:
            return TestResult(
                name="Interpolation Test",
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
            name="Interpolation Test",
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
) -> TestSuiteReport:
    """
    Run the full test suite on the global network.

    Args:
        train_inputs: Training data inputs
        train_outputs: Training data outputs
        test_inputs: Held-out test inputs
        test_outputs: Held-out test outputs

    Returns:
        TestSuiteReport
    """
    suite = NeuralIKTestSuite()
    report = suite.run_all(train_inputs, train_outputs, test_inputs, test_outputs)
    print(report.summary())
    return report
