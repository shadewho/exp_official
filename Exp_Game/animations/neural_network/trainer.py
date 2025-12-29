# Exp_Game/animations/neural_network/trainer.py
"""
Training Loop for Neural IK - Task-Aligned Losses

Trains the network with multiple loss terms:
    1. FK Loss (primary): Does the pose reach the target positions?
    2. Pose Loss (secondary): Does it match the training pose?
    3. Contact Loss: Are feet planted when grounded?
    4. Limit Penalty: Are rotations within joint limits?

This teaches the network to SOLVE IK, not just memorize poses.
"""

import numpy as np
import os
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .config import (
    BATCH_SIZE,
    EPOCHS_DEFAULT,
    LEARNING_RATE,
    WEIGHTS_DIR,
    BEST_WEIGHTS_PATH,
    FK_LOSS_WEIGHT,
    POSE_LOSS_WEIGHT,
    CONTACT_LOSS_WEIGHT,
    LIMIT_PENALTY_WEIGHT,
    LIMITS_MIN,
    LIMITS_MAX,
    NUM_BONES,
    INPUT_SIZE,
    OUTPUT_SIZE,
)
from .network import FullBodyIKNetwork, get_network, reset_network
from .forward_kinematics import (
    compute_fk_loss,
    compute_contact_loss,
    compute_fk_loss_gradient,
    clamp_rotations,
)
from .context import normalize_input


@dataclass
class TrainingMetrics:
    """Metrics from a training epoch."""
    epoch: int = 0
    total_loss: float = 0.0
    fk_loss: float = 0.0
    pose_loss: float = 0.0
    contact_loss: float = 0.0
    limit_penalty: float = 0.0
    test_fk_loss: float = 0.0
    test_pose_loss: float = 0.0
    time_seconds: float = 0.0
    is_best: bool = False


@dataclass
class TrainingReport:
    """Full report from a training run."""
    total_epochs: int = 0
    final_total_loss: float = 0.0
    final_fk_loss: float = 0.0
    final_pose_loss: float = 0.0
    best_fk_loss: float = float('inf')
    best_epoch: int = 0
    train_samples: int = 0
    test_samples: int = 0
    total_time: float = 0.0
    history: List[TrainingMetrics] = field(default_factory=list)

    def summary(self) -> str:
        """Generate human-readable summary."""
        fk_status = "REACHING TARGETS" if self.final_fk_loss < 0.05 else "NEEDS MORE TRAINING"

        return f"""
══════════════════════════════════════════════════════════════
 NEURAL IK TRAINING REPORT (Task-Aligned)
══════════════════════════════════════════════════════════════

 DATASET
   Training samples: {self.train_samples:,}
   Test samples:     {self.test_samples:,} (held out)

 TRAINING
   Epochs:           {self.total_epochs}
   Time:             {self.total_time:.1f}s

 LOSS BREAKDOWN (Final)
   FK Loss:          {self.final_fk_loss:.6f} (target reach error)
   Pose Loss:        {self.final_pose_loss:.6f} (pose matching)
   Total Loss:       {self.final_total_loss:.6f}

 BEST FK LOSS: {self.best_fk_loss:.6f} (epoch {self.best_epoch})

 STATUS: {fk_status}
   {"✓ Network can reach target positions" if self.final_fk_loss < 0.05 else "⚠ FK error still high - continue training or add data"}

 SAVED: {BEST_WEIGHTS_PATH}

══════════════════════════════════════════════════════════════
"""


class Trainer:
    """
    Trains the neural network with task-aligned losses.

    The key insight: we don't just want to match poses, we want to
    reach targets. FK loss measures this directly.
    """

    def __init__(self, network: FullBodyIKNetwork = None):
        """Initialize trainer."""
        self.network = network or get_network()
        self.report: Optional[TrainingReport] = None

    def train(
        self,
        data: Dict[str, np.ndarray],
        epochs: int = EPOCHS_DEFAULT,
        learning_rate: float = LEARNING_RATE,
        verbose: bool = True,
    ) -> TrainingReport:
        """
        Train the network with FK and pose losses.

        Args:
            data: Dict from AnimationDataExtractor.get_train_test_split()
                  Must include: train_inputs, train_outputs, train_effector_targets,
                               train_root_positions, train_ground_heights, train_contact_flags,
                               test_inputs, test_outputs, test_effector_targets, etc.
            epochs: Number of training epochs
            learning_rate: Learning rate
            verbose: Print progress

        Returns:
            TrainingReport with full metrics
        """
        # Extract arrays
        train_inputs = data['train_inputs']
        train_outputs = data['train_outputs']
        train_effector_targets = data['train_effector_targets']
        train_root_positions = data['train_root_positions']
        train_ground_heights = data['train_ground_heights']
        train_contact_flags = data['train_contact_flags']

        test_inputs = data['test_inputs']
        test_outputs = data['test_outputs']
        test_effector_targets = data['test_effector_targets']
        test_root_positions = data.get('test_root_positions', None)

        n_train = len(train_inputs)
        n_test = len(test_inputs)
        n_batches = max(1, n_train // BATCH_SIZE)

        # Normalize inputs for stable training
        train_inputs = normalize_input(train_inputs)
        test_inputs = normalize_input(test_inputs)

        report = TrainingReport(
            train_samples=n_train,
            test_samples=n_test,
        )

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # Shuffle training data
            perm = np.random.permutation(n_train)
            train_inputs_shuffled = train_inputs[perm]
            train_outputs_shuffled = train_outputs[perm]
            train_targets_shuffled = train_effector_targets[perm]
            train_roots_shuffled = train_root_positions[perm]
            train_ground_shuffled = train_ground_heights[perm]
            train_contact_shuffled = train_contact_flags[perm]

            # Epoch accumulators
            epoch_total = 0.0
            epoch_fk = 0.0
            epoch_pose = 0.0
            epoch_contact = 0.0
            epoch_limit = 0.0

            for batch_idx in range(n_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min(start_idx + BATCH_SIZE, n_train)

                batch_in = train_inputs_shuffled[start_idx:end_idx]
                batch_out = train_outputs_shuffled[start_idx:end_idx]
                batch_targets = train_targets_shuffled[start_idx:end_idx]
                batch_roots = train_roots_shuffled[start_idx:end_idx]
                batch_ground = train_ground_shuffled[start_idx:end_idx]
                batch_contact = train_contact_shuffled[start_idx:end_idx]

                # Forward pass
                predicted = self.network.forward(batch_in)

                # =============================================================
                # CLAMP OUTPUTS BEFORE FK LOSS
                # =============================================================
                # The FK loss should measure error AFTER clamping to joint limits.
                # This ensures the network learns to produce valid poses.
                predicted_reshaped = predicted.reshape(-1, NUM_BONES, 3)
                predicted_clamped, _ = clamp_rotations(
                    predicted_reshaped, LIMITS_MIN, LIMITS_MAX
                )
                predicted_clamped_flat = predicted_clamped.reshape(-1, OUTPUT_SIZE)

                # =============================================================
                # COMPUTE LOSSES
                # =============================================================

                # 1. FK Loss: Do predicted rotations reach target positions?
                #    Use CLAMPED outputs - this is what will actually be applied
                fk_loss, effector_errors = compute_fk_loss(
                    predicted_clamped_flat,
                    batch_targets.reshape(-1, 5, 3),
                    batch_roots,
                )

                # 2. Pose Loss: Match training poses (secondary)
                pose_error = predicted - batch_out
                pose_loss = float(np.mean(pose_error ** 2))

                # 3. Contact Loss: Feet grounded when contact flag set
                #    Also use clamped outputs
                contact_loss, _ = compute_contact_loss(
                    predicted_clamped_flat,
                    batch_ground,
                    batch_contact,
                    batch_roots,
                )

                # 4. Limit Penalty
                limit_penalty, limit_grad = self.network.compute_limit_penalty(predicted)

                # Combined loss
                total_loss = (
                    FK_LOSS_WEIGHT * fk_loss +
                    POSE_LOSS_WEIGHT * pose_loss +
                    CONTACT_LOSS_WEIGHT * contact_loss +
                    LIMIT_PENALTY_WEIGHT * limit_penalty
                )

                # =============================================================
                # COMPUTE GRADIENTS
                # =============================================================

                # Pose loss gradient (simple MSE gradient)
                pose_grad = 2 * pose_error / pose_error.size * POSE_LOSS_WEIGHT

                # FK loss gradient (numerical differentiation)
                # Note: gradient is w.r.t. raw predicted, but loss used clamped
                # This encourages network to stay within limits
                fk_grad = compute_fk_loss_gradient(
                    predicted_clamped_flat,
                    batch_targets.reshape(-1, 15),
                    batch_roots,
                ) * FK_LOSS_WEIGHT

                # Combined gradient
                total_grad = pose_grad + fk_grad + limit_grad * LIMIT_PENALTY_WEIGHT

                # Backward pass
                self.network.backward(total_grad, learning_rate)

                # Accumulate
                epoch_total += total_loss
                epoch_fk += fk_loss
                epoch_pose += pose_loss
                epoch_contact += contact_loss
                epoch_limit += limit_penalty

            # Average over batches
            epoch_total /= n_batches
            epoch_fk /= n_batches
            epoch_pose /= n_batches
            epoch_contact /= n_batches
            epoch_limit /= n_batches

            # =============================================================
            # TEST EVALUATION
            # =============================================================
            test_fk_loss = 0.0
            test_pose_loss = 0.0

            if n_test > 0:
                test_pred = self.network.forward(test_inputs)

                # Clamp test predictions for fair evaluation
                test_pred_reshaped = test_pred.reshape(-1, NUM_BONES, 3)
                test_pred_clamped, _ = clamp_rotations(
                    test_pred_reshaped, LIMITS_MIN, LIMITS_MAX
                )
                test_pred_clamped_flat = test_pred_clamped.reshape(-1, OUTPUT_SIZE)

                # FK loss on test set (using clamped)
                test_fk_loss, _ = compute_fk_loss(
                    test_pred_clamped_flat,
                    test_effector_targets.reshape(-1, 5, 3),
                    test_root_positions,
                )

                # Pose loss on test set
                test_pose_loss = float(np.mean((test_pred - test_outputs) ** 2))

            # Track best (by FK loss - the primary objective)
            is_best = test_fk_loss < report.best_fk_loss if n_test > 0 else epoch_fk < report.best_fk_loss
            if is_best:
                report.best_fk_loss = test_fk_loss if n_test > 0 else epoch_fk
                report.best_epoch = epoch
                self.network.best_loss = report.best_fk_loss
                self.network.save()

            # Record metrics
            metrics = TrainingMetrics(
                epoch=epoch,
                total_loss=epoch_total,
                fk_loss=epoch_fk,
                pose_loss=epoch_pose,
                contact_loss=epoch_contact,
                limit_penalty=epoch_limit,
                test_fk_loss=test_fk_loss,
                test_pose_loss=test_pose_loss,
                time_seconds=time.time() - epoch_start,
                is_best=is_best,
            )
            report.history.append(metrics)

            if verbose and (epoch % 10 == 0 or epoch == 1 or is_best):
                best_marker = " ★ BEST" if is_best else ""
                print(f"[Epoch {epoch:3d}] FK={epoch_fk:.4f} Pose={epoch_pose:.4f} "
                      f"Test FK={test_fk_loss:.4f}{best_marker}")

        # Finalize report
        report.total_epochs = epochs
        report.final_total_loss = report.history[-1].total_loss
        report.final_fk_loss = report.history[-1].fk_loss
        report.final_pose_loss = report.history[-1].pose_loss
        report.total_time = time.time() - start_time

        self.report = report

        if verbose:
            print(report.summary())

        return report


# =============================================================================
# Convenience functions
# =============================================================================

def train_network(
    data: Dict[str, np.ndarray],
    epochs: int = EPOCHS_DEFAULT,
    reset: bool = False,
) -> TrainingReport:
    """
    Train the global network instance.

    Args:
        data: Dataset dict from AnimationDataExtractor.get_train_test_split()
        epochs: Number of epochs
        reset: If True, reset network to random weights first

    Returns:
        TrainingReport
    """
    if reset:
        network = reset_network()
    else:
        network = get_network()

    trainer = Trainer(network)
    return trainer.train(data, epochs=epochs)


def quick_train(
    data: Dict[str, np.ndarray],
    epochs: int = 20,
) -> TrainingReport:
    """Quick training for testing - fewer epochs."""
    return train_network(data, epochs=epochs, reset=True)
