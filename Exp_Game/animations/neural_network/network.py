# Exp_Game/animations/neural_network/network.py
"""
Neural Network Architecture for Full-Body IK

A simple feedforward network that learns to map:
    Input: End-effector targets (positions + rotations)
    Output: Full body bone rotations

This network LEARNS from your animations. It is not hardcoded.
"""

import numpy as np
import os
from typing import Optional, Tuple, Dict

from .config import (
    INPUT_SIZE,
    OUTPUT_SIZE,
    HIDDEN_SIZE_1,
    HIDDEN_SIZE_2,
    HIDDEN_SIZE_3,
    LEARNING_RATE,
    LIMITS_ARRAY,
    LIMIT_PENALTY_WEIGHT,
    WEIGHTS_DIR,
    BEST_WEIGHTS_PATH,
)


class FullBodyIKNetwork:
    """
    Neural network for full-body IK.

    Architecture:
        Input (50) → Hidden1 (256, LeakyReLU) → Hidden2 (256, LeakyReLU)
        → Hidden3 (128, LeakyReLU) → Output (69)

    Input: Root-relative effector targets + environment context
    Output: Axis-angle bone rotations (23 bones × 3)

    Training uses:
        - FK loss (primary): do rotations reach targets?
        - Pose loss (secondary): match training poses
        - Contact loss: feet grounded when flagged
        - Limit penalty: soft joint constraints
    """

    def __init__(self, load_weights: bool = True):
        """
        Initialize network with random weights or load saved weights.

        Args:
            load_weights: If True and weights exist, load them.
        """
        # Layer sizes
        self.input_size = INPUT_SIZE
        self.hidden1_size = HIDDEN_SIZE_1
        self.hidden2_size = HIDDEN_SIZE_2
        self.hidden3_size = HIDDEN_SIZE_3
        self.output_size = OUTPUT_SIZE

        # Initialize weights with Xavier initialization
        self._init_weights()

        # Try to load saved weights
        if load_weights and os.path.exists(BEST_WEIGHTS_PATH):
            self.load(BEST_WEIGHTS_PATH)

        # Cache for backpropagation
        self._cache = {}

        # Training stats
        self.total_updates = 0
        self.best_loss = float('inf')

    def _init_weights(self):
        """Initialize weights with Xavier initialization."""
        # Layer 1: input → hidden1
        limit1 = np.sqrt(6 / (self.input_size + self.hidden1_size))
        self.W1 = np.random.uniform(-limit1, limit1,
                                     (self.input_size, self.hidden1_size)).astype(np.float32)
        self.b1 = np.zeros(self.hidden1_size, dtype=np.float32)

        # Layer 2: hidden1 → hidden2
        limit2 = np.sqrt(6 / (self.hidden1_size + self.hidden2_size))
        self.W2 = np.random.uniform(-limit2, limit2,
                                     (self.hidden1_size, self.hidden2_size)).astype(np.float32)
        self.b2 = np.zeros(self.hidden2_size, dtype=np.float32)

        # Layer 3: hidden2 → hidden3
        limit3 = np.sqrt(6 / (self.hidden2_size + self.hidden3_size))
        self.W3 = np.random.uniform(-limit3, limit3,
                                     (self.hidden2_size, self.hidden3_size)).astype(np.float32)
        self.b3 = np.zeros(self.hidden3_size, dtype=np.float32)

        # Layer 4: hidden3 → output
        limit4 = np.sqrt(6 / (self.hidden3_size + self.output_size))
        self.W4 = np.random.uniform(-limit4, limit4,
                                     (self.hidden3_size, self.output_size)).astype(np.float32)
        self.b4 = np.zeros(self.output_size, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.

        Args:
            x: Input array of shape (batch_size, 36) or (36,)

        Returns:
            Output array of shape (batch_size, 69) or (69,)
            Values are bone rotations in radians.
        """
        # Handle single sample
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)

        # Layer 1
        z1 = x @ self.W1 + self.b1
        a1 = np.maximum(z1, 0.1 * z1)  # LeakyReLU

        # Layer 2
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(z2, 0.1 * z2)  # LeakyReLU

        # Layer 3
        z3 = a2 @ self.W3 + self.b3
        a3 = np.maximum(z3, 0.1 * z3)  # LeakyReLU

        # Layer 4 (output - no activation, raw rotation values)
        z4 = a3 @ self.W4 + self.b4

        # Cache for backprop
        self._cache = {
            'x': x,
            'z1': z1, 'a1': a1,
            'z2': z2, 'a2': a2,
            'z3': z3, 'a3': a3,
            'z4': z4,
        }

        if single:
            return z4[0]
        return z4

    def backward(self, loss_grad: np.ndarray, learning_rate: float = None) -> Dict[str, float]:
        """
        Backward pass - update weights based on loss gradient.

        Args:
            loss_grad: Gradient of loss w.r.t. output, shape (batch_size, 69) or (69,)
            learning_rate: Learning rate (uses default if None)

        Returns:
            Dict with gradient norms for monitoring
        """
        if learning_rate is None:
            learning_rate = LEARNING_RATE

        # Handle single sample
        single = loss_grad.ndim == 1
        if single:
            loss_grad = loss_grad.reshape(1, -1)

        batch_size = loss_grad.shape[0]

        # Get cached values
        x = self._cache['x']
        a1, a2, a3 = self._cache['a1'], self._cache['a2'], self._cache['a3']
        z1, z2, z3 = self._cache['z1'], self._cache['z2'], self._cache['z3']

        # Output layer gradients
        dz4 = loss_grad  # No activation on output
        dW4 = (a3.T @ dz4) / batch_size
        db4 = np.mean(dz4, axis=0)

        # Hidden layer 3 gradients
        da3 = dz4 @ self.W4.T
        dz3 = da3 * np.where(z3 > 0, 1.0, 0.1)  # LeakyReLU derivative
        dW3 = (a2.T @ dz3) / batch_size
        db3 = np.mean(dz3, axis=0)

        # Hidden layer 2 gradients
        da2 = dz3 @ self.W3.T
        dz2 = da2 * np.where(z2 > 0, 1.0, 0.1)  # LeakyReLU derivative
        dW2 = (a1.T @ dz2) / batch_size
        db2 = np.mean(dz2, axis=0)

        # Hidden layer 1 gradients
        da1 = dz2 @ self.W2.T
        dz1 = da1 * np.where(z1 > 0, 1.0, 0.1)  # LeakyReLU derivative
        dW1 = (x.T @ dz1) / batch_size
        db1 = np.mean(dz1, axis=0)

        # Update weights
        self.W4 -= learning_rate * dW4
        self.b4 -= learning_rate * db4
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

        self.total_updates += 1

        return {
            'grad_W1': float(np.linalg.norm(dW1)),
            'grad_W2': float(np.linalg.norm(dW2)),
            'grad_W3': float(np.linalg.norm(dW3)),
        }

    def compute_limit_penalty(self, output: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute soft joint limit penalty.

        The network is FREE to output any value, but violations
        add to the loss. This guides learning toward valid poses.

        Args:
            output: Network output, shape (batch_size, 69) or (69,)

        Returns:
            (penalty_value, penalty_gradient)
        """
        single = output.ndim == 1
        if single:
            output = output.reshape(1, -1)

        batch_size = output.shape[0]

        # Reshape to (batch, 23 bones, 3 angles)
        rotations = output.reshape(batch_size, -1, 3)

        # Get limits: shape (23, 6) → (1, 23, 6)
        limits = LIMITS_ARRAY.reshape(1, -1, 6)

        # Extract min/max for each axis
        x_min, x_max = limits[:, :, 0], limits[:, :, 1]
        y_min, y_max = limits[:, :, 2], limits[:, :, 3]
        z_min, z_max = limits[:, :, 4], limits[:, :, 5]

        x_rot = rotations[:, :, 0]
        y_rot = rotations[:, :, 1]
        z_rot = rotations[:, :, 2]

        # Compute violations (how much outside limits)
        x_low = np.maximum(0, x_min - x_rot)
        x_high = np.maximum(0, x_rot - x_max)
        y_low = np.maximum(0, y_min - y_rot)
        y_high = np.maximum(0, y_rot - y_max)
        z_low = np.maximum(0, z_min - z_rot)
        z_high = np.maximum(0, z_rot - z_max)

        # Squared penalty (smooth gradient)
        penalty = (x_low**2 + x_high**2 +
                   y_low**2 + y_high**2 +
                   z_low**2 + z_high**2)

        total_penalty = float(np.mean(penalty) * LIMIT_PENALTY_WEIGHT)

        # Gradient of penalty w.r.t. rotations
        grad = np.zeros_like(rotations)
        grad[:, :, 0] = 2 * (-x_low + x_high)
        grad[:, :, 1] = 2 * (-y_low + y_high)
        grad[:, :, 2] = 2 * (-z_low + z_high)
        grad = grad.reshape(batch_size, -1) * LIMIT_PENALTY_WEIGHT

        if single:
            grad = grad[0]

        return total_penalty, grad

    def predict(self, context: np.ndarray) -> np.ndarray:
        """
        Predict bone rotations for given context.

        This is the inference method - used after training.

        Args:
            context: Shape (50,) - full context vector

        Returns:
            Shape (69,) - bone rotations (axis-angle)
        """
        return self.forward(context)

    def predict_clamped(self, context: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Predict and clamp rotations to joint limits.

        Use this at runtime to ensure valid poses.

        Args:
            context: Shape (50,) - full context vector

        Returns:
            (rotations, violation_amount)
            rotations: Shape (69,) - clamped bone rotations
            violation_amount: How much was clamped (0 = no clamping needed)
        """
        raw = self.forward(context)

        # Reshape to (23, 3) for per-bone limits
        rotations = raw.reshape(-1, 3)

        # Get limits
        limits = LIMITS_ARRAY  # (23, 6)
        x_min, x_max = limits[:, 0], limits[:, 1]
        y_min, y_max = limits[:, 2], limits[:, 3]
        z_min, z_max = limits[:, 4], limits[:, 5]

        # Clamp each axis
        clamped = rotations.copy()
        clamped[:, 0] = np.clip(rotations[:, 0], x_min, x_max)
        clamped[:, 1] = np.clip(rotations[:, 1], y_min, y_max)
        clamped[:, 2] = np.clip(rotations[:, 2], z_min, z_max)

        # Compute violation amount
        violation = float(np.sum(np.abs(clamped - rotations)))

        return clamped.flatten(), violation

    def save(self, path: str = None):
        """Save network weights to file."""
        if path is None:
            path = BEST_WEIGHTS_PATH

        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        weights = {
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2,
            'W3': self.W3, 'b3': self.b3,
            'W4': self.W4, 'b4': self.b4,
            'total_updates': self.total_updates,
            'best_loss': self.best_loss,
        }
        np.save(path, weights)

    def load(self, path: str = None) -> bool:
        """
        Load network weights from file.

        Returns:
            True if loaded successfully, False otherwise.
        """
        if path is None:
            path = BEST_WEIGHTS_PATH

        if not os.path.exists(path):
            return False

        try:
            weights = np.load(path, allow_pickle=True).item()
            self.W1 = weights['W1']
            self.b1 = weights['b1']
            self.W2 = weights['W2']
            self.b2 = weights['b2']
            self.W3 = weights['W3']
            self.b3 = weights['b3']
            self.W4 = weights['W4']
            self.b4 = weights['b4']
            self.total_updates = weights.get('total_updates', 0)
            self.best_loss = weights.get('best_loss', float('inf'))
            return True
        except Exception:
            return False

    def get_stats(self) -> Dict:
        """Get network statistics."""
        return {
            'total_updates': self.total_updates,
            'best_loss': self.best_loss,
            'param_count': (
                self.W1.size + self.b1.size +
                self.W2.size + self.b2.size +
                self.W3.size + self.b3.size +
                self.W4.size + self.b4.size
            ),
            'weights_exist': os.path.exists(BEST_WEIGHTS_PATH),
        }


# =============================================================================
# Module-level singleton
# =============================================================================

_network_instance: Optional[FullBodyIKNetwork] = None


def get_network() -> FullBodyIKNetwork:
    """Get the global network instance."""
    global _network_instance
    if _network_instance is None:
        _network_instance = FullBodyIKNetwork()
    return _network_instance


def reset_network():
    """Reset the network to fresh random weights."""
    global _network_instance
    _network_instance = FullBodyIKNetwork(load_weights=False)
    return _network_instance
