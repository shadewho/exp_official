# Exp_Game/animations/neural_network/trainer.py
"""
Training Data Structures (Training runs OUTSIDE Blender)

IMPORTANT: Actual training is done by standalone_trainer.py
Run it from command line for maximum speed:

    cd Exp_Game/animations/neural_network
    python standalone_trainer.py

This file only contains data structures for compatibility with
test results and UI status display.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from .config import BEST_WEIGHTS_PATH


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
 NEURAL IK TRAINING REPORT
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


# =============================================================================
# DEPRECATED - Training now runs outside Blender
# =============================================================================

class Trainer:
    """
    DEPRECATED: Use standalone_trainer.py instead.

    Training must run outside Blender for performance.
    This class is kept only for backwards compatibility.
    """

    def __init__(self, network=None):
        raise NotImplementedError(
            "Training no longer runs in Blender.\n"
            "Use standalone_trainer.py from command line:\n"
            "  cd Exp_Game/animations/neural_network\n"
            "  python standalone_trainer.py"
        )


def train_network(*args, **kwargs):
    """
    DEPRECATED: Use standalone_trainer.py instead.
    """
    raise NotImplementedError(
        "Training no longer runs in Blender.\n"
        "Use standalone_trainer.py from command line:\n"
        "  cd Exp_Game/animations/neural_network\n"
        "  python standalone_trainer.py"
    )


def quick_train(*args, **kwargs):
    """
    DEPRECATED: Use standalone_trainer.py instead.
    """
    raise NotImplementedError(
        "Training no longer runs in Blender.\n"
        "Use standalone_trainer.py from command line:\n"
        "  cd Exp_Game/animations/neural_network\n"
        "  python standalone_trainer.py"
    )
