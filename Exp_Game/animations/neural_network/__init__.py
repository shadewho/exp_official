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
        - runtime.py      Apply poses during gameplay

    STANDALONE (pure NumPy, runs outside Blender):
        - standalone_trainer.py   Training loop (run from terminal)
        - forward_kinematics.py   FK math for loss computation
        - network.py              Network architecture

    SHARED:
        - config.py       Rig data, hyperparameters, paths
        - context.py      Input normalization

WORKFLOW:
    1. In Blender: Extract data → Save to disk
    2. In Terminal: python standalone_trainer.py
    3. In Blender: Reload weights → Use in gameplay
"""

# Config
from .config import (
    CONTROLLED_BONES,
    NUM_BONES,
    INPUT_SIZE,
    OUTPUT_SIZE,
    JOINT_LIMITS_DEG,
    JOINT_LIMITS_RAD,
    END_EFFECTORS,
    IK_CHAINS,
    TASK_TYPES,
)

# Network (used by both training and runtime)
from .network import (
    FullBodyIKNetwork,
    get_network,
    reset_network,
)

# Data extraction (requires bpy - Blender only)
from .data import (
    AnimationDataExtractor,
    get_extractor,
    extract_all,
)

# Training data structures (actual training in standalone_trainer.py)
from .trainer import (
    TrainingReport,
    TrainingMetrics,
)

# Tests
from .tests import (
    NeuralIKTestSuite,
    TestSuiteReport,
    run_test_suite,
)

# Forward kinematics (pure NumPy)
from .forward_kinematics import (
    forward_kinematics,
    forward_kinematics_batch,
    compute_fk_loss,
    compute_contact_loss,
)

# Context (pure NumPy)
from .context import (
    ContextExtractor,
    build_input_from_targets,
    augment_input,
    normalize_input,
    denormalize_input,
)

# Runtime (requires bpy - Blender only)
from .runtime import (
    NeuralIKSolver,
    IKResult,
    get_solver,
    solve_ik,
    solve_from_full_body_target,
    rotations_to_bone_dict,
)
