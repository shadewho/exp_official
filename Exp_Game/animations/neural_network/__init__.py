# Exp_Game/animations/neural_network/__init__.py
"""
Neural Network IK System - Environment-Aware Version

A learning-based full-body IK solver that:
- Trains on your animations with task-aligned losses
- Uses FK loss to actually reach target positions
- Respects environment context (ground, contacts)
- Generalizes to novel poses

Structure:
    config.py       - Rig configuration, bone lists, joint limits
    context.py      - Environment context extraction
    forward_kinematics.py - FK computation for loss calculation
    network.py      - Neural network architecture
    data.py         - Extract training data from animations
    trainer.py      - Training loop with FK/pose/contact losses
    tests.py        - Test suite proving generalization
    runtime.py      - Runtime integration with IK refinement

Usage:
    1. Extract data from animations (data.py)
    2. Train network with FK loss (trainer.py)
    3. Run test suite (tests.py) - proves learning
    4. Use in gameplay via runtime.py
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

# Network
from .network import (
    FullBodyIKNetwork,
    get_network,
    reset_network,
)

# Data extraction
from .data import (
    AnimationDataExtractor,
    get_extractor,
    extract_all,
)

# Training
from .trainer import (
    Trainer,
    TrainingReport,
    train_network,
)

# Tests
from .tests import (
    NeuralIKTestSuite,
    TestSuiteReport,
    run_test_suite,
)

# Forward kinematics
from .forward_kinematics import (
    forward_kinematics,
    forward_kinematics_batch,
    compute_fk_loss,
    compute_contact_loss,
)

# Context
from .context import (
    ContextExtractor,
    build_input_from_targets,
    augment_input,
    normalize_input,
    denormalize_input,
)

# Runtime
from .runtime import (
    NeuralIKSolver,
    IKResult,
    get_solver,
    solve_ik,
    solve_from_full_body_target,
    rotations_to_bone_dict,
)
