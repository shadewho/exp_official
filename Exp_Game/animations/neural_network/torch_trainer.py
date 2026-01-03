"""
PyTorch GPU Neural IK Trainer
=============================
Single-file GPU-accelerated trainer using PyTorch autograd.

Usage:
    cd <addon_path>/Exp_Game/animations/neural_network
    python torch_trainer.py

Prerequisites:
    - PyTorch with CUDA (pip install torch --index-url https://download.pytorch.org/whl/cu121)
    - training_data.npz (extracted in Blender first)

Output:
    - weights/best.npy (compatible with existing Blender loader)
"""

import os
import sys
import time
import math
import numpy as np

# =============================================================================
# PyTorch setup
# =============================================================================
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    print("ERROR: PyTorch not installed. Run: pip install torch --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)

# Device selection
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    # Performance optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    DEVICE = torch.device("cpu")
    print("WARNING: CUDA not available, using CPU (will be slow)")

# =============================================================================
# Config imports (same directory - bypass package structure)
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from config import (
    DATA_DIR, WEIGHTS_DIR, BEST_WEIGHTS_PATH,
    INPUT_SIZE, HIDDEN_SIZE_1, HIDDEN_SIZE_2, OUTPUT_SIZE,
    NUM_BONES, END_EFFECTORS,
    PARENT_INDICES, LENGTHS_ARRAY, REST_POSITIONS_ARRAY,
    REST_ORIENTATIONS, LOCAL_OFFSETS,
    LIMITS_MIN, LIMITS_MAX, BONE_TO_INDEX,
)
from context import normalize_input

# =============================================================================
# Hyperparameters
# =============================================================================
BATCH_SIZE = 512          # Large batches for GPU saturation
MAX_EPOCHS = 500
EARLY_STOP_PATIENCE = 75  # More patience with new loss weights
LEARNING_RATE = 1e-3

# Loss weights
POSITION_WEIGHT = 5.0     # FK position loss - DOMINANT for IK accuracy
ORIENTATION_WEIGHT = 0.5  # Effector rotation loss
POSE_WEIGHT = 0.1         # Match training poses (low - positions matter more)
CONTACT_WEIGHT = 0.3      # Foot grounding
LIMIT_WEIGHT = 0.3        # Joint limits (anatomical)
SMOOTH_WEIGHT = 0.05      # Temporal smoothness (helps noise robustness)

# Ground plane
GROUND_HEIGHT = 0.0
CONTACT_THRESHOLD = 0.1

# Mixed precision (set False if NaN issues)
USE_AMP = True
INPUT_NOISE_STD = 0.01  # small noise on inputs for robustness

# =============================================================================
# Pre-computed tensors (move to GPU once)
# =============================================================================
PARENT_INDICES_T = torch.tensor(PARENT_INDICES, dtype=torch.long, device=DEVICE)
BONE_LENGTHS_T = torch.tensor(LENGTHS_ARRAY, dtype=torch.float32, device=DEVICE)
REST_POSITIONS_T = torch.tensor(REST_POSITIONS_ARRAY, dtype=torch.float32, device=DEVICE)
LOCAL_OFFSETS_T = torch.tensor(LOCAL_OFFSETS, dtype=torch.float32, device=DEVICE)
REST_ORIENTATIONS_T = torch.tensor(REST_ORIENTATIONS, dtype=torch.float32, device=DEVICE)
LIMITS_MIN_T = torch.tensor(LIMITS_MIN, dtype=torch.float32, device=DEVICE)
LIMITS_MAX_T = torch.tensor(LIMITS_MAX, dtype=torch.float32, device=DEVICE)

# Pre-compute relative rest orientations (parent^T @ child) for FK
# This transforms from parent's frame to child's rest frame
RELATIVE_REST_ORIENTATIONS_T = torch.zeros(NUM_BONES, 3, 3, dtype=torch.float32, device=DEVICE)
for i in range(NUM_BONES):
    parent_idx = PARENT_INDICES[i]
    if parent_idx < 0:
        RELATIVE_REST_ORIENTATIONS_T[i] = REST_ORIENTATIONS_T[i]  # Root uses its own rest orientation
    else:
        # relative = parent_rest.T @ child_rest
        RELATIVE_REST_ORIENTATIONS_T[i] = REST_ORIENTATIONS_T[parent_idx].T @ REST_ORIENTATIONS_T[i]

# Effector bone indices
EFFECTOR_INDICES = torch.tensor(
    [BONE_TO_INDEX[name] for name in END_EFFECTORS],
    dtype=torch.long, device=DEVICE
)

# Foot indices for contact loss
FOOT_EFFECTOR_INDICES = torch.tensor([2, 3], dtype=torch.long, device=DEVICE)  # LeftFoot, RightFoot

# =============================================================================
# Data loading
# =============================================================================
def load_data():
    """Load training data from npz file."""
    path = os.path.join(DATA_DIR, "training_data.npz")
    if not os.path.exists(path):
        print(f"ERROR: No training data at {path}")
        print("Extract data in Blender first using the Neural IK panel.")
        sys.exit(1)

    data = np.load(path)

    print(f"Loading from: {path}")
    print(f"Keys found: {list(data.keys())}")

    # Check for required keys (expects pre-split train/test datasets)
    required = [
        'train_inputs', 'train_outputs', 'train_effector_targets', 'train_effector_rotations',
        'train_root_positions', 'train_root_forwards', 'train_root_ups',
        'test_inputs', 'test_outputs', 'test_effector_targets', 'test_effector_rotations',
        'test_root_positions', 'test_root_forwards', 'test_root_ups',
    ]
    missing = [k for k in required if k not in data]
    if missing:
        print(f"ERROR: Missing keys in training data: {missing}")
        print("Re-extract data with updated data.py")
        sys.exit(1)

    return data


def prepare_data(data):
    """Convert numpy data to torch tensors on device (uses pre-split train/test)."""
    def to_tensor(arr):
        return torch.tensor(arr, dtype=torch.float32, device=DEVICE)

    # Normalize inputs (keep scale for now)
    train_inputs = normalize_input(data['train_inputs'].astype(np.float32))
    test_inputs = normalize_input(data['test_inputs'].astype(np.float32))

    result = {
        'train_inputs': to_tensor(train_inputs),
        'train_outputs': to_tensor(data['train_outputs']),
        'train_targets': to_tensor(data['train_effector_targets']),
        'train_target_rots': to_tensor(data['train_effector_rotations']),
        'train_root_pos': to_tensor(data['train_root_positions']),
        'train_root_fwd': to_tensor(data['train_root_forwards']),
        'train_root_up': to_tensor(data['train_root_ups']),

        'test_inputs': to_tensor(test_inputs),
        'test_outputs': to_tensor(data['test_outputs']),
        'test_targets': to_tensor(data['test_effector_targets']),
        'test_target_rots': to_tensor(data['test_effector_rotations']),
        'test_root_pos': to_tensor(data['test_root_positions']),
        'test_root_fwd': to_tensor(data['test_root_forwards']),
        'test_root_up': to_tensor(data['test_root_ups']),
    }

    print(f"Data loaded: {len(result['train_inputs'])} train, {len(result['test_inputs'])} test samples")
    return result


# =============================================================================
# Neural Network Model
# =============================================================================
class NeuralIK(nn.Module):
    """MLP for IK: 50 -> 256 -> 256 -> 128 -> 69 (LeakyReLU hidden)"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, OUTPUT_SIZE)
        self.act = nn.LeakyReLU(0.1)

        # Xavier initialization
        for layer in (self.fc1, self.fc2, self.fc3, self.fc4):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.fc4(x)
        return x


# =============================================================================
# Rodrigues: Axis-Angle -> Rotation Matrix
# =============================================================================
def axis_angle_to_rotmat(axis_angles):
    """
    Convert axis-angle to rotation matrices using Rodrigues formula.

    Args:
        axis_angles: (batch, num_bones, 3) axis-angle rotations

    Returns:
        (batch, num_bones, 3, 3) rotation matrices
    """
    batch, num_bones, _ = axis_angles.shape

    # Angle is the norm of axis-angle vector
    theta = torch.norm(axis_angles, dim=-1, keepdim=True).clamp(min=1e-8)  # (batch, bones, 1)

    # Normalized axis
    axis = axis_angles / theta  # (batch, bones, 3)

    # Components
    cos_t = torch.cos(theta)  # (batch, bones, 1)
    sin_t = torch.sin(theta)  # (batch, bones, 1)
    one_minus_cos = 1.0 - cos_t

    # Axis components
    x = axis[..., 0:1]  # (batch, bones, 1)
    y = axis[..., 1:2]
    z = axis[..., 2:3]

    # Build rotation matrix using Rodrigues formula
    # R = cos(θ)I + sin(θ)[k]× + (1-cos(θ))k⊗k

    # Create skew-symmetric matrix [k]× and outer product k⊗k inline
    # Row 0
    r00 = cos_t + one_minus_cos * x * x
    r01 = one_minus_cos * x * y - sin_t * z
    r02 = one_minus_cos * x * z + sin_t * y

    # Row 1
    r10 = one_minus_cos * y * x + sin_t * z
    r11 = cos_t + one_minus_cos * y * y
    r12 = one_minus_cos * y * z - sin_t * x

    # Row 2
    r20 = one_minus_cos * z * x - sin_t * y
    r21 = one_minus_cos * z * y + sin_t * x
    r22 = cos_t + one_minus_cos * z * z

    # Stack into rotation matrices
    row0 = torch.cat([r00, r01, r02], dim=-1)  # (batch, bones, 3)
    row1 = torch.cat([r10, r11, r12], dim=-1)
    row2 = torch.cat([r20, r21, r22], dim=-1)

    rotmat = torch.stack([row0, row1, row2], dim=-2)  # (batch, bones, 3, 3)

    return rotmat


def rotmat_to_quaternion(rotmat):
    """
    Convert rotation matrix to quaternion (w, x, y, z).
    Uses Shepperd's method for numerical stability.

    Args:
        rotmat: (..., 3, 3) rotation matrices

    Returns:
        (..., 4) quaternions (w, x, y, z)
    """
    batch_shape = rotmat.shape[:-2]

    # Flatten for processing
    R = rotmat.reshape(-1, 3, 3)
    n = R.shape[0]

    # Trace and diagonal
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    # Initialize output
    quat = torch.zeros(n, 4, device=R.device, dtype=R.dtype)

    # Case 1: trace > 0
    mask1 = trace > 0
    if mask1.any():
        s = torch.sqrt(trace[mask1] + 1.0) * 2  # s = 4w
        quat[mask1, 0] = 0.25 * s
        quat[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s
        quat[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s
        quat[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s

    # Case 2: R[0,0] is largest diagonal
    mask2 = (~mask1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if mask2.any():
        s = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2
        quat[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s
        quat[mask2, 1] = 0.25 * s
        quat[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s
        quat[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s

    # Case 3: R[1,1] is largest diagonal
    mask3 = (~mask1) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    if mask3.any():
        s = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2
        quat[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s
        quat[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s
        quat[mask3, 2] = 0.25 * s
        quat[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s

    # Case 4: R[2,2] is largest diagonal
    mask4 = (~mask1) & (~mask2) & (~mask3)
    if mask4.any():
        s = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2
        quat[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s
        quat[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s
        quat[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s
        quat[mask4, 3] = 0.25 * s

    # Normalize
    quat = quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-8)

    return quat.reshape(*batch_shape, 4)


def euler_to_quaternion(euler):
    """
    Convert Euler XYZ to quaternion.

    Args:
        euler: (..., 3) Euler angles in radians

    Returns:
        (..., 4) quaternions (w, x, y, z)
    """
    half = euler * 0.5
    cx = torch.cos(half[..., 0])
    sx = torch.sin(half[..., 0])
    cy = torch.cos(half[..., 1])
    sy = torch.sin(half[..., 1])
    cz = torch.cos(half[..., 2])
    sz = torch.sin(half[..., 2])

    w = cx * cy * cz + sx * sy * sz
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz

    return torch.stack([w, x, y, z], dim=-1)


def quaternion_geodesic_loss(q1, q2):
    """
    Geodesic distance between quaternions on SO(3).

    Args:
        q1, q2: (..., 4) quaternions (w, x, y, z)

    Returns:
        (...,) geodesic angle in radians
    """
    # Dot product (absolute to handle q/-q equivalence)
    dot = torch.abs(torch.sum(q1 * q2, dim=-1)).clamp(max=1.0 - 1e-7)

    # Geodesic angle
    angle = 2.0 * torch.acos(dot)

    return angle


# =============================================================================
# Forward Kinematics (Batched) - CORRECTED to match Blender's bone transforms
# =============================================================================
def forward_kinematics(rotations, root_positions, root_rotations):
    """
    Compute FK to get world positions and rotations of all bones.

    IMPORTANT: Local rotations from Blender are in bone-local space.
    Each bone has a rest orientation that defines its local coordinate frame.
    The FK must account for these rest orientations to match Blender's transforms.

    Args:
        rotations: (batch, 23, 3, 3) local rotation matrices
        root_positions: (batch, 3) world position of root (Hips)
        root_rotations: (batch, 3, 3) world rotation of root

    Returns:
        positions: (batch, 23, 3) world positions
        world_rotations: (batch, 23, 3, 3) world rotation matrices
    """
    batch = rotations.shape[0]

    # Output arrays
    positions = torch.zeros(batch, NUM_BONES, 3, device=DEVICE)
    world_rotations = torch.zeros(batch, NUM_BONES, 3, 3, device=DEVICE)

    # Process bones in order (parent before child guaranteed by CONTROLLED_BONES order)
    for bone_idx in range(NUM_BONES):
        parent_idx = PARENT_INDICES_T[bone_idx].item()
        local_rot = rotations[:, bone_idx]  # (batch, 3, 3)

        if parent_idx == -1:
            # Root bone (Hips)
            # World rotation = input root rotation @ rest orientation @ local rotation
            # The rest orientation defines the bone's local frame
            # The local rotation rotates within that frame
            rest_orient = RELATIVE_REST_ORIENTATIONS_T[bone_idx].unsqueeze(0).expand(batch, -1, -1)
            world_rotations[:, bone_idx] = root_rotations @ rest_orient @ local_rot
            positions[:, bone_idx] = root_positions
        else:
            # Child bone
            parent_world_rot = world_rotations[:, parent_idx]  # (batch, 3, 3)
            parent_pos = positions[:, parent_idx]  # (batch, 3)

            # Position: parent position + parent's world rotation applied to local offset
            # LOCAL_OFFSETS_T[i] is the offset from parent to child in parent's local frame
            local_offset = LOCAL_OFFSETS_T[bone_idx]  # (3,)
            offset = (parent_world_rot @ local_offset.unsqueeze(-1)).squeeze(-1)
            positions[:, bone_idx] = parent_pos + offset

            # World rotation: parent world @ relative rest orientation @ local rotation
            # relative_rest = how this bone's rest frame relates to parent's rest frame
            relative_rest = RELATIVE_REST_ORIENTATIONS_T[bone_idx].unsqueeze(0).expand(batch, -1, -1)
            world_rotations[:, bone_idx] = parent_world_rot @ relative_rest @ local_rot

    return positions, world_rotations


def build_root_rotation(forwards, ups):
    """
    Build rotation matrix from forward and up vectors.

    Args:
        forwards: (batch, 3) forward direction
        ups: (batch, 3) up direction

    Returns:
        (batch, 3, 3) rotation matrices
    """
    # Normalize
    forwards = forwards / (torch.norm(forwards, dim=-1, keepdim=True) + 1e-8)
    ups = ups / (torch.norm(ups, dim=-1, keepdim=True) + 1e-8)

    # Right = forward × up
    rights = torch.cross(forwards, ups, dim=-1)
    rights = rights / (torch.norm(rights, dim=-1, keepdim=True) + 1e-8)

    # Rebuild up for orthogonality
    ups = torch.cross(rights, forwards, dim=-1)

    # [right, forward, up] as columns
    rotmat = torch.stack([rights, forwards, ups], dim=-1)  # (batch, 3, 3)

    return rotmat


# =============================================================================
# Loss Functions
# =============================================================================
def compute_position_loss(pred_rotations, targets, root_pos, root_rot):
    """
    FK position loss - MSE of effector positions.

    Args:
        pred_rotations: (batch, 69) axis-angle predictions
        targets: (batch, 15) effector target positions (5 effectors × 3)
        root_pos: (batch, 3) root world position
        root_rot: (batch, 3, 3) root world rotation

    Returns:
        loss: scalar MSE loss
        pred_positions: (batch, 5, 3) predicted effector positions
    """
    batch = pred_rotations.shape[0]

    # Reshape and convert to rotation matrices
    axis_angles = pred_rotations.view(batch, NUM_BONES, 3)
    rotmats = axis_angle_to_rotmat(axis_angles)  # (batch, 23, 3, 3)

    # Forward kinematics
    positions, _ = forward_kinematics(rotmats, root_pos, root_rot)

    # Extract effector positions
    pred_effector_pos = positions[:, EFFECTOR_INDICES]  # (batch, 5, 3)

    # Target positions
    target_pos = targets.view(batch, 5, 3)

    # MSE loss
    loss = torch.mean((pred_effector_pos - target_pos) ** 2)

    return loss, pred_effector_pos


def compute_orientation_loss(pred_rotations, target_rots, root_pos, root_rot):
    """
    Geodesic orientation loss for effectors.

    Args:
        pred_rotations: (batch, 69) axis-angle predictions
        target_rots: (batch, 15) target effector rotations (Euler XYZ)
        root_pos: (batch, 3)
        root_rot: (batch, 3, 3)

    Returns:
        loss: mean geodesic angle squared
    """
    batch = pred_rotations.shape[0]

    # Reshape and convert to rotation matrices
    axis_angles = pred_rotations.view(batch, NUM_BONES, 3)
    rotmats = axis_angle_to_rotmat(axis_angles)

    # Forward kinematics to get world rotations
    _, world_rotations = forward_kinematics(rotmats, root_pos, root_rot)

    # Extract effector rotations and convert to quaternions
    pred_effector_rot = world_rotations[:, EFFECTOR_INDICES]  # (batch, 5, 3, 3)
    pred_quats = rotmat_to_quaternion(pred_effector_rot)  # (batch, 5, 4)

    # Target rotations (Euler) to quaternions
    target_euler = target_rots.view(batch, 5, 3)
    target_quats = euler_to_quaternion(target_euler)  # (batch, 5, 4)

    # Geodesic distance
    angles = quaternion_geodesic_loss(pred_quats, target_quats)  # (batch, 5)

    # Mean squared angle
    loss = torch.mean(angles ** 2)

    return loss


def compute_pose_loss(pred_rotations, ground_truth):
    """
    Pose regression loss - MSE between predicted and ground-truth rotations.

    Args:
        pred_rotations: (batch, 69) predicted axis-angle
        ground_truth: (batch, 69) ground-truth axis-angle

    Returns:
        loss: scalar MSE
    """
    return torch.mean((pred_rotations - ground_truth) ** 2)


def compute_contact_loss(pred_rotations, root_pos, root_rot, targets):
    """
    Contact loss - penalize feet above ground when they should be grounded.

    Args:
        pred_rotations: (batch, 69)
        root_pos: (batch, 3)
        root_rot: (batch, 3, 3)
        targets: (batch, 15) effector targets (to infer contact flags)

    Returns:
        loss: scalar
    """
    batch = pred_rotations.shape[0]

    # Get predicted foot positions via FK
    axis_angles = pred_rotations.view(batch, NUM_BONES, 3)
    rotmats = axis_angle_to_rotmat(axis_angles)
    positions, _ = forward_kinematics(rotmats, root_pos, root_rot)

    # Foot positions (indices 2, 3 = LeftFoot, RightFoot)
    foot_positions = positions[:, EFFECTOR_INDICES[FOOT_EFFECTOR_INDICES]]  # (batch, 2, 3)
    foot_z = foot_positions[:, :, 2]  # (batch, 2)

    # Infer contact flags from target positions
    target_pos = targets.view(batch, 5, 3)
    target_foot_z = target_pos[:, FOOT_EFFECTOR_INDICES, 2]  # (batch, 2)
    contact_flags = (target_foot_z - GROUND_HEIGHT < CONTACT_THRESHOLD).float()

    # Penalize feet above ground when grounded
    penetration = torch.relu(foot_z - GROUND_HEIGHT - 0.02)  # Allow 2cm tolerance
    loss = torch.mean(contact_flags * penetration ** 2)

    return loss


def compute_limit_penalty(pred_rotations):
    """
    Soft penalty for joint limit violations.

    Args:
        pred_rotations: (batch, 69)

    Returns:
        penalty: scalar
    """
    batch = pred_rotations.shape[0]
    rotations = pred_rotations.view(batch, NUM_BONES, 3)

    # Violations below min
    below_min = torch.relu(LIMITS_MIN_T - rotations)

    # Violations above max
    above_max = torch.relu(rotations - LIMITS_MAX_T)

    # Sum squared violations
    penalty = torch.mean(below_min ** 2 + above_max ** 2)

    return penalty


# =============================================================================
# Training
# =============================================================================
def train(data, fresh_start=False):
    """Main training loop. Set fresh_start=True to ignore saved weights."""
    # Unpack data
    train_inputs = data['train_inputs']
    train_outputs = data['train_outputs']
    train_targets = data['train_targets']
    train_target_rots = data['train_target_rots']
    train_root_pos = data['train_root_pos']
    train_root_fwd = data['train_root_fwd']
    train_root_up = data['train_root_up']

    test_inputs = data['test_inputs']
    test_outputs = data['test_outputs']
    test_targets = data['test_targets']
    test_target_rots = data['test_target_rots']
    test_root_pos = data['test_root_pos']
    test_root_fwd = data['test_root_fwd']
    test_root_up = data['test_root_up']

    n_train = len(train_inputs)
    n_test = len(test_inputs)
    n_batches = max(1, n_train // BATCH_SIZE)

    # Model
    model = NeuralIK().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    # Mixed precision scaler (new API to silence deprecation)
    scaler = torch.amp.GradScaler("cuda") if USE_AMP and DEVICE.type == 'cuda' else None

    # Try to load existing weights (unless fresh_start requested)
    if not fresh_start and os.path.exists(BEST_WEIGHTS_PATH):
        try:
            weights = np.load(BEST_WEIGHTS_PATH, allow_pickle=True).item()
            model.fc1.weight.data = torch.tensor(weights['W1'].T, device=DEVICE)
            model.fc1.bias.data = torch.tensor(weights['b1'], device=DEVICE)
            model.fc2.weight.data = torch.tensor(weights['W2'].T, device=DEVICE)
            model.fc2.bias.data = torch.tensor(weights['b2'], device=DEVICE)
            model.fc3.weight.data = torch.tensor(weights['W3'].T, device=DEVICE)
            model.fc3.bias.data = torch.tensor(weights['b3'], device=DEVICE)
            model.fc4.weight.data = torch.tensor(weights['W4'].T, device=DEVICE)
            model.fc4.bias.data = torch.tensor(weights['b4'], device=DEVICE)
            print(f"Resumed from saved weights")
        except Exception as e:
            print(f"Could not load weights: {e}, starting fresh")
    elif fresh_start:
        print("Fresh start requested - using random initialization")

    # Training state
    best_pos_loss = float('inf')
    best_epoch = 0
    no_improve = 0

    print(f"\n{'='*70}")
    print(f" PYTORCH GPU TRAINING")
    print(f"{'='*70}")
    print(f" Device:     {DEVICE}")
    print(f" Batch size: {BATCH_SIZE}")
    print(f" Batches:    {n_batches} per epoch")
    print(f" Optimizer:  Adam (lr={LEARNING_RATE})")
    print(f" AMP:        {'Enabled' if scaler else 'Disabled'}")
    print(f" Weights:    Position={POSITION_WEIGHT}, Orient={ORIENTATION_WEIGHT}, Pose={POSE_WEIGHT}")
    print(f"             Contact={CONTACT_WEIGHT}, Limit={LIMIT_WEIGHT}")
    print(f"{'='*70}")

    # =========================================================================
    # FK VALIDATION - Verify FK matches Blender's actual bone positions
    # =========================================================================
    print(f"\n{'-'*70}")
    print(f" FK VALIDATION - Testing if our math matches Blender")
    print(f"{'-'*70}")

    # Use first 100 test samples for validation
    n_val = min(100, n_test)

    # Get ground truth rotations and run FK
    gt_rots_flat = test_outputs[:n_val]  # (n_val, 69) axis-angle
    gt_rots = gt_rots_flat.view(n_val, NUM_BONES, 3)  # (n_val, 23, 3)

    # Build rotation matrices from axis-angle
    gt_rotmats = axis_angle_to_rotmat(gt_rots)  # (n_val, 23, 3, 3)

    # Build root rotation from forward/up vectors
    root_rot_mats = build_root_rotation(test_root_fwd[:n_val], test_root_up[:n_val])

    # Run FK with ground truth rotations
    fk_positions, _ = forward_kinematics(gt_rotmats, test_root_pos[:n_val], root_rot_mats)

    # Extract effector positions
    fk_effector_pos = fk_positions[:, EFFECTOR_INDICES]  # (n_val, 5, 3)

    # Get target effector positions from training data
    target_effector_pos = test_targets[:n_val].view(n_val, 5, 3)  # (n_val, 5, 3)

    # Compute per-effector errors
    effector_errors = torch.norm(fk_effector_pos - target_effector_pos, dim=-1)  # (n_val, 5)
    mean_errors = effector_errors.mean(dim=0) * 100  # to cm
    overall_rmse = torch.sqrt((effector_errors ** 2).mean()).item() * 100

    print(f" Testing FK on {n_val} ground truth samples...")
    print(f" (Ground truth rotations should produce ground truth positions)")
    print()
    for i, name in enumerate(END_EFFECTORS):
        err_cm = mean_errors[i].item()
        status = "OK" if err_cm < 1.0 else ("WARN" if err_cm < 5.0 else "BAD")
        print(f"   {name:12s}: {err_cm:6.2f} cm  [{status}]")

    print(f"\n   Overall RMSE: {overall_rmse:.2f} cm")

    if overall_rmse < 1.0:
        print(f"\n   [PASS] FK matches Blender (error < 1cm)")
        print(f"   Trained rotations will look correct in Blender!")
    elif overall_rmse < 5.0:
        print(f"\n   [WARN] FK close but not exact (error {overall_rmse:.1f}cm)")
        print(f"   Results may have slight differences from Blender.")
    else:
        print(f"\n   [FAIL] FK does NOT match Blender (error {overall_rmse:.1f}cm)")
        print(f"   WARNING: Trained network will NOT work correctly!")
        print(f"   Check REST_ORIENTATIONS and LOCAL_OFFSETS in config.py")

    print(f"{'-'*70}\n")

    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.time()
        model.train()

        # Shuffle
        perm = torch.randperm(n_train, device=DEVICE)
        train_in = train_inputs[perm]
        train_out = train_outputs[perm]
        train_tgt = train_targets[perm]
        train_tgt_rot = train_target_rots[perm]
        train_rp = train_root_pos[perm]
        train_rf = train_root_fwd[perm]
        train_ru = train_root_up[perm]

        epoch_pos = 0.0
        epoch_orient = 0.0
        epoch_pose = 0.0
        epoch_contact = 0.0
        epoch_limit = 0.0

        for b in range(n_batches):
            i0 = b * BATCH_SIZE
            i1 = min(i0 + BATCH_SIZE, n_train)

            batch_in = train_in[i0:i1]
            batch_out = train_out[i0:i1]
            batch_tgt = train_tgt[i0:i1]
            batch_tgt_rot = train_tgt_rot[i0:i1]
            batch_rp = train_rp[i0:i1]
            batch_rf = train_rf[i0:i1]
            batch_ru = train_ru[i0:i1]

            # Light input noise for robustness
            if INPUT_NOISE_STD > 0:
                batch_in = batch_in + torch.randn_like(batch_in) * INPUT_NOISE_STD

            # Build root rotation
            root_rot = build_root_rotation(batch_rf, batch_ru)

            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast("cuda"):
                    # Forward
                    pred = model(batch_in)

                    # Losses
                    pos_loss, _ = compute_position_loss(pred, batch_tgt, batch_rp, root_rot)
                    orient_loss = compute_orientation_loss(pred, batch_tgt_rot, batch_rp, root_rot)
                    pose_loss = compute_pose_loss(pred, batch_out)
                    contact_loss = compute_contact_loss(pred, batch_rp, root_rot, batch_tgt)
                    limit_loss = compute_limit_penalty(pred)

                    # Smoothness penalty: penalize differences between adjacent batch samples
                    smooth_loss = torch.mean((pred - pred.detach().roll(1, 0)) ** 2)

                    total_loss = (pos_loss * POSITION_WEIGHT +
                                 orient_loss * ORIENTATION_WEIGHT +
                                 pose_loss * POSE_WEIGHT +
                                 contact_loss * CONTACT_WEIGHT +
                                 limit_loss * LIMIT_WEIGHT +
                                 smooth_loss * SMOOTH_WEIGHT)

                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Forward
                pred = model(batch_in)

                # Losses
                pos_loss, _ = compute_position_loss(pred, batch_tgt, batch_rp, root_rot)
                orient_loss = compute_orientation_loss(pred, batch_tgt_rot, batch_rp, root_rot)
                pose_loss = compute_pose_loss(pred, batch_out)
                contact_loss = compute_contact_loss(pred, batch_rp, root_rot, batch_tgt)
                limit_loss = compute_limit_penalty(pred)

                # Smoothness penalty: penalize differences between adjacent batch samples
                smooth_loss = torch.mean((pred - pred.detach().roll(1, 0)) ** 2)

                total_loss = (pos_loss * POSITION_WEIGHT +
                             orient_loss * ORIENTATION_WEIGHT +
                             pose_loss * POSE_WEIGHT +
                             contact_loss * CONTACT_WEIGHT +
                             limit_loss * LIMIT_WEIGHT +
                             smooth_loss * SMOOTH_WEIGHT)

                total_loss.backward()
                optimizer.step()

            epoch_pos += pos_loss.item()
            epoch_orient += orient_loss.item()
            epoch_pose += pose_loss.item()
            epoch_contact += contact_loss.item()
            epoch_limit += limit_loss.item()

        epoch_pos /= n_batches
        epoch_orient /= n_batches
        epoch_pose /= n_batches
        epoch_contact /= n_batches
        epoch_limit /= n_batches

        # Test evaluation
        model.eval()
        with torch.no_grad():
            test_root_rot = build_root_rotation(test_root_fwd, test_root_up)
            test_pred = model(test_inputs)

            test_pos_loss, _ = compute_position_loss(test_pred, test_targets, test_root_pos, test_root_rot)
            test_orient_loss = compute_orientation_loss(test_pred, test_target_rots, test_root_pos, test_root_rot)

            test_pos = test_pos_loss.item()
            test_orient = test_orient_loss.item()

        # Track best (position only for early stopping)
        is_best = test_pos < best_pos_loss
        if is_best:
            best_pos_loss = test_pos
            best_epoch = epoch
            no_improve = 0
            save_weights(model, best_pos_loss, epoch)
        else:
            no_improve += 1

        # Early stopping
        if no_improve >= EARLY_STOP_PATIENCE:
            print(f"\n  Early stopping after {EARLY_STOP_PATIENCE} epochs without improvement")
            break

        # Scheduler step per epoch
        scheduler.step()

        # Progress
        t = time.time() - epoch_start
        if epoch % 5 == 0 or epoch == 1 or is_best:
            star = " *" if is_best else ""
            print(f"[{100*epoch/MAX_EPOCHS:5.1f}%] Epoch {epoch:3d} | "
                  f"Pos={epoch_pos:.4f} Ori={epoch_orient:.4f} Pose={epoch_pose:.4f} "
                  f"Contact={epoch_contact:.4f} Limit={epoch_limit:.4f} | "
                  f"Test P:{test_pos:.4f} O:{test_orient:.4f} | "
                  f"LR={scheduler.get_last_lr()[0]:.6f} | "
                  f"{t:.2f}s{star}")

    total_time = time.time() - start_time

    print(f"\n{'='*70}")
    print(f" TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f" Best Position Loss: {best_pos_loss:.6f} m² (epoch {best_epoch})")
    print(f" Best Position RMSE: {math.sqrt(best_pos_loss):.4f} m ({math.sqrt(best_pos_loss)*100:.1f} cm)")
    print(f" Total Time:         {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f" Weights saved:      {BEST_WEIGHTS_PATH}")
    print(f"{'='*70}\n")

    return model


def save_weights(model, best_loss=0.0, total_updates=0):
    """Save weights in numpy format (compatible with existing Blender loader)."""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    weights = {
        'W1': model.fc1.weight.data.cpu().numpy().T,  # Transpose for numpy convention
        'b1': model.fc1.bias.data.cpu().numpy(),
        'W2': model.fc2.weight.data.cpu().numpy().T,
        'b2': model.fc2.bias.data.cpu().numpy(),
        'W3': model.fc3.weight.data.cpu().numpy().T,
        'b3': model.fc3.bias.data.cpu().numpy(),
        'W4': model.fc4.weight.data.cpu().numpy().T,
        'b4': model.fc4.bias.data.cpu().numpy(),
        'best_loss': float(best_loss),
        'total_updates': int(total_updates),
    }

    np.save(BEST_WEIGHTS_PATH, weights)


# =============================================================================
# Test Suite
# =============================================================================
def run_tests(model, data):
    """Run all 5 tests on trained model."""
    print(f"\n{'='*70}")
    print(f" TEST SUITE")
    print(f"{'='*70}\n")

    model.eval()
    results = {}

    with torch.no_grad():
        # Test data
        test_inputs = data['test_inputs']
        test_outputs = data['test_outputs']
        test_targets = data['test_targets']
        test_target_rots = data['test_target_rots']
        test_root_pos = data['test_root_pos']
        test_root_fwd = data['test_root_fwd']
        test_root_up = data['test_root_up']
        test_root_rot = build_root_rotation(test_root_fwd, test_root_up)

        # 1. Holdout Position (FK)
        print("1. Holdout Position Test (FK)")
        pred = model(test_inputs)
        pos_loss, pred_pos = compute_position_loss(pred, test_targets, test_root_pos, test_root_rot)
        pos_rmse = math.sqrt(pos_loss.item())
        target_pos = test_targets.view(-1, 5, 3)
        errors = torch.norm(pred_pos - target_pos, dim=-1).mean(dim=0)  # Per effector

        print(f"   Position RMSE: {pos_rmse*100:.2f} cm")
        for i, name in enumerate(END_EFFECTORS):
            print(f"   - {name}: {errors[i].item()*100:.2f} cm")

        passed_pos = pos_rmse < 0.15  # 15cm threshold
        results['holdout_position'] = passed_pos
        print(f"   {'PASS' if passed_pos else 'FAIL'} (threshold: 15cm)\n")

        # 2. Holdout Orientation
        print("2. Holdout Orientation Test (Geodesic)")
        orient_loss = compute_orientation_loss(pred, test_target_rots, test_root_pos, test_root_rot)
        orient_rmse_deg = math.sqrt(orient_loss.item()) * 180 / math.pi

        print(f"   Orientation RMSE: {orient_rmse_deg:.1f} degrees")
        passed_orient = orient_rmse_deg < 45  # 45 degree threshold
        results['holdout_orientation'] = passed_orient
        print(f"   {'PASS' if passed_orient else 'FAIL'} (threshold: 45 deg)\n")

        # 3. Interpolation Test
        print("3. Interpolation Test")
        n = min(100, len(test_inputs))
        idx = torch.randperm(len(test_inputs))[:n*2].view(n, 2)

        interp_errors = []
        for i in range(n):
            i1, i2 = idx[i]
            for alpha in [0.25, 0.5, 0.75]:
                # Interpolate inputs
                inp = (1 - alpha) * test_inputs[i1] + alpha * test_inputs[i2]
                tgt = (1 - alpha) * test_targets[i1] + alpha * test_targets[i2]
                rp = (1 - alpha) * test_root_pos[i1] + alpha * test_root_pos[i2]
                rf = (1 - alpha) * test_root_fwd[i1] + alpha * test_root_fwd[i2]
                ru = (1 - alpha) * test_root_up[i1] + alpha * test_root_up[i2]

                rr = build_root_rotation(rf.unsqueeze(0), ru.unsqueeze(0))

                pred = model(inp.unsqueeze(0))
                loss, _ = compute_position_loss(pred, tgt.unsqueeze(0), rp.unsqueeze(0), rr)
                interp_errors.append(math.sqrt(loss.item()))

        interp_rmse = sum(interp_errors) / len(interp_errors)
        print(f"   Interpolation RMSE: {interp_rmse*100:.2f} cm")
        passed_interp = interp_rmse < 0.20  # 20cm threshold (harder)
        results['interpolation'] = passed_interp
        print(f"   {'PASS' if passed_interp else 'FAIL'} (threshold: 20cm)\n")

        # 4. Consistency Test
        print("4. Consistency Test (same input twice)")
        sample_input = test_inputs[0:1]
        pred1 = model(sample_input)
        pred2 = model(sample_input)
        diff = torch.max(torch.abs(pred1 - pred2)).item()

        print(f"   Max difference: {diff:.2e}")
        passed_consist = diff < 1e-5
        results['consistency'] = passed_consist
        print(f"   {'PASS' if passed_consist else 'FAIL'} (threshold: 1e-5)\n")

        # 5. Noise Robustness Test
        print("5. Noise Robustness Test")
        n = min(200, len(test_inputs))
        base_inputs = test_inputs[:n]
        base_targets = test_targets[:n]
        base_root_pos = test_root_pos[:n]
        base_root_rot = build_root_rotation(test_root_fwd[:n], test_root_up[:n])

        # Clean predictions
        clean_pred = model(base_inputs)
        clean_loss, _ = compute_position_loss(clean_pred, base_targets, base_root_pos, base_root_rot)
        clean_rmse = math.sqrt(clean_loss.item())

        # Noisy predictions
        noise = torch.randn_like(base_inputs) * 0.02  # 2% noise
        noisy_inputs = base_inputs + noise
        noisy_pred = model(noisy_inputs)
        noisy_loss, _ = compute_position_loss(noisy_pred, base_targets, base_root_pos, base_root_rot)
        noisy_rmse = math.sqrt(noisy_loss.item())

        degradation = (noisy_rmse - clean_rmse) / (clean_rmse + 1e-8)
        print(f"   Clean RMSE:  {clean_rmse*100:.2f} cm")
        print(f"   Noisy RMSE:  {noisy_rmse*100:.2f} cm")
        print(f"   Degradation: {degradation*100:.1f}%")

        passed_noise = degradation < 0.5  # Less than 50% degradation
        results['noise_robustness'] = passed_noise
        print(f"   {'PASS' if passed_noise else 'FAIL'} (threshold: 50% degradation)\n")

    # Summary
    n_passed = sum(results.values())
    n_total = len(results)

    print(f"{'='*70}")
    print(f" RESULTS: {n_passed}/{n_total} tests passed")
    print(f"{'='*70}")

    for name, passed in results.items():
        print(f"   {'PASS' if passed else 'FAIL'} - {name}")

    print(f"{'='*70}\n")

    return n_passed == n_total


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    import sys

    # Help text
    if '--help' in sys.argv or '-h' in sys.argv:
        print("""
Neural IK - PyTorch GPU Trainer
===============================

Usage: python torch_trainer.py [OPTIONS]

Options:
    --fresh     Start fresh training from random weights.
                Use this after fixing bugs or changing the FK computation.
                Without this flag, training resumes from saved weights.

    --help      Show this help message.

What the trainer does:
    1. Loads training data extracted from Blender animations
    2. Validates FK computation matches Blender's bone transforms
    3. Trains neural network to predict bone rotations from target positions
    4. Saves best weights to weights/best.npy
    5. Runs test suite to verify accuracy

Log column meanings:
    Pos     = Position loss (how far effectors are from targets, in meters^2)
    Ori     = Orientation loss (rotation error of effectors)
    Pose    = Pose matching loss (how close to training poses)
    Contact = Foot grounding loss (feet on ground when expected)
    Limit   = Joint limit penalty (anatomical constraints)
    Test P  = Test set position loss (generalization check)
    Test O  = Test set orientation loss
    *       = New best model saved
""")
        sys.exit(0)

    # Check for --fresh flag to start fresh training (ignore saved weights)
    FRESH_START = '--fresh' in sys.argv

    print(f"\n{'='*70}")
    print(f" NEURAL IK - PyTorch GPU Trainer")
    print(f"{'='*70}")
    print(f" Run with --help for usage information")
    print(f"{'='*70}\n")

    if FRESH_START:
        print("*** FRESH START MODE ***")
        print("    Ignoring saved weights, training from scratch.")
        print("    Use this after fixing FK bugs or changing architecture.\n")

    # Load and prepare data
    raw_data = load_data()
    data = prepare_data(raw_data)

    # Train
    model = train(data, fresh_start=FRESH_START)

    # Run tests
    all_passed = run_tests(model, data)

    print(f"\n{'='*70}")
    print(f" TRAINING COMPLETE")
    print(f"{'='*70}")

    if all_passed:
        print(" All tests PASSED!")
        print()
        print(" Next steps:")
        print("   1. Reinstall addon in Blender (copy Desktop -> AppData)")
        print("   2. Open Blender and go to the Neural IK panel")
        print("   3. Click 'Reload Weights' to load the trained model")
        print("   4. Use 'Toggle Pred/Truth' to compare predictions vs ground truth")
        print("   5. Predictions should now match ground truth visually!")
    else:
        print(" Some tests FAILED - review output above.")
        print()
        print(" Possible issues:")
        print("   - Not enough training data (extract more animations)")
        print("   - FK validation failed (check config.py bone data)")
        print("   - Need more training epochs (increase MAX_EPOCHS)")

    print(f"{'='*70}\n")
