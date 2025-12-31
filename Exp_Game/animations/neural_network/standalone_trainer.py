"""
Standalone Neural IK Trainer
============================
Run OUTSIDE Blender for maximum speed.

Usage:
    cd <addon_path>/Exp_Game/animations/neural_network
    python standalone_trainer.py

Prerequisites:
    - Python 3.10+ with NumPy
    - training_data.npz (extracted in Blender first)

Output:
    - weights/best.npy (load automatically in Blender)
"""

import os
import sys
import time

# Set thread count BEFORE importing NumPy (uses all cores)
NUM_CORES = os.cpu_count() or 4
os.environ['OMP_NUM_THREADS'] = str(NUM_CORES)
os.environ['MKL_NUM_THREADS'] = str(NUM_CORES)
os.environ['OPENBLAS_NUM_THREADS'] = str(NUM_CORES)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(NUM_CORES)
os.environ['NUMEXPR_NUM_THREADS'] = str(NUM_CORES)

import numpy as np

# CPU monitoring (optional - graceful fallback if psutil not installed)
try:
    import psutil
    HAS_PSUTIL = True
    PROCESS = psutil.Process()
except ImportError:
    HAS_PSUTIL = False
    PROCESS = None

def get_cpu_usage():
    """Get current CPU usage percentage."""
    if HAS_PSUTIL:
        try:
            # Get this process's CPU usage (across all cores)
            return PROCESS.cpu_percent(interval=None)
        except:
            return -1
    return -1

def init_cpu_monitor():
    """Initialize CPU monitoring (call once before training)."""
    if HAS_PSUTIL:
        try:
            PROCESS.cpu_percent(interval=None)  # First call returns 0, primes the counter
        except:
            pass

# Import from same directory directly (bypass Exp_Game/__init__.py which needs bpy)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Direct imports - no package traversal
from config import (
    DATA_DIR, WEIGHTS_DIR, BEST_WEIGHTS_PATH,
    FK_LOSS_WEIGHT, POSE_LOSS_WEIGHT, LIMIT_PENALTY_WEIGHT, CONTACT_LOSS_WEIGHT,
    LIMITS_MIN, LIMITS_MAX, NUM_BONES, OUTPUT_SIZE,
    INPUT_SIZE, HIDDEN_SIZE_1, HIDDEN_SIZE_2,
)
from forward_kinematics import (
    compute_fk_loss, compute_fk_loss_gradient, clamp_rotations,
    compute_contact_loss_simple, compute_contact_loss_gradient, infer_contact_flags,
)
from context import normalize_input

# =============================================================================
# TRAINING HYPERPARAMETERS (optimized for speed with Adam)
# =============================================================================
BATCH_SIZE = 128          # Larger batches = better core utilization
MAX_EPOCHS = 300          # Will early-stop much sooner
FK_GRAD_INTERVAL = 16     # Compute FK grad every N batches (balance cost vs. signal)
CONTACT_GRAD_INTERVAL = 8 # Compute contact grad every N batches
FK_GRAD_DISABLE_AFTER = 0.02  # Disable FK grad once loss below this
EARLY_STOP_PATIENCE = 40  # Stop after N epochs without improvement
GROUND_HEIGHT = 0.0       # Z coordinate of ground plane
CONTACT_THRESHOLD = 0.1   # Foot Z below this = grounded

# Adam optimizer parameters
ADAM_LR = 0.001
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8


# =============================================================================
# NETWORK WITH ADAM OPTIMIZER
# =============================================================================
class StandaloneNetwork:
    """Network with Adam optimizer for fast convergence."""

    def __init__(self):
        # Xavier initialization
        self.W1 = np.random.randn(INPUT_SIZE, HIDDEN_SIZE_1).astype(np.float32) * np.sqrt(2.0 / INPUT_SIZE)
        self.b1 = np.zeros(HIDDEN_SIZE_1, dtype=np.float32)
        self.W2 = np.random.randn(HIDDEN_SIZE_1, HIDDEN_SIZE_2).astype(np.float32) * np.sqrt(2.0 / HIDDEN_SIZE_1)
        self.b2 = np.zeros(HIDDEN_SIZE_2, dtype=np.float32)
        self.W3 = np.random.randn(HIDDEN_SIZE_2, OUTPUT_SIZE).astype(np.float32) * np.sqrt(2.0 / HIDDEN_SIZE_2)
        self.b3 = np.zeros(OUTPUT_SIZE, dtype=np.float32)

        # Cache for backward pass
        self._input = None
        self._h1 = None
        self._h1_act = None
        self._h2 = None
        self._h2_act = None

        # Adam momentum (first moment)
        self.m_W1 = np.zeros_like(self.W1)
        self.m_b1 = np.zeros_like(self.b1)
        self.m_W2 = np.zeros_like(self.W2)
        self.m_b2 = np.zeros_like(self.b2)
        self.m_W3 = np.zeros_like(self.W3)
        self.m_b3 = np.zeros_like(self.b3)

        # Adam velocity (second moment)
        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_b2 = np.zeros_like(self.b2)
        self.v_W3 = np.zeros_like(self.W3)
        self.v_b3 = np.zeros_like(self.b3)

        # Adam timestep
        self.t = 0

        self.best_loss = float('inf')

    def forward(self, x):
        """Forward pass with tanh activations."""
        self._input = x
        self._h1 = x @ self.W1 + self.b1
        self._h1_act = np.tanh(self._h1)
        self._h2 = self._h1_act @ self.W2 + self.b2
        self._h2_act = np.tanh(self._h2)
        out = self._h2_act @ self.W3 + self.b3
        return out

    def backward(self, grad_output, learning_rate=None):
        """Backward pass with Adam optimizer."""
        batch_size = grad_output.shape[0]
        self.t += 1

        # Compute gradients
        # Output layer
        grad_W3 = self._h2_act.T @ grad_output / batch_size
        grad_b3 = np.mean(grad_output, axis=0)
        grad_h2_act = grad_output @ self.W3.T

        # Hidden 2
        grad_h2 = grad_h2_act * (1 - self._h2_act ** 2)
        grad_W2 = self._h1_act.T @ grad_h2 / batch_size
        grad_b2 = np.mean(grad_h2, axis=0)
        grad_h1_act = grad_h2 @ self.W2.T

        # Hidden 1
        grad_h1 = grad_h1_act * (1 - self._h1_act ** 2)
        grad_W1 = self._input.T @ grad_h1 / batch_size
        grad_b1 = np.mean(grad_h1, axis=0)

        # Adam update for each parameter
        self._adam_update('W1', grad_W1)
        self._adam_update('b1', grad_b1)
        self._adam_update('W2', grad_W2)
        self._adam_update('b2', grad_b2)
        self._adam_update('W3', grad_W3)
        self._adam_update('b3', grad_b3)

    def _adam_update(self, name, grad):
        """Apply Adam update to a parameter."""
        # Get references
        param = getattr(self, name)
        m = getattr(self, f'm_{name}')
        v = getattr(self, f'v_{name}')

        # Update biased moments
        m[:] = ADAM_BETA1 * m + (1 - ADAM_BETA1) * grad
        v[:] = ADAM_BETA2 * v + (1 - ADAM_BETA2) * (grad ** 2)

        # Bias correction
        m_hat = m / (1 - ADAM_BETA1 ** self.t)
        v_hat = v / (1 - ADAM_BETA2 ** self.t)

        # Update parameter
        param -= ADAM_LR * m_hat / (np.sqrt(v_hat) + ADAM_EPSILON)

    def compute_limit_penalty(self, rotations):
        """Soft penalty for rotations outside joint limits."""
        reshaped = rotations.reshape(-1, NUM_BONES, 3)
        below = np.minimum(reshaped - LIMITS_MIN, 0)
        above = np.maximum(reshaped - LIMITS_MAX, 0)
        violations = below ** 2 + above ** 2
        penalty = float(np.mean(violations))
        grad = 2 * (below + above).reshape(rotations.shape) / rotations.size
        return penalty, grad

    def save(self, path=BEST_WEIGHTS_PATH):
        """Save weights to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, {
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2,
            'W3': self.W3, 'b3': self.b3,
            'best_loss': self.best_loss,
        })

    def load(self, path=BEST_WEIGHTS_PATH):
        """Load weights from disk."""
        if not os.path.exists(path):
            return False
        data = np.load(path, allow_pickle=True).item()
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        self.W3 = data['W3']
        self.b3 = data['b3']
        self.best_loss = data.get('best_loss', float('inf'))
        return True


# =============================================================================
# TRAINING
# =============================================================================
def load_data():
    """Load training data from disk."""
    path = os.path.join(DATA_DIR, "training_data.npz")
    if not os.path.exists(path):
        print(f"\n❌ ERROR: No training data found!")
        print(f"   Expected: {path}")
        print(f"   Run 'Extract from Actions' in Blender first!\n")
        sys.exit(1)

    # Load and validate
    data = dict(np.load(path))

    # Check for required keys
    required = ['train_inputs', 'train_outputs', 'train_effector_targets', 'train_root_positions',
                'test_inputs', 'test_outputs', 'test_effector_targets', 'test_root_positions']
    missing = [k for k in required if k not in data]
    if missing:
        print(f"\n❌ ERROR: Training data missing keys: {missing}")
        print(f"   Re-extract data in Blender with latest version.\n")
        sys.exit(1)

    n_train = len(data['train_inputs'])
    n_test = len(data['test_inputs'])

    # File info
    file_size = os.path.getsize(path) / (1024 * 1024)  # MB
    file_time = os.path.getmtime(path)
    from datetime import datetime
    file_date = datetime.fromtimestamp(file_time).strftime('%Y-%m-%d %H:%M')

    print(f"✓ Data loaded: {path}")
    print(f"  File:     {file_size:.1f} MB, created {file_date}")
    print(f"  Samples:  {n_train:,} train + {n_test:,} test ({100*n_test/(n_train+n_test):.0f}% holdout)")
    print(f"  Input:    {data['train_inputs'].shape[1]} dims, Output: {data['train_outputs'].shape[1]} dims")

    return data


def train(data, resume=True):
    """Main training loop."""
    # Prepare data
    train_inputs = normalize_input(data['train_inputs'])
    train_outputs = data['train_outputs']
    train_targets = data['train_effector_targets']
    train_roots = data['train_root_positions']

    test_inputs = normalize_input(data['test_inputs'])
    test_outputs = data['test_outputs']
    test_targets = data['test_effector_targets']
    test_roots = data['test_root_positions']

    n_train = len(train_inputs)
    n_test = len(test_inputs)
    n_batches = max(1, n_train // BATCH_SIZE)

    # Network
    network = StandaloneNetwork()
    if resume and network.load():
        print(f"✓ Resumed from saved weights (best loss: {network.best_loss:.6f})")
    else:
        print("✓ Starting with fresh random weights")

    # Training state
    best_fk = float('inf')
    best_epoch = 0
    no_improve = 0
    fk_grad_enabled = True

    print(f"\n{'═'*70}")
    print(f" TRAINING CONFIG")
    print(f"{'═'*70}")
    print(f" Batches:    {n_batches} per epoch (batch size {BATCH_SIZE})")
    print(f" Optimizer:  Adam (lr={ADAM_LR})")
    print(f" FK grad:    every {FK_GRAD_INTERVAL} batches → disabled when loss < {FK_GRAD_DISABLE_AFTER}")
    print(f" Contact:    every {CONTACT_GRAD_INTERVAL} batches (ground Z={GROUND_HEIGHT}, thresh={CONTACT_THRESHOLD}m)")
    print(f" Early stop: {EARLY_STOP_PATIENCE} epochs without improvement")
    print(f"{'═'*70}")
    print(f" NOTE: Contact flags inferred from foot Z position in training data.")
    print(f"       If your ground isn't at Z=0, adjust GROUND_HEIGHT in script.")
    print(f"{'═'*70}")
    if HAS_PSUTIL:
        print(f" CPU monitoring enabled (psutil installed)")
    else:
        print(f" CPU monitoring disabled (pip install psutil for CPU %)")
    print(f"{'═'*70}\n")

    init_cpu_monitor()
    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.time()

        # Shuffle
        perm = np.random.permutation(n_train)
        train_in = train_inputs[perm]
        train_out = train_outputs[perm]
        train_tgt = train_targets[perm]
        train_rt = train_roots[perm]

        epoch_fk = 0.0
        epoch_pose = 0.0
        epoch_contact = 0.0
        cached_fk_grad = None
        cached_contact_grad = None
        epoch_cpu_samples = []

        for b in range(n_batches):
            i0 = b * BATCH_SIZE
            i1 = min(i0 + BATCH_SIZE, n_train)

            batch_in = train_in[i0:i1]
            batch_out = train_out[i0:i1]
            batch_tgt = train_tgt[i0:i1]
            batch_rt = train_rt[i0:i1]

            # Forward
            pred = network.forward(batch_in)

            # Clamp for FK
            pred_r = pred.reshape(-1, NUM_BONES, 3)
            pred_c, _ = clamp_rotations(pred_r, LIMITS_MIN, LIMITS_MAX)
            pred_cf = pred_c.reshape(-1, OUTPUT_SIZE)

            # Losses
            fk_loss, _ = compute_fk_loss(pred_cf, batch_tgt.reshape(-1, 5, 3), batch_rt)
            pose_err = pred - batch_out
            pose_loss = float(np.mean(pose_err ** 2))
            limit_pen, limit_grad = network.compute_limit_penalty(pred)

            # Contact loss (infer grounded flags from target positions)
            contact_flags = infer_contact_flags(batch_tgt, GROUND_HEIGHT, CONTACT_THRESHOLD)
            contact_loss, _ = compute_contact_loss_simple(pred_cf, contact_flags, batch_rt, GROUND_HEIGHT)

            # Gradients
            pose_grad = 2 * pose_err / pose_err.size * POSE_LOSS_WEIGHT

            # FK gradient (expensive - compute sparingly)
            if fk_grad_enabled and b % FK_GRAD_INTERVAL == 0:
                fk_grad = compute_fk_loss_gradient(pred_cf, batch_tgt.reshape(-1, 5, 3), batch_rt)
                cached_fk_grad = fk_grad
            elif cached_fk_grad is not None:
                fk_grad = cached_fk_grad
            else:
                fk_grad = np.zeros_like(pose_grad)

            # Contact gradient (less expensive than FK - compute more often)
            if b % CONTACT_GRAD_INTERVAL == 0:
                contact_grad = compute_contact_loss_gradient(pred_cf, contact_flags, batch_rt, GROUND_HEIGHT)
                cached_contact_grad = contact_grad
            elif cached_contact_grad is not None:
                contact_grad = cached_contact_grad
            else:
                contact_grad = np.zeros_like(pose_grad)

            total_grad = (fk_grad * FK_LOSS_WEIGHT +
                         pose_grad +
                         contact_grad * CONTACT_LOSS_WEIGHT +
                         limit_grad * LIMIT_PENALTY_WEIGHT)
            network.backward(total_grad)

            epoch_fk += fk_loss
            epoch_pose += pose_loss
            epoch_contact += contact_loss

            # Sample CPU every few batches
            if b % 8 == 0:
                cpu_sample = get_cpu_usage()
                if cpu_sample >= 0:
                    epoch_cpu_samples.append(cpu_sample)

        epoch_fk /= n_batches
        epoch_pose /= n_batches
        epoch_contact /= n_batches

        # Test eval
        test_pred = network.forward(test_inputs)
        test_pred_r = test_pred.reshape(-1, NUM_BONES, 3)
        test_pred_c, _ = clamp_rotations(test_pred_r, LIMITS_MIN, LIMITS_MAX)
        test_pred_cf = test_pred_c.reshape(-1, OUTPUT_SIZE)
        test_fk, _ = compute_fk_loss(test_pred_cf, test_targets.reshape(-1, 5, 3), test_roots)

        # Test contact loss
        test_contact_flags = infer_contact_flags(test_targets, GROUND_HEIGHT, CONTACT_THRESHOLD)
        test_contact, _ = compute_contact_loss_simple(test_pred_cf, test_contact_flags, test_roots, GROUND_HEIGHT)

        # Track best
        is_best = test_fk < best_fk
        if is_best:
            best_fk = test_fk
            best_epoch = epoch
            network.best_loss = best_fk
            network.save()
            no_improve = 0
        else:
            no_improve += 1

        # Disable FK grad once converged (let pose loss refine)
        if fk_grad_enabled and test_fk < FK_GRAD_DISABLE_AFTER:
            fk_grad_enabled = False
            print(f"  → FK grad disabled (loss {test_fk:.4f} < {FK_GRAD_DISABLE_AFTER})")

        # Early stop
        if no_improve >= EARLY_STOP_PATIENCE:
            print(f"\n  ⚠ Early stopping after {EARLY_STOP_PATIENCE} epochs without improvement")
            break

        # Progress
        t = time.time() - epoch_start
        avg_cpu = np.mean(epoch_cpu_samples) if epoch_cpu_samples else -1
        if epoch % 5 == 0 or epoch == 1 or is_best:
            pct = 100 * epoch / MAX_EPOCHS
            star = " ★" if is_best else ""
            fk_status = "FK" if fk_grad_enabled else "pose-only"
            cpu_str = f" CPU:{avg_cpu:3.0f}%" if avg_cpu >= 0 else ""
            print(f"[{pct:5.1f}%] Epoch {epoch:3d} | FK={epoch_fk:.4f} Ct={epoch_contact:.4f} Test={test_fk:.4f} | {t:.1f}s ({fk_status}){cpu_str}{star}")

    total = time.time() - start_time
    print(f"\n{'═'*70}")
    print(f" TRAINING COMPLETE")
    print(f"{'═'*70}")
    print(f" Best FK Loss:      {best_fk:.6f} (epoch {best_epoch})")
    print(f" Final Contact:     {epoch_contact:.6f}")
    print(f" Total Time:        {total:.1f}s ({total/60:.1f} min)")
    print(f" Weights saved:     {BEST_WEIGHTS_PATH}")
    print(f"{'═'*70}")
    print(f" NEXT STEPS:")
    print(f"   1. In Blender: 'Reload Trained Weights'")
    print(f"   2. Run 'Test Suite' to verify (holdout, interpolation, noise)")
    print(f"   3. All 4 tests must pass before production use")
    print(f"{'═'*70}\n")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("\n" + "═"*70)
    print(f" NEURAL IK STANDALONE TRAINER")
    print(f"{'═'*70}")
    print(f" Runtime:    Python {sys.version.split()[0]} | NumPy {np.__version__}")
    print(f" Threads:    {NUM_CORES} CPU cores (auto-detected)")
    print(f" Optimizer:  Adam | Batch size {BATCH_SIZE}")
    print("═"*70)

    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    data = load_data()
    train(data, resume=False)  # Start fresh
