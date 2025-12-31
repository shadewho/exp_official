# Neural Network IK System

**Status**: TRAINED AND WORKING - All 4 tests passing
**Location**: `Exp_Game/animations/neural_network/`

---

## CURRENT RESULTS (2025-12-30)

```
══════════════════════════════════════════════════════════════
 NEURAL IK TEST SUITE
══════════════════════════════════════════════════════════════

 [✓ PASS] Holdout Test (FK)
         Score: 0.1095 (threshold: 0.2500)
         FK loss on unseen data: 0.1095m avg error

 [✓ PASS] Interpolation Test (FK)
         Score: 0.1094 (threshold: 0.3000)
         Avg interpolation FK error: 0.1094m

 [✓ PASS] Consistency Test
         Score: 0.0000 (threshold: 0.0000)
         Max difference between runs: 0.0000000000

 [✓ PASS] Noise Robustness Test
         Score: 0.0213 (threshold: 0.5000)
         Output/input change ratio: 0.02x

──────────────────────────────────────────────────────────────
 TOTAL: 4/4 tests passed
 STATUS: ✓ ALL TESTS PASSED - Network is learning
══════════════════════════════════════════════════════════════
```

**Training Stats:**
- Samples: 490 train + 62 test (307 raw samples from 7 actions, augmented 2x)
- Best FK Loss: 0.2149 (training) → 0.1095m (test holdout)
- Trained with Adam optimizer, batch size 128
- Early stopped at epoch 14 (best) after 40 epochs patience

---

## THE GOAL

Train a neural network to perform IK in REAL TIME during gameplay.

```
The network LEARNS from animations.
It sees thousands of poses, learns what works,
and generalizes to novel situations during gameplay.
```

---

## WORKFLOW (SIMPLIFIED)

### USE EXISTING DATA (90% of the time)
```
In Blender's Neural IK panel:
  1. Click "Load Saved Data"      ← loads training_data.npz
  2. Click "Load Weights"         ← loads best.npy
  3. Click "Run Tests"            ← verify 4/4 pass
```

### CREATE NEW DATA (only when animations change)
```
In Blender:
  1. Click "Extract Data"         ← scrapes all animations
  2. Click "Save to Disk"         ← saves training_data.npz

In Terminal:
  cd C:\Users\spenc\Desktop\Exploratory\addons\Exploratory\
     Exp_Game\animations\neural_network
  python standalone_trainer.py

Back in Blender:
  3. Click "Load Saved Data"
  4. Click "Load Weights"
  5. Click "Run Tests"
```

---

## ARCHITECTURE SEPARATION

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BLENDER vs STANDALONE                               │
└─────────────────────────────────────────────────────────────────────────────┘

    BLENDER (requires bpy):
    ├── data.py              Extract training data from animations
    ├── runtime.py           Apply poses during gameplay
    └── test_panel.py        UI operators

    STANDALONE (pure NumPy, runs in terminal):
    ├── standalone_trainer.py    Training loop with Adam optimizer
    ├── forward_kinematics.py    FK math for loss computation
    └── network.py               Network architecture (shared)

    SHARED (no bpy):
    ├── config.py            Rig data, paths, hyperparameters
    ├── context.py           Input normalization
    └── tests.py             Test suite (FK-based metrics)
```

**WHY SEPARATE?**
- Training is computationally expensive (FK gradient = slow)
- Running in Blender freezes the UI
- Standalone Python can use optimized NumPy/BLAS
- You can use Blender while training runs in terminal

---

## FILE LOCATIONS

```
Desktop (permanent - survives addon reinstalls):
  C:\Users\spenc\Desktop\Exploratory\addons\Exploratory\
  └── Exp_Game/animations/neural_network/
      ├── standalone_trainer.py    ← Run this in terminal
      ├── training_data/
      │   ├── training_data.npz    ← Your animation samples (552 total)
      │   └── weights/
      │       └── best.npy         ← Trained network (loss: 0.2149)
      └── *.py                     ← Code files

AppData (temporary - gets replaced on reinstall):
  C:\Users\spenc\AppData\...\addons\Exploratory
  └── This is where Blender loads from, but DON'T edit here
```

---

## BLENDER UI BUTTONS

| Button | What It Does |
|--------|--------------|
| **Load Saved Data** | training_data.npz → memory |
| **Load Weights** | best.npy → network |
| **Run Tests** | Verifies network learned (4 FK-based tests) |
| **Extract Data** | Scrapes all animations → memory (only when anims change) |
| **Save to Disk** | Memory → training_data.npz |
| **Show Full Path** | Prints standalone training commands to console |
| **Reset Weights** | Erases all learning (start over) |

---

## STANDALONE TRAINER

Located at: `neural_network/standalone_trainer.py`

**Run from terminal:**
```
cd C:\Users\spenc\Desktop\Exploratory\addons\Exploratory\Exp_Game\animations\neural_network
python standalone_trainer.py
```

**Features:**
- Adam optimizer (lr=0.001, β1=0.9, β2=0.999)
- Batch size 128 for speed
- FK gradient every 16 batches
- Contact gradient every 8 batches
- FK gradient disabled once loss < 0.02
- Contact flags inferred from foot Z positions (< 0.1m = grounded)
- Early stopping (40 epochs without improvement)
- CPU monitoring with psutil (optional)

**Last Training Output:**
```
═══════════════════════════════════════════════════════════════════════
 NEURAL IK STANDALONE TRAINER
═══════════════════════════════════════════════════════════════════════
 Runtime:    Python 3.11 | NumPy 1.26
 Threads:    12 CPU cores (auto-detected)
 Optimizer:  Adam | Batch size 128
═══════════════════════════════════════════════════════════════════════
✓ Data loaded: training_data.npz
  Samples:  490 train + 62 test (11% holdout)

[  4.7%] Epoch  14 | FK=0.2187 Test=0.2149 | ★ (best)
...
  ⚠ Early stopping after 40 epochs without improvement

═══════════════════════════════════════════════════════════════════════
 TRAINING COMPLETE
═══════════════════════════════════════════════════════════════════════
 Best FK Loss:      0.214926 (epoch 14)
 Weights saved:     weights/best.npy
═══════════════════════════════════════════════════════════════════════
```

---

## TEST SUITE CRITERIA

| Test | Threshold | Score | Status |
|------|-----------|-------|--------|
| Holdout (FK) | < 0.25m | 0.1095m | ✓ PASS |
| Interpolation (FK) | < 0.30m | 0.1094m | ✓ PASS |
| Consistency | = 0.00 | 0.0000 | ✓ PASS |
| Noise Robustness | < 0.50 | 0.0213 | ✓ PASS |

**IMPORTANT:** Tests now use FK loss (same metric as training).
Previous tests used rotation MSE which gave misleading scores (~175 instead of ~0.1).

---

## NETWORK ARCHITECTURE

```
INPUT (50 dimensions)
├── Effector Targets (30): 5 effectors × (pos[3] + rot[3])
├── Root Orientation (6): forward[3] + up[3]
├── Ground Context (12): 2 feet × 6 values
└── Motion State (2): phase + task_type
         ▼
┌─────────────────────────────────────────┐
│  HIDDEN LAYER 1: 128 neurons (tanh)     │
└─────────────────────────────────────────┘
         ▼
┌─────────────────────────────────────────┐
│  HIDDEN LAYER 2: 96 neurons (tanh)      │
└─────────────────────────────────────────┘
         ▼
OUTPUT (69 dimensions)
└── 23 bones × 3 axis-angle rotations
```

**Parameter Count**: ~23,000 weights
**Inference Time**: <0.1ms (fast enough for 30Hz game loop)

---

## TRAINING LOSSES

| Loss | Weight | Purpose |
|------|--------|---------|
| **FK Loss** | 1.0 | Primary - do rotations reach target positions? |
| **Contact Loss** | 0.5 | Feet stay on ground when grounded |
| Pose Loss | 0.3 | Secondary - match training animation poses |
| Limit Penalty | 0.1 | Soft joint constraint |

**Key**: FK Loss uses numerical gradient (slow but necessary).
Contact Loss ensures feet plant correctly - inferred from target foot positions.
Once FK loss is low, it's disabled and pose+contact refine the result (faster).

---

## CURRENT DATA

**Extracted from 7 actions:**
- Cube.005Action: 121 samples (locomotion)
- exp_fall: 34 samples (locomotion)
- exp_idle: 37 samples (idle)
- exp_jump: 27 samples (jump)
- exp_land: 41 samples (locomotion)
- exp_run: 17 samples (locomotion)
- exp_walk: 30 samples (locomotion)

**Total: 307 raw → 490 train + 62 test (with 2x augmentation)**

---

## NEXT STEPS (OPTIONAL)

To improve accuracy further:
1. Add more diverse animations (grabs, combat, climbing)
2. Retrain with lower learning rate (0.0003) for finer convergence
3. Current 0.11m error is already good for most animation use cases

---

**Last Updated**: 2025-12-30
