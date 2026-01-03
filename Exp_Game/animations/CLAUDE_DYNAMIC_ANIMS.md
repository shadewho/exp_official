# Neural Network IK System

**Location**: `Exp_Game/animations/neural_network/`

---

## CRITICAL: FILE PATHS

**ALWAYS save/edit files in:**
```
C:\Users\spenc\Desktop\Exploratory\addons\Exploratory\
```

**NEVER save/edit files in:**
```
C:\Users\spenc\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons\Exploratory\
```

**Why?** The AppData version is disposable - it gets deleted and replaced every time the addon is refreshed. Any changes there will be lost. The Desktop version is the source of truth.

Training data, weights, code changes - EVERYTHING permanent goes to Desktop.

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
  1. Click "Load Saved Data"      <- loads training_data.npz
  2. Click "Load Weights"         <- loads best.npy
  3. Click "Run Tests"            <- verify tests pass
```

### CREATE NEW DATA (when animations change)
```
In Blender:
  1. Click "Extract Data"         <- scrapes all animations
  2. Click "Save to Disk"         <- saves training_data.npz

In Terminal:
  cd C:\Users\spenc\Desktop\Exploratory\addons\Exploratory\
     Exp_Game\animations\neural_network
  python torch_trainer.py

Back in Blender:
  3. Click "Load Saved Data"
  4. Click "Load Weights"
  5. Click "Run Tests"
```

---

## ARCHITECTURE SEPARATION

```
BLENDER (requires bpy):
├── data.py              Extract training data from animations
└── test_panel.py        UI operators

STANDALONE (PyTorch GPU, runs in terminal):
├── torch_trainer.py         GPU training with autograd
├── forward_kinematics.py    FK math (NumPy fallback)
└── network.py               Network architecture (shared)

SHARED (no bpy):
├── config.py            Rig data, paths, hyperparameters
├── context.py           Input normalization
└── tests.py             Test suite (FK-based metrics)
```

**WHY SEPARATE?**
- Training benefits from GPU acceleration (PyTorch + CUDA)
- Running in Blender freezes the UI
- Autograd eliminates slow finite-difference gradients
- You can use Blender while training runs in terminal

---

## FILE LOCATIONS

```
Desktop (permanent - survives addon reinstalls):
  C:\Users\spenc\Desktop\Exploratory\addons\Exploratory\
  └── Exp_Game/animations/neural_network/
      ├── torch_trainer.py         <- Run this in terminal (GPU)
      ├── training_data/
      │   ├── training_data.npz    <- Your animation samples
      │   └── weights/
      │       └── best.npy         <- Trained network
      └── *.py                     <- Code files

AppData (temporary - gets replaced on reinstall):
  C:\Users\spenc\AppData\...\addons\Exploratory
  └── This is where Blender loads from, but DON'T edit here
```

---

## BLENDER UI BUTTONS

| Button | What It Does |
|--------|--------------|
| **Load Saved Data** | training_data.npz -> memory |
| **Load Weights** | best.npy -> network |
| **Run Tests** | Verifies network learned (FK-based tests) |
| **Extract Data** | Scrapes all animations -> memory |
| **Save to Disk** | Memory -> training_data.npz |
| **Reset Weights** | Erases all learning (start over) |

---

## PYTORCH GPU TRAINER

Located at: `neural_network/torch_trainer.py`

**Run from terminal:**
```
cd C:\Users\spenc\Desktop\Exploratory\addons\Exploratory\Exp_Game\animations\neural_network
python torch_trainer.py
```

**Prerequisites:**
- PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
- Verify: `python -c "import torch; print(torch.cuda.is_available())"`

**Features:**
- GPU-accelerated training (autograd - no finite differences)
- Adam optimizer
- Mixed precision (AMP) for speed
- Rodrigues axis-angle -> rotation matrix
- Quaternion geodesic orientation loss
- Early stopping

---

## NETWORK ARCHITECTURE

```
INPUT (50 dimensions)
├── Effector Targets (30): 5 effectors x (pos[3] + rot[3])
├── Root Orientation (6): forward[3] + up[3]
├── Ground Context (12): 2 feet x 6 values
└── Motion State (2): phase + task_type
         |
┌─────────────────────────────────────────┐
│  HIDDEN LAYER 1: 128 neurons (tanh)     │
└─────────────────────────────────────────┘
         |
┌─────────────────────────────────────────┐
│  HIDDEN LAYER 2: 96 neurons (tanh)      │
└─────────────────────────────────────────┘
         |
OUTPUT (69 dimensions)
└── 23 bones x 3 axis-angle rotations
```

**Parameter Count**: ~23,000 weights
**Inference Time**: <0.1ms (fast enough for 30Hz game loop)

---

## TRAINING LOSSES

| Loss | Weight | Purpose |
|------|--------|---------|
| **FK Loss** | 1.0 | Primary - do rotations reach target positions? |
| **Contact Loss** | 0.5 | Feet stay on ground when grounded |
| **Orientation Loss** | 0.7 | Match effector rotations |
| Pose Loss | 0.3 | Secondary - match training animation poses |
| Limit Penalty | 0.5 | Soft joint constraint |
| Smooth Penalty | 0.1 | Noise robustness |

---

---

## ⚠️ CRITICAL: FK PIPELINE VALIDATION

**Before training, you MUST validate that the FK math matches Blender's actual bone transforms.**

### Why This Test Exists

The neural network learns to produce **bone rotations** that will be applied in Blender. If our FK computation doesn't match how Blender actually positions bones, then:
- Ground truth rotations won't produce ground truth positions
- The network learns the wrong mapping
- Predictions look broken in Blender even if training loss is low

### The Validation Operator

In Blender's Neural IK panel, click **"Validate FK Pipeline"**. This runs 3 tests:

| Test | What It Checks |
|------|----------------|
| **1. REST_POSITIONS** | Do config.py rest positions match the actual armature? |
| **2. Identity Rotations** | Does FK with identity produce rest positions exactly? |
| **3. FK vs Blender** | Do computed positions match Blender's actual positions for all 23 bones? |

**ALL 3 TESTS MUST PASS** before training will work.

### The Root Rotation Bug (Critical)

The most common FK bug is **double-applying the rest orientation**.

**WRONG** (causes 100+ cm errors):
```python
# Don't use hips bone matrix for root_rot!
hips_rest_matrix = arm_matrix @ Matrix(hips.bone.matrix_local)
root_rot = build_from(hips_rest_matrix)  # WRONG - doubles rest orientation
```

**CORRECT**:
```python
# Use armature's world rotation (usually identity)
arm_rot_3x3 = arm_matrix.to_3x3()
root_rot = np.array([...arm_rot_3x3...])  # CORRECT
```

**Why?** FK already applies `REST_ORIENTATIONS[i]` internally for each bone. The `root_rotation` should only encode the armature's world transform (rotation of the whole character), not the bone's rest frame.

### Alignment Across Files

All 3 files MUST use the same root_rotation approach:

| File | Purpose | root_rotation Source |
|------|---------|---------------------|
| `test_panel.py` | Blender validation | `arm_matrix.to_3x3()` |
| `data.py` | Training data extraction | `arm_matrix.to_3x3()` columns |
| `torch_trainer.py` | GPU training FK | `build_root_rotation(fwd, up)` |

If any file uses the hips bone matrix instead, the pipeline breaks.

### After Making Changes

If you modify FK math or config.py:
1. Run **Validate FK Pipeline** in Blender
2. If it fails, fix the issue
3. **Re-extract training data** (old data has wrong root info!)
4. Train with `--fresh` flag to ignore old weights

---

## PyTorch FK Validation

The PyTorch trainer (`torch_trainer.py`) has its own FK validation that runs before training:

```
----------------------------------------------------------------------
 FK VALIDATION - Testing if our math matches Blender
----------------------------------------------------------------------
 Testing FK on 100 ground truth samples...
 (Ground truth rotations should produce ground truth positions)

   LeftHand    :   0.12 cm  [OK]
   RightHand   :   0.15 cm  [OK]
   LeftFoot    :   0.08 cm  [OK]
   RightFoot   :   0.09 cm  [OK]
   Head        :   0.05 cm  [OK]

   Overall RMSE: 0.10 cm

   [PASS] FK matches Blender (error < 1cm)
----------------------------------------------------------------------
```

If this shows errors > 5cm, **DO NOT TRAIN**. The network will learn garbage.

---

## TIPS

- More diverse training data = better generalization
- Re-extract data when you add new animations
- Test on poses the network hasn't seen (Test Set or Pose Library)
- Position error <15cm is good, <5cm is excellent
- **Always run FK validation after changing config.py or FK code**
- **Always re-extract data after fixing FK bugs** (old data is corrupted)

---

## RUNTIME INTEGRATION (Planned)

### What We Have

| Component | Status | Accuracy |
|-----------|--------|----------|
| Trained network | ✅ Complete | 2.2cm position, 6° rotation |
| FK validation | ✅ Passing | All 23 bones |
| Training data | ✅ 130 animations | ~17K samples |
| Blender test UI | ✅ Working | Pred/Truth toggle |

### Architecture: Engine Worker Integration

Neural IK inference MUST run in engine workers (no exceptions). This respects the engine-first mandate.

```
MAIN THREAD                              WORKER
──────────────────────────────────────────────────────────
1. Compute targets (bpy reads)
   - Object positions (reach targets)
   - Raycast results (foot planting)
   - Current root position/orientation

2. Build job data
   - effector_targets (5×3)
   - root_position (3)
   - root_forward (3)
   - root_up (3)
   - context (phase, task, ground)

3. Submit NEURAL_IK_SOLVE ────────────> 4. Build input vector (50 dims)
                                        5. Normalize input
                                        6. Network forward pass (~100µs)
                                        7. Clamp to joint limits
                                        8. Convert to quaternions

9. Poll result <─────────────────────── 10. Return bone_rotations (23×4)

11. Apply to bones (bpy writes)
12. Blend with animation (optional)
```

### Worker Cache (Loaded at Game Start)

Workers cache network weights and config at startup via `CACHE_NEURAL_IK` job:
- Network weights (~23K parameters)
- REST_ORIENTATIONS, LOCAL_OFFSETS
- Joint limits (LIMITS_MIN, LIMITS_MAX)

### New Files

```
engine/worker/
└── ik.py              # Neural IK inference (pure numpy, no bpy)

animations/
└── exp_neural_ik.py   # Main thread wrapper (job submit/apply)
```

### Job Type: NEURAL_IK_SOLVE

```python
# Input (from main thread)
{
    'effector_targets': (5, 3),    # World positions for hands/feet/head
    'root_position': (3,),         # Hips world position
    'root_forward': (3,),          # Character facing direction
    'root_up': (3,),               # Character up vector
    'motion_phase': float,         # 0-1 animation phase
    'task_type': int,              # 0=idle, 1=locomotion, 2=reach, etc.
    'ground_height': float,        # Ground plane Z
    'contact_flags': (2,),         # [left_foot_grounded, right_foot_grounded]
}

# Output (from worker)
{
    'bone_rotations': (23, 4),     # Quaternions for all bones
    'success': bool,
}
```

### Use Cases This Enables

| Feature | How It Works |
|---------|--------------|
| Foot IK | Raycast ground → foot targets → solve |
| Hand reach | Object position → hand target → solve |
| Look-at | Target position → head target → solve |
| Procedural grab | Trigger → animate hand to object |
| Pose-to-pose | Saved pose effectors → solve transition |

### Procedural Action System (Future)

Atomic actions that generate effector targets:
- `LookAt(target)` → sets head target
- `Reach(object)` → sets hand target
- `Lean(direction)` → adjusts spine targets
- `Crouch(amount)` → lowers hip target

Actions compose together. Neural IK solves all constraints simultaneously.

### Performance Budget

```
Frame budget:     33ms (30Hz)
Physics worker:   ~200µs
Neural IK worker: ~125µs
Total:            ~325µs = ~1% of budget ✓
```

### Implementation Order

1. **Worker IK module** (`worker/ik.py`) - Core inference, logging
2. **Cache job** (`CACHE_NEURAL_IK`) - Load weights at startup
3. **Solve job** (`NEURAL_IK_SOLVE`) - Single inference request
4. **Main thread wrapper** - Submit/poll/apply
5. **Debug visualization** - GPU overlay for targets/results
6. **Action generators** - LookAt, Reach, etc.
7. **Node integration** - Connect to node graph system

---

## CURRENT PROGRESS

**Status:** Engine integration working, blend system integration NOT applying poses

**Last Updated:** Jan 2025

### Architecture (Updated)

Network: 50 → 256 → 256 → 128 → 69 (120,645 params)

### Completed & Verified

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| `network_forward()` | `worker/ik.py` | ✅ VERIFIED | Matches network.py, ~400-850µs inference |
| `normalize_input()` | `worker/ik.py` | ✅ VERIFIED | Matches context.py |
| `axis_angle_to_quat()` | `worker/ik.py` | ✅ VERIFIED | Batch conversion, unit quaternions |
| `build_input()` | `worker/ik.py` | ✅ VERIFIED | Builds 50-dim input from effector targets |
| `CACHE_NEURAL_IK` job | `engine_worker_entry.py` | ✅ WORKING | All 4 workers cache weights at game start |
| `NEURAL_IK_SOLVE` job | `engine_worker_entry.py` | ✅ WORKING | Full pipeline: build→normalize→forward→quaternions |
| Weight loading | `exp_modal.py` | ✅ WORKING | 120,645 params loaded in ~1ms |
| K key test | `exp_modal.py` | ✅ WORKING | Uses pose library targets, comprehensive logging |
| `neural_ik_layer.py` | `animations/` | ✅ CREATED | Bridges worker output to blend system |

### Current Issue: Blend System Not Applying Poses

The network runs and produces reasonable output:
- Input range: [-0.9, 1.0] ✓
- Output axis-angles: [-1.3, 1.0] radians ✓
- Rotation magnitudes: 1-88° avg ~20-27° ✓
- Inference: ~400-850µs ✓

**BUT the character doesn't move.** Test logs show:
```
CURRENT_EFFECTORS (before IK): LeftHand dist=0.388m
RESULT_EFFECTORS (after IK):   LeftHand err=0.388m  ← SAME!
```

The `neural_ik_layer.py` creates an OVERRIDE layer with a `pose_provider` callback that returns the cached IK pose. The layer IS created (`LAYER_CREATED weight=1.0 mask_bones=23`), but:
- Either the blend system isn't calling the pose_provider
- Or `apply_to_armature()` doesn't run after the K test updates the cache
- Or the layer isn't being processed in the frame's blend evaluation

### Files Created/Modified

```
engine/worker/ik.py           - Neural IK inference (pure numpy)
animations/neural_ik_layer.py - Blend system integration layer
modal/exp_modal.py            - K key test, weight caching at startup
engine/engine_worker_entry.py - CACHE_NEURAL_IK, NEURAL_IK_SOLVE handlers
```

### K Test Flow (Working)

1. Get target pose from pose library (selected in UI)
2. Extract root position from CURRENT pose
3. Temporarily apply target pose to read effector world positions
4. Restore original pose
5. Submit NEURAL_IK_SOLVE job with target effector positions
6. Worker: build_input → normalize → forward → quaternions
7. Apply result to neural_ik_layer cached pose
8. Measure error (currently shows no movement)

### Diagnostic Logging (Comprehensive)

The K test now logs:
- TARGET_EFFECTORS (world coords from pose library)
- CURRENT_EFFECTORS (before IK, with dist_to_target)
- JOB_INPUT (root position, forward, up)
- INPUT_STATS (raw range, normalized range, means)
- OUTPUT_STATS (axis-angle range, rotation magnitudes in degrees)
- EFFECTOR_REL (root-relative positions network sees)
- RESULT_EFFECTORS (after IK, with error and movement)
- ACCURACY (total/avg error, REACHED/MISSED status)

### Next Steps

| Task | Purpose | Status |
|------|---------|--------|
| Fix blend system integration | Make IK poses actually apply to armature | ⬜ BLOCKING |
| Verify pose_provider called | Debug if blend system queries the layer | ⬜ Next |
| Force blend system update | May need to trigger re-evaluation after K | ⬜ Next |
| Joint limit clamping | Enforce constraints on output | ⬜ Later |
| Animated transitions | Smooth blend to IK pose over time | ⬜ Later |



### main goal:
the end goal is to prep and have the neural training able to guide or easily provide actions such as grab object etc. but right now we
 need to confirm that we can get it working in the worker engine with practical computation loads. thats the goal. but i hate when we try to jump straight
into it without ensuring that it can work. we always have to integrate IK overrides for full body or partial body into my layer system
so that the IK essentially gets permission to override the locomotion so we dont have any fighting. but its so important that we work on
 the fundamentals. i have the trained data working and i have results for accurate ground truth vs prediction. next are small steps to
getting real results from the trained data during game.
its critical we log results: Exp_Game\developer\CLAUDE_LOGGER.md and avoid guesswork as we develop
its critical we use the worker engine as much as possible and monitor its load and efficiency. optimized solutions are critical in the engine: Exp_Game\engine\CLAUDE_ENGINE_CONTEXT.md