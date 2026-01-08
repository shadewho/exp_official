# Exploratory Standard Rig

**Last Updated**: 2025-12-27

This is the official humanoid rig for the Exploratory game engine. All default animations and procedural features target this exact bone structure.

---

## Quick Reference

| Stat | Value |
|------|-------|
| **Total Bones** | 54 |
| **Naming Convention** | Mixamo-style (LeftArm, RightArm, etc.) |
| **Rest Pose** | T-Pose (arms horizontal) |
| **Total Height** | ~1.97m (head top) |
| **Hips Height** | 1.001m |
| **Floor Level** | Z = 0 |
| **Arm Span** | ~1.4m |
| **Units** | Meters |

---

## Hierarchy Tree

```
Root (ARMATURE ROOT - at origin, Z=0)
└── Hips (pelvis control - can translate/rotate for body mechanics)
    ├── Spine
    │   └── Spine1
    │       └── Spine2
    │           ├── NeckLower
    │           │   └── NeckUpper
    │           │       └── Head
    │           ├── LeftShoulder
    │           │   └── LeftArm
    │           │       └── LeftForeArm
    │           │           └── LeftHand
    │           │               ├── LeftHandThumb1 → Thumb2 → Thumb3
    │           │               ├── LeftHandIndex1 → Index2 → Index3
    │           │               ├── LeftHandMiddle1 → Middle2 → Middle3
    │           │               ├── LeftHandRing1 → Ring2 → Ring3
    │           │               └── LeftHandPinky1 → Pinky2 → Pinky3
    │           └── RightShoulder
    │               └── RightArm
    │                   └── RightForeArm
    │                       └── RightHand
    │                           └── [Same finger structure as left]
    ├── LeftThigh
    │   └── LeftShin
    │       └── LeftFoot
    │           └── LeftToeBase
    └── RightThigh
        └── RightShin
            └── RightFoot
                └── RightToeBase
```

---

## Bone Groups

### Root Bone (1)

| Bone | Parent | Length | Position (Head) | Purpose |
|------|--------|--------|-----------------|---------|
| `Root` | None (armature root) | 0m | (0, 0, 0) | World anchor - character's position in scene. Never animated directly. |

**Important:** The `Root` bone is the armature root, positioned at the origin (ground level). It defines WHERE the character is in the world.

### Core/Spine Bones (9)

| Bone | Parent | Length | Position (Head) | Purpose |
|------|--------|--------|-----------------|---------|
| `Hips` | Root | 0.136m | (0, 0.056, 1.001) | Pelvis control - can translate (crouch) and rotate (lean/tilt). |
| `Spine` | Hips | 0.145m | (0, 0.056, 1.137) | Lower back |
| `Spine1` | Spine | 0.191m | (0, 0.051, 1.282) | Mid back |
| `Spine2` | Spine1 | 0.149m | (0, 0.050, 1.473) | Upper back / chest |
| `NeckLower` | Spine2 | 0.039m | (0, 0.047, 1.623) | Base of neck |
| `NeckUpper` | NeckLower | 0.039m | (0, 0.047, 1.662) | Upper neck |
| `Head` | NeckUpper | 0.272m | (0, 0.047, 1.701) | Head (ends at ~1.97m) |
| `LeftShoulder` | Spine2 | 0.097m | (-0.045, 0.047, 1.605) | Clavicle left |
| `RightShoulder` | Spine2 | 0.097m | (0.045, 0.047, 1.605) | Clavicle right |

### Leg Bones (8)

| Bone | Parent | Length | Position (Head) |
|------|--------|--------|-----------------|
| `LeftThigh` | Hips | 0.495m | (-0.075, 0.054, 1.065) |
| `LeftShin` | LeftThigh | 0.478m | (-0.106, 0.054, 0.571) |
| `LeftFoot` | LeftShin | 0.175m | (-0.110, -0.016, 0.098) |
| `LeftToeBase` | LeftFoot | 0.066m | (-0.111, 0.132, 0.006) |
| `RightThigh` | Hips | 0.495m | (0.075, 0.054, 1.065) |
| `RightShin` | RightThigh | 0.478m | (0.106, 0.054, 0.571) |
| `RightFoot` | RightShin | 0.169m | (0.110, -0.010, 0.098) |
| `RightToeBase` | RightFoot | 0.063m | (0.111, 0.131, 0.006) |

### Arm Bones (6)

| Bone | Parent | Length | Position (Head) |
|------|--------|--------|-----------------|
| `LeftArm` | LeftShoulder | 0.278m | (-0.136, 0.047, 1.570) |
| `LeftForeArm` | LeftArm | 0.286m | (-0.414, 0.044, 1.557) |
| `LeftHand` | LeftForeArm | 0.072m | (-0.700, 0.044, 1.557) |
| `RightArm` | RightShoulder | 0.278m | (0.136, 0.048, 1.570) |
| `RightForeArm` | RightArm | 0.286m | (0.414, 0.044, 1.557) |
| `RightHand` | RightForeArm | 0.072m | (0.700, 0.045, 1.557) |

### Finger Bones (30)

Each hand has 5 fingers with 3 bones each:

**Left Hand:**
| Finger | Bone 1 | Bone 2 | Bone 3 |
|--------|--------|--------|--------|
| Thumb | `LeftHandThumb1` | `LeftHandThumb2` | `LeftHandThumb3` |
| Index | `LeftHandIndex1` | `LeftHandIndex2` | `LeftHandIndex3` |
| Middle | `LeftHandMiddle1` | `LeftHandMiddle2` | `LeftHandMiddle3` |
| Ring | `LeftHandRing1` | `LeftHandRing2` | `LeftHandRing3` |
| Pinky | `LeftHandPinky1` | `LeftHandPinky2` | `LeftHandPinky3` |

**Right Hand:** Same structure with `Right` prefix.

---

## Chain Definitions

### Look-At Chain

Used for head tracking and eye contact.

```python
LOOK_AT = {
    "bones": ["NeckLower", "NeckUpper", "Head"],
    "weights": [0.15, 0.25, 0.60],  # Head contributes most
    "limits": {
        "yaw": (-70, 70),           # Left/right degrees
        "pitch": (-40, 60),         # Down/up degrees
    },
}
```

### Spine Chain

Used for procedural leaning, twisting, breathing.

```python
SPINE = {
    "bones": ["Hips", "Spine", "Spine1", "Spine2"],
    "lengths": [0.136, 0.145, 0.191, 0.149],
    "total_length": 0.621,
}
```

### Finger Chains

Used for grip poses and procedural hand shapes.

```python
FINGERS = {
    # Left hand
    "thumb_L":  ["LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3"],
    "index_L":  ["LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3"],
    "middle_L": ["LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3"],
    "ring_L":   ["LeftHandRing1", "LeftHandRing2", "LeftHandRing3"],
    "pinky_L":  ["LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3"],
    # Right hand
    "thumb_R":  ["RightHandThumb1", "RightHandThumb2", "RightHandThumb3"],
    "index_R":  ["RightHandIndex1", "RightHandIndex2", "RightHandIndex3"],
    "middle_R": ["RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3"],
    "ring_R":   ["RightHandRing1", "RightHandRing2", "RightHandRing3"],
    "pinky_R":  ["RightHandPinky1", "RightHandPinky2", "RightHandPinky3"],
}
```

---

## Key Measurements

### Character Proportions

```
                    ┌─────┐ 1.97m (Head top)
                    │HEAD │
                    └──┬──┘ 1.70m (Head base)
                       │
                    ┌──┴──┐ 1.62m (Neck)
    ┌───────────────┤CHEST├───────────────┐
    │   SHOULDER    └──┬──┘   SHOULDER    │ 1.57m (Arms)
    │                  │                  │
   ARM               SPINE               ARM
    │                  │                  │
 FOREARM               │              FOREARM
    │                  │                  │
   HAND             ┌──┴──┐             HAND  ±0.70m X
                    │HIPS │ 1.00m
                    └──┬──┘
               ┌───────┴───────┐
             THIGH           THIGH
               │               │
             SHIN            SHIN
               │               │
             FOOT            FOOT  0.10m Z
               │               │
             ──┴──           ──┴──  0.00m (Floor)
```

### Critical Distances

| Measurement | Value | Use Case |
|-------------|-------|----------|
| Hip height | 1.001m | Character origin reference |
| Shoulder width | 0.27m | Collision capsule width hint |
| Arm reach (from shoulder) | 0.73m | Grab distance calculations |
| Leg length (hip to ankle) | 0.97m | Step height limits |
| Foot to floor (rest) | 0.098m | Ground offset |
| Eye height (approx) | ~1.65m | Camera placement |
| Total height | 1.97m | Collision capsule height |

### Limb Reach Limits

| Limb | Max Reach | Notes |
|------|-----------|-------|
| Left Leg | 0.973m | Thigh (0.495) + Shin (0.478) |
| Right Leg | 0.972m | Thigh (0.495) + Shin (0.478) |
| Left Arm | 0.565m | UpperArm (0.278) + ForeArm (0.286) |
| Right Arm | 0.565m | UpperArm (0.278) + ForeArm (0.286) |

---

## Bone Index Map (Alphabetical)

For numpy array operations, bones are indexed alphabetically. **Source of truth:** `animations/bone_groups.py`

```python
BONE_INDEX = {
    "Head": 0,
    "Hips": 1,
    "LeftArm": 2,
    "LeftFoot": 3,
    # ... (middle bones)
    "Root": 43,           # World anchor
    "RightHandThumb1": 44,
    # ... (remaining bones)
    "Spine": 51,
    "Spine1": 52,
    "Spine2": 53,
}
# TOTAL_BONES = 54
```

**Note:** Root is inserted alphabetically between RightHandRing3 and RightHandThumb1.

---

## Animation Bone Masks

For animation layers and masking:

```python
BONE_MASKS = {
    "full_body": ["*"],  # All 54 bones

    "upper_body": [
        "Spine", "Spine1", "Spine2",
        "NeckLower", "NeckUpper", "Head",
        "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
        "RightShoulder", "RightArm", "RightForeArm", "RightHand",
        # + all finger bones
    ],

    "lower_body": [
        "Hips",
        "LeftThigh", "LeftShin", "LeftFoot", "LeftToeBase",
        "RightThigh", "RightShin", "RightFoot", "RightToeBase",
    ],

    "arms_only": [
        "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
        "RightShoulder", "RightArm", "RightForeArm", "RightHand",
        # + all finger bones
    ],

    "legs_only": [
        "LeftThigh", "LeftShin", "LeftFoot", "LeftToeBase",
        "RightThigh", "RightShin", "RightFoot", "RightToeBase",
    ],

    "spine_and_head": [
        "Spine", "Spine1", "Spine2",
        "NeckLower", "NeckUpper", "Head",
    ],

    "left_arm": [
        "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
        # + left finger bones
    ],

    "right_arm": [
        "RightShoulder", "RightArm", "RightForeArm", "RightHand",
        # + right finger bones
    ],
}
```

---

## Rest Pose Reference

Key bone positions in rest (T-pose):

| Bone | Position (X, Y, Z) |
|------|-------------------|
| Hips | (0, 0.056, 1.001) |
| Head | (0, 0.047, 1.701) |
| LeftHand | (-0.700, 0.044, 1.557) |
| RightHand | (0.700, 0.045, 1.557) |
| LeftFoot | (-0.110, -0.016, 0.098) |
| RightFoot | (0.110, -0.010, 0.098) |
| LeftToeBase | (-0.111, 0.132, 0.006) |
| RightToeBase | (0.111, 0.131, 0.006) |

---

## Usage Notes

### For Users

- All characters must use this exact rig
- Weight paint your mesh to the provided armature
- Default animations will work automatically
- Procedural systems are pre-configured

### For Animation

- All animations use these exact bone names
- Rest pose is T-pose with palms facing down
- Root motion uses Hips bone
- Baked animations store 10 floats per bone (quat + loc + scale)

### For Procedural Animation

- Spine chain: additive leaning, breathing, recoil
- Neck/Head chain: look-at tracking
- Finger chains: procedural grip (curl based on object size)

### For Physics

- Each bone can map to a rigidbody capsule
- Key physics joints: Hips, Spine2, Head, UpperArms, ForeArms, Thighs, Shins
- Joint limits should match natural human range of motion

---

## Joint Rotation Limits

Anatomical rotation limits for each bone. Used by the engine to constrain poses during blending.

**Format**: `[min°, max°]` per axis. `[0, 0]` means no rotation allowed on that axis.

**Note on Hips bone**: The `Hips` bone is the ROOT MASTER PARENT of the entire armature. Rotating Hips rotates every single bone in the skeleton. It is intentionally **NOT included** in the engine's joint limit data (`default_limits.py`) because:
1. It's the root bone - all other bones are children of it
2. Applying limits would break root motion and character positioning
3. Any rotation applied to Hips affects the whole character's world-space orientation

The Hips bone is listed in the table below for documentation completeness, but the engine does not clamp its rotation.

### Spine & Torso

| Bone | X (Forward/Back) | Y (Twist) | Z (Side Bend) |
|------|------------------|-----------|---------------|
| Hips | No limit (root) | No limit | No limit |
| Spine | [-45, 45] | [-45, 45] | [-45, 45] |
| Spine1 | [-30, 30] | [-30, 30] | [-30, 30] |
| Spine2 | [-30, 30] | [-30, 30] | [-30, 30] |

### Neck & Head

| Bone | X (Nod) | Y (Turn) | Z (Tilt) |
|------|---------|----------|----------|
| NeckLower | [-45, 45] | [-45, 45] | [-30, 30] |
| NeckUpper | [-30, 30] | [-30, 30] | [-30, 30] |
| Head | [-45, 45] | [-60, 60] | [-30, 30] |

### Left Arm

| Bone | X | Y | Z |
|------|---|---|---|
| LeftArm | [-90, 90] | [-120, 120] | [-80, 140] |
| LeftForeArm | [0, 0] | [-170, 60] | [0, 90] |
| LeftHand | [-90, 90] | [0, 0] | [-60, 60] |

### Right Arm (Mirrored)

| Bone | X | Y | Z |
|------|---|---|---|
| RightArm | [-90, 90] | [-120, 120] | [-140, 80] |
| RightForeArm | [0, 0] | [-60, 170] | [-90, 0] |
| RightHand | [-90, 90] | [0, 0] | [-60, 60] |

### Left Leg

| Bone | X | Y | Z |
|------|---|---|---|
| LeftThigh | [-90, 120] | [-40, 40] | [-20, 80] |
| LeftShin | [-150, 10] | [-20, 20] | [-15, 15] |
| LeftFoot | [-80, 45] | [-30, 30] | [-40, 40] |
| LeftToeBase | [-40, 40] | [0, 0] | [0, 0] |

*Note: Shin Y/Z limits relaxed from [-10,10]/[0,0] to [-20,20]/[-15,15] based on animation test data analysis (2025-12-29).*

### Right Leg (Mirrored)

| Bone | X | Y | Z |
|------|---|---|---|
| RightThigh | [-90, 120] | [-40, 40] | [-80, 20] |
| RightShin | [-150, 10] | [-20, 20] | [-15, 15] |
| RightFoot | [-80, 45] | [-30, 30] | [-40, 40] |
| RightToeBase | [-40, 40] | [0, 0] | [0, 0] |

### Left Hand Fingers

| Bone | X (Curl) | Y | Z (Spread) |
|------|----------|---|------------|
| LeftHandThumb1 | [-30, 30] | [0, 0] | [-50, 40] |
| LeftHandThumb2 | [0, 0] | [0, 0] | [-60, 0] |
| LeftHandThumb3 | [0, 0] | [0, 0] | [-60, 0] |
| LeftHandIndex1 | [-20, 90] | [0, 0] | [-20, 20] |
| LeftHandIndex2 | [-20, 90] | [0, 0] | [-20, 20] |
| LeftHandIndex3 | [-20, 90] | [0, 0] | [-20, 20] |
| LeftHandMiddle1 | [-20, 90] | [0, 0] | [-20, 20] |
| LeftHandMiddle2 | [-20, 90] | [0, 0] | [-20, 20] |
| LeftHandMiddle3 | [-20, 90] | [0, 0] | [-20, 20] |
| LeftHandRing1 | [-20, 90] | [0, 0] | [-20, 20] |
| LeftHandRing2 | [-20, 90] | [0, 0] | [-20, 20] |
| LeftHandRing3 | [-20, 90] | [0, 0] | [-20, 20] |
| LeftHandPinky1 | [-20, 90] | [0, 0] | [-20, 20] |
| LeftHandPinky2 | [-20, 90] | [0, 0] | [-20, 20] |
| LeftHandPinky3 | [-20, 90] | [0, 0] | [-20, 20] |

### Right Hand Fingers (Mirrored)

| Bone | X (Curl) | Y | Z (Spread) |
|------|----------|---|------------|
| RightHandThumb1 | [-30, 30] | [0, 0] | [-40, 50] |
| RightHandThumb2 | [0, 0] | [0, 0] | [0, 60] |
| RightHandThumb3 | [0, 0] | [0, 0] | [0, 60] |
| RightHandIndex1 | [-20, 90] | [0, 0] | [-20, 20] |
| RightHandIndex2 | [-20, 90] | [0, 0] | [-20, 20] |
| RightHandIndex3 | [-20, 90] | [0, 0] | [-20, 20] |
| RightHandMiddle1 | [-20, 90] | [0, 0] | [-20, 20] |
| RightHandMiddle2 | [-20, 90] | [0, 0] | [-20, 20] |
| RightHandMiddle3 | [-20, 90] | [0, 0] | [-20, 20] |
| RightHandRing1 | [-20, 90] | [0, 0] | [-20, 20] |
| RightHandRing2 | [-20, 90] | [0, 0] | [-20, 20] |
| RightHandRing3 | [-20, 90] | [0, 0] | [-20, 20] |
| RightHandPinky1 | [-20, 90] | [0, 0] | [-20, 20] |
| RightHandPinky2 | [-20, 90] | [0, 0] | [-20, 20] |
| RightHandPinky3 | [-20, 90] | [0, 0] | [-20, 20] |

### JSON Format (For Engine)

```json
{
  "Spine": {"X": [-45, 45], "Y": [-45, 45], "Z": [-45, 45]},
  "Spine1": {"X": [-30, 30], "Y": [-30, 30], "Z": [-30, 30]},
  "Spine2": {"X": [-30, 30], "Y": [-30, 30], "Z": [-30, 30]},
  "NeckLower": {"X": [-45, 45], "Y": [-45, 45], "Z": [-30, 30]},
  "NeckUpper": {"X": [-30, 30], "Y": [-30, 30], "Z": [-30, 30]},
  "Head": {"X": [-45, 45], "Y": [-60, 60], "Z": [-30, 30]},

  "LeftArm": {"X": [-90, 90], "Y": [-120, 120], "Z": [-80, 140]},
  "LeftForeArm": {"X": [0, 0], "Y": [-170, 60], "Z": [0, 90]},
  "LeftHand": {"X": [-90, 90], "Y": [0, 0], "Z": [-60, 60]},
  "RightArm": {"X": [-90, 90], "Y": [-120, 120], "Z": [-140, 80]},
  "RightForeArm": {"X": [0, 0], "Y": [-60, 170], "Z": [-90, 0]},
  "RightHand": {"X": [-90, 90], "Y": [0, 0], "Z": [-60, 60]},

  "LeftThigh": {"X": [-90, 120], "Y": [-40, 40], "Z": [-20, 80]},
  "LeftShin": {"X": [-150, 10], "Y": [-20, 20], "Z": [-15, 15]},
  "LeftFoot": {"X": [-80, 45], "Y": [-30, 30], "Z": [-40, 40]},
  "LeftToeBase": {"X": [-40, 40], "Y": [0, 0], "Z": [0, 0]},
  "RightThigh": {"X": [-90, 120], "Y": [-40, 40], "Z": [-80, 20]},
  "RightShin": {"X": [-150, 10], "Y": [-20, 20], "Z": [-15, 15]},
  "RightFoot": {"X": [-80, 45], "Y": [-30, 30], "Z": [-40, 40]},
  "RightToeBase": {"X": [-40, 40], "Y": [0, 0], "Z": [0, 0]},

  "LeftHandThumb1": {"X": [-30, 30], "Y": [0, 0], "Z": [-50, 40]},
  "LeftHandThumb2": {"X": [0, 0], "Y": [0, 0], "Z": [-60, 0]},
  "LeftHandThumb3": {"X": [0, 0], "Y": [0, 0], "Z": [-60, 0]},
  "LeftHandIndex1": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},
  "LeftHandIndex2": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},
  "LeftHandIndex3": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},
  "LeftHandMiddle1": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},
  "LeftHandMiddle2": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},
  "LeftHandMiddle3": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},
  "LeftHandRing1": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},
  "LeftHandRing2": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},
  "LeftHandRing3": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},
  "LeftHandPinky1": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},
  "LeftHandPinky2": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},
  "LeftHandPinky3": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},

  "RightHandThumb1": {"X": [-30, 30], "Y": [0, 0], "Z": [-40, 50]},
  "RightHandThumb2": {"X": [0, 0], "Y": [0, 0], "Z": [0, 60]},
  "RightHandThumb3": {"X": [0, 0], "Y": [0, 0], "Z": [0, 60]},
  "RightHandIndex1": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},
  "RightHandIndex2": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},
  "RightHandIndex3": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},
  "RightHandMiddle1": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},
  "RightHandMiddle2": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},
  "RightHandMiddle3": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},
  "RightHandRing1": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},
  "RightHandRing2": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},
  "RightHandRing3": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},
  "RightHandPinky1": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},
  "RightHandPinky2": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]},
  "RightHandPinky3": {"X": [-20, 90], "Y": [0, 0], "Z": [-20, 20]}
}
```

---

## Bone Local Axis Orientations

### Purpose

**This section is critical for procedural animation.**

When we rotate a bone, we rotate around its LOCAL axes (X, Y, Z). But each bone has a different orientation in rest pose. Without knowing what each axis points to, we cannot:

1. **Interpret joint limits** - "LeftShin X: [-150, 10]" is meaningless unless we know what X means for that bone
2. **Validate poses** - We can't check if a knee is bending forward vs backward without knowing the axes

### Axis Conventions

- **Y-axis**: Points along the bone (head → tail). This is the "bone direction."
- **X-axis**: Perpendicular to Y, typically the "twist" axis (rotating the limb like turning a doorknob)
- **Z-axis**: Perpendicular to both, typically the "bend" axis for some bones

### Key Joint Bend Axes

| Joint | Bend Motion | Which Axis | Direction |
|-------|-------------|------------|-----------|
| Elbow | Forearm swings forward/back | **ForeArm local Z** | Z points DOWN, so rotating around Z moves forearm in sagittal plane |
| Knee | Shin swings forward/back | **Shin local X** | X points RIGHT, so rotating around X moves shin in sagittal plane |

### Full Bone Orientation Table

Each bone's local coordinate system in rest pose (T-pose).

| Bone | +X Points | +Y Points | +Z Points |
|------|-----------|-----------|-----------|
| `Head` | LEFT | UP | FORWARD |
| `Hips` | LEFT | UP | FORWARD |
| `Spine` | LEFT | UP | FORWARD |
| `Spine1` | LEFT | UP | FORWARD |
| `Spine2` | LEFT | UP | FORWARD |
| `NeckLower` | LEFT | UP | FORWARD |
| `NeckUpper` | LEFT | UP | FORWARD |
| `LeftShoulder` | BACK | LEFT | DOWN |
| `LeftArm` | BACK | LEFT | DOWN |
| `LeftForeArm` | BACK | LEFT | DOWN |
| `LeftHand` | BACK | LEFT | DOWN |
| `RightShoulder` | FORWARD | RIGHT | DOWN |
| `RightArm` | FORWARD | RIGHT | DOWN |
| `RightForeArm` | FORWARD | RIGHT | DOWN |
| `RightHand` | FORWARD | RIGHT | DOWN |
| `LeftThigh` | RIGHT | DOWN | FORWARD |
| `LeftShin` | RIGHT | DOWN | FORWARD |
| `LeftFoot` | RIGHT | FORWARD+DOWN | FORWARD+UP |
| `LeftToeBase` | RIGHT | FORWARD | UP |
| `RightThigh` | RIGHT | DOWN | FORWARD |
| `RightShin` | RIGHT | DOWN | FORWARD |
| `RightFoot` | RIGHT | FORWARD+DOWN | FORWARD+UP |
| `RightToeBase` | RIGHT | FORWARD | UP |

### Limb Axis Summary

**Arms (in T-pose, arms horizontal):**
```
LeftArm:     Y → LEFT (toward elbow)
             Z → DOWN (bend axis - elbow flexion)
             X → BACK (twist axis - forearm rotation)

RightArm:    Y → RIGHT (toward elbow)
             Z → DOWN (bend axis - elbow flexion)
             X → FORWARD (twist axis - forearm rotation)
```

**Legs (standing upright):**
```
LeftThigh:   Y → DOWN (toward knee)
             Z → FORWARD (not primary bend axis)
             X → RIGHT (bend axis - knee flexion)

RightThigh:  Y → DOWN (toward knee)
             Z → FORWARD (not primary bend axis)
             X → RIGHT (bend axis - knee flexion)
```

### How to Use This Data

**For Procedural Animation:**
```python
# To bend the right elbow (make forearm swing toward body):
right_forearm.rotation_euler.z = bend_angle  # Z is the bend axis

# To bend the right knee (make shin swing backward):
right_shin.rotation_euler.x = -bend_angle  # X is the bend axis, negative = back
```

**For Pose Validation:**
```python
# Check if knee is bending correctly (forward, not backward)
# Knee flexion is NEGATIVE X rotation (shin goes backward = knee bends forward)
if right_shin.rotation_euler.x > 10:  # More than 10° positive = bending wrong way
    print("ERROR: Knee bending backward!")
```

### Raw Vector Data

For precise calculations, here are the exact axis vectors:

| Bone | X Vector | Y Vector | Z Vector |
|------|----------|----------|----------|
| `RightArm` | (+0.02, +1.00, +0.02) | (+1.00, -0.01, -0.05) | (-0.05, +0.02, -1.00) |
| `RightForeArm` | (-0.00, +1.00, +0.02) | (+1.00, +0.00, -0.00) | (-0.00, +0.02, -1.00) |
| `RightHand` | (+0.00, +0.99, +0.13) | (+1.00, -0.01, +0.02) | (+0.02, +0.13, -0.99) |
| `LeftArm` | (+0.01, -1.00, -0.02) | (-1.00, -0.01, -0.05) | (+0.05, +0.02, -1.00) |
| `LeftForeArm` | (-0.00, -1.00, -0.02) | (-1.00, +0.00, -0.00) | (+0.00, +0.02, -1.00) |
| `LeftHand` | (+0.00, -0.99, -0.12) | (-1.00, -0.01, +0.03) | (-0.03, +0.12, -0.99) |
| `RightThigh` | (+1.00, +0.00, +0.06) | (+0.06, -0.00, -1.00) | (-0.00, +1.00, -0.00) |
| `RightShin` | (+1.00, -0.01, +0.01) | (+0.01, -0.13, -0.99) | (+0.01, +0.99, -0.13) |
| `RightFoot` | (+1.00, +0.01, +0.03) | (+0.01, +0.84, -0.55) | (-0.03, +0.54, +0.84) |
| `LeftThigh` | (+1.00, -0.00, -0.06) | (-0.06, -0.00, -1.00) | (+0.00, +1.00, -0.00) |
| `LeftShin` | (+1.00, +0.01, -0.01) | (-0.01, -0.15, -0.99) | (-0.01, +0.99, -0.15) |
| `LeftFoot` | (+1.00, -0.01, -0.03) | (-0.01, +0.85, -0.53) | (+0.03, +0.53, +0.85) |

---

## Version History

| Date | Change |
|------|--------|
| 2025-12-29 | Relaxed Shin Y/Z limits based on animation test data analysis |
| 2025-12-26 | Added Bone Local Axis Orientations section |
| 2025-12-25 | Added Joint Rotation Limits section |
| 2025-12-20 | Initial comprehensive documentation |