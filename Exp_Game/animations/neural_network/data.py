# Exp_Game/animations/neural_network/data.py
"""
Training Data Extraction - Environment-Aware Version

Extracts (input, output) pairs from Blender animations:
    Input: Root-relative effector targets + environment context (50 dims)
    Output: Axis-angle bone rotations (69 dims)

Also includes target effector positions for FK loss computation.
"""

import bpy
import numpy as np
from typing import List, Dict, Tuple, Optional
from mathutils import Vector, Quaternion, Matrix, Euler

from .config import (
    CONTROLLED_BONES,
    END_EFFECTORS,
    CONTACT_EFFECTORS,
    INPUT_SIZE,
    OUTPUT_SIZE,
    TRAIN_SPLIT,
    TASK_TYPES,
    BONE_TO_INDEX,
    DATA_DIR,
)
from .context import ContextExtractor, augment_input


def euler_to_axis_angle(euler: Euler) -> np.ndarray:
    """Convert Euler angles to axis-angle representation."""
    # Convert to quaternion first, then to axis-angle
    quat = euler.to_quaternion()
    axis, angle = quat.to_axis_angle()
    return np.array([axis.x * angle, axis.y * angle, axis.z * angle], dtype=np.float32)


def quaternion_to_axis_angle(quat: Quaternion) -> np.ndarray:
    """Convert quaternion to axis-angle representation."""
    axis, angle = quat.to_axis_angle()
    return np.array([axis.x * angle, axis.y * angle, axis.z * angle], dtype=np.float32)


class AnimationDataExtractor:
    """
    Extracts training data from Blender animations.

    For each frame of each animation:
        - Records root-relative effector positions/rotations
        - Records environment context (ground, contacts)
        - Records all bone rotations as axis-angle
        - Records target effector world positions (for FK loss)
    """

    def __init__(self, armature: bpy.types.Object):
        """
        Initialize extractor for an armature.

        Args:
            armature: The Blender armature object to extract from.
        """
        if armature is None or armature.type != 'ARMATURE':
            raise ValueError("Must provide a valid armature object")

        self.armature = armature
        self.context_extractor = ContextExtractor(armature)
        self.samples: List[Dict] = []

    def extract_from_action(
        self,
        action: bpy.types.Action,
        task_type: str = "locomotion",
        ground_height: float = 0.0,
    ) -> int:
        """
        Extract training samples from a single action.

        Args:
            action: The Blender action to extract from.
            task_type: Type of motion (idle, locomotion, reach, etc.)
            ground_height: Ground plane Z height

        Returns:
            Number of samples extracted.
        """
        if action is None:
            return 0

        # Store original action
        original_action = self.armature.animation_data.action if self.armature.animation_data else None

        # Assign the action
        if self.armature.animation_data is None:
            self.armature.animation_data_create()
        self.armature.animation_data.action = action

        # Get frame range
        frame_start = int(action.frame_range[0])
        frame_end = int(action.frame_range[1])
        total_frames = max(1, frame_end - frame_start)

        samples_before = len(self.samples)

        # Extract each frame
        for frame in range(frame_start, frame_end + 1):
            bpy.context.scene.frame_set(frame)
            bpy.context.view_layer.update()

            # Compute motion phase (0-1 within animation)
            motion_phase = (frame - frame_start) / total_frames

            sample = self._extract_frame(
                ground_height=ground_height,
                motion_phase=motion_phase,
                task_type=task_type,
            )

            if sample is not None:
                sample['action'] = action.name
                sample['frame'] = frame
                self.samples.append(sample)

        # Restore original action
        if original_action:
            self.armature.animation_data.action = original_action

        return len(self.samples) - samples_before

    def extract_from_all_actions(
        self,
        task_type: str = "locomotion",
        ground_height: float = 0.0,
    ) -> int:
        """
        Extract training samples from all actions in the blend file.

        Returns:
            Total number of samples extracted.
        """
        total = 0
        for action in bpy.data.actions:
            # Try to infer task type from action name
            action_task = task_type
            name_lower = action.name.lower()
            if "idle" in name_lower:
                action_task = "idle"
            elif "walk" in name_lower or "run" in name_lower:
                action_task = "locomotion"
            elif "grab" in name_lower or "reach" in name_lower:
                action_task = "reach"
            elif "crouch" in name_lower or "squat" in name_lower:
                action_task = "crouch"
            elif "jump" in name_lower:
                action_task = "jump"

            count = self.extract_from_action(action, action_task, ground_height)
            total += count
            print(f"[NeuralIK] Extracted {count} samples from '{action.name}' (task={action_task})")

        print(f"[NeuralIK] Total: {total} samples from {len(bpy.data.actions)} actions")
        return total

    def _extract_frame(
        self,
        ground_height: float = 0.0,
        motion_phase: float = 0.0,
        task_type: str = "idle",
    ) -> Optional[Dict]:
        """
        Extract input/output pair from current frame.

        Returns:
            Dict with 'input', 'output', 'effector_targets', or None if failed.
        """
        pose_bones = self.armature.pose.bones
        arm_matrix = self.armature.matrix_world

        # =====================================================================
        # INPUT: Context vector (50 dims)
        # =====================================================================
        try:
            input_data = self.context_extractor.extract_from_current_pose(
                ground_height=ground_height,
                motion_phase=motion_phase,
                task_type=task_type,
            )
        except Exception as e:
            print(f"[NeuralIK] Context extraction failed: {e}")
            return None

        if len(input_data) != INPUT_SIZE:
            print(f"[NeuralIK] Input size mismatch: {len(input_data)} != {INPUT_SIZE}")
            return None

        # =====================================================================
        # OUTPUT: Bone rotations as axis-angle (69 dims)
        # =====================================================================
        output_data = []

        for bone_name in CONTROLLED_BONES:
            bone = pose_bones.get(bone_name)
            if bone is None:
                output_data.extend([0.0, 0.0, 0.0])
                continue

            # Get bone's local rotation and convert to axis-angle
            if bone.rotation_mode == 'QUATERNION':
                axis_angle = quaternion_to_axis_angle(bone.rotation_quaternion)
            else:
                axis_angle = euler_to_axis_angle(bone.rotation_euler)

            output_data.extend(axis_angle.tolist())

        if len(output_data) != OUTPUT_SIZE:
            print(f"[NeuralIK] Output size mismatch: {len(output_data)} != {OUTPUT_SIZE}")
            return None

        # =====================================================================
        # EFFECTOR TARGETS: World positions AND rotations for FK loss
        # Positions: 15 dims (5 effectors × 3)
        # Rotations: 15 dims (5 effectors × 3 Euler XYZ)
        # =====================================================================
        effector_targets = []
        effector_rotations = []
        for effector_name in END_EFFECTORS:
            bone = pose_bones.get(effector_name)
            if bone is None:
                effector_targets.extend([0.0, 0.0, 0.0])
                effector_rotations.extend([0.0, 0.0, 0.0])
            else:
                # World position
                world_pos = arm_matrix @ bone.head
                effector_targets.extend([world_pos.x, world_pos.y, world_pos.z])
                # World rotation (Euler XYZ)
                bone_world_matrix = arm_matrix @ bone.matrix
                world_rot = bone_world_matrix.to_euler('XYZ')
                effector_rotations.extend([world_rot.x, world_rot.y, world_rot.z])

        # =====================================================================
        # ROOT INFO: For FK computation
        # =====================================================================
        # CRITICAL: Use armature's WORLD rotation, NOT hips bone matrix!
        # FK already applies REST_ORIENTATIONS internally for each bone.
        # Using hips bone matrix here would DOUBLE-APPLY the rest orientation.
        # The root_rotation in FK represents "how is the whole armature rotated in world space"
        hips = pose_bones.get("Hips")
        if hips:
            root_world_pos = arm_matrix @ hips.head
            # Extract armature's world rotation axes
            arm_rot = arm_matrix.to_3x3()
            root_forward = Vector((arm_rot[0][1], arm_rot[1][1], arm_rot[2][1])).normalized()  # Y column
            root_up = Vector((arm_rot[0][2], arm_rot[1][2], arm_rot[2][2])).normalized()  # Z column
        else:
            root_world_pos = Vector((0, 0, 1))
            root_forward = Vector((0, 1, 0))
            root_up = Vector((0, 0, 1))

        # =====================================================================
        # GROUND INFO: For contact loss
        # =====================================================================
        ground_heights = []
        contact_flags = []
        for foot_name in CONTACT_EFFECTORS:
            foot_bone = pose_bones.get(foot_name)
            if foot_bone:
                foot_z = (arm_matrix @ foot_bone.head).z
                ground_heights.append(ground_height)
                contact_flags.append(1.0 if foot_z - ground_height < 0.1 else 0.0)
            else:
                ground_heights.append(0.0)
                contact_flags.append(1.0)

        return {
            'input': np.array(input_data, dtype=np.float32),
            'output': np.array(output_data, dtype=np.float32),
            'effector_targets': np.array(effector_targets, dtype=np.float32),
            'effector_rotations': np.array(effector_rotations, dtype=np.float32),
            'root_position': np.array([root_world_pos.x, root_world_pos.y, root_world_pos.z], dtype=np.float32),
            'root_forward': np.array([root_forward.x, root_forward.y, root_forward.z], dtype=np.float32),
            'root_up': np.array([root_up.x, root_up.y, root_up.z], dtype=np.float32),
            'ground_heights': np.array(ground_heights, dtype=np.float32),
            'contact_flags': np.array(contact_flags, dtype=np.float32),
            'task_type': task_type,
            'motion_phase': motion_phase,
        }

    def get_dataset(self) -> Dict[str, np.ndarray]:
        """
        Get the full dataset as numpy arrays.

        Returns:
            Dict with:
                'inputs': (N, 50) - network inputs
                'outputs': (N, 69) - bone rotations
                'effector_targets': (N, 15) - effector world positions
                'effector_rotations': (N, 15) - effector world rotations (Euler XYZ)
                'root_positions': (N, 3) - root world positions
                'root_forwards': (N, 3) - root forward direction (for FK)
                'root_ups': (N, 3) - root up direction (for FK)
                'ground_heights': (N, 2) - ground height per foot
                'contact_flags': (N, 2) - contact state per foot
        """
        if not self.samples:
            return {
                'inputs': np.array([]),
                'outputs': np.array([]),
                'effector_targets': np.array([]),
                'effector_rotations': np.array([]),
                'root_positions': np.array([]),
                'root_forwards': np.array([]),
                'root_ups': np.array([]),
                'ground_heights': np.array([]),
                'contact_flags': np.array([]),
            }

        return {
            'inputs': np.stack([s['input'] for s in self.samples]),
            'outputs': np.stack([s['output'] for s in self.samples]),
            'effector_targets': np.stack([s['effector_targets'] for s in self.samples]),
            'effector_rotations': np.stack([s['effector_rotations'] for s in self.samples]),
            'root_positions': np.stack([s['root_position'] for s in self.samples]),
            'root_forwards': np.stack([s['root_forward'] for s in self.samples]),
            'root_ups': np.stack([s['root_up'] for s in self.samples]),
            'ground_heights': np.stack([s['ground_heights'] for s in self.samples]),
            'contact_flags': np.stack([s['contact_flags'] for s in self.samples]),
        }

    def get_train_test_split(
        self,
        augment: bool = True,
        augment_factor: int = 2,
    ) -> Dict[str, np.ndarray]:
        """
        Split dataset into training and test sets.

        Test set is NEVER used for training - only to verify generalization.

        Args:
            augment: Whether to augment training data
            augment_factor: How many augmented copies to add

        Returns:
            Dict with train_* and test_* arrays
        """
        dataset = self.get_dataset()

        if len(dataset['inputs']) == 0:
            return {
                'train_inputs': np.array([]),
                'train_outputs': np.array([]),
                'train_effector_targets': np.array([]),
                'train_effector_rotations': np.array([]),
                'train_root_positions': np.array([]),
                'train_root_forwards': np.array([]),
                'train_root_ups': np.array([]),
                'train_ground_heights': np.array([]),
                'train_contact_flags': np.array([]),
                'test_inputs': np.array([]),
                'test_outputs': np.array([]),
                'test_effector_targets': np.array([]),
                'test_effector_rotations': np.array([]),
                'test_root_positions': np.array([]),
                'test_root_forwards': np.array([]),
                'test_root_ups': np.array([]),
                'test_ground_heights': np.array([]),
                'test_contact_flags': np.array([]),
            }

        # Shuffle indices
        n = len(dataset['inputs'])
        indices = np.random.permutation(n)

        # Split point
        split_idx = int(n * TRAIN_SPLIT)

        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]

        result = {}

        # Split each array
        for key, arr in dataset.items():
            result[f'train_{key}'] = arr[train_idx]
            result[f'test_{key}'] = arr[test_idx]

        # Augment training data
        if augment and augment_factor > 1:
            augmented_inputs = [result['train_inputs']]
            augmented_outputs = [result['train_outputs']]

            for _ in range(augment_factor - 1):
                aug_inputs = augment_input(
                    result['train_inputs'],
                    noise_scale=0.02,
                    rotation_noise=0.03,
                )
                augmented_inputs.append(aug_inputs)
                # Outputs stay the same (we're adding input noise)
                augmented_outputs.append(result['train_outputs'])

            result['train_inputs'] = np.concatenate(augmented_inputs, axis=0)
            result['train_outputs'] = np.concatenate(augmented_outputs, axis=0)

            # Replicate other arrays to match
            for key in ['train_effector_targets', 'train_effector_rotations',
                       'train_root_positions', 'train_root_forwards', 'train_root_ups',
                       'train_ground_heights', 'train_contact_flags']:
                result[key] = np.tile(result[key], (augment_factor, 1))

        return result

    def clear(self):
        """Clear all extracted samples."""
        self.samples = []

    def save(self, path: str = None):
        """Save extracted data to disk."""
        import os
        if path is None:
            os.makedirs(DATA_DIR, exist_ok=True)
            path = os.path.join(DATA_DIR, "training_data.npz")

        dataset = self.get_dataset()
        np.savez_compressed(path, **dataset)
        print(f"[NeuralIK] Saved {len(self.samples)} samples to {path}")

    def load(self, path: str = None):
        """Load data from disk."""
        import os
        if path is None:
            path = os.path.join(DATA_DIR, "training_data.npz")

        if not os.path.exists(path):
            print(f"[NeuralIK] No saved data at {path}")
            return

        data = np.load(path)

        # Reconstruct samples from arrays
        n = len(data['inputs'])
        self.samples = []
        for i in range(n):
            self.samples.append({
                'input': data['inputs'][i],
                'output': data['outputs'][i],
                'effector_targets': data['effector_targets'][i],
                'root_position': data['root_positions'][i],
                'ground_heights': data['ground_heights'][i],
                'contact_flags': data['contact_flags'][i],
            })

        print(f"[NeuralIK] Loaded {n} samples from {path}")


# =============================================================================
# Module-level helpers
# =============================================================================

_extractor: Optional[AnimationDataExtractor] = None


def get_extractor(armature: bpy.types.Object = None) -> AnimationDataExtractor:
    """Get or create the global data extractor."""
    global _extractor

    if armature is not None:
        _extractor = AnimationDataExtractor(armature)
    elif _extractor is None:
        raise ValueError("No armature provided and no extractor exists")

    return _extractor


def extract_all(armature: bpy.types.Object) -> Tuple[int, int]:
    """Extract training data from all actions."""
    extractor = get_extractor(armature)
    extractor.clear()
    total = extractor.extract_from_all_actions()
    return total, len(bpy.data.actions)
