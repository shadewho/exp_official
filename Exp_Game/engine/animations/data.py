# Exp_Game/engine/animations/data.py
"""
BakedAnimation - Numpy-based animation data for maximum performance.

Supports both:
- Bone transforms (for armatures)
- Object transforms (for any object: mesh, empty, etc.)

Worker-safe: contains no Blender references, can be pickled.

Transform format (10 floats):
  (quat_w, quat_x, quat_y, quat_z, loc_x, loc_y, loc_z, scale_x, scale_y, scale_z)

NUMPY OPTIMIZATION:
  - Transforms stored as contiguous numpy arrays
  - Shape: (num_frames, num_bones, 10) for bones
  - Shape: (num_frames, 10) for object transforms
  - Enables vectorized blending (30-100x faster than Python loops)
  - Includes animated_mask to skip static bones
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


# Type alias: 10-float transform tuple (for external interface)
Transform = Tuple[float, float, float, float,   # quat wxyz
                  float, float, float,           # loc xyz
                  float, float, float]           # scale xyz

# Identity transform as numpy array
IDENTITY_TRANSFORM = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)


class BakedAnimation:
    """
    Pre-baked animation data using numpy arrays for performance.

    Attributes:
        name: Animation name (from Blender Action)
        duration: Total duration in seconds
        fps: Frames per second
        frame_count: Total number of frames

        # Bone data (for armatures)
        bone_names: List of bone names in order
        bone_index: Dict mapping bone name -> array index
        bone_transforms: np.ndarray shape (num_frames, num_bones, 10)
        animated_mask: np.ndarray bool (num_bones,) - True if bone animates

        # Object data (for object-level animation)
        object_transforms: np.ndarray shape (num_frames, 10) or None

        looping: Whether this animation loops by default
    """

    __slots__ = (
        'name', 'duration', 'fps', 'frame_count',
        'bone_names', 'bone_index', 'bone_transforms', 'animated_mask',
        'object_transforms', 'looping'
    )

    def __init__(
        self,
        name: str,
        duration: float,
        fps: float,
        bone_names: Optional[List[str]] = None,
        bone_transforms: Optional[np.ndarray] = None,
        animated_mask: Optional[np.ndarray] = None,
        object_transforms: Optional[np.ndarray] = None,
        looping: bool = True
    ):
        self.name = name
        self.duration = duration
        self.fps = fps
        self.frame_count = int(duration * fps) + 1

        # Bone data
        self.bone_names = bone_names or []
        self.bone_index = {name: i for i, name in enumerate(self.bone_names)}

        # Bone transforms: (num_frames, num_bones, 10)
        if bone_transforms is not None:
            self.bone_transforms = np.asarray(bone_transforms, dtype=np.float32)
        else:
            self.bone_transforms = np.empty((0, 0, 10), dtype=np.float32)

        # Animated mask: which bones actually move
        if animated_mask is not None:
            self.animated_mask = np.asarray(animated_mask, dtype=bool)
        elif len(self.bone_names) > 0:
            # Default: assume all bones animate (will be refined by baker)
            self.animated_mask = np.ones(len(self.bone_names), dtype=bool)
        else:
            self.animated_mask = np.empty(0, dtype=bool)

        # Object transforms: (num_frames, 10)
        if object_transforms is not None:
            self.object_transforms = np.asarray(object_transforms, dtype=np.float32)
        else:
            self.object_transforms = None

        self.looping = looping

    @property
    def num_bones(self) -> int:
        """Number of bones in this animation."""
        return len(self.bone_names)

    @property
    def num_animated_bones(self) -> int:
        """Number of bones that actually animate (non-static)."""
        return int(np.sum(self.animated_mask)) if len(self.animated_mask) > 0 else 0

    @property
    def has_bones(self) -> bool:
        """True if animation has bone data."""
        return len(self.bone_names) > 0

    @property
    def has_object(self) -> bool:
        """True if animation has object-level transforms."""
        return self.object_transforms is not None and len(self.object_transforms) > 0

    def get_bone_index(self, bone_name: str) -> int:
        """Get array index for a bone name. Returns -1 if not found."""
        return self.bone_index.get(bone_name, -1)

    def sample(self, time: float) -> np.ndarray:
        """
        Sample bone transforms at a specific time.

        Args:
            time: Time in seconds

        Returns:
            np.ndarray of shape (num_bones, 10) with interpolated transforms
        """
        if not self.has_bones or self.bone_transforms.size == 0:
            return np.empty((0, 10), dtype=np.float32)

        # Handle looping
        if self.looping and self.duration > 0:
            time = time % self.duration

        # Clamp to valid range
        time = max(0.0, min(time, self.duration))

        # Convert to frame
        frame_float = time * self.fps
        frame_int = int(frame_float)
        frac = frame_float - frame_int

        # Clamp frame to valid range
        max_frame = self.bone_transforms.shape[0] - 1
        frame_int = min(frame_int, max_frame)

        if frac < 0.001 or frame_int >= max_frame:
            # No interpolation needed
            return self.bone_transforms[frame_int].copy()

        # Interpolate between frames
        frame_a = self.bone_transforms[frame_int]
        frame_b = self.bone_transforms[frame_int + 1]

        # Simple lerp (quaternion lerp + normalize would be more correct)
        result = frame_a * (1.0 - frac) + frame_b * frac

        # Normalize quaternions (first 4 components)
        quat_norms = np.linalg.norm(result[:, 0:4], axis=1, keepdims=True)
        result[:, 0:4] /= (quat_norms + 1e-10)

        return result

    def to_dict(self) -> dict:
        """
        Convert to plain dict for serialization/pickling to worker.
        Numpy arrays are converted to lists for pickle compatibility.
        """
        return {
            "name": self.name,
            "duration": self.duration,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "bone_names": self.bone_names,
            "bone_transforms": self.bone_transforms.tolist() if self.bone_transforms.size > 0 else [],
            "animated_mask": self.animated_mask.tolist() if self.animated_mask.size > 0 else [],
            "object_transforms": self.object_transforms.tolist() if self.object_transforms is not None else None,
            "looping": self.looping,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BakedAnimation":
        """Reconstruct from plain dict (after unpickling in worker)."""
        bone_transforms = data.get("bone_transforms", [])
        if bone_transforms:
            bone_transforms = np.array(bone_transforms, dtype=np.float32)
        else:
            bone_transforms = None

        animated_mask = data.get("animated_mask", [])
        if animated_mask:
            animated_mask = np.array(animated_mask, dtype=bool)
        else:
            animated_mask = None

        object_transforms = data.get("object_transforms")
        if object_transforms is not None:
            object_transforms = np.array(object_transforms, dtype=np.float32)

        return cls(
            name=data["name"],
            duration=data["duration"],
            fps=data["fps"],
            bone_names=data.get("bone_names", []),
            bone_transforms=bone_transforms,
            animated_mask=animated_mask,
            object_transforms=object_transforms,
            looping=data.get("looping", True)
        )

    def __repr__(self) -> str:
        parts = [f"'{self.name}'", f"{self.duration:.2f}s", f"{self.fps}fps"]
        if self.has_bones:
            animated = self.num_animated_bones
            total = self.num_bones
            if animated < total:
                parts.append(f"{animated}/{total} bones animated")
            else:
                parts.append(f"{total} bones")
        if self.has_object:
            parts.append("object")
        return f"BakedAnimation({', '.join(parts)})"
