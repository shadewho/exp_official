# Exp_Game/engine/animations/data.py
"""
BakedAnimation - Unified animation data for any object.

Supports both:
- Bone transforms (for armatures)
- Object transforms (for any object: mesh, empty, etc.)

Worker-safe: contains no Blender references, can be pickled.

Transform format (10 floats):
  (quat_w, quat_x, quat_y, quat_z, loc_x, loc_y, loc_z, scale_x, scale_y, scale_z)
"""

from typing import Dict, List, Tuple, Optional


# Type alias: 10-float transform tuple
Transform = Tuple[float, float, float, float,   # quat wxyz
                  float, float, float,           # loc xyz
                  float, float, float]           # scale xyz


class BakedAnimation:
    """
    Pre-baked animation data, safe for worker processes.

    Attributes:
        name: Animation name (from Blender Action)
        duration: Total duration in seconds
        fps: Frames per second
        frame_count: Total number of frames
        bones: Dict[bone_name, List[Transform]] - for armature bones
        object_transforms: List[Transform] - for object-level animation
        looping: Whether this animation loops by default
    """

    __slots__ = ('name', 'duration', 'fps', 'frame_count', 'bones', 'object_transforms', 'looping')

    def __init__(
        self,
        name: str,
        duration: float,
        fps: float,
        bones: Optional[Dict[str, List[Transform]]] = None,
        object_transforms: Optional[List[Transform]] = None,
        looping: bool = True
    ):
        self.name = name
        self.duration = duration
        self.fps = fps
        self.frame_count = int(duration * fps) + 1
        self.bones = bones or {}
        self.object_transforms = object_transforms or []
        self.looping = looping

    @property
    def has_bones(self) -> bool:
        """True if animation has bone data."""
        return bool(self.bones)

    @property
    def has_object(self) -> bool:
        """True if animation has object-level transforms."""
        return bool(self.object_transforms)

    def to_dict(self) -> dict:
        """Convert to plain dict for serialization."""
        return {
            "name": self.name,
            "duration": self.duration,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "bones": self.bones,
            "object_transforms": self.object_transforms,
            "looping": self.looping,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BakedAnimation":
        """Reconstruct from plain dict."""
        anim = cls.__new__(cls)
        anim.name = data["name"]
        anim.duration = data["duration"]
        anim.fps = data["fps"]
        anim.frame_count = data["frame_count"]
        anim.bones = data.get("bones", {})
        anim.object_transforms = data.get("object_transforms", [])
        anim.looping = data.get("looping", True)
        return anim

    def __repr__(self) -> str:
        parts = [f"'{self.name}'", f"{self.duration:.2f}s", f"{self.fps}fps"]
        if self.bones:
            parts.append(f"{len(self.bones)} bones")
        if self.object_transforms:
            parts.append("object")
        return f"BakedAnimation({', '.join(parts)})"
