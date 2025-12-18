# Exp_Game/engine/animations/cache.py
"""
AnimationCache - Storage and lookup for baked animations.

Worker-safe: no bpy references, can be pickled for multiprocessing.

Provides:
- Global animation storage by name
- Per-object animation binding
- Fast lookup for runtime playback

NUMPY OPTIMIZATION (2025-12):
  - BakedAnimation now uses numpy arrays for transforms
  - Serialization (to_dict/from_dict) handles numpy conversion
  - Worker reconstructs arrays lazily on first use
"""

from typing import Dict, List, Optional, Set
from .data import BakedAnimation


class AnimationCache:
    """
    Centralized cache for baked animations.

    Stores animations globally by name, and tracks which objects
    have which animations bound to them.

    Usage:
        cache = AnimationCache()
        cache.add(baked_anim)
        anim = cache.get("Walk")

        # Bind animations to objects
        cache.bind("Player", "Walk")
        cache.bind("Player", "Run")
        anims = cache.get_bound("Player")  # ["Walk", "Run"]
    """

    __slots__ = ('_animations', '_bindings')

    def __init__(self):
        # Global animation storage: {name: BakedAnimation}
        self._animations: Dict[str, BakedAnimation] = {}

        # Object bindings: {object_id: set of animation names}
        self._bindings: Dict[str, Set[str]] = {}

    # =========================================================================
    # ANIMATION STORAGE
    # =========================================================================

    def add(self, animation: BakedAnimation) -> None:
        """Add or replace an animation in the cache."""
        self._animations[animation.name] = animation

    def add_many(self, animations: List[BakedAnimation]) -> int:
        """Add multiple animations. Returns count added."""
        for anim in animations:
            self._animations[anim.name] = anim
        return len(animations)

    def get(self, name: str) -> Optional[BakedAnimation]:
        """Get animation by name, or None if not found."""
        return self._animations.get(name)

    def remove(self, name: str) -> bool:
        """Remove animation by name. Returns True if removed."""
        if name in self._animations:
            del self._animations[name]
            # Also remove from all bindings
            for obj_bindings in self._bindings.values():
                obj_bindings.discard(name)
            return True
        return False

    def has(self, name: str) -> bool:
        """Check if animation exists in cache."""
        return name in self._animations

    def clear(self) -> None:
        """Clear all animations and bindings."""
        self._animations.clear()
        self._bindings.clear()

    @property
    def names(self) -> List[str]:
        """List all animation names in cache."""
        return list(self._animations.keys())

    @property
    def count(self) -> int:
        """Number of animations in cache."""
        return len(self._animations)

    # =========================================================================
    # OBJECT BINDINGS
    # =========================================================================

    def bind(self, object_id: str, animation_name: str) -> bool:
        """
        Bind an animation to an object.

        Args:
            object_id: Unique identifier for the object (e.g., object.name)
            animation_name: Name of animation to bind

        Returns:
            True if animation exists and was bound, False otherwise
        """
        if animation_name not in self._animations:
            return False

        if object_id not in self._bindings:
            self._bindings[object_id] = set()

        self._bindings[object_id].add(animation_name)
        return True

    def unbind(self, object_id: str, animation_name: str) -> bool:
        """Unbind an animation from an object."""
        if object_id in self._bindings:
            self._bindings[object_id].discard(animation_name)
            return True
        return False

    def unbind_all(self, object_id: str) -> None:
        """Remove all animation bindings for an object."""
        if object_id in self._bindings:
            del self._bindings[object_id]

    def get_bound(self, object_id: str) -> List[str]:
        """Get list of animation names bound to an object."""
        if object_id in self._bindings:
            return list(self._bindings[object_id])
        return []

    def get_bound_animations(self, object_id: str) -> List[BakedAnimation]:
        """Get actual BakedAnimation objects bound to an object."""
        result = []
        for name in self.get_bound(object_id):
            anim = self._animations.get(name)
            if anim:
                result.append(anim)
        return result

    def is_bound(self, object_id: str, animation_name: str) -> bool:
        """Check if animation is bound to object."""
        if object_id in self._bindings:
            return animation_name in self._bindings[object_id]
        return False

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_dict(self) -> dict:
        """Convert cache to plain dict for serialization."""
        return {
            "animations": {
                name: anim.to_dict()
                for name, anim in self._animations.items()
            },
            "bindings": {
                obj_id: list(names)
                for obj_id, names in self._bindings.items()
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AnimationCache":
        """Reconstruct cache from plain dict."""
        cache = cls()

        # Restore animations
        for name, anim_data in data.get("animations", {}).items():
            cache._animations[name] = BakedAnimation.from_dict(anim_data)

        # Restore bindings
        for obj_id, names in data.get("bindings", {}).items():
            cache._bindings[obj_id] = set(names)

        return cache

    def __repr__(self) -> str:
        bound_count = sum(len(b) for b in self._bindings.values())
        return f"AnimationCache({self.count} animations, {len(self._bindings)} objects, {bound_count} bindings)"
