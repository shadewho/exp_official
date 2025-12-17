# Exp_Game/animations/__init__.py
"""
Animations Module - Main thread animation system.

This module handles bpy writes for applying animations.
Worker-safe computation is in engine/animations/.

Components:
- AnimationController: Main orchestrator for all animation playback
- apply: Utility functions for applying transforms to bones/objects
- state_machine: Character locomotion state machine (IDLE, WALK, RUN, JUMP, FALL, LAND)
- test_panel: UI panel for testing animation playback
"""

from .controller import AnimationController, PlayingAnimation, ObjectAnimState
from . import apply

__all__ = [
    "AnimationController",
    "PlayingAnimation",
    "ObjectAnimState",
    "apply",
]
