# Exp_Game/animations/state_machine.py
"""
Character Animation State Machine

Determines which locomotion animation to play based on:
- Input keys (WASD, shift, space)
- Physics state (grounded, velocity)

States: IDLE, WALK, RUN, JUMP, FALL, LAND
"""

import bpy
import enum
from typing import Set, Optional, Tuple


class AnimState(enum.Enum):
    """Character locomotion states."""
    IDLE = 0
    WALK = 1
    RUN = 2
    JUMP = 3
    FALL = 4
    LAND = 5


def get_user_keymap() -> dict:
    """Get user-configured movement keys from preferences."""
    prefs = bpy.context.preferences.addons["Exploratory"].preferences
    return {
        'forward': prefs.key_forward,
        'backward': prefs.key_backward,
        'left': prefs.key_left,
        'right': prefs.key_right,
        'jump': prefs.key_jump,
        'run': prefs.key_run,
    }


class CharacterStateMachine:
    """
    Locomotion state machine for the player character.

    Tracks current state and determines transitions based on input and physics.
    Does NOT handle animation playback - just state logic.
    """

    def __init__(self):
        self.state = AnimState.IDLE

        # Jump/fall tracking
        self.air_time = 0.0
        self.fall_timer = 0.0
        self.min_fall_time = 0.9       # Time before JUMP → FALL
        self.min_fall_for_land = 0.20  # Min fall time to trigger LAND

        # State flags
        self.landing_in_progress = False
        self.jump_played_in_air = False
        self.jump_key_held = False

        # For one-shot animations
        self.one_shot_playing = False
        self.one_shot_start_time = 0.0
        self.one_shot_duration = 0.0

    def update(
        self,
        keys_pressed: Set[str],
        delta_time: float,
        is_grounded: bool,
        vertical_velocity: float = 0.0,
        game_time: float = 0.0
    ) -> Tuple[AnimState, bool]:
        """
        Update state machine and return current state.

        Args:
            keys_pressed: Set of currently pressed key names
            delta_time: Time since last update
            is_grounded: Whether character is on ground
            vertical_velocity: Current vertical velocity
            game_time: Current game time (for one-shot timing)

        Returns:
            (current_state, state_changed)
        """
        old_state = self.state

        # Check if one-shot animation finished
        if self.one_shot_playing:
            if game_time - self.one_shot_start_time >= self.one_shot_duration:
                self.one_shot_playing = False
                self.landing_in_progress = False

        # Update state based on input and physics
        new_state = self._evaluate_state(keys_pressed, delta_time, is_grounded, vertical_velocity)
        self.state = new_state

        return new_state, (new_state != old_state)

    def _evaluate_state(
        self,
        keys_pressed: Set[str],
        delta_time: float,
        is_grounded: bool,
        vertical_velocity: float
    ) -> AnimState:
        """Evaluate and return the appropriate state."""
        k = get_user_keymap()

        # Track jump key for press detection (not hold)
        jump_is_down = (k['jump'] in keys_pressed)
        just_pressed_jump = (jump_is_down and not self.jump_key_held)
        self.jump_key_held = jump_is_down

        # If landing animation is playing, wait for it to finish
        if self.landing_in_progress and self.one_shot_playing:
            return AnimState.LAND

        # Grounded logic
        if is_grounded:
            self.jump_played_in_air = False

            # Coming from FALL → check if we should LAND
            if self.state == AnimState.FALL:
                if self.fall_timer >= self.min_fall_for_land:
                    self.landing_in_progress = True
                    self.air_time = 0.0
                    self.fall_timer = 0.0
                    return AnimState.LAND
                else:
                    # Short fall, go straight to movement
                    self.air_time = 0.0
                    self.fall_timer = 0.0
                    return self._get_movement_state(keys_pressed, just_pressed_jump, is_grounded)

            # Coming from JUMP → go to movement
            if self.state == AnimState.JUMP:
                self.air_time = 0.0
                self.fall_timer = 0.0
                return self._get_movement_state(keys_pressed, just_pressed_jump, is_grounded)

            # Normal grounded movement
            self.air_time = 0.0
            self.fall_timer = 0.0
            return self._get_movement_state(keys_pressed, just_pressed_jump, is_grounded)

        # Airborne logic
        else:
            self.air_time += delta_time

            if self.state == AnimState.JUMP:
                # Been jumping long enough → transition to FALL
                if self.air_time >= self.min_fall_time:
                    self.fall_timer = 0.0
                    return AnimState.FALL
                return AnimState.JUMP

            elif self.state == AnimState.FALL:
                self.fall_timer += delta_time
                return AnimState.FALL

            else:
                # Walked off edge or other airborne entry
                if self.air_time >= self.min_fall_time:
                    self.fall_timer = 0.0
                    return AnimState.FALL
                return self.state

    def _get_movement_state(
        self,
        keys_pressed: Set[str],
        just_pressed_jump: bool,
        is_grounded: bool
    ) -> AnimState:
        """Determine movement state from input."""
        k = get_user_keymap()

        # Jump takes priority
        if just_pressed_jump and is_grounded and not self.jump_played_in_air:
            self.jump_played_in_air = True
            return AnimState.JUMP

        # Check movement keys
        move_keys = {k['forward'], k['backward'], k['left'], k['right']}
        pressing_move = bool(keys_pressed.intersection(move_keys))
        run_held = (k['run'] in keys_pressed)

        if pressing_move and run_held:
            return AnimState.RUN
        elif pressing_move:
            return AnimState.WALK
        else:
            return AnimState.IDLE

    def start_one_shot(self, duration: float, game_time: float):
        """Mark a one-shot animation as playing (blocks state changes)."""
        self.one_shot_playing = True
        self.one_shot_start_time = game_time
        self.one_shot_duration = duration

    def get_action_name(self) -> Optional[str]:
        """
        Get the action name for the current state from scene settings.

        Returns:
            Action name or None if not configured
        """
        scene = bpy.context.scene
        char = scene.character_actions

        state_to_action = {
            AnimState.IDLE: char.idle_action,
            AnimState.WALK: char.walk_action,
            AnimState.RUN: char.run_action,
            AnimState.JUMP: char.jump_action,
            AnimState.FALL: char.fall_action,
            AnimState.LAND: char.land_action,
        }

        action = state_to_action.get(self.state)
        return action.name if action else None

    def get_state_properties(self) -> dict:
        """
        Get properties for the current state animation.

        Returns:
            Dict with: loop, speed, is_one_shot
        """
        scene = bpy.context.scene
        char = scene.character_actions

        # One-shot states (play once, don't loop)
        one_shot_states = {AnimState.JUMP, AnimState.LAND}

        action = None
        if self.state == AnimState.IDLE:
            action = char.idle_action
        elif self.state == AnimState.WALK:
            action = char.walk_action
        elif self.state == AnimState.RUN:
            action = char.run_action
        elif self.state == AnimState.JUMP:
            action = char.jump_action
        elif self.state == AnimState.FALL:
            action = char.fall_action
        elif self.state == AnimState.LAND:
            action = char.land_action

        speed = 1.0
        if action and hasattr(action, 'action_speed'):
            speed = action.action_speed

        return {
            'loop': self.state not in one_shot_states,
            'speed': speed,
            'is_one_shot': self.state in one_shot_states,
        }
