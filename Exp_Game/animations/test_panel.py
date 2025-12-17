# Exp_Game/animations/test_panel.py
"""
Animation 2.0 Test Operators & Properties.

Operators and properties for testing the unified animation system.
UI is now in Developer Tools panel (dev_panel.py).
"""

import bpy
from bpy.types import Operator, PropertyGroup
from bpy.props import FloatProperty, BoolProperty, EnumProperty

from ..engine.animations.baker import bake_action
from ..engine.animations.data import BakedAnimation
from .controller import AnimationController


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL TEST CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════════

# Single controller instance for testing
_test_controller = None
_playback_timer = None


def get_test_controller() -> AnimationController:
    """Get or create the test controller."""
    global _test_controller
    if _test_controller is None:
        _test_controller = AnimationController()
    return _test_controller


def reset_test_controller():
    """Reset the test controller."""
    global _test_controller, _playback_timer
    if _playback_timer is not None:
        bpy.context.window_manager.event_timer_remove(_playback_timer)
        _playback_timer = None
    _test_controller = None


# ═══════════════════════════════════════════════════════════════════════════════
# OPERATORS
# ═══════════════════════════════════════════════════════════════════════════════

class ANIM2_OT_BakeAll(Operator):
    """Bake ALL actions in the blend file"""
    bl_idname = "anim2.bake_all"
    bl_label = "Bake All Actions"
    bl_options = {'REGISTER'}

    def execute(self, context):
        import time
        start_time = time.perf_counter()

        scene = context.scene
        armature = scene.target_armature

        if armature is None:
            self.report({'WARNING'}, "No target armature set in scene")
            return {'CANCELLED'}

        if armature.type != 'ARMATURE':
            self.report({'WARNING'}, "Target armature is not an armature object")
            return {'CANCELLED'}

        # Clear existing cache for fresh bake
        reset_test_controller()
        ctrl = get_test_controller()

        # Bake ALL actions in the file
        baked_count = 0
        failed = []

        for action in bpy.data.actions:
            try:
                anim = bake_action(action, armature)
                ctrl.add_animation(anim)
                baked_count += 1
            except Exception as e:
                failed.append(f"{action.name}: {e}")

        elapsed = (time.perf_counter() - start_time) * 1000

        if failed:
            self.report({'WARNING'}, f"Baked {baked_count} actions ({elapsed:.0f}ms). {len(failed)} failed.")
        else:
            self.report({'INFO'}, f"Baked {baked_count} actions in {elapsed:.0f}ms")

        return {'FINISHED'}


class ANIM2_OT_PlayAnimation(Operator):
    """Play an animation on the selected object"""
    bl_idname = "anim2.play_animation"
    bl_label = "Play"
    bl_options = {'REGISTER'}

    def execute(self, context):
        obj = context.active_object
        if obj is None:
            self.report({'WARNING'}, "No object selected")
            return {'CANCELLED'}

        props = context.scene.anim2_test
        anim_name = props.selected_animation

        if not anim_name:
            self.report({'WARNING'}, "No animation selected")
            return {'CANCELLED'}

        ctrl = get_test_controller()

        if not ctrl.has_animation(anim_name):
            self.report({'WARNING'}, f"Animation '{anim_name}' not in cache. Bake first.")
            return {'CANCELLED'}

        # Play with settings
        success = ctrl.play(
            obj.name,
            anim_name,
            weight=1.0,
            speed=props.play_speed,
            looping=props.loop_playback,
            fade_in=props.fade_time,
            replace=True
        )

        if success:
            self.report({'INFO'}, f"Playing '{anim_name}' on {obj.name}")
            # Start playback timer if not running
            if not bpy.app.timers.is_registered(playback_update):
                bpy.app.timers.register(playback_update, first_interval=1/60)
        else:
            self.report({'WARNING'}, f"Failed to play '{anim_name}'")

        return {'FINISHED'}


class ANIM2_OT_StopAnimation(Operator):
    """Stop all animations on the selected object"""
    bl_idname = "anim2.stop_animation"
    bl_label = "Stop"
    bl_options = {'REGISTER'}

    def execute(self, context):
        obj = context.active_object
        if obj is None:
            self.report({'WARNING'}, "No object selected")
            return {'CANCELLED'}

        props = context.scene.anim2_test
        ctrl = get_test_controller()
        ctrl.stop(obj.name, fade_out=props.fade_time)

        self.report({'INFO'}, f"Stopped animations on {obj.name}")
        return {'FINISHED'}


class ANIM2_OT_BlendAnimation(Operator):
    """Blend in a second animation"""
    bl_idname = "anim2.blend_animation"
    bl_label = "Blend In"
    bl_options = {'REGISTER'}

    def execute(self, context):
        obj = context.active_object
        if obj is None:
            self.report({'WARNING'}, "No object selected")
            return {'CANCELLED'}

        props = context.scene.anim2_test
        anim_name = props.blend_animation

        if not anim_name:
            self.report({'WARNING'}, "No blend animation selected")
            return {'CANCELLED'}

        ctrl = get_test_controller()

        if not ctrl.has_animation(anim_name):
            self.report({'WARNING'}, f"Animation '{anim_name}' not in cache")
            return {'CANCELLED'}

        # Add without replacing (blends with existing)
        success = ctrl.play(
            obj.name,
            anim_name,
            weight=props.blend_weight,
            speed=props.play_speed,
            looping=props.loop_playback,
            fade_in=props.fade_time,
            replace=False
        )

        if success:
            self.report({'INFO'}, f"Blending '{anim_name}' at {props.blend_weight:.0%}")
            if not bpy.app.timers.is_registered(playback_update):
                bpy.app.timers.register(playback_update, first_interval=1/60)

        return {'FINISHED'}


class ANIM2_OT_ClearCache(Operator):
    """Clear all cached animations"""
    bl_idname = "anim2.clear_cache"
    bl_label = "Clear Cache"
    bl_options = {'REGISTER'}

    def execute(self, context):
        reset_test_controller()
        self.report({'INFO'}, "Animation cache cleared")
        return {'FINISHED'}


# ═══════════════════════════════════════════════════════════════════════════════
# PLAYBACK TIMER
# ═══════════════════════════════════════════════════════════════════════════════

_last_time = None

def playback_update():
    """Timer callback for animation playback."""
    global _last_time
    import time

    ctrl = get_test_controller()

    # Calculate delta time
    current = time.perf_counter()
    if _last_time is None:
        _last_time = current
        dt = 1/60
    else:
        dt = current - _last_time
        _last_time = current

    # Update all animations
    ctrl.update(dt)

    # Force viewport redraw
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()

    # Check if anything is still playing
    has_playing = False
    for state in ctrl._states.values():
        if any(not p.finished for p in state.playing):
            has_playing = True
            break

    if has_playing:
        return 1/60  # Continue at 60fps
    else:
        _last_time = None
        return None  # Stop timer


# ═══════════════════════════════════════════════════════════════════════════════
# PROPERTIES
# ═══════════════════════════════════════════════════════════════════════════════

def get_animation_items(self, context):
    """Get list of cached animations for enum."""
    ctrl = get_test_controller()
    items = [("", "Select Animation", "")]
    for name in ctrl.cache.names:
        items.append((name, name, ""))
    return items


class ANIM2_TestProperties(PropertyGroup):
    """Properties for animation 2.0 testing."""

    selected_animation: EnumProperty(
        name="Animation",
        description="Animation to play",
        items=get_animation_items
    )

    blend_animation: EnumProperty(
        name="Blend",
        description="Animation to blend in",
        items=get_animation_items
    )

    blend_weight: FloatProperty(
        name="Blend Weight",
        description="Weight of blended animation",
        default=0.5,
        min=0.0,
        max=1.0
    )

    play_speed: FloatProperty(
        name="Speed",
        description="Playback speed multiplier",
        default=1.0,
        min=0.1,
        max=3.0
    )

    fade_time: FloatProperty(
        name="Fade",
        description="Fade in/out time in seconds",
        default=0.2,
        min=0.0,
        max=2.0
    )

    loop_playback: BoolProperty(
        name="Loop",
        description="Loop the animation",
        default=True
    )




# ═══════════════════════════════════════════════════════════════════════════════
# REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

classes = [
    ANIM2_TestProperties,
    ANIM2_OT_BakeAll,
    ANIM2_OT_PlayAnimation,
    ANIM2_OT_StopAnimation,
    ANIM2_OT_BlendAnimation,
    ANIM2_OT_ClearCache,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.anim2_test = bpy.props.PointerProperty(type=ANIM2_TestProperties)


def unregister():
    # Stop any running timer
    if bpy.app.timers.is_registered(playback_update):
        bpy.app.timers.unregister(playback_update)

    reset_test_controller()

    if hasattr(bpy.types.Scene, 'anim2_test'):
        del bpy.types.Scene.anim2_test

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
