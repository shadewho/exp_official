# Exp_Game/developer/dev_panel.py
"""
Developer Tools UI Panel

Provides a clean interface for toggling debug output categories.

UNIFIED PHYSICS: Static and dynamic meshes use identical physics code.
All physics logs show source (static/dynamic) - there is ONE system.
"""

import bpy

# Animation 2.0 imports
from ..animations.test_panel import (
    get_test_controller,
    reset_test_controller,
    playback_update,
)


def _is_create_panel_enabled(scene, key: str) -> bool:
    """Check if a specific Create panel is enabled in the filter."""
    flags = getattr(scene, "create_panels_filter", None)
    if flags is None:
        return True
    if hasattr(flags, "__len__") and len(flags) == 0:
        return False
    return (key in flags)


class DEV_PT_DeveloperTools(bpy.types.Panel):
    """Developer Tools panel in Create tab."""
    bl_label = "Developer Tools"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Exploratory"
    bl_idname = "DEV_PT_developer_tools"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return (context.scene.main_category == 'CREATE'
                and _is_create_panel_enabled(context.scene, 'DEV'))

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # ═══════════════════════════════════════════════════════════════
        # Dev Refresh Button
        # ═══════════════════════════════════════════════════════════════
        row = layout.row()
        row.scale_y = 1.5
        row.operator("exp.dev_refresh", text="Refresh Addon", icon='FILE_REFRESH')

        layout.separator()

        # ═══════════════════════════════════════════════════════════════
        # Master Frequency Control
        # ═══════════════════════════════════════════════════════════════
        box = layout.box()
        box.label(text="Master Frequency Control", icon='TIME')
        col = box.column(align=True)
        col.prop(scene, "dev_debug_master_hz", text="Output Frequency (Hz)")
        col.separator()
        col.prop(scene, "dev_export_session_log", text="Export Diagnostics Log")

        layout.separator()

        # ═══════════════════════════════════════════════════════════════
        # Engine Health
        # ═══════════════════════════════════════════════════════════════
        box = layout.box()
        box.label(text="Engine Health", icon='MEMORY')
        col = box.column(align=True)
        col.prop(scene, "dev_startup_logs", text="Startup Logs")
        col.prop(scene, "dev_debug_engine", text="Engine Diagnostics")

        col.separator()
        col.label(text="Manual Stress Tests:", icon='PLAY')
        row = col.row(align=True)
        row.operator("exp_engine.quick_test", text="Quick Test", icon='CHECKMARK')
        row.operator("exp_engine.stress_test", text="Full Stress Test", icon='COMMUNITY')

        layout.separator()

        # ═══════════════════════════════════════════════════════════════
        # Spatial Grid
        # ═══════════════════════════════════════════════════════════════
        box = layout.box()
        box.label(text="Spatial Grid", icon='MESH_GRID')
        col = box.column(align=True)
        col.prop(scene, "dev_debug_grid", text="Grid Diagnostics")

        layout.separator()

        # ═══════════════════════════════════════════════════════════════
        # Offload Systems
        # ═══════════════════════════════════════════════════════════════
        box = layout.box()
        box.label(text="Offload Systems", icon='FORCE_CHARGE')
        col = box.column(align=True)
        col.prop(scene, "dev_debug_kcc_physics", text="KCC Physics")
        col.prop(scene, "dev_debug_frame_numbers", text="Frame Numbers")
        col.prop(scene, "dev_debug_culling", text="Performance Culling")

        layout.separator()

        # ═══════════════════════════════════════════════════════════════
        # UNIFIED PHYSICS (Single section - static + dynamic identical)
        # ═══════════════════════════════════════════════════════════════
        box = layout.box()
        box.label(text="Unified Physics", icon='PHYSICS')

        col = box.column(align=True)
        col.prop(scene, "dev_debug_physics", text="Physics Summary")

        col.separator()
        col.label(text="Granular (shows source: static/dynamic):", icon='ALIGN_JUSTIFY')

        col.prop(scene, "dev_debug_physics_ground", text="Ground Detection")
        col.prop(scene, "dev_debug_physics_horizontal", text="Horizontal Collision")
        col.prop(scene, "dev_debug_physics_body", text="Body Integrity")
        col.prop(scene, "dev_debug_physics_ceiling", text="Ceiling Check")
        col.prop(scene, "dev_debug_physics_step", text="Step-Up")
        col.prop(scene, "dev_debug_physics_slide", text="Wall Slide")
        col.prop(scene, "dev_debug_physics_slopes", text="Slopes")

        layout.separator()

        # ═══════════════════════════════════════════════════════════════
        # Unified Camera (Static + Dynamic use same raycast)
        # ═══════════════════════════════════════════════════════════════
        box = layout.box()
        box.label(text="Unified Camera", icon='VIEW_CAMERA')
        col = box.column(align=True)
        col.prop(scene, "dev_debug_camera", text="Camera Raycast")

        layout.separator()

        # ═══════════════════════════════════════════════════════════════
        # Dynamic Mesh System (Unified with static)
        # ═══════════════════════════════════════════════════════════════
        box = layout.box()
        box.label(text="Dynamic Mesh System", icon='MESH_DATA')
        col = box.column(align=True)
        col.prop(scene, "dev_debug_dynamic_cache", text="Cache & Transforms")
        col.prop(scene, "dev_debug_dynamic_opt", text="Optimization Stats")

        layout.separator()

        # ═══════════════════════════════════════════════════════════════
        # KCC Visual Debug (3D Viewport)
        # ═══════════════════════════════════════════════════════════════
        box = layout.box()
        box.label(text="KCC Visual Debug (3D Viewport)", icon='SHADING_WIRE')
        col = box.column(align=True)

        row = col.row(align=True)
        row.prop(scene, "dev_debug_kcc_visual", text="Enable Visual Debug")

        if scene.dev_debug_kcc_visual:
            col.separator(factor=0.5)
            sub = col.column(align=True)
            sub.prop(scene, "dev_debug_kcc_visual_capsule", text="Capsule Shape")
            sub.prop(scene, "dev_debug_kcc_visual_normals", text="Hit Normals")
            sub.prop(scene, "dev_debug_kcc_visual_ground", text="Ground Ray")
            sub.prop(scene, "dev_debug_kcc_visual_movement", text="Movement Vectors")

            col.separator(factor=0.5)
            col.prop(scene, "dev_debug_kcc_visual_line_width", text="Line Width")
            col.prop(scene, "dev_debug_kcc_visual_vector_scale", text="Vector Scale")

        layout.separator()

        # ═══════════════════════════════════════════════════════════════
        # Game Systems
        # ═══════════════════════════════════════════════════════════════
        box = layout.box()
        box.label(text="Game Systems", icon='PLAY')
        col = box.column(align=True)
        col.prop(scene, "dev_debug_interactions", text="Interactions")
        col.prop(scene, "dev_debug_audio", text="Audio")
        col.prop(scene, "dev_debug_trackers", text="Trackers")

        layout.separator()

        # ═══════════════════════════════════════════════════════════════
        # Animation 2.0 (Test Tools)
        # ═══════════════════════════════════════════════════════════════
        box = layout.box()
        box.label(text="Animation 2.0", icon='ACTION')

        # ─── Debug Toggles ──────────────────────────────────────────────
        row = box.row(align=True)
        row.prop(scene, "dev_debug_animations", text="Animation Logs")
        row.prop(scene, "dev_debug_anim_cache", text="Cache Logs")

        row = box.row(align=True)
        row.prop(scene, "dev_debug_anim_worker", text="Animation Worker")

        props = scene.anim2_test
        ctrl = get_test_controller()
        obj = context.active_object

        # ─── Cache Status ───────────────────────────────────────────────
        row = box.row()
        row.label(text=f"Cache: {ctrl.cache.count} animations", icon='FILE_CACHE')
        row.operator("anim2.clear_cache", text="", icon='X')

        # ─── Bake ───────────────────────────────────────────────────────
        sub_box = box.box()
        sub_box.label(text="Bake", icon='IMPORT')

        armature = scene.target_armature
        if armature:
            sub_box.operator("anim2.bake_all", text=f"Bake All ({len(bpy.data.actions)} actions)", icon='ACTION')
        else:
            sub_box.label(text="Set target armature first", icon='ERROR')

        # ─── Playback ───────────────────────────────────────────────────
        sub_box = box.box()
        sub_box.label(text="Playback", icon='PLAY')

        if ctrl.cache.count > 0:
            sub_box.prop(props, "selected_animation", text="")

            row = sub_box.row(align=True)
            row.prop(props, "play_speed")
            row.prop(props, "fade_time")

            row = sub_box.row(align=True)
            row.prop(props, "loop_playback")
            row.prop(props, "playback_timeout")

            row = sub_box.row(align=True)
            row.scale_y = 1.3
            row.operator("anim2.play_animation", icon='PLAY')
            row.operator("anim2.stop_animation", icon='SNAP_FACE')

            # Stop All button (separate row)
            row = sub_box.row(align=True)
            row.scale_y = 1.2
            row.operator("anim2.stop_all", text="Stop All", icon='CANCEL')
        else:
            sub_box.label(text="Bake animations first", icon='INFO')

        # ─── Blending ───────────────────────────────────────────────────
        sub_box = box.box()
        sub_box.label(text="Blend Test", icon='MOD_VERTEX_WEIGHT')

        if ctrl.cache.count > 1:
            sub_box.prop(props, "blend_animation", text="")
            sub_box.prop(props, "blend_weight", slider=True)
            sub_box.operator("anim2.blend_animation", icon='OVERLAY')
        elif ctrl.cache.count == 1:
            sub_box.label(text="Need 2+ animations to blend", icon='INFO')
        else:
            sub_box.label(text="Bake animations first", icon='INFO')

        # ─── Status ─────────────────────────────────────────────────────
        if obj and obj.name in ctrl._states:
            state = ctrl._states[obj.name]
            playing = [p for p in state.playing if not p.finished]
            if playing:
                sub_box = box.box()
                sub_box.label(text="Now Playing:", icon='TIME')
                for p in playing:
                    row = sub_box.row()
                    row.label(text=f"  {p.animation_name}")
                    row.label(text=f"{p.weight * p.fade_progress:.0%}")

        # ─── Runtime IK (Gameplay) ─────────────────────────────────────
        sub_box = box.box()
        row = sub_box.row()
        row.label(text="Runtime IK", icon='CON_KINEMATIC')
        row.prop(scene, "runtime_ik_enabled", text="", icon='PLAY' if scene.runtime_ik_enabled else 'PAUSE')

        if scene.runtime_ik_enabled:
            col = sub_box.column(align=True)
            col.prop(scene, "runtime_ik_chain", text="")
            col.prop(scene, "runtime_ik_target", text="Target")
            col.prop(scene, "runtime_ik_influence", slider=True)

            # Show target position if set
            if scene.runtime_ik_target:
                loc = scene.runtime_ik_target.location
                col.label(text=f"Pos: ({loc.x:.2f}, {loc.y:.2f}, {loc.z:.2f})")

            # IK debug logs toggle
            row = sub_box.row()
            row.prop(scene, "dev_debug_runtime_ik", text="IK Logs")

        layout.separator()

        # ═══════════════════════════════════════════════════════════════
        # IK Test
        # ═══════════════════════════════════════════════════════════════
        box = layout.box()
        row = box.row()
        row.label(text="IK Test", icon='CON_KINEMATIC')
        row.prop(props, "ik_advanced_mode", text="Advanced", toggle=True)

        # Armature picker (no need to select the armature)
        box.prop(props, "ik_armature", text="Armature")

        ik_arm = props.ik_armature
        if ik_arm and ik_arm.type == 'ARMATURE':
            # Chain selector
            box.prop(props, "ik_chain", text="")

            # ─── Target Controls ───────────────────────────────────────
            sub = box.box()
            sub.label(text="Target", icon='EMPTY_AXIS')

            # Target object picker
            sub.prop(props, "ik_target_object", text="")

            if props.ik_target_object is not None:
                # Show target object's position
                loc = props.ik_target_object.matrix_world.translation
                sub.label(text=f"Pos: ({loc.x:.2f}, {loc.y:.2f}, {loc.z:.2f})")
            elif props.ik_advanced_mode:
                # Full XYZ control
                col = sub.column(align=True)
                col.prop(props, "ik_target", index=0, text="X")
                col.prop(props, "ik_target", index=1, text="Y")
                col.prop(props, "ik_target", index=2, text="Z")
            else:
                # Simple mode - legacy sliders
                if props.ik_chain.startswith("leg"):
                    sub.prop(props, "ik_target_z", slider=True)
                else:
                    sub.prop(props, "ik_arm_forward", slider=True)

            # ─── Pole Vector Controls ──────────────────────────────────
            sub = box.box()
            sub.label(text="Pole Vector", icon='ORIENTATION_NORMAL')
            row = sub.row(align=True)
            row.prop(props, "ik_pole_direction", text="")
            row.prop(props, "ik_pole_offset", text="Dist")

            # ─── Action Buttons ────────────────────────────────────────
            row = box.row(align=True)
            row.scale_y = 1.3
            row.operator("anim2.test_ik", text="Apply IK", icon='CON_KINEMATIC')
            row.operator("anim2.reset_pose", text="Reset", icon='LOOP_BACK')

            # ─── Visual Debug ──────────────────────────────────────────
            sub = box.box()
            row = sub.row()
            row.label(text="Visual Debug", icon='SHADING_WIRE')
            row.prop(scene, "dev_debug_ik_visual", text="", icon='HIDE_OFF' if scene.dev_debug_ik_visual else 'HIDE_ON')

            if scene.dev_debug_ik_visual:
                col = sub.column(align=True)
                col.prop(scene, "dev_debug_ik_visual_targets", text="Targets")
                col.prop(scene, "dev_debug_ik_visual_chains", text="Chains")
                col.prop(scene, "dev_debug_ik_visual_reach", text="Reach")
                col.prop(scene, "dev_debug_ik_visual_poles", text="Poles")
                col.prop(scene, "dev_debug_ik_visual_joints", text="Joints")
                sub.prop(scene, "dev_debug_ik_line_width", text="Line Width")

        else:
            box.label(text="Set an armature above", icon='INFO')


def register():
    bpy.utils.register_class(DEV_PT_DeveloperTools)


def unregister():
    bpy.utils.unregister_class(DEV_PT_DeveloperTools)