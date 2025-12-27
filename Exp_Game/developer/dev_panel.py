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
    is_test_modal_active,
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
        # Engine Health (Collapsible)
        # ═══════════════════════════════════════════════════════════════
        box = layout.box()
        row = box.row()
        row.prop(scene, "dev_section_engine",
                 icon='TRIA_DOWN' if scene.dev_section_engine else 'TRIA_RIGHT',
                 icon_only=True, emboss=False)
        row.label(text="Engine Health", icon='MEMORY')

        if scene.dev_section_engine:
            col = box.column(align=True)
            col.prop(scene, "dev_startup_logs", text="Startup Logs")
            col.prop(scene, "dev_debug_engine", text="Engine Diagnostics")
            col.separator()
            col.label(text="Manual Stress Tests:", icon='PLAY')
            row = col.row(align=True)
            row.operator("exp_engine.quick_test", text="Quick Test", icon='CHECKMARK')
            row.operator("exp_engine.stress_test", text="Full Stress Test", icon='COMMUNITY')

        # ═══════════════════════════════════════════════════════════════
        # Spatial Grid (Collapsible)
        # ═══════════════════════════════════════════════════════════════
        box = layout.box()
        row = box.row()
        row.prop(scene, "dev_section_grid",
                 icon='TRIA_DOWN' if scene.dev_section_grid else 'TRIA_RIGHT',
                 icon_only=True, emboss=False)
        row.label(text="Spatial Grid", icon='MESH_GRID')

        if scene.dev_section_grid:
            col = box.column(align=True)
            col.prop(scene, "dev_debug_grid", text="Grid Diagnostics")

        # ═══════════════════════════════════════════════════════════════
        # Offload Systems (Collapsible)
        # ═══════════════════════════════════════════════════════════════
        box = layout.box()
        row = box.row()
        row.prop(scene, "dev_section_offload",
                 icon='TRIA_DOWN' if scene.dev_section_offload else 'TRIA_RIGHT',
                 icon_only=True, emboss=False)
        row.label(text="Offload Systems", icon='FORCE_CHARGE')

        if scene.dev_section_offload:
            col = box.column(align=True)
            col.prop(scene, "dev_debug_kcc_physics", text="KCC Physics")
            col.prop(scene, "dev_debug_frame_numbers", text="Frame Numbers")
            col.prop(scene, "dev_debug_culling", text="Performance Culling")

        # ═══════════════════════════════════════════════════════════════
        # UNIFIED PHYSICS (Collapsible)
        # ═══════════════════════════════════════════════════════════════
        box = layout.box()
        row = box.row()
        row.prop(scene, "dev_section_physics",
                 icon='TRIA_DOWN' if scene.dev_section_physics else 'TRIA_RIGHT',
                 icon_only=True, emboss=False)
        row.label(text="Unified Physics", icon='PHYSICS')

        if scene.dev_section_physics:
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

        # ═══════════════════════════════════════════════════════════════
        # Unified Camera (Collapsible)
        # ═══════════════════════════════════════════════════════════════
        box = layout.box()
        row = box.row()
        row.prop(scene, "dev_section_camera",
                 icon='TRIA_DOWN' if scene.dev_section_camera else 'TRIA_RIGHT',
                 icon_only=True, emboss=False)
        row.label(text="Unified Camera", icon='VIEW_CAMERA')

        if scene.dev_section_camera:
            col = box.column(align=True)
            col.prop(scene, "dev_debug_camera", text="Camera Raycast")

        # ═══════════════════════════════════════════════════════════════
        # Dynamic Mesh System (Collapsible)
        # ═══════════════════════════════════════════════════════════════
        box = layout.box()
        row = box.row()
        row.prop(scene, "dev_section_dynamic",
                 icon='TRIA_DOWN' if scene.dev_section_dynamic else 'TRIA_RIGHT',
                 icon_only=True, emboss=False)
        row.label(text="Dynamic Mesh System", icon='MESH_DATA')

        if scene.dev_section_dynamic:
            col = box.column(align=True)
            col.prop(scene, "dev_debug_dynamic_cache", text="Cache & Transforms")
            col.prop(scene, "dev_debug_dynamic_opt", text="Optimization Stats")

        # ═══════════════════════════════════════════════════════════════
        # KCC Visual Debug (Collapsible)
        # ═══════════════════════════════════════════════════════════════
        box = layout.box()
        row = box.row()
        row.prop(scene, "dev_section_kcc_visual",
                 icon='TRIA_DOWN' if scene.dev_section_kcc_visual else 'TRIA_RIGHT',
                 icon_only=True, emboss=False)
        row.label(text="KCC Visual Debug", icon='SHADING_WIRE')

        if scene.dev_section_kcc_visual:
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

        # ═══════════════════════════════════════════════════════════════
        # Game Systems (Collapsible)
        # ═══════════════════════════════════════════════════════════════
        box = layout.box()
        row = box.row()
        row.prop(scene, "dev_section_game",
                 icon='TRIA_DOWN' if scene.dev_section_game else 'TRIA_RIGHT',
                 icon_only=True, emboss=False)
        row.label(text="Game Systems", icon='PLAY')

        if scene.dev_section_game:
            col = box.column(align=True)
            col.prop(scene, "dev_debug_interactions", text="Interactions")
            col.prop(scene, "dev_debug_audio", text="Audio")
            col.prop(scene, "dev_debug_trackers", text="Trackers")
            col.prop(scene, "dev_debug_world_state", text="World State Filter")
            col.prop(scene, "dev_debug_aabb_cache", text="AABB Cache")
            col.prop(scene, "dev_debug_projectiles", text="Projectiles")
            col.prop(scene, "dev_debug_hitscans", text="Hitscans")
            col.prop(scene, "dev_debug_transforms", text="Transforms")
            col.prop(scene, "dev_debug_tracking", text="Tracking (Track To)")

        # ═══════════════════════════════════════════════════════════════
        # Animation 2.0 (Collapsible)
        # ═══════════════════════════════════════════════════════════════
        box = layout.box()
        row = box.row()
        row.prop(scene, "dev_section_anim",
                 icon='TRIA_DOWN' if scene.dev_section_anim else 'TRIA_RIGHT',
                 icon_only=True, emboss=False)
        row.label(text="Animation 2.0", icon='ACTION')

        if scene.dev_section_anim:
            # ─── Rig Visualizer (3D Viewport) ─────────────────────────────
            sub_box = box.box()
            row = sub_box.row()
            row.label(text="Rig Visualizer", icon='ARMATURE_DATA')
            row.prop(scene, "dev_rig_visualizer_enabled", text="", icon='HIDE_OFF' if scene.dev_rig_visualizer_enabled else 'HIDE_ON')

            if scene.dev_rig_visualizer_enabled:
                col = sub_box.column(align=True)
                row = col.row(align=True)
                row.prop(scene, "dev_rig_vis_bone_groups", text="Bone Groups")
                if scene.dev_rig_vis_bone_groups:
                    col.prop(scene, "dev_rig_vis_selected_group", text="")
                col.separator(factor=0.5)
                col.label(text="IK Display:", icon='CON_KINEMATIC')
                sub = col.column(align=True)
                sub.prop(scene, "dev_rig_vis_ik_chains", text="Chains")
                sub.prop(scene, "dev_rig_vis_ik_targets", text="Targets")
                sub.prop(scene, "dev_rig_vis_ik_poles", text="Poles")
                sub.prop(scene, "dev_rig_vis_ik_reach", text="Reach Spheres")
                col.separator(factor=0.5)
                col.label(text="Advanced:", icon='PREFERENCES')
                sub = col.column(align=True)
                sub.prop(scene, "dev_rig_vis_bone_axes", text="Bone Axes (X/Y/Z)")
                sub.prop(scene, "dev_rig_vis_active_mask", text="Active Blend Mask")
                col.separator(factor=0.5)
                col.label(text="Text Overlay:", icon='SMALL_CAPS')
                sub = col.column(align=True)
                sub.prop(scene, "dev_rig_vis_text_overlay", text="Show State Text")
                if scene.dev_rig_vis_text_overlay:
                    row = sub.row(align=True)
                    row.prop(scene, "dev_rig_vis_text_size", text="Size")
                    row.prop(scene, "dev_rig_vis_text_background", text="", icon='TEXTURE')
                col.separator(factor=0.5)
                col.prop(scene, "dev_rig_vis_line_width", text="Line Width")
                if scene.dev_rig_vis_bone_axes:
                    col.prop(scene, "dev_rig_vis_axis_length", text="Axis Length")
                from .rig_visualizer import get_visualizer_state
                vis_state = get_visualizer_state()
                if vis_state["ik_chains"]:
                    col.separator(factor=0.5)
                    col.label(text=f"Active IK: {', '.join(vis_state['ik_chains'])}", icon='CHECKMARK')
                if vis_state["layers"]["additive"] or vis_state["layers"]["override"]:
                    layers_info = []
                    if vis_state["layers"]["additive"]:
                        layers_info.append(f"{len(vis_state['layers']['additive'])} add")
                    if vis_state["layers"]["override"]:
                        layers_info.append(f"{len(vis_state['layers']['override'])} ovr")
                    col.label(text=f"Layers: {', '.join(layers_info)}", icon='RENDERLAYERS')

            # ─── Debug Toggles ─────────────────────────────────────────────
            col = box.column(align=True)
            col.prop(scene, "dev_debug_animations", text="Animation Logs")
            col.prop(scene, "dev_debug_anim_cache", text="Cache Logs")
            col.prop(scene, "dev_debug_anim_worker", text="Worker Logs")
            col.prop(scene, "dev_debug_runtime_ik", text="IK Logs")
            col.prop(scene, "dev_debug_ik_solve", text="IK Solve Details")
            col.prop(scene, "dev_debug_rig_state", text="Rig State (verbose)")
            col.prop(scene, "dev_debug_pose_blend", text="Pose Blend Logs")

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

            # ═══════════════════════════════════════════════════════════════
            # UNIFIED TEST SUITE
            # ═══════════════════════════════════════════════════════════════
            sub_box = box.box()
            sub_box.label(text="Animation Test", icon='EXPERIMENTAL')

            # Armature (from Character panel)
            armature = scene.target_armature
            if armature:
                row = sub_box.row()
                row.label(text=f"Armature: {armature.name}", icon='ARMATURE_DATA')
            else:
                sub_box.label(text="Set target armature in Character panel", icon='ERROR')

            # Rig Probe buttons
            row = sub_box.row(align=True)
            row.operator("anim2.probe_rig", text="Probe Rig", icon='BONE_DATA')
            row.operator("anim2.dump_orientations", text="Dump Axes", icon='ORIENTATION_LOCAL')

            # ─── IK State Test ──────────────────────────────────────────────
            ik_box = sub_box.box()
            ik_box.label(text="IK State Analysis", icon='CON_KINEMATIC')
            col = ik_box.column(align=True)
            col.prop(scene, "ik_test_chain", text="Chain")
            col.prop(scene, "ik_test_target", text="Target")
            row = col.row(align=True)
            row.scale_y = 1.2
            row.operator("anim2.test_ik_state", text="Analyze", icon='VIEWZOOM')
            row.operator("anim2.apply_ik", text="Apply IK", icon='CON_KINEMATIC')
            col.separator(factor=0.5)
            col.prop(scene, "dev_debug_ik_visual", text="GPU Visualization")

            # Animation options
            options_box = sub_box.box()

            if ctrl.cache.count > 0:
                options_box.prop(props, "selected_animation", text="")
                row = options_box.row(align=True)
                row.prop(props, "play_speed")
                row = options_box.row(align=True)
                row.prop(props, "loop_playback")
                row.prop(props, "playback_timeout")
            else:
                options_box.label(text="Bake animations first", icon='INFO')

            # Play/Stop buttons
            row = sub_box.row(align=True)
            row.scale_y = 1.4
            row.operator("anim2.test_play", text="Play", icon='PLAY')
            row.operator("anim2.test_stop", text="Stop", icon='SNAP_FACE')
            row.operator("anim2.clear_cache", text="Clear", icon='LOOP_BACK')

            # Status
            if armature and armature.name in ctrl._states:
                state = ctrl._states[armature.name]
                playing = [p for p in state.playing if not p.finished]
                if playing:
                    status_box = sub_box.box()
                    status_box.label(text="Now Playing:", icon='TIME')
                    for p in playing:
                        row = status_box.row()
                        row.label(text=f"  {p.animation_name}")
                        row.label(text=f"{p.weight * p.fade_progress:.0%}")


def register():
    bpy.utils.register_class(DEV_PT_DeveloperTools)


def unregister():
    bpy.utils.unregister_class(DEV_PT_DeveloperTools)