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
            col.prop(scene, "dev_debug_ragdoll", text="Ragdoll")
            col.prop(scene, "dev_debug_health", text="Health")

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
                col.label(text="Display:", icon='PREFERENCES')
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

            # ─── Debug Toggles (cleaned up) ────────────────────────────────
            col = box.column(align=True)
            col.prop(scene, "dev_debug_animations", text="Animation Logs")
            col.prop(scene, "dev_debug_anim_cache", text="Cache Logs")
            col.prop(scene, "dev_debug_anim_worker", text="Anim Worker Logs")
            col.prop(scene, "dev_debug_pose_blend", text="Pose Blend Logs")
            col.prop(scene, "dev_debug_layers", text="Layer Logs")
            col.prop(scene, "dev_debug_neural_ik", text="Neural IK Logs")

            props = scene.anim2_test
            ctrl = get_test_controller()

            # ═══════════════════════════════════════════════════════════════
            # NEURAL IK SECTION
            # ═══════════════════════════════════════════════════════════════
            neural_box = box.box()
            neural_box.label(text="Neural IK", icon='RNA')

            armature = scene.target_armature
            if not armature:
                neural_box.label(text="Set target armature first", icon='ERROR')
            else:
                # Status display
                from ..animations.test_panel import get_neural_status
                status = get_neural_status()

                # Current state summary
                state_box = neural_box.box()
                if status['weights_exist']:
                    if status['best_loss'] is not None:
                        loss_val = status['best_loss']
                        if loss_val < 0.10:
                            state_box.label(text=f"Weights: GOOD (loss {loss_val:.4f})", icon='CHECKMARK')
                        else:
                            state_box.label(text=f"Weights: NEEDS WORK (loss {loss_val:.4f})", icon='ERROR')
                    else:
                        state_box.label(text="Weights: Loaded", icon='FILE_TICK')
                else:
                    state_box.label(text="Weights: Not trained yet", icon='BLANK1')

                if status['samples'] > 0:
                    state_box.label(text=f"Data: {status['samples']} samples loaded", icon='OUTLINER_DATA_MESH')
                else:
                    state_box.label(text="Data: None (extract or load first)", icon='BLANK1')

                neural_box.separator()

                # ─────────────────────────────────────────────────────────────
                # QUICK START: USE EXISTING DATA (most common)
                # ─────────────────────────────────────────────────────────────
                quick_box = neural_box.box()
                row = quick_box.row()
                row.label(text="USE EXISTING DATA", icon='CHECKMARK')
                row.label(text="(already trained)")
                col = quick_box.column(align=True)
                col.scale_y = 1.2
                col.operator("neural.load_data", text="1. Load Saved Data", icon='FILE_FOLDER')
                col.operator("neural.reload_weights", text="2. Load Weights", icon='FILE_REFRESH')
                col.operator("neural.test", text="3. Run Tests", icon='VIEWZOOM')
                col.scale_y = 0.7
                col.label(text="Use this if you already ran training before")
                col.scale_y = 1.0

                neural_box.separator()

                # ─────────────────────────────────────────────────────────────
                # CREATE NEW DATA (only when animations change)
                # ─────────────────────────────────────────────────────────────
                new_box = neural_box.box()
                row = new_box.row()
                row.prop(scene, "dev_neural_show_create",
                         icon='TRIA_DOWN' if getattr(scene, 'dev_neural_show_create', False) else 'TRIA_RIGHT',
                         icon_only=True, emboss=False)
                row.label(text="CREATE NEW DATA", icon='ADD')
                row.label(text="(re-extract)")

                if getattr(scene, 'dev_neural_show_create', False):
                    col = new_box.column(align=True)
                    col.scale_y = 0.7
                    col.label(text="Only needed if animations changed:")
                    col.scale_y = 1.0
                    col.separator()
                    col.operator("neural.extract_data", text=f"1. Extract ({len(bpy.data.actions)} actions)", icon='ACTION')
                    col.operator("neural.save_data", text="2. Save to Disk", icon='FILE_TICK')
                    col.separator()
                    col.scale_y = 0.7
                    col.label(text="Then run in PowerShell:")
                    col.label(text="  cd .../neural_network")
                    col.label(text="  python torch_trainer.py")
                    col.scale_y = 1.0
                    col.separator()
                    col.operator("neural.train", text="Show Full Path", icon='INFO')

                # ─────────────────────────────────────────────────────────────
                # Advanced (collapsed)
                # ─────────────────────────────────────────────────────────────
                adv = neural_box.box()
                row = adv.row()
                row.prop(scene, "dev_neural_show_advanced",
                         icon='TRIA_DOWN' if getattr(scene, 'dev_neural_show_advanced', False) else 'TRIA_RIGHT',
                         icon_only=True, emboss=False)
                row.label(text="Advanced", icon='PREFERENCES')

                if getattr(scene, 'dev_neural_show_advanced', False):
                    col = adv.column(align=True)
                    col.operator("neural.reset", text="Reset Weights (Start Over)", icon='TRASH')

                # ─────────────────────────────────────────────────────────────
                # Diagnostics (collapsed)
                # ─────────────────────────────────────────────────────────────
                diag = neural_box.box()
                row = diag.row()
                row.prop(scene, "dev_neural_show_diagnostics",
                         icon='TRIA_DOWN' if getattr(scene, 'dev_neural_show_diagnostics', False) else 'TRIA_RIGHT',
                         icon_only=True, emboss=False)
                row.label(text="Diagnostics", icon='VIEWZOOM')

                if getattr(scene, 'dev_neural_show_diagnostics', False):
                    col = diag.column(align=True)
                    col.scale_y = 0.7
                    col.label(text="Verify pipeline is working correctly:")
                    col.scale_y = 1.0
                    col.separator()
                    col.operator("neural.run_diagnostics", text="Run All Diagnostics", icon='CHECKMARK')

                    # Show last results if available
                    from ..animations.test_panel import get_diagnostic_results
                    results = get_diagnostic_results()
                    if results['last_run']:
                        col.separator()
                        col.scale_y = 0.7
                        col.label(text=f"Last run: {results['last_run']}")
                        col.scale_y = 1.0

                        results_box = diag.box()
                        results_box.scale_y = 0.8

                        def status_icon(ok):
                            return 'CHECKMARK' if ok else 'ERROR'

                        results_box.label(text=f"Rest Positions", icon=status_icon(results['rest_pos_ok']))
                        results_box.label(text=f"Bone Lengths", icon=status_icon(results['bone_len_ok']))
                        results_box.label(text=f"FK Computation", icon=status_icon(results['fk_ok']))
                        results_box.label(text=f"Root Rotation", icon=status_icon(results['root_rot_ok']))
                        results_box.label(text=f"Data Extraction", icon=status_icon(results['extraction_ok']))

                # ─────────────────────────────────────────────────────────────
                # Verification (collapsed) - Test trained model accuracy
                # ─────────────────────────────────────────────────────────────
                verify = neural_box.box()
                row = verify.row()
                row.prop(scene, "dev_neural_show_verification",
                         icon='TRIA_DOWN' if getattr(scene, 'dev_neural_show_verification', False) else 'TRIA_RIGHT',
                         icon_only=True, emboss=False)
                row.label(text="Verification", icon='OUTLINER_OB_ARMATURE')

                if getattr(scene, 'dev_neural_show_verification', False):
                    col = verify.column(align=True)

                    # Pipeline validation (FIRST - do this before anything else)
                    col.scale_y = 0.7
                    col.label(text="1. Validate FK pipeline (do first!):")
                    col.scale_y = 1.0
                    col.operator("neural.validate_pipeline", text="Validate FK Pipeline", icon='VIEWZOOM')
                    col.separator()

                    # Model verification
                    col.scale_y = 0.7
                    col.label(text="2. Test trained model accuracy:")
                    col.scale_y = 1.0
                    col.operator("neural.verify_model", text="Verify Trained Model", icon='CHECKMARK')
                    col.separator()

                    # Visual inspection
                    col.scale_y = 0.7
                    col.label(text="3. Visual inspection:")
                    col.scale_y = 1.0
                    row = col.row(align=True)
                    row.operator("neural.apply_test_pose", text="Prediction", icon='POSE_HLT')
                    row.operator("neural.apply_ground_truth", text="Ground Truth", icon='ARMATURE_DATA')
                    col.operator("neural.compare_visual", text="Toggle Compare", icon='FILE_REFRESH')

                    # Show verification results if available
                    from ..animations.test_panel import get_verification_results
                    vresults = get_verification_results()
                    if vresults.get('last_run'):
                        col.separator()
                        col.scale_y = 0.7
                        col.label(text=f"Last: {vresults['last_run']}")
                        col.scale_y = 1.0

                        summary = vresults.get('summary', {})
                        if summary:
                            vbox = verify.box()
                            vbox.scale_y = 0.8
                            grade = summary.get('grade', '?')
                            rmse = summary.get('position_rmse_cm', 0)
                            rot = summary.get('rotation_error_deg', 0)
                            passed = summary.get('passed', 0)
                            total = summary.get('total', 0)

                            grade_icon = 'CHECKMARK' if grade in ['EXCELLENT', 'GOOD'] else 'ERROR'
                            vbox.label(text=f"Grade: {grade}", icon=grade_icon)
                            vbox.label(text=f"Position: {rmse:.1f}cm RMSE")
                            vbox.label(text=f"Rotation: {rot:.1f}°")
                            vbox.label(text=f"Tests: {passed}/{total} passed")

            # ═══════════════════════════════════════════════════════════════
            # ANIMATION PLAYBACK
            # ═══════════════════════════════════════════════════════════════
            sub_box = box.box()
            sub_box.label(text="Animation Playback", icon='PLAY')

            # Cache status
            row = sub_box.row()
            row.label(text=f"Cache: {ctrl.cache.count} animations", icon='FILE_CACHE')
            row.operator("anim2.clear_cache", text="", icon='X')

            # Bake button
            armature = scene.target_armature
            if armature:
                sub_box.operator("anim2.bake_all", text=f"Bake All ({len(bpy.data.actions)} actions)", icon='ACTION')

            # Animation options
            if ctrl.cache.count > 0:
                options_box = sub_box.box()
                options_box.prop(props, "selected_animation", text="")
                row = options_box.row(align=True)
                row.prop(props, "play_speed")
                row = options_box.row(align=True)
                row.prop(props, "loop_playback")
                row.prop(props, "playback_timeout")

                # Play/Stop buttons
                row = sub_box.row(align=True)
                row.scale_y = 1.3
                row.operator("anim2.test_play", text="Play", icon='PLAY')
                row.operator("anim2.test_stop", text="Stop", icon='SNAP_FACE')

                # Reset pose
                sub_box.operator("anim2.reset_pose", text="Reset Pose", icon='LOOP_BACK')

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
