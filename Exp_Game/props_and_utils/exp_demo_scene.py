import bpy
import os
import time
from ...exp_preferences import get_addon_path
# UI text helpers (your module inside Exp_Game/reactions)
from ..reactions.exp_custom_ui import register_ui_draw, add_text_reaction, clear_all_text
from ..props_and_utils.exp_time import get_game_time

class EXPLORATORY_OT_AppendDemoScene(bpy.types.Operator):
    """Append the bundled demo scene and make it the active scene."""
    bl_idname  = "exploratory.append_demo_scene"
    bl_label   = "Append Demo World"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        # 1) Path to shipped .blend
        blend_path = os.path.join(
            get_addon_path(), "Exp_Game", "exp_assets", "Demo", "demo.blend"
        )
        if not os.path.isfile(blend_path):
            self.report({'ERROR'}, f"Demo .blend not found: {blend_path}")
            return {'CANCELLED'}

        # 2) Read available scenes and pick the first/only one
        try:
            with bpy.data.libraries.load(blend_path, link=False) as (df, _):
                scene_names = list(df.scenes or [])
        except Exception as e:
            self.report({'ERROR'}, f"Could not read scenes: {e}")
            return {'CANCELLED'}

        if not scene_names:
            self.report({'ERROR'}, "No scenes in demo .blend")
            return {'CANCELLED'}

        pick = scene_names[0]

        # 3) Append the scene
        before = set(bpy.data.scenes.keys())
        dir_path = os.path.join(blend_path, "Scene") + os.sep
        filepath = os.path.join(dir_path, pick)
        try:
            bpy.ops.wm.append(filepath=filepath, directory=dir_path, filename=pick)
        except Exception as e:
            self.report({'ERROR'}, f"Append failed: {e}")
            return {'CANCELLED'}

        # 4) Find the newly added scene & switch to it
        added = [s for s in bpy.data.scenes if s.name not in before]
        target = added[0] if added else bpy.data.scenes.get(pick)
        if not target:
            self.report({'ERROR'}, "Scene appended but not found")
            return {'CANCELLED'}

        try:
            context.window.scene = target
        except Exception as e:
            self.report({'WARNING'}, f"Appended '{target.name}', but couldn't switch: {e}")

        # 5) Flash short, left-aligned UI text for ~5 seconds (no wide lines)
        try:
            register_ui_draw()  # ensure draw handler is active

            base_now = get_game_time() or time.time()
            end_time = base_now + 10.0  # seconds

            # Top-left, small left margin; stack rows downward.
            # Keep lines short to avoid overlap with the N-panel.
            add_text_reaction(
                text_str="Welcome to the demo!",
                anchor='TOP_LEFT',
                margin_x=2,   # a bit in from the left edge
                margin_y=6,   # a bit down from the top
                scale=6,
                end_time=end_time,
                color=(1, 1, 1, 0.95),
            )
            add_text_reaction(
                text_str="To play:",
                anchor='TOP_LEFT',
                margin_x=2,
                margin_y=8,
                scale=5,
                end_time=end_time,
                color=(1, 1, 1, 0.9),
            )
            add_text_reaction(
                text_str="Open N-panel → 'Create' tab",
                anchor='TOP_LEFT',
                margin_x=2,
                margin_y=10,
                scale=4,
                end_time=end_time,
                color=(1, 1, 1, 0.9),
            )
            add_text_reaction(
                text_str="Click Play",
                anchor='TOP_LEFT',
                margin_x=2,
                margin_y=12,
                scale=4,
                end_time=end_time,
                color=(1, 1, 1, 0.9),
            )

            # Clear text after 5 seconds
            def _clear():
                try:
                    clear_all_text()
                    for area in bpy.context.screen.areas:
                        if area.type == 'VIEW_3D':
                            area.tag_redraw()
                except Exception:
                    pass
                return None

            bpy.app.timers.register(_clear, first_interval=5.0)

            # Immediate redraw nudge
            for area in bpy.context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()

        except Exception as e:
            # Non-fatal: scene append worked; UI text is optional
            print(f"[INFO] Demo appended; UI text display skipped due to: {e}")

        self.report({'INFO'}, f"Demo world ready: {target.name}")
        return {'FINISHED'}


# Registration
_classes = (EXPLORATORY_OT_AppendDemoScene,)

def register():
    from bpy.utils import register_class
    for cls in _classes:
        register_class(cls)

def unregister():
    from bpy.utils import unregister_class
    for cls in reversed(_classes):
        unregister_class(cls)
