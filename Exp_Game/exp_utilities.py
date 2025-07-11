#exp_utilities.py

import bpy


#---------------Game World---------------------------
def set_game_world(scene):
    """Store the name of the designated game world in a text datablock."""
    txt = bpy.data.texts.get("GAME_WORLD")
    if not txt:
        txt = bpy.data.texts.new("GAME_WORLD")
    txt.clear()
    txt.write(scene.name)
    print(f"Game world set to: {scene.name}")

def get_game_world():
    """Retrieve the game world scene from the text datablock."""
    txt = bpy.data.texts.get("GAME_WORLD")
    if txt:
        scene_name = txt.as_string().strip()
        return bpy.data.scenes.get(scene_name)
    return None

class EXPLORATORY_OT_SetGameWorld(bpy.types.Operator):
    """Set the current scene as the Game World for the add-on."""
    bl_idname = "exploratory.set_game_world"
    bl_label = "Set Game World"

    def execute(self, context):
        set_game_world(context.scene)
        self.report({'INFO'}, f"Game World set to {context.scene.name}")
        return {'FINISHED'}