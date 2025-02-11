# File: exp_ui.py
import bpy
# --------------------------------------------------------------------
# Exploratory Modal Panel
# --------------------------------------------------------------------
class ExploratoryPanel(bpy.types.Panel):
    bl_label = "Exploratory Modal Panel"
    bl_idname = "VIEW3D_PT_exploratory_modal"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Exploratory"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # Draw the mode toggle at the top:
        layout.prop(scene, "main_category", expand=True)
        layout.separator()
        layout.separator()       
        # Create the operator button and set the property:

    # # Show only when main_category is set to "CREATE"
    # @classmethod
    # def poll(cls, context):
    #     return context.scene.main_category == 'CREATE'
    
        op = layout.operator("view3d.exp_modal", text="Start Exploratory Modal")
        op.launched_from_ui = False  # or True, as desired

