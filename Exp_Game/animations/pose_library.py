# Exp_Game/animations/pose_library.py
"""
Pose Library - Scene-based pose storage and management.

Allows users to capture armature poses and save them to the scene.
Poses are stored as self-contained JSON data (no Action pointers).

Usage:
    # Capture current pose from target armature
    bpy.ops.exploratory.capture_pose(pose_name="GrabSword")

    # Preview pose in editor
    bpy.ops.exploratory.apply_pose(index=0)

    # Access pose library
    for pose in scene.pose_library:
        print(pose.name, len(json.loads(pose.bone_data_json)))
"""

import bpy
import json
import time
from bpy.types import PropertyGroup, Operator, UIList
from bpy.props import (
    StringProperty,
    FloatProperty,
    IntProperty,
    CollectionProperty,
)
from mathutils import Quaternion, Vector

from .bone_groups import BONE_INDEX, TOTAL_BONES
from ..developer.dev_logger import log_game


# =============================================================================
# PROPERTY GROUPS
# =============================================================================

class PoseEntry(PropertyGroup):
    """
    A single pose in the pose library.

    Stores bone transforms as JSON for self-contained storage.
    Format: {"BoneName": [qw, qx, qy, qz, lx, ly, lz, sx, sy, sz], ...}
    """
    name: StringProperty(
        name="Pose Name",
        default="Pose",
        description="Name of the saved pose"
    )

    description: StringProperty(
        name="Description",
        default="",
        description="Optional description of the pose"
    )

    bone_data_json: StringProperty(
        name="Bone Data (JSON)",
        default="{}",
        description="Serialized bone transform data"
    )

    source_armature_name: StringProperty(
        name="Source Armature",
        default="",
        description="Name of the armature this pose was captured from"
    )

    created_timestamp: FloatProperty(
        name="Created",
        default=0.0,
        description="Unix timestamp when pose was created"
    )

    def get_bone_count(self) -> int:
        """Get number of bones stored in this pose."""
        try:
            data = json.loads(self.bone_data_json)
            return len(data)
        except Exception:
            return 0


# =============================================================================
# UI LIST
# =============================================================================

class EXPLORATORY_UL_PoseLibraryList(UIList):
    """UIList for displaying poses in the pose library."""

    bl_idname = "EXPLORATORY_UL_pose_library_list"

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            row.prop(item, "name", text="", emboss=False, icon='ARMATURE_DATA')
            # Show bone count
            bone_count = item.get_bone_count()
            row.label(text=f"({bone_count})")
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text=item.name, icon='ARMATURE_DATA')


# =============================================================================
# OPERATORS
# =============================================================================

class EXPLORATORY_OT_CapturePose(Operator):
    """Capture the current pose from the target armature"""
    bl_idname = "exploratory.capture_pose"
    bl_label = "Capture Pose"
    bl_options = {'REGISTER', 'UNDO'}

    pose_name: StringProperty(
        name="Pose Name",
        default="New Pose",
        description="Name for the captured pose"
    )

    @classmethod
    def poll(cls, context):
        scene = context.scene
        armature = getattr(scene, 'target_armature', None)
        return armature is not None and armature.type == 'ARMATURE'

    def invoke(self, context, event):
        # Generate default name based on existing poses
        scene = context.scene
        existing_names = {p.name for p in scene.pose_library}
        base_name = "Pose"
        counter = 1
        while f"{base_name}_{counter}" in existing_names:
            counter += 1
        self.pose_name = f"{base_name}_{counter}"

        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        scene = context.scene
        armature = scene.target_armature

        if not armature or armature.type != 'ARMATURE':
            self.report({'WARNING'}, "No valid armature set as target")
            return {'CANCELLED'}

        # Capture bone transforms
        bone_data = {}
        pose_bones = armature.pose.bones

        for bone_name in BONE_INDEX.keys():
            pose_bone = pose_bones.get(bone_name)
            if pose_bone:
                # Ensure quaternion mode
                if pose_bone.rotation_mode != 'QUATERNION':
                    pose_bone.rotation_mode = 'QUATERNION'

                q = pose_bone.rotation_quaternion
                l = pose_bone.location
                s = pose_bone.scale

                bone_data[bone_name] = [
                    q.w, q.x, q.y, q.z,
                    l.x, l.y, l.z,
                    s.x, s.y, s.z
                ]

        # Create new pose entry
        new_pose = scene.pose_library.add()
        new_pose.name = self.pose_name
        new_pose.bone_data_json = json.dumps(bone_data)
        new_pose.source_armature_name = armature.name
        new_pose.created_timestamp = time.time()

        # Select the new pose
        scene.pose_library_index = len(scene.pose_library) - 1

        log_game("POSES", f"CAPTURE name={self.pose_name} bones={len(bone_data)} armature={armature.name}")
        self.report({'INFO'}, f"Captured pose: {self.pose_name} ({len(bone_data)} bones)")

        return {'FINISHED'}


class EXPLORATORY_OT_RemovePose(Operator):
    """Remove a pose from the library"""
    bl_idname = "exploratory.remove_pose"
    bl_label = "Remove Pose"
    bl_options = {'REGISTER', 'UNDO'}

    index: IntProperty(default=-1)

    @classmethod
    def poll(cls, context):
        return len(context.scene.pose_library) > 0

    def execute(self, context):
        scene = context.scene
        idx = self.index if self.index >= 0 else scene.pose_library_index

        if 0 <= idx < len(scene.pose_library):
            name = scene.pose_library[idx].name
            scene.pose_library.remove(idx)

            # Adjust index
            scene.pose_library_index = max(0, min(idx, len(scene.pose_library) - 1))

            log_game("POSES", f"REMOVE name={name} index={idx}")
            self.report({'INFO'}, f"Removed pose: {name}")
        else:
            self.report({'WARNING'}, "No pose selected")

        return {'FINISHED'}


class EXPLORATORY_OT_ApplyPose(Operator):
    """Apply a pose to the target armature (editor preview)"""
    bl_idname = "exploratory.apply_pose"
    bl_label = "Preview Pose"
    bl_options = {'REGISTER', 'UNDO'}

    index: IntProperty(default=-1)

    @classmethod
    def poll(cls, context):
        scene = context.scene
        armature = getattr(scene, 'target_armature', None)
        has_poses = len(scene.pose_library) > 0
        return armature is not None and armature.type == 'ARMATURE' and has_poses

    def execute(self, context):
        scene = context.scene
        idx = self.index if self.index >= 0 else scene.pose_library_index

        if not (0 <= idx < len(scene.pose_library)):
            self.report({'WARNING'}, "No valid pose selected")
            return {'CANCELLED'}

        armature = scene.target_armature
        if not armature or armature.type != 'ARMATURE':
            self.report({'WARNING'}, "No valid armature set as target")
            return {'CANCELLED'}

        pose_entry = scene.pose_library[idx]

        try:
            bone_data = json.loads(pose_entry.bone_data_json)
        except json.JSONDecodeError:
            self.report({'ERROR'}, "Invalid pose data")
            return {'CANCELLED'}

        pose_bones = armature.pose.bones
        applied_count = 0

        for bone_name, transform in bone_data.items():
            pose_bone = pose_bones.get(bone_name)
            if pose_bone:
                # Ensure quaternion mode
                pose_bone.rotation_mode = 'QUATERNION'

                # Apply transform: [qw, qx, qy, qz, lx, ly, lz, sx, sy, sz]
                pose_bone.rotation_quaternion = Quaternion((transform[0], transform[1], transform[2], transform[3]))
                pose_bone.location = Vector((transform[4], transform[5], transform[6]))
                pose_bone.scale = Vector((transform[7], transform[8], transform[9]))
                applied_count += 1

        # Update viewport
        context.view_layer.update()

        log_game("POSES", f"APPLY name={pose_entry.name} bones={applied_count}")
        self.report({'INFO'}, f"Applied pose: {pose_entry.name} ({applied_count} bones)")

        return {'FINISHED'}


class EXPLORATORY_OT_DuplicatePose(Operator):
    """Duplicate the selected pose"""
    bl_idname = "exploratory.duplicate_pose"
    bl_label = "Duplicate Pose"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return len(context.scene.pose_library) > 0

    def execute(self, context):
        scene = context.scene
        idx = scene.pose_library_index

        if not (0 <= idx < len(scene.pose_library)):
            self.report({'WARNING'}, "No pose selected")
            return {'CANCELLED'}

        source = scene.pose_library[idx]

        # Create duplicate
        new_pose = scene.pose_library.add()
        new_pose.name = f"{source.name}_copy"
        new_pose.description = source.description
        new_pose.bone_data_json = source.bone_data_json
        new_pose.source_armature_name = source.source_armature_name
        new_pose.created_timestamp = time.time()

        # Select the new pose
        scene.pose_library_index = len(scene.pose_library) - 1

        log_game("POSES", f"DUPLICATE source={source.name} new={new_pose.name}")
        self.report({'INFO'}, f"Duplicated pose: {new_pose.name}")

        return {'FINISHED'}


# NOTE: Legacy dev test operators (ANIM2_OT_PlayPose, ANIM2_OT_StopPose) removed.
# Pose playback is now handled by unified ANIM2_OT_TestPlay._play_pose() in test_panel.py


# =============================================================================
# REGISTRATION
# =============================================================================

# Classes to register
classes = [
    PoseEntry,
    EXPLORATORY_UL_PoseLibraryList,
    EXPLORATORY_OT_CapturePose,
    EXPLORATORY_OT_RemovePose,
    EXPLORATORY_OT_ApplyPose,
    EXPLORATORY_OT_DuplicatePose,
]


def register_pose_library():
    """Register pose library classes and scene properties."""
    for cls in classes:
        bpy.utils.register_class(cls)

    # Scene properties
    bpy.types.Scene.pose_library = CollectionProperty(type=PoseEntry)
    bpy.types.Scene.pose_library_index = IntProperty(
        name="Pose Library Index",
        default=0
    )

    log_game("POSES", "REGISTER pose_library module")


def unregister_pose_library():
    """Unregister pose library classes and scene properties."""
    # Remove scene properties
    if hasattr(bpy.types.Scene, 'pose_library'):
        del bpy.types.Scene.pose_library
    if hasattr(bpy.types.Scene, 'pose_library_index'):
        del bpy.types.Scene.pose_library_index

    # Unregister classes in reverse order
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass

    log_game("POSES", "UNREGISTER pose_library module")
