"""
Rig Test Suite - Visual testing of neural network IK predictions.

TWO MODES:

1. TRAINING DATA MODE - Test against saved training data
   - Pick a sample from train/test set
   - See original pose vs network prediction
   - Good for verifying training quality

2. POSE LIBRARY MODE - Test with YOUR custom poses
   - Select a pose from your Pose Library
   - Set it as the IK target
   - Watch the network solve for that pose
   - Great for testing on poses the network has never seen!

WORKFLOW FOR POSE LIBRARY MODE:
    1. Load trained weights (Neural IK panel → Load Weights)
    2. Select a pose from your Pose Library dropdown
    3. Click "Set as Target" - applies pose, captures effector positions
    4. Click "Solve IK" - network predicts rotations to reach those positions
    5. Compare the result!
"""

import bpy
import json
import math
import random
import numpy as np
from mathutils import Quaternion, Vector, Matrix, Euler
from bpy.types import Operator
from bpy.props import EnumProperty, IntProperty, BoolProperty

from .neural_network import get_network
from .neural_network.context import normalize_input, ContextExtractor
from .neural_network.config import (
    NUM_BONES,
    CONTROLLED_BONES,
    END_EFFECTORS,
    BONE_TO_INDEX,
    OUTPUT_SIZE,
    INPUT_SIZE,
)
from .neural_network.forward_kinematics import compute_fk_loss_with_orientation
from .test_panel import _neural_data
from ..developer.dev_logger import log_game


# =============================================================================
# STATE
# =============================================================================

# Training data mode state
_rigtest_state = {
    "last_pos_errors": None,
    "last_ori_errors": None,
    "last_pos_rmse": None,
    "last_ori_deg": None,
    "last_sample": None,
    "used_set": None,
    "last_action": None,
}

# Pose library target mode state
_pose_target_state = {
    "target_set": False,
    "target_pose_name": None,
    "target_effector_pos": None,  # (5, 3) world positions
    "target_effector_rot": None,  # (5, 3) euler rotations
    "target_root_pos": None,      # (3,) root position
    "target_root_fwd": None,      # (3,) forward vector
    "target_root_up": None,       # (3,) up vector
    "network_input": None,        # (50,) input vector
    "last_pos_rmse": None,
    "last_ori_deg": None,
    "last_pos_errors": None,
    "last_action": None,          # "target" or "solved"
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _axis_angle_to_quaternion(vec: np.ndarray) -> Quaternion:
    """Convert axis-angle vector to Blender Quaternion."""
    angle = float(np.linalg.norm(vec))
    if angle < 1e-8:
        return Quaternion((1.0, 0.0, 0.0, 0.0))
    axis = vec / angle
    return Quaternion((np.cos(angle / 2.0),
                       *(axis * np.sin(angle / 2.0))))


def _apply_rotations_to_armature(armature, rotations: np.ndarray):
    """Apply axis-angle rotations (NUM_BONES x 3) to pose bones."""
    pose_bones = armature.pose.bones
    for i, bone_name in enumerate(CONTROLLED_BONES):
        pb = pose_bones.get(bone_name)
        if pb:
            pb.rotation_mode = 'QUATERNION'
            quat = _axis_angle_to_quaternion(rotations[i])
            pb.rotation_quaternion = quat
            pb.location = Vector((0.0, 0.0, 0.0))
    bpy.context.view_layer.update()


def _build_root_rot(forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Build 3x3 root rotation from forward/up vectors."""
    f = forward / (np.linalg.norm(forward) + 1e-8)
    u = up / (np.linalg.norm(up) + 1e-8)
    r = np.cross(f, u)
    r /= (np.linalg.norm(r) + 1e-8)
    u = np.cross(r, f)
    R = np.stack([r, f, u], axis=1)
    return R.astype(np.float32)


def _get_sample_count(use_test: bool) -> int:
    """Get number of samples in the selected set."""
    dataset = _neural_data.get("dataset")
    if not dataset:
        return 0
    key = "test_inputs" if use_test else "train_inputs"
    arr = dataset.get(key)
    return len(arr) if arr is not None else 0


def _get_dataset_sample(use_test: bool, index: int):
    """Fetch a sample dict from the loaded dataset."""
    dataset = _neural_data.get("dataset")
    if not dataset:
        return None

    prefix = "test" if use_test else "train"
    inputs = dataset.get(f"{prefix}_inputs")
    outputs = dataset.get(f"{prefix}_outputs")
    eff_targets = dataset.get(f"{prefix}_effector_targets")
    eff_rots = dataset.get(f"{prefix}_effector_rotations")
    root_pos = dataset.get(f"{prefix}_root_positions")
    root_fwd = dataset.get(f"{prefix}_root_forwards")
    root_up = dataset.get(f"{prefix}_root_ups")

    if inputs is None or len(inputs) == 0:
        return None

    idx = max(0, min(index, len(inputs) - 1))

    return {
        "input": inputs[idx],
        "output": outputs[idx],
        "eff_targets": eff_targets[idx] if eff_targets is not None else None,
        "eff_rots": eff_rots[idx] if eff_rots is not None else None,
        "root_pos": root_pos[idx] if root_pos is not None else None,
        "root_fwd": root_fwd[idx] if root_fwd is not None else None,
        "root_up": root_up[idx] if root_up is not None else None,
        "index": idx,
    }


def _apply_pose_to_armature(armature, pose_entry):
    """Apply a pose library entry to an armature. Returns True on success."""
    try:
        bone_data = json.loads(pose_entry.bone_data_json)
    except json.JSONDecodeError:
        return False

    pose_bones = armature.pose.bones
    for bone_name, transform in bone_data.items():
        pose_bone = pose_bones.get(bone_name)
        if pose_bone:
            pose_bone.rotation_mode = 'QUATERNION'
            pose_bone.rotation_quaternion = Quaternion((transform[0], transform[1], transform[2], transform[3]))
            pose_bone.location = Vector((transform[4], transform[5], transform[6]))
            pose_bone.scale = Vector((transform[7], transform[8], transform[9]))

    bpy.context.view_layer.update()
    return True


def _extract_effector_data(armature):
    """Extract effector positions and rotations from current armature pose."""
    pose_bones = armature.pose.bones
    arm_matrix = armature.matrix_world

    # Get root (Hips) transform
    hips = pose_bones.get("Hips")
    if not hips:
        return None

    root_world_pos = arm_matrix @ hips.head
    root_world_matrix = arm_matrix @ hips.matrix
    root_forward = Vector(root_world_matrix.col[1][:3]).normalized()
    root_up = Vector(root_world_matrix.col[2][:3]).normalized()

    # Extract effector data
    effector_positions = []
    effector_rotations = []

    for eff_name in END_EFFECTORS:
        bone = pose_bones.get(eff_name)
        if bone:
            world_pos = arm_matrix @ bone.head
            bone_world_matrix = arm_matrix @ bone.matrix
            world_rot = bone_world_matrix.to_euler('XYZ')

            effector_positions.append([world_pos.x, world_pos.y, world_pos.z])
            effector_rotations.append([world_rot.x, world_rot.y, world_rot.z])
        else:
            effector_positions.append([0.0, 0.0, 0.0])
            effector_rotations.append([0.0, 0.0, 0.0])

    return {
        "eff_pos": np.array(effector_positions, dtype=np.float32),
        "eff_rot": np.array(effector_rotations, dtype=np.float32),
        "root_pos": np.array([root_world_pos.x, root_world_pos.y, root_world_pos.z], dtype=np.float32),
        "root_fwd": np.array([root_forward.x, root_forward.y, root_forward.z], dtype=np.float32),
        "root_up": np.array([root_up.x, root_up.y, root_up.z], dtype=np.float32),
    }


# =============================================================================
# TRAINING DATA MODE OPERATORS
# =============================================================================

class RIGTEST_OT_RandomSample(Operator):
    """Pick a random sample from the dataset"""
    bl_idname = "rigtest.random_sample"
    bl_label = "Random"
    bl_options = {'REGISTER'}

    def execute(self, context):
        scene = context.scene
        use_test = scene.rigtest_use_test == "test"
        count = _get_sample_count(use_test)
        if count == 0:
            self.report({'WARNING'}, "No data loaded - use Neural IK panel to Load Saved Data")
            return {'CANCELLED'}
        scene.rigtest_sample_index = random.randint(0, count - 1)
        return {'FINISHED'}


class RIGTEST_OT_ApplyGroundTruth(Operator):
    """Show the ORIGINAL pose from training data"""
    bl_idname = "rigtest.apply_ground_truth"
    bl_label = "Show Original Pose"
    bl_options = {'REGISTER', 'UNDO'}

    use_test: BoolProperty(default=True)
    sample_index: IntProperty(default=0, min=0)

    def execute(self, context):
        arm = getattr(context.scene, "target_armature", None)
        if not arm or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Set target armature first")
            return {'CANCELLED'}

        sample = _get_dataset_sample(self.use_test, self.sample_index)
        if sample is None or sample["output"] is None:
            self.report({'ERROR'}, "No data loaded")
            return {'CANCELLED'}

        rotations = sample["output"].reshape(NUM_BONES, 3)
        _apply_rotations_to_armature(arm, rotations)

        _rigtest_state.update({
            "last_pos_errors": None, "last_ori_errors": None,
            "last_pos_rmse": None, "last_ori_deg": None,
            "last_sample": sample["index"],
            "used_set": "test" if self.use_test else "train",
            "last_action": "original",
        })

        self.report({'INFO'}, f"Showing ORIGINAL pose #{sample['index']}")
        return {'FINISHED'}


class RIGTEST_OT_RunPrediction(Operator):
    """Show the PREDICTED pose from the neural network"""
    bl_idname = "rigtest.run_prediction"
    bl_label = "Show Network Prediction"
    bl_options = {'REGISTER', 'UNDO'}

    use_test: BoolProperty(default=True)
    sample_index: IntProperty(default=0, min=0)

    def execute(self, context):
        arm = getattr(context.scene, "target_armature", None)
        if not arm or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Set target armature first")
            return {'CANCELLED'}

        sample = _get_dataset_sample(self.use_test, self.sample_index)
        if sample is None or sample["input"] is None:
            self.report({'ERROR'}, "No data loaded")
            return {'CANCELLED'}

        net = get_network()
        x = normalize_input(sample["input"]).astype(np.float32)
        pred = net.predict_clamped(x)[0].reshape(NUM_BONES, 3)
        _apply_rotations_to_armature(arm, pred)

        # Compute errors
        pos_rmse, ori_deg, pos_errs = None, None, None
        if sample["eff_targets"] is not None and sample["eff_rots"] is not None:
            root_rot = None
            if sample["root_fwd"] is not None and sample["root_up"] is not None:
                root_rot = _build_root_rot(sample["root_fwd"], sample["root_up"]).reshape(1, 3, 3)

            pos_loss, ori_loss, pos_errs, _ = compute_fk_loss_with_orientation(
                pred.reshape(1, OUTPUT_SIZE),
                sample["eff_targets"].reshape(1, 5, 3),
                sample["eff_rots"].reshape(1, 5, 3),
                sample["root_pos"].reshape(1, 3) if sample["root_pos"] is not None else None,
                root_rot,
            )
            pos_rmse = float(np.sqrt(pos_loss))
            ori_deg = float(np.sqrt(ori_loss) * 57.2957795)

        _rigtest_state.update({
            "last_pos_errors": pos_errs[0] if pos_errs is not None else None,
            "last_ori_errors": None,
            "last_pos_rmse": pos_rmse,
            "last_ori_deg": ori_deg,
            "last_sample": sample["index"],
            "used_set": "test" if self.use_test else "train",
            "last_action": "prediction",
        })

        if pos_rmse is not None:
            self.report({'INFO'}, f"Prediction: {pos_rmse*100:.1f}cm error")
        return {'FINISHED'}


# =============================================================================
# POSE LIBRARY TARGET MODE OPERATORS
# =============================================================================

class RIGTEST_OT_SetPoseTarget(Operator):
    """Apply selected pose and capture as IK target"""
    bl_idname = "rigtest.set_pose_target"
    bl_label = "Set as Target"
    bl_description = "Apply the selected pose and capture effector positions as the IK target"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        arm = getattr(scene, "target_armature", None)

        if not arm or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Set target armature first")
            return {'CANCELLED'}

        if len(scene.pose_library) == 0:
            self.report({'ERROR'}, "No poses in library - capture some poses first")
            return {'CANCELLED'}

        idx = scene.pose_library_index
        if not (0 <= idx < len(scene.pose_library)):
            self.report({'ERROR'}, "Invalid pose selection")
            return {'CANCELLED'}

        pose_entry = scene.pose_library[idx]

        # Apply the pose
        if not _apply_pose_to_armature(arm, pose_entry):
            self.report({'ERROR'}, "Failed to apply pose")
            return {'CANCELLED'}

        # Extract effector data
        data = _extract_effector_data(arm)
        if data is None:
            self.report({'ERROR'}, "Failed to extract effector data (missing Hips bone?)")
            return {'CANCELLED'}

        # Build network input using ContextExtractor
        try:
            extractor = ContextExtractor(arm)
            network_input = extractor.extract(
                ground_height=0.0,
                motion_phase=0.0,
                task_type="idle",
            )
        except Exception as e:
            log_game("RIGTEST", f"ContextExtractor error: {e}")
            self.report({'ERROR'}, f"Failed to build network input: {e}")
            return {'CANCELLED'}

        # Store target state
        _pose_target_state.update({
            "target_set": True,
            "target_pose_name": pose_entry.name,
            "target_effector_pos": data["eff_pos"],
            "target_effector_rot": data["eff_rot"],
            "target_root_pos": data["root_pos"],
            "target_root_fwd": data["root_fwd"],
            "target_root_up": data["root_up"],
            "network_input": network_input,
            "last_pos_rmse": None,
            "last_ori_deg": None,
            "last_pos_errors": None,
            "last_action": "target",
        })

        log_game("RIGTEST", f"SET_TARGET pose={pose_entry.name}")
        self.report({'INFO'}, f"Target set: '{pose_entry.name}' - now click 'Solve IK'")
        return {'FINISHED'}


class RIGTEST_OT_SolvePoseIK(Operator):
    """Run neural network to solve for the target pose"""
    bl_idname = "rigtest.solve_pose_ik"
    bl_label = "Solve IK"
    bl_description = "Run the neural network to predict rotations that reach the target effector positions"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        arm = getattr(scene, "target_armature", None)

        if not arm or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Set target armature first")
            return {'CANCELLED'}

        if not _pose_target_state["target_set"]:
            self.report({'ERROR'}, "Set a target pose first (click 'Set as Target')")
            return {'CANCELLED'}

        network_input = _pose_target_state["network_input"]
        if network_input is None:
            self.report({'ERROR'}, "No network input captured")
            return {'CANCELLED'}

        # Run network
        net = get_network()
        x = normalize_input(network_input).astype(np.float32)
        pred = net.predict_clamped(x)[0].reshape(NUM_BONES, 3)

        # Apply prediction
        _apply_rotations_to_armature(arm, pred)

        # Compute errors vs target
        pos_rmse, ori_deg, pos_errs = None, None, None

        eff_targets = _pose_target_state["target_effector_pos"]
        eff_rots = _pose_target_state["target_effector_rot"]
        root_pos = _pose_target_state["target_root_pos"]
        root_fwd = _pose_target_state["target_root_fwd"]
        root_up = _pose_target_state["target_root_up"]

        if eff_targets is not None and eff_rots is not None:
            root_rot = None
            if root_fwd is not None and root_up is not None:
                root_rot = _build_root_rot(root_fwd, root_up).reshape(1, 3, 3)

            pos_loss, ori_loss, pos_errs, _ = compute_fk_loss_with_orientation(
                pred.reshape(1, OUTPUT_SIZE),
                eff_targets.reshape(1, 5, 3),
                eff_rots.reshape(1, 5, 3),
                root_pos.reshape(1, 3) if root_pos is not None else None,
                root_rot,
            )
            pos_rmse = float(np.sqrt(pos_loss))
            ori_deg = float(np.sqrt(ori_loss) * 57.2957795)

        _pose_target_state.update({
            "last_pos_rmse": pos_rmse,
            "last_ori_deg": ori_deg,
            "last_pos_errors": pos_errs[0] if pos_errs is not None else None,
            "last_action": "solved",
        })

        log_game("RIGTEST", f"SOLVE_IK pose={_pose_target_state['target_pose_name']} pos_rmse={pos_rmse}")

        if pos_rmse is not None:
            quality = "excellent" if pos_rmse < 0.05 else "good" if pos_rmse < 0.15 else "needs work"
            self.report({'INFO'}, f"IK solved: {pos_rmse*100:.1f}cm error ({quality})")
        else:
            self.report({'INFO'}, "IK solved - check armature")
        return {'FINISHED'}


class RIGTEST_OT_ShowTargetPose(Operator):
    """Show the target pose again (to compare with solution)"""
    bl_idname = "rigtest.show_target_pose"
    bl_label = "Show Target"
    bl_description = "Re-apply the target pose to compare with the IK solution"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        arm = getattr(scene, "target_armature", None)

        if not arm or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Set target armature first")
            return {'CANCELLED'}

        if not _pose_target_state["target_set"]:
            self.report({'ERROR'}, "No target set")
            return {'CANCELLED'}

        pose_name = _pose_target_state["target_pose_name"]

        # Find and apply the pose
        for pose_entry in scene.pose_library:
            if pose_entry.name == pose_name:
                _apply_pose_to_armature(arm, pose_entry)
                _pose_target_state["last_action"] = "target"
                self.report({'INFO'}, f"Showing target: '{pose_name}'")
                return {'FINISHED'}

        self.report({'WARNING'}, f"Pose '{pose_name}' no longer exists in library")
        return {'CANCELLED'}


# =============================================================================
# SCENE PROPERTIES
# =============================================================================

def register_props():
    # Training data mode
    bpy.types.Scene.rigtest_use_test = EnumProperty(
        name="Dataset",
        items=[
            ("test", "Test Set", "Unseen poses"),
            ("train", "Train Set", "Training poses"),
        ],
        default="test",
    )
    bpy.types.Scene.rigtest_sample_index = IntProperty(
        name="Pose #", default=0, min=0,
    )


def unregister_props():
    for attr in ("rigtest_use_test", "rigtest_sample_index"):
        if hasattr(bpy.types.Scene, attr):
            delattr(bpy.types.Scene, attr)


# =============================================================================
# UI DRAWING
# =============================================================================

def draw_rig_test_ui(layout, scene):
    """Draw the complete Rig Test Suite UI."""

    # =========================================================================
    # POSE LIBRARY TARGET MODE
    # =========================================================================
    box = layout.box()
    row = box.row()
    row.label(text="Pose Library IK Test", icon='ARMATURE_DATA')

    if len(scene.pose_library) == 0:
        warn = box.box()
        warn.label(text="No poses in library", icon='INFO')
        warn.label(text="Capture poses first (Pose Library panel)")
    else:
        col = box.column(align=True)

        # Pose selection
        col.label(text="1. Select a pose:")
        col.template_list(
            "EXPLORATORY_UL_pose_library_list", "",
            scene, "pose_library",
            scene, "pose_library_index",
            rows=3,
        )

        # Buttons
        col.separator()
        col.label(text="2. Test the network:")

        row = col.row(align=True)
        row.scale_y = 1.3
        row.operator("rigtest.set_pose_target", text="Set as Target", icon='TRACKER')
        row.operator("rigtest.solve_pose_ik", text="Solve IK", icon='CON_KINEMATIC')

        # Show target button (for comparison)
        if _pose_target_state["target_set"]:
            row = col.row(align=True)
            row.operator("rigtest.show_target_pose", text="Show Target Again", icon='LOOP_BACK')

        # Results
        state = _pose_target_state
        if state["target_set"]:
            results = box.box()

            if state["last_action"] == "target":
                results.label(text=f"Target: '{state['target_pose_name']}'", icon='TRACKER')
                results.label(text="Click 'Solve IK' to see prediction")

            elif state["last_action"] == "solved":
                results.label(text=f"IK Solution for '{state['target_pose_name']}'", icon='CON_KINEMATIC')

                if state["last_pos_rmse"] is not None:
                    err_cm = state["last_pos_rmse"] * 100
                    if err_cm < 5:
                        icon, quality = 'CHECKMARK', "Excellent"
                    elif err_cm < 15:
                        icon, quality = 'SOLO_ON', "Good"
                    else:
                        icon, quality = 'ERROR', "Needs work"

                    results.label(text=f"Position Error: {err_cm:.1f} cm ({quality})", icon=icon)

                    if state["last_ori_deg"] is not None:
                        results.label(text=f"Rotation Error: {state['last_ori_deg']:.1f} deg")

                    if state["last_pos_errors"] is not None:
                        results.separator()
                        results.label(text="Per-limb:")
                        for i, name in enumerate(END_EFFECTORS):
                            err = state["last_pos_errors"][i] * 100
                            results.label(text=f"  {name}: {err:.1f} cm")

    # =========================================================================
    # TRAINING DATA MODE
    # =========================================================================
    box = layout.box()
    row = box.row()
    row.label(text="Training Data Test", icon='FILE_VOLUME')

    dataset = _neural_data.get("dataset")
    if not dataset:
        warn = box.box()
        warn.label(text="No training data loaded", icon='INFO')
        warn.label(text="Neural IK panel → Load Saved Data")
    else:
        col = box.column(align=True)

        # Dataset selection
        row = col.row(align=True)
        row.prop(scene, "rigtest_use_test", expand=True)

        # Sample selection
        use_test = scene.rigtest_use_test == "test"
        count = _get_sample_count(use_test)

        row = col.row(align=True)
        row.prop(scene, "rigtest_sample_index", text="Pose")
        row.label(text=f"of {count}")
        row.operator("rigtest.random_sample", text="", icon='FILE_REFRESH')

        # Buttons
        row = col.row(align=True)
        row.scale_y = 1.2
        op1 = row.operator("rigtest.apply_ground_truth", text="Original", icon='ARMATURE_DATA')
        op1.use_test = use_test
        op1.sample_index = scene.rigtest_sample_index

        op2 = row.operator("rigtest.run_prediction", text="Prediction", icon='RNA')
        op2.use_test = use_test
        op2.sample_index = scene.rigtest_sample_index

        # Results
        state = _rigtest_state
        if state["last_sample"] is not None and state["last_action"] == "prediction":
            if state["last_pos_rmse"] is not None:
                err_cm = state["last_pos_rmse"] * 100
                col.label(text=f"Error: {err_cm:.1f} cm")


# =============================================================================
# REGISTRATION
# =============================================================================

classes = (
    RIGTEST_OT_RandomSample,
    RIGTEST_OT_ApplyGroundTruth,
    RIGTEST_OT_RunPrediction,
    RIGTEST_OT_SetPoseTarget,
    RIGTEST_OT_SolvePoseIK,
    RIGTEST_OT_ShowTargetPose,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    register_props()
    log_game("RIGTEST", "REGISTER rig_test_suite module")


def unregister():
    unregister_props()
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
    log_game("RIGTEST", "UNREGISTER rig_test_suite module")
