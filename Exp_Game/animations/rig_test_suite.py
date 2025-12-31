"""
Rig Test Suite - Visual comparison of neural network predictions vs training data.

PURPOSE:
    See how well the trained neural network reproduces poses from the training data.
    Pick any sample, view the original pose, then see what the network predicts.

WORKFLOW:
    1. Load training data (Neural IK panel → Load Saved Data)
    2. Load trained weights (Neural IK panel → Load Weights)
    3. Pick a sample index (or click Random)
    4. Click "Show Original" to see the ground-truth pose from training data
    5. Click "Show Prediction" to see what the neural network outputs
    6. Compare visually - how close is the prediction to the original?

WHAT THE ERRORS MEAN:
    - Position Error: How far (in cm) the predicted limb positions are from target
    - Lower is better. <10cm is good, <5cm is excellent
"""

import bpy
import math
import random
import numpy as np
from mathutils import Quaternion, Vector
from bpy.types import Operator
from bpy.props import EnumProperty, IntProperty, BoolProperty

from .neural_network import get_network
from .neural_network.context import normalize_input
from .neural_network.config import (
    NUM_BONES,
    CONTROLLED_BONES,
    END_EFFECTORS,
    BONE_TO_INDEX,
    OUTPUT_SIZE,
)
from .neural_network.forward_kinematics import compute_fk_loss_with_orientation
from .test_panel import _neural_data
from ..developer.dev_logger import log_game


# State for UI display
_rigtest_state = {
    "last_pos_errors": None,
    "last_ori_errors": None,
    "last_pos_rmse": None,
    "last_ori_deg": None,
    "last_sample": None,
    "used_set": None,
    "last_action": None,  # "original" or "prediction"
}


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
    """Show the ORIGINAL pose from training data (what the network should learn to reproduce)"""
    bl_idname = "rigtest.apply_ground_truth"
    bl_label = "Show Original Pose"
    bl_options = {'REGISTER', 'UNDO'}

    use_test: BoolProperty(default=True)
    sample_index: IntProperty(default=0, min=0)

    def execute(self, context):
        arm = getattr(context.scene, "target_armature", None)
        if not arm or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Set target armature in Scene properties first")
            return {'CANCELLED'}

        sample = _get_dataset_sample(self.use_test, self.sample_index)
        if sample is None or sample["output"] is None:
            self.report({'ERROR'}, "No data loaded - use Neural IK panel to Load Saved Data first")
            return {'CANCELLED'}

        rotations = sample["output"].reshape(NUM_BONES, 3)
        _apply_rotations_to_armature(arm, rotations)

        _rigtest_state.update({
            "last_pos_errors": None,
            "last_ori_errors": None,
            "last_pos_rmse": None,
            "last_ori_deg": None,
            "last_sample": sample["index"],
            "used_set": "test" if self.use_test else "train",
            "last_action": "original",
        })

        log_game("RIGTEST", f"SHOW_ORIGINAL set={_rigtest_state['used_set']} idx={sample['index']}")
        self.report({'INFO'}, f"Showing ORIGINAL pose #{sample['index']} (this is the target)")
        return {'FINISHED'}


class RIGTEST_OT_RunPrediction(Operator):
    """Show the PREDICTED pose from the neural network (compare this to the original)"""
    bl_idname = "rigtest.run_prediction"
    bl_label = "Show Network Prediction"
    bl_options = {'REGISTER', 'UNDO'}

    use_test: BoolProperty(default=True)
    sample_index: IntProperty(default=0, min=0)

    def execute(self, context):
        arm = getattr(context.scene, "target_armature", None)
        if not arm or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Set target armature in Scene properties first")
            return {'CANCELLED'}

        sample = _get_dataset_sample(self.use_test, self.sample_index)
        if sample is None or sample["input"] is None:
            self.report({'ERROR'}, "No data loaded - use Neural IK panel to Load Saved Data first")
            return {'CANCELLED'}

        net = get_network()
        x = normalize_input(sample["input"]).astype(np.float32)
        pred = net.predict_clamped(x)[0].reshape(NUM_BONES, 3)

        _apply_rotations_to_armature(arm, pred)

        # Compute errors
        pos_rmse = None
        ori_deg = None
        pos_errs = None
        ori_errs = None

        if sample["eff_targets"] is not None and sample["eff_rots"] is not None:
            root_rot = None
            if sample["root_fwd"] is not None and sample["root_up"] is not None:
                root_rot = _build_root_rot(sample["root_fwd"], sample["root_up"]).reshape(1, 3, 3)

            pos_loss, ori_loss, pos_errs, ori_errs = compute_fk_loss_with_orientation(
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
            "last_ori_errors": ori_errs[0] if ori_errs is not None else None,
            "last_pos_rmse": pos_rmse,
            "last_ori_deg": ori_deg,
            "last_sample": sample["index"],
            "used_set": "test" if self.use_test else "train",
            "last_action": "prediction",
        })

        log_game("RIGTEST", f"SHOW_PREDICTION set={_rigtest_state['used_set']} idx={sample['index']} pos_rmse={pos_rmse}")

        if pos_rmse is not None:
            quality = "excellent" if pos_rmse < 0.05 else "good" if pos_rmse < 0.15 else "needs work"
            self.report({'INFO'}, f"Prediction #{sample['index']}: {pos_rmse*100:.1f}cm error ({quality})")
        else:
            self.report({'INFO'}, f"Showing PREDICTION for pose #{sample['index']}")
        return {'FINISHED'}


# Scene properties
def register_props():
    bpy.types.Scene.rigtest_use_test = EnumProperty(
        name="Dataset",
        description="Which dataset to sample from",
        items=[
            ("test", "Test Set", "Poses the network has NOT seen during training (harder)"),
            ("train", "Train Set", "Poses the network trained on (should be accurate)"),
        ],
        default="test",
    )
    bpy.types.Scene.rigtest_sample_index = IntProperty(
        name="Pose #",
        description="Which pose to view (each number is a different pose)",
        default=0,
        min=0,
    )


def unregister_props():
    for attr in ("rigtest_use_test", "rigtest_sample_index"):
        if hasattr(bpy.types.Scene, attr):
            delattr(bpy.types.Scene, attr)


def draw_rig_test_ui(layout, scene):
    """Draw the Rig Test Suite UI."""
    box = layout.box()

    # Header
    row = box.row()
    row.label(text="Pose Comparison Tool", icon='POSE_HLT')

    # Check if data is loaded
    dataset = _neural_data.get("dataset")
    if not dataset:
        warn = box.box()
        warn.alert = True
        warn.label(text="No data loaded!", icon='ERROR')
        warn.label(text="Go to Neural IK panel above:")
        warn.label(text="1. Click 'Load Saved Data'")
        warn.label(text="2. Click 'Load Weights'")
        return

    # Dataset selection
    col = box.column(align=True)
    col.label(text="1. Pick a dataset:", icon='THREE_DOTS')
    row = col.row(align=True)
    row.prop(scene, "rigtest_use_test", expand=True)

    # Sample selection
    use_test = scene.rigtest_use_test == "test"
    count = _get_sample_count(use_test)

    col.separator()
    col.label(text="2. Pick a pose:", icon='THREE_DOTS')
    row = col.row(align=True)
    row.prop(scene, "rigtest_sample_index", text="Pose")
    row.label(text=f"of {count}")
    row.operator("rigtest.random_sample", text="", icon='FILE_REFRESH')

    # Action buttons
    col.separator()
    col.label(text="3. Compare:", icon='THREE_DOTS')

    row = col.row(align=True)
    row.scale_y = 1.3
    op1 = row.operator("rigtest.apply_ground_truth", text="Show Original", icon='ARMATURE_DATA')
    op1.use_test = use_test
    op1.sample_index = scene.rigtest_sample_index

    op2 = row.operator("rigtest.run_prediction", text="Show Prediction", icon='RNA')
    op2.use_test = use_test
    op2.sample_index = scene.rigtest_sample_index

    # Results
    state = _rigtest_state
    if state["last_sample"] is not None:
        results = box.box()

        if state["last_action"] == "original":
            results.label(text=f"Showing: ORIGINAL pose #{state['last_sample']}", icon='CHECKMARK')
            results.label(text="(This is what the network should match)")
        else:
            results.label(text=f"Showing: PREDICTION for pose #{state['last_sample']}", icon='RNA')

            if state["last_pos_rmse"] is not None:
                err_cm = state["last_pos_rmse"] * 100

                # Color-coded quality
                if err_cm < 5:
                    icon = 'CHECKMARK'
                    quality = "Excellent"
                elif err_cm < 15:
                    icon = 'SOLO_ON'
                    quality = "Good"
                else:
                    icon = 'ERROR'
                    quality = "Needs work"

                results.label(text=f"Position Error: {err_cm:.1f} cm ({quality})", icon=icon)

                if state["last_ori_deg"] is not None:
                    results.label(text=f"Rotation Error: {state['last_ori_deg']:.1f} deg")

                # Per-limb breakdown
                if state["last_pos_errors"] is not None:
                    results.separator()
                    results.label(text="Per-limb errors:")
                    for i, name in enumerate(END_EFFECTORS):
                        err = state["last_pos_errors"][i] * 100
                        results.label(text=f"  {name}: {err:.1f} cm")


# Registration
classes = (
    RIGTEST_OT_RandomSample,
    RIGTEST_OT_ApplyGroundTruth,
    RIGTEST_OT_RunPrediction,
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
