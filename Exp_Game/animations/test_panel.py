# Exp_Game/animations/test_panel.py
"""
Animation Test Panel - STRIPPED DOWN VERSION

Only contains:
- Animation playback testing
- Basic GPU visualization for targets
- Reset pose operator

ALL IK CODE HAS BEEN REMOVED - Starting fresh with neural network approach.
"""

import bpy
import time
from bpy.types import Operator, PropertyGroup
from bpy.props import FloatProperty, BoolProperty, EnumProperty

from ..engine.animations.baker import bake_action
from ..engine import EngineCore
from .controller import AnimationController
from ..developer.dev_logger import start_session, log_game, export_game_log, clear_log
import gpu
from gpu_extras.batch import batch_for_shader

from ..developer.gpu_utils import (
    get_cached_shader,
    CIRCLE_8,
    sphere_wire_verts,
    extend_batch_data,
    crosshair_verts,
)


# =============================================================================
# GPU TARGET VISUALIZER (kept - useful for any system)
# =============================================================================

_target_handler = None
_target_data = None

TARGET_COLORS = {
    "L_HAND": (0.2, 0.6, 1.0, 0.95),
    "R_HAND": (1.0, 0.4, 0.2, 0.95),
    "L_FOOT": (0.2, 1.0, 0.4, 0.95),
    "R_FOOT": (1.0, 1.0, 0.2, 0.95),
    "LOOK_AT": (1.0, 0.2, 1.0, 0.95),
    "HIPS": (0.6, 0.2, 1.0, 0.95),
}


def _draw_targets():
    """GPU draw callback for target visualization."""
    global _target_data
    if _target_data is None:
        return

    gpu.state.depth_test_set('NONE')
    gpu.state.blend_set('ALPHA')
    gpu.state.line_width_set(3.0)

    shader = get_cached_shader()
    all_verts = []
    all_colors = []

    for target in _target_data:
        tag = target.get('tag', '')
        pos = target.get('pos')
        if not pos:
            continue

        color = TARGET_COLORS.get(tag, (1.0, 1.0, 1.0, 0.9))
        extend_batch_data(all_verts, all_colors, crosshair_verts(pos, 0.08), color)
        extend_batch_data(all_verts, all_colors, sphere_wire_verts(pos, 0.06, CIRCLE_8), color)

        origin = target.get('origin')
        if origin:
            all_verts.extend([origin, pos])
            all_colors.extend([(*color[:3], 0.4), color])

    if all_verts:
        batch = batch_for_shader(shader, 'LINES', {"pos": all_verts, "color": all_colors})
        shader.bind()
        batch.draw(shader)

    gpu.state.line_width_set(1.0)
    gpu.state.blend_set('NONE')


def enable_target_visualizer():
    global _target_handler
    if _target_handler is None:
        _target_handler = bpy.types.SpaceView3D.draw_handler_add(
            _draw_targets, (), 'WINDOW', 'POST_VIEW'
        )


def disable_target_visualizer():
    global _target_handler, _target_data
    if _target_handler is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_target_handler, 'WINDOW')
        except:
            pass
        _target_handler = None
    _target_data = None


def set_target_data(targets: list):
    """Set target data for GPU visualization."""
    global _target_data
    _target_data = targets
    if _target_handler is None:
        enable_target_visualizer()


# =============================================================================
# TEST ENGINE & CONTROLLER
# =============================================================================

_test_engine = None
_test_controller = None


def get_test_engine() -> EngineCore:
    global _test_engine
    if _test_engine is None or not _test_engine.is_alive():
        _test_engine = EngineCore()
        _test_engine.start()
        _test_engine.wait_for_readiness(timeout=2.0)
        start_session()
    return _test_engine


def get_test_controller() -> AnimationController:
    global _test_controller
    if _test_controller is None:
        _test_controller = AnimationController()
    return _test_controller


def reset_test_controller():
    global _test_controller, _test_engine
    if _test_engine is not None:
        export_game_log("C:/Users/spenc/Desktop/engine_output_files/diagnostics_latest.txt")
        clear_log()
        _test_engine.shutdown()
        _test_engine = None
    disable_target_visualizer()
    _test_controller = None


# =============================================================================
# BASIC OPERATORS
# =============================================================================

class ANIM2_OT_BakeAll(Operator):
    """Bake all actions and cache in workers"""
    bl_idname = "anim2.bake_all"
    bl_label = "Bake All Actions"
    bl_options = {'REGISTER'}

    def execute(self, context):
        start_time = time.perf_counter()
        reset_test_controller()
        ctrl = get_test_controller()
        engine = get_test_engine()

        baked_count = 0
        for action in bpy.data.actions:
            try:
                anim = bake_action(action)
                ctrl.add_animation(anim)
                baked_count += 1
            except Exception as e:
                pass

        if baked_count > 0 and engine.is_alive():
            cache_data = ctrl.get_cache_data_for_workers()
            engine.broadcast_job("CACHE_ANIMATIONS", cache_data)
            time.sleep(0.1)

        elapsed = (time.perf_counter() - start_time) * 1000
        self.report({'INFO'}, f"Baked {baked_count} actions in {elapsed:.0f}ms")
        return {'FINISHED'}


class ANIM2_OT_ResetPose(Operator):
    """Reset armature to rest pose"""
    bl_idname = "anim2.reset_pose"
    bl_label = "Reset Pose"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        armature = getattr(context.scene, 'target_armature', None)
        return armature and armature.type == 'ARMATURE'

    def execute(self, context):
        armature = context.scene.target_armature
        for pose_bone in armature.pose.bones:
            pose_bone.rotation_mode = 'QUATERNION'
            pose_bone.rotation_quaternion = (1, 0, 0, 0)
            pose_bone.location = (0, 0, 0)
            pose_bone.scale = (1, 1, 1)
        context.view_layer.update()
        self.report({'INFO'}, "Pose reset")
        return {'FINISHED'}


# =============================================================================
# TEST MODAL (ANIMATION PLAYBACK)
# =============================================================================

_active_test_modal = None
_stop_requested = False


def is_test_modal_running() -> bool:
    return _active_test_modal is not None


# Alias for backwards compatibility
def is_test_modal_active() -> bool:
    return is_test_modal_running()


def stop_test_modal():
    global _stop_requested
    _stop_requested = True


class ANIM2_OT_TestModal(Operator):
    """Animation test modal - plays animations at 30Hz"""
    bl_idname = "anim2.test_modal"
    bl_label = "Animation Test Modal"
    bl_options = {'INTERNAL'}

    _timer = None
    _last_time: float = 0.0
    _start_time: float = 0.0

    def invoke(self, context, event):
        global _active_test_modal
        self._last_time = time.perf_counter()
        self._start_time = time.perf_counter()

        wm = context.window_manager
        self._timer = wm.event_timer_add(1/30, window=context.window)
        wm.modal_handler_add(self)
        _active_test_modal = self
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        global _active_test_modal, _stop_requested

        if _stop_requested:
            _stop_requested = False
            self.cancel(context)
            return {'CANCELLED'}

        if event.type == 'ESC' and event.value == 'PRESS':
            self.cancel(context)
            return {'CANCELLED'}

        if event.type == 'TIMER':
            props = context.scene.anim2_test
            armature = getattr(context.scene, 'target_armature', None)
            if not armature:
                self.cancel(context)
                return {'CANCELLED'}

            timeout = getattr(props, 'playback_timeout', 20.0)
            if timeout > 0 and (time.perf_counter() - self._start_time) >= timeout:
                self.cancel(context)
                return {'CANCELLED'}

            current = time.perf_counter()
            dt = current - self._last_time
            self._last_time = current
            self._step_animation(context, dt, armature)
            return {'RUNNING_MODAL'}

        return {'PASS_THROUGH'}

    def _step_animation(self, context, dt: float, armature):
        ctrl = get_test_controller()
        engine = get_test_engine()
        if not ctrl or not engine or not engine.is_alive():
            return

        ctrl.update_state(dt)
        jobs_data = ctrl.get_compute_job_data()
        if not jobs_data:
            return

        job_id = engine.submit_job("ANIMATION_COMPUTE_BATCH", {"objects": jobs_data})
        if job_id is None or job_id < 0:
            return

        poll_start = time.perf_counter()
        while (time.perf_counter() - poll_start) < 0.005:
            results = list(engine.poll_results(max_results=10))
            for result in results:
                if result.job_type == "ANIMATION_COMPUTE_BATCH" and result.job_id == job_id:
                    if result.success:
                        self._apply_result(armature, result.result)
                    return
            time.sleep(0.0001)

    def _apply_result(self, armature, result_data: dict):
        results_dict = result_data.get("results", {})
        for object_name, obj_result in results_dict.items():
            if object_name != armature.name:
                continue
            bone_transforms = obj_result.get("bone_transforms", {})
            pose_bones = armature.pose.bones
            for bone_name, transform in bone_transforms.items():
                pose_bone = pose_bones.get(bone_name)
                if pose_bone:
                    pose_bone.rotation_mode = 'QUATERNION'
                    pose_bone.rotation_quaternion = (transform[0], transform[1], transform[2], transform[3])
                    pose_bone.location = (transform[4], transform[5], transform[6])
                    pose_bone.scale = (transform[7], transform[8], transform[9])
            bpy.context.view_layer.update()

    def cancel(self, context):
        global _active_test_modal
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        _active_test_modal = None


# =============================================================================
# PROPERTIES
# =============================================================================

def get_animation_items(self, context):
    ctrl = get_test_controller()
    items = [("", "Select Animation", "")]
    for name in ctrl.cache.names:
        items.append((name, name, ""))
    return items


class ANIM2_TestProperties(PropertyGroup):
    selected_animation: EnumProperty(
        name="Animation",
        items=get_animation_items
    )
    play_speed: FloatProperty(
        name="Speed",
        default=1.0,
        min=0.1,
        max=3.0
    )
    loop_playback: BoolProperty(
        name="Loop",
        default=True
    )
    playback_timeout: FloatProperty(
        name="Timeout",
        default=20.0,
        min=0.0,
        max=300.0
    )


# =============================================================================
# PLAY/STOP
# =============================================================================

class ANIM2_OT_TestPlay(Operator):
    """Play selected animation"""
    bl_idname = "anim2.test_play"
    bl_label = "Play"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        armature = getattr(context.scene, 'target_armature', None)
        if not armature or armature.type != 'ARMATURE':
            return False
        ctrl = get_test_controller()
        return ctrl.cache.count > 0

    def execute(self, context):
        props = context.scene.anim2_test
        armature = context.scene.target_armature
        anim_name = props.selected_animation

        if not anim_name:
            self.report({'WARNING'}, "No animation selected")
            return {'CANCELLED'}

        ctrl = get_test_controller()
        ctrl.play(
            armature.name, anim_name,
            weight=1.0, speed=props.play_speed,
            looping=props.loop_playback, fade_in=0.2, replace=True
        )

        if not is_test_modal_running():
            bpy.ops.anim2.test_modal('INVOKE_DEFAULT')

        return {'FINISHED'}


class ANIM2_OT_TestStop(Operator):
    """Stop animation playback"""
    bl_idname = "anim2.test_stop"
    bl_label = "Stop"
    bl_options = {'REGISTER'}

    def execute(self, context):
        stop_test_modal()
        ctrl = get_test_controller()
        if ctrl:
            ctrl.clear_all()
        return {'FINISHED'}


class ANIM2_OT_ClearCache(Operator):
    """Clear animation cache"""
    bl_idname = "anim2.clear_cache"
    bl_label = "Clear Cache"
    bl_options = {'REGISTER'}

    def execute(self, context):
        ctrl = get_test_controller()
        if ctrl:
            ctrl.cache.clear()
        return {'FINISHED'}


# =============================================================================
# NEURAL IK OPERATORS
# =============================================================================

# Module-level storage for neural data (persists during session)
_neural_data = {
    'dataset': None,  # Full dataset dict from get_train_test_split()
    'samples_extracted': 0,
    'last_report': None,
}


def get_neural_status() -> dict:
    """Get current neural network status for UI display."""
    from .neural_network import get_network
    import os
    from .neural_network.config import BEST_WEIGHTS_PATH

    net = get_network()
    stats = net.get_stats()

    dataset = _neural_data['dataset']
    train_samples = len(dataset['train_inputs']) if dataset and 'train_inputs' in dataset else 0
    test_samples = len(dataset['test_inputs']) if dataset and 'test_inputs' in dataset else 0

    return {
        'samples': _neural_data['samples_extracted'],
        'train_samples': train_samples,
        'test_samples': test_samples,
        'weights_exist': os.path.exists(BEST_WEIGHTS_PATH),
        'total_updates': stats['total_updates'],
        'best_loss': stats['best_loss'] if stats['best_loss'] < float('inf') else None,
    }


class NEURAL_OT_ExtractData(Operator):
    """Extract training data from all animations"""
    bl_idname = "neural.extract_data"
    bl_label = "Extract Data"
    bl_options = {'REGISTER'}

    def execute(self, context):
        scene = context.scene
        armature = scene.target_armature

        if not armature:
            self.report({'ERROR'}, "Set target armature first")
            return {'CANCELLED'}

        from .neural_network import AnimationDataExtractor

        try:
            extractor = AnimationDataExtractor(armature)
            total = extractor.extract_from_all_actions()

            if total == 0:
                self.report({'WARNING'}, "No animation data extracted")
                return {'CANCELLED'}

            # Get full dataset with train/test split and augmentation
            dataset = extractor.get_train_test_split(augment=True, augment_factor=2)

            # Store globally
            _neural_data['dataset'] = dataset
            _neural_data['samples_extracted'] = total

            train_count = len(dataset['train_inputs'])
            test_count = len(dataset['test_inputs'])

            self.report({'INFO'}, f"Extracted {total} samples ({train_count} train w/aug, {test_count} test)")
            return {'FINISHED'}

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, f"Extraction failed: {e}")
            return {'CANCELLED'}


class NEURAL_OT_Train(Operator):
    """Show instructions for standalone training"""
    bl_idname = "neural.train"
    bl_label = "Train (Standalone)"
    bl_options = {'REGISTER'}

    def execute(self, context):
        # Print clear instructions to console
        print("\n" + "="*70)
        print(" NEURAL IK TRAINING - RUN OUTSIDE BLENDER")
        print("="*70)
        print("")
        print(" Training runs in a separate Python process for speed.")
        print(" Blender stays responsive while training runs.")
        print("")
        print(" STEPS:")
        print(" 1. Open Command Prompt or Terminal")
        print(" 2. Run these commands:")
        print("")
        print("    cd C:\\Users\\spenc\\Desktop\\Exploratory\\addons\\Exploratory\\Exp_Game\\animations\\neural_network")
        print("    python torch_trainer.py")
        print("")
        print(" 3. Wait for training to complete")
        print(" 4. Weights auto-save to weights/best.npy")
        print(" 5. Click 'Reload Weights' in Blender to use them")
        print("")
        print("="*70 + "\n")

        self.report({'INFO'}, "See System Console for training instructions (Window > Toggle System Console)")
        return {'FINISHED'}


class NEURAL_OT_Test(Operator):
    """Run test suite to verify learning"""
    bl_idname = "neural.test"
    bl_label = "Run Tests"
    bl_options = {'REGISTER'}

    def execute(self, context):
        dataset = _neural_data['dataset']
        if dataset is None:
            self.report({'ERROR'}, "Load data first (click 'Load Saved Data')")
            return {'CANCELLED'}

        from .neural_network import run_test_suite
        from .neural_network.context import normalize_input

        try:
            # Normalize inputs same as training!
            train_inputs_norm = normalize_input(dataset['train_inputs'])
            test_inputs_norm = normalize_input(dataset['test_inputs'])

            # Get FK data for proper loss calculation
            train_targets = dataset.get('train_effector_targets')
            train_roots = dataset.get('train_root_positions')
            test_targets = dataset.get('test_effector_targets')
            test_roots = dataset.get('test_root_positions')

            # Get rotation data for orientation tests
            train_effector_rots = dataset.get('train_effector_rotations')
            test_effector_rots = dataset.get('test_effector_rotations')
            train_root_fwd = dataset.get('train_root_forwards')
            train_root_up = dataset.get('train_root_ups')
            test_root_fwd = dataset.get('test_root_forwards')
            test_root_up = dataset.get('test_root_ups')

            report = run_test_suite(
                train_inputs_norm,
                dataset['train_outputs'],
                test_inputs_norm,
                dataset['test_outputs'],
                train_targets,
                train_roots,
                test_targets,
                test_roots,
                train_effector_rots,
                test_effector_rots,
                train_root_fwd,
                train_root_up,
                test_root_fwd,
                test_root_up,
            )

            if report.failed == 0:
                self.report({'INFO'}, f"All {report.total} tests passed!")
            else:
                self.report({'WARNING'}, f"{report.passed}/{report.total} tests passed")

            return {'FINISHED'}

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, f"Tests failed: {e}")
            return {'CANCELLED'}


class NEURAL_OT_ReloadWeights(Operator):
    """Reload trained weights from disk (after standalone training)"""
    bl_idname = "neural.reload_weights"
    bl_label = "Reload Weights"
    bl_options = {'REGISTER'}

    def execute(self, context):
        from .neural_network import reset_network
        from .neural_network.config import BEST_WEIGHTS_PATH
        import os

        if not os.path.exists(BEST_WEIGHTS_PATH):
            self.report({'ERROR'}, "No weights file found. Train first.")
            return {'CANCELLED'}

        net = reset_network()
        if net.load():
            self.report({'INFO'}, f"Weights reloaded (best loss: {net.best_loss:.6f})")
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "Failed to load weights")
            return {'CANCELLED'}


class NEURAL_OT_Reset(Operator):
    """Reset network to random weights"""
    bl_idname = "neural.reset"
    bl_label = "Reset Network"
    bl_options = {'REGISTER'}

    def execute(self, context):
        from .neural_network import reset_network

        reset_network()
        _neural_data['last_report'] = None

        self.report({'INFO'}, "Network reset to random weights")
        return {'FINISHED'}


class NEURAL_OT_SaveData(Operator):
    """Save training data to disk (persists across sessions)"""
    bl_idname = "neural.save_data"
    bl_label = "Save Training Data"
    bl_options = {'REGISTER'}

    def execute(self, context):
        if _neural_data['dataset'] is None:
            self.report({'ERROR'}, "No data to save. Extract first.")
            return {'CANCELLED'}

        from .neural_network.config import DATA_DIR
        import os
        import numpy as np

        os.makedirs(DATA_DIR, exist_ok=True)
        path = os.path.join(DATA_DIR, "training_data.npz")

        dataset = _neural_data['dataset']
        np.savez_compressed(path, **dataset)

        total = len(dataset.get('train_inputs', [])) + len(dataset.get('test_inputs', []))
        self.report({'INFO'}, f"Saved {total} samples to {path}")
        return {'FINISHED'}


class NEURAL_OT_LoadData(Operator):
    """Load training data from disk"""
    bl_idname = "neural.load_data"
    bl_label = "Load Training Data"
    bl_options = {'REGISTER'}

    def execute(self, context):
        from .neural_network.config import DATA_DIR
        import os
        import numpy as np

        path = os.path.join(DATA_DIR, "training_data.npz")

        if not os.path.exists(path):
            self.report({'ERROR'}, f"No saved data at {path}")
            return {'CANCELLED'}

        try:
            data = np.load(path, allow_pickle=True)
            dataset = {key: data[key] for key in data.files}

            _neural_data['dataset'] = dataset
            train_count = len(dataset.get('train_inputs', []))
            test_count = len(dataset.get('test_inputs', []))
            _neural_data['samples_extracted'] = train_count + test_count

            self.report({'INFO'}, f"Loaded {train_count} train + {test_count} test samples")
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Load failed: {e}")
            return {'CANCELLED'}


class NEURAL_OT_AppendData(Operator):
    """Extract new animations and ADD to existing saved data"""
    bl_idname = "neural.append_data"
    bl_label = "Append New Data"
    bl_options = {'REGISTER'}

    def execute(self, context):
        scene = context.scene
        armature = scene.target_armature

        if not armature:
            self.report({'ERROR'}, "Set target armature first")
            return {'CANCELLED'}

        from .neural_network import AnimationDataExtractor
        from .neural_network.config import DATA_DIR
        import os
        import numpy as np

        # Load existing data if present
        path = os.path.join(DATA_DIR, "training_data.npz")
        existing_dataset = None
        if os.path.exists(path):
            try:
                data = np.load(path, allow_pickle=True)
                existing_dataset = {key: data[key] for key in data.files}
            except:
                pass

        # Extract new data
        try:
            extractor = AnimationDataExtractor(armature)
            new_count = extractor.extract_from_all_actions()

            if new_count == 0:
                self.report({'WARNING'}, "No new animation data extracted")
                return {'CANCELLED'}

            new_dataset = extractor.get_train_test_split(augment=True, augment_factor=2)

            # Merge if existing
            if existing_dataset is not None:
                for key in new_dataset:
                    if key in existing_dataset and len(existing_dataset[key]) > 0:
                        new_dataset[key] = np.concatenate([
                            existing_dataset[key],
                            new_dataset[key]
                        ], axis=0)

            # Save merged
            os.makedirs(DATA_DIR, exist_ok=True)
            np.savez_compressed(path, **new_dataset)

            # Update in-memory
            _neural_data['dataset'] = new_dataset
            total = len(new_dataset['train_inputs']) + len(new_dataset['test_inputs'])
            _neural_data['samples_extracted'] = total

            self.report({'INFO'}, f"Appended! Total: {total} samples")
            return {'FINISHED'}

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, f"Append failed: {e}")
            return {'CANCELLED'}


# =============================================================================
# NEURAL IK DIAGNOSTICS
# =============================================================================

# Store diagnostic results for UI display
_diagnostic_results = {
    'last_run': None,
    'rest_pos_ok': None,
    'bone_len_ok': None,
    'fk_ok': None,
    'root_rot_ok': None,
    'extraction_ok': None,
    'log': [],
}


def get_diagnostic_results() -> dict:
    """Get last diagnostic results for UI display."""
    return _diagnostic_results.copy()


def _diag_log(category: str, message: str, level: str = "INFO"):
    """Add to diagnostic log."""
    entry = f"[{level}] [{category}] {message}"
    _diagnostic_results['log'].append(entry)
    print(entry)


def _verify_rest_positions(armature) -> bool:
    """Compare config REST_POSITIONS with actual Blender rest positions."""
    from .neural_network.config import CONTROLLED_BONES, REST_POSITIONS
    import numpy as np

    _diag_log("REST_POS", "=" * 50)
    _diag_log("REST_POS", "VERIFYING REST POSITIONS")

    arm_data = armature.data
    arm_matrix = armature.matrix_world
    max_error = 0.0

    for bone_name in CONTROLLED_BONES:
        bone = arm_data.bones.get(bone_name)
        if bone is None:
            _diag_log("REST_POS", f"  {bone_name}: MISSING", "ERROR")
            continue

        blender_world = arm_matrix @ bone.head_local
        config_pos = np.array(REST_POSITIONS[bone_name])
        blender_arr = np.array([blender_world.x, blender_world.y, blender_world.z])
        error = np.linalg.norm(blender_arr - config_pos)
        max_error = max(max_error, error)

        if error > 0.001:
            _diag_log("REST_POS", f"  {bone_name}: Error={error:.4f}m", "WARN")

    passed = max_error < 0.01
    _diag_log("REST_POS", f"Max error: {max_error:.4f}m - {'PASS' if passed else 'FAIL'}")
    return passed


def _verify_bone_lengths(armature) -> bool:
    """Compare config BONE_LENGTHS with actual Blender bone lengths."""
    from .neural_network.config import CONTROLLED_BONES, BONE_LENGTHS

    _diag_log("BONE_LEN", "=" * 50)
    _diag_log("BONE_LEN", "VERIFYING BONE LENGTHS")

    arm_data = armature.data
    max_error_pct = 0.0

    for bone_name in CONTROLLED_BONES:
        bone = arm_data.bones.get(bone_name)
        if bone is None:
            continue

        blender_len = bone.length
        config_len = BONE_LENGTHS[bone_name]
        error_pct = abs(blender_len - config_len) / config_len * 100
        max_error_pct = max(max_error_pct, error_pct)

        if error_pct > 1:
            _diag_log("BONE_LEN", f"  {bone_name}: {error_pct:.1f}% diff", "WARN")

    passed = max_error_pct < 5
    _diag_log("BONE_LEN", f"Max error: {max_error_pct:.1f}% - {'PASS' if passed else 'FAIL'}")
    return passed


def _verify_fk_against_blender(armature) -> bool:
    """Compare our FK computation with Blender's actual bone positions."""
    from mathutils import Vector, Matrix
    from .neural_network.config import CONTROLLED_BONES, END_EFFECTORS, NUM_BONES, BONE_TO_INDEX
    from .neural_network.forward_kinematics import forward_kinematics, get_effector_positions
    import numpy as np

    _diag_log("FK_VERIFY", "=" * 50)
    _diag_log("FK_VERIFY", "VERIFYING FK COMPUTATION")

    pose_bones = armature.pose.bones
    arm_matrix = armature.matrix_world

    hips = pose_bones.get("Hips")
    if hips is None:
        _diag_log("FK_VERIFY", "No Hips bone!", "ERROR")
        return False

    # Get root info - armature world rotation (not hips bone matrix!)
    hips_world_pos = arm_matrix @ hips.head
    root_pos = np.array([hips_world_pos.x, hips_world_pos.y, hips_world_pos.z])

    # root_rot = armature world rotation
    arm_rot_3x3 = arm_matrix.to_3x3()
    root_rot = np.array([
        [arm_rot_3x3[0][0], arm_rot_3x3[0][1], arm_rot_3x3[0][2]],
        [arm_rot_3x3[1][0], arm_rot_3x3[1][1], arm_rot_3x3[1][2]],
        [arm_rot_3x3[2][0], arm_rot_3x3[2][1], arm_rot_3x3[2][2]],
    ], dtype=np.float32)

    # Extract current rotations
    rotations = np.zeros((NUM_BONES, 3), dtype=np.float32)
    for i, bone_name in enumerate(CONTROLLED_BONES):
        bone = pose_bones.get(bone_name)
        if bone:
            if bone.rotation_mode == 'QUATERNION':
                quat = bone.rotation_quaternion
            else:
                quat = bone.rotation_euler.to_quaternion()
            axis, angle = quat.to_axis_angle()
            rotations[i] = [axis.x * angle, axis.y * angle, axis.z * angle]

    # Run our FK
    fk_positions, _ = forward_kinematics(rotations, root_pos, root_rot)

    # Compare effectors
    _diag_log("FK_VERIFY", "Effector comparison:")
    errors = []
    for name in END_EFFECTORS:
        bone = pose_bones.get(name)
        blender_pos = arm_matrix @ bone.head
        blender_arr = np.array([blender_pos.x, blender_pos.y, blender_pos.z])
        fk_pos = fk_positions[BONE_TO_INDEX[name]]
        error = np.linalg.norm(blender_arr - fk_pos)
        errors.append(error)
        status = "OK" if error < 0.02 else "ERROR"
        _diag_log("FK_VERIFY", f"  {name}: {error*100:.1f}cm [{status}]")

    mean_error = np.mean(errors)
    passed = mean_error < 0.05
    _diag_log("FK_VERIFY", f"Mean effector error: {mean_error*100:.1f}cm - {'PASS' if passed else 'FAIL'}")
    return passed


def _verify_root_rotation_handling(armature) -> bool:
    """Test if Hips rotation is being double-applied."""
    from mathutils import Quaternion
    from .neural_network.config import NUM_BONES, BONE_TO_INDEX
    from .neural_network.forward_kinematics import forward_kinematics
    import numpy as np

    _diag_log("ROOT_ROT", "=" * 50)
    _diag_log("ROOT_ROT", "TESTING ROOT ROTATION HANDLING")

    pose_bones = armature.pose.bones
    arm_matrix = armature.matrix_world
    hips = pose_bones.get("Hips")

    if hips is None:
        _diag_log("ROOT_ROT", "No Hips bone!", "ERROR")
        return False

    # Store original
    original_mode = hips.rotation_mode
    if original_mode != 'QUATERNION':
        hips.rotation_mode = 'QUATERNION'
    original_quat = hips.rotation_quaternion.copy()

    # Apply test rotation: 45° around Y
    test_angle = np.pi / 4
    test_quat = Quaternion((0, 1, 0), test_angle)
    hips.rotation_quaternion = test_quat
    bpy.context.view_layer.update()

    _diag_log("ROOT_ROT", "Applied 45° Y rotation to Hips")

    # Get root info - armature world rotation (not hips bone matrix!)
    hips_world_pos = arm_matrix @ hips.head
    root_pos = np.array([hips_world_pos.x, hips_world_pos.y, hips_world_pos.z])

    # root_rot = armature world rotation (FK applies REST_ORIENTATIONS internally)
    arm_rot_3x3 = arm_matrix.to_3x3()
    root_rot = np.array([
        [arm_rot_3x3[0][0], arm_rot_3x3[0][1], arm_rot_3x3[0][2]],
        [arm_rot_3x3[1][0], arm_rot_3x3[1][1], arm_rot_3x3[1][2]],
        [arm_rot_3x3[2][0], arm_rot_3x3[2][1], arm_rot_3x3[2][2]],
    ], dtype=np.float32)

    # Extract Hips local rotation
    axis, angle = test_quat.to_axis_angle()
    hips_local_rot = np.array([axis.x * angle, axis.y * angle, axis.z * angle])

    # Build rotation array
    rotations = np.zeros((NUM_BONES, 3), dtype=np.float32)
    rotations[0] = hips_local_rot

    # Run FK
    fk_positions, _ = forward_kinematics(rotations, root_pos, root_rot)

    # Compare Spine position
    spine = pose_bones.get("Spine")
    double_applied = False
    if spine:
        blender_spine = arm_matrix @ spine.head
        blender_arr = np.array([blender_spine.x, blender_spine.y, blender_spine.z])
        fk_spine = fk_positions[BONE_TO_INDEX["Spine"]]
        error = np.linalg.norm(blender_arr - fk_spine)

        _diag_log("ROOT_ROT", f"Spine position error: {error*100:.1f}cm")

        if error > 0.05:  # 5cm threshold (in meters)
            _diag_log("ROOT_ROT", "ROOT ROTATION IS DOUBLE-APPLIED!", "ERROR")
            _diag_log("ROOT_ROT", "Fix: FK should not apply Hips local_rot when root_rot already includes it", "ERROR")
            double_applied = True
        else:
            _diag_log("ROOT_ROT", "Root rotation handling OK")

    # Restore
    hips.rotation_quaternion = original_quat
    hips.rotation_mode = original_mode
    bpy.context.view_layer.update()

    return not double_applied


def _verify_data_extraction(armature) -> bool:
    """Verify round-trip: extract → FK → compare."""
    from .neural_network.config import CONTROLLED_BONES, END_EFFECTORS, NUM_BONES, BONE_TO_INDEX
    from .neural_network.forward_kinematics import forward_kinematics, get_effector_positions
    import numpy as np

    _diag_log("EXTRACT", "=" * 50)
    _diag_log("EXTRACT", "VERIFYING DATA EXTRACTION ROUND-TRIP")

    pose_bones = armature.pose.bones
    arm_matrix = armature.matrix_world

    hips = pose_bones.get("Hips")
    hips_world_pos = arm_matrix @ hips.head
    root_pos = np.array([hips_world_pos.x, hips_world_pos.y, hips_world_pos.z])

    # root_rot = armature world rotation (FK applies REST_ORIENTATIONS internally)
    arm_rot_3x3 = arm_matrix.to_3x3()
    root_rot = np.array([
        [arm_rot_3x3[0][0], arm_rot_3x3[0][1], arm_rot_3x3[0][2]],
        [arm_rot_3x3[1][0], arm_rot_3x3[1][1], arm_rot_3x3[1][2]],
        [arm_rot_3x3[2][0], arm_rot_3x3[2][1], arm_rot_3x3[2][2]],
    ], dtype=np.float32)

    # Extract bone rotations
    rotations = np.zeros((NUM_BONES, 3), dtype=np.float32)
    for i, bone_name in enumerate(CONTROLLED_BONES):
        bone = pose_bones.get(bone_name)
        if bone:
            if bone.rotation_mode == 'QUATERNION':
                quat = bone.rotation_quaternion
            else:
                quat = bone.rotation_euler.to_quaternion()
            axis, angle = quat.to_axis_angle()
            rotations[i] = [axis.x * angle, axis.y * angle, axis.z * angle]

    # Extract effector targets
    effector_targets = []
    for name in END_EFFECTORS:
        bone = pose_bones.get(name)
        world_pos = arm_matrix @ bone.head
        effector_targets.append([world_pos.x, world_pos.y, world_pos.z])
    effector_targets = np.array(effector_targets)

    # Run FK
    fk_positions, _ = forward_kinematics(rotations, root_pos, root_rot)
    fk_effectors = get_effector_positions(fk_positions)

    # Compare
    errors = []
    for i, name in enumerate(END_EFFECTORS):
        error = np.linalg.norm(effector_targets[i] - fk_effectors[i])
        errors.append(error)
        status = "OK" if error < 0.02 else "ERROR"
        _diag_log("EXTRACT", f"  {name}: {error*100:.1f}cm [{status}]")

    mean_error = np.mean(errors)
    passed = mean_error < 0.03
    _diag_log("EXTRACT", f"Mean error: {mean_error*100:.1f}cm - {'PASS' if passed else 'FAIL'}")
    return passed


class NEURAL_OT_RunDiagnostics(Operator):
    """Run diagnostic tests to verify Neural IK pipeline"""
    bl_idname = "neural.run_diagnostics"
    bl_label = "Run Diagnostics"
    bl_options = {'REGISTER'}

    def execute(self, context):
        import time

        armature = context.scene.target_armature
        if not armature:
            self.report({'ERROR'}, "Set target armature first")
            return {'CANCELLED'}

        # Clear previous results
        _diagnostic_results['log'] = []
        _diagnostic_results['last_run'] = time.strftime("%Y-%m-%d %H:%M:%S")

        _diag_log("MAIN", "=" * 60)
        _diag_log("MAIN", "NEURAL IK DIAGNOSTIC SUITE")
        _diag_log("MAIN", "=" * 60)

        # Run all tests
        _diagnostic_results['rest_pos_ok'] = _verify_rest_positions(armature)
        _diagnostic_results['bone_len_ok'] = _verify_bone_lengths(armature)
        _diagnostic_results['fk_ok'] = _verify_fk_against_blender(armature)
        _diagnostic_results['root_rot_ok'] = _verify_root_rotation_handling(armature)
        _diagnostic_results['extraction_ok'] = _verify_data_extraction(armature)

        # Summary
        _diag_log("MAIN", "=" * 60)
        _diag_log("MAIN", "SUMMARY")
        _diag_log("MAIN", f"  Rest Positions: {'PASS' if _diagnostic_results['rest_pos_ok'] else 'FAIL'}")
        _diag_log("MAIN", f"  Bone Lengths:   {'PASS' if _diagnostic_results['bone_len_ok'] else 'FAIL'}")
        _diag_log("MAIN", f"  FK Computation: {'PASS' if _diagnostic_results['fk_ok'] else 'FAIL'}")
        _diag_log("MAIN", f"  Root Rotation:  {'PASS' if _diagnostic_results['root_rot_ok'] else 'FAIL'}")
        _diag_log("MAIN", f"  Data Extraction:{'PASS' if _diagnostic_results['extraction_ok'] else 'FAIL'}")
        _diag_log("MAIN", "=" * 60)

        # Report
        all_passed = all([
            _diagnostic_results['rest_pos_ok'],
            _diagnostic_results['bone_len_ok'],
            _diagnostic_results['fk_ok'],
            _diagnostic_results['root_rot_ok'],
            _diagnostic_results['extraction_ok'],
        ])

        if all_passed:
            self.report({'INFO'}, "All diagnostics passed!")
        else:
            self.report({'WARNING'}, "Some diagnostics failed - check console")

        return {'FINISHED'}


# =============================================================================
# TRAINED MODEL VERIFICATION
# =============================================================================

_verification_results = {
    'last_run': None,
    'tests': [],
    'summary': None,
}


def get_verification_results() -> dict:
    """Get verification results for UI display."""
    return _verification_results.copy()


class NEURAL_OT_VerifyModel(Operator):
    """Verify trained model accuracy using test data - Human readable results"""
    bl_idname = "neural.verify_model"
    bl_label = "Verify Trained Model"
    bl_options = {'REGISTER'}

    def execute(self, context):
        import time
        import numpy as np
        from .neural_network import get_network
        from .neural_network.config import (
            BEST_WEIGHTS_PATH, DATA_DIR, END_EFFECTORS,
            CONTROLLED_BONES, BONE_TO_INDEX, NUM_BONES
        )
        from .neural_network.context import normalize_input
        from .neural_network.forward_kinematics import (
            forward_kinematics,
            get_effector_positions,
            compute_fk_loss_with_orientation,
        )
        import os

        armature = context.scene.target_armature
        if not armature:
            self.report({'ERROR'}, "Set target armature first")
            return {'CANCELLED'}

        # Check weights exist
        if not os.path.exists(BEST_WEIGHTS_PATH):
            self.report({'ERROR'}, "No trained weights found. Train first!")
            return {'CANCELLED'}

        # Load test data
        data_path = os.path.join(DATA_DIR, "training_data.npz")
        if not os.path.exists(data_path):
            self.report({'ERROR'}, "No training data found. Extract data first!")
            return {'CANCELLED'}

        data = np.load(data_path, allow_pickle=True)

        # Check for test data
        if 'test_inputs' not in data:
            self.report({'ERROR'}, "Training data missing test split. Re-extract data.")
            return {'CANCELLED'}

        test_inputs = data['test_inputs']
        test_outputs = data['test_outputs']
        test_targets = data['test_effector_targets']
        test_target_rots = data.get('test_effector_rotations')
        test_root_pos = data['test_root_positions']
        test_root_fwd = data.get('test_root_forwards')
        test_root_up = data.get('test_root_ups')

        n_samples = len(test_inputs)
        if n_samples == 0:
            self.report({'ERROR'}, "No test samples in data")
            return {'CANCELLED'}

        # Clear previous results
        _verification_results['tests'] = []
        _verification_results['last_run'] = time.strftime("%Y-%m-%d %H:%M:%S")

        # Get network
        net = get_network()
        if not net.load():
            self.report({'ERROR'}, "Failed to load weights")
            return {'CANCELLED'}

        print("\n" + "=" * 70)
        print(" TRAINED MODEL VERIFICATION")
        print(" Testing neural network predictions against ground truth")
        print("=" * 70)

        # =====================================================================
        # DIAGNOSTIC: Verify FK produces correct positions from ground truth
        # =====================================================================
        print("\n" + "-" * 70)
        print(" FK SANITY CHECK (ground truth rotations → FK → positions)")
        print("-" * 70)

        gt_fk_errors = []
        diagnostic_samples = min(50, n_samples)
        diagnostic_indices = np.random.choice(n_samples, diagnostic_samples, replace=False)

        for idx in diagnostic_indices:
            gt_output = test_outputs[idx]
            target_pos = test_targets[idx].reshape(5, 3)
            root_pos = test_root_pos[idx]

            # Build root rotation matrix
            if test_root_fwd is not None and test_root_up is not None:
                root_fwd = test_root_fwd[idx]
                root_up_vec = test_root_up[idx]
                root_fwd = root_fwd / (np.linalg.norm(root_fwd) + 1e-8)
                root_up_vec = root_up_vec / (np.linalg.norm(root_up_vec) + 1e-8)
                root_right = np.cross(root_fwd, root_up_vec)
                root_right = root_right / (np.linalg.norm(root_right) + 1e-8)
                root_up_vec = np.cross(root_right, root_fwd)
                root_rot = np.array([
                    [root_right[0], root_fwd[0], root_up_vec[0]],
                    [root_right[1], root_fwd[1], root_up_vec[1]],
                    [root_right[2], root_fwd[2], root_up_vec[2]],
                ], dtype=np.float32)
            else:
                root_rot = np.eye(3, dtype=np.float32)

            # Run FK on GROUND TRUTH rotations
            gt_rots = gt_output.reshape(NUM_BONES, 3)
            fk_pos, _ = forward_kinematics(gt_rots, root_pos, root_rot)
            fk_effector_pos = get_effector_positions(fk_pos)

            # Error between FK output and target positions
            for i in range(5):
                err = np.linalg.norm(fk_effector_pos[i] - target_pos[i])
                gt_fk_errors.append(err)

        gt_fk_mean = np.mean(gt_fk_errors) * 100  # to cm
        gt_fk_max = np.max(gt_fk_errors) * 100

        if gt_fk_mean > 5.0:
            print(f"  ⚠️  FK MISMATCH: Ground truth rotations produce positions {gt_fk_mean:.1f}cm off target!")
            print(f"      This means FK computation doesn't match how Blender computes bone positions.")
            print(f"      Max error: {gt_fk_max:.1f}cm")
            print("")
            print("  The FK function needs to be fixed before network accuracy can be evaluated.")
        else:
            print(f"  ✓ FK VALID: Ground truth rotations produce positions within {gt_fk_mean:.1f}cm of target")
            print(f"      (Max error: {gt_fk_max:.1f}cm)")

        print("-" * 70)

        # Normalize inputs
        test_inputs_norm = normalize_input(test_inputs)

        # Run predictions on sampled test cases
        all_pos_errors = {name: [] for name in END_EFFECTORS}
        all_rot_errors = []

        sample_size = min(100, n_samples)
        sample_indices = np.random.choice(n_samples, sample_size, replace=False)

        for idx in sample_indices:
            inp = test_inputs_norm[idx]
            gt_output = test_outputs[idx]
            target_pos = test_targets[idx].reshape(5, 3)
            target_rots = test_target_rots[idx].reshape(5, 3) if test_target_rots is not None else None
            root_pos = test_root_pos[idx]

            # Build root rotation matrix
            if test_root_fwd is not None and test_root_up is not None:
                root_fwd = test_root_fwd[idx]
                root_up = test_root_up[idx]
                root_fwd = root_fwd / (np.linalg.norm(root_fwd) + 1e-8)
                root_up = root_up / (np.linalg.norm(root_up) + 1e-8)
                root_right = np.cross(root_fwd, root_up)
                root_right = root_right / (np.linalg.norm(root_right) + 1e-8)
                root_up = np.cross(root_right, root_fwd)
                root_rot = np.array([
                    [root_right[0], root_fwd[0], root_up[0]],
                    [root_right[1], root_fwd[1], root_up[1]],
                    [root_right[2], root_fwd[2], root_up[2]],
                ], dtype=np.float32)
            else:
                root_rot = np.eye(3, dtype=np.float32)

            # Predict (clamped to limits)
            pred_output, _ = net.predict_clamped(inp)
            pred_output = pred_output.reshape(1, -1)

            # Compute FK-based errors with orientation if targets available
            pos_loss, ori_loss, pos_errs, ori_errs = compute_fk_loss_with_orientation(
                pred_output,
                target_pos.reshape(1, 5, 3),
                target_rots.reshape(1, 5, 3) if target_rots is not None else np.zeros((1, 5, 3), dtype=np.float32),
                root_pos.reshape(1, 3),
                root_rot.reshape(1, 3, 3),
            )

            # Position errors per effector (meters)
            for i, name in enumerate(END_EFFECTORS):
                all_pos_errors[name].append(pos_errs[0][i])

            # Rotation error per effector (radians)
            all_rot_errors.extend(ori_errs.flatten().tolist())

        # Compute statistics
        print("\n" + "-" * 70)
        print(" POSITION ACCURACY (Lower is better)")
        print("-" * 70)

        total_passed = 0
        total_tests = 0
        test_results = []

        for name in END_EFFECTORS:
            errors = np.array(all_pos_errors[name])
            mean_err = np.mean(errors) * 100
            std_err = np.std(errors) * 100
            max_err = np.max(errors) * 100

            passed = mean_err < 15.0
            total_tests += 1
            if passed:
                total_passed += 1

            status = "PASS" if passed else "FAIL"
            print(f"  {name:12s}: Mean={mean_err:6.2f}cm  Std={std_err:5.2f}cm  Max={max_err:5.1f}cm  [{status}]")

            test_results.append({
                'name': f"{name} Position",
                'mean_error_cm': round(mean_err, 2),
                'std_cm': round(std_err, 2),
                'max_cm': round(max_err, 2),
                'passed': passed,
            })

        # Overall position
        all_errors = []
        for errs in all_pos_errors.values():
            all_errors.extend(errs)
        overall_mean = np.mean(all_errors) * 100
        overall_rmse = np.sqrt(np.mean(np.array(all_errors) ** 2)) * 100

        print("-" * 70)
        print(f"  OVERALL:      Mean={overall_mean:6.2f}cm  RMSE={overall_rmse:5.2f}cm")

        # Rotation accuracy
        print("\n" + "-" * 70)
        print(" ROTATION ACCURACY")
        print("-" * 70)

        mean_rot_err = np.mean(all_rot_errors)
        rot_err_deg = np.degrees(mean_rot_err)
        rot_passed = rot_err_deg < 30.0
        total_tests += 1
        if rot_passed:
            total_passed += 1

        status = "PASS" if rot_passed else "FAIL"
        print(f"  Mean Rotation Error: {rot_err_deg:.2f}°  [{status}]")

        test_results.append({
            'name': "Rotation Accuracy",
            'mean_error_deg': round(rot_err_deg, 2),
            'passed': rot_passed,
        })

        # Summary
        print("\n" + "=" * 70)
        print(f" SUMMARY: {total_passed}/{total_tests} tests passed")
        print("=" * 70)

        accuracy_pct = (total_passed / total_tests) * 100 if total_tests > 0 else 0

        if accuracy_pct == 100 and overall_rmse < 5.0:
            grade = "EXCELLENT"
        elif accuracy_pct >= 80:
            grade = "GOOD"
        elif accuracy_pct >= 50:
            grade = "FAIR"
        else:
            grade = "POOR"

        print(f"\n Position RMSE: {overall_rmse:.2f}cm")
        print(f" Rotation Error: {rot_err_deg:.2f}°")
        print(f" Grade: {grade}")
        print("=" * 70 + "\n")

        # Store results
        _verification_results['tests'] = test_results
        _verification_results['summary'] = {
            'passed': total_passed,
            'total': total_tests,
            'accuracy_pct': accuracy_pct,
            'position_rmse_cm': round(overall_rmse, 2),
            'rotation_error_deg': round(rot_err_deg, 2),
            'grade': grade,
            'samples_tested': len(sample_indices),
        }

        if accuracy_pct >= 80:
            self.report({'INFO'}, f"Verification: {grade} - {overall_rmse:.1f}cm RMSE")
        else:
            self.report({'WARNING'}, f"Verification: {grade} - {overall_rmse:.1f}cm RMSE")

        return {'FINISHED'}


class NEURAL_OT_ApplyTestPose(Operator):
    """Apply network prediction to armature for visual inspection"""
    bl_idname = "neural.apply_test_pose"
    bl_label = "Apply Prediction"
    bl_options = {'REGISTER', 'UNDO'}

    sample_index: bpy.props.IntProperty(name="Sample", default=0, min=0)

    def execute(self, context):
        import numpy as np
        from mathutils import Quaternion
        from .neural_network import reset_network
        from .neural_network.config import DATA_DIR, CONTROLLED_BONES, NUM_BONES, BEST_WEIGHTS_PATH
        from .neural_network.context import normalize_input
        import os

        armature = context.scene.target_armature
        if not armature:
            self.report({'ERROR'}, "Set target armature first")
            return {'CANCELLED'}

        if not os.path.exists(BEST_WEIGHTS_PATH):
            self.report({'ERROR'}, "No trained weights")
            return {'CANCELLED'}

        data_path = os.path.join(DATA_DIR, "training_data.npz")
        if not os.path.exists(data_path):
            self.report({'ERROR'}, "No test data")
            return {'CANCELLED'}

        data = np.load(data_path, allow_pickle=True)
        test_inputs = data['test_inputs']
        test_outputs = data['test_outputs']  # Load GT for comparison

        if len(test_inputs) == 0:
            self.report({'ERROR'}, "No test samples")
            return {'CANCELLED'}

        idx = self.sample_index % len(test_inputs)

        net = reset_network()
        net.load()

        # Normalize input
        inp = normalize_input(test_inputs[idx:idx+1])[0]
        # Predict with clamping to limits
        pred = net.predict_clamped(inp)[0]
        pred_rots = pred.reshape(NUM_BONES, 3)

        # Get ground truth for comparison
        gt_rots = test_outputs[idx].reshape(NUM_BONES, 3)

        pose_bones = armature.pose.bones

        # Key bones to debug (limbs that show criss-crossing)
        debug_bones = ["LeftArm", "LeftForeArm", "RightArm", "RightForeArm",
                       "LeftThigh", "LeftShin", "RightThigh", "RightShin"]

        print("\n" + "="*70)
        print(f"PREDICTION DEBUG - Sample {idx}")
        print("="*70)
        print(f"\nArmature: {armature.name}")
        print(f"World Location: {armature.location}")
        print(f"World Rotation: {armature.rotation_euler}")
        print(f"World Scale: {armature.scale}")

        print("\n" + "-"*70)
        print("KEY LIMB BONES - PREDICTION vs GROUND TRUTH")
        print("-"*70)
        print(f"{'Bone':<15} {'Pred Axis-Angle':<30} {'GT Axis-Angle':<30} {'Diff':<10}")
        print("-"*70)

        for i, bone_name in enumerate(CONTROLLED_BONES):
            bone = pose_bones.get(bone_name)
            if bone is None:
                continue

            axis_angle = pred_rots[i]
            gt_aa = gt_rots[i]
            angle = np.linalg.norm(axis_angle)

            if angle > 1e-6:
                axis = axis_angle / angle
                quat = Quaternion((axis[0], axis[1], axis[2]), angle)
            else:
                quat = Quaternion((1, 0, 0, 0))

            bone.rotation_mode = 'QUATERNION'
            bone.rotation_quaternion = quat.normalized()

            # Debug output for key bones
            if bone_name in debug_bones:
                diff = np.linalg.norm(axis_angle - gt_aa)
                pred_str = f"[{axis_angle[0]:+.3f}, {axis_angle[1]:+.3f}, {axis_angle[2]:+.3f}]"
                gt_str = f"[{gt_aa[0]:+.3f}, {gt_aa[1]:+.3f}, {gt_aa[2]:+.3f}]"
                print(f"{bone_name:<15} {pred_str:<30} {gt_str:<30} {diff:.3f}")

        print("\n" + "-"*70)
        print("DETAILED QUATERNION CONVERSION (Key Bones)")
        print("-"*70)

        for bone_name in debug_bones:
            i = CONTROLLED_BONES.index(bone_name)
            axis_angle = pred_rots[i]
            angle = np.linalg.norm(axis_angle)

            print(f"\n{bone_name}:")
            print(f"  Raw axis-angle: [{axis_angle[0]:+.4f}, {axis_angle[1]:+.4f}, {axis_angle[2]:+.4f}]")
            print(f"  Angle (radians): {angle:.4f} ({np.degrees(angle):.1f} deg)")

            if angle > 1e-6:
                axis = axis_angle / angle
                print(f"  Normalized axis: [{axis[0]:+.4f}, {axis[1]:+.4f}, {axis[2]:+.4f}]")
                quat = Quaternion((axis[0], axis[1], axis[2]), angle)
                print(f"  Blender Quaternion(axis, angle): w={quat.w:.4f}, x={quat.x:.4f}, y={quat.y:.4f}, z={quat.z:.4f}")
            else:
                print(f"  (Near-zero rotation - using identity)")

            # Show what's actually on the bone now
            bone = pose_bones.get(bone_name)
            if bone:
                print(f"  Applied quaternion: w={bone.rotation_quaternion.w:.4f}, x={bone.rotation_quaternion.x:.4f}, y={bone.rotation_quaternion.y:.4f}, z={bone.rotation_quaternion.z:.4f}")

        print("\n" + "="*70)
        print("END PREDICTION DEBUG")
        print("="*70 + "\n")

        context.view_layer.update()
        self.report({'INFO'}, f"Applied PREDICTION for sample {idx}")
        return {'FINISHED'}


class NEURAL_OT_ApplyGroundTruth(Operator):
    """Apply ground truth pose from test data for comparison"""
    bl_idname = "neural.apply_ground_truth"
    bl_label = "Apply Ground Truth"
    bl_options = {'REGISTER', 'UNDO'}

    sample_index: bpy.props.IntProperty(name="Sample", default=0, min=0)

    def execute(self, context):
        import numpy as np
        from mathutils import Quaternion
        from .neural_network.config import DATA_DIR, CONTROLLED_BONES, NUM_BONES
        import os

        armature = context.scene.target_armature
        if not armature:
            self.report({'ERROR'}, "Set target armature first")
            return {'CANCELLED'}

        data_path = os.path.join(DATA_DIR, "training_data.npz")
        if not os.path.exists(data_path):
            self.report({'ERROR'}, "No test data")
            return {'CANCELLED'}

        data = np.load(data_path, allow_pickle=True)
        test_outputs = data['test_outputs']

        if len(test_outputs) == 0:
            self.report({'ERROR'}, "No test samples")
            return {'CANCELLED'}

        idx = self.sample_index % len(test_outputs)
        gt_rots = test_outputs[idx].reshape(NUM_BONES, 3)

        pose_bones = armature.pose.bones

        # Key bones to debug
        debug_bones = ["LeftArm", "LeftForeArm", "RightArm", "RightForeArm",
                       "LeftThigh", "LeftShin", "RightThigh", "RightShin"]

        print("\n" + "="*70)
        print(f"GROUND TRUTH DEBUG - Sample {idx}")
        print("="*70)
        print(f"\nArmature: {armature.name}")
        print(f"World Location: {armature.location}")
        print(f"World Rotation: {armature.rotation_euler}")
        print(f"World Scale: {armature.scale}")

        print("\n" + "-"*70)
        print("KEY LIMB BONES - GROUND TRUTH VALUES")
        print("-"*70)
        print(f"{'Bone':<15} {'GT Axis-Angle':<35} {'Angle (deg)':<15}")
        print("-"*70)

        for i, bone_name in enumerate(CONTROLLED_BONES):
            bone = pose_bones.get(bone_name)
            if bone is None:
                continue

            axis_angle = gt_rots[i]
            angle = np.linalg.norm(axis_angle)

            if angle > 1e-6:
                axis = axis_angle / angle
                quat = Quaternion((axis[0], axis[1], axis[2]), angle)
            else:
                quat = Quaternion((1, 0, 0, 0))

            bone.rotation_mode = 'QUATERNION'
            bone.rotation_quaternion = quat.normalized()

            # Debug output for key bones
            if bone_name in debug_bones:
                gt_str = f"[{axis_angle[0]:+.4f}, {axis_angle[1]:+.4f}, {axis_angle[2]:+.4f}]"
                print(f"{bone_name:<15} {gt_str:<35} {np.degrees(angle):.1f}")

        print("\n" + "-"*70)
        print("DETAILED QUATERNION CONVERSION (Key Bones)")
        print("-"*70)

        for bone_name in debug_bones:
            i = CONTROLLED_BONES.index(bone_name)
            axis_angle = gt_rots[i]
            angle = np.linalg.norm(axis_angle)

            print(f"\n{bone_name}:")
            print(f"  Raw axis-angle: [{axis_angle[0]:+.4f}, {axis_angle[1]:+.4f}, {axis_angle[2]:+.4f}]")
            print(f"  Angle (radians): {angle:.4f} ({np.degrees(angle):.1f} deg)")

            if angle > 1e-6:
                axis = axis_angle / angle
                print(f"  Normalized axis: [{axis[0]:+.4f}, {axis[1]:+.4f}, {axis[2]:+.4f}]")
                quat = Quaternion((axis[0], axis[1], axis[2]), angle)
                print(f"  Blender Quaternion(axis, angle): w={quat.w:.4f}, x={quat.x:.4f}, y={quat.y:.4f}, z={quat.z:.4f}")
            else:
                print(f"  (Near-zero rotation - using identity)")

            # Show what's actually on the bone now
            bone = pose_bones.get(bone_name)
            if bone:
                print(f"  Applied quaternion: w={bone.rotation_quaternion.w:.4f}, x={bone.rotation_quaternion.x:.4f}, y={bone.rotation_quaternion.y:.4f}, z={bone.rotation_quaternion.z:.4f}")

        print("\n" + "="*70)
        print("END GROUND TRUTH DEBUG")
        print("="*70 + "\n")

        context.view_layer.update()
        self.report({'INFO'}, f"Applied GROUND TRUTH for sample {idx}")
        return {'FINISHED'}


class NEURAL_OT_CompareVisual(Operator):
    """Toggle between prediction and ground truth for visual comparison"""
    bl_idname = "neural.compare_visual"
    bl_label = "Toggle Pred/Truth"
    bl_options = {'REGISTER', 'UNDO'}

    _current_idx = 0
    _showing_prediction = True

    def execute(self, context):
        import numpy as np
        from .neural_network.config import DATA_DIR
        import os

        data_path = os.path.join(DATA_DIR, "training_data.npz")
        if not os.path.exists(data_path):
            self.report({'ERROR'}, "No test data")
            return {'CANCELLED'}

        data = np.load(data_path, allow_pickle=True)
        n_samples = len(data['test_outputs'])

        if n_samples == 0:
            self.report({'ERROR'}, "No test samples")
            return {'CANCELLED'}

        cls = NEURAL_OT_CompareVisual

        if cls._showing_prediction:
            bpy.ops.neural.apply_ground_truth(sample_index=cls._current_idx)
            cls._showing_prediction = False
            self.report({'INFO'}, f"Sample {cls._current_idx}: GROUND TRUTH")
        else:
            cls._current_idx = (cls._current_idx + 1) % n_samples
            bpy.ops.neural.apply_test_pose(sample_index=cls._current_idx)
            cls._showing_prediction = True
            self.report({'INFO'}, f"Sample {cls._current_idx}: PREDICTION")

        return {'FINISHED'}


# =============================================================================
# DIAGNOSTIC: Validate entire FK pipeline
# =============================================================================

class NEURAL_OT_ValidatePipeline(Operator):
    """Validate the entire extraction->FK->Blender pipeline"""
    bl_idname = "neural.validate_pipeline"
    bl_label = "Validate FK Pipeline"
    bl_options = {'REGISTER'}

    def _print_correct_orientations(self, armature):
        """Print the correct REST_ORIENTATIONS values that should go in config.py"""
        import numpy as np
        from mathutils import Quaternion
        from .neural_network.config import CONTROLLED_BONES, PARENT_INDICES, NUM_BONES

        # Reset to rest pose
        for bone in armature.pose.bones:
            bone.rotation_quaternion = Quaternion((1, 0, 0, 0))
        import bpy
        bpy.context.view_layer.update()

        arm_matrix = armature.matrix_world
        arm_rot = arm_matrix.to_3x3()

        print("\n" + "=" * 70)
        print(" CORRECT REST_ORIENTATIONS FOR config.py")
        print("=" * 70)
        print(" Copy this into config.py to fix the FK computation:")
        print("-" * 70)

        # Collect orientations
        rest_orientations = np.zeros((NUM_BONES, 3, 3), dtype=np.float32)
        for i, bone_name in enumerate(CONTROLLED_BONES):
            bone = armature.pose.bones.get(bone_name)
            if bone:
                rest_mat = bone.bone.matrix_local.to_3x3()
                world_rest_mat = arm_rot @ rest_mat
                rest_orientations[i] = np.array([
                    [world_rest_mat[0][0], world_rest_mat[0][1], world_rest_mat[0][2]],
                    [world_rest_mat[1][0], world_rest_mat[1][1], world_rest_mat[1][2]],
                    [world_rest_mat[2][0], world_rest_mat[2][1], world_rest_mat[2][2]],
                ], dtype=np.float32)

        # Print as Python code
        print("\nREST_ORIENTATIONS = np.array([")
        for i, bone_name in enumerate(CONTROLLED_BONES):
            mat = rest_orientations[i]
            print(f"    # {bone_name}")
            print(f"    [[{mat[0,0]:+.6f}, {mat[0,1]:+.6f}, {mat[0,2]:+.6f}],")
            print(f"     [{mat[1,0]:+.6f}, {mat[1,1]:+.6f}, {mat[1,2]:+.6f}],")
            print(f"     [{mat[2,0]:+.6f}, {mat[2,1]:+.6f}, {mat[2,2]:+.6f}]],")
        print("], dtype=np.float32)")

        print("\n" + "=" * 70)

    def execute(self, context):
        import numpy as np
        from mathutils import Vector, Quaternion, Matrix
        from .neural_network.config import (
            CONTROLLED_BONES, REST_POSITIONS, END_EFFECTORS,
            PARENT_INDICES, NUM_BONES, BONE_TO_INDEX,
            REST_POSITIONS_ARRAY, REST_ORIENTATIONS, LOCAL_OFFSETS,
        )

        armature = context.scene.target_armature
        if not armature:
            self.report({'ERROR'}, "Set target armature first")
            return {'CANCELLED'}

        print("\n" + "=" * 70)
        print(" FK PIPELINE VALIDATION")
        print("=" * 70)
        print(" This test verifies that our FK math matches Blender's actual transforms.")
        print(" If this fails, training will NOT produce usable results.")
        print("=" * 70)

        # =====================================================================
        # TEST 1: Do config.py REST_POSITIONS match the actual armature?
        # =====================================================================
        print("\n" + "-" * 70)
        print(" TEST 1: REST_POSITIONS vs Actual Armature (in rest pose)")
        print("-" * 70)

        # We need to check rest pose, so temporarily clear all pose transforms
        # Save current pose first
        saved_rotations = {}
        for bone in armature.pose.bones:
            saved_rotations[bone.name] = bone.rotation_quaternion.copy()
            bone.rotation_quaternion = Quaternion((1, 0, 0, 0))  # Identity

        context.view_layer.update()

        arm_matrix = armature.matrix_world
        rest_errors = []
        print(f"\n {'Bone':<15} {'Config Pos':<25} {'Actual Pos':<25} {'Error':<10}")
        print("-" * 70)

        for bone_name in CONTROLLED_BONES:
            bone = armature.pose.bones.get(bone_name)
            if not bone:
                print(f" {bone_name:<15} MISSING IN ARMATURE!")
                continue

            config_pos = np.array(REST_POSITIONS[bone_name])
            actual_pos = arm_matrix @ bone.head
            actual_pos_np = np.array([actual_pos.x, actual_pos.y, actual_pos.z])

            error = np.linalg.norm(config_pos - actual_pos_np)
            rest_errors.append(error)

            status = "OK" if error < 0.01 else "BAD"
            config_str = f"({config_pos[0]:+.3f}, {config_pos[1]:+.3f}, {config_pos[2]:+.3f})"
            actual_str = f"({actual_pos_np[0]:+.3f}, {actual_pos_np[1]:+.3f}, {actual_pos_np[2]:+.3f})"
            print(f" {bone_name:<15} {config_str:<25} {actual_str:<25} {error*100:.1f}cm [{status}]")

        avg_rest_error = np.mean(rest_errors) * 100
        max_rest_error = np.max(rest_errors) * 100

        print("-" * 70)
        print(f" Average error: {avg_rest_error:.2f} cm")
        print(f" Max error: {max_rest_error:.2f} cm")

        if max_rest_error > 1.0:
            print("\n [FAIL] REST_POSITIONS in config.py don't match armature!")
            print(" Update the REST_POSITIONS dict in config.py with the actual values above.")
            test1_passed = False
        else:
            print("\n [PASS] REST_POSITIONS match armature.")
            test1_passed = True

        # =====================================================================
        # TEST 2: FK with IDENTITY rotations (should match rest positions exactly)
        # =====================================================================
        print("\n" + "-" * 70)
        print(" TEST 2: FK with Identity Rotations (sanity check)")
        print("-" * 70)
        print(" If all local rotations are identity, FK should output REST_POSITIONS exactly.")

        # Keep armature in rest pose (identity rotations)
        # root at its rest position, identity rotation
        from .neural_network.forward_kinematics import forward_kinematics

        identity_rotations = np.zeros((NUM_BONES, 3), dtype=np.float32)  # All zeros = identity
        hips_rest_pos = np.array(REST_POSITIONS["Hips"], dtype=np.float32)
        identity_root_rot = np.eye(3, dtype=np.float32)

        fk_positions_identity, _ = forward_kinematics(
            identity_rotations, hips_rest_pos, identity_root_rot, use_axis_angle=True
        )

        print(f"\n {'Bone':<15} {'REST_POSITIONS':<25} {'FK Output':<25} {'Error':<10}")
        print("-" * 70)

        identity_errors = []
        for i, bone_name in enumerate(CONTROLLED_BONES):
            rest_pos = REST_POSITIONS_ARRAY[i]
            fk_pos = fk_positions_identity[i]
            error = np.linalg.norm(fk_pos - rest_pos)
            identity_errors.append(error)

            status = "OK" if error < 0.001 else "BAD"
            rest_str = f"({rest_pos[0]:+.3f}, {rest_pos[1]:+.3f}, {rest_pos[2]:+.3f})"
            fk_str = f"({fk_pos[0]:+.3f}, {fk_pos[1]:+.3f}, {fk_pos[2]:+.3f})"
            if error > 0.001:  # Only print mismatches
                print(f" {bone_name:<15} {rest_str:<25} {fk_str:<25} {error*100:.1f}cm [{status}]")

        max_identity_error = np.max(identity_errors) * 100
        if max_identity_error < 0.1:
            print(f" All bones match! Max error: {max_identity_error:.3f}cm")
            print("\n [PASS] FK identity test passed.")
            test2_passed = True
        else:
            print(f"\n Max error: {max_identity_error:.2f}cm")
            print("\n [FAIL] FK doesn't even work with identity rotations!")
            print(" The FK structure itself is broken.")
            test2_passed = False

        # =====================================================================
        # TEST 3: FK computation on a real pose
        # =====================================================================
        print("\n" + "-" * 70)
        print(" TEST 3: FK Computation vs Blender (on actual pose)")
        print("-" * 70)

        # Restore the original pose
        for bone_name, rot in saved_rotations.items():
            bone = armature.pose.bones.get(bone_name)
            if bone:
                bone.rotation_quaternion = rot

        context.view_layer.update()

        # Get Blender's actual world positions for ALL bones
        print("\n Getting Blender's actual bone positions...")
        actual_bone_pos = {}
        for bone_name in CONTROLLED_BONES:
            bone = armature.pose.bones.get(bone_name)
            if bone:
                world_pos = arm_matrix @ bone.head
                actual_bone_pos[bone_name] = np.array([world_pos.x, world_pos.y, world_pos.z])

        # Get local rotations and run our FK
        print("\n Extracting local rotations...")
        local_rotations = np.zeros((NUM_BONES, 3), dtype=np.float32)
        for i, bone_name in enumerate(CONTROLLED_BONES):
            bone = armature.pose.bones.get(bone_name)
            if bone:
                quat = bone.rotation_quaternion
                axis, angle = quat.to_axis_angle()
                local_rotations[i] = np.array([axis.x * angle, axis.y * angle, axis.z * angle])

        # Get root info
        hips = armature.pose.bones.get("Hips")
        root_pos = arm_matrix @ hips.head
        root_pos_np = np.array([root_pos.x, root_pos.y, root_pos.z])

        # root_rotation = armature's world rotation (NOT the hips bone matrix!)
        # The FK already applies REST_ORIENTATIONS internally, so root_rotation
        # should only encode the armature's world transform.
        arm_rot_3x3 = arm_matrix.to_3x3()
        root_rot = np.array([
            [arm_rot_3x3[0][0], arm_rot_3x3[0][1], arm_rot_3x3[0][2]],
            [arm_rot_3x3[1][0], arm_rot_3x3[1][1], arm_rot_3x3[1][2]],
            [arm_rot_3x3[2][0], arm_rot_3x3[2][1], arm_rot_3x3[2][2]],
        ], dtype=np.float32)

        print(f"\n Root position: ({root_pos_np[0]:.4f}, {root_pos_np[1]:.4f}, {root_pos_np[2]:.4f})")
        print(f" Armature rotation: identity={np.allclose(root_rot, np.eye(3))}")

        # Run our FK
        print("\n Running our FK computation...")
        from .neural_network.forward_kinematics import forward_kinematics

        fk_positions, fk_rotations = forward_kinematics(
            local_rotations, root_pos_np, root_rot, use_axis_angle=True
        )

        # Compare ALL bone positions
        print("\n" + "-" * 70)
        print(f" {'Bone':<15} {'Blender Pos':<25} {'FK Pos':<25} {'Error':<10}")
        print("-" * 70)

        fk_errors = []
        failed_bones = []
        for i, bone_name in enumerate(CONTROLLED_BONES):
            fk_pos = fk_positions[i]
            actual_pos = actual_bone_pos[bone_name]

            error = np.linalg.norm(fk_pos - actual_pos)
            fk_errors.append(error)

            status = "OK" if error < 0.01 else ("WARN" if error < 0.05 else "BAD")
            actual_str = f"({actual_pos[0]:+.3f}, {actual_pos[1]:+.3f}, {actual_pos[2]:+.3f})"
            fk_str = f"({fk_pos[0]:+.3f}, {fk_pos[1]:+.3f}, {fk_pos[2]:+.3f})"

            if error > 0.01:  # Only print bones with errors
                print(f" {bone_name:<15} {actual_str:<25} {fk_str:<25} {error*100:.1f}cm [{status}]")
                failed_bones.append(bone_name)

        if not failed_bones:
            print(" All 23 bones match! No errors.")

        avg_fk_error = np.mean(fk_errors) * 100
        max_fk_error = np.max(fk_errors) * 100

        print("-" * 70)
        print(f" Bones tested: {len(CONTROLLED_BONES)}")
        print(f" Bones with errors: {len(failed_bones)}")
        print(f" Average FK error: {avg_fk_error:.2f} cm")
        print(f" Max FK error: {max_fk_error:.2f} cm")

        if max_fk_error > 5.0:
            print("\n [FAIL] FK computation doesn't match Blender!")
            print(" Our math is wrong. The FK function needs to be fixed.")
            test2_passed = False
        elif max_fk_error > 1.0:
            print("\n [WARN] FK has small errors. May affect accuracy.")
            test2_passed = True
        else:
            print("\n [PASS] FK matches Blender!")
            test2_passed = True

        # =====================================================================
        # SUMMARY
        # =====================================================================
        print("\n" + "=" * 70)
        print(" VALIDATION SUMMARY")
        print("=" * 70)
        print(f" Test 1 (REST_POSITIONS): {'PASS' if test1_passed else 'FAIL'}")
        print(f" Test 2 (FK Computation): {'PASS' if test2_passed else 'FAIL'}")

        if test1_passed and test2_passed:
            print("\n [ALL PASS] Pipeline is valid! Training should work.")
        elif not test1_passed:
            print("\n [ACTION REQUIRED] Update REST_POSITIONS in config.py")
        else:
            print("\n [ACTION REQUIRED] FK math needs debugging.")
            print(" Check forward_kinematics.py")
            # Print the correct REST_ORIENTATIONS for config.py
            self._print_correct_orientations(armature)

        print("=" * 70 + "\n")

        if test1_passed and test2_passed:
            self.report({'INFO'}, "Pipeline validation PASSED!")
        else:
            self.report({'WARNING'}, "Pipeline validation FAILED - see console")

        return {'FINISHED'}


# =============================================================================
# REGISTRATION
# =============================================================================

def register():
    bpy.utils.register_class(ANIM2_OT_BakeAll)
    bpy.utils.register_class(ANIM2_OT_ResetPose)
    bpy.utils.register_class(ANIM2_OT_TestModal)
    bpy.utils.register_class(ANIM2_TestProperties)
    bpy.utils.register_class(ANIM2_OT_TestPlay)
    bpy.utils.register_class(ANIM2_OT_TestStop)
    bpy.utils.register_class(ANIM2_OT_ClearCache)
    # Neural IK operators
    bpy.utils.register_class(NEURAL_OT_ExtractData)
    bpy.utils.register_class(NEURAL_OT_Train)
    bpy.utils.register_class(NEURAL_OT_Test)
    bpy.utils.register_class(NEURAL_OT_ReloadWeights)
    bpy.utils.register_class(NEURAL_OT_Reset)
    bpy.utils.register_class(NEURAL_OT_SaveData)
    bpy.utils.register_class(NEURAL_OT_LoadData)
    bpy.utils.register_class(NEURAL_OT_AppendData)
    bpy.utils.register_class(NEURAL_OT_RunDiagnostics)
    # Verification operators
    bpy.utils.register_class(NEURAL_OT_VerifyModel)
    bpy.utils.register_class(NEURAL_OT_ApplyTestPose)
    bpy.utils.register_class(NEURAL_OT_ApplyGroundTruth)
    bpy.utils.register_class(NEURAL_OT_CompareVisual)
    # Pipeline validation operators
    bpy.utils.register_class(NEURAL_OT_ValidatePipeline)
    bpy.types.Scene.anim2_test = bpy.props.PointerProperty(type=ANIM2_TestProperties)


def unregister():
    del bpy.types.Scene.anim2_test
    # Pipeline validation operators
    bpy.utils.unregister_class(NEURAL_OT_ValidatePipeline)
    # Verification operators
    bpy.utils.unregister_class(NEURAL_OT_CompareVisual)
    bpy.utils.unregister_class(NEURAL_OT_ApplyGroundTruth)
    bpy.utils.unregister_class(NEURAL_OT_ApplyTestPose)
    bpy.utils.unregister_class(NEURAL_OT_VerifyModel)
    # Neural IK operators
    bpy.utils.unregister_class(NEURAL_OT_RunDiagnostics)
    bpy.utils.unregister_class(NEURAL_OT_AppendData)
    bpy.utils.unregister_class(NEURAL_OT_LoadData)
    bpy.utils.unregister_class(NEURAL_OT_SaveData)
    bpy.utils.unregister_class(NEURAL_OT_Reset)
    bpy.utils.unregister_class(NEURAL_OT_ReloadWeights)
    bpy.utils.unregister_class(NEURAL_OT_Test)
    bpy.utils.unregister_class(NEURAL_OT_Train)
    bpy.utils.unregister_class(NEURAL_OT_ExtractData)
    bpy.utils.unregister_class(ANIM2_OT_ClearCache)
    bpy.utils.unregister_class(ANIM2_OT_TestStop)
    bpy.utils.unregister_class(ANIM2_OT_TestPlay)
    bpy.utils.unregister_class(ANIM2_TestProperties)
    bpy.utils.unregister_class(ANIM2_OT_TestModal)
    bpy.utils.unregister_class(ANIM2_OT_ResetPose)
    bpy.utils.unregister_class(ANIM2_OT_BakeAll)
