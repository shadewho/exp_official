# Exploratory/Exp_game/reactions/exp_transforms.py
import bpy
from mathutils import Vector, Euler, Matrix
from ..props_and_utils.exp_time import get_game_time
# ------------------------------
# TransformTask + Manager
# ------------------------------

_active_transform_tasks = []

class TransformTask:
    def __init__(self, obj, start_loc, start_rot, start_scl,
                 end_loc, end_rot, end_scl,
                 start_time, duration,
                 rot_interp='euler',
                 delta_euler=None):
        self.obj = obj

        # Store starting transforms (Euler = local space)
        self.start_loc = start_loc
        self.start_rot = start_rot           # Euler
        self.start_scl = start_scl

        # Targets
        self.end_loc = end_loc
        self.end_rot = end_rot               # Euler
        self.end_scl = end_scl

        # Rotation interpolation mode
        self.rot_interp  = rot_interp        # 'euler' | 'quat' | 'local_delta'
        self.delta_euler = delta_euler       # Euler (only for 'local_delta')

        # Quats for slerp path (used by 'quat')
        self.start_rot_q = start_rot.to_quaternion()
        self.end_rot_q   = end_rot.to_quaternion()

        self.start_time = start_time
        self.duration   = duration

    def update(self, now):
        if self.duration <= 0.0:
            self.obj.location       = self.end_loc
            self.obj.rotation_euler = self.end_rot
            self.obj.scale          = self.end_scl
            return True

        t = (now - self.start_time) / self.duration
        if t >= 1.0:
            self.obj.location       = self.end_loc
            self.obj.rotation_euler = self.end_rot
            self.obj.scale          = self.end_scl
            return True

        # Lerp location
        self.obj.location = self.start_loc.lerp(self.end_loc, t)

        # Rotation
        if self.rot_interp == 'local_delta':
            # True Blender-like local rotation (supports multi-turn spins)
            # q(t) = q_start @ quat(Euler(t * delta_euler))
            qdelta_t = Euler((
                self.delta_euler.x * t,
                self.delta_euler.y * t,
                self.delta_euler.z * t,
            ), 'XYZ').to_quaternion()
            cur_q = self.start_rot_q @ qdelta_t
            self.obj.rotation_euler = cur_q.to_euler('XYZ')

        elif self.rot_interp == 'quat':
            cur_q = self.start_rot_q.slerp(self.end_rot_q, t)
            self.obj.rotation_euler = cur_q.to_euler('XYZ')

        else:
            # Per-channel Euler lerp (old behavior)
            current_rot = self.start_rot.copy()
            current_rot.x = (1.0 - t) * self.start_rot.x + (t * self.end_rot.x)
            current_rot.y = (1.0 - t) * self.start_rot.y + (t * self.end_rot.y)
            current_rot.z = (1.0 - t) * self.start_rot.z + (t * self.end_rot.z)
            self.obj.rotation_euler = current_rot

        # Lerp scale
        self.obj.scale = self.start_scl.lerp(self.end_scl, t)
        return False


def schedule_transform(obj, end_loc, end_rot, end_scl, duration,
                       rot_interp='euler', delta_euler=None):
    """
    rot_interp: 'euler' | 'quat' | 'local_delta'
    If rot_interp == 'local_delta', pass delta_euler (Euler local offset).
    """
    if not obj:
        return

    start_loc = obj.location.copy()
    start_rot = obj.rotation_euler.copy()
    start_scl = obj.scale.copy()

    start_time = get_game_time()

    task = TransformTask(
        obj,
        start_loc, start_rot, start_scl,
        end_loc, end_rot, end_scl,
        start_time, duration,
        rot_interp=rot_interp,
        delta_euler=delta_euler
    )
    _active_transform_tasks.append(task)


def update_transform_tasks():
    """
    Called once per frame.
    Removes tasks that have finished.
    """
    now = get_game_time()
    finished_indices = []
    for i, task in enumerate(_active_transform_tasks):
        done = task.update(now)
        if done:
            finished_indices.append(i)
    # remove in reverse so indexes don't shift
    for i in reversed(finished_indices):
        _active_transform_tasks.pop(i)

        # In exp_reactions.py

def execute_transform_reaction(reaction):
    """
    Applies a transform reaction to either:
      • the scene’s target_armature (if use_character=True), or
      • the specified transform_object.
    """
    scene = bpy.context.scene

    # 1) Pick target: character vs. user-picked object
    if getattr(reaction, "use_character", False):
        target_obj = scene.target_armature
    else:
        target_obj = reaction.transform_object

    # 2) Bail if nothing to move
    if not target_obj:
        return

    # 3) Ensure Euler XYZ rotation mode
    target_obj.rotation_mode = 'XYZ'

    # 4) Clamp duration
    duration = reaction.transform_duration
    if duration < 0.0:
        duration = 0.0

    # 5) Dispatch based on transform_mode
    mode = reaction.transform_mode

    if mode == "OFFSET":
        # The old approach: interpret location/rotation/scale as global offsets
        apply_offset_transform(reaction, target_obj, duration)

    elif mode == "TO_LOCATION":
        # Interpret location/rotation/scale as absolute world transforms
        apply_to_location_transform(reaction, target_obj, duration)

    elif mode == "TO_OBJECT":
        # NEW: allow using the character as the "to object" source
        to_obj = (
            scene.target_armature
            if getattr(reaction, "transform_to_use_character", False)
            else reaction.transform_to_object
        )
        if not to_obj:
            return

        # capture the original transforms
        start_loc = target_obj.location.copy()
        start_rot = target_obj.rotation_euler.copy()
        start_scl = target_obj.scale.copy()

        # pick which channels to override
        end_loc = (
            to_obj.location.copy()
            if reaction.transform_use_location
            else start_loc
        )
        end_rot = (
            to_obj.rotation_euler.copy()
            if reaction.transform_use_rotation
            else start_rot
        )
        end_scl = (
            to_obj.scale.copy()
            if reaction.transform_use_scale
            else start_scl
        )
        # schedule the transform with only the selected channels
        schedule_transform(target_obj, end_loc, end_rot, end_scl, duration)

    elif mode == "LOCAL_OFFSET":
        # Interpret location/rotation/scale as offsets in local space
        apply_local_offset_transform(reaction, target_obj, duration)
        
    elif mode == "TO_BONE":
        # NEW: move to a specific armature bone (by name)
        apply_to_bone_transform(reaction, target_obj, duration)

def apply_offset_transform(reaction, target_obj, duration):
    """
    Global offset around the object's current origin, preserving spin.
    We still use a matrix to compute the translated/scaled end location,
    but we compute end Euler by adding the offset eulers so 360° isn't lost.
    """
    loc_off = Vector(reaction.transform_location)
    rot_off = Euler(reaction.transform_rotation, 'XYZ')
    scl_off = Vector(reaction.transform_scale)

    # Current
    start_loc = target_obj.location.copy()
    start_rot = target_obj.rotation_euler.copy()
    start_scl = target_obj.scale.copy()

    # Build the user offset in GLOBAL space for location/scale computation
    T_off = Matrix.Translation(loc_off)
    R_off = rot_off.to_matrix().to_4x4()  # used only to rotate the loc offset when pivoting
    S_off = Matrix.Diagonal((scl_off.x, scl_off.y, scl_off.z, 1.0))
    user_offset_mat = T_off @ R_off @ S_off

    # Pivot is object's current world location
    pivot_world = target_obj.matrix_world.translation
    pivot_inv = Matrix.Translation(-pivot_world)
    pivot_fwd = Matrix.Translation(pivot_world)

    start_mat  = target_obj.matrix_world.copy()
    offset_mat = pivot_fwd @ user_offset_mat @ pivot_inv
    final_mat  = offset_mat @ start_mat

    # Decompose ONLY for end location/scale (rotation from decompose loses spin)
    end_loc, _end_rot_q, end_scl = final_mat.decompose()

    # END ROTATION: preserve spin by explicit Euler addition
    end_rot = Euler((
        start_rot.x + rot_off.x,
        start_rot.y + rot_off.y,
        start_rot.z + rot_off.z,
    ), 'XYZ')

    schedule_transform(target_obj, end_loc, end_rot, end_scl, duration)

def apply_to_bone_transform(reaction, target_obj, duration):
    """
    Copy transforms from a specific bone (by name) onto target_obj.
    Armature source can be the character or a user-picked armature.
    Respects transform_use_location/rotation/scale.
    """
    scn = bpy.context.scene

    # Choose armature source
    use_char = bool(getattr(reaction, "transform_to_bone_use_character", True))
    arm_obj = scn.target_armature if use_char else getattr(reaction, "transform_to_armature", None)
    if not arm_obj or getattr(arm_obj, "type", "") != 'ARMATURE':
        return

    # Resolve bone by name (string, survives rebuilds)
    bone_name = (getattr(reaction, "transform_bone_name", "") or "").strip()
    if not bone_name:
        return
    try:
        pb = arm_obj.pose.bones.get(bone_name)
    except Exception:
        pb = None
    if not pb:
        return

    # Current target state
    start_loc = target_obj.location.copy()
    start_rot = target_obj.rotation_euler.copy()
    start_scl = target_obj.scale.copy()

    # Bone world transform
    world_mat = arm_obj.matrix_world @ pb.matrix
    try:
        end_loc_w, end_rot_q, end_scl_v = world_mat.decompose()
        end_rot_e = end_rot_q.to_euler('XYZ')
    except Exception:
        end_loc_w = world_mat.translation
        end_rot_e = world_mat.to_euler('XYZ')
        end_scl_v = Vector((1.0, 1.0, 1.0))

    # Per-channel application
    use_loc = bool(getattr(reaction, "transform_use_location", True))
    use_rot = bool(getattr(reaction, "transform_use_rotation", True))
    use_scl = bool(getattr(reaction, "transform_use_scale", True))

    end_loc = end_loc_w if use_loc else start_loc
    end_rot = end_rot_e if use_rot else start_rot
    end_scl = end_scl_v if use_scl else start_scl

    schedule_transform(target_obj, end_loc, end_rot, end_scl, duration)

def apply_to_location_transform(reaction, target_obj, duration):

    # interpret transform_location, transform_rotation, transform_scale
    # as the actual final world transforms
    end_loc = Vector(reaction.transform_location)
    end_rot = Euler(reaction.transform_rotation, 'XYZ')
    end_scl = Vector(reaction.transform_scale)

    # read current
    current_loc = target_obj.location.copy()
    current_rot = target_obj.rotation_euler.copy()
    current_scl = target_obj.scale.copy()

    schedule_transform(target_obj, end_loc, end_rot, end_scl, duration)

def apply_to_object_transform(reaction, target_obj, to_obj, duration):

    # We'll read the to_obj's location, rotation_euler, scale as the final
    end_loc = to_obj.location.copy()
    end_rot = to_obj.rotation_euler.copy()
    end_scl = to_obj.scale.copy()

    # current
    start_loc = target_obj.location.copy()
    start_rot = target_obj.rotation_euler.copy()
    start_scl = target_obj.scale.copy()

    # schedule
    schedule_transform(target_obj, end_loc, end_rot, end_scl, duration)

def apply_local_offset_transform(reaction, target_obj, duration):
    """
    LOCAL offset:
      - Rotation: about the object's CURRENT LOCAL axes (like Blender's local rotation).
      - Translation: along local axes (normalized basis avoids skew from non-uniform scale).
      - Scale: component-wise in local space.
    """
    from math import tau
    loc_off = Vector(reaction.transform_location)
    rot_off = Euler(reaction.transform_rotation, 'XYZ')  # radians
    scl_off = Vector(reaction.transform_scale)

    # --- START (current state)
    start_loc = target_obj.location.copy()
    start_rot_eul = target_obj.rotation_euler.copy()
    start_rot_q = start_rot_eul.to_quaternion()
    start_scl = target_obj.scale.copy()

    # --- TRUE LOCAL ROTATION ---
    # Build the delta in LOCAL space and right-multiply:
    #   q_end = q_start @ q_delta_local
    delta_q_local = rot_off.to_quaternion()
    end_rot_q = start_rot_q @ delta_q_local
    end_rot_base = end_rot_q.to_euler('XYZ')

    # Preserve multi-turn spins so big angles (e.g. 720°) don’t collapse:
    def unwrap_to_target(base_val, target_val):
        k = round((target_val - base_val) / tau)
        return base_val + k * tau

    target_eul = (
        start_rot_eul.x + rot_off.x,
        start_rot_eul.y + rot_off.y,
        start_rot_eul.z + rot_off.z,
    )
    end_rot = Euler((
        unwrap_to_target(end_rot_base.x, target_eul[0]),
        unwrap_to_target(end_rot_base.y, target_eul[1]),
        unwrap_to_target(end_rot_base.z, target_eul[2]),
    ), 'XYZ')

    # --- LOCAL TRANSLATION ---
    # Use the normalized rotation basis so non-uniform scale doesn't skew direction.
    R_world_n = target_obj.matrix_world.to_3x3().normalized()
    end_loc = start_loc + (R_world_n @ loc_off)

    # --- LOCAL SCALE (component-wise) ---
    end_scl = Vector((
        start_scl.x * scl_off.x,
        start_scl.y * scl_off.y,
        start_scl.z * scl_off.z,
    ))

    schedule_transform(
    target_obj,
    end_loc, end_rot, end_scl,
    duration,
    rot_interp='local_delta',
    delta_euler=rot_off  # <<— the exact local offset the user requested
)
