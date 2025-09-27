#Exp_Game/modal/exp_view_helpers.py

from ..physics.exp_view import shortest_angle_diff
import math

# ---------------------------
# Utility / Helper Methods
# ---------------------------
def smooth_rotate_towards_camera(self):
    if not self.target_object:
        return

    pressed_keys = {
        self.pref_forward_key,
        self.pref_backward_key,
        self.pref_left_key,
        self.pref_right_key,
    }
    actual_pressed = _resolved_move_keys(self).intersection(pressed_keys)
    if not actual_pressed:
        return

    desired_yaw = determine_desired_yaw(self, actual_pressed)
    current_yaw = self.target_object.rotation_euler.z
    yaw_diff = shortest_angle_diff(current_yaw, desired_yaw)
    if abs(yaw_diff) > 0.001:
        self.target_object.rotation_euler.z += yaw_diff * self.rotation_smoothness


def determine_desired_yaw(self, actual_pressed):
    """Calculate desired yaw based on user-chosen forward/back/left/right keys + camera yaw."""
    base_yaw = self.yaw

    # Check combos: forward+right => ~45°, forward+left => -45°, etc.
    # We'll unify the user-chosen keys to match typical movement combos
    # Or any logic you wish to implement:
    if (self.pref_forward_key in actual_pressed and self.pref_right_key in actual_pressed):
        return base_yaw - math.radians(45)
    if (self.pref_forward_key in actual_pressed and self.pref_left_key in actual_pressed):
        return base_yaw + math.radians(45)
    if (self.pref_backward_key in actual_pressed and self.pref_right_key in actual_pressed):
        return base_yaw - math.radians(135)
    if (self.pref_backward_key in actual_pressed and self.pref_left_key in actual_pressed):
        return base_yaw + math.radians(135)

    if self.pref_forward_key in actual_pressed:
        return base_yaw
    if self.pref_backward_key in actual_pressed:
        return base_yaw + math.pi
    if self.pref_left_key in actual_pressed:
        return base_yaw + (math.pi / 2)
    if self.pref_right_key in actual_pressed:
        return base_yaw - (math.pi / 2)

    # fallback
    return base_yaw

def _axis_of_key(self, key):
    if key in (self.pref_left_key, self.pref_right_key):  return 'x'
    if key in (self.pref_forward_key, self.pref_backward_key): return 'y'
    return None

def _resolved_move_keys(self):
    """
    Return at most one X key and one Y key (last-pressed on each axis),
    plus run key if held.
    """
    resolved = set()
    kx = self._axis_last.get('x')
    if kx and (kx in self.keys_pressed):
        resolved.add(kx)
    ky = self._axis_last.get('y')
    if ky and (ky in self.keys_pressed):
        resolved.add(ky)
    if self.pref_run_key in self.keys_pressed:
        resolved.add(self.pref_run_key)
    return resolved

def _update_axis_resolution_on_release(self, key):
    ax = _axis_of_key(self, key)   # ← was: self._axis_of_key(key)
    if not ax:
        return
    # drop from candidates
    self._axis_candidates[ax].pop(key, None)
    # if it was selected, pick newest remaining on that axis
    if self._axis_last.get(ax) == key:
        if self._axis_candidates[ax]:
            self._axis_last[ax] = max(self._axis_candidates[ax], key=self._axis_candidates[ax].get)
        else:
            self._axis_last[ax] = None



def _bind_view3d_once(self, context) -> bool:
    """
    Pick exactly one VIEW_3D + WINDOW region in the CURRENT WINDOW and cache:
    - _view3d_window, _view3d_screen, _view3d_area, _view3d_rv3d
    - _clip_start_cached
    Selection order:
    1) If context.area is a VIEW_3D, use that.
    2) Else choose the largest VIEW_3D in context.window.screen.
    Returns True on success, False otherwise.
    """
    self._view3d_window = None
    self._view3d_screen = None
    self._view3d_area   = None
    self._view3d_rv3d   = None
    self._clip_start_cached = 0.1

    win = getattr(context, "window", None)
    scr = getattr(win, "screen", None) if win else None
    if not win or not scr:
        return False

    # 1) Prefer the current context area if it's a VIEW_3D
    area = getattr(context, "area", None)
    if area and getattr(area, "type", None) == 'VIEW_3D':
        region = next((r for r in area.regions if r.type == 'WINDOW'), None)
        if region is not None:
            self._view3d_window = win
            self._view3d_screen = scr
            self._view3d_area   = area
            self._view3d_rv3d   = area.spaces.active.region_3d
            try:
                self._clip_start_cached = float(area.spaces.active.clip_start)
                if self._clip_start_cached <= 0.0:
                    self._clip_start_cached = 0.1
            except Exception:
                self._clip_start_cached = 0.1
            return True

    # 2) Otherwise, choose the largest VIEW_3D on this screen
    best = None
    best_pixels = -1
    for a in scr.areas:
        if a.type != 'VIEW_3D':
            continue
        region = next((r for r in a.regions if r.type == 'WINDOW'), None)
        if not region:
            continue
        pixels = int(region.width) * int(region.height)
        if pixels > best_pixels:
            best_pixels = pixels
            best = (a, region)

    if best is None:
        return False

    area, region = best
    self._view3d_window = win
    self._view3d_screen = scr
    self._view3d_area   = area
    self._view3d_rv3d   = area.spaces.active.region_3d
    try:
        self._clip_start_cached = float(area.spaces.active.clip_start)
        if self._clip_start_cached <= 0.0:
            self._clip_start_cached = 0.1
    except Exception:
        self._clip_start_cached = 0.1
    return True


def _maybe_rebind_view3d(self, context) -> bool:
    """
    Re-validate cached VIEW_3D handles. If the cached area is gone/invalid,
    attempt a single rebind using the same selection policy as _bind_view3d_once.
    Returns True if a valid rv3d is available after the call.
    """
    try:
        # Happy path: cached area & rv3d still valid
        if (self._view3d_window and self._view3d_screen and
            self._view3d_area and getattr(self._view3d_area, "type", None) == 'VIEW_3D' and
            self._view3d_rv3d is not None and self._view3d_area in self._view3d_screen.areas):
            return True
    except Exception:
        pass

    # Attempt to rebind (e.g., after fullscreen/layout change)
    return bool(self._bind_view3d_once(context))
