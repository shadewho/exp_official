#Exploratory.Exp_Game/exp_view.py

import sys
import os
import mathutils
import math
import platform
from .exp_raycastutils import raycast_to_ground
from ctypes.util import find_library
import ctypes
import ctypes.wintypes
def update_view(context, obj, pitch, yaw, bvh_tree, orbit_distance, zoom_factor, dynamic_bvh_map=None):
    # Calculate the direction vector
    direction = mathutils.Vector((
        math.cos(pitch) * math.sin(yaw),
        -math.cos(pitch) * math.cos(yaw),
        math.sin(pitch)
    ))

    # Set the desired view location
    head_height = 2.0
    scan_point = obj.location + mathutils.Vector((0, 0, head_height))
    view_location = scan_point + direction * orbit_distance

    # Pull your detection range out by the buffer amount
    buf = context.scene.camera_collision_buffer
    extended_scan_distance = orbit_distance + zoom_factor + buf
    if extended_scan_distance < 0.0:
        extended_scan_distance = 0.0

        
    # Perform obstruction checking
    if bvh_tree:
        # Calculate the direction vector from the head to the extended view location
        direction_to_view = (view_location - scan_point).normalized()

        # Perform the raycast from the scan point towards the extended view location
        extended_view_location = scan_point + direction_to_view * extended_scan_distance
        hit_location = raycast_to_ground(bvh_tree, scan_point, direction_to_view)

        # If the ray hits something, adjust the view location
        if hit_location:
            distance_to_hit = (scan_point - hit_location).length
            if distance_to_hit < extended_scan_distance:
                buf = context.scene.camera_collision_buffer
                # compute raw pull‑in distance
                adjusted_distance = distance_to_hit - buf
                # clamp so we never go farther out than orbit_distance
                adjusted_distance = min(adjusted_distance, orbit_distance)
                # apply it
                view_location = scan_point + direction_to_view * adjusted_distance


    #####################################################################
    # Dynamic BVH Check (without removing or changing original logic)
    #####################################################################
    if dynamic_bvh_map:
        # We'll do a second pass and see if any dynamic mesh is closer than our current camera offset.
        # We'll replicate the logic above but for each dynamic BVH, then compare distances.
        direction_to_view_dyn = (view_location - scan_point).normalized()
        extended_view_location_dyn = scan_point + direction_to_view_dyn * extended_scan_distance

        closest_distance_dyn = None
        closest_hit_dyn = None

        for dyn_obj, (dyn_bvh, dyn_radius) in dynamic_bvh_map.items():
            if not dyn_bvh:
                continue
            # Raycast to dynamic object
            hit_dyn = raycast_to_ground(dyn_bvh, scan_point, direction_to_view_dyn)
            if hit_dyn:
                dist_dyn = (scan_point - hit_dyn).length
                # Track the closest dynamic obstacle
                if (closest_distance_dyn is None) or (dist_dyn < closest_distance_dyn):
                    closest_distance_dyn = dist_dyn
                    closest_hit_dyn = hit_dyn

        # If a dynamic obstacle is closer than our current view, clamp again
        if closest_hit_dyn and closest_distance_dyn < extended_scan_distance:
            buf = context.scene.camera_collision_buffer
            adjusted_distance_dyn = closest_distance_dyn - buf
            adjusted_distance_dyn = min(adjusted_distance_dyn, orbit_distance)
            if adjusted_distance_dyn < (view_location - scan_point).length:
                view_location = scan_point + direction_to_view_dyn * adjusted_distance_dyn

    #####################################################################
    # END of new dynamic block
    #####################################################################

    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            for region in area.regions:
                if region.type == 'WINDOW':
                    region_3d = area.spaces.active.region_3d
                    region_3d.view_location = view_location
                    region_3d.view_rotation = direction.to_track_quat('Z', 'Y').to_matrix().to_3x3().to_quaternion()

def shortest_angle_diff(current, target):
    diff = (target - current + math.pi) % (2 * math.pi) - math.pi
    return diff


# — only load X11 when truly on Linux —
_x11 = _disp = _root = None
if platform.system() == "Linux":
    try:
        _x11 = ctypes.CDLL(find_library("X11"))
        _disp = _x11.XOpenDisplay(None)
        _root = _x11.XDefaultRootWindow(_disp)
    except OSError:
        _x11 = None

# — only load CoreGraphics when truly on macOS —
_core = None
if platform.system() == "Darwin":
    try:
        _core = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
        )
    except OSError:
        _core = None


def confine_cursor_to_window():

    system = platform.system()

    """Grab & confine the cursor to the Blender window on Win/Linux/macOS."""
    if system == "Windows":
        hwnd = ctypes.windll.user32.GetActiveWindow()
        rect = ctypes.wintypes.RECT()
        ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rect))
        # Convert client coordinates to screen coordinates
        pt = ctypes.wintypes.POINT(rect.left, rect.top)
        ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(pt))
        rect.left, rect.top = pt.x, pt.y
        pt = ctypes.wintypes.POINT(rect.right, rect.bottom)
        ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(pt))
        rect.right, rect.bottom = pt.x, pt.y
        ctypes.windll.user32.ClipCursor(ctypes.byref(rect))

    elif system == "Linux" and _x11:
        # skip if Wayland
        if "WAYLAND_DISPLAY" in os.environ:
            return
        # TODO: narrow to Blender’s X11 window rather than root
        _x11.XGrabPointer(
            _disp,
            ctypes.c_ulong(_root),
            False,
            (1 << 21) | (1 << 2) | (1 << 3),
            1, 1,
            ctypes.c_ulong(_root),
            0,
            0,
        )

    elif system == "Darwin" and _core:
        # decouple cursor & mouse movement
        _core.CGAssociateMouseAndMouseCursorPosition(False)

def release_cursor_clip():
    """Release whatever confine_cursor_to_window grabbed."""
    system = platform.system()

    if system == "Windows":
        ctypes.windll.user32.ClipCursor(None)

    elif system == "Linux" and _x11:
        _x11.XUngrabPointer(_disp, 0)

    elif system == "Darwin" and _core:
        _core.CGAssociateMouseAndMouseCursorPosition(True)


# Yes, on paper it runs on all three OSes, but:

# Windows: may clip the wrong window or the wrong-resolution rectangle.

# Linux: will lock the pointer to the whole screen, not just Blender’s view, and will no-op on Wayland.

# macOS: won’t actually confine at all—you’ll need to combine event taps + warping for real confinement.