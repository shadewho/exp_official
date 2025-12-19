# Exploratory/Exp_UI/interface/operators/display.py
# Detail-only overlay (no browse). Safe teardown from anywhere.

import os
import tempfile
import requests
import bpy
from bpy.props import BoolProperty

from ...auth.helpers import load_token
from ...internet.helpers import is_internet_available
from ...main_config import PACKAGES_ENDPOINT, PACKAGE_DETAILS_ENDPOINT
from ..drawing.draw_master import load_image_buttons
from ..drawing.utilities import draw_image_buttons_callback
from ...download_and_explore.explore_main import explore_icon_handler, reset_download_progress

# ---------------------------------------------------------------------------
# Global overlay registry + hard kill
# ---------------------------------------------------------------------------

_OVERLAY = {"handler": None, "timer": None, "area": None, "window": None}

def force_hide_overlay():
    """
    Hard-stop the overlay regardless of which window/scene launched it.
    Idempotent: safe to call anytime.
    """
    # 1) draw handler
    h = _OVERLAY.get("handler")
    if h is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(h, 'WINDOW')
        except Exception:
            pass
        _OVERLAY["handler"] = None

    # 2) timer
    t = _OVERLAY.get("timer")
    if t is not None:
        try:
            bpy.context.window_manager.event_timer_remove(t)
        except Exception:
            pass
        _OVERLAY["timer"] = None

    # 3) clear draw data so nothing renders even if a handler slipped through
    try:
        if getattr(bpy.types, "Scene", None) is not None:
            if getattr(bpy.types.Scene, "gpu_image_buttons_data", None) is None:
                bpy.types.Scene.gpu_image_buttons_data = []
            try:
                bpy.types.Scene.gpu_image_buttons_data.clear()
            except Exception:
                bpy.types.Scene.gpu_image_buttons_data = []
    except Exception:
        pass

    # 4) UI state and cursor
    try:
        bpy.context.scene.ui_current_mode = "GAME"
    except Exception:
        pass
    try:
        bpy.context.window.cursor_modal_restore()
    except Exception:
        pass

    # 5) flush one redraw
    try:
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
    except Exception:
        pass


# Derive /api/download from PACKAGES_ENDPOINT once
DOWNLOAD_ENDPOINT = PACKAGES_ENDPOINT.rsplit('/', 1)[0] + "/download"

# ---------------------------------------------------------------------------
# Overlay operator
# ---------------------------------------------------------------------------

class PACKAGE_OT_Display(bpy.types.Operator):
    """Detail-only overlay (no browse, no pagination, no cache)."""
    bl_idname = "view3d.add_package_display"
    bl_label  = "Show Exploratory Detail"
    bl_options = {'REGISTER'}

    keep_mode: BoolProperty(
        name="Keep Mode",
        default=True,
        description="Keep current UI mode"
    )

    _handler = None
    _timer   = None
    _dirty   = True
    _last_progress = -1.0
    _original_area_type = None

    def invoke(self, context, event):
        self._original_area_type = getattr(context.area, "type", None)
        if not self.keep_mode:
            context.scene.ui_current_mode = "DETAIL"

        # ensure draw data list exists
        bpy.types.Scene.gpu_image_buttons_data = load_image_buttons()

        # draw handler
        self._handler = bpy.types.SpaceView3D.draw_handler_add(
            draw_image_buttons_callback, (), 'WINDOW', 'POST_PIXEL'
        )

        # keep a sentinel item (optional – harmless)
        try:
            data = bpy.types.Scene.gpu_image_buttons_data
            if data is None:
                bpy.types.Scene.gpu_image_buttons_data = []
                data = bpy.types.Scene.gpu_image_buttons_data
            if isinstance(data, list):
                data.append({"name": "handler", "handler": self._handler})
        except Exception:
            pass

        # timer + modal loop
        self._timer = context.window_manager.event_timer_add(0.1, window=context.window)
        context.window_manager.modal_handler_add(self)
        if context.area:
            context.area.tag_redraw()

        # publish to global registry so others can kill us
        global _OVERLAY
        _OVERLAY["handler"] = self._handler
        _OVERLAY["timer"]   = self._timer
        _OVERLAY["area"]    = context.area
        _OVERLAY["window"]  = context.window

        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if not is_internet_available():
            self.report({'ERROR'}, "No internet. Closing UI.")
            return self.cancel(context)

        # programmatic shutdown: mode flipped to GAME
        if getattr(context.scene, "ui_current_mode", "GAME") == "GAME":
            return self.cancel(context)

        if event.type == 'TIMER':
            # if area type changed under us (fullscreen, layout switch), bail
            if (not context.area) or (getattr(context.area, "type", None) != self._original_area_type):
                return self.cancel(context)

            if self._dirty:
                bpy.types.Scene.gpu_image_buttons_data = load_image_buttons()
                if context.area:
                    context.area.tag_redraw()
                self._dirty = False

            # refresh when download progress changes
            cur = float(getattr(context.scene, "download_progress", 0.0))
            if abs(cur - self._last_progress) > 1e-6:
                self._last_progress = cur
                self._dirty = True

        # hover hand
        if event.type == 'MOUSEMOVE':
            mx, my = event.mouse_region_x, event.mouse_region_y
            hover = False
            for button in (bpy.types.Scene.gpu_image_buttons_data or []):
                name = button.get("name")
                if name not in {"Close_Icon", "Back_Icon", "Explore_Icon"}:
                    continue
                x1, y1, x2, y2 = button.get("pos", (0, 0, 0, 0))
                if x1 <= mx <= x2 and y1 <= my <= y2:
                    hover = True
                    break
            context.window.cursor_modal_set('HAND' if hover else 'DEFAULT')

        # clicks
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            mx, my = event.mouse_region_x, event.mouse_region_y
            for button in (bpy.types.Scene.gpu_image_buttons_data or []):
                name = button.get("name")
                if not name:
                    continue
                x1, y1, x2, y2 = button.get("pos", (0, 0, 0, 0))
                if not (x1 <= mx <= x2 and y1 <= my <= y2):
                    continue

                if name == "Close_Icon":
                    # cancel current download if any, then close
                    try:
                        from ...download_and_explore.explore_main import current_download_task
                        if current_download_task is not None:
                            current_download_task.cancel()
                    except Exception:
                        pass
                    return self.cancel(context)

                if name == "Back_Icon":
                    return self.cancel(context)

                if name == "Explore_Icon":
                    code = (context.scene.download_code or "").strip()
                    if not code or int(getattr(context.scene.my_addon_data, "file_id", 0) or 0) <= 0:
                        self.report({'ERROR'}, "Enter a download code and load details first.")
                        return {'RUNNING_MODAL'}
                    token = load_token()
                    if not token:
                        self.report({'ERROR'}, "Not logged in.")
                        return {'RUNNING_MODAL'}
                    reset_download_progress()
                    context.scene.download_progress = 0.0
                    context.scene.ui_current_mode = "LOADING"
                    self._dirty = True
                    explore_icon_handler(context, code)
                    return {'RUNNING_MODAL'}

        if event.type == 'ESC':
            return self.cancel(context)

        return {'PASS_THROUGH'}

    def cancel(self, context):
        # hard kill + clear registry
        force_hide_overlay()
        global _OVERLAY
        _OVERLAY.update({"handler": None, "timer": None, "area": None, "window": None})
        return {'CANCELLED'}


# ---------------------------------------------------------------------------
# Fetch detail by code (no browse)
# ---------------------------------------------------------------------------

class WEBAPP_OT_ShowDetailByCode(bpy.types.Operator):
    """Open Exploratory UI, retreive item details (Download Code)"""
    bl_idname = "webapp.show_detail_by_code"
    bl_label  = "Display world details (download code)"
    bl_options = {'REGISTER'}

    def execute(self, context):
        if not is_internet_available():
            self.report({'ERROR'}, "No internet.")
            return {'CANCELLED'}

        token = load_token()
        if not token:
            self.report({'ERROR'}, "Not logged in.")
            return {'CANCELLED'}

        code = (context.scene.download_code or "").strip()
        if not code:
            self.report({'ERROR'}, "Enter a download code.")
            return {'CANCELLED'}

        headers = {"Authorization": f"Bearer {token}"}

        # 1) Resolve code -> file info
        try:
            r = requests.post(DOWNLOAD_ENDPOINT, json={"download_code": code}, headers=headers, timeout=20)
            r.raise_for_status()
            d = r.json()
        except Exception as e:
            self.report({'ERROR'}, f"Download code lookup failed: {e}")
            return {'CANCELLED'}

        if not d.get("success"):
            self.report({'ERROR'}, d.get("message", "Invalid code"))
            return {'CANCELLED'}

        pkg = d.get("package") or {}
        file_id  = int(pkg.get("file_id", 0) or 0)
        thumb_url = pkg.get("thumbnail_url")

        if file_id <= 0:
            self.report({'ERROR'}, "Bad server response.")
            return {'CANCELLED'}

        # 2) Fetch detail JSON for UI fields
        try:
            r2 = requests.get(f"{PACKAGE_DETAILS_ENDPOINT}/{file_id}", headers=headers, timeout=15)
            r2.raise_for_status()
            detail = r2.json()
        except Exception as e:
            self.report({'ERROR'}, f"Detail fetch failed: {e}")
            return {'CANCELLED'}

        if not detail.get("success"):
            self.report({'ERROR'}, detail.get("message", "Detail not available"))
            return {'CANCELLED'}

        # 3) Download thumbnail to a temp file (no disk DB)
        local_thumb = ""
        if thumb_url:
            try:
                rt = requests.get(thumb_url, timeout=20)
                rt.raise_for_status()
                folder = tempfile.gettempdir()
                local_thumb = os.path.join(folder, f"exploratory_thumb_{file_id}.png")
                with open(local_thumb, "wb") as f:
                    f.write(rt.content)
            except Exception:
                local_thumb = ""

        # 4) Pump into your PG
        ad = context.scene.my_addon_data
        if hasattr(ad, "init_from_package"):
            ad.init_from_package(detail)
            ad.comments.clear()
            for c in (detail.get("comments") or []):
                c_item = ad.comments.add()
                c_item.author    = c.get("author", "")
                c_item.text      = c.get("content", "")
                c_item.timestamp = c.get("timestamp", "")
        else:
            ad.file_id        = file_id
            ad.package_name   = detail.get("package_name", "")
            ad.author         = detail.get("uploader", "")
            ad.description    = detail.get("description", "")
            ad.likes          = int(detail.get("likes", 0) or 0)
            ad.download_count = int(detail.get("download_count", 0) or 0)
            ad.file_type      = detail.get("file_type", "world")
            ad.vote_count     = int(detail.get("vote_count", 0) or 0)
            ad.upload_date    = detail.get("upload_date", "")

        context.scene.selected_thumbnail = local_thumb if local_thumb else ""
        context.scene.ui_current_mode = "DETAIL"

        # Ensure overlay is visible or refresh it
        if _OVERLAY.get("handler") is None:
            bpy.ops.view3d.add_package_display('INVOKE_DEFAULT', keep_mode=True)
        else:
            bpy.types.Scene.gpu_image_buttons_data = load_image_buttons()
            if context.area:
                context.area.tag_redraw()

        self.report({'INFO'}, "Details loaded.")
        return {'FINISHED'}



class WEBAPP_OT_PasteCode(bpy.types.Operator):
    """Paste download code from clipboard"""
    bl_idname = "webapp.paste_code"
    bl_label  = "Paste download code from clipboard"
    bl_options = {'REGISTER', 'INTERNAL'}

    MAX_PASTE_LENGTH = 20

    def execute(self, context):
        try:
            clip = (context.window_manager.clipboard or "").strip()
        except Exception:
            clip = ""
        if not clip:
            self.report({'ERROR'}, "Clipboard is empty.")
            return {'CANCELLED'}

        if len(clip) > self.MAX_PASTE_LENGTH:
            self.report({'ERROR'}, "Invalid paste: too long.")
            return {'CANCELLED'}

        try:
            context.scene.download_code = clip
        except Exception:
            self.report({'ERROR'}, "Scene.download_code is missing.")
            return {'CANCELLED'}

        self.report({'INFO'}, "Code pasted.")
        return {'FINISHED'}



# ---------------------------------------------------------------------------
# Remove overlay (compat for existing calls)
# ---------------------------------------------------------------------------

class REMOVE_PACKAGE_OT_Display(bpy.types.Operator):
    """Hide the overlay; safe if not running."""
    bl_idname = "view3d.remove_package_display"
    bl_label  = "Hide Exploratory Overlay"
    bl_options = {'REGISTER'}

    def execute(self, context):
        force_hide_overlay()
        return {'FINISHED'}
