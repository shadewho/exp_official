# Exp_Game/startup_and_reset/exp_fullscreen.py
# Safe BASIC fullscreen enter/exit with UI hide/restore (no wm.redraw_timer)

import bpy

_UI_DELAY_SEC = 0.01  # tiny defer so the fullscreen screen exists before we edit its space

_state = {
    "entering": False,
    "exiting":  False,
    "did_enter": False,     # True iff we toggled into fullscreen this session
    "win_ptr": None,        # window we acted in
    "ui_hidden": False,     # have we hidden space UI
    "orig_ui": {},          # sid -> {flag_name: original_value, "_overlay_show": bool}
    "screen_statusbars": {},# screen_ptr -> original show_statusbar bool
}

# ───────────── helpers ─────────────

def _ptr(x): return int(x.as_pointer()) if x else None

def _is_area_fullscreen(win) -> bool:
    scr = getattr(win, "screen", None)
    return bool(scr and len(scr.areas) == 1)

def _find_view3d(win):
    """Return (area, region, space) for first 3D View WINDOW region in win."""
    if not win or not win.screen:
        return None, None, None
    for area in win.screen.areas:
        if area.type == 'VIEW_3D':
            reg = next((r for r in area.regions if r.type == 'WINDOW'), None)
            if reg:
                return area, reg, area.spaces.active
    return None, None, None

def _remember_and_set(space, attr, value, bucket):
    if hasattr(space, attr):
        try:
            if attr not in bucket:
                bucket[attr] = getattr(space, attr)
            setattr(space, attr, value)
        except Exception:
            pass

def _hide_view3d_ui(space):
    """Hide header/tool header/toolbar/N-panel/gizmos/overlays and remember originals."""
    if not space:
        return
    sid = _ptr(space)
    if sid not in _state["orig_ui"]:
        _state["orig_ui"][sid] = {}
    B = _state["orig_ui"][sid]

    _remember_and_set(space, "show_region_header",       False, B)
    _remember_and_set(space, "show_region_tool_header",  False, B)
    _remember_and_set(space, "show_region_toolbar",      False, B)
    _remember_and_set(space, "show_region_ui",           False, B)
    _remember_and_set(space, "show_gizmo",               False, B)

    try:
        if hasattr(space, "overlay") and hasattr(space.overlay, "show_overlays"):
            if "_overlay_show" not in B:
                B["_overlay_show"] = space.overlay.show_overlays
            space.overlay.show_overlays = False
    except Exception:
        pass

    _state["ui_hidden"] = True

def _restore_view3d_ui(space):
    """Restore any UI flags we changed earlier on this SpaceView3D."""
    if not space:
        return
    sid = _ptr(space)
    B = _state["orig_ui"].get(sid)
    if not B:
        return

    for k, v in list(B.items()):
        try:
            if k == "_overlay_show":
                space.overlay.show_overlays = v
            else:
                if hasattr(space, k):
                    setattr(space, k, v)
        except Exception:
            pass

    _state["ui_hidden"] = False
    _state["orig_ui"].pop(sid, None)

# ---- statusbar per-screen helpers ----

def _set_screen_statusbar(screen, visible: bool):
    """Set screen.show_statusbar, remembering original once per Screen."""
    if not screen or not hasattr(screen, "show_statusbar"):
        return
    sptr = _ptr(screen)
    if sptr not in _state["screen_statusbars"]:
        _state["screen_statusbars"][sptr] = bool(screen.show_statusbar)
    try:
        screen.show_statusbar = bool(visible)
    except Exception:
        pass

def _restore_all_statusbars():
    """Restore show_statusbar for all screens we touched."""
    try:
        for scr in bpy.data.screens:
            sptr = _ptr(scr)
            if sptr in _state["screen_statusbars"]:
                try:
                    scr.show_statusbar = _state["screen_statusbars"][sptr]
                except Exception:
                    pass
    finally:
        _state["screen_statusbars"].clear()

# ───────────── Public API ─────────────

def enter_fullscreen_once():
    """
    If already in area-fullscreen:
        - Do NOT toggle (would exit). Just hide UI + turn off screen.show_statusbar.
    Else:
        - Toggle ON now; next tick hide UI on the fullscreen copy and turn off its screen.show_statusbar.
    """
    if _state["entering"]:
        return
    _state["entering"] = True

    win = bpy.context.window
    if not win:
        _state["entering"] = False
        return

    _state["win_ptr"] = _ptr(win)

    # Always turn off statusbar on the current screen
    _set_screen_statusbar(win.screen, False)

    # Case A: already fullscreen → no toggle; just hide UI now.
    if _is_area_fullscreen(win):
        try:
            _a, _r, s = _find_view3d(win)
            _hide_view3d_ui(s)
        finally:
            _state["did_enter"] = False
            _state["entering"] = False
        return

    # Case B: not fullscreen → toggle ON now, then hide UI + hide statusbar on the fullscreen screen next tick.
    area, region, _ = _find_view3d(win)
    if not area or not region:
        _state["entering"] = False
        return

    try:
        with bpy.context.temp_override(window=win, screen=win.screen, area=area, region=region):
            bpy.ops.screen.screen_full_area('EXEC_DEFAULT')
        _state["did_enter"] = True
    except Exception:
        _state["entering"] = False
        return

    def _hide_next_tick():
        try:
            # After toggle, Blender switches to a temp Screen; hide UI + its statusbar too.
            w = bpy.context.window
            a2, r2, s2 = _find_view3d(w)
            _hide_view3d_ui(s2)
            if w and w.screen:
                _set_screen_statusbar(w.screen, False)
        finally:
            _state["entering"] = False
        return None
    bpy.app.timers.register(_hide_next_tick, first_interval=_UI_DELAY_SEC)


def exit_fullscreen_once():
    """
    Restore any hidden UI first. Then, if we toggled in ourselves (did_enter=True),
    toggle OFF once to return to the original layout. Finally, restore statusbars.
    """
    if _state["exiting"]:
        return
    _state["exiting"] = True

    def _restore_and_maybe_toggle_off():
        try:
            # Restore UI on current VIEW_3D (fullscreen copy or original)
            a, r, s = _find_view3d(bpy.context.window)
            if s:
                _restore_view3d_ui(s)

            # Only leave fullscreen if we actually entered it here
            if _state["did_enter"]:
                win = None
                for w in bpy.context.window_manager.windows:
                    if _ptr(w) == _state["win_ptr"]:
                        win = w
                        break
                if win and win.screen:
                    a2, r2, _s2 = _find_view3d(win)
                    if a2 and r2:
                        try:
                            with bpy.context.temp_override(window=win, screen=win.screen, area=a2, region=r2):
                                bpy.ops.screen.screen_full_area('EXEC_DEFAULT')
                        except Exception:
                            pass
        finally:
            # Restore all screens we modified
            _restore_all_statusbars()
            _state["did_enter"] = False
            _state["exiting"] = False
        return None

    bpy.app.timers.register(_restore_and_maybe_toggle_off, first_interval=_UI_DELAY_SEC)


def reset_fullscreen_state():
    """
    Reset fullscreen state. Call on game end to clear accumulated state.
    This prevents stale UI references from persisting across game sessions.
    """
    _state["entering"] = False
    _state["exiting"] = False
    _state["did_enter"] = False
    _state["win_ptr"] = None
    _state["ui_hidden"] = False
    _state["orig_ui"].clear()
    _state["screen_statusbars"].clear()
