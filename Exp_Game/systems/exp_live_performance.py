# Exp_Game/systems/exp_live_performance.py
"""
Top-right live performance HUD (compact & human-readable).

Displays:
  • Summary grade, smoothed work ms, smoothed FPS (from wall-clock tick spacing), and 1% low FPS
  • Budget bar vs 60 FPS (16.7 ms) using smoothed work ms
  • Top "hot" stage with a mini bar and % of frame
  • Top 3 stage breakdown (friendly names)
  • Physics (steps this frame, target Hz, ~steps/sec)
  • World activity (dynamic/static proxies, active dyn BVHs)
  • Gameplay snapshot (anim state, grounded, vZ, time & time-scale)

No changes to the anchor or perf_mark() usage.
"""

import time
from collections import deque
import bpy

from ..reactions import exp_custom_ui as ui
from ..props_and_utils.exp_time import get_game_time
from ..animations.exp_animations import get_global_animation_manager

# ------- Layout & cadence (unchanged anchor) -------
_UPDATE_HZ  = 4
_ANCHOR     = 'TOP_RIGHT'   # do not change
_MARGIN_X   = 1
_START_Y    = 1
_ROW_GAP    = 1
_ROWS       = 8             # 8 visible lines (no hints line)

# ------- Friendly labels for stages -------
_FRIENDLY = {
    "time":                "Timing",
    "anim_audio":          "Anim+Audio",
    "custom_tasks":        "Custom",
    "dynamic_meshes":      "Dynamics",
    "culling":             "Culling",
    "physics":             "Physics",
    "interact_ui_audio":   "UI/Interact/Audio",
}

def _count_proxies(scene):
    dyn_total = 0
    stat_total = 0
    for pm in getattr(scene, "proxy_meshes", []):
        if getattr(pm, "is_moving", False):
            dyn_total += 1
        else:
            stat_total += 1
    return dyn_total, stat_total

def _grade_from_ms(ema_ms: float, one_low_fps: float, ema_fps: float):
    """Grade & header color based on smoothed frame work time; degrade for stutter."""
    if ema_ms <= 12.0:
        grade, color = ("EXCELLENT", (0.65, 1.00, 0.65, 1.0))
    elif ema_ms <= 16.7:
        grade, color = ("GREAT",     (0.75, 1.00, 0.75, 1.0))
    elif ema_ms <= 25.0:
        grade, color = ("GOOD",      (1.00, 1.00, 0.70, 1.0))
    elif ema_ms <= 33.3:
        grade, color = ("OK",        (1.00, 0.90, 0.60, 1.0))
    elif ema_ms <= 50.0:
        grade, color = ("POOR",      (1.00, 0.70, 0.60, 1.0))
    else:
        grade, color = ("CRITICAL",  (1.00, 0.55, 0.55, 1.0))

    # if 1% low is much lower than EMA FPS, degrade one step
    if ema_fps > 0 and one_low_fps > 0 and one_low_fps < 0.7 * ema_fps:
        ladder = ["EXCELLENT", "GREAT", "GOOD", "OK", "POOR", "CRITICAL"]
        try:
            idx = ladder.index(grade)
            grade = ladder[min(idx + 1, len(ladder) - 1)]
            color = (min(1.0, color[0] + 0.05), color[1] * 0.98, color[2] * 0.98, 1.0)
        except ValueError:
            pass
    return grade, color

def _severity_color(ms: float, budget_ms: float = 16.7):
    """Color by how much a cost consumes of a 60 FPS budget."""
    if budget_ms <= 0.0:
        return (1,1,1,0.95)
    r = ms / budget_ms
    if r <= 0.6:  return (0.85, 1.00, 0.85, 0.98)  # light green
    if r <= 1.0:  return (1.00, 0.97, 0.80, 0.98)  # yellow-ish
    if r <= 1.5:  return (1.00, 0.80, 0.65, 0.98)  # orange
    return (1.00, 0.55, 0.55, 0.98)                # red

def _bar(value: float, max_value: float, width: int = 16):
    """ASCII budget bar using █ and ░; falls back if width < 1."""
    width = max(1, int(width))
    if max_value <= 1e-9:
        return "░" * width
    frac = max(0.0, min(1.0, float(value) / float(max_value)))
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)

def _friendly(name: str) -> str:
    return _FRIENDLY.get(name, name or "—")


class _LivePerfOverlay:
    def __init__(self, modal):
        self.modal = modal

        # smoothing / history
        self.history_ms = deque(maxlen=600)   # work ms history (~10s @ 60fps)
        self._ema_ms = None                   # EMA of work ms (handler time)
        self._ema_fps = None                  # EMA of FPS (from tick spacing)
        self._sps_ema = None                  # EMA of physics steps/sec
        self.stage_ms = {}                    # filled by perf_mark()

        # wall-clock tick spacing for realistic FPS + 1% low
        self._last_tick_t = None
        self._tick_hist = deque(maxlen=600)   # store dt between HUD updates

        self._frame_start = 0.0               # start time for work ms in this frame

        self._items = []
        self._created = False
        self._last_ui_update = 0.0

    # ---- lifecycle ----
    def begin_frame(self):
        self._frame_start = time.perf_counter()
        self.stage_ms.clear()

    def end_frame(self, context):
        scene = context.scene
        if not getattr(scene, "show_live_performance_overlay", False):
            self._teardown_items()
            # reset tick anchor so next enable starts fresh
            self._last_tick_t = None
            return

        # A) Work time (handler duration) in ms
        total_ms = max(0.0001, (time.perf_counter() - self._frame_start) * 1000.0)
        self.history_ms.append(total_ms)
        self._ema_ms = total_ms if self._ema_ms is None else (self._ema_ms * 0.90 + total_ms * 0.10)

        # B) Realistic FPS based on wall-clock tick spacing (time between HUD updates)
        now = time.perf_counter()
        fps_inst = 0.0
        if self._last_tick_t is not None:
            tick_dt = max(1e-6, now - self._last_tick_t)
            fps_inst = 1.0 / tick_dt
            self._tick_hist.append(tick_dt)
            self._ema_fps = fps_inst if self._ema_fps is None else (self._ema_fps * 0.85 + fps_inst * 0.15)
        self._last_tick_t = now

        # C) 1% low FPS based on tick spacing
        one_low_fps = 0.0
        if len(self._tick_hist) >= 60:
            dts = sorted(self._tick_hist)
            idx = min(len(dts) - 1, max(0, int(len(dts) * 0.99) - 1))
            worst_dt = dts[idx]
            one_low_fps = (1.0 / worst_dt) if worst_dt > 0 else 0.0

        # D) Physics info
        steps_last = int(getattr(self.modal, "_perf_last_physics_steps", 0) or 0)
        target_hz  = int(getattr(self.modal, "physics_hz", 0) or 0)
        sps_inst   = steps_last * (fps_inst if fps_inst > 0 else (self._ema_fps or 0.0))
        self._sps_ema = sps_inst if self._sps_ema is None else (self._sps_ema * 0.85 + sps_inst * 0.15)

        # E) World activity
        dyn_active = int(getattr(self.modal, "_dyn_active_count", 0) or 0)
        dyn_bvhs   = len(getattr(self.modal, "dynamic_bvh_map", {}) or {})
        dyn_total, stat_total = _count_proxies(scene)

        # F) Hot stage & top 3 from perf_mark slices
        top_name, top_ms, top_pct = "—", 0.0, 0
        if self.stage_ms:
            top_name, top_ms = max(self.stage_ms.items(), key=lambda kv: kv[1])
            top_pct = int(round((top_ms / total_ms) * 100.0))
        top3 = sorted(self.stage_ms.items(), key=lambda kv: kv[1], reverse=True)

        # G) Grade & header color
        ema_fps_val = self._ema_fps if self._ema_fps is not None else 0.0
        grade, hdr_col = _grade_from_ms(self._ema_ms, one_low_fps, ema_fps_val)

        # H) throttle HUD writes
        if (now - self._last_ui_update) < (1.0 / _UPDATE_HZ):
            return
        self._last_ui_update = now

        self._ensure_items(scene)

        # ---------- line 0: title ----------
        self._items[0]["text"]  = "PERFORMANCE"
        self._items[0]["color"] = hdr_col

        # ---------- line 1: summary ----------
        fps_s = f"{int(round(ema_fps_val))}" if ema_fps_val > 0 else "—"
        low_s = f"{int(round(one_low_fps))}" if one_low_fps > 0 else "—"
        ms_s  = f"{self._ema_ms:4.1f}"
        self._items[1]["text"]  = f"{grade} • {ms_s} ms • ~{fps_s} FPS (1% {low_s})"
        self._items[1]["color"] = (1,1,1,0.95)

        # ---------- line 2: 60 FPS budget bar ----------
        budget = 16.7  # ms for 60 FPS
        self._items[2]["text"]  = f"Budget 60FPS: [{_bar(self._ema_ms, budget, 16)}]  {self._ema_ms:4.1f}/{budget:.1f} ms"
        self._items[2]["color"] = _severity_color(self._ema_ms, budget)

        # ---------- line 3: top stage ----------
        if top_ms > 0.0:
            top_label = _friendly(top_name)
            self._items[3]["text"]  = f"Hot: {top_label}  {top_ms:.1f} ms  ({top_pct}%)  [{_bar(top_ms, total_ms, 14)}]"
            self._items[3]["color"] = _severity_color(top_ms, budget)
        else:
            self._items[3]["text"]  = "Hot: —"
            self._items[3]["color"] = (1,1,1,0.85)

        # ---------- line 4: compact breakdown (top 3) ----------
        if top3:
            parts = [f"{_friendly(n)} {ms:.1f}ms" for n, ms in top3[:3]]
            self._items[4]["text"]  = "Breakdown: " + " • ".join(parts)
        else:
            self._items[4]["text"]  = "Breakdown: —"
        self._items[4]["color"] = (1,1,1,0.85)

        # ---------- line 5: world activity ----------
        self._items[5]["text"]  = f"Dynamic Proxies {dyn_active}/{dyn_total}  •  Dynamic Proxies {dyn_bvhs}  •  Static Proxies {stat_total}"
        self._items[5]["color"] = (0.95,0.95,1.00,0.95)

        # ---------- line 6: physics ----------
        sps_s = f"{int(round(self._sps_ema))}" if self._sps_ema is not None else "0"
        self._items[6]["text"]  = f"Physics: {steps_last} step{' ' if steps_last==1 else 's'} @ {target_hz} Hz  (~{sps_s}/s)"
        self._items[6]["color"] = (0.90,0.95,1.00,0.95)

        # ---------- line 7: gameplay snapshot ----------
        try:
            mgr = get_global_animation_manager()
            anim_state = mgr.anim_state.name if (mgr and getattr(mgr, "anim_state", None) is not None) else "—"
        except Exception:
            anim_state = "—"
        grounded = bool(getattr(self.modal, "is_grounded", False))
        vz = float(getattr(self.modal, "z_velocity", 0.0) or 0.0)
        gtime = get_game_time()
        tscale = float(getattr(self.modal, "time_scale", 1.0))
        gflag = "✓" if grounded else "✕"
        vz_s = f"{vz:+.2f}"
        self._items[7]["text"]  = f"State: {anim_state}  •  Grounded {gflag}  •  vZ {vz_s}  •  t {gtime:5.2f}s ×{tscale:.2f}"
        self._items[7]["color"] = (1,1,1,0.85)

    # ---- timing marks (unchanged) ----
    def mark(self, stage, start=None):
        if start is None:
            return time.perf_counter()
        dur_ms = (time.perf_counter() - start) * 1000.0
        self.stage_ms[stage] = self.stage_ms.get(stage, 0.0) + dur_ms

    # ---- UI plumbing ----
    def _ensure_items(self, scene):
        scale = int(getattr(scene, "live_perf_scale", 2) or 2)
        if self._created:
            # rebuild if vanished, or if scale changed
            if any(it not in ui._reaction_texts for it in self._items):
                self._teardown_items()
            else:
                current_scale = None
                try:
                    current_scale = self._items[0].get("scale", None)
                except Exception:
                    pass
                if current_scale == scale:
                    return
                self._teardown_items()

        self._items = []
        for i in range(_ROWS):
            item = ui.add_text_reaction(
                text_str="",
                anchor=_ANCHOR,
                margin_x=_MARGIN_X,
                margin_y=_START_Y + i * _ROW_GAP,
                scale=scale,
                end_time=None,
                color=(1,1,1,0.95),
                font_name="DEFAULT",
            )
            self._items.append(item)
        self._created = True

    def _teardown_items(self):
        if not self._created:
            return
        try:
            for it in self._items:
                if it in ui._reaction_texts:
                    ui._reaction_texts.remove(it)
        except Exception:
            pass
        self._items.clear()
        self._created = False


# ---- Public API (same names you already use) ----

def init_live_performance(modal):
    modal._live_perf = _LivePerfOverlay(modal)

def perf_frame_begin(modal):
    lp = getattr(modal, "_live_perf", None)
    if lp:
        lp.begin_frame()

def perf_mark(modal, stage, t0=None):
    lp = getattr(modal, "_live_perf", None)
    if not lp:
        return None
    if t0 is None:
        return lp.mark(stage, None)
    lp.mark(stage, t0)

def perf_frame_end(modal, context):
    lp = getattr(modal, "_live_perf", None)
    if lp:
        lp.end_frame(context)
