# Exp_Game/systems/exp_live_performance.py
"""
Right-side tall performance column (no temperatures, minimal options).
Only depends on:
  • Scene.show_live_performance_overlay (Bool)
  • Scene.live_perf_scale (Int)

What it shows:
  • Current Performance (grade, smoothed ms, smoothed FPS, 1% low)
  • Physics: steps last frame, target Hz, ~steps/sec actually executed
  • Dynamic Proxy Meshes: ACTIVE / TOTAL  • Dyn BVHs count
  • Static Proxy Meshes: TOTAL (combined into static BVH)
  • Cull entries
  • Game time & time scale
  • Hot stage (largest slice) and Top 3 stage breakdown (from perf_mark)
"""

import time
from collections import deque
import bpy

from ..reactions import exp_custom_ui as ui
from ..props_and_utils.exp_time import get_game_time

# Fixed layout + update cadence (no user options beyond scale + show/hide)
_UPDATE_HZ  = 4          # throttle UI updates to reduce flashing
_ANCHOR     = 'TOP_RIGHT'
_MARGIN_X   = 1
_START_Y    = 1
_ROW_GAP    = 1
_ROWS       = 9          # number of lines in the column

def _grade_from_ms(ema_ms: float, one_low_fps: float, ema_fps: float):
    """Map frame time to a friendly grade; degrade if 1% low is much worse than EMA FPS."""
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

    # If stutter is notable (1% low << EMA), bump grade down one step
    if ema_fps > 0 and one_low_fps > 0 and one_low_fps < 0.7 * ema_fps:
        ladder = ["EXCELLENT", "GREAT", "GOOD", "OK", "POOR", "CRITICAL"]
        try:
            idx = ladder.index(grade)
            grade = ladder[min(idx + 1, len(ladder) - 1)]
            color = (min(1.0, color[0] + 0.05), color[1] * 0.98, color[2] * 0.98, 1.0)
        except ValueError:
            pass
    return grade, color

def _count_proxies(scene):
    dyn_total = 0
    stat_total = 0
    for pm in getattr(scene, "proxy_meshes", []):
        if getattr(pm, "is_moving", False):
            dyn_total += 1
        else:
            stat_total += 1
    return dyn_total, stat_total

class _LivePerfOverlay:
    def __init__(self, modal):
        self.modal = modal

        # smoothing / history
        self.history_ms = deque(maxlen=600)   # ~10s @ 60fps
        self._ema_ms = None
        self._ema_fps = None
        self._sps_ema = None  # physics steps/sec EMA
        self.stage_ms = {}

        self._frame_start = 0.0

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
            return

        total_ms = max(0.0001, (time.perf_counter() - self._frame_start) * 1000.0)
        self.history_ms.append(total_ms)

        fps_inst = 1000.0 / total_ms
        self._ema_ms  = total_ms if self._ema_ms  is None else (self._ema_ms  * 0.90 + total_ms * 0.10)
        self._ema_fps = fps_inst  if self._ema_fps is None else (self._ema_fps * 0.85 + fps_inst  * 0.15)

        one_low_fps = 0.0
        if len(self.history_ms) >= 60:
            ms_sorted = sorted(self.history_ms)
            idx = max(0, int(len(ms_sorted) * 0.99) - 1)
            worst_ms = ms_sorted[idx]
            if worst_ms > 0:
                one_low_fps = 1000.0 / worst_ms

        steps_last = int(getattr(self.modal, "_perf_last_physics_steps", 0) or 0)
        fixed_dt = 0.0
        target_hz = 0
        if getattr(self.modal, "fixed_clock", None):
            fixed_dt = float(getattr(self.modal.fixed_clock, "fixed_dt", 0.0) or 0.0)
            target_hz = int(round(1.0 / fixed_dt)) if fixed_dt > 0 else 0

        sps_inst = steps_last * fps_inst
        self._sps_ema = sps_inst if self._sps_ema is None else (self._sps_ema * 0.85 + sps_inst * 0.15)

        dyn_active = int(getattr(self.modal, "_dyn_active_count", 0) or 0)
        dyn_bvhs   = len(getattr(self.modal, "dynamic_bvh_map", {}) or {})
        dyn_total, stat_total = _count_proxies(scene)

        cull_entries = len(getattr(scene, "performance_entries", []))
        gtime = get_game_time()
        tscale = float(getattr(self.modal, "time_scale", 1.0))

        top_name, top_ms, top_pct = "—", 0.0, 0
        if self.stage_ms:
            top_name, top_ms = max(self.stage_ms.items(), key=lambda kv: kv[1])
            top_pct = int(round((top_ms / total_ms) * 100.0))
        top3 = sorted(self.stage_ms.items(), key=lambda kv: kv[1], reverse=True)
        top3_txt = " • ".join(f"{k}:{v:.1f}ms" for k, v in top3[:3]) if top3 else ""

        grade, hdr_col = _grade_from_ms(self._ema_ms, one_low_fps, self._ema_fps)

        # throttle UI
        now = time.perf_counter()
        if (now - self._last_ui_update) < (1.0 / _UPDATE_HZ):
            return
        self._last_ui_update = now

        self._ensure_items(scene)

        # ---------- write lines ----------
        # 0: title
        self._items[0]["text"]  = "PERFORMANCE"
        self._items[0]["color"] = hdr_col

        # 1: current perf summary
        fps_s = f"{int(round(self._ema_fps))}"
        low_s = f"{int(round(one_low_fps))}" if one_low_fps > 0 else "—"
        ms_s  = f"{self._ema_ms:4.1f}"
        self._items[1]["text"]  = f"Current: {grade} • {ms_s} ms • ~{fps_s} FPS (1% {low_s})"
        self._items[1]["color"] = (1,1,1,0.95)

        # 2: physics
        sps_s = f"{int(round(self._sps_ema))}" if self._sps_ema is not None else "0"
        self._items[2]["text"]  = f"Physics: {steps_last} steps @ {target_hz} Hz  (~{sps_s}/s)"
        self._items[2]["color"] = (0.90,0.95,1.00,0.95)

        # 3: dynamic proxies
        self._items[3]["text"]  = f"Dynamic Proxy Meshes: {dyn_active}/{dyn_total}  •  Dyn BVHs: {dyn_bvhs}"
        self._items[3]["color"] = (0.95,0.95,1.00,0.95)

        # 4: static proxies
        self._items[4]["text"]  = f"Static Proxy Meshes: {stat_total}  (in static BVH)"
        self._items[4]["color"] = (0.95,0.95,1.00,0.95)

        # 5: culling
        self._items[5]["text"]  = f"Culling Entries: {cull_entries}"
        self._items[5]["color"] = (0.90,1.00,0.90,0.95)

        # 6: clock
        self._items[6]["text"]  = f"Game Time: {gtime:6.2f}s  •  Time Scale ×{tscale:.2f}"
        self._items[6]["color"] = (1,1,1,0.85)

        # 7: hot stage
        self._items[7]["text"]  = f"Hot: {top_name}  {top_ms:.1f} ms  {top_pct}%"
        self._items[7]["color"] = (1.00,0.90,0.80,0.95)

        # 8: top 3
        self._items[8]["text"]  = f"Top: {top3_txt}"
        self._items[8]["color"] = (1,1,1,0.80)

    # ---- timing marks ----
    def mark(self, stage, start=None):
        if start is None:
            return time.perf_counter()
        dur_ms = (time.perf_counter() - start) * 1000.0
        self.stage_ms[stage] = self.stage_ms.get(stage, 0.0) + dur_ms

    # ---- UI plumbing ----
    def _ensure_items(self, scene):
        scale = int(getattr(scene, "live_perf_scale", 2) or 2)
        if self._created:
            # if item objects vanished (unlikely), rebuild
            if any(it not in ui._reaction_texts for it in self._items):
                self._teardown_items()
            else:
                # update scale if changed by recreating (simplest)
                try:
                    current_scale = self._items[0].get("scale", None)
                except Exception:
                    current_scale = None
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


# ---- Public API (unchanged) ----

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
