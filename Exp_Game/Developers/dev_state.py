from __future__ import annotations
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Deque, List, Tuple, Any

import bpy
from .dev_utils import pyget

# -------- series and meters --------

@dataclass
class Series:
    name: str
    maxlen: int
    values: Deque[float]
    def push(self, v: float): self.values.append(float(v))
    def set_maxlen(self, m: int):
        m = int(max(30, min(2000, m)))
        if m == self.maxlen: return
        old = list(self.values)
        self.values = deque(old[-m:], maxlen=m)
        self.maxlen = m
    @property
    def last(self) -> float | None:
        return self.values[-1] if self.values else None
    def minmax(self, fallback=(0.0, 1.0)):
        if not self.values: return fallback
        lo, hi = min(self.values), max(self.values)
        return (lo, hi if hi > lo else lo + 1.0)

class EventMeter:
    __slots__ = ("ts",)
    def __init__(self):
        self.ts: Deque[float] = deque()
    def record(self, t: float):
        self.ts.append(float(t))
    def rate(self, now: float, window: float = 1.0) -> float:
        w = max(0.1, float(window)); cutoff = now - w
        while self.ts and self.ts[0] < cutoff:
            self.ts.popleft()
        return (len(self.ts) / w) if w > 0 else 0.0

class _State:
    def __init__(self):
        # Core series
        self.series: Dict[str, Series] = {
            "frame_ms":    Series("frame_ms",    300, deque(maxlen=300)),
            "fps":         Series("fps",         300, deque(maxlen=300)),
            "rtt_ms":      Series("rtt_ms",      300, deque(maxlen=300)),
            "phase_ms":    Series("phase_ms",    300, deque(maxlen=300)),
            "view_allowed":   Series("view_allowed",   300, deque(maxlen=300)),
            "view_candidate": Series("view_candidate", 300, deque(maxlen=300)),
            "view_delta":     Series("view_delta",     300, deque(maxlen=300)),
            "view_lag_ms":    Series("view_lag_ms",    300, deque(maxlen=300)),
            "phys_down_dist": Series("phys_down_dist", 300, deque(maxlen=300)),
            "phys_n_up":      Series("phys_n_up",      300, deque(maxlen=300)),
            "phys_vel_z":     Series("phys_vel_z",     300, deque(maxlen=300)),
        }
        # Frame timing
        self.enabled = False
        self.frame_start = 0.0
        self.last_end_t  = 0.0
        self.frame_idx   = 0
        self.fps_ema = 0.0
        self.ms_ema  = 0.0
        # Stable custom key catalog
        self.custom_groups_order: List[str] = []
        self.custom_keys_by_group: Dict[str, List[str]] = {}
        self.custom_last_seen: Dict[str, int] = {}
        self.custom_last_val: Dict[str, Any] = {}
        # View metrics
        self.view_last_allowed: float | None = None
        self.view_last_allowed_wall: float = 0.0
        self.view_last_candidate_wall: float = 0.0
        self.view_lag_ema_ms: float = 0.0
        self.view_jitter_ema: float = 0.0
        # Event meters
        self.meter_timer      = EventMeter()
        self.meter_view_queue = EventMeter()
        self.meter_view_apply = EventMeter()
        self.meter_xr_req   = EventMeter()
        self.meter_xr_ok    = EventMeter()
        self.meter_xr_fail  = EventMeter()
        # XR deltas
        self.xr_req_last = 0
        self.xr_ok_last  = 0
        self.xr_fail_last= 0
        # rate-limited console emits
        self.last_console_emit = 0.0
        self.last_view_emit    = 0.0
        self.last_xr_emit      = 0.0

        #phys
        self.last_phys_emit = 0.0

STATE = _State()

# -------- helpers used by draw and end-frame --------

def world_counts(scene, modal) -> Tuple[int,int,int,int]:
    dyn_total = stat_total = 0
    try:
        for pm in getattr(scene, "proxy_meshes", []):
            if getattr(pm, "is_moving", False): dyn_total += 1
            else: stat_total += 1
    except Exception:
        pass
    dyn_bvhs = len(pyget(modal, "dynamic_bvh_map", {}) or {}) if modal else 0
    dyn_active = 0
    das = pyget(modal, "_dyn_active_state", None)
    if isinstance(das, dict):
        dyn_active = sum(1 for v in das.values() if v is True)
    elif modal:
        dyn_active = dyn_bvhs
    return dyn_active, dyn_bvhs, dyn_total, stat_total

def catalog_update_from_bus(BUS):
    merged = {}
    merged.update(BUS.scalars)
    merged.update(BUS.temp)
    for k, v in merged.items():
        if not isinstance(k, str): continue
        grp = k.split(".", 1)[0].upper()
        if grp not in STATE.custom_keys_by_group:
            STATE.custom_keys_by_group[grp] = []
            STATE.custom_groups_order.append(grp)
        if k not in STATE.custom_keys_by_group[grp]:
            STATE.custom_keys_by_group[grp].append(k)
        STATE.custom_last_val[k]  = v
        STATE.custom_last_seen[k] = STATE.frame_idx

# -------- frame lifecycle --------

def frame_begin(modal, BUS):
    scene = bpy.context.scene
    if scene is None: return
    # Clear volatile bus each TIMER tick
    try:
        BUS.temp.clear()
    except Exception:
        pass
    STATE.frame_start = time.perf_counter()
    STATE.frame_idx  += 1
    STATE.meter_timer.record(STATE.frame_start)

def _parse_first_float(x) -> float | None:
    try:
        if isinstance(x, (int, float)): return float(x)
        if not isinstance(x, str): return None
        import re
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", x)
        return float(m.group(0)) if m else None
    except Exception:
        return None

def frame_end(modal, context, BUS):
    scene = context.scene if context else None
    if scene is None: return
    now = time.perf_counter()

    # frame ms
    if STATE.frame_start > 0.0:
        ms = (now - STATE.frame_start) * 1000.0
        STATE.series["frame_ms"].push(ms)
        STATE.ms_ema = ms if STATE.ms_ema == 0.0 else (STATE.ms_ema * 0.90 + ms * 0.10)

    # fps EMA
    if STATE.last_end_t > 0.0:
        dt = max(1e-6, now - STATE.last_end_t)
        fps_inst = 1.0 / dt
        STATE.fps_ema = fps_inst if STATE.fps_ema == 0.0 else (STATE.fps_ema * 0.85 + fps_inst * 0.15)
        STATE.series["fps"].push(STATE.fps_ema)
    STATE.last_end_t = now

    # XR counters (delta → meters)
    xs = pyget(modal, "_xr_stats", None) if modal else None
    if isinstance(xs, dict):
        rtt = float(xs.get("last_lat_ms", 0.0))
        ph  = float(xs.get("phase_ms", 0.0))
        STATE.series["rtt_ms"].push(rtt)
        STATE.series["phase_ms"].push(ph)
        req = int(xs.get("req", 0) or 0); ok = int(xs.get("ok", 0) or 0); fail= int(xs.get("fail", 0) or 0)
        if req > STATE.xr_req_last:
            for _ in range(req - STATE.xr_req_last): STATE.meter_xr_req.record(now)
        if ok > STATE.xr_ok_last:
            for _ in range(ok - STATE.xr_ok_last): STATE.meter_xr_ok.record(now)
        if fail > STATE.xr_fail_last:
            for _ in range(fail - STATE.xr_fail_last): STATE.meter_xr_fail.record(now)
        STATE.xr_req_last, STATE.xr_ok_last, STATE.xr_fail_last = req, ok, fail

    # View series mirroring
    cand_v = None
    if "view_candidate" in BUS.series and BUS.series["view_candidate"].values:
        cand_v = float(BUS.series["view_candidate"].values[-1])
        STATE.series["view_candidate"].push(cand_v)
    else:
        v = BUS.temp.get("VIEW.candidate", None)
        v = _parse_first_float(v)
        if v is not None:
            cand_v = float(v); STATE.series["view_candidate"].push(cand_v)

    allowed_v = None
    if "view_allowed" in BUS.series and BUS.series["view_allowed"].values:
        allowed_v = float(BUS.series["view_allowed"].values[-1])
        STATE.series["view_allowed"].push(allowed_v)
    else:
        v = BUS.temp.get("VIEW.allowed", None)
        v = _parse_first_float(v)
        if v is not None:
            allowed_v = float(v)
            STATE.series["view_allowed"].push(allowed_v)
            if (now - STATE.view_last_allowed_wall) > (1.0 / 90.0):
                STATE.meter_view_apply.record(now)
                if STATE.view_last_candidate_wall > 0.0:
                    lag_ms = (now - STATE.view_last_candidate_wall) * 1000.0
                    STATE.series["view_lag_ms"].push(lag_ms)
                    STATE.view_lag_ema_ms = lag_ms if STATE.view_lag_ema_ms == 0.0 else (STATE.view_lag_ema_ms*0.8 + lag_ms*0.2)
                STATE.view_last_allowed_wall = now

    va = STATE.series["view_allowed"].last
    vc = STATE.series["view_candidate"].last
    if (va is not None) and (vc is not None):
        STATE.series["view_delta"].push(float(vc) - float(va))

    # ---- NEW: physics series mirroring ----
    # Down ray distance (string in BUS -> float)
    ddist_txt = BUS.temp.get("PHYS.down.dist", BUS.scalars.get("PHYS.down.dist", None))
    ddist = _parse_first_float(ddist_txt)
    if ddist is not None and "phys_down_dist" in STATE.series:
        STATE.series["phys_down_dist"].push(float(ddist))

    # n·up if KCC provided (else try BUS)
    nup_txt = BUS.temp.get("PHYS.nup", BUS.scalars.get("PHYS.nup", None))
    nup = _parse_first_float(nup_txt)
    if nup is None:
        pc = pyget(modal, "physics_controller", None) if modal else None
        if pc and getattr(pc, "ground_norm", None) is not None:
            gn = pc.ground_norm
            try:
                nup = float(gn.z)
            except Exception:
                nup = None
    if (nup is not None) and ("phys_n_up" in STATE.series):
        STATE.series["phys_n_up"].push(float(nup))

    # vel.z from modal
    zvel = float(pyget(modal, "z_velocity", 0.0) or 0.0) if modal else 0.0
    if "phys_vel_z" in STATE.series:
        STATE.series["phys_vel_z"].push(zvel)

    # resize buffers
    maxlen = int(getattr(scene, "dev_hud_max_samples", 300))
    for s in STATE.series.values(): s.set_maxlen(maxlen)
    for s in BUS.series.values():   s.set_maxlen(maxlen)

    # -------- Console logging (existing) --------
    if scene.dev_hud_log_console and (now - STATE.last_console_emit) >= 1.0:
        STATE.last_console_emit = now
        fps_txt = f"{STATE.fps_ema:.1f}" if STATE.fps_ema > 0 else "—"
        ms_txt  = f"{STATE.ms_ema:.2f}" if STATE.ms_ema > 0 else "—"
        rtt = STATE.series["rtt_ms"].last;  rtt_txt = f"{(rtt or 0):.2f}"
        ph  = STATE.series["phase_ms"].last; ph_txt  = f"{(ph or 0):+.2f}"
        print(f"[DEVHUD] frame {ms_txt} ms  ~{fps_txt} FPS  XR rtt {rtt_txt} ms  phase {ph_txt} ms")

    if getattr(scene, "dev_log_xr_console", False):
        period = 1.0 / max(0.1, float(getattr(scene, "dev_log_xr_hz", 2.0)))
        if (now - STATE.last_xr_emit) >= period:
            STATE.last_xr_emit = now
            xr_rq = STATE.meter_xr_req.rate(now, 1.0)
            xr_ok = STATE.meter_xr_ok.rate(now, 1.0)
            xr_fl = STATE.meter_xr_fail.rate(now, 1.0)
            rtt = STATE.series["rtt_ms"].last
            ph  = STATE.series["phase_ms"].last
            print(f"[XR] req/s {xr_rq:.1f}  ok/s {xr_ok:.1f}  fail/s {xr_fl:.1f}  rtt {(rtt or 0):.2f} ms  phase {(ph or 0):+.2f} ms")

    if getattr(scene, "dev_log_view_console", False):
        period = 1.0 / max(0.1, float(getattr(scene, "dev_log_view_hz", 4.0)))
        if (now - STATE.last_view_emit) >= period:
            STATE.last_view_emit = now
            qhz = STATE.meter_view_queue.rate(now, 1.0)
            ahz = STATE.meter_view_apply.rate(now, 1.0)
            va  = STATE.series["view_allowed"].last
            vc  = STATE.series["view_candidate"].last
            dd  = (vc - va) if (vc is not None and va is not None) else float("nan")
            hit = BUS.temp.get("VIEW.hit", BUS.scalars.get("VIEW.hit", None))
            src = BUS.temp.get("VIEW.src", BUS.scalars.get("VIEW.src", None))
            lag = STATE.series["view_lag_ms"].last
            print(f"[VIEW] qHz {qhz:.1f}  aHz {ahz:.1f}  lag {(lag or 0):.1f} ms  dist {(va or 0):.3f} m  cand {(vc or 0):.3f} m  Δ {(dd if dd==dd else 0):+.3f} m  src {src or '—'}  hit {hit or '—'}")

    # ---- NEW: Physics console logging ----
    if getattr(scene, "dev_log_physics_console", False):
        period = 1.0 / max(0.1, float(getattr(scene, "dev_log_physics_hz", 3.0)))
        if (now - STATE.last_phys_emit) >= period:
            STATE.last_phys_emit = now
            steps = int(pyget(modal, "_perf_last_physics_steps", 0) or 0) if modal else 0
            hz    = int(pyget(modal, "physics_hz", 30) or 30)
            pc    = pyget(modal, "physics_controller", None) if modal else None
            on_g  = bool(pyget(modal, "is_grounded", False))
            walk  = bool(getattr(pc, "on_walkable", False)) if pc else False
            zvel  = float(pyget(modal, "z_velocity", 0.0) or 0.0) if modal else 0.0
            coy   = float(getattr(pc, "_coyote", 0.0) if pc else 0.0)
            snap  = float(getattr(scene, "char_physics", None).snap_down) if getattr(scene, "char_physics", None) else float('nan')
            nup   = STATE.series["phys_n_up"].last
            ddist = STATE.series["phys_down_dist"].last
            dhit  = BUS.temp.get("PHYS.down.hit", BUS.scalars.get("PHYS.down.hit", "—"))
            dobj  = BUS.temp.get("PHYS.down.obj", BUS.scalars.get("PHYS.down.obj", "—"))
            print(f"[PHYS] {steps}@{hz}Hz on={on_g} walk={walk} n·up={(nup if nup is not None else float('nan')):0.3f} z={zvel:+0.3f} "
                  f"down={dhit} dist={(ddist if ddist is not None else float('nan')):0.3f} obj={dobj} coyote={coy:0.3f}s snap={snap:0.3f}m")

