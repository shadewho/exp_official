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


def _compute_xr_health(*, alive: bool, last_ok_age_ms: float,
                       req_hz: float, ok_hz: float, fail_hz: float,
                       rtt_ema: float, rtt_p95: float, phase_fr: float,
                       df_p95: float, backlog_now: int, backlog_max: int) -> tuple[str, str]:
    """
    Pipeline-tolerant XR health classifier.
    Returns (grade, reason).
    """
    # Tunables (match your existing constants)
    TRACK_MIN   = 0.85   # ok/s must track req/s
    PIPE_TOL    = 6      # tolerate ~6 in-flight frames when tracking OK
    FAIL_STRESS = 10.0   # >=10% fail   -> STRESSED
    FAIL_OVW    = 25.0   # >=25% fail   -> OVERWHELMED
    FAIL_MIN_RQ = 5.0    # only care about fail% when there is real traffic

    fail_pct    = (fail_hz / max(1e-6, req_hz)) * 100.0 if req_hz > 0 else 0.0
    tracking_ok = (req_hz < 0.1) or (ok_hz >= TRACK_MIN * req_hz)
    pipe_ok     = (backlog_now <= PIPE_TOL) or (df_p95 <= 1.0) or tracking_ok

    # 0) Warmup / idle
    if not alive:
        return ("IDLE", "XR not running")
    # If XR is alive but we haven't seen any reply yet and traffic is low -> WARMUP
    if (last_ok_age_ms > 1500.0) and (req_hz < 1.0) and (ok_hz < 1.0):
        return ("WARMUP", "waiting first reply")

    # 1) Hard overwhelmed
    if last_ok_age_ms > 1500.0:
        return ("OVERWHELMED", "no replies >1.5s")
    if (fail_pct >= FAIL_OVW) and (req_hz >= FAIL_MIN_RQ):
        return ("OVERWHELMED", f"fail {fail_pct:0.1f}%")
    # falling behind + deep pipe or real frame lag
    if (not tracking_ok) and ((backlog_now > PIPE_TOL) or (df_p95 >= 2.0)):
        return ("OVERWHELMED", ("ok/s < req/s" if ok_hz < TRACK_MIN * req_hz else f"Δframes p95 {df_p95:0.1f}"))

    # 2) Stressed (but not overwhelmed)
    if (fail_pct >= FAIL_STRESS and req_hz >= FAIL_MIN_RQ):
        return ("STRESSED", f"fail {fail_pct:0.1f}%")
    if rtt_ema > 50.0:
        return ("STRESSED", f"rtt {rtt_ema:0.2f} ms")
    if rtt_p95 > 65.0:
        return ("STRESSED", f"rtt_p95 {rtt_p95:0.1f} ms")
    if abs(phase_fr) >= 4:
        return ("STRESSED", f"phase {phase_fr:+.1f} fr")
    # Only treat Δframes as stress when the pipe is NOT OK
    if (df_p95 >= 2.0) and (not pipe_ok):
        return ("STRESSED", f"Δframes p95 {df_p95:0.1f}")

    # 3) Healthy; annotate common pipeline state
    if tracking_ok and pipe_ok and 0.8 <= df_p95 <= 1.2:
        return ("HEALTHY", "pipelined (≈1 frame)")
    return ("HEALTHY", "OK")


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
            "xr_proc_ms": Series("xr_proc_ms", 300, deque(maxlen=300)),
            "xr_pend_now": Series("xr_pend_now", 300, deque(maxlen=300)),

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
# -------- XR raw-dump catalog (stable, no flicker) --------

def xr_catalog_update_from_bus(BUS):
    """
    Build/maintain a stable list of XR.* keys so HUD rows never flicker.
    We never remove keys; they persist for the session. Last values/seen
    are stored and reused when a key doesn't arrive this frame.
    """
    # Lazily create fields on STATE (no __init__ changes needed)
    if not hasattr(STATE, "xr_keys_order"):
        STATE.xr_keys_order = []          # insertion order; never shrinks
        STATE.xr_last_seen  = {}          # key -> frame_idx last seen
        STATE.xr_last_val   = {}          # key -> last value (any)

    merged = {}
    try:
        merged.update(BUS.scalars or {})
    except Exception:
        pass
    try:
        merged.update(BUS.temp or {})
    except Exception:
        pass

    for k, v in merged.items():
        if not isinstance(k, str):
            continue
        if not (k.startswith("XR") or k.startswith("XR.")):
            continue
        if k not in STATE.xr_last_seen:
            STATE.xr_keys_order.append(k)
        STATE.xr_last_seen[k] = int(STATE.frame_idx)
        STATE.xr_last_val[k]  = v

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
    
def _p95_list(vals) -> float:
    try:
        arr = list(float(v) for v in vals if v is not None)
        if not arr: return 0.0
        arr.sort()
        k = int(0.95 * (len(arr)-1))
        return float(arr[k])
    except Exception:
        return 0.0

def frame_end(modal, context, BUS):
    scene = context.scene if context else None
    if scene is None:
        return
    now = time.perf_counter()

    # -------- Frame timing (EMAs + series) --------
    if STATE.frame_start > 0.0:
        ms = (now - STATE.frame_start) * 1000.0
        STATE.series["frame_ms"].push(ms)
        STATE.ms_ema = ms if STATE.ms_ema == 0.0 else (STATE.ms_ema * 0.90 + ms * 0.10)
    if STATE.last_end_t > 0.0:
        dt = max(1e-6, now - STATE.last_end_t)
        fps_inst = 1.0 / dt
        STATE.fps_ema = fps_inst if STATE.fps_ema == 0.0 else (STATE.fps_ema * 0.85 + fps_inst * 0.15)
        STATE.series["fps"].push(STATE.fps_ema)
    STATE.last_end_t = now

    # -------- XR stats snapshot (once) --------
    xs = pyget(modal, "_xr_stats", None) if modal else None

    # Push RTT/Phase series + meters from deltas
    if isinstance(xs, dict):
        rtt_last = float(xs.get("last_lat_ms", 0.0) or 0.0)
        phase_ms = float(xs.get("phase_ms", 0.0) or 0.0)
        STATE.series["rtt_ms"].push(rtt_last)
        STATE.series["phase_ms"].push(phase_ms)

        req = int(xs.get("req", 0) or 0)
        ok  = int(xs.get("ok", 0) or 0)
        fl  = int(xs.get("fail", 0) or 0)
        if req > STATE.xr_req_last:
            for _ in range(req - STATE.xr_req_last): STATE.meter_xr_req.record(now)
        if ok > STATE.xr_ok_last:
            for _ in range(ok - STATE.xr_ok_last):  STATE.meter_xr_ok.record(now)
        if fl > STATE.xr_fail_last:
            for _ in range(fl - STATE.xr_fail_last): STATE.meter_xr_fail.record(now)
        STATE.xr_req_last, STATE.xr_ok_last, STATE.xr_fail_last = req, ok, fl

        # Optional runtime diagnostics → BUS + series
        try:
            BUS.scalars["XR.core.proc_ms"]       = round(float(xs.get("proc_ms", 0.0) or 0.0), 2)
            BUS.scalars["XR.core.jobs_n"]        = int(xs.get("jobs_n", 0) or 0)
            BUS.scalars["XR.core.jobs_ms"]       = round(float(xs.get("jobs_ms", 0.0) or 0.0), 2)
            BUS.scalars["XR.core.job_max_ms"]    = round(float(xs.get("job_max_ms", 0.0) or 0.0), 2)
            if "proc_ms" in xs:
                STATE.series["xr_proc_ms"].push(float(xs.get("proc_ms", 0.0) or 0.0))
        except Exception:
            pass

        # NEW: wire/queue component (rtt – proc)
        try:
            from collections import deque as _dq
            if "xr_wire_ms" not in STATE.series:
                STATE.series["xr_wire_ms"] = Series("xr_wire_ms", 300, _dq(maxlen=300))
            wire_last = max(0.0, rtt_last - float(xs.get("proc_ms", 0.0) or 0.0))
            STATE.series["xr_wire_ms"].push(wire_last)
        except Exception:
            pass


    # -------- Resize buffers to user setting (before p95) --------
    maxlen = int(getattr(scene, "dev_hud_max_samples", 300))
    for s in STATE.series.values(): s.set_maxlen(maxlen)
    for s in BUS.series.values():   s.set_maxlen(maxlen)

    # -------- Derive p95 + raw sync metrics (no tunables) --------
    try:
        rtt_p95   = _p95_list(STATE.series["rtt_ms"].values)
        proc_p95  = _p95_list(STATE.series["xr_proc_ms"].values)
        wire_p95  = _p95_list(STATE.series["xr_wire_ms"].values) if "xr_wire_ms" in STATE.series else 0.0
        pend_now  = int(STATE.series["xr_pend_now"].last or 0)

        BUS.scalars["XR.core.rtt_p95"]   = round(rtt_p95, 2)
        BUS.scalars["XR.core.proc_p95"]  = round(proc_p95, 2)
        BUS.scalars["XR.core.wire_p95"]  = round(wire_p95, 2)
        BUS.scalars["XR.core.backlog_now"] = int(pend_now)

        # Sync over recent trace (raw Δframes stats)
        trace  = pyget(modal, "_frame_trace", []) if modal else []
        last30 = trace[-30:] if isinstance(trace, list) else []
        dframes = []
        worst = 0
        if last30:
            for rec in last30:
                try:
                    s = int(rec.get("step", 0)); t = int(rec.get("tick", 0))
                    df = abs(s - t); worst = max(worst, df); dframes.append(float(df))
                except Exception:
                    pass
        df_p95 = _p95_list(dframes) if dframes else 0.0
        BUS.scalars["XR.core.sync_df_p95"]   = round(df_p95, 1)
        BUS.scalars["XR.core.sync_df_worst"] = int(worst)
    except Exception:
        pass


    # -------- XR.core scalars + pipeline-tolerant grade --------
    try:
        xr = pyget(modal, "_xr", None) if modal else None
        alive = bool(xr)
        host = getattr(xr, "host", None) or getattr(xr, "_host", None)
        port = getattr(xr, "port", None) or getattr(xr, "_port", None)
        if not host:
            addr = getattr(xr, "addr", None) or getattr(xr, "_addr", None)
            if isinstance(addr, (tuple, list)) and len(addr) == 2:
                host, port = addr[0], addr[1]

        req_hz = STATE.meter_xr_req.rate(now, 1.0)
        ok_hz  = STATE.meter_xr_ok.rate(now, 1.0)
        fl_hz  = STATE.meter_xr_fail.rate(now, 1.0)
        fail_pct = (fl_hz / max(1e-6, req_hz)) * 100.0 if req_hz > 0 else 0.0

        xs = pyget(modal, "_xr_stats", None) if modal else None
        rtt_last = float(xs.get("last_lat_ms", 0.0)) if isinstance(xs, dict) else 0.0
        rtt_ema  = float(xs.get("lat_ema_ms", 0.0))  if isinstance(xs, dict) else 0.0
        phase_ms = float(xs.get("phase_ms", 0.0))    if isinstance(xs, dict) else 0.0
        phase_fr = float(xs.get("phase_frames", 0.0))if isinstance(xs, dict) else 0.0
        last_ok  = float(xs.get("last_ok_wall", 0.0))if isinstance(xs, dict) else 0.0
        last_ok_age_ms = (now - last_ok) * 1000.0 if last_ok > 0.0 else 9999.0

        bnow    = int(BUS.scalars.get("XR.core.backlog_now", 0) or 0)
        bmax    = int(BUS.scalars.get("XR.core.backlog_max", 0) or 0)
        rtt_p95 = float(BUS.scalars.get("XR.core.rtt_p95", 0.0) or 0.0)
        df_p95  = float(BUS.scalars.get("XR.core.sync_df_p95", 0.0) or 0.0)

        # Identity + scalars
        BUS.scalars["XR.core.alive"]          = ("Y" if alive else "N")
        BUS.scalars["XR.core.host"]           = (str(host) if host else "—")
        BUS.scalars["XR.core.port"]           = (int(port) if port not in (None, "",) else "—")
        BUS.scalars["XR.core.req_s"]          = round(req_hz, 2)
        BUS.scalars["XR.core.ok_s"]           = round(ok_hz, 2)
        BUS.scalars["XR.core.fail_s"]         = round(fl_hz, 2)
        BUS.scalars["XR.core.fail_pct"]       = round(fail_pct, 2)
        BUS.scalars["XR.core.rtt_ms"]         = round(rtt_last, 3)
        BUS.scalars["XR.core.lat_ema_ms"]     = round(rtt_ema, 3)
        BUS.scalars["XR.core.phase_ms"]       = round(phase_ms, 3)
        BUS.scalars["XR.core.phase_frames"]   = round(phase_fr, 3)
        BUS.scalars["XR.core.last_ok_age_ms"] = round(last_ok_age_ms, 1)

        # Grade (shared policy)
        grade, reason = _compute_xr_health(
            alive=alive, last_ok_age_ms=last_ok_age_ms,
            req_hz=req_hz, ok_hz=ok_hz, fail_hz=fl_hz,
            rtt_ema=rtt_ema, rtt_p95=rtt_p95, phase_fr=phase_fr,
            df_p95=df_p95, backlog_now=bnow, backlog_max=bmax,
        )

        BUS.scalars["XR.core.grade"]  = grade
        BUS.scalars["XR.core.reason"] = reason

    except Exception:
        pass


    # -------- Console logging (master-gated) --------
    master = bool(getattr(scene, "dev_hud_log_console", True))

    if master and (now - STATE.last_console_emit) >= 1.0:
        STATE.last_console_emit = now
        fps_txt = f"{STATE.fps_ema:.1f}" if STATE.fps_ema > 0 else "—"
        ms_txt  = f"{STATE.ms_ema:.2f}"  if STATE.ms_ema > 0 else "—"
        rtt_txt = f"{(STATE.series['rtt_ms'].last or 0):.2f}"
        ph_txt  = f"{(STATE.series['phase_ms'].last or 0):+.2f}"
        print(f"[DEVHUD] frame {ms_txt} ms  ~{fps_txt} FPS  XR rtt {rtt_txt} ms  phase {ph_txt} ms")

    if master and getattr(scene, "dev_log_xr_console", False):
        period = 1.0 / max(0.1, float(getattr(scene, "dev_log_xr_hz", 2.0)))
        if (now - STATE.last_xr_emit) >= period:
            STATE.last_xr_emit = now
            xr_rq = STATE.meter_xr_req.rate(now, 1.0)
            xr_ok = STATE.meter_xr_ok.rate(now, 1.0)
            xr_fl = STATE.meter_xr_fail.rate(now, 1.0)
            fail_pct = (xr_fl / max(1e-6, xr_rq)) * 100.0 if xr_rq > 0 else 0.0
            rtt = STATE.series["rtt_ms"].last or 0.0
            ph  = STATE.series["phase_ms"].last or 0.0
            print(f"[XR] req/s {xr_rq:.1f}  ok/s {xr_ok:.1f}  fail/s {xr_fl:.1f}  fail {fail_pct:.1f}%  rtt {rtt:.2f} ms  phase {ph:+.2f} ms")

    if master and getattr(scene, "dev_log_view_console", False):
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

    if master and getattr(scene, "dev_log_physics_console", False):
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

    # --- XR Health (single, raw, no badges) ---
    if master and getattr(scene, "dev_log_xr_health_console", False):
        if not hasattr(STATE, "last_xr_health_emit"):
            STATE.last_xr_health_emit = 0.0
        period_h = 1.0 / max(0.1, float(getattr(scene, "dev_log_xr_health_hz", 2.0)))
        if (now - float(getattr(STATE, "last_xr_health_emit", 0.0) or 0.0)) >= period_h:
            STATE.last_xr_health_emit = now

            xs      = pyget(modal, "_xr_stats", {}) if modal else {}
            alive   = bool(pyget(modal, "_xr", None)) if modal else False
            rtt_ema = float(xs.get("lat_ema_ms", 0.0))   if isinstance(xs, dict) else 0.0
            phase_fr= float(xs.get("phase_frames", 0.0)) if isinstance(xs, dict) else 0.0
            last_ok = float(xs.get("last_ok_wall", 0.0)) if isinstance(xs, dict) else 0.0
            last_age= (now - last_ok) * 1000.0 if last_ok > 0.0 else 9999.0

            req_hz = STATE.meter_xr_req.rate(now, 1.0)
            ok_hz  = STATE.meter_xr_ok.rate(now, 1.0)
            fl_hz  = STATE.meter_xr_fail.rate(now, 1.0)
            fail_pct = (fl_hz / max(1e-6, req_hz)) * 100.0 if req_hz > 0 else 0.0

            rtt_p95 = float(BUS.scalars.get("XR.core.rtt_p95", 0.0) or 0.0)
            wire_p95= float(BUS.scalars.get("XR.core.wire_p95", 0.0) or 0.0)
            df_p95  = float(BUS.scalars.get("XR.core.sync_df_p95", 0.0) or 0.0)
            bnow    = int(BUS.scalars.get("XR.core.backlog_now", 0) or 0)
            bmax    = int(BUS.scalars.get("XR.core.backlog_max", 0) or 0)
            pp95    = float(BUS.scalars.get("XR.core.proc_p95", 0.0) or 0.0)

            grade, reason = _compute_xr_health(
                alive=alive, last_ok_age_ms=last_age,
                req_hz=req_hz, ok_hz=ok_hz, fail_hz=fl_hz,
                rtt_ema=rtt_ema, rtt_p95=rtt_p95, phase_fr=phase_fr,
                df_p95=df_p95, backlog_now=bnow, backlog_max=bmax,
            )

            # Truth flags (evidence inline)
            TRACK_MIN = 0.85
            tracking_ok = (req_hz < 0.1) or (ok_hz >= TRACK_MIN * req_hz)
            pipe_ok = (bnow <= 6) or (df_p95 <= 1.0) or tracking_ok

            print(
                f"[XR.HEALTH] {grade} — {reason}  | "
                f"rtt_ema {rtt_ema:0.2f} ms  rtt_p95 {rtt_p95:0.2f} ms  wire_p95 {wire_p95:0.2f} ms  "
                f"phase {phase_fr:+.1f} fr  Δframes p95 {df_p95} (worst {BUS.scalars.get('XR.core.sync_df_worst','—')})  "
                f"backlog {bnow}/{bmax}  track {'OK' if tracking_ok else 'MISS'}  pipe {'OK' if pipe_ok else 'DEEP'}  "
                f"fail {fail_pct:0.1f}%  ok/s {ok_hz:0.1f} req/s {req_hz:0.1f}  proc_p95 {pp95:0.2f} ms"
            )



