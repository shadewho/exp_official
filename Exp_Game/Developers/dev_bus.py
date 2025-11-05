from __future__ import annotations
import time
from collections import deque
from typing import Dict

# NOTE: We import bpy lazily/defensively so these helpers are safe in unit tests.
try:
    import bpy  # type: ignore
except Exception:
    bpy = None  # noqa: N816

from .dev_state import STATE, Series

class _Bus:
    """Global debug bus."""
    def __init__(self):
        self.scalars: Dict[str, object] = {}
        self.temp:    Dict[str, object] = {}
        self.series:  Dict[str, Series] = {}
        self.logs = deque(maxlen=200)

BUS = _Bus()

def _hud_enabled() -> bool:
    """Fast, defensive check: when HUD is OFF, all helpers become NO-OPs."""
    if bpy is None:
        return False
    try:
        scn = bpy.context.scene
        return bool(scn and getattr(scn, "dev_hud_enable", False))
    except Exception:
        return False

def devhud_set(key: str, value, volatile: bool=False):
    if not _hud_enabled():
        return
    if volatile: BUS.temp[key] = value
    else:        BUS.scalars[key] = value

def devhud_post(msg: str):
    if not _hud_enabled():
        return
    ts = time.strftime("%H:%M:%S")
    BUS.logs.append(f"[{ts}] {msg}")

def devhud_series_push(name: str, value: float, maxlen: int = 300):
    if not _hud_enabled():
        return

    # Always keep a Bus series (for custom renderers / external readers)
    s = BUS.series.get(name)
    if s is None:
        BUS.series[name] = Series(name=name, maxlen=maxlen, values=deque(maxlen=maxlen))
        s = BUS.series[name]
    if s.maxlen != maxlen:
        s.set_maxlen(maxlen)
    v = float(value)
    s.push(v)

    # Internal timing + meters for specific view keys
    t = time.perf_counter()
    if name == "view_candidate":
        STATE.meter_view_queue.record(t)
        STATE.view_last_candidate_wall = t
        # mirror into core STATE series if present
        if "view_candidate" in STATE.series:
            STATE.series["view_candidate"].push(v)

    elif name == "view_allowed":
        STATE.meter_view_apply.record(t)
        if STATE.view_last_candidate_wall > 0.0:
            lag_ms = (t - STATE.view_last_candidate_wall) * 1000.0
            STATE.series["view_lag_ms"].push(lag_ms)
            STATE.view_lag_ema_ms = lag_ms if STATE.view_lag_ema_ms == 0.0 else (STATE.view_lag_ema_ms*0.8 + lag_ms*0.2)
        if STATE.view_last_allowed is not None and STATE.view_last_allowed_wall > 0.0:
            dt = max(1e-6, t - STATE.view_last_allowed_wall)
            d  = abs(v - float(STATE.view_last_allowed)) / dt
            STATE.view_jitter_ema = d if STATE.view_jitter_ema == 0.0 else (STATE.view_jitter_ema*0.85 + d*0.15)
        STATE.view_last_allowed = v
        STATE.view_last_allowed_wall = t
        if "view_allowed" in STATE.series:
            STATE.series["view_allowed"].push(v)
