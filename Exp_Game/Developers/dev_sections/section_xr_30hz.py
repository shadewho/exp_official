# Exp_Game/Developers/dev_sections/section_xr_30hz.py
from __future__ import annotations
import time
from ..dev_registry import register_section
from ..dev_draw_prims import draw_text, draw_rect, draw_line_strip
from ..dev_utils import pyget, safe_modal
from ..dev_state import STATE

class XRThirtyHzSection:
    """
    XR @30Hz — Modal Sync
    Left-pinned section that shows:
      • 30-step timeline (exact modal cadence via _frame_trace)
      • XR↔Modal health stats (req/ok/fail Hz, fail%, RTT last/EMA, phase frames/ms)
      • Micro-graph of RTT over the last 30 steps
    """
    key = "xr.30hz"
    column = "LEFT"
    order = 11
    prop_toggle = "dev_xr_core_30hz"
    sticky_left = True  # always left column, no reflow

    def measure(self, scene, STATE, BUS, scale, lh, width):
        if not getattr(scene, "dev_xr_core_30hz", False) or not getattr(scene, "dev_hud_show_xr", False):
            return 0
        # Fixed, compact height (stable layout; avoids column shuffle on toggle)
        # 4 rows text + timeline bar + spacing + micro-graph + spacing
        bar_h = int(1.1 * lh)
        gh    = int(2.0 * lh)
        spacing = int(0.6 * lh)
        return 4 * lh + bar_h + spacing + gh + spacing

    def _colors(self):
        OK   = (0.55, 0.95, 0.65, 1.0)
        MISS = (1.00, 0.45, 0.45, 1.0)
        DUPE = (1.00, 0.75, 0.45, 1.0)
        OTHER= (0.70, 0.75, 0.90, 1.0)
        BG   = (0.10, 0.12, 0.15, 0.85)
        GRID = (0.45, 0.50, 0.58, 0.8)
        LINE = (0.95, 0.80, 0.40, 1.0)  # RTT sparkline
        return OK, MISS, DUPE, OTHER, BG, GRID, LINE

    def _last_30(self, trace):
        if not isinstance(trace, list) or not trace:
            return []
        return trace[-30:] if len(trace) >= 30 else trace[:]

    def draw(self, x, y, scene, STATE, BUS, scale, lh, width):
        if not (getattr(scene, "dev_xr_core_30hz", False) and getattr(scene, "dev_hud_show_xr", False)):
            return y

        now = time.perf_counter()
        modal = safe_modal()
        xs = pyget(modal, "_xr_stats", {}) if modal else {}
        xr  = pyget(modal, "_xr", None) if modal else None
        alive = bool(xr)

        # Flow rates (1s meters)
        req_hz = STATE.meter_xr_req.rate(now, 1.0)
        ok_hz  = STATE.meter_xr_ok.rate(now, 1.0)
        fl_hz  = STATE.meter_xr_fail.rate(now, 1.0)
        fail_pct = (fl_hz / max(1e-6, req_hz)) * 100.0 if req_hz > 0 else 0.0

        # Latency / phase
        rtt_last = float(xs.get("last_lat_ms", 0.0)) if xs else 0.0
        rtt_ema  = float(xs.get("lat_ema_ms", 0.0))  if xs else 0.0
        phase_fr = float(xs.get("phase_frames", 0.0))if xs else 0.0
        phase_ms = float(xs.get("phase_ms", 0.0))    if xs else 0.0
        last_ok  = float(xs.get("last_ok_wall", 0.0)) if xs else 0.0
        last_ok_age_ms = (now - last_ok) * 1000.0 if last_ok > 0.0 else 9999.0
        dupes  = int(xs.get("dupes", 0) or 0) if xs else 0
        misses = int(xs.get("misses", 0) or 0) if xs else 0
        gate_ema = float(xs.get("gate_wait_ms_ema", 0.0)) if xs else 0.0

        # Modal step timeline — exactly the 30 most recent XR frame replies,
        # pushed by GameLoop._xr_trace_push per physics step
        trace = pyget(modal, "_frame_trace", []) if modal else []
        last30 = self._last_30(trace)

        # Modal 30Hz check: recent step rate over 1s using wall times from the trace
        steps_1s = 0
        if last30:
            cutoff = now - 1.0
            for rec in reversed(last30):
                t = float(rec.get("wall", 0.0) or 0.0)
                if t >= cutoff:
                    steps_1s += 1
                else:
                    break

        # Δframes (step - tick) for last record
        delta_frames_txt = "—"
        if last30:
            rec = last30[-1]
            step = rec.get("step", None)
            tick = rec.get("tick", None)
            if isinstance(step, int) and isinstance(tick, int):
                delta_frames_txt = f"{int(step) - int(tick):+d}"

        # Title + summary rows
        draw_text(x, y, "XR @30Hz — Modal Sync", 12 * scale); y -= lh
        draw_text(x, y, f"Modal step Hz {steps_1s:0.1f}   Alive {'Y' if alive else 'N'}", 12 * scale); y -= lh
        draw_text(x, y, f"Flow  req/s {req_hz:0.1f}   ok/s {ok_hz:0.1f}   fail/s {fl_hz:0.1f}   fail {fail_pct:0.1f}%", 12 * scale); y -= lh
        draw_text(x, y, f"Latency  rtt {rtt_last:0.2f} ms (ema {rtt_ema:0.2f})   phase {phase_fr:+.1f} fr {phase_ms:+.2f} ms   last_ok {last_ok_age_ms:0.0f} ms   Δframes {delta_frames_txt}", 12 * scale); y -= lh

        # Timeline (30 bars)
        OK, MISS, DUPE, OTHER, BG, GRID, LINE = self._colors()
        bar_h = int(1.1 * lh)
        gap   = max(1, int(round(1 * scale)))
        n     = 30
        bar_w = max(3, int((width - (n - 1) * gap) / max(1, n)))
        total = n * bar_w + (n - 1) * gap
        sx    = x + max(0, (width - total))
        base_y= y - bar_h

        # Background behind bars
        draw_rect(x, base_y, width, bar_h, BG)

        # Fill missing left slots (if <30 recs) as OTHER (dim)
        pad = n - len(last30) if last30 and len(last30) < n else (n if not last30 else 0)
        i = 0
        while i < pad:
            bx = sx + i * (bar_w + gap)
            draw_rect(bx, base_y, bar_w, bar_h, (0.55, 0.60, 0.70, 0.25))
            i += 1

        if last30:
            start_idx = pad
            for j, rec in enumerate(last30):
                st = str(rec.get("status", "?")).upper()
                col = OK if st == "OK" else (MISS if st == "MISS" else (DUPE if st == "DUPE" else OTHER))
                bx = sx + (start_idx + j) * (bar_w + gap)
                draw_rect(bx, base_y, bar_w, bar_h, col)

        y = base_y - int(0.6 * lh)

        # Micro-graph (RTT over last 30 steps)
        gh = int(2.0 * lh)
        gy = y - gh
        draw_rect(x, gy, width, gh, BG)
        # y grid @ 10ms
        y_max = 0.0
        rtts = []
        if last30:
            rtts = [float(rec.get("rtt_ms", 0.0) or 0.0) for rec in last30]
            y_max = max(12.0, max(rtts) * 1.2)
        else:
            y_max = 12.0

        # optional 10ms line
        if y_max > 0.0:
            q = min(1.0, 10.0 / y_max)
            zy = gy + (1.0 - q) * gh
            draw_line_strip([(x, zy), (x + width, zy)], GRID, width=1.0)

        if rtts:
            pts = []
            denom = max(1, len(rtts) - 1)
            for k, v in enumerate(rtts):
                tx = x + (k / denom) * width
                q  = 0.0 if y_max <= 1e-6 else max(0.0, min(1.0, v / y_max))
                ty = gy + (1.0 - q) * gh
                pts.append((tx, ty))
            draw_line_strip(pts, LINE, width=max(1.0, 1.4 * scale))

        # Caption
        y = gy - int(0.6 * lh)
        draw_text(x, y, "RTT (ms) — last 30 steps", 12 * scale); y -= lh
        return y

register_section(XRThirtyHzSection())
