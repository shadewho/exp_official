# Exp_Game/Developers/dev_sections/section_xr_modal_general.py
from __future__ import annotations
import time
from ..dev_registry import register_section
from ..dev_draw_prims import draw_text, draw_rect, draw_line_strip
from ..dev_utils import pyget, safe_modal

class GeneralXRModalSection:
    """
    One place for general XR + modal stats:
      • Title + alive/port + Grade — Reason
      • Flow (req/ok/fail Hz & %)
      • Latency (last, EMA) + phase (frames, ms) + gate wait
      • Sync (last_ok age, Δframes, misses/dupes, backlog)
      • 30-step timeline (modal cadence)
      • RTT and Phase (ms) micro-graphs (last 30 steps, with guides)
    Pinned left; fixed height to prevent column shuffling.
    """
    key = "xr.general_modal"
    column = "LEFT"
    order = 9
    prop_toggle = "dev_xr_modal_general"
    sticky_left = True  # never reflows to other columns

    # ---- layout constants (keep stable to stop reflow) ----
    def _total_height(self, lh: int) -> int:
        rows    = 5  # title + (alive/grade) + flow + latency + sync
        bar_h   = int(1.1 * lh)
        gap     = int(0.6 * lh)
        g_h     = int(2.0 * lh)
        return rows*lh + bar_h + gap + g_h + gap + g_h + int(0.4*lh)

    def measure(self, scene, STATE, BUS, scale, lh, width):
        if not (getattr(scene, "dev_hud_show_xr", False) and getattr(scene, "dev_xr_modal_general", False)):
            return 0
        return self._total_height(int(lh))

    # ---- helpers ----
    def _port(self, xr):
        try:
            host = getattr(xr, "host", None) or getattr(xr, "_host", None)
            port = getattr(xr, "port", None) or getattr(xr, "_port", None)
            if host and port:
                return f"{host}:{int(port)}"
            addr = getattr(xr, "addr", None) or getattr(xr, "_addr", None)
            if isinstance(addr, (tuple, list)) and len(addr) == 2:
                return f"{addr[0]}:{int(addr[1])}"
        except Exception:
            pass
        return "—"

    def _last30(self, trace):
        if not isinstance(trace, list) or not trace:
            return []
        return trace[-30:] if len(trace) >= 30 else trace[:]

    def _col(self, grade: str):
        return {
            "HEALTHY":     (0.65, 1.00, 0.65, 1.0),
            "STRESSED":    (1.00, 0.90, 0.60, 1.0),
            "OVERWHELMED": (1.00, 0.55, 0.55, 1.0),
            "WARMUP":      (0.70, 0.85, 1.00, 1.0),
        }.get(grade, (0.9, 0.9, 0.9, 1.0))

    def draw(self, x, y, scene, STATE, BUS, scale, lh, width):
        if not (getattr(scene, "dev_hud_show_xr", False) and getattr(scene, "dev_xr_modal_general", False)):
            return y

        now = time.perf_counter()
        modal = safe_modal()
        xr  = pyget(modal, "_xr", None) if modal else None
        xs  = pyget(modal, "_xr_stats", {}) if modal else {}

        alive = bool(xr)
        port  = self._port(xr) if xr else "—"

        # Flow (1s meters)
        req_hz   = STATE.meter_xr_req.rate(now, 1.0)
        ok_hz    = STATE.meter_xr_ok.rate(now, 1.0)
        fail_hz  = STATE.meter_xr_fail.rate(now, 1.0)
        fail_pct = (fail_hz / max(1e-6, req_hz)) * 100.0 if req_hz > 0 else 0.0

        # Latency / phase (last + EMA)
        rtt_last   = float(xs.get("last_lat_ms", 0.0) or 0.0)
        rtt_ema    = float(xs.get("lat_ema_ms", 0.0)  or 0.0)
        phase_fr   = float(xs.get("phase_frames", 0.0)or 0.0)
        phase_ms   = float(xs.get("phase_ms", 0.0)    or 0.0)
        gate_ms    = float(xs.get("gate_wait_ms_ema", 0.0) or 0.0)
        last_ok    = float(xs.get("last_ok_wall", 0.0) or 0.0)
        last_ok_age_ms = (now - last_ok) * 1000.0 if last_ok > 0.0 else 9999.0

        # Source-of-truth p95s + sync/backlog from frame_end (keeps HUD == console)
        rtt_p95  = float(BUS.scalars.get("XR.core.rtt_p95", 0.0) or 0.0)
        wire_p95 = float(BUS.scalars.get("XR.core.wire_p95", 0.0) or 0.0)
        proc_p95 = float(BUS.scalars.get("XR.core.proc_p95", 0.0) or 0.0)
        df_p95   = float(BUS.scalars.get("XR.core.sync_df_p95", 0.0) or 0.0)
        df_worst = int(BUS.scalars.get("XR.core.sync_df_worst", 0) or 0)
        bnow     = int(BUS.scalars.get("XR.core.backlog_now", 0) or 0)
        bmax     = int(BUS.scalars.get("XR.core.backlog_max", 0) or 0)

        # === SINGLE SOURCE OF TRUTH (from frame_end): grade + reason ===
        grade  = str(BUS.scalars.get("XR.core.grade", "—"))
        reason = str(BUS.scalars.get("XR.core.reason", "—"))

        fs = int(12 * scale)

        # Header
        draw_text(x, y, "General XR and Modal", fs); y -= lh
        draw_text(x, y, f"XR {'ALIVE' if alive else 'Idle'}   port {port}   Grade:", fs)
        draw_text(x + int(210*scale), y, f"{grade} — {reason}", fs, self._col(grade)); y -= lh

        # Flow row
        draw_text(x, y, f"Flow  req/s {req_hz:0.1f}   ok/s {ok_hz:0.1f}   fail/s {fail_hz:0.1f}   fail {fail_pct:0.1f}%", fs); y -= lh

        # Latency/phase row (show proc_p95 for visibility; wire_p95 if you want, comment kept lean)
        draw_text(x, y, f"Latency  rtt {rtt_last:0.2f} ms (ema {rtt_ema:0.2f})   phase {phase_fr:+.1f} fr {phase_ms:+.2f} ms   gate {gate_ms:0.2f} ms   proc {proc_p95:0.2f} ms", fs); y -= lh

        # Sync row (use df_p95/df_worst from frame_end for consistency)
        draw_text(x, y, f"Sync  last_ok {last_ok_age_ms:0.0f} ms   Δframes {df_worst:+d}  p95 {df_p95:0.1f}   backlog {bnow}/{bmax}", fs); y -= lh

        # ---- 30-step timeline (modal cadence) ----
        bar_h = int(1.1 * lh)
        gap   = max(1, int(round(1*scale)))
        n     = 30
        bar_w = max(3, int((width - (n-1)*gap) / max(1, n)))
        total = n * bar_w + (n-1) * gap
        sx    = x + max(0, width - total)
        base_y= y - bar_h

        BG   = (0.10, 0.12, 0.15, 0.85)
        OK   = (0.55, 0.95, 0.65, 1.0)
        MISS = (1.00, 0.45, 0.45, 1.0)
        DUPE = (1.00, 0.75, 0.45, 1.0)
        OTHER= (0.70, 0.75, 0.90, 1.0)

        # Trace for last 30
        trace = pyget(modal, "_frame_trace", []) if modal else []
        last30 = trace[-30:] if isinstance(trace, list) else []

        from ..dev_draw_prims import draw_rect, draw_line_strip
        draw_rect(x, base_y, width, bar_h, BG)

        pad = n - len(last30) if last30 and len(last30) < n else (n if not last30 else 0)
        for i in range(pad):
            bx = sx + i * (bar_w + gap)
            draw_rect(bx, base_y, bar_w, bar_h, (0.55, 0.60, 0.70, 0.25))

        if last30:
            start_idx = pad
            for j, rec in enumerate(last30):
                st = str(rec.get("status", "?")).upper()
                colb = OK if st == "OK" else (MISS if st == "MISS" else (DUPE if st == "DUPE" else OTHER))
                bx = sx + (start_idx + j) * (bar_w + gap)
                draw_rect(bx, base_y, bar_w, bar_h, colb)

        y = base_y - int(0.6 * lh)

        # ---- RTT micro-graph (ms) ----
        gh = int(2.0 * lh); gy = y - gh
        draw_rect(x, gy, width, gh, BG)
        GRID = (0.45, 0.50, 0.58, 0.8)
        LINE = (0.95, 0.80, 0.40, 1.0)

        rtts = [float(rec.get("rtt_ms", 0.0) or 0.0) for rec in (last30 or [])]
        ymax_rtt = max(12.0, (max(rtts) * 1.2) if rtts else 0.0)
        zy = gy + (1.0 - min(1.0, 10.0 / ymax_rtt)) * gh
        draw_line_strip([(x, zy), (x + width, zy)], GRID, width=1.0)

        if rtts:
            denom = max(1, len(rtts) - 1)
            pts = []
            for k, v in enumerate(rtts):
                tx = x + (k / denom) * width
                q  = 0.0 if ymax_rtt <= 1e-6 else max(0.0, min(1.0, v / ymax_rtt))
                ty = gy + (1.0 - q) * gh
                pts.append((tx, ty))
            draw_line_strip(pts, LINE, width=max(1.0, 1.4*scale))

        y = gy - int(0.4 * lh)
        from ..dev_draw_prims import draw_text as _dt
        _dt(x, y, "XR RTT (ms) — last 30 steps", fs); y -= lh

        # ---- Phase micro-graph (ms, centered with 0-line) ----
        gh2 = int(2.0 * lh); gy2 = y - gh2
        draw_rect(x, gy2, width, gh2, BG)

        pvals = [float(rec.get("phase_ms", 0.0) or 0.0) for rec in (last30 or [])]
        max_abs = max((abs(v) for v in pvals), default=0.0)
        cap = 66.7
        ymax_phase = max(10.0, min(max_abs * 1.25, cap))
        mid_y = gy2 + gh2 * 0.5
        draw_line_strip([(x, mid_y), (x + width, mid_y)], GRID, width=1.0)

        if pvals:
            denom = max(1, len(pvals) - 1)
            pts = []
            for k, v in enumerate(pvals):
                tx = x + (k / denom) * width
                q = 0.5 if ymax_phase <= 1e-6 else (0.5 - (v / (2.0 * ymax_phase)))
                q = max(0.0, min(1.0, q))
                ty = gy2 + q * gh2
                pts.append((tx, ty))
            draw_line_strip(pts, (0.65, 0.75, 1.0, 1.0), width=max(1.0, 1.4*scale))

        y = gy2 - int(0.4 * lh)
        _dt(x, y, "XR Phase (ms, 0 = synced) — last 30 steps", fs); y -= lh

        return y

register_section(GeneralXRModalSection())
