from __future__ import annotations
import bpy
import gpu
import blf
from gpu_extras.batch import batch_for_shader
import time
from .dev_utils import pyget, safe_modal
from .dev_state import STATE, world_counts, catalog_update_from_bus

# --- low-level primitives ---

def _corner_xy(W, H, w, h, pos: str, pad: int):
    return {'TR': (W - w - pad, pad), 'TL': (pad, pad),
            'BR': (W - w - pad, H - h - pad), 'BL': (pad, H - h - pad)}.get(pos, (W - w - pad, pad))

def _draw_rect(x, y, w, h, col):
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    verts = [(x,y), (x+w,y), (x+w,y+h), (x,y+h)]
    batch = batch_for_shader(shader, 'TRI_FAN', {"pos": verts})
    gpu.state.blend_set('ALPHA')
    shader.bind(); shader.uniform_float("color", col); batch.draw(shader)

def _draw_line_strip(points, col, width=1.0):
    if len(points) < 2: return
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    batch  = batch_for_shader(shader, 'LINE_STRIP', {"pos": points})
    gpu.state.blend_set('ALPHA')
    try: gpu.state.line_width_set(max(1.0, float(width)))
    except Exception: pass
    shader.bind(); shader.uniform_float("color", col); batch.draw(shader)
    try: gpu.state.line_width_set(1.0)
    except Exception: pass

def _draw_text(x, y, text, size_px=12, color=(1,1,1,1)):
    blf.position(0, x, y, 0); blf.size(0, size_px); blf.color(0, *color); blf.draw(0, text)

def _samples_to_poly(series, x, y, w, h, ymin, ymax):
    vals = list(series.values); n = len(vals)
    if n == 0: return []
    rng = max(1e-6, (ymax - ymin)); pts = []
    for i, v in enumerate(vals):
        t = (i / (n - 1)) if n > 1 else 1.0
        vx = x + t * w
        q  = max(0.0, min(1.0, (v - ymin) / rng))
        vy = y + (1.0 - q) * h
        pts.append((vx, vy))
    return pts

# --- helpers for summary grading ---

def _p01_low_fps():
    vals = list(STATE.series["frame_ms"].values)
    if len(vals) < 60: return 0.0
    vals_sorted = sorted(vals)
    worst_idx = max(0, int(len(vals_sorted) * 0.99) - 1)
    p01_ms = vals_sorted[worst_idx]
    return (1000.0 / p01_ms) if p01_ms > 0.0 else 0.0

def _grade(ema_ms: float, ema_fps: float):
    budget_ms = 33.3333
    ratio = (ema_ms / budget_ms) if budget_ms > 0 else 0.0
    if   ratio <= 0.60: lab, col = ("EXCELLENT", (0.65, 1.00, 0.65, 1.0))
    elif ratio <= 0.85: lab, col = ("GREAT",     (0.75, 1.00, 0.75, 1.0))
    elif ratio <= 1.00: lab, col = ("GOOD",      (1.00, 1.00, 0.70, 1.0))
    elif ratio <= 1.50: lab, col = ("OK",        (1.00, 0.90, 0.60, 1.0))
    elif ratio <= 2.00: lab, col = ("POOR",      (1.00, 0.70, 0.60, 1.0))
    else:               lab, col = ("CRITICAL",  (1.00, 0.55, 0.55, 1.0))
    low_fps = _p01_low_fps()
    if ema_fps > 0.0 and low_fps > 0.0 and low_fps < 0.7 * ema_fps:
        ladder = ["EXCELLENT","GREAT","GOOD","OK","POOR","CRITICAL"]
        try:
            i = ladder.index(lab)
            lab = ladder[min(i+1, len(ladder)-1)]
            col = (min(1.0, col[0] + 0.05), col[1] * 0.98, col[2] * 0.98, 1.0)
        except ValueError:
            pass
    return lab, col, budget_ms, low_fps

# --- main draw ---

def draw_2d(BUS):
    scene = bpy.context.scene
    if not (scene and getattr(scene, "dev_hud_enable", False)):
        return

    # Resolve region
    try:
        area = next((a for a in bpy.context.screen.areas if a.type == 'VIEW_3D'), None)
        if not area: return
        region = next((r for r in area.regions if r.type == 'WINDOW'), None)
        if not region: return
    except Exception:
        return

    modal = safe_modal()

    # scale/layout
    pos   = scene.dev_hud_position
    scale = max(1, int(scene.dev_hud_scale))
    lh    = 16 * scale
    pad   = 12 * scale
    gap   = 22 * scale
    col_w_left  = 460 * scale
    col_w_right = 460 * scale
    box_w = col_w_left + gap + col_w_right

    # keep custom catalog stable
    catalog_update_from_bus(BUS)

    # measure
    def _measure_left_height():
        h = int(1.3*lh) + lh  # title + summary
        if scene.dev_hud_show_xr:       h += 2*lh
        if scene.dev_hud_show_world:    h += 1*lh
        if scene.dev_hud_show_physics:  h += 1*lh
        if scene.dev_hud_show_camera:   h += 1*lh
        if scene.dev_hud_show_view:     h += 3*lh
        if scene.dev_hud_graphs:
            per = int(0.6*lh) + int(2.0*lh) + int(0.5*lh)
            h += 3 * per
        return h

    def _measure_right_height():
        if not scene.dev_hud_show_custom: return 0
        h = 0
        for grp in STATE.custom_groups_order:
            keys = STATE.custom_keys_by_group.get(grp) or []
            if not keys: continue
            h += lh + lh*len(keys)
        return h

    left_h  = _measure_left_height()
    right_h = _measure_right_height()
    box_h   = max(left_h, right_h) + 2*pad

    x0, y0 = _corner_xy(region.width, region.height, box_w, box_h, pos, pad)

    # background frame
    _draw_rect(x0-4, y0-4, box_w+8, box_h+8, (0.07, 0.08, 0.10, 0.92))
    _draw_rect(x0-4, y0-4, box_w+8, 1, (0.22,0.26,0.32,1.0))
    _draw_rect(x0-4, y0-4+box_h+8-1, box_w+8, 1, (0.22,0.26,0.32,1.0))

    xL = x0 + 8*scale
    xR = x0 + col_w_left + gap + 8*scale
    y_top = y0 + box_h - lh

    # left column
    yL = y_top + int(0.3*lh)
    _draw_text(xL, yL, "DEVELOPER HUD", 14*scale); yL -= lh

    ema_ms  = float(STATE.ms_ema or 0.0)
    ema_fps = float(STATE.fps_ema or 0.0)
    lab, col, budget, low_fps = _grade(ema_ms, ema_fps)
    _draw_text(xL, yL, f"Frame {ema_ms:5.2f} ms   ~{ema_fps:0.1f} FPS  (1% {low_fps:0.0f})   Budget {budget:0.1f} ms   Rating:", 12*scale)
    _draw_text(xL + int(360*scale), yL, lab, 12*scale, col); yL -= lh

    if scene.dev_hud_show_xr:
        xs = pyget(modal, "_xr_stats", {}) if modal else {}
        xr_ok = bool(pyget(modal, "_xr", None)) if modal else False
        rtt = float(xs.get("last_lat_ms", 0.0)) if xs else 0.0
        ph  = float(xs.get("phase_ms", 0.0))    if xs else 0.0
        rqhz = STATE.meter_xr_req.rate(time.perf_counter(), 1.0)
        okhz = STATE.meter_xr_ok.rate(time.perf_counter(), 1.0)
        flhz = STATE.meter_xr_fail.rate(time.perf_counter(), 1.0)
        status = "✓ XR ALIVE" if xr_ok else "… XR Idle"
        _draw_text(xL, yL, f"{status}   rtt {rtt:.2f} ms   phase {ph:+.2f} ms", 12*scale); yL -= lh
        _draw_text(xL, yL, f"XR req/s {rqhz:0.1f}  ok/s {okhz:0.1f}  fail/s {flhz:0.1f}", 12*scale); yL -= lh

    if scene.dev_hud_show_world:
        dyn_active, dyn_bvhs, dyn_total, stat_total = world_counts(scene, modal)
        _draw_text(xL, yL, f"World  Dyn Active {dyn_active}/{dyn_total}   Dyn BVHs {dyn_bvhs}   Static {stat_total}", 12*scale); yL -= lh

    if scene.dev_hud_show_physics:
        steps = int(pyget(modal, "_perf_last_physics_steps", 0) or 0) if modal else 0
        hz = int(pyget(modal, "physics_hz", 30) or 30)
        timer_hz = STATE.meter_timer.rate(time.perf_counter(), 1.0)
        _draw_text(xL, yL, f"Physics  {steps} step{'s' if steps!=1 else ''} @ {hz} Hz   TIMER ~{timer_hz:0.1f} Hz", 12*scale); yL -= lh

    if scene.dev_hud_show_camera:
        mode = getattr(scene, "view_mode", "THIRD")
        dist = float(pyget(modal, "_cam_allowed_last", 0.0) or 0.0) if modal else 0.0
        _draw_text(xL, yL, f"Camera  mode {mode}   boom {dist:0.2f} m", 12*scale); yL -= lh

    if scene.dev_hud_show_view:
        nowt = time.perf_counter()
        qhz = STATE.meter_view_queue.rate(nowt, 1.0)
        ahz = STATE.meter_view_apply.rate(nowt, 1.0)
        va  = STATE.series["view_allowed"].last
        vc  = STATE.series["view_candidate"].last
        dd  = (vc - va) if (vc is not None and va is not None) else 0.0
        lag = STATE.view_lag_ema_ms
        jit = STATE.view_jitter_ema
        _draw_text(xL, yL, f"View Hz   queue {qhz:0.1f}   apply {ahz:0.1f}", 12*scale); yL -= lh
        _draw_text(xL, yL, f"View dist  allowed {(va or 0):0.3f} m   cand {(vc or 0):0.3f} m   Δ {dd:+0.3f} m", 12*scale); yL -= lh
        hit = BUS.temp.get("VIEW.hit", BUS.scalars.get("VIEW.hit", "—"))
        src = BUS.temp.get("VIEW.src", BUS.scalars.get("VIEW.src", "—"))
        _draw_text(xL, yL, f"View lag {lag:0.1f} ms   jitter |d|/s ~{jit:0.2f}   src {src}   hit {hit}", 12*scale); yL -= lh

    # graphs
    if scene.dev_hud_graphs:
        gx = xL
        gw = col_w_left - int(16*scale)
        gh = int(2.0 * lh)
        def graph(label, series_key, ymin=None, ymax=None, color=(0.31,0.66,1.0,1.0), zero=None):
            nonlocal yL
            _draw_text(gx, yL, label, 12*scale); yL -= int(0.6*lh)
            sy = yL - gh
            _draw_rect(gx, sy, gw, gh, (0.10,0.12,0.15,0.85))
            s = STATE.series.get(series_key) or BUS.series.get(series_key)
            if s:
                lo, hi = s.minmax()
                if ymin is not None or ymax is not None:
                    lo = lo if ymin is None else float(ymin)
                    hi = hi if ymax is None else float(ymax)
                    if hi <= lo: hi = lo + 1.0
                if (zero is not None) and (lo < zero < hi):
                    zq = (zero - lo)/(hi-lo); zy = sy + (1.0 - zq)*gh
                    _draw_line_strip([(gx, zy), (gx+gw, zy)], (0.45,0.50,0.58,0.8), width=1.0)
                pts = _samples_to_poly(s, gx, sy, gw, gh, lo, hi)
                _draw_line_strip(pts, color, width=1.4*scale)
            yL = sy - int(0.5*lh)

        graph("Frame ms", "frame_ms", ymin=0.0, ymax=max(40.0, STATE.ms_ema*1.5), color=(0.50,1.0,0.60,1.0))
        graph("XR RTT (ms)", "rtt_ms", ymin=0.0, ymax=max(12.0, (STATE.series['rtt_ms'].last or 0.0)*2.0), color=(0.95,0.80,0.40,1.0))
        graph("View Allowed (m)", "view_allowed", ymin=0.0, ymax=None, color=(0.65,0.75,1.0,1.0))

    # right column (stable custom keys)
    if scene.dev_hud_show_custom:
        yR = y_top
        NORMAL = (1.0, 1.0, 1.0, 1.0)
        STALE  = (0.78, 0.82, 0.90, 0.90)
        for grp in STATE.custom_groups_order:
            keys = STATE.custom_keys_by_group.get(grp) or []
            if not keys: continue
            _draw_text(xR, yR, f"{grp}", 12*scale); yR -= lh
            for full_key in keys:
                short = full_key.split(".", 1)[-1]
                val   = STATE.custom_last_val.get(full_key, "—")
                age   = STATE.frame_idx - int(STATE.custom_last_seen.get(full_key, 0))
                colr  = NORMAL if age == 0 else STALE
                _draw_text(xR + int(16*scale), yR, f"{short}: {val}", 12*scale, colr); yR -= lh
