from __future__ import annotations
import gpu
import blf
from gpu_extras.batch import batch_for_shader

def corner_xy(W, H, w, h, pos: str, pad: int):
    return {'TR': (W - w - pad, pad), 'TL': (pad, pad),
            'BR': (W - w - pad, H - h - pad), 'BL': (pad, H - h - pad)}.get(pos, (W - w - pad, pad))

def draw_rect(x, y, w, h, col):
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    verts = [(x,y), (x+w,y), (x+w,y+h), (x,y+h)]
    batch = batch_for_shader(shader, 'TRI_FAN', {"pos": verts})
    gpu.state.blend_set('ALPHA')
    shader.bind(); shader.uniform_float("color", col); batch.draw(shader)

def draw_line_strip(points, col, width=1.0):
    if len(points) < 2: return
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    batch  = batch_for_shader(shader, 'LINE_STRIP', {"pos": points})
    gpu.state.blend_set('ALPHA')
    try: gpu.state.line_width_set(max(1.0, float(width)))
    except Exception: pass
    shader.bind(); shader.uniform_float("color", col); batch.draw(shader)
    try: gpu.state.line_width_set(1.0)
    except Exception: pass

def draw_text(x, y, text, size_px=12, color=(1,1,1,1)):
    blf.position(0, x, y, 0); blf.size(0, size_px); blf.color(0, *color); blf.draw(0, text)

def samples_to_poly(series, x, y, w, h, ymin, ymax):
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
