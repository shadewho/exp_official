# Exploratory/Exp_Game/xr_systems/xr_jobs/kcc_ops.py
import math

def _normalize2(x, y):
    l2 = x*x + y*y
    if l2 <= 1.0e-12:
        return 0.0, 0.0
    inv = 1.0 / math.sqrt(l2)
    return x*inv, y*inv

def _remove_uphill_xy(mx, my, nx, ny, nz, max_slope_dot):
    # up·n = nz when up=(0,0,1)
    up_dot = max(-1.0, min(1.0, nz))
    if up_dot >= float(max_slope_dot):
        return mx, my
    # uphill = up - n*(up·n) = (0,0,1) - n*nz
    gx = -nx * nz
    gy = -ny * nz
    gl2 = gx*gx + gy*gy
    if gl2 <= 1.0e-12:
        return mx, my
    inv = 1.0 / math.sqrt(gl2)
    gx *= inv; gy *= inv
    comp = mx*gx + my*gy
    if comp > 0.0:
        mx -= gx * comp
        my -= gy * comp
    return mx, my

def _move_xy(payload):
    dx = float(payload.get("dx", 0.0))
    dy = float(payload.get("dy", 0.0))
    yaw = float(payload.get("yaw", 0.0))
    on_ground = bool(payload.get("on_ground", False))
    n = payload.get("normal", None)
    max_slope_dot = float(payload.get("max_slope_dot", 0.7))

    # 1) normalize input intent
    ix, iy = _normalize2(dx, dy)
    if ix == 0.0 and iy == 0.0:
        return {"xy": (0.0, 0.0)}

    # 2) rotate by yaw about Z
    cy = math.cos(yaw); sy = math.sin(yaw)
    mx = ix * cy - iy * sy
    my = ix * sy + iy * cy

    # 3) optional steep-slope clamp
    if on_ground and isinstance(n, (list, tuple)) and len(n) == 3:
        nx, ny, nz = float(n[0]), float(n[1]), float(n[2])
        nn = math.sqrt(nx*nx + ny*ny + nz*nz)
        if nn > 1.0e-12:
            nx, ny, nz = nx/nn, ny/nn, nz/nn
        mx, my = _remove_uphill_xy(mx, my, nx, ny, nz, max_slope_dot)

    # 4) renormalize (cosmetic)
    mx, my = _normalize2(mx, my)
    return {"xy": (mx, my)}

def register(register_job):
    # Public job name
    register_job("kcc.move_xy.v1", _move_xy)
