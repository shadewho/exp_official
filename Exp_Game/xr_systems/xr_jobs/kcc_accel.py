# Exploratory/Exp_Game/xr_systems/xr_jobs/kcc_accel.py
import math

def _normalize2(x, y):
    l2 = x * x + y * y
    if l2 <= 1.0e-12:
        return 0.0, 0.0
    inv = 1.0 / math.sqrt(l2)
    return x * inv, y * inv

def _accel_xy(payload: dict):
    vx = float(payload.get("vx", 0.0))
    vy = float(payload.get("vy", 0.0))
    wx = float(payload.get("wx", 0.0))
    wy = float(payload.get("wy", 0.0))
    target_speed = float(payload.get("target_speed", 0.0))
    accel = float(payload.get("accel", 0.0))
    dt = float(payload.get("dt", 0.0))

    # Desired = normalized wish * target_speed
    wx, wy = _normalize2(wx, wy)
    dx = wx * target_speed
    dy = wy * target_speed

    # Exact blend: cur.lerp(desired, t) with t ∈ [0,1]
    t = accel * dt
    if t < 0.0: t = 0.0
    if t > 1.0: t = 1.0

    nx = vx + (dx - vx) * t
    ny = vy + (dy - vy) * t
    return {"xy": (nx, ny)}

def register(register_job):
    register_job("kcc.accel_xy.v1", _accel_xy)
