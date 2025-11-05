# XR job: clamp horizontal movement against a steep slope normal
import math

def _clamp_xy(payload: dict):
    hx = float(payload.get("hx", 0.0))
    hy = float(payload.get("hy", 0.0))
    n  = payload.get("normal", (0.0, 0.0, 1.0))
    floor_cos = float(payload.get("floor_cos", 0.7))

    nx, ny, nz = (float(n[0]), float(n[1]), float(n[2])) if isinstance(n, (list, tuple)) and len(n) == 3 else (0.0, 0.0, 1.0)
    nn = math.sqrt(nx*nx + ny*ny + nz*nz)
    if nn > 1.0e-12:
        nx, ny, nz = nx/nn, ny/nn, nz/nn

    # Walkable? then unchanged
    if nz >= floor_cos:
        return {"xy": (hx, hy)}

    # Uphill vector in-plane with the slope
    gx, gy = -nx * nz, -ny * nz
    gl2 = gx*gx + gy*gy
    if gl2 <= 1.0e-12:
        return {"xy": (hx, hy)}
    inv = 1.0 / math.sqrt(gl2)
    gx *= inv; gy *= inv

    # Remove uphill component from (hx,hy)
    comp = hx*gx + hy*gy
    if comp > 0.0:
        hx -= gx * comp
        hy -= gy * comp

    return {"xy": (hx, hy)}

def register(register_job):
    register_job("kcc.clamp_xy.v1", _clamp_xy)
