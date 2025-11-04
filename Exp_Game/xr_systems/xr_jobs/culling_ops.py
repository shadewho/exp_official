#xr_systems/xr_jobs/culling_ops.py
# Minimal per-object distance check (keeps algorithm trivial).
# All thresholds and batching decisions are made on the Blender side.

def _compute_cull_batch(entry_ptr, obj_names, obj_positions, ref_loc, thresh, start, max_count):
    try:
        rx, ry, rz = (float(ref_loc[0]), float(ref_loc[1]), float(ref_loc[2]))
    except Exception:
        rx = ry = rz = 0.0
    try:
        t2 = float(thresh) * float(thresh)
    except Exception:
        t2 = 0.0

    n = len(obj_names)
    if n == 0:
        return {"entry_ptr": int(entry_ptr), "next_idx": int(start), "changes": []}

    out = []
    idx = int(start) % n
    max_count = int(max_count) if max_count > 0 else n
    i = 0
    while i < n and len(out) < max_count:
        name = obj_names[idx]
        try:
            px, py, pz = obj_positions[idx]
            dx = float(px) - rx
            dy = float(py) - ry
            dz = float(pz) - rz
            far = (dx*dx + dy*dy + dz*dz) > t2
        except Exception:
            far = True
        out.append((name, far))
        i += 1
        idx = (idx + 1) % n

    return {"entry_ptr": int(entry_ptr), "next_idx": int(idx), "changes": out}

def register(register_job):
    register_job("cull.batch.v1", lambda payload: _compute_cull_batch(
        payload.get("entry_ptr", 0),
        payload.get("obj_names", []),
        payload.get("obj_positions", []),
        payload.get("ref_loc", (0.0, 0.0, 0.0)),
        payload.get("thresh", 0.0),
        payload.get("start", 0),
        payload.get("max_count", 0),
    ))
