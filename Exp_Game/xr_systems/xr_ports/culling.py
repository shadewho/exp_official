# Blender-side helper used from exp_performance.update_performance_culling()
# This file stays tiny and focused; all decisions live in exp_performance.

from ..xr_queue import xr_enqueue

_JOB_NAME = "cull.batch.v1"

def queue_cull_batch(
    op,
    entry_ptr: int,
    names: list[str],
    positions: list[tuple[float, float, float]],
    ref_loc_xyz: tuple[float, float, float],
    thresh: float,
    start_idx: int,
    max_count: int,
    sim_time: float,
    apply_fn,          # callable(result_dict) -> writes_applied:int
    set_next_idx_fn,   # callable(int) -> None
):
    payload = {
        "entry_ptr": int(entry_ptr),
        "obj_names": names,
        "obj_positions": positions,
        "ref_loc": (float(ref_loc_xyz[0]), float(ref_loc_xyz[1]), float(ref_loc_xyz[2])),
        "thresh": float(thresh),
        "start": int(start_idx),
        "max_count": int(max_count),
        "t": float(sim_time),
    }

    def _apply(result_dict):
        # result looks like: {"entry_ptr":..., "next_idx":..., "changes":[(name, desired_hidden), ...]}
        writes = 0
        try:
            if callable(apply_fn):
                writes = int(apply_fn(result_dict) or 0)
            if callable(set_next_idx_fn) and isinstance(result_dict, dict) and "next_idx" in result_dict:
                set_next_idx_fn(int(result_dict["next_idx"]))
        except Exception:
            pass

        # Update HUD counters
        xr_stats = getattr(op, "_xr_stats", None)
        if isinstance(xr_stats, dict) and isinstance(result_dict, dict):
            scanned = len(result_dict.get("changes", [])) if result_dict.get("changes") else 0
            xr_stats["scan_total"] = int(xr_stats.get("scan_total", 0)) + scanned
            xr_stats["writes_total"] = int(xr_stats.get("writes_total", 0)) + max(0, writes)
            xr_stats["last_batch"] = scanned
            xr_stats["last_total"] = len(names)

    xr_enqueue(_JOB_NAME, payload, _apply)
