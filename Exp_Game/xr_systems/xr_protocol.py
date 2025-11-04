# XR message protocol helpers (intentionally tiny)

PROTOCOL_VERSION = 1

def make_job(name: str, payload: dict, job_id: str) -> dict:
    return {"id": str(job_id), "name": str(name), "payload": payload}

def make_batch(frame_seq: int, sim_time: float, jobs: list[dict]) -> dict:
    return {
        "type": "frame_input",
        "frame_seq": int(frame_seq),
        "t": float(sim_time),
        "jobs": jobs,
        # You can include other cosmetic fields here if you want (e.g., hud "set")
    }
