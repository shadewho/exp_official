# XR message protocol helpers (intentionally tiny)

"""Developers manifesto: develop a strong output of the character, actions, 
environment, utilties etc so that we can eliminate guess work from XR development.
Visualize and output statistics and save time debugging and confusion in the 
future. Gate everything by ON/OFF toggles so that we can quickly enable/disable 
and don't bloat the system when not needed. It's critical every new component 
and is backed by data and visualization to help us understand the system and develop it.
Use /Developers module and the framwork within to build these systems out.
Use DevHud to output text and graphs as needed. 
"""

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
