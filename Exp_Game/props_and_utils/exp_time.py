# File: Exp_Game/props_and_utils/exp_time.py
import time

# -------- Clocks --------
# Real-time (wall clock) – for UI/camera smoothing etc.
_last_real_time = 0.0

# Simulation time – advance only on fixed simulation ticks (e.g., physics)
_sim_time = 0.0


def init_time():
    """
    Call once at the start of your game/scene to reset the clocks.
    - Real time is captured from a monotonic source.
    - Simulation time starts at 0.0 and is ONLY advanced via tick_sim_time().
    """
    global _last_real_time, _sim_time
    _last_real_time = time.perf_counter()
    _sim_time = 0.0


def update_real_time():
    """
    Returns wall-clock delta since last call and updates the internal marker.
    Does NOT advance simulation time.
    """
    global _last_real_time
    now = time.perf_counter()
    dt = now - _last_real_time
    _last_real_time = now
    return dt


def tick_sim_time(dt):
    """
    Advance the simulation clock by dt seconds (must be called on fixed ticks).
    """
    global _sim_time
    if dt > 0.0:
        _sim_time += float(dt)


def get_game_time():
    """
    Returns simulation time in seconds (advanced by tick_sim_time()).
    Use this for gameplay, animations, audio durations, timers, etc.
    """
    return _sim_time


# ---- Back-compat helper ----
def update_time():
    """
    Back-compat alias retained for existing imports.
    NOTE: Unlike the old version, this only returns wall-clock delta and
    does NOT advance simulation time. Use tick_sim_time() on fixed steps.
    """
    return update_real_time()


class FixedStepClock:
    """
    Deterministic accumulator. Add real dt each frame; step physics at fixed_dt up to max_steps.
    (Unchanged from your original; provided for completeness.)
    """
    def __init__(self, fixed_dt=1.0/60, max_steps=5):
        self.fixed_dt = float(fixed_dt)
        self.max_steps = int(max_steps)
        self._acc = 0.0

    def add(self, real_dt):
        self._acc += max(0.0, real_dt)

    def steps(self):
        eps = self.fixed_dt * 1e-3
        n = int((self._acc + 0.5 * self.fixed_dt - eps) / self.fixed_dt)
        if n > self.max_steps:
            n = self.max_steps
        if n > 0:
            self._acc -= n * self.fixed_dt
        return n
