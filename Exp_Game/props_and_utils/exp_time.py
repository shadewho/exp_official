# File: exp_time.py
import time

# Global “game clock” state
_game_time = 0.0        # This is our small, ever-increasing timer
_last_real_time = 0.0   # We'll store the real perf_counter() from last update

def init_time():
    """
    Call once at the start of your game/scene to reset the clock to 0.0.
    """
    global _game_time, _last_real_time
    _game_time = 0.0
    _last_real_time = time.perf_counter()  # Mark the current real (monotonic) time

def update_time():
    """
    Call ONCE per frame or per 'tick' to increment the _game_time
    by the actual real-time delta since last frame.
    Returns: float `delta` (the time in seconds since last update).
    """
    global _game_time, _last_real_time
    now = time.perf_counter()
    delta = now - _last_real_time  # time since last update
    _last_real_time = now
    # Accumulate into our game clock
    _game_time += delta
    return delta

def get_game_time():
    """
    Returns the current 'game time' in seconds since init_time().
    This is always a small, monotonically increasing number.
    """
    return _game_time

class FixedStepClock:
    """
    Deterministic accumulator. Add real dt each frame; step physics at fixed_dt up to max_steps.
    """
    def __init__(self, fixed_dt=1.0/60, max_steps=5):
        self.fixed_dt = float(fixed_dt)
        self.max_steps = int(max_steps)
        self._acc = 0.0

    def add(self, real_dt):
        self._acc += max(0.0, real_dt)

    def steps(self):
        # ROUND rather than FLOOR to prevent N / N+1 alternation
        eps = self.fixed_dt * 1e-3
        n = int((self._acc + 0.5 * self.fixed_dt - eps) / self.fixed_dt)

        if n > self.max_steps:  # keep your existing clamp
            n = self.max_steps
        if n > 0:
            self._acc -= n * self.fixed_dt
        return n