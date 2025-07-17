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