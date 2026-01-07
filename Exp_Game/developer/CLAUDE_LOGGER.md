# Fast Buffer Logger System

## Critical Performance Issue

**IMPORTANT**: Console `print()` statements during gameplay cause severe performance degradation (~1000μs+ per call). This was causing frame drops, stuttering, and unacceptable game performance.

## The Solution: Fast Buffer Logger

The logging system (`dev_logger.py`) solves this by:
- Writing to an **in-memory buffer** during gameplay (~1μs per call)
- **Zero I/O** during game loop (no console writes, no file writes)
- **Batch export** to file when game stops
- **Frequency gating** via master Hz control (1-30 Hz)

Performance improvement: **1000x faster** than console prints.

---

## How It Works

### 1. Session Start
```python
from Exp_Game.developer.dev_logger import start_session

# Call once when game starts
start_session()  # Resets buffer, starts frame tracking
```

### 2. During Gameplay (Fast Logging)
```python
from Exp_Game.developer.dev_logger import log_game, increment_frame

# Log game events (fast - just appends to memory buffer)
# UNIFIED: Physics logs show source (static or dynamic)
log_game("KCC", "GROUND pos=(10,5,3) ground=True")
log_game("GROUND", "source=static z=5.0 normal=(0,0,1)")
log_game("GROUND", "source=dynamic_12345 z=5.5 normal=(0,0,1)")
log_game("HORIZONTAL", "clear move=0.3m | 4rays 12tris")

# Call once per frame
increment_frame()
```

### 3. Session End (Export)
```python
from Exp_Game.developer.dev_logger import export_game_log, clear_log

# When game stops, write all logs to file
if context.scene.dev_export_session_log:
    export_game_log("C:/Users/spenc/Desktop/engine_output_files/diagnostics_latest.txt")
    clear_log()  # Prepare for next session
```

### 4. Frequency Gating
All logging respects the **Master Hz** setting in Developer Tools panel:
- `30 Hz` = Every frame (verbose, for short tests only)
- `5 Hz` = Every 5th frame (~0.17s intervals)
- `1 Hz` = Once per second (recommended for most debugging)

**IMPORTANT: Per-Message-Type Gating**

Frequency gating is applied per **message type**, not per category. The first word of each message is used as the gate key. This ensures different log types don't block each other:

```
# These have SEPARATE gates (both can log in same frame):
log_game("ANIMATIONS", "BATCH_SUBMIT job=123")  # Gate: animations:BATCH_SUBMIT
log_game("ANIMATIONS", "BATCH_RESULT job=123")  # Gate: animations:BATCH_RESULT

# These share a gate (only one per Hz interval):
log_game("ANIMATIONS", "BATCH_SUBMIT job=123")  # Gate: animations:BATCH_SUBMIT
log_game("ANIMATIONS", "BATCH_SUBMIT job=456")  # Gate: animations:BATCH_SUBMIT (blocked)
```

This prevents important result logs from being blocked by submit logs when they occur in the same frame.

---

## Log Categories

UNIFIED PHYSICS: Static and dynamic meshes use identical physics code.
All physics logs show source (static/dynamic) - there is ONE system.

| Category | Debug Property | Description |
|----------|---------------|-------------|
| **Engine** |
| `ENGINE` | `engine` | Core engine diagnostics |
| **Offload Systems** |
| `KCC` | `kcc_physics` | KCC physics step results |
| `CAMERA` | `camera` | Camera occlusion raycasts |
| `FRAME` | `frame_numbers` | Frame numbers with timestamps |
| `CULLING` | `culling` | Distance-based object culling |
| **Optimization Systems** |
| `WORLD-STATE` | `world_state` | World state filtering (Phase 1.1) - shows collected vs total objects |
| `AABB-CACHE` | `aabb_cache` | AABB cache for collisions (Phase 1.2) - shows cache hits/misses per frame |
| **Unified Physics** (all show source: static/dynamic) |
| `PHYSICS` | `physics` | Physics summary per frame |
| `GROUND` | `physics_ground` | Ground detection raycasts |
| `HORIZONTAL` | `physics_horizontal` | Horizontal collision (walls/obstacles) |
| `BODY` | `physics_body` | Body integrity (embedding detection) |
| `CEILING` | `physics_ceiling` | Ceiling check |
| `STEP` | `physics_step` | Step-up stair climbing |
| `SLIDE` | `physics_slide` | Wall slide |
| `SLOPES` | `physics_slopes` | Slope handling |
| **Dynamic Mesh System** (unified with static) |
| `DYN-CACHE` | `dynamic_cache` | Dynamic mesh caching & transforms |
| `PLATFORM` | `dynamic_cache` | Platform attach/detach (relative position) |
| **Game Systems** |
| `INTERACTIONS` | `interactions` | Interaction and reaction system |
| `AUDIO` | `audio` | Audio playback and state |
| `TRACKERS` | `trackers` | Tracker evaluation system |
| **Animation System** |
| `ANIMATIONS` | `animations` | Animation batch jobs, blending, bone updates |
| `ANIM-CACHE` | `anim_cache` | Animation caching in workers |
| `ANIM-WORKER` | `anim_worker` | Animation worker job routing |
| `TEST_MODAL` | `animations` | Animation 2.0 test modal playback |
| **Pose System** (Development/Testing) |
| `POSES` | `poses` | Pose library capture/apply |
| `POSE-CACHE` | `poses` | Pose cache transfer to workers |
| `POSE-BLEND` | `pose_blend` | Pose-to-pose blending diagnostics |

---

## Worker Process Logs

Worker processes (engine offload) can't access the main buffer directly, so they:
1. Collect logs during computation: `worker_logs.append(("KCC", "message"))`
2. Return logs in engine result: `result["logs"]`
3. Main thread adds them via: `log_worker_messages(worker_logs)`

The frequency gate is applied when worker logs are added to the buffer.

---

## When to Use Logger vs Print

### MUST Use Logger (Performance Critical)
Anything called during the **game loop** that could impact performance:
- Physics calculations (KCC, ground, horizontal, body, ceiling, step, slide, slopes)
- Camera occlusion checks
- Performance culling operations
- Dynamic mesh activation
- Platform carry motion
- Any worker process diagnostics
- Frame-by-frame tracking

**Why**: These are called every frame (30+ times per second). Console prints would destroy performance.

### Can Use Print (Non-Critical)
Things that happen **outside the game loop** or are **one-time events**:
- Startup logs (addon initialization)
- Engine stress tests (standalone operators)
- Error messages (exceptions, failures)
- User actions (button clicks, operator execution)
- Shutdown/cleanup messages
- Development debug statements (temporary)

**Why**: These are infrequent and don't run during gameplay.

---

## Output Format

Exported logs are formatted as:
```
[CATEGORY F#### T##.###s] message
```

Example (UNIFIED PHYSICS - shows source):
```
[KCC F0042 T1.400s] GROUND pos=(10.5,5.2,3.1) vel=(2.3,1.5,0.0) ground=True
[GROUND F0042 T1.401s] HIT source=static z=3.10m normal=(0.00,0.00,1.00) | player_z=3.10 tris=12
[GROUND F0042 T1.401s] HIT source=dynamic_12345 z=3.50m normal=(0.00,0.00,1.00) | player_z=3.50 tris=8
[HORIZONTAL F0042 T1.402s] clear move=0.3m | 4rays 12tris
[PHYSICS F0042 T1.403s] total=150us | static+dynamic=2 | rays=6 tris=45 | ground=static
```

Where:
- `F####` = Frame number
- `T##.###s` = Timestamp since session start
- Message = Diagnostic data

---

## File Structure

### `dev_logger.py`
Core logging system:
- `start_session()` - Initialize session
- `increment_frame()` - Track frame numbers
- `log_game(category, message)` - Fast logging with frequency gating
- `log_worker_messages(worker_logs)` - Process worker logs with frequency gating
- `export_game_log(filepath)` - Write buffer to file
- `clear_log()` - Clear buffer for next session

### `dev_debug_gate.py`
Frequency gating system:
- `should_print_debug(category)` - Check if category should log based on Hz setting
- `reset_debug_timers()` - Reset frequency timers on session start/end

### `dev_properties.py`
Debug properties:
- `dev_debug_master_hz` - Master frequency control (1-30 Hz)
- Individual category toggles (`dev_debug_kcc_physics`, `dev_debug_physics_ground`, etc.)

### `dev_panel.py`
UI panel:
- Master Frequency Control box
- Debug category toggles
- Export Session Log toggle

---

## Best Practices

1. **Always use logger in game loop** - Never print() during gameplay
2. **Use descriptive categories** - Makes filtering/analysis easier
3. **Keep messages concise** - Reduce buffer memory usage
4. **Enable categories selectively** - Only debug what you need
5. **Adjust Hz for context**:
   - Testing specific frame: Use 30 Hz
   - General debugging: Use 1-5 Hz
   - Long sessions: Use 1 Hz
6. **Export logs after every test** - Don't rely on memory between sessions
7. **Share logs with Claude** - Read diagnostics_latest.txt for frame-by-frame analysis

---

## Performance Stats

### Before (Console Prints)
- **Per print**: ~1000μs (1 millisecond)
- **100 prints/frame**: 100ms = **3 FPS**
- **Impact**: Severe stuttering, unplayable

### After (Fast Buffer Logger)
- **Per log**: ~1μs (1 microsecond)
- **100 logs/frame**: 0.1ms = **minimal impact**
- **Impact**: Smooth gameplay, full diagnostics

**Result**: 1000x performance improvement while maintaining full diagnostic capability.

---

## Example: Replacing Prints

### Bad (Console Print)
```python
# In game loop - DESTROYS PERFORMANCE
if scene.dev_debug_kcc_physics:
    print(f"KCC pos=({pos.x:.2f},{pos.y:.2f},{pos.z:.2f})")  # ~1000us
```

### Good (Fast Buffer Logger)
```python
# In game loop - FAST
if scene.dev_debug_kcc_physics:
    log_game("KCC", f"pos=({pos.x:.2f},{pos.y:.2f},{pos.z:.2f})")  # ~1us
```

### Also Good (Non-Critical Print)
```python
# Outside game loop - OK to use print
def invoke(self, context, event):
    print("[ExpModal] Game starting...")  # One-time startup message
    return {'RUNNING_MODAL'}
```

---

## Troubleshooting

**Issue**: Too many logs in output file
- **Solution**: Lower Master Hz setting (use 1 Hz for most debugging)
- **Solution**: Disable unnecessary debug categories

**Issue**: Missing logs for category
- **Solution**: Check that category toggle is enabled in Developer Tools panel
- **Solution**: Verify Export Session Log is enabled
- **Solution**: Check category name matches `_CATEGORY_MAP` in dev_logger.py

**Issue**: Game still stuttering with logs enabled
- **Solution**: This shouldn't happen - logger is designed to be fast
- **Solution**: Check if you have rogue print() statements in game loop
- **Solution**: Use profiler to identify bottleneck

---

## Future Improvements

Potential enhancements:
- [ ] Per-category Hz controls (if needed)
- [ ] Log filtering by severity level
- [ ] Real-time log viewer (non-blocking)
- [ ] Automatic log rotation (keep last N sessions)
- [ ] Performance profiling integration
- [ ] Export to JSON for automated analysis

---