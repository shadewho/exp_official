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
log_game("KCC", "COMPLETE pos=(10,5,3) ground=True")
log_game("CAMERA", "Raycast hit dist=2.5m")
log_game("PHYS-CAPSULE", "clear move=0.3m | 4rays 12tris")

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

The frequency gate prevents log spam while maintaining diagnostic visibility.

---

## Log Categories

Each log has a category that maps to a debug toggle:

| Category | Debug Property | Description |
|----------|---------------|-------------|
| `KCC` | `kcc_offload` | KCC physics step results |
| `CAMERA` | `camera_offload` | Camera occlusion raycasts |
| `DYNAMIC` | `dynamic_offload` | Dynamic mesh activation |
| `FRAME` | `frame_numbers` | Frame numbers with timestamps |
| `PHYS-CAPSULE` | `physics_capsule` | Capsule sweep collision testing |
| `PHYS-GROUND` | `physics_ground` | Ground detection raycasts |
| `PHYS-TIMING` | `physics_timing` | Physics timing breakdown |
| `PHYS-CATCHUP` | `physics_catchup` | Multi-step catchup tracking |
| `PHYS-PLATFORM` | `physics_platform` | Platform carry motion |
| `PHYS-CONSISTENCY` | `physics_consistency` | Frame-to-frame consistency |
| `PHYS-STEP` | `physics_step_up` | Step-up stair climbing |
| `PHYS-SLOPES` | `physics_slopes` | Slope handling |
| `PHYS-SLIDE` | `physics_slide` | Multi-plane wall sliding |
| `PHYS-VERTICAL` | `physics_vertical` | Jumping, gravity, ceiling hits |
| `PHYS-ENHANCED` | `physics_enhanced` | Enhanced diagnostic details |
| `CULLING` | `performance` | Distance-based object culling |
| `INTERACTIONS` | `interactions` | Interaction and reaction system |
| `AUDIO` | `audio` | Audio playback and state |
| `ANIMATIONS` | `animations` | Animation and NLA system |

---

## Worker Process Logs

Worker processes (engine offload) can't access the main buffer directly, so they:
1. Collect logs during computation: `worker_logs.append(("KCC", "message"))`
2. Return logs in engine result: `result["logs"]`
3. Main thread adds them via: `log_worker_messages(worker_logs)`

The frequency gate is applied when worker logs are added to the buffer.

---

## When to Use Logger vs Print

### ✅ MUST Use Logger (Performance Critical)
Anything called during the **game loop** that could impact performance:
- Physics calculations (KCC, collisions, raycasts)
- Camera occlusion checks
- Performance culling operations
- Dynamic mesh activation
- Platform carry motion
- Any worker process diagnostics
- Frame-by-frame tracking

**Why**: These are called every frame (30+ times per second). Console prints would destroy performance.

### ✅ Can Use Print (Non-Critical)
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

Example:
```
[KCC F0042 T1.400s] COMPLETE pos=(10.5,5.2,3.1) vel=(2.3,1.5,0.0) ground=True
[CAMERA F0042 T1.402s] SUBMIT job=17 origin=(10.5,5.2,5.1) dir=(0.0,-0.97,0.26)
[PHYS-CAPSULE F0042 T1.403s] clear move=0.3m | 4rays 12tris
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
- Individual category toggles (`dev_debug_kcc_offload`, etc.)

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

### ❌ Bad (Console Print)
```python
# In game loop - DESTROYS PERFORMANCE
if scene.dev_debug_kcc_offload:
    print(f"KCC pos=({pos.x:.2f},{pos.y:.2f},{pos.z:.2f})")  # ~1000μs
```

### ✅ Good (Fast Buffer Logger)
```python
# In game loop - FAST
if scene.dev_debug_kcc_offload:
    log_game("KCC", f"pos=({pos.x:.2f},{pos.y:.2f},{pos.z:.2f})")  # ~1μs
```

### ✅ Also Good (Non-Critical Print)
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

**Last Updated**: 2025-12-03
**Status**: Fully operational, tested, performance-critical
