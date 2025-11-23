# Engine Testing Guide

Complete guide to testing the multiprocessing engine for production readiness.

## Quick Start

### Method 1: Python Console (Recommended)

1. Open Blender
2. Open System Console: `Window → Toggle System Console`
3. Open Python Console in Blender
4. Copy and paste the contents of `CONSOLE_TEST.py`
5. Watch the output in System Console

### Method 2: Search Menu

1. Press `F3` (search)
2. Type "Quick Engine Test" or "Engine Stress Test"
3. Run the operator
4. Check System Console for detailed output

### Method 3: Manual Testing

```python
from Exploratory.Exp_Game.engine import EngineCore
from Exploratory.Exp_Game.engine.stress_test import run_stress_test

# Start engine
engine = EngineCore()
engine.start()

# Run test
if engine.is_alive():
    result = run_stress_test(engine, duration=5.0)
    print(f"Grade: {result['grade']}")
    engine.shutdown()
```

---

## What Gets Tested

The stress test validates:

### 1. Worker Health
- All workers start successfully
- Workers remain alive during operation
- Worker process isolation (no bpy imports)

### 2. Job Submission
- Jobs can be submitted to queue
- Queue handles burst loads
- Non-blocking submission works

### 3. Job Processing
- Workers process jobs correctly
- Data integrity maintained
- Error handling works

### 4. Result Retrieval
- Results returned successfully
- No results lost
- Correct job-result matching

### 5. Performance Metrics
- **Throughput**: Jobs processed per second
- **Latency**: Round-trip time for jobs
- **Completion Rate**: % of jobs that return
- **Queue Saturation**: Rejected jobs under load

### 6. Synchronization
- Main thread and workers communicate properly
- No race conditions
- Graceful shutdown

---

## Understanding Results

### Grade Scale

- **A (✓✓✓)**: Ready for production
  - Throughput > 200 jobs/sec
  - Latency < 50ms
  - 99%+ completion rate
  - All workers alive

- **B (✓✓)**: Functional with minor issues
  - Throughput 100-200 jobs/sec
  - Latency 50-100ms
  - 95-99% completion rate

- **C (⚠)**: Needs optimization
  - Throughput 50-100 jobs/sec
  - Latency 100-200ms
  - 80-95% completion rate

- **F (✗✗✗)**: Not ready
  - Throughput < 50 jobs/sec
  - Latency > 200ms
  - < 80% completion rate
  - Workers dying

### Sample Output

```
======================================================================
  ENGINE STRESS TEST RESULTS - GRADE: A ✓✓✓
======================================================================
  Status: READY

  WORKERS:
    Alive: 4/4

  JOBS:
    Submitted:  1250
    Received:   1250 (100.0%)
    Success:    1250
    Failed:     0
    Lost:       0
    Rejected:   0

  PERFORMANCE:
    Throughput: 250 jobs/sec
    Latency:    45.2ms avg (12.3-98.7ms)

  TIMING:
    Test:       5.00s
    Submit:     5.00s
    Collect:    0.24s

  VERDICT:
    ✓ Engine is READY for production use
    ✓ All systems performing optimally
======================================================================
```

---

## Common Issues

### Workers Not Starting

**Symptom**: `0/4 workers alive`

**Causes**:
- BPY import guard missing in `__init__.py`
- Workers trying to import `bpy` module
- Multiprocessing spawn issue

**Fix**:
1. Check `Exploratory/__init__.py` has import guard (lines 29-58)
2. Ensure all `bpy` imports are inside the `else` block
3. Restart Blender completely

### Jobs Timing Out

**Symptom**: `JOB TIMEOUT` error

**Causes**:
- Workers stuck or crashed
- Job type not implemented in `engine_worker_entry.py`
- Queue communication broken

**Fix**:
1. Check System Console for worker errors
2. Verify `COMPUTE_HEAVY` and `ECHO` handlers exist in `process_job()`
3. Check workers are alive: `engine.get_stats()['workers_alive']`

### Low Throughput

**Symptom**: Grade C or F, `Low throughput` message

**Causes**:
- Not enough workers
- Job complexity too high
- CPU throttling

**Fix**:
1. Increase `WORKER_COUNT` in `engine_config.py`
2. Reduce job complexity
3. Check CPU usage is not maxed out

### High Latency

**Symptom**: `High latency` warning

**Causes**:
- Workers overloaded
- Job queue saturated
- Too few workers for workload

**Fix**:
1. Reduce job complexity
2. Increase `WORKER_COUNT`
3. Batch jobs instead of submitting individually

### Queue Saturation

**Symptom**: `Queue saturated` warning, many rejections

**Causes**:
- Submitting faster than workers process
- `JOB_QUEUE_SIZE` too small

**Fix**:
1. Increase `JOB_QUEUE_SIZE` in `engine_config.py`
2. Throttle job submission rate
3. Add more workers

### Jobs Being Lost

**Symptom**: Completion rate < 100%, lost jobs > 0

**Causes**:
- Workers crashing
- Result queue overflow
- Pickling errors

**Fix**:
1. Check System Console for worker exceptions
2. Increase `RESULT_QUEUE_SIZE`
3. Ensure job data is picklable (no bpy objects!)

---

## Performance Tuning

### For High Throughput

```python
# engine_config.py
WORKER_COUNT = 8           # More workers
JOB_QUEUE_SIZE = 100       # Bigger queue
RESULT_QUEUE_SIZE = 100
```

### For Low Latency

```python
# engine_config.py
WORKER_COUNT = 2           # Fewer workers, less contention
JOB_QUEUE_SIZE = 5         # Small queue
RESULT_QUEUE_SIZE = 5
```

### For Stability

```python
# engine_config.py
WORKER_COUNT = 4
JOB_QUEUE_SIZE = 10        # Balanced
RESULT_QUEUE_SIZE = 10
DEBUG_ENGINE = True        # Enable logging
```

---

## Integration Testing

To test engine within the modal operator:

```python
# In Blender, start the game normally
# Then check System Console for:

[ExpModal] Multiprocessing engine started successfully
[Engine Core] Started successfully with 4 workers
[Engine Core] HEARTBEAT #1 - Workers: 4/4, Jobs: 0, Uptime: 1.0s
[Engine Core] HEARTBEAT #2 - Workers: 4/4, Jobs: 0, Uptime: 2.0s
...
[ExpModal] Shutting down multiprocessing engine...
[Engine Core] Shutdown complete
```

### Expected Behavior

✓ Engine starts on game start
✓ Heartbeats every 1 second
✓ All workers alive
✓ Clean shutdown on game end

### Red Flags

❌ `WARNING: Engine failed to start`
❌ `Workers: 0/4`
❌ `ModuleNotFoundError: No module named '_bpy'`
❌ Workers crashing/restarting

---

## Testing Before Adding Game Logic

**CRITICAL**: Always test the engine FIRST before adding real game code:

1. Run `CONSOLE_TEST.py` - should get Grade A or B
2. Start/stop game 10 times - engine should always start
3. Leave game running for 60s - all workers should stay alive
4. Submit 1000 test jobs - should get 100% completion

Only after all these pass should you add real game logic to the engine.

---

## Debugging Tips

### Enable Verbose Logging

```python
# engine_config.py
DEBUG_ENGINE = True

# engine_worker_entry.py (line 22)
DEBUG_ENGINE = True
```

### Check Worker Process IDs

```python
engine = EngineCore()
engine.start()

for i, worker in enumerate(engine._workers):
    print(f"Worker {i}: PID {worker.pid}, Alive: {worker.is_alive()}")
```

### Monitor Queue Sizes

```python
# In modal loop
stats = engine.get_stats()
print(f"Pending jobs: {stats['jobs_pending']}")
```

### Test Individual Job Types

```python
# Test ECHO
job_id = engine.submit_job("ECHO", {"test": 123})

# Test COMPUTE_HEAVY
job_id = engine.submit_job("COMPUTE_HEAVY", {"iterations": 1000, "data": [1,2,3]})

# Wait and check result
results = engine.poll_results()
for r in results:
    if r.job_id == job_id:
        print(f"Success: {r.success}, Error: {r.error}")
```

---

## What Comes Next

Once stress test passes with Grade A or B:

1. ✅ Engine core is solid
2. ✅ Workers are stable
3. ✅ Communication works
4. ✅ Ready for game logic

Now you can safely add:
- AI pathfinding jobs
- Physics predictions
- Complex calculations
- Expensive computations

All without blocking the main game loop!

---

## Support

If tests fail and you can't figure out why:

1. Check System Console first
2. Enable `DEBUG_ENGINE = True`
3. Review `CLAUDE.md` multiprocessing section
4. Check that BPY import guard exists in `__init__.py`
5. Verify Python version (3.11+ recommended)
