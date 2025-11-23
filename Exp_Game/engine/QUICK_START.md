# Engine Testing - Quick Start

## Fastest Way to Test (30 seconds)

### Option 1: Python Console

1. Open Blender
2. `Window → Toggle System Console`
3. Open Python Console (in Blender)
4. Paste this:

```python
from Exploratory.Exp_Game.engine.engine_core import EngineCore
from Exploratory.Exp_Game.engine.stress_test import quick_test

engine = EngineCore()
engine.start()

if quick_test(engine):
    print("\n✓ ENGINE READY")
else:
    print("\n❌ ENGINE FAILED")

engine.shutdown()
```

### Option 2: Search Menu

1. Press `F3`
2. Type "Quick Engine Test"
3. Press Enter
4. Check System Console

---

## Full Stress Test (5 minutes)

### Using CONSOLE_TEST.py

1. Open `Exp_Game/engine/CONSOLE_TEST.py`
2. Copy entire file contents
3. Paste into Blender's Python Console
4. Wait for results

### Using Operator

1. Press `F3`
2. Type "Engine Stress Test"
3. Press Enter
4. Watch System Console for detailed output

---

## What to Look For

### ✓ Good Signs

```
✓ Engine started: 4/4 workers
✓ Job processed successfully (latency: 15.2ms)
✓ Submitted 1250 jobs in 5.00s (250 jobs/sec)
✓ Received 1250/1250 results

GRADE: A ✓✓✓
Status: READY
```

### ❌ Bad Signs

```
❌ ENGINE FAILED TO START
Workers: 0/4
ModuleNotFoundError: No module named '_bpy'

GRADE: F ✗✗✗
Status: NOT READY
```

---

## Next Steps

### If Grade A or B
✅ Engine is ready
✅ Proceed with adding game logic
✅ Test in modal operator

### If Grade C
⚠ Engine works but needs optimization
⚠ Check `README_TESTING.md` for tuning
⚠ Address issues before production

### If Grade F or Errors
❌ Critical issues exist
❌ Check System Console for errors
❌ Review `README_TESTING.md` troubleshooting
❌ Verify BPY import guard in `__init__.py`

---

## Testing in Modal Operator

After standalone test passes:

1. Start game normally (from Exploratory panel)
2. Check System Console for:
   ```
   [ExpModal] Multiprocessing engine started successfully
   [Engine Core] Started successfully with 4 workers
   ```
3. Play for 30 seconds
4. Check heartbeats appear every 1 second:
   ```
   [Engine Core] HEARTBEAT #1 - Workers: 4/4
   ```
5. End game
6. Verify clean shutdown:
   ```
   [Engine Core] Shutdown complete
   ```

---

## Common Problems

### "ModuleNotFoundError: No module named '_bpy'"
→ BPY import guard issue
→ Check `Exploratory/__init__.py` lines 29-58

### "Engine failed to start"
→ Restart Blender completely
→ Check System Console for details

### "Workers: 0/4"
→ Worker processes crashed
→ Enable `DEBUG_ENGINE = True` in `engine_config.py`

### "Jobs timing out"
→ Job type not implemented
→ Check `engine_worker_entry.py` has handler for job type

---

## Files Overview

```
engine/
├── QUICK_START.md          ← You are here
├── README_TESTING.md       ← Detailed testing guide
├── CONSOLE_TEST.py         ← Paste into Python Console
├── test_operator.py        ← Blender operators (F3 menu)
├── stress_test.py          ← Test implementation
├── engine_core.py          ← Main engine
├── engine_worker_entry.py  ← Worker logic
└── engine_config.py        ← Configuration
```

---

## Support

Full documentation: `README_TESTING.md`
Architecture docs: `../../../CLAUDE.md` (multiprocessing section)
