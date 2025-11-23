# Engine Stress Test

Simple stress test to verify engine readiness.

## Quick Start

**Paste this into Blender's Python Console:**

```python
from Exploratory.Exp_Game.engine import engine_config
engine_config.DEBUG_ENGINE = False

from Exploratory.Exp_Game.engine.stress_test import run_stress_test
from Exploratory.Exp_Game.engine.engine_core import EngineCore

engine = EngineCore()
engine.start()

if engine.is_alive():
    print(run_stress_test(engine, duration=5.0))
    engine.shutdown()
else:
    print("❌ ENGINE FAILED TO START")
```

---

## What You Get

```
============================================================
  ENGINE STRESS TEST - GRADE: A ✓✓✓
============================================================
  Status: READY
  Duration: 5.0s

  PERFORMANCE:
    Speed:      250 jobs/sec
    Response:   45ms average
    Completion: 100% (1250/1250)
    Success:    1250/1250
    Failures:   0 queue rejections

  VERDICT:
    Engine is READY for production.
============================================================
```

---

## Letter Grades

- **A** = Ready for production ✓✓✓
- **B** = Usable with minor issues ✓✓
- **C** = Needs work ⚠
- **F** = Not ready ✗✗✗

---

## Metrics

**Speed (jobs/sec):**
- `< 50` = Too slow
- `50-100` = OK
- `100-200` = Good
- `> 200` = Excellent

**Response (ms):**
- `< 50ms` = Excellent
- `50-100ms` = Good
- `100-200ms` = Acceptable
- `> 200ms` = Too slow

**Completion (%):**
- `100%` = Perfect
- `95-99%` = Acceptable
- `< 95%` = Problem

---

## Common Issues

**"Low throughput"**
→ Increase `WORKER_COUNT` in `engine_config.py`

**"High latency"**
→ Reduce job complexity

**"Jobs being lost"**
→ Check System Console for worker crashes

**"Queue saturation"**
→ Increase `JOB_QUEUE_SIZE` in `engine_config.py`

---

## Custom Duration

```python
run_stress_test(engine, duration=10.0)  # 10 second test
```

---

## Bottom Line

**Grade A or B** → Ship it
**Grade C** → Fix issues first
**Grade F** → Don't ship
