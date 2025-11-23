# Paste this entire block into Blender's Python Console

from Exploratory.Exp_Game.engine import engine_config
engine_config.DEBUG_ENGINE = False

from Exploratory.Exp_Game.engine.stress_test import run_stress_test
from Exploratory.Exp_Game.engine.engine_core import EngineCore

print("\nStarting engine stress test...\n")

engine = EngineCore()
engine.start()

if engine.is_alive():
    report = run_stress_test(engine, duration=5.0)
    print(report)
    engine.shutdown()
else:
    print("❌ ENGINE FAILED TO START\n")
