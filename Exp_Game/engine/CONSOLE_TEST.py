# PASTE THIS INTO BLENDER'S PYTHON CONSOLE
# Tests the multiprocessing engine standalone (not in modal)

from Exploratory.Exp_Game.engine.engine_core import EngineCore
from Exploratory.Exp_Game.engine.stress_test import run_stress_test, quick_test

print("\n" + "="*70)
print("  STARTING ENGINE STRESS TEST")
print("="*70)

# Create and start engine
engine = EngineCore()
engine.start()

# Quick check first
if not engine.is_alive():
    print("\n❌ ENGINE FAILED TO START!")
    print("Check System Console (Window → Toggle System Console) for errors")
else:
    print("\n✓ Engine started successfully")
    print(f"✓ Workers: {engine.get_stats()['workers_alive']}/{engine.get_stats()['workers_total']}")

    # Run quick test
    print("\nRunning quick test...")
    if quick_test(engine):
        print("\n" + "="*70)
        print("  RUNNING FULL STRESS TEST (5 seconds)")
        print("="*70)

        # Run full stress test
        result = run_stress_test(engine, duration=5.0, verbose=True)

        # Shutdown
        print("\nShutting down engine...")
        engine.shutdown()
        print("✓ Test complete")
    else:
        print("\n❌ Quick test failed - skipping full test")
        engine.shutdown()
