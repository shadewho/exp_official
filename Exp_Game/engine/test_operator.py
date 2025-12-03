# Exp_Game/engine/test_operator.py
"""
Blender operator for running engine stress tests from the UI.
"""

import bpy
from .engine_core import EngineCore
from .stress_test import run_stress_test, quick_test


class EXP_ENGINE_OT_StressTest(bpy.types.Operator):
    """Test ENGINE CORE in isolation - worker performance, job throughput, and readiness checks"""
    bl_idname = "exp_engine.stress_test"
    bl_label = "Engine Stress Test"
    bl_description = (
        "Test ENGINE CORE in isolation:\n"
        "• Worker spawning and readiness\n"
        "• Job throughput (jobs/sec)\n"
        "• Latency (round-trip time)\n"
        "• Queue saturation handling\n"
        "• Data integrity\n\n"
        "This tests the engine's raw capabilities, NOT engine+modal integration"
    )
    bl_options = {'REGISTER'}

    intensity: bpy.props.EnumProperty(
        name="Intensity",
        description="Test intensity level",
        items=[
            ('LIGHT', "Light (3s)", "Quick test with light load - verifies basic functionality", 0),
            ('NORMAL', "Normal (5s)", "Standard stress test - recommended for initial validation", 1),
            ('HEAVY', "Heavy (10s)", "Heavy stress test - pushes engine to limits", 2),
            ('EXTREME', "Extreme (20s)", "Extreme stress test - maximum load for production validation", 3),
        ],
        default='NORMAL'
    )

    def execute(self, context):
        # Determine test duration based on intensity
        durations = {
            'LIGHT': 3.0,
            'NORMAL': 5.0,
            'HEAVY': 10.0,
            'EXTREME': 20.0
        }
        duration = durations[self.intensity]

        intensity_names = {
            'LIGHT': "LIGHT",
            'NORMAL': "NORMAL",
            'HEAVY': "HEAVY",
            'EXTREME': "EXTREME"
        }

        print("\n" + "="*70)
        print(f"  ENGINE STRESS TEST - {intensity_names[self.intensity]} INTENSITY")
        print("="*70)
        print(f"  Duration: {duration}s")
        print(f"  This will submit jobs as fast as possible and measure:")
        print(f"    • Worker health and stability")
        print(f"    • Concurrent job processing")
        print(f"    • Throughput (jobs/second)")
        print(f"    • Latency (round-trip time)")
        print(f"    • Queue saturation handling")
        print(f"    • Data integrity")
        print("="*70)

        self.report({'INFO'}, f"Starting {intensity_names[self.intensity]} stress test ({duration}s)...")

        # Create and start engine
        engine = EngineCore()

        print("\n[STARTUP] Creating engine...")
        engine.start()

        if not engine.is_alive():
            self.report({'ERROR'}, "Engine failed to start! Check System Console")
            print("\n" + "="*70)
            print("  ❌ ENGINE STARTUP FAILED")
            print("="*70)
            print("  Check Window → Toggle System Console for error details")
            print("  Common causes:")
            print("    • Workers crashed on startup")
            print("    • BPY import guard issue")
            print("    • Multiprocessing configuration problem")
            print("="*70 + "\n")
            return {'CANCELLED'}

        stats = engine.get_stats()
        print(f"\n[STARTUP] ✓ Engine started successfully")
        print(f"          Workers: {stats['workers_alive']}/{stats['workers_total']} alive")
        print(f"          Ready for stress test")

        # Run full stress test with verbose output
        print(f"\n[TESTING] Running {intensity_names[self.intensity]} stress test...")
        print(f"          Watch below for real-time progress\n")

        result = run_stress_test(engine, duration=duration, verbose=True)

        # Shutdown
        print("\n[SHUTDOWN] Shutting down engine...")
        engine.shutdown()
        print("[SHUTDOWN] ✓ Clean shutdown complete\n")

        # Report results to user
        if result.get("success"):
            grade = result.get("grade", "?")
            status = result.get("status", "UNKNOWN")
            metrics = result.get("metrics", {})
            issues = result.get("issues", [])

            # Print summary box
            print("="*70)
            print(f"  FINAL RESULTS - GRADE: {grade}")
            print("="*70)
            print(f"  Status: {status}")
            print(f"  Throughput: {metrics.get('throughput_jobs_per_sec', 0):.0f} jobs/sec")
            print(f"  Latency: {metrics.get('latency_avg_ms', 0):.1f}ms avg")
            print(f"  Completion: {metrics.get('completion_rate_pct', 0):.1f}%")
            print(f"  Workers: {metrics.get('workers_alive', 0)}/{metrics.get('workers_total', 0)} alive")

            if issues:
                print(f"\n  Issues Found: {len(issues)}")
                for issue in issues:
                    print(f"    • {issue}")

            print("="*70 + "\n")

            # User report based on grade
            if grade == "A":
                self.report({'INFO'}, f"Grade A - Engine READY! {metrics.get('throughput_jobs_per_sec', 0):.0f} jobs/sec")
            elif grade == "B":
                self.report({'WARNING'}, f"Grade B - Functional with minor issues ({len(issues)} issues)")
            elif grade == "C":
                self.report({'WARNING'}, f"Grade C - Needs optimization ({len(issues)} issues)")
            else:
                self.report({'ERROR'}, f"Grade F - NOT READY ({len(issues)} critical issues)")

            return {'FINISHED'}
        else:
            error = result.get("error", "Unknown error")
            message = result.get("message", "")

            print("="*70)
            print(f"  ❌ TEST FAILED: {error}")
            print("="*70)
            if message:
                print(f"  {message}")
            print("="*70 + "\n")

            self.report({'ERROR'}, f"Test failed: {error}")
            return {'CANCELLED'}

    def invoke(self, context, event):
        # Show dialog to select intensity
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        layout.label(text="Select test intensity:")
        layout.prop(self, "intensity", expand=True)
        layout.separator()
        layout.label(text="Higher intensity = longer test + more stress", icon='INFO')


class EXP_ENGINE_OT_QuickTest(bpy.types.Operator):
    """Quick sanity check - verify engine core can start and process jobs"""
    bl_idname = "exp_engine.quick_test"
    bl_label = "Quick Engine Test"
    bl_description = (
        "Quick 3-second sanity check:\n"
        "• Spawn workers\n"
        "• Verify readiness\n"
        "• Process test job\n\n"
        "Tests ENGINE CORE only, not modal integration"
    )
    bl_options = {'REGISTER'}

    def execute(self, context):
        print("\n" + "="*70)
        print("  QUICK ENGINE TEST")
        print("="*70)

        # Create and start engine
        engine = EngineCore()
        engine.start()

        if not engine.is_alive():
            self.report({'ERROR'}, "Engine failed to start!")
            print("❌ Engine not alive")
            print("="*70 + "\n")
            return {'CANCELLED'}

        stats = engine.get_stats()
        print(f"✓ Engine started: {stats['workers_alive']}/{stats['workers_total']} workers")

        # Run quick test
        if quick_test(engine):
            self.report({'INFO'}, "Engine working correctly!")
            print("="*70 + "\n")
        else:
            self.report({'ERROR'}, "Engine test failed!")
            print("="*70 + "\n")

        # Shutdown
        engine.shutdown()
        return {'FINISHED'}


def register():
    bpy.utils.register_class(EXP_ENGINE_OT_QuickTest)
    bpy.utils.register_class(EXP_ENGINE_OT_StressTest)


def unregister():
    bpy.utils.unregister_class(EXP_ENGINE_OT_StressTest)
    bpy.utils.unregister_class(EXP_ENGINE_OT_QuickTest)
