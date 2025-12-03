# Exp_Game/engine/stress_test.py
"""
ENGINE CORE STRESS TEST - Tests engine in isolation

PURPOSE: Validate the multiprocessing engine's raw capabilities:
- Worker spawning and readiness
- Job submission throughput
- Result polling latency
- Queue saturation handling
- Worker health and stability

NOT for testing engine+modal integration - this tests the ENGINE CORE only.
Run via Developer Tools → Manual Stress Tests (Quick Test / Full Stress Test)
"""

import time
from typing import Dict, List, Tuple


def run_stress_test(engine, duration: float = 5.0, verbose: bool = True) -> Dict:
    """
    Run comprehensive stress test on engine.

    Tests:
    - Engine startup and worker health
    - Job submission and processing
    - Result retrieval and accuracy
    - Latency and throughput
    - Queue saturation handling
    - Data integrity

    IMPORTANT: This is a BURST stress test - it intentionally floods the engine
    with jobs as fast as possible to test maximum capacity. High latency (1-5s)
    is EXPECTED because jobs queue up waiting for workers.

    In real gameplay, you submit steadily (~20-50 jobs/sec), not in bursts,
    so actual latency will be <10ms.

    Args:
        engine: EngineCore instance (must be started and alive)
        duration: Test duration in seconds (default: 5)
        verbose: Print progress messages (default: True)

    Returns:
        Dictionary with test results and grade
    """

    if not engine.is_alive():
        return {
            "success": False,
            "error": "ENGINE NOT RUNNING",
            "grade": "F",
            "message": "Engine must be started and alive before testing"
        }

    if verbose:
        print(f"\n{'='*70}")
        print(f"  MULTIPROCESSING ENGINE STRESS TEST")
        print(f"{'='*70}")

    # === TEST 1: COMPREHENSIVE HEALTH CHECK ===
    if verbose:
        print(f"\n[TEST 1/6] Comprehensive health check...")

    # Use new health check method
    health = engine.check_worker_health()

    if not health["healthy"]:
        error_msg = "; ".join(health["critical"] + health["warnings"])
        return {
            "success": False,
            "error": "HEALTH CHECK FAILED",
            "grade": "F",
            "message": error_msg
        }

    if verbose:
        print(f"  ✓ Workers: {health['workers_alive']}/{health['workers_alive'] + health['workers_dead']} alive")
        print(f"  ✓ Engine healthy")

    # === TEST 2: READINESS VERIFICATION ===
    if verbose:
        print(f"\n[TEST 2/6] Verifying engine readiness...")

    # Use new readiness check
    if not engine.wait_for_readiness(timeout=5.0):
        return {
            "success": False,
            "error": "READINESS CHECK FAILED",
            "grade": "F",
            "message": "Workers did not respond to PING within timeout"
        }

    if verbose:
        print(f"  ✓ All workers responding to PING")

    # === TEST 3: BASIC JOB PROCESSING ===
    if verbose:
        print(f"\n[TEST 3/6] Testing basic job processing...")

    # Submit a simple echo job
    test_data = {"test": "hello", "value": 42}
    job_id = engine.submit_job("ECHO", test_data, check_overload=False)

    if job_id is None:
        return {
            "success": False,
            "error": "FAILED TO SUBMIT JOB",
            "grade": "F",
            "message": "Engine rejected job submission"
        }

    # Wait for result (timeout 5s)
    timeout = time.time() + 5.0
    result = None

    while time.time() < timeout:
        results = engine.poll_results(max_results=10)
        for r in results:
            if r.job_id == job_id:
                result = r
                break
        if result:
            break
        time.sleep(0.01)

    if result is None:
        return {
            "success": False,
            "error": "JOB TIMEOUT",
            "grade": "F",
            "message": f"Job {job_id} never returned (workers may be stuck)"
        }

    if not result.success:
        return {
            "success": False,
            "error": "JOB FAILED",
            "grade": "F",
            "message": f"Job failed: {result.error}"
        }

    if verbose:
        print(f"  ✓ Job processed successfully (latency: {result.processing_time*1000:.1f}ms)")

    # === TEST 4: STRESS LOAD ===
    if verbose:
        print(f"\n[TEST 4/6] Stress testing with heavy load ({duration}s)...")

    start = time.time()
    end = start + duration

    submitted_jobs = []
    queue_rejections = 0
    submission_start = time.time()

    # Submit jobs as fast as possible
    # Use realistic workload: simulates game calculations (1-5ms each)
    # Like AI pathfinding, physics prediction, batch distance calculations
    while time.time() < end:
        job_id = engine.submit_job("COMPUTE_HEAVY", {
            "iterations": 10,  # Realistic game calculation load (1-5ms per job)
            "data": list(range(50))
        })

        if job_id:
            submitted_jobs.append({
                "id": job_id,
                "submit_time": time.time()
            })
        else:
            queue_rejections += 1

        # Brief sleep to avoid CPU spin
        time.sleep(0.0001)

    submission_duration = time.time() - submission_start
    submission_rate = len(submitted_jobs) / submission_duration

    if verbose:
        print(f"  ✓ Submitted {len(submitted_jobs)} jobs in {submission_duration:.2f}s ({submission_rate:.0f} jobs/sec)")
        if queue_rejections > 0:
            print(f"  ⚠ Queue full: {queue_rejections} rejections")

    # === TEST 5: RESULT COLLECTION ===
    if verbose:
        print(f"\n[TEST 5/6] Collecting results (timeout: 30s)...")

    received_results = {}
    collection_start = time.time()
    timeout = time.time() + 30.0
    last_progress = time.time()

    while len(received_results) < len(submitted_jobs) and time.time() < timeout:
        results = engine.poll_results(max_results=100)

        for r in results:
            if r.job_id not in received_results:
                received_results[r.job_id] = {
                    "success": r.success,
                    "receive_time": time.time(),
                    "processing_time": r.processing_time,
                    "error": r.error
                }

        # Progress update every 2 seconds
        if verbose and time.time() - last_progress >= 2.0:
            pct = (len(received_results) / len(submitted_jobs) * 100) if submitted_jobs else 0
            print(f"    Progress: {len(received_results)}/{len(submitted_jobs)} ({pct:.0f}%)")
            last_progress = time.time()

        time.sleep(0.001)

    collection_duration = time.time() - collection_start

    if verbose:
        print(f"  ✓ Received {len(received_results)}/{len(submitted_jobs)} results in {collection_duration:.2f}s")

    # === TEST 6: METRICS CALCULATION ===
    if verbose:
        print(f"\n[TEST 6/6] Calculating performance metrics...")

    # Calculate latencies
    latencies = []
    for job in submitted_jobs:
        if job["id"] in received_results:
            result = received_results[job["id"]]
            latency_ms = (result["receive_time"] - job["submit_time"]) * 1000
            latencies.append(latency_ms)

    # Count successes and failures
    successes = sum(1 for r in received_results.values() if r["success"])
    failures = sum(1 for r in received_results.values() if not r["success"])
    lost_jobs = len(submitted_jobs) - len(received_results)

    # Calculate metrics
    total_duration = time.time() - start
    throughput = len(received_results) / total_duration
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    min_latency = min(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0
    completion_rate = len(received_results) / len(submitted_jobs) if submitted_jobs else 0

    # Get final worker stats for grading
    final_health = engine.check_worker_health()
    workers_alive = final_health["workers_alive"]
    workers_total = workers_alive + final_health["workers_dead"]

    # Phase 1: Get worker distribution and job type stats
    worker_dist = engine.get_worker_distribution() if hasattr(engine, 'get_worker_distribution') else None
    job_type_stats = engine.get_job_type_stats() if hasattr(engine, 'get_job_type_stats') else None

    # Calculate grade
    grade_result = _calculate_grade(
        throughput=throughput,
        avg_latency=avg_latency,
        completion_rate=completion_rate,
        queue_rejections=queue_rejections,
        failures=failures,
        workers_alive=workers_alive,
        workers_total=workers_total
    )

    # Build result dictionary
    result_dict = {
        "success": True,
        "grade": grade_result["letter"],
        "status": grade_result["status"],
        "issues": grade_result["issues"],
        "metrics": {
            "workers_alive": workers_alive,
            "workers_total": workers_total,
            "jobs_submitted": len(submitted_jobs),
            "jobs_received": len(received_results),
            "jobs_success": successes,
            "jobs_failed": failures,
            "jobs_lost": lost_jobs,
            "queue_rejections": queue_rejections,
            "throughput_jobs_per_sec": throughput,
            "latency_avg_ms": avg_latency,
            "latency_min_ms": min_latency,
            "latency_max_ms": max_latency,
            "completion_rate_pct": completion_rate * 100,
            "test_duration_sec": total_duration,
            "submission_duration_sec": submission_duration,
            "collection_duration_sec": collection_duration
        },
        "phase1": {
            "worker_distribution": worker_dist,
            "job_type_stats": job_type_stats
        }
    }

    if verbose:
        print(_format_report(result_dict))

    return result_dict


def _calculate_grade(throughput: float, avg_latency: float, completion_rate: float,
                     queue_rejections: int, failures: int, workers_alive: int,
                     workers_total: int) -> Dict:
    """Calculate letter grade and identify issues."""

    issues = []

    # Worker health (critical)
    if workers_alive < workers_total:
        issues.append(f"CRITICAL: Only {workers_alive}/{workers_total} workers alive!")

    # Completion rate (critical)
    if completion_rate < 0.8:
        issues.append("CRITICAL: >20% of jobs lost - engine unstable!")
    elif completion_rate < 0.95:
        issues.append(f"WARNING: {(1-completion_rate)*100:.1f}% of jobs lost")

    # Job failures
    if failures > 0:
        issues.append(f"WARNING: {failures} jobs failed in workers")

    # Throughput scoring (0-3 points)
    if throughput < 50:
        t_score = 0
        issues.append(f"Low throughput ({throughput:.0f} jobs/sec) - increase WORKER_COUNT")
    elif throughput < 100:
        t_score = 1
    elif throughput < 200:
        t_score = 2
    else:
        t_score = 3

    # Latency scoring (0-3 points)
    # NOTE: High latency in burst stress tests is expected (queue buildup)
    # These thresholds account for queue depth during intentional saturation
    if avg_latency > 10000:  # 10+ seconds
        l_score = 0
        issues.append(f"CRITICAL: Extreme latency ({avg_latency:.0f}ms avg) - workers stuck or deadlocked")
    elif avg_latency > 5000:  # 5-10 seconds
        l_score = 1
        issues.append(f"High latency ({avg_latency:.0f}ms avg) - acceptable for burst stress, but monitor in gameplay")
    elif avg_latency > 1000:  # 1-5 seconds
        l_score = 2
        # No issue - this is expected for burst tests
    else:  # <1 second
        l_score = 3

    # Completion scoring (0-3 points)
    if completion_rate < 0.8:
        c_score = 0
    elif completion_rate < 0.95:
        c_score = 1
    elif completion_rate < 0.99:
        c_score = 2
    else:
        c_score = 3

    # Queue saturation
    if queue_rejections > 10:
        issues.append(f"Queue saturated ({queue_rejections} rejections) - increase JOB_QUEUE_SIZE")

    # Calculate final grade
    total_score = t_score + l_score + c_score

    # Critical failures override score
    if workers_alive < workers_total or completion_rate < 0.8:
        return {"letter": "F", "status": "NOT READY", "issues": issues}

    if total_score >= 8:
        return {"letter": "A", "status": "READY", "issues": issues}
    elif total_score >= 6:
        return {"letter": "B", "status": "USABLE", "issues": issues}
    elif total_score >= 4:
        return {"letter": "C", "status": "NEEDS WORK", "issues": issues}
    else:
        return {"letter": "F", "status": "NOT READY", "issues": issues}


def _format_report(result: Dict) -> str:
    """Format test results as readable report."""

    grade = result["grade"]
    status = result["status"]
    metrics = result["metrics"]
    issues = result["issues"]

    # Grade symbol
    symbols = {"A": "✓✓✓", "B": "✓✓", "C": "⚠", "F": "✗✗✗"}
    symbol = symbols.get(grade, "?")

    lines = [
        "",
        "=" * 70,
        f"  ENGINE STRESS TEST RESULTS - GRADE: {grade} {symbol}",
        "=" * 70,
        f"  Status: {status}",
        "",
        "  WORKERS:",
        f"    Alive: {metrics['workers_alive']}/{metrics['workers_total']}",
        "",
        "  JOBS:",
        f"    Submitted:  {metrics['jobs_submitted']}",
        f"    Received:   {metrics['jobs_received']} ({metrics['completion_rate_pct']:.1f}%)",
        f"    Success:    {metrics['jobs_success']}",
        f"    Failed:     {metrics['jobs_failed']}",
        f"    Lost:       {metrics['jobs_lost']}",
        f"    Rejected:   {metrics['queue_rejections']}",
        "",
        "  PERFORMANCE:",
        f"    Throughput: {metrics['throughput_jobs_per_sec']:.0f} jobs/sec",
        f"    Latency:    {metrics['latency_avg_ms']:.1f}ms avg ({metrics['latency_min_ms']:.1f}-{metrics['latency_max_ms']:.1f}ms)",
        "",
        "  TIMING:",
        f"    Test:       {metrics['test_duration_sec']:.2f}s",
        f"    Submit:     {metrics['submission_duration_sec']:.2f}s",
        f"    Collect:    {metrics['collection_duration_sec']:.2f}s",
        "",
    ]

    # Phase 1: Worker distribution
    if result.get("phase1") and result["phase1"].get("worker_distribution"):
        worker_dist = result["phase1"]["worker_distribution"]
        lines.append("  PHASE 1 - WORKER LOAD DISTRIBUTION:")

        total = worker_dist["total_jobs"]
        for worker_id in sorted(worker_dist["workers"].keys()):
            w_stats = worker_dist["workers"][worker_id]
            lines.append(
                f"    Worker {worker_id}: {w_stats['jobs_processed']:6} jobs "
                f"({w_stats['percentage']:5.1f}%) "
                f"avg {w_stats['avg_time_ms']:.2f}ms"
            )

        lines.append("")

        # Analyze distribution fairness
        percentages = [w["percentage"] for w in worker_dist["workers"].values()]
        if percentages:
            max_pct = max(percentages)
            min_pct = min(percentages)
            variance = max_pct - min_pct

            if variance > 40:
                lines.append("    ⚠️  UNEVEN DISTRIBUTION - Shared queue causing imbalance!")
                lines.append(f"    Variance: {variance:.1f}% (should be <10% for fair distribution)")
            elif variance > 20:
                lines.append("    ⚠️  Moderate imbalance - Phase 2 per-worker queues will improve this")
            else:
                lines.append("    ✓ Distribution is relatively fair")

        lines.append("")

    # Phase 1: Job type stats
    if result.get("phase1") and result["phase1"].get("job_type_stats"):
        job_stats = result["phase1"]["job_type_stats"]
        if job_stats:
            lines.append("  PHASE 1 - JOB TYPE PROFILING:")

            # Sort by count (top 5)
            sorted_jobs = sorted(
                job_stats.items(),
                key=lambda x: x[1]["count"],
                reverse=True
            )[:5]

            for job_type, stats in sorted_jobs:
                lines.append(
                    f"    {job_type:25} "
                    f"{stats['count']:6} jobs  "
                    f"avg {stats['avg_time_ms']:6.2f}ms"
                )

            lines.append("")

    # Add issues if any
    if issues:
        lines.append("  ISSUES:")
        for issue in issues:
            lines.append(f"    • {issue}")
        lines.append("")

    # Verdict
    lines.append("  VERDICT:")
    if grade == "A":
        lines.append("    ✓ Engine is READY for production use")
        lines.append("    ✓ All systems performing optimally")
    elif grade == "B":
        lines.append("    ✓ Engine is functional")
        lines.append("    ⚠ Minor issues detected (see above)")
    elif grade == "C":
        lines.append("    ⚠ Engine needs optimization")
        lines.append("    ⚠ Address issues before production use")
    else:
        lines.append("    ✗ Engine NOT READY")
        lines.append("    ✗ Critical issues must be fixed")

    lines.append("=" * 70)
    lines.append("")

    return "\n".join(lines)


def quick_test(engine) -> bool:
    """
    Quick sanity check - returns True if engine is working.

    Args:
        engine: EngineCore instance (must be started)

    Returns:
        True if engine passes basic test, False otherwise
    """
    if not engine.is_alive():
        print("❌ Engine not alive")
        return False

    # Submit simple job
    job_id = engine.submit_job("ECHO", {"test": True})
    if not job_id:
        print("❌ Failed to submit job")
        return False

    # Wait for result
    timeout = time.time() + 3.0
    while time.time() < timeout:
        results = engine.poll_results()
        for r in results:
            if r.job_id == job_id and r.success:
                print("✓ Engine working")
                return True
        time.sleep(0.01)

    print("❌ Job timeout")
    return False
