#!/usr/bin/env python3
"""
Unified Analysis Runner

Auto-discovers all completed trials and runs:
  1. Per-trial planner graphs (trajectories, info gain, reconstruction, etc.)
  2. Cross-trial statistical comparison (paired t-tests, Cohen's d, publication figures)

Results are saved to existing locations:
  - Per-trial graphs:  reconstruction/results/trial_N/planner/
  - Statistics:         statistics/results/

Idempotent: running again overwrites previous results with updated data.

Usage:
    python3 run_analysis.py
"""

import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).parent.absolute()
TRIALS_DIR = WORKSPACE / "src" / "info_gain" / "data" / "trials"
RECON_VENV = WORKSPACE / "reconstruction" / "venv" / "bin" / "python3"


def discover_trials():
    """Find all completed trials (both exact and pose_aware have summary.json)."""
    fields = ["radial", "x_compress", "y_compress", "x_compress_tilt", "y_compress_tilt"]
    trial_nums = set()
    paired_count = 0

    for field in fields:
        exact_dir = TRIALS_DIR / "exact" / field
        pose_dir = TRIALS_DIR / "pose_aware" / field

        if not exact_dir.exists() or not pose_dir.exists():
            continue

        for trial_dir in sorted(exact_dir.glob("trial_*")):
            trial_name = trial_dir.name
            exact_summary = trial_dir / "summary.json"
            pose_summary = pose_dir / trial_name / "summary.json"

            if exact_summary.exists():
                num = int(trial_name.split("_")[-1])
                trial_nums.add(num)
                if pose_summary.exists():
                    paired_count += 1

    return sorted(trial_nums), paired_count


def run_planner_graphs(trial_nums):
    """Run analyze_planners.py for all trials."""
    script = WORKSPACE / "reconstruction" / "analyze_planners.py"
    python = str(RECON_VENV) if RECON_VENV.exists() else sys.executable

    print("=" * 70)
    print("STEP 1: Per-Trial Planner Graphs")
    print(f"  Script:  {script}")
    print(f"  Python:  {python}")
    print(f"  Trials:  {trial_nums}")
    print(f"  Output:  reconstruction/results/trial_N/planner/")
    print("=" * 70)

    trial_args = " ".join(str(t) for t in trial_nums)
    result = subprocess.run(
        [python, str(script), "--trials"] + [str(t) for t in trial_nums],
        cwd=str(WORKSPACE / "reconstruction"),
        capture_output=False,
    )

    if result.returncode != 0:
        print(f"\n[WARNING] analyze_planners.py exited with code {result.returncode}")
    else:
        print(f"\n[OK] Per-trial graphs complete")

    return result.returncode == 0


def run_statistics():
    """Run compare_planners.py for cross-trial statistics."""
    script = WORKSPACE / "statistics" / "compare_planners.py"
    python = str(RECON_VENV) if RECON_VENV.exists() else sys.executable

    print("\n" + "=" * 70)
    print("STEP 2: Cross-Trial Statistical Comparison")
    print(f"  Script:  {script}")
    print(f"  Python:  {python}")
    print(f"  Output:  statistics/results/")
    print("=" * 70)

    result = subprocess.run(
        [python, str(script)],
        cwd=str(WORKSPACE / "statistics"),
        capture_output=False,
    )

    if result.returncode != 0:
        print(f"\n[WARNING] compare_planners.py exited with code {result.returncode}")
    else:
        print(f"\n[OK] Statistical comparison complete")

    return result.returncode == 0


def main():
    print("=" * 70)
    print("  UNIFIED ANALYSIS RUNNER")
    print("=" * 70)

    # Discover trials
    trial_nums, paired_count = discover_trials()

    if not trial_nums:
        print("\n[ERROR] No completed trials found in:")
        print(f"  {TRIALS_DIR}")
        print("Run some trials first.")
        sys.exit(1)

    print(f"\nDiscovered {len(trial_nums)} trial(s): {trial_nums}")
    print(f"Paired comparisons (exact + pose_aware): {paired_count}")
    print()

    # Step 1: Per-trial graphs
    ok1 = run_planner_graphs(trial_nums)

    # Step 2: Statistics (needs at least some paired trials)
    if paired_count > 0:
        ok2 = run_statistics()
    else:
        print("\n[SKIP] Statistics: need at least 1 paired trial (both exact + pose_aware)")
        ok2 = True

    # Summary
    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)
    print(f"  Per-trial graphs:  {'OK' if ok1 else 'FAILED'}")
    print(f"  Statistics:        {'OK' if ok2 else 'FAILED'}")
    print()
    print(f"  Graphs:  {WORKSPACE / 'reconstruction' / 'results'}/")
    print(f"  Stats:   {WORKSPACE / 'statistics' / 'results'}/")
    print("=" * 70)

    sys.exit(0 if (ok1 and ok2) else 1)


if __name__ == "__main__":
    main()
