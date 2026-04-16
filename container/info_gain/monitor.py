#!/usr/bin/env python3
"""
Terminal-based monitoring dashboard for orchestrated trials.

Displays real-time progress of exact and pose_aware planners running
in Docker containers.

Usage:
    python3 monitor.py
    python3 monitor.py --results-dir /path/to/results
    python3 monitor.py --watch  # Continuously refresh

Features:
    - Shows current trial/field progress
    - Displays compute metrics (CPU, GPU, RAM) for each planner
    - Shows sample counts and estimated completion
    - Color-coded status indicators
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BG_BLACK = "\033[40m"


def clear_screen():
    os.system('clear' if os.name != 'nt' else 'cls')


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def get_sample_count(trial_dir: Path) -> int:
    """Get current sample count from samples.csv."""
    samples_file = trial_dir / "samples.csv"
    if samples_file.exists():
        try:
            with open(samples_file, 'r') as f:
                return sum(1 for _ in f) - 1  # Subtract header
        except:
            pass
    return 0


def get_compute_metrics(trial_dir: Path) -> dict:
    """Get latest compute metrics from compute_metrics.csv."""
    metrics_file = trial_dir / "compute_metrics.csv"
    if not metrics_file.exists():
        return None

    try:
        with open(metrics_file, 'r') as f:
            lines = f.readlines()
            if len(lines) < 2:
                return None

            # Parse header and last line
            header = lines[0].strip().split(',')
            last = lines[-1].strip().split(',')

            return {
                header[i]: float(last[i]) if i > 0 else last[i]
                for i in range(len(header))
            }
    except:
        return None


def count_completed(results_dir: Path, planner: str) -> int:
    """Count completed trials (those with summary.json)."""
    planner_dir = results_dir / planner
    if not planner_dir.exists():
        return 0

    count = 0
    for field_dir in planner_dir.iterdir():
        if field_dir.is_dir():
            for trial_dir in field_dir.iterdir():
                if trial_dir.is_dir() and (trial_dir / "summary.json").exists():
                    count += 1
    return count


def draw_progress_bar(percent: float, width: int = 30) -> str:
    """Draw a progress bar."""
    filled = int(width * percent / 100)
    empty = width - filled
    bar = f"{'█' * filled}{'░' * empty}"
    return f"[{bar}] {percent:.1f}%"


def print_dashboard(results_dir: Path, status: dict):
    """Print the monitoring dashboard."""
    clear_screen()

    # Header
    print(f"{Colors.BOLD}{Colors.CYAN}╔{'═'*68}╗{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}║{'AQUATIC MAPPING - TRIAL ORCHESTRATOR':^68}║{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}╚{'═'*68}╝{Colors.RESET}")
    print()

    # Status info
    state = status.get('state', 'unknown')
    state_color = Colors.GREEN if state == 'running' else Colors.YELLOW if state == 'finished' else Colors.RED

    trial = status.get('current_trial', 0)
    total_trials = status.get('total_trials', 0)
    field = status.get('current_field', '')
    fields = status.get('fields', [])

    print(f"  {Colors.BOLD}Status:{Colors.RESET} {state_color}{state.upper()}{Colors.RESET}")
    print(f"  {Colors.BOLD}Trial:{Colors.RESET}  {trial}/{total_trials}")
    print(f"  {Colors.BOLD}Field:{Colors.RESET}  {field}")

    if 'elapsed_seconds' in status:
        elapsed = format_duration(status['elapsed_seconds'])
        print(f"  {Colors.BOLD}Elapsed:{Colors.RESET} {elapsed}")

    print()

    # Progress bars
    completed = status.get('completed', {})
    failed = status.get('failed', {})
    total_expected = total_trials * len(fields)

    print(f"  {Colors.BOLD}{'─'*66}{Colors.RESET}")
    print(f"  {Colors.BOLD}PLANNER PROGRESS{Colors.RESET}")
    print(f"  {Colors.BOLD}{'─'*66}{Colors.RESET}")

    for planner in ['exact', 'pose_aware']:
        comp = completed.get(planner, 0)
        fail = failed.get(planner, 0)
        pct = (comp / total_expected * 100) if total_expected > 0 else 0

        color = Colors.GREEN if fail == 0 else Colors.YELLOW
        bar = draw_progress_bar(pct, 35)

        print(f"  {color}{planner:12}{Colors.RESET} {bar}  {comp}/{total_expected} done, {fail} failed")

    print()

    # Current trial details
    if trial > 0 and field:
        print(f"  {Colors.BOLD}{'─'*66}{Colors.RESET}")
        print(f"  {Colors.BOLD}CURRENT FIELD: {field.upper()} (Trial {trial}){Colors.RESET}")
        print(f"  {Colors.BOLD}{'─'*66}{Colors.RESET}")

        for planner in ['exact', 'pose_aware']:
            trial_dir = results_dir / planner / field / f"trial_{trial:03d}"

            samples = get_sample_count(trial_dir)
            metrics = get_compute_metrics(trial_dir)

            has_summary = (trial_dir / "summary.json").exists()

            if has_summary:
                status_str = f"{Colors.GREEN}COMPLETE{Colors.RESET}"
            elif samples > 0:
                status_str = f"{Colors.YELLOW}RUNNING{Colors.RESET} ({samples}/100 samples)"
            else:
                status_str = f"{Colors.BLUE}STARTING{Colors.RESET}"

            print(f"  {Colors.BOLD}{planner:12}{Colors.RESET} {status_str}")

            if metrics:
                cpu = metrics.get('cpu_percent', 0)
                ram = metrics.get('ram_used_mb', 0)
                gpu = metrics.get('gpu_util', 0)
                gpu_mem = metrics.get('gpu_mem_used_mb', 0)
                gpu_temp = metrics.get('gpu_temp', 0)

                print(f"               CPU: {cpu:5.1f}%  RAM: {ram:6.0f}MB  GPU: {gpu:5.1f}% ({gpu_mem:.0f}MB, {gpu_temp:.0f}°C)")

    print()

    # Completed fields summary
    print(f"  {Colors.BOLD}{'─'*66}{Colors.RESET}")
    print(f"  {Colors.BOLD}COMPLETED TRIALS{Colors.RESET}")
    print(f"  {Colors.BOLD}{'─'*66}{Colors.RESET}")

    for planner in ['exact', 'pose_aware']:
        for field_name in fields:
            field_dir = results_dir / planner / field_name
            if not field_dir.exists():
                continue

            trial_count = 0
            for td in sorted(field_dir.iterdir()):
                if td.is_dir() and (td / "summary.json").exists():
                    trial_count += 1

            if trial_count > 0:
                pct = trial_count / total_trials * 100
                bar = '█' * trial_count + '░' * (total_trials - trial_count)
                print(f"  {planner:12} {field_name:18} [{bar}] {trial_count}/{total_trials}")

    # Footer
    print()
    print(f"  {Colors.BOLD}{'─'*66}{Colors.RESET}")
    print(f"  Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Results: {results_dir}")
    print(f"  {Colors.BOLD}{'─'*66}{Colors.RESET}")
    print()
    print(f"  Press Ctrl+C to exit")


def main():
    parser = argparse.ArgumentParser(description="Monitor orchestrated trials")
    parser.add_argument(
        "--results-dir", "-r",
        type=str,
        default=None,
        help="Path to results directory (auto-detected if not specified)"
    )
    parser.add_argument(
        "--watch", "-w",
        action="store_true",
        help="Continuously refresh display"
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=5,
        help="Refresh interval in seconds (default: 5)"
    )

    args = parser.parse_args()

    # Find results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        # Auto-detect
        script_dir = Path(__file__).parent.absolute()
        workspace = script_dir.parent.parent
        results_dir = workspace / "data" / "trials"

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    status_file = results_dir / "orchestrator_status.json"

    try:
        while True:
            # Read status
            if status_file.exists():
                try:
                    with open(status_file, 'r') as f:
                        status = json.load(f)
                except:
                    status = {"state": "unknown", "message": "Cannot read status file"}
            else:
                status = {
                    "state": "waiting",
                    "message": "Orchestrator not started",
                    "current_trial": 0,
                    "total_trials": 0,
                    "current_field": "",
                    "fields": [],
                    "completed": {"exact": 0, "pose_aware": 0},
                    "failed": {"exact": 0, "pose_aware": 0}
                }

            print_dashboard(results_dir, status)

            if not args.watch:
                break

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n  Monitoring stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
