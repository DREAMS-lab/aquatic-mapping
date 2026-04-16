#!/usr/bin/env python3
"""
Host-side orchestrator with work-queue architecture.

Runs 2, 4, 5, or 6 containers continuously - when one finishes, immediately starts next job.
No waiting for paired containers. Maximum throughput.

Usage:
    python3 orchestrator.py --start-trial 1 --end-trial 10
    python3 orchestrator.py --start-trial 11 --end-trial 20 --fields radial,x_compress
    python3 orchestrator.py --trials 10  # legacy: runs trials 1-10
"""

import argparse
import subprocess
import time
import json
import os
import sys
import signal
from pathlib import Path
from datetime import datetime
from queue import Queue
from threading import Thread, Lock
import threading

# Configuration
ALL_FIELDS = ["radial", "x_compress", "y_compress", "x_compress_tilt", "y_compress_tilt"]
PLANNERS = ["exact", "pose_aware", "analytical", "nonstationary_exact", "nonstationary_pose_aware",
            "nonstationary_hotspot_exact", "nonstationary_hotspot_pose_aware"]

IMAGE_NAME = os.environ.get("AQUATIC_IMAGE", "aquatic-sim")
CONTAINER_TIMEOUT = 3600  # 1 hour max per job
MAX_SAMPLES = 100
DEFAULT_WORKERS = 2

# Fixed VNC ports per worker slot
VNC_PORTS = {
    0: (5902, 6090),  # Worker 0
    1: (5903, 6091),  # Worker 1
    2: (5904, 6092),  # Worker 2
    3: (5905, 6093),  # Worker 3
    4: (5906, 6094),  # Worker 4
    5: (5907, 6095),  # Worker 5
}

SCRIPT_DIR = Path(__file__).parent.absolute()
WORKSPACE_ROOT = SCRIPT_DIR.parent.parent


class Orchestrator:
    def __init__(self, start_trial: int, end_trial: int, fields: list, workspace_path: Path, max_workers: int = DEFAULT_WORKERS, retry_missing: bool = False, planners: list = None, uncertainty_scale: float = None):
        self.start_trial = start_trial
        self.end_trial = end_trial
        self.num_trials = end_trial - start_trial + 1
        self.fields = fields
        self.workspace = workspace_path
        self.max_workers = max_workers
        self.retry_missing = retry_missing
        self.planners = planners if planners else PLANNERS
        self.uncertainty_scale = uncertainty_scale
        self.results_dir = workspace_path / "data" / "trials"
        self.status_file = self.results_dir / "orchestrator_status.json"
        self.log_file = self.results_dir / "orchestrator.log"

        # Job queue: (planner, field, trial)
        self.job_queue = Queue()
        self.total_jobs = 0
        self.completed_jobs = 0
        self.failed_jobs = 0

        # Worker state
        self.workers = {}  # slot -> {job, container_name, start_time, samples}
        self.workers_lock = Lock()

        # Lock for counters and job_history (thread-safe updates)
        self.counters_lock = Lock()

        # Lock for status file writes
        self.status_lock = Lock()

        # Stats (dynamic based on selected planners)
        self.completed = {p: 0 for p in self.planners}
        self.failed = {p: 0 for p in self.planners}
        self.start_time = datetime.now()
        self.running = True

        # Job tracking: key = "planner/field/trial_NNN" -> status
        self.job_tracker = {}  # job_key -> {"status": pending/running/completed/failed, "retries": N}
        self.job_tracker_lock = Lock()
        self.max_retries = None  # Infinite retries - keep trying until success

        # Job history for logging
        self.job_history = []  # [(timestamp, event, job, elapsed, details)]

        # Setup
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        self.results_dir.mkdir(parents=True, exist_ok=True)
        for planner in self.planners:
            for field in self.fields:
                (self.results_dir / planner / field).mkdir(parents=True, exist_ok=True)

        # Clear log file
        with open(self.log_file, "w") as f:
            f.write("")

    def _handle_signal(self, signum, frame):
        self.log(f"SIGNAL {signum} received - shutting down")
        self.running = False
        self._stop_all_containers()
        sys.exit(1)

    def log(self, message: str):
        """Log with timestamp to console and file."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {message}"
        print(line)
        with open(self.log_file, "a") as f:
            f.write(line + "\n")

    def log_event(self, event: str, planner: str, field: str, trial: int,
                  elapsed: float = 0, details: str = ""):
        """Log structured event."""
        ts = datetime.now().strftime("%H:%M:%S")
        job_str = f"{planner}/{field}/trial_{trial:03d}"

        if event == "START":
            self.log(f">>> START  {job_str}")
        elif event == "DONE":
            self.log(f"<<< DONE   {job_str} ({elapsed:.1f}s) {details}")
        elif event == "FAIL":
            self.log(f"!!! FAIL   {job_str} ({elapsed:.1f}s) {details}")
        else:
            self.log(f"    {event:6} {job_str} {details}")

        with self.counters_lock:
            self.job_history.append({
                "timestamp": datetime.now().isoformat(),
                "event": event,
                "planner": planner,
                "field": field,
                "trial": trial,
                "elapsed": elapsed,
                "details": details
            })

    def update_status(self):
        """Update status file for UI."""
        with self.workers_lock:
            workers_copy = {
                k: {
                    "job": v.get("job"),
                    "container": v.get("container_name"),
                    "samples": v.get("samples", 0),
                    "start_time": v.get("start_time").isoformat() if v.get("start_time") else None,
                    "elapsed": (datetime.now() - v["start_time"]).total_seconds() if v.get("start_time") else 0
                }
                for k, v in self.workers.items()
            }

        with self.job_tracker_lock:
            job_list = dict(self.job_tracker)

        with self.counters_lock:
            status = {
                "state": "running" if self.running else "stopped",
                "total_jobs": self.total_jobs,
                "completed_jobs": self.completed_jobs,
                "failed_jobs": self.failed_jobs,
                "jobs_remaining": self.job_queue.qsize(),
                "workers": workers_copy,
                "completed": dict(self.completed),
                "failed": dict(self.failed),
                "fields": self.fields,
                "start_trial": self.start_trial,
                "end_trial": self.end_trial,
                "num_trials": self.num_trials,
                "max_workers": self.max_workers,
                "start_time": self.start_time.isoformat(),
                "elapsed_seconds": (datetime.now() - self.start_time).total_seconds(),
                "updated": datetime.now().isoformat(),
                "job_list": job_list,
            }

        # Atomic write: write to temp file then rename (locked to prevent races)
        with self.status_lock:
            tmp_file = self.status_file.with_suffix('.tmp')
            with open(tmp_file, "w") as f:
                json.dump(status, f, indent=2)
            os.replace(tmp_file, self.status_file)

    def _stop_all_containers(self):
        """Stop all worker containers."""
        for slot in range(self.max_workers):
            name = f"worker_{slot}"
            subprocess.run(["docker", "stop", name], capture_output=True)
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)

    def get_sample_count(self, trial_dir: Path) -> int:
        """Count samples from planner.log (samples.csv only written at end)."""
        log_file = trial_dir / "planner.log"
        if not log_file.exists():
            return 0
        try:
            import subprocess
            result = subprocess.run(
                ["grep", "-c", "Sample.*info=", str(log_file)],
                capture_output=True, text=True
            )
            return int(result.stdout.strip()) if result.returncode == 0 else 0
        except:
            return 0

    def _job_key(self, planner, field, trial):
        return f"{planner}/{field}/trial_{trial:03d}"

    def run_job(self, slot: int, planner: str, field: str, trial: int):
        """Run a single job in a container."""
        container_name = f"worker_{slot}"
        job_key = self._job_key(planner, field, trial)
        trial_dir = self.results_dir / planner / field / f"trial_{trial:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        chown_path = f"/workspace/{trial_dir.relative_to(self.workspace)}"

        # Update job tracker
        with self.job_tracker_lock:
            self.job_tracker[job_key]["status"] = "running"

        # Force remove existing container + wait to avoid rc=125
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
        time.sleep(1)

        vnc_port, novnc_port = VNC_PORTS[slot]

        # Update worker state
        with self.workers_lock:
            self.workers[slot] = {
                "job": f"{planner}/{field}/{trial}",
                "container_name": container_name,
                "start_time": datetime.now(),
                "samples": 0,
                "planner": planner,
                "field": field,
                "trial": trial,
            }

        self.log_event("START", planner, field, trial)
        self.update_status()

        # CRITICAL: ROS_DOMAIN_ID and GZ_PARTITION isolate each container's
        # ROS2 and Gazebo networks. Without this, containers on the same
        # Docker network will discover each other's topics and interfere!
        ros_domain_id = slot + 10
        gz_partition = f"worker_{slot}"

        cmd = [
            "docker", "run",
            "--name", container_name,
            "--rm",
            "--gpus", "all",
            "--user", "root",
            "-e", f"PLANNER_TYPE={planner}",
            "-e", "WORKSPACE_DIR=/home/simuser/aquatic-mapping",
            "-e", "PX4_DIR=/home/simuser/PX4-Autopilot",
            "-e", f"ROS_DOMAIN_ID={ros_domain_id}",
            "-e", f"GZ_PARTITION={gz_partition}",
            *(["-e", f"UNCERTAINTY_SCALE={self.uncertainty_scale}"] if self.uncertainty_scale is not None else []),
            "-p", f"{novnc_port}:{novnc_port}",
            "-p", f"{vnc_port}:{vnc_port}",
            "-v", f"{self.workspace}:/home/simuser/aquatic-mapping:rw",
            "-v", f"{SCRIPT_DIR}/run_single_field.sh:/tmp/run_single_field.sh:ro",
            "--entrypoint", "/bin/bash",
            IMAGE_NAME,
            "-c", f"/tmp/run_single_field.sh {planner} {field} {trial} {slot}"
        ]

        start_time = time.time()

        # Start sample monitor thread with auto-kill for stuck jobs
        monitor_running = threading.Event()
        monitor_running.set()
        process_container = [None]  # Mutable reference for container process

        def monitor_samples():
            stuck_start = None  # When did we first notice 0 samples?
            STUCK_TIMEOUT = 180  # Kill if 0 samples for 3 minutes

            while monitor_running.is_set() and self.running:
                samples = self.get_sample_count(trial_dir)
                with self.workers_lock:
                    if slot in self.workers:
                        self.workers[slot]["samples"] = samples
                self.update_status()

                # Auto-kill stuck jobs (0 samples after initial period)
                elapsed = time.time() - start_time
                if elapsed > 60:  # Wait 60s for initialization
                    if samples == 0:
                        if stuck_start is None:
                            stuck_start = time.time()
                        elif time.time() - stuck_start > STUCK_TIMEOUT:
                            self.log(f"Worker {slot}: STUCK at 0 samples for {STUCK_TIMEOUT}s - auto-killing")
                            subprocess.run(["docker", "kill", container_name], capture_output=True)
                            monitor_running.clear()
                            break
                    else:
                        stuck_start = None  # Reset if samples start coming in

                time.sleep(2)

        monitor = Thread(target=monitor_samples, daemon=True)
        monitor.start()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=CONTAINER_TIMEOUT
            )

            elapsed = time.time() - start_time
            success = result.returncode == 0 and (trial_dir / "summary.json").exists()

            # Save container log
            with open(trial_dir / "container.log", "w") as f:
                f.write(f"=== STDOUT ===\n{result.stdout}\n")
                f.write(f"=== STDERR ===\n{result.stderr}\n")
                f.write(f"=== Return code: {result.returncode} ===\n")

            # Fix ownership of all files created by container (run as root)
            host_uid = os.getuid()
            host_gid = os.getgid()
            subprocess.run(
                ["docker", "run", "--rm", "-v", f"{self.workspace}:/workspace",
                 "ubuntu:24.04", "chown", "-R", f"{host_uid}:{host_gid}",
                 chown_path],
                capture_output=True
            )

            if success:
                with self.counters_lock:
                    self.completed[planner] += 1
                    self.completed_jobs += 1
                with self.job_tracker_lock:
                    self.job_tracker[job_key]["status"] = "completed"
                samples = self.get_sample_count(trial_dir)
                self.log_event("DONE", planner, field, trial, elapsed, f"samples={samples}")
            else:
                self._handle_job_failure(planner, field, trial, elapsed, f"rc={result.returncode}")

            return success

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            subprocess.run(["docker", "kill", container_name], capture_output=True)
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
            time.sleep(2)  # Wait for container cleanup to prevent rc=125 cascade

            # Fix ownership even on timeout
            host_uid = os.getuid()
            host_gid = os.getgid()
            subprocess.run(
                ["docker", "run", "--rm", "-v", f"{self.workspace}:/workspace",
                 "ubuntu:24.04", "chown", "-R", f"{host_uid}:{host_gid}",
                 chown_path],
                capture_output=True
            )

            self._handle_job_failure(planner, field, trial, elapsed, "TIMEOUT")
            return False

        except Exception as e:
            elapsed = time.time() - start_time

            # Fix ownership even on exception
            host_uid = os.getuid()
            host_gid = os.getgid()
            subprocess.run(
                ["docker", "run", "--rm", "-v", f"{self.workspace}:/workspace",
                 "ubuntu:24.04", "chown", "-R", f"{host_uid}:{host_gid}",
                 chown_path],
                capture_output=True
            )

            self._handle_job_failure(planner, field, trial, elapsed, str(e))
            return False

        finally:
            monitor_running.clear()
            with self.workers_lock:
                if slot in self.workers:
                    self.workers[slot] = {"job": None, "container_name": None, "samples": 0}
            self.update_status()

    def _handle_job_failure(self, planner, field, trial, elapsed, details):
        """Handle job failure with auto-retry (infinite retries)."""
        job_key = self._job_key(planner, field, trial)
        with self.job_tracker_lock:
            retries = self.job_tracker[job_key].get("retries", 0)

        # Always re-queue for retry (infinite retries)
        with self.job_tracker_lock:
            self.job_tracker[job_key]["retries"] = retries + 1
            self.job_tracker[job_key]["status"] = "pending"
        self.job_queue.put((planner, field, trial))
        self.log_event("FAIL", planner, field, trial, elapsed, f"{details} -> RETRY {retries + 1}")

    def worker_loop(self, slot: int):
        """Worker thread that continuously processes jobs."""
        self.log(f"Worker {slot} started (VNC: localhost:{VNC_PORTS[slot][1]})")

        while self.running:
            try:
                # Get next job (non-blocking with timeout)
                job = self.job_queue.get(timeout=1)
            except:
                continue

            if job is None:  # Poison pill
                break

            planner, field, trial = job
            self.run_job(slot, planner, field, trial)
            self.job_queue.task_done()

        self.log(f"Worker {slot} stopped")

    def run(self):
        """Main orchestration loop."""
        self.log("=" * 60)
        self.log("  ORCHESTRATOR - WORK QUEUE MODE")
        self.log("=" * 60)
        self.log(f"  Trials:    {self.start_trial}-{self.end_trial} ({self.num_trials} total)")
        self.log(f"  Fields:    {', '.join(self.fields)}")
        self.log(f"  Planners:  {', '.join(self.planners)}")
        self.log(f"  Workers:   {self.max_workers} (rolling - no waiting)")
        self.log(f"  Workspace: {self.workspace}")
        self.log("=" * 60)
        self.log("  VNC Access:")
        for slot in range(self.max_workers):
            vnc, novnc = VNC_PORTS[slot]
            self.log(f"    Worker {slot}: http://localhost:{novnc}/vnc.html")
        self.log("=" * 60)

        # Build job queue - all combinations of planner × field × trial
        jobs = []
        skipped = 0
        for trial in range(self.start_trial, self.end_trial + 1):
            for field in self.fields:
                for planner in self.planners:
                    if self.retry_missing:
                        summary = self.results_dir / planner / field / f"trial_{trial:03d}" / "summary.json"
                        if summary.exists():
                            skipped += 1
                            continue
                    jobs.append((planner, field, trial))

        self.total_jobs = len(jobs)
        for job in jobs:
            planner, field, trial = job
            job_key = self._job_key(planner, field, trial)
            with self.job_tracker_lock:
                self.job_tracker[job_key] = {"status": "pending", "retries": 0}
            self.job_queue.put(job)

        if self.retry_missing and skipped > 0:
            self.log(f"  Mode:      RETRY MISSING ({skipped} already complete, {self.total_jobs} to run)")
        self.log(f"Queued {self.total_jobs} jobs")
        self.log("")

        self.update_status()

        # Start worker threads
        workers = []
        for slot in range(self.max_workers):
            t = Thread(target=self.worker_loop, args=(slot,), daemon=True)
            t.start()
            workers.append(t)

        # Wait for all jobs to complete (queue empty + no running jobs)
        try:
            while self.running:
                with self.counters_lock:
                    done = self.completed_jobs + self.failed_jobs >= self.total_jobs
                # Also check queue is empty (retries add to queue but not total_jobs)
                if done and self.job_queue.empty():
                    # Double-check no workers are still running
                    with self.workers_lock:
                        active = any(w.get("job") for w in self.workers.values())
                    if not active:
                        break
                self.update_status()
                time.sleep(2)
        except KeyboardInterrupt:
            self.running = False

        # Signal workers to stop
        for _ in range(self.max_workers):
            self.job_queue.put(None)

        for t in workers:
            t.join(timeout=5)

        self._stop_all_containers()

        # Final summary
        total_elapsed = (datetime.now() - self.start_time).total_seconds()
        self.log("")
        self.log("=" * 60)
        self.log("  COMPLETE")
        self.log("=" * 60)
        self.log(f"  Total time:  {total_elapsed/60:.1f} minutes")
        self.log(f"  Completed:   {self.completed_jobs}/{self.total_jobs}")
        self.log(f"  Failed:      {self.failed_jobs}")
        for planner in self.planners:
            self.log(f"  {planner:12} {self.completed[planner]} done, {self.failed[planner]} failed")
        if self.completed_jobs > 0:
            avg = total_elapsed / self.completed_jobs
            self.log(f"  Avg/job:     {avg:.1f}s ({avg/60:.1f}min)")
        self.log("=" * 60)

        # Save job history
        with open(self.results_dir / "job_history.json", "w") as f:
            json.dump(self.job_history, f, indent=2)

        self.update_status()
        return self.failed_jobs == 0


def main():
    parser = argparse.ArgumentParser(description="Work-queue orchestrator for trials")
    parser.add_argument("--start-trial", type=int, default=1, help="First trial number")
    parser.add_argument("--end-trial", type=int, default=10, help="Last trial number")
    parser.add_argument("--trials", "-t", type=int, default=None, help="Number of trials (legacy, overrides start/end)")
    parser.add_argument("--fields", "-f", type=str, default=None, help="Comma-separated fields")
    parser.add_argument("--skip-fields", type=str, default=None, help="Fields to skip")
    parser.add_argument("--workspace", "-w", type=str, default=str(WORKSPACE_ROOT))
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, choices=[2, 4, 5, 6], help="Number of worker containers (2, 4, 5, or 6)")
    parser.add_argument("--planners", type=str, default=None, help="Comma-separated planners (exact,pose_aware,analytical)")
    parser.add_argument("--retry-missing", action="store_true", help="Only run trials missing summary.json")
    # Keep --parallel for compatibility but ignore it
    parser.add_argument("--parallel", "-p", type=int, default=1, help="(ignored)")
    parser.add_argument("--uncertainty-scale", type=float, default=None, help="EKF covariance multiplier for pose-aware planners")

    args = parser.parse_args()

    # Handle legacy --trials argument (overrides start/end)
    if args.trials is not None:
        start_trial = 1
        end_trial = args.trials
    else:
        start_trial = args.start_trial
        end_trial = args.end_trial

    if start_trial < 1:
        print("ERROR: Start trial must be >= 1")
        sys.exit(1)
    if end_trial < start_trial:
        print("ERROR: End trial must be >= start trial")
        sys.exit(1)

    if args.fields:
        fields = [f.strip() for f in args.fields.split(",")]
        for f in fields:
            if f not in ALL_FIELDS:
                print(f"ERROR: Unknown field '{f}'")
                sys.exit(1)
    else:
        fields = ALL_FIELDS.copy()

    if args.skip_fields:
        skip = [f.strip() for f in args.skip_fields.split(",")]
        fields = [f for f in fields if f not in skip]

    if not fields:
        print("ERROR: No fields!")
        sys.exit(1)

    # Parse planners
    if args.planners:
        planners = [p.strip() for p in args.planners.split(",")]
        for p in planners:
            if p not in PLANNERS:
                print(f"ERROR: Unknown planner '{p}'. Available: {', '.join(PLANNERS)}")
                sys.exit(1)
    else:
        planners = None  # Use default

    orch = Orchestrator(start_trial, end_trial, fields, Path(args.workspace), max_workers=args.workers, retry_missing=args.retry_missing, planners=planners, uncertainty_scale=args.uncertainty_scale)
    success = orch.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
