#!/usr/bin/env python3
"""
Trial Manager - Tkinter UI for orchestrator.

Shows worker status with live progress images, job queue, and logs.
Supports 2-6 concurrent worker containers.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import subprocess
import threading
import json
import time
import os
import webbrowser
from pathlib import Path
from datetime import datetime

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

SCRIPT_DIR = Path(__file__).parent.absolute()
WORKSPACE_ROOT = SCRIPT_DIR.parent.parent
RESULTS_DIR = WORKSPACE_ROOT / "data" / "trials"
ALL_FIELDS = ["radial", "x_compress", "y_compress", "x_compress_tilt", "y_compress_tilt"]
ALL_PLANNERS = [
    "exact", "pose_aware", "analytical",
    "nonstationary_exact", "nonstationary_pose_aware",
    "nonstationary_hotspot_exact", "nonstationary_hotspot_pose_aware",
]

# Short display names for UI
PLANNER_SHORT = {
    "exact": "exact",
    "pose_aware": "pose_aware",
    "analytical": "analytical",
    "nonstationary_exact": "ns_exact",
    "nonstationary_pose_aware": "ns_pose_aware",
    "nonstationary_hotspot_exact": "ns_hs_exact",
    "nonstationary_hotspot_pose_aware": "ns_hs_pose",
}

VNC_PORTS = {0: 6090, 1: 6091, 2: 6092, 3: 6093, 4: 6094, 5: 6095}

IMG_THUMB_SIZE = (380, 240)
IMG_REFRESH_MS = 5000


class TrialManager:
    def __init__(self, root):
        self.root = root
        self.root.title("Trial Manager")
        self.root.geometry("1200x950")

        self.running = False
        self.process = None
        self.worker_widgets = {}

        self._build_ui()
        self._start_status_monitor()
        if HAS_PIL:
            self.root.after(IMG_REFRESH_MS, self._refresh_worker_images)

    # ================================================================
    # UI CONSTRUCTION
    # ================================================================

    def _build_ui(self):
        # --- Row 1: Controls ---
        ctrl = ttk.Frame(self.root, padding=(10, 6))
        ctrl.pack(fill=tk.X)

        ttk.Label(ctrl, text="Start:").pack(side=tk.LEFT)
        self.start_trial_var = tk.StringVar(value="1")
        ttk.Entry(ctrl, textvariable=self.start_trial_var, width=5).pack(side=tk.LEFT, padx=2)

        ttk.Label(ctrl, text="End:").pack(side=tk.LEFT, padx=(8, 0))
        self.end_trial_var = tk.StringVar(value="10")
        ttk.Entry(ctrl, textvariable=self.end_trial_var, width=5).pack(side=tk.LEFT, padx=2)

        ttk.Label(ctrl, text="Workers:").pack(side=tk.LEFT, padx=(8, 0))
        self.workers_var = tk.StringVar(value="2")
        ttk.Combobox(ctrl, textvariable=self.workers_var,
                      values=["2", "4", "5", "6"], width=3,
                      state="readonly").pack(side=tk.LEFT, padx=2)

        ttk.Separator(ctrl, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        self.start_btn = ttk.Button(ctrl, text="START", command=self.start)
        self.start_btn.pack(side=tk.LEFT, padx=3)
        self.stop_btn = ttk.Button(ctrl, text="STOP", command=self.stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=3)
        self.retry_btn = ttk.Button(ctrl, text="RETRY FAILED", command=self.retry_failed)
        self.retry_btn.pack(side=tk.LEFT, padx=3)
        ttk.Button(ctrl, text="CLEAN", command=self.clean).pack(side=tk.LEFT, padx=3)

        # --- Row 2: Status summary ---
        status_bar = ttk.Frame(self.root, padding=(10, 2))
        status_bar.pack(fill=tk.X)

        self.state_label = ttk.Label(status_bar, text="IDLE", font=("Helvetica", 10, "bold"))
        self.state_label.pack(side=tk.LEFT, padx=(0, 15))
        self.jobs_label = ttk.Label(status_bar, text="Jobs: 0/0")
        self.jobs_label.pack(side=tk.LEFT, padx=(0, 15))
        self.failed_label = ttk.Label(status_bar, text="Failed: 0")
        self.failed_label.pack(side=tk.LEFT, padx=(0, 15))
        self.time_label = ttk.Label(status_bar, text="Time: 0:00")
        self.time_label.pack(side=tk.LEFT)

        ttk.Separator(self.root).pack(fill=tk.X, padx=10, pady=2)

        # --- Configuration: Fields + Planners ---
        config_frame = ttk.LabelFrame(self.root, text="Configuration", padding=5)
        config_frame.pack(fill=tk.X, padx=10, pady=(2, 4))

        # Fields row
        fields_row = ttk.Frame(config_frame)
        fields_row.pack(fill=tk.X)
        ttk.Label(fields_row, text="Fields:", width=8).pack(side=tk.LEFT)
        self.field_vars = {}
        for field in ALL_FIELDS:
            var = tk.BooleanVar(value=True)
            self.field_vars[field] = var
            ttk.Checkbutton(fields_row, text=field, variable=var).pack(side=tk.LEFT, padx=6)

        # Planners grid (3 rows)
        planner_row = ttk.Frame(config_frame)
        planner_row.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(planner_row, text="Planners:", width=8).pack(side=tk.LEFT, anchor=tk.N)

        planner_grid = ttk.Frame(planner_row)
        planner_grid.pack(side=tk.LEFT)

        self.planner_vars = {}
        for i, planner in enumerate(ALL_PLANNERS):
            var = tk.BooleanVar(value=True)
            self.planner_vars[planner] = var
            row = i // 3
            col = i % 3
            ttk.Checkbutton(planner_grid, text=PLANNER_SHORT[planner],
                            variable=var).grid(row=row, column=col, sticky=tk.W, padx=8, pady=1)

        ttk.Separator(self.root).pack(fill=tk.X, padx=10, pady=2)

        # --- Main area: Workers on top, Queue+Log on bottom ---
        main_pane = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=(2, 5))

        # Workers section
        workers_frame = ttk.LabelFrame(main_pane, text="Workers", padding=5)
        main_pane.add(workers_frame, weight=3)

        self.workers_container = ttk.Frame(workers_frame)
        self.workers_container.pack(fill=tk.BOTH, expand=True)

        # Create worker panels (up to 6, show/hide based on count)
        for slot in range(6):
            self._create_worker_panel(slot)

        # Bottom section: Queue + Log
        bottom_pane = ttk.PanedWindow(main_pane, orient=tk.HORIZONTAL)
        main_pane.add(bottom_pane, weight=2)

        # Job queue
        queue_frame = ttk.LabelFrame(bottom_pane, text="Job Queue", padding=5)
        bottom_pane.add(queue_frame, weight=1)

        self.queue_summary = ttk.Label(queue_frame, text="No jobs", font=("Helvetica", 9))
        self.queue_summary.pack(anchor=tk.W)

        tree_container = ttk.Frame(queue_frame)
        tree_container.pack(fill=tk.BOTH, expand=True)

        self.job_tree = ttk.Treeview(tree_container,
                                      columns=("planner", "field", "trial", "status"),
                                      show="headings", height=8)
        self.job_tree.heading("planner", text="Planner")
        self.job_tree.heading("field", text="Field")
        self.job_tree.heading("trial", text="Trial")
        self.job_tree.heading("status", text="Status")
        self.job_tree.column("planner", width=100)
        self.job_tree.column("field", width=100)
        self.job_tree.column("trial", width=60)
        self.job_tree.column("status", width=90)

        tree_scroll = ttk.Scrollbar(tree_container, orient=tk.VERTICAL,
                                     command=self.job_tree.yview)
        self.job_tree.configure(yscrollcommand=tree_scroll.set)
        self.job_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Log
        log_frame = ttk.LabelFrame(bottom_pane, text="Log", padding=5)
        bottom_pane.add(log_frame, weight=1)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=8,
                                                   font=("Courier", 9), wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _create_worker_panel(self, slot):
        """Create a worker panel with status text + image placeholder."""
        # 2-column grid: slots 0,1 in row 0; slots 2,3 in row 1; etc.
        row = slot // 2
        col = slot % 2

        frame = ttk.LabelFrame(self.workers_container, text=f"Worker {slot}", padding=4)
        frame.grid(row=row, column=col, sticky="nsew", padx=4, pady=3)

        # Make columns expand equally
        self.workers_container.columnconfigure(col, weight=1)
        self.workers_container.rowconfigure(row, weight=1)

        # Top line: job info
        info_lbl = ttk.Label(frame, text="(idle)", font=("Helvetica", 9, "bold"))
        info_lbl.pack(anchor=tk.W)

        # Progress row
        prog_frame = ttk.Frame(frame)
        prog_frame.pack(fill=tk.X, pady=2)

        samples_lbl = ttk.Label(prog_frame, text="--/100", width=8)
        samples_lbl.pack(side=tk.LEFT)
        progress_bar = ttk.Progressbar(prog_frame, length=120, mode='determinate')
        progress_bar.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)

        ttk.Button(prog_frame, text="VNC", width=4,
                    command=lambda s=slot: self.open_vnc(s)).pack(side=tk.LEFT, padx=2)
        ttk.Button(prog_frame, text="Kill", width=4,
                    command=lambda s=slot: self.kill_worker(s)).pack(side=tk.LEFT, padx=2)

        # Image area
        if HAS_PIL:
            img_label = ttk.Label(frame, text="No image yet", anchor=tk.CENTER,
                                   relief=tk.SUNKEN)
            img_label.pack(fill=tk.BOTH, expand=True, pady=(3, 0))
        else:
            img_label = ttk.Label(frame, text="(Pillow not installed — no preview)",
                                   anchor=tk.CENTER)
            img_label.pack(fill=tk.BOTH, expand=True, pady=(3, 0))

        self.worker_widgets[slot] = {
            "frame": frame,
            "info": info_lbl,
            "samples": samples_lbl,
            "progress": progress_bar,
            "img_label": img_label,
            "img_path": None,
        }

    # ================================================================
    # IMAGE REFRESH
    # ================================================================

    def _refresh_worker_images(self):
        """Periodically reload progress.png thumbnails for active workers."""
        if not HAS_PIL:
            return

        for slot, ww in self.worker_widgets.items():
            img_path = ww.get("img_path")
            if img_path and Path(img_path).exists():
                try:
                    img = Image.open(img_path)
                    img.thumbnail(IMG_THUMB_SIZE, Image.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    ww["img_label"].config(image=photo, text="")
                    ww["img_label"].image = photo  # prevent GC
                except Exception:
                    ww["img_label"].config(image='', text="Error loading image")
                    ww["img_label"].image = None
            else:
                ww["img_label"].config(image='', text="No image yet")
                ww["img_label"].image = None

        self.root.after(IMG_REFRESH_MS, self._refresh_worker_images)

    # ================================================================
    # HELPERS
    # ================================================================

    def log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{ts}] {msg}\n")
        self.log_text.see(tk.END)

    def open_vnc(self, slot):
        url = f"http://localhost:{VNC_PORTS[slot]}/vnc.html"
        self.log(f"Opening {url}")
        webbrowser.open(url)

    def kill_worker(self, slot):
        container_name = f"worker_{slot}"
        self.log(f"Killing worker_{slot}...")
        subprocess.run(["docker", "kill", container_name], capture_output=True)
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
        self.log(f"Worker {slot} killed — job will be retried")

    def get_fields(self):
        return [f for f, v in self.field_vars.items() if v.get()]

    def get_planners(self):
        return [p for p, v in self.planner_vars.items() if v.get()]

    def parse_job(self, job_str):
        if not job_str:
            return None, None, None
        parts = job_str.split("/")
        if len(parts) >= 3:
            return parts[0], parts[1], parts[2].replace("trial_", "")
        return None, None, None

    # ================================================================
    # START / STOP / RETRY
    # ================================================================

    def start(self):
        fields = self.get_fields()
        if not fields:
            messagebox.showerror("Error", "Select at least one field")
            return
        planners = self.get_planners()
        if not planners:
            messagebox.showerror("Error", "Select at least one planner")
            return
        try:
            start_trial = int(self.start_trial_var.get())
            end_trial = int(self.end_trial_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid trial numbers")
            return
        if start_trial < 1 or end_trial < start_trial:
            messagebox.showerror("Error", "Invalid trial range")
            return

        num_workers = int(self.workers_var.get())
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)

        self.log(f"Starting trials {start_trial}-{end_trial}, {num_workers} workers")
        self.log(f"  Fields: {', '.join(fields)}")
        self.log(f"  Planners: {', '.join(planners)}")

        thread = threading.Thread(
            target=self._run_orchestrator,
            args=(start_trial, end_trial, fields, planners, num_workers, False),
            daemon=True)
        thread.start()

    def retry_failed(self):
        if self.running:
            messagebox.showerror("Error", "Already running")
            return
        fields = self.get_fields()
        planners = self.get_planners()
        if not fields or not planners:
            messagebox.showerror("Error", "Select fields and planners")
            return
        try:
            start_trial = int(self.start_trial_var.get())
            end_trial = int(self.end_trial_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid trial numbers")
            return

        # Count missing
        missing = 0
        for t in range(start_trial, end_trial + 1):
            for field in fields:
                for planner in planners:
                    summary = RESULTS_DIR / planner / field / f"trial_{t:03d}" / "summary.json"
                    if not summary.exists():
                        missing += 1

        if missing == 0:
            messagebox.showinfo("Info", f"No missing trials in range {start_trial}-{end_trial}")
            return
        if not messagebox.askyesno("Confirm", f"Retry {missing} missing trials?"):
            return

        num_workers = int(self.workers_var.get())
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        self.retry_btn.config(state=tk.DISABLED)
        self.log(f"Retrying {missing} missing trials, {num_workers} workers")

        thread = threading.Thread(
            target=self._run_orchestrator,
            args=(start_trial, end_trial, fields, planners, num_workers, True),
            daemon=True)
        thread.start()

    def stop(self):
        if self.process:
            self.log("Stopping...")
            self.running = False
            self.process.terminate()
            for slot in range(6):
                subprocess.run(["docker", "stop", f"worker_{slot}"], capture_output=True)
                subprocess.run(["docker", "rm", "-f", f"worker_{slot}"], capture_output=True)
            self.log("Stopped")

    def _run_orchestrator(self, start_trial, end_trial, fields, planners, num_workers, retry_missing):
        cmd = [
            "python3", str(SCRIPT_DIR / "orchestrator.py"),
            "--start-trial", str(start_trial),
            "--end-trial", str(end_trial),
            "--fields", ",".join(fields),
            "--planners", ",".join(planners),
            "--workers", str(num_workers),
            "--workspace", str(WORKSPACE_ROOT),
        ]
        if retry_missing:
            cmd.append("--retry-missing")

        try:
            self.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1)
            self.root.after(0, lambda: self.stop_btn.config(state=tk.NORMAL))

            for line in iter(self.process.stdout.readline, ''):
                if not self.running:
                    break
                self.root.after(0, self.log, line.strip())
            self.process.wait()
        except Exception as e:
            self.root.after(0, self.log, f"Error: {e}")
        finally:
            self.root.after(0, self._on_done)

    def _on_done(self):
        self.running = False
        self.process = None
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.retry_btn.config(state=tk.NORMAL)
        self.log("Done")

    # ================================================================
    # STATUS MONITOR
    # ================================================================

    def _start_status_monitor(self):
        def monitor():
            while True:
                self._update_status()
                time.sleep(1)
        threading.Thread(target=monitor, daemon=True).start()

    def _update_status(self):
        status_file = RESULTS_DIR / "orchestrator_status.json"
        try:
            if not status_file.exists():
                self.root.after(0, lambda: self.state_label.config(text="IDLE"))
                return

            with open(status_file) as f:
                s = json.load(f)

            state = s.get("state", "unknown").upper()
            total = s.get("total_jobs", 0)
            done = s.get("completed_jobs", 0)
            failed = s.get("failed_jobs", 0)
            elapsed = s.get("elapsed_seconds", 0)
            workers = s.get("workers", {})

            mins = int(elapsed // 60)
            secs = int(elapsed % 60)

            self.root.after(0, lambda: self.state_label.config(text=state))
            self.root.after(0, lambda: self.jobs_label.config(text=f"Jobs: {done}/{total}"))
            self.root.after(0, lambda: self.failed_label.config(text=f"Failed: {failed}"))
            self.root.after(0, lambda: self.time_label.config(text=f"Time: {mins}:{secs:02d}"))

            # Update worker panels
            for slot in range(len(self.worker_widgets)):
                w = workers.get(str(slot), workers.get(slot, {}))
                ww = self.worker_widgets[slot]
                job = w.get("job")
                if job:
                    planner, field, trial = self.parse_job(job)
                    samples = w.get("samples", 0)
                    short_p = PLANNER_SHORT.get(planner, planner or "--")
                    trial_dir = f"trial_{int(trial):03d}" if trial else None
                    info_text = f"{short_p}  |  {field or '--'}  |  {trial_dir or '--'}"
                    img_path = str(RESULTS_DIR / planner / field / trial_dir / "figures" / "progress.png") if planner and field and trial_dir else None

                    self.root.after(0, lambda ww=ww, t=info_text: ww["info"].config(text=t))
                    self.root.after(0, lambda ww=ww, s=samples: ww["samples"].config(text=f"{s}/100"))
                    self.root.after(0, lambda ww=ww, s=samples: ww["progress"].config(value=s))
                    ww["img_path"] = img_path
                else:
                    self.root.after(0, lambda ww=ww: ww["info"].config(text="(idle)"))
                    self.root.after(0, lambda ww=ww: ww["samples"].config(text="--/100"))
                    self.root.after(0, lambda ww=ww: ww["progress"].config(value=0))
                    ww["img_path"] = None

            # Update job queue
            job_list = s.get("job_list", {})
            if job_list:
                self.root.after(0, self._update_job_tree, job_list)

        except Exception:
            pass

    def _update_job_tree(self, job_list):
        counts = {"pending": 0, "running": 0, "completed": 0, "failed": 0}
        for info in job_list.values():
            st = info.get("status", "pending")
            counts[st] = counts.get(st, 0) + 1

        self.queue_summary.config(
            text=f"Pending: {counts['pending']}  Running: {counts['running']}  "
                 f"Done: {counts['completed']}  Failed: {counts['failed']}")

        for item in self.job_tree.get_children():
            self.job_tree.delete(item)

        status_order = {"running": 0, "failed": 1, "pending": 2, "completed": 3}
        sorted_jobs = sorted(job_list.items(),
                              key=lambda x: (status_order.get(x[1].get("status", ""), 4), x[0]))

        for job_key, info in sorted_jobs:
            parts = job_key.split("/")
            if len(parts) < 3:
                continue
            planner = PLANNER_SHORT.get(parts[0], parts[0])
            field = parts[1]
            trial = parts[2].replace("trial_", "")
            status = info.get("status", "?")
            retries = info.get("retries", 0)
            display = status.upper()
            if retries > 0 and status != "completed":
                display += f" (r{retries})"

            self.job_tree.insert("", tk.END,
                                  values=(planner, field, trial, display),
                                  tags=(status,))

        self.job_tree.tag_configure("running", foreground="blue")
        self.job_tree.tag_configure("failed", foreground="red")
        self.job_tree.tag_configure("completed", foreground="gray")

    # ================================================================
    # CLEAN DATA
    # ================================================================

    def clean(self):
        if self.running:
            messagebox.showerror("Error", "Stop first")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Clean Trial Data")
        dialog.geometry("420x450")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="What to clean:", font=("Helvetica", 11, "bold")).pack(pady=(10, 5))

        mode_var = tk.StringVar(value="range")
        ttk.Radiobutton(dialog, text="Specific trial range",
                         variable=mode_var, value="range").pack(anchor=tk.W, padx=20)

        range_frame = ttk.Frame(dialog)
        range_frame.pack(fill=tk.X, padx=40, pady=5)
        ttk.Label(range_frame, text="Start:").pack(side=tk.LEFT)
        clean_start = tk.StringVar(value="1")
        ttk.Entry(range_frame, textvariable=clean_start, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(range_frame, text="End:").pack(side=tk.LEFT, padx=(10, 0))
        clean_end = tk.StringVar(value="10")
        ttk.Entry(range_frame, textvariable=clean_end, width=5).pack(side=tk.LEFT, padx=2)

        ttk.Radiobutton(dialog, text="All trial data (backup everything)",
                         variable=mode_var, value="all").pack(anchor=tk.W, padx=20, pady=(10, 0))

        ttk.Label(dialog, text="Planners:", font=("Helvetica", 10)).pack(anchor=tk.W, padx=20, pady=(15, 0))
        clean_planner_vars = {}
        planner_grid = ttk.Frame(dialog)
        planner_grid.pack(fill=tk.X, padx=40)
        for i, planner in enumerate(ALL_PLANNERS):
            var = tk.BooleanVar(value=True)
            clean_planner_vars[planner] = var
            ttk.Checkbutton(planner_grid, text=PLANNER_SHORT[planner],
                            variable=var).grid(row=i // 3, column=i % 3, sticky=tk.W, padx=6, pady=1)

        backup_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(dialog, text="Backup before deleting",
                         variable=backup_var).pack(anchor=tk.W, padx=20, pady=(10, 0))

        def do_clean():
            mode = mode_var.get()
            planners = [p for p, v in clean_planner_vars.items() if v.get()]
            if not planners:
                messagebox.showerror("Error", "Select at least one planner", parent=dialog)
                return

            if mode == "range":
                try:
                    s = int(clean_start.get())
                    e = int(clean_end.get())
                    if s < 1 or e < s:
                        messagebox.showerror("Error", "Invalid range", parent=dialog)
                        return
                    desc = f"trials {s}-{e} for {len(planners)} planners"
                except ValueError:
                    messagebox.showerror("Error", "Invalid trial numbers", parent=dialog)
                    return
            else:
                desc = f"ALL trials for {len(planners)} planners"
                s, e = None, None

            if not messagebox.askyesno("Confirm", f"Delete {desc}?", parent=dialog):
                return
            dialog.destroy()
            self._execute_clean(mode, planners, backup_var.get(), s, e)

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=15)
        ttk.Button(btn_frame, text="Clean", command=do_clean).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def _execute_clean(self, mode, planners, backup, start=None, end=None):
        try:
            uid = os.getuid()
            gid = os.getgid()

            if mode == "all" and backup:
                backup_name = f"trials_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                cmd = [
                    "docker", "run", "--rm",
                    "-v", f"{RESULTS_DIR.parent}:/data",
                    "ubuntu:24.04", "bash", "-c",
                    f"cd /data && cp -r trials {backup_name} && "
                    f"chown -R {uid}:{gid} {backup_name} 2>/dev/null || true"
                ]
                subprocess.run(cmd, capture_output=True)
                self.log(f"Backed up to {backup_name}")

            if mode == "all":
                for planner in planners:
                    cmd = [
                        "docker", "run", "--rm",
                        "-v", f"{RESULTS_DIR}:/data",
                        "ubuntu:24.04", "bash", "-c",
                        f"rm -rf /data/{planner}/* && "
                        f"chown -R {uid}:{gid} /data/{planner} 2>/dev/null || true"
                    ]
                    subprocess.run(cmd, capture_output=True)
                self.log(f"Cleaned all trials for {len(planners)} planners")
            else:
                rm_cmds = []
                for planner in planners:
                    for t in range(start, end + 1):
                        trial_str = f"trial_{t:03d}"
                        for field in ALL_FIELDS:
                            rm_cmds.append(f"rm -rf /data/{planner}/{field}/{trial_str}")

                if backup:
                    backup_name = f"trials_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}_t{start}-{end}"
                    cp_cmds = []
                    for planner in planners:
                        for t in range(start, end + 1):
                            for field in ALL_FIELDS:
                                src = f"/data/{planner}/{field}/trial_{t:03d}"
                                dst = f"/backup/{planner}/{field}/"
                                cp_cmds.append(f"mkdir -p {dst} && cp -r {src} {dst} 2>/dev/null || true")
                    cmd = [
                        "docker", "run", "--rm",
                        "-v", f"{RESULTS_DIR}:/data",
                        "-v", f"{RESULTS_DIR.parent / backup_name}:/backup",
                        "ubuntu:24.04", "bash", "-c",
                        " && ".join(cp_cmds) + f" && chown -R {uid}:{gid} /backup 2>/dev/null || true"
                    ]
                    subprocess.run(cmd, capture_output=True)
                    self.log(f"Backed up trials {start}-{end}")

                if rm_cmds:
                    cmd = [
                        "docker", "run", "--rm",
                        "-v", f"{RESULTS_DIR}:/data",
                        "ubuntu:24.04", "bash", "-c",
                        " && ".join(rm_cmds)
                    ]
                    subprocess.run(cmd, capture_output=True)
                self.log(f"Cleaned trials {start}-{end} for {len(planners)} planners")

        except Exception as e:
            self.log(f"Error cleaning: {e}")


def main():
    root = tk.Tk()
    app = TrialManager(root)

    def on_close():
        if app.running:
            if messagebox.askyesno("Confirm", "Stop and exit?"):
                app.stop()
                root.destroy()
        else:
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
