#!/usr/bin/env python3
"""
Aquatic Mapping Simulation Control Panel
Backend API Server
"""

from fastapi import FastAPI, HTTPException, Depends, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
import zipfile
import tempfile
import shutil
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
import secrets
import docker
import asyncio
import json
import subprocess
import os
from pathlib import Path
from datetime import datetime
from typing import Optional
from pydantic import BaseModel
import re

# Get absolute paths
BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "frontend" / "templates"
STATIC_DIR = BASE_DIR / "frontend" / "static"

app = FastAPI(title="Aquatic Mapping Control Panel")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates and static files
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Environment detection
IS_PRODUCTION = os.environ.get("RAILWAY_ENVIRONMENT") or os.environ.get("RENDER") or os.environ.get("HEROKU")
HOST_PC_URL = os.environ.get("HOST_PC_URL", "http://localhost:8000")  # URL to local PC when deployed

# Simple auth - CHANGE THIS PASSWORD!
USERNAME = os.environ.get("SIM_USERNAME", "bakin")
PASSWORD = os.environ.get("SIM_PASSWORD", "ozhugu")

security = HTTPBasic()

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, USERNAME)
    correct_password = secrets.compare_digest(credentials.password, PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# Docker client (only available when running on host PC)
if IS_PRODUCTION:
    docker_client = None
    print("ℹ️  Running in production mode - Docker features disabled")
    print("ℹ️  Connect to host PC to enable container management")
else:
    try:
        docker_client = docker.from_env()
        docker_client.ping()  # Test connection
        print("✓ Docker client connected")
    except Exception as e:
        print(f"✗ Docker client failed: {e}")
        docker_client = None

# Data models
class BatchRequest(BaseModel):
    start_trial: int
    end_trial: int
    concurrent: int = 3

class TrialAction(BaseModel):
    trial_id: int
    action: str  # start, stop, remove

class ReconstructionRequest(BaseModel):
    field: str = "all"  # radial, x_compress, y_compress, x_compress_tilt, y_compress_tilt, all
    method: str = "all"  # standard, mchutchon, girard, all
    kernel: str = "all"  # rbf, exponential, matern15, matern25, all

# Store for running reconstructions
reconstruction_processes = {}

# ============================================================================
# Batch Manager - Continuous batch execution with auto-start
# ============================================================================
class BatchManager:
    def __init__(self):
        self.active_batch = None
        self.pending_trials = []
        self.completed_trials = []
        self.failed_trials = []
        self.concurrent_limit = 3
        self.running = False
        self._task = None

    def start_batch(self, start_trial: int, end_trial: int, concurrent: int):
        """Initialize a new batch job"""
        self.pending_trials = list(range(start_trial, end_trial + 1))
        self.completed_trials = []
        self.failed_trials = []
        self.concurrent_limit = concurrent
        self.active_batch = {
            "start_trial": start_trial,
            "end_trial": end_trial,
            "concurrent": concurrent,
            "total": end_trial - start_trial + 1,
            "started_at": datetime.now().isoformat()
        }
        self.running = True

    def stop_batch(self):
        """Cancel the active batch"""
        self.running = False
        self.pending_trials = []
        self.active_batch = None

    def get_status(self):
        """Get current batch status"""
        if not self.active_batch:
            return {"active": False}

        return {
            "active": self.running,
            "batch": self.active_batch,
            "pending": len(self.pending_trials),
            "pending_trials": self.pending_trials[:10],  # First 10 for display
            "completed": len(self.completed_trials),
            "completed_trials": self.completed_trials,
            "failed": len(self.failed_trials),
            "failed_trials": self.failed_trials,
            "progress": round((len(self.completed_trials) + len(self.failed_trials)) / self.active_batch["total"] * 100, 1) if self.active_batch else 0
        }

    def mark_completed(self, trial_id: int):
        """Mark a trial as completed"""
        if trial_id in self.pending_trials:
            self.pending_trials.remove(trial_id)
        if trial_id not in self.completed_trials:
            self.completed_trials.append(trial_id)

    def mark_failed(self, trial_id: int):
        """Mark a trial as failed"""
        if trial_id in self.pending_trials:
            self.pending_trials.remove(trial_id)
        if trial_id not in self.failed_trials:
            self.failed_trials.append(trial_id)

batch_manager = BatchManager()

# Store for active connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

# ============================================================================
# API Routes
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    """Public landing page - no auth required"""
    return templates.TemplateResponse("landing.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, username: str = Depends(verify_credentials)):
    """Protected dashboard - requires auth"""
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "username": username
    })

@app.get("/api/status")
async def get_status(username: str = Depends(verify_credentials)):
    """Get overall system and container status"""
    containers = get_simulation_containers()
    system = get_system_stats()
    return {
        "containers": containers,
        "system": system,
        "timestamp": datetime.now().isoformat(),
        "is_production": IS_PRODUCTION is not None,
        "host_pc_connected": docker_client is not None
    }

@app.get("/api/containers")
async def get_containers(username: str = Depends(verify_credentials)):
    """Get all simulation containers"""
    return get_simulation_containers()

@app.post("/api/trial/start/{trial_id}")
async def start_trial(trial_id: int, username: str = Depends(verify_credentials)):
    """Start a single trial"""
    if not docker_client:
        raise HTTPException(status_code=500, detail="Docker not available. Make sure Docker is running.")
    try:
        result = start_single_trial(trial_id)
        await manager.broadcast({"event": "trial_started", "trial_id": trial_id})
        return {"success": True, "message": f"Trial {trial_id} started", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/trial/stop/{trial_id}")
async def stop_trial(trial_id: int, username: str = Depends(verify_credentials)):
    """Stop a running trial"""
    try:
        container_name = f"aquatic-trial-{trial_id}"
        container = docker_client.containers.get(container_name)
        container.stop(timeout=10)
        await manager.broadcast({"event": "trial_stopped", "trial_id": trial_id})
        return {"success": True, "message": f"Trial {trial_id} stopped"}
    except docker.errors.NotFound:
        raise HTTPException(status_code=404, detail=f"Trial {trial_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/trial/{trial_id}")
async def remove_trial(trial_id: int, username: str = Depends(verify_credentials)):
    """Remove a trial container"""
    try:
        container_name = f"aquatic-trial-{trial_id}"
        container = docker_client.containers.get(container_name)
        container.remove(force=True)
        await manager.broadcast({"event": "trial_removed", "trial_id": trial_id})
        return {"success": True, "message": f"Trial {trial_id} removed"}
    except docker.errors.NotFound:
        raise HTTPException(status_code=404, detail=f"Trial {trial_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def batch_monitor_task():
    """Background task that monitors containers and starts new trials as capacity becomes available"""
    while batch_manager.running and (batch_manager.pending_trials or get_running_trial_count() > 0):
        try:
            containers = get_simulation_containers()
            running_trials = [c for c in containers if c["status"] == "running"]
            running_count = len(running_trials)
            running_trial_ids = [c["trial_id"] for c in running_trials]

            # Check for completed containers (exited but data exists)
            for container in containers:
                if container["status"] == "exited":
                    trial_id = container["trial_id"]
                    # Check if data was saved (successful completion)
                    data_dir = os.path.expanduser(f"~/workspaces/aquatic-mapping/src/sampling/data/missions/trial_{trial_id}")
                    if os.path.exists(data_dir) and any(f.endswith('_samples.csv') for root, dirs, files in os.walk(data_dir) for f in files):
                        batch_manager.mark_completed(trial_id)
                    else:
                        batch_manager.mark_failed(trial_id)

            # Start new trials if capacity is available
            while running_count < batch_manager.concurrent_limit and batch_manager.pending_trials:
                next_trial = batch_manager.pending_trials[0]
                try:
                    start_single_trial(next_trial)
                    batch_manager.pending_trials.pop(0)  # Remove from pending after successful start
                    running_count += 1
                    await manager.broadcast({
                        "event": "trial_started",
                        "trial_id": next_trial,
                        "batch_status": batch_manager.get_status()
                    })
                except Exception as e:
                    print(f"Failed to start trial {next_trial}: {e}")
                    batch_manager.mark_failed(next_trial)

            # Broadcast batch status update
            await manager.broadcast({
                "event": "batch_update",
                "batch_status": batch_manager.get_status()
            })

            # Check if batch is complete
            if not batch_manager.pending_trials and running_count == 0:
                batch_manager.running = False
                await manager.broadcast({
                    "event": "batch_complete",
                    "batch_status": batch_manager.get_status()
                })
                break

        except Exception as e:
            print(f"Batch monitor error: {e}")

        await asyncio.sleep(5)  # Check every 5 seconds

def get_running_trial_count():
    """Get count of running trial containers"""
    if not docker_client:
        return 0
    return len([c for c in get_simulation_containers() if c["status"] == "running"])

@app.post("/api/batch/start")
async def start_batch(batch: BatchRequest, background_tasks=None, username: str = Depends(verify_credentials)):
    """Start a batch of trials with continuous monitoring"""
    if not docker_client:
        raise HTTPException(status_code=500, detail="Docker not available")

    # Stop any existing batch
    if batch_manager.running:
        batch_manager.stop_batch()

    try:
        # Initialize batch manager
        batch_manager.start_batch(batch.start_trial, batch.end_trial, batch.concurrent)

        # Start initial trials up to concurrent limit
        started = []
        running_count = get_running_trial_count()

        while running_count < batch.concurrent and batch_manager.pending_trials:
            next_trial = batch_manager.pending_trials.pop(0)
            try:
                start_single_trial(next_trial)
                started.append(next_trial)
                running_count += 1
            except Exception as e:
                batch_manager.mark_failed(next_trial)
                print(f"Failed to start trial {next_trial}: {e}")

        # Start background monitoring task
        asyncio.create_task(batch_monitor_task())

        await manager.broadcast({
            "event": "batch_started",
            "trials": started,
            "batch_status": batch_manager.get_status()
        })

        return {
            "success": True,
            "started": started,
            "total": batch.end_trial - batch.start_trial + 1,
            "pending": len(batch_manager.pending_trials),
            "message": f"Batch started: {len(started)} trials running, {len(batch_manager.pending_trials)} pending"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/batch/status")
async def get_batch_status(username: str = Depends(verify_credentials)):
    """Get current batch status"""
    return batch_manager.get_status()

@app.post("/api/batch/cancel")
async def cancel_batch(username: str = Depends(verify_credentials)):
    """Cancel the active batch (doesn't stop running containers)"""
    batch_manager.stop_batch()
    await manager.broadcast({"event": "batch_cancelled"})
    return {"success": True, "message": "Batch cancelled - running containers will continue"}

@app.post("/api/batch/stop")
async def stop_all(username: str = Depends(verify_credentials)):
    """Stop all running trials and cancel batch"""
    try:
        # Cancel batch first
        batch_manager.stop_batch()

        stopped = []
        for container in docker_client.containers.list():
            if container.name.startswith("aquatic-trial-"):
                container.stop(timeout=10)
                stopped.append(container.name)

        await manager.broadcast({"event": "batch_stopped", "containers": stopped})
        return {"success": True, "stopped": stopped}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/system")
async def get_system(username: str = Depends(verify_credentials)):
    """Get system resource usage"""
    return get_system_stats()

@app.get("/api/trials/completed")
async def get_completed_trials(username: str = Depends(verify_credentials)):
    """Get list of completed trials with data"""
    data_dir = os.path.expanduser("~/workspaces/aquatic-mapping/src/sampling/data/missions")
    trials = []

    if os.path.exists(data_dir):
        for item in os.listdir(data_dir):
            if item.startswith("trial_"):
                trial_path = os.path.join(data_dir, item)
                if os.path.isdir(trial_path):
                    trial_id = item.replace("trial_", "")
                    fields = []
                    for field in os.listdir(trial_path):
                        field_path = os.path.join(trial_path, field)
                        if os.path.isdir(field_path):
                            csv_file = os.path.join(field_path, f"{field}_samples.csv")
                            if os.path.exists(csv_file):
                                size = os.path.getsize(csv_file)
                                fields.append({"name": field, "size": size})

                    trials.append({
                        "id": trial_id,
                        "path": trial_path,
                        "fields": fields,
                        "field_count": len(fields)
                    })

    return sorted(trials, key=lambda x: int(x["id"]) if x["id"].isdigit() else 0)

@app.get("/api/logs/{trial_id}")
async def get_trial_logs(trial_id: int, lines: int = 100, username: str = Depends(verify_credentials)):
    """Get logs from a trial container"""
    try:
        container_name = f"aquatic-trial-{trial_id}"
        container = docker_client.containers.get(container_name)
        logs = container.logs(tail=lines).decode('utf-8')
        return {"trial_id": trial_id, "logs": logs}
    except docker.errors.NotFound:
        raise HTTPException(status_code=404, detail=f"Trial {trial_id} not found")

@app.get("/api/download/{trial_id}")
async def download_trial_data(trial_id: int, username: str = Depends(verify_credentials)):
    """Download trial data as ZIP file"""
    data_dir = os.path.expanduser("~/workspaces/aquatic-mapping/src/sampling/data/missions")
    trial_path = os.path.join(data_dir, f"trial_{trial_id}")

    if not os.path.exists(trial_path):
        raise HTTPException(status_code=404, detail=f"Trial {trial_id} data not found")

    # Create ZIP file
    zip_filename = f"trial_{trial_id}_data.zip"
    zip_path = os.path.join(tempfile.gettempdir(), zip_filename)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(trial_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, trial_path)
                zipf.write(file_path, arcname)

    return FileResponse(
        zip_path,
        media_type='application/zip',
        filename=zip_filename,
        headers={"Content-Disposition": f"attachment; filename={zip_filename}"}
    )

@app.get("/api/trial/{trial_id}/data")
async def get_trial_data_preview(trial_id: int, field: str = "radial", username: str = Depends(verify_credentials)):
    """Get preview of trial CSV data"""
    data_dir = os.path.expanduser("~/workspaces/aquatic-mapping/src/sampling/data/missions")
    csv_path = os.path.join(data_dir, f"trial_{trial_id}", field, f"{field}_samples.csv")

    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail=f"Data not found for trial {trial_id}, field {field}")

    import csv
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 100:  # Limit to first 100 rows
                break
            rows.append(row)

    return {"trial_id": trial_id, "field": field, "rows": rows, "total_preview": len(rows)}

@app.delete("/api/trial/{trial_id}/data")
async def delete_trial_data(trial_id: int, username: str = Depends(verify_credentials)):
    """Delete trial data and reconstruction results"""
    data_dir = os.path.expanduser("~/workspaces/aquatic-mapping/src/sampling/data/missions")
    trial_data_path = os.path.join(data_dir, f"trial_{trial_id}")

    results_dir = os.path.expanduser("~/workspaces/aquatic-mapping/reconstruction/results")
    trial_results_path = os.path.join(results_dir, f"trial_{trial_id}")

    deleted = []
    errors = []

    # Delete trial data
    if os.path.exists(trial_data_path):
        try:
            # Try to fix permissions first (files created by Docker may have different ownership)
            try:
                for root, dirs, files in os.walk(trial_data_path):
                    for d in dirs:
                        os.chmod(os.path.join(root, d), 0o755)
                    for f in files:
                        os.chmod(os.path.join(root, f), 0o644)
            except:
                pass  # Ignore permission change errors, will try deletion anyway

            shutil.rmtree(trial_data_path)
            deleted.append(f"trial data ({trial_data_path})")
        except PermissionError as e:
            errors.append(f"Permission denied deleting trial data. Files may be owned by Docker. Try: sudo rm -rf {trial_data_path}")
        except Exception as e:
            errors.append(f"Failed to delete trial data: {str(e)}")

    # Delete reconstruction results
    if os.path.exists(trial_results_path):
        try:
            shutil.rmtree(trial_results_path)
            deleted.append(f"reconstruction results ({trial_results_path})")
        except Exception as e:
            errors.append(f"Failed to delete reconstruction results: {str(e)}")

    # Remove from reconstruction processes dict if present
    if trial_id in reconstruction_processes:
        proc = reconstruction_processes[trial_id]
        if proc.poll() is None:  # Still running
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except:
                proc.kill()
        del reconstruction_processes[trial_id]
        deleted.append("running reconstruction process")

    if not deleted and not errors:
        raise HTTPException(status_code=404, detail=f"No data found for trial {trial_id}")

    return {
        "success": len(errors) == 0,
        "deleted": deleted,
        "errors": errors,
        "message": f"Deleted trial {trial_id}" if len(errors) == 0 else f"Partially deleted trial {trial_id}"
    }

# ============================================================================
# Reconstruction API Routes
# ============================================================================

@app.post("/api/reconstruct/{trial_id}")
async def start_reconstruction(trial_id: int, request: ReconstructionRequest, username: str = Depends(verify_credentials)):
    """Start GP reconstruction for a trial"""
    reconstruction_dir = os.path.expanduser("~/workspaces/aquatic-mapping/reconstruction")
    venv_python = os.path.join(reconstruction_dir, "venv", "bin", "python")
    script_path = os.path.join(reconstruction_dir, "run_reconstruction.py")

    # Check if trial data exists
    data_dir = os.path.expanduser("~/workspaces/aquatic-mapping/src/sampling/data/missions")
    trial_path = os.path.join(data_dir, f"trial_{trial_id}")

    if not os.path.exists(trial_path):
        raise HTTPException(status_code=404, detail=f"Trial {trial_id} data not found")

    # Clean up finished processes and check if actually running
    if trial_id in reconstruction_processes:
        proc = reconstruction_processes[trial_id]
        if proc.poll() is None:  # Still running
            # Kill the old one and start fresh
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except:
                proc.kill()
        # Remove from dict either way
        del reconstruction_processes[trial_id]

    try:
        # Build command - always run all fields and all methods
        cmd = [venv_python, script_path, "all", str(trial_id), "all"]

        # Create results directory and log file for output
        results_dir = os.path.expanduser(f"~/workspaces/aquatic-mapping/reconstruction/results/trial_{trial_id}")
        os.makedirs(results_dir, exist_ok=True)
        log_file = os.path.join(results_dir, "reconstruction.log")

        # Start reconstruction process in background, writing to log file
        with open(log_file, 'w') as log_f:
            proc = subprocess.Popen(
                cmd,
                cwd=reconstruction_dir,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True
            )
        reconstruction_processes[trial_id] = proc

        await manager.broadcast({
            "event": "reconstruction_started",
            "trial_id": trial_id
        })

        return {
            "success": True,
            "message": f"Reconstruction started for trial {trial_id}",
            "pid": proc.pid,
            "log_file": log_file
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/reconstruct/{trial_id}/status")
async def get_reconstruction_status(trial_id: int, username: str = Depends(verify_credentials)):
    """Get reconstruction status for a trial"""
    if trial_id not in reconstruction_processes:
        return {"running": False, "message": "No reconstruction process found"}

    proc = reconstruction_processes[trial_id]
    if proc.poll() is None:
        return {"running": True, "message": "Reconstruction in progress", "pid": proc.pid}
    else:
        # Process finished - check if it succeeded or failed
        return_code = proc.returncode
        status = {
            "running": False,
            "return_code": return_code,
            "success": return_code == 0
        }

        # If failed, try to read error from log
        if return_code != 0:
            log_file = os.path.expanduser(f"~/workspaces/aquatic-mapping/reconstruction/results/trial_{trial_id}/reconstruction.log")
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r') as f:
                        log_content = f.read()
                        # Extract error messages
                        error_lines = [line for line in log_content.split('\n') if 'ERROR' in line or 'error' in line.lower()]
                        if error_lines:
                            status["error"] = '\n'.join(error_lines[-5:])  # Last 5 error lines
                        else:
                            status["error"] = "Reconstruction failed. Check logs for details."
                        status["message"] = "Reconstruction failed"
                except:
                    status["message"] = "Reconstruction failed"
                    status["error"] = "Could not read error log"
            else:
                status["message"] = "Reconstruction failed"
                status["error"] = "No log file found"
        else:
            status["message"] = "Reconstruction completed successfully"

        return status

@app.get("/api/reconstruct/{trial_id}/results")
async def get_reconstruction_results(trial_id: int, username: str = Depends(verify_credentials)):
    """Get reconstruction results (metrics) for a trial"""
    results_dir = os.path.expanduser(f"~/workspaces/aquatic-mapping/reconstruction/results/trial_{trial_id}")

    if not os.path.exists(results_dir):
        raise HTTPException(status_code=404, detail=f"No reconstruction results for trial {trial_id}")

    results = []
    fields = ['radial', 'x_compress', 'y_compress', 'x_compress_tilt', 'y_compress_tilt']
    methods = ['standard_gp', 'mchutchon_nigp', 'girard']
    kernels = ['rbf', 'exponential', 'matern15', 'matern25']

    for method in methods:
        for field in fields:
            for kernel in kernels:
                # Girard only supports RBF
                if method == 'girard' and kernel != 'rbf':
                    continue

                metrics_path = os.path.join(results_dir, method, field, kernel, f"{field}_{kernel}_metrics.csv")
                if os.path.exists(metrics_path):
                    import csv
                    with open(metrics_path, 'r') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            results.append({
                                "method": method,
                                "field": field,
                                "kernel": kernel,
                                "rmse": float(row.get('rmse', 0)),
                                "nrmse": float(row.get('nrmse', 0))
                            })

    # Sort by field, then method, then kernel
    results.sort(key=lambda x: (x['field'], x['method'], x['kernel']))

    return {"trial_id": trial_id, "results": results}

@app.get("/api/reconstruct/{trial_id}/logs")
async def get_reconstruction_logs(trial_id: int, username: str = Depends(verify_credentials)):
    """Get reconstruction output logs"""
    if trial_id not in reconstruction_processes:
        return {"logs": "", "running": False}

    proc = reconstruction_processes[trial_id]
    running = proc.poll() is None

    # Try to read available output (non-blocking)
    logs = ""
    try:
        if proc.stdout:
            import select
            if select.select([proc.stdout], [], [], 0)[0]:
                logs = proc.stdout.read()
    except:
        pass

    return {"logs": logs, "running": running}

@app.get("/api/reconstruct/{trial_id}/images")
async def get_reconstruction_images(trial_id: int, username: str = Depends(verify_credentials)):
    """Get list of reconstruction result images"""
    results_dir = os.path.expanduser(f"~/workspaces/aquatic-mapping/reconstruction/results/trial_{trial_id}")

    if not os.path.exists(results_dir):
        return {"images": []}

    images = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith('.png'):
                rel_path = os.path.relpath(os.path.join(root, file), results_dir)
                images.append({
                    "name": file,
                    "path": rel_path,
                    "url": f"/api/reconstruct/{trial_id}/image/{rel_path}"
                })

    return {"trial_id": trial_id, "images": images}

@app.get("/api/reconstruct/{trial_id}/image/{image_path:path}")
async def get_reconstruction_image(trial_id: int, image_path: str):
    """Serve a reconstruction result image (no auth - public images)"""
    results_dir = os.path.expanduser(f"~/workspaces/aquatic-mapping/reconstruction/results/trial_{trial_id}")
    full_path = os.path.join(results_dir, image_path)

    # Security check - ensure path is within results dir
    if not os.path.abspath(full_path).startswith(os.path.abspath(results_dir)):
        raise HTTPException(status_code=403, detail="Access denied")

    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(full_path, media_type="image/png")

@app.post("/api/reconstruct/{trial_id}/generate-heatmap")
async def generate_comparison_heatmap(trial_id: int, username: str = Depends(verify_credentials)):
    """Generate comparison heatmap for a trial"""
    reconstruction_dir = os.path.expanduser("~/workspaces/aquatic-mapping/reconstruction")
    venv_python = os.path.join(reconstruction_dir, "venv", "bin", "python")
    script_path = os.path.join(reconstruction_dir, "compare_all_methods.py")

    # Check if trial results exist
    results_dir = os.path.expanduser(f"~/workspaces/aquatic-mapping/reconstruction/results/trial_{trial_id}")
    if not os.path.exists(results_dir):
        raise HTTPException(status_code=404, detail=f"No reconstruction results for trial {trial_id}")

    try:
        # Run the comparison script
        result = subprocess.run(
            [venv_python, script_path, str(trial_id)],
            cwd=reconstruction_dir,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            # Check if heatmap was created
            heatmap_path = os.path.join(results_dir, "comparison", "comparison_heatmaps.png")
            if os.path.exists(heatmap_path):
                return {
                    "success": True,
                    "message": f"Comparison heatmap generated for trial {trial_id}",
                    "path": f"/api/reconstruct/{trial_id}/image/comparison/comparison_heatmaps.png"
                }
            else:
                return {
                    "success": False,
                    "message": "Heatmap script completed but no image was created",
                    "output": result.stdout
                }
        else:
            return {
                "success": False,
                "message": "Failed to generate heatmap",
                "error": result.stderr or result.stdout
            }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Heatmap generation timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Host VNC Control Routes (x11vnc + noVNC for web access)
# ============================================================================

@app.post("/api/host/vnc/start")
async def start_host_vnc(username: str = Depends(verify_credentials)):
    """Start VNC server with noVNC web access for remote desktop"""
    scripts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
    start_script = os.path.join(scripts_dir, "start-remote-desktop.sh")

    # Check if services are already running
    x11vnc_running = subprocess.run(["pgrep", "-f", "x11vnc.*-rfbport 5900"],
                                     capture_output=True).returncode == 0
    novnc_running = subprocess.run(["pgrep", "-f", "websockify.*6080"],
                                    capture_output=True).returncode == 0

    if x11vnc_running and novnc_running:
        return {
            "success": True,
            "message": "Remote desktop already running",
            "vnc_port": 5900,
            "novnc_port": 6080,
            "novnc_url": "/vnc/vnc.html"
        }

    # Try to start using the script if it exists
    if os.path.exists(start_script):
        try:
            result = subprocess.run([start_script], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return {
                    "success": True,
                    "message": "Remote desktop started",
                    "vnc_port": 5900,
                    "novnc_port": 6080,
                    "novnc_url": "/vnc/vnc.html"
                }
            else:
                raise Exception(result.stderr or "Script failed")
        except subprocess.TimeoutExpired:
            # Script is running in background, that's expected
            pass
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Fallback: start services directly
    try:
        # Start x11vnc
        if not x11vnc_running:
            vnc_passwd = os.path.expanduser("~/.vnc/passwd")
            vnc_cmd = ["x11vnc", "-display", ":0", "-forever", "-shared", "-rfbport", "5900", "-bg"]
            if os.path.exists(vnc_passwd):
                vnc_cmd.extend(["-rfbauth", vnc_passwd])
            else:
                vnc_cmd.append("-nopw")

            subprocess.Popen(vnc_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            await asyncio.sleep(1)

        # Start noVNC
        if not novnc_running:
            novnc_cmd = ["websockify", "--web=/usr/share/novnc/", "6080", "localhost:5900"]
            subprocess.Popen(novnc_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        return {
            "success": True,
            "message": "Remote desktop started",
            "vnc_port": 5900,
            "novnc_port": 6080,
            "novnc_url": "/vnc/vnc.html"
        }

    except FileNotFoundError as e:
        missing = "x11vnc" if "x11vnc" in str(e) else "novnc/websockify"
        raise HTTPException(
            status_code=500,
            detail=f"{missing} not installed. Run: ./scripts/setup-remote-desktop.sh"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/host/vnc/stop")
async def stop_host_vnc(username: str = Depends(verify_credentials)):
    """Stop the host VNC and noVNC servers"""
    scripts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
    stop_script = os.path.join(scripts_dir, "stop-remote-desktop.sh")

    if os.path.exists(stop_script):
        subprocess.run([stop_script], capture_output=True)
    else:
        # Stop directly
        subprocess.run(["pkill", "-f", "x11vnc.*-rfbport 5900"], capture_output=True)
        subprocess.run(["pkill", "-f", "websockify.*6080"], capture_output=True)

    return {"success": True, "message": "Remote desktop stopped"}

@app.get("/api/host/vnc/status")
async def get_host_vnc_status(username: str = Depends(verify_credentials)):
    """Get host VNC and noVNC server status"""
    x11vnc_running = subprocess.run(["pgrep", "-f", "x11vnc.*-rfbport 5900"],
                                     capture_output=True).returncode == 0
    novnc_running = subprocess.run(["pgrep", "-f", "websockify.*6080"],
                                    capture_output=True).returncode == 0

    return {
        "x11vnc_running": x11vnc_running,
        "novnc_running": novnc_running,
        "running": x11vnc_running and novnc_running,
        "vnc_port": 5900 if x11vnc_running else None,
        "novnc_port": 6080 if novnc_running else None,
        "novnc_url": "http://localhost:6080/vnc.html" if novnc_running else None
    }

# WebSocket for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Send status update every 2 seconds
            status = {
                "containers": get_simulation_containers(),
                "system": get_system_stats(),
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send_json(status)
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# ============================================================================
# Helper Functions
# ============================================================================

def parse_mission_progress(logs: str) -> dict:
    """Parse container logs to extract mission progress"""
    progress = {
        "current_waypoint": 0,
        "total_waypoints": 0,
        "mission_complete": False,
        "progress_percent": 0
    }

    # Find the latest waypoint progress line
    # Format: "Waypoint 5/25: (x, y)"
    waypoint_matches = re.findall(r'Waypoint (\d+)/(\d+):', logs)
    if waypoint_matches:
        last_match = waypoint_matches[-1]
        progress["current_waypoint"] = int(last_match[0])
        progress["total_waypoints"] = int(last_match[1])
        progress["progress_percent"] = round((int(last_match[0]) / int(last_match[1])) * 100, 1)

    # Check if mission is complete
    if 'MISSION COMPLETE!' in logs:
        progress["mission_complete"] = True
        progress["progress_percent"] = 100

    return progress

def get_simulation_containers():
    """Get all simulation containers and their status"""
    if not docker_client:
        return []

    containers = []
    for container in docker_client.containers.list(all=True):
        if container.name.startswith("aquatic-trial-"):
            trial_id = container.name.replace("aquatic-trial-", "")

            # Get container stats
            status = container.status
            stats = {}
            mission_progress = {}

            if status == "running":
                try:
                    raw_stats = container.stats(stream=False)
                    # Calculate CPU percentage
                    cpu_delta = raw_stats['cpu_stats']['cpu_usage']['total_usage'] - \
                                raw_stats['precpu_stats']['cpu_usage']['total_usage']
                    system_delta = raw_stats['cpu_stats']['system_cpu_usage'] - \
                                   raw_stats['precpu_stats']['system_cpu_usage']
                    cpu_percent = (cpu_delta / system_delta) * 100.0 if system_delta > 0 else 0

                    # Memory usage
                    mem_usage = raw_stats['memory_stats'].get('usage', 0)
                    mem_limit = raw_stats['memory_stats'].get('limit', 1)
                    mem_percent = (mem_usage / mem_limit) * 100.0

                    stats = {
                        "cpu_percent": round(cpu_percent, 1),
                        "mem_usage_mb": round(mem_usage / 1024 / 1024, 1),
                        "mem_percent": round(mem_percent, 1)
                    }
                except:
                    pass

                # Get mission progress from logs
                try:
                    logs = container.logs(tail=50).decode('utf-8')
                    mission_progress = parse_mission_progress(logs)
                except:
                    pass

            # Get VNC port
            vnc_port = None
            try:
                ports = container.attrs['NetworkSettings']['Ports']
                if ports and '6080/tcp' in ports and ports['6080/tcp']:
                    vnc_port = ports['6080/tcp'][0]['HostPort']
            except:
                pass

            containers.append({
                "name": container.name,
                "trial_id": trial_id,
                "status": status,
                "vnc_port": vnc_port,
                "stats": stats,
                "mission": mission_progress,
                "created": container.attrs['Created']
            })

    return sorted(containers, key=lambda x: int(x["trial_id"]) if x["trial_id"].isdigit() else 0)

def get_system_stats():
    """Get system resource usage"""
    stats = {
        "cpu_percent": 0,
        "memory_percent": 0,
        "memory_used_gb": 0,
        "memory_total_gb": 0,
        "gpu": None
    }

    # CPU and Memory
    try:
        import psutil
        stats["cpu_percent"] = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        stats["memory_percent"] = mem.percent
        stats["memory_used_gb"] = round(mem.used / 1024 / 1024 / 1024, 1)
        stats["memory_total_gb"] = round(mem.total / 1024 / 1024 / 1024, 1)
    except ImportError:
        pass

    # GPU
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,name', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            if len(parts) >= 4:
                stats["gpu"] = {
                    "utilization": int(parts[0]),
                    "memory_used_mb": int(parts[1]),
                    "memory_total_mb": int(parts[2]),
                    "name": parts[3]
                }
    except:
        pass

    return stats

def start_single_trial(trial_id: int):
    """Start a single simulation trial"""
    host_data_dir = os.path.expanduser("~/workspaces/aquatic-mapping/src/sampling/data/missions")
    container_name = f"aquatic-trial-{trial_id}"
    domain_id = trial_id % 100
    novnc_port = 6080 + trial_id

    # Create data directory
    trial_dir = os.path.join(host_data_dir, f"trial_{trial_id}")
    os.makedirs(trial_dir, exist_ok=True)
    os.chmod(host_data_dir, 0o777)
    os.chmod(trial_dir, 0o777)

    # Remove existing container if exists
    try:
        existing = docker_client.containers.get(container_name)
        existing.remove(force=True)
    except docker.errors.NotFound:
        pass

    # Start new container
    container = docker_client.containers.run(
        "aquatic-sim:latest",
        "mission",
        name=container_name,
        detach=True,
        environment={
            "TRIAL_ID": str(trial_id),
            "ROS_DOMAIN_ID": str(domain_id),
            "HEADLESS": "1"
        },
        volumes={
            host_data_dir: {"bind": "/home/simuser/aquatic-mapping/src/sampling/data/missions", "mode": "rw"}
        },
        ports={
            "6080/tcp": novnc_port
        }
    )

    return {"container_id": container.id, "name": container_name, "vnc_port": novnc_port}

# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
