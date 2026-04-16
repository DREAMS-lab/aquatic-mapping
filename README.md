# Aquatic Mapping

Autonomous adaptive sampling for aquatic environmental monitoring using Gaussian Process-based informative path planning on a PX4 rover in Gazebo simulation.

Seven planners with increasing sophistication — from a baseline that ignores position uncertainty, through Monte Carlo and analytical approaches, to nonstationary Gibbs kernel planners with online hotspot detection.

<p align="center">
  <img src="https://img.shields.io/badge/ROS2-Jazzy-blue" alt="ROS2 Jazzy"/>
  <img src="https://img.shields.io/badge/PX4-SITL-orange" alt="PX4 SITL"/>
  <img src="https://img.shields.io/badge/Gazebo-Harmonic-green" alt="Gazebo Harmonic"/>
  <img src="https://img.shields.io/badge/Python-3.12-yellow" alt="Python 3.12"/>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quickstart — Docker](#quickstart--docker)
- [Quickstart — Local](#quickstart--local)
- [Packages](#packages)
- [Planners](#planners)
- [Fields](#fields)
- [Running Experiments](#running-experiments)
- [Container Orchestration](#container-orchestration)
- [Reconstruction](#reconstruction)
- [Statistics](#statistics)
- [Data Structure](#data-structure)
- [Repo Layout](#repo-layout)

---

## Overview

A PX4 R1 rover drives through a 25m × 25m domain containing a Gaussian temperature field. At each step, a planner selects the next sampling location by maximizing an information-theoretic acquisition function (mutual information) minus a travel cost. After 100 samples the mission ends and reconstruction metrics are computed against ground truth.

The seven planners differ in how they handle **position uncertainty** (from the PX4 EKF) and **spatial nonstationarity** (via Gibbs kernels with spatially varying lengthscales):

| Planner | Kernel | Uncertainty | Hotspot |
|---------|--------|-------------|---------|
| `exact` | RBF | ignored | post-mission |
| `pose_aware` | RBF | Monte Carlo | post-mission |
| `analytical` | RBF | Girard closed-form | post-mission |
| `nonstationary_exact` | Gibbs | ignored | post-mission |
| `nonstationary_pose_aware` | Gibbs | Monte Carlo | post-mission |
| `nonstationary_hotspot_exact` | Gibbs | ignored | online 2-phase |
| `nonstationary_hotspot_pose_aware` | Gibbs | Monte Carlo | online 2-phase |

---

## Prerequisites

**For Docker (recommended):**
- Docker with NVIDIA GPU support (`nvidia-docker2`)
- ~15 GB disk for the image

**For local development:**
- Ubuntu 24.04
- [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/Installation.html) (desktop)
- [Gazebo Harmonic](https://gazebosim.org/docs/harmonic/install_ubuntu)
- [PX4-Autopilot](https://docs.px4.io/main/en/dev_setup/dev_env_linux_ubuntu.html) (SITL)
- [Micro XRCE DDS Agent](https://micro-xrce-dds.docs.eprosima.com/en/latest/)
- Python 3.12, NVIDIA GPU (optional but recommended for PyTorch)

---

## Quickstart — Docker

```bash
git clone --recurse-submodules https://github.com/DREAMS-lab/aquatic-mapping.git
cd aquatic-mapping

# Build the simulation image (~30-60 min first time)
cd infra/docker
docker build -t aquatic-sim .
cd ../..

# Run a single trial (exact planner, radial field)
cd container/info_gain
python3 orchestrator.py --trials 1 --planners exact --fields radial --workers 1
```

Watch via VNC at `http://localhost:6090/vnc.html`.

For batch runs:

```bash
# 10 trials, all 5 fields, exact + pose_aware, 2 parallel workers
python3 orchestrator.py --trials 10 --planners exact,pose_aware --workers 2

# Or use the GUI
python3 trial_manager.py
```

---

## Quickstart — Local

```bash
git clone --recurse-submodules https://github.com/DREAMS-lab/aquatic-mapping.git
cd aquatic-mapping

# Setup venv + build packages
./setup.sh        # CPU
./setup.sh --gpu  # with CUDA 12.4

# In one terminal — start PX4 SITL
cd ~/PX4-Autopilot
make px4_sitl gz_r1_rover

# In another terminal — start DDS bridge
MicroXRCEAgent udp4 -p 8888

# In a third terminal — run a planner
source venv/bin/activate
source install/setup.bash
ros2 launch info_gain exact.launch.py field_type:=radial trial:=1
```

The planner will arm the rover, collect 100 samples, save results to `data/trials/exact/radial/trial_001/`, and exit.

---

## Packages

### `sampling`

Rover infrastructure and field generation. Provides:

- **5 field generators** — temperature fields with different anisotropy and rotation
- **Rover monitor** — PX4 odometry → ROS2 TF bridge
- **Lawnmower mission** — fixed-path data collection baseline
- **Data recorder** — CSV + rosbag logging

```bash
ros2 launch sampling mission.launch.py trial_number:=1       # lawnmower
ros2 launch sampling rover_fields.launch.py                   # rover + fields only
```

### `info_gain`

Three stationary-kernel planners (RBF):

```bash
ros2 launch info_gain exact.launch.py field_type:=radial trial:=1
ros2 launch info_gain pose_aware.launch.py field_type:=radial trial:=1
ros2 launch info_gain analytical.launch.py field_type:=radial trial:=1
```

Core modules: `gp_model.py` (GPyTorch GP wrapper), `peak_detection.py` (Kac-Rice hotspot detection).

### `nonstationary_planning`

Four Gibbs-kernel planners with spatially varying lengthscales:

```bash
ros2 launch nonstationary_planning nonstationary_exact.launch.py field_type:=radial trial:=1
ros2 launch nonstationary_planning nonstationary_pose_aware.launch.py field_type:=radial trial:=1
ros2 launch nonstationary_planning nonstationary_hotspot_exact.launch.py field_type:=radial trial:=1
ros2 launch nonstationary_planning nonstationary_hotspot_pose_aware.launch.py field_type:=radial trial:=1
```

Core modules: `gibbs_kernel.py` (anisotropic Paciorek kernel, 76 parameters), `gibbs_gp_model.py` (online MAP optimization).

### `px4_msgs`

PX4 message definitions (git submodule). Provides `VehicleOdometry`, `TrajectorySetpoint`, `OffboardControlMode`, etc.

---

## Planners

### Exact (baseline)

Greedy single-step. Acquisition: `score = info_gain - λ × travel_cost`. Ignores position uncertainty entirely.

### Pose-Aware (Monte Carlo)

Averages information gain over M=30 Monte Carlo samples drawn from the EKF position covariance. Vectorized: one batch GP prediction for all N×M noisy candidates.

### Analytical (Girard closed-form)

Closed-form expected variance using the Girard/Dallaire expected RBF kernel. Deterministic, no sampling noise.

### Nonstationary Exact / Pose-Aware

Same acquisition functions but with a Gibbs kernel (spatially varying lengthscale). Online MAP optimization every 10 samples learns anisotropic lengthscale fields l₁(x), l₂(x), θ(x).

### Hotspot Exact / Pose-Aware

Two-phase: explore for 40 samples (pure info gain), then exploit detected hotspots via Kac-Rice expected number of peaks. Acquisition weighted toward detected peaks with 4-layer edge filtering.

---

## Fields

All fields are 25m × 25m Gaussian temperature surfaces. Base temperature 20°C, hotspot amplitude 10°C, center at (12.5, 12.5), measurement noise σ = 0.6°C.

| Field | σ_x | σ_y | Rotation |
|-------|-----|-----|----------|
| `radial` | 5.0 | 5.0 | 0° |
| `x_compress` | 2.5 | 7.0 | 0° |
| `y_compress` | 7.0 | 2.5 | 0° |
| `x_compress_tilt` | 2.5 | 7.0 | 45° |
| `y_compress_tilt` | 7.0 | 2.5 | 45° |

---

## Running Experiments

### Single planner, single field

```bash
ros2 launch info_gain exact.launch.py field_type:=radial trial:=1
```

### Common launch arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `field_type` | `radial` | One of the 5 fields |
| `trial` | `-1` (auto) | Trial number |
| `lambda_cost` | `0.1` | Info vs travel trade-off |
| `noise_var` | `0.36` | GP observation noise (σ² = 0.6²) |
| `lengthscale` | `2.0` | GP kernel lengthscale |
| `uncertainty_scale` | `1.0` | Multiply EKF variance (pose-aware only) |
| `n_mc_samples` | `30` | Monte Carlo samples (pose-aware only) |

---

## Container Orchestration

The `container/info_gain/` directory has everything for running automated batch experiments in Docker.

### Scripts

| Script | Description |
|--------|-------------|
| `orchestrator.py` | Work-queue orchestrator — runs N parallel Docker containers |
| `trial_manager.py` | Tkinter GUI — field/planner selection, VNC buttons, progress tracking |
| `run_single_field.sh` | Container entrypoint — 7-phase startup (display → build → PX4 → DDS → metrics → planner → wait) |
| `monitor.py` | Terminal dashboard with live progress |
| `start_exact_sim.sh` | Manual mode — opens 3 xterm windows for interactive dev |
| `stop_sim.sh` | Kill all sim processes |

### Orchestrator usage

```bash
# 10 trials, all fields, all planners, 2 workers
python3 orchestrator.py --trials 10

# Specific planners and fields
python3 orchestrator.py --trials 5 --planners exact,pose_aware --fields radial,x_compress

# 4 parallel workers
python3 orchestrator.py --trials 10 --workers 4

# Test pose-aware with higher uncertainty
python3 orchestrator.py --trials 5 --planners pose_aware --uncertainty-scale 25
```

### Container isolation

Each worker slot gets unique ROS and Gazebo isolation:

- `ROS_DOMAIN_ID = slot + 10`
- `GZ_PARTITION = worker_{slot}`
- VNC: `590{2+slot}`, noVNC: `609{slot}`

---

## Reconstruction

Offline GP reconstruction comparison using data from planner trials. Three methods:

| Method | Approach |
|--------|----------|
| Standard GP | Baseline, deterministic inputs |
| McHutchon NIGP | Input noise as heteroscedastic output noise via gradients |
| Girard | Analytic expected RBF kernel for uncertain inputs |

```bash
cd reconstruction
source venv/bin/activate      # or use workspace venv
pip install -r requirements.txt

python run_reconstruction.py radial 1 all       # all methods, radial field, trial 1
python run_reconstruction.py all 1 standard     # standard GP, all fields
python analyze_planners.py --all                 # cross-planner analysis
```

Output: `data/reconstruction/trial_N/{method}/{field}/{kernel}/`

---

## Statistics

Planner comparison with publication-quality plots and statistical tests.

```bash
cd statistics
python compare_planners.py           # exact vs pose-aware (paired)
python compare_all_planners.py       # all 7 planners
```

Tests: Wilcoxon signed-rank, paired t-test, Cohen's d with 95% CI, Holm-Bonferroni correction.

Output: `data/statistics/` — forest plots, bar charts, paired slope graphs, summary CSVs.

---

## Data Structure

All output goes to `data/` (gitignored):

```
data/
├── trials/
│   ├── exact/{field}/trial_001/
│   │   ├── config.json
│   │   ├── samples.csv
│   │   ├── decisions.csv
│   │   ├── summary.json
│   │   ├── ground_truth.npz
│   │   ├── gp_reconstruction.npz
│   │   ├── compute_metrics.csv
│   │   └── figures/
│   ├── pose_aware/...
│   ├── analytical/...
│   ├── nonstationary_exact/...
│   ├── nonstationary_pose_aware/...
│   ├── nonstationary_hotspot_exact/...
│   └── nonstationary_hotspot_pose_aware/...
├── reconstruction/
│   └── trial_N/{method}/{field}/{kernel}/
└── statistics/
    └── *.png, *.csv
```

---

## Repo Layout

```
aquatic-mapping/
├── src/
│   ├── sampling/                  # Rover, fields, missions, logging
│   ├── info_gain/                 # 3 stationary planners + GP model
│   ├── nonstationary_planning/    # 4 Gibbs kernel planners
│   └── px4_msgs/                  # PX4 message definitions (submodule)
├── container/
│   └── info_gain/                 # Docker orchestration scripts
├── infra/
│   ├── docker/                    # Dockerfile, docker-compose, entrypoints
│   └── scripts/                   # Build/run helpers
├── reconstruction/                # GP reconstruction methods
├── statistics/                    # Planner comparison scripts
├── requirements.txt               # Python dependencies
├── setup.sh                       # One-command workspace setup
└── README.md
```

---

## License

DREAMS Lab, Arizona State University
