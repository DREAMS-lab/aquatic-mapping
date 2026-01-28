# Info Gain Simulation Scripts

Local scripts to run informative path planning experiments with different uncertainty considerations.

## Quick Start

```bash
# Build the workspace first
cd ~/workspaces/aquatic-mapping
colcon build --packages-select sampling info_gain

# Run exact planner (no uncertainty)
./container/info_gain/start_exact_sim.sh radial 1

# Run pose-aware planner (position uncertainty)
./container/info_gain/start_pose_aware_sim.sh radial 1

# Stop simulation
./container/info_gain/stop_sim.sh
```

## Planners Available

### 1. Exact Planner (No Uncertainty)
```bash
./start_exact_sim.sh [field_type] [trial]
```
- **Method**: Standard information gain Δ(x)
- **Assumption**: Robot reaches commanded positions exactly
- **Data saved to**: `src/info_gain/data/trials/exact/`

### 2. Planner-Aware (Position Uncertainty in Planning)
```bash
./start_pose_aware_sim.sh [field_type] [trial]
```
- **Method**: Expected information gain U(x) = E[Δ(x̃)] under pose noise
- **GP Model**: Standard (exact inputs) - no change to GP
- **Planner**: Uses Monte Carlo to average over position uncertainty
- **Uncertainty source**: **Real-time PX4 EKF position variance** (`/fmu/out/vehicle_odometry`)
- **Data saved to**: `src/info_gain/data/trials/pose_aware/`
- **Extra parameters**:
  - `POSITION_STD`: Fallback only (default 0.5m, used ONLY if PX4 variance unavailable)
  - `N_MC_SAMPLES`: Monte Carlo samples (default 30)

### 3. Model-Aware Planner [TODO]
- GP model accounts for uncertain inputs

### 4. Both-Aware Planner [TODO]
- Both GP and planner account for uncertainty

## Usage Examples

### Exact Planner:
```bash
# Auto trial number
./start_exact_sim.sh radial

# Specific trial
./start_exact_sim.sh radial 1

# Custom parameters
HORIZON=5 LAMBDA_COST=0.2 ./start_exact_sim.sh x_compress 2
```

### Pose-Aware Planner:
```bash
# Default settings (σ_pos = 0.5m, MC samples = 30)
./start_pose_aware_sim.sh radial 1

# Custom position uncertainty
POSITION_STD=1.0 ./start_pose_aware_sim.sh radial 2

# More Monte Carlo samples for better approximation
N_MC_SAMPLES=50 ./start_pose_aware_sim.sh y_compress 1

# Custom all parameters
HORIZON=3 POSITION_STD=0.3 N_MC_SAMPLES=50 ./start_pose_aware_sim.sh radial 3
```

## Field Types

- `radial` (default) - Symmetric Gaussian hotspot
- `x_compress` - Compressed in X direction
- `y_compress` - Compressed in Y direction
- `x_compress_tilt` - X-compressed with 30° tilt
- `y_compress_tilt` - Y-compressed with 30° tilt

## Configuration Parameters

### Common Parameters (Both Planners)

| Variable | Default | Description |
|----------|---------|-------------|
| `HORIZON` | 2 | Planning horizon H (1=greedy, 2=default) |
| `LAMBDA_COST` | 0.1 | Info vs travel trade-off |
| `PX4_DIR` | ~/PX4-Autopilot | PX4 installation path |

### Pose-Aware Only

| Variable | Default | Description |
|----------|---------|-------------|
| `POSITION_STD` | 0.5 | Fallback position std if PX4 variance unavailable |
| `N_MC_SAMPLES` | 30 | Monte Carlo samples for E[Δ(x̃)] |

**Notes:**
- Max samples is hardcoded to 100 in both planners
- Trial number can be specified as 2nd argument, or omitted for auto-increment
- Sampling region: [0.5, 24.5] × [0.5, 24.5] m (0.5m buffer from field edges)

## Position Uncertainty from PX4 EKF

**Both planners now use real-time position variance from PX4 Extended Kalman Filter:**

- **Exact planner**: Logs PX4 variance in `samples.csv` but doesn't use it for planning (assumes exact positions)
- **Pose-aware planner**: **Uses PX4 variance dynamically** for computing expected information gain

The pose-aware planner extracts `position_variance` from `/fmu/out/vehicle_odometry`:
- `position_variance[0]` = σ²_north (NED frame)
- `position_variance[1]` = σ²_east (NED frame)

These are converted to ENU frame and used to build the covariance matrix `Σ_x = diag(σ²_x, σ²_y)` for Monte Carlo sampling.

**CSV columns added:**
- `pos_var_x`: Variance in x-direction (m²)
- `pos_var_y`: Variance in y-direction (m²)
- `pos_std_x`: Standard deviation in x-direction (m)
- `pos_std_y`: Standard deviation in y-direction (m)

This makes the pose-aware planner **adaptive** - uncertainty is typically high initially and decreases as the EKF converges.

## What It Starts

3 terminal windows:
1. **PX4 SITL + Gazebo** - Rover simulation
2. **DDS Agent** - Micro XRCE DDS bridge
3. **ROS2 Launch** - All ROS2 nodes:
   - Static TF (world -> odom)
   - Rover monitor (odom -> base_link TF)
   - Robot state publisher (URDF)
   - Field generator (selected field only)
   - RViz (field-specific config)
   - Planner node (exact or pose-aware)

## Data Output

Results saved to:
```
src/info_gain/data/trials/
├── exact/                       # Exact planner (no uncertainty)
│   ├── radial/
│   │   ├── trial_001/
│   │   ├── trial_002/
│   │   └── ...
│   └── ...
└── pose_aware/                  # Pose-aware planner (position uncertainty)
    ├── radial/
    │   └── trial_001/
    └── ...

Each trial folder contains:
├── config.json                    # Experiment configuration (includes planner type)
├── samples.csv                    # All collected samples
├── decisions.csv                  # Planning decisions per step
├── decisions.json                 # Full decision details (JSON)
├── summary.json                   # Final results (includes reconstruction metrics)
├── ground_truth.npz               # Ground truth field (X, Y, field)
├── gp_reconstruction.npz          # GP predictions (mean, variance, error)
├── reconstruction_metrics.json    # RMSE, MAE, max error, mean variance
└── figures/
    ├── final.png                  # Final live visualization
    ├── reconstruction_comparison.png  # Ground truth vs GP comparison (4 panels)
    └── convex_hull.png            # Sampling coverage visualization
```

## Reconstruction Evaluation

Both planners automatically evaluate GP reconstruction quality against ground truth:

**Metrics computed:**
- **RMSE** (Root Mean Square Error): Overall reconstruction accuracy
- **MAE** (Mean Absolute Error): Average pointwise error
- **Max Error**: Worst-case reconstruction error
- **Mean Variance**: Average GP uncertainty

**Output files:**
- `ground_truth.npz`: True field values on 0.5m grid
- `gp_reconstruction.npz`: GP predictions (mean, variance, error maps)
- `reconstruction_metrics.json`: All metrics
- `figures/reconstruction_comparison.png`: Visual comparison (4 panels with RMSE)
- `figures/convex_hull.png`: Sampling coverage (hull area, % coverage)

**Loading results in Python:**
```python
import numpy as np
import json

# Load ground truth
gt = np.load('ground_truth.npz')
X, Y, field = gt['X'], gt['Y'], gt['field']

# Load GP reconstruction
gp = np.load('gp_reconstruction.npz')
gp_mean, gp_var, error = gp['mean'], gp['variance'], gp['error']

# Load metrics
with open('reconstruction_metrics.json') as f:
    metrics = json.load(f)
    print(f"RMSE: {metrics['rmse']:.3f}°C")
```

## Comparison Between Planners

To compare exact vs pose-aware performance:

1. Run both planners on same field
2. Compare `summary.json` metrics:
   - `reconstruction_rmse`: Field reconstruction quality
   - `cumulative_info_gain`: Total information gathered
   - `total_travel_cost`: Distance traveled
3. Compare visualizations

**Expected result:** Under real position noise, pose-aware planner should achieve better reconstruction by planning for uncertainty.

## Docker Container

For containerized experiments with noVNC, see `../../infra/docker/`
