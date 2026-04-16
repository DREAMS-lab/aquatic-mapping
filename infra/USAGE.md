# Container Infrastructure Usage Guide

## Quick Start

### Build the Container Image
```bash
cd ~/workspaces/aquatic-mapping/infra/docker
./build-image.sh
```

### Run a Single Trial
```bash
cd ~/workspaces/aquatic-mapping/infra/scripts
./run-trial.sh 1
```

This will:
- Start container with trial ID 1
- Run **all 5 field types** simultaneously (radial, x_compress, y_compress, x_compress_tilt, y_compress_tilt)
- Save data to `~/workspaces/aquatic-mapping/src/sampling/data/missions/*/trial_1/`
- Make GUI available at http://localhost:6081/vnc.html

### Run Multiple Trials in Parallel
```bash
cd ~/workspaces/aquatic-mapping/infra/scripts
./run-batch.sh 1 25 5
```

This will:
- Run trials 1-25 (all 5 fields per trial)
- Maximum 5 concurrent containers
- Auto-queue when slots are full

## Scripts

### `run-trial.sh`
Run a single trial simulation. All 5 field types run automatically.

**Usage:**
```bash
./run-trial.sh <trial_number>
```

**Examples:**
```bash
./run-trial.sh 1     # Trial 1 (all 5 fields)
./run-trial.sh 10    # Trial 10 (all 5 fields)
./run-trial.sh 50    # Trial 50 (all 5 fields)
```

### `run-batch.sh`
Run multiple trials in parallel with automatic queueing.

**Usage:**
```bash
./run-batch.sh <start_trial> <end_trial> [max_concurrent]
```

**Examples:**
```bash
./run-batch.sh 1 25              # Trials 1-25, max 5 concurrent (default)
./run-batch.sh 1 25 8            # Trials 1-25, max 8 concurrent
./run-batch.sh 26 50 3           # Trials 26-50, max 3 concurrent
./run-batch.sh 1 100 10          # Trials 1-100, max 10 concurrent
```

## Container Management

### View Running Containers
```bash
docker ps --filter "name=aquatic-trial-"
```

### View Container Logs
```bash
docker logs -f aquatic-trial-1
```

### Stop a Trial
```bash
docker stop aquatic-trial-1
```

### Stop All Trials
```bash
docker ps --filter "name=aquatic-trial-" --format '{{.Names}}' | xargs docker stop
```

### Remove All Trial Containers
```bash
docker ps -a --filter "name=aquatic-trial-" --format '{{.Names}}' | xargs docker rm
```

## Access GUI

Each trial gets a unique noVNC port based on trial ID:
- Trial 1: http://localhost:6081/vnc.html
- Trial 2: http://localhost:6082/vnc.html
- Trial N: http://localhost:608N/vnc.html

**Internet Access:**
Set up Cloudflare tunnel pointing to localhost:6081-6090 for remote access.

## Data Persistence

All trial data is saved to the host at:
```
~/workspaces/aquatic-mapping/src/sampling/data/missions/
```

**Structure for each trial:**
```
missions/
├── radial/
│   ├── trial_1/
│   │   ├── radial_samples.csv
│   │   └── radial_bag/
│   │       ├── metadata.yaml
│   │       └── radial_bag_0.mcap
│   ├── trial_2/
│   └── ...
├── x_compress/
│   ├── trial_1/
│   ├── trial_2/
│   └── ...
├── y_compress/
│   └── ...
├── x_compress_tilt/
│   └── ...
└── y_compress_tilt/
    └── ...
```

**CSV Format:**
```
x,y,temperature,vx,vy,cov_xx,cov_xy,cov_yy,vel_var_x,vel_var_y
```

Data is written directly to the host (not stored in container). Containers can be deleted without losing data.

## Monitor Data Collection

### Watch all trial 2 data directories
```bash
watch -n 5 'ls -lh ~/workspaces/aquatic-mapping/src/sampling/data/missions/*/trial_2/'
```

### Count samples in radial field trial 1
```bash
wc -l ~/workspaces/aquatic-mapping/src/sampling/data/missions/radial/trial_1/radial_samples.csv
```

### View CSV data
```bash
head ~/workspaces/aquatic-mapping/src/sampling/data/missions/radial/trial_1/radial_samples.csv
```

### Monitor all trials
```bash
watch -n 2 'docker ps --filter "name=aquatic-trial-"'
```

## Resource Allocation

**Recommended limits based on hardware:**
- 7800X3D (8C/16T) + 32GB RAM: **5-8 concurrent containers**
- Each container uses ~2 CPU cores, ~3GB RAM

Adjust `max_concurrent` parameter based on your system performance.

## Example Workflows

### Run 25 trials (recommended approach)
```bash
cd ~/workspaces/aquatic-mapping/infra/scripts

# Run all 25 in one batch with 5 concurrent max
./run-batch.sh 1 25 5

# Monitor progress
watch -n 10 'docker ps --filter "name=aquatic-trial-"'
```

### Run 100 trials in batches
```bash
cd ~/workspaces/aquatic-mapping/infra/scripts

# Batch 1: trials 1-25
./run-batch.sh 1 25 5

# Batch 2: trials 26-50  
./run-batch.sh 26 50 5

# Batch 3: trials 51-75
./run-batch.sh 51 75 5

# Batch 4: trials 76-100
./run-batch.sh 76 100 5
```

### Test single trial before batch
```bash
cd ~/workspaces/aquatic-mapping/infra/scripts

# Test trial 1
./run-trial.sh 1

# Wait 2 minutes for data collection
sleep 120

# Check data for all fields
ls -lh ~/workspaces/aquatic-mapping/src/sampling/data/missions/*/trial_1/

# If good, stop and run full batch
docker stop aquatic-trial-1
docker rm aquatic-trial-1
./run-batch.sh 1 25 5
```

### Run trials with conservative resource usage
```bash
# Only 3 concurrent for lower system load
./run-batch.sh 1 50 3
```

## Troubleshooting

### Container exits immediately
Check logs:
```bash
docker logs aquatic-trial-1
```

Rebuild image if needed:
```bash
cd ~/workspaces/aquatic-mapping/infra/docker
docker build --no-cache -t aquatic-sim:latest .
```

### No data being saved
1. Check volume mount in container:
   ```bash
   docker exec aquatic-trial-1 ls -la /home/simuser/aquatic-mapping/src/sampling/data/missions/
   ```

2. Check host directory permissions:
   ```bash
   ls -la ~/workspaces/aquatic-mapping/src/sampling/data/missions/
   ```

### Port already in use
Each trial needs a unique ID. If trial 1 is running, start trial 2 or stop trial 1 first:
```bash
docker stop aquatic-trial-1
```

### Out of memory
Reduce `max_concurrent` parameter:
```bash
./run-batch.sh 1 25 3  # Only 3 concurrent instead of 5
```

### Check system resources
```bash
# CPU and memory usage
htop

# Docker container resources
docker stats
```

## Fields Collected

Each trial automatically collects data for all 5 temperature field types:

1. **radial** - Radial temperature gradient
2. **x_compress** - X-axis compression field
3. **y_compress** - Y-axis compression field  
4. **x_compress_tilt** - X-axis compression with tilt
5. **y_compress_tilt** - Y-axis compression with tilt

You don't need to specify fields - they all run automatically in parallel for each trial.
