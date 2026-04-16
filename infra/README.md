# Aquatic Mapping Simulation Infrastructure

Docker-based infrastructure for running parallel PX4/ROS2/Gazebo simulations.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         HOST MACHINE                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Docker Container (aquatic-sim)              │    │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐    │    │
│  │  │  Xvfb   │ │   VNC   │ │ noVNC   │ │   PX4 SITL  │    │    │
│  │  │ :99     │ │  :590X  │ │  :608X  │ │             │    │    │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────────┘    │    │
│  │  ┌─────────┐ ┌─────────┐ ┌───────────────────────────┐  │    │
│  │  │ Gazebo  │ │  XRCE   │ │   ROS2 Nodes (sampling)   │  │    │
│  │  │ Harmonic│ │  Agent  │ │                           │  │    │
│  │  └─────────┘ └─────────┘ └───────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Shared Data Volume                     │   │
│  │    ~/aquatic-mapping/src/sampling/data/missions/          │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Build the Docker Image

```bash
cd ~/workspaces/aquatic-mapping/infra/scripts
chmod +x *.sh
./build-image.sh
```

This takes 30-60 minutes on first build (downloads ~10GB of dependencies).

### 2. Run a Single Simulation

```bash
# Basic run
docker run -d --name sim-test \
    -e TRIAL_ID=1 \
    -e FIELD_TYPE=radial \
    -p 6081:6080 \
    aquatic-sim:latest

# View logs
docker logs -f sim-test

# Access GUI via browser
open http://localhost:6081/vnc.html
```

### 3. Run Multiple Simulations

Using docker-compose:

```bash
cd ~/workspaces/aquatic-mapping/infra/docker

# Start 3 simulations
docker compose up sim1 sim2 sim3

# Or use the batch script
cd ~/workspaces/aquatic-mapping/infra/scripts
./run-batch.sh 1 10 radial 5   # Trials 1-10, 5 concurrent
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TRIAL_ID` | 1 | Trial number for data output |
| `FIELD_TYPE` | radial | Field type: radial, x_compress, y_compress, x_compress_tilt, y_compress_tilt |
| `ROS_DOMAIN_ID` | 0 | ROS2 domain ID (unique per container to prevent crosstalk) |
| `HEADLESS` | 1 | 1=no GUI rendering, 0=full GUI |
| `VNC_PORT` | 5900 | VNC server port |
| `NOVNC_PORT` | 6080 | noVNC web interface port |

### Resource Limits (Recommended)

| Resource | Per Container | 10 Containers |
|----------|---------------|---------------|
| CPU | 2 cores | 20 cores |
| RAM | 4 GB | 40 GB |
| GPU | None (headless) | - |

Your system (7800X3D 8C/16T, 32GB RAM): **Run 5-8 containers concurrently**

## Network Access

### Local Access

```
http://localhost:6081/vnc.html    # Container 1
http://localhost:6082/vnc.html    # Container 2
...
```

### Remote/Internet Access

Option 1: **Port Forwarding** (router config)
- Forward ports 6081-6090 to your machine

Option 2: **Tailscale** (recommended for security)
```bash
# Install Tailscale
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up

# Access via Tailscale IP
http://100.x.x.x:6081/vnc.html
```

Option 3: **Cloudflare Tunnel** (free, secure)
```bash
# Install cloudflared
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared-linux-amd64
sudo mv cloudflared-linux-amd64 /usr/local/bin/cloudflared

# Create tunnel
cloudflared tunnel --url http://localhost:6081
# Gives you a public URL like: https://xxxxx.trycloudflare.com
```

Option 4: **ngrok** (quick testing)
```bash
ngrok http 6081
```

## Container Commands

### Entry Points

```bash
# Full mission (default)
docker run aquatic-sim:latest mission

# Just rover stack (debugging)
docker run aquatic-sim:latest rover

# Interactive shell
docker run -it aquatic-sim:latest shell

# Headless test (no display)
docker run aquatic-sim:latest headless-test
```

### Managing Containers

```bash
# List running containers
docker ps --filter "name=aquatic-sim"

# Stop all simulations
docker stop $(docker ps -q --filter "name=aquatic-sim")

# Remove all simulation containers
docker rm $(docker ps -aq --filter "name=aquatic-sim")

# View container logs
docker logs -f aquatic-sim-1

# Execute command in running container
docker exec -it aquatic-sim-1 bash
```

## Data Output

Data is saved to:
```
~/workspaces/aquatic-mapping/src/sampling/data/missions/
├── radial/
│   ├── trial_1/
│   │   ├── radial_bag/
│   │   └── radial_samples.csv
│   ├── trial_2/
│   └── ...
├── x_compress/
└── ...
```

## Troubleshooting

### Container won't start
```bash
# Check logs
docker logs aquatic-sim-1

# Common issues:
# - Port already in use: Change VNC_PORT/NOVNC_PORT
# - Out of memory: Reduce concurrent containers
```

### ROS2 nodes not communicating
- Ensure unique `ROS_DOMAIN_ID` per container
- Check `MicroXRCEAgent` is running: `docker exec -it <container> ps aux | grep XRCE`

### Gazebo not starting
- Check headless mode: `HEADLESS=1` for no GUI
- Increase container memory: `--memory=6g`

### VNC connection refused
```bash
# Check VNC server is running
docker exec -it aquatic-sim-1 ps aux | grep x11vnc

# Check port mapping
docker port aquatic-sim-1
```

## Building from Scratch

If you need to modify the image:

1. Edit `infra/docker/Dockerfile`
2. Rebuild: `./scripts/build-image.sh --no-cache`
3. Test: `docker run -it aquatic-sim:latest shell`

## File Structure

```
infra/
├── docker/
│   ├── Dockerfile           # Main image definition
│   ├── docker-compose.yml   # Multi-container orchestration
│   ├── entrypoint.sh        # Container startup script
│   └── supervisord.conf     # Process management
├── scripts/
│   ├── build-image.sh       # Build helper
│   └── run-batch.sh         # Batch execution
├── controller/              # (Future) Central control plane
└── README.md                # This file
```
