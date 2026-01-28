# Sampling Container (Lawnmower)

Container setup for lawnmower sampling experiments.

**See main container files in:** `../../infra/docker/`

This folder is a reference - the actual Docker setup is in `infra/docker/`.

## Quick Start

```bash
# From infra/docker/ directory
cd ../../infra/docker/

# Build the container
docker build -t aquatic-sim .

# Run a single trial
../scripts/run-trial.sh 1
```

## Key Files in infra/docker/

- `Dockerfile` - Container image definition
- `docker-compose.yml` - Multi-container orchestration
- `entrypoint-simple.sh` - Container startup script
- `start-mission-headless.sh` - Headless mission runner
