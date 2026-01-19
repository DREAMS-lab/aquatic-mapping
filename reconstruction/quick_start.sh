#!/bin/bash
# quick start script for GP reconstruction

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "  GP Reconstruction Quick Start"
echo "========================================"
echo ""

# activate venv
if [ -d "venv" ]; then
    echo "activating virtual environment..."
    source venv/bin/activate
else
    echo "ERROR: virtual environment not found"
    echo "run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# get arguments
FIELD_TYPE="${1:-radial}"
TRIAL="${2:-1}"
METHOD="${3:-both}"

echo "field type: $FIELD_TYPE"
echo "trial: $TRIAL"
echo "method: $METHOD"
echo ""

# run reconstruction
python run_reconstruction.py "$FIELD_TYPE" "$TRIAL" "$METHOD"

echo ""
echo "========================================"
echo "  Complete!"
echo "========================================"
echo "results saved to: results/trial_${TRIAL}/"
