#!/bin/bash
# Start the Aquatic Mapping Web Control Panel

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "  Aquatic Mapping - Web Control Panel"
echo "=============================================="

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "[1/3] Creating virtual environment..."
    python3 -m venv venv
fi

echo "[2/3] Installing dependencies..."
source venv/bin/activate
pip install -q -r requirements.txt

echo "[3/3] Starting server..."
echo ""
echo "=============================================="
echo "  Server starting on http://localhost:8000"
echo "=============================================="
echo ""
echo "  Username: bakin"
echo "  Password: ozhugu"
echo ""
echo "  Change password by setting environment variables:"
echo "    export SIM_USERNAME=youruser"
echo "    export SIM_PASSWORD=yourpassword"
echo ""
echo "=============================================="
echo ""

cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
