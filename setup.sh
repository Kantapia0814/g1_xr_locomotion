#!/usr/bin/env bash
# =============================================================================
# setup.sh — Automated setup for g1_xr_locomotion
#
# This script:
#   1. Clones unitree_sim_isaaclab (from our fork, patches already included)
#   2. Creates conda environments from exported yml files
#   3. Installs Python packages for each component (editable mode)
#
# Prerequisites:
#   - Conda (Miniconda/Anaconda) installed
#   - Isaac Sim / Isaac Lab installed (will be in unitree_sim_env)
#   - NVIDIA GPU with appropriate drivers
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_DIR="$SCRIPT_DIR/unitree_sim_isaaclab"

echo "============================================================"
echo "  g1_xr_locomotion — Setup Script"
echo "============================================================"

# ----------------------------------------------------------------
# Step 1: Clone unitree_sim_isaaclab from our fork
#         (all integration patches are already committed)
# ----------------------------------------------------------------
if [ -d "$ISAACLAB_DIR" ]; then
    echo "[INFO] unitree_sim_isaaclab already exists, skipping clone."
else
    echo "[Step 1/3] Cloning unitree_sim_isaaclab (fork with patches)..."
    git clone https://github.com/Kantapia0814/unitree_sim_isaaclab.git "$ISAACLAB_DIR"
    echo "[Step 1/3] Done."
fi

# ----------------------------------------------------------------
# Step 2: Create conda environments from exported yml files
# ----------------------------------------------------------------
echo "[Step 2/3] Creating conda environments..."
echo ""

ENVS_DIR="$SCRIPT_DIR/envs"

if [ -d "$ENVS_DIR" ]; then
    for yml in "$ENVS_DIR"/*.yml; do
        ENV_NAME=$(head -1 "$yml" | sed 's/name: //')
        if conda env list | grep -q "^${ENV_NAME} "; then
            echo "  [INFO] Environment '$ENV_NAME' already exists, skipping."
        else
            echo "  Creating environment '$ENV_NAME' from $(basename "$yml")..."
            conda env create -f "$yml"
            echo "  Done."
        fi
    done
else
    echo "  [WARN] envs/ directory not found. Create environments manually."
fi

echo "[Step 2/3] Done."

# ----------------------------------------------------------------
# Step 3: Install Python packages (editable mode)
# ----------------------------------------------------------------
echo "[Step 3/3] Installing Python packages..."
echo ""
echo "  Please install packages manually in each conda environment:"
echo ""
echo "  # Environment: tv (xr_teleoperate)"
echo "  conda activate tv"
echo "  cd $SCRIPT_DIR/xr_teleoperate/teleop/teleimager && pip install -e . --no-deps"
echo "  cd $SCRIPT_DIR/xr_teleoperate/teleop/televuer && pip install -e ."
echo "  cd $SCRIPT_DIR/xr_teleoperate/teleop/robot_control/dex-retargeting && pip install -e ."
echo "  cd $SCRIPT_DIR/xr_teleoperate && pip install -r requirements.txt"
echo "  cd $SCRIPT_DIR/unitree_sdk2_python && pip install -e ."
echo ""
echo "  # Environment: gr00t_wbc_env (GR00T-WholeBodyControl)"
echo "  conda activate gr00t_wbc_env"
echo "  cd $SCRIPT_DIR/GR00T-WholeBodyControl && pip install -e ."
echo "  cd $SCRIPT_DIR/unitree_sdk2_python && pip install -e ."
echo ""
echo "  # Environment: unitree_sim_env (Isaac Sim)"
echo "  conda activate unitree_sim_env"
echo "  # Follow unitree_sim_isaaclab README for Isaac Lab setup first"
echo "  cd $SCRIPT_DIR/unitree_sdk2_python && pip install -e ."
echo ""
echo "============================================================"
echo "  Setup complete!"
echo "  See README.md for launch instructions."
echo "============================================================"
