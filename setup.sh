#!/usr/bin/env bash
# =============================================================================
# setup.sh — Automated setup for g1_xr_locomotion
#
# This script:
#   1. Clones unitree_sim_isaaclab and applies our integration patches
#   2. Installs Python packages for each component (editable mode)
#
# Prerequisites:
#   - Conda environments already created (unitree_sim_env, gr00t_wbc_env, tv)
#   - Isaac Sim / Isaac Lab installed in unitree_sim_env
#   - NVIDIA GPU with appropriate drivers
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_DIR="$SCRIPT_DIR/unitree_sim_isaaclab"
PATCHES_DIR="$SCRIPT_DIR/patches/unitree_sim_isaaclab"

echo "============================================================"
echo "  g1_xr_locomotion — Setup Script"
echo "============================================================"

# ----------------------------------------------------------------
# Step 1: Clone unitree_sim_isaaclab (if not already present)
# ----------------------------------------------------------------
if [ -d "$ISAACLAB_DIR" ]; then
    echo "[INFO] unitree_sim_isaaclab already exists, skipping clone."
else
    echo "[Step 1/3] Cloning unitree_sim_isaaclab..."
    git clone https://github.com/unitreerobotics/unitree_sim_isaaclab.git "$ISAACLAB_DIR"
    echo "[Step 1/3] Done."
fi

# ----------------------------------------------------------------
# Step 2: Apply integration patches to unitree_sim_isaaclab
# ----------------------------------------------------------------
echo "[Step 2/3] Applying integration patches to unitree_sim_isaaclab..."

# Copy modified/new files, preserving directory structure
cp -v "$PATCHES_DIR/sim_main.py"                                          "$ISAACLAB_DIR/sim_main.py"
cp -v "$PATCHES_DIR/action_provider/action_provider_wh_dds.py"            "$ISAACLAB_DIR/action_provider/action_provider_wh_dds.py"
cp -v "$PATCHES_DIR/action_provider/action_provider_dds.py"               "$ISAACLAB_DIR/action_provider/action_provider_dds.py"
cp -v "$PATCHES_DIR/dds/dds_create.py"                                    "$ISAACLAB_DIR/dds/dds_create.py"
cp -v "$PATCHES_DIR/dds/dds_master.py"                                    "$ISAACLAB_DIR/dds/dds_master.py"
cp -v "$PATCHES_DIR/dds/g1_robot_dds.py"                                  "$ISAACLAB_DIR/dds/g1_robot_dds.py"
cp -v "$PATCHES_DIR/dds/odo_imu_dds.py"                                   "$ISAACLAB_DIR/dds/odo_imu_dds.py"
cp -v "$PATCHES_DIR/robots/unitree.py"                                    "$ISAACLAB_DIR/robots/unitree.py"
cp -v "$PATCHES_DIR/tasks/common_observations/g1_29dof_state.py"          "$ISAACLAB_DIR/tasks/common_observations/g1_29dof_state.py"
cp -v "$PATCHES_DIR/tasks/common_scene/base_scene_minimal_ground_wholebody.py" "$ISAACLAB_DIR/tasks/common_scene/base_scene_minimal_ground_wholebody.py"
cp -v "$PATCHES_DIR/tasks/g1_tasks/__init__.py"                           "$ISAACLAB_DIR/tasks/g1_tasks/__init__.py"

# New task directory
mkdir -p "$ISAACLAB_DIR/tasks/g1_tasks/minimal_ground_g1_29dof_dex3_wholebody/mdp"
cp -v "$PATCHES_DIR/tasks/g1_tasks/minimal_ground_g1_29dof_dex3_wholebody/__init__.py" \
      "$ISAACLAB_DIR/tasks/g1_tasks/minimal_ground_g1_29dof_dex3_wholebody/"
cp -v "$PATCHES_DIR/tasks/g1_tasks/minimal_ground_g1_29dof_dex3_wholebody/minimal_ground_g1_29dof_dex3_wh_env_cfg.py" \
      "$ISAACLAB_DIR/tasks/g1_tasks/minimal_ground_g1_29dof_dex3_wholebody/"
cp -v "$PATCHES_DIR/tasks/g1_tasks/minimal_ground_g1_29dof_dex3_wholebody/mdp/__init__.py" \
      "$ISAACLAB_DIR/tasks/g1_tasks/minimal_ground_g1_29dof_dex3_wholebody/mdp/"
cp -v "$PATCHES_DIR/tasks/g1_tasks/minimal_ground_g1_29dof_dex3_wholebody/mdp/observations.py" \
      "$ISAACLAB_DIR/tasks/g1_tasks/minimal_ground_g1_29dof_dex3_wholebody/mdp/"
cp -v "$PATCHES_DIR/tasks/g1_tasks/minimal_ground_g1_29dof_dex3_wholebody/mdp/terminations.py" \
      "$ISAACLAB_DIR/tasks/g1_tasks/minimal_ground_g1_29dof_dex3_wholebody/mdp/"
cp -v "$PATCHES_DIR/tasks/g1_tasks/minimal_ground_g1_29dof_dex3_wholebody/mdp/rewards.py" \
      "$ISAACLAB_DIR/tasks/g1_tasks/minimal_ground_g1_29dof_dex3_wholebody/mdp/"

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
