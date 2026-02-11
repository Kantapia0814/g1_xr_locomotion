# g1_xr_locomotion

Full-body teleoperation of Unitree G1: XR for arms/hands, GR00T-WBC for lower body.

This project integrates **[xr_teleoperate](https://github.com/unitreerobotics/xr_teleoperate)** (upper body control via XR devices) with **[GR00T-WholeBodyControl](https://github.com/NVIDIA-Omniverse/GR00T-WholeBodyControl)** (lower body locomotion via WBC policy) to achieve full-body teleoperation of the Unitree G1 humanoid robot within **Isaac Sim**.

- **Upper body** (arms + dexterous hands): Controlled in real-time via Apple Vision Pro hand tracking
- **Lower body** (legs + waist): Controlled by NVIDIA GR00T Whole-Body-Control policy with keyboard walking commands
- **Communication**: All three processes coordinate via DDS (CycloneDDS), with no modifications to the core logic of either project

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Prerequisites](#2-prerequisites)
3. [Installation](#3-installation)
4. [Simulation Deployment](#4-simulation-deployment)
5. [Physical Robot Deployment](#5-physical-robot-deployment)
6. [Technical Details](#6-technical-details)
7. [Repository Structure](#7-repository-structure)
8. [Acknowledgements](#8-acknowledgements)

---

## 1. Architecture Overview

The system runs as **three independent processes** communicating over DDS:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Apple Vision Pro                             │
│                   (Hand Tracking + Head Pose)                       │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ WebSocket / WebRTC
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Process 3: xr_teleoperate  (conda: tv)                            │
│                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐                       │
│  │ TeleVuer │→ │ Arm IK   │→ │ DDS Publish│──→ rt/arm_sdk         │
│  │ (VR data)│  │ Solver   │  │            │──→ rt/dex3/left/cmd   │
│  └──────────┘  └──────────┘  │            │──→ rt/dex3/right/cmd  │
│                               └────────────┘                       │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
┌──────────────────────┐  ┌──────────────────────────────────────────┐
│ Process 2: GR00T-WBC │  │  Process 1: Isaac Sim                    │
│ (conda: gr00t_wbc)   │  │  (conda: unitree_sim_env)                │
│                      │  │                                          │
│ Subscribe:           │  │  Subscribes:                             │
│  rt/arm_sdk ─────┐   │  │   rt/lowcmd ──→ Torque Control (body)   │
│  rt/lowstate     │   │  │   rt/dex3/*  ──→ Position Control (hand) │
│  rt/odostate     │   │  │                                          │
│  rt/secondary_imu│   │  │  Publishes:                              │
│                  ▼   │  │   rt/lowstate                            │
│ ┌──────────────────┐ │  │   rt/odostate                           │
│ │ G1GearWbcPolicy  │ │  │   rt/secondary_imu                     │
│ │ (Balance + Walk) │ │  │                                          │
│ │ + Interpolation  │ │  │  Camera → teleimager → Vision Pro        │
│ │   (Arms)         │ │  │                                          │
│ └────────┬─────────┘ │  └──────────────────────────────────────────┘
│          │            │                   ▲
│          ▼            │                   │
│   rt/lowcmd ──────────┼───────────────────┘
│   (29-DOF unified)    │
└──────────────────────┘

Keyboard (wasd): Walking commands → GR00T-WBC
```

| Process | Conda Env | Role | Controls |
|---------|-----------|------|----------|
| Isaac Sim | `unitree_sim_env` | Physics simulation, camera streaming | Robot simulation (PhysX) |
| GR00T-WBC | `gr00t_wbc_env` | Whole-body locomotion + balance | Legs (12 DOF) + Waist (3 DOF) + Arms (14 DOF) |
| xr_teleoperate | `tv` | XR hand tracking, arm IK, hand retargeting | Arms (14 DOF) + Hands (14 DOF) |

---

## 2. Prerequisites

### Hardware
- **GPU**: NVIDIA RTX GPU (for Isaac Sim, see [system requirements](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/requirements.html))
- **XR Device**: Apple Vision Pro, Meta Quest 3, or PICO 4 Ultra Enterprise
- **Router**: For connecting XR device and host PC to the same network

### Software
- Ubuntu 20.04 or 22.04
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- NVIDIA Isaac Sim 4.5.0 (via Isaac Lab)
- Git

---

## 3. Installation

### 3.1 Clone this repository

```bash
git clone https://github.com/Kantapia0814/g1_xr_locomotion.git
cd g1_xr_locomotion
```

### 3.2 Run the setup script

The setup script clones `unitree_sim_isaaclab` and applies our integration patches:

```bash
chmod +x setup.sh
./setup.sh
```

> **Note**: `unitree_sim_isaaclab` requires Isaac Lab to be installed first. Follow the [unitree_sim_isaaclab README](https://github.com/unitreerobotics/unitree_sim_isaaclab) for Isaac Lab setup.

### 3.3 Create conda environments and install packages

#### Environment 1: Isaac Sim (`unitree_sim_env`)

```bash
# Create environment (follow unitree_sim_isaaclab README for Isaac Lab setup)
conda create -n unitree_sim_env python=3.10
conda activate unitree_sim_env
# Install Isaac Lab first (see unitree_sim_isaaclab README)
# Then install unitree_sdk2_python:
cd unitree_sdk2_python
pip install -e .
```

#### Environment 2: GR00T-WBC (`gr00t_wbc_env`)

```bash
conda create -n gr00t_wbc_env python=3.10
conda activate gr00t_wbc_env
cd GR00T-WholeBodyControl
pip install -e .
cd ../unitree_sdk2_python
pip install -e .
```

#### Environment 3: xr_teleoperate (`tv`)

```bash
conda create -n tv python=3.10 pinocchio=3.1.0 numpy=1.26.4 -c conda-forge
conda activate tv

# Install submodules
cd xr_teleoperate/teleop/teleimager
pip install -e . --no-deps

cd ../televuer
pip install -e .

# Generate SSL certificates for XR device connection
# For Pico / Quest:
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout key.pem -out cert.pem

# For Apple Vision Pro (detailed setup):
openssl genrsa -out rootCA.key 2048
openssl req -x509 -new -nodes -key rootCA.key -sha256 -days 365 -out rootCA.pem -subj "/CN=xr-teleoperate"
openssl genrsa -out key.pem 2048
openssl req -new -key key.pem -out server.csr -subj "/CN=localhost"
# Create server_ext.cnf — replace IP.2 with your host IP
cat > server_ext.cnf << 'CERTEOF'
subjectAltName = @alt_names
[alt_names]
DNS.1 = localhost
IP.1 = 192.168.123.164
IP.2 = 192.168.123.2
CERTEOF
openssl x509 -req -in server.csr -CA rootCA.pem -CAkey rootCA.key \
  -CAcreateserial -out cert.pem -days 365 -sha256 -extfile server_ext.cnf
# Copy rootCA.pem to Apple Vision Pro via AirDrop and install it

# Configure certificate paths
mkdir -p ~/.config/xr_teleoperate/
cp cert.pem key.pem ~/.config/xr_teleoperate/

# Install dex-retargeting
cd ../robot_control/dex-retargeting
pip install -e .

# Install remaining dependencies
cd ../../../
pip install -r requirements.txt

# Install unitree_sdk2_python
cd ../unitree_sdk2_python
pip install -e .
```

---

## 4. Simulation Deployment

### 4.1 Launch Order

All three processes must be started **in order**. Open three separate terminals:

#### Terminal 1: Isaac Sim

```bash
conda activate unitree_sim_env
cd unitree_sim_isaaclab

python sim_main.py \
  --task=Isaac-MinimalGround-G129-Dex3-Wholebody \
  --enable_fullbody_dds \
  --enable_cameras \
  --enable_dex3_dds \
  --device cpu \
  --robot_type g129
```

Wait until the simulation window appears and you see `controller started, start main loop...`.

> **Note**: Click once in the simulation window to activate it.

#### Terminal 2: GR00T-WBC

```bash
conda activate gr00t_wbc_env
cd GR00T-WholeBodyControl

python -m gr00t_wbc.control.main.teleop.run_g1_xr_control_loop
```

Wait until you see `Waiting for arm commands on rt/arm_sdk from xr_teleoperate...`.

**Keyboard controls for walking:**

| Key | Action |
|-----|--------|
| `]` | Activate WBC policy (start balancing) |
| `o` | Deactivate policy |
| `w` / `s` | Move forward / backward |
| `a` / `d` | Strafe left / right |
| `q` / `e` | Rotate left / right |

#### Terminal 3: xr_teleoperate

```bash
conda activate tv
cd xr_teleoperate/teleop

python teleop_hand_and_arm.py \
  --sim \
  --arm=G1_29 \
  --ee=dex3 \
  --img-server-ip=localhost
```

> **Important**: Use `--img-server-ip=localhost` in simulation mode.

### 4.2 Connect XR Device

1. Put on your XR headset and connect to the same Wi-Fi network as the host PC
2. Open a browser and navigate to: `https://<HOST_IP>:8012?ws=wss://<HOST_IP>:8012`
   - Replace `<HOST_IP>` with your host machine's IP (check with `ifconfig`)
   - Accept any SSL certificate warnings
3. Click **Virtual Reality** and allow all prompts
4. Align your arms to the robot's initial pose (arms down at sides)
5. Press **`r`** in Terminal 3 to start teleoperation

### 4.3 Additional Options

```bash
# With data recording
python teleop_hand_and_arm.py --sim --arm=G1_29 --ee=dex3 --img-server-ip=localhost --record

# With controller tracking (instead of hand tracking)
python teleop_hand_and_arm.py --sim --arm=G1_29 --ee=dex3 --img-server-ip=localhost --input-mode=controller

# Headless mode (no GUI on xr_teleoperate side)
python teleop_hand_and_arm.py --sim --arm=G1_29 --ee=dex3 --img-server-ip=localhost --headless
```

### 4.4 Exit

1. Position the robot's arms close to the initial pose
2. Press **`q`** in Terminal 3 (xr_teleoperate)
3. Press `Ctrl+C` in Terminal 2 (GR00T-WBC)
4. Close Isaac Sim

---

## 5. Physical Robot Deployment

Physical deployment follows the same 3-process architecture with these differences:

### 5.1 Image Service (on Robot PC2)

```bash
# On the robot's Development Computing Unit (PC2)
# Install and configure teleimager: https://github.com/silencht/teleimager
cd ~/teleimager
python -m teleimager.image_server
```

### 5.2 Launch

#### Terminal 1: GR00T-WBC (on Host PC)

```bash
conda activate gr00t_wbc_env
cd GR00T-WholeBodyControl

python -m gr00t_wbc.control.main.teleop.run_g1_xr_control_loop \
  --interface=real
```

#### Terminal 2: xr_teleoperate (on Host PC)

```bash
conda activate tv
cd xr_teleoperate/teleop

python teleop_hand_and_arm.py \
  --arm=G1_29 \
  --ee=dex3 \
  --motion \
  --img-server-ip=192.168.123.164
```

> **Warning**:
> 1. Keep a safe distance from the robot at all times.
> 2. Ensure the robot is in control mode (via R3 remote) before using `--motion`.
> 3. Align arms to the initial pose before pressing `r`.
> 4. Position arms near the initial pose before pressing `q` to exit.

---

## 6. Technical Details

### 6.1 Sim-to-Sim Migration of GR00T-WBC

GR00T-WholeBodyControl was originally designed to run with its own **MuJoCo** simulator. We performed a **sim-to-sim migration** — replacing MuJoCo with Isaac Sim as the physics backend while preserving the WBC policy logic.

**The Solution: External Simulator Mode**

We created a new control loop (`run_g1_xr_control_loop.py`) that sets `SIMULATOR = "external"`, preventing MuJoCo from being instantiated. `G1Env` communicates with Isaac Sim via DDS exactly as it would with a real robot.

Isaac Sim receives `rt/lowcmd` and computes impedance-control torques internally:

```
torque = tau + kp * (q_target - q_current) + kd * (dq_target - dq_current)
```

**Additional Isaac Sim modifications** (in `patches/unitree_sim_isaaclab/`):
- `action_provider_wh_dds.py`: Full-body torque control mode matching MuJoCo's actuator model
- `odo_imu_dds.py`: New DDS publisher for `rt/odostate` and `rt/secondary_imu`
- `g1_29dof_state.py`: Publishes floating-base odometry and torso IMU for GR00T-WBC
- `unitree.py`: Effort limits aligned with MuJoCo values (50 Nm for waist/ankle)
- New `Isaac-MinimalGround-G129-Dex3-Wholebody` task with camera support

### 6.2 DDS Communication Architecture

All inter-process communication uses **CycloneDDS** with **Domain ID 1** (simulation mode).

| DDS Topic | Message Type | Publisher | Subscriber | Content |
|-----------|-------------|-----------|------------|---------|
| `rt/lowstate` | `LowState_` | Isaac Sim | GR00T-WBC, xr_teleoperate | 29-DOF joint states + IMU |
| `rt/odostate` | `OdoState_` | Isaac Sim | GR00T-WBC | Floating-base odometry |
| `rt/secondary_imu` | `IMUState_` | Isaac Sim | GR00T-WBC | Torso IMU |
| `rt/lowcmd` | `LowCmd_` | GR00T-WBC | Isaac Sim | 29-DOF body command (q, dq, tau, kp, kd) |
| `rt/arm_sdk` | `LowCmd_` | xr_teleoperate | GR00T-WBC | 14 arm joint positions |
| `rt/dex3/left/cmd` | `HandCmd_` | xr_teleoperate | Isaac Sim | 7-DOF left hand commands |
| `rt/dex3/right/cmd` | `HandCmd_` | xr_teleoperate | Isaac Sim | 7-DOF right hand commands |

**Dual Control Mode in Isaac Sim:**
- **Body joints (29 DOF)**: Torque control (PD gains zeroed, torques from GR00T-WBC applied directly)
- **Hand joints (14 DOF)**: Position control (PD gains retained, position targets from xr_teleoperate)

### 6.3 Key Integration Challenges & Solutions

**DDS Topic Conflict Prevention**: xr_teleoperate publishes arm commands to `rt/arm_sdk` (not `rt/lowcmd`). GR00T-WBC merges them with lower-body output and publishes the unified 29-DOF command to `rt/lowcmd`.

**CycloneDDS Fork Safety**: Hand retargeting runs in a child process via `fork()`. CycloneDDS threads don't survive `fork()`, so the main process reads from shared memory and publishes hand commands via `publish_from_main_process()`.

**Head Yaw Compensation**: Inverse yaw rotation applied to wrist position and orientation to correct misalignment between VR headset and robot forward direction.

**Hand Commands in Torque Control Mode**: Modified Isaac Sim's `DDSRLActionProvider` to apply hand position targets alongside body torque targets in every simulation sub-step.

**OdoState/IMU for GR00T-WBC**: Created new `OdoImuDDS` publisher to provide floating-base odometry and torso IMU data that GR00T-WBC's `BodyStateProcessor` requires.

---

## 7. Repository Structure

```
g1_xr_locomotion/
│
├── README.md                       # This file
├── README_ko.md                    # Korean version
├── setup.sh                        # Automated setup (clones isaaclab, applies patches)
├── .gitignore
│
├── xr_teleoperate/                 # Upper body control via XR (modified)
│   ├── assets/                     # Robot URDF files
│   ├── teleop/
│   │   ├── teleop_hand_and_arm.py  # Main script (modified: DDS republish for hands)
│   │   ├── robot_control/
│   │   │   ├── robot_hand_unitree.py   # (modified: publish_from_main_process)
│   │   │   ├── robot_arm_ik.py
│   │   │   └── robot_arm.py
│   │   ├── televuer/
│   │   │   └── src/televuer/
│   │   │       └── tv_wrapper.py       # (modified: head yaw compensation)
│   │   ├── teleimager/             # Image streaming
│   │   └── utils/                  # Recording, filtering, visualization
│   └── requirements.txt
│
├── GR00T-WholeBodyControl/         # Lower body locomotion (modified)
│   └── gr00t_wbc/
│       └── control/main/teleop/
│           └── run_g1_xr_control_loop.py   # NEW: XR control loop for Isaac Sim
│
├── unitree_sdk2_python/            # DDS communication library (modified)
│   └── unitree_sdk2py/idl/
│       ├── default.py              # (modified: OdoState_ factory)
│       └── unitree_hg/msg/dds_/
│           └── _OdoState_.py       # NEW: OdoState IDL dataclass
│
└── patches/
    └── unitree_sim_isaaclab/       # Patches for Isaac Sim (applied by setup.sh)
        ├── sim_main.py             # (modified: --enable_fullbody_dds flag)
        ├── action_provider/
        │   ├── action_provider_wh_dds.py  # (modified: torque control mode)
        │   └── action_provider_dds.py     # (modified: fullbody joint mapping)
        ├── dds/
        │   ├── odo_imu_dds.py      # NEW: OdoState + SecondaryIMU publisher
        │   ├── dds_create.py       # (modified: register OdoImuDDS)
        │   ├── dds_master.py       # (modified: bind to wlo1 interface)
        │   └── g1_robot_dds.py     # (modified: IMU quaternion convention)
        ├── robots/unitree.py       # (modified: effort limits aligned to MuJoCo)
        └── tasks/                  # (modified + new: MinimalGround wholebody task)
```

---

## 8. Acknowledgements

This project builds upon the following open-source projects:

- [xr_teleoperate](https://github.com/unitreerobotics/xr_teleoperate) — Unitree XR teleoperation
- [GR00T-WholeBodyControl](https://github.com/NVIDIA-Omniverse/GR00T-WholeBodyControl) — NVIDIA whole-body control
- [unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab) — Unitree Isaac Lab simulation
- [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python) — Unitree Python DDS SDK
- [OpenTeleVision](https://github.com/OpenTeleVision/TeleVision) — Teleoperation framework
- [dex-retargeting](https://github.com/dexsuite/dex-retargeting) — Dexterous hand retargeting
- [vuer](https://github.com/vuer-ai/vuer) — WebXR framework
- [pinocchio](https://github.com/stack-of-tasks/pinocchio) — Rigid body dynamics library
