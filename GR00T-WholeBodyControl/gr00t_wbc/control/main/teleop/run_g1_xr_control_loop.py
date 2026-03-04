"""
XR Teleoperate Control Loop for GR00T-WBC
==========================================

This module implements a DDS-based control loop that receives wrist poses
from xr_teleoperate (via rt/wrist_poses), solves IK locally using WBC's
own robot state, and combines results with GR00T-WBC's lower body
balance/locomotion policy (G1GearWbcPolicy).

Architecture (3-process):
    Process 1: Isaac Sim (unitree_sim_env)
        - Publishes: rt/lowstate, rt/odostate, rt/secondary_imu
        - Subscribes: rt/lowcmd, rt/dex3/left/cmd, rt/dex3/right/cmd

    Process 2: This control loop (gr00t_wbc_env)
        - Subscribes: rt/wrist_poses (wrist 4x4 poses from xr_teleoperate)
        - Subscribes: rt/lowstate, rt/odostate, rt/secondary_imu (via G1Env)
        - Publishes: rt/lowcmd (unified 29-DOF body command)
        - Solves IK at 50Hz using WBC's own observation for arm state

    Process 3: xr_teleoperate (tv env)
        - Publishes: rt/wrist_poses (left/right 4x4 wrist poses)
        - Publishes: rt/dex3/left/cmd, rt/dex3/right/cmd (hand commands)

Walking control is via keyboard (wasd keys) in this process.
Hand control is handled separately by xr_teleoperate → Isaac Sim.
"""

from copy import deepcopy
from datetime import datetime
import os
from pathlib import Path
import signal
import sys
import time
import threading

import numpy as np
import tyro

from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
    HandCmd_,
    LowCmd_ as hg_LowCmd,
)

# Add xr_teleoperate to path for IK solver import
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
_xr_teleop_root = os.path.join(_project_root, "xr_teleoperate")
if _xr_teleop_root not in sys.path:
    sys.path.insert(0, _xr_teleop_root)

from teleop.robot_control.robot_arm_ik import G1_29_ArmIK

from gr00t_wbc.control.envs.g1.g1_env import G1Env
from gr00t_wbc.control.main.constants import (
    DEFAULT_BASE_HEIGHT,
    DEFAULT_NAV_CMD,
    DEFAULT_WRIST_POSE,
)
from gr00t_wbc.control.main.teleop.configs.configs import ControlLoopConfig
from gr00t_wbc.control.policy.wbc_policy_factory import get_wbc_policy
from gr00t_wbc.control.robot_model.instantiation.g1 import (
    instantiate_g1_robot_model,
)
from gr00t_wbc.control.utils.keyboard_dispatcher import (
    KeyboardDispatcher,
    KeyboardEStop,
    KeyboardListener,
)
from gr00t_wbc.control.utils.telemetry import Telemetry

CONTROL_NODE_NAME = "XRControlPolicy"

# NOTE: obs["q"] is 43-DOF (body 29 + hands 14), NOT 29-DOF.
# In the 43-DOF model: left arm=[15..21], left hand=[22..28], right arm=[29..35], right hand=[36..42]
# We use robot_model.get_joint_group_indices("arms") to get correct indices at runtime.
# (see arm_obs_indices initialization in main())

# Arm velocity safety constants
MAX_ARM_VELOCITY = 10.0  # rad/s max commanded position step (0.2 rad/step at 50Hz)
SAFETY_VELOCITY_LIMIT = 6.0  # JointSafetyMonitor.ARM_VELOCITY_LIMIT
VELOCITY_DAMPING_START = 3.0  # Start reducing commanded delta when actual vel exceeds this


class DDSWristSubscriber:
    """
    Subscribes to rt/wrist_poses DDS topic and extracts left/right wrist poses.

    xr_teleoperate (in --motion mode) publishes LowCmd_ messages to rt/wrist_poses
    with motor_cmd[0..15].q encoding the left wrist 4x4 pose (flattened)
    and motor_cmd[16..31].q encoding the right wrist 4x4 pose (flattened).
    """

    def __init__(self):
        self._subscriber = ChannelSubscriber("rt/wrist_poses", hg_LowCmd)
        self._subscriber.Init(None, 0)
        self._lock = threading.Lock()
        self._last_left_wrist = None
        self._last_right_wrist = None
        self._connected = False

        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        while True:
            msg = self._subscriber.Read()
            if msg is not None:
                left_flat = np.array([msg.motor_cmd[i].q for i in range(16)])
                right_flat = np.array([msg.motor_cmd[16 + i].q for i in range(16)])
                with self._lock:
                    self._last_left_wrist = left_flat.reshape(4, 4)
                    self._last_right_wrist = right_flat.reshape(4, 4)
                    self._connected = True
            time.sleep(0.002)  # 500Hz polling

    @property
    def connected(self):
        with self._lock:
            return self._connected

    def get_wrist_poses(self):
        """
        Get the latest wrist poses from xr_teleoperate.

        Returns:
            Tuple of (left_wrist_4x4, right_wrist_4x4) or (None, None) if no data yet.
        """
        with self._lock:
            if self._last_left_wrist is not None:
                return self._last_left_wrist.copy(), self._last_right_wrist.copy()
            return None, None


HAND_NUM_DOF = 7  # per hand


class DDSHandCmdSubscriber:
    """
    Subscribes to rt/dex3/left/cmd and rt/dex3/right/cmd to capture
    hand commands sent by xr_teleoperate directly to the robot.
    """

    def __init__(self):
        self._left_sub = ChannelSubscriber("rt/dex3/left/cmd", HandCmd_)
        self._left_sub.Init(None, 0)
        self._right_sub = ChannelSubscriber("rt/dex3/right/cmd", HandCmd_)
        self._right_sub.Init(None, 0)
        self._lock = threading.Lock()
        self._left_cmd = None  # dict with q, kp, kd arrays
        self._right_cmd = None

        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        while True:
            left_msg = self._left_sub.Read()
            right_msg = self._right_sub.Read()
            with self._lock:
                if left_msg is not None:
                    self._left_cmd = {
                        "q": np.array([left_msg.motor_cmd[i].q for i in range(HAND_NUM_DOF)]),
                        "kp": np.array([left_msg.motor_cmd[i].kp for i in range(HAND_NUM_DOF)]),
                        "kd": np.array([left_msg.motor_cmd[i].kd for i in range(HAND_NUM_DOF)]),
                    }
                if right_msg is not None:
                    self._right_cmd = {
                        "q": np.array([right_msg.motor_cmd[i].q for i in range(HAND_NUM_DOF)]),
                        "kp": np.array([right_msg.motor_cmd[i].kp for i in range(HAND_NUM_DOF)]),
                        "kd": np.array([right_msg.motor_cmd[i].kd for i in range(HAND_NUM_DOF)]),
                    }
            time.sleep(0.002)

    def get_hand_commands(self):
        """Returns (left_cmd_dict, right_cmd_dict) or (None, None)."""
        with self._lock:
            if self._left_cmd is not None and self._right_cmd is not None:
                return (
                    {k: v.copy() for k, v in self._left_cmd.items()},
                    {k: v.copy() for k, v in self._right_cmd.items()},
                )
            return None, None


# ---------------------------------------------------------------------------
# Camera image subscriber (lightweight ZMQ, no teleimager dependency)
# ---------------------------------------------------------------------------

class ZMQCameraSubscriber:
    """Subscribe to teleimager's ZMQ stream for head camera JPEG frames.

    Connects to tcp://{host}:{port}, receives JPEG bytes, decodes to RGB numpy.
    """

    def __init__(self, host: str, port: int):
        import zmq

        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.RCVHWM, 1)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(f"tcp://{host}:{port}")
        self._sock.setsockopt_string(zmq.SUBSCRIBE, "")
        self._poller = zmq.Poller()
        self._poller.register(self._sock, zmq.POLLIN)

        self._lock = threading.Lock()
        self._latest_frame = None  # RGB uint8 numpy array
        self._running = True

        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        print(f"[Camera] ZMQ subscriber connecting to tcp://{host}:{port}")

    def _recv_loop(self):
        import cv2

        while self._running:
            try:
                events = dict(self._poller.poll(timeout=100))
                if self._sock in events:
                    jpg_bytes = self._sock.recv()
                    np_buf = np.frombuffer(jpg_bytes, dtype=np.uint8)
                    bgr = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
                    if bgr is not None:
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        with self._lock:
                            self._latest_frame = rgb
            except Exception:
                if self._running:
                    time.sleep(0.01)

    def get_frame(self):
        """Returns latest RGB frame (H, W, 3) or None."""
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
            return None

    def close(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self._sock.close()
        self._ctx.term()


# ---------------------------------------------------------------------------
# Data recording helpers
# ---------------------------------------------------------------------------

def _wrist_4x4_to_eef(left_wrist_4x4, right_wrist_4x4):
    """Convert two 4x4 wrist poses to 14D eef vector (pos3+quat4 per hand)."""
    from scipy.spatial.transform import Rotation

    def mat_to_pos_quat(mat):
        pos = mat[:3, 3]
        quat = Rotation.from_matrix(mat[:3, :3]).as_quat()  # x,y,z,w
        # LeRobot convention: w,x,y,z
        quat = np.array([quat[3], quat[0], quat[1], quat[2]])
        return np.concatenate([pos, quat])

    left_pq = mat_to_pos_quat(left_wrist_4x4)
    right_pq = mat_to_pos_quat(right_wrist_4x4)
    return np.concatenate([left_pq, right_pq]).astype(np.float64)


def create_xr_data_exporter(config, robot_model, has_camera=False):
    """Create Gr00tDataExporter for XR teleoperation data collection."""
    from gr00t_wbc.data.exporter import Gr00tDataExporter
    from gr00t_wbc.data.utils import get_dataset_features, get_modality_config

    if config.record_save_path:
        save_path = Path(config.record_save_path)
    else:
        save_path = Path(
            f"./outputs/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-g1-xr-{config.task_name}/"
        )

    features = get_dataset_features(robot_model)

    # Remove image/video features if no camera is available
    if not has_camera:
        features.pop("observation.images.ego_view", None)

    # Remove img_state_delta (not applicable for XR teleop)
    features.pop("observation.img_state_delta", None)

    # Add hand command features (kp, kd, target_q per hand)
    for side in ["left", "right"]:
        for field in ["q", "kp", "kd"]:
            features[f"observation.hand_cmd.{side}_{field}"] = {
                "dtype": "float64",
                "shape": (HAND_NUM_DOF,),
            }

    modality_config = get_modality_config(robot_model)

    exporter = Gr00tDataExporter.create(
        save_root=save_path,
        fps=config.control_frequency,
        features=features,
        modality_config=modality_config,
        task=config.task_name,
        robot_type="g1_29dof",
        vcodec="libx264",
    )
    print(f"[Recording] Data exporter created at: {save_path}")
    print(f"[Recording] Camera recording: {'enabled' if has_camera else 'disabled'}")
    return exporter


def generate_xr_frame(
    obs, wbc_action, left_wrist, right_wrist,
    hand_cmd_sub, robot_model, camera_sub=None,
    nav_cmd=None, height_cmd=None, body_orientation_cmd=None,
):
    """Generate a single data frame for recording.

    Args:
        camera_sub: Optional ZMQCameraSubscriber. If provided and a frame is
            available, ``observation.images.ego_view`` is included.
        nav_cmd: Current navigation command (3D: vx, vy, yaw_rate) from
            G1GearWbcPolicy. If None, uses DEFAULT_NAV_CMD.
        height_cmd: Current base height command from G1GearWbcPolicy.
            If None, uses DEFAULT_BASE_HEIGHT.
        body_orientation_cmd: Current body orientation command (3D: roll, pitch,
            yaw) from G1GearWbcPolicy. If None, uses [0, 0, 0].
    """
    # Build 43-DOF action: wbc_action["q"] is already 43-DOF,
    # but hand joints are default values — overwrite with actual hand commands
    action_43 = np.array(wbc_action["q"], dtype=np.float64)

    left_cmd, right_cmd = hand_cmd_sub.get_hand_commands()
    if left_cmd is not None:
        left_hand_indices = robot_model.get_joint_group_indices("left_hand")
        right_hand_indices = robot_model.get_joint_group_indices("right_hand")
        action_43[left_hand_indices] = left_cmd["q"]
        action_43[right_hand_indices] = right_cmd["q"]

    # Build eef from wrist 4x4 matrices
    if left_wrist is not None:
        eef_14 = _wrist_4x4_to_eef(left_wrist, right_wrist)
    else:
        eef_14 = np.zeros(14, dtype=np.float64)

    if nav_cmd is None:
        nav_cmd = DEFAULT_NAV_CMD
    if height_cmd is None:
        height_cmd = DEFAULT_BASE_HEIGHT
    if body_orientation_cmd is None:
        body_orientation_cmd = [0.0, 0.0, 0.0]

    frame = {
        "observation.state": np.array(obs["q"], dtype=np.float64),
        "observation.eef_state": np.array(
            obs.get("wrist_pose", eef_14), dtype=np.float64
        ),
        "action": action_43,
        "action.eef": eef_14,
        "teleop.navigate_command": np.array(nav_cmd, dtype=np.float64),
        "teleop.base_height_command": np.array(
            [height_cmd], dtype=np.float64
        ),
        "teleop.body_orientation_command": np.array(
            body_orientation_cmd, dtype=np.float64
        ),
    }

    # Camera image (RGB) — blank frame if no image available yet
    if camera_sub is not None:
        img = camera_sub.get_frame()
        if img is None:
            from gr00t_wbc.data.constants import RS_VIEW_CAMERA_HEIGHT, RS_VIEW_CAMERA_WIDTH
            img = np.zeros((RS_VIEW_CAMERA_HEIGHT, RS_VIEW_CAMERA_WIDTH, 3), dtype=np.uint8)
        frame["observation.images.ego_view"] = img

    # Hand commands (kp, kd, target_q)
    if left_cmd is not None:
        for field in ["q", "kp", "kd"]:
            frame[f"observation.hand_cmd.left_{field}"] = left_cmd[field].astype(
                np.float64
            )
            frame[f"observation.hand_cmd.right_{field}"] = right_cmd[field].astype(
                np.float64
            )
    else:
        for side in ["left", "right"]:
            for field in ["q", "kp", "kd"]:
                frame[f"observation.hand_cmd.{side}_{field}"] = np.zeros(
                    HAND_NUM_DOF, dtype=np.float64
                )

    return frame


def main(config: ControlLoopConfig):
    # Signal handling for clean shutdown
    shutdown_event = threading.Event()

    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, shutting down...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Load WBC config
    wbc_config = config.load_wbc_yaml()

    # Override settings for external simulator mode (Isaac Sim via DDS)
    # - "external" prevents MuJoCo simulator creation (SimulatorFactory returns None)
    # - with_hands=False because hands are controlled separately by xr_teleoperate
    #   via rt/dex3/left/cmd and rt/dex3/right/cmd directly to Isaac Sim
    wbc_config["SIMULATOR"] = "external"
    wbc_config["with_hands"] = False

    # Set correct Domain ID based on environment type
    # Isaac Sim uses Domain ID 1, real robot uses Domain ID 0
    if config.env_type == "real":
        wbc_config["DOMAIN_ID"] = 0
    # sim env_type keeps DOMAIN_ID: 1 from yaml

    # Setup robot model
    waist_location = "lower_and_upper_body" if config.enable_waist else "lower_body"
    robot_model = instantiate_g1_robot_model(
        waist_location=waist_location, high_elbow_pose=config.high_elbow_pose
    )

    # Create G1Env for DDS communication with Isaac Sim (no local simulator)
    env = G1Env(
        env_name=config.env_name,
        robot_model=robot_model,
        config=wbc_config,
        wbc_version=config.wbc_version,
    )
    # env.sim is None since SIMULATOR="external", so no simulator to start

    # Create WBC policy (InterpolationPolicy for upper body + G1GearWbcPolicy for lower body)
    wbc_policy = get_wbc_policy("g1", robot_model, wbc_config, config.upper_body_joint_speed)

    # Keyboard dispatcher for walking commands (wasd) and safety (e-stop)
    # NOTE: We don't use KeyboardListenerPublisher since it requires ROS topic publishing
    keyboard_estop = KeyboardEStop()
    keyboard_listener = KeyboardListener()  # for data collection keys (c, x)
    dispatcher = KeyboardDispatcher()
    dispatcher.register(env)
    dispatcher.register(wbc_policy)
    dispatcher.register(keyboard_estop)
    dispatcher.register(keyboard_listener)
    dispatcher.start()

    # Voice control setup (only when --voice is enabled)
    voice_controller = None
    if config.voice:
        from gr00t_wbc.control.utils.voice_controller import VoiceController

        # Initialize AudioClient for robot speaker feedback (real robot only)
        audio_client = None
        if config.env_type == "real":
            try:
                from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

                audio_client = AudioClient()
                audio_client.SetTimeout(10.0)
                audio_client.Init()
                audio_client.SetVolume(80)
                print("[Voice] AudioClient initialized (robot speaker enabled).")
            except Exception as e:
                print(f"[Voice] AudioClient init failed, no robot speaker: {e}")
                audio_client = None

        voice_controller = VoiceController(
            policy=wbc_policy,
            audio_client=audio_client,
            gpu_id=config.voice_gpu_id,
            whisper_model=config.whisper_model,
            qwen_model=config.qwen_model,
            language=config.voice_language,
        )
        voice_controller.start()

    # Data recording setup (only when --record is enabled)
    exporter = None
    hand_cmd_subscriber = None
    camera_subscriber = None
    recording = False
    episode_count = 0
    frame_count = 0
    if config.record:
        # Try to connect to camera and verify it's actually streaming
        try:
            camera_subscriber = ZMQCameraSubscriber(
                config.img_server_ip, config.img_server_port
            )
            # Wait up to 3 seconds for an actual frame
            print("[Recording] Checking camera connection (waiting up to 3s)...")
            for _ in range(30):
                if camera_subscriber.get_frame() is not None:
                    break
                time.sleep(0.1)
            if camera_subscriber.get_frame() is None:
                print("[Recording] No camera frames received, recording without video")
                camera_subscriber.close()
                camera_subscriber = None
        except Exception as e:
            print(f"[Recording] Camera unavailable ({e}), recording without video")
            camera_subscriber = None

        exporter = create_xr_data_exporter(
            config, robot_model, has_camera=(camera_subscriber is not None)
        )
        hand_cmd_subscriber = DDSHandCmdSubscriber()
        print("[Recording] Press 'c' to start/stop recording, 'x' to discard episode")

    # Initialize telemetry for timing measurements
    telemetry = Telemetry(window_size=100)

    # DDS subscriber for wrist poses from xr_teleoperate
    wrist_subscriber = DDSWristSubscriber()

    # IK solver (runs at WBC's 50Hz using WBC's own robot state)
    _xr_assets_dir = os.path.join(_xr_teleop_root, "assets", "g1")
    arm_ik = G1_29_ArmIK(
        urdf_path=os.path.join(_xr_assets_dir, "g1_body29_hand14.urdf"),
        model_dir=_xr_assets_dir,
        cache_path=os.path.join(_xr_assets_dir, "g1_29_model_cache.pkl"),
    )

    # Get joint group information for constructing target_upper_body_pose
    upper_body_indices = robot_model.get_joint_group_indices("upper_body")
    arm_indices = robot_model.get_joint_group_indices("arms")
    num_upper_body = len(upper_body_indices)
    num_arms = len(arm_indices)

    # Arm indices in the 43-DOF obs["q"] (NOT 29-DOF!)
    # arm_indices = [15,16,...,21, 29,30,...,35] for left+right arm in 43-DOF model
    arm_obs_indices = np.array(arm_indices)

    # Compute arm positions within the upper_body array
    # upper_body_indices is sorted ascending, arms come before hands in joint order
    # So arm positions are [0..13] and hand positions are [14..27]
    arm_positions_in_upper_body = []
    for arm_idx in arm_indices:
        pos = upper_body_indices.index(arm_idx)
        arm_positions_in_upper_body.append(pos)

    # Initial upper body pose (from robot model default)
    initial_upper_body_pose = robot_model.get_initial_upper_body_pose()

    print("=" * 60)
    print("  XR Teleoperate Control Loop for GR00T-WBC")
    print("=" * 60)
    print(f"  Upper body joints: {num_upper_body} (arms: {num_arms}, hands: {num_upper_body - num_arms})")
    print(f"  Control frequency: {config.control_frequency} Hz")
    print(f"  Arm control: IK solved locally at {config.control_frequency}Hz using WBC state")
    print(f"  Wrist pose source: rt/wrist_poses from xr_teleoperate")
    print(f"  Arm obs indices in 43-DOF: L={list(arm_obs_indices[:7])}, R={list(arm_obs_indices[7:])}")
    print(f"  Arm mapping: IK(rt/wrist_poses + obs[q][arm_indices]) -> upper_body positions:")
    print(f"    Left arm  -> upper_body[{arm_positions_in_upper_body[:7]}]")
    print(f"    Right arm -> upper_body[{arm_positions_in_upper_body[7:]}]")
    print(f"  Hand positions in upper_body: set to initial pose (controlled separately via Dex3 DDS)")
    print(f"  Simulator: external (Isaac Sim via DDS)")
    print(f"  Interface: {wbc_config.get('INTERFACE', 'N/A')}, Domain ID: {wbc_config.get('DOMAIN_ID', 'N/A')}")
    print("-" * 60)
    print("  Waiting for wrist poses on rt/wrist_poses from xr_teleoperate...")
    print("  Walking control: keyboard (] activate, o deactivate, wasd direction, q/e turn)")
    if config.voice:
        print(f"  Voice control: ENABLED (say 'hey mycroft' to give commands)")
        print(f"    Whisper: {config.whisper_model}, Qwen: {config.qwen_model}, GPU: {config.voice_gpu_id}")
    else:
        print("  Voice control: disabled (use --voice to enable)")
    print("=" * 60)

    loop_dt = 1.0 / config.control_frequency
    arm_connected_logged = False

    try:
        while not shutdown_event.is_set():
            t_start = time.monotonic()

            with telemetry.timer("total_loop"):
                # Step simulator if in sync mode (no-op since sim is None)
                with telemetry.timer("step_simulator"):
                    if env.sim and config.sim_sync_mode:
                        env.step_simulator()

                # Observe robot state from Isaac Sim via DDS
                with telemetry.timer("observe"):
                    obs = env.observe()
                    wbc_policy.set_observation(obs)

                # Get wrist poses from xr_teleoperate, solve IK using WBC's own state
                with telemetry.timer("policy_setup"):
                    t_now = time.monotonic()
                    wbc_goal = {}

                    left_wrist, right_wrist = wrist_subscriber.get_wrist_poses()

                    if left_wrist is not None:
                        if not arm_connected_logged:
                            print("[XR Control Loop] Receiving wrist poses from xr_teleoperate!")
                            arm_connected_logged = True
                            _debug_counter = 0

                        # Extract arm state from WBC's own observation (fresh, 50Hz)
                        # arm_obs_indices = [15..21, 29..35] in 43-DOF model
                        arm_q_from_obs = obs["q"][arm_obs_indices]    # 14D
                        arm_dq_from_obs = obs["dq"][arm_obs_indices]  # 14D

                        # Solve IK using WBC's actual robot state
                        sol_q_raw, _ = arm_ik.solve_ik(
                            left_wrist, right_wrist,
                            arm_q_from_obs, arm_dq_from_obs
                        )

                        # Step 1: Clamp position delta to limit commanded velocity
                        max_delta = MAX_ARM_VELOCITY / config.control_frequency
                        delta = sol_q_raw - arm_q_from_obs
                        delta_clamped = np.clip(delta, -max_delta, max_delta)

                        # Step 2: Velocity-aware damping — reduce step when joint is
                        # already moving fast to prevent PD overshoot triggering safety
                        vel_abs = np.abs(arm_dq_from_obs)
                        damping = np.where(
                            vel_abs > VELOCITY_DAMPING_START,
                            np.clip(
                                (SAFETY_VELOCITY_LIMIT - vel_abs)
                                / (SAFETY_VELOCITY_LIMIT - VELOCITY_DAMPING_START),
                                0.0, 1.0,
                            ),
                            1.0,
                        )
                        delta_clamped *= damping

                        sol_q = arm_q_from_obs + delta_clamped

                        # Debug: log arm data every 50 iterations (~1 second)
                        _debug_counter += 1
                        if _debug_counter % 50 == 1:
                            np.set_printoptions(precision=3, suppress=True)
                            print(f"\n[DEBUG iter={_debug_counter}]")
                            print(f"  L wrist pos: {left_wrist[:3,3]}")
                            print(f"  R wrist pos: {right_wrist[:3,3]}")
                            print(f"  obs L arm:   {arm_q_from_obs[:7]}")
                            print(f"  obs R arm:   {arm_q_from_obs[7:]}")
                            print(f"  IK raw L:    {sol_q_raw[:7]}")
                            print(f"  IK raw R:    {sol_q_raw[7:]}")
                            print(f"  delta L:     {delta[:7]}")
                            print(f"  delta R:     {delta[7:]}")
                            print(f"  clamped L:   {sol_q[:7]}")
                            print(f"  clamped R:   {sol_q[7:]}")

                        # Construct target_upper_body_pose
                        # - Arms (14 joints): from local IK solution
                        # - Hands (14 joints): initial pose (controlled separately via Dex3 DDS)
                        target_upper_body_pose = initial_upper_body_pose.copy()
                        for i, pos in enumerate(arm_positions_in_upper_body):
                            target_upper_body_pose[pos] = sol_q[i]

                        wbc_goal = {
                            "target_upper_body_pose": target_upper_body_pose,
                            "target_time": t_now + loop_dt,
                        }

                    # Send goal to policy (interpolation + lower body)
                    if wbc_goal:
                        wbc_goal["interpolation_garbage_collection_time"] = t_now - 2 * loop_dt
                        wbc_policy.set_goal(wbc_goal)

                # Compute action (InterpolationPolicy for arms + G1GearWbcPolicy for legs)
                with telemetry.timer("policy_action"):
                    wbc_action = wbc_policy.get_action(time=t_now)

                # Send 29-DOF body command to Isaac Sim via rt/lowcmd
                with telemetry.timer("queue_action"):
                    env.queue_action(wbc_action)

                # Data recording (only when --record is enabled)
                if config.record:
                    # Capture current RL policy commands (7D):
                    #   [0:3] cmd (vx, vy, yaw_rate)
                    #   [3]   height_cmd
                    #   [4:7] roll_cmd, pitch_cmd, yaw_cmd
                    lbp = wbc_policy.lower_body_policy
                    current_policy_cmd = np.array([
                        lbp.cmd[0], lbp.cmd[1], lbp.cmd[2],
                        lbp.height_cmd,
                        lbp.roll_cmd, lbp.pitch_cmd, lbp.yaw_cmd,
                    ])

                    key = keyboard_listener.pop_key()
                    if key == "c":
                        if not recording:
                            recording = True
                            frame_count = 0
                            print(f"\n[Recording] Episode {episode_count} STARTED")
                        else:
                            recording = False
                            exporter.save_episode()
                            print(
                                f"[Recording] Episode {episode_count} SAVED "
                                f"({frame_count} frames)"
                            )
                            episode_count += 1
                            frame_count = 0
                    elif key == "x" and recording:
                        recording = False
                        exporter.skip_and_start_new_episode()
                        print(f"[Recording] Episode {episode_count} DISCARDED")
                        frame_count = 0

                    if recording:
                        frame = generate_xr_frame(
                            obs, wbc_action,
                            left_wrist, right_wrist,
                            hand_cmd_subscriber, robot_model,
                            camera_sub=camera_subscriber,
                            nav_cmd=current_policy_cmd[:3],
                            height_cmd=current_policy_cmd[3],
                            body_orientation_cmd=current_policy_cmd[4:7],
                        )
                        exporter.add_frame(frame)
                        frame_count += 1

            # Check simulator health (no-op since sim is None)
            if env.sim and (not env.sim.sim_thread or not env.sim.sim_thread.is_alive()):
                raise RuntimeError("Simulator thread is not alive")

            # Maintain control loop frequency
            end_time = time.monotonic()
            elapsed = end_time - t_start
            sleep_time = max(0, loop_dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Log timing information
            if config.verbose_timing:
                telemetry.log_timing_info(context="XR Control Loop", threshold=0.0)
            elif elapsed > loop_dt and not config.sim_sync_mode:
                telemetry.log_timing_info(context="XR Control Loop Missed", threshold=0.001)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received")
    finally:
        print("Cleaning up...")
        if voice_controller is not None:
            voice_controller.close()
        if camera_subscriber is not None:
            camera_subscriber.close()
        dispatcher.stop()
        env.close()


if __name__ == "__main__":
    config = tyro.cli(ControlLoopConfig)
    main(config)
