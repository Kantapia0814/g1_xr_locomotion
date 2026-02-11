"""
XR Teleoperate Control Loop for GR00T-WBC
==========================================

This module implements a DDS-based control loop that receives arm commands
from xr_teleoperate (via rt/arm_sdk) and combines them with GR00T-WBC's
lower body balance/locomotion policy (G1GearWbcPolicy).

Architecture (3-process):
    Process 1: Isaac Sim (unitree_sim_env)
        - Publishes: rt/lowstate, rt/odostate, rt/secondary_imu
        - Subscribes: rt/lowcmd, rt/dex3/left/cmd, rt/dex3/right/cmd

    Process 2: This control loop (gr00t_wbc_env)
        - Subscribes: rt/arm_sdk (arm joint positions from xr_teleoperate)
        - Subscribes: rt/lowstate, rt/odostate, rt/secondary_imu (via G1Env)
        - Publishes: rt/lowcmd (unified 29-DOF body command)

    Process 3: xr_teleoperate (tv env)
        - Publishes: rt/arm_sdk (14 arm joint positions)
        - Publishes: rt/dex3/left/cmd, rt/dex3/right/cmd (hand commands)
        - Subscribes: rt/lowstate (for joint state feedback)

Walking control is via keyboard (wasd keys) in this process.
Hand control is handled separately by xr_teleoperate → Isaac Sim.
"""

from copy import deepcopy
import signal
import time
import threading

import numpy as np
import tyro

from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as hg_LowCmd

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
)
from gr00t_wbc.control.utils.telemetry import Telemetry

CONTROL_NODE_NAME = "XRControlPolicy"

# Arm motor indices in the DDS LowCmd_ message
# Matches xr_teleoperate's G1_29_JointArmIndex:
#   [15..21] = left arm  (shoulder_pitch, shoulder_roll, shoulder_yaw,
#                          elbow, wrist_roll, wrist_pitch, wrist_yaw)
#   [22..28] = right arm (same order)
ARM_MOTOR_INDICES = list(range(15, 29))


class DDSArmSubscriber:
    """
    Subscribes to rt/arm_sdk DDS topic and extracts arm joint positions.

    xr_teleoperate (in --motion mode) publishes LowCmd_ messages to rt/arm_sdk
    with motor_cmd[15..28] containing arm joint positions (q) for both arms.
    """

    def __init__(self):
        self._subscriber = ChannelSubscriber("rt/arm_sdk", hg_LowCmd)
        self._subscriber.Init(None, 0)
        self._lock = threading.Lock()
        self._last_arm_q = None
        self._connected = False

        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        while True:
            msg = self._subscriber.Read()
            if msg is not None:
                arm_q = np.array([msg.motor_cmd[i].q for i in ARM_MOTOR_INDICES])
                with self._lock:
                    self._last_arm_q = arm_q
                    self._connected = True
            time.sleep(0.002)  # 500Hz polling

    @property
    def connected(self):
        with self._lock:
            return self._connected

    def get_arm_q(self):
        """
        Get the latest arm joint positions from xr_teleoperate.

        Returns:
            np.ndarray of shape (14,) or None if no message received yet.
            Order: [left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw,
                    left_elbow, left_wrist_roll, left_wrist_pitch, left_wrist_yaw,
                    right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw,
                    right_elbow, right_wrist_roll, right_wrist_pitch, right_wrist_yaw]
        """
        with self._lock:
            return self._last_arm_q.copy() if self._last_arm_q is not None else None


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
    dispatcher = KeyboardDispatcher()
    dispatcher.register(env)
    dispatcher.register(wbc_policy)
    dispatcher.register(keyboard_estop)
    dispatcher.start()

    # Initialize telemetry for timing measurements
    telemetry = Telemetry(window_size=100)

    # DDS subscriber for arm commands from xr_teleoperate
    arm_subscriber = DDSArmSubscriber()

    # Get joint group information for constructing target_upper_body_pose
    upper_body_indices = robot_model.get_joint_group_indices("upper_body")
    arm_indices = robot_model.get_joint_group_indices("arms")
    num_upper_body = len(upper_body_indices)
    num_arms = len(arm_indices)

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
    print(f"  Arm mapping: rt/arm_sdk motor[15..28] -> target_upper_body_pose positions:")
    print(f"    Left arm  (motor 15-21) -> upper_body[{arm_positions_in_upper_body[:7]}]")
    print(f"    Right arm (motor 22-28) -> upper_body[{arm_positions_in_upper_body[7:]}]")
    print(f"  Hand positions in upper_body: set to initial pose (controlled separately via Dex3 DDS)")
    print(f"  Simulator: external (Isaac Sim via DDS)")
    print(f"  Interface: {wbc_config.get('INTERFACE', 'N/A')}, Domain ID: {wbc_config.get('DOMAIN_ID', 'N/A')}")
    print("-" * 60)
    print("  Waiting for arm commands on rt/arm_sdk from xr_teleoperate...")
    print("  Walking control: keyboard (] activate, o deactivate, wasd direction, q/e turn)")
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

                # Get arm commands from xr_teleoperate via DDS
                with telemetry.timer("policy_setup"):
                    t_now = time.monotonic()
                    wbc_goal = {}

                    arm_q = arm_subscriber.get_arm_q()

                    if arm_q is not None:
                        if not arm_connected_logged:
                            print("[XR Control Loop] Receiving arm commands from xr_teleoperate!")
                            arm_connected_logged = True

                        # Construct target_upper_body_pose
                        # - Arms (14 joints): from xr_teleoperate's rt/arm_sdk
                        # - Hands (14 joints): initial pose (hands controlled separately via Dex3 DDS)
                        target_upper_body_pose = initial_upper_body_pose.copy()
                        for i, pos in enumerate(arm_positions_in_upper_body):
                            target_upper_body_pose[pos] = arm_q[i]

                        # Build goal for the interpolation policy
                        # NOTE: We intentionally do NOT include navigate_cmd or base_height_command
                        # in the goal. This keeps use_teleop_policy_cmd=False in the lower body
                        # policy, allowing keyboard walking commands (wasd) to work directly
                        # through handle_keyboard_button().
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
        dispatcher.stop()
        env.close()


if __name__ == "__main__":
    config = tyro.cli(ControlLoopConfig)
    main(config)
