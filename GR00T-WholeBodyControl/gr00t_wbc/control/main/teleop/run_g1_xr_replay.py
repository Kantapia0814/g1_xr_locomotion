"""
XR Teleoperation Data Replay for Isaac Sim
============================================

Replays recorded LeRobot datasets through the WBC control loop to Isaac Sim.
Feeds recorded arm positions to WBC as upper body targets, while the lower
body RL balance policy maintains balance in real-time. Locomotion commands
(teleop.navigate_command, teleop.base_height_command) from the parquet are
fed back to the RL policy so walking is reproduced.

Startup sequence:
  1. Stabilization (3s) — activate RL balance, hold default pose
  2. Warm-up (3s) — interpolate arms from current to first frame
  3. Replay — arms from data + RL balance with recorded nav/height commands

Usage:
    python -m gr00t_wbc.control.main.teleop.run_g1_xr_replay \
        --dataset-path outputs/2026-02-25-14-32-18-g1-xr-hello_world_4 \
        --episode 2 --interface sim
"""

import os
import signal
import sys
import time
import threading

import numpy as np
import pandas as pd
import tyro

from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_

from gr00t_wbc.control.envs.g1.g1_env import G1Env
from gr00t_wbc.control.envs.g1.utils.command_sender import make_hand_mode
from gr00t_wbc.control.main.teleop.configs.configs import ReplayConfig
from gr00t_wbc.control.policy.wbc_policy_factory import get_wbc_policy
from gr00t_wbc.control.robot_model.instantiation.g1 import (
    instantiate_g1_robot_model,
)
from gr00t_wbc.control.utils.telemetry import Telemetry

HAND_NUM_DOF = 7


def publish_hand_cmd(publisher, q_7, kp_7, kd_7):
    """Publish a single hand command via DDS."""
    msg = unitree_hg_msg_dds__HandCmd_()
    for i in range(HAND_NUM_DOF):
        msg.motor_cmd[i].mode = make_hand_mode(i)
        msg.motor_cmd[i].q = float(q_7[i])
        msg.motor_cmd[i].kp = float(kp_7[i])
        msg.motor_cmd[i].kd = float(kd_7[i])
        msg.motor_cmd[i].dq = 0.0
        msg.motor_cmd[i].tau = 0.0
    publisher.Write(msg)


def load_episode(dataset_path: str, episode: int) -> pd.DataFrame:
    """Load a single episode from a LeRobot dataset."""
    parquet_path = os.path.join(
        dataset_path, "data", "chunk-000", f"episode_{episode:06d}.parquet"
    )
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Episode file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    print(f"[Replay] Loaded episode {episode}: {len(df)} frames")
    return df


def main(config: ReplayConfig):
    if not config.dataset_path:
        print("Error: --dataset-path is required")
        sys.exit(1)

    # Signal handling
    shutdown_event = threading.Event()

    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, shutting down...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Load episode data
    df = load_episode(config.dataset_path, config.episode)

    # Determine which column to replay
    source_col = "action" if config.use_actions else "observation.state"

    # Load locomotion commands from parquet columns
    has_nav_cmd = "teleop.navigate_command" in df.columns
    has_height_cmd = "teleop.base_height_command" in df.columns
    has_orientation_cmd = "teleop.body_orientation_command" in df.columns
    if has_nav_cmd:
        nav_cmds = np.array(df["teleop.navigate_command"].tolist())  # N×3
        print(f"[Replay] Loaded teleop.navigate_command from parquet ({len(nav_cmds)} frames)")
    if has_height_cmd:
        height_cmds = np.array(df["teleop.base_height_command"].tolist())  # N×1
        print(f"[Replay] Loaded teleop.base_height_command from parquet ({len(height_cmds)} frames)")
    if has_orientation_cmd:
        orientation_cmds = np.array(df["teleop.body_orientation_command"].tolist())  # N×3
        print(f"[Replay] Loaded teleop.body_orientation_command from parquet ({len(orientation_cmds)} frames)")

    # Load WBC config (same as control loop)
    wbc_config = config.load_wbc_yaml()
    wbc_config["SIMULATOR"] = "external"
    wbc_config["with_hands"] = False

    if config.env_type == "real":
        wbc_config["DOMAIN_ID"] = 0

    # Setup robot model
    waist_location = "lower_and_upper_body" if config.enable_waist else "lower_body"
    robot_model = instantiate_g1_robot_model(
        waist_location=waist_location, high_elbow_pose=config.high_elbow_pose
    )

    # Create G1Env
    env = G1Env(
        env_name=config.env_name,
        robot_model=robot_model,
        config=wbc_config,
        wbc_version=config.wbc_version,
    )

    # Create WBC policy (RL balance for legs + interpolation for arms)
    wbc_policy = get_wbc_policy(
        "g1", robot_model, wbc_config, config.upper_body_joint_speed
    )

    # Hand command publishers
    left_hand_pub = ChannelPublisher("rt/dex3/left/cmd", HandCmd_)
    left_hand_pub.Init()
    right_hand_pub = ChannelPublisher("rt/dex3/right/cmd", HandCmd_)
    right_hand_pub.Init()

    # Joint group info for mapping arm data → upper body targets
    upper_body_indices = robot_model.get_joint_group_indices("upper_body")
    arm_indices = robot_model.get_joint_group_indices("arms")
    arm_obs_indices = np.array(arm_indices)

    arm_positions_in_upper_body = []
    for arm_idx in arm_indices:
        pos = upper_body_indices.index(arm_idx)
        arm_positions_in_upper_body.append(pos)

    initial_upper_body_pose = robot_model.get_initial_upper_body_pose()

    telemetry = Telemetry(window_size=100)
    loop_dt = 1.0 / config.control_frequency

    # Check which hand command columns exist
    has_hand_cmd = "observation.hand_cmd.left_q" in df.columns

    np.set_printoptions(precision=3, suppress=True)

    print("=" * 60)
    print("  XR Teleoperation Replay")
    print("=" * 60)
    print(f"  Dataset: {config.dataset_path}")
    print(f"  Episode: {config.episode} ({len(df)} frames)")
    print(f"  Source: {source_col}")
    print(f"  Playback speed: {config.playback_speed}x")
    print(f"  Hand commands: {'yes' if has_hand_cmd else 'no'}")
    print(f"  Nav command (wasd/qe): {'yes' if has_nav_cmd else 'no'}")
    print(f"  Height command (1/2): {'yes' if has_height_cmd else 'no'}")
    print(f"  Orientation command (3-8): {'yes' if has_orientation_cmd else 'no'}")
    print(f"  Stabilize: {config.stabilize_duration}s (RL balance)")
    print(f"  Warmup: {config.warmup_duration}s (arm interpolation)")
    print(f"  Replay: arms from data, legs via RL balance")
    print(f"  Loop: {config.loop}")
    print(f"  Interface: {wbc_config.get('INTERFACE', 'N/A')}, "
          f"Domain ID: {wbc_config.get('DOMAIN_ID', 'N/A')}")
    print("-" * 60)
    print("  Press Ctrl+C to stop")
    print("=" * 60)

    total_frames = len(df)

    # Wait for Isaac Sim connection
    print("\n[Replay] Waiting for Isaac Sim data...", flush=True)
    obs = env.observe()
    print("[Replay] Isaac Sim connected!", flush=True)
    wbc_policy.set_observation(obs)

    # =================================================================
    # Phase 1: Stabilization — activate RL balance, hold default pose
    # =================================================================
    stabilize_steps = int(config.stabilize_duration * config.control_frequency)

    if stabilize_steps > 0:
        print(f"[Replay] Phase 1: Stabilization — activating RL balance, "
              f"holding {config.stabilize_duration}s ({stabilize_steps} steps)...")

        # Activate the lower body RL balance policy (equivalent to pressing ']')
        wbc_policy.activate_policy()

        for step in range(stabilize_steps):
            if shutdown_event.is_set():
                break
            t_start = time.monotonic()

            obs = env.observe()
            wbc_policy.set_observation(obs)
            t_now = time.monotonic()

            # No goal — upper body holds initial pose, RL handles legs
            wbc_action = wbc_policy.get_action(time=t_now)
            env.queue_action(wbc_action)

            elapsed = time.monotonic() - t_start
            sleep_time = max(0, loop_dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        print("[Replay] Phase 1 complete — robot balanced.")

    # =================================================================
    # Phase 2: Warm-up — interpolate arms to first frame
    # =================================================================
    first_action = np.array(df.iloc[0][source_col])
    first_arm_q = first_action[arm_obs_indices]

    first_upper_body_target = initial_upper_body_pose.copy()
    for j, pos in enumerate(arm_positions_in_upper_body):
        first_upper_body_target[pos] = first_arm_q[j]

    # Re-read current arm positions after stabilization
    obs = env.observe()
    wbc_policy.set_observation(obs)
    current_arm_q = obs["q"][arm_obs_indices]
    warmup_start_upper = initial_upper_body_pose.copy()
    for j, pos in enumerate(arm_positions_in_upper_body):
        warmup_start_upper[pos] = current_arm_q[j]

    warmup_steps = int(config.warmup_duration * config.control_frequency)

    if warmup_steps > 0:
        print(f"[Replay] Phase 2: Warm-up — interpolating arms to first frame, "
              f"{config.warmup_duration}s ({warmup_steps} steps)...")

        for step in range(warmup_steps):
            if shutdown_event.is_set():
                break
            t_start = time.monotonic()

            obs = env.observe()
            wbc_policy.set_observation(obs)
            t_now = time.monotonic()

            alpha = (step + 1) / warmup_steps
            target = warmup_start_upper + alpha * (first_upper_body_target - warmup_start_upper)

            wbc_goal = {
                "target_upper_body_pose": target,
                "target_time": t_now + loop_dt,
                "interpolation_garbage_collection_time": t_now - 2 * loop_dt,
            }
            wbc_policy.set_goal(wbc_goal)
            wbc_action = wbc_policy.get_action(time=t_now)
            env.queue_action(wbc_action)

            elapsed = time.monotonic() - t_start
            sleep_time = max(0, loop_dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        print("[Replay] Phase 2 complete — arms at first frame pose.")

    # =================================================================
    # Phase 3: Replay — recorded arms via WBC + RL balance for legs
    #          + recorded nav_commands for walking (if available)
    # =================================================================
    print("[Replay] Phase 3: Replaying recorded data...")
    frame_idx = 0

    try:
        while not shutdown_event.is_set():
            t_start = time.monotonic()

            with telemetry.timer("total_loop"):
                with telemetry.timer("observe"):
                    obs = env.observe()
                    wbc_policy.set_observation(obs)

                with telemetry.timer("policy_setup"):
                    t_now = time.monotonic()

                    row = df.iloc[frame_idx]
                    action_43 = np.array(row[source_col])
                    arm_q = action_43[arm_obs_indices]

                    target_upper_body_pose = initial_upper_body_pose.copy()
                    for j, pos in enumerate(arm_positions_in_upper_body):
                        target_upper_body_pose[pos] = arm_q[j]

                    wbc_goal = {
                        "target_upper_body_pose": target_upper_body_pose,
                        "target_time": t_now + loop_dt,
                        "interpolation_garbage_collection_time": t_now - 2 * loop_dt,
                    }
                    wbc_policy.set_goal(wbc_goal)

                    # Set recorded locomotion commands for RL balance policy
                    lbp = wbc_policy.lower_body_policy
                    if has_nav_cmd:
                        lbp.cmd[:] = nav_cmds[frame_idx]  # vx, vy, yaw_rate
                    if has_height_cmd:
                        val = height_cmds[frame_idx]
                        lbp.height_cmd = float(val[0]) if hasattr(val, '__len__') else float(val)
                    if has_orientation_cmd:
                        ori = orientation_cmds[frame_idx]  # roll, pitch, yaw
                        lbp.roll_cmd = float(ori[0])
                        lbp.pitch_cmd = float(ori[1])
                        lbp.yaw_cmd = float(ori[2])

                with telemetry.timer("policy_action"):
                    wbc_action = wbc_policy.get_action(time=t_now)

                with telemetry.timer("queue_action"):
                    env.queue_action(wbc_action)

                # Send hand commands via separate DDS topics
                if has_hand_cmd:
                    left_q = np.array(row["observation.hand_cmd.left_q"])
                    left_kp = np.array(row["observation.hand_cmd.left_kp"])
                    left_kd = np.array(row["observation.hand_cmd.left_kd"])
                    right_q = np.array(row["observation.hand_cmd.right_q"])
                    right_kp = np.array(row["observation.hand_cmd.right_kp"])
                    right_kd = np.array(row["observation.hand_cmd.right_kd"])

                    publish_hand_cmd(left_hand_pub, left_q, left_kp, left_kd)
                    publish_hand_cmd(right_hand_pub, right_q, right_kp, right_kd)

            # Progress
            frame_idx += 1
            if frame_idx % 50 == 0 or frame_idx == total_frames:
                pct = frame_idx / total_frames * 100
                nav_str = ""
                if has_nav_cmd:
                    nav_str = f" cmd={nav_cmds[frame_idx - 1]}"
                print(f"[Replay] Frame {frame_idx}/{total_frames} ({pct:.0f}%){nav_str}")

            # End of episode
            if frame_idx >= total_frames:
                if config.loop:
                    print(f"[Replay] Looping episode {config.episode}")
                    frame_idx = 0
                else:
                    print(f"[Replay] Episode {config.episode} complete")
                    break

            # Maintain timing
            elapsed = time.monotonic() - t_start
            target_dt = loop_dt / config.playback_speed
            sleep_time = max(0, target_dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received")
    finally:
        print("Cleaning up...")
        env.close()


if __name__ == "__main__":
    config = tyro.cli(ReplayConfig)
    main(config)
