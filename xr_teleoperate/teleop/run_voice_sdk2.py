"""
SDK2 Voice Control Script
=========================

Standalone script that uses voice commands (OpenWakeWord + Whisper + Qwen3)
to control the G1 robot via Unitree SDK2 built-in motions.

No GR00T-WBC needed. Directly calls LocoClient and G1ArmActionClient.

Usage:
    conda activate gr00t_wbc_env
    python -m teleop.run_voice_sdk2 --interface eth0
    python -m teleop.run_voice_sdk2 --interface eth0 --gpu-id 0
"""

import argparse
import signal
import sys
import time

# Add unitree_sdk2_python to path
import os

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "unitree_sdk2_python"
    ),
)


def main():
    parser = argparse.ArgumentParser(description="SDK2 Voice Control for G1")
    parser.add_argument(
        "--interface",
        type=str,
        default="eth0",
        help="Network interface for DDS (default: eth0)",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=1,
        help="GPU ID for Whisper and Qwen3 (default: 1)",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="large-v3-turbo",
        help="Whisper model name (default: large-v3-turbo)",
    )
    parser.add_argument(
        "--qwen-model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Qwen3 model name (default: Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="ko",
        help="Language for STT (default: ko)",
    )
    args = parser.parse_args()

    # ---- Step 1: Initialize DDS ----
    print(f"[SDK2] Initializing DDS on interface '{args.interface}'...")
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    ChannelFactoryInitialize(0, args.interface)
    print("[SDK2] DDS initialized (domain 0).")

    # ---- Step 2: Ensure AI mode is active ----
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
        MotionSwitcherClient,
    )

    motion_switcher = MotionSwitcherClient()
    motion_switcher.SetTimeout(5.0)
    motion_switcher.Init()

    code, current_mode = motion_switcher.CheckMode()
    if code == 0:
        print(f"[SDK2] Current mode: {current_mode}")
        if current_mode.get("name") != "ai":
            print("[SDK2] Switching to AI mode...")
            code, _ = motion_switcher.SelectMode("ai")
            print(f"[SDK2] SelectMode('ai') code: {code}")
            time.sleep(2)
        else:
            print("[SDK2] Already in AI mode.")

    # ---- Step 3: Initialize SDK2 clients ----
    print("[SDK2] Initializing LocoClient...")
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

    loco_client = LocoClient()
    loco_client.SetTimeout(10.0)
    loco_client.Init()

    print("[SDK2] LocoClient ready.")
    print("[SDK2] NOTE: Use Unitree Explore app to set robot to 'Walk (Control Waist)' mode first.")

    print("[SDK2] Initializing G1ArmActionClient...")
    from unitree_sdk2py.g1.arm.g1_arm_action_client import (
        G1ArmActionClient,
        action_map,
    )

    arm_client = G1ArmActionClient()
    arm_client.SetTimeout(10.0)
    arm_client.Init()
    print("[SDK2] G1ArmActionClient ready.")

    print("[SDK2] Initializing AudioClient...")
    from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

    audio_client = AudioClient()
    audio_client.SetTimeout(10.0)
    audio_client.Init()
    audio_client.SetVolume(80)
    print("[SDK2] AudioClient ready (volume=80).")

    # ---- Step 4: Start voice controller ----
    from teleop.utils.sdk2_voice_controller import SDK2VoiceController

    voice = SDK2VoiceController(
        loco_client=loco_client,
        arm_client=arm_client,
        arm_action_map=action_map,
        audio_client=audio_client,
        gpu_id=args.gpu_id,
        whisper_model=args.whisper_model,
        qwen_model=args.qwen_model,
        language=args.language,
    )
    voice.start()

    # ---- Step 5: Wait for Ctrl+C ----
    print("\n" + "=" * 50)
    print("  SDK2 Voice Control Active")
    print("  Say 'hey mycroft' followed by a command.")
    print("  Press Ctrl+C to exit.")
    print("=" * 50 + "\n")

    def signal_handler(sig, frame):
        print("\n[SDK2] Shutting down...")
        voice.close()
        print("[SDK2] Done.")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Keep main thread alive
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
