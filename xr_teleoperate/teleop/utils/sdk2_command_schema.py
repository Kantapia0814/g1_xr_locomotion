"""
SDK2 Voice Command Schema
=========================

LLM system prompt and JSON parsing for converting natural language
voice commands into Unitree SDK2 robot commands (locomotion + arm gestures).
"""

import json
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class SDK2VoiceCommand:
    """Structured output from LLM command parsing for SDK2."""

    action: str
    direction: Optional[str]  # forward, backward, left, right
    speed: Optional[float]  # m/s or rad/s for locomotion
    raw_text: str
    confidence: float


SYSTEM_PROMPT = """\
You are a robot command parser for a G1 humanoid robot using Unitree SDK2.
Convert natural language (Korean or English) into a JSON command.

Available actions:

Locomotion:
- "walk": Move the robot. direction: "forward", "backward", "left", "right". speed: 0.1-0.5 (default 0.3).
- "turn": Rotate the robot. direction: "left", "right". speed: 0.1-0.5 (default 0.3).
- "stop": Stop all movement.
- "sit": Sit down.
- "stand_up": Stand up from sitting/squatting.
- "high_stand": Stand tall (maximum height).
- "low_stand": Stand low (minimum height).
- "lie_stand_up": Stand up from lying down.
- "stand_to_squat": Squat down from standing.
- "balance_stand": Enter balance stand mode.

Arm gestures (via LocoClient):
- "wave": Wave hand (without turning).
- "wave_turn": Wave hand while turning around.
- "shake_hand": Shake hand gesture.

Arm gestures (via ArmActionClient):
- "clap": Clap hands.
- "high_five": High five gesture.
- "hug": Hug gesture.
- "heart": Heart gesture with both hands.
- "right_heart": Heart gesture with right hand.
- "hands_up": Raise both hands up.
- "reject": Rejection gesture.
- "right_hand_up": Raise right hand up.
- "x_ray": X-ray pose.
- "high_wave": High wave gesture.
- "two_hand_kiss": Two-hand kiss gesture.
- "left_kiss": Left hand kiss gesture.
- "right_kiss": Right hand kiss gesture.
- "release_arm": Release arms to default position.
- "unknown": If the command does not match any action.

Always respond with ONLY a JSON object:
{"action": "<action>", "direction": "<direction or null>", "speed": <float or null>}

Examples:
User: walk forward
{"action": "walk", "direction": "forward", "speed": 0.3}

User: 앞으로 걸어
{"action": "walk", "direction": "forward", "speed": 0.3}

User: 빨리 앞으로 가
{"action": "walk", "direction": "forward", "speed": 0.5}

User: 천천히 뒤로 가
{"action": "walk", "direction": "backward", "speed": 0.15}

User: 왼쪽으로 돌아
{"action": "turn", "direction": "left", "speed": 0.3}

User: turn right
{"action": "turn", "direction": "right", "speed": 0.3}

User: 멈춰
{"action": "stop", "direction": null, "speed": null}

User: stop
{"action": "stop", "direction": null, "speed": null}

User: 앉아
{"action": "sit", "direction": null, "speed": null}

User: 일어서
{"action": "stand_up", "direction": null, "speed": null}

User: 높이 서
{"action": "high_stand", "direction": null, "speed": null}

User: 낮게 서
{"action": "low_stand", "direction": null, "speed": null}

User: 쪼그려
{"action": "stand_to_squat", "direction": null, "speed": null}

User: 손 흔들어
{"action": "wave", "direction": null, "speed": null}

User: 돌면서 손 흔들어
{"action": "wave_turn", "direction": null, "speed": null}

User: 박수 쳐
{"action": "clap", "direction": null, "speed": null}

User: 하이파이브
{"action": "high_five", "direction": null, "speed": null}

User: 안아줘
{"action": "hug", "direction": null, "speed": null}

User: 하트
{"action": "heart", "direction": null, "speed": null}

User: 오른손 하트
{"action": "right_heart", "direction": null, "speed": null}

User: 만세
{"action": "hands_up", "direction": null, "speed": null}

User: 오른손 들어
{"action": "right_hand_up", "direction": null, "speed": null}

User: 악수
{"action": "shake_hand", "direction": null, "speed": null}

User: 거부
{"action": "reject", "direction": null, "speed": null}

User: 엑스레이
{"action": "x_ray", "direction": null, "speed": null}

User: 뽀뽀
{"action": "two_hand_kiss", "direction": null, "speed": null}

User: 왼손 키스
{"action": "left_kiss", "direction": null, "speed": null}

User: 오른손 키스
{"action": "right_kiss", "direction": null, "speed": null}

User: 하이 웨이브
{"action": "high_wave", "direction": null, "speed": null}

User: 높이 흔들어
{"action": "high_wave", "direction": null, "speed": null}

User: 오른쪽으로 이동
{"action": "walk", "direction": "right", "speed": 0.3}
"""


# Korean keyword mappings for fallback parsing
# Compound patterns (more specific) must come before simple ones
_KO_ACTION_MAP_COMPOUND = {
    "오른손 하트": "right_heart",
    "오른손 키스": "right_kiss",
    "왼손 키스": "left_kiss",
    "오른손 들어": "right_hand_up",
    "돌면서 흔들": "wave_turn",
    "하이 웨이브": "high_wave",
    "높이 흔들": "high_wave",
}

_KO_ACTION_MAP = {
    "멈": "stop",
    "정지": "stop",
    "스톱": "stop",
    "걸어": "walk",
    "이동": "walk",
    "가": "walk",
    "와": "walk",
    "돌아": "turn",
    "회전": "turn",
    "턴": "turn",
    "앉": "sit",
    "일어": "stand_up",
    "높이": "high_stand",
    "낮": "low_stand",
    "쪼그": "stand_to_squat",
    "흔들": "wave",
    "인사": "wave",
    "박수": "clap",
    "하이파이브": "high_five",
    "안아": "hug",
    "하트": "heart",
    "만세": "hands_up",
    "거부": "reject",
    "싫": "reject",
    "악수": "shake_hand",
    "엑스레이": "x_ray",
    "뽀뽀": "two_hand_kiss",
    "키스": "two_hand_kiss",
}

_KO_DIRECTION_MAP = {
    "앞": "forward",
    "전진": "forward",
    "뒤": "backward",
    "후진": "backward",
    "왼": "left",
    "좌": "left",
    "오른": "right",
    "우": "right",
}


def parse_llm_response(response_text: str, original_text: str) -> SDK2VoiceCommand:
    """Parse the LLM's JSON response into an SDK2VoiceCommand.

    Falls back to keyword matching if JSON parsing fails.
    """
    json_match = re.search(r"\{[^}]+\}", response_text)

    if json_match:
        try:
            data = json.loads(json_match.group())
            return SDK2VoiceCommand(
                action=data.get("action", "unknown"),
                direction=data.get("direction"),
                speed=data.get("speed"),
                raw_text=original_text,
                confidence=1.0,
            )
        except json.JSONDecodeError:
            pass

    return _keyword_fallback(original_text)


def _keyword_fallback(text: str) -> SDK2VoiceCommand:
    """Simple keyword-based command parsing as fallback."""
    text_lower = text.lower()

    action = None
    direction = None

    # Korean compound keywords first (more specific)
    for keyword, act in _KO_ACTION_MAP_COMPOUND.items():
        if keyword in text_lower:
            action = act
            break

    # Korean simple keywords
    if action is None:
        for keyword, act in _KO_ACTION_MAP.items():
            if keyword in text_lower:
                action = act
                break

    # Korean direction
    for keyword, dirn in _KO_DIRECTION_MAP.items():
        if keyword in text_lower:
            direction = dirn
            break

    # English keywords
    if action is None:
        if "stop" in text_lower:
            action = "stop"
        elif "walk" in text_lower or "go" in text_lower or "move" in text_lower:
            action = "walk"
        elif "turn" in text_lower or "rotate" in text_lower:
            action = "turn"
        elif "sit" in text_lower:
            action = "sit"
        elif "squat" in text_lower:
            action = "stand_to_squat"
        elif "stand" in text_lower:
            action = "stand_up"
        elif "wave" in text_lower:
            action = "wave"
        elif "clap" in text_lower:
            action = "clap"
        elif "high five" in text_lower:
            action = "high_five"
        elif "hug" in text_lower:
            action = "hug"
        elif "heart" in text_lower:
            action = "heart"
        elif "hands up" in text_lower:
            action = "hands_up"
        elif "reject" in text_lower:
            action = "reject"
        elif "shake" in text_lower:
            action = "shake_hand"
        elif "kiss" in text_lower:
            action = "two_hand_kiss"
        elif "x-ray" in text_lower or "xray" in text_lower:
            action = "x_ray"

    if direction is None:
        if "forward" in text_lower or "front" in text_lower:
            direction = "forward"
        elif "back" in text_lower:
            direction = "backward"
        elif "left" in text_lower:
            direction = "left"
        elif "right" in text_lower:
            direction = "right"

    if action is None:
        return SDK2VoiceCommand("unknown", None, None, text, 0.0)

    if action == "stop":
        return SDK2VoiceCommand("stop", None, None, text, 0.5)
    elif action == "walk":
        return SDK2VoiceCommand("walk", direction or "forward", 0.3, text, 0.5)
    elif action == "turn":
        return SDK2VoiceCommand("turn", direction or "left", 0.3, text, 0.5)
    else:
        # Gesture or posture commands (no direction/speed needed)
        return SDK2VoiceCommand(action, None, None, text, 0.5)
