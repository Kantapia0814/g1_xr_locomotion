"""
Voice Command Schema for G1 Robot
==================================

Defines the LLM system prompt and JSON parsing for converting
natural language voice commands into structured robot commands.
"""

import json
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class VoiceCommand:
    """Structured output from LLM command parsing."""

    action: str  # walk, turn, stop, height, activate, deactivate, speed, unknown
    direction: Optional[str]  # forward, backward, left, right, up, down, faster, slower
    magnitude: Optional[float]  # 0.0-1.0
    raw_text: str  # Original transcription
    confidence: float  # Parsing confidence


SYSTEM_PROMPT = """\
You are a robot command parser for a G1 humanoid robot.
Convert natural language (Korean or English) into a JSON command.

Available actions:
- "walk": Move the robot. direction: "forward", "backward", "left", "right". magnitude: 0.0-1.0 (0.3=slow, 0.5=normal, 0.8=fast, 1.0=maximum).
- "turn": Rotate the robot. direction: "left", "right". magnitude: 0.0-1.0.
- "stop": Stop all movement. No direction or magnitude needed.
- "activate": Activate the walking policy. No direction or magnitude.
- "deactivate": Deactivate the walking policy. No direction or magnitude.
- "height": Adjust body height. direction: "up", "down". magnitude: 0.0-1.0.
- "speed": Adjust gait speed. direction: "faster", "slower".
- "unknown": If the command does not match any action.

Always respond with ONLY a JSON object:
{"action": "<action>", "direction": "<direction or null>", "magnitude": <float or null>}

Examples:
User: walk forward slowly
{"action": "walk", "direction": "forward", "magnitude": 0.3}

User: 앞으로 걸어
{"action": "walk", "direction": "forward", "magnitude": 0.5}

User: turn left
{"action": "turn", "direction": "left", "magnitude": 0.5}

User: 왼쪽으로 돌아
{"action": "turn", "direction": "left", "magnitude": 0.5}

User: stop
{"action": "stop", "direction": null, "magnitude": null}

User: 멈춰
{"action": "stop", "direction": null, "magnitude": null}

User: 빨리 앞으로 가
{"action": "walk", "direction": "forward", "magnitude": 0.8}

User: 천천히 뒤로 가
{"action": "walk", "direction": "backward", "magnitude": 0.3}

User: 일어서
{"action": "height", "direction": "up", "magnitude": 0.5}

User: 앉아
{"action": "height", "direction": "down", "magnitude": 0.5}

User: 빨리 걸어
{"action": "speed", "direction": "faster", "magnitude": null}

User: 오른쪽으로 이동
{"action": "walk", "direction": "right", "magnitude": 0.5}
"""


# Korean keyword mappings for fallback parsing
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
    "일어": "height_up",
    "높이": "height_up",
    "앉": "height_down",
    "낮": "height_down",
    "빨리": "speed_faster",
    "느리": "speed_slower",
    "천천": "speed_slower",
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


def parse_llm_response(response_text: str, original_text: str) -> VoiceCommand:
    """Parse the LLM's JSON response into a VoiceCommand.

    Falls back to keyword matching if JSON parsing fails.
    """
    # Try to extract JSON from response
    json_match = re.search(r"\{[^}]+\}", response_text)

    if json_match:
        try:
            data = json.loads(json_match.group())
            return VoiceCommand(
                action=data.get("action", "unknown"),
                direction=data.get("direction"),
                magnitude=data.get("magnitude"),
                raw_text=original_text,
                confidence=1.0,
            )
        except json.JSONDecodeError:
            pass

    # Fallback: keyword matching (Korean + English)
    return _keyword_fallback(original_text)


def _keyword_fallback(text: str) -> VoiceCommand:
    """Simple keyword-based command parsing as fallback."""
    text_lower = text.lower()

    # Detect action
    action = None
    direction = None

    # Korean keywords
    for keyword, act in _KO_ACTION_MAP.items():
        if keyword in text_lower:
            if act == "height_up":
                return VoiceCommand("height", "up", 0.5, text, 0.5)
            elif act == "height_down":
                return VoiceCommand("height", "down", 0.5, text, 0.5)
            elif act == "speed_faster":
                return VoiceCommand("speed", "faster", None, text, 0.5)
            elif act == "speed_slower":
                return VoiceCommand("speed", "slower", None, text, 0.5)
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

    if direction is None:
        if "forward" in text_lower or "front" in text_lower:
            direction = "forward"
        elif "back" in text_lower:
            direction = "backward"
        elif "left" in text_lower:
            direction = "left"
        elif "right" in text_lower:
            direction = "right"

    if action == "stop":
        return VoiceCommand("stop", None, None, text, 0.5)
    elif action == "walk":
        return VoiceCommand("walk", direction or "forward", 0.5, text, 0.5)
    elif action == "turn":
        return VoiceCommand("turn", direction or "left", 0.5, text, 0.5)

    return VoiceCommand("unknown", None, None, text, 0.0)
