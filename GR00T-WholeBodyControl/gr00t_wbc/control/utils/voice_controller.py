"""
Voice Controller for G1 Robot
==============================

Wake word detection (OpenWakeWord) + Speech-to-Text (Faster-Whisper)
+ LLM command parsing (Qwen3) for hands-free voice control.
Audio feedback (ding + TTS) plays on the robot's built-in speaker
via Unitree SDK2 AudioClient.

Pipeline:
    Microphone (always on) → OpenWakeWord ("hey mycroft")
    → Record until silence (VAD) → Whisper Large v3 Turbo (STT)
    → Qwen3 (LLM) → Robot command (modify lbp.cmd etc.)

Usage:
    voice = VoiceController(policy=wbc_policy, audio_client=audio_client, gpu_id=1)
    voice.start()   # starts background listening thread
    ...
    voice.close()   # cleanup
"""

import asyncio
import math
import struct
import subprocess
import tempfile
import threading
import time
from typing import Optional

import numpy as np

from gr00t_wbc.control.utils.voice_command_schema import (
    SYSTEM_PROMPT,
    VoiceCommand,
    parse_llm_response,
)

# Audio constants
SAMPLE_RATE = 16000  # Whisper expects 16kHz
CHANNELS = 1
CHUNK_1280 = 1280  # 80ms at 16kHz (OpenWakeWord frame size)
SILENCE_THRESHOLD = 500  # RMS amplitude threshold for silence detection
SILENCE_DURATION = 0.7  # seconds of silence to stop recording
MAX_RECORD_SECONDS = 5  # safety limit
WAKEWORD_THRESHOLD = 0.5  # OpenWakeWord confidence threshold


def _generate_ding_pcm() -> bytes:
    """Generate a short 'ding' sound as raw PCM bytes (16kHz, 16-bit, mono)."""
    freq = 1200  # Hz
    duration = 0.15  # seconds
    sample_rate = 16000
    n_samples = int(sample_rate * duration)
    pcm = bytearray()
    for i in range(n_samples):
        t = i / sample_rate
        envelope = math.exp(-t * 20)
        value = int(32767 * envelope * math.sin(2 * math.pi * freq * t))
        value = max(-32768, min(32767, value))
        pcm += struct.pack("<h", value)
    return bytes(pcm)


# Pre-generate ding PCM bytes once at import time
_DING_PCM = _generate_ding_pcm()

# All possible TTS phrases for GR00T-WBC commands
_TTS_PHRASES = [
    "walk forward", "walk backward", "walk left", "walk right",
    "turn left", "turn right",
    "stop",
    "activate", "deactivate",
    "higher", "lower",
    "faster", "slower",
]


def _edge_tts_to_pcm(text: str, voice: str = "en-US-GuyNeural") -> Optional[bytes]:
    """Generate PCM (16kHz, mono, 16-bit) from text using edge-tts + ffmpeg."""
    try:
        import edge_tts

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as mp3_f:
            mp3_path = mp3_f.name

        # edge-tts is async, run in a temporary event loop
        async def _gen():
            comm = edge_tts.Communicate(text, voice, rate="+10%")
            await comm.save(mp3_path)

        asyncio.run(_gen())

        # Convert MP3 → raw PCM with ffmpeg
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", mp3_path,
                "-ar", "16000", "-ac", "1", "-f", "s16le", "-",
            ],
            capture_output=True,
            timeout=10,
        )
        # Clean up temp file
        import os
        try:
            os.unlink(mp3_path)
        except OSError:
            pass

        if result.returncode == 0 and result.stdout:
            return result.stdout
        return None
    except Exception as e:
        print(f"[Voice] edge-tts generation failed for '{text}': {e}")
        return None


def _build_tts_cache(voice: str = "en-US-GuyNeural") -> dict:
    """Pre-generate PCM for all command phrases."""
    cache = {}
    print(f"[Voice] Pre-generating TTS cache ({len(_TTS_PHRASES)} phrases)...")
    for phrase in _TTS_PHRASES:
        pcm = _edge_tts_to_pcm(phrase, voice)
        if pcm:
            cache[phrase] = pcm
    print(f"[Voice] TTS cache ready: {len(cache)}/{len(_TTS_PHRASES)} phrases cached.")
    return cache


class VoiceController:
    """Hands-free voice controller using wake word + STT + LLM.

    Runs as a background thread. Listens for "hey mycroft", then
    records speech, transcribes it, parses the command with an LLM,
    and modifies the lower body policy state directly.
    """

    def __init__(
        self,
        policy,
        audio_client=None,
        gpu_id: int = 1,
        whisper_model: str = "large-v3-turbo",
        qwen_model: str = "Qwen/Qwen3-0.6B",
        language: str = "ko",
    ):
        self._policy = policy
        self._audio = audio_client
        self._gpu_id = gpu_id
        self._language = language
        self._stop_event = threading.Event()
        self._listen_thread = None
        self._pyaudio_instance = None
        self._stream_counter = 0

        # ---- Load models ----
        print("[Voice] Loading OpenWakeWord model...")
        from openwakeword.model import Model as OWWModel

        self._wakeword = OWWModel(wakeword_models=["hey_mycroft"])
        print("[Voice] OpenWakeWord loaded (CPU).")

        print(f"[Voice] Loading Whisper '{whisper_model}' on GPU {gpu_id}...")
        from faster_whisper import WhisperModel

        self._whisper = WhisperModel(
            whisper_model,
            device="cuda",
            device_index=gpu_id,
            compute_type="float16",
        )
        print("[Voice] Whisper loaded.")

        print(f"[Voice] Loading Qwen3 '{qwen_model}' on GPU {gpu_id}...")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(qwen_model)
        self._llm = AutoModelForCausalLM.from_pretrained(
            qwen_model,
            torch_dtype="auto",
            device_map=f"cuda:{gpu_id}",
        )
        print("[Voice] Qwen3 loaded.")

        # Pre-generate TTS audio for all commands (robot speaker)
        if self._audio is not None:
            self._tts_cache = _build_tts_cache()
        else:
            self._tts_cache = {}

        print("[Voice] Voice controller ready. Say 'hey mycroft' to give a command.")

    # ----------------------------------------------------------------
    # Audio feedback (robot speaker via AudioClient)
    # ----------------------------------------------------------------

    def _play_ding(self):
        """Play a short ding sound on the robot speaker."""
        if self._audio is None:
            return
        try:
            self._stream_counter += 1
            stream_id = f"ding_{self._stream_counter}"
            self._audio.PlayStream("voice_ctrl", stream_id, _DING_PCM)
        except Exception as e:
            print(f"[Voice] Ding sound failed: {e}")

    def _speak(self, text: str):
        """Play pre-cached TTS audio on robot speaker."""
        if self._audio is None:
            return
        pcm = self._tts_cache.get(text)
        if pcm is None:
            # Fallback: generate on the fly (slower)
            pcm = _edge_tts_to_pcm(text)
            if pcm is None:
                print(f"[Voice] No TTS audio for '{text}'")
                return
            self._tts_cache[text] = pcm

        try:
            self._stream_counter += 1
            stream_id = f"tts_{self._stream_counter}"
            chunk_size = 96000  # 3s at 16kHz
            for offset in range(0, len(pcm), chunk_size):
                chunk = pcm[offset : offset + chunk_size]
                self._audio.PlayStream("voice_ctrl", stream_id, chunk)
                if offset + chunk_size < len(pcm):
                    time.sleep(0.5)
        except Exception as e:
            print(f"[Voice] Robot TTS failed: {e}")

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def start(self):
        """Start the background listening thread."""
        if self._listen_thread is not None:
            return
        self._stop_event.clear()
        self._listen_thread = threading.Thread(
            target=self._listen_loop, daemon=True
        )
        self._listen_thread.start()

    def close(self):
        """Stop listening and release resources."""
        self._stop_event.set()
        if self._listen_thread is not None:
            self._listen_thread.join(timeout=3)
            self._listen_thread = None
        if self._pyaudio_instance is not None:
            self._pyaudio_instance.terminate()
            self._pyaudio_instance = None

        # Free GPU memory
        import torch

        del self._whisper
        del self._llm
        del self._tokenizer
        torch.cuda.empty_cache()
        print("[Voice] Voice controller shut down.")

    # ----------------------------------------------------------------
    # Background listening loop
    # ----------------------------------------------------------------

    def _listen_loop(self):
        """Main loop: wake word → record → STT → LLM → execute."""
        import pyaudio

        self._pyaudio_instance = pyaudio.PyAudio()

        try:
            stream = self._pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_1280,
            )
        except Exception as e:
            print(f"[Voice] Failed to open microphone: {e}")
            print("[Voice] Make sure a microphone is connected and portaudio is installed.")
            return

        print("[Voice] Microphone opened. Listening for wake word...")

        try:
            while not self._stop_event.is_set():
                # Read 80ms audio frame
                try:
                    data = stream.read(CHUNK_1280, exception_on_overflow=False)
                except Exception:
                    continue

                # Convert to int16 numpy array for OpenWakeWord
                audio_i16 = np.frombuffer(data, dtype=np.int16)

                # Wake word detection
                prediction = self._wakeword.predict(audio_i16)
                score = prediction.get("hey_mycroft", 0)

                if score > WAKEWORD_THRESHOLD:
                    print(f"[Voice] Wake word detected! (confidence: {score:.2f})")
                    self._wakeword.reset()  # reset to avoid re-triggering
                    self._play_ding()

                    # Record until silence
                    audio_frames = self._record_until_silence(stream)

                    if audio_frames:
                        # Process in background to resume wake word listening quickly
                        threading.Thread(
                            target=self._process_audio,
                            args=(audio_frames,),
                            daemon=True,
                        ).start()
                    else:
                        print("[Voice] No speech detected after wake word.")

        except Exception as e:
            print(f"[Voice] Listening loop error: {e}")
        finally:
            stream.stop_stream()
            stream.close()

    # ----------------------------------------------------------------
    # Audio recording with VAD
    # ----------------------------------------------------------------

    def _record_until_silence(self, stream) -> list:
        """Record audio until silence is detected.

        Returns list of raw audio byte frames.
        """
        frames = []
        silence_start = None
        record_start = time.monotonic()

        print("[Voice] Recording... (speak now)")

        while not self._stop_event.is_set():
            try:
                data = stream.read(CHUNK_1280, exception_on_overflow=False)
            except Exception:
                continue

            frames.append(data)

            # Check RMS amplitude for silence detection
            audio_i16 = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_i16.astype(np.float32) ** 2))

            if rms < SILENCE_THRESHOLD:
                if silence_start is None:
                    silence_start = time.monotonic()
                elif time.monotonic() - silence_start > SILENCE_DURATION:
                    print("[Voice] Silence detected, stopping recording.")
                    break
            else:
                silence_start = None

            # Safety timeout
            if time.monotonic() - record_start > MAX_RECORD_SECONDS:
                print("[Voice] Max recording duration reached.")
                break

        duration = time.monotonic() - record_start
        print(f"[Voice] Recorded {duration:.1f}s of audio.")
        return frames

    # ----------------------------------------------------------------
    # STT + LLM pipeline
    # ----------------------------------------------------------------

    def _process_audio(self, frames: list):
        """Transcribe audio, parse command, execute action."""
        try:
            # Convert to float32 for Whisper
            audio_bytes = b"".join(frames)
            audio_np = (
                np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                / 32768.0
            )

            # STT
            t0 = time.monotonic()
            segments, info = self._whisper.transcribe(
                audio_np,
                language=self._language,
                beam_size=5,
                vad_filter=True,
            )
            text = " ".join(seg.text for seg in segments).strip()
            stt_time = time.monotonic() - t0

            if not text:
                print("[Voice] No speech recognized.")
                return

            print(f"[Voice] Heard: '{text}' ({stt_time:.2f}s)")

            # LLM parsing
            t0 = time.monotonic()
            command = self._parse_command(text)
            llm_time = time.monotonic() - t0

            # Build human-readable command name for TTS
            tts_name = self._make_tts_name(command)

            print(
                f"[Voice] Command: {tts_name} "
                f"(mag={command.magnitude}) "
                f"({llm_time:.2f}s)"
            )

            # Speak the command name on robot speaker before executing
            if command.action != "unknown":
                self._speak(tts_name)

            # Execute
            self._execute_command(command)

        except Exception as e:
            print(f"[Voice] Error processing audio: {e}")

    def _parse_command(self, text: str) -> VoiceCommand:
        """Send transcribed text to Qwen3 and parse JSON response."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # Qwen3: skip thinking for speed
        )

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._llm.device)

        import torch

        with torch.no_grad():
            output_ids = self._llm.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.1,
                do_sample=False,
            )

        response = self._tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        return parse_llm_response(response, text)

    @staticmethod
    def _make_tts_name(cmd: VoiceCommand) -> str:
        """Build a human-readable phrase for TTS from a VoiceCommand."""
        if cmd.action == "walk":
            return f"walk {cmd.direction}" if cmd.direction else "walk forward"
        elif cmd.action == "turn":
            return f"turn {cmd.direction}" if cmd.direction else "turn left"
        elif cmd.action == "stop":
            return "stop"
        elif cmd.action == "activate":
            return "activate"
        elif cmd.action == "deactivate":
            return "deactivate"
        elif cmd.action == "height":
            return "higher" if cmd.direction == "up" else "lower"
        elif cmd.action == "speed":
            return "faster" if cmd.direction == "faster" else "slower"
        return cmd.action

    # ----------------------------------------------------------------
    # Command execution
    # ----------------------------------------------------------------

    def _execute_command(self, cmd: VoiceCommand):
        """Translate VoiceCommand into policy state changes."""
        # Get lower_body_policy reference
        lbp = getattr(self._policy, "lower_body_policy", self._policy)

        if cmd.action == "walk":
            magnitude = (cmd.magnitude or 0.5) * 0.6
            if cmd.direction == "forward":
                lbp.cmd[0] = magnitude
            elif cmd.direction == "backward":
                lbp.cmd[0] = -magnitude
            elif cmd.direction == "left":
                lbp.cmd[1] = magnitude
            elif cmd.direction == "right":
                lbp.cmd[1] = -magnitude
            print(f"[Voice] → cmd = {lbp.cmd}")

        elif cmd.action == "turn":
            magnitude = (cmd.magnitude or 0.5) * 0.6
            if cmd.direction == "left":
                lbp.cmd[2] = magnitude
            elif cmd.direction == "right":
                lbp.cmd[2] = -magnitude
            print(f"[Voice] → cmd = {lbp.cmd}")

        elif cmd.action == "stop":
            lbp.cmd[0] = 0.0
            lbp.cmd[1] = 0.0
            lbp.cmd[2] = 0.0
            print("[Voice] → STOP (cmd zeroed)")

        elif cmd.action == "activate":
            lbp.use_policy_action = True
            print("[Voice] → Policy ACTIVATED")

        elif cmd.action == "deactivate":
            lbp.use_policy_action = False
            print("[Voice] → Policy DEACTIVATED")

        elif cmd.action == "height":
            delta = (cmd.magnitude or 0.5) * 0.2
            if cmd.direction == "up":
                lbp.height_cmd += delta
            elif cmd.direction == "down":
                lbp.height_cmd -= delta
            print(f"[Voice] → height_cmd = {lbp.height_cmd}")

        elif cmd.action == "speed":
            if cmd.direction == "faster":
                lbp.freq_cmd = min(2.0, lbp.freq_cmd + 0.2)
            elif cmd.direction == "slower":
                lbp.freq_cmd = max(1.0, lbp.freq_cmd - 0.2)
            print(f"[Voice] → freq_cmd = {lbp.freq_cmd}")

        else:
            print(f"[Voice] → Unknown command: '{cmd.raw_text}'")
