"""
qwen3_tts.py
────────────
Local Qwen3-TTS engine for RedditVideoMakerBot.
ONE file. Works on iMac M4 and Linux. Auto-detects backend.

BACKENDS (detected once at import, in priority order):
──────────────────────────────────────────────────────
  mlx        → iMac M4, mlx_audio installed        ~1-3s/sentence  ✅ BEST
  mlx_bridge → iMac M4, mlx in venv_mlx/           ~2-4s/sentence
  mps        → iMac M4, PyTorch only               ~3-6s/sentence
  qwen       → Linux, qwen-tts package             ~1-2s/sentence on GPU
  cuda       → Linux, PyTorch + NVIDIA GPU         ~1-2s/sentence
  cpu        → anything, PyTorch CPU               ~15-40s/sentence
  gtts       → internet fallback, always works

VOICE SELECTION (in priority order):
──────────────────────────────────────
  1. voice_design=true + voice_prompt  → natural language voice design
  2. reference_audio set + file exists → voice cloning
  3. random_voice=True                 → random English speaker
  4. QWEN_VOICE_MAP[sentiment]         → auto per sentiment
  5. config voice_name                 → manual override
  6. "Ryan"                            → hardcoded fallback

MODEL LOADING:
──────────────
  Loaded ONCE, cached for entire run. Never reloads between sentences.
  _model_failed sentinel: if load fails once, gTTS is used for ALL
  subsequent sentences without retrying (no spam).
"""

import platform
import re
import subprocess
import time
import random
from pathlib import Path

from utils import settings
from utils.console import print_substep


# ─────────────────────────────────────────────────────────────────────────────
# Platform helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"

def _is_linux() -> bool:
    return platform.system() == "Linux"


# ─────────────────────────────────────────────────────────────────────────────
# Backend detection — runs once at import time
# ─────────────────────────────────────────────────────────────────────────────

def _detect_backend() -> str:
    # 1. MLX native — best option on iMac M4
    if _is_apple_silicon():
        try:
            import mlx_audio  # noqa
            import mlx.core   # noqa
            return "mlx"
        except ImportError:
            pass

    # 2. MLX via subprocess bridge — mlx in separate venv_mlx/ due to dep conflicts
    if _is_apple_silicon():
        root = Path(__file__).parent.parent
        if (root / "venv_mlx").exists() and (root / "mlx_bridge.py").exists():
            return "mlx_bridge"

    # 3. PyTorch MPS — Apple Silicon without mlx_audio
    if _is_apple_silicon():
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass

    # 4. qwen-tts package — Linux primary path (handles CUDA internally)
    try:
        import qwen_tts  # noqa
        return "qwen"
    except ImportError:
        pass

    # 5. PyTorch CUDA — Linux NVIDIA GPU, no qwen-tts
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass

    # 6. PyTorch CPU — slow but works anywhere
    try:
        import torch  # noqa
        return "cpu"
    except ImportError:
        pass

    # 7. gTTS — internet fallback
    return "gtts"


_BACKEND = _detect_backend()

_BACKEND_LABELS = {
    "mlx":        "MLX — Apple Silicon native 🍎 (~1-3s/sentence)",
    "mlx_bridge": "MLX via subprocess bridge 🍎 (~2-4s/sentence)",
    "mps":        "PyTorch MPS — Apple GPU (~3-6s/sentence)",
    "qwen":       "qwen-tts — Linux official pipeline (~1-2s/sentence GPU)",
    "cuda":       "PyTorch CUDA — NVIDIA GPU (~1-2s/sentence)",
    "cpu":        "PyTorch CPU (~15-40s/sentence)",
    "gtts":       "gTTS — internet fallback",
}


# ─────────────────────────────────────────────────────────────────────────────
# Model ID remapping
# mlx-community/ models are MLX-quantized — transformers/qwen-tts can't load them.
# Auto-remap to Qwen/ HF originals when MLX is not the active backend.
# ─────────────────────────────────────────────────────────────────────────────

_MLX_TO_HF = {
    "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit":        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit":        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}

def _resolve_model_id(model_id: str) -> str:
    if _BACKEND == "mlx":
        return model_id  # MLX loads mlx-community/ natively
    remapped = _MLX_TO_HF.get(model_id)
    if remapped:
        print_substep(f"Model remapped: {model_id.split('/')[-1]} → {remapped.split('/')[-1]}", style="dim")
        return remapped
    return model_id


# ─────────────────────────────────────────────────────────────────────────────
# Model singleton
# ─────────────────────────────────────────────────────────────────────────────

_model_instance  = None
_model_loaded_id = None
_model_failed    = False  # True after first failure — never retry


def _get_model(model_id: str):
    global _model_instance, _model_loaded_id, _model_failed

    if _model_failed:
        return None

    if _model_instance is not None and _model_loaded_id == model_id:
        return _model_instance

    if _model_instance is not None and _model_loaded_id != model_id:
        print_substep(f"Switching model: {_model_loaded_id} → {model_id}", style="yellow")
        _model_instance  = None
        _model_loaded_id = None
        _model_failed    = False

    resolved_id = _resolve_model_id(model_id)

    print_substep(f"Loading Qwen3-TTS [{_BACKEND_LABELS.get(_BACKEND, _BACKEND)}]", style="bold blue")
    print_substep(f"Model: {resolved_id}", style="dim")

    try:
        if _BACKEND == "mlx":
            _model_instance = _load_mlx(model_id)
        elif _BACKEND == "mps":
            _model_instance = _load_torch(resolved_id, device="mps")
        elif _BACKEND == "qwen":
            _model_instance = _load_qwen_tts(resolved_id)
        elif _BACKEND == "cuda":
            _model_instance = _load_torch(resolved_id, device="cuda")
        elif _BACKEND == "cpu":
            _model_instance = _load_torch(resolved_id, device="cpu")
        else:
            _model_failed = True
            return None

        if _model_instance is not None:
            _model_loaded_id = model_id
            print_substep("Model loaded ✓", style="bold green")
        else:
            _model_failed = True

        return _model_instance

    except Exception as e:
        hints = {
            "mlx":  "pip install mlx mlx-audio",
            "mps":  "pip install torch",
            "qwen": "pip install 'qwen-tts==0.1.1' 'transformers==4.57.3' 'huggingface-hub<1.0'",
            "cuda": "pip install torch (CUDA wheel from pytorch.org)",
            "cpu":  "pip install torch",
        }
        print_substep(
            f"Model load failed ({_BACKEND}): {e}\n"
            f"  Fix: {hints.get(_BACKEND, '')}\n"
            f"  Using gTTS for this entire run.",
            style="yellow",
        )
        _model_failed = True
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Loaders — one per backend
# ─────────────────────────────────────────────────────────────────────────────

def _load_mlx(model_id: str):
    """iMac M4 — MLX native. Fastest."""
    from mlx_audio.tts.utils import load_model
    return load_model(model_id)


def _load_qwen_tts(model_id: str):
    """
    Linux — official qwen-tts package.
    API: Qwen3TTSModel.from_pretrained(model_id)
         model.generate_custom_voice(text, language, speaker) → (wavs, sr)
         model.generate_voice_clone(text, voice_sample_path, language) → (wavs, sr)
    """
    from qwen_tts import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained(model_id)
    return {"type": "qwen_tts", "model": model}


def _load_torch(model_id: str, device: str):
    """PyTorch loader for MPS / CUDA / CPU."""
    import torch
    dtype_map = {"mps": torch.bfloat16, "cuda": torch.float16, "cpu": torch.float32}
    dtype = dtype_map.get(device, torch.float32)
    print_substep(f"PyTorch: device={device} dtype={dtype}", style="dim")

    try:
        from transformers import Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor
        model = Qwen3TTSForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device, attn_implementation="sdpa"
        )
        processor = Qwen3TTSProcessor.from_pretrained(model_id)
        model.eval()
        return {"type": "torch_qwen3tts", "model": model, "processor": processor, "device": device}
    except (ImportError, AttributeError):
        pass

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.eval()
    return {"type": "torch_auto", "model": model, "tokenizer": tokenizer, "device": device}


# ─────────────────────────────────────────────────────────────────────────────
# Generators — one per backend, all return True/False, never raise
# ─────────────────────────────────────────────────────────────────────────────

def _generate_mlx(model, text: str, filepath: str, voice_cfg: dict) -> bool:
    """
    iMac M4 — MLX native generation via mlx_audio.

    Correct mlx_audio API (confirmed from PyPI docs):
        results = list(model.generate(text=..., voice=..., language=...))
        audio_mx = results[0].audio   # mx.array — NOT a plain numpy array
        audio_np = np.array(audio_mx) # convert after mx.eval()

    model.generate() returns an iterator of GenerationResult objects.
    Each GenerationResult has a .audio attribute (MLX array).
    voice= is the correct kwarg (NOT speaker=).
    """
    try:
        import numpy as np
        import mlx.core as mx
        import soundfile as sf

        speaker   = voice_cfg.get("speaker", "Ethan")
        v_design  = voice_cfg.get("voice_design", False)
        v_prompt  = voice_cfg.get("voice_prompt", "")
        ref_audio = voice_cfg.get("reference_audio", "")

        gen_kwargs = {
            "text":     text,
            "language": "English",
        }

        if v_design and v_prompt:
            gen_kwargs["voice_design"] = v_prompt
        elif ref_audio and Path(ref_audio).exists():
            gen_kwargs["ref_audio"] = ref_audio
        else:
            gen_kwargs["voice"] = speaker  # mlx_audio uses voice=, NOT speaker=

        # Consume the generator — returns list of GenerationResult objects
        results = list(model.generate(**gen_kwargs))

        if not results:
            print_substep("MLX generation returned empty results", style="yellow")
            return False

        # Extract audio from first GenerationResult — it has a .audio attribute
        audio_mx = results[0].audio
        mx.eval(audio_mx)  # force MLX lazy evaluation before numpy conversion
        audio_np = np.array(audio_mx).astype(np.float32)

        # Flatten to 1D if needed
        if audio_np.ndim > 1:
            audio_np = audio_np.flatten()

        if len(audio_np) == 0:
            print_substep("MLX generation produced empty audio array", style="yellow")
            return False

        # Normalize
        peak = np.abs(audio_np).max()
        if peak > 0:
            audio_np = audio_np / peak * 0.95

        wav_path = filepath.replace(".mp3", "_tmp.wav")
        sf.write(wav_path, audio_np, samplerate=24000)
        _wav_to_mp3(wav_path, filepath)
        return True

    except Exception as e:
        print_substep(f"MLX generation error: {e}", style="yellow")
        return False


def _generate_qwen_tts(model_wrapper: dict, text: str, filepath: str, voice_cfg: dict) -> bool:
    """
    Linux — qwen-tts package generation.
    API: generate_custom_voice(text, language, speaker) → (wavs, sr)
         generate_voice_clone(text, voice_sample_path, language) → (wavs, sr)
    wavs is a list of numpy arrays — use wavs[0].
    """
    try:
        import numpy as np
        import soundfile as sf

        model     = model_wrapper["model"]
        speaker   = voice_cfg.get("speaker", "Ryan")
        ref_audio = voice_cfg.get("reference_audio", "")

        if ref_audio and Path(ref_audio).exists():
            wavs, sample_rate = model.generate_voice_clone(
                text=text,
                voice_sample_path=ref_audio,
                language="English",
            )
        else:
            wavs, sample_rate = model.generate_custom_voice(
                text=text,
                language="English",
                speaker=speaker,
            )

        # wavs is a list of numpy arrays — take the first one
        audio_np = wavs[0].astype(np.float32)

        # Normalize
        peak = np.abs(audio_np).max()
        if peak > 0:
            audio_np = audio_np / peak * 0.95

        wav_path = filepath.replace(".mp3", "_tmp.wav")
        sf.write(wav_path, audio_np, samplerate=sample_rate)
        _wav_to_mp3(wav_path, filepath)
        return True

    except Exception as e:
        print_substep(f"qwen-tts generation error: {e}", style="yellow")
        return False


def _generate_torch(model_wrapper: dict, text: str, filepath: str, voice_cfg: dict) -> bool:
    """PyTorch generation for MPS / CUDA / CPU backends."""
    try:
        import torch
        import numpy as np
        import soundfile as sf

        model_type = model_wrapper["type"]
        device     = model_wrapper["device"]
        speaker    = voice_cfg.get("speaker", "Ryan")

        if model_type == "torch_qwen3tts":
            model     = model_wrapper["model"]
            processor = model_wrapper["processor"]
            prompt    = f"<|speaker|>{speaker}<|/speaker|>{text}"
            inputs    = processor(text=prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=2048)
            if hasattr(processor, "decode_audio"):
                audio_np = processor.decode_audio(output).cpu().numpy().astype(np.float32)
            else:
                audio_np = output[0].cpu().float().numpy()

        elif model_type == "torch_auto":
            model     = model_wrapper["model"]
            tokenizer = model_wrapper["tokenizer"]
            inputs    = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=2048)
            audio_np = output[0].cpu().float().numpy()

        else:
            return False

        # Flatten to 1D if needed
        if audio_np.ndim > 1:
            audio_np = audio_np.flatten()

        # Normalize
        peak = np.abs(audio_np).max()
        if peak > 0:
            audio_np = audio_np / peak * 0.95

        wav_path = filepath.replace(".mp3", "_tmp.wav")
        sf.write(wav_path, audio_np.astype(np.float32), samplerate=24000)
        _wav_to_mp3(wav_path, filepath)
        return True

    except Exception as e:
        print_substep(f"PyTorch generation error ({model_wrapper.get('device', '?')}): {e}", style="yellow")
        return False


def _generate_mlx_bridge(text: str, filepath: str, voice_cfg: dict) -> bool:
    """
    MLX via subprocess bridge.
    Used when mlx is in venv_mlx/ due to dep conflicts with whisperx in main venv.
    """
    try:
        root        = Path(__file__).parent.parent
        bridge_py   = root / "mlx_bridge.py"
        venv_python = root / "venv_mlx" / "bin" / "python"
        model_id    = settings.config.get("qwen_tts", {}).get(
            "model", "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit"
        )
        speaker      = voice_cfg.get("speaker", "Ryan")
        voice_design = voice_cfg.get("voice_prompt", "")
        ref_audio    = voice_cfg.get("reference_audio", "")
        wav_path     = filepath.replace(".mp3", "_tmp.wav")

        cmd = [str(venv_python), str(bridge_py),
               "--text", text, "--speaker", speaker,
               "--output", wav_path, "--model", model_id]
        if voice_design:
            cmd += ["--voice-design", voice_design]
        if ref_audio:
            cmd += ["--reference-audio", ref_audio]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            print_substep(f"MLX bridge error: {result.stderr[:200]}", style="yellow")
            return False
        if not Path(wav_path).exists():
            return False

        _wav_to_mp3(wav_path, filepath)
        return True

    except subprocess.TimeoutExpired:
        print_substep("MLX bridge timed out (120s)", style="yellow")
        return False
    except Exception as e:
        print_substep(f"MLX bridge failed: {e}", style="yellow")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# WAV → MP3 via FFmpeg (already a pipeline dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _wav_to_mp3(wav_path: str, mp3_path: str) -> None:
    result = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", wav_path, "-ar", "44100", "-ab", "192k", mp3_path],
        capture_output=True,
    )
    try:
        Path(wav_path).unlink(missing_ok=True)
    except Exception:
        pass
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg WAV→MP3 failed: {result.stderr.decode()[:300]}")


# ─────────────────────────────────────────────────────────────────────────────
# Instruction prefix stripper
# ─────────────────────────────────────────────────────────────────────────────

def _strip_instruction(text: str) -> str:
    """Strips [instruction] prefix written by DeepSeek. Qwen would read it literally."""
    return re.sub(r'^\[[^\]]+\]\s*', '', text).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Voice resolution
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_voice(random_voice: bool) -> dict:
    cfg = settings.config.get("qwen_tts", {})
    base = {
        "speaker":         "Ethan",  # valid mlx_audio voices: Ethan, Chelsie, Vivian
        "voice_design":    False,
        "voice_prompt":    "",
        "reference_audio": cfg.get("reference_audio", ""),
    }

    # 1. Voice design mode
    if cfg.get("voice_design", False) and cfg.get("voice_prompt", "").strip():
        base["voice_design"] = True
        base["voice_prompt"] = cfg["voice_prompt"].strip()
        return base

    # 2. Voice cloning
    ref = cfg.get("reference_audio", "").strip()
    if ref and Path(ref).exists():
        base["reference_audio"] = ref
        return base

    # 3. Random
    if random_voice:
        try:
            from utils.sentiment_map import QWEN_SPEAKERS
            base["speaker"] = random.choice(QWEN_SPEAKERS)
        except Exception:
            pass
        return base

    # 4. Sentiment-aware
    try:
        from utils.sentiment_map import QWEN_VOICE_MAP
        sentiment = settings.config["settings"].get("sentiment", "dramatic")
        if sentiment in QWEN_VOICE_MAP:
            base["speaker"] = QWEN_VOICE_MAP[sentiment]
            return base
    except Exception:
        pass

    # 5. Manual config
    voice_name = cfg.get("voice_name", "").strip()
    if voice_name:
        base["speaker"] = voice_name
        return base

    # 6. Fallback
    return base


# ─────────────────────────────────────────────────────────────────────────────
# gTTS fallback — internet, always works
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_gtts(text: str, filepath: str) -> None:
    try:
        from gtts import gTTS
        clean = _strip_instruction(text)
        if not clean:
            return
        gTTS(text=clean, lang="en", slow=False).save(filepath)
        print_substep(f"gTTS fallback: {clean[:60]}", style="dim")
    except Exception as e:
        print_substep(f"gTTS fallback failed: {e}", style="red")


# ─────────────────────────────────────────────────────────────────────────────
# Public engine class
# ─────────────────────────────────────────────────────────────────────────────

class Qwen3TTS:
    """
    Local Qwen3-TTS. Same interface as every other TTS module.
    Backend is auto-detected. Works on iMac M4 and Linux unchanged.

        tts = Qwen3TTS()
        tts.run(text, filepath="assets/temp/.../mp3/postaudio-0.mp3")
    """

    max_chars: int = 1000

    def __init__(self):
        self._backend = _BACKEND
        if _BACKEND == "gtts":
            print_substep(
                f"⚠️  No local backend found. {_install_hint()}\n   Using gTTS.",
                style="yellow",
            )
        else:
            print_substep(
                f"Qwen3-TTS ready — {_BACKEND_LABELS.get(_BACKEND, _BACKEND)}",
                style="bold blue",
            )

    def run(self, text: str, filepath: str, random_voice: bool = False) -> None:
        if not text or not text.strip():
            return

        clean_text = _strip_instruction(text)
        if not clean_text:
            return

        # No local backend
        if self._backend == "gtts":
            _fallback_gtts(clean_text, filepath)
            return

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # MLX bridge — no model object, runs as subprocess
        if self._backend == "mlx_bridge":
            voice_cfg = _resolve_voice(random_voice)
            try:
                if not _generate_mlx_bridge(clean_text, filepath, voice_cfg):
                    raise RuntimeError("Bridge returned False")
            except Exception as e:
                print_substep(f"MLX bridge failed: {e}. Using gTTS.", style="yellow")
                _fallback_gtts(clean_text, filepath)
            self._cooldown()
            return

        # All other backends — load model, generate
        model_id = settings.config.get("qwen_tts", {}).get(
            "model", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
        )
        model = _get_model(model_id)

        if model is None:
            _fallback_gtts(clean_text, filepath)
            return

        voice_cfg = _resolve_voice(random_voice)

        try:
            success = False

            if self._backend == "mlx":
                success = _generate_mlx(model, clean_text, filepath, voice_cfg)
            elif self._backend == "qwen":
                success = _generate_qwen_tts(model, clean_text, filepath, voice_cfg)
            elif self._backend in ("mps", "cuda", "cpu"):
                success = _generate_torch(model, clean_text, filepath, voice_cfg)

            if not success:
                raise RuntimeError("Generator returned False")

        except Exception as e:
            print_substep(f"Qwen3-TTS failed: {e}. Using gTTS.", style="yellow")
            _fallback_gtts(clean_text, filepath)
            return

        self._cooldown()

    def _cooldown(self):
        cooldown = settings.config.get("qwen_tts", {}).get("cooldown_seconds", 0.2)
        if cooldown > 0:
            time.sleep(cooldown)


# ─────────────────────────────────────────────────────────────────────────────
# Install hint
# ─────────────────────────────────────────────────────────────────────────────

def _install_hint() -> str:
    if _is_apple_silicon():
        return "Run: ./install_qwen_tts_mac.sh"
    if _is_linux():
        return "Run: ./install_qwen_tts_linux.sh"
    return "Install: pip install qwen-tts==0.1.1 torch"