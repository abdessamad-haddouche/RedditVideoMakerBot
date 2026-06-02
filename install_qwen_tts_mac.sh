#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# install_qwen_tts_mac.sh
# Apple Silicon (M1/M2/M3/M4) — MLX backend
# ─────────────────────────────────────────────────────────────────────────────
# Uses mlx + mlx-audio. These require a SEPARATE venv from whisperx
# because mlx-audio needs transformers>=5.5 and huggingface-hub>=1.0,
# which conflict with whisperx's requirements.
#
# SOLUTION: Two venvs.
#   venv/         ← main project venv (whisperx + qwen-tts, Python deps)
#   venv_mlx/     ← MLX-only venv (mlx + mlx-audio, called as subprocess)
#
# The engine detects which venv it's in and routes accordingly.
#
# Usage:
#   chmod +x install_qwen_tts_mac.sh
#   ./install_qwen_tts_mac.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e
RED='\033[0;31m'; GREEN='\033[0;32m'; BLUE='\033[0;34m'
YELLOW='\033[1;33m'; BOLD='\033[1m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓ $1${NC}"; }
step() { echo -e "\n${BOLD}${BLUE}▶ $1${NC}"; }
warn() { echo -e "${YELLOW}⚠ $1${NC}"; }
err()  { echo -e "${RED}✗ $1${NC}"; }

echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}  Qwen3-TTS — macOS Apple Silicon (MLX) Setup${NC}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# ── Verify Apple Silicon ──────────────────────────────────────────────────────
step "1/5  Checking hardware..."
ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
    err "Not Apple Silicon ($ARCH). Use install_qwen_tts_linux.sh instead."
    exit 1
fi
ok "Apple Silicon confirmed (arm64)"

# ── Verify we're in the main project venv ────────────────────────────────────
step "2/5  Checking environment..."
if [[ -z "$VIRTUAL_ENV" ]]; then
    warn "No venv active. Activating ./venv ..."
    source ./venv/bin/activate
fi
ok "Venv: $VIRTUAL_ENV"

# ── Install soundfile in main venv (needed for WAV writing) ──────────────────
step "3/5  Installing soundfile in main venv..."
pip install soundfile --quiet
ok "soundfile installed"

# ── Try installing mlx in the SAME venv first ────────────────────────────────
# This works if whisperx isn't installed or you're on a clean venv.
step "4/5  Attempting MLX install in current venv..."

pip install mlx mlx-audio --quiet 2>/dev/null && {
    ok "mlx + mlx-audio installed in current venv"
    python -c "import mlx_audio, mlx.core; print('MLX import OK')" 2>/dev/null && {
        ok "MLX verified ✓"
        echo ""
        echo -e "${BOLD}Result: MLX backend active in main venv${NC}"
        echo -e "Update config.toml:"
        echo -e "  ${YELLOW}model = \"mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit\"${NC}"
        echo -e "  ${YELLOW}device = \"mps\"${NC}"
        exit 0
    }
}

# MLX conflicted with whisperx — create a separate venv
warn "MLX conflicts with whisperx in main venv. Creating venv_mlx/ ..."

step "4/5 (b)  Creating dedicated MLX venv..."
python3 -m venv venv_mlx
source venv_mlx/bin/activate

pip install --upgrade pip --quiet
pip install mlx mlx-audio soundfile --quiet
ok "mlx + mlx-audio + soundfile installed in venv_mlx/"

python -c "import mlx_audio, mlx.core; print('MLX import OK')"
ok "MLX verified in venv_mlx/ ✓"

# Switch back to main venv
deactivate
source ./venv/bin/activate

# ── Install a thin subprocess bridge ─────────────────────────────────────────
step "5/5  Setting up MLX subprocess bridge..."

# The bridge lets the main venv call MLX generation via subprocess
# without the dependency conflict. qwen3_tts.py detects venv_mlx/ and
# routes generation through it automatically.
cat > mlx_bridge.py << 'BRIDGE'
#!/usr/bin/env python3
"""
mlx_bridge.py
─────────────
Subprocess bridge: main venv → venv_mlx/ → MLX generation.
Called by qwen3_tts.py when venv_mlx/ exists but mlx_audio is not
importable in the current venv.

Usage (called by qwen3_tts.py, not by you directly):
  python mlx_bridge.py --text "Hello world" --speaker Ryan --output /path/to/out.wav
"""
import argparse
import sys
import numpy as np
import soundfile as sf

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text",     required=True)
    p.add_argument("--speaker",  default="Ryan")
    p.add_argument("--output",   required=True)
    p.add_argument("--model",    default="mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit")
    p.add_argument("--voice-design",  default="")
    p.add_argument("--reference-audio", default="")
    args = p.parse_args()

    try:
        from mlx_audio.tts.utils import load_model
        model = load_model(args.model)

        gen_kwargs = {"text": args.text}
        if args.voice_design:
            gen_kwargs["voice_design"] = args.voice_design
        elif args.reference_audio:
            gen_kwargs["reference_audio"] = args.reference_audio
        else:
            gen_kwargs["speaker"] = args.speaker

        audio = model.generate(**gen_kwargs)
        audio_np = np.array(audio).astype(np.float32)
        peak = np.abs(audio_np).max()
        if peak > 0:
            audio_np = audio_np / peak * 0.95

        sf.write(args.output, audio_np, samplerate=24000)
        print("OK")
        sys.exit(0)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
BRIDGE

ok "mlx_bridge.py created"
echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}${BOLD}  Setup complete ✓${NC}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  MLX venv:   ./venv_mlx/"
echo "  Bridge:     ./mlx_bridge.py"
echo ""
echo "  Update config.toml:"
echo -e "    ${YELLOW}[settings.tts]${NC}"
echo -e "    ${YELLOW}voice_choice = \"qwen\"${NC}"
echo ""
echo -e "    ${YELLOW}[qwen_tts]${NC}"
echo -e "    ${YELLOW}model = \"mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit\"${NC}"
echo -e "    ${YELLOW}device = \"mps\"${NC}"
echo -e "    ${YELLOW}voice_name = \"Ryan\"${NC}"
echo ""