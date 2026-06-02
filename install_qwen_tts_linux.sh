#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# install_qwen_tts_linux.sh
# Linux x86_64 — qwen-tts package (CPU or CUDA)
# ─────────────────────────────────────────────────────────────────────────────
# Uses the official qwen-tts==0.1.1 package with transformers==4.57.3.
# This conflicts with mlx-audio (which needs transformers>=5.5) but is
# compatible with whisperx.
#
# GPU (CUDA):  ~1-2s/sentence
# CPU only:    ~8-20s/sentence  ← still usable for batch pipeline
#
# Usage:
#   chmod +x install_qwen_tts_linux.sh
#   ./install_qwen_tts_linux.sh           # auto-detects CUDA
#   ./install_qwen_tts_linux.sh --cpu     # force CPU mode
# ─────────────────────────────────────────────────────────────────────────────

set -e
RED='\033[0;31m'; GREEN='\033[0;32m'; BLUE='\033[0;34m'
YELLOW='\033[1;33m'; BOLD='\033[1m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓ $1${NC}"; }
step() { echo -e "\n${BOLD}${BLUE}▶ $1${NC}"; }
warn() { echo -e "${YELLOW}⚠ $1${NC}"; }
err()  { echo -e "${RED}✗ $1${NC}"; }

FORCE_CPU=false
for arg in "$@"; do [[ "$arg" == "--cpu" ]] && FORCE_CPU=true; done

echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}  Qwen3-TTS — Linux (qwen-tts package) Setup${NC}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# ── Verify Linux ──────────────────────────────────────────────────────────────
step "1/6  Checking platform..."
if [[ "$(uname -s)" != "Linux" ]]; then
    warn "Not Linux — this script is for Linux. Use install_qwen_tts_mac.sh on macOS."
fi
ok "Linux confirmed"

# ── Venv check ────────────────────────────────────────────────────────────────
step "2/6  Checking environment..."
if [[ -z "$VIRTUAL_ENV" ]]; then
    warn "No venv active. Activating ./venv ..."
    source ./venv/bin/activate
fi
ok "Venv: $VIRTUAL_ENV"
PYTHON_VERSION=$(python --version 2>&1)
ok "Python: $PYTHON_VERSION"

# ── CUDA detection ────────────────────────────────────────────────────────────
step "3/6  Detecting GPU..."
HAS_CUDA=false
if [[ "$FORCE_CPU" == false ]]; then
    if command -v nvidia-smi &>/dev/null; then
        CUDA_VER=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' 2>/dev/null || echo "unknown")
        ok "NVIDIA GPU detected (CUDA $CUDA_VER)"
        HAS_CUDA=true
    else
        warn "No NVIDIA GPU detected. Using CPU mode."
        warn "Expected: ~8-20s per sentence on CPU."
    fi
else
    warn "CPU mode forced via --cpu flag."
fi

# ── Pin conflicting packages to whisperx-compatible versions ─────────────────
step "4/6  Pinning transformers + huggingface-hub for whisperx compatibility..."

# These versions satisfy BOTH qwen-tts AND whisperx:
#   qwen-tts==0.1.1  requires transformers==4.57.3
#   whisperx         requires huggingface-hub<1.0
pip install \
    "transformers==4.57.3" \
    "huggingface-hub>=0.20.0,<1.0.0" \
    --quiet
ok "transformers==4.57.3, huggingface-hub<1.0 pinned"

# ── Install qwen-tts ──────────────────────────────────────────────────────────
step "5/6  Installing qwen-tts..."
pip install "qwen-tts==0.1.1" soundfile --quiet
ok "qwen-tts==0.1.1 + soundfile installed"

# ── Install PyTorch (CUDA or CPU) ────────────────────────────────────────────
step "6/6  Installing PyTorch..."
if [[ "$HAS_CUDA" == true ]]; then
    # Install latest stable PyTorch with CUDA
    # Adjust cu121 to match your CUDA version if needed
    pip install torch --index-url https://download.pytorch.org/whl/cu121 --quiet
    ok "PyTorch + CUDA installed"
else
    pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
    ok "PyTorch CPU installed"
fi

# ── Verify ────────────────────────────────────────────────────────────────────
echo ""
python -c "
import sys

# Test qwen-tts
try:
    import qwen_tts
    print('  qwen-tts:       OK')
except Exception as e:
    print(f'  qwen-tts:       FAILED — {e}')
    sys.exit(1)

# Test transformers version
try:
    import transformers
    print(f'  transformers:   {transformers.__version__}')
except:
    pass

# Test torch
try:
    import torch
    cuda = torch.cuda.is_available()
    print(f'  torch:          {torch.__version__}  CUDA={cuda}')
except Exception as e:
    print(f'  torch:          FAILED — {e}')

# Test soundfile
try:
    import soundfile
    print('  soundfile:      OK')
except Exception as e:
    print(f'  soundfile:      FAILED — {e}')

# Test whisperx still works
try:
    import whisperx
    print('  whisperx:       OK (not broken)')
except Exception as e:
    print(f'  whisperx:       WARNING — {e}')
    print('  (whisperx optional — pipeline still works without it)')
"

echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}${BOLD}  Setup complete ✓${NC}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Backend active: $(python -c "
try:
    import qwen_tts
    import torch
    cuda = torch.cuda.is_available()
    print(f'qwen-tts  (CUDA={cuda})')
except:
    print('unknown — run python main.py to see backend log')
")"
echo ""
echo "  Update config.toml:"
echo -e "    ${YELLOW}[settings.tts]${NC}"
echo -e "    ${YELLOW}voice_choice = \"qwen\"${NC}"
echo ""
echo -e "    ${YELLOW}[qwen_tts]${NC}"
echo -e "    ${YELLOW}model = \"Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice\"${NC}"
echo -e "    ${YELLOW}device = \"$([ "$HAS_CUDA" == true ] && echo "cuda" || echo "cpu")\"${NC}"
echo -e "    ${YELLOW}voice_name = \"Ryan\"${NC}"
echo ""
if [[ "$HAS_CUDA" == false ]]; then
    warn "CPU mode: expect ~8-20s per sentence."
    warn "A 30-sentence post will take ~5-10 minutes to generate audio."
    warn "The video will still be correct — just slow on first run."
fi