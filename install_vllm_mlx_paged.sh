#!/bin/bash
# ==============================================================================
# vLLM MLX DISTRIBUTED RAY ZERO-CLICK INSTALLER (PAGED ATTENTION)
# Target: Apple Silicon (M1/M2/M3) Bare-Metal Cluster
# ==============================================================================
set -euo pipefail

VENV_DIR="$HOME/.venv-vLLM_0160_Ray_02540_vllm-metal_010_paged_attn"
REPO_DIR="$HOME/DEV/bakari_vllm-metal_paged_attn"
TARGET_PYTHON="3.12.10"

echo "============================================================"
echo "🚀 INITIATING ATOMIC PAGED DEPLOYMENT (v3.12.10 FORCE)"
echo "============================================================"

# 1. Pyenv Forge for Exact Version Parity
echo "=> [1] Forging Python $TARGET_PYTHON Environment..."
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"

if ! command -v pyenv >/dev/null 2>&1; then
    echo "   [*] Installing pyenv..."
    brew install pyenv >/dev/null 2>&1 || true
fi

eval "$(pyenv init -)"

if ! pyenv versions --bare | grep -q "^${TARGET_PYTHON}$"; then
    echo "   [*] Compiling Python $TARGET_PYTHON from source (takes ~3-5 mins)..."
    export PYTHON_CONFIGURE_OPTS="--enable-shared"
    CFLAGS="-I$(brew --prefix xz)/include" LDFLAGS="-L$(brew --prefix xz)/lib" pyenv install --skip-existing "$TARGET_PYTHON"
fi

PYTHON_EXE="$PYENV_ROOT/versions/$TARGET_PYTHON/bin/python"

# 2. Workspace & Environment Isolation
echo "=> [2] Forging pristine workspace and virtual environment..."
mkdir -p "$REPO_DIR" && cd "$REPO_DIR"
rm -rf "$VENV_DIR"
"$PYTHON_EXE" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# 3. Rust Toolchain Check
echo "=> [3] Verifying Rust Compiler..."
if ! command -v cargo >/dev/null 2>&1; then
    echo "   📥 Installing Rust toolchain..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

# 4. Core Dependencies & Build Tools
echo "=> [4] Upgrading build toolchain & Core ML bounds..."
pip install --upgrade pip cmake ninja wheel setuptools_scm packaging maturin >/dev/null
pip install torch==2.10.0 torchaudio==2.10.0 torchvision==0.25.0 "transformers<5"
pip install "ray[default]==2.54.0"
pip install opencv-python-headless>=4.13.0 av==16.1.0

# 5. Public HTTPS Cloning
echo "=> [5] Cloning optimized repositories..."
rm -rf vllm vllm-metal
git clone -b fix/bakari-apple-silicon-ray-sync https://github.com/ProtoAI-Bakari/vllm.git
git clone -b feat/paged-attention-testing https://github.com/ProtoAI-Bakari/vllm-metal.git

# 6. Pure Constraint Enforcement & Paged Install
echo "=> [6] Injecting strict dependency bounds and building Paged kernels..."
sed 's/\[.*\]//g' vllm-metal/requirements.txt > pure_constraints.txt
pip install -r vllm-metal/requirements.txt
pip install -c pure_constraints.txt -e ./vllm
pip install -c pure_constraints.txt --no-build-isolation -e "./vllm-metal[paged]"

# 7. macOS Specific OS-Level Fixes
echo "=> [7] Applying macOS Dylib and Ray Environment bypasses..."
mkdir -p "$HOME/.config/vllm"
echo '["LD_LIBRARY_PATH"]' > "$HOME/.config/vllm/ray_non_carry_over_env_vars.json"

SP_PATH="$VENV_DIR/lib/python3.12/site-packages"
AV_DYLIB=$(find "$SP_PATH/av" -name "libavdevice*.dylib" | head -n 1 || true)
CV2_DYLIB=$(find "$SP_PATH/cv2" -name "libavdevice*.dylib" | head -n 1 || true)

if [[ -n "$AV_DYLIB" && -n "$CV2_DYLIB" ]]; then
    echo "   🔗 Symlinking Dylib: $CV2_DYLIB -> $AV_DYLIB"
    ln -sf "$AV_DYLIB" "$CV2_DYLIB"
fi

echo "============================================================"
echo "✅ PAGED DEPLOYMENT SUCCESSFUL (Python $(python -V))"
echo "Activate environment: source $VENV_DIR/bin/activate"
echo "============================================================"
