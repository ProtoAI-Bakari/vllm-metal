#!/bin/bash
set -e

ENV_NAME=".venv-vLLM_0160_Ray_02540_vllm-metal_010_paged_attn"
WORKSPACE_DIR="$HOME/DEV/vllm-metal/bakari_vllm-metal_paged_attn"

echo "============================================================"
echo "🚀 INITIATING ATOMIC PAGED DEPLOYMENT (vLLM 0.16.0 + MLX Paged)"
echo "============================================================"

# 1. Forge pristine virtual environment
echo "=> [1] Forging pristine virtual environment..."
rm -rf "$HOME/$ENV_NAME"
python3.12 -m venv "$HOME/$ENV_NAME"
source "$HOME/$ENV_NAME/bin/activate"

# 2. Rust Toolchain Check
echo "=> [2] Verifying Rust Compiler..."
if ! command -v cargo &> /dev/null; then
    echo "   📥 Installing Rust toolchain..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

# 3. Upgrade Bounds
echo "=> [3] Upgrading build toolchain & Core bounds..."
pip install --upgrade pip wheel setuptools maturin
pip install cmake ninja accelerate psutil mlx_lm ray[default]==2.41.0

# 4. Workspace Prep
echo "=> [4] Cloning repositories..."
mkdir -p "$WORKSPACE_DIR"
cd "$WORKSPACE_DIR"
if [ ! -d "vllm" ]; then
    git clone -b fix/bakari-apple-silicon-ray-sync https://github.com/ProtoAI-Bakari/vllm.git
fi
if [ ! -d "vllm-metal" ]; then
    git clone -b feat/paged-attention-testing https://github.com/ProtoAI-Bakari/vllm-metal.git
fi

# 5. Compile vLLM & Paged Plugin
echo "=> [5] Compiling Core vLLM & Paged Metal Backend..."
cd "$WORKSPACE_DIR/vllm"
pip install -e .
cd "$WORKSPACE_DIR/vllm-metal"
pip install --no-build-isolation -e ".[paged]"

# 6. Dylib bypasses
echo "=> [6] Applying macOS Dylib bypasses..."
AV_DYLIB=$(find "$HOME/$ENV_NAME/lib/" -name "libavdevice.*.dylib" | grep -i "av" | head -n 1)
CV2_DIR=$(dirname $(find "$HOME/$ENV_NAME/lib/" -name "cv2" -type d | head -n 1))/.dylibs
if [ -n "$AV_DYLIB" ] && [ -d "$CV2_DIR" ]; then
    AV_FILE=$(basename "$AV_DYLIB")
    ln -sf "$AV_DYLIB" "$CV2_DIR/$AV_FILE"
    echo "   🔗 Symlinked Dylib: $AV_FILE"
fi

echo "============================================================"
echo "✅ PAGED DEPLOYMENT SUCCESSFUL"
echo "Activate: source ~/$ENV_NAME/bin/activate"
echo "============================================================"
