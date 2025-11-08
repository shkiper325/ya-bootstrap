#!/usr/bin/env bash
# Quick‑deploy script for a personal‑data processing cloud worker
#  - Installs an isolated Miniconda env (Python 3.10)
#  - Installs either CPU‑only or CUDA 12.6 PyTorch stack
#  - Provides a `futaba` conda env activation alias
#  - Optional secure cleanup workflow that wipes caches & free space then powers off
#
# Usage examples:
#   ./deploy_futaba.sh cpu      # install CPU‑only stack
#   ./deploy_futaba.sh gpu      # install GPU stack (CUDA 12.6 wheels)
#   ./deploy_futaba.sh cleanup  # wipe environment & shut down (irreversible!)
#
# NOTE: The script expects to run under an unprivileged user that can `sudo poweroff`.
#       Remove or adapt that line if sudo rights are not available.

set -euo pipefail

usage() {
  echo "Usage: $0 {cpu|gpu|cleanup}" >&2
  exit 1
}

[[ $# -ne 1 ]] && usage
readonly ACTION="$1"
readonly INSTALLER="Miniconda3-latest-Linux-x86_64.sh"

case "$ACTION" in
  cleanup)
    echo "[+] Removing Miniconda and model caches…"
    rm -rf "$HOME/.cache $HOME/$INSTALLER"

    if command -v pv &>/dev/null; then
      echo "[+] Zero‑filling free space (this can take a while)…"
      dd if=/dev/zero bs=1M status=none | pv | dd of="$HOME/full.disk" bs=1M status=none
      sync && rm -f "$HOME/full.disk"
    fi

    echo "[+] Powering off host…"
    sudo poweroff || echo "sudo poweroff failed – manually shut down if desired."

    ;;

  cpu|gpu)
    sudo apt update && sudo apt upgrade -y
    sudo apt install build-essential ffmpeg aptitude git-lfs libcurl4-openssl-dev cmake -y
    if [[ "$ACTION" == gpu ]]; then
        sudo apt install nvidia-driver-580-server nvidia-cuda-toolkit-*
    fi

    echo "[+] Fetching Miniconda installer…"
    wget -q "https://repo.anaconda.com/miniconda/$INSTALLER" -O "$HOME/$INSTALLER"

    echo "[+] Installing Miniconda silently…"
    bash "$HOME/$INSTALLER" -b -p "$HOME/miniconda3"
    rm -f "$HOME/$INSTALLER"

    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"

    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

    echo "[+] Creating \"futaba\" environment…"
    conda create -y -n futaba python=3.10
    conda activate futaba

    echo "[+] Installing PyTorch stack…"
    if [[ "$ACTION" == cpu ]]; then
      pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    else
      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
    fi

    echo "[+] Installing project Python dependencies…"
    pip install \
        numpy \
        openai-whisper \
        huggingface-hub \
        transformers \
        accelerate \
        bitsandbytes \
        peft \
        opencv-python \
        datasets \
        matplotlib \
        pandas \
        pillow \
        scipy \
        six \
        tensorboard \
        tqdm \
        sentencepiece

    if ! grep -q "alias futaba=" "$HOME/.bashrc"; then
      printf "\n# Futaba ML env\nalias futaba=\"eval \$(~/miniconda3/bin/conda shell.bash hook) && conda activate futaba\"\n" >> "$HOME/.bashrc"
    fi

    echo "export PATH=\"\$PATH:\$HOME/bin\"" >> "$HOME/.bashrc"

    cp -rp bin llm whisper "$HOME"

    echo "[+] Done. Start a new shell or run 'source ~/.bashrc' then 'futaba' to activate the env."

    cd $HOME
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    mkdir build
    cd build
    
    if [[ "$ACTION" == gpu ]]; then
        cmake .. -DGGML_CUDA=ON
    else
        cmake ..
    fi
    cmake --build . --config Release -j$(nproc)
    sudo make install

    ;;

  *)
    usage
    ;;
esac
