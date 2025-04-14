#!/bin/bash

if [[ "$1" != "cleanup" && "$1" != "cpu" && "$1" != "gpu" ]]; then
    echo "Неверный аргумент: $1. Используйте один из следующих: clear, cpu, gpu."
    exit 1
fi

if [ "$1" = "cleanup" ]; then
    cd ~/
    rm -rf ~/miniconda3
    rm -rf ~/.cache/huggingface
    rm -rf ~/.cache/whisper
    rm -rf ~/Miniconda3-latest-Linux-x86_64.sh
    dd if=/dev/zero | pv > full.disk
    sync
    rm full.disk
    message "Выключение"
    sudo poweroff
    exit
fi

sudo apt update && sudo apt upgrade
sudo apt install build-essential nvidia-driver-570-server

cd ~/
rm -rf Miniconda3-latest-Linux-x86_64.sh
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
rm -rf Miniconda3-latest-Linux-x86_64.sh
source "$HOME/miniconda3/bin/activate"
conda create -n futaba python==3.10
conda activate futaba
if [ "$1" = "cpu" ]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
fi
pip install -r requirements.txt
