#!/usr/bin/env bash
# Creates the `manualarm` conda env used by everything under bythebook/.
# Separate from the `lerobot` recording env on purpose: nothing there is touched.
set -euo pipefail
CONDA="${CONDA:-$HOME/miniforge3/bin/conda}"
ENV_DIR="$("$CONDA" info --base)/envs/manualarm"
[ -d "$ENV_DIR" ] || "$CONDA" create -y -n manualarm python=3.12
PIP="$ENV_DIR/bin/pip"
# RTX 50xx needs a CUDA 12.8 build of torch.
"$PIP" install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
"$PIP" install -r "$(dirname "$0")/requirements.txt"
"$PIP" install numpy==2.2.6      # placo's dependencies pull a newer numpy; lerobot needs this one
"$ENV_DIR/bin/python" -c "import torch; x = torch.ones(2, device='cuda'); assert (x + x).sum().item() == 4.0; print('GPU OK:', torch.cuda.get_device_name(0))"
