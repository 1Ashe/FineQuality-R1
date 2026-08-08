#!/usr/bin/env bash
set -e

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 torchvision==0.21.0 packaging==25.0 ninja==1.13.0 psutil==7.0.0
python -m pip install --no-build-isolation -r requirements.txt
python -m pip install --no-deps -e src/open-r1-multimodal
