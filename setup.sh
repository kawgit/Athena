#!/bin/bash

deactivate
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
pip install transformers tokenizers datasets wandb

pip list --not-required