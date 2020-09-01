#!/bin/bash
conda create -n pruning_toy python=3 pytorch torchvision &&
conda activate pruning_toy &&
pip install -r requirements.txt
