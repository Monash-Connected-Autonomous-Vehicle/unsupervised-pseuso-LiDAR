#! /bin/bash

python train.py

dvc add ./pretrained/generic_sfm.pth
dvc push
