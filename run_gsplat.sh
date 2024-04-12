#!/bin/bash

data=$1
pose=$2

python train_exr.py --source_path /CT/LS_BRM02/static00/FlashingLights/$data/pose_0$pose/L_203 --model_path /scratch/inf0/user/pgera/FlashingLights/3dgs/alphas/output/$data/pose_0$pose --eval --sh_degree 3 --iteration 7000

python render_exr.py --source_path /CT/LS_BRM02/static00/FlashingLights/$data/pose_0$pose/L_203 --model_path /scratch/inf0/user/pgera/FlashingLights/3dgs/alphas/output/$data/pose_0$pose --eval --sh_degree 3 --iteration 7000

