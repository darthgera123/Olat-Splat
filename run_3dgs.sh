#!/bin/bash


for i in {130..164};
do
num=$i
light=$(printf "L_%03d" "$num")

python render_exr.py --source_path /CT/LS_BRM02/static00/FlashingLights/sandhoodie/pose_01/$light/ \
--model_path /CT/LS_BRM02/static00/3dgs/output/base/sandhoodie/pose_01/$light --sh_degree 3 --skip_test & 

done