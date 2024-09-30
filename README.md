# Gaussian Splatting HumanOLAT dataset

## Training OLAT 3DGS

```
python train_exr.py --source_path /CT/LS_BRM03/nobackup/relight_3dgs/data/sunrise_pullover/pose_01/9C4A0003-e05009bcad/ --model_path /CT/LS_BRM03/nobackup/relight_3dgs/output/sunrise_pullover/pose_01/9C4A0003-e05009bcad_3dgs_nodense/ --eval --sh_degree 0 --iteration 30000
```

## Training OLAT 3DGS with Normal Reg

```
python train_exr.py --source_path /CT/prithvi2/static00/3dgs/data/bluesweater_exr/gshader/full_light --eval --model_path /CT/prithvi2/static00/3dgs/output/bluesweater_exr/gshader/shader_brdf --eval --sh_degree 3 --iteration 10000 --brdf
```

## Training 3DGS with scale regularization
```
python train_exr_mask.py --source_path /CT/LS_BRM03/nobackup/relight_3dgs/data/sunrise_pullover/pose_01/full_light_mask --model_path /CT/LS_BRM03/nobackup/october/output/sunrise_pullover/pose_01/full_light_mask_exr --eval --sh_degree 3 --iteration 20000 --mask_path /CT/LS_BRM03/nobackup/relight_3dgs/data/sunrise_pullover/pose_01/full_light_mask
```

## Rendering 3DGS

```
python render_exr.py --source_path /CT/LS_BRM03/nobackup/relight_3dgs/data/sunrise_pullover/pose_01/full_light_mask_512 --model_path /CT/LS_BRM03/nobackup/october/output/sunrise_pullover/pose_01/full_light_mask_512  --sh_degree 3
```
