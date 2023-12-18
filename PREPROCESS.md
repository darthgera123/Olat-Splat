# Preprocess Data
To preprocess captured data, we need to follow the steps

## Metashape Calibration
Take gpu22/20
```
cd /CT/prithvi/work/studio-tools/LightStage/calibration
```
Follow instructions in `README.md`
SCALE=1.0 as everything is in metres


## Undistort and Mask generation
Take gpu22/20
```
/CT/prithvi/work/studio-tools/LightStage/segmentation
```
Follow instructions in `README.md`

## Generate full_light data
```
cd /CT/prithvi2/work/gsplat_codes/gaussian-splatting
python calib_camera.py \        
--calib /CT/prithvi2/nobackup/spacehoodie/OLATS/full_light/cameras.calib \
--points_in /CT/prithvi2/nobackup/spacehoodie/OLATS/full_light/points_meshlab.ply \
--mask_path /CT/prithvi2/nobackup/spacehoodie/OLATS/C001_foregroundSegmentation_olat2/crf \
--img_path /CT/prithvi2/nobackup/spacehoodie/OLATS/C001_foregroundSegmentation_olat2/fg/ \
--ext jpg \
--output /CT/prithvi2/static00/3dgs/data/spacehoodie/full_light_1 \
--scale 4 \
--create_alpha \
--lights_txt /CT/prithvi/work/studio-tools/LightStage/calibration/LSX_light_positions_aligned.txt \
--lights_order /CT/prithvi2/work/lightstage/LSX_python3_command/LSX/Data/LSX3_light_z_spiral.txt 
```

## Generate light cam paired data
```
python create_light_cam.py \
--cam_json /CT/prithvi2/static00/3dgs/data/bluesweater_exr/mlp/full_light/transforms_train.json \
--light_json /CT/prithvi2/static00/3dgs/data/bluesweater_exr/mlp/full_light/light_dir.json \
--output /CT/prithvi2/static00/3dgs/data/bluesweater_exr/mlp/olat_5/ \
--mask_path /CT/prithvi2/nobackup/bluesweater/OLATS/C002_foregroundSegmentation/crf \
--img_path /CT/LS_FRM01/nobackup/bluesweater/OLATS/ \
--scale 4 \
--ext exr \
--create_alpha
```