# Video Preprocessing

Here we use a video captured by ourself as an example.
Please do the following steps one by one.

## Step 0: Data preparation
Download an example video from [here](https://cloud.tsinghua.edu.cn/f/b7820a31fbbf496190fc/?dl=1) and put it to the `video` folder. If you just want to process your own captured video, you can skip this step and directly put your own video into the `video` folder.

```
|- video
    |- old_neutral.MOV
```

## Step 1: Extract video frames
For every 10 frames, we select the sharpest one as the training data and drop out other frames.

Dump all the frames to disk.
```
python scripts/extract_frames.py \
    --video_path video/old_neutral.MOV \
    --data_root workspace/data/old_neutral
```

Compute the sharpness value of each frame.
```
python scripts/compute_sharpness.py --data_root workspace/data/old_neutral
```

For every 10 frames, select the sharpest one.
```
python scripts/select_sharp_frames.py --data_root workspace/data/old_neutral
```

(OPTIONAL) Now you can delete the raw_frames folder and the sharpness.pkl.
```
rm -rf workspace/data/old_neutral/raw_frames
rm -rf workspace/data/old_neutral/sharpness.pkl
```

## Step 2: Matting and face parsing
Video matting to compute foreground mask. Before running the code, you should first download `rvm_resnet50.pth` from [here](https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.pth) (if it does not work please check their [repo](https://github.com/PeterL1n/RobustVideoMatting)) and put it into `scripts/RobustVideoMatting/model`.
```
cd scripts/RobustVideoMatting
python inference.py \
    --variant resnet50 \
    --checkpoint model/rvm_resnet50.pth \
    --device cuda:0 \
    --input-source ../../workspace/data/old_neutral/image \
    --output-type png_sequence \
    --output-composition ../../workspace/data/old_neutral/mats \
    --num-workers 8
cd ../..
```

<!-- * Compute face parsing masks. For eye masks, there are two options:
    * use the eye mask predicted by the face parsing mask (low-quality but fully automatic)
    * manually label the eye mask on the reconstructed UV map (**our default option**, will be detailed in RUN.md, high-quality but requires a little manual efforts) -->

Compute face parsing masks.
```
python scripts/face_parsing.py --data_root workspace/data/old_neutral
```

Check the results in `workspace/data/old_neutral/debug` manually and delete the image with **(1) eyes blinking or (2) poor face/hair mask**. After that, run the following command to sync the results:
```
# for example, if you find the mask's quality of workspace/data/old_neutral/debug/00012.png is poor
# and delete workspace/data/old_neutral/debug/00012.png manually
# the following command would remove 00012.png in all the subfolder, i.e. image, mask, hair_mask, and mats
python scripts/sync_select.py --data_root workspace/data/old_neutral
```

Apply mask to the image.
```
python scripts/apply_mask.py --data_root workspace/data/old_neutral
```

## Step 3: Camera calibration
Use [MetaShape](https://www.agisoft.com/) to calibrate the camera parameters. After running MetaShape, you would get the results like [this](https://cloud.tsinghua.edu.cn/f/4fcdca6b686540ee8db7/?dl=1). Then, unzip and put them into `workspace/data/old_neutral` and rename them to `metashape_recon.obj` and `metashape_recon.xml`. The folder structure should be:

```
|- workspace/data/old_neutral
    |- metashape_recon.obj  # the reconstructed mesh
    |- metashape_recon.xml  # the estimated camera parameters
    |- image
    |- hair_mask
    |- ...
```

*NOTE: the quality of camera parameters play an important role in our method, and we find COLMAP **cannot** give high-quality reconstruction results in our pilot experiments.*

*If you are not familar with MetaShape, we will provide a video toturial to show how to obtain these files. Stay tuned.*

Next, convert MetaShape camera coordinates to `transforms.json`.
```
python scripts/agi2nerf.py --data_root workspace/data/old_neutral
```


## Step 4: AlbedoMM fitting
Fitting the AlbedoMM model to the frames to obtain pseudo GT of specular albedo maps.

### Run MICA
First, clone the official [MICA](https://github.com/Zielon/MICA) repo and prepare the MICA conda environment following their instructions.

Then, compute the identity parameters.
```
conda activate MICA
python scripts/mica_compute_identity.py \
    --img_path workspace/data/old_neutral/image/00312.png \  # for your custom dataset you should select a frontal-view frame
    --data_root workspace/data/old_neutral \
    --mica_root /home/yxuhan/Research/MICA  # replace it to your MICA repo's absolute path
```

### Run fitting
First you need to prepare some files in `scripts/AlbedoMMFitting/data/FLAME2020`, i.e. `generic_model.pkl` and `albedoModel2020_FLAME_albedoPart.npz`:
```
|- scripts/AlbedoMMFitting/data/FLAME2020
    |- albedoModel2020_FLAME_albedoPart.npz  # download from https://github.com/waps101/AlbedoMM/releases
    |- generic_model.pkl  # copy this file from the MICA repo
```

Now, you can run AlbedoMM fitting.
```
cd scripts/AlbedoMMFitting
python fitting.py \
    --data_root ../../workspace/data/old_neutral \
    --save_root ../../workspace/fitting/old_neutral
python enlarge_specular.py --data_root ../../workspace/data/old_neutral
cd ../..
```

## In the end
Congratulations! If you have successfully done the following steps, you can goto RUN.md to reconstruct your own high-quality relightable scan! 
