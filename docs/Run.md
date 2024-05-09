# Running

## Training on the preprocessed dataset
Download a dataset from [here](https://cloud.tsinghua.edu.cn/d/0d9bef2214dd42dc95d7/), e.g. old_neutral.zip.
Then, unzip it in the data folder, the directory structure should be:
```
|-data
    |- old_neutral
        |- hair_mask
        |- image
        |- ...
        |- config.yaml
        |- transforms.json
    |- ...
```

Running on the preprocessed dataset, using `cuda:0`:
```
python trainer.py --config_path data/old_neutral/config.yaml --device 0
```

## Export relightable assets
```
python trainer.py \
    --config_path data/old_neutral/config.yaml \
    --ckpt_path workspace/old_neutral/latest.pth \
    --save_visual_dir workspace/export/old_neutral \
    --mode export_eyeball \
    --chunk_size 16384 \
    --device 0
```

## Training on your own dataset
### Step 1
In the first step, the goal is to obtain the (1) right and left eye mask for each frame and (2) some 3D landmarks on the eyeballs. We disable the hybrid representation in this step: 
```
python trainer.py --config_path data/old_neutral/config_wo-hybrid.yaml --device 0
```

After training, we export the relightable assets from it:
```
python trainer.py \
    --config_path data/old_neutral/config_wo-hybrid.yaml \
    --ckpt_path workspace/cora/old_neutral_wo-hybrid/latest.pth \
    --save_visual_dir workspace/export/old_neutral_wo-hybrid \
    --mode export_eyeball \
    --chunk_size 16384 \
    --device 0
```

We manually label the left eye and right eye mask on the UV diffuse map. See the example images `diffuse_leye_mask.png` and `diffuse_reye_mask.png` in the [provided dataset](https://cloud.tsinghua.edu.cn/d/0d9bef2214dd42dc95d7/) (the eyes region is painted with RGB [1,1,1]) for reference::

<img src="../misc/diffuse_leye_mask.png" width="40%" >
<img src="../misc/diffuse_reye_mask.png" width="40%" >

Then, we can render the left and right eye mask by running this command:
```
python scripts/render_uv_eye_mask.py \
    --data_root data/old_neutral \
    --mesh_uv_path workspace/export/old_neutral_wo-hybrid/mesh_uv.obj
```

Next, we manually label the eyeball landmarks on the UV diffuse map. See the example images `diffuse_leye_landmark.png` and `diffuse_reye_landmark.png` in the [provided dataset](https://cloud.tsinghua.edu.cn/d/0d9bef2214dd42dc95d7/)(4 landmarks are painted with RGB [0,1,0] for each eye) for reference:

<img src="../misc/diffuse_leye_landmark.jpg" width="40%" >
<img src="../misc/diffuse_reye_landmark.jpg" width="40%" >

Using the following scripts, we can convert it to 3D landmarks.
```

```

After running the above scripts, you need to copy the printed 3D landmark position list to the config file like this:
```
data: {
    ...
    left_eye_ldm: [
        [-0.3164336383342743, -0.18279047310352325, 0.19412003457546234],
        [-0.2668660283088684, -0.19570477306842804, 0.20923510193824768],
        [-0.28190210461616516, -0.17798882722854614, 0.23827917873859406],
        [-0.21853820979595184, -0.171295166015625, 0.22779381275177002],
    ],
    right_eye_ldm: [
        [0.019469600170850754, -0.1688525378704071, 0.28486162424087524],
        [0.07669556140899658, -0.19646050035953522, 0.29582440853118896],
        [0.06701074540615082, -0.16873115301132202, 0.3271498382091522],
        [0.12527015805244446, -0.1823343187570572, 0.315157413482666],
    ],
    ...
}
```
See `data/old_neutral/config.yaml` for the reference.

*Note: all the things done in Step 1 can be replaced by the automatic method. We can use the face parsing network to obtain the left and right eye mask for each frame and the landmark detection method (triangularize multi-view 2D landmarks) to obtain 3D eyeball landmarks.*

### Step 2
In the second step, we have already obtain the eyes mask and the 3D eyeball landmarks, we can diretly run the full method as using our preprocessed dataset:

```
python trainer.py --config_path data/old_neutral/config.yaml --device 0
```
