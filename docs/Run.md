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
In the first step, the goal is to obtain the (1) right and left eye mask for each frame and (2) place the eyeball into a plausible position manually in Blender and set a reasonable size. We disable the hybrid representation in this step: 
```
python trainer.py --config_path data/old_neutral/config_wo-hybrid.yaml --device 0
```

Then, we export the relightable assets from it, and manually label the left eye and right eye mask on the UV diffuse map. See the example images, i.e. in .

Next, we can render the left and right eye mask by running this command:
```

```

### Step 2
In the second step, we have already obtain the eyes mask and the eyes position, we can diretly run the full method as using our preprocessed dataset:

```

```
