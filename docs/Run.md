# Training on the preprocessed dataset

```
# download a dataset from https://cloud.tsinghua.edu.cn/d/0d9bef2214dd42dc95d7/, e.g. old_neutral.zip
# then unzip it in the data folder, the directory structure should be
|-data
    |- old_neutral
        |- hair_mask
        |- image
        |- ...
        |- config.yaml
        |- transforms.json
    |- ...

# running on the preprocessed dataset, using cuda:0
python trainer.py --config_path data/old_neutral/config.yaml --device 0
```

# Training on your own dataset

TODO.

# Export relightable assets from the trained neural fields
```
python trainer.py \
    --config_path data/old_neutral/config.yaml \
    --ckpt_path workspace/old_neutral/latest.pth \
    --save_visual_dir workspace/export/old_neutral \
    --mode export_eyeball \
    --chunk_size 16384 \
    --device 0
```
