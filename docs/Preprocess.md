# Video Preprocessing

Here we use a video captured by ourself as an example.

* Download an example video from [here](https://cloud.tsinghua.edu.cn/f/b7820a31fbbf496190fc/?dl=1) and put it to the `video` folder. If you just want process your own captured video, you can skip this download step and directly put your own video into the `video` folder.

```
|- video
    |- old_neutral.MOV
```

* Extract the frames. For every 10 frames, we select the sharpest one as the training data and drop out other frames.
```
# dump all the frames to disk
python scripts/extract_frames.py \
    --video_path video/old_neutral.MOV \
    --data_root workspace/data/old_neutral

# compute the sharpness value of each frame
python scripts/compute_sharpness.py --data_root workspace/data/old_neutral

# for every 10 frames, select the sharpest one
python scripts/select_sharp_frames.py --data_root workspace/data/old_neutral

# (OPTION) now you can delete the raw_frames folder and the sharpness.pkl
rm -rf workspace/data/old_neutral/raw_frames
rm -rf workspace/data/old_neutral/sharpness.pkl

```

* Compute face parsing masks. For eye masks, there are two options:
    * use the eye mask predicted by the face parsing mask (low-quality but fully automatic)
    * manually label the eye mask on the reconstructed UV map (**our default option**, will be detailed in RUN.md, high-quality but requires a little manual efforts)

```

```

* Fitting the AlbedoMM model to the frames to obtain pseudo GT of specular albedo maps.

```

```


* Use [MetaShape](https://www.agisoft.com/) to calibrate the camera parameters. NOTE: the quality of camera parameters play an important role in our method, and we find COLMAP **cannot** give high-quality reconstruction results in our pilot experiments.
