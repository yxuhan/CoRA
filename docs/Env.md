# Environment
A single RTX 3090 graphics card is enough to run the code.

1. Create an virtual environment using conda:
```
conda create -n cora python=3.9
conda activate cora
```

2. Install pytorch and torchvision. Here we use `cuda 11.6` as an example. For other cuda version, please find the corresponding `whl` file path at [here](https://download.pytorch.org/whl/torch_stable.html).
```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    https://download.pytorch.org/whl/cu116/torch-1.13.1%2Bcu116-cp39-cp39-linux_x86_64.whl \
    https://download.pytorch.org/whl/cu116/torchvision-0.14.1%2Bcu116-cp39-cp39-linux_x86_64.whl
```

3. Install pytorch3d. Again, if you want to install pytorch3d in other cuda version, please find the corresponding file path at [here](https://anaconda.org/pytorch3d/pytorch3d/files).
```
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.4/download/linux-64/pytorch3d-0.7.4-py39_cu116_pyt1131.tar.bz2
```

4. Install tinycudann:
```
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

5. Install other libs:
```
pip install nerfacc==0.3.1 \
    moviepy \
    kornia \
    pyyaml \
    lpips \
    tensorboard \
    scikit-learn \
    PyMCubes==0.1.2 \
    trimesh \
    pymeshlab \
    fake-bpy-module-3.1 \
    networks \
    scikit-image==0.19.1 \
    fvcore \
    iopath \
    pymeshfix \
    pyfacer \
    av \
    pims \
    timm \
    face_alignment \
    mediapipe \
    opencv-python==4.5.2.52 \
    opencv-python-headless==4.5.2.52 \
    chumpy \
    numpy==1.22.4
```

6. Install Blender for UV unwrap:
```
cd blender

# Download Blender
wget https://mirror.clarkson.edu/blender/release/Blender3.1/blender-3.1.0-linux-x64.tar.xz or download from https://cloud.tsinghua.edu.cn/f/44a204b6c5824133ad92/?dl=1


# extract files
tar -xvf blender-3.1.0-linux-x64.tar.xz
```
