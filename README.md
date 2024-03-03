### News
* We have released the full code to run on our preprocessed dataset, see [Run.md](docs/Run.md) for more details. The data preprocessing code and the data capture instructions is coming soon! For users who want to capture their own data, stay tuned!


# *CoRA*: *Co*-located *R*elightable *A*vatar

<img src="misc/teaser.gif" width="100%" >

This is a PyTorch implementation of the following paper:

**High-Quality Facial Geometry and Appearance Capture at Home**, CVPR 2024.

Yuxuan Han, Junfeng Lyu, and Feng Xu

[Project Page](https://yxuhan.github.io/CoRA/index.html) | [Video](https://www.youtube.com/watch?v=eqkTmNvlbgc) | [Paper](https://arxiv.org/abs/2312.03442)

**Abstract**: *Facial geometry and appearance capture have demonstrated tremendous success in 3D scanning real humans in studios. Recent works propose to democratize this technique while keeping the results high quality. However, they are still inconvenient for daily usage. In addition, they focus on an easier problem of only capturing facial skin. This paper proposes a novel method for high-quality face capture, featuring an easy-to-use system and the capability to model the complete face with skin, mouth interior, hair, and eyes. We reconstruct facial geometry and appearance from a single co-located smartphone flashlight sequence captured in a dim room where the flashlight is the dominant light source (e.g. rooms with curtains or at night). To model the complete face, we propose a novel hybrid representation to effectively model both eyes and other facial regions, along with novel techniques to learn it from images. We apply a combined lighting model to compactly represent real illuminations and exploit a morphable face albedo model as a reflectance prior to disentangle diffuse and specular. Experiments show that our method can capture high-quality 3D relightable scans.*

## Document
To use our codebase to create your own 3D relightable avatar, we provide the following documents:
1. [Env.md](docs/Env.md) for code environment setup.
1. [Capture.md](docs/Capture.md) for instructions to capture video under our setup, i.e. co-located video in a dim room where the smartphone flashlight is the dominant light source.
    * We provide some example videos captured by ourself at [here](). If you want to test our code quickly, you can just use these videos.
1. [Preprocess.md](docs/Preprocess.md) for video preprocessing. 
    * We provide the processed version of our captured video at [here](). If you want to test our code quickly, you can just use this dataset.
1. [Run.md](docs/Run.md) for scripts to train our method on the preprocessed dataset to reconstruct relightable avatar.

We also plan to create a video toturial to help users to create their own relightable avatar using our codebase. Stay tuned.

## Contact
If you have any questions, please contact Yuxuan Han (hanyx22@mails.tsinghua.edu.cn).

## License and Citation
This repository can <b>only be used for personal/research/non-commercial purposes</b>.
Please cite the following paper if this model helps your research:

    @inproceedings{han2024cora,
        author = {Han, Yuxuan and Lyu, Junfeng and Xu, Feng},
        title = {High-Quality Facial Geometry and Appearance Capture at Home},
        journal={CVPR 2024},
        year={2024}
    }

## Acknowledgments
* The code is built on a bunch of wonderful projects, including [Nerfacc](https://github.com/nerfstudio-project/nerfacc), [tinycudann](https://github.com/NVlabs/tiny-cuda-nn), [facer](https://github.com/FacePerceiver/facer), [metrical-tracker](https://github.com/Zielon/metrical-tracker), and [WildLight](https://github.com/za-cheng/WildLight).
* Thanks [SoulShell](http://soulshell.cn/) for providing their Light Stage to help us conduct comparison experiments.
* Thanks [Jingwang Ling](https://gerwang.github.io/) and [Zhibo Wang](https://sireer.github.io/) for helpful discussions.
