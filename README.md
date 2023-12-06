# CoRA: **Co**-located **R**elightable **A**vatar

<img src="misc/teaser.gif" width="100%" >

This is a PyTorch implementation of the following paper:

**High-Quality Facial Geometry and Appearance Capture at Home**

Yuxuan Han, Junfeng Lyu, and Feng Xu

[Project Page]() | [Video]() | [Paper]()

**Abstract**: *Facial geometry and appearance capture have demonstrated tremendous success in 3D scanning real humans in studios. Recent works propose to democratize this technique while keeping the results high quality. However, they are still inconvenient for daily usage. In addition, they focus on an easier problem of only capturing facial skin. This paper proposes a novel method for high-quality face capture, featuring an easy-to-use system and the capability to model the complete face with skin, mouth interior, hair, and eyes. We reconstruct facial geometry and appearance from a single co-located smartphone flashlight sequence captured in a dim room where the flashlight is the dominant light source (\emph{e.g.} rooms with curtains or at night). To model the complete face, we propose a novel hybrid representation to effectively model both eyes and other facial regions, along with novel techniques to learn it from images. We apply a combined lighting model to compactly represent real illuminations and exploit a morphable face albedo model as a reflectance prior to disentangle diffuse and specular. Experiments show that our method can capture high-quality 3D relightable scans.*

## About Code Release
We plan to release the full code (including the data preprocessing pipeline, training code, and the automatic 3D assets export scripts) to support the users to use our project to create their own authentic relightable scan at home.

The code will be released upon the paper is accepted, or maybe more eailer. Stay tuned.

## Contact
If you have any questions, please contact Yuxuan Han (hanyx22@mails.tsinghua.edu.cn).

## License and Citation
This repository can only be used for personal/research/non-commercial purposes.
Please cite the following paper if this model helps your research:

    @inproceedings{han2023cora,
        author = {Han, Yuxuan and Lyu, Junfeng and Xu, Feng},
        title = {High-Quality Facial Geometry and Appearance Capture at Home},
        booktitle = {technical report},
        year={2023}
    }

## Acknowledgments
* The code is built on a bunch of wonderful projects, including [Nerfacc](https://github.com/nerfstudio-project/nerfacc), [tinycudann](https://github.com/NVlabs/tiny-cuda-nn), [facer](https://github.com/FacePerceiver/facer), [metrical-tracker](https://github.com/Zielon/metrical-tracker), and [WildLight](https://github.com/za-cheng/WildLight).
* Thanks [SoulShell](http://soulshell.cn/) for providing their Light Stage to help us conduct comparison experiments.
* Thanks [Jingwang Ling](https://gerwang.github.io/) for helpful discussions.
