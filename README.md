# 3DGS Crowd Viewer [[Project Page]](https://bigmmgz.github.io/CrowdSplat/)

This repository is a modified version of [GaussianSplattingViewer](https://github.com/limacv/GaussianSplattingViewer). For rasterization, we use [diff-gaussian-rasterization-memory-optimized
](https://github.com/RockyXu66/diff-gaussian-rasterization-memory-optimized), a memory-optimized version of [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization) designed for our crowd rendering. The avatar templates are trained using [GaussianAvatar](https://github.com/aipixel/GaussianAvatar).

We use this lightweight viewer to visualize our crowd rendering, evaluate rendering speed and memory usage.


![UI demo](assets/Result-white-bg-cropped.png)
Screenshot of CrowdSplat running at 31 FPS on an RTX4090 with 3,500 animated characters.

## Install

Test under Ubuntu 22.04 with cuda 11.8

```
git clone https://github.com/RockyXu66/GaussianSplattingViewer.git

conda create -n gs-crowd-viewer python=3.9 -y

conda activate gs-crowd-viewer

pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118

mkdir submodules

git clone https://github.com/RockyXu66/diff-gaussian-rasterization-memory-optimized.git --recursive submodules/diff-gaussian-rasterization-memory-optimized

pip install -r requirements.txt

pip install cuda-python loguru

pip install submodules/diff-gaussian-rasterization-memory-optimized
```

Download the sample characters and motions from [GoogleDrive](https://drive.google.com/file/d/1-EPfsME-CKIEIKlUVeQRAsVbtuutWbbS/view?usp=sharing), unzip the file and put them under the root.
```
<location>
|---gs-crowd-avatars
|   |---models
|   |---motions
|---
    |---submodules
        |---diff-gaussian-rasterization-memory-optimized
```

## Usage

```
python main.py
```

## TODO

- [ ] Add script to generate motion files from AMASS
- [ ] Add script to train and export avatar templates using [GaussianAvatar](https://github.com/aipixel/GaussianAvatar)

## Citation

If you find our work useful, please consider citing:

```
@inproceedings{sun2024crowdsplat,
    title={CrowdSplat: exploring Gaussian Splatting for crowd rendering},
    author={Sun, Xiaohan and Xu, Yinghan and Dingliana, John and O’Sullivan, Carol},
    booktitle={IET Conference Proceedings CP887},
    volume={2024},
    number={10},
    pages={311--314},
    year={2024},
    organization={IET}
}
```

## Acknowledgements

Thanks to the author of

* [GaussianSplattingViewer](https://github.com/limacv/GaussianSplattingViewer)
* [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
* [GaussianAvatar](https://github.com/aipixel/GaussianAvatar)

A huge thanks to the original authors for their contributions and for open-sourcing their work! 🚀
