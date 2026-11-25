# Data licenses

This file lists the datasets and models this playbook uses, where they come from, and their licenses. The code in this repository is MIT-licensed. That license does not cover third-party data or models; each one keeps its own license, listed below.

## Datasets

| Dataset | Source | License | In this repo |
|---|---|---|---|
| Oxford RobotCar sample (INS, GPS, camera frames and LiDAR scans from one drive on 2014-12-12) | [Oxford RobotCar Dataset](https://robotcar-dataset.robots.ox.ac.uk/) | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) | `oxfordrobocar_sample_small/` (197 MB). `oxfordrobocar_sample_small/README.md` lists every file and which ones the code uses |

## Models

| Model | Source | License | How it's used |
|---|---|---|---|
| ResNet-18, ImageNet weights (`ResNet18_Weights.DEFAULT`) | [torchvision](https://pytorch.org/vision/stable/models.html) | BSD-3-Clause (torchvision), as the README states | Camera-frame features (`src/features/video_features.py`; `embedding_model` in `config/params.yaml`). Downloaded by torchvision on first use |
| ResNet-50, ImageNet weights (`ResNet50_Weights.DEFAULT`, optional) | [torchvision](https://pytorch.org/vision/stable/models.html) | Not stated in this repo. torchvision's code is BSD-3-Clause | Used only if `embedding_model` is set to `resnet50`. Downloaded by torchvision on first use |

## Notes

- The Oxford RobotCar sample is CC BY-NC-SA 4.0 (non-commercial, attribution, share-alike). That license, not this repository's MIT license, governs use of the data and of anything adapted from it. For commercial use, contact the dataset's authors.
- The camera images show public streets and may include people and vehicles. See the dataset's privacy page before you republish them.

## Citations

- Oxford RobotCar: W. Maddern, G. Pascoe, C. Linegar and P. Newman, "1 Year, 1000km: The Oxford RobotCar Dataset", *The International Journal of Robotics Research* 36(1), 2017.
- ResNet: K. He, X. Zhang, S. Ren and J. Sun, "Deep Residual Learning for Image Recognition", arXiv:1512.03385, 2015.
