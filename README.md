# Vicinity-Vision-Transformer

[[Project Page]](https://opennlplab.github.io/vvt/)

This repo is the official implementations of [Vicinity Vision Transformer](https://arxiv.org/abs/2206.10552).
It serves as a general-purpose backbone for image classification, semantic segmentation, object detction tasks. 

if you use this code, please cite:

```
@misc{sun2022vicinity,
      title={Vicinity Vision Transformer}, 
      author={Weixuan Sun and Zhen Qin and Hui Deng and Jianyuan Wang and Yi Zhang and Kaihao Zhang and Nick Barnes and Stan Birchfield and Lingpeng Kong and Yiran Zhong},
      year={2022},
      eprint={2206.10552},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

![image](https://user-images.githubusercontent.com/13931546/175231586-5e7c46f3-29a2-4497-9ddf-bc40aeee88b3.png)


## Todo
- [ ] Segmentation code.
- [ ] ImageNet21k pre-training.
- [ ] pretrain weights of 384x384 resolution.

## Usage
### Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

### environment
First, clone the repository locally:
```
git clone https://github.com/OpenNLPLab/Vicinity-Vision-Transformer.git
```
Then, install PyTorch 1.8.1 and torchvision 0.9.1 and pytorch-image-models 0.4.12


### Evaluation
```
sh test.sh configs/vvt/vvt_small.py 1 /path/to/checkpoint
```

### Training
```
sh train.sh configs/vvt/vvt_small.py 8 
```


## Weight
### VVT on ImageNet-1K

| Method           | Size | Acc@1 | #Params (M) | Link |
|------------------|:----:|:-----:|:-----------:|:-----:|
| VVT-tiny          |  224 |  79.2 |     12.9    | [link](https://1drv.ms/u/s!Ak3sXyXVg781gtRFwwHih3Yu9G3FGg?e=yHKvuc)    |
| VVT-tiny          |  384 |  80.3 |     12.9    |  -   |
| VVT-small        |  224 |  82.6 |     25.5    |   [link](https://1drv.ms/u/s!Ak3sXyXVg781gtREWfCdlLJVy1IgpA?e=l4h3Wi)   |
| VVT-small        |  384 |  83.4 |     25.5    |   -   |
| VVT-medium       |  224 |  83.8 |     47.9    |   [link](https://1drv.ms/u/s!Ak3sXyXVg781gtRG4lD_uEVyj7cPYw?e=ihjjtO)   |
| VVT-medium        |  384 |  84.5 |     47.9    |   -   |
| VVT-large        |  224 |  84.1 |     61.8    |  [link](https://1drv.ms/u/s!Ak3sXyXVg781gtRHmfu0BybCZ8k1FQ?e=fLskgG)    |
| VVT-large        |  384 |  84.7 |     61.8    |   -   |








---
Our code is developed based on [TIMM](https://github.com/rwightman/pytorch-image-models) and [PVT](https://github.com/whai362/PVT)


