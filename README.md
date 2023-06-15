# Vicinity-Vision-Transformer

[[Project Page]](https://opennlplab.github.io/vvt/)

This repo is the official implementations of [Vicinity Vision Transformer](https://arxiv.org/abs/2206.10552).
It serves as a general-purpose backbone for image classification, semantic segmentation, object detction tasks. 

if you use this code, please cite:

```
@ARTICLE {10149455,
author = {W. Sun and Z. Qin and H. Deng and J. Wang and Y. Zhang and K. Zhang and N. Barnes and S. Birchfield and L. Kong and Y. Zhong},
journal = {IEEE Transactions on Pattern Analysis &amp; Machine Intelligence},
title = {Vicinity Vision Transformer},
year = {5555},
volume = {},
number = {01},
issn = {1939-3539},
pages = {1-14},
abstract = {Vision transformers have shown great success on numerous computer vision tasks. However, their central component, softmax attention, prohibits vision transformers from scaling up to high-resolution images, due to both the computational complexity and memory footprint being quadratic. Linear attention was introduced in natural language processing (NLP) which reorders the self-attention mechanism to mitigate a similar issue, but directly applying existing linear attention to vision may not lead to satisfactory results. We investigate this problem and point out that existing linear attention methods ignore an inductive bias in vision tasks, i.e., 2D locality. In this paper, we propose Vicinity Attention, which is a type of linear attention that integrates 2D locality. Specifically, for each image patch, we adjust its attention weight based on its 2D Manhattan distance from its neighbouring patches. In this case, we achieve 2D locality in a linear complexity where the neighbouring image patches receive stronger attention than far away patches. In addition, we propose a novel Vicinity Attention Block that is comprised of Feature Reduction Attention (FRA) and Feature Preserving Connection (FPC) in order to address the computational bottleneck of linear attention approaches, including our Vicinity Attention, whose complexity grows quadratically with respect to the feature dimension. The Vicinity Attention Block computes attention in a compressed feature space with an extra skip connection to retrieve the original feature distribution. We experimentally validate that the block further reduces computation without degenerating the accuracy. Finally, to validate the proposed methods, we build a linear vision transformer backbone named Vicinity Vision Transformer (VVT). Targeting general vision tasks, we build VVT in a pyramid structure with progressively reduced sequence length. We perform extensive experiments on CIFAR-100, ImageNet-1 k, and ADE20 K datasets to validate the effectiveness of our method. Our method has a slower growth rate in terms of computational overhead than previous transformer-based and convolution-based networks when the input resolution increases. In particular, our approach achieves state-of-the-art image classification accuracy with 50% fewer parameters than previous approaches.},
keywords = {transformers;task analysis;computer vision;standards;image resolution;sun;image classification},
doi = {10.1109/TPAMI.2023.3285569},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month = {jun}
}

```

![image](https://user-images.githubusercontent.com/13931546/175231586-5e7c46f3-29a2-4497-9ddf-bc40aeee88b3.png)



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


