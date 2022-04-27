



<div align="center">

# DSAL: Deeply Supervised Active Learning from Strong and Weak Labelers for Biomedical Image Segmentation

[![JBHI2021](https://img.shields.io/badge/arXiv-2101.09057-blue)](https://arxiv.org/abs/2101.09057)
[![JBHI2021](https://img.shields.io/badge/Journal-JBHI2021-green)](https://ieeexplore.ieee.org/document/9326423)


</div>

Keras implementation of our method for IEEE JBHI 2021 paper: "DSAL: Deeply Supervised Active Learning from Strong and Weak Labelers for Biomedical Image Segmentation".

Contents
---
- [Abstract](#Abstract)
- [Dataset](#Dataset)
- [Training](#Training)
- [Evaluation](#Evaluation)
- [Citation](#Citation)
- [Acknowledgement](#Acknowledgement)
  

Abstract
---
Image segmentation is one of the most essential biomedical image processing problems for different imaging modalities, including microscopy and X-ray in the Internet-of-Medical-Things (IoMT) domain. However, annotating biomedical images is knowledge-driven, time-consuming, and labor-intensive, making it difficult to obtain abundant labels with limited costs. Active learning strategies come into ease the burden of human annotation, which queries only a subset of training data for annotation. Despite receiving attention, most of active learning methods still require huge computational costs and utilize unlabeled data inefficiently. They also tend to ignore the intermediate knowledge within networks. In this work, we propose a deep active semi-supervised learning framework, DSAL, combining active learning and semi-supervised learning strategies. In DSAL, a new criterion based on deep supervision mechanism is proposed to select informative samples with high uncertainties and low uncertainties for strong labelers and weak labelers respectively. The internal criterion leverages the disagreement of intermediate features within the deep learning network for active sample selection, which subsequently reduces the computational costs. We use the proposed criteria to select samples for strong and weak labelers to produce oracle labels and pseudo labels simultaneously at each active learning iteration in an ensemble learning manner, which can be examined with IoMT Platform. Extensive experiments on multiple medical image datasets demonstrate the superiority of the proposed method over state-of-the-art active learning methods.

<p align="center">
<img src="https://github.com/jacobzhaoziyuan/DSAL/blob/main/assets/archi.png" width="700">
</p>



Dataset
---
- __ISIC 2017__: [[Download](https://challenge.isic-archive.com/data/)] composes of 2000 RGB dermoscopy images with binary masks of lesions.
- __RSNA Bone Age dataset__: [[Download](https://www.kaggle.com/kmader/rsna-bone-age)]. We follow the image processing and sampling methods from [BHI 2019](https://arxiv.org/pdf/1903.04778.pdf) & [EMBC 2020](https://arxiv.org/pdf/2005.03225) and obtain a small balanced dataset of 139 samples with masks of finger bones.

The data is under `./orgData`, with the following file structure:
```
./orgData
├── testGT
├── testImg
├── trainGT
├── trainImg
├── valGT
└── valImg
```

Training
--- 
#### 1. Preparing Environment
- Python 3.6.10, Keras 2.3.1 
- Install the dependencies in [`requirements.txt`](src/requirements.txt)

#### 2. Run the Training
```
cd src
python main.py
```
#### 3. Another Configuations
- If you wish to explore different experiment settings, simply specify the configurations in [`constants.py`](src/constants.py).+

Evaluation
---
- (Optional) Make sure the `constants.py` under the repo root is the correct experiment you want to evaluate on. 
   You can do this by simply:
    ```
    cp {global_path/exp/constants.py} .
    ```
- Run the test:
    ```
    python test.py
    ```


Citation
---
If you find the codebase useful for your research, please cite the paper:
```
@article{zhao2021dsal,
  title={Dsal: Deeply supervised active learning from strong and weak labelers for biomedical image segmentation},
  author={Zhao, Ziyuan and Zeng, Zeng and Xu, Kaixin and Chen, Cen and Guan, Cuntai},
  journal={IEEE Journal of Biomedical and Health Informatics},
  volume={25},
  number={10},
  pages={3744--3751},
  year={2021},
  publisher={IEEE}
}
```

Acknowledgement
---
Part of the code is adopted from [CEAL](https://github.com/marc-gorriz/CEAL-Medical-Image-Segmentation) codebase. We thank the authors for their fantastic and efficient codebase.
