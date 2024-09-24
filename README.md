# A Four-Dimensional Variational Constrained Neural Network-based Data Assimilation Method

## Abstract
Advances in data assimilation (DA) methods and the increasing amount of observations have continuously improved the accuracy of initial fields in numerical weather prediction during the last decades. Meanwhile, in order to effectively utilize the rapidly increasing data, Earth scientists must further improve DA methods. Recent studies have introduced machine learning (ML) methods to assist the DA process. In this paper, we explore the potential of a four-dimensional variational (4DVar) constrained neural network (NN) method for accurate DA. Our NN is trained to approximate the solution of the variational problem, thereby avoiding the need for expensive online optimization when generating the initial fields. In the context that the full-field system truths are unavailable, our approach embeds the system's kinetic features described by a series of analysis fields into the NN through a 4DVar-form loss function. Numerical experiments on the Lorenz96 physical model demonstrate that our method can generate better initial fields than most traditional DA methods at a low computational cost, and is robust when assimilating observations with higher error outside of the distributions where it is trained. Furthermore, our NN-based DA model is effective against Lorenz96 physical models with larger variable numbers. Our approach exemplifies how ML methods can be leveraged to improve both the efficiency and accuracy of DA techniques.

## Introduction

This is the official repository for the paper [A Four-Dimensional Variational Constrained Neural Network-based Data Assimilation Method](https://doi.org/10.1029/2023MS003687) accepted by Joural of Advances in Modeling Earth Systems

Resources including traing/validation/inference codes are released here.

## Cite
```
@article{wangFourDimensionalVariational2024,
  title = {A {{Four}}‐{{Dimensional Variational Constrained Neural Network}}‐{{Based Data Assimilation Method}}},
  author = {Wang, Wuxin and Ren, Kaijun and Duan, Boheng and Zhu, Junxing and Li, Xiaoyong and Ni, Weicheng and Lu, Jingze and Yuan, Taikang},
  date = {2024-01},
  journaltitle = {Journal of Advances in Modeling Earth Systems},
  shortjournal = {J Adv Model Earth Syst},
  volume = {16},
  number = {1},
  pages = {e2023MS003687},
  issn = {1942-2466, 1942-2466},
  doi = {10.1029/2023MS003687},
  url = {https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023MS003687},
}

```

## Contact
If you have any questions, please contact me via email: wuxinwang@nudt.edu.cn