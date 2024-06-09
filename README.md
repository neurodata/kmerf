# Learning Interpretable Characteristic Kernels via Decision Forests

[![Google Scholar](https://img.shields.io/badge/Google%20Scholar-%234285F4?style=for-the-badge)](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C21&q=Learning+Interpretable+Characteristic+Kernels+via+Decision+Forests&btnG=)
[![Static Badge](https://img.shields.io/badge/arXiv-1812.00029-%23B31B1B?style=for-the-badge)](https://arxiv.org/abs/1812.00029)
[![Static Badge](https://img.shields.io/badge/sampan.me-%23007ab3?style=for-the-badge)](https://sampan.me/panda2023learning.html)

By: Sambit Panda, Cencheng Shen, and Joshua T. Vogelstein

This repo contains figure replication code for our paper.

## Abstract
Decision forests are widely used for classification and regression tasks. A lesser known property of tree-based methods is that one can construct a proximity matrix from the tree(s), and these proximity matrices are induced kernels. While there has been extensive research on the applications and properties of kernels, there is relatively little research on kernels induced by decision forests. We construct Kernel Mean Embedding Random Forests (KMERF), which induce kernels from random trees and/or forests using leaf-node proximity. We introduce the notion of an asymptotically characteristic kernel, and prove that KMERF kernels are asymptotically characteristic for both discrete and continuous data. Because KMERF is data-adaptive, we suspected it would outperform kernels selected a priori on finite sample data. We illustrate that KMERF nearly dominates current state-of-the-art kernel-based tests across a diverse range of high-dimensional two-sample and independence testing settings. Furthermore, our forest-based approach is interpretable, and provides feature importance metrics that readily distinguish important dimensions, unlike other high-dimensional non-parametric testing procedures. Hence, this work demonstrates the decision forest-based kernel can be more powerful and more interpretable than existing methods, flying in the face of conventional wisdom of the trade-off between the two.

## Notes
The real data figure in the manuscript was created by modifying this MATLAB script and running our test: https://github.com/neurodata/MGC-paper/blob/master/Code/Experiments/run_realData3.m This has been reproduced in the real_data.ipynb file within this repo.