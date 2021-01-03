# Joint Segmentation and Path Classification of Curvilinear Structures

[toc]

- 词短句翻译

  simultaneously：同时

  index terms：索引词

  investigate：研究

  inception：开始，开端[n.]

  sustain：持续

  substantial：大量的

  elusive：难捉摸的

  advent：出现

  in large part：很大程度上
  
  dichotomy：二分法
  
  to this end：因此

## Abstract

目的：检测图片中的曲型结构

最具挑战的问题在于：推断出曲线网络的图结构表示

当前方法：

- 首先获得图片的二值化分割结果
- 然后或是进行先验规则，或是进行通过单独的分类网络，从像素级的分割中获得路径的近似概率，进而微调网络结果（then refine it using either a set of hand-designed heuristics or a separate classifier that assigns likelihood to paths extracted from the pixel-wise prediction）

我们的方法：

使用深度网络同时训练分割和路径分类任务

我们验证了这么做的有效性，同时这么做能够消除两个模块之间的差异，能够更加统一的针对该任务做出优化

我们在道路和神经数据集上进行了应用测试

## Introduction

早在1960s到1970s年间计算机视觉领域开创时，曲型结构的自动化描述就开始被研究。

但图片存在较多噪声，结果比较复杂时，自动提取曲型结构仍比较困难。

在其他领域，机器学习尤其是深度学习从大规模的数据上学习到了更加鲁棒的结果

针对这一领域，有两大完全不同的方法。

- 利用分割结果，区分每一像素分别归属于前景和背景的可能性
- 利用分割结果作为输入，结合一个标量图像（阈值图？用来表明系统对所分配的标签的置信度，然后生成曲型线性结构的图形表示

我们早期的工作就属于这两种方法。

我们使用一种分类器产生分割结果，另一个分类器去评测图中的边的可能性得分，这些边属于全图的子图。

最近更多的深度学习多是依靠两个独立训练的分类器，一个用于分割，另一个用于区分可能的连通区。

本文中，我们构建了联合两个关键步骤的通道，使得两份部分能够一起被优化。