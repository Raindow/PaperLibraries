# VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION

[toc]

- 词短句翻译

  investigate：调查，研究

  thorough：彻底的，完全的

  prior-art：先前技术

  localization：本地化，定位

  secure：获得

  generalize：推广，泛化

  facilitate：促进

  cluster：簇，集群

  utilis：利用

  to this end：为了这个目的

  assess：评估评价

  evaluate：评估评价

  revision：修订

## ABSTRACT

卷积神经网络深度和它在大规模图像识别上的准确度之间的联系。

主要贡献：对使用了很小卷积核的网络结构，彻底清晰地分析评测了增加其深度的带来的影响，而结果显示当网络深度提升至16-19层时，先前网络能有极大的提升。这些发现是我们在Image Challenge 2014的提交上的基础，我们也分别获得了定位方面一等奖和分类方面二等奖（These findings were the basis of our ImageNet Challenge 2014 submission, where **our team secured the first and the second places in the localisation and classification tracks respectively**）我们也证明了我们的方法能够很好的推广到其他数据集，并实现了最先进优异的结果。我们已经公开了两个表现最好的的卷积网络模型，以促进在计算机视觉中使用深度学习的进一步研究。

## 1 INTRODUCTION

卷积神经网络在大规模的图片、影像的识别上取得了较大的成功，它的成功基于大型公共图片数据集库，比如ImageNet，以及高性能计算机系统，譬如GPU和大规模分布式集群。此外，ImageNet Large-Scale Visual Recognition Challenge在深度视觉识别体系结构的发展中发挥了重要作用，它已经成为了几代大规模图像分类系统的测试平台，从高维浅层特征编码到深层卷积网络。

随着卷积神经网络在计算机视觉领域愈加的普遍，大家做了很多关于如何提高最初的结构的准确率的尝试。举例而言（for instance），ILSVRC 2013中最佳的提交方案在网络的第一层使用了较小的接收窗口（卷积核？）和更小的步长，以及在整个图片和多个维度上对网络进行密集地训练测试。

本文中，我们指出了影响卷积网络的另一个重要因素：网络的深度。为此，我们固定了网络结构的其他参数，仅仅逐步的通过增加卷积层的方式，增加网络的深度，在所有层都是用了$3 \times 3$卷积核的条件下，这样的操作是可行的。

最后，我们提出了一个更加准确的卷积神经网络结构，不仅能在ILSVRC分类和定位任务上取得最先进的准确率，而且能够应用于其他图片识别数据集，即使该网络结构仅仅作为相对简单流程的一部分，也能够有较好的提升效果。我们已经发布了两个性能最好的模型，以供进一步的研究。

接下来的文章将以如下结构安排。

Section 2，我们会描述我们的卷积网络的配置

Section 3，图片分类网络训练和评测的细节

Section 4，在ILSVRC分类任务上对配置进行比较

Section 5，paper concludes

为了完整性，我们在附录A中描述并评估了我们的ILSVRC-2014目标定位检测系统，在附录B中则讨论了如何在其他数据集中进行深层网络特征的泛化推广，附录C主要是论文修改的列表

## 2 CONVNET CONFIGURATIONS

