# clDice-a_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure

[toc]

- 词短句翻译

  vascular：血管
  
  numerous：许多的
  
  microscopic：显微的
  
  optoacoustic：光声的
  
  radiology：放射性的

## Abstract

对管道、网状结构（血管、神经、道路）的准确分割与很多领域研究都有关。对于类似结构拓扑特性是他们最重要的特征，尤其是保持他们的连通特性。在血管的例子中，联通血管的遗漏会违反血流动力学。

我们引入了一种崭新的相似性度量——中心线Dice（clDice），基于分割掩码和形态学上的骨架进行计算。同时在理论上证明了clDice在二值化的2D和3D分割结果上保证了同伦等价的拓扑保留。在此基础上我们提出了计算上更具效率、可微并且能够结合到任意分割分割网络的损失（soft-Dice）。我们在五个公开数据集上使用了soft-clDice，包括血管、道路、神经（2D、3D）。soft-clDice是分割结果有更准确的连通信息、更高的图相似性和更好的体积精度（volumetric scores）。

## Introduction

通道和曲线结构在许多领域中都是重要的问题，比如临床和生物应用（从显微图像、光声图像和放生图像中分割出血管和神经）或是工业质量管控 。