# Topology-Preserving_Deep_Image_Segmentation

[toc]

- 词短句翻译

  be prone to：倾向于
  
  metric：度量，指标
  
  spectrum：范围，频谱
  
  fine-scale：精细的
  
  grasp：抓紧，抓牢
  
  membrane：膜
  
  neuron membranes：神经元细胞膜
  
  morphological：形态学上的
  
  structural quantification：结构量化
  
  catastrophic：灾难性的
  
  fidelity：准确地，精确地
  
  discrete：离散的
  
  inspect：检查
  
  unified：统一的
  
  derive：得到，获得，源自
  
  empirically：以经验为主地
  
  mutually：相互地
  
  empower：授权，允许
  
  estimate：估计，评测
  
  denoised：去噪
  
  elongated：拉长的
  
  alleviate：减轻，缓和
  
  revolve：旋转
  
  revolve around：围绕，围绕某一话题、方法
  
  intrinsically：本质上

## Abstract

关键的几个概念：

Betti数，Rand index（兰德里数），

本文主要提出了continuous-valued loss function，来限制拓扑形状

## Introduction

关键问题：

对于在分析物体功能信息上极为重要的fine-scale structures，包括small object instances， instances with multiple connected components， and thin connections，

A broken connection or a missing component may only induce marginal per-pixel error, but can cause catastrophic functional mistakes（一个连接点或丢失部分片段在像素层级可能只在边缘上产生，似乎并不严重，但却会对功能层面分析上产生重大错误）

功能层面分析包括但不局限于：规划机器人行动时，提取出细长对象，比如绳或是手柄

Betti数：Betti number (number of connected components and handles)