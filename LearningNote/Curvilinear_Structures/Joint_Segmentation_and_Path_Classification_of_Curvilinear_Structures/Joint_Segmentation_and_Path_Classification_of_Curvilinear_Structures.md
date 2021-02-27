# Joint Segmentation and Path Classification of Curvilinear Structures

[toc]

**本文还存疑惑：**

**？推导过程时进行切片了么？**

**感觉不是。。？**

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
  
  separate：不同的
  
  streamlined：合理化的，流线型的，改进的
  
  intuition：直觉
  
  be prone to：倾向于
  
  have an edge：有优势
  
  supersede：取代代替
  
  advent：出现
  
  biomedical：生物医学的
  
  continuity：连续性
  
  tackle：应付，处理，解决
  
  ambiguities：模棱两可，歧义
  
  fragments：碎片
  
  tubularity：管状
  
  sophisticated：复杂的，精致的
  
  geometric：几何的
  
  topological：拓扑的
  
  constraints：约束
  
  sake：目的，利益
  
  yield：产出，生成
  
  topology：拓扑学的
  
  convergence：收敛（n.）
  
  overlaid：覆盖
  
  unevenly：不平均的，不平衡的
  
  metrics：指标

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
- 利用分割结果作为输入，结合一个标量图像（阈值图？用来表明系统对所分配的标签的置信度，然后生成曲型线性结构的图形表示）

我们早期的工作就属于这两种方法。

我们使用一种分类器产生分割结果，另一个分类器去评测图中的边的可能性得分，这些边属于全图的子图。

最近更多的深度学习多是依靠两个独立训练的分类器，一个用于分割，另一个用于区分可能的连通区。

本文中，我们构建了联合两个关键步骤的通道，使得两份部分能够一起被优化。为此，我们设计了一个新的网络，由一个编码器两个解码器构成，不同的解码器应对上述两个不同的问题。

首先，只给出一张图片，产生分割结果

然后，给出两点之间的路径作为额外输入，返回该路径与图像中实际存在的线性结构相对应的可能性。

这能够让我们去计算分割结果，建立一个边与候选线性结构相对应的图，并权衡它们，只保留最好的。

它比起我们以往的工作更加的流畅，合理，整体化。正因为它强制流程的一致性，他的效果更佳。

我们的贡献：

提出了将分割线性解构和分类先行路径统一起来的方法。其背后的直觉源于两个任务是紧密相关的，而且依赖于同一网络的产生的同一特征。基于此，我们提出了一种通用的的方法，而且在不同类型的图片上都能有很好的效果。

## Related Work

多数的寻找曲线算法总是由寻找图中弯曲路径开始。我们认为它主要基于分割，逐像素的标记该像素是否属于目标结构。

在多数应用中，譬如神经医学和制图学，上述做法可以视为很好的第一步，但是不够好。其中最关键的问题在于推断线性结构构成的图的连通性。

接下来，将讨论我们将讨论解决这两个任务的现有方法，这两个任务通常被视为是割裂的，这也是我们希望去解决的。

### 2.1 Segmentation

检测线性结构

既可以通过一定的算法，比如经典的**Optimally Oriented Flux (OOF)** 和 **Multi-Dimensional Oriented Flux (MDOF)**算法（需要去看一下），他们不需要训练数据，但是他们也有局限，难以胜任复杂且具有挑战性的任务，尤其当输入图片有极广的尺度范围或者极为不同的外观或是图像结构非常不规则的时候。

而面对上述的情景，基于学习特征的方法则有较大的优势，近几年也有较多的方法提出。将图像经过哈尔小波变换后结果或是图像的光谱特征作为分类器的输入，而在“**Multiscale Centerline Detection**”中分类器被预测到最近的中心线距离的回归器替代，进而能够通过回归器得到线性结构的宽度。

正如计算机视觉中的其他领域，许多早期基于学习的方法，正在被基于深度学习的方法所代替。

对于道路结构的提取，

- 第一次将深度学习引入流程中的是“**Learning to Detect Roads in High-Resolution Aerial Images**”，直接使用图片切片输入到全连接网络中。虽然图片提供了一些有关于线性结构的上下文，但由于内存的限制，他仍然相对较小。
- 卷积神经网络的出现代表着能够使用更大的感受野，在“**N4-Fields: Neural Network Nearest Neighbor Fields for Image Transforms**”，卷积被用来提取特征，然后可以与学习字典中的单词进行匹配（”could then be matched against words in a learned dictionary”没理解具体意思），最终的预测结果基于特征空间内的相邻者的共同影响。
- 在“**Machine Learning for Aerial Image Labeling**”中则使用了卷积网络代替了全连接网络
- “**Deep Roadmapper: Extracting Road Topology from Aerial Images**”一文提出了一种可微分交集连接损失（**a differentiable Intersection-over Union loss**，可微分IoU损失）用于道路分割，进而提取道路网的结构。
- 在边检测任务中“**Holistically-Nested Edge Detection**”中提到的检测器利用多尺度的卷积特征直接生成全图的边网络图。

生物医学领域，

- “**Deep Retinal Image Understanding**”先利用`VGG`网络在自然图像上进行预训练，而后使用特殊的层经过细调或增强去提取视网膜血管

- `U-Net`在生物图像分割领域有不错的表现，是目前在`ISBI’12`挑战中神经元边界检测结果最好的方法之一

- 最近的方法则更关注特征和结构，

  特征：“**Unsupervised Domain Adaptation by Backpropagation**”，“**EL-GAN: Embedding Loss Driven Generative Adversarial Networks for Lane Detection**”和“**Self-Supervised Feature Learning for Semantic Segmentation of Overhead Imagery**”

  结构：“**Automatic Road Detection and Centerline Extraction via Cascaded End-To-End Convolutional Neural Network**”和“**Holistically-Nested Edge Detection**”

我们的方法引入了新的损失衡量去捕捉高阶的线性结构特征，比如结构的平滑性和连续性。

### 2.2 Delineation

从像素概率掩码推断连通性信息的最简单方法是对其进行阈值化和骨架化，然后应用一组先验启发式方法来弥合差距并消除错误检测。但是，往往这种人工设计的先验启发式的方法是针对某一项问题，并且有大量的超参数，很大可能只能解决众多问题中的一小部分。下图展示了一些能够被机器学习很好地解决的问题

![image-20210114100701629](D:\Documents\PaperLibraries\LearningNote\Curvilinear_Structures\Joint_Segmentation_and_Path_Classification_of_Curvilinear_Structures\assets\image-20210114100701629.png)

传统方法容易产生模糊两可的情况，上图中的`a`图中的间断是不合理的，，从图中可以看到这是另一条路垂直穿插过另两条路，而`b`中的间隔却是合理的，因为原图中间隔部分是一个停车场，两张图中相似的间隔情况却应该有不同的处理方式，这是先验启发式规则难以解决的。

当前已有的方法按照下述流程：

1. 使用Section 2.1 中提及的方法去判断平面像素属于线性结构（线性图）或是立体像素属于管状结构的概率（管状图），我们早期的工作或是使用OOF算法（“**Three Dimensional Curvilinear Structure Detection Using Optimally Oriented Flux**”和“**Detecting Irregular Curvilinear Structures in Gray Scale and Color Imagery Using Multi-Directional Oriented Flux**”），或是使用基于决策树的分类器（“**Multiscale Centerline Detection**”）
2. 定义一个图，他的边连接空间位置同时能够匹配上目标结构的可能碎片，举个例子，考虑到图片中管状结构在图片中的位置和距离，找到最短路径进而定义最短路径（“**Hierarchical Discriminative Framework for Detecting Tubular Structures in 3D Images**”，“**Mind the Gap: Modeling Local and Global Context in (Road) Networks**”，“**Road Networks as Collections of Minimum Cost Paths**”，“**Reconstructing Curvilinear Networks Using Path Classifiers and Integer Programming**”），图中的边往往被称为路径
3. 在上述图中找到子图，子图的边与候选目标结构匹配，且最终将在线性结构结果中保留，找到这样的子图的最简单的方法，就是寻找最小生成树，然后对其进行修剪（“**Automated Reconstruction of Tree Structures Using Path Classifiers and Mixed Integer Programming**”，“**Automatic 3D Neuron Tracing Using All-Path Pruning**”,“**FMST: an Automatic Neuron Tracing Method Based on Fast Marching and Minimum Spanning Tree**”）。更复杂精细的做法是通过线性规划或二次规划制定搜索最优子图，进一步加强全局的几何和拓扑约束，获得最终的轮廓和环路结构（“**Reconstructing Curvilinear Networks Using Path Classifiers and Integer Programming**”，“**Active Learning and Proofreading for Delineation of Curvilinear Structures**”）近阶段的趋势同样使用了两个深度学习网络完成上述目的，其一用于分割，另一用于寻找可能的路径或是逐像素跟踪线性结构中的像素点并检测端点像素。

需要注意的是：尽管分割和描绘轮廓的部分已经用深度网络实现，但他们之间仍然保持着独立，并且不一起训练，同时这也是我们将要解决的事情。

## 3 METHOD

我们的方法仍然按Section 2.2中提及的流程进行。我们首先计算一个管状图，用它来建立一个边与候选线性结构相对应的过完备图（overcomplete graph），给每个边打分，之后只保留最好的，此过程如下图：

![image-20210118170410176](D:\Documents\PaperLibraries\LearningNote\Curvilinear_Structures\Joint_Segmentation_and_Path_Classification_of_Curvilinear_Structures\assets\image-20210118170410176.png)

但是和“**Reconstructing Curvilinear Networks Using Path Classifiers and Integer Programming**”中的方法不同是我们并不认为进行复杂的线性规划或是二次规划来寻找最优秀子图是有必要的。由于我们使用了相同的网络去计算管道图（tubularity map）和边的权重确保了网络的一致性同时也无需复杂方法生成最终的轮廓线来提高网络能力去超越最先进的方法。接下来我们将描述网络的结构，以及相关的训练和测试流程。

### 3.1 Formalization

为了简单，我们通过2D的图片进行方法的讲解，但是我们的方法也能很自然的应用到3D图像上，网络整体流程如下：

![image-20210119143823811](D:\Documents\PaperLibraries\LearningNote\Curvilinear_Structures\Joint_Segmentation_and_Path_Classification_of_Curvilinear_Structures\assets\image-20210119143823811.png)

网络有两个分支，两个分支共享相同的U-Net编码器，编码器以图片和二值候选路径掩码作为输入。第一个分支使用解码器同时跳过连接去生成管状图（这里有一点懵逼🤨）。第二个分支则依赖于一个简单网络生成路径的分类分数，相关符号表示及说明如下：

$x_i \in \mathbb{R}^{H \cdot W \cdot C}$：大小为$W \times H$，通道数为$C$的图片

$x_p \in \mathbb{R}^{H \times W}$：候选路径的$W \times H$的二值图片，值为1代表该像素属于路径

他们将被输入到编码网络$f_{enc}$，此流程如上图中的左侧所示。

而上图中的右侧

将潜在的表征$h=f_{enc}(x_i,x_p)$输入到解码器$f_s$中，最后输出分割图片$y_s=f_s(h)$，同时另一个解码器$f_p$则输出路径分类分数$y_p=f_p(h)$

对于图片$x_i$中的像素点$q$，$y_s[q]$代表着$q$像素点被视为“1”的可能性，即属于线性结构的可能性。

$y_s^{gt}$：$W \times H$二值分割结果图

$\mathcal{L}_{seg}$：分割分支$y_s$和$y_s^{gt}$的二值交叉损失值，训练时降低其值。

$\mathcal{L}_p$：分类分支$y_p$和$y_p^{gt}$二值交叉损失值，训练时降低其值。

总损失函数：
$$
\mathcal{L}_{seg}(y_s, y_s^{gt}) + \eta_p \mathcal{L}_p(y_p, y_p^{gt})
$$
$\eta_p$：常量，确保两个损失值有相同的量级。

上述两部分损失都使用二值交叉熵。此外，我们还可以添加“**Beyond the Pixel-Wise Loss for Topology-Aware Delineation**”中的拓扑学损失到$\mathcal{L}_{seg}$确保分割结果的全局统计特性和ground-truth相同（这一点可以标注一下，因为不是很懂，**全局统计特性**的意思，好像包括像素值全部的直方图特性，就是原来ground-truth的全图数值特性）

### 3.2 Architecture

为实现上述的网络$f$，我们基于U-Net网络结构，它是全连接网络，包括一个编码器和一个解码器，之间有跳跃连接。它是目前线性结构二值分割最先进的方法。为了能够在路径分类中使用，我们调整了U-Net网络使他能同时接受二值掩码$x_p$和原图$x_i$，就如Fig.3中的右侧所示。而后，我们添加了图片中的第二个分支。他连接了编码器和分类的输出分数$y_p$，第一个分支仍然产生分割概率图$y_s$。上述情况如Fig.3中的右侧。

更准确详细地说，分割分支遵循标准U-Net设计，有四个最大池化操作和相应的编码器和解码器层之间的跳跃连接。在每次下采样，滤波器的数量按照因子2增加（即变为原来的两倍），而在上采样时恰好相反。分类分支使用相同的编码器但使用更加简单的解码器，由最大池化层，两个全连接层以及一层额外的卷积层组成。为了加速模型的收敛，减少训练耗时，我们使用批归一化（batch normalization），并且我们还在测试时使用当前批统计信息（？？？什么意思，不能完全理解“**3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation**”）

对于3D数据集，由于内存的限制，我们则使用两个最大池化层和两个全连接路径分类层编码器。

在实践中，为了计算$y_s$和$y_p$，我们第一次传入网络时 ，将$x_p$用一张全为0的图片替代，传入网络中，这样结果将只以原图为条件。然后我们将第二次传入$x_p$去获得同时取决于图像和输入路径的$y_p$值，这能保障分割和分类两分支使用相同的中间特征。

### 3.3 Training

训练网络时，需要成对的图片$(x_i,y_s^{gt})$和标明了是否为候选路径$x_p$。

Section 4将会详细讨论能够提供分割的ground-truth和对应图片的公开数据集。

考虑到这些，找到候选路径的一个简单方法就是从ground-truth分割结果中抽取正面的样本，并使用随机路径作为负面的样本。但是这将会导致产生的候选路径对于分类来说过于简单，缺少有效监督信息（。。🤣有一点缺少理解）。

因此，我们最初只训练网络的分割部分，当损失稳定后，我们再用它计算管状图。然后使用该图片去获得正路径样本和负路径样本。这能让我们联合训练网络的两个分支，同时我们在训练路径评分分支中使用的路径是真实的，而且与在推理阶段可能遇见的路径相似。如下图中所示：

![image-20210121092902519](D:\Documents\PaperLibraries\LearningNote\Curvilinear_Structures\Joint_Segmentation_and_Path_Classification_of_Curvilinear_Structures\assets\image-20210121092902519.png)

上下两排分别展示了在不同场景（道路场景和神经通道）下的候选路径，上排图片中的黄色线条和下方的白色线条代表候选。在这两个场景中，只要路径仍然在线性结构中，即使他并不沿着中心线，我们也认为他是正样本，而对于一些从一个线性结构跨越到另一结构，或是走过了捷径的，我们认为他是个负样本。

对于一张分割的管状图，将按照下面三个步骤获得候选路径：

1. 找到图中的节点。我们首先阈值化管道图，描绘出管道图中的道路结构，寻找到其中的关键节点（包括交点和端点）。端点仅有一个邻居节点，交点有至少两个邻居。由于这些关键点在图片中出现的并不均衡，我们以规则的间隔对线性结构进行采样，限制相邻节点之间的最大距离。

2. 连接图中节点，如图：

   ![image-20210121144418064](D:\Documents\PaperLibraries\LearningNote\Curvilinear_Structures\Joint_Segmentation_and_Path_Classification_of_Curvilinear_Structures\assets\image-20210121144418064.png)

   图中绿色点代表ground-truth分割图或是管状图中拓扑学上的关键点（即交点或端点），网络将基于这些点生成过充分的连通图（over-complete graph）但是有些节点间距离可能过大，为了对其做出一些限定，我们引入了虚线构成的规则网格，他与路径的交点，作为额外的节点，在图上表现为蓝色节点。在此基础上，为了防止节点之间过于接近，我们引入了半径为ε的排除区，同时也是绿色节点的半径，在此范围内不可再有其他的节点。这确保了相连的节点之间的距离在$\varepsilon$和$d$之间（离谱，，，没提到$d$怎么来的。。我裂幵🙄），即节点之间的距离小于$d$时，我们将其相连，这就产生了一个过完备图，如之前提及的Fig.4所示，每条边对应两个节点之间的最短路径，然后我们用$A^{*}$算法来提取这些路径。对于一个起点和目标点，通过向路径添加与当前路径端点相邻的像素$(x, y)$来迭代地增长路径，并使下列值最小化：
   $$
   \begin{eqnarray}
   f(x, y) &=& c(x, y) + h(x, y), \\
   c(x, y) &=& 1.1 − p(x, y), \\
   h(x, y) &=& 0.5 * d(x, y)
   \end{eqnarray}
   $$
   $p$：分类网络输出点$(x, y)$属于通道的分数

   $d$：点$(x, y)$到目标的欧氏距离

   事实上，$c$代表了将一个新的像素点添加到路径中的花费，而$h$近似于从$(x, y)$到目标的最短路径的损耗。这对于可能非常大的图而言，在计算速度和贴近真实最短路径之间提供了一个很好的折中方法。

3. 选取正负样本。**与分割真实值重叠90%以上的路径视为正样本，其余的则被认为负样本**，就如Fig.5中所示，候选路径并不沿着中心线，但只要他仍处于显现结构中，他依然会被视为有效路径。而那些从一个线性结构穿到另一个线性结构或是采取了一个捷径的候选则被视为无效者。从我们的早期工作中“**Reconstructing Curvilinear Networks Using Path Classifiers and Integer Programming**”发现，消除后者引起的错误是极为关键的。

为了Section 3的简洁，我们以整张图片进行描述。但在实践中，我们的网络从训练图像中裁剪出patches进行训练。为了获得这样的patches用来训练，我们使用上面讨论过方法选出正负样本，然后裁剪其所在的部分图片，如果一个路径尺寸不符合我们的U-Net编码器输入要求，我们将对路径进行切割。

### 3.4 Inference

我们先使用双流U-Net网络进行分割，然后将其结构化，按照Fig.6的方法拓扑化，然后按照$A^{*}$算法获得路径，最后保留得分高的路径。

这流程的关键在于我们用同样的网络去计算管状图，并评测路径，确保了两个分支使用了同意图片特征。

## 4 RESULTS

### 4.1 Datasets and Baselines

我们在两个线性结构非常不同的两个数据集上进行了评测。

- Roads数据集，图片大小为$4096 \times 4096$，由于计算内存限制，我们会将原图和ground-truth以因子2进行下采样，之后为了获得overcomplete graph，我们进一步对管道图进行一半的下采样，而在评测结果的时候，进行上采样与原图同等大小，然后进行评价。
- Axons数据集，神经元数据集

对于Roads数据集我们设置平均节点采样距离$d=250$，而最大的连接距离为$1.1d$

对于Axons数据集我们设置平均节点采样距离$d=30$，而最大的连接距离为$1.5d$

我们将把我们的方法与两种当前对道路检测的方法和神经元描绘方法进行比较

- **RoadTracer**，通过CNN网络训练跟踪路径上下一像素的方向，我们在Roads数据集上与其进行性能比较
- **QMIP**，步骤与我们的类似，但需要两个独立的分类器，分别获得管道图和路径分类结果。然后使用混合整数规划优化来寻找最优子图。

我们还将我们的结果与那些通过简单的分割勾勒出路径结果进行比较。

### 4.2 Evaluation Metrics

评价指标

我们的算法的目标是产生近似线性结构中心线的路径，同时保持网络的整体拓扑结构

因此我们使用拓扑敏感的指标对轮廓算法进行性能评估。

这些算法都很关注拓扑学中的关键点（交点或是端点），以及关键点之间的连线，往往都通过ground-truth中找到关键点和最短路径，然后在预测图中找到相应的有效点和最短路径。它们的不同之处在于测量和使用两条路径相似性的方法上。

论文中接着呈现了

- 与独立训练分割和分类相比，同时训练分割和分类，能同时提高两个分支的表现

- 定量的分析两类指标

  - Normalized Path Difference（考虑的是全局）

    当ground-truth中存在一对连接的点$a^{*}$和$b^{*}$，他们之间的最短距离为$l^{*}$，如果在预测结果中，在上述两个点附近不超过半径为$R$的距离中存在各自对应最近的点$a$和$b$，然后计算两者的距离$l$，然后计算这两个距离的差异性：
    $$
    min\left\{\frac{\left|l-l^*\right|}{l^*}, 1\right\}
    $$
    如果预测结果中不存在对应$a$和$b$，则其差异性为$1$。我们会计算ground-truth中所有“点对”，实际上，在Roads数据集上，我们将$R=40$，在Axons数据集上我们设置为$10$，基本上这是各数据集上的交点之间的最小值，然后我们绘制所有不同分数的累积分布

  - Topological precision and recall（关键点附近，局部）

    论文的选择是逐像素而不是逐路径，对于所有关键点对之间的路径，选择逐像素点的去判断是否与真实值相同。
    $$
    \begin{equation}
    \begin{split}
    & precision &= &\frac{1}{\sum n_m} \sum \frac{n_m}{n_t} n_m \\
    & recall &= &\frac{sumn_m^*}{n_t^*}
    \end{split}
    \end{equation}
    $$
    $n_m$：预测路径的像素点与对应ground-truth的像素点距离超过$m$大小的像素个数

    $n_t$：预测路径中的所有像素点

    $n_m^*$：ground-truth路径中像素点和预测路径中对应像素点距离不超过$m$的像素个数

    $n_t^*$：ground-truth中所有像素点个数

    如果不存在匹配路径，则$n_m$和$n_m^*$都视为$0$。

    precision表示预测的路径位置的精确程度，并根据路径的长度加权。

    recall表明了由这些路径有效建模的地面真相的总体比例

- 可视化结果

### 4.3 Analysis and Comparisons

我们首先研究了联合分割和路径分类训练对这两种任务的效果。虽然我们发现它对分割没有明显的影响，但却大大提高了路径分类的性能，提高了两者之间的相互依赖性，同时加强了对负样本的检测

所有三种道路提取方法都具有相似的精度(除了道路边缘小，分割的精度优于其他两种方法)，但我们的方法有更高的召回率。归一化路径差异直方图也表明，联合路径分类和分割减少了不连通段的数量。

## 5 CONCLUSION

提出了一种曲线结构的节点分割和连接分类方法。我们利用这两个任务的相似性来训练一个可以同时执行这两个任务的网络。结果它们彼此都得到了优化，我们的方法在描述道路和神经元的非常不同的数据集上比最先进的方法表现更好。