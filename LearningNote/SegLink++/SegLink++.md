# SegLink ++ : Detecting Dense and Arbitrary-shaped Scene Text by Instance-aware Component Grouping

- 词短句翻译

	curved：弯曲的

	commodity：日常的

	aspect ratio：宽高比
	
	extract：提取
	
	granularity：粒度
	
	surpass：超出，胜过
	
	overperform：超出，胜过
	
	bottleneck：瓶颈
	
	address：设法解决
	
	explicitly：明白地，明确地
	
	repulsive：排斥的
	
	estimation：评估，估计
	
	exploit：利用
	
	facilitate：促进
	
	commodity：商品，有价值的东西
	
	aspect ratio variation：大纵横比变化
	
	centroid：形心，质心
	
	holistic：整体的
	
	utilize：利用
	
	stochastic：随机的
	
	recurrent：周期性的
	
	thorough：完全的，十分的
	
	compensates：弥补
	
	differentiable：可微的，可分辨的
	
	aforementioned：之前提及的
	
	depict：描述，描绘
	
	leverage：利用
	
	polygons：多边形
	
	sake：目的，利益
	
	alleviate：减轻缓和
	
	integrate：使完整，使成为整体
	
	hyperparameter：超参数，机器学习算法中的调优参数，需要人为设定
	
	in term of：根据
	
	vice versa ：反之亦然
	
	coherent：连接的，相关的
	
	redundant：过剩的，多余的

## Abstract

针对什么领域，当前这个领域里的方法已经很好了，但仍然有哪些不足？我们基于什么思想提出了

什么方法解决了什么问题？基本思想是从前就存在的，依据同一思想的方法也是有的，那么同一思想中方法的问题，你是如何解决的？在你的算法里具体是什么样的？算法的效果如何？

论文中的一些词有不同的层次，实验效果特别好：outperforms；还可以的：achieves very competitive performance，很有趣🤣。

针对文字检测，但当前方法对密集文字和弯曲文字的识别度仍然不佳，我们依据“instance- aware component grouping (ICG)”自底向上地检测密集或是任意形状的文字，此外为了解决大多数自底向上方法在分离密集文本时遇到的困难，我们提出了文本组件之间的吸引链接和排斥链接，能够更好的处理紧密文字，更好的利用上下文进行实例感知，最后在网络训练所得的吸引和排斥的基础上，依据最小生成树（MST）算法进行最后文本的检测。通过DAST1500这种针对密集性、弯曲性文字的数据集进行算法的评判，在多个数据集上的表现都比较优异。

## Introduction

场景文字检测的难点：多纵横比，大小相差大，多方向，外形多样，同时文字所处环境光影和视角的变化都比较大，传统OCR对此极难处理。

传统方法往往通过多个阶段逐步解决上述问题：

1. 通过被选择设计的特征（engineered features，这个词组的理解还没到位）提取出组合元素区域（component regions，指的应该是可能组成字串的小区域）
2. 筛选出候选组件区域
3. 组合提取出的要素通过串联等形式形成文字。

传统方法往往受限于被选择设计的特征而且需要复杂的后处理（heavy post-processing）。

深度学习在这一方向的应用，使得相关工作有了极大的发展进步。总体来说，当前基于深度学习的文字检测算法可以分为自顶向下和自底向上两类。顶-下方法主要受到一般物体检测的影响，专注解决文字检测的多方向性和大宽高比问题，它通常直接回归得到水平/转向的矩形或四边形。而底-上方法则是依据传统文字检测的思路，首先通过CNN获得组合要素然后通过要素的聚类分组组合（group）。自底向上方法更加的灵活更加能够检测出任意形状的文字。根据底-上方法中组合要素的粒度，它可以进一步被分为：像素级（pixel-level）底-上方法，零件级（part-level）底-上方法。

顶-下方法虽然已经在多方向性文字检测领域取得了重大的成果，但是他们仍然对弯曲文字和大纵横比文字感到头疼。

在弯曲文字检测方面，底-上方法远超顶-下方法（Recently, on curved text detection benchmarks, bottom- up methods **surpass** the top-down methods **by a large margin**），但由于需要繁重的后处理任务，底-上方法仍具有效率瓶颈。此外，底-下方法对于如何区分紧密文字块同样有问题。虽然在当前弯曲文字测试标准中较少出现，但他们在日常生活场景中是普遍存在的，因而需要形成一个密集多方向性文字检测的数据库并且提出一种能够很好解决相关性问题的场景文字检测方法。

论文受分水岭算法的启发（It is inspired by the mutex watershed algorithm for neuron segmentation）【旁白：这个算法没听过，可以以后单独成章，冈萨雷斯《数字图象处理》中有提到的】提出了一种**组件感知聚合方法**（instance-aware component grouping）。通过引入排斥链和吸引链，ICG能够很好辨识密集多向文字。

整体网络基于SegLink进行了再创造，吸引链和排斥链通过同一CNN feature训练得到。同时为了更好的利用文本上下文以达到区分密集文字的目的，提出了一种instance-aware损失，使得备选组合要素中不存在文字的区域有更大的损失权重（loss weight）。值得一提的是这个思想并不局限在SegLink这一方法中，所有底-上方法都可以参考这一思想。

我们根据标准PASCAL VOC的格式制作了DAST1500数据集，全部由包装纸上写有商品描述的产品图片组成，这个数据集能够使训练模型更好的专注于密集任意方向的文字。在此数据集上，ICG表现更佳。

本篇论文主要贡献在三个方面：

- 提出了ICG（Instance-aware component grouping）结构，能够结合到底-上方法中，并进一步推动这一方法在密集，多向文字检测的应用发展
- 提出了一个新的数据集
- ICG在弯曲多向文字上的检测效果更好

接下来的文章结构如下：

- 文字检测相关工作介绍（Section2）
- 提出的方法细节（Section3）
- 实验结果（Section4）
- 总结，观点（Section5）


## Related work

### Scene text detection

在深度学习介入之前，文字检测有一套经典自底向上的流程：文本模块提取筛选，文本区域组合以及对文本候选区域进行筛选。大连的工作专注于通过特定的特征提取文本组合元素（component），比如MSER（Maximally Stable Extremal Regions）和SWT（Stroke Width Transform）。近些年，深度学习在这领域广泛应用，同时在准确性和效率上也远超以往的方法。一般来说深度学习可以分为两个方向：自顶向下和自底向上两种。

顶-下方法受一般物体检测影响，基于先验框，通过回归得到文本框。在检测中，面对文本的大纵横比变化，通过SSD网络，使用长先验框和长卷积核来处理文本的大纵横比变化（applies long default boxes as well as convolution kernel to deal with large aspect ratio variation of text）。TextBoxes++在此基础上更进一步，面对多向文字通过回归多边形角点坐标进行检测。SSTD则是通过FCN引入注意力模块去增强文字检测的训练过程和多尺度的检测。面对多变的方向问题，相继产生了很多方法：

- R2CNN调整了Fast-RCNN的流程，添加了倾斜框的预测
- Deep Matching Prior Network，使用了两个分支，一个是旋转相关的特征用于检测，另一个是旋转无关的特征用于分类，进而得到更好的长的多向文本
- Instance transformation network，学习几何感知表示信息（啊，这……先看看吧，不知道，先余着）来测定文本方向

底-上场景文字检测，则是与传统方法类似的流程，首先检测出文本组件区域（text components），然后让他们串联。一般而言，底-上方法可以被分为像素级和组件级方法。

- **像素级**，将文字检测视为分割任务，因此在这方面，FCN网络经常被用于生成像素级分类特征图，然后通过后处理将文字像素组合成实例对象。
  1. 得到文本分割图，获得字符质心，然后获得文本实例
  2. 通过FCN以整体方式提取字符文字区域（mark一下，“Scene text detection via holis- tic, multi-channel prediction”）
  3. 利用FCN的多尺度输出，通过级联FCN产生文字
  4. 将文字的分割视为三类的分割，除了文字区，非文字区，还有一个边界区域。这一观念的进一步发展则是，使用语义敏感文本边界以及自展技术（bootstrapping technique）去生成更多训练实例（Bootstrapping算法，指的就是利用有限的样本资料经由多次重复抽样，重新建立起足以代表母体样本分布的新样本）
  5. Deep Direct Regression和EAST则是预测估计（estimate）基于像素的多边形
  6. PixelLink 则是使用八个方向的链接（8-direction link）识别文字边界然后组成文字实例
  7. 将图片在像素层面视为随机流图（SFG），然后使用马尔可夫聚类网络聚合成文字区域（Markov clustering Network）

- **组件级**，文字区域被视为一块块的文本组件的组合。
  1. CTPN利用固定宽度的文本块进行水平文字的检测，然后通过RNN网络进行组件的链接
  2. SegLink则是通过学习分割区域以及8-邻域之间的联系，进而组合为文字实例。其作者认为可以利用四个检测得到的角点与四个部分的分割图生成文本实例（os：这不是Coner算法？下论文可以看出Corner算法的思想，“Multi-oriented scene text detection via corner localization and region segmentation”）
  3. CTD，回归得到文本内容的多个角点，然后通过TLOC提炼出结果
  4. TextSnake将文本区域视为一组Disks，实现曲线文本检测（可以想象贪吃蛇🐍）

### Comparison with related works

与传统方法相比，ICG具有类似的流程，但通过学习得到的文字组件和吸引/排斥链接极大加强了检测能力（包括准确性以及效率）。

与顶-下深度学习方法相比，ICG不但在多方向文字检测方面具有很强竞争能力，而且在多形状文本方面更加准确（the proposed ICG has the advantages to accurately detect arbitraryshaped texts while maintaining competitive results on multioriented text detection）。

ICG作为底-上方法，旨在解决一个以往的底-上方法并没有投入过多的关注的问题——密集多形状文本检测。ICG提出几个重要想法：

- 吸引/排斥链能够区分紧密的文本内容
- 定义的实例敏感损失（Instance-aware loss）能够弥补一般底-上方法后处理过于复杂难以达到端到端训练目的的问题（The proposed instance-aware loss somehow compensates the drawback of bottom-up methods which usually involve a postprocessing that cannot be trained in an end-to-end way）
- 上述方法可以被普遍的应用于密集，形状多变文本的检测

## Methodology

### 3.1. Overview

顶-下方法受一般目标检测的影响，在多向文本检测中大放异彩，但他们在自然场景中多见的弯曲文本检测方面遇到了问题。

底-上方法在任意形状文本处理中，显得更加游刃有余，也因此成为这一领域的砥柱中流。但底-下方法有如下两个主要问题：

- 难以区分距离较近的不同实例文本，对于密集文本而言，相对较近的文本区域，可能会被检测为同一文字领域
- 繁重的后处理流程难以在端到端方法中实现。底-上方法通常先检测文本组件或文本像素，然后进行组合。后处理模块不被包括在网络中，难以通过训练进行优化

为了通过底-下方法解决上述的两个主要问题——密集，多形状的文字检测，新提出一种ICG结构（instance-aware component grouping framework），网络工作流程如下：

![image-20201019200350904](assets/image-20201019200350904.png)

网络通过VGG16进行特征提取，获得多层次的特征输出，在此基础上，根据文本组合元素以及各元素之间的吸引/排斥关系，通过类似最小生成树的算法（modified minimum spanning tree），组合小部分，进而得到多个多边形的文本检测框，再通过多边形NMS获得了最终结果。在这一流程中，我们利用文本实例敏感的损失函数，使得后处理的过程能够更好的与网络结合，通过训练调节后处理的效用。

ICG方法在Section 3.2中进行了详细描述（The proposed instance-aware component grouping framework
detailed in Section 3.2 is **rather general**），同时要认识到，这个思想能够被广泛的应用到各种底-上方法中。

在本论文中，我们改进了SegLink，将SSD作为主干网，新的网络结构呈现在Section 3.3。Section 3.4主要是训练的标签初始化过程；Section 3.5是网络的优化；Section 3.6则是文本推断和后处理的过程（The inference and post-processing，嘿嘿嘿这里本来不应该这么翻译，但能够感受下就行🤣！！）

### 3.2. Instance-aware component grouping framework Bottom-up

（os：第一段老生常谈，但英语描述让人觉得相对还是有新意的）

底-上文本检测通常在密集文字和任意形状场景文字检测上更加的零花，为了减轻解决分离间距过小、以及后处理难以优化的问题（To alleviate the two major problems of bottom-up text detection methods: diffi- culty in separating close text instances and non-optimized post- processing），ICG闪亮登场。

ICG由两个模块组成，分别解决上述的两个难题。

- **文本组件区域通过吸引/排斥链进行拼接**（Text component grouping with attractive and repulsive links），图片中的文本内容往往被认为是一组组具有相同几何特性邻近字符组成的序列。底-上方法处理任意形状的文本非常灵活，先提取文本组合区，然后将其组合。而后处理流程或是基于先验规则（heuristic rules，经验），或是组合规则，或是学习得到的文本区域之间的关系。我们延续了底-上方法，继承了他们的灵活性。我们利用SSD，通过卷积，根据先验框获得文本组件区域。而除了学习传统的两组件区域之间的吸引链接外，我们也认为需要学习文本组件区域之间的排斥关系，便于区分紧密文字。同时在另一阶段也可以继续利用吸引和排斥关系解决多尺度文本检测问题。网络中吸引排斥链接不仅在同层相邻文本组件的之间，也存在于跨层相邻文本间（这里的层代表的是[Section 3.3](#3.3. Network architecture)中提及的各卷积层输出），最终组成了类似于边赋权图$G$，可以如下方式表示：
	$$
	G=(V,E)
	$$
	$V$是不同分辨率特征金字塔中的点的集合，$E$则是同层或跨层相邻点之间连接边的集合，每条边$e$有两个权重

	1. 吸引力（attractive force），$w_a(e)$
	2. 排斥力（repulsive force），$w_r(e)$

- **网络使用实例敏感损失函数**（Network training with instance-aware loss），以往的底-上方法难以将繁重的后处理任务结合到网络训练中进行优化。为了减轻这个问题，我们将基于最小生成树（MST算法，一般有Prim算法和Kruskal算法，基于图中文本组件区域是比较多的，边的数量也是极大的，或许使用Prim算法更加合适，此外Prim算法可以利用Fib堆进行优化。不过这里的细节。。emmm，先记录下😂）的后处理集合到网络训练中，并由此提出了ICG损失函数。此外我们还使用了IoU（intersection-over-union），将检测结果与标签刻画的实际文本区域的重合度作为一个考量网络损失的方面，如下是几个基本量：

	$g_i$：第$i$个groud-truth text instances（标记的文字实例）

	$d_i$：相对于第$i$个groud-truth text instances（标记的文字实例）的检测结果的集合

	$IoU_i^m$：$g_i$和$d_i$中每一检测结果计算IoU取得的最大值

	$d_i^m$：$d_i$中的所有检测框和$g_i$计算IoU，当取得$IoU_i^m$（当IoU为maximum）时的检测结果

	在此基础上，我们通过$\frac{1}{IoU_i^m}$来衡量基本符合$g_i$（ground-truth）回归文本组件以及组件之间吸引排斥关系（或者是说”对应标注框与最优检测框的匹配程度“，具体使用可见[Section 3.5](#3.5. Optimization))
	
	总结：
	
	ICG在大多数底-上方法中都能够使用，并能很好地分离相近文本，将后处理结合到网络训练之中。
	
	### 3.3. Network architecture
	
	新的网络结构在SegLink的基础上，借鉴SSD网络进行搭建，具体如下：
	
	![image-20201019090954930](assets/image-20201019090954930.png)
	
	

我们将VGG16作为主干网络去提取图片特征，并将最后两层全连接网络$fc_6$和$fc_7$替换为卷积层$conv_6$和$conv_7$，在此基础上又添加了$conv_{8\_1}$到 $conv_{11}$的卷积层，更深层次的卷积特征能够扩大网络的感受野进而解决处理多尺度的文本检测。我们选择了六个卷积层（$l=\left\{1,2,...,6\right\}:conv_{4\_3},conv_7,conv_{8\_2},conv_{9\_2},conv_{10\_2}, conv_{11}$），利用$3*3$卷积层进行处理，获得文本提取区域和吸引/排斥链接。

- **文本提取区域的学习**，对于形状多样的文本内容的检测，我们首先用$(x,y,w,h,\theta)$元组表示一个旋转的文本组件，$(x,y)$是文本元素的中心点，$w$和$h$则是文本组件的宽高，$\theta$则是旋转角度。因此，在设置SSD网络中的先验框时，我们对不同卷积层设置了不同高度的先验框$h_d^l,l=1,2,...,6$（这里需要注意，对于文字组件区域的宽度，我们并没有调整，一方面是没必要，另一方面减少变量，我们将每一块宽度视为定值，但不同深度的文本组件区域，他的高度可能是变化比较大的）：
  $$
  \begin{equation}
    \begin{split}
      & h_d^1=12 \\
      & h_d^2=24 \\
      & h_d^3=45 \\
      & h_d^4=90 \\
      & h_d^5=150 \\
      & h_d^6=285 \\
    \end{split}
  \end{equation}
  $$

  每层先验框的高度按照如下比例进行设置：
  $$
  h_d^l=a_l\approx\gamma\frac{w_I}{w_l}
  $$
  $w_I$：图片的宽度

  $w_l$：特征图$l$的宽度

  $\gamma$：超参数，文中默认为1.5

  文本组件提取模块会输出8通道特征图，其中两维会经过Softmax层得到分类结果分数$s$，其余六通道则会被用于预测定向文本组件的几何特性（the other 6-channel is reserved for the geometrical properties of oriented text component representation）。

  值得一提的是，网络并不直接回归得到文本的旋转方向$\theta$，而是通过回归获得$sin\theta$和$cos\theta$。

  综上所述，对于某一卷积层的输出特征层$l$，我们用元组$(x_g,y_g,w_g,h_g,\theta _g)$表示$(x_d^l,y_d^l)$（检测过程中以点$(x_d,y_d)$为预测文本组件中心点）处文本的真实位置（监督学习，由标签给出），然后在文本分类之外，文本组件检测模块还会去回归$(\Delta x,\Delta y,\Delta w,\Delta h,\Delta sin \theta,\Delta cos \theta)$中的各项，回归方式如下：
  $$
  \begin{align}
    & \Delta x=\frac{x_g-x_d^l}{a_l}, \\
    & \Delta y=\frac{y_g-y_d^l}{a_l}, \\
    & \Delta w=log(\frac{w_g}{a_l}), \\
    & \Delta h=log(\frac{h_g}{a_l}), \\
  \end{align}
  $$

- **吸引/排斥链接的训练**，网络在上述模块后会进行吸引/排斥链的训练，我们将这一训练过程视为两个二元分类问题，通过Softmax去对特征图中的四个通道进行处理。正如Section 3.2中所介绍，我们会同时预测同层和跨层之间的排斥吸引链。对于同层链（For the within-layer links），我们采用8-连通性进行衡量。而对于跨层链，除第一层外，当前层$l$中的每一个点$p$都会被连接到上一层$l-1$中与$p$对应能汇聚为$p$代表区域的四个点（什么意思呢😶？就是说在$l-1$层有四个区域，当他们再次经过卷积后，变为$l$中的一个区域，而$l-1$中四个区域的中心点就与$l$中的区域中心点$p$有着对应关系），因此这一模块在第一层的输出结果中有$8*4=32$个通道数，剩余层则有$(8+4)*4=48$通道数

### 3.4. Ground-truths generation

在场景文字检测时，多项文字往往用定向矩形或四边形描述，弯曲文本多用有多个四边形组成的多边形描述。

在我们的网络中，我们需要先对文本组件的标签和局部吸引排斥链标签进行初始化（generate text component level ground-truths and local attractive and repulsive link ground-truths）。遵循着一般性原则（Without loss of generality），我们在下图展示了怎样通过弯曲文本的标签文件根据由多个四边形组成的多边形生成我们需要的训练基准（ground-truth）

![image-20201020095602646](assets/image-20201020095602646.png)

对于一个文本像素$p=(x_p,y_p)$，$Rect_p$代表最符合对应包含$p$的分割四边形的定向矩形。为了求取在$p$处文本组件的真实位置标签$(x_g,y_g,w_g,h_g,\theta_g)$。$\theta_p$是定向矩形$Rect_p$的旋转角度，剩余的四个几何特性，我们先将矩形$Rect_p$以$p=(x_p,y_p)$为旋转中心顺时针旋转$\theta_p$角度，与$p$点处的水平先验框对齐（就是矩形$Rect_p$转为水平，以$p=(x_p,y_p)$为旋转中心）生成$Rect_p'$，如$Fig.4(b)$。然后我们在$Rect_p$中找到与先验框同等大小，竖直方向相同的框（可以这么思考，就是将先验框上移与$Rect_p'$水平重合，如$Fig.4(c)$，如果先验框的水平长度超出$Rect_p'$则对先验框的长度进行修减。最后我们再将$Rect_p'$绕着$p$点逆时针旋转回原方向，$Rect_p'$中的黄色矩阵则是对应着原始的先验框的文本组件区域。对于像素点$p$，根据下面的条件进行选择什么规模的先验框：
$$
max(\frac{h_d}{h_g},\frac{h_g}{h_d})\leq1.5
$$
$h_d$是先验框的高度，当多个ground-truth文本实例包含$p$时（密集文字中可能遇见的情况），能够使上述公式结果最小的ground-truth作为可信的ground-truth，点$p$也被视为属于此ground-truth，并在此基础上进行吸引排斥关系的学习。如果所有先验框与ground-truth的公式计算结果都不符合上述条件，则该先验框是不合理的。除此之外，所有中心点$p$在分类时划分为非文本类时的先验框都是被认为是负样本，也正因如此，我们不必去为非文本类像素生成ground-truth text components。

对于给定的链接$e=(p_1,p_2)\in E$，

当$p_1$和$p_2$属于同一个ground-truth时，则$e$的吸引力设置为$w_a(e)=1$。当$p_1$和$p_2$属于不同的ground-truth时，则他们之间的排斥力$w_r(e)=1$，对于其他情况则吸引和排斥都为0，$w_a(e)=w_r(e)=0$

### 3.5. Optimization

- Training obejective。新网络可以视为一个多任务网络，分别需要文本元素提取以及吸引排斥链的学习。

  - 对于检测文本组合元素，我们改进了 传统目标检测的损失函数。

    $l_g$：ground-truth的location

    $l_p$：predictted text的location

    $s_g$：ground-truth的分数

    $s_p$：predicted text的Score

    $N_d$：以匹配的先验框的数量

    $\alpha$：超参数，人为设定

    $L_{conf}$：二分类Softmax损失函数

    $L_{loc}$：smooth L1 loss

    $w$：实例敏感损失权重，详情可见[Section 3.2](#3.2. Instance-aware component grouping framework Bottom-up)

    这里$s_g$可能指，如果这个区域是文字，则$Score=1$，否则为0

    检测模块损失率$L_C$由下式给出：
    $$
    L_C(s_g,l_g,s_p,l_p)=\frac{L_conf(s_g,s_p,w)+\alpha\times L_{loc}(l_g,l_p,w)}{N_d}
    $$
  
- 对于吸引排斥检测，
  
  $w_a^g$：ground-truth的attractive的weight
  
  $w_r^g$：ground-truth的repulsive的weight
  
  $w_a^p$：predicted text的attractive的weight
  
  $w_r^p$：predicted text的repulsive的weight
  
  $N_a$：the number of attractive links
  
  $N_r$：the number of repulsive links
  
  $\beta$：超参数
  
  $L_{conf}$：二类Softmax损失函数
  
  损失率$L_E$如下：
  $$
    L_E=\frac{L_{conf}(w_a^g,w_a^p,w)+\beta\times L_{conf}(w_r^g,w_r^p,w)}{N_a+N_r}
  $$
  
  
  最后，将这两部分损失率按一定比例相加，得到最终损失函数$L$
  
  $\lambda_1$和$\lambda_2$：是两个hyper-parameter。
  $$
    L=\lambda_1\times L_C+\lambda_2\times L_E
  $$
  
- Online hard negative mining（负难样本挖掘）

  在两个模块中都设置正负样本比为3，需要注意的时，吸引排斥训练模块中，我们只考虑那些至少与一个文本类型像素的负样本边

### 3.6. Inference and post-processing

向网络输入一张图片，我们将获得文本组件分类分数$s_p$，对应的相关几何特性和吸引/排斥链的权重$w_a$和$w_r$。需要注意的是，在同一层卷积输出时，每条边$e$会被预测两次，$e=(p_1,p_2)$和$e'=(p_2,p_1)$，最终$e$的$w_a$和$w_r$由两次预测中的最大值决定
$$
\begin{equation}
  \begin{split}
    & w_a(e)=max(w_a(e),w_a(e')) \\
    & w_r(e)=max(w_r(e)，w_r(e')) \\
  \end{split}
\end{equation}
$$
然后，我们在互斥分水岭算法（mutex watershed，具体是什么……mark一下）的启发下，修改最小生成树算法将文本组件组合成宇轩文本实例，然后利用多边形组合和多边形极大抑制（NMS）完成文本检测工作。

- Modified MST（改进的最小生成树算法）

  | **Algorithm 1：**Modified MST for text component grouping based on learned text score s p , attractive w a and repulsive w r links. |
  | ------------------------------------------------------------ |
  | 1 $Text_Inference(w_a,w_r,s_p,t_s,t_l)$<br />2 // **Initialization**<br />3 $E^+={e=(p_1,p_2)\in E|w_a(e)>t_l}$ **and** $max(s_p(p_1),s_p(p_2))>t_s$；<br />4 $E^-={e=(p_1,p_2)\in E|w_r(e)>t_l}$ **and** $max(s_p(p_1),s_p(p_2))>t_s$；<br />5 $A^+\leftarrow\O,A^-\leftarrow\O$；// set of attractive ( resp. repulsive) links in final MST；<br />6 **for** $e=(p_1,p_2)\in E^+\cup E^-$in **descending** order of $max(w_a,w_r)$ **do**<br />7      **if** $w_a(e)>w_r(e)$ **then**<br />8          **if not** $connect(p_q,p_2)$ and **not** $mutex(p_1,p_2)$ **then**<br />9           // merge $p_1$ and $p_2$ and update mutex constraints；<br />10             $merge(p_1,p_2),A^+\leftarrow A+\cup e$； <br />11    **else if** $w_a(e)\leq w_r(e)$ **then**<br />12         **if not** $connect(p_1,p_2)$ **then**<br />13              $addmutex(p_1,p_2):A^-\leftarrow A^-\cup e$ // add mutex constraint between $p_1$ and $p_2$；<br />14 $D\leftarrow CC_Labeling(A^+)$；//Grouping by connected labeling；<br />15 **return** $D$； |
  
  推论阶段，我们**先**对预选文本组件考察他的文本类别预测分数$s_p$和他的吸引链$w_a$和排斥链$w_r$，$t_s$是类别阈值，$t_l$是吸引/排斥阈值
  
  当$s_p>t_s \&\& ((w_a > t_l) \left|\right|(w_r>t_l))$时，将边$e$保留在集合$E^+\cup E^-$中。
  
  **接下来**，我们将上述集合中的所有边按每条边的吸引力和排斥力中的最大值（$max(w_a,w_r)$），进行降序排列。
  
  **然后**对每一条边$e=(p_1,p_2)$（underlying linking edge，没想好怎么理解）,通过比较吸引和排斥力的大小，决定是否将包含点$p_1$和$p_2$的文本组件合并，或是将$p_1$，$p_2$所在组件设置为互斥，进行文本的区分。对两个已被较大排斥力所区分的文本组件将不会被较小的吸引力所影响，反之亦然。（原文：For two text components already having a mutex constraint added by a larger repulsive weight, the current linking edge with a lower attractive weight will not result in a merging process, and vice versa）。对于每次合并，我们也会所有文本组件之间的互斥情况（就像是最小生成树Prim算法中的更新距离矩阵的概念）。
  
  **最终**，根据所选的吸引链$A^+$将所有的文本组件合并成标签图$D$（图解结构）。
  
- polygon combination（多边形联结）。上一步骤结束后，我们获得了已经被分组文本组件，接下来，我们需要将其变得连贯。我们**首先**筛掉了一些孤立的文本组件，**然后**对于多向文本检测，我们将每个分组的文本组件转换为最小定向的矩形，对于弯曲文本检测，我们将每个分组的文本组件转换为边缘最少的多边形（原文：Then for multioriented ( resp., curved) text detection, we transform each grouped text component into a minimum oriented rectangle ( resp., a polygon with minimum edges). 初始译文应该是”对于多向（弯曲）文本检测，我们将每个分组的文本组件转换为最小定向的矩形（边缘最少的多边形）“这个”resp.“多熟悉熟悉），**随后**我们再过滤掉，一些平均高度过小或范围区域过小的检测结果。

- polygon NMS（多边形极大抑制）。为了进一步去除一些多余的几何图形，我们采取了多边形极大抑制，参考了如下$IoU$计算方式，进行调整

  对于多边形$A$和$B$，计算他们的$IoU'$如下：
  $$
  IoU'=\frac{\left|A\cap B\right|}{min(\left|A\right|,\left|Bs\right|)}
  $$
  最终我们的NMS的分数是多边形面积与其平均高度之间的比率。 该多边形NMS有助于摆脱一些小的重复检测。（原文：we adopt a polygon non maximum suppression (NMS) based on a modified $IoU'=\frac{\left|A\cap B\right|}{min(\left|A\right|,\left|Bs\right|)}$ for two polygons A and B , where $\left|\cdot\right|$ denotes the cardinality. The score for NMS is the ratio between area of polygon and its average height. This polygon NMS helps to get rid of some small and redundant detections.这里的语义理解有点问题……容我再思考思考🤣）
  
  ## 4. Experiments
  
  新算法主要针对密集多形状的场景文字，为了论证算法的有效性，我们第一次引入了一个多由商品图片组成的数据集——DAST1500，该数据集包含大量密集任意形状的文本内容，我们对这一数据集做了彻底的研究（We conduct ablation study on this dataset），并将研究结果与其他最先进的方法进行了比较，此外我们也在其他两个多向文本数据集和另两个弯曲文本数据集上，进行了网络的测验。