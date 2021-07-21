# A Topology Layer for Machine Learning

这里可能是不全的，下次做全些

[toc]

- 词短句翻译

## Abstract

基于持续同调性的拓扑在机器学习包括深度学习中已经能有一定的应用，

本文提出一种基于水平集过滤和边过滤计算持续同调性的可微拓扑层，……

## Topological Preliminaries

这里建议重看下**Topological_Persistence_And_Simplification**

本文中使用的*Loss function*：
$$
\varepsilon(p,q,i_0;PD_k）= \sum_{i=i_0}^{\left| \mathcal{I}_k \right|} \left | d_i - b_i \right |^p \left (\frac{d_i + b_i}{2} \right )^q
$$

从存活时间最长的$i_0$点开始计算所有的存活时间，不同的$i_0$设置代表着$PD_k$​（*k维Persistence Diagram*）有不同数量的*k*维特征不需要被惩罚，举例而言，

当$i_0$为`2`时，对于$PD_0$则损失函数只会计算存活时间第二长以及之后的连通区的总存活时间并进行抑制，总的而言，就是允许结构中存在一个连通区，同理，对于$PD_1$，则倾向于保留一个孔（*hole*）

$p$​则代表对于持续的特征的惩罚力度，存活时间越长的结果惩罚力度越大

$q$则衡量着某一特征在后续滤波中的权重（水平集滤波或是别的什么……​

