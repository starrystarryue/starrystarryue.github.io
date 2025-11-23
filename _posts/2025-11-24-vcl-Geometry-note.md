---
layout: post
title: vcl Geometry I
image: /assets/images/my_avatar.png
category: Notes
author: Starryue
---

# Geometry 几何

几何：

1. 几何表示
   计算机中如何编码几何？
   polygon mesh表示的优劣？triangle mesh的几种表示？
   什么是细分曲面？举例两个具体算法？
   什么是网格参数化？举出具体原因的例子？举例一个具体的算法流程？
2. 几何处理
   离散微分几何：平均邻域(3)、法向量、梯度、拉普拉斯算子(2)？
   网格平滑，从扩散模型到Laplace处理？两种Laplace算子的效果？
   Poisson图像编辑和Laplace网格编辑的关联？条件的表达？
   网格简化的动机？流程？Lab2是怎么处理的？
3. 几何重建
   什么是几何重建？输入、处理、输出是什么？处理时有什么核心步骤？
   二维和三维的平移变换怎么用矩阵表示？考虑三维绕不同轴的旋转？
   复述ICP的算法思路？初始化和SVD的数学计算？
   两种几何重建的方法分别是什么？具体过程？
   model fitting的方法？如何利用采样一致性？
4. 几何变换


## **08 Geometry Representation (几何表示)**

如何在计算机中编码几何？显式表示：点云，*polygon mesh*, subdivision surface ；隐式表示

### Polygon Mesh -- Triangle mesh (三角网格)

顶点存储为坐标三元组（x, y, z），三角形通过索引表示 ( i, j, k )

关于半边数据结构: Half-Edge Data Structures](https://jerryyin.info/geometry-processing-algorithms/half-edge/)

Polygon surface是简便表示且易于处理的表示，但是对于曲面则需要更多的面片来表示。

### Subdivision Surface (细分曲面)

用于创建复杂的曲面模型,，核心思想是通过不断*细分一个初始网格*，使得网格逐渐变得平滑，最终生成一个贴近原模型的光滑曲面。

#### 1）Catmull-Clark Subdivision

通常用于处理 *四边形网格*，每次细分增加顶点数量，并通过对顶点、面、边的更新来实现光滑效果。

##### 算法具体流程如下，在每次迭代中：

###### 1. 新增“边点”、面心

对于每个面（通常四边形），添加一个新的“面点“，位置是该面上所有顶点的位置的平均值，即对于四边形面 ，面点 f 的位置为：
$$
P = \frac{v_1 + v_2 + v_3 + v_4}{4}
$$
对于每条边，生成一个新的“边点”, 位置计算由该边的两个顶点的平均值，再加上它相邻两个面的面点的一部分，即边点e：
$$
e = \frac{v_1 + v_2 + f_1 + f_2}{4}
$$

###### 2. 更新顶点

生成新的点后，每个原始顶点的位置会根据它周围相邻的面和边的信息进行加权平均更新，按照公式：
$$
v = \frac{1}{n} \left( \frac{f_1 + f_2 + f_3 + f_4}{4} + 2\frac{m_1 + m_2 + m_3 + m_4}{4} + (n - 3) p \right)
$$
期中，m是边的中心（***不是“边点”!!***), p是顶点原来的位置，n是这个顶点邻近的边/面的数量。

###### 3. 将更新过的点与点相连，更新边

在更新了所有顶点的位置后，下一步连接这些新顶点，形成新的网格。

###### 4. 储存更新的面：

经过细分后，一个原始的面经过处理后会生成4个新的面，面的数将增加。每次细分后，网格的光滑度会提高，逐渐逼近一个理想的平滑曲面。

#### 2）Loop subdivision

常用于 *三角面片* 的细分，算法流程如下：

###### 1.每条边上新增顶点

v0 v1是一条边的两个端点，则：

如果这条边有 两个相邻面，新的顶点位置为：
$$
v_{\text{new}} = \frac{3}{8} (v_0 + v_1) + \frac{1}{8} (v_2 + v_3)
$$
其中v2 v3是相邻三角形的第三个顶点；

如果这条边只有 一个相邻面，新的顶点位置是该边的两个端点的中点：
$$
v_{\text{new}} = \frac{v_0 + v_1}{2}
$$

###### 2. 更新旧顶点

更新顶点 v 的位置：
$$
v' = (1 - n * u) v + \sum_{i=1}^{n} u v_i
$$
其中，

- v 是顶点的原始位置
- n 是该顶点的度数（即相邻的面数）
- u 是加权系数：n=3 则 u=3/16, 否则u=3/8n
- vi 是与该顶点相邻的每个顶点

通过以上操作，每个三角形都会被细分为 *四个子三角形*。

### Mesh Parameterization(网格参数化)

#### 网格参数化

网格参数化是将一个*三维表面（如一个三角形网格、四边形网格等）映射到 二维空间* 的过程，使得每个 3D 点都能与一个 2D 坐标对 (u, v) 相对应。

应用：纹理映射，世界地图绘制(立体投影——角度不变^面积变化、墨卡托投影——无角度变形^面积变形显著、朗伯投影——保面积和角度^地图外围的区域形变会比较大)

> ​	按照参数域的不同，网格参数化可以分为：(1)平面(planar)参数化，即将网格映射到平面；(2) 球面(spherical) 参数化，即将网格映射到规则的球面；(3) 基域(simplicial) 参数化，采用一个同构的简化模型作为参数域；以及基于其它参数域的参数化． 
>
> ​	按照参数化过程中保持哪种几何量，可以将参数化分为保长度 (Isometric) 的参数化、 保角度(Conformal) 的参数化、保面积 (Equiareal) 的参数化等等，即：将网格曲面映射到 平面上时，分别保证长度、角度、面积不发生扭曲．事实上，保长度的映射等价于既保角度 又保面积的映射．从三维网格表面映射到平面或其他参数域，理想的参数化是保形参数化， 即保持局部形状不发生改变．这里的形状描述包括长度、角度、面积等几何度量．但理论上只有可展曲面(developable surface) 存在保形参数化，如圆柱侧面，而一般的曲面在进行映 射时都会发生不同程度的扭曲． 	
>
> ​	寻找网格表面顶点和平面上点之间的一一对应，又可以分为两种类型的平面参数化：(1) 开网格参数化，这种情形下网格是具有边界的开网格，又分为固定边界映射和自由边界映射；(2) 闭网格参数化，这种情形下网格构成一个封闭曲面，通常人为指定一条边界将封闭 网格切开，然后转化为开网格进行参数化
>
> 来源：[VCL Lab - Peking University](https://vcl.pku.edu.cn/course/vci)

#### Spring System

对于平面参数化，引入一种基础的参数化方法：Spring System——通过 ***最小化能量*** 来找到最优的网格参数化或顶点位置

###### 1. 能量函数

定义弹簧系统的总能量函数E：
$$
E = \frac{1}{2} \sum_{i=1}^{n} \sum_{j \in N_i} \frac{1}{2} D_{ij} \| \mathbf{t_i} - \mathbf{t_j} \|^2
$$
其中，

- ti=(ui,vi) 是二维参数空间中点 i 的坐标，tj 同理。
- Dij 是顶点 i 和 j 之间的弹簧系数。
- Ni 是与顶点 i 相邻的顶点集。

###### 2. 最优化方程

为了最小化系统的总能量E，通过求解能量函数的 *梯度* 来优化顶点的位置，即：
$$
\frac{\partial E}{\partial t_i} = \sum_{j \in N_i} D_{ij} (\mathbf{t_i} - \mathbf{t_j}) = 0
$$
做简单的变换，有：
$$
t_i = \sum_{j \in N_i} \lambda_{ij} t_j
$$
并定义系数（coefficient)为：
$$
\lambda_{ij} = \frac{D_{ij}}{\sum_{k \in N_i} D_{ik}}
$$
常用的仿射组合系数有三种：平均系数、值坐标系数、 调和坐标系数

###### 3. 解线性方程

即转化为这个线性方程：
$$
t_i - \sum_{j\in N_i} \lambda_{ij} t_j = 0 
$$
而在网格的边界上，某些顶点的位置是已知的，固定边界映射过去，作为给定的边界条件来限制网格的参数化。

构建线性方程组 At=b，A 是表示网格中每个顶点和相邻顶点之间关系的矩阵，t 是需要计算的未知的顶点位置， b是已知的边界条件约束。再用如 [线性方程组-迭代法 2：Jacobi迭代和Gauss-Seidel迭代 - 知乎](https://zhuanlan.zhihu.com/p/389389672)) 迭代求解即可。





## **09 Geometry Processing (几何处理)**

### Discrete differential geometry (离散微分几何)

通过离散化技术将连续的几何概念和工具转化为离散的形式，以便计算机处理。
详细内容推导见 slides: [几何编辑](https://vcl.pku.edu.cn/course/vci/slides/09-geometry_processing.pdf)、tutorial: [CS 15-458/858: Discrete Differential Geometry – CARNEGIE MELLON UNIVERSITY | SPRING 2022 | TUE/THU 11:50-1:10 | GHC 4215](https://brickisland.net/DDGSpring2022/)

四个基本概念以及表示：

- #### Local Averaging Region (局部平均区域)

  定义三角网格顶点上的微分量

  ![img1](notes-vcl-part2-local area.png)

- #### Normal Vectors (法向量)

  定义三角形内部点和顶点的法向量

  <img src="E:\vcI\garbage\notes-vcl-part2-normal vector.png" alt="image" style="zoom:35%;" />

- #### Gradients (梯度)

  线性插值, 三角形内部点的梯度

  <img src="E:\vcI\garbage\notes-vcl-part2-gradient.png" alt="image" style="zoom:35%;" />

- #### Laplace-Beltrami Operator (拉普拉斯)

  顶点Laplace算子的一般形式，均匀、余切情况

  <img src="E:\vcI\garbage\notes-vcl-part2-Laplace.png" alt="image" style="zoom:35%;" />

<img src="E:\vcI\garbage\ef532cf2-629b-4169-90d1-11bc96fdde2d.png" alt="ef532cf2-629b-4169-90d1-11bc96fdde2d" style="zoom:35%;" />



- 关于 Gradient 和 Laplacian

梯度一致的时候，Laplace为0。

------

在上一讲，我们已经学习了几何处理中的网格细分和参数化；接下继续学习三个不同的几何处理方法。

### Mesh Smoothing (网格平滑)

#### 从 “扩散” 到平滑

类比图像处理，平滑就是 “去噪” 的过程，<u>改善网格的顶点分布</u>使得其更加平滑、自然。

借助 “扩散流” 的模型来理解，对于一个时空信号 f(x,t), 扩散流公式：
$$
\frac{\partial f(x,t)}{\partial t}=\lambda \Delta f(x,t)
$$
应用到网格平滑上，可以吧将带噪音的顶点坐标理解成杂乱分布的热量，遂扩散的进行，热量分布逐渐趋于平衡，于是顶点构成的表面趋于平滑所以，利用定义在表面的拉普拉斯算子，这里关注 *拉普拉斯平滑*：修改网格顶点的位置，使得每个顶点的位置趋向其邻域的平均位置。

#### 流程

①空间离散化，将函数值离散为网格顶点上的值，f(t) = (f(v1,t), f(v2,t),....f(vn,t)), 对每个顶点：
$$
\frac{\partial f(v_i,t)}{\partial t}=\lambda \Delta f(v_i,t)
$$
于是，有矩阵形式：
$$
\frac {\partial \mathbf{f}(t)}{\partial t} = \lambda \cdot L\mathbf{f}(t)
$$
②对于时间的离散化，采用均匀间隔的时间步长h，利用有限差分近似等式左侧的偏微分：
$$
\frac {\partial \mathbf{f}(t)}{\partial t} =\frac{ \mathbf{f}(t+h) - \mathbf{f}(t)}{h}
$$
显示欧拉，形式可以改写成：
$$
\mathbf{f}(t+h)  = \mathbf{f}(t)+h\frac {\partial \mathbf{f}(t)}{\partial t}  = \mathbf{f} + h\lambda \cdot L \mathbf{f}(t)
$$
即， 每经过一个h的时间步，用此式更新每个顶点上的函数值，直至达到设定的迭代停止条件。带入“网格平滑” 的背景，令函数 f 就表示顶点的坐标，则得到了**第（k+1）步的 Laplacian smoothing更新公式**：
$$
\mathbf{x}_i^{k+1} = \mathbf{x}_i^{k}+ h \lambda \cdot \Delta\mathbf{x}_i^{k}
$$

#### 拉普拉斯算子的两种形式

①均匀算子情况下，
$$
\Delta v_i = \frac{1}{|N(v_i)|} \sum_{j \in N(v_i)} (v_j - v_i)
$$
只考虑邻居的数量以及邻顶点的位置信息，忽略了面的曲率等信息，因而平滑 ***沿重心方向*** 移动，即相邻点的平均位置；

②余切算子情况下，
$$
\Delta v_i = \frac{1}{2A_i} \sum_{j \in N(i)} \left( \cot \theta_{ij} + \cot \theta_{ji} \right) (v_j - v_i)
$$
考虑了每个顶点相邻三角形的几何特性（邻接边的角度），够使顶点沿着网格的几何结构进行移动，***更能够保留几何特征***。

------



### Detail-Preserving Mesh Editing (网格编辑)

操纵和修改网格表面的几何形状，同时能够保留原始网格几何细节的操作。

回顾：泊松图像编辑，类比到三维的应用即 拉普拉斯网格编辑。

目标：指定一些顶点的目标位置后，重建一个既满足这些顶点约束、又尽可能保持局部拉普拉斯微分坐标的网格。数学公式：
$$
\arg \min_{v'_1, \dots, v'_n} \sum_{i=1}^n \left\| L(v'_i) - \Delta_i \right\|^2 + \sum_{j \in C} \left\| v'_j - u_j \right\|^2
$$
其中：

- vi' 是顶点 i 的新位置。
- L(vi') 是拉普拉斯算子作用于顶点 vi'，它表示每个顶点相对于其邻居的偏移量。
- delta i 是目标拉普拉斯误差, 通常是原始网格或参考网格的拉普拉斯算子值。
- C 是需要满足约束的顶点集合（例如边界顶点）。
- uj 是约束条件下顶点 j 的目标位置。

这是一个标准的最小二乘问题，转化为线性系统迭代求解即可。

应用：涂层迁移、表面移植等。

------



### Mesh Simplification (网格简化)

#### 网格简化

为什么需要简化？
--> 用plygon mesh来表示物体模型，但处理的物体模型通常由很多块面片组成，逐一处理渲染在游戏、可视化等场景十分影响效率；同时，并非每时每刻都需要用最高精度的模型，比如模型在远处时，本身细节就不用很精致，精度较低不影响渲染质量。所以，可以用更多的面片来表示近处物体，较少的面片来表示远处物体。

这就是网格简化：用更少的面片数来表达原有模型。

#### 流程

为了删除多余的顶点、边和面，可以通过移除顶点、边坍塌、半边坍塌来实现；但是在坍塌边的时候需要注意，防止坍缩某条边后对模型造成较大影响 (如干扰拓扑结构），于是引入一种误差度量，计算坍塌一条边的损失，比如可以采用最基本的 Quadric Error Metrics (二次误差度量)。

详细参考 Lab2 Task3

参考资料：[Surface simplification using quadric error metrics](https://dl.acm.org/doi/pdf/10.1145/258734.258849)





