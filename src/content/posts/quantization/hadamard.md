---
title: "Hadamard 矩阵与快速变换：从 ±1 到 d log d"
published: 2026-10-26T16:00:00+08:00
description: 从 Sylvester 构造、正交性与峰值下界讲到快速 Walsh–Hadamard 变换的蝶形实现、分块、随机符号与 fp16 数值行为，并落到 QuaRot/SpinQuant 里的在线旋转。
tags: [Hadamard, 快速变换, 量化, 线性代数, 推理优化]
category: Quantization
draft: false
---

一个 7B 模型的激活里，常常只有几个通道的值比其它通道大一个数量级。4-bit 量化按范围定步长，这几个通道就把 16 个码字的动态范围吃掉了一大半。一个自然的补救是**旋转**：乘一个正交矩阵，向量长度不变，尖峰被摊到所有坐标上。

最常用的旋转不是学出来的，而是一个只含 $\pm1$ 的矩阵——**Hadamard 矩阵**。它有两个少见的性质：第一，在所有正交矩阵里，它的最大元素绝对值最小（下一节给出证明）；第二，乘它不需要真正的乘法，快速变换只要 $d\log_2 d$ 次加减。

这篇笔记把这两件事讲清楚：定义与构造、峰值下界、快速变换的蝶形推导、维度不是 2 的幂怎么办、随机符号与 fp16 下会出什么问题，最后回到 QuaRot / SpinQuant 里的在线旋转。文中的构造、变换和量化例子都能用附录 A 的代码复现，完整实验设置见附录 B；矩阵默认用归一化版本，$H^{\top}H=I$。

> **五分钟版**
>
> - $H_d$ 是 $d\times d$ 正交矩阵，元素为 $\pm1/\sqrt d$；Sylvester 构造下它还对称，且 $H_d^2=I$。
> - $d=2^k$ 时有蝶形快速变换：$d\log_2 d$ 次加减，而不是 $d^2$ 次乘加。$d=128$ 是 $896$ 对 $16384$。
> - 任意正交矩阵的最大元素至少是 $1/\sqrt d$，Hadamard 恰好取到这个下界；对单位 $k$ 稀疏向量，峰值上界是 $\sqrt{k/d}$。
> - 但它不是“对任何输入都最优”：全同号向量会被它挤成一个尖峰，高斯输入则完全不变。
> - 维度不是 2 的幂时，用块对角（$11008=86\times128$）或填充；推理里通常与量化 kernel 融合。

## 0. 记号与约定

读后面公式前，先把几个反复出现的符号对齐：

| 符号 | 类型 | 含义 | 备注 |
|---|---|---|---|
| $d$ | 整数 | 矩阵阶数 / 向量维度 | 通常是 $2^k$；分块时可以是 $86\times128$ |
| $\mathcal H_d$ | 矩阵 | 未归一化 Hadamard，元素为 $\pm1$ | $\mathcal H^{\top}\mathcal H=dI$ |
| $H_d$ | 矩阵 | 归一化 Hadamard，$\mathcal H_d/\sqrt d$ | $H^{\top}H=I$，本文默认用这个 |
| $e_i$ | 向量 | 第 $i$ 个坐标单位向量 | 用来定义“尖峰” |
| $\mathbf 1$ | 向量 | 全 $1$ 向量 | $\mathbf 1/\sqrt d$ 是单位向量 |
| $k$ | 整数 | 非零分量个数（稀疏度） | 峰值界是 $\sqrt{k/d}$ |
| $D$ | 对角矩阵 | 对角元为随机 $\pm1$ | $DH$ 仍然正交 |
| $R_i$ | 矩阵 | SpinQuant 里的旋转 | $R_1,R_2$ 可学；$R_3,R_4$ 固定 Hadamard |
| $\operatorname{popcount}$ | 函数 | 二进制表示中 $1$ 的个数 | 逐元素公式用 |
| $\operatorname{bitreverse}_k$ | 置换 | 把 $k$ 位二进制反过来 | sequency 排序用 |
| $\odot$ | 运算 | 逐元素乘积 | 和 Hadamard 矩阵无关 |
| $\oplus$ | 运算 | 按位异或 | 用于 XOR 卷积 |
| $s$ | 标量 | 量化步长 | 对称/非对称定义不同 |
| $b$ | 整数 | 量化比特数 | 4-bit 即 $b=4$ |

## 1. 它是什么

### 1.1 先约定归一化

Hadamard 矩阵通常指元素为 $\pm1$ 的 $d\times d$ 矩阵，行与行正交。记未归一化的版本为 $\mathcal H_d$，它满足

$$
\mathcal H_d^{\top}\mathcal H_d=dI_d .
$$

把它除以 $\sqrt d$，得到归一化版本

$$
H_d=\frac{1}{\sqrt d}\mathcal H_d,\qquad H_d^{\top}H_d=I_d .
$$

论文和代码里这两个版本经常混用：说“元素是 $\pm1$”时指 $\mathcal H_d$，说“正交矩阵”时指 $H_d$。本文除特别说明外都用 $H_d$。这个 $1/\sqrt d$ 不是装饰：它决定 $H$ 是否保范数，也决定量化时能不能把它折进 scale。

最小的两个例子是

$$
H_2=\frac{1}{\sqrt2}\begin{pmatrix}1&1\\[2pt]1&-1\end{pmatrix},
\qquad
H_4=\frac12\begin{pmatrix}
1&1&1&1\\
1&-1&1&-1\\
1&1&-1&-1\\
1&-1&-1&1
\end{pmatrix}.
$$

$H_4$ 可以随手验：任取两列做内积都是 $0$，每列长度都是 $1$。比如第 1 列与第 2 列的内积是 $(1-1+1-1)/4=0$，第 1 列的长度平方是 $(1+1+1+1)/4=1$。

### 1.2 五个基本事实

对 Sylvester 构造的 $H_d$（$d=2^k\ge2$）：

| 性质 | 式子 | 说明 |
|---|---|---|
| 正交 | $H^{\top}H=HH^{\top}=I$ | 保长度、保内积 |
| 对称 | $H^{\top}=H$ | 所以 $H^{-1}=H$ |
| 对合 | $H^2=I$ | 转两次回到原处 |
| 特征值 | $\lambda=\pm1$，各 $d/2$ 个 | 迹为 $0$，$(I+H)/2$ 是秩 $d/2$ 的投影 |
| 行列式 | $\det H=\pm1$ | $H_2$ 为 $-1$，$d\ge4$ 时为 $+1$ |

前三条互相咬合：正交给出 $H^{\top}H=I$，对称把它变成 $H^2=I$。这也意味着 $H$ 的特征值只能是 $\pm1$，且因为 $\operatorname{tr}H=0$（$d\ge2$），正负特征值各占一半。于是

$$
P_+=\frac{I+H}{2},\qquad P_-=\frac{I-H}{2}
$$

是一对正交投影，秩都是 $d/2$。注意 $I+H=2P_+$ 是**奇异**的：它把 $P_-$ 方向整个压掉。这一点在优化里很关键——Cayley 参数化 $R=(I-A)(I+A)^{-1}$ 取不到特征值 $-1$，所以“从 $H$ 出发”不能写成 $H+\text{Cayley}$，只能写成 $H\cdot\text{Cayley}(A)$。

### 1.3 它不是“转 45°”

二维旋转只有一个角度，所以可以问“转了多少度”。到了 $d$ 维，正交矩阵的自由度是 $d(d-1)/2$：$d=128$ 时有 $8128$ 个。$H_{128}$ 只是这个巨大空间里的一个离散点，没有单一的“旋转角”。

更准确地说，对 Sylvester 构造的 $H$，它是一个**对合**：在一个 $d/2$ 维子空间上是恒等，在另一个 $d/2$ 维子空间上是 $-I$。$d\ge4$ 时后者的维数是偶数，可以拆成 $d/4$ 个平面，每个平面转 $180^\circ$。所以它不是“轻轻转一点”，而是一次离散的翻转。$H_2$ 更特殊：$\det H_2=-1$，它是反射，不是旋转。工程上叫它“旋转”只是沿用“正交变换”的习惯说法；量化只用到保范数，不关心行列式。

这也解释了为什么方法里没有“旋转强度”这个超参。若把连续旋转写成 $R(t)=\exp(tA)$，$t$ 很小时 $xR(t)\approx x+t\,xA$，尖峰几乎没动，量化步长也不会变小。$H$ 不在“小 $t$”附近，它是 $t=\pi$ 那一端的离散点。

> **一个容易混的记号**：Hadamard 乘积（逐元素乘积 $A\odot B$）和 Hadamard 矩阵没有关系，只是都冠了同一个名字。本文说的 $H$ 永远是那个 $\pm1$ 正交矩阵。

### 1.4 它最初为什么重要：最大行列式

Hadamard 1893 年的问题是：元素为 $\pm1$ 的 $d\times d$ 矩阵，行列式最大能到多少？Hadamard 不等式给出

$$
\lvert\det A\rvert\le\prod_{i=1}^{d}\lVert a_i\rVert_2,
$$

其中 $a_i$ 是 $A$ 的行。对 $\pm1$ 矩阵，每行长度都是 $\sqrt d$，所以

$$
\lvert\det A\rvert\le d^{d/2}.
$$

等号成立当且仅当各行两两正交——这正是 Hadamard 矩阵的定义。所以 Hadamard 矩阵就是“行列式达到上界的 $\pm1$ 矩阵”，这也是它名字的由来。Sylvester 在 1867 年就构造出了 $2^k$ 阶的那一族；Hadamard 在 1893 年证明了上界并提出了存在性问题。

## 2. 怎么造

### 2.1 Sylvester 递推

从 $1\times1$ 的 $H_1=(1)$ 出发，反复把边长翻倍：

$$
H_{2n}=\frac{1}{\sqrt2}
\begin{pmatrix}
H_n & H_n\\
H_n & -H_n
\end{pmatrix}.
$$

验证正交性只用一块矩阵乘法：

$$
H_{2n}^{\top}H_{2n}
=\frac12
\begin{pmatrix}
H_n^{\top} & H_n^{\top}\\
H_n^{\top} & -H_n^{\top}
\end{pmatrix}
\begin{pmatrix}
H_n & H_n\\
H_n & -H_n
\end{pmatrix}
=
\begin{pmatrix}
H_n^{\top}H_n & 0\\
0 & H_n^{\top}H_n
\end{pmatrix}
=I_{2n}.
$$

这里用了 $H_n^{\top}H_n=I_n$。翻 $k$ 次就得到 $H_{2^k}$，元素全是 $\pm1/\sqrt{2^k}$。代码只有四行：

```python
H = np.array([[1.0]])
while H.shape[0] < n:
    H = np.block([[H, H], [H, -H]]) / np.sqrt(2)
```

不需要训练，也不需要优化：这个矩阵在推理前构造一次，之后一直用。

### 2.2 三种等价写法

**Kronecker 积.** 上面的递推可以写成

$$
H_{2^k}=\underbrace{H_2\otimes H_2\otimes\cdots\otimes H_2}_{k\ \text{factors}}.
$$

因为 $(A\otimes B)^{\top}(A\otimes B)=(A^{\top}A)\otimes(B^{\top}B)$，而 $H_2^{\top}H_2=I_2$，所以这个张量积自然正交。

**逐元素公式.** 设 $i,j$ 从 $0$ 数起，$\operatorname{popcount}(i\&j)$ 表示 $i$ 与 $j$ 的二进制表示中同为 $1$ 的位数，则

$$
(H_d)_{ij}=\frac{(-1)^{\operatorname{popcount}(i\&j)}}{\sqrt d}.
$$

这个公式直接给出符号图案，也解释了为什么 $H$ 的行是“方波”：固定 $i$，随着 $j$ 从 $0$ 数到 $d-1$，$i$ 的每一个为 $1$ 的二进制位都会让符号周期性地翻转，叠加起来就是一条方波。

**$8\times8$ 的符号图案**（先不看 $1/\sqrt8$）：

```
+ + + + + + + +
+ - + - + - + -
+ + - - + + - -
+ - - + + - - +
+ + + + - - - -
+ - + - - + - +
+ + - - - - + +
+ - - + - + + -
```

第 1 行、第 1 列全是 $+$。这不是随便排的：Walsh 函数就是按 $\operatorname{popcount}(i\&j)$ 的奇偶来排符号的。把符号全取 $+$ 会得到秩 1 的“求平均”，信息全丢；符号随机取 $\pm1$ 又不是正交的。真正让行两两正交的，是这套按二进制位排出来的 Walsh 符号。

### 2.3 两种顺序：自然序与 sequency 序

同一个 $H_d$，行的排列方式可以不同：

- **自然序（Hadamard 序）**：第 $i$ 行就是 $\operatorname{popcount}(i\&j)$ 定义的顺序，也就是上面的递推顺序。
- **sequency 序（Walsh 序）**：按一行里符号变化的次数（sequency）从小到大排。变化越少，波形越“慢”。

<img src="/images/quantization/hadamard-pattern.png" data-light-src="/images/quantization/hadamard-pattern.png" data-dark-src="/images/quantization/hadamard-pattern-dark.png" alt="图 1：H32 的符号图案，左为自然序，右为 sequency 序。" loading="lazy">

*图 1：$H_{32}$ 的符号图案。左图是 Sylvester 自然序，右图按符号变化次数重排。浅色为 $+1$，深色为 $-1$。sequency 序下每一行就是一条越来越快的方波。*

自然序与 sequency 序之间的置换是“Gray 码 + 位反转”：若 $s$ 是 sequency，则自然序的行号是

$$
i=\operatorname{bitreverse}_k\big(s\oplus(s\gg1)\big),
\qquad k=\log_2 d.
$$

以 $d=8$ 为例，$s=0,1,\dots,7$ 对应的自然序行号是 $0,4,6,2,3,7,5,1$。对量化来说，行序不重要——它只是把坐标重新排了一下；只要 $q$ 和 $k$ 用同一个顺序，点积照样抵消。真正重要的是**符号图案**。

### 2.4 哪些阶数存在

- $d=1,2$：$H_1=(1)$，$H_2$ 如上。
- $d=2^k$：Sylvester 构造，任意 $k$ 都有。
- $d>2$：**必要条件**是 $4\mid d$。证明很短，见下。
- 充分性：Hadamard 猜想认为每个 $4$ 的倍数都存在；目前大量阶数已构造出来（Paley 构造覆盖 $q+1$，$q\equiv3\pmod4$ 为素数幂，例如 $d=12,20,24,\dots$），但猜想本身仍未解决。
- $d$ 为奇数且 $d>2$：不存在。取两行做内积，它是奇数个 $\pm1$ 之和，必为奇数，不可能等于 $0$。

**为什么 $4\mid d$.** 设 $H$ 的行两两正交。先把第一行通过列符号翻转变成全 $+1$——列翻转不改变正交性。其它任意一行与第一行正交，所以它的 $d$ 个 $\pm1$ 之和为 $0$：$d$ 为偶数，且该行有 $d/2$ 个 $+1$。

取第二、三行。设

- $a$：两行同为 $+1$ 的列数；
- $b$：两行同为 $-1$ 的列数；
- $c$：第二行 $+1$、第三行 $-1$ 的列数；
- $e$：第二行 $-1$、第三行 $+1$ 的列数。

第二行有 $d/2$ 个 $+1$，所以 $a+c=d/2$；第三行同理 $a+e=d/2$，故 $c=e$。两行内积为 $0$，所以 $a+b-c-e=0$，即 $a+b=2c$。总列数是

$$
d=a+b+c+e=2c+2c=4c,
$$

所以 $4\mid d$。这个论证对 $d\ge3$ 成立；$d=2$ 是单独的例外。

**存在不等于有快速变换.** 即便某个 $11008$ 阶的 Hadamard 矩阵存在，也未必有 Sylvester 那样规整的蝶形结构。工程上不会去赌这一点，直接用块对角或填充——见第 5 节。

### 2.5 它还在哪里出现

Hadamard 矩阵不只用于量化：

- **编码**：Hadamard 码用它的行当码字，行正交性直接变成纠错距离。
- **信号处理**：Walsh 谱把信号分解成方波，和 Fourier 谱是两套基；图像、通信里都有快速变换。
- **实验设计**：正交阵列用 Hadamard 矩阵来平衡多因素实验的主效应。
- **量子计算**：$H_2$ 就是 Hadamard 门，用来制造叠加态。
- **随机数值线性代数**：亚采样随机 Hadamard 变换（SRHT）用 $d\log d$ 的代价做快速随机投影。

这些应用和量化里的用法共享同一个核心：**用 $\pm1$ 的正交性，把“混合”做成廉价的加减**。

## 3. 它把向量怎么了

### 3.1 尖峰与全同号：一对镜像

先看两个极端。取 $e_0=(1,0,\dots,0)$：

$$
e_0H_d=\Big(\frac{1}{\sqrt d},\frac{1}{\sqrt d},\dots,\frac{1}{\sqrt d}\Big).
$$

一个尖峰被彻底摊平，每个坐标都拿到 $1/\sqrt d$。反过来，取全同号向量 $\mathbf 1/\sqrt d=(1,\dots,1)/\sqrt d$：

$$
\frac{\mathbf 1^{\top}}{\sqrt d}H_d=(1,0,\dots,0).
$$

全同号向量被挤成一个尖峰。两件事其实是同一件事：$H^2=I$，$H$ 把 $e_0$ 和 $\mathbf 1/\sqrt d$ 这两个方向**互换**了。它没有“把一切都变均匀”的魔法，只是在坐标基和“全同号方向”之间搬能量。输入在坐标基下越稀疏，它摊得越平；输入越接近全同号，它挤得越尖。

### 3.2 手算一个：(7,0,0,1)

取 $x=(7,0,0,1)$——一个坐标很大、其余接近 $0$，正是离群值的样子。

**直接乘矩阵：**

$$
xH_4=\frac12(7+0+0+1,\ 7-0+0-1,\ 7+0-0-1,\ 7-0-0+1)
=\frac12(8,6,6,8)=(4,3,3,4).
$$

**快速变换：**

第 1 轮（相邻两两一加一减）：

$$
(7,0,0,1)\ \longrightarrow\ (7+0,\ 7-0,\ 0+1,\ 0-1)=(7,7,1,-1).
$$

第 2 轮（隔一个再做同样的事）：

$$
(7,7,1,-1)\ \longrightarrow\ (7+1,\ 7-1,\ 7-1,\ 7+1)=(8,6,6,8).
$$

除以 $\sqrt4=2$，得 $(4,3,3,4)$，与直接乘矩阵一致。

**看范围：** 旋转前 $\max-\min=7-0=7$，旋转后 $4-3=1$，缩小 7 倍。

**接着量化**（4-bit 非对称 min-max）：旋转前步长 $s=7/15\approx0.467$，旋转后 $s=1/15\approx0.0667$，细了 7 倍。旋转后的四个数落在 $3$、$4$ 两端，编码成 $0$ 或 $15$，反量化回来正好是 $3.0$ 与 $4.0$，没有损失；不旋转时，$0$ 与 $1$ 只能分到相隔两格的粗格子上，反量化后 $1$ 变成 $0.933$；旋转后四个数正好落在端点，误差为 $0$。

这个例子说明旋转的收益不是“把数变小”，而是**把范围变小**。长度 $\lVert x\rVert_2=\sqrt{50}\approx7.07$ 完全没变，变的是坐标分布。

把同样的场景放大到 $d=128$，区别更直观：

<img src="/images/quantization/hadamard-flatten.png" data-light-src="/images/quantization/hadamard-flatten.png" data-dark-src="/images/quantization/hadamard-flatten-dark.png" alt="图 2：一个 d=128 的 LLM 式 token 在 Hadamard 旋转前后的值，以及 4-bit 量化电平。" loading="lazy">

*图 2：上图是带 3 个离群通道的 token，下图是 $H_{128}$ 的输出；虚线是 4-bit 非对称量化的 16 个电平。范围从 $1.002$ 缩到 $0.488$，步长从 $0.0668$ 缩到 $0.0325$。旋转没有让任何数变小，只是把尖峰摊开，让电平落在更细的网格上。*

### 3.3 峰值下界：为什么 1/√d 是最优的

设 $Q$ 是任意 $d\times d$ 正交矩阵，第 $i$ 行记为 $q_i$。因为 $q_i$ 是单位向量，

$$
1=\lVert q_i\rVert_2^2=\sum_{j=1}^{d}q_{ij}^2\le d\cdot\max_j q_{ij}^2.
$$

所以

$$
\max_j\lvert q_{ij}\rvert\ge\frac{1}{\sqrt d}.
$$

这个下界对每一行都成立。要让整个矩阵的最大元素恰好等于 $1/\sqrt d$，每一行都必须取等号；而一行取等号的唯一方式是这 $d$ 个平方数全都等于 $1/d$，也就是每个元素绝对值都恰好是 $1/\sqrt d$。再加上“本身正交”这个前提，这就是归一化 Hadamard 矩阵的定义。

换句话说：**在所有正交矩阵里，Hadamard 矩阵的最大元素最小，且恰好是 $1/\sqrt d$。** 这不是形容词，是等号成立条件的直接翻译。

由此得到对任意向量 $x$ 的界：

$$
\lVert xH_d\rVert_\infty
=\max_j\Big\lvert\sum_i x_iH_{ij}\Big\rvert
\le\frac{1}{\sqrt d}\sum_i\lvert x_i\rvert
=\frac{\lVert x\rVert_1}{\sqrt d}.
$$

如果 $x$ 只有 $k$ 个非零分量，由 Cauchy–Schwarz，$\lVert x\rVert_1\le\sqrt k\lVert x\rVert_2$，于是

$$
\lVert xH_d\rVert_\infty\le\sqrt{\frac{k}{d}}\,\lVert x\rVert_2.
$$

这个界是紧的：取 $x=(1,\dots,1,0,\dots,0)/\sqrt k$（前 $k$ 个坐标为正），第一行全 $+1$，所以第一个输出恰好是 $\sqrt{k/d}$。$d=128$ 时，$k=1$ 得 $0.0884$，$k=3$ 得 $0.1531$，$k=10$ 得 $0.2795$，$k=16$ 得 $0.3536$。

**“对一个尖峰最优”和“对所有尖峰最优”是两件事.** 想让 $e_0$ 被摊平，只要第 $0$ 行全平就行——任意正交矩阵都能安排这样一行。但要让 $e_0,e_1,\dots,e_{d-1}$ 这 $d$ 个坐标尖峰**同时**被摊平，就必须每一行都全平。Hadamard 做到了这一点：

$$
\max_i\lVert e_iH\rVert_\infty=\max_{i,j}\lvert H_{ij}\rvert=\frac{1}{\sqrt d},
$$

而任意正交 $Q$ 都有 $\max_i\lVert e_iQ\rVert_\infty=\max_{i,j}\lvert Q_{ij}\rvert\ge1/\sqrt d$。所以在这个“所有坐标尖峰”的准则下，Hadamard 是最优的。

注意 $1/\sqrt d$ 首先是**矩阵元素**的下界；对固定的单位向量 $x$，$\lVert xH\rVert_\infty$ 也至少是 $1/\sqrt d$（任何单位向量都如此），但可以远大于它——全同号给 $1$。Hadamard 保证的是：$d$ 个坐标尖峰 $e_i$ 同时取到这个下界；它不保证任意 $x$ 都取到。

但这**不**意味着它对每个具体的 $x$ 都最优。下一节给出反例。

### 3.4 它不是对每个向量都最优

对固定的 $x$，峰值 $\lVert xH\rVert_\infty$ 取决于 $x$ 长什么样。$d=128$ 上取几类单位向量，比较不旋转、Hadamard、随机正交（200 次取中位数）。表中“不转”=不旋转，“随机”=随机正交：

| 向量类型 | 不转 | Hadamard | 随机 | 说明 |
|---|---:|---:|---:|---|
| 单尖峰 $e_0$ | 1.000 | **0.088** | 0.243 | 正好是下界 |
| 3 尖峰 ± | 0.577 | **0.153** | 0.247 | 接近 $\sqrt{3/128}$ |
| 10 尖峰 ± | 0.316 | **0.224** | 0.251 | 有符号相消 |
| 16 尖峰 ± | 0.250 | 0.221 | 0.246 | 交叉点附近 |
| 高斯随机 | 0.248 | 0.246 | 0.245 | 三者一样 |
| 全同号 | **0.088** | 1.000 | 0.247 | Hadamard 最差 |
| LLM 式 | 0.552 | **0.229** | 0.248 | 3 离群 + 高斯 |

这张表把“Hadamard 是万能的”这个印象打掉了：

- **稀疏尖峰**：Hadamard 把每个尖峰摊到全维，峰值压到下界附近；随机正交的峰值约 $0.25$，比它大。
- **高斯**：$x\sim\mathcal N(0,I/d)$ 时，$xH$ 与 $x$ 同分布（因为 $H$ 正交），三者的峰值和量化误差都应该完全一样；表里的微小差别只是采样噪声。
- **全同号**：Hadamard 把 $\mathbf 1/\sqrt d$ 变成 $e_0$，峰值从 $0.088$ 涨到 $1$，是**最差**的选择。完全相等的全 $1$ 向量是个例外：它被变成 $e_0$，只有两个不同值，min-max 量化误差为 $0$；但只要加一点扰动，尖峰就会把范围撑大，剩下的小分量被压到同一个格子里——第 8 节的“稠密同号”实验会看到这一点。
- **LLM 式**：真实的激活既不是纯稀疏，也不是全同号，而是“少数通道特别大 + 其余接近零”。这正是 Hadamard 擅长的形状，所以它比随机正交略好（$0.229$ 对 $0.248$），比不旋转好得多（$0.552$）。

这也解释了为什么不能把“Hadamard 最优”理解成“对任何输入都最优”。它最优的是**峰值上界**和**坐标尖峰**；对具体的分布，它只是一个与数据无关的强先验。

### 3.5 能量守恒，峰值不守恒

正交变换保范数：

$$
\lVert xH\rVert_2^2=xHH^{\top}x^{\top}=\lVert x\rVert_2^2.
$$

所以“旋转让数值变小”是错的——总能量一点没少。变的只是**坐标分布**：原来集中在一个坐标上的能量，被分到 $d$ 个坐标上。量化误差由坐标的范围决定，所以这个“重新分配”才有意义。

反过来，$\lVert xH\rVert_\infty\le\lVert x\rVert_2$ 对任何正交矩阵都成立，这是“单个坐标不能超过总长度”。Hadamard 的特别之处是它把这个上界压到了 $1/\sqrt d$ 的**矩阵元素**层面，从而对稀疏 $x$ 给出与分布无关的紧界；它不是让所有 $x$ 的峰值都最小。

### 3.6 一个闭环：稀疏 → 稠密 → 稀疏

下面的动画把 $X\mapsto XH$ 走两遍。$X$ 是一个 $32\times32$ 的激活矩阵：行是 token，列是通道；少数通道和几个孤立位置带有离群值。右乘 $H$ 后，能量摊到所有通道；再右乘**同一个** $H$，因为 $H^2=I$，严格回到 $X$。

<video src="/videos/hadamard-loop.mp4" poster="/images/quantization/hadamard-loop-poster.png" autoplay loop muted playsinline style="width:100%;border-radius:12px;display:block;background:#0f172a">
  <img src="/videos/hadamard-loop.gif" alt="Hadamard loop: sparse → dense → sparse">
</video>

*动画 1：$X\to XH\to XH^2=X$。热图每帧按自身最大值归一化，颜色只表示相对结构；变换本身保 Frobenius 范数 $\lVert XH\rVert_F=\lVert X\rVert_F$。注意第二次必须用同一个 $H$；若换成随机符号变体 $H'=DH$，则 $H'^2\ne I$，第二次要用 $H'^{-1}=HD$ 才能回到原矩阵。*

这里演示的是**沿通道维右乘** $H$，也就是量化里对激活做的事，不是把整张矩阵做二维变换。二维版本 $H_T X H_d$ 同样满足平方为恒等（对 token 维和通道维分别用相应阶数的 $H$），但混合的是两个方向。

## 4. 快速变换：d log d 是怎么来的

### 4.1 从递推到蝶形

把 $x$ 切成前后两半 $x=(u,v)$，每半 $d/2$ 维。利用 $H_{2n}$ 的分块结构：

$$
(u,v)\,H_{2n}
=\frac{1}{\sqrt2}\Big((u+v)H_n,\ (u-v)H_n\Big).
$$

也就是说：**先做 $2n$ 次加减（$n$ 个加法、$n$ 个减法），再分别对两半做规模为 $n$ 的变换**。递归下去：

$$
T(2n)=2T(n)+2n,\qquad T(1)=0,
$$

解得 $T(d)=d\log_2 d$。每一层做 $d$ 次加减（$d/2$ 个蝴蝶，每个一加一减），共 $\log_2 d$ 层。

<img src="/images/quantization/hadamard-butterfly.png" data-light-src="/images/quantization/hadamard-butterfly.png" data-dark-src="/images/quantization/hadamard-butterfly-dark.png" alt="图 3：d=8 的快速 Walsh–Hadamard 变换蝶形图。" loading="lazy">

*图 3：$d=8$ 的 FWHT 蝶形图。第 $s$ 层的步长是 $h=2^{s-1}$，每个蝴蝶把 $(u,v)$ 变成 $(u+v,u-v)$。三层共 $4\times3=12$ 个蝴蝶、$24$ 次加减，正好是 $d\log_2 d$。*

图里每一层的连接看起来是“跨很多行”的，但每个蝴蝶只连两个相距 $h$ 的位置：第 1 层 $h=1$（相邻），第 2 层 $h=2$，第 3 层 $h=4$。这就是原位算法的全部结构。

### 4.2 算法

```python
def fwht(x):
    """x: (..., n), n 为 2 的幂；返回 x @ H_n（归一化）。"""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[-1]
    assert n >= 1 and (n & (n - 1)) == 0, "n must be a power of two"
    lead = x.shape[:-1]
    x = x.reshape(-1, n) / np.sqrt(n)   # 预缩放，见 7.2
    h = 1
    while h < n:
        x = x.reshape(-1, n // (2 * h), 2, h)
        a, b = x[:, :, 0, :], x[:, :, 1, :]
        x = np.concatenate([a + b, a - b], axis=2).reshape(-1, n)
        h *= 2
    return x.reshape(*lead, n)
```

几个要点：

- **输出是自然序**，与 $xH_n$ 一致，不需要额外的位反转。若需要 sequency 序，再乘一个置换。
- **逆变换就是它自己**：归一化后 $H^{-1}=H$，所以同一个函数再跑一遍就回到 $x$。如果你用的是未归一化的 $\mathcal H$，逆变换要除以 $d$。
- **预缩放**：先把输入除以 $\sqrt n$，后面全用未归一化的 $\pm1$ 蝴蝶。这样输出直接是 $xH_n$，而且中间值的上界是 $\lVert x\rVert_2$，不会像“最后再除”那样把中间值放大到 $\lVert x\rVert_1$。细节见 7.2。
- **原位**：真实 kernel 里可以把 $a,b$ 写回同一块寄存器/共享内存，不需要每层分配新数组；上面的代码为了清楚用了 `concatenate`。

### 4.3 手算 d=4

用算法跑 $x=(1,2,3,4)$：

| 步骤 | 结果 |
|---|---|
| 第 1 层（$h=1$） | $(1+2,\ 1-2,\ 3+4,\ 3-4)=(3,-1,7,-1)$ |
| 第 2 层（$h=2$） | $(3+7,\ -1+(-1),\ 3-7,\ -1-(-1))=(10,-2,-4,0)$ |
| 除以 $\sqrt4=2$ | $(5,-1,-2,0)$ |

直接乘矩阵：

$$
(1,2,3,4)H_4=\frac12(10,-2,-4,0)=(5,-1,-2,0).
$$

一致。对 $d=8$，图 3 就是同一件事多一层。

### 4.4 代价对比

| $d$ | 直接乘矩阵（乘加） | FWHT（加减） | 倍数 |
|---:|---:|---:|---:|
| $128$ | $16{,}384$ | $896$ | $18.3\times$ |
| $256$ | $65{,}536$ | $2{,}048$ | $32\times$ |
| $1024$ | $1{,}048{,}576$ | $10{,}240$ | $102.4\times$ |
| $4096$ | $16{,}777{,}216$ | $49{,}152$ | $341\times$ |
| $11008$（$86\times128$ 分块） | $121{,}176{,}064$ | $77{,}056$ | $1572\times$ |

这些是**操作数**，不是实测加速比。$d=128$ 时 GPU 上的 $128\times128$ 矩阵乘本身很快，FWHT 的收益主要来自“不读矩阵、只做加减、可与量化融合”；$d=11008$ 时差距才是决定性的。分块本身已经省了一次：$86$ 个 $128\times128$ 的块乘是 $86\times128^2=1{,}409{,}024$ 次乘加，FWHT 又比它少 $18.3$ 倍。

### 4.5 它到底是什么：(Z/2)^k 上的 Fourier 变换

Hadamard 变换不是“随便找了一个正交矩阵”。设 $G=({\mathbb Z}/2)^k$，把向量 $x$ 看成 $G$ 上的函数 $f$。$G$ 的特征标是

$$
\chi_s(t)=(-1)^{s\cdot t},\qquad s,t\in\{0,1\}^k,
$$

其中 $s\cdot t$ 是逐位与再求和（模 2）。$H_{2^k}$ 的第 $s$ 行、第 $t$ 列正好是 $\chi_s(t)/\sqrt d$；所以 $H$ 就是 $G$ 的特征标表，FWHT 就是 $G$ 上的 Fourier 变换。

这带来两个推论：

1. **正交性是群特征标的正交性**，不是偶然。
2. **它把 XOR 卷积变成逐点乘积**：对 $f,g:G\to\mathbb R$，定义 $(f*g)(z)=\sum_t f(t)g(z\oplus t)$，则

$$
\widehat{f*g}(s)=\hat f(s)\,\hat g(s).
$$

这和 DFT 把循环卷积变成逐点乘积是同一件事，只是群从 ${\mathbb Z}/d$ 换成了 $({\mathbb Z}/2)^k$。FWHT 的快算法就是 Cooley–Tukey 在这组群上的特例。

**和 DFT 的区别：**

| | Hadamard $H_d$ | DFT $F_d$ |
|---|---|---|
| 元素 | $\pm1/\sqrt d$ | $e^{-2\pi ijk/d}/\sqrt d$ |
| 运算 | 只有加减 | 复数乘加 |
| 逆 | $H$ 自己 | $F^{H}$，且 $F^4=I$ |
| 对角化什么 | XOR 卷积 | 循环卷积 |
| 群 | $({\mathbb Z}/2)^k$ | ${\mathbb Z}/d$ |
| 输出顺序 | 自然序 / sequency 序 | 自然序 / 位反转序 |
| 实数输入 | 实数输出 | 复数输出（共轭对称） |
| 快速算法 | FWHT，$d\log d$ | FFT，$d\log d$ |

两者都是 $d\log d$，但 Hadamard 只做加减、没有旋转因子（twiddle factor），所以在硬件上更便宜。它不是 DFT 的近似，也不是 DFT 的特例；它们是不同群上的同一类变换。

## 5. 维度不是 2 的幂怎么办

$d_h=128$ 没问题；但 $d_{ff}=11008=86\times128$ 不是 2 的幂，造不出这个尺寸的 Sylvester 矩阵。实际做法是**块对角**：把 11008 切成 86 个 128 的块，每块各自乘一个 $H_{128}$：

$$
H_{\text{blk}}=\operatorname{blkdiag}\big(\underbrace{H_{128},\dots,H_{128}}_{86\ \text{blocks}}\big)
=I_{86}\otimes H_{128}.
$$

分块正交就意味着整体正交：

$$
H_{\text{blk}}^{\top}H_{\text{blk}}
=(I\otimes H)^{\top}(I\otimes H)
=I\otimes H^{\top}H=I.
$$

所以输出不变、量化收益照旧；而一次变换仍然只是 86 个小快速变换，代价是 $86\times896=77{,}056$ 次加减，而不是 $11008^2\approx1.2$ 亿次乘加。

**为什么不填充到 16384.** 填充到下一个 2 的幂也很简单：补 5376 个零，做 $16384$ 点的 FWHT，再截断。代价是 $16384\times14=229{,}376$ 次加减，约是分块方案的 $3$ 倍；显存和带宽也多 $49\%$（$16384/11008$）。除非你的 kernel 只支持 2 的幂，否则分块更划算。

**块大小怎么选.** $128$ 是 head 维度，也是常见的 warp/寄存器块大小；$256$ 也可以，块越大，$d\log d$ 的 $\log$ 项越大，但块内并行度更高。关键是让块大小和硬件的数据布局对齐——一块最好是一个 warp 能放进寄存器、或一次 shared-memory 读写能覆盖的范围。

**其他阶数.** 对一般的 $4k$ 阶 Hadamard 矩阵，存在性由 Paley 构造等覆盖很多阶，但不一定有 Sylvester 那样规整的蝶形分解；工程上不会为了一个非 2 的幂的维度去实现一套复杂的快速算法。分块/填充是默认答案。

## 6. 随机化：DH 还算数吗

算。取对角阵 $D=\operatorname{diag}(d_1,\dots,d_d)$，每个 $d_i$ 独立取 $\pm1$，令 $H'=DH$。$D$ 自己就是正交矩阵，所以 $H'$ 仍然正交：

$$
(H')^{\top}H'=H^{\top}D^{\top}DH=H^{\top}H=I.
$$

它等价于先把输入的第 $i$ 个坐标乘上 $d_i$，再做 $H$；也可以看成给 $H$ 的每一行换符号。它仍然保留“每个元素绝对值都是 $1/\sqrt d$”的性质，所以峰值下界照旧；但它不再是 Sylvester 那个固定的符号图案。

几个容易忽略的细节：

- **$DH$ 一般不对称，也不对合**。$(DH)^{\top}=HD$，只有当 $D$ 与 $H$ 交换时才等于 $DH$；而 $H$ 的元素全非零，所以只有标量对角阵与它交换；非平凡的 $D$ 都会破坏对称性。$(DH)^2=DHDH\ne I$。逆是 $(DH)^{-1}=HD$，不是它自己。
- **符号变体有多少个**：$D$ 有 $2^d$ 种取法；$D$ 与 $-D$ 只差一个全局负号，作为变换输出整体取负，对量化范围没有影响，所以本质上有 $2^{d-1}$ 种。论文脚注写的是 $2n$，按定义应为 $2^n$（归一化前的元素也应写作 $\pm1$，而不是 $\pm\sqrt n$）——这是两处笔误，不影响结论。
- **为什么要随机化**：固定的 $H$ 对不同层、不同 head 用同一套符号，可能让某些坐标系统性地占优；随机符号让每个坐标都有机会被“翻正/翻负”，减少这种系统性偏差。QuIP#、QuaRot 一类方法常用随机符号变体。
- **随机化不改变幅度**：$|H'_{ij}|=1/\sqrt d$，所以 $1/\sqrt d$ 的下界和 $\sqrt{k/d}$ 的上界都还在。

**随机 Hadamard 的方差.** SpinQuant 报告：100 次随机旋转中，最好的和最差的下游精度差 13 分；换成随机 Hadamard 后，方差仍有 6 分。也就是说，$H$ 把方差压小了一半，但没有压到零。本文的峰值实验也看到同样的趋势：对 LLM 式向量，$DH$ 的峰值均值 $0.232$、标准差 $0.024$；随机正交是 $0.250$、$0.030$。$H$ 更好、更稳，但不是“随便一个 $H$ 都一样好”。

**随机 $\pm1$ 矩阵不是 Hadamard 矩阵.** 如果只把每个元素独立取 $\pm1/\sqrt d$，行与行不再正交：$d=128$ 时 $\max\lvert S^{\top}S-I\rvert\approx0.34$，离 $0$ 很远。真正让行正交的是 Walsh 符号的编排，不是“幅度一样”。这一点在实现里很重要：不能用一个随机符号矩阵冒充旋转，否则全精度输出就变了。

## 7. 数值与实现

### 7.1 条件数

$H$ 是正交矩阵，$\lVert H\rVert_2=\lVert H^{-1}\rVert_2=1$，条件数是 $1$。在精确算术下，变换不放大误差；在浮点下，误差主要来自每一步加法的舍入，而不是矩阵本身病态。

### 7.2 fp16 下的中间值

未归一化的蝴蝶每层做 $u\pm v$，中间值的绝对值最坏可以到 $\lVert x\rVert_1$。对 $d=4096$、$x\sim\mathcal N(0,I)$，$\lVert x\rVert_1\approx3260$，离 fp16 的 $65504$ 还很远；但如果激活整体乘了 $1000$，$\lVert x\rVert_1\approx3.3\times10^6$，就会溢出。

两种处理方式：

| 方式 | 中间值上界 | fp16 误差（$d=4096$） | 备注 |
|---|---|---:|---|
| 不缩放，最后除 $\sqrt d$ | $\lVert x\rVert_1$ | $7.9\times10^{-4}$ | 幅值大时会溢出 |
| 每层乘 $1/\sqrt2$ | $\lVert x\rVert_2$ | $1.7\times10^{-3}$ | 多 $\log d$ 次舍入，反而更差 |
| 输入预乘 $1/\sqrt d$，最后不除 | $\lVert x\rVert_2$ | $7.7\times10^{-4}$ | 推荐：一次缩放，不溢出 |

表中的误差是相对 L2 误差，输入为随机高斯，fp16 计算、fp64 参考。结论有点反直觉：**每层缩放虽然安全，但会多出 $\log d$ 次舍入，精度约差 2 倍**；输入预缩放只做一次乘法，既避免溢出又不增加每层舍入。预缩放后的向量 $x'=x/\sqrt d$ 满足 $\lVert x'\rVert_2=\lVert x\rVert_2/\sqrt d$，而未归一化变换的中间值不超过 $\lVert x'\rVert_1\le\sqrt d\lVert x'\rVert_2=\lVert x\rVert_2$。

fp32 下相对误差约 $10^{-7}$，基本可以忽略。bf16 的指数范围大、不会溢出，但有效尾数只有 8 位（fp16 是 11 位），误差大约是 fp16 的 $8$ 倍；对精度敏感的路径建议 fp32 累加，或者把缩放折进量化 scale。

### 7.3 GPU 上怎么算

$d=128$ 的一行是 $128$ 个 bf16，$256$ 字节；一个 warp 的 32 个 lane 每个拿 4 个元素，正好放进寄存器。7 层蝴蝶只需要 warp shuffle 和加减，**不需要读 $H$ 矩阵**——符号是算法的一部分，不是数据。这是 FWHT 相对“读一个 $128\times128$ 矩阵再乘”的最大优势：省掉了 $32$ KB 的常量读取和矩阵乘的调度。

$d=11008$ 按 $128$ 分块后，每一块也是同样的 warp 级操作；块与块之间独立，天然并行。

一个典型的融合 kernel 是：

1. 读入 fp16/bf16 激活；
2. 预乘 $1/\sqrt{128}$（或把缩放折进 scale）；
3. 7 层原地蝴蝶；
4. 求 min/max，算量化 scale；
5. 量化成 int4，写回量化值和 scale。

这样“旋转 + 量化”只读一次、写一次，不额外走一遍显存。SpinQuant 报告的在线 Hadamard 开销约 $8\%$ 网络延迟；从量级上看，成本主要在这套“读、算、量化、写”的访存，而不是那 $896$ 次加减。

### 7.4 代价量级

以 LLaMA-2 7B 为例：

| 在线旋转 | 维度 | 直接乘矩阵 | 快速变换 |
|---|---:|---:|---:|
| 每个 head 的 $q$ 或 $k$（单个向量） | $128$ | $128^2\approx1.6$ 万次乘加 | $128\times7=896$ 次加减 |
| FFN 中间激活 | $11008$（$86$ 块） | $11008^2\approx1.2$ 亿次 | $86\times896\approx7.7$ 万次 |

单个向量的 $896$ 次加减，相对一次 $4096\times128$ 的投影（$52$ 万次乘加）只占 $0.17\%$；FFN 的 $7.7$ 万次相对 $4096\times11008\approx4500$ 万次乘加，也约 $0.17\%$。所以纯算术上它很便宜；$8\%$ 的端到端开销主要来自“访存 + 量化 + kernel launch”这些工程成本。

## 8. 在量化里：它到底帮了什么

### 8.1 量化器

对张量 $X$，$b$-bit 非对称 min-max 量化是

$$
s=\frac{\max X-\min X}{2^b-1},\qquad
q=\operatorname{clamp}\Big(\operatorname{round}\Big(\frac{X-\min X}{s}\Big),0,2^b-1\Big),\qquad
\hat X=sq+\min X.
$$

对称 absmax 则是

$$
s=\frac{\max\lvert X\rvert}{2^{b-1}-1},\qquad
\hat X=s\cdot\operatorname{clamp}\Big(\operatorname{round}\Big(\frac{X}{s}\Big),-(2^{b-1}-1),2^{b-1}-1\Big).
$$

两种约定下，步长 $s$ 都正比于一个“范围”量：非对称看 $\max-\min$，对称看 $\max\lvert X\rvert$。旋转把尖峰摊开，就是在压这个量。在不裁剪、误差近似均匀的假设下，取整误差最多是 $s/2$；步长减半，误差也大致减半。这就是“旋转 → 范围变小 → 量化变准”的全部逻辑。

**但“峰值下界”只对对称量化是硬结论.** 对称量化的步长由 $\max\lvert z\rvert$ 决定，而任意单位向量都满足 $\max_i\lvert z_i\rvert\ge1/\sqrt d$，Hadamard 在单尖峰上取到这个下界，所以“最优”可以严格证明。非对称量化的步长由范围 $\max-\min$ 决定，而范围**没有**类似的普适下界——一个常数向量的范围是 $0$，量化误差也是 $0$。所以非对称下只能说“旋转通常把范围压窄”，不能把 $1/\sqrt d$ 当成量化误差的下界。好在两个量高度相关，实测中两种约定的改善幅度相近。

### 8.2 为什么输出不变

对一个线性层 $y=xW^{\top}$，在 $x$ 和 $W$ 之间插入 $H$：

$$
y=(xH)(WH)^{\top}=xHH^{\top}W^{\top}=xW^{\top}.
$$

所以只要把 $H$ 同时乘进权重，输出完全不变。注意力里更直接：$q$ 和 $k$ 用同一个 $H$，

$$
(qH)(kH)^{\top}=qHH^{\top}k^{\top}=qk^{\top}.
$$

点积不变，softmax 前的分数就不变。V 和输出投影也可以成对吸收：$vH$ 之后乘 $W_oH$，净效果还是 $vW_o^{\top}$。

**不能吸收的地方**是 RoPE 和逐元素乘法：

- $q,k$ 上的 $H$ 在 RoPE 之后。RoPE 按 token 位置转角度，$H$ 混合坐标，$H\Theta_t\ne\Theta_tH$，所以 $H$ 不能提前并进 $W_q,W_k$。
- FFN 里的 $z=\operatorname{SiLU}(uW_{\text{gate}}^{\top})\odot(uW_{\text{up}}^{\top})$，逐元素乘法只认同一坐标，$H$ 不能穿过 $\odot$。

这两处只能在线算，这正是 SpinQuant 里 $R_3,R_4$ 保留为固定 Hadamard 的原因：它们不可吸收，但足够便宜。

### 8.3 实测：旋转收益与“并非万能”

合成一组 LLM 式激活：$d=128$，4000 个 token；每个 token 先取 $\mathcal N(0,I)$，再在 3 个固定通道上加 $U(5,12)\cdot\pm1$ 的离群值，最后归一化到单位长度。按 token 做 4-bit 量化，比较不旋转、Hadamard、随机正交：

<img src="/images/quantization/hadamard-ecdf.png" data-light-src="/images/quantization/hadamard-ecdf.png" data-dark-src="/images/quantization/hadamard-ecdf-dark.png" alt="图 4：4-bit 非对称量化下，逐 token 相对 L2 误差的 ECDF。" loading="lazy">

*图 4：4-bit 非对称 per-token 量化的误差分布。不旋转的中位数是 $20.0\%$，随机正交 $9.9\%$，Hadamard $9.3\%$。旋转把误差减半；Hadamard 比随机正交略好，但差距不大。*

| 变换 | 峰值 | 范围 | 峰度 | 非对称 | 对称 |
|---|---:|---:|---:|---:|---:|
| 不转 | 0.554 | 0.898 | 21.1 | 19.4% | 25.7% |
| Hadamard | **0.232** | **0.432** | **2.67** | **9.3%** | **10.8%** |
| 随机 | 0.249 | 0.459 | 2.94 | 9.9% | 11.6% |

表中“不转”=不旋转，“随机”=随机正交（40 次平均）；峰值、范围、峰度都是均值，误差是逐 token 的相对 L2 均值，峰度是 Pearson 峰度（高斯为 3），非对称/对称分别指两种 4-bit 量化约定。旋转把峰度从 $21$ 压到 $2.7$，把误差从 $19.4\%$ 压到 $9.3\%$——约 $2.1$ 倍。Hadamard 相对随机正交只再好 $6\%$ 左右，所以**主要收益来自“旋转”这件事本身，Hadamard 是其中便宜又稳的那个选择**。

但换成不同的输入分布，结论会变（“稠密同号”指 $\lvert\mathcal N(0,1)\rvert$ 归一化后的向量）：

| 输入 | 不转 | Hadamard | 随机 | 说明 |
|---|---:|---:|---:|---|
| LLM 式 | 19.4% | **9.3%** | 10.0% | 3 离群 + 高斯 |
| 高斯随机 | 9.9% | 9.9% | 9.9% | 三者完全一样 |
| 稠密同号 | **5.4%** | 20.3% | 9.7% | Hadamard 最差 |

高斯行三者完全相同，不是巧合：$x\sim\mathcal N(0,I/d)$ 时 $xH$ 与 $x$ 同分布，任何正交变换的期望量化误差都一样。旋转只能改变**坐标基**，不能改变各向同性分布。稠密同号行则是 Hadamard 的反例：全同号方向被它映射成 $e_0$，尖峰把范围撑大，原本很小但数量众多的分量被压到同一个格子里，误差反而从 $5.4\%$ 涨到 $20.3\%$。

所以正确的结论是：**Hadamard 是一个针对“坐标稀疏 + 少数离群通道”的先验，不是通用最优旋转。** LLM 的激活恰好长这样，所以它好用；换成稠密同号的数据，它会把事情弄糟。

### 8.4 它和学出来的旋转是什么关系

Hadamard 不参与训练，也不需要校准；它是一块“固定的好默认值”。SpinQuant 的做法是：$R_1,R_2$ 用随机 Hadamard 初始化，再在 Stiefel 流形上用 Cayley SGD 优化；$R_3,R_4$ 保持固定 Hadamard，因为在线算得起。

论文在 LLaMA-2 7B 的 W4A4KV4 上报告平均 $64.0$，距全精度 $2.9$ 分。论文的消融里，学到的旋转确实比随机 Hadamard 更好（Mistral-7B 上高出 $15.7$ 分以上）；这同时说明 Hadamard 初始化本身就是一个很强的起点。在单层的 $q/k$ 旋转上，Hadamard 已经把峰值压到接近下界，可学的空间相对有限；学习预算通常花在 $R_1,R_2$ 这种能利用残差、$W_o$、多层累积的位置回报更高。

一句话：**Hadamard 不是终点，但它是默认值、初始化和“没时间调参时的答案”。**

### 8.5 放在 Transformer 的哪里

| 位置 | 维度 | 能不能吸收 | 作用 |
|---|---:|---|---|
| 残差流 $R_1$ | $d_{\text{model}}$ | 能（嵌入/权重） | 整体旋转，压残差里的离群通道 |
| $V$/$W_o$ 配对 $R_2$ | $d_h$ | 能（$W_v$、$W_o$） | 压 $V$ 的离群值 |
| $q,k$ 的 $R_3$（固定 Hadamard） | $d_h$ | 不能（RoPE 挡住） | 压 KV cache 的离群值 |
| FFN 中间 $R_4$（固定 Hadamard） | $d_{ff}$ | 不能（$\odot$ 挡住） | 压 down_proj 输入的离群值 |

两处在线旋转都在**量化之前**做：先转、再按转过的张量定 scale，否则收益白丢。$q$ 和 $k$ 必须用同一个 $R_3$，否则点积对不上；$k$ 转完再写 KV cache，$q$ 转完直接算分数。

## 9. 常见误区

| 说法 | 哪里不对 |
|---|---|
| “Hadamard 是转 45°” | $d$ 维没有单一角度；$H$ 是对合，$d\ge4$ 时在 $d/4$ 个平面里各转 $180^\circ$。 |
| “Hadamard 把所有向量都变均匀” | 它保范数、只换基。$e_0$ 变均匀，全同号向量反而变尖峰。 |
| “Hadamard 是最优旋转” | 只在“峰值下界/坐标尖峰”意义下最优；对稠密同号、甚至某些高斯样本，它不是最优。 |
| “Hadamard 是随机旋转” | $H$ 是确定性矩阵；$DH$ 是离散随机族；Haar 随机正交是另一回事。 |
| “快速变换是近似” | FWHT 精确计算同一个矩阵乘，只是换了求和顺序；误差只来自浮点舍入。 |
| “FWHT 只能做 2 的幂” | Sylvester 蝶形需要 $2^k$；其他维度用块对角/填充，或别的快速分解。 |
| “Hadamard 是 DFT 的一种” | 它是 $({\mathbb Z}/2)^k$ 上的 Fourier 变换，不是 ${\mathbb Z}/d$ 上的 DFT。 |
| “随机 $\pm1$ 矩阵就是 Hadamard” | 行不正交；$d=128$ 时 $\max\lvert S^{\top}S-I\rvert\approx0.34$。Walsh 符号才是关键。 |
| “旋转会改变模型输出” | 正确吸收后全精度输出不变；变的只是中间激活的坐标基。 |
| “$H$ 要占很多显存” | $H_{128}$ 以 fp16 存是 $32$ KB；实际实现通常现算，不存矩阵。 |

## 10. 速查

**矩阵与构造**

$$
H_{2n}=\frac{1}{\sqrt2}\begin{pmatrix}H_n&H_n\\H_n&-H_n\end{pmatrix},
\qquad
(H_d)_{ij}=\frac{(-1)^{\operatorname{popcount}(i\&j)}}{\sqrt d}.
$$

**保范数与对合**

$$
H^{\top}H=I,\qquad H^{\top}=H,\qquad H^2=I,\qquad \lVert xH\rVert_2=\lVert x\rVert_2.
$$

**峰值**

$$
\max_{i,j}\lvert Q_{ij}\rvert\ge\frac{1}{\sqrt d}
\quad(Q\ \text{orthogonal}),
\qquad
\max_{i,j}\lvert H_{ij}\rvert=\frac{1}{\sqrt d},
$$

$$
\lVert xH\rVert_\infty\le\frac{\lVert x\rVert_1}{\sqrt d},
\qquad
\lVert xH\rVert_\infty\le\sqrt{\frac{k}{d}}\,\lVert x\rVert_2\ \ (x\ \text{is }k\text{-sparse}).
$$

**快速变换**

$$
T(d)=d\log_2 d,\qquad
d=128:\ 896\ \text{adds},\qquad
d=11008=86\times128:\ 77{,}056\ \text{adds}.
$$

**实践**

- 维度是 2 的幂：直接 FWHT；否则按 $128$ 分块，或填充到下一个 2 的幂。
- fp16：输入预乘 $1/\sqrt d$，不要每层缩放；关键路径用 fp32 累加。
- 量化：先旋转、再定 scale；$q,k$ 用同一个 $H$；随机符号用 $DH$，不要用随机 $\pm1$ 矩阵。
- 不要默认“Hadamard 一定更好”：先看激活是不是“少数通道大 + 其余小”；如果是稠密同号，旋转可能帮倒忙。

## 11. 自测

**1. 证明：任意 $d\times d$ 正交矩阵的最大元素至少是 $1/\sqrt d$；等号成立时矩阵长什么样？**

<details>
<summary>答案</summary>

设第 $i$ 行是单位向量 $q_i$。则 $1=\sum_j q_{ij}^2\le d\max_j q_{ij}^2$，所以 $\max_j\lvert q_{ij}\rvert\ge1/\sqrt d$。对每一行都成立，所以整个矩阵的最大元素也至少是 $1/\sqrt d$。等号成立要求每一行都取等号，即每个 $\lvert q_{ij}\rvert=1/\sqrt d$；再加上正交性，这就是归一化 Hadamard 矩阵。

</details>

**2. 用 FWHT 手算 $x=(1,2,3,4)$，并数一数一共做了多少次加减。**

<details>
<summary>答案</summary>

第 1 层（$h=1$）：$(3,-1,7,-1)$；第 2 层（$h=2$）：$(10,-2,-4,0)$；除以 $\sqrt4=2$ 得 $(5,-1,-2,0)$。两层、每层 4 次加减，共 $8=d\log_2 d$ 次。

</details>

**3. 取 $d=128$，$x$ 是前 $k=4$ 个坐标全为 $1/\sqrt k$、其余为 $0$ 的单位向量。它经过 $H_{128}$ 后的峰值是多少？和 $1/\sqrt d$ 差多少？**

<details>
<summary>答案</summary>

第一行全 $+1$，所以第一个输出是 $k/\sqrt k/\sqrt d=\sqrt{k/d}=\sqrt{4/128}\approx0.1768$。这是 $k$ 稀疏向量的上界，也是这个特定 $x$ 的实际峰值。普适下界是 $1/\sqrt{128}\approx0.0884$，所以还差 $2$ 倍；随机正交矩阵的典型峰值约 $0.25$，反而更差。

</details>

**4. 为什么不能用一个元素独立取 $\pm1/\sqrt d$ 的随机矩阵冒充 Hadamard 矩阵？**

<details>
<summary>答案</summary>

因为行与行不再正交。$d=128$ 时 $\max\lvert S^{\top}S-I\rvert\approx0.34$，用它做“旋转”会改变全精度输出。Hadamard 的关键不是“幅度都一样”，而是 Walsh 符号让行两两正交；随机化要用 $DH$（$D$ 是随机 $\pm1$ 对角阵），而不是随机符号矩阵。

</details>

**5. 为什么高斯输入经过 Hadamard 后，量化误差几乎不变？**

<details>
<summary>答案</summary>

若 $x\sim\mathcal N(0,I/d)$，则 $xH$ 与 $x$ 同分布（正交变换保持各向同性高斯分布）。逐坐标量化的期望误差只取决于分布，所以任何正交变换都不会改善；旋转的收益来自原分布的**非各向同性**（少数通道特别大）。

</details>

## 附录 A：可运行代码

下面这段 NumPy 代码覆盖构造、FWHT、分块、随机正交和量化对比。`fwht` 已经预缩放，输出就是 $xH_n$；在 CPU 上跑几秒。

```python
import numpy as np
import math


def hadamard(n):
    """归一化 Sylvester Hadamard 矩阵，n 必须是 2 的幂。"""
    assert n >= 1 and (n & (n - 1)) == 0
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]]) / math.sqrt(2)
    return H


def fwht(x):
    """x: (..., n) -> x @ H_n，n 为 2 的幂。"""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[-1]
    assert n >= 1 and (n & (n - 1)) == 0, "n must be a power of two"
    lead = x.shape[:-1]
    x = x.reshape(-1, n) / math.sqrt(n)      # 预缩放：中间值 <= ||x||_2
    h = 1
    while h < n:
        x = x.reshape(-1, n // (2 * h), 2, h)
        a, b = x[:, :, 0, :], x[:, :, 1, :]
        x = np.concatenate([a + b, a - b], axis=2).reshape(-1, n)
        h *= 2
    return x.reshape(*lead, n)


def block_hadamard(dim, block=128):
    """dim 必须是 block 的倍数；返回 I ⊗ H_block。"""
    assert dim % block == 0
    return np.kron(np.eye(dim // block), hadamard(block))


def fwht_block(x, block=128):
    """对最后一维按 block 分块做 FWHT。"""
    assert x.shape[-1] % block == 0
    lead = x.shape[:-1]
    x = x.reshape(*lead, -1, block)
    return fwht(x).reshape(*lead, -1)


def haar(d, rng):
    """Haar 随机正交矩阵。"""
    Q, R = np.linalg.qr(rng.standard_normal((d, d)))
    return Q * np.sign(np.diag(R))


def quant_asym(x, bits=4):
    """逐行非对称 min-max 量化，返回反量化结果。"""
    lo = x.min(axis=-1, keepdims=True)
    hi = x.max(axis=-1, keepdims=True)
    s = (hi - lo) / (2 ** bits - 1)
    s = np.where(s == 0, 1.0, s)
    q = np.clip(np.round((x - lo) / s), 0, 2 ** bits - 1)
    return q * s + lo


# 1) 正确性：FWHT == 矩阵乘
rng = np.random.default_rng(0)
for n in (4, 8, 128, 1024):
    x = rng.standard_normal((3, n))
    assert np.allclose(fwht(x), x @ hadamard(n), atol=1e-12)

# 2) 手算例子
print(fwht(np.array([1.0, 2.0, 3.0, 4.0])))   # [ 5. -1. -2.  0.]

# 3) 分块 == 块对角矩阵乘（这里用 512 阶，免得构造 11008² 的大矩阵）
x = rng.standard_normal(512)
assert np.allclose(fwht_block(x, 128), x @ block_hadamard(512, 128))

# 4) 量化对比：2000 个 LLM 式 token
d = 128
H = hadamard(d)
rngq = np.random.default_rng(1)
X = rngq.standard_normal((2000, d))
X[:, [3, 50, 90]] += rngq.uniform(5, 12, (2000, 3)) * rngq.choice([-1, 1], (2000, 3))
X /= np.linalg.norm(X, axis=1, keepdims=True)
for name, R in [("none", np.eye(d)), ("H", H), ("random", haar(d, rngq))]:
    Z = X @ R
    Zq = quant_asym(Z)
    err = np.linalg.norm(Zq - Z, axis=1) / np.linalg.norm(Z, axis=1)
    print(f"{name:7s} peak={np.abs(Z).max(1).mean():.3f}  rel_err={err.mean():.3f}")
```

PyTorch 的等价写法（把 `np` 换成 `torch`，`concatenate(axis=2)` 换成 `torch.cat(dim=2)`）：

```python
import torch

def fwht_torch(x):
    n = x.shape[-1]
    lead = x.shape[:-1]
    x = x.reshape(-1, n) / (n ** 0.5)
    h = 1
    while h < n:
        x = x.reshape(-1, n // (2 * h), 2, h)
        a, b = x[:, :, 0, :], x[:, :, 1, :]
        x = torch.cat([a + b, a - b], dim=2).reshape(-1, n)
        h *= 2
    return x.reshape(*lead, n)
```

## 附录 B：实验设置

- 维度 $d=128$；4000 个 token；每个 token 先取 $\mathcal N(0,I)$，再在 3 个固定通道上加 $U(5,12)\cdot\pm1$，最后归一化到单位 L2。
- 量化：4-bit per-token 非对称 min-max（主表）和对称 absmax（对照）；误差为逐 token 的 $\lVert\hat z-z\rVert_2/\lVert z\rVert_2$，表中为均值。
- 随机正交：Haar 分布，40 次取平均；峰值表为 200 次取中位数。
- 峰度：Pearson 峰度，高斯为 3。
- 图 1–4 用 matplotlib 生成，亮/暗两套同一份数据、同一 seed，只换配色。
- 这些数字是合成数据上的几何实验，不是模型 benchmark；它们用来隔离“旋转—峰值—量化误差”的关系，不替代 SpinQuant 在真实模型上的评测。

## 参考文献

1. J. J. Sylvester. Thoughts on inverse orthogonal matrices, simultaneous sign successions, and tessellated pavements in two or more colours. *Philosophical Magazine*, 34:461–475, 1867.
2. J. Hadamard. Résolution d'une question relative aux déterminants. *Bulletin des Sciences Mathématiques*, 17:240–246, 1893.
3. J. L. Walsh. A closed set of normal orthogonal functions. *American Journal of Mathematics*, 45(1):5–24, 1923.
4. J. W. Cooley, J. W. Tukey. An algorithm for the machine calculation of complex Fourier series. *Mathematics of Computation*, 19(90):297–301, 1965.
5. R. E. A. C. Paley. On orthogonal matrices. *Journal of Mathematics and Physics*, 12:311–320, 1933.
6. J. Seberry, M. Yamada. Hadamard matrices, sequences, and block designs. In *Contemporary Design Theory*, 431–560, 1992.
7. K. J. Horadam. *Hadamard Matrices and Their Applications*. Princeton University Press, 2007.
8. S. Ashkboos, A. Mohtashami, M. L. Croci, B. Li, P. Cameron, M. Jaggi, D. Alistarh, T. Hoefler, J. Hensman. QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs. NeurIPS 2024. arXiv:2404.00456.
9. Z. Liu, C. Zhao, I. Fedorov, B. Soran, D. Choudhary, R. Krishnamoorthi, V. Chandra, Y. Tian, T. Blankevoort. SpinQuant: LLM Quantization with Learned Rotations. ICLR 2025. arXiv:2405.16406.
10. A. Tseng, J. Chee, Q. Sun, V. Kuleshov, C. De Sa. QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks. ICML 2024. arXiv:2402.04396.
11. T. Dao. fast-hadamard-transform: CUDA implementation of the fast Hadamard transform. https://github.com/Dao-AILab/fast-hadamard-transform
12. Wikipedia. Hadamard matrix. https://en.wikipedia.org/wiki/Hadamard_matrix
