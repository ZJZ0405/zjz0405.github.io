---
title: "SpinQuant：基于可学习旋转的大语言模型量化（中英对照精读）"
published: 2026-10-27
description: ICLR 2025 SpinQuant 中英对照精读：用旋转不变性插入 R1–R4 四段旋转，在 Stiefel 流形上用 Cayley SGD 学 R1/R2；7 模型 × 4 比特配置的主结果、与 QuaRot 的对比、约 8% 在线延迟，附读前预备与 7 处补充推导。
tags: [Quantization, SpinQuant, PTQ, 论文精读, 中英对照, 旋转, 低精度推理]
category: Quantization
draft: false
bilingual: true
---

> **SpinQuant: LLM Quantization with Learned Rotations**  
> Zechun Liu, Changsheng Zhao, Igor Fedorov, Bilge Soran, Dhruv Choudhary, Raghuraman Krishnamoorthi, Vikas Chandra, Yuandong Tian, Tijmen Blankevoort (Meta)  
> ICLR 2025 · [arXiv:2405.16406](https://arxiv.org/abs/2405.16406) · [code](https://github.com/facebookresearch/SpinQuant)

:::en
> **How to read:** this is a source-grounded bilingual reader. Each block starts with a page/block reference; `Original` preserves the paper text and `中文` is the translation. The switch above toggles **bilingual / Chinese only / English only**; Chinese-only notes (prerequisites, derivations, reading notes) are hidden in English mode. Block prefixes: `S` body text, `C` captions, `F` figures, `T` tables, `E` equations, `D` reader-added derivations (7), `N` prerequisites (4).
>
> The front matter below (metadata, prerequisites N001–N004, page index) and the seven derivations are written in Chinese only; switch to **Bilingual** or **中文** to read them.
:::

:::zh
> **中英对照精读版（source-grounded reader）**  
> 原文：Zechun Liu, Changsheng Zhao, Igor Fedorov, Bilge Soran, Dhruv Choudhary, Raghuraman Krishnamoorthi, Vikas Chandra, Yuandong Tian, Tijmen Blankevoort.  
> *SpinQuant: LLM Quantization with Learned Rotations*. ICLR 2025；arXiv:2405.16406 — 本地文件 `SpinQuant_2405.16406.pdf`（24 页）。

**阅读方式**：每个块给出 `Source: p.页码 · 块号`，原句以 `Original:` 起、译文以 `中文:` 起。图表以 300 dpi 紧裁切图片内嵌，图注/表注原文与译文并列，公式用 LaTeX 重排并带 `E...` 锚点。块号规则：`S` 正文、`C` 图注/表注、`F` 图、`T` 表、`E` 公式、`D` 读者补充推导（非论文原文，共 7 处）、`N` 读前预备（非论文原文，共 4 处）。

---

## 0. 元数据与速览

| 项目 | 内容 |
|---|---|
| 标题 | SpinQuant: LLM Quantization with Learned Rotations / 基于可学习旋转的大语言模型量化 |
| 作者 | Zechun Liu\*, Changsheng Zhao\*, Igor Fedorov, Bilge Soran, Dhruv Choudhary, Raghuraman Krishnamoorthi, Vikas Chandra, Yuandong Tian, Tijmen Blankevoort（Meta） |
| 发表 | ICLR 2025（arXiv:2405.16406v4, 20 Feb 2025） |
| 类型 | 方法/算法论文（LLM 训练后量化） |
| 规模 | 24 页：正文 10 页、参考文献 3 页、附录 11 页；11 图、19 表、5 个独立公式 |
| 关键结论 | 4-bit 权重+激活+KV-cache 下，LLaMA-2 7B 与全精度仅差 2.9 分；学到的旋转普遍优于随机/Hadamard 旋转 |
| 补充推导 | 7 处（D1–D7），插在对应公式与段落之后；面向只学过基础线性代数的读者，用最小例子讲思路，严格版另存 |
| 读前预备 | §0.5 讲 RMSNorm（公式、手算例子、为何对旋转免疫、$\gamma$ 为何是麻烦）；§0.6 讲残差与残差流（为什么能整条一起旋转）；§0.7 讲 RoPE（位置编码，以及它为什么挡住 S3）；§0.8 讲 Hadamard 矩阵与快速变换；不熟悉这些的话从这里开始 |

**一句话概括**：论文指出“旋转”本身不改变全精度网络输出，但不同旋转对量化误差影响巨大；于是把旋转矩阵当作可学习参数，在 Stiefel 流形上用 Cayley SGD 直接最小化量化网络的损失，得到比随机旋转、Hadamard 旋转更稳健的 4-bit 量化方案。

不熟悉 RMSNorm 的话，先花两分钟看 [§0.5](#N001)——后面 §3.1 的推导（D2）全靠它。

---

## 0.5 读前预备：RMSNorm 是什么

<a id="N001"></a>

### 为什么需要它

同一个模型里，不同 token、不同层的激活数值大小能差好几个数量级。下一层的权重是按某个固定尺度学出来的，输入忽大忽小会不好用；对量化更直接——一个张量的数值范围越大，量化步长就越粗。归一化的作用就是**把每个向量缩放到统一尺度**，让后面的层和量化器都省心。

### 公式

对 $d$ 维向量 $x$：

$$
\mathrm{RMSNorm}(x)=\frac{x}{\mathrm{RMS}(x)}\odot\gamma,\qquad
\mathrm{RMS}(x)=\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^{2}+\epsilon}.
$$

分母叫**均方根**（root mean square）：先平方、再取平均、再开方。$\epsilon$ 是个很小的常数（$10^{-5}$ 量级），只用来防止除以 0，对后面的推导没有影响。$\gamma$ 是每个坐标一个可学习的缩放系数，训练开始时全为 1。

### 手算一遍

取 $x=(3,4)$，则

- $\mathrm{RMS}=\sqrt{(9+16)/2}=\sqrt{12.5}=3.54$
- $x/\mathrm{RMS}=(0.85,\,1.13)$，此时长度变成 $\sqrt{0.85^{2}+1.13^{2}}=\sqrt{2}$——**归一化后的向量长度恒等于 $\sqrt d$**
- 若 $\gamma=(2,1)$，输出是 $(1.70,\,1.13)$

对照一下 LayerNorm：它还会先减去均值 $\bar x$，RMSNorm 省掉这一步，少一次减法、效果相当，所以 LLaMA 系列用的都是它。论文 Figure 1 没有把归一化层画出来，但它就在每个模块的最前面。

### 为什么它对旋转“免疫”

关键在分母：$\mathrm{RMS}(x)=\sqrt{\|x\|_2^2/d+\epsilon}$ 只由向量长度决定。而旋转不改变长度（$\|xR\|_2=\|x\|_2$），所以分母一模一样，于是

$$
\mathrm{RMSNorm}(xR)=\frac{xR}{s(x)}=\frac{x}{s(x)}R=\mathrm{RMSNorm}(x)\,R .
$$

用大白话说：**先归一化再旋转，等于先旋转再归一化**。这正是 D2 的第一步，也是整个 SpinQuant 能成立的地基。

### 顺带说说 $\gamma$ 为什么是个麻烦

$\gamma$ 是“按坐标各自缩放”，而 $R$ 是“把不同坐标混起来”，两者换个顺序结果就不同。举个最小的例子：取 $x=(1,0)$、$\gamma=(2,1)$、$R$ 是“交换两个坐标”——

- 先缩放再旋转：$x\odot\gamma=(2,0)$，交换后得 $(0,2)$
- 先旋转再缩放：$xR=(0,1)$，再乘 $\gamma$ 得 $(0,1)$

不一样。所以 $\gamma$ 不能留在归一化层里，得先乘进下一层的权重矩阵（论文脚注 2 说的就是这件事，D2 的第二步会再算一遍）。

### 它在 LLaMA 里出现几次

每个 block 两次（attention 前一次、MLP 前一次），最后输出前再一次，共 $2L+1$ 次（$L$ 为层数）。它没有除 $\gamma$ 以外的参数，也不进 KV cache；每次只做一次求和与一次缩放，推理开销很小。

---

## 0.6 读前预备：残差与残差流是什么

<a id="N002"></a>

### 一句话

每个模块不是“改写”主干上的数据，而是算出一个**增量**再加回去：$h\leftarrow h+f(h)$。这条被反复读、反复加、一直传到最后一层的主干，就叫**残差流**（residual stream）。

### 为什么要这么设计

如果每一层都把 $h$ 整个换掉，信息经过几十层会被反复覆盖，梯度也不好传。改成“只加增量”之后，最差情况是 $f(h)=0$（这一层什么都不做），网络至少不会变坏；每层只需学“该补什么”，而不是“整个重写”。名字里的“残差”指的就是这个增量——它等于“想要的输出”减去“输入”。

### 手算一遍

取 $h=(1,2)$，某个模块算出 $f(h)=(0.1,-0.3)$，那么新的 $h=(1.1,\,1.7)$。主干只挪了一点点，大部分信息原样带过去。

### 为什么它能“整条一起旋转”（本文的关键）

加法是线性的。若把主干整体右乘 $R$，即 $h\to hR$，同时让每个模块的增量也变成 $f(h)R$，则

$$
\big(h+f(h)\big)R=hR+f(h)R .
$$

也就是说：**主干和增量一起转，加完之后仍然是“转过的”结果**。这正是 D2 里“整条残差流一起转”的依据——只要每个模块入口把 $R$ 乘回去、出口再乘上 $R$，主干上永远是 $hR$，而模块内部算的还是没旋转过的数。

### 在 LLaMA 里长什么样

- **形状**：$h$ 是“token 数 × 4096”（LLaMA-2 7B 的隐藏维度 $d=4096$），32 个 block 依次读写它。
- **一个 block 写两次**：attention 的输出加一次，MLP 的输出加一次。
- **读它的有 5 处**：$W_q,W_k,W_v,W_{up},W_{gate}$；**写它的有 2 处**：$W_o,W_{down}$——D5 那张吸收清单就是这么来的。
- **pre-norm**：归一化只作用在“读出来的副本”上，主干本身从不被归一化。这一点让“主干整体旋转”变得很干净。
- Figure 1(a) 里那条粉色的线就是残差流（灰色表示未旋转的原始激活）。

### 和量化有什么关系

残差流的主干本身不做量化，量化发生在“读数”的地方，也就是各个线性层的输入。而残差流里往往有少数通道数值特别大（离群值），一旦读数就要量化，量化步长会被这几个大数撑粗——这就是必须在读数之前先做旋转的原因。

---

## 0.7 读前预备：RoPE 是什么

<a id="N003"></a>

### 先说问题：注意力本身不知道顺序

注意力分数是 $q$ 与 $k$ 的点积。把 token 顺序全部打乱，每一对点积还是原来那些，输出只是跟着重排——也就是说，光靠 attention，模型分不出“猫追狗”和“狗追猫”。所以必须有东西把位置信息塞进去。

### 传统做法与 RoPE 的区别

- 老办法是**加**：给每个位置配一个向量，加到词向量上（绝对位置嵌入）。“位置 5 和位置 6 有什么关系”要靠模型自己从数据里学。
- RoPE 是**转**：位置 $t$ 的 $q,k$ 各转一个正比于 $t$ 的角度。两个 token 的点积于是只取决于它们的位置**差**，因为转过的夹角之差正好是 $(t-s)\theta$。相对位置信息就这样直接写进了点积里，不用学。

### 具体怎么转

把 $d_h$ 维拆成 $d_h/2$ 个二维对 $(x_{2i},x_{2i+1})$，第 $i$ 对用固定角度 $\theta_i=10000^{-2i/d_h}$，位置 $t$ 就转 $t\theta_i$：

$$
\Theta_t=\mathrm{blkdiag}\big(R(t\theta_1),\dots,R(t\theta_{d_h/2})\big),\qquad q\mapsto q\Theta_t,
$$

其中每个 $R(\cdot)$ 是普通的二维旋转矩阵。所以 RoPE 是一个**块对角的正交矩阵**：只改方向、不改长度。

### 手算一个（为了让数字好看，把转速取成每 token 1 弧度）

$q=(1,0)$ 放在位置 1，$k=(1,0)$ 放在位置 2：

- 位置 1 转 1 弧度：$(0.540,\,0.841)$
- 位置 2 转 2 弧度：$(-0.416,\,0.909)$
- 点积：$0.540\times(-0.416)+0.841\times0.909=0.540=\cos 1$

结果只跟“位置差 1”有关，与它们各自处在第几个位置无关——这正是 RoPE 想要的性质。（真实 LLaMA 的转速非常小，$10000^{-2i/d_h}$，每个位置只转一点点。）

### 在 LLaMA 里

只作用在 $q,k$ 上，$v$ 不动；嵌入层里没有位置向量；每层、每个 head 各自转。

### 为什么本文要专门提它（见 D4）

因为 RoPE 是“**按位置**转、每个 token 角度不同”，而 S3 的 $H$ 是“**所有 token 用同一个**混合矩阵”，两者的乘法顺序换不了（$H\Theta_t\neq\Theta_tH$）。所以 $H$ 没法提前并进 $W_q,W_k$，只能放在 RoPE 之后现算——这就是 S3 必须“在线”的原因（在线那一步具体怎么算、为什么便宜，见 D4 后半段）。

### 别和 SpinQuant 的旋转搞混

两个都叫 rotation，但完全不是同一件事：

| | RoPE | SpinQuant 的 $R$ |
|---|---|---|
| 目的 | 编码位置 | 摊平离群值、改善量化 |
| 谁用 | 每个 token 按位置不同 | 所有 token 共用同一个 |
| 什么时候有 | 预训练时就在 | 量化前才加上 |

---

## 0.8 读前预备：Hadamard 矩阵与快速变换

<a id="N004"></a>

### 它是什么

一个 $d\times d$ 的矩阵，元素只有 $\pm1$（归一化后是 $\pm1/\sqrt d$），且任意两列互相垂直。于是它就是一个旋转矩阵：

$$
H^{\top}H=I .
$$

$d$ 取 2 的幂时按 Sylvester 方式构造，$4\times4$ 长这样：

$$
H_4=\frac{1}{2}\begin{pmatrix}
1&1&1&1\\
1&-1&1&-1\\
1&1&-1&-1\\
1&-1&-1&1
\end{pmatrix}.
$$

可以随手验一下：任取两列做内积都是 $0$，每列长度都是 $1$。

### 它做的事：只加减，不做乘法

输出的第 $j$ 个分量，就是输入各分量带 $\pm1$ 符号相加。所以“乘 Hadamard”不需要真正的乘法。单尖峰会被彻底摊平：

$$
(1,0,0,0)\,H_4=(0.5,\,0.5,\,0.5,\,0.5),
$$

最大值 $1\to0.5$、范围 $1\to0$。这就是 D1 说的“把尖峰摊开”最干净的例子，也是 S3/S4 想做的事。

### 为什么便宜：快速变换

这个矩阵是递归拼出来的（$H_{2d}$ 由两块 $H_d$ 组成），所以可以分轮做加减：

$$
(1,2,3,4)\ \xrightarrow{\text{第 1 轮}}\ (3,-1,7,-1)\ \xrightarrow{\text{第 2 轮}}\ (10,-2,-4,0)\ \xrightarrow{\div 2}\ (5,-1,-2,0),
$$

与直接算出的 $(1,2,3,4)H_4=(5,-1,-2,0)$ 一致。两轮共 $8$ 次加减，直接乘矩阵则要 $4\times4=16$ 次乘加；$d$ 越大差距越大（$d\log_2 d$ 对 $d^{2}$）。这正是 S4 里 $d_{ff}=11008$ 也用得起的原因——按 $128$ 分块，每块做小的快速变换。

### 拿一个具体向量走一遍

取 $x=(7,0,0,1)$——一个坐标很大、其余接近 $0$，正是离群值的样子。

**第 1 轮**（相邻两两一加一减）：

$$
(7,0,0,1)\ \longrightarrow\ (7+0,\ 7-0,\ 0+1,\ 0-1)=(7,\,7,\,1,\,-1)
$$

**第 2 轮**（隔一个再做同样的事）：

$$
(7,7,1,-1)\ \longrightarrow\ (7+1,\ 7-1,\ 7-1,\ 7+1)=(8,\,6,\,6,\,8)
$$

**除以 $\sqrt4=2$**，得 $(4,3,3,4)$。和直接乘矩阵的结果一致：

$$
xH_4=\tfrac12(7+0+0+1,\ 7-0+0-1,\ 7+0-0-1,\ 7-0-0+1)=\tfrac12(8,6,6,8)=(4,3,3,4).
$$

**看范围**：旋转前 $\max-\min=7-0=7$，旋转后 $4-3=1$，缩小 7 倍。

**接着量化**（4-bit 非对称 min-max，就是式 (1)）：步长 $s=(4-3)/15=0.0667$，四个数落在 $3$、$4$ 两端，编码成 $0$ 或 $15$，反量化回来正好是 $3.0$ 与 $4.0$，没有损失。若不旋转，步长是 $7/15=0.467$，粗了 7 倍，那些接近 $0$ 的小数会被压进同一个格子。

**推广到真实尺寸**：$d=128$ 就是这条规则做 $7$ 轮；$d_{ff}=11008$ 拆成 $86$ 个 $128$ 的块，每块做 $7$ 轮。S3 对每个 head 的 $q,k$（$128$ 维）各做一次，S4 对 $z$ 的 $86$ 块各做一次——算法完全一样，只是作用对象不同。

### “转多少度”这个问题在 $d$ 维里没有答案

二维旋转只有一个角度，所以能问“转多少度”。到了 $d$ 维，一个正交矩阵的自由度是 $d(d-1)/2$ 个角度——$d=128$ 时是 $8128$ 个——没有单一的“转多少”。$H_{128}$ 也不是“转 45°”，它是**把所有坐标完全混合**：每个输出分量都是全部输入分量的等权加减。描述它的正确说法是“混得多均匀”，不是“转了多少”。

这也意味着**“只转一点点”没有意义**。若把旋转写成连续形式 $R(t)=\exp(tA)$，$t$ 很小时

$$
xR(t)\approx x+t\,xA,
$$

离群值几乎没动，量化步长也不会变小，白花力气。所以方法里根本没有“旋转强度”这个超参。

真正可调的是几个离散选择：分块大小（128）、要不要乘随机符号对角阵、每层是否共用同一个 $H$、以及 $R_1,R_2$ 优化多少步。

当然“转得多”也不总更好：本节后面那个 $(1,1,1,1)H_4=(2,0,0,0)$ 就是“混合过头”的例子——同一个向量的分量全同号时，完全混合反而把能量挤到了一个坐标上。

### 它到底“均分”了没有

“均分”有两个意思，混着用会得出相反的结论。

**保范数意义上的均分，所有正交矩阵都做到了.** $\lVert xH\rVert_2=\lVert x\rVert_2$ 对任何正交矩阵成立（Parseval），这是“旋转”的定义，不是 $H$ 的功劳。所以若说的是“能量被摊开”，$H$ 和随便一个随机正交矩阵没有区别。

**峰值意义上的均分，只有 Hadamard 做到了极致.** 正交矩阵的每一行是单位向量，所以 $\max|\text{元素}|\ge 1/\sqrt d$；所有行同时取到这个下界，当且仅当每个元素都是 $\pm1/\sqrt d$——这正是 Hadamard 的定义。$d=128$ 时是 $0.0884$；随机正交矩阵的典型峰值约 $\sqrt{2\ln d/d}\approx0.28$，大 3 倍多。$1/\sqrt d$ 是下界，不可能更“均分”。

**但它不是“把什么都变均匀”，是在搬.** 保范数就意味着总量守恒，$H$ 只能把“尖”和“平”这两个方向**互换**：尖峰 $e_0$ 转完全平，而全平向量 $(1,\dots,1)/\sqrt d$ 转完是 $(1,0,\dots,0)$——一个尖峰。这也解释了 $H^2=I$：搬过去还能搬回来（S4 的逆折进 $W_{down}$ 靠的就是这一条）。所以 $H$ 的适用场景很明确：**输入在坐标基下是稀疏尖峰**。LLaMA 的残留离群值恰好是这个形状；换成本来就均匀的输入，$H$ 反而会把它挤成一个尖峰。

**幅度均匀不等于“纯均分”，符号才是干活的.** 符号全取 $+1$（整个矩阵都是 $+1/\sqrt d$）得到的是秩 1 的“求平均”，信息全丢，范数也不再守恒。符号随机取 $\pm1$ 呢？行与行不再正交，$d=128$ 时 $\max|R^{\top}R-I|\approx0.33$，差得远。真正让行两两正交的，是按 $\langle i,j\rangle$ 奇偶排出来的那套 Walsh 符号。所以准确说法是“**均匀幅度 + 特定符号编排**”，两者缺一不可。

顺带一个后果：$H=H^{\top}=H^{-1}$，所以它的特征值只能是 $\pm1$——D4 里量到的“$I+H_{128}$ 不满秩、$-1$ 重数 64”，根子就在这里。

### 为什么 $1/\sqrt d$ 是下界，“唯一”又是什么意思

三行就证完了。正交矩阵的每一行是单位向量，所以

$$
\sum_{j=1}^{d}H_{ij}^{2}=1
$$

$d$ 个平方数加起来是 $1$，平均值就是 $1/d$；最大值不可能小于平均值，于是 $\max_j|H_{ij}|\ge1/\sqrt d$。取等号只有一种情况：这 $d$ 个平方数全都等于 $1/d$，即**这一行的每个元素绝对值都恰好是 $1/\sqrt d$**。

把这两步写成不等式更清楚。记 $a_j=H_{ij}^2\ge0$，一方面 $\sum_j a_j=1$，另一方面每个 $a_j$ 都不超过其中最大的那个，所以 $\sum_j a_j\le d\cdot\max_j a_j$。两式一夹：

$$
\max_j a_j\ \ge\ \frac{1}{d}\sum_j a_j\ =\ \frac{1}{d}.
$$

等号被这两步同时钉死：若 $\max_j a_j=1/d$，则 $\sum_j a_j\le d\cdot(1/d)=1$ 且左边已知等于 $1$，两边都取到等号，于是每个 $a_j$ 都必须等于 $1/d$——**一行里不可能有的元素大、有的元素小**。证明里没有用到“正交”以外的任何信息：$a_j$ 的和为 $1$ 来自“行是单位向量”，而“行是单位向量”来自正交。

对每一行都这样看：任何正交矩阵的 $\max|\text{元素}|\ge1/\sqrt d$，而要让整个矩阵的 $\max|\text{元素}|$ 恰好等于 $1/\sqrt d$，必须所有行同时取等号，也就是每个元素都是 $\pm1/\sqrt d$。再加上“本身正交”这个前提，这就是归一化 Hadamard 矩阵的定义。所以“只有 Hadamard 做到极致”不是形容词，是等号成立条件的直接翻译。

**“对一个尖峰最优”和“对所有尖峰最优”是两件事.** 想让 $e_0$ 被摊平，只要第 $0$ 行全平就行——任意正交矩阵都能安排这样一行，再补几条正交的行填满即可。但要让 $e_0,e_1,\dots,e_{d-1}$ 这 $d$ 个尖峰**同时**被摊平，就必须每一行都全平。用 $d=8$ 做对照：一个第 $0$ 行全平的正交矩阵，第 $0$ 行峰值是 $0.354$（正好下界），其余各行的峰值在 $0.51\sim0.70$ 之间；它只对 $1$ 个尖峰最优，Hadamard 对 $8$ 个全中。

**尖峰不止一个时，缺口是 $\sqrt k$.** 单位向量若有 $k$ 个非零分量，Hadamard 保证 $\lVert xH\rVert_\infty\le\sqrt k/\sqrt d$（对等幅同号的输入这个上界取得到），而普适下界仍是 $1/\sqrt d$。$d=128$ 时逐个数：$k=1$ 得 $0.0884$，$k=3$ 得 $0.1531=\sqrt3/\sqrt{128}$，$k=10$ 得 $0.2795=\sqrt{10}/\sqrt{128}$。D4 里那个峰值因子 $2.45$，就是这 $\sqrt k$ 加上噪声、再加上“所有 token 共用同一个 $R$”堆出来的。

**这个下界是对“对称量化”说的.** 上面 $\max_i|z_i|$ 之所以是决定性的量，前提是**步长由最大绝对值定**：零点固定在 $0$、码字 $[-7,7]$、步长 $=\max_i|z_i|/7$，这就是“以 $0$ 为中心对称缩放”。论文式 (1) 用的却是**非对称 min-max**：步长 $=(\max-\min)/15$，零点挪到 $\min$。这时决定误差的是**范围** $\max-\min$，而范围**没有**类似的普适下界——一个全均匀的向量范围是 $0$，量化误差也是 $0$。所以：

- 对称量化下，$\max_i|z_i|\ge\lVert z\rVert/\sqrt d$ 是硬下界，Hadamard 在单尖峰上取到它，整套“最优”论证成立；
- 非对称量化下，对应的量是半宽 $(\max-\min)/2$，没有普遍下界，只能得到“旋转把范围压窄”这个方向性结论。

两种约定下“旋转带来数倍改善、$H$ 接近最优”都成立，只是“最优”的精确含义只在对称情形下可证。$d=128$、400 个 token（每 token 自带 scale）分别按两种步长算：

| $R$ | $\max_i\lvert z_i\rvert$ | 半宽 | 对称 4-bit | 非对称 4-bit |
|---|---|---|---|---|
| 不转 | 0.672 | 0.470 | 29.7% | 20.3% |
| 随机正交 | 0.250 | 0.230 | 11.6% | 10.0% |
| Hadamard | 0.216 | 0.202 | 10.0% | 8.7% |

误差是逐 token 的相对 L2。非对称在三种 $R$ 下都更低，因为它不浪费负半轴；但两者的相对改善幅度几乎一样（约 3 倍）。

**论文自己选的是非对称.** §A.4 的消融（Table 12）在 W4A4 上：对称 $61.2$、非对称 $61.8$（RTN），对称 $63.3$、非对称 $64.0$（GPTQ）；KV cache 上非对称只多 $0.1$ 分；裁剪与否影响不显著。所以论文最终对**激活与 KV cache 都用非对称 min-max、不裁剪**。这意味着论文语境里被旋转压下去的量严格来说是**范围**；$1/\sqrt d$ 这个下界是理解“旋转为什么有用”的辅助工具（它给出了对称情形下的精确最优点），而不是论文的量化准则。好在两个量高度相关——上表里 $\max_i|z_i|$ 与半宽几乎同步变化——所以用哪个都不会改变“旋转值数倍改善、$H$ 接近最优”这个结论。

### 怎么一步步造出来

从 $1\times1$ 的 $H_1=(1)$ 出发，反复用同一条规则把边长翻倍：

$$
H_{2n}=\frac{1}{\sqrt2}\begin{pmatrix}H_n & H_n\\ H_n & -H_n\end{pmatrix}.
$$

翻七次就得到 $H_{128}$，元素全是 $\pm1/\sqrt{128}=\pm0.0884$。写成代码只有四行：

```
H = [[1.0]]
while len(H) < n:
    H = [[H, H], [H, -H]] / sqrt(2)     # 逐块拼接，n 取 2 的幂
```

不需要任何训练，也不需要优化：这个矩阵在推理前构造一次，之后一直用。

也可以跳过递推，直接写出每个元素。设 $i,j$ 从 $1$ 数起，令 $\langle i,j\rangle$ 表示 $i-1$ 与 $j-1$ 的二进制表示中**同为 1 的位的个数**，则

$$
H_{ij}=\frac{(-1)^{\langle i,j\rangle}}{\sqrt d}.
$$

$8\times8$ 的符号图案（先不看 $1/\sqrt8$）长这样，第 1 行、第 1 列全是 $+$：

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

同一个东西还有第三种写法——Kronecker 积：$H_{2^k}=\underbrace{H_2\otimes\cdots\otimes H_2}_{k\ \text{个}}$，$H_{128}$ 就是 $7$ 个 $\frac{1}{\sqrt2}\begin{pmatrix}1&1\\1&-1\end{pmatrix}$ 乘起来。

**要强调的是：这三个写法都只是“把 $\pm1$ 排好”，没有任何一步是从数据或权重里算出来的。** $H_3,H_4$ 是常量；真正从数据学出来的只有 $R_1,R_2$。

可运行脚本：`tools/build_hadamard.py` 打印 $H_{128}$ 的规模、正交误差、块数与常量占用；`S34.py` 有五个可手算的小例子（构造、点积不变、KV cache 4-bit、FFN 4-bit、RoPE 不可换序）；`compare_s3_cayley_vs_hadamard.py` 在 $d=128$ 上比“Cayley 学一个 $R_3$”与“直接用 $H$”（结论见 D4 的“能不能用 Cayley 学一个更好的 $R_3$”）。

### 维度不是 2 的幂怎么办

$d_h=128$ 没问题；但 $d_{ff}=11008=86\times128$ 不是 2 的幂，造不出这个尺寸的 Hadamard。实际做法是用**块对角**：把 11008 切成 86 个 128 的块，每块各自乘一个 $H_{128}$：

$$
H_4=\mathrm{blkdiag}\big(\underbrace{H_{128},\ \dots,\ H_{128}}_{86\ \text{块}}\big).
$$

分块正交就意味着整体正交，所以抵消与等价性全部照旧；而一次变换仍然只是 86 个小快速变换，代价不变。论文正文只写了“插入 Hadamard 矩阵”，没写分块细节——这是实现层面绕不开的一步，属于合理推断而非原文。

### 乘一个随机符号对角阵还算数吗

算。取对角阵 $D$，对角线上每个元素独立地取 $\pm1$，令 $H'=DH$。$D$ 自己就是正交矩阵，所以 $H'$ 仍然正交，S3 的抵消照样成立：

$$
(qDH)(kDH)^{\top}=qD\,H H^{\top}D^{\top}k^{\top}=qDD^{\top}k^{\top}=qk^{\top}.
$$

作用只是给每个坐标随机换个正负号，让不同层、不同 head 的变换不完全一样。论文脚注 1 说的“用对角阵 $S$ 生成随机 Hadamard 矩阵”就是这件事。

### 论文里用在哪

- **S3**：每个 head 的 $q,k$，维度 $d_h=128$；
- **S4**：FFN 中间激活，维度 $d_{ff}=11008$（按 $128$ 分块）。

常量占用很小：一份 $H_{128}$ 以 fp16 存是 $32$ KB。LLaMA-2 7B 有 $32$ 层、每层 $32$ 个 head，若真的每 head 存一份也才 $32$ MB；实际实现通常全局共用一份 $H_{128}$，需要随机化时只额外存每层 $128$ 位的符号掩码。

两处都在推理时现算，所以叫“在线”，见 D4。

### 为什么默认用 Hadamard，但它并不是最优的

**好在哪里.** 它的每个元素幅度相同（$\pm1/\sqrt d$），于是对任何输入都能给出一个与分布无关、而且紧的上界：$|(xH)_j|\le\lVert x\rVert_1/\sqrt d$。在“不知道 $x$ 长什么样”的前提下，这是最稳的选择——不会因为某个通道特别大而突然变差。论文 §2.2 说“Hadamard 矩阵对权重最大值的界更紧”，就是这个意思。

**但它不是对每个 $x$ 都最优.** 反例：取 $x=(1,1,1,1)$，

$$
(1,1,1,1)H_4=(2,0,0,0).
$$

能量全被推到第一个坐标上，最大分量是 $2$；而“完全摊平”的理想值是 $\lVert x\rVert_2/\sqrt d=1$。也就是说，同一个向量的每个分量都同号时，Hadamard 反而比随机旋转差。真实权重和激活接近零均值、正负相消，所以实测中它通常很好——但这是经验，不是定理。

**论文也没有声称 S3/S4 最优.** 论文里“更优的旋转”这一主张（Figure 4、Table 2/4）针对的是**学出来的 $R_1,R_2$**；$H_3,H_4$ 从头到尾不参与优化，保留 Hadamard 的理由是“在线算得起”（§3.2），不是“它最优”。

### 一处勘误

论文脚注 1 有两处记法笔误：一是元素写作 $\pm\sqrt n$，按定义应为 $\pm1/\sqrt n$（归一化前是 $\pm1$）；二是符号变体的个数写作 $2n$，按定义应为 $2^{n}$（$n$ 个坐标各自选一次正负）。两处都不影响结论。

---

## 0.9 目录与页码索引

- [0.5 读前预备：RMSNorm](#N001) — 阅读器补充，非原文
- [0.6 读前预备：残差与残差流](#N002) — 阅读器补充，非原文
- [0.7 读前预备：RoPE](#N003) — 阅读器补充，非原文
- [0.8 读前预备：Hadamard 矩阵](#N004) — 阅读器补充，非原文
- [ABSTRACT](#S004) — p.1
- [1 INTRODUCTION](#S006) — p.1
- [2 MOTIVATION](#S018) — p.3
  - [2.1 OUTLIER REDUCTION](#S022) — p.3
  - [2.2 RANDOM ROTATIONS PRODUCE LARGE VARIANCE](#S025) — p.4
- [3 METHOD](#S028) — p.4
  - [3.1 ROTATION PARAMETERIZATION](#S030) — p.4
  - [3.2 Cayley-OPTIMIZED ROTATION](#S037) — p.5
- [4 EXPERIMENTS](#S045) — p.6
  - [4.1 EXPERIMENTAL SETTINGS](#S047) — p.6
  - [4.2 MAIN RESULTS](#S050) — p.6
  - [4.3 ABLATION STUDIES](#S057) — p.8
  - [4.3.1 LEARNED ROTATION VS RANDOM ROTATION](#S058) — p.8
  - [4.3.2 COMPATIBILITY WITH GPTQ](#S060) — p.8
  - [4.3.3 ROTATION TYPE](#S062) — p.8
  - [4.3.4 COMPARISON WITH QUAROT](#S064) — p.8
  - [4.4 ILLUSTRATIVE ANALYSIS OF THE ROTATION EFFICACY](#S068) — p.9
  - [4.5 SPEED MEASUREMENT](#S074) — p.10
- [5 RELATED WORK](#S076) — p.10
- [6 CONCLUSIONS](#S079) — p.10
- [ACKNOWLEDGMENT](#S081) — p.11
- [REFERENCES](#S083) — p.11
- [A APPENDIX / SUPPLEMENTAL MATERIAL](#S134) — p.14
  - [A.1 COMPLETE RESULTS OF MAIN RESULT TABLE](#S135) — p.14
  - [A.2 RESULTS ON 3-BIT WEIGHT QUANTIZATION](#S137) — p.14
  - [A.3 Cayley OPTIMIZATION CHOICE](#S139) — p.14
  - [A.4 QUANTIZATION CHOICE](#S141) — p.14
  - [A.5 CALIBRATION DATA CHOICE](#S143) — p.14
  - [A.6 LATENCY MEASUREMENT ON GPU](#S145) — p.14
  - [A.7 OPTIMIZATION TIME](#S150) — p.18
  - [A.8 ABLATION STUDY ON RTN VS GPTQ](#S152) — p.18
  - [A.9 WEIGHT-ONLY QUANTIZATION](#S154) — p.18
  - [A.10 FEW-SHOT RESULTS ON INSTRUCTION-FINETUNED MODELS](#S156) — p.18
- [B ANALYSIS](#S158) — p.18
  - [B.1 GRADIENT ANALYSIS](#S159) — p.18
  - [B.2 LOSS ANALYSIS](#S163) — p.19
  - [C DISTRIBUTION VISUALIZATIONS BEFORE AND AFTER ROTATION](#S165) — p.20

**补充推导（读者补充，非原文）**

- [D1 旋转到底帮了量化什么忙](#D001) — p.3 · §2.1 末
- [D2 S1（残差旋转）为什么等于没动](#D002) — p.5 · S033 后
- [D3 S2（V 上的旋转）为什么能穿过注意力](#D003) — p.5 · 图 5 后
- [D4 S3 和 S4 为什么必须“在线”，在线在算哪一步](#D004) — p.5 · S036 后
- [D5 四段合起来](#D005) — p.5 · §3.1 末
- [D6 Cayley 更新为什么保正交，$R_1,R_2$ 怎么算出来](#D006) — p.6 · S042 后
- [D7 既然输出不变，凭什么还能学 R](#D007) — p.19 · B.1 末

---

:::

<a id="S004"></a>
## 摘要 / Abstract

<a id="S005"></a>
**Source:** p.1 S005

:::en
**Original:** Post-training quantization (PTQ) techniques applied to weights, activations, and the KV cache greatly reduce memory usage, latency, and power consumption of Large Language Models (LLMs), but may lead to large quantization errors when outliers are present. Rotating activation or weight matrices helps remove outliers and benefits quantization. In this work, we identify a collection of applicable rotation parameterizations that lead to identical outputs in full-precision Transformer architectures while enhancing quantization accuracy. In addition, we find that some random rotations lead to much better quantization than others, with an up to 13 points difference in downstream zero-shot reasoning performance. As a result, we propose SpinQuant, a novel approach that incorporates learned rotation matrices for optimal quantized network accuracy. With 4-bit quantization of weight, activation, and KV-cache, SpinQuant narrows the accuracy gap on zero-shot reasoning tasks with full precision to merely 2.9 points on the LLaMA-2 7B model, surpassing LLM-QAT by 19.1 points and SmoothQuant by 25.0 points. Furthermore, SpinQuant also outperforms concurrent work QuaRot, which applies random rotations to remove outliers. In particular, for LLaMA-3 8B models that are hard to quantize, SpinQuant reduces the gap to full precision by up to 45.1% relative to QuaRot. Code is available at github.com/facebookresearch/SpinQuant.
:::

:::zh
**中文:** 训练后量化（PTQ）技术作用于权重、激活与 KV cache，能显著降低大语言模型（LLM）的内存占用、延迟与功耗，但当存在离群值（outlier）时可能带来较大的量化误差。对激活或权重矩阵施加旋转有助于消除离群值并改善量化。在本工作中，我们识别出一类适用的旋转参数化方式：它们在完整精度的 Transformer 架构中产生完全相同的输出，同时提升量化精度。此外，我们发现某些随机旋转带来的量化效果远好于其他随机旋转，在下游零样本推理性能上最多可相差 13 分。因此，我们提出 SpinQuant，一种引入可学习旋转矩阵以获得最优量化网络精度的方法。在权重、激活与 KV-cache 全部量化为 4-bit 时，SpinQuant 在 LLaMA-2 7B 模型上将零样本推理任务相对全精度的精度差距缩小到仅 2.9 分，分别比 LLM-QAT 高 19.1 分、比 SmoothQuant 高 25.0 分。此外，SpinQuant 也优于同期工作 QuaRot（后者用随机旋转来消除离群值）。特别是对于难以量化的 LLaMA-3 8B 模型，SpinQuant 将相对 QuaRot 与全精度的差距最多缩小 45.1%。代码见 github.com/facebookresearch/SpinQuant。
:::

<a id="S006"></a>
## 1. 引言 / Introduction

<a id="S007"></a>
**Source:** p.1 S007

:::en
**Original:** Large Language models (LLMs) have demonstrated impressive performance across many disciplines. SoTA open source models (e.g., LLaMA (Touvron et al., 2023b), Mistral (Jiang et al., 2023), etc) and proprietary LLMs (e.g., GPT (Achiam et al., 2023), Gemini(Team et al., 2023), etc) have been used in general purpose chatting assistants, medical diagnosticians (Thirunavukarasu et al., 2023), computer game content generators (Cox and Ooi, 2023), coding co-pilots (Roziere et al., 2023), and much more.
:::

:::zh
**中文:** 大语言模型（LLM）已在众多学科中展现出令人瞩目的表现。SoTA 开源模型（如 LLaMA (Touvron et al., 2023b)、Mistral (Jiang et al., 2023) 等）与专有 LLM（如 GPT (Achiam et al., 2023)、Gemini (Team et al., 2023) 等）已被用于通用聊天助手、医疗诊断 (Thirunavukarasu et al., 2023)、游戏内容生成 (Cox and Ooi, 2023)、编程助手 (Roziere et al., 2023) 等众多场景。
:::

<a id="S008"></a>
**Source:** p.1 S008

:::en
**Original:** To serve such a high demand, the inference cost becomes a real issue. Many effective techniques have been developed. Post-training Quantization (PTQ), as one effective category of techniques, quantizes the weights (or activations) into low-precision and thus reduces the memory usage and may significantly improve latency. This is not only important for server-side inference, but also for on-device scenarios with small-sized LLMs (Liu et al., 2024; AI@Meta, 2024).
:::

:::zh
**中文:** 为满足如此巨大的需求，推理成本成为现实问题，已有许多有效的技术被提出。训练后量化（PTQ）作为其中一类有效技术，将权重（或激活）量化为低精度，从而降低内存占用并可能显著改善延迟。这不仅对服务端推理重要，对部署小型 LLM 的端侧场景同样重要 (Liu et al., 2024; AI@Meta, 2024)。
:::

<a id="S009"></a>
**Source:** p.1 S009

:::en
**Original:** When applying quantization, outliers remain an open challenge because they stretch the quantization range, leaving fewer effective bits available for the majority of values. Prior research mitigates this challenge by trading quantization difficulty between weights and activations (Xiao et al., 2022; Lin et al., 2023) or employing mixed-precision to handle outliers (Zhao et al., 2023). In this work, we focus on a new angle: multiplying the weight matrix with a rotation matrix to reduce outliers and enhance quantizability. Inspired by (Elhage et al., 2023) and SliceGPT (Ashkboos et al., 2023a), we leverage the property of rotational invariance to construct rotation matrices in pairs from identity mapping, which can be integrated into nearby weights without affecting the overall network outputs. By applying these random rotations, we produce a distribution of weight or activation entries that is outlier-less, facilitating easy quantization.
:::

:::zh
**中文:** 在应用量化时，离群值仍是未解的挑战：它们会拉伸量化范围，使大多数数值只分到更少的有效比特。既有研究通过在权重与激活之间转移量化难度 (Xiao et al., 2022; Lin et al., 2023)，或采用混合精度处理离群值 (Zhao et al., 2023) 来缓解这一问题。在本工作中，我们从一个新角度切入：将权重矩阵与旋转矩阵相乘，以减少离群值并增强可量化性。受 (Elhage et al., 2023) 与 SliceGPT (Ashkboos et al., 2023a) 启发，我们利用旋转不变性，从恒等映射出发成对构造旋转矩阵，使其可以并入相邻权重而不影响网络的整体输出。施加这些随机旋转后，权重或激活元素呈现出无离群值的分布，从而便于量化。
:::

<a id="S010"></a>

:::en
> **Footnote:** ∗ Equal contribution. Correspondence to: Zechun Liu <zechunliu@meta.com>.
:::

:::zh
> **脚注：** ∗ 同等贡献。通讯作者：Zechun Liu <zechunliu@meta.com>。
:::

<a id="F001"></a>
#### Fig. 1. Overall Diagram of Rotation <span class="lang-zh">· 旋转的整体示意图</span>

**Placed near:** p.1 S009  
**Source:** p.2 F001 · C001

<img src="/images/quantization/spinquant/figures/F001.png" alt="Figure 1" width="1641" height="607" loading="lazy" decoding="async">

<a id="C001"></a>

:::en
**Original caption:** Figure 1: Overall diagram of rotation. (a) The residual stream can be rotated in the transformer network, resulting in numerically equivalent floating point networks before and after rotation. The rotated activations exhibit fewer outliers and are easier to quantize. (b) & (c) The rotation matrix can be integrated with the corresponding weight matrices and we further define R2, R3, and R4 for reducing outliers inside the block.
:::

:::zh
**中文图注:** 图 1：旋转的整体示意图。(a) 残差流可以在 Transformer 网络中被旋转，使旋转前后的浮点网络在数值上等价。旋转后的激活离群值更少，更易量化。(b) 与 (c) 旋转矩阵可以与相应的权重矩阵整合，我们进一步定义 $R_2$、$R_3$、$R_4$ 以削减模块内部的离群值。
:::

:::zh
**Reading note:** 读图要点：(a) 残差流整体旋转 $R_1$；(b) 注意力模块内 $R_2$ 与在线 Hadamard $R_3$；(c) MLP 内 $R_4$。绿色/紫色框分别表示可合并旋转与在线旋转。
:::

<a id="S011"></a>
**Source:** p.2 S011

:::en
**Original:** In addition to using random rotation, which statistically works well, we find that the performance of quantized network could vary a lot with different rotation matrices. For example, the downstream averaged accuracy on zero-shot reasoning tasks may change up to 13 points with different rotations. As a result, we propose SpinQuant that integrates and optimizes the rotation matrix to minimize the final loss of the quantized network, with fixed weight parameters, by employing the Cayley SGD (Li et al., 2020), a proficient technique for optimizing orthonormal matrices. This optimization does not alter the full-precision network output but refines the intermediate activations and weights, making them more quantization-friendly.
:::

:::zh
**中文:** 除了使用统计上效果良好的随机旋转之外，我们还发现量化网络的性能会随不同旋转矩阵而大幅波动。例如，在零样本推理任务上，下游平均准确率在不同旋转下最多可变化 13 分。因此，我们提出 SpinQuant：在权重参数固定的条件下，采用 Cayley SGD (Li et al., 2020)（一种擅长优化正交矩阵的技术）对旋转矩阵进行整合与优化，以最小化量化网络的最终损失。该优化不改变全精度网络的输出，而是改善中间激活与权重，使其更利于量化。
:::

<a id="S012"></a>
**Source:** p.2 S012

:::en
**Original:** In SpinQuant, we introduce two rotation strategies tailored for different complexity levels: SpinQuant$_{no\ had}$ and SpinQuant$_{had}$. Here, had refers to hadamard rotation matrix. In SpinQuant$_{no\ had}$, as depicted in Figure 1(b), we implement shortcut rotation (R1) and Wv- Wo pair rotation (R2), which can be directly absorbed into the respective weight matrices. During inference, the original weights are simply replaced with the rotated quantized weights, eliminating the need for modification in the forward pass. Conversely, in SpinQuant$_{had}$, designed for scenarios with low-bit quantization of KV cache or activations (e.g., 4-bit), we further incorporate online Hadamard rotation matrices (R3 , R4 ) to address activation outliers inside MLP block and KV cache.
:::

:::zh
**中文:** 在 SpinQuant 中，我们针对不同的复杂度需求引入两种旋转策略：SpinQuant$_{no\ had}$ 与 SpinQuant$_{had}$。其中 had 指 Hadamard 旋转矩阵。如 Figure 1(b) 所示，在 SpinQuant$_{no\ had}$ 中我们实现捷径旋转（$R_1$）与 $W_v$-$W_o$ 配对旋转（$R_2$），它们可直接并入相应的权重矩阵。推理时只需用旋转后的量化权重替换原始权重，无需改动前向过程。相反，SpinQuant$_{had}$ 面向 KV cache 或激活的低比特量化（如 4-bit）场景，我们进一步引入在线 Hadamard 旋转矩阵（$R_3$、$R_4$），以处理 MLP 模块内部与 KV cache 中的激活离群值。
:::

<a id="S013"></a>
**Source:** p.2 S013

:::en
**Original:** To rigorously assess the effectiveness of SpinQuant, we executed comprehensive experiments across seven leading Large Language Models (LLMs), including LLaMA-2(Touvron et al., 2023b) models (7B/13B/70B), LLaMA-3(AI@Meta, 2024) models (1B/3B/8B), and the Mistral (Jiang et al., 2023) 7B model. The key contributions of this study are summarized as follows:
:::

:::zh
**中文:** 为严格评估 SpinQuant 的有效性，我们在七个主流大语言模型（LLM）上开展了全面实验，包括 LLaMA-2 (Touvron et al., 2023b) 系列（7B/13B/70B）、LLaMA-3 (AI@Meta, 2024) 系列（1B/3B/8B）以及 Mistral (Jiang et al., 2023) 7B 模型。本工作的主要贡献总结如下：
:::

<a id="S014"></a>
**Source:** p.2 S014

:::en
**Original:** • We introduce SpinQuant, the first method that employs learned rotations to mitigate outliers in weight and activation distributions, boosting the performance of quantized LLMs.
:::

:::zh
**中文:** • 我们提出 SpinQuant，这是首个采用可学习旋转来缓解权重与激活分布中离群值的方法，从而提升量化 LLM 的性能。
:::

<a id="S015"></a>
**Source:** p.2 S015

:::en
**Original:** • We reveal that random rotations introduce substantial variance in quantized network performance. We propose optimizing rotation matrices within Stiefel manifold, directly minimizing the final loss of rotated quantized network. Ablation studies validate that our learned rotations consistently outperform random rotations, with improvements up to 16.2 points.
:::

:::zh
**中文:** • 我们揭示随机旋转会给量化网络性能带来显著方差。我们提出在 Stiefel 流形内优化旋转矩阵，直接最小化旋转后量化网络的最终损失。消融研究表明，我们学到的旋转始终优于随机旋转，提升最多达 16.2 分。
:::

<a id="S016"></a>
**Source:** p.2 S016

:::en
**Original:** • SpinQuant$_{no\ had}$ merges rotation matrices into pre-trained weights without altering the network architecture, significantly narrowing the W4A8KV8 quantization performance gap from 12.1 to 1.6 on the Mistral-7B model in zero-shot commonsense reasoning tasks. Noteworthily, SpinQuant$_{no\ had}$ W4A8 quantization achieves comparable performance as state-of-the-art weight only quantization methods like QuIP# (Tseng et al., 2024) and OminiQuant (Shao et al., 2023) on LLaMA-2.
:::

:::zh
**中文:** • SpinQuant$_{no\ had}$ 将旋转矩阵并入预训练权重而不改变网络架构，在 Mistral-7B 模型的零样本常识推理任务上，将 W4A8KV8 量化的性能差距从 12.1 显著缩小到 1.6。值得注意的是，SpinQuant$_{no\ had}$ 的 W4A8 量化在 LLaMA-2 上达到了与 QuIP# (Tseng et al., 2024)、OminiQuant (Shao et al., 2023) 等最先进权重仅量化方法相当的性能。
:::

<a id="S017"></a>
**Source:** p.2 S017

:::en
**Original:** • SpinQuant$_{had}$ attains an average accuracy of 64.0 in extreme W4A4KV4 quantization settings on LLaMA-2 7B. This represents a mere 2.9 point gap from the full-precision network, a substantial improvement over the previous LLM-QAT (Liu et al., 2023c) approach, which exhibited a 22.0 point gap under identical precision conditions.
:::

:::zh
**中文:** • SpinQuant$_{had}$ 在 LLaMA-2 7B 的极端 W4A4KV4 量化设置下取得 64.0 的平均准确率。这与全精度网络仅相差 2.9 分，相比先前 LLM-QAT (Liu et al., 2023c) 方法（在相同精度条件下差距为 22.0 分）是显著改进。
:::

<a id="F002"></a>
#### Fig. 2. Activation Distribution Before and After Rotation <span class="lang-zh">· 旋转前后的激活分布</span>

**Placed near:** p.1 S017  
**Source:** p.3 F002 · C002

<img src="/images/quantization/spinquant/figures/F002.png" alt="Figure 2" width="1473" height="366" loading="lazy" decoding="async">

<a id="C002"></a>

:::en
**Original caption:** Figure 2: Activation distribution in LLaMA-2 7B model before and after rotation. Outliers exist in particular channels before rotation. Since channel-wise quantization is not supported in most hardware, outlier removal using rotation enables accurate token-wise or tensor-wise quantization.
:::

:::zh
**中文图注:** 图 2：LLaMA-2 7B 模型旋转前后的激活分布。旋转前特定通道存在离群值。由于多数硬件不支持逐通道量化，用旋转去除离群值可使逐 token 或逐张量量化变得准确。
:::

:::zh
**Reading note:** 四个子图为 (a) 原始 MHA 输入、(b) 旋转后 MHSA 输入、(c) 原始 FFN 输入、(d) 旋转后 FFN 输入；注意纵轴刻度的变化。
:::

<a id="F003"></a>
#### Fig. 3. Outlier Measurement and Quantization Error <span class="lang-zh">· 离群值度量与量化误差</span>

**Placed near:** p.1 S017  
**Source:** p.3 F003 · C003

<img src="/images/quantization/spinquant/figures/F003.png" alt="Figure 3" width="1640" height="398" loading="lazy" decoding="async">

<a id="C003"></a>

:::en
**Original caption:** Figure 3: Outlier measurement and quantization error across input activation and weights in the five layers that take inputs from the residual (Q/K/V/Up/Gate-projection) of each block in the LLaMA-2 7B model. (a) After rotation, kurtosis of activation distributions is significantly reduced to approximately three across all layers. Quantization error is reduced after rotation in both (b) activations and (c) weights.
:::

:::zh
**中文图注:** 图 3：LLaMA-2 7B 模型中每个模块从残差取输入的五层（Q/K/V/Up/Gate-projection）的输入激活与权重上的离群值度量与量化误差。(a) 旋转后，所有层的激活分布峰度显著降至约 3。(b) 激活与 (c) 权重的量化误差在旋转后都减小。
:::

:::zh
**Reading note:** 左图 (a) 峰度随层数从 >200 降到 ≈3；(b)(c) 分别给出激活与权重的量化误差。
:::

<a id="S018"></a>
## 2. 动机 / Motivation

<a id="S019"></a>
**Source:** p.3 S019

:::en
**Original:** Quantization reduces the precision of weights (and/or activations) in a neural network in order to save memory and lower the latency. The quantization process can be formulated as:
:::

:::zh
**中文:** 量化通过降低神经网络中权重（和/或激活）的精度来节省内存、降低延迟。该量化过程可以写成：
:::

<a id="E001"></a>
**Source:** p.3 E001 · Eq. (1)

$$
X_Q = \alpha\left\lfloor \frac{X_R-\beta}{\alpha}\right\rceil+\beta
$$

<div align="right">(1)</div>

:::zh
**中文说明:** 对称/非对称量化的统一形式：$X_Q$ 为量化张量，$X_R$ 为 FP16 实值张量

*（原始裁切核对：[E001](/images/quantization/spinquant/equations/E001.png)）*
:::

<a id="S020"></a>
**Source:** p.3 S020

:::en
**Original:** where $\alpha = \frac{\max(|X_R|)}{2^{N-1}-1}$, $\beta = 0$ in symmetric quantization or $\alpha = \frac{\max(X_R)-\min(X_R)}{2^{N}-1}$, $\beta = \min(X_R)$
:::

:::zh
**中文:** 其中 $\alpha = \frac{\max(|X_R|)}{2^{N-1}-1}$、$\beta = 0$（对称量化），或 $\alpha = \frac{\max(X_R)-\min(X_R)}{2^{N}-1}$、$\beta = \min(X_R)$（非对称量化）
:::

<a id="S021"></a>
**Source:** p.3 S021

:::en
**Original:** in asymmetric quantization. Here XQ is a quantized tensor and XR is a real-valued FP16 tensor. N is number of bits. For Large language models (LLMs), the presence of outliers extends the range of weight/activation values and increases the reconstruction errors for normal values (Dettmers et al., 2022; Liu et al., 2023b; Yelysei Bondarenko, 2023) (Figures 2 (a)&(c)).
:::

:::zh
**中文:** 在非对称量化中，$X_Q$ 为量化后的张量，$X_R$ 为实值 FP16 张量，$N$ 为比特数。对于大语言模型（LLM），离群值的存在拉伸了权重/激活的数值范围，并增大了正常数值的重构误差 (Dettmers et al., 2022; Liu et al., 2023b; Yelysei Bondarenko, 2023)（Figures 2 (a)&(c)）。
:::

<a id="S022"></a>
### 2.1 离群值削减 / Outlier Reduction

<a id="S023"></a>
**Source:** p.3 S023

:::en
**Original:** There exist many ways to mitigate the effect of outliers (Xiao et al., 2022; Dettmers et al., 2022). In this paper, we propose to use optimized rotation to reduce outliers. Intuitively, a random rotation matrix statistically blends large and small weights together into a well-behaved distribution with fewer outliers (Elhage et al., 2023), and thus is easier to quantize.
:::

:::zh
**中文:** 缓解离群值影响的方法有很多 (Xiao et al., 2022; Dettmers et al., 2022)。本文提出用优化后的旋转来减少离群值。直观上，随机旋转矩阵会在统计意义上将大权重与小权重混合，形成离群值更少的良好分布 (Elhage et al., 2023)，因而更易量化。
:::

<a id="S024"></a>
**Source:** p.3 S024

:::en
**Original:** Figure 3 (a) illustrates the measurement of the Kurtosis κ of the activations before and after rotation. κ quantifies the “tailedness” of a real-valued random variable’s probability distribution. A larger κ indicates more outliers, while κ ≈ 3 suggests a Gaussian-like distribution. In Figure 3 (a), the activation distribution in the transformer contains numerous outliers, with κ of many layers exceeding 200. However, after multiplying these activations with a random rotation matrix, the κ across all layers becomes approximately 3, indicating a more Gaussian-shaped distribution that is easier to quantize. This is corroborated by Figure 3 (b), where the quantization error of the activation tensor significantly decreases after rotation.
:::

:::zh
**中文:** Figure 3 (a) 展示了旋转前后激活峰度 κ 的测量。κ 刻画实值随机变量概率分布的“尾部厚度”：κ 越大表明离群值越多，而 κ ≈ 3 提示接近高斯分布。在 Figure 3 (a) 中，Transformer 的激活分布包含大量离群值，许多层的 κ 超过 200。然而在将这些激活与随机旋转矩阵相乘后，所有层的 κ 都变为约 3，说明分布更接近高斯形，因而更易量化。Figure 3 (b) 印证了这一点：旋转后激活张量的量化误差显著下降。
:::

<a id="F004"></a>
#### Fig. 4. Performance Distribution Under Different Random Rotations <span class="lang-zh">· 不同随机旋转下的性能分布</span>

**Placed near:** p.3 S024  
**Source:** p.4 F004 · C004

<img src="/images/quantization/spinquant/figures/F004.png" alt="Figure 4" width="859" height="626" loading="lazy" decoding="async">

<a id="C004"></a>

:::en
**Original caption:** Figure 4: The performance distributions of W4A4 quantized LLaMA-2 7B under different random rotations, using network-level parameterization (Sec. 3.1). We compare the distributions using random floating-point rotations, random Hadamard matrices, and optimized rotation matrices with Cayley optimization (Sec. 3.2). Despite that Hadamard matrices mostly perform better than random rotations, both random groups demonstrate large variance. In contrast, by optimizing the rotation matrix with Cayley optimization (i.e., SpinQuant), the performance is improved significantly and the variance becomes much smaller.
:::

:::zh
**中文图注:** 图 4：在网络级参数化（第 3.1 节）下，W4A4 量化的 LLaMA-2 7B 在不同随机旋转下的性能分布。我们比较了随机浮点旋转、随机 Hadamard 矩阵，以及用 Cayley 优化（第 3.2 节）得到的优化旋转矩阵三种分布。尽管 Hadamard 矩阵总体上优于随机旋转，但两组随机方法的方差都很大。相反，用 Cayley 优化（即 SpinQuant）优化旋转矩阵后，性能显著提升且方差大幅减小。
:::

:::zh
**Reading note:** 小提琴图：随机旋转方差极大（最好与最差相差 13 分），Cayley 优化后的分布更集中且均值更高。
:::

:::zh
<a id="D001"></a>
### 推导 D1 · 旋转到底帮了量化什么忙

**插入位置：** p.3–4 · §2.1 末尾（紧随 S024 与图 4）

**性质：** 读者补充推导，非论文原文。这里只用最小例子讲思路；严格证明见 `DERIVATION_PACKAGE.md` §6

**一句话.** 旋转不改变这堆数的“总能量”，只把极端值摊到别的坐标上。而量化误差的大小由数值范围决定，所以摊平就等于提高精度。

**先看一个 2 维例子.** 取 $x=(10,0)$：范围是 $10-0=10$，量化步长必须按这个范围定。把它转 45°：

$$
(10,0)\cdot\frac{1}{\sqrt2}\begin{pmatrix}1&1\\-1&1\end{pmatrix}=(7.07,\,7.07),
$$

范围变成 $0$（这是极端情形，只说明方向），而长度没变：$\sqrt{10^2+0^2}=\sqrt{7.07^2+7.07^2}=10$。

**为什么长度一定不变.** 正交矩阵的定义就是 $R^{\top}R=I$，于是

$$
\|xR\|_2^2=xRR^{\top}x^{\top}=\|x\|_2^2,\qquad \lvert (xR)_j\rvert\le\|x\|_2 .
$$

第二个式子说：任何一个坐标都不会超过原来的长度。所以旋转没法让整体数值变小（长度守恒），只能把尖峰摊开——最理想是让 $d$ 个坐标各分到 $\|x\|_2/\sqrt d$。

**这和量化误差什么关系.** 对称量化把范围 $\rho$ 均分成 $2^b-1$ 格，步长 $\alpha=\rho/(2^b-1)$，取整误差最多 $\alpha/2$。所以**范围缩小 $g$ 倍，误差就缩小 $g$ 倍，相当于多出 $\log_2 g$ 个比特**。

**实测与上限.** 合成数据（$d=4096$，4 个坐标放大 25 倍）：范围 $36.2\to8.2$（缩小 4.4 倍，约 2.1 个比特），峰度 $121\to3.0$，长度恒为 $72.5$。理论上最多能缩到 $\sqrt d/2\approx32$ 倍，但真实激活只是少数坐标偏大，所以实际收益离上限很远——这就是论文 §2.1 画的那张图。
:::

<a id="S025"></a>
### 2.2 随机旋转带来巨大方差 / Random Rotations Produce Large Variance

<a id="S026"></a>
**Source:** p.4 S026

:::en
**Original:** Interestingly, while statistically random rotation leads to better quantization, not all random rotations give the same quantization outcome. To show this, we tested the zero-shot average accuracy of the rotated version of LLaMA-2 7B, quantized to 4-bit weight and 4-bit activation, under 100 randomized trials. As shown in Figure 4, the performance variance is substantial, with the best random rotation matrix outperforming the worst by 13 points. Random Hadamard matrices 1 outperform random rotation matrices, in consistent with the findings in (Tseng et al., 2024) that Hadamard matrices yield tighter bounds on weight maximal value. However, even random Hadamard rotation matrices exhibit a non-negligible variance in final performance, as large as 6 points.
:::

:::zh
**中文:** 有趣的是，尽管统计意义上随机旋转能带来更好的量化，但并非所有随机旋转都给出相同的量化结果。为说明这一点，我们在 100 次随机试验下测试了 LLaMA-2 7B 旋转版本（权重 4-bit、激活 4-bit）的零样本平均准确率。如 Figure 4 所示，性能方差相当大，最好的随机旋转矩阵比最差的高出 13 分。随机 Hadamard 矩阵 ¹ 优于随机旋转矩阵，这与 (Tseng et al., 2024) 的发现一致，即 Hadamard 矩阵对权重最大值的界更紧。然而即便使用随机 Hadamard 旋转矩阵，最终性能仍存在不可忽略的方差，高达 6 分。
:::

<a id="S027"></a>
**Source:** p.4 S027

:::en
**Original:** Given the huge variance across multiple trials of rotations, a natural question arises: Is it possible to optimize the rotation to maximize the benefit of quantization? We affirmatively answer this question by presenting a viable framework with quantization-oriented rotation learning that consistently achieves high accuracy across 7 models and 4 low-bit quantization settings.
:::

:::zh
**中文:** 鉴于多次旋转试验之间的巨大方差，一个自然的问题浮现出来：能否优化旋转以最大化量化收益？我们对这个问题给出肯定回答，提出了一个可行的、面向量化的旋转学习框架，在 7 个模型与 4 种低比特量化设置下都能稳定取得高准确率。
:::

<a id="S028"></a>
## 3. 方法 / Method

<a id="S029"></a>
**Source:** p.4 S029

:::en
**Original:** In this section, we introduce SpinQuant, a framework that integrates and optimizes rotations in LLMs targeting at quantization loss. We start with defining rotation parameterization of popular LLM architectures, which includes two mergeable rotation matrices (R1, R2) that produce rotationally invariant full-precision network, and two online Hadamard rotation (R3, R4) to further reduce the outliers for extreme activation and KV-cache quantization. Then, we present how to optimize these rotation matrices on Stiefel manifold with target loss.
:::

:::zh
**中文:** 本节介绍 SpinQuant——一个以量化损失为目标、在 LLM 中整合并优化旋转的框架。我们首先定义主流 LLM 架构的旋转参数化：其中包含两个可合并的旋转矩阵（$R_1$、$R_2$），它们产生旋转不变的全精度网络；以及两个在线 Hadamard 旋转（$R_3$、$R_4$），用于在极端激活与 KV-cache 量化下进一步削减离群值。随后，我们说明如何在 Stiefel 流形上以目标损失优化这些旋转矩阵。
:::

<a id="S030"></a>
### 3.1 旋转参数化 / Rotation Parameterization

<a id="S031"></a>
**Source:** p.4 S031

:::en
**Original:** Rotating activations in residual As shown in Figure 1(a), we rotate the activations in the residual path by multiplying the embedding output X with a random rotation matrix (R1). This rotation removes outliers and eases the quantization of the input activations to the fully-connected layers that read from the residual. To maintain numerical invariance, we reverse the rotation of the activation by
:::

:::zh
**中文:** **残差路径上的激活旋转**  如 Figure 1(a) 所示，我们将嵌入输出 $X$ 与随机旋转矩阵（$R_1$）相乘，从而旋转残差路径上的激活。该旋转消除离群值，并使读取残差的各全连接层的输入激活更易量化。为保持数值不变性，我们在激活进入包含非线性的注意力模块与前馈网络之前，用下式反向旋转：
:::

<a id="S032"></a>
**Source:** p.4 S032

:::en
**Original:** 1A Hadamard matrix H is a special type of rotation matrix, where the entries of the matrix are solely ±√n. Given a Hadamard matrix H, we can generate 2n different random Hadamard matrices by multiplying with S, a diagonal matrix with elements si randomly chosen from {−1, 1}.
:::

:::zh
**中文:** ¹Hadamard 矩阵 $H$ 是一类特殊的旋转矩阵，其元素仅为 $\pm\sqrt{n}$。给定一个 Hadamard 矩阵 $H$，通过与对角矩阵 $S$（其元素 $s_i$ 从 $\{-1, 1\}$ 中随机选取）相乘，可以生成 $2n$ 个不同的随机 Hadamard 矩阵。
:::

<a id="S033"></a>
**Source:** p.5 S033

:::en
**Original:** multiplying it with $R_1^{T}\ (=R_1^{-1})$ prior to its passage through the attention block and feed-forward network, which contains non-linearity. When the quantization is not present, the full-precision network remains intact no matter which rotation is applied.² The rotation matrices can be merged into corresponding weight matrices, as illustrated in Figures 1(b)&(c). After absorption, no new parameters are introduced in the network. We can now modify $R_1$ freely without impacting the floating-point network’s accuracy or parameter count.
:::

:::zh
**中文:** 将其乘以 $R_1^{T}\ (=R_1^{-1})$。当不存在量化时，无论施加何种旋转，全精度网络都保持完好。² 如 Figures 1(b)&(c) 所示，这些旋转矩阵可以并入对应的权重矩阵。吸收之后，网络中不引入任何新参数。因此我们现在可以自由修改 $R_1$，而不影响浮点网络的精度或参数数量。
:::

:::zh
<a id="D002"></a>
### 推导 D2 · S1（残差旋转）为什么等于没动

**插入位置：** p.5 · 紧随 S033

**性质：** 读者补充推导，非论文原文。严格版见 `DERIVATION_PACKAGE.md` §0–§1

**一句话.** RMSNorm 只关心“向量有多长”，旋转不改变长度；所以“先归一化再旋转”和“先旋转再归一化”是一回事。把整条残差流乘上 $R_1$，再在每个模块入口把 $R_1$ 乘回去，网络输出就一点没变。

**第一步：归一化只看长度.**（RMSNorm 的定义与手算见 [§0.5](#N001)）$s(x)=\sqrt{\tfrac1d\|x\|_2^2+\epsilon}$ 只用到长度，而长度对旋转免疫，于是

$$
\frac{xR_1}{s(xR_1)}=\frac{x}{s(x)}\,R_1 .
$$

**第二步：把 $\gamma$ 挪进权重.** 真实 RMSNorm 还带一个逐元素缩放 $\gamma$，它是“按坐标缩放”，跟“跨坐标搬运”的旋转不能换顺序——这也是很多人推不下去的地方。但

$$
\big(\mathrm{Norm}(x)\odot\gamma\big)W=\mathrm{Norm}(x)\big(\mathrm{diag}(\gamma)W\big),
$$

说明 $\gamma$ 可以提前乘进下一层权重，网络行为完全不变（这是改写，不是近似）。挪完之后归一化层就不带 $\gamma$ 了，第一步可以直接用。

**第三步：整条残差流一起转.**（残差流是什么、为什么能整条转，见 [§0.6](#N002)）让每个残差状态都写成 $hR_1$：

- **读残差的层**（$W_q,W_k,W_v,W_{up},W_{gate}$）：把 $R_1^{\top}$ 并到权重左边，输入乘上去正好抵消（$R_1R_1^{\top}=I$）。这就是 Figure 1 里那个 $R_1^{-1}$ 方框。
- **写残差的层**（$W_o,W_{down},W_{lm}$）：输出再乘一次 $R_1$，让残差继续待在旋转后的坐标系里；嵌入表 $E$ 也乘 $R_1$，它是残差的起点。
- **非线性算子**（softmax、SiLU）：一个都不碰，它们拿到的已经是没旋转过的数。

**为什么能一路推下去.** 第 0 层（嵌入）已经转了；假设第 $\ell$ 层还成立，第 $\ell+1$ 层的输入是“残差 + 模块输出”，两部分在同一坐标系里，加法是线性的，所以依然成立。逐层传下去，最后 logits 一模一样。

**一句提醒.** 撤销必须发生在进入模块**之前**。若让注意力直接吃旋转过的输入，分数会变成 $(uR_1W_q)(uR_1W_k)^{\top}\neq uW_qW_k^{\top}u^{\top}$，softmax 之后就不是原来的数了。
:::

<a id="S034"></a>
**Source:** p.5 S034

:::en
**Original:** **Rotating activations in the attention block** As depicted in Figure 1(b), in the attention block, we propose to rotate the value matrix by multiplying $R_2$, and the activations to out-projection layer by $R_2^{T}$ head-wisely. $R_2$ has the shape of $(D_{head}, D_{head})$ and can be independently chosen across layers. The numerical in-variance is illustrated in Figure 5, these two rotations can be offset in a full-precision network since there are no operators between $R_2$ and $R_2^{T}$. Meanwhile, it can improve quantization for value cache and input activations to out-projection layer without introducing any new parameters in the network.
:::

:::zh
**中文:** **注意力模块内的激活旋转**  如 Figure 1(b) 所示，在注意力模块中，我们提出将 value 矩阵乘以 $R_2$，并按 head 将进入 out-projection 层的激活乘以 $R_2^{T}$。$R_2$ 的形状为 $(D_{head}, D_{head})$，可以在各层独立选择。Figure 5 说明了其数值不变性：由于 $R_2$ 与 $R_2^{T}$ 之间没有算子，这两个旋转在全精度网络中可以相互抵消。同时，它还能改善 value cache 与进入 out-projection 层输入激活的量化，且不在网络中引入任何新参数。
:::

<a id="F005"></a>
#### Fig. 5. Rotation Equivalence in Multi-Head Self-Attention <span class="lang-zh">· 多头自注意力中的旋转等价性</span>

**Placed near:** p.4 S034  
**Source:** p.5 F005

<img src="/images/quantization/spinquant/figures/F005.png" alt="Figure 5" width="891" height="112" loading="lazy" decoding="async">

:::zh
**Reading note:** 一行图示：$\mathrm{out}=\mathrm{Attn}\cdot X\cdot (W_v R_2)(R_2^{-1}W_o)$，即 $R_2$ 与 $R_2^{T}$ 相互抵消。
:::

:::zh
<a id="D003"></a>
### 推导 D3 · S2（V 上的旋转）为什么能穿过注意力

**插入位置：** p.5 · 紧随图 5

**性质：** 读者补充推导，非论文原文。严格版见 `DERIVATION_PACKAGE.md` §2

**一句话.** 注意力做的是“按 token 加权平均”，$R_2$ 做的是“在 head 内部重新组合特征”。一个动行、一个动列，互不干扰。

**手算一个 $2\times2$ 例子.** 两个 token、head 两维：权重 $A=\begin{pmatrix}0.6&0.4\\0.2&0.8\end{pmatrix}$，value $X=\begin{pmatrix}1&2\\3&4\end{pmatrix}$，取 $R_2=\begin{pmatrix}0&-1\\1&0\end{pmatrix}$（转 90°）。

- 先转再平均：$XR_2=\begin{pmatrix}2&-1\\4&-3\end{pmatrix}$，$A(XR_2)=\begin{pmatrix}2.8&-1.8\\3.6&-2.6\end{pmatrix}$。
- 先平均再转：$AX=\begin{pmatrix}1.8&2.8\\2.6&3.6\end{pmatrix}$，$(AX)R_2=\begin{pmatrix}2.8&-1.8\\3.6&-2.6\end{pmatrix}$。

结果相同，因为 $A$ 只管行（token）、$R_2$ 只管列（head 内坐标），可以交换顺序：

$$
A(XR_2)=(AX)R_2 .
$$

**代入 S2.** 把 value 先转好再进 cache，出来时在 $W_o$ 左边补一个 $R_2^{\top}$：

$$
\big(A^{(h)}XW_v^{(h)}R_2\big)R_2^{\top}W_o^{(h)}=A^{(h)}XW_v^{(h)}W_o^{(h)},
$$

$R_2R_2^{\top}=I$，精确抵消。

**两点要知道.** 一是 $q,k$ 没被动过，所以注意力权重逐位不变，只有 value 与 $W_o$ 的输入被改写；二是这两个位置正好对应两个量化点——value cache 与 out-projection 的输入。
:::

<a id="S035"></a>
**Source:** p.5 S035

:::en
**Original:** We denote the method with only $R_1$ and $R_2$ inserted and optimized as SpinQuant$_{no\ had}$, which can readily achieve significant accuracy improvement than previous quantization methods, and closing the gap between W4A8 quantized LLMs and their full-precision counterparts to 0.1 − 2.5 points on zero-shot commonsense reasoning averaged accuracy.
:::

:::zh
**中文:** 我们将仅插入并优化 $R_1$ 与 $R_2$ 的方法记为 SpinQuant$_{no\ had}$，它能比先前量化方法轻松取得显著的精度提升，在零样本常识推理平均准确率上把 W4A8 量化 LLM 与全精度对应模型的差距缩小到 0.1 − 2.5 分。
:::

<a id="S036"></a>
**Source:** p.5 S036

:::en
**Original:** **Additional unabsorbed rotations** To further enhance outlier suppression for lower-bit (e.g. 4-bit) activation quantization, we incorporate a Hadamard matrix multiplication ($R_4$ in Figure 1(c)) inside the feed-forward block, reducing the outliers in the input to the down projection layer, similar to (Tseng et al., 2024; Ashkboos et al., 2023b). Hadamard rotation can be computed with fast hadamard transform and introduce marginal overhead to the inference latency. Similarly, Hadamard matrix ($R_3$ in Figure 1(b)) can be inserted when low-bit KV cache quantization is required. We denote the resulting method, equipped with all rotations, as SpinQuant$_{had}$. Next, we demonstrate how to jointly optimize these rotations.
:::

:::zh
**中文:** **额外的未吸收旋转**  为进一步增强低比特（如 4-bit）激活量化下的离群值抑制，我们在前馈模块内部引入 Hadamard 矩阵乘法（Figure 1(c) 中的 $R_4$），以减少进入 down projection 层的输入离群值，类似 (Tseng et al., 2024; Ashkboos et al., 2023b)。Hadamard 旋转可用快速 Hadamard 变换计算，对推理延迟只带来边缘开销。类似地，当需要低比特 KV cache 量化时，可以插入 Hadamard 矩阵（Figure 1(b) 中的 $R_3$）。我们将装备了全部旋转的最终方法记为 SpinQuant$_{had}$。接下来我们说明如何联合优化这些旋转。
:::

:::zh
<a id="D004"></a>
### 推导 D4 · S3 和 S4 为什么必须“在线”，在线到底在算哪一步

**插入位置：** p.5 · 紧随 S036

**性质：** 读者补充推导，非论文原文。严格版见 `DERIVATION_PACKAGE.md` §3–§4

S1/S2 能吸收，是因为它们的“撤销”就是一个权重矩阵。S3/S4 不行，原因各不相同。

**① S3：点积不怕“两边一起转”.** 注意力分数是 $q$ 与 $k$ 的点积；用同一个旋转矩阵转两个向量，长度和夹角都不变，点积自然不变：

$$
(qH)\cdot(kH)=q\cdot k .
$$

所以 K 可以先转好再存进 cache（转过的数更好量化），q 也照样转一下，分数完全一致——压根不需要“撤销”。

**为什么不能提前并进 $W_q,W_k$.** 因为 RoPE 夹在中间（RoPE 是什么见 [§0.7](#N003)）：RoPE 按 token 位置转一个角度，$H$ 把坐标混一混，两件事顺序不能换。用 $2\times2$ 的 Hadamard 和 30° 旋转试一下：$H\Theta$ 的第一行是 $(0.97,\,0.26)$，$\Theta H$ 的第一行是 $(0.26,\,0.97)$，不相等。所以 $H$ 只能放在 RoPE 之后算。

**② S4：逐元素乘法不认“混合坐标”.** FFN 里 $z=\mathrm{SiLU}(a)\odot b$，逐元素相乘只看“同一位置上的两个数”。取 $a=(1,0)$、$b=(0,1)$，则 $a\odot b=(0,0)$；把两个向量都转 45° 得 $a'=(0.71,0.71)$、$b'=(-0.71,0.71)$，而 $a'\odot b'=(-0.5,0.5)\neq(0,0)$。所以任何“混合坐标”的旋转都过不去；唯一能过的是“只换位置”的置换，但置换对离群值没有任何帮助。

**于是只能放在乘法之后.** 旋转施加在 $z$ 上，再由下一层线性变换转回来：

$$
(zH_4)\big(H_4^{\top}W_{down}\big)=zW_{down},\qquad W_{down}\mapsto H_4^{\top}W_{down}.
$$

线性层最“听话”，左乘右乘都可以，所以 $H_4^{\top}$ 直接进 $W_{down}$。

#### “在线”到底在线在哪

先把两个词分清楚：**离线**指的是权重——模型导出时就能算好、直接塞进矩阵里的东西；**在线**指的是数据——每次前向都得按当前的 token 现算的东西。S1/S2 转的是权重，所以全部离线；S3/S4 转的是激活，只能在线。

**S3 的一次前向（每个 token、每个 head）：**

1. $q=uW_q$、$k=uW_k$，各过一遍 RoPE；
2. **现算** $q'=qH_3$、$k'=kH_3$ ← 这就是“在线”的那一步；
3. $k'$ 量化后写进 KV cache；
4. 分数 $=q'k'^{\top}$，与不旋转时完全一样。

**S4 的一次前向：**

1. $z=\mathrm{SiLU}(uW_{gate})\odot(uW_{up})$；
2. **现算** $z'=zH_4$ ← 在线的那一步；
3. $z'$ 量化，再乘上已经吸好 $H_4^{\top}$ 的 $W_{down}$。

注意这两条流程里都**没有“转回来”的动作**：逆转角早就进了权重，运行时只付一次“转过去”的代价。相比之下 S1 的旋转进了嵌入表和权重、S2 的进了 $W_v$ 与 $W_o$，推理时连一次乘法都不多做——这就是“可吸收”与“必须在线”的实际差别。

#### 为什么这一步便宜

Hadamard 的元素只有 $\pm 1/\sqrt d$（它是怎么构造的、为什么能只做加减，见 [§0.8](#N004)），所以乘它不需要 $d^{2}$ 次乘法，可以用快速变换做到 $d\log_2 d$ 次加减。以 LLaMA-2 7B 为例：

| 在线旋转 | 维度 | 直接乘矩阵 | 快速变换 |
|---|---|---|---|
| S3（每个 head 的 $q$ 与 $k$） | 128 | $128^{2}\approx1.6$ 万次乘加 | $128\times7=896$ 次加减 |
| S4（FFN 中间激活） | 11008（按 128 分块） | $11008^{2}\approx1.2$ 亿次 | $86\times896\approx7.7$ 万次 |

注意 S3 的 896 是**单个 head** 的：32 个 head、$q$ 与 $k$ 各转一次，合计约 $5.7$ 万次；而一次 $4096\times4096$ 的矩阵乘就要 $1670$ 万次乘加，前者不到后者的 $0.4\%$。S4 那边是 $7.7$ 万对 $4500$ 万，约 $0.17\%$。所以论文实测两个在线旋转合计只多约 8% 延迟（Table 6/14）——开销主要来自访存与量化，而不是这点加减。

#### 一个常见误解：$H_3$ 固定，不等于可以提前算好

$H_3$ 确实是**固定**的：它是 Hadamard 矩阵（或它乘一个随机的 $\pm1$ 对角阵），不参与训练，也不是学出来的。但“固定”只说明矩阵本身不变，**被它乘的那个向量每来一个 token 都是新的**：

- $q'=qH_3$ 里的 $q$ 是当前 token 过完 $W_q$ 与 RoPE 之后的结果，不是常量；
- $z'=zH_4$ 里的 $z$ 依赖当前输入。

所以这次乘法只能在前向时做。“提前算好”的唯一办法是把它挪进某个权重矩阵———而那正是前面被 RoPE 与 $\odot$ 挡住的事情，与 $H_3$ 是否固定毫无关系。

顺带把两件容易混的事分开：

| | 要训练吗 | 能并进权重吗 | 推理时要现算吗 |
|---|---|---|---|
| $R_1$（残差） | 要 | 能 | 不用 |
| $R_2$（V） | 要 | 能 | 不用 |
| $R_3=H_3$（q/k） | 不要，固定 | 不能 | 要，每个 token 一次 |
| $R_4=H_4$（FFN） | 不要，固定 | 不能 | 要，每个 token 一次 |

也就是说，**“可学习”和“可吸收”是两件独立的事**：$R_1,R_2$ 可学又可吸收，$R_3,R_4$ 不学但也不可吸收。能否吸收取决于“它乘的是权重还是激活”，不取决于它是不是学出来的。

固定反而有好处：矩阵已知，就能把它写死成“若干次加减”的快速变换、直接融进量化 kernel，不必从显存里读一个 $128\times128$ 的矩阵。论文能把这份开销压到约 8%，靠的就是这一点。

#### 在 4-bit 下，这一步的实际顺序

关键是**先转、后量化**：量化步长必须按“转过之后的”张量来定；如果先量化再转，就等于用还没摊平的范围定了步长，收益白丢。

**S3（KV cache 4-bit）：**

1. $u=\mathrm{Norm}(h)$，$k=uW_k$——权重是 4-bit，取用时反量化，累加结果仍是 FP16；
2. 过 RoPE，仍是 FP16；
3. **在线转**：$k'=kH_3$，FP16，只是加减；
4. **量化**：按 token（每个 head 的 128 维）取 min-max，得到 4-bit 整数加一个缩放因子，写进 cache；
5. 读 cache 时反量化，分数用 $q'=qH_3$ 与 $k'$ 算点积——$q$ 不进 cache，通常保持高精度。

同一个 head 里，写 cache 的 $k$ 与查询的 $q$ 必须用**同一个** $H_3$，否则点积对不上。

**S4（激活 4-bit）：**

1. $z=\mathrm{SiLU}(uW_{gate})\odot(uW_{up})$，FP16；
2. **在线转**：$z'=zH_4$，FP16，按 128 分块做加减；
3. **量化**：按 token 取 min-max 到 4-bit（这一步本来就是 A4 的量化点）；
4. 反量化后乘 $W_{down}$（4-bit，$H_4^{\top}$ 已折进去）。

两处都可以把“转 + 量化”写进同一个 kernel：读进 FP16 激活、做几次加减、直接写 int4 与缩放因子，不多一次显存往返。

**顺带解释一个数字**：前面按 FP16 矩阵乘估算，旋转只占 $0.3\%$ 量级；但到 4-bit 时矩阵乘快了约 3 倍（Table 6 的端到端加速），分母变小，旋转的相对占比会涨到 $1\%$ 量级。剩下的差距来自量化与反量化的访存和 kernel 开销——所以 $8\%$ 是“转 + 量化”整套管线的价格，不能只算在几次加减头上。

#### S3/S4 到底有没有必要

先看收益：Table 1 里把 $H_3,H_4$ 打开与关掉的对比（0-shot 平均准确率，越高越好）：

| 模型 | 配置（W-A-KV） | no had | had | 差值 |
|---|---|---|---|---|
| LLaMA-2 7B | W4A8（4-8-8） | 65.8 | 65.8 | $0.0$ |
| LLaMA-2 7B | W4A4（4-4-16） | 57.0 | 64.1 | $+7.1$ |
| LLaMA-2 7B | W4A4KV4（4-4-4） | 56.0 | 64.0 | $+8.0$ |
| LLaMA-3 8B | W4A8（4-8-16） | 68.6 | 68.4 | $-0.2$ |
| LLaMA-3 8B | W4A4（4-4-16） | 51.9 | 65.8 | $+13.9$ |

所以必要性是**有条件的**：只做 W4A8 时，$H_3,H_4$ 基本等于白花 8% 延迟，论文自己也说这种场合用 SpinQuant$_{no\ had}$ 就够了；一旦把激活或 KV cache 压到 4-bit，它们值 $7\sim14$ 分。

**为什么没有便宜的替代品.** S1/S2 的旋转能“藏”进上游的线性层（嵌入表、$W_v$），所以免费；而 $H_3$ 要转的 $k$ 上游是 RoPE（按位置变，折不进权重），下游是点积（没有权重可折）；$H_4$ 要转的 $z$ 上游是逐元素乘法（非线性，折不进）。这两个位置**结构上没有地方可折**，只能现场算——这才是它们存在的理由，而不是“它比别的方法更优”。

一句诚实的补充：正因为不优化，$H$ 的符号变体、分块大小都没被搜过，这块留白本身是可以继续做的地方。

#### 能不能用 Cayley 学一个更好的 $R_3$

$R_1,R_2$ 是学出来的，那 $R_3$ 为什么不也学一个？把例 3 放大到真实规模量一次：$d=128$，512 个 token，KV cache 走 4-bit 对称 per-token absmax，$k$ 留 3 个固定的离群通道（真实 LLM 的激活就是这样），比的是注意力分数的相对误差。换成论文式 (1) 的非对称 min-max 结论不变：不转 $0.270$、Hadamard $0.107$、Cayley 从 $H$ 出发 $0.105$；两种约定下 $H$ 与 Cayley 的差距都在 $\pm2\%$ 噪声带内。量化器约定与峰值下界的关系见 §0.8「这个下界是对“对称量化”说的」。

| $R$ | 训练? | 训练集 | 同分布 | 离群通道漂移 | 峰值因子 |
|---|---|---|---|---|---|
| 不转 | 否 | 0.402 | 0.392 | 0.406 | 7.56 |
| 随机正交（200 次取最好） | 否 | 0.138 | 0.129 | 0.149 | 2.66 |
| Hadamard | 否 | 0.119 | 0.118 | 0.125 | 2.45 |
| Cayley（从 $H$ 出发，800 步） | 是 | 0.104 | 0.123 | 0.129 | 2.45 |

旋转本身的价值是压倒性的：$0.392\to0.118$，三倍多的改善。但在 $H$ 之上再学 800 步，训练集从 $0.119$ 降到 $0.104$，**独立数据反而变差**——它记住的是这 512 个样本各自的取整格点，不是分布。

原因是 4-bit 的误差由 $\max_i|z_i|$ 决定，而任何正交矩阵都满足 $\max_i|z_i|\ge\lVert z\rVert_2/\sqrt d$（峰值因子 $\ge 1$）。Hadamard 把尖峰摊到全维、符号错开，峰值因子已经压到 $2.45$，随机正交是 $2.66$。换个目标直接对峰值求导，$H$ 处的梯度比单位阵处小近 7 个数量级——它已经是驻点。在 $H$ 周围撒 600 个随机小旋转，有 179 个能让手头这份数据好最多 $2.6\%$，可换一份独立数据就全部打回原样。

所以结论不是“$H$ 最优”，而是“**单层 S3 上它已经贴着天花板**”：这一点点上没有可学的结构，学习预算花在 $R_1,R_2$ 上（那里有 $W_{out}$、残差、多层累积可以利用）回报高得多。顺带一个实现细节：$H_{128}$ 有特征值 $-1$（重数 64），而 Cayley 参数化 $R=(I-A)(I+A)^{-1}$ 取不到 $-1$，所以“从 $H$ 出发”要写成 $R=H\cdot\mathrm{Cayley}(A)$，$A=0$ 就正好是 $H$。

脚本：`compare_s3_cayley_vs_hadamard.py`，约 20 秒，含梯度校验与上述全部诊断。

**小结.** S3 靠“两边一起转”免费抵消，S4 靠“转完立刻由下一层转回来”；两者的共同点是**转的是激活而不是权重**，所以每次前向都要现算一次——这就是“在线”的含义。
:::

:::zh
<a id="D005"></a>
### 推导 D5 · 四段合起来

**插入位置：** p.5 · §3.1 末尾

**性质：** 读者补充推导，非论文原文。严格版见 `DERIVATION_PACKAGE.md` §5

四个插入点互不重叠：S1 在残差口，S2 在 V 与 $W_o$ 之间，S3 在 RoPE 之后，S4 在乘法之后。所以可以像搭积木一样逐个装上去，各管一段，互不影响——整网的浮点输出逐点不变，参数个数也不变，只有 S3/S4 各多一次在线计算。

| 阶段 | 转什么 | 什么时候转回来 | 塞进哪个矩阵 | 哪个张量因此更好量化 |
|---|---|---|---|---|
| S1 | 整条残差流 | 进模块之前 | 读残差的 5 个权重左乘 $R_1^{\top}$；$W_o,W_{down},W_{lm}$ 右乘 $R_1$ | 读残差的输入激活 |
| S2 | 每个 head 的 value | 进 $W_o$ 之前 | $W_v$ 右乘 $R_2$；$W_o$ 左乘 $R_2^{\top}$ | value cache、$W_o$ 的输入 |
| S3 | 每个 head 的 $q,k$ | 不用转回来 | 无，在线算子 | K cache、$q,k$ 激活 |
| S4 | FFN 中间结果 | 进 $W_{down}$ 之前 | $W_{down}$ 左乘 $H_4^{\top}$ | $W_{down}$ 的输入 |

在一个小模型（3 层、RoPE、SwiGLU）上依次叠加四段，logits 与全精度版本的差距是 $1.2\times10^{-14}$、$1.1\times10^{-14}$、$1.4\times10^{-14}$、$1.4\times10^{-14}$——都是浮点舍入的量级，等于没变。
:::

<a id="S037"></a>
### 3.2 Cayley 优化的旋转 / Cayley-Optimized Rotation

<a id="S038"></a>
**Source:** p.5 S038

:::en
**Original:** As illustrated in Figure 1, we have determined that the incorporation of four rotation matrices ($R_1$, $R_2$, $R_3$, $R_4$) can improve quantization performance while preserving numerical consistency in a full-precision network. Given that $R_3$ and $R_4$ are online rotation operations, meaning they cannot be absorbed into the weight matrix, we retain them as Hadamard matrices. This is because online Hadamard transforms can be efficiently implemented without significant overhead. We then define the optimization objective as identifying the optimal rotation matrix $R_1$ and $R_2$ that minimizes the final loss of the quantized network:
:::

:::zh
**中文:** 如 Figure 1 所示，我们已确定引入四个旋转矩阵（$R_1$、$R_2$、$R_3$、$R_4$）可以在保持全精度网络数值一致性的同时提升量化性能。鉴于 $R_3$ 与 $R_4$ 是在线旋转操作，即无法并入权重矩阵，我们将它们保留为 Hadamard 矩阵，因为在线 Hadamard 变换可以高效实现且没有显著开销。随后我们将优化目标定义为：寻找使量化网络最终损失最小的最优旋转矩阵 $R_1$ 与 $R_2$：
:::

<a id="E002"></a>
**Source:** p.5 E002 · Eq. (2)

$$
\arg\min_{R\in\mathcal{M}}\ \mathcal{L}_Q(R_1,R_2\mid W,X)
$$

<div align="right">(2)</div>

:::zh
**中文说明:** 在 Stiefel 流形上寻找使量化网络损失最小的旋转 $R_1,R_2$

*（原始裁切核对：[E002](/images/quantization/spinquant/equations/E002.png)）*
:::

<a id="S039"></a>
**Source:** p.5 S039

:::en
**Original:** Here, $\mathcal{M}$ represents the *Stiefel* manifold i.e., the set of all orthonormal matrices. $\mathcal{L}_Q(\cdot)$ denotes the task loss, such as cross-entropy, on the calibration set. It is a function of $\{R_1, R_2\}$, given the fixed pretrained weights $W$ and the input tensor $X$ and with the quantization function $Q$ in the network. To optimize the rotation matrix on the *Stiefel* manifold, we employ the *Cayley SGD* method (Li et al., 2020), which is an efficient optimization algorithm on the *Stiefel* manifold. More specifically, in each iteration, the update of the rotation $R$ is parameterized as the following:
:::

:::zh
**中文:** 这里 $\mathcal{M}$ 表示 *Stiefel* 流形，即所有正交矩阵的集合。$\mathcal{L}_Q(\cdot)$ 表示校准集上的任务损失（如交叉熵）。在固定预训练权重 $W$ 与输入张量 $X$、且网络中含有量化函数 $Q$ 的情况下，它是 $\{R_1, R_2\}$ 的函数。为了在 *Stiefel* 流形上优化旋转矩阵，我们采用 *Cayley SGD* 方法 (Li et al., 2020)，这是一种在 *Stiefel* 流形上高效的优化算法。具体而言，每次迭代中旋转 $R$ 的更新参数化为：
:::

<a id="E003"></a>
**Source:** p.5 E003 · Eq. (3)

$$
R' = \Delta R(Y)R := \left(I-\frac{\alpha}{2}Y\right)^{-1}\left(I+\frac{\alpha}{2}Y\right)R
$$

<div align="right">(3)</div>

:::zh
**中文说明:** Cayley 变换给出的旋转更新，保证 $R$ 始终正交

*（原始裁切核对：[E003](/images/quantization/spinquant/equations/E003.png)）*
:::

<a id="S040"></a>
**Source:** p.5 S040

:::en
**Original:** 2In a pre-norm LLM like LLaMA (Touvron et al., 2023a), we can convert a transformer network into a rotation-invariant network by incorporating the RMSNorm scale parameters α into the weight matrix right after the RMSNorm layer (Ashkboos et al., 2023a).
:::

:::zh
**中文:** ²在类似 LLaMA (Touvron et al., 2023a) 的 pre-norm LLM 中，我们可以把 RMSNorm 的缩放参数 α 并入紧随 RMSNorm 层之后的权重矩阵，从而将 Transformer 网络转换为旋转不变网络 (Ashkboos et al., 2023a)。
:::

<a id="S041"></a>
**Source:** p.6 S041

:::en
**Original:** where ∆R(Y ) := (I − α2 Y )−1(I + α2 Y ) is the Cayley Transform of a skew-symmetric matrix Y (i.e., Y ⊤ = −Y ). Y is computed from a projection Gˆ of the gradient G := ∇RLQ of the loss function:
:::

:::zh
**中文:** 其中 $\Delta R(Y) := (I - \frac{\alpha}{2} Y)^{-1}(I + \frac{\alpha}{2} Y)$ 是反对称矩阵 $Y$（即 $Y^{\top} = -Y$）的 Cayley 变换。$Y$ 由损失函数梯度 $G := \nabla_R \mathcal{L}_Q$ 的投影 $\hat{G}$ 计算得到：
:::

<a id="E004"></a>
**Source:** p.6 E004 · Eq. (4)

$$
Y=\hat{G}-\hat{G}^{\top},\qquad \hat{G}:=GR^{\top}-\frac{1}{2}RR^{\top}GR^{\top}
$$

<div align="right">(4)</div>

:::zh
**中文说明:** 由损失梯度 $G:=\nabla_R\mathcal{L}_Q$ 构造的反对称矩阵 $Y$

*（原始裁切核对：[E004](/images/quantization/spinquant/equations/E004.png)）*
:::

<a id="S042"></a>
**Source:** p.6 S042

:::en
**Original:** It can be shown that ∆R(Y ) is always orthonormal and thus R′ is guaranteed to be orthonormal (R′⊤R′ = I) if R is orthonormal. While Eqn. 3 requires a matrix inverse, the new rotation matrix R′ can be computed via an efficient fixed point iteration (Li et al., 2020). Overall, the approach maintains the property of orthonormality with only ∼2 times the computation time per iteration compared to a naive SGD algorithm.
:::

:::zh
**中文:** 可以证明 $\Delta R(Y)$ 始终是正交的，因此只要 $R$ 正交，$R'$ 就保证正交（$R'^{\top}R' = I$）。虽然式 3 需要矩阵求逆，但新的旋转矩阵 $R'$ 可以通过高效的定点迭代计算 (Li et al., 2020)。总体而言，与朴素 SGD 算法相比，该方法在每次迭代仅需约 2 倍计算时间即可保持正交性。
:::

:::zh
<a id="D006"></a>
### 推导 D6 · Cayley 更新为什么还是旋转，以及 $R_1,R_2$ 怎么算出来

**插入位置：** p.6 · 紧随 S042

**性质：** 读者补充推导，非论文原文。严格版见 `DERIVATION_PACKAGE.md` §7

**一句话.** 式 (3) 那个看着复杂的更新，其实是用矩阵算出来的一个旋转：不管梯度给什么，乘完还是正交矩阵，所以 $R$ 永远不会“跑出”正交矩阵的集合。

**用 $2\times2$ 算一遍.** 取 $Y=\begin{pmatrix}0&-1\\1&0\end{pmatrix}$（反对称）、$\alpha=1$，则

$$
\Delta R=\Big(I-\tfrac{1}{2}Y\Big)^{-1}\Big(I+\tfrac{1}{2}Y\Big)
=\begin{pmatrix}0.6&-0.8\\0.8&0.6\end{pmatrix}.
$$

右边正好是一个旋转矩阵（$\cos\theta=0.6$，$\sin\theta=0.8$，即转 $53.13^\circ$），因为它的两列长度都是 1、且互相垂直——这就是正交。你可以把它和它的转置相乘，会得到单位矩阵。

**为什么一般也对.** 关键在于 $Y$ 被刻意做成反对称（$Y^{\top}=-Y$）：这样 $I+\tfrac{\alpha}{2}Y$ 的转置等于 $I-\tfrac{\alpha}{2}Y$。转置、相乘、分子分母互相抵消，最后只剩 $I$。所以 $\Delta R$ 一定正交，而正交矩阵乘正交矩阵还是正交，$R'=\Delta R\cdot R$ 就保住了正交性。

**式 (4) 在做什么.** 它把梯度 $G$ 变成反对称的 $Y$（取 $Y=\hat G-\hat G^{\top}$ 这一步就是在“取反对称部分”），因为只有反对称矩阵才能生成旋转。至于 $\hat G$ 里那个 $-\tfrac12RR^{\top}GR^{\top}$，在 $RR^{\top}=I$ 时不起作用，可以直接忽略。

#### 四个矩阵的来源不一样

先把“谁要训练、谁直接构造”分清：

| | 怎么得到 | 要训练吗 |
|---|---|---|
| $R_1$（残差） | 随机 Hadamard 初始化，再用 Cayley SGD 优化 100 步 | 要 |
| $R_2$（V，每层一个） | 同上 | 要 |
| $H_3$（q/k） | 直接构造 Hadamard（可乘一个随机 $\pm1$ 对角阵） | 不要 |
| $H_4$（FFN） | 同上，$d_{ff}$ 按 128 分块 | 不要 |

$H_3,H_4$ 是“算出来”的，但只是把 $\pm1$ 按 Sylvester 规则排好（见 §0.8），不涉及任何优化。真正被“训练出来”的只有 $R_1,R_2$。

顺带一提，Hadamard 还兼任 $R_1,R_2$ 的**初始化**。Table 4 显示：优化之前，Hadamard 初始化就比随机浮点旋转好；优化之后两者差距缩小（§4.3.3）。这说明固定的 Hadamard 本身就是一个很强的默认值，不是权宜之计——它不需要知道数据长什么样，就能把能量摊开。

#### $R_1,R_2$ 的一轮迭代

论文 §3.2 的式 (2)–(5) 就是这一轮循环，展开成六步：

1. **冻结全部权重**：$W$ 不动，只留 $R_1,R_2$ 可训练（合计约占权重规模的 $0.26\%$）。
2. **前向**：把旋转插进网络，跑 800 条 WikiText-2 校准样本；激活（与权重，视设置而定）照常量化，量化器用直通估计——反向时把它当作“什么都不做”，梯度直接穿过去。
3. **算损失**：校准集上的交叉熵 $L_Q$。
4. **反向**：得到 $\partial L/\partial R$；单层形式就是式 (5)。
5. **投影**：$Y=\hat G-\hat G^{\top}$（式 4），把梯度变成反对称矩阵——只有反对称矩阵才能“生成旋转”。
6. **更新**：$R\leftarrow\big(I-\tfrac{\alpha}{2}Y\big)^{-1}\big(I+\tfrac{\alpha}{2}Y\big)R$（式 3），一步保证 $R$ 仍然正交。

重复 100 次，学习率从 1.5 线性衰减到 0。**为什么不直接 $R\leftarrow R-\eta G$**？因为减完就不再正交，得每一步额外做一次 SVD 或 QR 重新正交化；Cayley 这一形式自带正交性，省掉那一步。

#### 为什么非要“保正交”的更新

先分清一件事：需要 Cayley 的不是 $R_1,R_2$ 这两个身份，而是“**正在被优化的正交矩阵**”这个身份。$H_3,H_4$ 不参与优化，所以根本用不上它；哪天要学 $R_3$，同样得用这套（学不学得动是另一回事，见 D4 末尾的实测）。

那为什么必须一直正交？因为折进权重用的是 $R^{\top}$。只有正交时 $R^{\top}=R^{-1}$，你才不必在推理时求逆，也不必存第二个矩阵。S3 的要求更硬：

$$
(qR)(kR)^{\top}=q\,RR^{\top}k^{\top}
$$

要让它恒等于 $qk^{\top}$，就必须有 $RR^{\top}=I$——这条不是“最好满足”，是必须满足。于是一旦 $R$ 在训练中漂出正交矩阵的集合，“插进去不改变全精度输出”这个前提当场失效，被最小化的 $\mathcal{L}_Q$ 就是另一个网络的损失，优化也就没有意义了。这是约束，不是审美。

而朴素的 $R\leftarrow R-\eta G$ 走一步就不正交。可以每步补一次 SVD 或 QR 拉回来，代价是每步一次分解；Cayley 的做法是把“落在流形上”直接写进更新式，省掉这一步补救。

#### 为什么是乘性小步，而不是 $R=(I-A)(I+A)^{-1}$

后者把整个 $R$ 表示成反对称矩阵的函数，有两个麻烦：$(I+A)^{-1}$ 里的 $A$ 可以任意大、可以病态；而且想从给定的 $R$ 反解 $A$ 时，只要 $R$ 有特征值 $-1$ 就解不出来——$H$ 恰好就有（D4 里量到的 $\mathrm{rank}(I+H_{128})=64$）。

论文的式 (3) 是乘性的：$R\leftarrow\Delta R(Y)R$，每次只是在左边乘一个 $I$ 附近的小旋转。$(I-\tfrac{\alpha}{2}Y)^{-1}$ 用截断级数算，全是矩阵乘法；而且**始终不需要从 $R$ 反解 $A$**，所以 $-1$ 特征值在这里根本不是问题。论文 §5（S078）把这点讲得很直白：skew 参数化加 Cayley、或者矩阵指数，都要每步一次昂贵的求逆或矩阵指数，而 Cayley SGD 只用矩阵乘法。

#### 为什么偏偏 $R_1,R_2$ 值得优化

回到 D4 末尾那个对比：$R_3$ 是“转完就点积”，没有下游权重可以借力，单层能拿到的结构在 $H$ 那里已经到顶。$R_1$ 不一样——它在残差口，一份 $d\times d$ 被所有层共用、折进每层读残差的 5 个权重矩阵，梯度里带着整网的信息；$R_2$ 影响的不只是 V cache，还有 $W_o$ 的输入。所以这两个位置的优化有真实抓手：S039 里“不同旋转能让准确率差 13 分”，正是“方向没选好”留出来的空间。

#### 代价与两个容易忽略的细节

- **代价**：每步一次完整的前向 + 反向。LLaMA-3 1B/3B/8B 约 13/18/30 分钟，LLaMA-2 7B/13B 约 25/30 分钟，70B 约 3.5 小时，Mistral-7B 约 16 分钟（§4.1）。校准样本从 800 减到 128、迭代超过 100 步，困惑度都不再明显变化（Table 11）。
- **旋转是“每个模型算一次”**，算完写进 checkpoint；推理时不会对每个输入再优化。
- **和 GPTQ 一起用时**，旋转是在“只量化激活”的网络上学的（§4.3.2）：权重误差留给 GPTQ，旋转只负责激活误差，两者不抢同一件事。
:::

<a id="S043"></a>
**Source:** p.6 S043

:::en
**Original:** We apply the Cayley SGD method to solve Eqn. 2 for {R1,R2}, while the underlying weight parameters in the network remain frozen. {R1, R2} count for only ∼0.26% of the weight size and is constrained to be orthonormal. Consequently, the underlying floating-point network remains unchanged, and the rotation only influences the quantization performance.
:::

:::zh
**中文:** 我们应用 Cayley SGD 方法求解关于 $\{R_1,R_2\}$ 的式 2，而网络中底层的权重参数保持冻结。$\{R_1, R_2\}$ 仅占权重规模的约 0.26%，并被约束为正交。因此底层浮点网络保持不变，旋转只影响量化性能。
:::

<a id="S044"></a>
**Source:** p.6 S044

:::en
**Original:** By employing Cayley optimization to update the rotation for 100 iterations on an 800-sample WikiText2 calibration dataset, we obtain a rotation matrix that outperforms the best random matrix and random Hadamard matrix in 100 random seeds, shown in Figure 4. The Cayley-optimized rotation exhibits minimal variance when initiated from different random seeds. The rotation matrices are initialized with random Hadamard matrices for optimization and our ablation study in Section 4.3.3 demonstrates that the optimized rotation is robust to random rotation initialization as well.
:::

:::zh
**中文:** 以 800 个样本的 WikiText2 校准集执行 Cayley 优化、把旋转更新 100 次后，我们得到的旋转矩阵在 100 个随机种子下都优于最好的随机矩阵与随机 Hadamard 矩阵，见 Figure 4。从不同随机种子出发，Cayley 优化得到的旋转方差极小。旋转矩阵以随机 Hadamard 矩阵初始化，第 4.3.3 节的消融研究表明优化后的旋转对随机旋转初始化同样稳健。
:::

<a id="S045"></a>
## 4. 实验 / Experiments

<a id="S046"></a>
**Source:** p.6 S046

:::en
**Original:** We conduct experiments on the LLaMA-2 (Touvron et al., 2023b) models (7B/13B/70B), LLaMA- 3 (AI@Meta, 2024) models (1B/3B/8B) and Mistral (Jiang et al., 2023) 7B model. Our evaluation of the proposed SpinQuant was carried out on eight zero-shot commonsense reasoning tasks. These tasks include BoolQ (Clark et al., 2019), PIQA (Bisk et al., 2020), SIQA (Sap et al., 2019), HellaSwag (Zellers et al., 2019), WinoGrande (Sakaguchi et al., 2021), ARC-easy and ARC- challenge (Clark et al., 2018), and OBQA (Mihaylov et al., 2018). Additionally, we also report the perplexity score on WikiText2 testset (Merity et al., 2016) for our evaluation.
:::

:::zh
**中文:** 我们在 LLaMA-2 (Touvron et al., 2023b)（7B/13B/70B）、LLaMA-3 (AI@Meta, 2024)（1B/3B/8B）与 Mistral (Jiang et al., 2023) 7B 模型上进行实验。对 SpinQuant 的评估在八个零样本常识推理任务上展开，包括 BoolQ (Clark et al., 2019)、PIQA (Bisk et al., 2020)、SIQA (Sap et al., 2019)、HellaSwag (Zellers et al., 2019)、WinoGrande (Sakaguchi et al., 2021)、ARC-easy 与 ARC-challenge (Clark et al., 2018) 以及 OBQA (Mihaylov et al., 2018)。此外，我们还报告 WikiText2 测试集上的困惑度 (Merity et al., 2016) 作为评估指标。
:::

<a id="S047"></a>
### 4.1 实验设置 / Experimental Settings

<a id="S048"></a>
**Source:** p.6 S048

:::en
**Original:** We employ Cayley SGD (Li et al., 2020) to optimize the rotation matrix, R1 and R2, both initialized as a random Hadamard matrix, while maintaining all network weights constant. R1 is the residual rotation, shaped as (Dtoken, Dtoken). R2 is head-wise rotation in each attention block, shaped as (Dhead, Dhead) and is separately learned in each layer. The learning rate starts at 1.5 and linearly decays to 0. We utilize 800 samples from WikiText-2 to optimize rotation for 100 iterations. It takes only ∼ 13 / 18 / 30 minutes for LLaMA-3 1B / 3B / 8B, respectively, and ∼ 25 / 30 minutes for LLaMA-2 7B / 13B, respectively. For LLaMA-2 70B, it takes ∼ 3.5 hours and for Mistral-7B it takes ∼ 16 minutes.
:::

:::zh
**中文:** 我们采用 Cayley SGD (Li et al., 2020) 优化旋转矩阵 $R_1$ 与 $R_2$，二者均以随机 Hadamard 矩阵初始化，同时保持所有网络权重不变。$R_1$ 是残差旋转，形状为 $(D_{token}, D_{token})$。$R_2$ 是每个注意力模块内按 head 的旋转，形状为 $(D_{head}, D_{head})$，并在每层分别学习。学习率从 1.5 开始并线性衰减到 0。我们使用 WikiText-2 的 800 个样本将旋转优化 100 次迭代。LLaMA-3 1B / 3B / 8B 分别仅需约 13 / 18 / 30 分钟，LLaMA-2 7B / 13B 分别约 25 / 30 分钟。LLaMA-2 70B 约需 3.5 小时，Mistral-7B 约需 16 分钟。
:::

<a id="S049"></a>
**Source:** p.6 S049

:::en
**Original:** In the main results, we optimize the rotation with respect to the activation quantized network, where the weights remain 16-bit. After rotation is learned, we apply GPTQ on the rotated weights (Frantar et al., 2022), for which we adhere to the standard GPTQ settings by using 128 samples from WikiText- 2 with a sequence length of 2048 as the calibration set for GPTQ quantization. In the main table, we present the results of SpinQuant with GPTQ, and in the ablation study, while we also show the results of employing simple round-to-nearest (RTN) quantization in the ablation study.
:::

:::zh
**中文:** 在主要结果中，我们针对激活被量化的网络优化旋转，此时权重保持 16-bit。旋转学习完成后，我们对旋转后的权重应用 GPTQ (Frantar et al., 2022)；为此遵循标准 GPTQ 设置，使用 WikiText-2 中 128 个样本、序列长度 2048 作为 GPTQ 量化的校准集。主表中我们给出 SpinQuant 结合 GPTQ 的结果，而在消融研究中也给出采用简单最近取整（RTN）量化的结果。
:::

<a id="S050"></a>
### 4.2 主要结果 / Main Results

<a id="S051"></a>
**Source:** p.6 S051

:::en
**Original:** We present two rotation schemes SpinQuant$_{no\ had}$ and SpinQuant$_{had}$ to accommodate different scenarios. In Table 1, we use seven models and four most commonly used bit-width settings to provide a guideline on which rotation scheme should be chosen in practice.
:::

:::zh
**中文:** 我们给出两种旋转方案 SpinQuant$_{no\ had}$ 与 SpinQuant$_{had}$，以适应不同场景。在 Table 1 中，我们使用七个模型和四种最常用的比特宽度设置，为实践中应选择哪种旋转方案提供指引。
:::

<a id="S052"></a>
**Source:** p.6 S052

:::en
**Original:** **Recap** SpinQuant$_{no\ had}$ uses learned rotation $R_1$ and $R_2$ only, which can be merged into corresponding model weights during inference time after the rotation is learned. Using SpinQuant$_{no\ had}$
:::

:::zh
**中文:** **回顾**  SpinQuant$_{no\ had}$ 仅使用学到的旋转 $R_1$ 与 $R_2$，它们在旋转学习完成后可在推理时并入对应模型权重。使用 SpinQuant$_{no\ had}$
:::

<a id="T001"></a>
#### Table 1. Main Results: PPL and Average Zero-Shot Accuracy <span class="lang-zh">· 主结果：PPL 与零样本常识推理平均准确率</span>

**Placed near:** p.6 S052  
**Source:** p.7 T001 · C005

<img src="/images/quantization/spinquant/tables/T001.png" alt="Table 1" width="1650" height="1013" loading="lazy" decoding="async">

<a id="C005"></a>

:::en
**Original caption:** Table 1: Comparison of the perplexity score on WikiText2 and averaged accuracy on eight Zero-shot Common Sense Reasoning tasks. Results for SmoothQuant (Xiao et al., 2022), LLM-QAT (Liu et al., 2023c), GPTQ (Frantar et al., 2022) were obtained using their publicly released codebase. While OmniQuant (Shao et al., 2023), AWQ (Lin et al., 2023), and QuIP# (Tseng et al., 2024) results were quoted from their papers. Full results are in the Appendix.
:::

:::zh
**中文表注:** 表 1：WikiText2 上的困惑度与八个零样本常识推理任务平均准确率的比较。SmoothQuant (Xiao et al., 2022)、LLM-QAT (Liu et al., 2023c)、GPTQ (Frantar et al., 2022) 的结果由各自公开的代码库获得；OmniQuant (Shao et al., 2023)、AWQ (Lin et al., 2023)、QuIP# (Tseng et al., 2024) 的结果引自其论文。完整结果见附录。
:::

:::zh
**Reading note:** 主表：7 个模型 × 4 种比特配置。加粗行为 SpinQuant 结果；W4A4KV4 一栏是关键对比。
:::

<a id="S053"></a>
**Source:** p.7 S053

:::en
**Original:** only needs to replace the original model weights with the rotated model weights, necessitating no modification to the forward pass nor any additional kernel support. While SpinQuant$_{had}$ comprises both learned rotations (R1, R2) and the online Hadamard rotations (R3, R4). During inference time, R3 and R4 can be computed with fast Hadamard kernel (Tseng et al., 2024) and we show in Sec. 4.5, the online Hadamard rotation only introduces ∼8% of the network latency overhead.
:::

:::zh
**中文:** 只需用旋转后的模型权重替换原始权重，既无需改动前向过程，也不需要额外的算子支持。而 SpinQuant$_{had}$ 同时包含学到的旋转（$R_1$、$R_2$）与在线 Hadamard 旋转（$R_3$、$R_4$）。推理时 $R_3$ 与 $R_4$ 可用快速 Hadamard 算子计算 (Tseng et al., 2024)，我们在第 4.5 节表明在线 Hadamard 旋转只带来约 8% 的网络延迟开销。
:::

<a id="S054"></a>
**Source:** p.7 S054

:::en
**Original:** As shown in Table 1, in the scenarios where weights are quantized to 4-bit and activations are quantized to 8-bit, using SpinQuant$_{no\ had}$ can readily achieve good performance. For example, SpinQuant$_{no\ had}$ enhances the 4-8-8 quantized Mistral 7B by 10.5 points. In llama3-8B, SpinQuant$_{no\ had}$ achieves more than 4.1 point improvements compared to GPTQ (Frantar et al., 2022) on 4-8-16 setting, and leaving the gap to full-precision network to only 1.0 point. In these settings with activations not extremely quantized, using SpinQuant$_{no\ had}$ is a viable solution, and adding additional online Hadamard rotation yields marginal benefit.
:::

:::zh
**中文:** 如 Table 1 所示，在权重 4-bit、激活 8-bit 的场景下，使用 SpinQuant$_{no\ had}$ 即可轻松取得良好性能。例如，SpinQuant$_{no\ had}$ 使 4-8-8 量化的 Mistral 7B 提升 10.5 分。在 llama3-8B 上，SpinQuant$_{no\ had}$ 在 4-8-16 设置下比 GPTQ (Frantar et al., 2022) 提升 4.1 分以上，与全精度网络的差距仅 1.0 分。在这些激活未被极端量化的设置中，SpinQuant$_{no\ had}$ 是可行方案，额外加入在线 Hadamard 旋转带来的收益有限。
:::

<a id="S055"></a>
**Source:** p.7 S055

:::en
**Original:** In contrast, when activations are quantized to 4 bits, the accuracy drops significantly and most previous methods fail to produce meaningful results. SpinQuant$_{no\ had}$ bridge the gap by up to 20 points. In 4-4-4 quantized LLaMA-2 models, SpinQuant$_{no\ had}$ significantly surpasses LLM-QAT (Liu et al., 2023c), by 11.1 points on 7B model and outperforms SmoothQuant (Xiao et al., 2023) by 20.2 on the 13B model, thereby reducing the gap to the corresponding full-precision network from 22.0 / 27.8 points to 10.9 / 7.6 points respectively. Still, the gap to the full-precision network is non-negligible. In this scenario, SpinQuant$_{had}$ can further improve the accuracy by more than 5 points and close the gap to the respective FP network to 2-4 points. In 4-4-4 quantized LLaMA-2 7B/13B/70B models, SpinQuant$_{had}$ leaves only a 2.9/1.4/1.7 accuracy gap to the corresponding full-precision network, significantly surpassing the previous SoTA methods by 19.1/16.4/15.3 points, respectively.
:::

:::zh
**中文:** 相反，当激活量化为 4-bit 时，准确率显著下降，多数先前方法无法给出有意义的结果。SpinQuant$_{no\ had}$ 最多可弥补 20 分的差距。在 4-4-4 量化的 LLaMA-2 模型上，SpinQuant$_{no\ had}$ 在 7B 上比 LLM-QAT (Liu et al., 2023c) 高 11.1 分，在 13B 上比 SmoothQuant (Xiao et al., 2023) 高 20.2 分，从而将与对应全精度网络的差距分别从 22.0 / 27.8 分降到 10.9 / 7.6 分。但与全精度网络的差距仍不可忽略。在该场景下，SpinQuant$_{had}$ 可进一步将准确率提升 5 分以上，把与相应 FP 网络的差距缩小到 2-4 分。在 4-4-4 量化的 LLaMA-2 7B/13B/70B 上，SpinQuant$_{had}$ 与对应全精度网络仅剩 2.9/1.4/1.7 的准确率差距，分别比先前的 SoTA 方法高出 19.1/16.4/15.3 分。
:::

<a id="S056"></a>
**Source:** p.7 S056

:::en
**Original:** In addition, compared to the state-of-the-art weight-only quantization methods, OmniQuant (Shao et al., 2023), AWQ (Lin et al., 2023) and QuIP# (Tseng et al., 2024), SpinQuant achieves similar evaluation perplexity on Wiki dataset with 4-bit weights and 8-bit activations, and without using advance vector quantization technique. These results show SpinQuant is suitable for various scenarios and achieves state-of-the-art performance.
:::

:::zh
**中文:** 此外，与最先进的权重仅量化方法 OmniQuant (Shao et al., 2023)、AWQ (Lin et al., 2023) 和 QuIP# (Tseng et al., 2024) 相比，SpinQuant 在 4-bit 权重、8-bit 激活下于 Wiki 数据集上取得了相近的评估困惑度，且没有使用更高级的向量量化技术。这些结果表明 SpinQuant 适用于多种场景，并达到最先进的性能。
:::

<a id="T002"></a>
#### Table 2. Learned Rotation vs Hadamard Rotation <span class="lang-zh">· 学到的旋转 vs Hadamard 旋转</span>

**Placed near:** p.6 S056  
**Source:** p.8 T002 · C006

<img src="/images/quantization/spinquant/tables/T002.png" alt="Table 2" width="1353" height="277" loading="lazy" decoding="async">

<a id="C006"></a>

:::en
**Original caption:** Table 2: Compared to Hadamard rotation, SpinQuant learned rotation consistently outperform by a significant margin. Results are averaged accuracy on eight Zero-shot CommonSense Reasoning tasks.
:::

:::zh
**中文表注:** 表 2：与 Hadamard 旋转相比，SpinQuant 学到的旋转始终以显著幅度更优。结果为八个零样本常识推理任务上的平均准确率。
:::

:::zh
**Reading note:** 学习旋转相对 Hadamard 旋转的提升（括号内为增量）。
:::

<a id="T003"></a>
#### Table 3. Compatibility with GPTQ <span class="lang-zh">· 与 GPTQ 的兼容性消融</span>

**Placed near:** p.6 S056  
**Source:** p.8 T003 · C007

<img src="/images/quantization/spinquant/tables/T003.png" alt="Table 3" width="946" height="258" loading="lazy" decoding="async">

<a id="C007"></a>

:::en
**Original caption:** Table 3: Ablation study on compatibility with GPTQ (Frantar et al., 2022) on a LLaMA2-7B model.
:::

:::zh
**中文表注:** 表 3：在 LLaMA2-7B 模型上关于与 GPTQ (Frantar et al., 2022) 兼容性的消融研究。
:::

:::zh
**Reading note:** 仅量化激活（而非权重）来学习旋转，再叠加 GPTQ，效果最好。
:::

<a id="S057"></a>
### 4.3 消融研究 / Ablation Studies

<a id="S058"></a>
### 4.3.1 学习得到的旋转与随机旋转对比 / Learned Rotation vs Random Rotation

<a id="S059"></a>
**Source:** p.8 S059

:::en
**Original:** In Table 2, we contrast the use of random Hadamard rotations with SpinQuant’s optimized rotations. Employing learned rotations, whether under R1,2 settings or R1,2,3,4 settings, consistently enhances accuracy across various models and bit-width configurations. Notably, in the quantization of Mistral-7B, SpinQuant$_{had}$ secures an improvement exceeding 15.7 points over using random Hadamard rotations. Given that rotation optimization incurs a minimal time cost (only 30 minutes for smaller models and up to 3.5 hours for a 70B model) we advocate for the adoption of optimized rotations for precise quantization of LLMs.
:::

:::zh
**中文:** 在 Table 2 中，我们将随机 Hadamard 旋转与 SpinQuant 优化后的旋转进行对比。使用学到的旋转，无论在 $R_{1,2}$ 还是 $R_{1,2,3,4}$ 设置下，都能在多种模型与比特宽度配置上稳定提升准确率。值得注意的是，在 Mistral-7B 的量化中，SpinQuant$_{had}$ 相比使用随机 Hadamard 旋转取得超过 15.7 分的提升。考虑到旋转优化所需时间极少（小模型仅 30 分钟，70B 模型最多 3.5 小时），我们主张对 LLM 的精确量化采用优化后的旋转。
:::

<a id="S060"></a>
### 4.3.2 与 GPTQ 的兼容性 / Compatibility with GPTQ

<a id="S061"></a>
**Source:** p.8 S061

:::en
**Original:** In the context where both weights and activations are quantized, we observed that the learned rotations tend to adapt effectively to both weight and activation quantization. Given that GPTQ significantly helps mitigate the errors due to weight quantization, but leaves activation quantization untouched, we elect to optimize the rotation matrices with respect to a network where only activations are quantized. This approach allows the rotation to more efficiently manage the activation quantization error while leaving the weight quantization error to be addressed by GPTQ. As shown in Table 3, this modification resulted in superior performance in both W4A4 and W4A4KV4 settings in the LLaMA-2 7B model, which is the configuration we choose to utilize throughout the rest of this paper.
:::

:::zh
**中文:** 在权重与激活都被量化的情形下，我们观察到学到的旋转倾向于同时适配权重量化与激活量化。鉴于 GPTQ 能显著缓解权重量化带来的误差、却不触及激活量化，我们选择在「只对激活做量化」的网络上优化旋转矩阵。这样旋转可以更有效地处理激活量化误差，而把权重量化误差留给 GPTQ。如 Table 3 所示，这一改动在 LLaMA-2 7B 的 W4A4 与 W4A4KV4 设置下都带来更好的性能，因此本文其余部分都采用这一配置。
:::

<a id="S062"></a>
### 4.3.3 旋转类型 / Rotation Type

<a id="S063"></a>
**Source:** p.8 S063

:::en
**Original:** In Table 4, we evaluate the impact of random orthogonal floating-point rotation matrices and random Hadamard matrices on quantization accuracy, utilizing round-to-nearest quantization for our analysis. Prior to optimization, the Hadamard matrices yield a better-quantized network performance compared to floating-point rotation matrices. However, after optimization, the initial choice of rotation, whether floating-point or Hadamard, becomes less significant. This is likely due to the loss-aware rotation optimization’s ability to locate an optimal local minima that effectively minimizes quantization error, thereby enhancing robustness to varying types of rotation initialization.
:::

:::zh
**中文:** 在 Table 4 中，我们评估随机正交浮点旋转矩阵与随机 Hadamard 矩阵对量化精度的影响，分析中使用最近取整量化。优化之前，Hadamard 矩阵得到的量化网络性能优于浮点旋转矩阵。然而在优化之后，旋转的初始选择（浮点或 Hadamard）变得不那么重要。这很可能是因为损失感知的旋转优化能够找到有效最小化量化误差的最优局部极小，从而增强了对不同旋转初始化类型的鲁棒性。
:::

<a id="S064"></a>
### 4.3.4 与 QuaRot 的对比 / Comparison with QuaRot

<a id="S065"></a>
**Source:** p.8 S065

:::en
**Original:** Compared to QuaRot (Ashkboos et al., 2023b), which exhibits significant accuracy variances in quantized networks—experiencing drops of 28.1 and 33.2 points when quantizing a 70B model with round-to-nearest methods to W4A4 and W4A4KV4—this degradation stems from inherent noise in using random rotation matrix that introduce high variance and compromise robustness. In contrast, SpinQuant$_{had}$ consistently maintains high accuracy across various configurations, achieving improvements of 2.0 to 28.6 points over QuaRot (Table 5), while utilizing fewer online Hadamard matrices (two per block in SpinQuant$_{had}$ versus four per block in QuaRot).
:::

:::zh
**中文:** QuaRot (Ashkboos et al., 2023b) 在量化网络中表现出显著的精度方差——用最近取整方法把 70B 模型量化到 W4A4 与 W4A4KV4 时分别下降 28.1 与 33.2 分；这种退化源于使用随机旋转矩阵所固有的噪声，它带来高方差并损害鲁棒性。相比之下，SpinQuant$_{had}$ 在各类配置下始终维持高准确率，相比 QuaRot 取得 2.0 至 28.6 分的提升（Table 5），同时使用的在线 Hadamard 矩阵更少（SpinQuant$_{had}$ 每个模块两个，QuaRot 每个模块四个）。
:::

<a id="S066"></a>
**Source:** p.8 S066

:::en
**Original:** Furthermore, the integration of R2 in SpinQuant effectively reduces in-block outliers, thereby enabling SpinQuant$_{no\ had}$ to deliver optimal performance in W4A8 settings. SpinQuant$_{no\ had}$ can be achieved by simply substituting the model weights with rotated weights, making it a more
:::

:::zh
**中文:** 此外，SpinQuant 中引入 $R_2$ 有效减少了模块内部的离群值，从而使 SpinQuant$_{no\ had}$ 在 W4A8 设置下发挥最优性能。SpinQuant$_{no\ had}$ 只需用旋转后的权重替换模型权重即可实现，因此相比需要修改模型架构并提供特殊算子支持的 QuaRot，是一种更
:::

<a id="T004"></a>
#### Table 4. Floating-Point Rotation vs Hadamard Rotation <span class="lang-zh">· 浮点旋转 vs Hadamard 旋转</span>

**Placed near:** p.8 S066  
**Source:** p.9 T004 · C008

<img src="/images/quantization/spinquant/tables/T004.png" alt="Table 4" width="1155" height="395" loading="lazy" decoding="async">

<a id="C008"></a>

:::en
**Original caption:** Table 4: Floating-point(FP) rotation vs Hadamard rotation on a LLaMA-2 7B model.
:::

:::zh
**中文表注:** 表 4：LLaMA-2 7B 模型上浮点（FP）旋转与 Hadamard 旋转的对比。
:::

:::zh
**Reading note:** 优化前 Hadamard 初始化更好，优化后初始化方式的影响变小。
:::

<a id="T005"></a>
#### Table 5. Comparison with QuaRot <span class="lang-zh">· 与 QuaRot 的对比</span>

**Placed near:** p.8 S066  
**Source:** p.9 T005 · C009

<img src="/images/quantization/spinquant/tables/T005.png" alt="Table 5" width="1279" height="342" loading="lazy" decoding="async">

<a id="C009"></a>

:::en
**Original caption:** Table 5: Comparison with QuaRot (Ashkboos et al., 2023b).
:::

:::zh
**中文表注:** 表 5：与 QuaRot (Ashkboos et al., 2023b) 的对比。
:::

:::zh
**Reading note:** QuaRot 在 70B 上 RTN 时崩溃（41.5/91.2），SpinQuant$_{had}$ 仍保持稳定。
:::

<a id="S067"></a>
**Source:** p.9 S067

:::en
**Original:** straightforward and efficient approach compared to QuaRot, which requires modifying the model architecture and special kernel support.
:::

:::zh
**中文:** 这与 QuaRot 相比是一种直接而高效的方式，后者需要修改模型架构并支持特殊算子。
:::

<a id="S068"></a>
### 4.4 旋转有效性的示例分析 / Illustrative Analysis of the Rotation Efficacy

<a id="S069"></a>
**Source:** p.9 S069

:::en
**Original:** The rationale behind rotating network weights and activations can be elucidated through a straightforward example. Consider an activation (X) represented as a 2D vector, where one entry x1 consistently receives higher magnitude activations than x2 (as depicted in Figure 6(a)). Quantizing these components together typically results in a quantization range dominated by x1, thereby compromising the precision for x2.
:::

:::zh
**中文:** 旋转网络权重与激活背后的原理可以通过一个简单例子说明。考虑一个表示为二维向量的激活（$X$），其中一个分量 $x_1$ 的幅值始终大于 $x_2$（如 Figure 6(a) 所示）。把这些分量放在一起量化，量化范围通常会被 $x_1$ 主导，从而损害 $x_2$ 的精度。
:::

<a id="S070"></a>
**Source:** p.9 S070

:::en
**Original:** From an information entropy standpoint, expanding each axis to fully utilize the available quantization range maximizes the representational capacity of each axis. Thus, matrix rotation emerges as an intuitive solution. In a 2D scenario, rotating the axis by 45° equalizes the value representation range across axes (illustrated in Figure 6(b)). Assuming the network as a black box without knowledge of the exact activation distribution, uniformly rotating all axes by the maximal degree (45° in 2D) can optimize distribution evenness across each axis, partially explaining why Hadamard rotation often outperforms random rotation matrices.
:::

:::zh
**中文:** 从信息熵的角度看，让每个轴充分用满可用的量化范围，可以最大化每个轴的表征能力。因此矩阵旋转成为直观的解决方案。在二维情形下，把坐标轴旋转 45° 就能使各轴的数值表征范围趋于均衡（如 Figure 6(b) 所示）。若把网络视为黑箱、不知道确切的激活分布，那么将所有轴统一旋转到最大角度（二维中为 45°）可以优化各轴间分布的均匀性，这在一定程度上解释了为什么 Hadamard 旋转往往优于随机旋转矩阵。
:::

<a id="S071"></a>
**Source:** p.9 S071

:::en
**Original:** Taking this further, if the activation distribution is known, treating the network as a white box during quantization allows for the identification of more optimal rotations than Hadamard. For instance, in a 3D scenario depicted in Figure 6(c-d), where x1’s magnitude is four times that of x2 and x3, rotating the distribution by 45° along x3 and x2 redistributes the maximum values from [2, 0.5, 0.5] to [1, 1, 1.414]. However, even more optimal rotation strategies may exist, and learning the rotation can help pinpoint the most effective rotation for a given distribution.
:::

:::zh
**中文:** 进一步说，如果激活分布已知，在量化时把网络视为白箱，就有可能找到比 Hadamard 更优的旋转。例如在 Figure 6(c-d) 所示的 3D 情形中，$x_1$ 的幅值是 $x_2$ 与 $x_3$ 的四倍，沿 $x_3$ 与 $x_2$ 旋转 45° 可将最大值从 [2, 0.5, 0.5] 重新分布为 [1, 1, 1.414]。然而可能存在更优的旋转策略，学习旋转有助于为给定分布找到最有效的旋转。
:::

<a id="S072"></a>
**Source:** p.9 S072

:::en
**Original:** This opens up intriguing research avenues, such as determining if, given an activation distribution with known outlier axes and magnitudes, a closed-form solution for the optimal rotation matrix that evenly distributes magnitude across different axes can be derived. Additionally, it raises the question
:::

:::zh
**中文:** 这开启了一些有趣的研究方向，例如：在已知离群轴与幅值的激活分布下，能否推导出让幅值在各轴上均匀分摊的最优旋转矩阵的闭式解。此外它还引出一个问题：
:::

<a id="F006"></a>
#### Fig. 6. How Rotation Reduces Outliers (2D/3D Examples) <span class="lang-zh">· 旋转如何削减离群值（2D/3D 示例）</span>

**Placed near:** p.9 S072  
**Source:** p.9 F006 · C010

<img src="/images/quantization/spinquant/figures/F006.png" alt="Figure 6" width="1393" height="424" loading="lazy" decoding="async">

<a id="C010"></a>

:::en
**Original caption:** Figure 6: An illustration of how rotation helps reduce outliers and maximize quantization range utilization. 9
:::

:::zh
**中文图注:** 图 6：旋转如何帮助削减离群值并最大化量化范围利用率的图示。
:::

:::zh
**Reading note:** 二维/三维玩具示例说明旋转如何把主轴方向的离群幅值摊平到各轴。
:::

<a id="T006"></a>
#### Table 6. Speed Measurement on MacBook M1 Pro <span class="lang-zh">· MacBook M1 Pro 上的速度测量</span>

**Placed near:** p.9 S072  
**Source:** p.10 T006 · C011

<img src="/images/quantization/spinquant/tables/T006.png" alt="Table 6" width="779" height="183" loading="lazy" decoding="async">

<a id="C011"></a>

:::en
**Original caption:** Table 6: Real-time end-to-end speed measurement of LLaMA-3 8B on MacBook M1 Pro CPU.
:::

:::zh
**中文表注:** 表 6：LLaMA-3 8B 在 MacBook M1 Pro CPU 上的实时端到端速度测量。
:::

:::zh
**Reading note:** 4-bit 相比 16-bit 约 3× 加速；在线 Hadamard 带来约 8% 额外延迟。
:::

<a id="S073"></a>
**Source:** p.10 S073

:::en
**Original:** of whether this theoretically calculated rotation yields the best quantization performance. We leave this question to future research.
:::

:::zh
**中文:** 这样理论计算出的旋转是否能带来最佳的量化性能。我们把这个问题留待未来研究。
:::

<a id="S074"></a>
### 4.5 速度测量 / Speed Measurement

<a id="S075"></a>
**Source:** p.10 S075

:::en
**Original:** We conduct an end-to-end speed measurement of the LLaMA-3 8B model with W16A16 and W4A8 configurations on a MacBook M1 Pro CPU (OS version 14.5). The results in Table 6 demonstrate that 4-bit quantization yields a ∼3× speedup compared to the 16-bit model. Comparing SpinQuant$_{had}$ to SpinQuant$_{no\ had}$, online Hadamard processing introduced a modest 8% increase in latency. Therefore, it is a trade-off between using SpinQuant$_{no\ had}$ without online Hadamard for its simpleness or using SpinQuant$_{had}$ with online Hadamard rotations for higher accuracy in lower- bit activation quantization. Detailed GPU latency results are provided in the Appendix.
:::

:::zh
**中文:** 我们在 MacBook M1 Pro CPU（操作系统版本 14.5）上对 LLaMA-3 8B 模型的 W16A16 与 W4A8 配置做了端到端速度测量。Table 6 的结果表明，4-bit 量化相比 16-bit 模型带来约 3 倍加速。比较 SpinQuant$_{had}$ 与 SpinQuant$_{no\ had}$，在线 Hadamard 处理带来约 8% 的适度延迟增加。因此这里存在权衡：要么使用不含在线 Hadamard 的 SpinQuant$_{no\ had}$ 以取其简洁，要么使用带在线 Hadamard 旋转的 SpinQuant$_{had}$ 以在更低比特激活量化下取得更高精度。更详细的 GPU 延迟结果见附录。
:::

<a id="S076"></a>
## 5. 相关工作 / Related Work

<a id="S077"></a>
**Source:** p.10 S077

:::en
**Original:** Quantization Neural network quantization has been demonstrated as an effective tool for model size compression and storage reduction (Nagel et al., 2020; Krishnamoorthi, 2018; Nagel et al., 2019; Li et al., 2021). However, in large language models (LLMs), quantization presents unique challenges due to the presence of numerous outliers. These outliers dominate the quantization range, leaving only a few effective bits for the majority of values. Various strategies have been proposed to address the difficulties in LLM quantization. These include separating outliers and using mixed precision (Dettmers et al., 2022; van Baalen et al., 2023; Kim et al., 2023; Huang et al., 2024; Egiazarian et al., 2024), employing Hessian-based methods to mitigate quantization difficulty (Frantar et al., 2022), trading outliers between weights and activations (Xiao et al., 2022; Lin et al., 2023; Liu et al., 2023b) utilizing weight equalization (Nagel et al., 2019), outlier suppression (Wei et al., 2022; 2023), channel reassembly (Liu et al., 2023a) and even suggesting architectural modifications to handle outliers during pre-training(Yelysei Bondarenko, 2023). Recently two QuIP papers (Chee et al., 2024; Tseng et al., 2024) introduce the incoherence processing using random rotation matrices and applying vector quantization on the weights for compression. This does introduce extra overhead and imposes some constraints on the devices the LLM is deployed to in the availability of vector quantization kernels.
:::

:::zh
**中文:** **量化**  神经网络量化已被证明是压缩模型规模、减少存储的有效工具 (Nagel et al., 2020; Krishnamoorthi, 2018; Nagel et al., 2019; Li et al., 2021)。然而在大语言模型（LLM）中，由于存在大量离群值，量化面临独特挑战。这些离群值主导了量化范围，使大多数数值只能获得很少的有效比特。已有多种策略被提出来应对 LLM 量化的困难，包括：分离离群值并使用混合精度 (Dettmers et al., 2022; van Baalen et al., 2023; Kim et al., 2023; Huang et al., 2024; Egiazarian et al., 2024)；采用基于 Hessian 的方法缓解量化难度 (Frantar et al., 2022)；在权重与激活之间转移离群值 (Xiao et al., 2022; Lin et al., 2023; Liu et al., 2023b)；利用权重均衡 (Nagel et al., 2019)、离群值抑制 (Wei et al., 2022; 2023)、通道重排 (Liu et al., 2023a)，甚至建议在预训练阶段修改架构以处理离群值 (Yelysei Bondarenko, 2023)。近期两篇 QuIP 论文 (Chee et al., 2024; Tseng et al., 2024) 提出利用随机旋转矩阵做非相干处理，并对权重施加向量量化以压缩。这确实引入了额外开销，并在向量量化核的可用性上对 LLM 部署设备施加了一些限制。
:::

<a id="S078"></a>
**Source:** p.10 S078

:::en
**Original:** Optimization in orthonormal space The optimization of rotation matrices is carried out within the Stiefel Manifold (James, 1976), which encompasses all orthonormal matrices. Optimization while staying on this manifold can be done by e.g., parameterizing a skew-symmetric matrix and applying the Cayley transformation on top of it (Nishimori and Akaho, 2005), or using a matrix exponential (Absil and Malick, 2012; Lezcano-Casado and Martinez-Rubio, 2019). However, these methods rely on expensive inverse or matrix-exponential functions that are applied every iteration. Instead, we follow the more efficient method named Cayley SGD (Li et al., 2020), which can be applied to optimize a rotation matrix R for arbitrary loss functions efficiently. Cayley SGD relies on an iterative approximation of the Cayley Transform that is conducted solely with matrix multiplications.
:::

:::zh
**中文:** **正交空间中的优化**  旋转矩阵的优化在 Stiefel 流形 (James, 1976) 中进行，该流形包含所有正交矩阵。在保持位于该流形上的同时做优化，可以通过例如参数化一个反对称矩阵并在其上施加 Cayley 变换 (Nishimori and Akaho, 2005)，或使用矩阵指数 (Absil and Malick, 2012; Lezcano-Casado and Martinez-Rubio, 2019) 来实现。然而这些方法依赖每次迭代都要执行的昂贵求逆或矩阵指数运算。我们转而采用更高效的方法 Cayley SGD (Li et al., 2020)，它可以针对任意损失函数高效地优化旋转矩阵 $R$。Cayley SGD 依赖 Cayley 变换的迭代近似，且只通过矩阵乘法完成。
:::

<a id="S079"></a>
## 6. 结论 / Conclusions

<a id="S080"></a>
**Source:** p.10 S080

:::en
**Original:** In this paper, we present SpinQuant, a novel quantization technique that utilizes learned rotation to effectively bridge the performance gap between full precision and 4-bit weight, activation, and kv-cache quantization. At its core, SpinQuant leverages the rotation invariance property of LLM models to insert rotation matrices that diminish outliers in the weights and intermediate activations while maintaining the network’s full-precision output numerically identical. Additionally, SpinQuant incorporates Cayley SGD for optimizing rotation matrices, resulting in improved and robust quantization outcomes. Importantly, SpinQuant is compatible with more advanced weight quantization techniques (e.g., GPTQ) and demonstrates state-of-the-art performance.
:::

:::zh
**中文:** 本文提出 SpinQuant，这是一种利用可学习旋转的量化技术，能有效弥合全精度与 4-bit 权重、激活及 kv-cache 量化之间的性能差距。其核心在于利用 LLM 的旋转不变性插入旋转矩阵，在保持网络全精度输出数值不变的同时削减权重与中间激活中的离群值。此外，SpinQuant 引入 Cayley SGD 来优化旋转矩阵，带来更优且更稳健的量化结果。重要的是，SpinQuant 与更先进的权重量化技术（如 GPTQ）兼容，并展现出最先进的性能。
:::

<a id="S081"></a>
## 致谢 / Acknowledgment

<a id="S082"></a>
**Source:** p.11 S082

:::en
**Original:** We extend our gratitude to Scott Wolchok and Chen Lai for their crucial contributions to latency measurement on the MacBook M1 Pro CPU, and to Sijia Chen, Geonhwa Jeong and Jiecao Yu for their expert support in GPU latency assessment.
:::

:::zh
**中文:** 我们感谢 Scott Wolchok 与 Chen Lai 在 MacBook M1 Pro CPU 延迟测量上的关键贡献，并感谢 Sijia Chen、Geonhwa Jeong 与 Jiecao Yu 在 GPU 延迟评估上的专业支持。
:::

<a id="S083"></a>
## 参考文献 / References

:::zh
**说明**：参考文献条目按原文照录（不逐条翻译），保留原始 `S...` 锚点以便溯源。
:::

:::en
<a id="S084"></a>P-A Absil and Jérôme Malick. Projection-like retractions on matrix manifolds. SIAM Journal on Optimization, 22(1):135–158, 2012.  
<a id="S085"></a>Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.  
<a id="S086"></a>AI@Meta. Llama 3 model card. 2024. URL https://github.com/meta-llama/llama3/blob/ main/MODEL_CARD.md.  
<a id="S087"></a>Saleh Ashkboos, Maximilian L Croci, Marcelo Gennari do Nascimento, Torsten Hoefler, and James Hensman. Slicegpt: Compress large language models by deleting rows and columns. In The Twelfth International Conference on Learning Representations, 2023a.  
<a id="S088"></a>Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L. Croci, Bo Li, Martin Jaggi, Dan Alistarh, Torsten Hoefler, and James Hensman. Quarot: Outlier-free 4-bit inference in rotated llms. 2023b.  
<a id="S089"></a>Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 7432–7439, 2020.  
<a id="S090"></a>Jerry Chee, Yaohui Cai, Volodymyr Kuleshov, and Christopher M De Sa. Quip: 2-bit quantization of large language models with guarantees. Advances in Neural Information Processing Systems, 36, 2024.  
<a id="S091"></a>Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions. arXiv preprint arXiv:1905.10044, 2019.  
<a id="S092"></a>Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.  
<a id="S093"></a>Samuel Rhys Cox and Wei Tsang Ooi. Conversational interactions with npcs in llm-driven gaming: Guidelines from a content analysis of player feedback. In International Workshop on Chatbot Research and Design, pages 167–184. Springer, 2023.  
<a id="S094"></a>Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. Llm.int8(): 8-bit matrix multiplication for transformers at scale. 2022.  
<a id="S095"></a>Vage Egiazarian, Andrei Panferov, Denis Kuznedelev, Elias Frantar, Artem Babenko, and Dan Alistarh. Extreme compression of large language models via additive quantization. arXiv preprint arXiv:2401.06118, 2024.  
<a id="S096"></a>Nelson Elhage, Robert Lasenby, and Christopher Olah. Privileged bases in the transformer residual stream. Transformer Circuits Thread, 2023.  
<a id="S097"></a>Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. Gptq: Accurate post-training quantization for generative pre-trained transformers. arXiv preprint arXiv:2210.17323, 2022.  
<a id="S098"></a>Wei Huang, Haotong Qin, Yangdong Liu, Yawei Li, Xianglong Liu, Luca Benini, Michele Magno, and Xiaojuan Qi. Slim-llm: Salience-driven mixed-precision quantization for large language models. arXiv preprint arXiv:2405.14917, 2024.  
<a id="S099"></a>Ioan Mackenzie James. The topology of Stiefel manifolds, volume 24. Cambridge University Press, 1976.  
<a id="S100"></a>Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.  
<a id="S101"></a>Sehoon Kim, Coleman Hooper, Amir Gholami, Zhen Dong, Xiuyu Li, Sheng Shen, Michael W Mahoney, and Kurt Keutzer. Squeezellm: Dense-and-sparse quantization. arXiv preprint arXiv:2306.07629, 2023.  
<a id="S102"></a>Raghuraman Krishnamoorthi. Quantizing deep convolutional networks for efficient inference: A whitepaper. arXiv preprint arXiv:1806.08342, 2018.  
<a id="S103"></a>Mario Lezcano-Casado and David Martinez-Rubio. Cheap orthogonal constraints in neural networks: A simple parametrization of the orthogonal and unitary group. 2019.  
<a id="S104"></a>Jun Li, Li Fuxin, and Sinisa Todorovic. Efficient riemannian optimization on the stiefel manifold via the cayley transform. arXiv preprint arXiv:2002.01113, 2020.  
<a id="S105"></a>Yuhang Li, Ruihao Gong, Xu Tan, Yang Yang, Peng Hu, Qi Zhang, Fengwei Yu, Wei Wang, and Shi Gu. Brecq: Pushing the limit of post-training quantization by block reconstruction. In International Conference on Learning Representations (ICLR), 2021.  
<a id="S106"></a>Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Xingyu Dang, and Song Han. Awq: Activation-aware weight quantization for llm compression and acceleration. arXiv preprint arXiv:2306.00978, 2023.  
<a id="S107"></a>Jing Liu, Ruihao Gong, Xiuying Wei, Zhiwei Dong, Jianfei Cai, and Bohan Zhuang. Qllm: Accurate and efficient low-bitwidth quantization for large language models. arXiv preprint arXiv:2310.08041, 2023a.  
<a id="S108"></a>Shih-yang Liu, Zechun Liu, Xijie Huang, Pingcheng Dong, and Kwang-Ting Cheng. Llm-fp4: 4-bit floating- point quantized transformers. arXiv preprint arXiv:2310.16836, 2023b.  
<a id="S109"></a>Zechun Liu, Barlas Oguz, Changsheng Zhao, Ernie Chang, Pierre Stock, Yashar Mehdad, Yangyang Shi, Raghuraman Krishnamoorthi, and Vikas Chandra. Llm-qat: Data-free quantization aware training for large language models. arXiv preprint arXiv:2305.17888, 2023c.  
<a id="S110"></a>Zechun Liu, Changsheng Zhao, Forrest Iandola, Chen Lai, Yuandong Tian, Igor Fedorov, Yunyang Xiong, Ernie Chang, Yangyang Shi, Raghuraman Krishnamoorthi, et al. Mobilellm: Optimizing sub-billion parameter language models for on-device use cases. arXiv preprint arXiv:2402.14905, 2024.  
<a id="S111"></a>Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models. arXiv preprint arXiv:1609.07843, 2016.  
<a id="S112"></a>Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. arXiv preprint arXiv:1809.02789, 2018.  
<a id="S113"></a>Markus Nagel, Mart van Baalen, Tijmen Blankevoort, and Max Welling. Data-free quantization through weight equalization and bias correction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1325–1334, 2019.  
<a id="S114"></a>Markus Nagel, Rana Ali Amjad, Mart Van Baalen, Christos Louizos, and Tijmen Blankevoort. Up or down? Adaptive rounding for post-training quantization. In International Conference on Machine Learning (ICML), 2020.  
<a id="S115"></a>Yasunori Nishimori and Shotaro Akaho. Learning algorithms utilizing quasi-geodesic flows on the stiefel manifold. Neurocomputing, 67:106–135, 2005.  
<a id="S116"></a>Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research, 21(1):5485–5551, 2020.  
<a id="S117"></a>Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, et al. Code llama: Open foundation models for code. arXiv preprint arXiv:2308.12950, 2023.  
<a id="S118"></a>Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM, 64(9):99–106, 2021.  
<a id="S119"></a>Maarten Sap, Hannah Rashkin, Derek Chen, Ronan LeBras, and Yejin Choi. Socialiqa: Commonsense reasoning about social interactions. arXiv preprint arXiv:1904.09728, 2019.  
<a id="S120"></a>Wenqi Shao, Mengzhao Chen, Zhaoyang Zhang, Peng Xu, Lirui Zhao, Zhiqian Li, Kaipeng Zhang, Peng Gao, Yu Qiao, and Ping Luo. Omniquant: Omnidirectionally calibrated quantization for large language models. arXiv preprint arXiv:2308.13137, 2023.  
<a id="S121"></a>Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.  
<a id="S122"></a>Arun James Thirunavukarasu, Darren Shu Jeng Ting, Kabilan Elangovan, Laura Gutierrez, Ting Fang Tan, and Daniel Shu Wei Ting. Large language models in medicine. Nature medicine, 29(8):1930–1940, 2023.  
<a id="S123"></a>Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023a.  
<a id="S124"></a>Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023b.  
<a id="S125"></a>Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, and Christopher De Sa. Quip#: Even better llm quantization with hadamard incoherence and lattice codebooks. arXiv preprint arXiv:2402.04396, 2024.  
<a id="S126"></a>Mart van Baalen, Markus Nagel Andrey Kuzmin, Peter Couperus, Cedric Bastoul, Eric Mahurin, Tijmen Blankevoort, and Paul Whatmough. Gptvq: The blessing of dimensionality for llm quantization. 2023.  
<a id="S127"></a>Xiuying Wei, Yunchen Zhang, Xiangguo Zhang, Ruihao Gong, Shanghang Zhang, Qi Zhang, Fengwei Yu, and Xianglong Liu. Outlier suppression: Pushing the limit of low-bit transformer language models. arXiv preprint arXiv:2209.13325, 2022.  
<a id="S128"></a>Xiuying Wei, Yunchen Zhang, Yuhang Li, Xiangguo Zhang, Ruihao Gong, Jinyang Guo, and Xianglong Liu. Outlier suppression+: Accurate quantization of large language models by equivalent and optimal shifting and scaling. arXiv preprint arXiv:2304.09145, 2023.  
<a id="S129"></a>Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. Smoothquant: Accurate and efficient post-training quantization for large language models. In CVPR, 2022.  
<a id="S130"></a>Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. Smoothquant: Accurate and efficient post-training quantization for large language models. In International Conference on Machine Learning, pages 38087–38099. PMLR, 2023.  
<a id="S131"></a>Tijmen Blankevoort Yelysei Bondarenko, Markus Nagel. Quantizable transformers: Removing outliers by helping attention heads do nothing. 2023.  
<a id="S132"></a>Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830, 2019.  
<a id="S133"></a>Yilong Zhao, Chien-Yu Lin, Kan Zhu, Zihao Ye, Lequn Chen, Size Zheng, Luis Ceze, Arvind Krishnamurthy, Tianqi Chen, and Baris Kasikci. Atom: Low-bit quantization for efficient and accurate llm serving. arXiv preprint arXiv:2310.19102, 2023.
:::

<a id="S134"></a>
## 附录（补充材料）/ Appendix / Supplemental Material

<a id="S135"></a>
### A.1 主结果表的完整结果 / Complete Results of Main Result Table

<a id="S136"></a>
**Source:** p.14 S136

:::en
**Original:** In Tables 7, 8 and 9, we show the complete results of Table 1. We compare the accuracy on eight zero-shot commonsense reasoning tasks including ARC-easy, ARC-challenge (Clark et al., 2018), BoolQ (Clark et al., 2019), PIQA (Bisk et al., 2020), SIQA (Sap et al., 2019), HellaSwag (Zellers et al., 2019), OBQA (Mihaylov et al., 2018), and WinoGrande (Sakaguchi et al., 2021) as well as the perplexity score on WikiText2 testset (Merity et al., 2016). We compare our results with previous works including SmoothQuant(Xiao et al., 2022), LLM-QAT(Liu et al., 2023c), GPTQ (Frantar et al., 2022), OmniQuant (Shao et al., 2023), QuIP# (Tseng et al., 2024).
:::

:::zh
**中文:** 在 Tables 7、8、9 中，我们给出 Table 1 的完整结果。我们比较了八个零样本常识推理任务上的准确率，包括 ARC-easy、ARC-challenge (Clark et al., 2018)、BoolQ (Clark et al., 2019)、PIQA (Bisk et al., 2020)、SIQA (Sap et al., 2019)、HellaSwag (Zellers et al., 2019)、OBQA (Mihaylov et al., 2018) 与 WinoGrande (Sakaguchi et al., 2021)，以及 WikiText2 测试集 (Merity et al., 2016) 上的困惑度。我们将结果与 SmoothQuant (Xiao et al., 2022)、LLM-QAT (Liu et al., 2023c)、GPTQ (Frantar et al., 2022)、OmniQuant (Shao et al., 2023)、QuIP# (Tseng et al., 2024) 等先前工作进行比较。
:::

<a id="S137"></a>
### A.2 3-bit 权重量化结果 / Results on 3-Bit Weight Quantization

<a id="S138"></a>
**Source:** p.14 S138

:::en
**Original:** We present the 3-bit weight and 8-bit activation quantization results across seven models in Table 10. Our method, SpinQuant, successfully reduces the gap to the full-precision network from the previous 9.0 − 28.0 points to 1.2 − 5.3 points, demonstrating its effectiveness for low-bit quantization.
:::

:::zh
**中文:** 我们在 Table 10 中给出七个模型上 3-bit 权重、8-bit 激活的量化结果。我们的方法 SpinQuant 成功地将与全精度网络的差距从此前的 9.0 − 28.0 分缩小到 1.2 − 5.3 分，展示了其在低比特量化上的有效性。
:::

<a id="S139"></a>
### A.3 Cayley 优化中的选择 / Cayley Optimization Choice

<a id="S140"></a>
**Source:** p.14 S140

:::en
**Original:** In Table 11, we evaluate the impact of varying the number of samples and iterations used in Cayley optimization. Given the limited trainable parameters in the rotation matrix and its constraint optimization nature, minimal calibration data and iterations are sufficient to optimize the rotation for better quantization. The findings indicate that rotation optimization is resilient to modifications in the number of samples. Even though we used 800 samples in our experiments, reducing this to 128 samples does not lead to a significant change in the perplexity. Furthermore, we examined the optimal number of iterations and found that the wiki perplexity ceases to decrease and stabilizes at 100 iterations. Consequently, we chose to use 100 iterations in all our experiments.
:::

:::zh
**中文:** 在 Table 11 中，我们评估 Cayley 优化中样本数与迭代次数变化的影响。鉴于旋转矩阵中可训练参数有限、且优化具有约束性质，极少的校准数据与迭代次数就足以优化旋转以获得更好的量化。结果显示，旋转优化对样本数的改变具有韧性。尽管我们在实验中使用了 800 个样本，将其减少到 128 个样本并不会使困惑度发生显著变化。此外我们考察了最优迭代次数，发现 wiki 困惑度在 100 次迭代时不再下降并趋于稳定。因此我们在所有实验中选择使用 100 次迭代。
:::

<a id="S141"></a>
### A.4 量化方式的选择 / Quantization Choice

<a id="S142"></a>
**Source:** p.14 S142

:::en
**Original:** We conduct an ablation study on symmetric vs asymmetric quantization and whether to clip the min-max ranges or not during activation and KV-cache quantization. The results in Table 12 show that for both activation quantization and KV-cache quantization, asymmetric quantization outperforms symmetric quantization. In the clip settings, we set the activation clipping ratio to 0.9 and the KV-cache clipping ratio to 0.95 as suggested in the previous works (Zhao et al., 2023). However, the results show that clipping the range or not does not impact the final result significantly. Therefore we opt for no clipping, i.e., using the min-max quantization for activation and KV cache quantization across our experiments due to its simplicity.
:::

:::zh
**中文:** 我们针对对称量化与非对称量化、以及在激活与 KV-cache 量化时是否裁剪 min-max 范围做了消融研究。Table 12 的结果表明，无论是激活量化还是 KV-cache 量化，非对称量化都优于对称量化。在裁剪设置中，按照先前工作 (Zhao et al., 2023) 的建议，我们将激活裁剪比例设为 0.9、KV-cache 裁剪比例设为 0.95。然而结果显示，是否裁剪范围对最终结果影响不大。因此我们选择不裁剪，即所有实验中对激活与 KV cache 量化都采用 min-max 量化，因为这样更简单。
:::

<a id="S143"></a>
### A.5 校准数据的选择 / Calibration Data Choice

<a id="S144"></a>
**Source:** p.14 S144

:::en
**Original:** To assess the robustness of SpinQuant with respect to calibration data used in rotation optimization we use C4 dataset (Raffel et al., 2020) as calibration data and performe experiments on the LLaMA-2 7B model. The results in Table 13 reflect that using C4 datasets yields consistent results with utilizing the Wiki dataset, showing that SpinQuant is robust to calibration data choice.
:::

:::zh
**中文:** 为评估 SpinQuant 对旋转优化所用校准数据的鲁棒性，我们使用 C4 数据集 (Raffel et al., 2020) 作为校准数据，并在 LLaMA-2 7B 模型上进行实验。Table 13 的结果表明，使用 C4 数据集与使用 Wiki 数据集的结果一致，说明 SpinQuant 对校准数据的选择具有鲁棒性。
:::

<a id="S145"></a>
### A.6 GPU 上的延迟测量 / Latency Measurement on GPU

<a id="S146"></a>
**Source:** p.14 S146

:::en
**Original:** In light of the available Tensor cores in NVIDIA’s Hopper (H100) architecture, we provide the whole network end-to-end speed test result of W-fp8-A-fp8 quantization on H100 GPU, both with and without Hadamard transformations. Specifically, we utilize FP8 GEMM from the FBGEMM repo 3, which incorporates dequantization via epilogue fusion. We also leverage the Tensor Core-based Hadamard transform kernel 4 to minimize the overhead of the online Hadamard transform. The end-
:::

:::zh
**中文:** 鉴于 NVIDIA Hopper（H100）架构中可用的 Tensor Core，我们给出 H100 GPU 上 W-fp8-A-fp8 量化的全网络端到端速度测试结果，分别包含与不包含 Hadamard 变换。具体而言，我们使用 FBGEMM 仓库 ³ 中的 FP8 GEMM，它通过 epilogue 融合实现反量化。我们还利用基于 Tensor Core 的 Hadamard 变换核 ⁴ 来最小化在线 Hadamard 变换的开销。端
:::

<a id="S147"></a>
**Source:** p.14 S147

> **Footnote ³:** https://github.com/pytorch/FBGEMM/blob/main/fbgemm_gpu/experimental/gemm/triton_gemm/fp8_gemm.py

<a id="S148"></a>
**Source:** p.14 S148

> **Footnote ⁴:** https://github.com/pytorch-labs/applied-ai/blob/main/kernels/cuda/inference/hadamard_transform/hadamard_transform.cpp

<a id="T007"></a>
#### Table 7. Complete Results on LLaMA-2 <span class="lang-zh">· LLaMA-2 完整结果</span>

**Placed near:** p.14 S146  
**Source:** p.15 T007 · C012

<img src="/images/quantization/spinquant/tables/T007.png" alt="Table 7" width="1486" height="2431" loading="lazy" decoding="async">

<a id="C012"></a>

:::en
**Original caption:** Table 7: Complete comparison of the perplexity score on WikiText2 and averaged accuracy on Zero-shot Common Sense Reasoning tasks on LLaMA-2.
:::

:::zh
**中文表注:** 表 7：LLaMA-2 上 WikiText2 困惑度与零样本常识推理任务平均准确率的完整比较。
:::

:::zh
**Reading note:** 按任务的完整数值（ARC-e/ARC-c/BoolQ/PIQA/SIQA/HellaSwag/OBQA/WinoGrande）。
:::

<a id="T008"></a>
#### Table 8. Complete Results on LLaMA-3 <span class="lang-zh">· LLaMA-3 完整结果</span>

**Placed near:** p.14 S146  
**Source:** p.16 T008 · C013

<img src="/images/quantization/spinquant/tables/T008.png" alt="Table 8" width="1485" height="2274" loading="lazy" decoding="async">

<a id="C013"></a>

:::en
**Original caption:** Table 8: Complete omparison of the perplexity score on WikiText2 and averaged accuracy on Zero-shot Common Sense Reasoning tasks on LLaMA-3.
:::

:::zh
**中文表注:** 表 8：LLaMA-3 上 WikiText2 困惑度与零样本常识推理任务平均准确率的完整比较。
:::

:::zh
**Reading note:** LLaMA-3 系列三款模型的完整数值。
:::

<a id="S149"></a>
**Source:** p.16 S149

:::en
**Original:** to-end speed test results of LLaMA-3 70B are detailed in Table 14. When implemented meticulously, SpinQuant with Hadamard rotation sees marginal difference in the latency compared to without Hadamard rotation.
:::

:::zh
**中文:** 到端的速度测试结果在 Table 14 中给出，对象为 LLaMA-3 70B。在实现得当时，带 Hadamard 旋转的 SpinQuant 与不带 Hadamard 旋转的版本在延迟上差异很小。
:::

<a id="T009"></a>
#### Table 9. Complete Results on Mistral-7B <span class="lang-zh">· Mistral-7B 完整结果</span>

**Placed near:** p.14 S149  
**Source:** p.17 T009 · C014

<img src="/images/quantization/spinquant/tables/T009.png" alt="Table 9" width="1485" height="683" loading="lazy" decoding="async">

<a id="C014"></a>

:::en
**Original caption:** Table 9: Complete comparison of the perplexity score on WikiText2 and averaged accuracy on Zero-shot Common Sense Reasoning tasks on Mistral-7B-v0.3.
:::

:::zh
**中文表注:** 表 9：Mistral-7B-v0.3 上 WikiText2 困惑度与零样本常识推理任务平均准确率的完整比较。
:::

:::zh
**Reading note:** Mistral-7B 的完整数值。
:::

<a id="T010"></a>
#### Table 10. 3-Bit Weight Quantization Results <span class="lang-zh">· 3-bit 权重量化结果</span>

**Placed near:** p.14 S149  
**Source:** p.17 T010 · C015

<img src="/images/quantization/spinquant/tables/T010.png" alt="Table 10" width="1485" height="1315" loading="lazy" decoding="async">

<a id="C015"></a>

:::en
**Original caption:** Table 10: 3-bit weight 8-bit activation quantization results on WikiText2 and Zero-shot Common Sense Reasoning tasks.
:::

:::zh
**中文表注:** 表 10：3-bit 权重、8-bit 激活量化在 WikiText2 与零样本常识推理任务上的结果。
:::

:::zh
**Reading note:** 3-bit 权重 + 8-bit 激活：差距从 9.0–28.0 分缩小到 1.2–5.3 分。
:::

<a id="T011"></a>
#### Table 11. Ablation on Calibration Samples and Iterations <span class="lang-zh">· Cayley 优化的样本/迭代次数消融</span>

**Placed near:** p.14 S149  
**Source:** p.17 T011 · C016

<img src="/images/quantization/spinquant/tables/T011.png" alt="Table 11" width="1484" height="148" loading="lazy" decoding="async">

<a id="C016"></a>

:::en
**Original caption:** Table 11: Ablation study on Number of training samples and iterations in Cayley SGD optimization, using LLaMA-2 7B.
:::

:::zh
**中文表注:** 表 11：在 LLaMA-2 7B 上关于 Cayley SGD 优化中训练样本数与迭代次数的消融研究。
:::

:::zh
**Reading note:** 样本数从 800 减到 128、迭代 100 次即可收敛。
:::

<a id="T012"></a>
#### Table 12. Symmetric vs Asymmetric Quantization and Clipping <span class="lang-zh">· 对称 vs 非对称量化与裁剪</span>

**Placed near:** p.14 S149  
**Source:** p.18 T012 · C017

<img src="/images/quantization/spinquant/tables/T012.png" alt="Table 12" width="1485" height="305" loading="lazy" decoding="async">

<a id="C017"></a>

:::en
**Original caption:** Table 12: Ablation of symmetric and asymmetric quantization and range clipping options on LLaMA-2 7B.
:::

:::zh
**中文表注:** 表 12：LLaMA-2 7B 上对称/非对称量化与范围裁剪选项的消融。
:::

:::zh
**Reading note:** 非对称量化优于对称量化；是否裁剪影响不大。
:::

<a id="T013"></a>
#### Table 13. Calibration Data Choice <span class="lang-zh">· 校准数据选择</span>

**Placed near:** p.14 S149  
**Source:** p.18 T013 · C018

<img src="/images/quantization/spinquant/tables/T013.png" alt="Table 13" width="1574" height="376" loading="lazy" decoding="async">

<a id="C018"></a>

:::en
**Original caption:** Table 13: Ablation study on calibration data choice using LLaMA-2 7B.
:::

:::zh
**中文表注:** 表 13：在 LLaMA-2 7B 上关于校准数据选择的消融研究。
:::

:::zh
**Reading note:** C4 与 WikiText2 校准结果一致。
:::

<a id="S150"></a>
### A.7 优化时间 / Optimization Time

<a id="S151"></a>
**Source:** p.18 S151

:::en
**Original:** In Table 15, we show the comparison of optimization time between GPTQ and SpinQuant. SpinQuant requires a scale of optimization time similar to that of GPTQ. The additional optimization time required by SpinQuant is worthwhile considering the substantial improvements it offers over GPTQ.
:::

:::zh
**中文:** 在 Table 15 中，我们展示 GPTQ 与 SpinQuant 在优化时间上的比较。SpinQuant 所需的优化时间与 GPTQ 处于相似量级。考虑到 SpinQuant 相对 GPTQ 带来的实质提升，额外的优化时间是值得的。
:::

<a id="S152"></a>
### A.8 RTN 与 GPTQ 的消融研究 / Ablation Study on RTN vs GPTQ

<a id="S153"></a>
**Source:** p.18 S153

:::en
**Original:** SpinQuant is fully compatible with both GPTQ and naive RTN. To isolate the contributions of GPTQ and rotation to overall performance, we present results for SpinQuant combined with RTN in the W4A4KV16 quantization scenario in Table 16. Our analysis indicates that the primary accuracy gains are attributed to the incorporation of learned rotations, which enhances accuracy by 6.5 ∼ 20.9 percentage points over previous methods (including GPTQ). The subsequent integration of GPTQ further boosts performance by up to 2.3 percentage points.
:::

:::zh
**中文:** SpinQuant 与 GPTQ 以及朴素 RTN 都完全兼容。为分离 GPTQ 与旋转对整体性能的贡献，我们在 W4A4KV16 量化场景下给出 SpinQuant 结合 RTN 的结果（Table 16）。分析表明，主要精度收益来自可学习旋转的引入，它相比先前方法（包括 GPTQ）带来 6.5 ∼ 20.9 个百分点的准确率提升。随后再结合 GPTQ 可进一步把性能提高最多 2.3 个百分点。
:::

<a id="S154"></a>
### A.9 权重仅量化 / Weight-Only Quantization

<a id="S155"></a>
**Source:** p.18 S155

:::en
**Original:** We also include a comparison of SpinQuant performance under weight-only quantization in Table 17. The weight-only results show that SpinQuant consistently achieves higher accuracy than AWQ and other previous work.
:::

:::zh
**中文:** 我们还在 Table 17 中给出权重仅量化下 SpinQuant 性能的比较。权重仅量化的结果表明，SpinQuant 始终取得比 AWQ 及其他先前工作更高的准确率。
:::

<a id="S156"></a>
### A.10 指令微调模型上的少样本结果 / Few-Shot Results on Instruction-Finetuned Models

<a id="S157"></a>
**Source:** p.18 S157

:::en
**Original:** We further conduct experiments applying SpinQuant to instruction-finetuned LLaMA 3.2 1B and 3B models in Table 18. We present the results for few-shot learning scenarios. SpinQuant W4A8 quantized models demonstrate significant improvements in 5-shot accuracy on the MMLU benchmark and 1-shot rouge score on the TLDR9 summarization benchmark. It significantly closed the gap to the BF16 baseline.
:::

:::zh
**中文:** 我们进一步在 Table 18 中对指令微调的 LLaMA 3.2 1B 与 3B 模型应用 SpinQuant 开展实验。我们给出少样本学习场景的结果。SpinQuant W4A8 量化模型在 MMLU 基准的 5-shot 准确率与 TLDR9 摘要基准的 1-shot rouge 分数上都有显著提升，明显缩小了与 BF16 基线的差距。
:::

<a id="S158"></a>
## B 分析 / Analysis

<a id="S159"></a>
### B.1 梯度分析 / Gradient Analysis

<a id="S160"></a>
**Source:** p.18 S160

:::en
**Original:** On the one hand, we have shown that the class of LLMs we are interested in are rotation invariant, i.e. the full-precision model output does not change regardless of what R is. On the other hand, we are claiming that some R are better than others for quantized LLM and that better R can be learned with backpropagation on equation (2). To reconcile these seemingly conflicting claims, we inspect the gradient of the output of a single linear, W , and activations, X , which are both rotated and quantized:
:::

:::zh
**中文:** 一方面，我们已经说明所关注的这类 LLM 具有旋转不变性，即无论 $R$ 是什么，全精度模型的输出都不改变。另一方面，我们主张某些 $R$ 对量化 LLM 优于其他 $R$，且更好的 $R$ 可以通过对式 (2) 的反向传播学到。为调和这两个看似矛盾的论断，我们考察单个线性层输出对权重 $W$ 与激活 $X$ 的梯度，二者都被旋转并量化：
:::

<a id="E005"></a>
**Source:** p.18 E005 · Eq. (5)

$$
\frac{\partial\sum_{ij}\left(Q(WR^{-1})Q(RX)\right)_{ij}}{\partial R_{mn}}=\sum_{ij}\left[-(WR^{-1})_{im}\left(R^{-1}Q(RX)\right)_{nj}+Q(WR^{-1})_{im}X_{nj}\right]
$$

<div align="right">(5)</div>

:::zh
**中文说明:** 量化输出对旋转矩阵的梯度：解释为何可以反向传播学习 $R$

*（原始裁切核对：[E005](/images/quantization/spinquant/equations/E005.png)）*
:::

<a id="T014"></a>
#### Table 14. Speed Measurement on H100 <span class="lang-zh">· H100 上的速度测量</span>

**Placed near:** p.18 S160  
**Source:** p.19 T014 · C019

<img src="/images/quantization/spinquant/tables/T014.png" alt="Table 14" width="1055" height="192" loading="lazy" decoding="async">

<a id="C019"></a>

:::en
**Original caption:** Table 14: Real-time end-to-end speed measurement of LLaMA-3 70B on NVIDIA’s Hopper (H100) GPU. TTFT (Time to First Token) and TTIT (Time to Iterative Token) are performance metrics to measure the pre-filling speed and decoding speed, respectively.
:::

:::zh
**中文表注:** 表 14：LLaMA-3 70B 在 NVIDIA Hopper（H100）GPU 上的实时端到端速度测量。TTFT（Time to First Token）与 TTIT（Time to Iterative Token）分别是衡量预填充速度与解码速度的性能指标。
:::

:::zh
**Reading note:** H100 上的 TTFT/TTIT；带与不带 Hadamard 差异很小。
:::

<a id="T015"></a>
#### Table 15. Optimization Time Comparison <span class="lang-zh">· 优化时间对比</span>

**Placed near:** p.18 S160  
**Source:** p.19 T015 · C020

<img src="/images/quantization/spinquant/tables/T015.png" alt="Table 15" width="1657" height="507" loading="lazy" decoding="async">

<a id="C020"></a>

:::en
**Original caption:** Table 15: Optimization time comparison between GPTQ and SpinQuant.
:::

:::zh
**中文表注:** 表 15：GPTQ 与 SpinQuant 的优化时间比较。
:::

:::zh
**Reading note:** SpinQuant 优化时间与 GPTQ 同量级。
:::

<a id="S161"></a>
**Source:** p.19 S161

:::en
**Original:** • is non-zero in general, which validates our approach of using backpropagation to learn $R$ • reduces to 0 when quantization is not present, which validates the claim that it only makes sense to learn $R$ for quantized models
:::

:::zh
**中文:**

- 一般情况下不为零，这验证了我们用反向传播学习 $R$ 的做法
- 在不存在量化时变为 0，这验证了“只有对量化模型学习 $R$ 才有意义”的说法
:::

<a id="S162"></a>
**Source:** p.19 S162

:::en
**Original:** • demonstrates that two components move the gradient with respect to R away from 0: 1) differences in quantized and unquantized rotated weights; 2) differences in quantized and unquantized rotated activations
:::

:::zh
**中文:** • 说明有两个分量使关于 $R$ 的梯度偏离 0：1）量化与未量化的旋转权重之间的差异；2）量化与未量化的旋转激活之间的差异
:::

:::zh
<a id="D007"></a>
### 推导 D7 · 既然输出不变，凭什么还能学 R

**插入位置：** p.19 · 紧随 S162（附录 B.1 末）

**性质：** 读者补充推导，非论文原文。严格版见 `DERIVATION_PACKAGE.md` §7

这里有个看起来矛盾的地方：全精度网络对 $R$ 是**完全不敏感**的，那梯度不就应该恒等于 0、根本学不动吗？答案是：梯度确实来自量化器。

**把一层单独拿出来看.** 一层做的事是“激活乘权重”，旋转后变成

$$
\text{输出}=Q\big(W R^{-1}\big)\,Q\big(RX\big),
$$

其中 $Q$ 是量化（取整）。如果 $Q$ 什么都不做，$R^{-1}$ 和 $R$ 精确抵消，输出跟 $R$ 毫无关系——这时候对 $R$ 求导当然是 0。一旦 $Q$ 把数截断了，抵消不再精确，剩下的差就是量化误差，输出重新依赖 $R$。

**所以梯度长什么样.** 把上式对 $R$ 求导再整理，结果会分成两项：一项由**激活的量化误差**产生，一项由**权重的量化误差**产生。

- $Q$ 是恒等（不量化）时两个误差都是 0，两项同时消失，梯度为 0 —— 这就是 S161 说的“不存在量化时降为 0”。
- 两项分别对应权重误差与激活误差 —— 这正是 S162 说的“两个分量”。
- 每一项都是“一列数 × 一行数”的形式，所以整个梯度矩阵很“薄”。这解释了为什么 100 次迭代就够（Table 11）。

**直觉.** 学 $R$ 其实是在挑一个坐标系，让量化误差尽量小；误差不存在的场合（全精度）没有可挑的，梯度自然为零。

**边界.** 上面是论文为了讲清机制用的单层模型。真实损失是交叉熵、多层耦合，梯度不会有这么干净的形式；但“只有量化才让 $R$ 变得可学”这个结论不变。
:::

<a id="S163"></a>
### B.2 损失分析 / Loss Analysis

<a id="F007"></a>
#### Fig. 7. Training Curves and SNR Analysis <span class="lang-zh">· 训练曲线与 SNR 分析</span>

**Placed near:** p.19 S163  
**Source:** p.19 F007 · C021

<img src="/images/quantization/spinquant/figures/F007.png" alt="Figure 7" width="1662" height="441" loading="lazy" decoding="async">

<a id="C021"></a>

:::en
**Original caption:** Figure 7: Training curves for LLaMA-2 7B with 4-bit weights and 4-bit activations in wiki2 train set. (a) End- to-end quantization SNR. R0 and RT denote randomly initialized rotation and learned rotation after T = 200 iterations; (b) Activation quantization. SNR for layer 27 attention out projection; (c) Improvement in activation quantization SNR after optimization of R for each layer.
:::

:::zh
**中文图注:** 图 7：LLaMA-2 7B（4-bit 权重、4-bit 激活）在 wiki2 训练集上的训练曲线。(a) 端到端量化 SNR，$R_0$ 与 $R_T$ 分别表示随机初始化的旋转与 $T=200$ 次迭代后学到的旋转；(b) 激活量化：第 27 层 attention out projection 的 SNR；(c) 对每一层优化 $R$ 后激活量化 SNR 的改善。
:::

:::zh
**Reading note:** (a) 端到端量化 SNR 随迭代上升；(b) 第 27 层 out projection 的 SNR；(c) 各层 SNR 改善量。
:::

<a id="S164"></a>
**Source:** p.19 S164

:::en
**Original:** While Sec. 4 shows that learning R yields significant benefits on zero-shot reasoning tasks, in this section we shed some light on why our method is able to achieve accuracy gains. Intuitively, we expect the end-to-end signal to (quantization) noise ratio (SNR) to improve as a result of learning R. In other words, learning R should bring the quantized model output closer to the floating point model output. As Table 19 shows, we observe an SNR improvement of 3.8 dB when introducing a random R into LLaMA-2 7B with weights/activations quantized to 4 bits, and then an additional 5.9dB improvement after learning R, all measured on the WikiText2 (Merity et al., 2016) test set. Figure 7a shows that the batch-level training set SNR during R training progressively improves as expected, as well as the layer-level SNR for a particular layer in Figure 7b. Digging a bit deeper, Figure 7c shows the layer-level SNR improvement for each layer as a result of training R. We see that, perhaps counter-intuitively, layer-level SNR improves significantly for a few layers, but does not change much for most layers, and even gets worse for one of the layers. We hypothesize that: 1) certain layers have a disproportionate impact on model output or have a disproportionately low quantization SNR without rotation; 2) The process of optimizing R rotates the residual stream basis such as to prioritize improving the SNR of such layers, possibly at the cost of hurting less important layers.
:::

:::zh
**中文:** 虽然第 4 节表明学习 $R$ 在零样本推理任务上带来显著收益，本节则进一步阐释我们的方法为何能取得准确率提升。直观上，我们期望端到端信号与（量化）噪声比（SNR）因学习 $R$ 而改善。换句话说，学习 $R$ 应使量化模型的输出更接近浮点模型的输出。如 Table 19 所示，我们观察到在权重/激活量化为 4 bits 的 LLaMA-2 7B 中引入随机 $R$ 时 SNR 提升 3.8 dB，随后在学习 $R$ 之后又额外提升 5.9 dB，均在 WikiText2 (Merity et al., 2016) 测试集上测得。Figure 7a 表明，$R$ 训练过程中批级训练集 SNR 如预期逐步改善，Figure 7b 给出某一特定层的层级 SNR。进一步观察，Figure 7c 展示了训练 $R$ 后每一层的层级 SNR 改善。我们看到——或许与直觉相反——少数层的层级 SNR 显著提升，但大多数层变化不大，甚至有一层变差。我们推测：1）某些层对模型输出的影响不成比例地大，或在不做旋转时量化 SNR 不成比例地低；2）优化 $R$ 的过程会旋转残差流的基，从而优先改善这些层的 SNR，代价可能是牺牲较不重要的层。
:::

<a id="T016"></a>
#### Table 16. RTN vs GPTQ Ablation <span class="lang-zh">· RTN vs GPTQ 消融</span>

**Placed near:** p.19 S164  
**Source:** p.20 T016 · C022

<img src="/images/quantization/spinquant/tables/T016.png" alt="Table 16" width="1371" height="291" loading="lazy" decoding="async">

<a id="C022"></a>

:::en
**Original caption:** Table 16: Ablation study on SpinQuant combined with RTN or GPTQ in the W4A4KV16 quantization scenario.
:::

:::zh
**中文表注:** 表 16：W4A4KV16 量化场景下 SpinQuant 结合 RTN 或 GPTQ 的消融研究。
:::

:::zh
**Reading note:** 旋转贡献主要增益，GPTQ 再补 2.3 个百分点。
:::

<a id="T017"></a>
#### Table 17. Weight-Only Quantization <span class="lang-zh">· 权重仅量化</span>

**Placed near:** p.19 S164  
**Source:** p.20 T017 · C023

<img src="/images/quantization/spinquant/tables/T017.png" alt="Table 17" width="1485" height="261" loading="lazy" decoding="async">

<a id="C023"></a>

:::en
**Original caption:** Table 17: A comparison of SpinQuant performance under 4-bit weight-only quantization.
:::

:::zh
**中文表注:** 表 17：SpinQuant 在 4-bit 权重仅量化下的性能比较。
:::

:::zh
**Reading note:** 权重仅量化下与 AWQ 等方法对比。
:::

<a id="T018"></a>
#### Table 18. Few-Shot Results on Instruction-Finetuned Models <span class="lang-zh">· 指令微调模型上的少样本结果</span>

**Placed near:** p.19 S164  
**Source:** p.20 T018 · C024

<img src="/images/quantization/spinquant/tables/T018.png" alt="Table 18" width="1319" height="161" loading="lazy" decoding="async">

<a id="C024"></a>

:::en
**Original caption:** Table 18: Results of applying SpinQuant to instruction-finetuned LLaMA 3.2 1B and 3B models.
:::

:::zh
**中文表注:** 表 18：将 SpinQuant 应用于指令微调的 LLaMA 3.2 1B 与 3B 模型的结果。
:::

:::zh
**Reading note:** MMLU 5-shot 与 TLDR9 rougeL 上的少样本结果。
:::

<a id="T019"></a>
#### Table 19. End-to-End Signal-to-Quantization-Noise Ratio <span class="lang-zh">· 端到端信号量化噪声比</span>

**Placed near:** p.19 S164  
**Source:** p.20 T019 · C025

<img src="/images/quantization/spinquant/tables/T019.png" alt="Table 19" width="1436" height="179" loading="lazy" decoding="async">

<a id="C025"></a>

:::en
**Original caption:** Table 19: Average end-to-end signal to quantization noise ratio (dB) for LLaMA-2 7B with weights and activations quantized to 4 bits on wiki2 test set
:::

:::zh
**中文表注:** 表 19：LLaMA-2 7B（权重与激活量化为 4 bits）在 wiki2 测试集上的平均端到端信号量化噪声比（dB）。
:::

:::zh
**Reading note:** SNR 从 -2.9 dB（$R=I$）提升到 6.8 dB（学到的 $R$）。
:::

<a id="S165"></a>
## C 旋转前后的分布可视化 / Distribution Visualizations Before and After Rotation

<a id="S166"></a>
**Source:** p.20 S166

:::en
**Original:** We present visualizations of the activation distributions before and after rotation in Figures 8 and 9, respectively. Similarly, the weight distributions before and after rotation are depicted in Figures 10 and 11. Overall, after rotation, the extreme values are attenuated, and the distribution exhibits no noteworthy outliers across the token dimension. Additionally, we make an interesting observation: in several activation layers, the first token displays substantial values in multiple channels. After rotation, this outlier is distributed across all channels of the first token. Although per-token activation quantization can readily manage this distribution, investigating the source of these outliers and reducing them prior to applying SpinQuant might further enhance quantization accuracy, which could be a potential future research direction.
:::

:::zh
**中文:** 我们在 Figures 8 与 9 中分别展示旋转前后的激活分布可视化；类似地，Figures 10 与 11 展示旋转前后的权重分布。总体而言，旋转后极端值被削弱，分布在 token 维度上不再出现值得注意的离群值。此外我们有一个有趣的观察：在若干激活层中，第一个 token 在多个通道上呈现较大的数值。旋转后，该离群值被分散到第一个 token 的所有通道上。尽管逐 token 的激活量化可以轻松处理这种分布，但研究这些离群值的来源并在应用 SpinQuant 之前削减它们，可能进一步提升量化精度，这也是一个有潜力的未来研究方向。
:::

<a id="F008"></a>
#### Fig. 8. Activation Magnitude (Before Rotation) <span class="lang-zh">· 激活幅值（旋转前）</span>

**Placed near:** p.20 S166  
**Source:** p.21 F008 · C026

<img src="/images/quantization/spinquant/figures/F008.png" alt="Figure 8" width="1365" height="2459" loading="lazy" decoding="async">

<a id="C026"></a>

:::en
**Original caption:** Figure 8: Magnitude of the input activations of a linear layer in {1st, 11th, 21st, and 31st} blocks in LLaMA-2 7B model before rotation.
:::

:::zh
**中文图注:** 图 8：LLaMA-2 7B 模型中第 {1, 11, 21, 31} 个模块内某线性层输入激活在旋转前的幅值。
:::

<a id="F009"></a>
#### Fig. 9. Activation Magnitude (After Rotation) <span class="lang-zh">· 激活幅值（旋转后）</span>

**Placed near:** p.20 S166  
**Source:** p.22 F009 · C027

<img src="/images/quantization/spinquant/figures/F009.png" alt="Figure 9" width="1360" height="2459" loading="lazy" decoding="async">

<a id="C027"></a>

:::en
**Original caption:** Figure 9: Magnitude of the input activations of a linear layer in {1st, 11th, 21st, and 31st} blocks in LLaMA-2 7B model after rotation.
:::

:::zh
**中文图注:** 图 9：LLaMA-2 7B 模型中第 {1, 11, 21, 31} 个模块内某线性层输入激活在旋转后的幅值。
:::

<a id="F010"></a>
#### Fig. 10. Weight Magnitude (Before Rotation) <span class="lang-zh">· 权重幅值（旋转前）</span>

**Placed near:** p.20 S166  
**Source:** p.23 F010 · C028

<img src="/images/quantization/spinquant/figures/F010.png" alt="Figure 10" width="1364" height="2459" loading="lazy" decoding="async">

<a id="C028"></a>

:::en
**Original caption:** Figure 10: Magnitude of the weights of a linear layer in {1st, 11th, 21st, and 31st} blocks in LLaMA-2 7B before rotation.
:::

:::zh
**中文图注:** 图 10：LLaMA-2 7B 中第 {1, 11, 21, 31} 个模块内某线性层权重在旋转前的幅值。
:::

<a id="F011"></a>
#### Fig. 11. Weight Magnitude (After Rotation) <span class="lang-zh">· 权重幅值（旋转后）</span>

**Placed near:** p.20 S166  
**Source:** p.24 F011 · C029

<img src="/images/quantization/spinquant/figures/F011.png" alt="Figure 11" width="1364" height="2459" loading="lazy" decoding="async">

<a id="C029"></a>

:::en
**Original caption:** Figure 11: Magnitude of the weights of a linear layer in {1st, 11th, 21st, and 31st} blocks in LLaMA-2 7B after rotation.
:::

:::zh
**中文图注:** 图 11：LLaMA-2 7B 中第 {1, 11, 21, 31} 个模块内某线性层权重在旋转后的幅值。
:::

:::zh

---

## 公式索引

| 式号 | 位置 | 含义 |
|---|---|---|
| [E001](#E001) | p.3 | 对称/非对称量化的统一形式：$X_Q$ 为量化张量，$X_R$ 为 FP16 实值张量 |
| [E002](#E002) | p.5 | 在 Stiefel 流形上寻找使量化网络损失最小的旋转 $R_1,R_2$ |
| [E003](#E003) | p.5 | Cayley 变换给出的旋转更新，保证 $R$ 始终正交 |
| [E004](#E004) | p.6 | 由损失梯度 $G:=\nabla_R\mathcal{L}_Q$ 构造的反对称矩阵 $Y$ |
| [E005](#E005) | p.18 | 量化输出对旋转矩阵的梯度：解释为何可以反向传播学习 $R$ |

式 (1)–(5) 为论文原式。补充推导中的公式不占式号，按所在推导块登记：

| 推导块 | 位置 | 关键内容 |
|---|---|---|
| [D1](#D001) | p.3 | 用 2 维例子说明旋转如何缩小数值范围，估算等效比特数 |
| [D2](#D002) | p.5 | 归一化只看长度、$\gamma$ 折进权重、逐层归纳 |
| [D3](#D003) | p.5 | 手算 $2\times2$ 例子验证 $A(XR_2)=(AX)R_2$ |
| [D4](#D004) | p.5 | 点积两边同转不变；RoPE 与 $H$ 不可换序；逐元素积的反例；在线那一步的流程、开销，以及“固定 ≠ 可吸收” |
| [D5](#D005) | p.5 | 四段如何搭积木，以及各自塞进哪个矩阵 |
| [D6](#D006) | p.6 | 用 $2\times2$ 例算出 $\Delta R$ 就是一个旋转矩阵；四个矩阵的来源与 $R_1,R_2$ 的六步迭代流程 |
| [D7](#D007) | p.19 | 为什么没有量化器梯度就是 0，以及两项误差来源 |

## 术语对照表（Terminology Ledger）

| 英文（canonical） | 中文 | 说明 / 决定 |
|---|---|---|
| PTQ (post-training quantization) | 训练后量化 | 全文统一；不使用“离线量化” |
| rotation matrix ($R$) | 旋转矩阵 | 符号 $R$ 保留；四个可学习/在线旋转分别记作 $R_1$–$R_4$ |
| SpinQuant$_{no\ had}$ | SpinQuant$_{no\ had}$（不含在线 Hadamard） | 保留英文方法名与下标，不音译 |
| SpinQuant$_{had}$ | SpinQuant$_{had}$（含在线 Hadamard） | 同上 |
| outlier | 离群值 | 不使用“异常值”，以与量化文献一致 |
| activation | 激活 | — |
| RMSNorm (root mean square normalization) | 均方根归一化 | 不减均值；定义与手算见 §0.5 |
| residual connection / residual stream | 残差连接 / 残差流 | 主干 $h\leftarrow h+f(h)$；“整条残差流一起旋转”指对 $h$ 右乘 $R_1$；见 §0.6 |
| RoPE (rotary position embedding) | 旋转位置编码 | 按 token 位置把 $q,k$ 旋转一个角度；与 SpinQuant 的 $R$ 无关，见 §0.7 |
| KV cache | KV cache（键值缓存） | 首次出现给中文，其后用 KV cache |
| perplexity (PPL) | 困惑度 | — |
| zero-shot commonsense reasoning | 零样本常识推理 | — |
| quantization error | 量化误差 | — |
| kurtosis $\kappa$ | 峰度 $\kappa$ | — |
| Cayley SGD / Cayley transform | Cayley SGD / Cayley 变换 | 人名保留 |
| Stiefel manifold | Stiefel 流形 | 人名保留 |
| orthonormal / orthogonal | 正交（规范正交） | orthonormal 统一译“正交”，必要时加注 |
| Hadamard matrix/rotation | Hadamard 矩阵/旋转 | 人名保留；元素 $\pm1/\sqrt d$，可用快速变换，构造与例子见 §0.8 |
| W4A4KV4 等 | W4A4KV4 等 | 位宽缩写不翻译；W-A-KV 依次为权重、激活、KV-cache |
| GPTQ / RTN / AWQ / QuaRot / QuIP# | GPTQ / RTN / AWQ / QuaRot / QuIP# | 方法名保留原文 |
| SNR | 信噪比（SNR） | 首次出现给中文 |
| token-wise / channel-wise | 逐 token / 逐通道 | — |
| mergeable rotation | 可合并旋转 | 指可吸收进权重矩阵的旋转 |
| online rotation | 在线旋转 | 推理时需实时计算的旋转 |
| round-to-nearest (RTN) | 最近取整（RTN） | — |
| calibration set | 校准集 | — |

## 阅读提示（critical reading notes）

1. **核心论点**：旋转不改变全精度输出（旋转不变性），但显著改变量化误差；因此“选择/学习哪个旋转”是一个可优化的自由度。论文把它变成在 Stiefel 流形上的损失最小化问题，而不是启发式地挑 Hadamard 矩阵。
2. **证据链**：Figure 2–3（离群值与峰度）→ Figure 4（随机旋转的 13 分方差）→ Table 1（7 模型 × 4 比特配置）→ Table 2/4（学习旋转 vs 随机/Hadamard）→ Table 5（vs QuaRot）→ Table 6/14（速度代价）。逻辑是“现象→方法→主结果→代价”。
3. **需要留意的点**：(i) 旋转优化只用 800 条 WikiText2 样本、100 次迭代，代价小但依赖校准集分布；(ii) 主结果中权重用 GPTQ、旋转针对“仅激活量化”的网络学习，这一选择由 Table 3 支撑；(iii) W4A4KV4 的 2.9 分差距是在 LLaMA-2 7B 上，70B 为 1.7 分，但 LLaMA-3 8B 等“难量化”模型差距更大（这也是相对 QuaRot 提升 45.1% 的来源）；(iv) 在线 Hadamard 带来约 8% 延迟，属于精度/速度权衡。
4. **优化范围只到 $R_1,R_2$**：论文里“学到的旋转更好”是针对 $R_1,R_2$ 的实验结论（Figure 4、Table 2/4）；$H_3,H_4$ 是固定 Hadamard，不参与优化，因此不存在“S3/S4 最优”的声明，它们的作用是“在可接受的在线开销下进一步压离群值”。读 Table 1 时要注意：开/关 $H_3,H_4$ 的差别只在 W4A4/KV4 这类极端设置里明显，W4A8 下基本无差（具体数字见 D4 的“S3/S4 到底有没有必要”）。
5. **可复现性**：超参见 §4.1（$R_1$ 形状 $(D_{token},D_{token})$、$R_2$ 形状 $(D_{head},D_{head})$、lr 1.5 线性衰减、100 次迭代）；代码 github.com/facebookresearch/SpinQuant。
6. **推导位置**：D1–D7 是本阅读器补充的推导（非论文原文），分别落在 §2.1 末、§3.1 的 S1/S2/S3+S4/合成四处、§3.2 的 Cayley 更新处与附录 B.1 末。写法是“最小例子 + 一句话结论”，只需要矩阵乘法、转置、单位矩阵三样东西；想看完整证明（含引理、边界与数值自检脚本）再翻 `DERIVATION_PACKAGE.md`。
7. **局限（作者未单列 Limitations 章节）**：Appendix 提到首 token 的大幅值通道在旋转后被摊平，但若能在旋转前削减这些离群值可能进一步提升精度——作者将其列为未来方向。

:::
