---
title: "量化从零开始：从仿射量化到 FP4 的工程笔记"
published: 2026-10-26T12:30:00+08:00
description: 用 NumPy 和 PyTorch 从零实现 affine quantizer、校准、PTQ/QAT、AdaRound/GPTQ/AWQ/SmoothQuant，并在玩具网络上验证 6 dB/bit、outlier、FP8/FP4 与 KV cache 的真实取舍。
tags: [Quantization, PyTorch, 推理优化, FP8]
category: Quantization
draft: false
---

把一个 7B 模型从 FP16 压到 INT4，权重显存会从 14 GB 降到 3.5 GB。这个账很好算，也很容易让人误以为量化就是“把每个数砍掉几位”。

真正麻烦的是另外三件事：误差从哪里来，为什么同样 4-bit 有的层没事、有的层直接崩，以及省下来的 bit 到底变成了显存、带宽还是更快的 kernel。这篇笔记从一条实数轴开始，走完仿射量化、校准、PTQ/QAT、GPTQ/AWQ/SmoothQuant，最后落到 FP8/FP4 和 KV cache。

所有实验来自一份可运行的 notebook：数据和网络都是刻意做小的，CPU 几秒跑完。它们不是模型榜单，作用是让公式和现象对得上。

## 先算账：省的是显存，还是带宽？

权重显存可以粗略写成：

$$
\text{weight bytes} \approx N_{\text{params}} \times \frac{b}{8}
$$

7B 模型在 FP16 下约 14 GB，INT8 约 7 GB，INT4 约 3.5 GB。但压缩率不是简单的 $32/4=8$ 倍，因为每组量化参数、没有量化的 bias / LayerNorm / embedding、对齐 padding 和反量化缓冲区都要占地方。INT4 如果每 128 个权重共享一个 FP16 scale，有效位宽是 $4+16/128=4.125$ bit，70B 模型的 scale 本身仍有约 1 GB。

| 格式 | 有效 bit/权重 | 7B 权重 | 70B 权重 |
|---|---:|---:|---:|
| FP16 / BF16 | 16.0 | 14.00 GB | 140.00 GB |
| INT8 | 8.0 | 7.00 GB | 70.00 GB |
| INT4 | 4.0 | 3.50 GB | 35.00 GB |
| INT4, group=128 | 4.125 | 3.61 GB | 36.09 GB |

只算权重还不够。KV cache 的容量是：

$$
\text{KV bytes} = 2 \times B \times L \times H \times D \times T \times \frac{b_{\text{kv}}}{8}
$$

其中 $2$ 是 K 和 V，$B$ 是 batch，$L$ 是层数，$H$ 是 KV head 数，$D$ 是 head_dim，$T$ 是上下文长度。Llama-2 70B（80 层、64 head、head_dim=128、batch=1）在 4K context 下 FP16 KV cache 约 10.7 GB，32K 时约 85.9 GB。即使权重压到 36 GB，32K 上下文的 KV cache 仍可能比权重更大。

这也解释了大模型推理里一个很实际的现象。**prefill** 一次处理整段 prompt，矩阵乘法的 batch 很大，通常更接近 compute-bound；**decode** 每生成一个 token 都要把全部权重读一遍，batch 往往很小，通常是 memory-bound。decode 的算术强度只有“每个权重做一两次乘加”，GPU 算力大量闲置，真正卡住的是 HBM 到 SM 的带宽。把权重从 FP16 压到 INT4，即使计算前仍要反量化到 FP16，搬运量也少了 4 倍，所以 **weight-only 量化（W4A16）在 decode 上可能有很大收益，在 prefill 或大 batch 下收益有限**。

这里要区分三件经常被混为一谈的事：

- **存储格式**：权重在显存里占几个 bit；
- **计算内核**：矩阵乘法是否真的用了整数或低比特指令；
- **模拟量化**：训练或分析时用浮点 QDQ 模拟误差，速度不一定变快。

## 一个实数，怎么变成整数？

最常用的是仿射量化。给定浮点张量 $x$，把它映射到 $b$ 比特的整数码字：

$$
q = \operatorname{clamp}\left(\operatorname{round}\left(\frac{x}{s}\right) + z,\; q_{\min},\; q_{\max}\right)
$$

反量化是：

$$
\hat{x} = s\,(q-z)
$$

$s>0$ 是步长，$z$ 是零点，$\operatorname{clamp}$ 负责处理越界值。我们希望把 $[x_{\min},x_{\max}]$ 线性铺满 $[q_{\min},q_{\max}]$，于是：

$$
s = \frac{x_{\max}-x_{\min}}{q_{\max}-q_{\min}},
\qquad
z = \operatorname{round}\left(q_{\min} - \frac{x_{\min}}{s}\right)
$$

如果分布关于 0 对称，可以强制 $z=0$，只留一个 scale：

$$
s = \frac{\max |x|}{q_{\max}},
\qquad
\hat{x} = s\,q
$$

权重通常用对称量化，因为它们大多以 0 为中心；ReLU 之后的激活通常用非对称无符号量化，避免把一半码字浪费在永远不会出现的负数上。

<details>
<summary>核心实现（NumPy）</summary>

```python
def qparams(x, bits=8, axis=None, symmetric=False):
    if symmetric:                      # 权重：零点固定为 0
        qmin, qmax = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        scale = np.max(np.abs(x), axis=axis, keepdims=True) / qmax
        zero = np.zeros_like(scale)
    else:                              # ReLU 激活：用满 [0, 2^b-1]
        qmin, qmax = 0, (1 << bits) - 1
        xmin = np.min(x, axis=axis, keepdims=True)
        xmax = np.max(x, axis=axis, keepdims=True)
        scale = (xmax - xmin) / (qmax - qmin)
        zero = np.rint(qmin - xmin / np.where(scale == 0, 1.0, scale))

    scale = np.where(scale < 1e-12, 1.0, scale)
    return scale, np.clip(zero, qmin, qmax), qmin, qmax

def fake_quant(x, bits=8, axis=None, symmetric=False):
    s, z, qmin, qmax = qparams(x, bits, axis, symmetric)
    q = np.clip(np.rint(x / s) + z, qmin, qmax)
    return (q - z) * s
```

</details>

拿 $[-1.2,-0.7,0,0.3,2.4]$ 做 4-bit 非对称量化，会得到 $s=0.24$、$z=5$，码字是 $[0,2,5,6,15]$，反量化回来是 $[-1.2,-0.72,0,0.24,2.4]$。$x=0$ 精确落在码字 5 上，两个相邻码字的间距就是 $s$。这个例子里最大误差是 $0.06$，出现在 $0.3$ 附近。

对称和非对称的差别在偏斜分布上最明显。对 ReLU 输出 $[0,0.02,0.1,0.25,0.5,1,2]$ 做 4-bit 量化，非对称 uint4 会把 $0.1$ 量成 $0.133$，误差 $0.033$；对称 int4 把 $[-2,2]$ 铺满 16 个码字，$0.1$ 直接落到 0，误差 $0.1$。**同样 4-bit，码字有没有被浪费，误差可以差三倍。**

### 粒度往往比 bit 数更影响精度

同一个 bit 数下，一个 scale 管多少个数，结果可能差一个数量级：

| 粒度 | 一个 scale 管多少 | 4-bit SQNR |
|---|---|---:|
| per-tensor | 整个张量 | 7.99 dB |
| per-channel | 每个输出通道 | 14.13 dB |
| group=32 | 每 32 个连续元素 | 20.47 dB |

实验里的权重矩阵每行尺度不同，还藏了少量极端值。per-tensor 为了不让 $10.0$ 被截断，把 scale 拉得很大，绝大多数正常权重只剩两三个有效电平；per-channel 解决了行间尺度差；group=32 连行内 outlier 也隔离了。

但更细的粒度不是免费的。group=32 的 scale 数量是 per-channel 的 8 倍，硬件读取 scale 和反量化指令本身也要带宽。LLM 里常见的 $g=128$ 就是精度和开销的折中，有效位宽可以写成：

$$
\text{effective bits} \approx b + \frac{\text{scale bits}}{g}
$$

## 误差从哪里来？

量化误差可以拆成两部分：

$$
x-\hat{x}
=
\underbrace{x-Q_{\text{no-clip}}(x)}_{\text{rounding}}
+
\underbrace{Q_{\text{no-clip}}(x)-\hat{x}}_{\text{clipping}}
$$

rounding 是“值在范围内，但只能落到最近的离散电平”；clipping 是“值超出范围，被硬压到边界”。两者此消彼长：范围收窄，rounding 更细但 clipping 更多；范围放宽，clipping 少但每个电平更粗。

假设没有 clipping，量化误差在 $[-s/2,s/2]$ 内近似均匀，误差方差是 $s^2/12$。信号功率为 $\sigma_x^2$ 时：

$$
\mathrm{SQNR}
= 10\log_{10}\frac{\sigma_x^2}{s^2/12}
= 10\log_{10}12 + 20\log_{10}\frac{\sigma_x}{s}
$$

若满量程为 $R$，$s\approx R/2^b$，就有：

$$
\mathrm{SQNR} \approx 6.02b + 10.79 + 20\log_{10}\frac{\sigma_x}{R}
$$

**每多 1 bit，SQNR 约增加 6 dB。** 常数项取决于信号的峰值因子：对满量程正弦，$20\log_{10}(\sigma_x/R)=-9.03$ dB，公式退化成常见的 $6.02b+1.76$。用 20 万个标准正态样本实测，拟合斜率是 5.97 dB/bit，和理论值很接近。

这也解释了“8-bit 的精度是 4-bit 的两倍”为什么不对。按 SQNR 算，8-bit 比 4-bit 多约 24 dB，噪声功率小约 250 倍；但这是在同一归一化范围、没有 clipping、误差均匀的理想条件下。真实模型里的 outlier 和分布偏移会让这个数字大幅缩水。

### clipping：省出来的范围可能很贵

真实分布往往是重尾的。用 Student-t（自由度 3）模拟激活，比较不同裁剪阈值下的 4-bit MSE：

<img src="/images/quantization/clipping-tradeoff.png" data-light-src="/images/quantization/clipping-tradeoff.png" data-dark-src="/images/quantization/clipping-tradeoff-dark.png" alt="图 1：重尾分布下，min-max 阈值与 MSE 最优阈值的差别。" loading="lazy">

*图 1：min-max 阈值 75.31 时 MSE 为 2.028；MSE 最优阈值 8.37 时只有 0.295，而且只丢掉了 0.37% 的样本。*

min-max 直接把样本最大值当阈值，几乎总是太靠右。少数极端值被保留下来，代价是主体分布只剩很粗的电平。

### outlier 是量化的头号敌人

一个张量里少数几个极大值，会把 per-tensor 的 scale 拉得很大，让其余 99.9% 的值挤在很少几个电平里，这叫 **outlier domination**。LLM 的激活尤其明显：某些通道的幅值可以是其他通道的几十倍，而且会随 token 变化。

所以权重量化通常比激活量化容易：per-channel / group-wise 对权重有效；激活则需要 per-token、SmoothQuant、旋转或 QAT。只看平均绝对误差也不够，outlier 通道的误差会沿着权重传播到输出。

### 校准：scale 不是取 max 就完了

PTQ 不训练模型，只用一小批代表性数据跑一遍，统计激活范围，再决定每层 scale。这个过程叫校准。常见方法有四类：

| 方法 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| min-max | $T=\max_i x_i$ | 简单，绝不截断 | 对 outlier 极敏感 |
| percentile | $T=Q_p(x)$，如 99.9% | 抗 outlier、可解释 | $p$ 要选，可能裁掉重要值 |
| MSE | 网格搜索 $T$，最小化重建 MSE | 直接优化重建误差 | 只优化单层，未必端到端最优 |
| KL | 最小化参考分布与量化分布的 KL | 关注分布形状 | 对分箱和候选范围敏感 |

在一个带 0.5% 极端 outlier 的非负激活上，用 12 万样本校准、另 12 万样本做 holdout，结果很直观：

| 方法 | 阈值 | 校准集截断 | holdout MSE | SQNR |
|---|---:|---:|---:|---:|
| min-max | 21.36 | 0% | 0.169 | 8.72 dB |
| percentile 99.9 | 8.09 | 0.10% | 0.054 | 13.70 dB |
| percentile 99.99 | 14.74 | 0.01% | 0.082 | 11.83 dB |
| MSE search | 8.32 | 0.10% | 0.053 | 13.76 dB |
| KL calibration | 4.09 | 0.27% | 0.103 | 10.85 dB |

<img src="/images/quantization/calibration.png" data-light-src="/images/quantization/calibration.png" data-dark-src="/images/quantization/calibration-dark.png" alt="图 2：重尾激活的分布，以及不同裁剪阈值下的校准 MSE。" loading="lazy">

*图 2：min-max 被单个 outlier 拖到 21 附近；MSE 最优在 8.3 左右；KL 在这组数据上反而偏激进。*

几个结论：

- **min-max 最差**，一个 20 多的 outlier 就能毁掉整层的有效分辨率；
- **99.9% 分位数和 MSE 搜索接近**，都裁掉了约 0.1% 的极端值；
- **KL 不一定最好**，它优化的是分布形状，不是任务损失，分箱方式和候选范围都会影响结果；
- 校准集要足够大、覆盖真实分布，**测试集不能参与选阈值**。

实践上可以先用 percentile 99.9 / 99.99 做 baseline，对敏感层再用 MSE 搜索。静态校准把 scale 固定写进模型，kernel 简单；动态量化在推理时按 token 现算 scale，更抗分布偏移，但多一次 reduce。LLM 的激活常用 per-token dynamic。

## 把量化放进网络：PTQ 的第一课

单张量实验只能看局部。现在训练一个三分类螺旋数据上的小 MLP（2 → 64 → 64 → 3），FP32 测试精度 99.3%，然后逐层替换权重：

| 权重 bit | 2 | 3 | 4 | 5 | 8 |
|---|---:|---:|---:|---:|---:|
| test acc | 28.7% | 40.2% | 79.5% | 98.7% | 99.3% |

8-bit 几乎无损，5-bit 开始明显下降，4-bit 掉到约 80%，2-bit 基本失效。但这不是“4-bit 不行”的普适结论——真实 LLM 的权重冗余更强，配上 GPTQ/AdaRound 或 QAT，4-bit 可以很好。这里先看到的是**误差确实会传播**。

再同时量化激活。一个配方通常写成 **W?A?**：前一个数字是权重 bit，后一个是激活 bit。所有配置都不训练，激活 scale 来自训练集，结果如下：

| 配置 | 权重 | 激活 | test acc | 相对 FP32 |
|---|---|---:|---:|---:|
| FP32 | 32 | 32 | 99.3% | — |
| W4A16 | 4 | 16 | 79.5% | -19.8 |
| W8A8 | 8 | 8 | 99.2% | -0.2 |
| W4A8 | 4 | 8 | 79.7% | -19.7 |
| W8A4 | 8 | 4 | 77.7% | -21.7 |
| W4A4 | 4 | 4 | 66.0% | -33.3 |

<img src="/images/quantization/ptq-bits.png" data-light-src="/images/quantization/ptq-bits.png" data-dark-src="/images/quantization/ptq-bits-dark.png" alt="图 3：静态 PTQ 下，权重和激活 bit 数对测试精度的影响。" loading="lazy">

*图 3：在这个小 MLP 上，4-bit 权重是主要瓶颈；8-bit 激活几乎没有额外损失；两边都粗时误差会叠加。*

几个观察：

1. **8-bit 激活不自动等于安全**。这里安全，是因为 MLP 的激活范围不算极端；换成 LLM 的 outlier 激活，静态 per-tensor W8A8 完全可能掉很多。
2. **W4A16 和 W4A8 几乎一样**。把激活从 FP16 降到 8-bit 没有引入多少额外误差，但带宽和算力收益可能很实际。这正是 weight-only 量化在 decode 中流行的原因。
3. **W4A4 不是两个独立误差的简单相加**。误差会叠加、放大，甚至改变决策边界。
4. **bit 数不是唯一变量**。per-tensor → per-channel、static → dynamic、min-max → percentile，都会改变同一组 bit 数的结果。

## QAT 与 STE：让梯度穿过 round

PTQ 的极限很清楚：权重和激活都被固定成离散值，模型没有机会适应。QAT 在训练的前向里模拟量化，让优化器看到量化后的损失，从而调整权重。

问题是 `round` 的导数几乎处处为 0，在跳变点不可导。工程上的解法是 **STE（Straight-Through Estimator）**：

- 前向照常做 `round + clamp + dequantize`；
- 反向假装 `round` 是恒等映射，把上游梯度直接传给输入；
- 对超出范围的值，让梯度为 0，因为 clamp 已经饱和。

写成公式：

$$
\text{forward:}\quad \hat{x}=s\cdot\operatorname{clamp}(\operatorname{round}(x/s),q_{\min},q_{\max})
$$

$$
\text{backward:}\quad
\frac{\partial L}{\partial x}
\approx
\frac{\partial L}{\partial \hat{x}}\cdot
\mathbf{1}[q_{\min}\le x/s\le q_{\max}]
$$

STE 是有偏的：它优化的不是真实量化函数的梯度，而是一个 surrogate。但实践中非常有效，是现代 QAT 的基础。

<details>
<summary>STE 的核心实现（PyTorch）</summary>

```python
class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, qmin, qmax):
        ctx.qmin, ctx.qmax = qmin, qmax
        ctx.save_for_backward(x)
        return torch.clamp(torch.round(x), qmin, qmax)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        in_range = (x >= ctx.qmin) & (x <= ctx.qmax)
        return grad_out * in_range.to(grad_out.dtype), None, None
```

</details>

在 W4A16 的设置下，用固定 scale 的 fake quant 包住三个线性层，再从 FP32 模型继续训练 500 个 epoch：

| | FP32 | W4 PTQ | W4 QAT |
|---|---:|---:|---:|
| test acc | 99.3% | 79.5% | 98.3% |

<img src="/images/quantization/qat-recovery.png" data-light-src="/images/quantization/qat-recovery.png" data-dark-src="/images/quantization/qat-recovery-dark.png" alt="图 4：4-bit 权重的 QAT 恢复效果。" loading="lazy">

*图 4：QAT 把 4-bit 权重的测试精度从 79.5% 拉回 98.3%，接近 FP32 的 99.3%。*

前向的权重始终是 4-bit 可表示的值，反向通过 STE 把梯度传给全精度权重，优化器把权重移动到“量化后仍然好用”的位置。scale 固定，但权重可以重新分布到量化格点上。

QAT 有几个关键细节：

- 通常从预训练模型微调，而不是从头训练；学习率要小（原训练的 1/10～1/100），步数也远少于原训练。
- 必须模拟推理图。Conv-BN 要在训练时 fold，或者在导出前 fold，否则训练和部署的数值不一致。
- scale 也可以学习。LSQ 把 $s$ 变成可训练参数，在 STE 下 $\partial\hat{x}/\partial s\approx q$；PACT 则学习 clipping 阈值。
- 激活 QAT 通常比权重 QAT 更敏感，因为激活随输入变化，outlier 会移动。
- QAT 不是免费的：需要训练代码、标签和算力，还可能过拟合校准分布。PTQ 加好的校准往往已经够用。

## 逐层重建：不要只优化权重空间

PTQ 的核心问题可以写成：给定一层输入 $X$ 和权重 $W$，怎样选量化后的 $\hat{W}$，让输出 $X\hat{W}^\top$ 尽量接近 $XW^\top$？

$$
\min_{\hat{W}}\;\|XW^\top-X\hat{W}^\top\|_F^2
=
\min_{\hat{W}}\;\operatorname{tr}\left((W-\hat{W})H(W-\hat{W})^\top\right),
\qquad H=X^\top X
$$

这比“让 $\hat{W}$ 尽量接近 $W$”更合理，因为不是所有方向的权重误差都同等重要：输入协方差 $H$ 会放大某些方向。

从一个训练好的 MLP 第 2 层（64×64）取真实激活做实验，4-bit 的结果：

| 方法 | 改变什么 | 层相对误差 | 全模型精度 |
|---|---|---:|---:|
| FP32 | — | 0 | 99.3% |
| RTN | 每个权重独立取最近整数 | 12.6% | 83.7% |
| AdaRound | 每个权重的取整方向 | 3.4% | 99.0% |
| GPTQ | 未量化列的误差补偿 | 7.5% | 93.2% |

<img src="/images/quantization/layer-reconstruction.png" data-light-src="/images/quantization/layer-reconstruction.png" data-dark-src="/images/quantization/layer-reconstruction-dark.png" alt="图 5：第 2 层 4-bit 量化后的输出相对误差。" loading="lazy">

*图 5：RTN 的每个权重都“离原值最近”，但整层输出误差不是最小；AdaRound 和 GPTQ 优化的是输出空间。*

### AdaRound：四舍五入到让输出误差最小

RTN 对每个权重独立取最近整数。但量化误差会经过输入协方差传播，“每个权重都离原值最近”并不等于“整层输出误差最小”。AdaRound 把取整方向看成一个二元选择：

$$
\hat{W}=s\left(\lfloor W/s\rfloor+\Delta\right),
\qquad \Delta\in\{0,1\}
$$

直接优化 0/1 变量是离散优化，所以用连续松弛：

$$
h(v)=\operatorname{clamp}\left(\sigma(v)(\zeta-\gamma)+\gamma,\;0,\;1\right),
\qquad \gamma=-0.1,\;\zeta=1.1
$$

目标函数是重建误差加上一个把 $h$ 推向 0/1 的正则项：

$$
\min_v\;\|XW^\top-X\hat{W}(v)^\top\|_F^2
+\lambda\sum_{i,j}\left(1-\left|2h_{ij}(v)-1\right|^\beta\right)
$$

训练结束后用 $h>0.5$ 得到最终的 0/1 选择。AdaRound 只改变“每个权重往哪边取整”，不改变 scale，也不改变量化格点。

### GPTQ：把当前列的误差补偿给后面的列

GPTQ 从 OBQ 演化而来，逐列量化权重，并把当前列的量化误差补偿到还没量化的列上。目标仍然是 $\operatorname{tr}((W-\hat{W})H(W-\hat{W})^\top)$。假设按输入维度 $j=1,\dots,d$ 顺序量化，当前列的误差为：

$$
\delta_j=\frac{w_j-\hat{w}_j}{[H^{-1}]_{jj}}
$$

剩余列更新为：

$$
w_{j+1:}\;\leftarrow\;w_{j+1:}-\delta_j\,[H^{-1}]_{j,j+1:}
$$

直观理解：当前列犯的错，会在后续列里“反向拉回来”，让整体二次型误差下降。所有输出通道可以并行做，因为 $H$ 只由输入决定。

<details>
<summary>教学版 GPTQ 的核心循环</summary>

```python
# w 是 W 的可写副本；s 是每行的 scale
w = W.copy()
H = X.T @ X / len(X)
H += damp * np.mean(np.diag(H)) * np.eye(d)   # damping，缓解病态
Hinv = np.linalg.inv(H)

for j in range(d):
    q = np.clip(np.rint(w[:, j] / s), -qmax - 1, qmax)
    wq = q * s
    err = (w[:, j] - wq) / Hinv[j, j]
    w[:, j + 1:] -= np.outer(err, Hinv[j, j + 1:])
```

</details>

真实 GPTQ 还会做 Cholesky 分解、act-order（按激活重要性重排输入列）、分块更新和 group-wise scale。教学版只保留更新公式，方便看清机制。

### AWQ：保护与大幅值激活相连的权重通道

AWQ 的出发点是：不是所有权重通道同等重要，和大幅值激活相连的输入通道更值得保护。它利用一个恒等变换：

$$
Y=XW^\top=\left(X\,\mathrm{diag}(s)^{-1}\right)\left(W\,\mathrm{diag}(s)\right)^\top
$$

把输入通道 $j$ 除以 $s_j$，把权重第 $j$ 列乘以 $s_j$，输出完全不变，但量化误差的分配变了。AWQ 根据激活幅值搜索 $s$，让重要通道的权重量化更精细。

在一个刻意构造的合成层上，RTN 的测试集相对误差是 0.226，AWQ 式缩放把它降到 0.159，cosine 从 0.978 升到 0.988。最佳 $\alpha$ 是 0.15。**这个数字只说明机制有效，不代表真实模型上的收益**；真实 AWQ 还会按 group 搜索、做 weight clipping，并在校准集上最小化输出误差。

### SmoothQuant：把激活 outlier 迁移到权重

SmoothQuant 针对的是 LLM 激活的通道级 outlier，用的是同一个恒等变换，但目标不同：AWQ 主要保护权重，SmoothQuant 主要让激活更容易量化。它选择：

$$
s_j=\frac{\max|X_j|^\alpha}{\max|W_j|^{1-\alpha}},
\qquad \alpha\in[0,1]
$$

$\alpha=0$ 时全部难度留给激活，$\alpha=1$ 时全部迁移到权重，$\alpha=0.5$ 时两边各承担一半。

<img src="/images/quantization/smoothquant.png" data-light-src="/images/quantization/smoothquant.png" data-dark-src="/images/quantization/smoothquant-dark.png" alt="图 6：SmoothQuant 的 alpha 搜索与 outlier 迁移。" loading="lazy">

*图 6：合成层上 W8A8 相对误差从 6.53% 降到 1.80%，最佳 alpha=0.8；右图能看到通道 7 的尖峰被摊平。*

关键不是“数学上等价”本身，而是等价变换改变了量化误差的分配：

- 原来激活的一个通道幅值是 20，其他通道约 1，per-tensor scale 被它支配；
- 平滑后这个通道被缩小，权重对应列被放大；
- 权重是 per-channel 量化的，放大主要影响该通道自己的 scale，不会像 per-tensor 那样拖累其他通道；
- 激活的最大值下降，per-tensor scale 变细。

真实部署时，$s$ 必须 fold 进前一层，否则会多一次逐通道乘法；$\alpha$ 要在验证集上选；LayerNorm、残差连接和不同 attention 头的 fold 位置都要仔细处理。

### 旋转：让 outlier 摊到所有通道

另一个思路是正交旋转。对任意正交矩阵 $R$：

$$
Y=XW^\top=(XR)(WR)^\top
$$

输出不变，但激活的坐标变了。Hadamard 矩阵（归一化后）是一种常用的快速正交变换：

$$
H_2=\frac{1}{\sqrt2}\begin{bmatrix}1&1\\1&-1\end{bmatrix},
\qquad
H_{2^k}=H_2^{\otimes k}
$$

在同一个合成层上，不旋转、Hadamard 旋转、随机正交旋转的结果：

| 变换 | 最大激活 | kurtosis | W8A8 相对误差 |
|---|---:|---:|---:|
| 不旋转 | 35.54 | 58.69 | 6.53% |
| Hadamard | 6.18 | 3.00 | 1.32% |
| 随机正交 | 8.10 | 3.34 | 1.63% |

Hadamard 把最大激活从 35 降到 6 左右，kurtosis 从几十降到 3 附近（高斯约为 3）。QuaRot、SpinQuant 这类方法会在模型里插入旋转，再配合 GPTQ 等量化算法。代价是旋转必须 fuse 进相邻线性层，否则推理会多出矩阵乘法；旋转也会改变权重分布，通常还要重新校准。

### 方法地图

| 方法 | 改什么 | 需要校准 | 需要训练 | 典型场景 |
|---|---|---|---|---|
| RTN | 取整 | 否 | 否 | baseline、权重量化 |
| percentile / MSE | 裁剪阈值 | 是 | 否 | 激活校准 |
| AdaRound | 取整方向 | 是 | 否（局部优化） | 4-bit 权重 PTQ |
| GPTQ | 未量化列的补偿 | 是（Hessian） | 否 | LLM W4/W3 |
| AWQ | 通道缩放 | 是 | 否（搜索） | LLM W4 weight-only |
| SmoothQuant | 激活→权重迁移 | 是 | 否 | LLM W8A8 |
| QuaRot / SpinQuant | 正交旋转 | 是/学习 | 可选 | 极端 outlier、KV cache |
| QAT / LSQ | 权重和 scale | 训练集 | 是 | ≤3-bit、严苛精度 |

这些方法不是互斥的。一个典型的 LLM 4-bit pipeline 可能是：**旋转 → SmoothQuant/AWQ 缩放 → GPTQ → 校准评估**。

顺便说一句，压缩不只有量化。剪枝、蒸馏、低秩是正交路线：非结构化剪枝把权重变成很多 0，但普通 dense 矩阵乘法不会因此变快；低秩分解需要低秩 kernel，否则可能多出额外的 launch 开销。**参数量减少不等于推理加速**，量化也一样。

## FP8 / FP4：换一种网格

前面所有量化器都建立在整数网格上：步长均匀，$q=\operatorname{round}(x/s)+z$。FP8/FP4 换了一条路：让绝对步长随数值大小按 2 的幂变化。这带来两个直接后果：

- 相对精度更均匀：大值和小值用不同的绝对步长，小值不会因为一个全局 scale 被整体压成 0；
- 动态范围靠指数位撑起来，尾数位决定精度。

代价是硬件要做浮点解码、分块缩放和更高精度的累加；而且 **FP8/FP4 仍然需要 scale**，只是 scale 的粒度和存放方式与 INT 不同。

| 格式 | S/E/M | 最大有限值 | 最小 normal | 特殊值 |
|---|---|---:|---:|---|
| E4M3 (FP8) | 1/4/3 | 448 | $2^{-6}=0.015625$ | 2 个 NaN，无 Inf |
| E5M2 (FP8) | 1/5/2 | 57344 | $2^{-14}=6.10\times10^{-5}$ | 2 个 Inf，6 个 NaN |
| E2M1 (FP4) | 1/2/1 | 6 | 1 | 无 Inf/NaN |

几个容易踩的点：

- **E4M3 没有 Inf**：超过 448 只能饱和，`0x7F/0xFF` 是 NaN。它是推理最常用的 FP8 格式。
- **E5M2 范围大但精度低**：5 位指数覆盖到 57344，2 位尾数意味着一个 binade 只有 4 个台阶；它更适合梯度，不太适合权重/激活。
- **E2M1 只有 7 个非零正电平**：$0.5,1,1.5,2,3,4,6$。没有 block scale 时，FP4 对 LLM 基本不可用。

<img src="/images/quantization/fp-grid.png" data-light-src="/images/quantization/fp-grid.png" data-dark-src="/images/quantization/fp-grid-dark.png" alt="图 7：INT8、E4M3、E2M1 的可表示电平，以及同一组数上的相对误差。" loading="lazy">

*图 7：INT8 是等距网格，小值没有电平只能变成 0；E4M3 的相对误差几乎水平；E2M1 在 per-tensor 下大部分小值直接归零。*

这张图不是说 FP8 一定比 INT8 好。如果数据分布很窄、没有 outlier，均匀网格的 INT8 反而更划算；FP8 的优势要在“跨数量级”和“重尾”出现时才兑现。

### 分块缩放：FP4 的救命稻草

FP8/FP4 的 scale 不是可选项。以 per-tensor FP8 为例：

$$
s=\frac{\max|x|}{448},
\qquad
q=\mathrm{FP8}(x/s),
\qquad
\hat{x}=q\cdot s
$$

per-tensor 的 scale 仍然会被一个 outlier 拉大；把张量切成 block，每块单独算 scale，就能把 outlier 关在局部。DeepSeek-V3 的 FP8 训练就是一个典型：权重按 $128\times128$ 分块，激活按 $1\times128$ 分块，scale 用 FP32，矩阵乘加在 FP32 中累加。

4-bit 的 MX/NV 格式把 block scale 写进了格式本身：

| 格式 | 元素 | block 大小 | block scale | scale 开销 |
|---|---|---|---:|---:|
| FP8 per-tensor | E4M3 | 整个张量 | FP32 | ≈0 |
| FP8 block=128 | E4M3 | 128 | FP32 | 0.25 bit/elem |
| INT4 group=32 | INT4 | 32 | FP16 | 0.5 bit/elem |
| MXFP4 | E2M1 | 32 | E8M0（2 的幂） | 0.25 bit/elem |
| NVFP4 | E2M1 | 16 | E4M3 + FP32 全局 | 0.5 bit/elem |

MXFP4 的 block scale 是 2 的幂，通常向上取整，避免最大值饱和：

$$
s_{\text{block}}=2^{\lceil \log_2(\max_{\text{block}}|x|/6)\rceil}
$$

NVFP4 用两级缩放：16 个元素共享一个 E4M3 block scale，再加一个 per-tensor 的 FP32 全局 scale，把 E4M3 scale 也放进可表示范围。

这里有个常被忽略的账：**“4-bit 模型”不是 4.0 bit/权重。** MXFP4 约 4.25，NVFP4 约 4.5，INT4 group=32 约 4.5。比较格式时要在相同的有效 bit 下比，而不是只看元素位宽。

### 同一个张量，不同格式

在一个刻意构造的重尾权重上（每行尺度跨约 3 个数量级，另有 0.05% 的 40 倍 outlier），各格式的 weight SQNR：

<img src="/images/quantization/format-sqnr.png" data-light-src="/images/quantization/format-sqnr.png" data-dark-src="/images/quantization/format-sqnr-dark.png" alt="图 8：同一个重尾权重张量上，不同格式与 scale 粒度的 SQNR。" loading="lazy">

*图 8：INT8 per-tensor 只有 12.4 dB；FP8 E4M3 per-tensor 31.9 dB，block=128 34.5 dB；4-bit 一档里 NVFP4 21.3 dB、INT4 group=32 19.2 dB、MXFP4 17.5 dB。*

读这张图要注意：

- **INT8 per-tensor 只有 12.4 dB**：一个全局 amax 被 outlier 拉大，主体权重的有效位数被吃掉；
- **FP8 E4M3 per-tensor 31.9 dB**，比 INT8 per-tensor 高约 19 dB：浮点网格对大小值一视同仁；
- **E5M2 比 E4M3 低约 6 dB**：2 位尾数买来的范围，在权重量化上通常是亏的；
- **FP8 block=128 34.5 dB**：分块把 outlier 关在局部，是 DeepSeek 式 FP8 的默认做法；
- **4-bit 一档**：NVFP4（4.5 bit）21.3 dB > INT4 group=32（4.5 bit）19.2 dB > MXFP4（4.25 bit）17.5 dB。

**这是这颗张量的结果，不是普适排名。** 换成高斯或均匀分布，INT4 可能反超。格式选择和 scale 粒度必须一起选；说“FP8 比 INT8 好”或“FP4 比 INT4 好”都不完整，必须带上 per-tensor/per-block、block 大小、有效 bit 和分布。

### 端到端：格式只是第一步

把同样的格式套到前面的三层 MLP 上，只量化权重、激活保持 FP32：

- FP8 per-tensor 的 test acc 从 99.3% 掉到 97.3%，logit 相对误差 0.18；改成 block=32 后回到 99.3%，logit 误差 0.12；
- 4-bit 一档掉到 86.3%～88.3%，logit 误差 0.43～0.54。这颗网络很小，FP4 的误差会被 ReLU 和分类边界放大；这不代表 FP4 在 LLM 上一定这么差，但说明 4-bit 需要更认真的 recipe；
- 测试集只有 600 个样本，一两个百分点的差异在噪声范围内，MXFP4 与 NVFP4 的排名不要从这张表下结论。

### KV cache：FP8/FP4 最直接的收益

长上下文推理里，KV cache 往往比权重更早成为瓶颈。以 Llama-2 70B（80 层、64 head、head_dim=128、batch=1）为例，把 K/V 从 FP16 换到 FP8/FP4，并计入 block scale 开销：

| KV 格式 | 有效 bit/elem | 4K KV cache | 32K KV cache | 相对 FP16 |
|---|---:|---:|---:|---:|
| FP16 | 16.00 | 10.74 GB | 85.90 GB | 1.00 |
| FP8, scale/128 | 8.25 | 5.54 GB | 44.29 GB | 0.52 |
| INT8, scale/128 | 8.25 | 5.54 GB | 44.29 GB | 0.52 |
| MXFP4, E8M0/32 | 4.25 | 2.85 GB | 22.82 GB | 0.27 |
| NVFP4, E4M3/16 | 4.50 | 3.02 GB | 24.16 GB | 0.28 |

FP8/INT8 的存储都是 1 byte/elem，差别在 kernel 和数值行为；MXFP4 的元素是 0.5 byte，但每 32 个元素要带一个 8-bit E8M0 scale；NVFP4 每 16 个元素带一个 E4M3 scale，再加一个可忽略的 FP32 全局 scale。表里只算容量，没有算 scale 的读取、对齐和 kernel 开销；真实 KV 量化常用 per-head/per-token scale，结果会不同。

### 工程现实

FP8/FP4 的收益最终取决于硬件和 kernel：

- **存储/带宽**：FP8 是 1 byte/elem，FP4 是 0.5 byte/elem（packed）。如果 kernel 不支持 packed 格式，权重仍按 FP16 存，显存不会降。
- **计算**：数据中心 GPU 从 Hopper 开始有原生 FP8，Blackwell 开始有原生 FP4。更老的卡或部分消费卡可能只能模拟，数值上能验证，速度不会变快。
- **scale layout**：block scale 要按 tensor core 的要求 swizzle/对齐；scale 读取得太碎会成为新的带宽瓶颈。
- **累加精度**：FP8/FP4 的乘加通常累加到 FP16/FP32；累加精度不够时，长 K 维的误差会累积。
- **混合精度**：embedding、lm_head、LayerNorm、attention 的 softmax/输出往往保留 BF16/FP16；不要为了平均 bit 数把敏感层一起压。

FP8 训练和 FP8 推理也不是同一套 recipe：训练常用 delayed scaling（用历史 amax 决定当前 scale）或 current scaling，推理更常用 per-tensor 静态权重 + per-token 动态激活；DeepSeek-V3 则用 blockwise FP8，并在 FP32 中累加。FP4 训练目前更依赖 RHT（随机 Hadamard 变换）、stochastic rounding、2D block scaling 和保留敏感层；FP4 推理直接做 PTQ 也往往要配合旋转、GPTQ/AWQ 或混合精度，不能只靠一次 cast。

## 工程清单与选型

最后把容易踩的坑压缩成一份清单。

**校准**

- **不要**用测试集选 scale 或 $\alpha$；
- 校准集太小、分布单一，或只用短文本校准却部署长上下文；
- 只校准一层，忽略量化后激活分布漂移。

**实现**

- 权重 per-channel 的 axis 搞错（应该是输出通道）；
- 对称量化用了 $[-128,127]$，但 scale 用 127 算，边界处理不一致；
- 忘记 fold BN / LayerNorm 的缩放；
- zero-point 用了浮点，或者没有 clamp 到整数范围；
- QDQ 模拟和真实 kernel 的 rounding / saturation 行为不一致；
- 量化了 bias（通常不应该，bias 很小且对精度敏感）。

**评估**

- 只看 perplexity，不看任务指标；
- 用不同长度的序列比较；
- 忽略 KV cache 的显存和带宽；
- 把“模型文件变小”当成“推理变快”；
- 在 batch=1 的 decode 上测出收益，却在 prefill 场景部署。

**工程**

- 4-bit 权重没有 packed，实际还是 FP16 存储；
- group scale 的读取成为新瓶颈；
- kernel 只支持 per-channel，不支持 per-group；
- 混合精度导致额外 kernel launch，抵消了带宽收益。

如果要从零开始选一个方案，可以按场景查表：

| 场景 | 优先尝试 |
|---|---|
| 只想省 LLM 显存，decode 为主 | W4A16 + GPTQ/AWQ，group=128 |
| 服务端要高吞吐 | W8A8 + 好的校准，或 FP8 |
| Blackwell 上想再省一半带宽/显存 | NVFP4（或 MXFP4）+ 分块 scale；敏感层留 FP8/BF16 |
| 激活 outlier 很严重 | SmoothQuant、per-token dynamic、旋转 |
| 4-bit 权重精度不够 | AdaRound/GPTQ，QAT，混合精度 |
| ≤3-bit | QAT（LSQ/PACT）通常必需 |
| 长上下文 KV cache 爆了 | KV INT8/FP8、GQA、分页/稀疏 |
| 权重文件小了但没变快 | 检查是否真的 packed、kernel 是否支持、是否 memory-bound |

量化的本质不是“把浮点变成整数”，而是在**误差、范围、粒度、硬件和校准数据**之间做一组显式的取舍。每一个 bit 的节省，都要问清楚它从哪里来、代价是什么、在哪里验证。
