---
title: "量化论文汇总：从 INT8 到 FP4 的阅读地图"
published: 2026-10-26T22:50:00+08:00
description: 按主题整理 70 篇左右的 LLM 低精度推理论文：基础 PTQ/QAT、outlier、GPTQ/AWQ、旋转 W4A4、KV cache、FP8/FP4、kernel 与评测，附阅读顺序、选型建议和追踪新论文的方法。
tags: [Quantization, 论文, 阅读清单, 推理优化, 低精度]
category: Quantization
draft: false
---

这个分类里已经有两篇长文：一篇从仿射量化一路写到 FP4，另一篇专门讲 Hadamard 与快速变换。它们按“问题”组织，适合从头读；但要查某条线做到哪了、某篇论文值不值得看，还是得自己去 arXiv 翻。这份汇总就是那张地图。

收录 70 篇左右，主线是 LLM / Transformer 的低精度推理，卷积网络时代几篇奠基工作也保留下来——PTQ、QAT、校准、STE 这些零件都是那时候定型的。选择标准不是引用数，而是：对后续工作有影响、有公开代码或可复现实验、能解释一个具体的工程现象。所以它必然有偏，文末会说明偏在哪。

所有数字都是论文自报。硬件、校准集、序列长度、评测框架不同，**不要直接横向比较**；本文只在必要处引用数字，并尽量写成“论文报告”。2026 年的条目还很新，只代表“值得跟”，不代表已经被时间检验。

> **五分钟版**
>
> - **8-bit 基本无痛**：FP8 接近无损，INT8 通常掉 1–3 个点；真正的难点在 4-bit 激活和 KV cache。
> - **4-bit 权重-only 是部署默认**：GPTQ / AWQ + group=128 + Marlin 一类 kernel；decode 收益大，prefill 收益有限。
> - **W4A4 要靠旋转或学习变换**：QuaRot → SpinQuant → DuQuant / FlatQuant；outlier 从激活挪走，代价是 online transform 和 kernel 融合。
> - **长上下文先爆的是 KV cache**：K per-channel、V per-token、pre-RoPE、residual window 是常见配方。
> - **FP4 是 2025–2026 的主线**：推理侧 NVFP4 / MXFP4，训练侧 RHT + 2D scaling + stochastic rounding；硬件支持决定收益。
> - **“lossless” 多半有前缀**：通常是若干 benchmark 上掉点小于 1%，不代表长尾、指令遵循和多轮对话也无损。

## 技术栈一览

| 层 | 要解决的问题 | 代表工作 | 典型代价 |
|---|---|---|---|
| 数值格式 | 用几个 bit、指数位和尾数位怎么分 | FP8 E4M3/E5M2、MXFP4、NVFP4 | 格式越窄，越依赖 block scale 和硬件 |
| 粒度 | 一个 scale 管多少个数 | per-channel、group=128、block=16/32 | 粒度越细越准，scale 读取和 kernel 越复杂 |
| 权重 PTQ | 给定校准集，怎么选量化后的权重 | GPTQ、AWQ、SpQR、OmniQuant | 需要校准数据；低 bit 下可能过拟合校准集 |
| 激活与旋转 | 激活 outlier 怎么摊平 | SmoothQuant、QuaRot、SpinQuant、FlatQuant | online 变换要 fuse，否则白做 |
| KV cache | 长上下文下 K/V 怎么压 | KIVI、KVQuant、GEAR、PolarQuant | 与 attention kernel 强耦合 |
| QAT | PTQ 掉点太多时怎么救 | LLM-QAT、EfficientQAT、BitNet | 训练成本；需要 STE 和量化模拟 |
| 系统 | 省下的字节怎么变成吞吐 | Marlin、Atom、QServe、Nunchaku | dequant 开销、weight packing、layout |
| 评测 | 怎么判断量化模型还能不能用 | ppl、zero-shot、KL/flips、端到端吞吐 | 单一指标几乎总会高估 |

## 三条阅读路线

**只想把模型部署起来**：先看 [LLM.int8](#llm-的-outlier8-bit-为什么也会坏) 理解 outlier，再看 [SmoothQuant](#llm-的-outlier8-bit-为什么也会坏)，然后直接读 [GPTQ](#权重-only-4-bitgptqawq-与它们的后代) 和 [AWQ](#权重-only-4-bitgptqawq-与它们的后代)，KV 部分看 [KIVI / KVQuant](#kv-cache长上下文真正的瓶颈)，最后用 [Give Me BF16](#评测与综述怎么判断一篇量化论文靠不靠谱) 校准预期。kernel 和框架文档比论文更常更新，部署时以 vLLM / TensorRT-LLM 的支持列表为准。

**想做量化算法**：从 [Jacob](#基础整数推理与-ptqqat-的四个零件) 的 integer-only 配方开始，补 [LSQ / AdaRound](#基础整数推理与-ptqqat-的四个零件)，然后精读 [GPTQ](#权重-only-4-bitgptqawq-与它们的后代) 的推导和 [QuIP / QuIP#](#权重-only-4-bitgptqawq-与它们的后代) 的 incoherence 分析，再看 [QuaRot / SpinQuant](#4-bit-激活与旋转从-quarot-到-flatquant) 如何把旋转变成计算不变性，最后跟 [FlatQuant](#4-bit-激活与旋转从-quarot-到-flatquant) 和 2026 年的 [MPQ](#2026-年值得跟的五篇新工作) 这类“把量化当推断问题”的工作。

**想做系统 / kernel**：先读 [AWQ 的 TinyChat](#权重-only-4-bitgptqawq-与它们的后代) 理解 weight packing，再看 [Atom](#系统与-kernel论文里的-3x-从哪里来)、[QServe](#系统与-kernel论文里的-3x-从哪里来) 怎么把 dequant 融进 GEMM，然后读 Marlin 的实现和 [SVDQuant / Nunchaku](#系统与-kernel论文里的-3x-从哪里来) 的 low-rank 分支融合。FP4 部分从 [NVFP4 预训练](#fp8fp4格式训练与推理) 和 [Quartet](#fp8fp4格式训练与推理) 入手。

## 基础：整数推理与 PTQ/QAT 的四个零件

现代量化器可以拆成四个决策：**映射**（对称 / 非对称）、**粒度**（per-tensor / channel / group）、**校准**（min-max、percentile、MSE）、**是否训练**（PTQ / QAT）。2018–2020 年的论文把这四件事基本讲清楚了，后面的 LLM 工作大多是在“outlier 怎么处理”和“粒度怎么和 kernel 对齐”上做文章。

如果只读三篇：Jacob 建立 integer-only 推理的完整配方；Nagel 的 DFQ 解释没有校准数据时怎么救；AdaRound 把 rounding 从 heuristic 变成可学习的问题。这三篇合起来，足够看懂今天大多数 PTQ 代码在做什么。

| 论文 | 年份/会议 | 做了什么 | 读它的理由 |
|---|---|---|---|
| [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877) | 2018 CVPR | 权重 per-axis、激活 per-tensor，整数算子完成推理；训练时用 STE 模拟量化 | 今天 uint8 / int8 推理的默认配方 |
| [Quantizing deep convolutional networks for efficient inference: A whitepaper](https://arxiv.org/abs/1806.08342) | 2018 | 系统总结 per-channel 权重、对称/非对称、校准和偏差修正 | 工程细节最全的早期文档 |
| [Post-training 4-bit quantization of convolution networks for rapid-deployment](https://arxiv.org/abs/1810.05723) | 2019 NeurIPS | 不微调、不需要完整数据集，用解析方法选 clipping 做 4-bit PTQ | 证明 4-bit 不一定需要 QAT |
| [Data-Free Quantization Through Weight Equalization and Bias Correction](https://arxiv.org/abs/1906.04721) | 2019 ICCV | 跨层权重均衡 + 偏差修正，不需要校准数据 | 没有数据时的第一道补救 |
| [Learned Step Size Quantization](https://arxiv.org/abs/1902.08153) | 2020 ICLR | 把 step size 变成可学习参数，并对梯度做尺度修正 | QAT 里最常用的 quantizer 之一 |
| [Up or Down? Adaptive Rounding for Post-Training Quantization](https://arxiv.org/abs/2004.10568) | 2020 ICML | 把取整方向放松成连续变量，逐层最小化重建误差 | GPTQ 之前最重要的 rounding 改进 |
| [Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning](https://arxiv.org/abs/2208.11580) | 2022 | 用二阶信息做逐层量化/剪枝，给出 OBS 框架的现代版本 | GPTQ 的直接前身 |

这一节的方法在 CNN 上很成熟，但直接搬到 LLM 会碰到两个新问题：激活的 channel 级 outlier，以及 decode 阶段的 memory-bound。前者引出下一节，后者决定了权重-only 量化为什么在 LLM 上特别有效。

## LLM 的 outlier：8-bit 为什么也会坏

LLM 的激活有两类 outlier。一类是 **normal outlier**：某些通道在所有 token 上都偏大；另一类是 **massive outlier**：极少数 token 的个别通道能比周围大几个数量级。per-tensor scale 被这几列绑架，剩下 99% 的值只能分到很粗的格子，W8A8 也会明显掉点。

处理路线大致两条：

- **不量化 outlier**：LLM.int8 把 outlier 维度拆出来做 FP16 矩阵乘，其余走 INT8；SpQR 把 outlier 权重单独存成稀疏高精度。简单直接，但 kernel 要做混合精度，实际加速常被吃掉。
- **把 outlier 迁走**：SmoothQuant 用 per-channel 缩放把激活的量化难度搬到权重；Outlier Suppression 把 LayerNorm 的 $\gamma$ 迁到后续模块，再做 token-wise clipping；Outlier Suppression+ 额外处理通道间的非对称；RPTQ 按通道范围重排后再聚类量化。迁移本身是等价变换，clipping 则需要校准；代价是变换要 fuse 进相邻层。

LLM-FP4 提供了另一个视角：FP 格式的指数位本身就能吸收一部分长尾，但激活的通道间方差仍然需要 per-channel scale。它把 per-channel scale 重参数化成权重的指数偏置，几乎不增加推理开销——这个技巧后来在 FP4 工作里很常见。

| 论文 | 年份/会议 | 做了什么 | 读它的理由 |
|---|---|---|---|
| [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) · [code](https://github.com/TimDettmers/bitsandbytes) | 2022 NeurIPS | vector-wise 量化 + 混合精度分解，把 outlier 维度留给 FP16 | 理解 LLM outlier 的起点 |
| [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438) · [code](https://github.com/mit-han-lab/smoothquant) | 2023 ICML | per-channel 平滑，把激活量化难度等价迁移到权重，实现 W8A8 | 工业界 W8A8 的事实基线 |
| [Outlier Suppression: Pushing the Limit of Low-bit Transformer Language Models](https://arxiv.org/abs/2209.13325) · [code](https://github.com/wimh966/outlier_suppression) | 2022 NeurIPS | 发现 LayerNorm $\gamma$ 放大 outlier；Gamma Migration + token-wise clipping | 解释 outlier 从哪来 |
| [Outlier Suppression+: Accurate quantization of large language models by equivalent and optimal shifting and scaling](https://arxiv.org/abs/2304.09145) · [code](https://github.com/ModelTC/Outlier_Suppression_Plus) | 2023 EMNLP | 通道级 shifting 处理非对称，scaling 处理集中度，均可等价迁移 | 比 OS 更完整的通道级修正 |
| [RPTQ: Reorder-based Post-training Quantization for Large Language Models](https://arxiv.org/abs/2304.01089) | 2023 | 按通道范围重排、聚类量化，再把重排 fuse 进 LayerNorm 和权重 | 通道范围差异也是一种 outlier |
| [ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers](https://arxiv.org/abs/2206.01861) | 2022 | per-token 动态激活量化 + group-wise 权重量化 + 逐层蒸馏 | 最早把 W8A8 做到 GPT 规模的工作之一 |
| [LLM-FP4: 4-Bit Floating-Point Quantized Transformers](https://arxiv.org/abs/2310.16836) · [code](https://github.com/nbasyl/LLM-FP4) | 2023 EMNLP | FP4 权重和激活，搜索指数偏置与 clipping；per-channel 激活 scale 重参数化进权重 | FP4 PTQ 的代表作 |

这一节的方法大多只解决 W8A8。把它们直接套到 W4A4 通常不够，因为 4-bit 的格点更粗，激活里剩下的 normal outlier 依然致命；这时需要后面旋转那一节的办法，或者 QAT。

## 权重-only 4-bit：GPTQ、AWQ 与它们的后代

权重-only 量化的目标函数是逐层重建：

$$
\min_{\hat W}\ \lVert XW-X\hat W\rVert_2^2
$$

其中 $X$ 是校准集上的输入。GPTQ 用 Hessian 的逆做逐列量化，并把当前列的误差补偿到还没量化的列；AWQ 发现保护 1% 的 salient channel 就能大幅降低误差，于是用激活统计搜一个 per-channel scale，把重要通道放大后再量化；SpQR 和 SqueezeLLM 把 outlier 或敏感值单独存成稀疏高精度；OmniQuant 把 clipping 和等价变换都变成可学习参数。

4-bit 以下，scalar quantization 的格点不够用，于是有了向量量化（VQ）家族：GPTVQ 用 codebook 提高量化维度，AQLM 用多码本加性量化，QuIP / QuIP# 用随机正交（含 Hadamard）做 incoherence processing，QuIP# 再加 $E_8$ lattice codebook，QTIP 用 trellis 编码把有效维度做得更高，VPTQ 则用二阶优化和 channel-independent 的码本初始化。QuIP 系列的“先旋转再量化”和后面旋转那一节是同一个思想，只是它作用在权重侧。

| 论文 | 年份/会议 | 做了什么 | 读它的理由 |
|---|---|---|---|
| [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) · [code](https://github.com/IST-DASLab/gptq) | 2023 ICLR | 用 Hessian 逆逐列量化，把误差补偿到未量化列；175B 约 4 GPU 小时 | 4-bit 权重 PTQ 的必读基线 |
| [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) · [code](https://github.com/mit-han-lab/llm-awq) | 2024 MLSys | 用激活统计找 salient channel，per-channel 放大后量化；不依赖反向传播 | 部署最广的 W4A16 方案之一 |
| [SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression](https://arxiv.org/abs/2306.03078) · [code](https://github.com/Vahe1994/SpQR) | 2023 | 把 outlier 权重挑出来存高精度，其余压到 3–4 bit | 低 bit 下“保 outlier”的典型做法 |
| [SqueezeLLM: Dense-and-Sparse Quantization](https://arxiv.org/abs/2306.07629) · [code](https://github.com/SqueezeAILab/SqueezeLLM) | 2023 ICML | 基于二阶敏感度的非均匀量化 + dense-and-sparse 分解 | 非均匀量化和稀疏修正的组合 |
| [OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models](https://arxiv.org/abs/2308.13137) · [code](https://github.com/OpenGVLab/OmniQuant) | 2024 ICLR | 可学习权重裁剪（LWC）+ 可学习等价变换（LET） | 把 PTQ 的超参搜索变成优化 |
| [GPTVQ: The Blessing of Dimensionality for LLM Quantization](https://arxiv.org/abs/2402.15319) | 2024 | 用向量量化 codebook 提高量化维度，改善 size–accuracy 折中 | 从 scalar 走向 VQ 的过渡 |
| [Extreme Compression of Large Language Models via Additive Quantization](https://arxiv.org/abs/2401.06118) · [code](https://github.com/Vahe1994/AQLM) | 2024 ICML | 多码本加性量化，把 2–3 bit 的权重表示成多个码本向量之和 | 极限压缩的代表，2–3 bit 可跑 |
| [QuIP: 2-Bit Quantization of Large Language Models With Guarantees](https://arxiv.org/abs/2307.13304) · [code](https://github.com/Cornell-RelaxML/QuIP) | 2023 NeurIPS | 随机正交做 incoherence processing + 自适应 rounding，给出误差分析 | 2-bit 的理论起点 |
| [QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks](https://arxiv.org/abs/2402.04396) · [code](https://github.com/Cornell-RelaxML/quip-sharp) | 2024 ICML | 随机 Hadamard + $E_8$ 格点码本 + 微调 | ≤4 bit 的强基线，Hadamard 的实战案例 |
| [QTIP: Quantization with Trellises and Incoherence Processing](https://arxiv.org/abs/2406.11235) · [code](https://github.com/Cornell-RelaxML/qtip) | 2024 NeurIPS | 用 trellis coded quantization 做到超高维量化，码本大小与 bitrate 解耦 | VQ 之后的下一个方向 |
| [VPTQ: Extreme Low-bit Vector Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2409.17066) · [code](https://github.com/microsoft/VPTQ) | 2024 EMNLP | 二阶优化指导 VQ，channel-independent 码本初始化，支持 residual/outlier | 2-bit 工程化较好的一条线 |
| [MPQ: A Message-Passing View of Post-Training Quantization](https://arxiv.org/abs/2610.32127) · [code](https://github.com/poseidonchan/MPQ) | 2026 NeurIPS | 把 PTQ 看成离散码本上的后验推断，用 message passing 保留块间软信息，最后才硬判决 | 2026 年 PTQ solver 的新思路 |

**怎么选**：

- 4-bit 默认 AWQ 或 GPTQ。AWQ 不依赖反向传播、跨域泛化好；GPTQ 在更低 bit 和需要误差补偿时更稳，但要校准集。
- group size 128 是 kernel 友好的折中；64 / 32 更准，但 scale 数量、读取开销和 kernel 支持都要确认。
- 2–3 bit 先看 AQLM / QuIP# / QTIP / VPTQ，再考虑 QAT；不要指望 2-bit scalar PTQ 还能保住指令模型。
- 权重-only 的收益主要在 decode。prefill 是 compute-bound，别把 decode 的加速比直接承诺给大 batch 服务。
- 论文里的 ppl 差距要结合任务指标看；指令模型、代码、长上下文通常比 base model 更敏感。

## 4-bit 激活与旋转：从 QuaRot 到 FlatQuant

W4A4 的难点在激活：权重可以 per-channel，激活一般只能 per-token；一个 outlier 就能把整个 token 的 scale 拉大。旋转的思路是乘一个正交矩阵，向量长度不变，尖峰被摊到所有通道，量化步长随之变小。Hadamard 是最便宜的旋转：元素只有 $\pm1$，快速变换只要 $d\log d$ 次加减。

QuaRot 的关键贡献不是“旋转能压 outlier”这个观察，而是证明可以把它做成 **computational invariance**：把旋转吸进相邻的权重、LayerNorm 和 embedding，输出完全不变，连 KV cache 都能一起转。SpinQuant 进一步指出随机旋转之间差异很大（论文报告最多 13 个点的 zero-shot 差距），于是用 Cayley 优化学习旋转。DuQuant 用 block rotation + zigzag permutation 处理 massive outlier；FlatQuant 把 per-layer 的仿射变换也变成可学习参数，用 Kronecker 分解和 fused kernel 控制 online 开销。

2026 年的两条新线值得跟：PeRQ 在 block rotation 之前做 permutation，按 block 均衡 $\ell_1$ 质量，并把 permutation 折进权重；DuQuant++ 把旋转块对齐到 MXFP4 的 32 元素组，利用每组独立 scale 的特点，只做一次 outlier-aware rotation。

除了改分布，还有一条正交的线是改调度。SliderQuant 先做逐层敏感度分析：浅层和深层比中间层更容易被量化破坏，其中最敏感的是第一层和最后一层。于是它把固定滑窗换成随深度变形的滑窗——浅层从 1 层开始逐步扩大、始终锚定第一层，中间层保持 $\{s=2,i=1\}$ 的固定重叠，深层从整段开始逐步收缩、始终锚定最后一层；窗口内再按 $\gamma=1/2$ 分两阶段量化。它不引入任何 online 变换，W4A4 的 Llama2-7B 就把 WikiText2 从 CBQ 的 12.73 压到 8.34；再叠上旋转（SliderQuant+）还能低一点。代价在量化阶段：W4A4 的 Llama2-7B 默认 20 个 epoch、约 6 个 A6000 小时；推理时学到的 scale 和 LoRA 都折进权重，没有额外开销。逐段中英对照、全部实验表与批判性笔记见 [SliderQuant 中英对照精读](/posts/quantization/sliderquant/)。

旋转为什么有效、Hadamard 的峰值下界和 FWHT 的推导，见 [Hadamard 矩阵与快速变换](/posts/quantization/hadamard/)。

| 论文 | 年份/会议 | 做了什么 | 读它的理由 |
|---|---|---|---|
| [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://arxiv.org/abs/2404.00456) · [code](https://github.com/spcl/QuaRot) | 2024 NeurIPS | 用计算不变性把旋转吸进权重/LayerNorm，权重、激活、KV 全 4-bit | W4A4 的里程碑，必读 |
| [SpinQuant: LLM Quantization with Learned Rotations](https://arxiv.org/abs/2405.16406) · [code](https://github.com/facebookresearch/SpinQuant) | 2025 ICLR | 参数化旋转并用 Cayley 优化学习；比较不同随机旋转的方差 | 说明“转哪里”比“转不转”更重要 |
| [DuQuant: Distributing Outliers via Dual Transformation Makes Stronger Quantized LLMs](https://arxiv.org/abs/2406.01721) · [code](https://github.com/Hsu1023/DuQuant) | 2024 NeurIPS | block rotation + zigzag permutation + 二次旋转，专门处理 massive outlier | W4A4 下 outlier 更强的解法 |
| [FlatQuant: Flatness Matters for LLM Quantization](https://arxiv.org/abs/2410.09426) · [code](https://github.com/ruikangliu/FlatQuant) | 2025 ICML | 逐层学习仿射变换，Kronecker 分解 + fused kernel；W4A4 LLaMA-3-70B 掉点 <1% | 从“旋转”走到“学习变换” |
| [OstQuant: Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for Better Distribution Fitting](https://arxiv.org/abs/2501.13987) · [code](https://github.com/BrotherHappy/OSTQuant) | 2025 | 提出 QSUR 度量，学习正交 + scaling 变换，让分布更贴合量化空间 | 用“空间利用率”解释变换的好坏 |
| [Pushing the Limits of Block Rotations in Post-Training Quantization](https://arxiv.org/abs/2601.22347) | 2026 | 分析 block Hadamard 的 outlier 抑制极限；PeRQ 先 permutation 再旋转，block=16 恢复 90% full-vector 效果 | block rotation 的理论补课 |
| [DuQuant++: Fine-grained Rotation Enhances Microscaling FP4 Quantization](https://arxiv.org/abs/2604.17789) · [code](https://github.com/Hsu1023/DuQuant-v2) | 2026 | 旋转块对齐 MXFP4 的 32 元素组，单次 outlier-aware rotation，online 开销减半 | MXFP4 + 旋转的最新组合 |
| [SliderQuant: Accurate Post-Training Quantization for LLMs](https://arxiv.org/abs/2603.25284) · [code](https://github.com/deep-optimization/SliderQuant) · [精读](/posts/quantization/sliderquant/) | 2026 ICLR | 把固定滑窗改成随深度变形的滑窗（浅层扩张、中层固定、深层收缩），窗口内分阶段量化；W4A16/W2A16/W4A4 都测 | 不改分布、只改量化调度，和旋转正交 |

**工程注意**：

- 旋转必须在定 scale 之前；先转再量化，否则收益白丢。
- attention 里 $q$ 和 $k$ 必须用同一个旋转，否则点积对不上；$k$ 要先转再写 KV cache。
- online transform 必须和量化 kernel 融合。单独跑一次旋转会多一次读写，端到端收益可能归零。
- block rotation 比 full-vector 便宜，但 outlier 抑制更依赖输入几何；permutation 如果能折进权重，就尽量不要留在 runtime。
- 学到的变换会过拟合校准集。换模型、换领域、换序列长度，都要重新校准并重测。

## KV cache：长上下文真正的瓶颈

KV cache 的容量随上下文线性增长。Llama-2 70B 在 4K context 下 FP16 KV 约 10.7 GB，32K 时约 85.9 GB（batch=1）——比 4-bit 权重还大。量化的关键观察是 K 和 V 的分布不同：K 的 outlier 集中在 channel 维，适合 per-channel；V 的 outlier 少，适合 per-token。

KIVI 用 per-channel K + per-token V，再加一个 residual window 保留最近的 token；KVQuant 同样用 per-channel K，再加 pre-RoPE 量化、per-layer 非均匀 datatype、per-vector dense-and-sparse；GEAR 用 low-rank 矩阵拟合量化误差、稀疏矩阵修正 outlier；QAQ 和 ZipCache 用 attention saliency 做混合精度；PolarQuant 用随机预条件 + 极坐标变换，把角度量化，省掉 per-block 的 scale/zero-point。

| 论文 | 年份/会议 | 做了什么 | 读它的理由 |
|---|---|---|---|
| [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750) · [code](https://github.com/jy-yuan/KIVI) | 2024 ICML | K per-channel、V per-token，2-bit KV + residual window | KV 2-bit 的代表作，配方常被复用 |
| [KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization](https://arxiv.org/abs/2401.18079) · [code](https://github.com/SqueezeAILab/KVQuant) | 2024 NeurIPS | per-channel K、pre-RoPE、非均匀 datatype、per-vector dense-and-sparse | 3-bit 下 ppl 掉 <0.1 的强基线 |
| [GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference of LLM](https://arxiv.org/abs/2403.05527) · [code](https://github.com/HaoKang-Timmy/GEAR) | 2024 | 量化 + low-rank 误差拟合 + 稀疏 outlier 修正 | 把“量化误差”当成可修正项 |
| [QAQ: Quality Adaptive Quantization for LLM KV Cache](https://arxiv.org/abs/2403.04643) · [code](https://github.com/ClubieDong/KVCacheQuantization) | 2024 | 分别处理 K/V 敏感度，outlier 处理 + attention-aware 的混合精度 | 质量自适应的代表 |
| [ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification](https://arxiv.org/abs/2405.14256) | 2024 | channel-separable tokenwise 量化 + normalized attention saliency，兼容 FlashAttention | 压缩率和 kernel 兼容性兼顾 |
| [PolarQuant: Quantizing KV Caches with Polar Transformation](https://arxiv.org/abs/2502.02617) | 2025 | 随机预条件 + 极坐标变换，量化角度，省掉显式归一化 | 换坐标系消除 scale 开销的思路 |

**工程注意**：

- KV 量化和 attention kernel 强耦合。FlashAttention、PagedAttention 的 layout 决定能不能做 per-head / per-token scale，论文里的 kernel 不一定能直接搬。
- 长上下文的误差会累积。评估至少覆盖 4K / 32K / 128K，不要只测 2K 的 ppl。
- K 在 RoPE 之前量化通常更好；V 的 outlier 少，但误差会直接进输出，别只盯着 K。
- KV 4-bit 的收益主要在 memory-bound decode；prefill 仍受 compute 限制，收益有限。
- KV 量化和 eviction / sparsity 是正交的，可以叠加，但误差也会叠加，要分别做 ablation。

## QAT 与极低比特：什么时候必须训练

PTQ 在 2-bit 以下通常崩。QAT 把量化放进训练前向，反向用 STE 传梯度；问题是 LLM 的 QAT 很贵，于是出现了三条降本路线：

- **data-free distillation**：LLM-QAT 用模型自己生成的文本做蒸馏，不依赖原始训练集，同时量化 KV cache。
- **block-wise + 参数高效**：EfficientQAT 先逐 block 训练全部参数，再端到端只训练量化参数；LR-QAT 用 low-rank 辅助权重、定点 downcast 和 checkpointing，把 7B 的 QAT 压到单张 24GB 卡。
- **从零训练低比特架构**：BitNet 系列直接训练 1-bit 权重 / ternary 权重，BitNet b1.58 在 3B 以上可以匹配 FP16；后续 a4.8 和 v2 继续把激活压到 4-bit，v2 用 Hadamard 处理激活 outlier。

注意：BitNet 的结论是 **pretraining** 的结论，不是“把已有模型 PTQ 到 1.58 bit”。已有模型做 1-bit，通常走 OneBit 这种“矩阵分解初始化 + 训练”的路线，或者 PB-LLM 这种“部分二值化 + 保留少量高精度”的路线，掉点都很明显。

| 论文 | 年份/会议 | 做了什么 | 读它的理由 |
|---|---|---|---|
| [LLM-QAT: Data-Free Quantization Aware Training for Large Language Models](https://arxiv.org/abs/2305.17888) · [code](https://github.com/facebookresearch/LLM-QAT) | 2023 | 用模型自生成文本做 data-free 蒸馏，量化权重、激活和 KV cache | LLM QAT 的起点 |
| [EfficientQAT: Efficient Quantization-Aware Training for Large Language Models](https://arxiv.org/abs/2407.11062) · [code](https://github.com/OpenGVLab/EfficientQAT) | 2025 ACL | Block-AP 逐 block 训练全部参数 + E2E-QP 只训练量化参数；2-bit 70B 单卡 41 小时 | 目前最实用的 LLM QAT 配方之一 |
| [Low-Rank Quantization-Aware Training for LLMs](https://arxiv.org/abs/2406.06385) · [code](https://github.com/qualcomm-ai-research/LR-QAT) | 2024 | low-rank 辅助权重 + 定点 downcast + checkpointing，7B QAT 单卡 24GB | 显存受限时的 QAT 方案 |
| [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) · [code](https://github.com/artidoro/qlora) | 2023 NeurIPS | NF4 + double quantization + paged optimizer，4-bit 基座微调 | 4-bit 微调的默认起点 |
| [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453) · [code](https://github.com/microsoft/unilm/tree/master/bitnet) | 2023 | 1-bit 权重 + 8-bit 激活，从零训练 | 低比特架构的早期尝试 |
| [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764) | 2024 | ternary 权重 $\{-1,0,1\}$，3B 以上匹配 FP16 | 1-bit 路线的标志性结果 |
| [BitNet a4.8: 4-bit Activations for 1-bit LLMs](https://arxiv.org/abs/2411.04965) | 2024 | 1-bit 权重 + 4-bit 激活，混合稀疏 | 把激活也压到 4-bit |
| [BitNet v2: Native 4-bit Activations with Hadamard Transformation for 1-bit LLMs](https://arxiv.org/abs/2504.18415) | 2025 | 用 Hadamard 变换处理 4-bit 激活的 outlier | 旋转思想进入 1-bit 架构 |
| [OneBit: Towards Extremely Low-bit Large Language Models](https://arxiv.org/abs/2402.11295) | 2024 NeurIPS | 1-bit 权重，矩阵分解初始化 + 训练，保留约 81% 性能 | 已有模型压到 1-bit 的参考 |
| [PB-LLM: Partially Binarized Large Language Models](https://arxiv.org/abs/2310.00034) | 2023 | 只把一部分权重二值化，显著权重保留高精度 | “部分二值化”的另一种取舍 |

**什么时候值得 QAT**：

- 目标 ≤3 bit，且 PTQ 掉点超过可接受范围。
- 有几十到几百 GPU 小时的预算；EfficientQAT 报告 2-bit 70B 单卡 41 小时，已经不是遥不可及。
- 需要同时量化激活或 KV cache，而不只是权重。
- 不能依赖校准集，愿意用 self-generated data。
- 只是想省显存、4-bit 够用：优先 GPTQ / AWQ，不必上 QAT。

## FP8/FP4：格式、训练与推理

FP8 已经是训练和高端推理的常用低精度格式：E4M3 要精度，E5M2 要范围；Hopper 之后有原生 FP8 tensor core。MX 格式把 scale 降到 block 级——32 个元素共享一个 E8M0 指数，FP4 才有可用的动态范围。NVFP4 用 16 个元素共享一个 E4M3 scale，再加一个 FP32 全局 scale；Blackwell 有原生 FP4 支持。

FP4 训练的关键不是格式本身，而是 recipe：RHT 压 block 级 outlier、2D scaling 让前向和反向的量化一致、stochastic rounding 做无偏梯度估计、敏感层保留高精度。NVIDIA 的 NVFP4 预训练在 12B 模型、10T token 上验证了这套组合，loss 和下游任务接近 FP8 基线；Quartet 则研究“所有主要线性层都用 FP4”能否与 FP16 / FP8 竞争，并在 Blackwell 上写了专用 kernel。

推理侧，FP4 的收益很大程度上取决于硬件和 kernel：Blackwell 有原生 FP4；老卡只能模拟，数值上能验证，速度不会变快。权重是否 packed、scale layout 是否对齐 tensor core、累加精度够不够，都是坑。

| 论文 | 年份/会议 | 做了什么 | 读它的理由 |
|---|---|---|---|
| [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) | 2022 | 提出 E4M3 / E5M2 两种 FP8 格式及数值取舍 | FP8 的格式定义 |
| [Microscaling Data Formats for Deep Learning](https://arxiv.org/abs/2310.10537) | 2023 | MX 格式：block 共享 E8M0 scale，覆盖 MXFP8/6/4 | 理解 MXFP4 的必读 |
| [FP8-LM: Training FP8 Large Language Models](https://arxiv.org/abs/2310.18313) | 2023 NeurIPS | 梯度、优化器状态等也用 FP8，训练框架级实现 | FP8 训练的系统性工作 |
| [COAT: Compressing Optimizer states and Activation for Memory-Efficient FP8 Training](https://arxiv.org/abs/2410.19313) | 2025 ICLR | 同时压优化器状态和激活的 FP8 训练 | FP8 训练的显存优化 |
| [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) · [code](https://github.com/deepseek-ai/DeepSeek-V3) | 2024 | blockwise FP8 训练，细粒度 scale，FP32 累加 | 大规模 FP8 训练的工程样本 |
| [Pretraining Large Language Models with NVFP4](https://arxiv.org/abs/2509.25149) | 2025 | NVFP4 + RHT + 2D scaling + stochastic rounding + 敏感层保留；12B/10T token | FP4 预训练目前最完整的公开配方 |
| [Quartet: Native FP4 Training Can Be Optimal for Large Language Models](https://arxiv.org/abs/2505.14669) · [code](https://github.com/IST-DASLab/Quartet) | 2025 | 线性层全 FP4 训练，Blackwell 专用 kernel，与 FP16/FP8 对比 | 全 FP4 训练的上限在哪 |
| [What Matters for NVFP4 Training? A Comparative Study of Low-Precision Pretraining Recipes](https://arxiv.org/abs/2610.24343) | 2026 | 8B dense / 30B-A3B MoE、最多 1T token，系统比较 NVFP4 recipe 的各组件 | 2026 年最实用的 FP4 训练消融 |
| [Fractional Power-of-Two Quantization for Efficient and Effective Multiplier-Free LLM Inference](https://arxiv.org/abs/2610.21257) | 2026 NeurIPS | 用 base-$\sqrt2$ 的 fractional PoT 网格做 4-bit 权重和激活，保持 shift-add 数据通路 | 无乘法器推理的另一条路 |

**格式选择（粗略）**：

- 服务端 8-bit：FP8 E4M3，kernel 成熟，精度接近无损。
- 权重-only 4-bit：INT4 + group=128，生态最好，Marlin / vLLM 支持最全。
- 4-bit 权重 + 4-bit 激活：NVFP4 / MXFP4 + 旋转或 block scale，硬件和 kernel 是门槛。
- 训练：BF16 → FP8 已是常规；FP4 还在 recipe 阶段，先确认框架和卡的支持。

## 系统与 kernel：论文里的 3x 从哪里来

decode 是 memory-bound：每生成一个 token 都要把全部权重读一遍。量化把字节数降下来，但如果 dequant 在 CUDA core 上做，省下的带宽会被指令开销吃掉。Marlin、Atom、QServe 的核心都是“让 dequant 和 GEMM 融合，并让数据布局对 tensor core 友好”。

Atom 用 W4A4 + 混合精度 + group quantization，把 4-bit 算子真正用起来；QServe 用 W4A8KV4 + progressive quantization + SmoothAttention，重点解决大 batch 下 dequant 的 20–90% 运行时开销；SVDQuant 把 outlier 吸到 low-rank 分支，再和 4-bit 分支 fuse——它主要面向 diffusion 模型，但和 LLM 的 W4A4 是同一条系统路线。

| 论文 | 年份/会议 | 做了什么 | 读它的理由 |
|---|---|---|---|
| [Atom: Low-bit Quantization for Efficient and Accurate LLM Serving](https://arxiv.org/abs/2310.19102) · [code](https://github.com/efeslab/Atom) | 2024 MLSys | W4A4，混合精度 + group quantization + 低比特算子 | 服务端 4-bit 的系统样本 |
| [QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving](https://arxiv.org/abs/2405.04532) · [code](https://github.com/mit-han-lab/omniserve) | 2025 MLSys | W4A8KV4（QoQ）+ progressive quantization + SmoothAttention + 专用 kernel | 算法和 kernel 协同设计的范本 |
| [SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models](https://arxiv.org/abs/2411.05007) · [code](https://github.com/mit-han-lab/nunchaku) | 2025 ICLR | 用 low-rank 分支吸收 outlier，和 4-bit 分支融合；Nunchaku 引擎 | W4A4 在 diffusion 上的落地 |
| [LUT-GEMM: Quantized Matrix Multiplication based on LUTs for Efficient Inference in Large-Scale Generative Language Models](https://arxiv.org/abs/2206.09557) | 2024 ICLR | 用 lookup table 做 W4A16 矩阵乘，binary-coding quantization | 早期 LUT 路线的代表 |

**工程资源**（比论文更新更快）：

- [vLLM quantization 文档](https://docs.vllm.ai/en/latest/features/quantization/)：支持哪些格式、哪些 kernel、怎么起服务。
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)：FP8 / FP4 / INT4 的生产级 kernel 和量化流程。
- [CUTLASS](https://github.com/NVIDIA/cutlass)：FP8 / FP4 / microscaling 的底层实现。
- [Marlin](https://github.com/IST-DASLab/marlin)：W4A16 的高性能 GEMM kernel，异步拷贝 + FP32 累加，vLLM 的常用后端。
- [llm-compressor](https://github.com/vllm-project/llm-compressor)：GPTQ / AWQ / SmoothQuant 等压缩流程。
- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)：LLM.int8 / NF4 的实现。
- [Nunchaku](https://github.com/mit-han-lab/nunchaku)：SVDQuant 的推理引擎。

## 评测与综述：怎么判断一篇量化论文靠不靠谱

量化论文最容易糊弄的地方是评测：ppl 降 0.1 不代表任务不掉点；accuracy 一样不代表输出分布一样（flips）；speedup 可能是在 batch=1、特定 kernel、特定序列长度上测的。读的时候至少要问：校准集和测试集有没有重叠？序列长度是多少？baseline 是 FP16 还是另一个量化方法？加速比是在 prefill 还是 decode 上测的？

“Give Me BF16 or Give Me Death?” 在 Llama-3.1 全家族、50 万次评测上给出了一组比较实用的结论：FP8 接近无损，INT8 掉 1–3%，W4A16 比预期更有竞争力；同步 serving 选 W4A16，异步 continuous batching 选 W8A8。“Accuracy is Not All You Need” 则用 KL 和 flips 说明：即使 accuracy 接近，压缩模型的输出分布也可能明显变了，MT-Bench 上差距更大。

| 论文 | 年份/会议 | 做了什么 | 读它的理由 |
|---|---|---|---|
| [A Survey of Quantization Methods for Efficient Neural Network Inference](https://arxiv.org/abs/2103.13630) | 2021 | PTQ / QAT / 硬件友好的分类综述 | 建立整体 taxonomy |
| [A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295) | 2021 | Qualcomm 的量化工程手册：校准、QAT、硬件约束 | 工程细节比论文全 |
| [A Survey on Model Compression for Large Language Models](https://arxiv.org/abs/2308.07633) | 2023 | 量化、剪枝、蒸馏、低秩的 LLM 压缩综述 | 把量化放回压缩工具箱 |
| [A Survey of Low-bit Large Language Models: Basics, Systems, and Algorithms](https://arxiv.org/abs/2409.16694) | 2024 | 低比特 LLM 的算法 + 系统综述 | 查漏补缺的入口 |
| [The case for 4-bit precision: k-bit Inference Scaling Laws](https://arxiv.org/abs/2212.09720) | 2023 ICML | 大规模实验拟合 bit 数与精度的 scaling law | 解释为什么 4-bit 是甜点 |
| ["Give Me BF16 or Give Me Death"? Accuracy-Performance Trade-Offs in LLM Quantization](https://arxiv.org/abs/2411.02355) | 2025 ACL | FP8 / INT8 / INT4 在 Llama-3.1 全家族上的系统评测与部署建议 | 选型时最实用的一篇 |
| [Evaluating Quantized Large Language Models](https://arxiv.org/abs/2402.18158) | 2024 | 11 个模型家族、5 类任务，覆盖权重/激活/KV | 评测维度的参考 |
| [Accuracy is Not All You Need](https://arxiv.org/abs/2407.09141) | 2024 | 提出 flips 和 KL 指标，揭示 accuracy 相近但行为变化 | 别只看 accuracy |
| [Massive Activations in Large Language Models](https://arxiv.org/abs/2402.17762) | 2024 COLM | 发现极少数激活比周围大几个数量级，并分析其作用 | 理解 outlier 的来源 |

**验收清单**：

- 校准集和测试集严格分开；不要用测试集选 scale、$\alpha$ 或 clipping 阈值。
- 同时报告 ppl、zero-shot / 任务指标、KL 或 flips；指令模型加 MT-Bench 一类生成评测。
- 标清楚序列长度；长上下文要单独测 4K / 32K / 128K。
- 加速比区分 prefill 和 decode、batch size、GPU 型号、是否含 dequant。
- 确认权重真的 packed；很多“4-bit 模型”只是存成 int4，kernel 仍按 FP16 读。
- 确认 kernel 支持当前粒度（per-channel / group / per-token），否则会 fallback。
- 检查敏感层：embedding、lm_head、LayerNorm、attention 的 softmax 是否保留高精度。
- KV cache 单独报显存和吞吐；长上下文下它可能比权重更重要。
- 量化后的模型要跑一遍安全 / 拒答 / 多语言，低比特对长尾行为的影响通常更大。

## 2026 年值得跟的五篇新工作

前面各节已经提到它们，这里单独列一下，方便按时间跟：

1. **PeRQ**（[2601.22347](https://arxiv.org/abs/2601.22347)）：block rotation 的理论分析 + permutation，解决“块内 $\ell_1$ 质量不均”的问题。
2. **DuQuant++**（[2604.17789](https://arxiv.org/abs/2604.17789)）：把旋转对齐到 MXFP4 的 32 元素组，单次 rotation，online 开销减半。
3. **MPQ**（[2610.32127](https://arxiv.org/abs/2610.32127)）：把 PTQ 看成离散码本上的后验推断，用 message passing 保留软信息，再落到硬判决。
4. **FPoT**（[2610.21257](https://arxiv.org/abs/2610.21257)）：base-$\sqrt2$ 的 fractional power-of-two 网格，4-bit 权重和激活 + shift-add，无乘法器推理。
5. **NVFP4 recipe 消融**（[2610.24343](https://arxiv.org/abs/2610.24343)）：8B / 30B-A3B MoE、1T token 规模上比较 FP4 训练的各个组件，工程价值很高。

这五篇都还很新，结论要等复现和后续引用。把它们放进来，是因为它们分别落在三条线上：**block rotation 的理论化**（PeRQ、DuQuant++）、**PTQ 与推断算法的结合**（MPQ）、**FP4 的格式与训练 recipe**（FPoT、NVFP4 消融）。

## 怎么跟新论文

这份清单会定期更新，但论文不会等清单。平时可以用下面几条线追踪：

```text
# arXiv：最近两周的量化论文
http://export.arxiv.org/api/query?search_query=cat:cs.LG+AND+abs:%22quantization%22&sortBy=submittedDate&sortOrder=descending&max_results=50

# 只看 KV cache 量化
http://export.arxiv.org/api/query?search_query=abs:%22KV+cache%22+AND+abs:%22quantization%22&sortBy=submittedDate&sortOrder=descending&max_results=30

# 只看 FP4
http://export.arxiv.org/api/query?search_query=abs:%22FP4%22+AND+abs:%22LLM%22&sortBy=submittedDate&sortOrder=descending&max_results=30
```

- **Hugging Face Papers**（<https://huggingface.co/papers>）：每天的 trending，适合发现被社区讨论的工作。
- **OpenReview**：ICLR / NeurIPS / MLSys 的审稿意见比 abstract 更能看出方法的边界。
- **Semantic Scholar / Google Scholar 的引用提醒**：盯住 GPTQ、AWQ、QuaRot、KIVI 这几篇的 citing papers，比泛搜“quantization”有效得多。
- **vLLM / TensorRT-LLM 的 release notes**：新 kernel 和新格式往往先出现在这里，而不是论文里。
- **复现优先**：看到新方法，先找 code 和 calibration 脚本；没有 code 的“SOTA”先打问号。

一个 30 分钟的 triage 流程：读 title / abstract → 看它和哪篇 baseline 比、bit-width 和模型规模是否可比 → 找 code 和 kernel 支持 → 看 evaluation 是否覆盖长上下文和指令模型 → 决定要不要精读。大多数论文到第三步就可以放下了。

## 收录说明与偏差

- **不追求穷举**。这份清单偏向 LLM 推理、有公开代码、能被后续工作复用的论文；视觉、语音、推荐系统的量化只收了少数跨界的（如 SVDQuant）。
- **会议信息以 arXiv comment 为准**，抓取时间 2026-10-26。预印本可能随后被接收，以正式出版为准。
- **数字不横向比较**。同一张表里的“掉点 1%”可能来自不同模型、不同校准集、不同评测框架；需要严格对比时回到原文。
- **代码链接已检查可访问**（2026-10-26），但仓库可能迁移或归档；找不到时以论文为准。
- **2026 年条目只代表“值得跟”**，还没有经过充分复现和时间检验。
- 这份清单会随分类更新；发现错误或漏掉重要工作，欢迎在[仓库](https://github.com/ZJZ0405/zjz0405.github.io/issues)提 issue。

## 如果只读七篇

1. [LLM.int8](https://arxiv.org/abs/2208.07339)：理解 outlier，知道 8-bit 为什么也会坏。
2. [SmoothQuant](https://arxiv.org/abs/2211.10438)：W8A8 的工程基线，等价变换的范本。
3. [GPTQ](https://arxiv.org/abs/2210.17323)：权重-only 量化的数学，Hessian 误差补偿。
4. [AWQ](https://arxiv.org/abs/2306.00978)：从算法到 kernel 的完整落地。
5. [QuaRot](https://arxiv.org/abs/2404.00456) 或 [SpinQuant](https://arxiv.org/abs/2405.16406)：W4A4 的旋转路线。
6. [KIVI](https://arxiv.org/abs/2402.02750) 或 [KVQuant](https://arxiv.org/abs/2401.18079)：长上下文的 KV 压缩。
7. [Give Me BF16](https://arxiv.org/abs/2411.02355)：选型和评测，知道什么时候该停在 8-bit。

读完这七篇，再看具体模型和框架的量化配置，基本不会迷路。
