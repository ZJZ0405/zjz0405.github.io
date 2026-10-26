---
title: "SliderQuant：面向大语言模型的精确训练后量化（中英对照精读）"
published: 2026-10-27
description: ICLR 2026 SliderQuant 中英对照精读：从逐层量化敏感度出发，用 PESW/FSSW/PCSW 三段式滑窗和层内分阶段量化降低跨层误差，附全部实验表、消融与批判性笔记。
tags: [Quantization, SliderQuant, PTQ, 论文精读, 中英对照, 低精度推理]
category: Quantization
draft: false
bilingual: true
---

> **SliderQuant: Accurate Post-Training Quantization for LLMs**  
> Shigeng Wang, Chao Li, Yangyuxuan Kang, Jiawei Fan, Zhonghong Ou, Anbang Yao  
> ICLR 2026 · [arXiv:2603.25284](https://arxiv.org/abs/2603.25284) · [code](https://github.com/deep-optimization/SliderQuant)

:::en
> **How to read:** this is a source-grounded bilingual reader. Every block starts with a page/block reference; `Original` keeps the paper text and `中文` is the translation. The switch above toggles **bilingual / Chinese only / English only**; Chinese-only reading notes are hidden in English mode.
:::

:::zh
> **中英对照精读版（source-grounded reader）**  
> 原文：Shigeng Wang, Chao Li, Yangyuxuan Kang, Jiawei Fan, Zhonghong Ou, Anbang Yao.  
> *SliderQuant: Accurate Post-Training Quantization for LLMs*. ICLR 2026；arXiv:2603.25284v1（30 页）。

**阅读方式**：每个块给出 `**Source:** p.页码 块号`；`**Original:**` 为原文照录，`**中文:**` 为逐段译文。图表以 300 dpi 紧裁切图片内嵌，图注/表注原文与译文并列。公式用 LaTeX 重排，带 `E...` 锚点与页内出处。块号规则：`S` 正文、`C` 图注/表注、`F` 图、`T` 表、`E` 公式、`N` 读前预备（非论文原文）。同一段落若被分页截断，会拆成两个块（如 S006 → S007），按 `Source:` 的页码即可看出接续关系。

**翻译约定**：`post-training quantization` 统一译作“训练后量化（PTQ）”；`weight-only quantization` 译“仅权重量化”；`weight-activation quantization` 译“权重-激活量化”；`inter-layer / intra-layer sliding quantization` 译“层间／层内滑动量化”；方法名、数据集名、指标名（PESW、PCSW、WikiText2、C4、PIQA、W4A4 等）与符号一律保留原文。

---

## 0. 元数据与速览

| 项目 | 内容 |
|---|---|
| 标题 | SliderQuant: Accurate Post-Training Quantization for LLMs / 面向大语言模型的精确训练后量化 |
| 作者 | Shigeng Wang\*, Chao Li\*, Yangyuxuan Kang, Jiawei Fan, Zhonghong Ou, Anbang Yao†（\* 共同贡献，† 通讯作者） |
| 单位 | Intel Labs China；北京邮电大学（BUPT） |
| 发表 | ICLR 2026（arXiv:2603.25284v1，2026-03-26） |
| 类型 | 方法/算法论文（LLM 训练后量化） |
| 规模 | 30 页：正文 10 页、参考文献 4 页、附录 16 页；20 图、30 表、3 个独立公式 |
| 代码 | https://github.com/deep-optimization/SliderQuant |
| 核心结论 | W4A4 下 Llama2-7B 的 WikiText2/C4 困惑度为 8.34/11.10（FP16 为 5.47/6.97，此前最好的 CBQ 为 12.73/14.45）；W4A16 下 DeepSeek-R1-Distill-Qwen-32B 在数学与代码任务上接近无损 |
| 一句话概括 | 先实测“不同层对量化的敏感度并不相同”（浅/深层更敏感，首/末层最敏感），再把固定的滑动窗口改造成“浅层渐扩—中间固定—深层渐缩”的三段式调度，并在窗口内做分阶段增量量化，用极少的可学习参数把序列 PTQ 的跨层误差压下来 |

**一句话概括（展开版）**：现有 PTQ 大多把模型切成等长、互不重叠的片段逐段量化，隐含假设“每层同等重要”。论文用三类代表性方法（SmoothQuant 逐层、OmniQuant 逐块、CBQ 多块）在 W4A4 下逐层/累计量化，看到三条规律：中间层损伤小、浅深层损伤大、首末层最大，且随量化层数增加而放大。于是提出 SliderQuant：用重叠滑动窗口把相邻层的量化“绑”在一起传递误差信息，并让窗口尺寸随层深度自适应变化；窗口内再按比例 $\gamma$ 分 $N=1/\gamma$ 个阶段由局部到整体联合量化。所有新增参数（通道缩放 $\alpha$、LoRA 的 $A/B$）在量化后并入权重，推理时零额外开销。

---

## 0.4 论文地图与阅读路径

| 步骤 | 位置 | 在讲什么 |
|---|---|---|
| ① 观察 | §1，[Fig. 1](#F001) | 逐层量化单层 / 前 $l$ 层，看困惑度怎么变，得到三条经验规律 |
| ② 统一框架 | §3.1，[Eq. 1](#E001) | 固定尺寸滑动量化是总框架；逐层、逐块、多块量化都是其特例（$i=s$ 即不重叠） |
| ③ 方法 | §3.2，[Fig. 2](#F002) | 层间滑动（PESW/FSSW/PCSW）+ 层内滑动（$\gamma$ 分阶段）+ 可学习参数 |
| ④ 实验 | §4，[Table 1](#T001)–[Table 12](#T012) | W4A16/W2A16/W4A4 × Llama/Llama2/Llama3/Qwen2.5/MoE/DeepSeek-R1 蒸馏 |
| ⑤ 附录 | §A–§I | 数据集与实现细节、旋转版 SliderQuant+、量化效率、更多消融与结果、可视化 |

**赶时间的三分钟版**：看 [Fig. 1](#F001)（为什么层与层不一样）→ [Fig. 2](#F002)（窗口怎么滑）→ [Table 1](#T001)（赢了多少）→ §5 结论。

**想复现**：附录 [B.1](#S050) 给出量化器定义（[Eq. A](#E003)），[Table A](#TA) 给出全部超参（$L_s=L_d=4$、$s=2$、$i=1$、$\gamma=0.5$、LoRA rank $r=4$、校准样本 128、AdamW、20 epoch），§B.5 说明部署用 `llama.cpp`、无需定制算子。

---

## 0.5 读前预备：读懂本文所需的三个背景

> 以下 `N` 块为**读者补充**，非论文原文，目的是让不熟悉 PTQ 的读者能直接进入 §3。

<a id="N001"></a>
### N001 一个量化器到底在做什么

给定一个张量 $Z$（权重或激活），把浮点压成 $b$ 比特整数，最常用的是均匀（线性）量化：

$$
\text{quantizer}(Z)=\text{clamp}\left(\left\lfloor \frac{Z}{\alpha}\right\rceil-\beta,\ 0,\ 2^{b}-1\right),\qquad
\alpha=\frac{Z_{\max}-Z_{\min}}{2^{b}-1},\qquad
\beta=\left\lfloor\frac{Z_{\min}}{\alpha}\right\rceil .
$$

三个量各管一件事：$\alpha$ 是**步长**（把浮点区间等分成 $2^b$ 格），$\beta$ 是**零点偏移**（让量化后的整数 0 对齐浮点里的某个值，本文取整后即非对称量化），$\lfloor\cdot\rceil$ 是四舍五入。`clamp` 把越界值截断——这一步就是量化误差的主要来源：区间外的值被压平，区间内的值被舍入到最近的格点。

两个粒度决定了“区间”怎么取：**逐通道（per-channel）** 对权重每个输出通道各取一组 $Z_{\min},Z_{\max}$；**逐 token（per-token）** 对激活每行（一个 token 的向量）各取一组。粒度越细，区间越贴合局部数值分布，误差越小，但要存的量化参数（scale/zero-point）也越多。**分组（group-wise）** 是折中：把通道再切成大小为 $g$ 的组，每组一套参数。论文默认逐通道 + 逐 token，只在 MoE 和数学/代码任务上改用 $g=128$ 的分组量化（§B.1）。

<a id="N002"></a>
### N002 序列量化：误差为什么会“攒”起来

LLM 有 $L$ 层，逐层量化的做法是：量化第 1 层 → 用**量化后**的第 1 层跑一遍校准数据，得到第 2 层的输入 → 再量化第 2 层 → …… 直到第 $L$ 层。注意第 2 层的输入已经带了第 1 层的误差，第 3 层的输入又叠加了前两层的误差。所以误差不是独立事件，而是**沿层传播并累积**的。

这就解释了论文的两个经验现象为什么重要：如果某一层本身很敏感（例如首层把原始输入映射到隐藏空间、末层把隐藏状态映射回词表分布），它的误差会被后面所有层继续放大；反之，把相邻若干层放进同一个窗口联合优化（窗口内反向传播可以“补偿”彼此的误差），就能在局部把误差按下去。窗口越大、重叠越多，补偿越充分，但显存与计算也越贵——这正是 [Table 11](#T011) 与 [Table Q](#TQ) 在权衡的东西。

<a id="N003"></a>
### N003 滑动窗口里的“重叠”为什么有用

设窗口大小为 $s$、每步移动 $i$ 层，则相邻两个窗口重叠 $s-i$ 层。当 $i<s$ 时，重叠层会被**量化两次以上**：第一次作为前一个窗口的尾部，第二次作为后一个窗口的头部，而第二次优化时它已经能看到“后一段”的信息。于是重叠层充当了相邻窗口之间的**信息桥梁**——前一段的量化误差在这里被后一段“接住”并重新优化。

论文把这个直觉做成了两类设计：层间滑动（窗口沿层方向移动，重叠层数随深度变化）与层内滑动（窗口内部的层按比例分批加入联合优化）。极端情形 $s=1$ 或 $i=s$ 时没有重叠，就退化成逐层/逐块量化——这也正是 §3.1 说“它们是滑动量化的特例”的含义。

---

## 0.6 记号表

| 记号 | 含义 | 首次出现 |
|---|---|---|
| $L$ | 模型层数 | [S010](#S010) |
| $s,\ i$ | 滑动窗口大小（层数）、每步移动间隔；重叠为 $s-i$ | [S010](#S010) |
| $\mathbf{W},\mathbf{X}$ | 当前窗口内 $s$ 层的预训练权重集、输入特征 | [S010](#S010) |
| $\hat{\mathbf{W}},\hat{\mathbf{X}}$ | 对应的低精度（量化后）权重集、输入 | [S010](#S010) |
| $\mathcal{F}(\cdot,\cdot)$ | 当前窗口的输出特征（前向函数） | [S011](#S011) |
| $c$ | 校准样本数（默认 128） | [S010](#S010) |
| $\alpha_i$ | 第 $i$ 层可学习的逐通道缩放向量 | [S016](#S016) |
| $\mathbf{A}_i,\mathbf{B}_i,\ r$ | LoRA 的两个低秩矩阵与秩（默认 $r=4$） | [S016](#S016) |
| $\oslash,\ \odot$ | 逐元素除、逐元素乘 | [S016](#S016) |
| $L_s,\ L_i,\ L_d$ | 浅层、中间层、深层数量（默认 $L_s=L_d=4$） | [S014](#S014) |
| $\gamma,\ N$ | 层内滑动比例、阶段数 $N=1/\gamma$（默认 $\gamma=0.5,\ N=2$） | [S015](#S015) |
| $b$ | 量化位宽 | [E003](#E003) |
| CS / LoRA | 通道缩放（channel scaling）/ 低秩适配 | [S016](#S016) |

:::

---

## 摘要 / Abstract

<a id="S001"></a>
**Source:** p.1 S001

:::en
**Original:** In this paper, we address post-training quantization (PTQ) for large language models (LLMs) from an overlooked perspective: given a pre-trained high-precision LLM, the predominant sequential quantization framework treats different layers equally, but this may be not optimal in challenging low bit-width settings. Based on three major categories of sequential PTQ methods, we empirically study the quantization impact of different layers on model accuracy, and observe that: (1) shallow/deep layers are usually more sensitive to quantization than intermediate layers; (2) among shallow/deep layers, the most sensitive one is the first/last layer, which exhibits significantly larger quantization error than others. These empirical observations imply that the quantization design for different layers of LLMs is required on multiple levels instead of a single level shared to all layers. Motivated by this, we propose a new PTQ framework termed Sliding-layer Quantization (SliderQuant) that relies on a simple adaptive sliding quantization concept facilitated by few learnable parameters. The base component of SliderQuant is called inter-layer sliding quantization, which incorporates three types of novel sliding window designs tailored for addressing the varying quantization sensitivity of shallow, intermediate and deep layers. The other component is called intra-layer sliding quantization that leverages an incremental strategy to quantize each window. As a result, SliderQuant has a strong ability to reduce quantization errors across layers. Extensive experiments on basic language generation, zero-shot commonsense reasoning and challenging math and code tasks with various LLMs (including Llama/Llama2/Llama3/Qwen2.5 model families, DeepSeek-R1 distilled models and large MoE models) show that our method outperforms existing PTQ methods (including the latest PTQ methods using rotation transformations) for both weight-only quantization and weight-activation quantization under diverse bit width settings. Code is available at https://github.com/deep-optimization/SliderQuant.
:::

:::zh
**中文:** 本文从一个被忽视的角度研究大语言模型（LLM）的训练后量化（PTQ）：给定一个预训练的高精度 LLM，主流的序列量化框架对不同层一视同仁，但在困难的低位宽设定下这未必最优。基于序列 PTQ 方法的三大类别，我们实证研究了不同层对模型精度的影响，并观察到：(1) 浅层/深层通常比中间层对量化更敏感；(2) 在浅层/深层之中，最敏感的是第一层/最后一层，其量化误差显著大于其他层。这些实证观察意味着：LLM 不同层的量化设计需要在多个层次上进行，而不是所有层共用同一个层次的设计。受此启发，我们提出一个新的 PTQ 框架，称为滑动层量化（Sliding-layer Quantization，SliderQuant），它依托一种简单的自适应滑动量化概念，只需少量可学习参数支撑。SliderQuant 的基础组件称为**层间滑动量化**，它包含三种新颖的滑动窗口设计，分别针对浅层、中间层与深层各异的量化敏感度。另一个组件称为**层内滑动量化**，它利用增量策略对每个窗口进行量化。因此，SliderQuant 具有很强的跨层降低量化误差的能力。在基础语言生成、零样本常识推理以及具有挑战性的数学与代码任务上，针对多种 LLM（包括 Llama/Llama2/Llama3/Qwen2.5 系列、DeepSeek-R1 蒸馏模型与大规模 MoE 模型）的大量实验表明：在多种位宽设定下，无论仅权重量化还是权重-激活量化，我们的方法都优于现有 PTQ 方法（包括使用旋转变换的最新 PTQ 方法）。代码见 https://github.com/deep-optimization/SliderQuant。
:::

---

## 1. 引言 / Introduction

<a id="S002"></a>
**Source:** p.1 S002

:::en
**Original:** Transformer-based large language models (LLMs) (Vaswani et al., 2017; Brown et al., 2020; Achiam et al., 2023; Anil et al., 2023; Liu et al., 2024a; Jaech et al., 2024; Guo et al., 2025) have demonstrated extraordinary performance on a wide range of natural language processing tasks. However, deploying them in real-world scenarios poses a great challenge due to their huge model sizes. Post-training quantization (PTQ) is a practically appealing way to reduce memory and computation demands of LLMs at inference. It approximates pre-trained high-precision models with low-precision replacements conditioned on a small number of calibration samples, without the need of the expensive retraining pipeline used in quantization-aware training (Zafrir et al., 2019; Ma et al., 2024a; Xu et al., 2024). Because of this, PTQ research has gained increasing attention in the LLM community.
:::

:::zh
**中文:** 基于 Transformer 的大语言模型（LLM）（Vaswani et al., 2017; Brown et al., 2020; Achiam et al., 2023; Anil et al., 2023; Liu et al., 2024a; Jaech et al., 2024; Guo et al., 2025）在广泛的自然语言处理任务上展现出非凡的性能。然而，由于其庞大的模型规模，把它们部署到真实场景中是一个巨大挑战。训练后量化（PTQ）是降低 LLM 推理时内存与计算需求的一种实用且诱人的途径。它在一小批校准样本的条件下，用低精度替代品逼近预训练的高精度模型，无需量化感知训练那套昂贵的重训练流程（Zafrir et al., 2019; Ma et al., 2024a; Xu et al., 2024）。正因如此，PTQ 研究在 LLM 社区获得了越来越多的关注。
:::

<a id="S003"></a>
**Source:** p.1 S003

:::en
**Original:** Existing PTQ methods for LLMs generally use a sequential quantization framework: splitting a pre-trained LLM into the same-sized disjoint parts, and then quantizing them from the first to the last part separately. Early seminal works, such as LLM.int8() (Dettmers et al., 2022), ZeroQuant (Yao et al., 2022), GPTQ (Frantar et al., 2023) and SmoothQuant (Xiao et al., 2023), assume that different layers of a pre-trained LLM are independent to each other, and use the layer-wise quantization.
:::

:::zh
**中文:** 现有的 LLM PTQ 方法通常采用**序列量化框架**：把预训练 LLM 切成大小相同、彼此不相交的若干部分，然后从第一部分到最后一个部分依次分别量化。早期开创性工作，如 LLM.int8()（Dettmers et al., 2022）、ZeroQuant（Yao et al., 2022）、GPTQ（Frantar et al., 2023）与 SmoothQuant（Xiao et al., 2023），假定预训练 LLM 的不同层彼此独立，采用**逐层量化**。
:::

<a id="S004"></a>
**Source:** p.2 S004

:::en
**Original:** This simple framework has been popularly adopted by many subsequent works (Lin et al., 2024b; Ashkboos et al., 2024b; Lin et al., 2024a; Dettmers et al., 2024; Duanmu et al., 2024). Instead, OmniQuant (Shao et al., 2024) and FlatQuant (Sun et al., 2025) employ the block-wise quantization in which layers within each attention block are quantized simultaneously. To explore longer-distance dependencies than the block-wise quantization, QLLM (Liu et al., 2024b) and CBQ (Ding et al., 2025) utilize the multi-block-wise quantization based on a fixed-size sliding window, resembling the insights of prior works (Li et al., 2021; Zheng et al., 2022) tailored for quantizing convolutional neural networks in computer vision. These methods have significantly advanced the post-training quantization research for LLMs. However, in formulation, they typically treat different layers of a pre-trained LLM equally, no matter using the layer-wise or block-wise or multi-block-wise quantization. Such an assumption appears reasonable under moderate quantization settings, e.g., 8-bit weight quantization, as the total quantization error tends to be small. However, we conjecture it may be not optimal under challenging quantization settings, e.g., 4-bit weight-activation quantization.
:::

:::zh
**中文:** 这个简单的框架被许多后续工作广泛采用（Lin et al., 2024b; Ashkboos et al., 2024b; Lin et al., 2024a; Dettmers et al., 2024; Duanmu et al., 2024）。与之不同，OmniQuant（Shao et al., 2024）与 FlatQuant（Sun et al., 2025）采用**逐块量化**，即每个注意力块内的层被同时量化。为了探索比逐块量化更长距离的依赖，QLLM（Liu et al., 2024b）与 CBQ（Ding et al., 2025）利用基于固定尺寸滑动窗口的**多块量化**，这与面向计算机视觉中卷积神经网络量化的先前工作（Li et al., 2021; Zheng et al., 2022）的思路相似。这些方法显著推进了 LLM 训练后量化研究。然而在形式化上，无论采用逐层、逐块还是多块量化，它们通常都把预训练 LLM 的不同层**同等对待**。在适中的量化设定下（例如 8 比特权重量化），这一假设看似合理，因为总量化误差往往很小；但我们推测，在困难的量化设定下（例如 4 比特权重-激活量化），它可能并非最优。
:::

<a id="S005"></a>
**Source:** p.2 S005

:::en
**Original:** For the sequential quantization framework described above, some previous works (Nagel et al., 2020; Li et al., 2021; Frantar et al., 2023; Shao et al., 2024; Ding et al., 2025) have established a solid theoretical foundation based on second-order Taylor expansion to get a better approximation solution for post-training quantization. This paper moves one step further: we revisit the predominant sequential quantization framework via questioning whether different layers of currently prevailing LLMs have similar quantization impacts on model accuracy. To explore this question, we select three representative layer-wise, block-wise and multi-block-wise quantization methods, SmoothQuant, OmniQuant and CBQ, and examine them on a lot of popular LLMs in the challenging 4-bit weight-activation quantization regime. Of particular interest, we have observed several properties. Firstly, for each of our tested LLMs, intermediate layers usually have smaller quantization impacts on model accuracy compared to shallow/deep layers. This implies that shallow/deep layers are more sensitive to quantization than intermediate layers which are relatively easy to quantize. Secondly, among shallow/deep layers, the first/last layer has the largest quantization impact on model accuracy, exhibiting significantly larger quantization error than others. This implies that the first and last layers are greatly important in the quantization process, as they are responsible for the very basic feature extraction and the final feature abstraction, respectively. Thirdly, as more layers are sequentially quantized, the quantization impact on model accuracy will be magnified gradually. However, SmoothQuant, OmniQuant and CBQ show unsatisfactory abilities to suppress this issue, suffering from the underlying premise that all layers are treated equally in their layer-wise, block-wise and multi-block-wise quantization frameworks. Figure 1 illustrates these properties on Llama2-7B, Llama2-13B (Touvron et al., 2023b) and Qwen2.5-14B (Yang et al., 2024).
:::

:::zh
**中文:** 对于上述序列量化框架，一些先前工作（Nagel et al., 2020; Li et al., 2021; Frantar et al., 2023; Shao et al., 2024; Ding et al., 2025）已基于二阶 Taylor 展开建立了坚实的理论基础，为训练后量化求得更优的近似解。本文再进一步：我们重新审视这一主流序列量化框架，追问当前主流 LLM 的不同层对模型精度是否具有相似的量化影响。为探究这个问题，我们选取三种代表性的逐层、逐块与多块量化方法——SmoothQuant、OmniQuant 与 CBQ——在困难的 4 比特权重-激活量化体制下，于大量流行 LLM 上进行了考察。特别值得关注的是，我们观察到若干性质。第一，对我们测试的每个 LLM，中间层对模型精度的影响通常小于浅层/深层。这意味着浅层/深层比相对易于量化的中间层对量化更敏感。第二，在浅层/深层之中，第一层/最后一层对模型精度的影响最大，其量化误差显著大于其他层。这意味着第一层与最后一层在量化过程中极其重要，因为它们分别负责最基础的特征提取与最终的特征抽象。第三，随着被序列量化的层数增多，量化对模型精度的影响会逐步放大。然而 SmoothQuant、OmniQuant 与 CBQ 抑制这一问题的能力并不理想，根源在于其逐层、逐块与多块量化框架都把所有权重层同等对待这一前提。图 1 在 Llama2-7B、Llama2-13B（Touvron et al., 2023b）与 Qwen2.5-14B（Yang et al., 2024）上展示了这些性质。
:::

<a id="F001"></a>
#### Fig. 1. <span class="lang-zh">不同层对模型精度的量化影响</span>

**Placed near:** p.2 S005<br>
**Source:** p.2 C001

<img src="/images/quantization/sliderquant/figures/fig01.png" alt="Figure 1" width="1326" height="736" loading="lazy" decoding="async">

:::en
**Original caption:** Illustrations on the quantization impact of different layers to model accuracy: (1) quantizing a single layer (the first row) and (2) quantizing the first $l$ layers (the second row) of Llama2-7B, Llama2-13B and Qwen2.5-14B. Here, we select 3 representative layer-wise, block-wise and multi-block-wise quantization methods, SmoothQuant, OmniQuant and CBQ, and examine them under 4-bit weight-activation (W4A4) quantization on WikiText2. In the Appendix, Figure C provides more illustrations on Llama3-8B, Qwen2.5-7B and Qwen2.5-32B, showing similar observations.
:::

:::zh
**中文图注:** 不同层对模型精度的量化影响示意：(1) 量化单个层（第一行）；(2) 量化前 $l$ 层（第二行），模型为 Llama2-7B、Llama2-13B 与 Qwen2.5-14B。这里选取三种代表性的逐层、逐块与多块量化方法 SmoothQuant、OmniQuant 与 CBQ，在 WikiText2 上以 4 比特权重-激活（W4A4）量化进行考察。附录图 C 给出了 Llama3-8B、Qwen2.5-7B 与 Qwen2.5-32B 上的更多示意，观察结论类似。
:::

:::zh
**Reading note:** 上排横轴是“层索引”，纵轴是只量化该层时的 WikiText2 困惑度，虚线是 FP16 基线。可以看到 **U 形**：首层最高（SmoothQuant 约 5.9），中间层贴近虚线，末层又抬升。下排横轴是“被量化的层数”，曲线随层数单调上升，但 SliderQuant（红线）几乎贴住虚线，说明重叠窗口确实抑制了误差累积——这张图就是全文的动机来源。
:::

<a id="S006"></a>
**Source:** p.2 S006

:::en
**Original:** These empirical properties highlight two ingredients that are essential to formulate an improved sequential quantization framework adopting a fixed bit-width: (1) special attention during quantization is required on shallow and deep layers, particularly the first and last layers; (2) the quantization synergy of successive layers is required to reduce quantization errors across layers. Driven by this, we
:::

:::zh
**中文:** 这些实证性质凸显了两个要素，它们对构建一个采用固定位宽的改进型序列量化框架至关重要：(1) 量化过程中需要对浅层与深层、尤其是第一层与最后一层给予特别关注；(2) 需要相邻层之间的**量化协同**，以降低跨层的量化误差。受此驱动，我们
:::

<a id="S007"></a>
**Source:** p.3 S007

:::en
**Original:** present a new sequential quantization framework, Sliding-layer Quantization (SliderQuant) shown in Figure 2, which relies on a simple adaptive sliding quantization concept. In principle, by adopting a sliding window, the layers of a pre-trained LLM are sequentially divided into overlapping windows with the same size first, and then the quantization is performed window by window facilitated by few learnable quantization parameters. The overlaps of consecutive windows establish a basic quantization synergy path to reduce quantization errors across layers. However, simply using a fixed-size sliding window (Duanmu et al., 2024; Ding et al., 2025) still has a large gap to endow SliderQuant with the desired two ingredients, as shallow, intermediate and deep layers will be quantized with the same window size and moving interval per step and treated equally important. We fill this gap by presenting two novel sliding quantization components. Our base component, inter-layer sliding quantization, incorporates three types of sliding window designs tailored for adaptively quantizing shallow, intermediate and deep layers with a smart optimization relay across them. Specifically, it first allocates a progressively expanded sliding window along shallow layers, a fixed-size sliding window along intermediate layers and a progressively contracted sliding window along deep layers, and then performs the sliding quantization progressively.
:::

:::zh
**中文:** 提出一个新的序列量化框架——滑动层量化（SliderQuant），如图 2 所示，它依托一种简单的自适应滑动量化概念。原则上，借助一个滑动窗口，预训练 LLM 的各层先被顺序划分成大小相同、彼此重叠的窗口，然后在少量可学习量化参数的辅助下逐窗口进行量化。相邻窗口的重叠建立了一条基本的量化协同路径，用以降低跨层量化误差。然而，仅仅使用固定尺寸的滑动窗口（Duanmu et al., 2024; Ding et al., 2025）仍与 SliderQuant 所需的两个要素相去甚远，因为浅层、中间层与深层会以相同的窗口大小和每步移动间隔被量化，并被视作同等重要。为填补这一差距，我们提出两个新颖的滑动量化组件。我们的基础组件——**层间滑动量化**——融合了三种滑动窗口设计，通过它们之间巧妙的优化接力，自适应地量化浅层、中间层与深层。具体而言，它先在浅层上分配一个**渐进扩张**的滑动窗口，在中间层上分配一个**固定尺寸**的滑动窗口，在深层上分配一个**渐进收缩**的滑动窗口，然后渐进地执行滑动量化。
:::

<a id="S008"></a>
**Source:** p.3 S008

:::en
**Original:** With these three types of sliding window designs, our inter-layer sliding quantization component can leverage the aforementioned empirical properties about the varying layer sensitivity to quantization. To exploit these empirical properties further, we present another complementary component, called intra-layer sliding quantization. It extends the progressively expanded sliding design within each window of inter-layer sliding quantization component, by which all layers in each window are jointly quantized in an incremental manner. Coupling these two components in this way forms a neat implementation of SliderQuant.
:::

:::zh
**中文:** 借助这三类滑动窗口设计，我们的层间滑动量化组件能够利用前述关于“各层量化敏感度不同”的实证性质。为进一步利用这些性质，我们提出另一个互补组件，称为**层内滑动量化**。它把渐进扩张的滑动设计延伸到层间滑动量化组件的每一个窗口之内，使得每个窗口中的所有层以增量的方式被联合量化。以这种方式耦合两个组件，就构成了 SliderQuant 的一套简洁实现。
:::

<a id="S009"></a>
**Source:** p.3 S009

:::en
**Original:** Overall, SliderQuant is a flexible PTQ framework which can be used for both weight-only and weight-activation quantization. Our experimental results show that SliderQuant outperforms existing PTQ methods across a broad range of quantization settings (W4A16, W3A16, W2A16, W4A4), model families (Llama, Llama2, Llama3, Qwen2.5, Qwen3) and model sizes (7B, 8B, 13B, 14B, 32B, 65B, 70B) on 2 basic language generation and 6 commonsense reasoning benchmarks. Incorporating advanced techniques (e.g., rotation transformations) into SliderQuant further improves its performance. In addition, we validate the effectiveness of our SliderQuant on the advanced MoE model Qwen3-30B-A3B. Notably, we also apply SliderQuant to the recently popular DeepSeek-R1 (Guo et al., 2025) distilled models with strong chain-of-thought reasoning abilities, achieving near-lossless accuracy under 4-bit weight-only quantization on 5 challenging math and code tasks.
:::

:::zh
**中文:** 总体而言，SliderQuant 是一个灵活的 PTQ 框架，既可用于仅权重量化，也可用于权重-激活量化。实验结果表明，在广泛的量化设定（W4A16、W3A16、W2A16、W4A4）、模型家族（Llama、Llama2、Llama3、Qwen2.5、Qwen3）与模型规模（7B、8B、13B、14B、32B、65B、70B）上，SliderQuant 在 2 个基础语言生成与 6 个常识推理基准上均优于现有 PTQ 方法。把先进技术（例如旋转变换）引入 SliderQuant 可进一步提升其性能。此外，我们在先进的 MoE 模型 Qwen3-30B-A3B 上验证了 SliderQuant 的有效性。值得注意的是，我们还将 SliderQuant 应用于近期流行的、具有强思维链推理能力的 DeepSeek-R1（Guo et al., 2025）蒸馏模型，在 4 比特仅权重量化下，于 5 个具有挑战性的数学与代码任务上取得了接近无损的精度。
:::

---

## 2. 相关工作 / Related Work

<a id="S010"></a>
**Source:** p.3 S010

:::en
**Original:** Many PTQ works focus on weight-only quantization. Early works, such as Q-BERT (Shen et al., 2020) and GOBO (Zadeh et al., 2020), use a mixed-precision decomposition scheme in which the weight outliers (i.e., a small fraction of weights causing large quantization errors) are retained in high-precision format while the other weights are quantized into low-precision format. They consider small language models like BERT (Devlin et al., 2019). GPTQ (Frantar et al., 2023) formulates an efficient weight-only quantization approach using approximate second-order Hessian matrices. Compared to the popular round-to-nearest (RTN) quantization method (Dettmers et al., 2022), GPTQ shows much better performance when quantizing weights to 3-bit/4-bit. QuIP (Chee et al., 2024) introduces an incoherence-driven scheme to promote GPTQ, especially for the 2-bit quantization of weights, and an improved variant is further presented in (Tseng et al., 2024). SpQR (Dettmers et al., 2024) and AWQ (Lin et al., 2024b) extend the mixed-precision decomposition scheme through designing more effective strategies to identify weight outliers, achieving superior performance to GPTQ. To avoid the inefficiency of the mixed-precision implementation on hardware systems (Cai et al., 2020), AWQ searches channel-wise factors to scale down weight outliers, enabling full-weight quantization. Other works, including but not limited to (Shen et al., 2020), (Tang et al., 2023), (Park et al., 2022) and (Kim et al., 2024), try to advance the weight-only PTQ research from other aspects.
:::

:::zh
**中文:** 许多 PTQ 工作聚焦于仅权重量化。早期工作如 Q-BERT（Shen et al., 2020）与 GOBO（Zadeh et al., 2020）采用混合精度分解方案：把权重离群值（即一小部分会造成大量化误差的权重）保留为高精度格式，其余权重则量化为低精度格式。它们考虑的是 BERT（Devlin et al., 2019）这类小型语言模型。GPTQ（Frantar et al., 2023）利用近似二阶 Hessian 矩阵，构建了一种高效的仅权重量化方法。与流行的最近舍入（RTN）量化方法（Dettmers et al., 2022）相比，GPTQ 在把权重压到 3 比特/4 比特时表现明显更好。QuIP（Chee et al., 2024）引入一种以“不相干性”为驱动的方案来提升 GPTQ，尤其针对权重的 2 比特量化，其改进变体在（Tseng et al., 2024）中进一步给出。SpQR（Dettmers et al., 2024）与 AWQ（Lin et al., 2024b）通过设计更有效的权重离群值识别策略来扩展混合精度分解方案，性能优于 GPTQ。为避免混合精度实现在硬件系统上的低效（Cai et al., 2020），AWQ 搜索逐通道因子来缩小权重离群值，从而实现全权重量化。其他工作——包括但不限于（Shen et al., 2020）、（Tang et al., 2023）、（Park et al., 2022）与（Kim et al., 2024）——则试图从其他方面推进仅权重量化 PTQ 研究。
:::

<a id="S011"></a>
**Source:** p.3 S011

:::en
**Original:** Compared to weight-only quantization, weight-activation quantization can bring more significant reductions in both compute and storage costs at inference, but it is more challenging due to its higher sensitivity to quantization error. LLM.int8() (Dettmers et al., 2022), a pioneering work for weight-activation quantization, uses a mixed-precision scheme that quantizes all weights and most of activations into INT8 format, but isolates activation outliers (i.e., a small fraction of activations that have large magnitudes than the others) into FP16 format. To suppress activation outliers, the authors of (Wei et al., 2023) employ the non-scaling layer normalization and the token-wise clipping, making activations to be more friendly for 8-bit quantization. Unlike LLM.int8() using the vanilla vector-wise quantization with RTN, ZeroQuant (Yao et al., 2022) applies a fine-grained INT-8 quantization scheme consisting of group-wise quantization for weights and token-wise quantization for activations. Based on a linear equivalent transformation, SmoothQuant (Xiao et al., 2023) uses per-channel smoothing factors to scale down activation outliers and scale up the corresponding weights, mitigating the quantization difficulty from activations to weights which are easier to quantize. SmoothQuant calculates channel-wise smoothing factors over a randomly sampled calibration set in an offline manner, and is tailored for 8-bit weight-activation quantization. Instead, OmniQuant (Shao et al., 2024) dynamically learns activation-smoothing factors and weight-clipping thresholds, and considers more diverse bit-width settings down to 4-bit. QUIK (Ashkboos et al., 2024a) addresses 4-bit weight-activation quantization by extending the mixed-precision scheme. QLLM (Liu et al., 2024b) formulates a channel disassembly and channel assembly scheme facilitated by the low-rank adaptation (Hu et al., 2022) to suppress outliers in some channels. However, this scheme modifies LLM architectures, and thus introduces extra inference-time costs. QuaRot (Ashkboos et al., 2024b) uses random rotation transformations to remove outliers from the hidden state. Some subsequent works improve QuaRot by making rotation transformations learnable (Liu et al., 2025; Sun et al., 2025) or combining rotation and permutation transformations (Lin et al., 2024a). Similar to QLLM, these rotation-based methods also introduce extra computational costs at inference, as rotation transformations added to some layers are not absorbable due to non-linear operations.
:::

:::zh
**中文:** 与仅权重量化相比，权重-激活量化能在推理时带来更显著的计算与存储开销下降，但由于对量化误差更敏感，它也更困难。LLM.int8()（Dettmers et al., 2022）是权重-激活量化的开创性工作，它采用混合精度方案：所有权重与大部分激活被量化为 INT8，但激活离群值（即一小部分幅度远大于其他的激活）被单独保留为 FP16。为抑制激活离群值，（Wei et al., 2023）的作者采用不带缩放的层归一化与逐 token 裁剪，使激活对 8 比特量化更友好。与 LLM.int8() 使用朴素的向量级 RTN 量化不同，ZeroQuant（Yao et al., 2022）采用细粒度 INT8 量化方案：权重做分组量化、激活做逐 token 量化。基于一种线性等价变换，SmoothQuant（Xiao et al., 2023）用逐通道平滑因子把激活离群值缩小、同时把对应权重放大，从而把量化困难从激活转移到更易量化的权重上。SmoothQuant 以离线方式在随机采样的校准集上计算逐通道平滑因子，专为 8 比特权重-激活量化设计。与之不同，OmniQuant（Shao et al., 2024）动态学习激活平滑因子与权重裁剪阈值，并考虑低至 4 比特的更多样位宽设定。QUIK（Ashkboos et al., 2024a）通过扩展混合精度方案来处理 4 比特权重-激活量化。QLLM（Liu et al., 2024b）借助低秩适配（Hu et al., 2022）构建通道拆解与通道组装方案，以抑制某些通道的离群值；然而该方案改动了 LLM 架构，因而引入额外的推理时开销。QuaRot（Ashkboos et al., 2024b）使用随机旋转变换从隐藏状态中去除离群值。后续一些工作通过让旋转变换可学习（Liu et al., 2025; Sun et al., 2025）或把旋转与置换变换结合（Lin et al., 2024a）来改进 QuaRot。与 QLLM 类似，这些基于旋转的方法在推理时同样引入额外计算开销，因为加在某些层上的旋转变换由于非线性操作而无法被吸收。
:::

---

## 3. 方法 / Method

<a id="S012"></a>
**Source:** p.4 S012

:::en
**Original:** In this section, we describe the formulation of our SliderQuant and detail its implementation.
:::

:::zh
**中文:** 本节我们描述 SliderQuant 的形式化表述，并详述其实现。
:::

### 3.1 基本概念：固定尺寸滑动量化 / Fixed-Size Sliding Quantization

<a id="S013"></a>
**Source:** p.4 S013

:::en
**Original:** Given a pre-trained high-precision LLM having $L$ layers, we start with the vanilla sliding quantization (Duanmu et al., 2024; Ding et al., 2025). It uses a fixed-size sliding window $\{s,i\}$ moving along the layer direction of the given model and performs the sequential quantization in a window-wise manner, where $s$ denotes the window size and $i$ denotes the moving interval per step. The overlap between two consecutive windows is $s-i$. Let $\mathbf{W}=\{\mathbf{W}_1,...\mathbf{W}_s\}$ be the pre-trained weight matrix set for $s$ layers in the current window and let $\mathbf{X}$ be its input feature corresponding to a small set of $c$ calibration samples. Then, for weight-only quantization, the optimization objective is defined as
:::

:::zh
**中文:** 给定一个有 $L$ 层的预训练高精度 LLM，我们从朴素的滑动量化出发（Duanmu et al., 2024; Ding et al., 2025）。它使用一个固定尺寸的滑动窗口 $\{s,i\}$ 沿给定模型的层方向移动，以逐窗口的方式执行序列量化；其中 $s$ 表示窗口大小，$i$ 表示每步移动间隔。相邻两个窗口之间的重叠为 $s-i$。设 $\mathbf{W}=\{\mathbf{W}_1,...\mathbf{W}_s\}$ 为当前窗口中 $s$ 层的预训练权重矩阵集合，$\mathbf{X}$ 为其对应于 $c$ 个校准样本的输入特征。那么，对仅权重量化而言，优化目标定义为
:::

<a id="E001"></a>
**Source:** p.4 E001 · Eq. (1)

$$
\underset{\hat{\mathbf{W}}}{\operatorname{argmin}}\ \big\| \mathcal{F}(\mathbf{W},\mathbf{X}) - \mathcal{F}(\hat{\mathbf{W}},\mathbf{X}) \big\|_2^2
$$

:::zh
**中文说明：** 该式即“用当前窗口的输入输出对齐误差来挑选低精度权重”。$\mathcal{F}(\cdot,\cdot)$ 是窗口的前向输出，$\hat{\mathbf{W}}$ 是待定的低精度权重集；最小化二者输出特征的 $L_2$ 距离平方，等价于要求量化窗口在**校准数据上**复现 FP16 窗口的行为。注意这里优化的是整个窗口的**输出**，而不是逐层的权重近似误差——这是滑动窗口类方法能“互相补偿”误差的根源。
:::

<a id="S014"></a>
**Source:** p.4 S014

:::en
**Original:** where $\mathcal{F}(\cdot,\cdot)$ denotes the output feature of the current window, and $\hat{\mathbf{W}}=\{\hat{\mathbf{W}}_1,...\hat{\mathbf{W}}_s\}$ denotes the low-precision weight matrix set needs to be determined. For weight-activation quantization, the low-precision input feature $\hat{\mathbf{X}}$ is obtained from the quantization of $\mathbf{X}$ beforehand, and then its optimization objective can be defined by simply replacing $\mathcal{F}(\hat{\mathbf{W}},\mathbf{X})$ in Eq. 1 by $\mathcal{F}(\hat{\mathbf{W}},\hat{\mathbf{X}})$.
:::

:::zh
**中文:** 其中 $\mathcal{F}(\cdot,\cdot)$ 表示当前窗口的输出特征，$\hat{\mathbf{W}}=\{\hat{\mathbf{W}}_1,...\hat{\mathbf{W}}_s\}$ 表示待确定的低精度权重矩阵集合。对于权重-激活量化，低精度输入特征 $\hat{\mathbf{X}}$ 由 $\mathbf{X}$ 预先量化得到，此时只需把式 1 中的 $\mathcal{F}(\hat{\mathbf{W}},\mathbf{X})$ 替换为 $\mathcal{F}(\hat{\mathbf{W}},\hat{\mathbf{X}})$，即可定义其优化目标。
:::

<a id="S015"></a>
**Source:** p.4 S015

:::en
**Original:** According to the above definition, when the window size $s$ is one layer or one attention block or multiple attention blocks and the moving interval per step $i$ is equal to the window size $s$ (i.e., there is no overlap between two consecutive windows), we will get layer-wise, block-wise and multi-block-wise quantization frameworks popularly used in existing PTQ works (Yao et al., 2022; Frantar et al., 2023; Xiao et al., 2023; Shao et al., 2024; Liu et al., 2024b). That is, they are special cases of fixed-sized sliding quantization. For sliding PTQ methods including ours, a larger window size leads to increased memory cost but enjoys much better accuracy (see Table 1) compared to layer-wise PTQ methods like GPTQ (the most efficient PTQ method tailored for weight-only quantization).
:::

:::zh
**中文:** 按照上述定义，当窗口大小 $s$ 为一层、一个注意力块或多个注意力块，且每步移动间隔 $i$ 等于窗口大小 $s$（即相邻窗口之间没有重叠）时，我们就得到现有 PTQ 工作广泛使用的逐层、逐块与多块量化框架（Yao et al., 2022; Frantar et al., 2023; Xiao et al., 2023; Shao et al., 2024; Liu et al., 2024b）。也就是说，它们是固定尺寸滑动量化的**特例**。对于包括本文在内的滑动 PTQ 方法，更大的窗口会带来更高的显存开销，但与 GPTQ（为仅权重量化量身定制的最有效 PTQ 方法）这类逐层 PTQ 方法相比，精度要好得多（见表 1）。
:::

### 3.2 SliderQuant

<a id="S016"></a>
**Source:** p.4–5 S016

:::en
**Original:** Recall that our empirical observations underlie two ingredients that are crucial to improve the sequential quantization framework. Firstly, special attention during quantization is required on shallow and deep layers, particularly the first and last layers, as they are more sensitive to quantization than intermediate layers. Secondly, the quantization synergy of successive layers is required to reduce quantization errors across layers. For fixed-size sliding quantization (we use it as the baseline sliding quantization design in our ablations), the existence of an overlap $s-i\geq1$ between any two consecutive windows builds a basic synergy path to reduce quantization errors across layers. However, when using a fixed-size sliding window, all layers of a pre-trained LLM will be quantized with the same window size and moving interval per step. That is, shallow, intermediate and deep layers are still treated equally to a great extent, leading to a large gap to have the desired quantization design. We present SliderQuant to fill this gap via designing a more adaptive sliding quantization framework, which consists of two novel sliding quantization components, as shown in Figure 2.
:::

:::zh
**中文:** 回顾我们的实证观察：有两个要素对改进序列量化框架至关重要。第一，量化过程中需要对浅层与深层、尤其是第一层与最后一层给予特别关注，因为它们比中间层对量化更敏感。第二，需要相邻层之间的量化协同，以降低跨层量化误差。对于固定尺寸滑动量化（我们在消融中把它作为基线滑动量化设计），任意两个相邻窗口之间存在重叠 $s-i\geq1$，这建立了一条降低跨层量化误差的基本协同路径。然而使用固定尺寸滑动窗口时，预训练 LLM 的所有层都会以相同的窗口大小与每步移动间隔被量化。也就是说，浅层、中间层与深层在很大程度上仍被同等对待，与理想的量化设计存在很大差距。我们提出 SliderQuant 来填补这一差距：设计一个更自适应的滑动量化框架，它由两个新颖的滑动量化组件构成，如图 2 所示。
:::

<a id="F002"></a>
#### Fig. 2. <span class="lang-zh">SliderQuant 总览：层间滑动 + 层内滑动</span>

**Placed near:** p.5 S016<br>
**Source:** p.5 C002

<img src="/images/quantization/sliderquant/figures/fig02.png" alt="Figure 2" width="1660" height="594" loading="lazy" decoding="async">

:::en
**Original caption:** Overview of SliderQuant consisting of two components based on a simple adaptive sliding quantization concept. The base component of SliderQuant, inter-layer sliding quantization, has three sliding window designs along shallow, intermediate and deep layers of any pre-trained high-precision LLM, which are tailored for addressing their varying layer sensitivity to quantization. To establish a smooth sliding quantization relay from shallow to intermediate layers then from intermediate to deep layers, we set one overlapped layer between shallow and intermediate layers and one overlapped layer between intermediate and deep layers. Besides, this also makes each intermediate layer have an even quantization frequency. The other component of SliderQuant, intra-layer sliding quantization, is applied within the current window of inter-layer sliding quantization component, by which all layers in the current window are jointly quantized in an incremental manner.
:::

:::zh
**中文图注:** SliderQuant 总览，它由两个组件构成，基于一个简单的自适应滑动量化概念。SliderQuant 的基础组件——层间滑动量化——在任意预训练高精度 LLM 的浅层、中间层与深层上采用三种滑动窗口设计，分别针对它们各异的量化敏感度。为了建立从浅层到中间层、再从中间层到深层的平滑滑动量化接力，我们在浅层与中间层之间设置一个重叠层，在中间层与深层之间也设置一个重叠层；此外，这也使得每个中间层具有均匀的量化频次。SliderQuant 的另一个组件——层内滑动量化——施加在层间滑动量化组件的当前窗口内部，使当前窗口中的所有层以增量方式被联合量化。
:::

:::zh
**Reading note:** 左半部分是**层间**：三组箭头分别对应渐进扩张（PESW，窗口 1→2→3→4，首层始终在窗内）、固定尺寸（FSSW，$s=2,i=1$，每次滑动一层）、渐进收缩（PCSW，窗口 4→3→2→1，末层始终在窗内）；下方矩阵的灰度表示“已量化/正在量化/未量化”，箭头方向就是量化顺序。右半部分是**层内**：在一个窗口内部，Stage 1 先量化“前一半”（灰色格），Stage 2 再把整个窗口一起量化；红框标出的 group 从 `1 0 2.6 5.3` 变成 `1 0 3 5`，示意同一组权重被重估。注意两层含义的“滑”是**正交**的：层间决定哪些层一起优化，层内决定窗口内按什么比例分批加入。
:::

<a id="S017"></a>
**Source:** p.5 S017

:::en
**Original:** **Inter-Layer Sliding Quantization.** Our base component, inter-layer sliding quantization, incorporates three types of sliding window designs tailored for adaptively quantizing shallow, intermediate and deep layers with a smart optimization relay across them. For $L_s$ shallow layers, a progressively expanded sliding window (PESW) is designed, which starts from quantizing the first layer with the window size of 1, then takes the first layer as the anchor layer and gradually increases the window size by 1 per step until including all shallow layers. With PESW, the first layer is always involved in the quantization process with every expanded sliding window, building dense local to global synergies to ease the quantization of shallow layers. Reversely, a progressively contracted sliding window (PCSW) is designed for $L_d$ deep layers, which starts from quantizing all deep layers, and then gradually decreases the window size by 1 per step until to only include the last layer. With PCSW, the last layer is always used as the anchor layer, and it is involved in the quantization process with every contracted sliding window, building dense global to local synergies to ease the quantization of deep layers. For $L_i$ intermediate layers, we adopt a fixed-size sliding window $\{s=2,i=1\}$ (FSSW) and an even optimization frequency across layers. Specifically, we set one overlapped layer between shallow and intermediate layers, and also set one overlapped layer between intermediate and deep layers. Alternatively, when the window size of fixed-size sliding quantization at intermediate layers is larger than 2 (i.e., $s>2$), we can easily ensure an even optimization frequency by changing the number of overlapped layers between intermediate layers and shallow/deep layers, e.g., 2 overlapped layers when $s=3$. Sequentially applying these three types of sliding window designs establishes the desired optimization relay across all layers of any pre-trained LLM, flexibly leveraging our identified empirical properties about the varying layer sensitivity to quantization. According to the ablative experiments, we set $L_s=4,L_d=4$ as default for an accuracy-efficiency trade-off.
:::

:::zh
**中文:** **层间滑动量化。** 我们的基础组件——层间滑动量化——融合三类滑动窗口设计，通过它们之间巧妙的优化接力来自适应地量化浅层、中间层与深层。对于 $L_s$ 个浅层，设计了**渐进扩张滑动窗口（PESW）**：它从窗口大小为 1 量化第一层开始，然后以第一层为锚层，每步把窗口大小增加 1，直到覆盖所有浅层。借助 PESW，第一层始终参与每一个扩张窗口的量化过程，从而建立从局部到全局的稠密协同，减轻浅层的量化难度。相反地，对于 $L_d$ 个深层，设计了**渐进收缩滑动窗口（PCSW）**：它从量化所有深层开始，然后每步把窗口大小减 1，直到只包含最后一层。借助 PCSW，最后一层始终被用作锚层，参与每一个收缩窗口的量化过程，从而建立从全局到局部的稠密协同，减轻深层的量化难度。对于 $L_i$ 个中间层，我们采用固定尺寸滑动窗口 $\{s=2,i=1\}$（FSSW），并保证各层优化频次均匀。具体而言，我们在浅层与中间层之间设置一个重叠层，也在中间层与深层之间设置一个重叠层。或者，当中间层固定尺寸滑动量化的窗口大于 2（即 $s>2$）时，我们可以通过改变中间层与浅层/深层之间的重叠层数来轻松保证优化频次均匀，例如 $s=3$ 时取 2 个重叠层。顺序应用这三类滑动窗口设计，就在任意预训练 LLM 的所有层上建立了所需的优化接力，灵活利用了我们所发现的“各层量化敏感度不同”这一实证性质。根据消融实验，为兼顾精度与效率，我们默认设置 $L_s=4,L_d=4$。
:::

:::zh
**阅读提示（读者补充）**：这一段是全文的“设计核心”，可对照 [Fig. 2](#F002) 左半部分读。三个窗口共享同一个思想——**锚层固定、窗口大小单调变化**：

- 浅层用 PESW：窗口从 `{1}` 长到 `{1..L_s}`，首层是锚，越往后“看得越远”，把浅层逐步接入全局；
- 深层用 PCSW：窗口从 `{全部深层}` 缩到 `{最后一层}`，末层是锚，越往后“收得越紧”，把深层逐步收回到局部；
- 中间层用 FSSW（$s=2,i=1$）：每步只滑一层，让中间层各被优化两次、频次均匀。

这样做的效果是：越敏感的层（首/末层）被优化的次数越多，而中间层保持均等的“曝光”。两种锚定方向的对称性（首层局部→全局、末层全局→局部）是论文刻意设计的，对应 §4 消融中 PESW 与 PCSW 缺一不可的结论（[Table 8](#T008)）。
:::

<a id="S018"></a>
**Source:** p.5 S018

:::en
**Original:** **Intra-Layer Sliding Quantization.** To exploit our identified empirical properties further, another complementary component, called intra-layer sliding quantization, is presented. This component is applied into the current sliding window of our base component. Specifically, it extends the progressively expanded sliding design to each window of inter-layer sliding quantization, where its $s$ layers parallelly apply the progressively expanded sliding by a ratio $\gamma$ along weight and activation dimensions. As a result, the joint quantization of all $s$ layers is completed incrementally in $N = 1/\gamma$ sliding stages. We set $\gamma = 0.5, N = 2$ as default, as illustrated in the right part of Figure 2. In the first stage, the first half of weight/activation matrices in the current window of inter-layer sliding quantization is quantized jointly. In the second stage, the whole of weight/activation matrices (including the first half) in the current window of inter-layer sliding quantization is quantized jointly. As a result, intra-layer sliding quantization builds a local to global parameter synergy across layers within the current sliding window of inter-layer sliding quantization to reduce quantization error. Coupling these two components in this way also leads to a neat formulation of SliderQuant.
:::

:::zh
**中文:** **层内滑动量化。** 为进一步利用我们所发现的实证性质，我们提出另一个互补组件，称为层内滑动量化。该组件被施加到基础组件的当前滑动窗口内部。具体而言，它把渐进扩张的滑动设计延伸到层间滑动量化的每个窗口：在该窗口中，$s$ 层沿权重与激活维度按比例 $\gamma$ 并行地执行渐进扩张滑动。于是，所有 $s$ 层的联合量化在 $N=1/\gamma$ 个滑动阶段中增量式完成。我们默认设置 $\gamma=0.5,N=2$，如图 2 右半部分所示。在第一阶段，当前层间滑动量化窗口中权重/激活矩阵的前一半被联合量化；在第二阶段，该窗口中全部权重/激活矩阵（含前一半）被联合量化。因此，层内滑动量化在层间滑动量化的当前滑动窗口内部，跨层建立了从局部到全局的参数协同，以降低量化误差。以这种方式耦合两个组件，也带来了 SliderQuant 简洁的公式表述。
:::

<a id="S019"></a>
**Source:** p.6 S019

:::en
**Original:** **Learnable Parameters and Quantizer.** SliderQuant builds a new sequential quantization framework by a flexible schedule of multiple sliding window designs, enabling dense synergies among shallow/intermediate/deep layers and a smart quantization relay across them. Next, we describe how SliderQuant quantizes weights and activations at each layer in a sliding window. It is well known that effectively removing outliers in weights and activations at each layer of a pre-trained LLM is important to reduce the total quantization error. Prior works (Xiao et al., 2023; Shao et al., 2024; Liu et al., 2024b; Ding et al., 2025) popularly use channel scaling (CS) (Meller et al., 2019) and low-rank adaptation (LoRA) (Hu et al., 2022; Dettmers et al., 2023) to handle this issue. Inspired by them, we simply combine CS and LoRA in our method. Let $\mathbf{W}_i \in \mathbb{R}^{n \times m}$ be the weight matrix of the $i^{th}$ layer in a sliding window, and let $\mathbf{X}_i \in \mathbb{R}^{k \times n}$ be its input feature corresponding to a small set of $c$ calibration samples ($c=128$ as default). Then, the quantization process is defined as
:::

:::zh
**中文:** **可学习参数与量化器。** SliderQuant 通过多种滑动窗口设计的灵活调度，构建了一个新的序列量化框架，使浅层/中间层/深层之间形成稠密协同，并在它们之间实现巧妙的量化接力。接下来我们描述 SliderQuant 如何在滑动窗口内对每一层的权重与激活进行量化。众所周知，有效去除预训练 LLM 每一层权重与激活中的离群值，对降低总量化误差十分重要。先前工作（Xiao et al., 2023; Shao et al., 2024; Liu et al., 2024b; Ding et al., 2025）普遍使用通道缩放（CS）（Meller et al., 2019）与低秩适配（LoRA）（Hu et al., 2022; Dettmers et al., 2023）来处理这一问题。受其启发，我们在方法中简单地组合了 CS 与 LoRA。设 $\mathbf{W}_i \in \mathbb{R}^{n \times m}$ 为滑动窗口中第 $i$ 层的权重矩阵，$\mathbf{X}_i \in \mathbb{R}^{k \times n}$ 为其对应于 $c$ 个校准样本的输入特征（默认 $c=128$）。那么量化过程定义为
:::

<a id="E002"></a>
**Source:** p.6 E002 · Eq. (2)

$$
\tilde{\mathbf{X}}_i = \mathbf{X}_i \oslash \alpha_i,\qquad
\tilde{\mathbf{W}}_i = \mathbf{W}_i \odot \alpha_i + \mathbf{A}_i\mathbf{B}_i,\qquad
\mathbf{X}_{i+1} = \text{quantizer}(\tilde{\mathbf{X}}_i)\cdot \text{quantizer}(\tilde{\mathbf{W}}_i)
$$

:::zh
**中文说明：** 三步可一起读：① 输入激活除以通道缩放向量 $\alpha_i$（逐通道缩小），② 权重乘上同一个 $\alpha_i$（逐通道放大）再叠加低秩修正 $\mathbf{A}_i\mathbf{B}_i$，③ 二者分别量化后相乘得到下一层输入。$\alpha_i$ 的作用是把“激活里的离群值”搬到权重上——激活除以 $\alpha$ 后动态范围变窄、更易量化，权重乘以 $\alpha$ 后范围变大，但权重通常比激活更耐受量化；$\mathbf{A}_i\mathbf{B}_i$ 则提供秩 $r=4$ 的加性修正，补偿缩放后仍残留的量化损失。两者在量化结束后都可并入权重，因此推理时没有额外算子。
:::

<a id="S020"></a>
**Source:** p.6 S020

:::en
**Original:** where $\alpha_i \in \mathbb{R}^{n}$ denotes a learnable channel-wise scaling vector to scale $\mathbf{X}_i$ and reversely scale $\mathbf{W}_i$, $\mathbf{A}_i \in \mathbb{R}^{n \times r}$ and $\mathbf{B}_i \in \mathbb{R}^{r \times m}$ (we set $r=4$ in experiments) are two low-rank matrices to get a refined weight matrix $\tilde{\mathbf{W}}_i \in \mathbb{R}^{n \times m}$ for quantization, $\oslash$ and $\odot$ denote element-wise division and multiplication operations, respectively. With the refined weight matrices $\tilde{\mathbf{W}}$ and input $\tilde{\mathbf{X}}$ defined by Eq. 2, the corresponding quantized weight matrices $\hat{\mathbf{W}} = \text{quantizer}(\tilde{\mathbf{W}})$ and input $\hat{\mathbf{X}} = \text{quantizer}(\tilde{\mathbf{X}})$ are obtained to minimize the mean square error defined in Eq. 1. In implementation, we use a uniform quantizer for both weights and activations, for simplicity and fair performance comparisons with existing post-training quantization methods. We put its definition in the Appendix B.1.
:::

:::zh
**中文:** 其中 $\alpha_i \in \mathbb{R}^{n}$ 表示可学习的逐通道缩放向量，用来缩放 $\mathbf{X}_i$ 并反向缩放 $\mathbf{W}_i$；$\mathbf{A}_i \in \mathbb{R}^{n \times r}$ 与 $\mathbf{B}_i \in \mathbb{R}^{r \times m}$（实验中设 $r=4$）是两个低秩矩阵，用于得到供量化的精化权重矩阵 $\tilde{\mathbf{W}}_i \in \mathbb{R}^{n \times m}$；$\oslash$ 与 $\odot$ 分别表示逐元素除法与乘法。由式 2 定义精化后的权重矩阵 $\tilde{\mathbf{W}}$ 与输入 $\tilde{\mathbf{X}}$，即可得到相应的量化权重矩阵 $\hat{\mathbf{W}} = \text{quantizer}(\tilde{\mathbf{W}})$ 与量化输入 $\hat{\mathbf{X}} = \text{quantizer}(\tilde{\mathbf{X}})$，以最小化式 1 所定义的均方误差。在实现中，为简洁并与现有训练后量化方法公平比较，我们对权重与激活都使用均匀量化器。其定义放在附录 B.1。
:::

---

## 4. 实验 / Experiments

<a id="S021"></a>
**Source:** p.6 S021

:::en
**Original:** **Models.** We first select widely used Llama (Touvron et al., 2023a), Llama2 (Touvron et al., 2023b), Llama3 (Dubey et al., 2024) and Qwen2.5 (Yang et al., 2024) families for experiments. To further explore the potential of SliderQuant, we evaluate it on more advanced LLMs, including a Mixture of Experts (MoE) model Qwen3-30B-A3B (Yang et al., 2025) and the recently popular DeepSeek-R1 (Guo et al., 2025) distilled models with chain-of-thought reasoning abilities.
:::

:::zh
**中文:** **模型。** 我们首先选取广泛使用的 Llama（Touvron et al., 2023a）、Llama2（Touvron et al., 2023b）、Llama3（Dubey et al., 2024）与 Qwen2.5（Yang et al., 2024）系列进行实验。为进一步探索 SliderQuant 的潜力，我们在更先进的 LLM 上评估它，包括混合专家（MoE）模型 Qwen3-30B-A3B（Yang et al., 2025），以及近期流行的、具备思维链推理能力的 DeepSeek-R1（Guo et al., 2025）蒸馏模型。
:::

<a id="S022"></a>
**Source:** p.6 S022

:::en
**Original:** **Evaluations.** For most models (including the Llama series, Qwen2.5, and Qwen3), following the commonly adopted settings in post-training quantization research, we evaluate their two fundamental capabilities: basic language generation and commonsense reasoning. For language generation, we report perplexity on WikiText2 (Merity et al., 2017) and C4 (Raffel et al., 2020). For commonsense reasoning, we report zero-shot accuracy on 6 widely adopted benchmarks: PIQA (Bisk et al., 2020), ARC (Clark et al., 2018), HellaSwag (HS) (Zellers et al., 2019), Winogrande (WG) (Sakaguchi et al., 2021), BoolQ (Clark et al., 2019), and MMLU (Hendrycks et al., 2020). For the DeepSeek-R1 distilled models with chain-of-thought reasoning abilities, we evaluate their performance on challenging mathematical reasoning (MATH-500 (Hendrycks et al., 2021), AIME-2024 (Jia, 2024), and GSM8K (Cobbe et al., 2021)) and code generation benchmarks (HumanEval+ and MBPP+ (Liu et al., 2023)). To ensure fair and reproducible results, we adopt standard evaluations: LM Evaluation Harness (Gao et al., 2024) for commonsense reasoning, OpenCompass (Contributors, 2023) for mathematical reasoning, and EvalPlus (Liu et al., 2023) for code generation.
:::

:::zh
**中文:** **评测。** 对大多数模型（包括 Llama 系列、Qwen2.5 与 Qwen3），遵循训练后量化研究中常用的设定，我们评测其两项基本能力：基础语言生成与常识推理。语言生成方面，我们报告在 WikiText2（Merity et al., 2017）与 C4（Raffel et al., 2020）上的困惑度。常识推理方面，我们报告在 6 个广泛采用的基准上的零样本准确率：PIQA（Bisk et al., 2020）、ARC（Clark et al., 2018）、HellaSwag（HS）（Zellers et al., 2019）、Winogrande（WG）（Sakaguchi et al., 2021）、BoolQ（Clark et al., 2019）与 MMLU（Hendrycks et al., 2020）。对具备思维链推理能力的 DeepSeek-R1 蒸馏模型，我们评测其在具有挑战性的数学推理（MATH-500（Hendrycks et al., 2021）、AIME-2024（Jia, 2024）与 GSM8K（Cobbe et al., 2021））与代码生成基准（HumanEval+ 与 MBPP+（Liu et al., 2023））上的表现。为保证结果公平且可复现，我们采用标准评测工具：常识推理由 LM Evaluation Harness（Gao et al., 2024）完成，数学推理由 OpenCompass（Contributors, 2023）完成，代码生成由 EvalPlus（Liu et al., 2023）完成。
:::

<a id="S023"></a>
**Source:** p.6 S023

:::en
**Original:** **Counterpart Methods.** Recent top-performing PTQ methods usually introduce architectural modifications, some of which are not absorable at inference. For a fair comparison, we categorize existing PTQ methods into two distinct groups based on whether they introduce extra inference-time costs. Notably, as a flexible quantization framework, SliderQuant maintains compatibility with both paradigms. For quantization without extra inference-time costs, our experiments cover both weight-only and weight-activation quantization: for weight-only quantization, we compare SliderQuant with round-to-nearest quantization (RTN), GPTQ (Frantar et al., 2023), AWQ (Lin et al., 2024b), and QuIP (Chee et al., 2024); for weight-activation quantization, we compare SliderQuant with RTN, SmoothQuant (Xiao et al., 2023), OmniQuant (Shao et al., 2024), AffineQuant (Ma et al., 2024b), and CBQ (Ding et al., 2025). For quantization with extra inference-time costs, we implement a variant of SliderQuant named SliderQuant+ for a fair comparison, which additionally incorporates rotation transformations. We compare it with QLLM (Liu et al., 2024b), Atom (Zhao et al., 2024), DuQuant (Lin et al., 2024a), QuaRot (Ashkboos et al., 2024b), SpinQuant (Liu et al., 2025), and FlatQuant (Sun et al., 2025). Implementation details are provided in the Appendix B.
:::

:::zh
**中文:** **对比方法。** 近期表现最好的 PTQ 方法通常会引入架构改动，其中一些在推理时无法被吸收。为公平比较，我们根据是否引入额外推理时开销，把现有 PTQ 方法分为两类。值得注意的是，作为一个灵活的量化框架，SliderQuant 与两种范式都兼容。对于**无额外推理时开销**的量化，我们的实验同时覆盖仅权重量化与权重-激活量化：仅权重量化上，把 SliderQuant 与最近舍入量化（RTN）、GPTQ（Frantar et al., 2023）、AWQ（Lin et al., 2024b）与 QuIP（Chee et al., 2024）比较；权重-激活量化上，与 RTN、SmoothQuant（Xiao et al., 2023）、OmniQuant（Shao et al., 2024）、AffineQuant（Ma et al., 2024b）与 CBQ（Ding et al., 2025）比较。对于**有额外推理时开销**的量化，为实现公平比较，我们实现了 SliderQuant 的一个变体 SliderQuant+，它额外引入了旋转变换；我们将其与 QLLM（Liu et al., 2024b）、Atom（Zhao et al., 2024）、DuQuant（Lin et al., 2024a）、QuaRot（Ashkboos et al., 2024b）、SpinQuant（Liu et al., 2025）与 FlatQuant（Sun et al., 2025）比较。实现细节见附录 B。
:::

### 4.1 主要结果 / Main Results

<a id="S024"></a>
**Source:** p.7 S024

:::en
**Original:** **Quantization without Extra Inference-Time Costs.** We first report the results for quantization methods without extra inference-time costs. As shown in Table 1, SliderQuant consistently achieves lower perplexity on WikiText2 and C4 than counterpart PTQ methods across a broad range of quantization settings, model families and model sizes. Under the extremely low-bit configuration of W4A4, SliderQuant gets more prominent performance. These results demonstrate the robustness and effectiveness of SliderQuant in preserving generation quality even under aggressive quantization. In Table 2, we provide the results comparison on 6 commonsense QA benchmarks. We can observe that SliderQuant consistently surpasses counterpart PTQ methods. These results validate the effectiveness of SliderQuant in preserving various capabilities of LLMs under low-bit quantization.
:::

:::zh
**中文:** **无额外推理时开销的量化。** 我们首先报告无额外推理时开销的量化方法的结果。如表 1 所示，在广泛的量化设定、模型家族与模型规模上，SliderQuant 在 WikiText2 与 C4 上始终取得比对比 PTQ 方法更低的困惑度。在极低比特的 W4A4 配置下，SliderQuant 的表现更为突出。这些结果证明了 SliderQuant 在激进量化下仍能保持生成质量的鲁棒性与有效性。表 2 给出了在 6 个常识问答基准上的结果对比，可以观察到 SliderQuant 始终超过对比 PTQ 方法。这些结果验证了 SliderQuant 在低位宽量化下保持 LLM 多种能力的有效性。
:::

<a id="T001"></a>
#### Table 1. <span class="lang-zh">语言生成任务：无额外推理时开销的方法对比（困惑度）</span>

**Placed near:** p.7 S024<br>
**Source:** p.7 C003

<img src="/images/quantization/sliderquant/tables/tab01.png" alt="Table 1" width="1660" height="830" loading="lazy" decoding="async">

:::en
**Original caption:** Results comparison of different quantization methods without extra inference-time costs on the language generation tasks. The metric is perplexity. Best results are shown in bold.
:::

:::zh
**中文表注:** 无额外推理时开销的不同量化方法在语言生成任务上的结果对比。指标为困惑度，最优结果加粗。
:::

:::zh
**Reading note:** 三块分别是 W4A16、W2A16、W4A4。看两处：① **W4A4 的 Llama2-7B**，SliderQuant 8.34/11.10，CBQ 12.73/14.45、OmniQuant 14.26/18.02，而 FP16 是 5.47/6.97——相对次优方法困惑度降了约 35%；② **W2A16** 上多数方法已经崩坏（RTN 达 $3.8\times10^4$），SliderQuant 仍保持 9.59/13.83。注意 W4A16 的 SliderQuant（5.61/7.19）已经非常接近 FP16（5.47/6.97），说明该框架在温和设定下也没有副作用。
:::

<a id="T002"></a>
#### Table 2. <span class="lang-zh">零样本常识推理：无额外推理时开销的方法对比（准确率）</span>

**Placed near:** p.7 S024<br>
**Source:** p.7 C004

<img src="/images/quantization/sliderquant/tables/tab02.png" alt="Table 2" width="1660" height="400" loading="lazy" decoding="async">

:::en
**Original caption:** Results comparison of different quantization methods without extra inference-time costs on the zero-shot commonsense reasoning tasks. The metric is accuracy (%).
:::

:::zh
**中文表注:** 无额外推理时开销的不同量化方法在零样本常识推理任务上的结果对比。指标为准确率（%）。
:::

:::zh
**Reading note:** W4A4 下差距最直观的是 MMLU：Llama2-13B 上 SliderQuant 27.04 对 CBQ 24.78、OmniQuant 23.56；Qwen2.5-14B 上 SliderQuant 43.50 对 CBQ 31.61——MMLU 是这批基准里最“脆”的一个。平均分：Llama2-13B 56.77 vs CBQ 54.67（FP16 70.26）；Qwen2.5-14B 58.96 vs CBQ 53.50（FP16 77.42）。
:::

<a id="S025"></a>
**Source:** p.7 S025

:::en
**Original:** **Quantization with Extra Inference-Time Costs.** The comparative results of quantization methods with extra inference-time costs are shown in Table 3. We can see that SliderQuant+ achieves the best results on average across different models and benchmarks. While SliderQuant demonstrates strong quantization capabilities, incorporating rotation transformations further enhances its effectiveness, enabling it to handle more precision-sensitive scenarios. Combining the results in Table 1, Table 2 and Table 3, we demonstrate the generalizability of our sliding-layer quantization framework, which consistently gets superior performance compared to the state-of-the-art PTQ methods with or without extra inference-time costs. Experimental details are provided in the Appendix C.
:::

:::zh
**中文:** **有额外推理时开销的量化。** 有额外推理时开销的量化方法对比结果见表 3。可以看到，SliderQuant+ 在不同模型与基准上的平均结果最好。SliderQuant 本身已展现出很强的量化能力，而引入旋转变换进一步增强了其效力，使其能够应对对精度更敏感的场景。结合表 1、表 2 与表 3 的结果，我们展示了滑动层量化框架的泛化性：无论有无额外推理时开销，它都始终优于最先进的 PTQ 方法。实验细节见附录 C。
:::

<a id="T003"></a>
#### Table 3. <span class="lang-zh">有额外推理时开销的方法对比（W4A4，含旋转的 SliderQuant+）</span>

**Placed near:** p.7 S025<br>
**Source:** p.8 C005

<img src="/images/quantization/sliderquant/tables/tab03.png" alt="Table 3" width="1660" height="1122" loading="lazy" decoding="async">

:::en
**Original caption:** Results comparison of different quantization methods with extra inference-time costs. SliderQuant+ denotes SliderQuant using rotation transformations.
:::

:::zh
**中文表注:** 有额外推理时开销的不同量化方法结果对比。SliderQuant+ 表示使用旋转变换的 SliderQuant。
:::

:::zh
**Reading note:** 这一栏的对手都是旋转类方法（QuaRot、SpinQuant、FlatQuant、DuQuant）与 QLLM/Atom。Llama2-7B 上 SliderQuant+ 平均 67.58，略胜 FlatQuant 67.15、SpinQuant 65.23（FP16 69.05）；Llama2-13B 71.22 vs FlatQuant 70.38；Llama-2-70B 76.32 vs 76.02；Llama3-8B 71.50 vs 70.89。增益幅度不大但一致——说明旋转带来的收益与滑动窗口带来的收益**可以叠加**，两者不是替代关系。
:::

<a id="S026"></a>
**Source:** p.7 S026

:::en
**Original:** **Extension to Mixture of Experts Architectures.** As shown in Table 4, when applying SliderQuant to Qwen3-30B-A3B, an advanced MoE architecture, we can observe a similar performance improvement trend as on dense LLMs. The results further validate the generalizability of SliderQuant.
:::

:::zh
**中文:** **扩展到混合专家架构。** 如表 4 所示，当把 SliderQuant 应用于先进 MoE 架构 Qwen3-30B-A3B 时，可以观察到与稠密 LLM 上相似的性能提升趋势。这些结果进一步验证了 SliderQuant 的泛化性。
:::

<a id="T004"></a>
#### Table 4. <span class="lang-zh">在 MoE 模型 Qwen3-30B-A3B 上的探索</span>

**Placed near:** p.7 S026<br>
**Source:** p.8 C006

<img src="/images/quantization/sliderquant/tables/tab04.png" alt="Table 4" width="1660" height="438" loading="lazy" decoding="async">

:::en
**Original caption:** Exploration of applying SliderQuant to the Mixture of Experts (MoE) model Qwen3-30B-A3B. We take OmniQuant as the baseline using its public code for a comparative evaluation.
:::

:::zh
**中文表注:** 将 SliderQuant 应用于混合专家（MoE）模型 Qwen3-30B-A3B 的探索。我们使用 OmniQuant 的公开代码作为基线进行对比评估。
:::

:::zh
**Reading note:** MoE 的难点在于专家路由对数值扰动敏感。W4A4 上 OmniQuant 已经崩到 52.43/41.98（平均 50.11），SliderQuant 是 15.49/17.77（平均 66.86），FP16 为 8.71/12.08（75.64）。W4A16/W3A16 上两者差距缩小（75.00 vs 74.08、74.48 vs 73.01），说明滑动窗口主要在**极端位宽**下起决定作用。
:::

<a id="S027"></a>
**Source:** p.7–8 S027

:::en
**Original:** **Challenging Math and Code Tasks with Reasoning Language Models.** Most existing PTQ works are limited to evaluating LLMs on basic language generation and commonsense reasoning tasks. However, for real-world applications, it is also crucial to assess their complex reasoning abilities on more challenging tasks. In the experiments, we apply SliderQuant to the state-of-the-art DeepSeek-R1 distilled models, exploring its performance on challenging mathematical reasoning and code generation benchmarks. The results are shown in Table 5. Under the W4A16 setting, SliderQuant remains near-lossless relative to FP16 on both 14B and 32B models, while consistently outperforming OmniQuant. Under the more aggressive W2A16 setting, SliderQuant achieves substantially higher accuracy than OmniQuant across all benchmarks and model scales. These results indicate that SliderQuant preserves reasoning fidelity at 4-bit setting and remains robust even at 2-bit setting.
:::

:::zh
**中文:** **用推理语言模型挑战数学与代码任务。** 现有 PTQ 工作大多只在基础语言生成与常识推理任务上评测 LLM。然而对真实应用而言，在更困难的任务上评估其复杂推理能力同样至关重要。实验中，我们把 SliderQuant 应用于最先进的 DeepSeek-R1 蒸馏模型，探索其在具有挑战性的数学推理与代码生成基准上的表现。结果见表 5。在 W4A16 设定下，SliderQuant 在 14B 与 32B 模型上相对 FP16 都接近无损，同时始终优于 OmniQuant。在更激进的 W2A16 设定下，SliderQuant 在所有基准与模型规模上都取得了远高于 OmniQuant 的准确率。这些结果表明，SliderQuant 在 4 比特设定下保住了推理保真度，即便在 2 比特设定下也依然稳健。
:::

<a id="T005"></a>
#### Table 5. <span class="lang-zh">DeepSeek-R1 蒸馏模型上的数学与代码任务</span>

**Placed near:** p.8 S027<br>
**Source:** p.8 C007

<img src="/images/quantization/sliderquant/tables/tab05.png" alt="Table 5" width="1660" height="471" loading="lazy" decoding="async">

:::en
**Original caption:** Exploration of applying SliderQuant to the DeepSeek-R1 distilled models with chain-of-thought reasoning abilities on the challenging mathematical reasoning and code generation tasks.
:::

:::zh
**中文表注:** 将 SliderQuant 应用于具备思维链推理能力的 DeepSeek-R1 蒸馏模型，在具有挑战性的数学推理与代码生成任务上的探索。
:::

:::zh
**Reading note:** 这张表最有说服力的是 **W4A16 的 32B**：SliderQuant 平均 82.71，FP16 是 83.17——只差 0.46 分，其中 AIME-2024 甚至打平（76.67）且优于 OmniQuant 的 56.66。W2A16 上所有方法都明显退化（SliderQuant 36.59 vs OmniQuant 8.05），但相对优势被拉大：这说明“推理能力”对量化误差的容忍度极低，也解释了为什么作者要专门在推理模型上验证。
:::

<a id="S028"></a>
**Source:** p.9 S028

:::en
**Original:** **Comparison with Mixed-Precision Methods.** In Table 6 and Table 7, we compare SliderQuant with the state-of-the-art mixed-precision quantization methods including LLM-MQ (Li et al., 2024) and SpQR (Dettmers et al., 2024) for weight-only quantization, and QUIK for weight-activation quantization. Across all settings, SliderQuant consistently achieves higher accuracy, even with lower bit widths. Notably, SliderQuant surpasses LLM-MQ and QUIK with higher bit widths preserving a fixed portion of FP16 outliers, and also yields clear gains over SpQR. These results show that our sliding-layer quantization mechanism is superior to existing mixed-precision quantization schemes.
:::

:::zh
**中文:** **与混合精度方法的比较。** 在表 6 与表 7 中，我们把 SliderQuant 与最先进的混合精度量化方法比较，包括用于仅权重量化的 LLM-MQ（Li et al., 2024）与 SpQR（Dettmers et al., 2024），以及用于权重-激活量化的 QUIK。在所有设定下，即便位宽更低，SliderQuant 也始终取得更高的准确率。值得注意的是，SliderQuant 超过了保留固定比例 FP16 离群值、位宽更高的 LLM-MQ 与 QUIK，相对 SpQR 也有明显增益。这些结果表明，我们的滑动层量化机制优于现有的混合精度量化方案。
:::

<a id="T006"></a>
#### Table 6. <span class="lang-zh">与混合精度方法（LLM-MQ、QUIK）的对比（Llama2-13B）</span>

**Placed near:** p.9 S028<br>
**Source:** p.9 C008

<img src="/images/quantization/sliderquant/tables/tab06.png" alt="Table 6" width="1660" height="625" loading="lazy" decoding="async">

:::en
**Original caption:** Results comparison of SliderQuant with mixed-precision quantization methods on Llama2-13B. In the table, (1) for LLM-MQ/QUIK, 0.5% in FP16 or 5% in FP16 indicates the ratio of weight or weight-activation outliers at each layer stored in FP16; (2) for 2.1-bit weight quantization by SliderQuant, we simply use 4-bit quantization to the first and last layers while quantizing all other layers to 2-bit, and SliderQuant+ denotes our best version of SliderQuant using rotation transformations.
:::

:::zh
**中文表注:** SliderQuant 与混合精度量化方法在 Llama2-13B 上的结果对比。表中：(1) 对 LLM-MQ/QUIK，“0.5% in FP16”或“5% in FP16”表示每层以 FP16 存储的权重或权重-激活离群值比例；(2) SliderQuant 的 2.1 比特权重量化，是对第一层与最后一层直接用 4 比特、其余层全部用 2 比特；SliderQuant+ 表示使用旋转变换的最佳版本。
:::

:::zh
**Reading note:** 看点是**同等或更低比特下取胜**：LLM-MQ/QUIK 需要保留 0.5%–5% 的 FP16 离群值（等效位宽反而更高），SliderQuant 用纯低位宽（2.1 bit 的极端设定只给首末层 4 bit）就超过了它们。这张表回应的是“混合精度是不是更划算”这一常见质疑。
:::

<a id="T007"></a>
#### Table 7. <span class="lang-zh">与 SpQR 的对比（Llama-7B / Llama-13B）</span>

**Placed near:** p.9 S028<br>
**Source:** p.9 C009

<img src="/images/quantization/sliderquant/tables/tab07.png" alt="Table 7" width="1660" height="272" loading="lazy" decoding="async">

:::en
**Original caption:** Results comparison of SliderQuant with the mixed-precision quantization method SpQR on Llama-7B and Llama-13B.
:::

:::zh
**中文表注:** SliderQuant 与混合精度量化方法 SpQR 在 Llama-7B 与 Llama-13B 上的结果对比。
:::

:::zh
**Reading note:** SpQR 是仅权重量化的强基线（W3.45A16）。这里 SliderQuant 在 W4A16/W3A16 下与之对比，落差主要体现在 C4 与常识推理子项上；结合正文“yields clear gains over SpQR”，可把它读作“不需要稀疏离群值通路也能达到同级或更好”。
:::

### 4.2 消融研究 / Ablation Studies

<a id="S029"></a>
**Source:** p.9 S029

:::en
**Original:** To have a better understanding of our proposed SliderQuant, we further conduct a lot of ablative experiments under both weight-only quantization and weight-activation quantization with Llama2-7B. In the ablations, we use the fixed-size sliding quantization $\{s=2,i=1\}$ as the baseline.
:::

:::zh
**中文:** 为更好地理解我们提出的 SliderQuant，我们进一步在 Llama2-7B 上、于仅权重量化与权重-激活量化两种情形下开展了大量消融实验。消融中，我们以固定尺寸滑动量化 $\{s=2,i=1\}$ 作为基线。
:::

<a id="S030"></a>
**Source:** p.9 S030

:::en
**Original:** **Overall Design.** We first conduct experiments to study the two core components of SliderQuant, inter-layer sliding quantization **(Inter-S)** and intra-layer sliding quantization **(Intra-S)**. The results are shown in Table 8. Recall that compared to the baseline fixed-size sliding quantization, Inter-S additionally introduces a progressively expanded sliding window **(PESW)** for shallow layers and a progressively contracted sliding window **(PCSW)** for deep layers. We can find that both PESW and PCSW significantly improve the performance, demonstrating the importance to consider the varying layer sensitivity to quantization of any pre-trained LLM. By extending the progressively expanded sliding design within each window, Intra-S further enhances the performance. Coupling Inter-S and Intra-S leads to the best results, confirming the effectiveness of SliderQuant's multi-level design.
:::

:::zh
**中文:** **整体设计。** 我们首先通过实验研究 SliderQuant 的两个核心组件——层间滑动量化**（Inter-S）**与层内滑动量化**（Intra-S）**，结果见表 8。回顾一下：与基线固定尺寸滑动量化相比，Inter-S 额外为浅层引入了渐进扩张滑动窗口**（PESW）**、为深层引入了渐进收缩滑动窗口**（PCSW）**。可以看到 PESW 与 PCSW 都显著提升了性能，说明考虑任意预训练 LLM 各层量化敏感度差异的重要性。通过把渐进扩张的滑动设计延伸到每个窗口内部，Intra-S 进一步提升了性能。将 Inter-S 与 Intra-S 耦合得到最佳结果，证实了 SliderQuant 多层次设计的有效性。
:::

<a id="T008"></a>
#### Table 8. <span class="lang-zh">整体设计消融：Inter-S 与 Intra-S</span>

**Placed near:** p.9 S030<br>
**Source:** p.10 C010

<img src="/images/quantization/sliderquant/tables/tab08.png" alt="Table 8" width="914" height="313" loading="lazy" decoding="async">

:::en
**Original caption:** Ablation of inter-layer sliding quantization (Inter-S) and intra-layer sliding quantization (Intra-S).
:::

:::zh
**中文表注:** 层间滑动量化（Inter-S）与层内滑动量化（Intra-S）的消融。
:::

:::zh
**Reading note:** 逐行读这张表就是方法的故事线：基线 12.73/14.45 → 只加 Intra-S 10.34/13.46 → 只加 PESW 10.30/13.78 → 只加 PCSW 9.84/13.31 → PESW+PCSW 9.13/11.78 → 三者齐全 **8.34/11.10**。注意两个单独方向的收益接近（都降约 2.4），合起来又进一步降 1.5——说明浅层与深层的协同是**互补**的，不是重复计功。
:::

<a id="S031"></a>
**Source:** p.9 S031

:::en
**Original:** **Effect of $L_{s}$ and $L_{d}$ in Inter-Layer Sliding Quantization.** After validating the effectiveness of Inter-S and Intra-S, we next study the effect of different settings for them independently. In Table 9, we provide results of Inter-S with different $L_{s}$ and $L_{d}$, namely the number of shallow layers and the number of deep layers in our design. For simplicity, we always keep $L_{s}$ and $L_{d}$ the same in the experiments. As $L_{s}$ and $L_{d}$ gradually increase from 2 to 6, the model performance steadily improves. While it also leads to larger memory and computational costs during quantization accordingly. Considering the trade-off between quantization performance and efficiency, we set $L_{s}=L_{d}=4$ as default, as further scaling beyond this point tends to bring marginal performance improvements.
:::

:::zh
**中文:** **层间滑动量化中 $L_{s}$ 与 $L_{d}$ 的影响。** 在验证 Inter-S 与 Intra-S 的有效性之后，我们接下来分别研究它们不同设置的影响。表 9 给出了 Inter-S 在不同 $L_{s}$ 与 $L_{d}$（即设计中浅层数与深层数）下的结果。为简洁起见，实验中我们始终让 $L_{s}$ 与 $L_{d}$ 相等。随着 $L_{s}$ 与 $L_{d}$ 从 2 逐步增至 6，模型性能稳步提升，但量化时的显存与计算开销也相应增大。综合考虑量化性能与效率的折中，我们默认设置 $L_{s}=L_{d}=4$，因为超过该点继续扩大往往只带来边际收益。
:::

<a id="T009"></a>
#### Table 9. <span class="lang-zh">$L_s$ / $L_d$ 的消融</span>

**Placed near:** p.9 S031<br>
**Source:** p.10 C011

<img src="/images/quantization/sliderquant/tables/tab09.png" alt="Table 9" width="735" height="251" loading="lazy" decoding="async">

:::en
**Original caption:** Ablation of inter-layer sliding quantization with different $L_{s}$ and $L_{d}$.
:::

:::zh
**中文表注:** 不同 $L_{s}$ 与 $L_{d}$ 下层间滑动量化的消融。
:::

:::zh
**Reading note:** W4A4 从 2/2 的 10.23 一路降到 6/6 的 8.94，但 4/4→5/5 只降 0.15、5/5→6/6 只降 0.04——典型的饱和曲线。默认 4/4 就是这条曲线的“拐点”，兼顾显存。
:::

<a id="S032"></a>
**Source:** p.10 S032

:::en
**Original:** **Effect of $\gamma$ in Intra-Layer Sliding Quantization.** In this set of ablative experiments, we study the performance of Intra-S with different settings of the sliding ratio $\gamma$. For a given value of $\gamma$, all layers in each window of Inter-S are parallelly quantized along the weight/activation dimension incrementally, and the quantization process is completed in $N=1/\gamma$ stages. As shown in Table 10, Intra-S with $N>1$ always achieves better performance than the baseline ($N=1$). For SliderQuant, we set $\gamma=0.5,N=2$ as default, considering the performance and the simplicity of implementation.
:::

:::zh
**中文:** **层内滑动量化中 $\gamma$ 的影响。** 这组消融实验研究 Intra-S 在不同滑动比例 $\gamma$ 下的表现。给定 $\gamma$ 后，Inter-S 每个窗口中的所有层沿权重/激活维度并行地增量量化，量化过程在 $N=1/\gamma$ 个阶段内完成。如表 10 所示，$N>1$ 的 Intra-S 始终优于基线（$N=1$）。兼顾性能与实现简洁性，我们为 SliderQuant 默认设置 $\gamma=0.5,N=2$。
:::

<a id="T010"></a>
#### Table 10. <span class="lang-zh">层内滑动比例 $\gamma$ 的消融</span>

**Placed near:** p.10 S032<br>
**Source:** p.10 C012

<img src="/images/quantization/sliderquant/tables/tab10.png" alt="Table 10" width="772" height="238" loading="lazy" decoding="async">

:::en
**Original caption:** Ablation of intra-layer sliding quantization with different $\gamma$. Note $N=1/\gamma$.
:::

:::zh
**中文表注:** 不同 $\gamma$ 下层内滑动量化的消融。注意 $N=1/\gamma$。
:::

:::zh
**Reading note:** $\gamma=1.0$（即不分阶段）12.73 → $\gamma=0.5$ 10.34 → 0.33 时 C4 更低但 WikiText2 回升（10.56）→ 0.25 全面变差（11.32）。收益在 2 阶段就基本吃完，更多阶段反而因为“每阶段可见的层太少”而损失全局协同——所以默认 $N=2$ 不是随便选的。
:::

<a id="S033"></a>
**Source:** p.10 S033

:::en
**Original:** **Effect of $s$ in Fixed-Size Sliding Quantization.** To mitigate the performance differences potentially caused by varying window sizes, we increase the fixed window size $s$ for the baseline. As shown in Table 11, the performance of the baseline improves as $s$ increases from 2 to 4. Nevertheless, a substantial performance gap still remains compared to Inter-S which uses the same window size of 4 only in shallow and deep layers. This highlights the limitation of fixed-size sliding quantization, where a uniform strategy overlooks the varying quantization sensitivity across layers. These results further validate the necessity of our adaptive sliding window designs.
:::

:::zh
**中文:** **固定尺寸滑动量化中 $s$ 的影响。** 为消除窗口大小不同可能带来的性能差异，我们增大基线的固定窗口尺寸 $s$。如表 11 所示，随着 $s$ 从 2 增至 4，基线性能有所提升。然而，与“仅在浅层和深层使用相同窗口大小 4”的 Inter-S 相比，仍存在明显差距。这凸显了固定尺寸滑动量化的局限：统一的策略忽视了各层量化敏感度的差异。这些结果进一步验证了我们的自适应滑动窗口设计的必要性。
:::

<a id="T011"></a>
#### Table 11. <span class="lang-zh">增大固定窗口尺寸 $s$ 的对比</span>

**Placed near:** p.10 S033<br>
**Source:** p.10 C013

<img src="/images/quantization/sliderquant/tables/tab11.png" alt="Table 11" width="839" height="259" loading="lazy" decoding="async">

:::en
**Original caption:** Ablation of fixed-size sliding quantization with larger $s$.
:::

:::zh
**中文表注:** 固定尺寸滑动量化在更大 $s$ 下的消融。
:::

:::zh
**Reading note:** 这是回应审稿人“是不是只要把窗口调大就行”的关键一张：基线把 $s$ 从 2 加到 4，W4A4 仅从 12.73 降到 11.13；而 Inter-S 在同样的 $s=4$ 下（仅用于浅/深层）是 9.13。**同样的窗口预算，用在不同层上效果差一倍以上**——这直接支撑了“自适应”而非“统一加大”的论点。
:::

<a id="S034"></a>
**Source:** p.10 S034

:::en
**Original:** **Channel Scaling and LoRA.** Table 12 shows the effect of Channel Scaling (CS) and LoRA under W4A4 quantization. We observe that neither component alone achieves the best performance. Regardless of whether applied to the baseline or our SliderQuant, using both CS and LoRA together consistently yields better results than using either one individually. This demonstrates the complementary nature of the two techniques under our proposed sliding quantzation framework, well suppressing outliers in weights and activations at different layers.
:::

:::zh
**中文:** **通道缩放与 LoRA。** 表 12 展示了 W4A4 量化下通道缩放（CS）与 LoRA 的作用。我们观察到，单独使用任一组件都无法取得最佳性能。无论应用于基线还是我们的 SliderQuant，同时使用 CS 与 LoRA 都始终优于单独使用其中之一。这说明在我们提出的滑动量化框架下，两种技术具有互补性，能够很好地抑制不同层中权重与激活的离群值。
:::

<a id="T012"></a>
#### Table 12. <span class="lang-zh">通道缩放（CS）与 LoRA 的作用</span>

**Placed near:** p.10 S034<br>
**Source:** p.10 C014

<img src="/images/quantization/sliderquant/tables/tab12.png" alt="Table 12" width="676" height="305" loading="lazy" decoding="async">

:::en
**Original caption:** Effect of channel scaling (CS) and LoRA.
:::

:::zh
**中文表注:** 通道缩放（CS）与 LoRA 的作用。
:::

:::zh
**Reading note:** 两个值得注意的点：① 基线只加 CS 时是 20.41/29.67（很差），说明**没有 LoRA 的滑动窗口很难训**；② 基线 +LoRA 就是 12.73/14.45，恰好等于全文的基线数字——可见论文的“baseline”本来就含 LoRA。SliderQuant 上 CS+LoRA 达 8.34/11.10，比只加 CS 的 9.18/12.21 更好，说明两者在自适应窗口下才真正互补。
:::

<a id="S035"></a>
**Source:** p.10 S035

:::en
**Original:** **Additional Experiments and Analyses.** We include further implementation details and more experimental results in the Appendix, covering the following aspects: (1) the analysis of SliderQuant's quantization efficiency; (2) the ablation studies on the number of calibration samples and the group-wise quantization; (3) extended evaluation across diverse model families and quantization settings; (4) design details of SliderQuant with rotation transformations; (5) other results and visualizations; (6) other experiments and discussions conducted for the rebuttal.
:::

:::zh
**中文:** **补充实验与分析。** 我们在附录中给出更多实现细节与实验结果，涵盖以下方面：(1) SliderQuant 量化效率的分析；(2) 关于校准样本数量与分组量化的消融；(3) 在多样模型家族与量化设定上的扩展评测；(4) 带旋转变换的 SliderQuant 的设计细节；(5) 其他结果与可视化；(6) 为回复审稿意见而开展的其他实验与讨论。
:::

---

## 5. 结论 / Conclusion

<a id="S036"></a>
**Source:** p.10 S036

:::en
**Original:** In this paper, we propose SliderQuant, a new post-training quantization framework that explicitly accounts for the varying layer sensitivity to quantization of large language models. By coupling inter-layer and intra-layer sliding quantization components, SliderQuant reduces quantization errors across layers. Extensive experiments across various model families, quantization settings (e.g., W2A16, W4A4), and benchmarks demonstrate the effectiveness and generalizability of our method.
:::

:::zh
**中文:** 本文提出 SliderQuant，一个新的训练后量化框架，它显式地考虑了大语言模型各层对量化敏感度的差异。通过耦合层间与层内滑动量化组件，SliderQuant 降低了跨层量化误差。跨多种模型家族、量化设定（如 W2A16、W4A4）与基准的大量实验，证明了我们方法的有效性与泛化性。
:::

---

## 参考文献 / References

:::zh
> 原文共 63 条参考文献（p.11–14）。参考文献条目不做翻译，以下按原文（作者. *标题*. 出处）照录，供检索核对。
:::

<details>
<summary>展开：完整参考文献列表（63 条） / Full reference list (63)</summary>

1. Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. *Gpt-4 technical report*. *arXiv preprint arXiv:2303.08774*, 2023.
2. Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. *Gemini: A family of highly capable multimodal models*. *arXiv preprint arXiv.2312.11805*, 2023.
3. Saleh Ashkboos, Ilia Markov, Elias Frantar, Tingxuan Zhong, Xincheng Wang, Jie Ren, Torsten Hoefler, and Dan Alistarh. *Towards end-to-end 4-bit inference on generative large language models*. In *EMNLP*, 2024a.
4. Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L Croci, Bo Li, Martin Jaggi, Dan Alistarh, Torsten Hoefler, and James Hensman. *Quarot: Outlier-free 4-bit inference in rotated llms*. In *NeurIPS*, 2024b.
5. Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. *Piqa: Reasoning about physical commonsense in natural language*. In *AAAI*, 2020.
6. Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. *Language models are few-shot learners*. In *NeurIPS*, 2020.
7. Yaohui Cai, Zhewei Yao, Zhen Dong, Amir Gholami, Michael W Mahoney, and Kurt Keutzer. *Zeroq: A novel zero shot quantization framework*. In *CVPR*, 2020.
8. Jerry Chee, Yaohui Cai, Volodymyr Kuleshov, and Christopher M De Sa. *Quip: 2-bit quantization of large language models with guarantees*. In *NeurIPS*, 2024.
9. Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. *Boolq: Exploring the surprising difficulty of natural yes/no questions*. In *NAACL*, 2019.
10. Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. *Think you have solved question answering? try arc, the ai2 reasoning challenge*. *arXiv preprint arXiv:1803.05457*, 2018.
11. Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. *Training verifiers to solve math word problems*. *arXiv preprint arXiv:2110.14168*, 2021.
12. OpenCompass Contributors. *Opencompass: A universal evaluation platform for foundation models*. https://github.com/open-compass/opencompass, 2023.
13. Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. *Llm.int8(): 8-bit matrix multiplication for transformers at scale*. In *NeurIPS*, 2022.
14. Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. *Qlora: Efficient finetuning of quantized llms*. In *NeurIPS*, 2023.
15. Tim Dettmers, Ruslan Svirschevski, Vage Egiazarian, Denis Kuznedelev, Elias Frantar, Saleh Ashkboos, Alexander Borzunov, Torsten Hoefler, and Dan Alistarh. *Spqr: A sparse-quantized representation for near-lossless llm weight compression*. In *ICLR*, 2024.
16. Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. *Bert: Pre-training of deep bidirectional transformers for language understanding*. In *NAACL-HLT*, 2019.
17. Xin Ding, Xiaoyu Liu, Zhijun Tu, Yun Zhang, Wei Li, Jie Hu, Hanting Chen, Yehui Tang, Zhiwei Xiong, Baoqun Yin, et al. *Cbq: Cross-block quantization for large language models*. In *ICLR*, 2025.
18. Haojie Duanmu, Zhihang Yuan, Xiuhong Li, Jiangfei Duan, Xingcheng Zhang, and Dahua Lin. *Skvq: Sliding-window key and value cache quantization for large language models*. In *COLM*, 2024.
19. Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. *The llama 3 herd of models*. *arXiv preprint arXiv:2407.21783*, 2024.
20. Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. *Gptq: Accurate post-training quantization for generative pre-trained transformers*. In *ICLR*, 2023.
21. Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac'h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. *A framework for few-shot language model evaluation, 2024*. .
22. Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. *Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning*. *arXiv preprint arXiv:2501.12948*, 2025.
23. Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. *Measuring massive multitask language understanding*. *arXiv preprint arXiv:2009.03300*, 2020.
24. Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. *Measuring mathematical problem solving with the math dataset*. *arXiv preprint arXiv:2103.03874*, 2021.
25. Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. *Lora: Low-rank adaptation of large language models*. In *ICLR*, 2022.
26. Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. *Openai o1 system card*. *arXiv preprint arXiv:2412.16720*, 2024.
27. Maxwell Jia. *Aime 2024 dataset*. https://huggingface.co/datasets/Maxwell-Jia/AIME_2024, 2024.
28. Sehoon Kim, Coleman Hooper, Amir Gholami, Zhen Dong, Xiuyu Li, Sheng Shen, Michael W Mahoney, and Kurt Keutzer. *Squeezellm: Dense-and-sparse quantization*. In *ICML*, 2024.
29. Shiyao Li, Xuefei Ning, Ke Hong, Tengxuan Liu, Luning Wang, Xiuhong Li, Kai Zhong, Guohao Dai, Huazhong Yang, and Yu Wang. *Llm-mq: Mixed-precision quantization for efficient llm deployment*. In *NeurIPS Workshop*, 2023.
30. Yuhang Li, Ruihao Gong, Xu Tan, Yang Yang, Peng Hu, Qi Zhang, Fengwei Yu, Wei Wang, and Shi Gu. *Brecq: Pushing the limit of post-training quantization by block reconstruction*. In *ICLR*, 2021.
31. Haokun Lin, Haobo Xu, Yichen Wu, Jingzhi Cui, Yingtao Zhang, Linzhan Mou, Linqi Song, Zhenan Sun, and Ying Wei. *Duquant: Distributing outliers via dual transformation makes stronger quantized llms*. In *NeurIPS*, 2024a.
32. Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. *Awq: Activation-aware weight quantization for llm compression and acceleration*. In *MLSys*, 2024b.
33. Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. *Deepseek-v3 technical report*. *arXiv preprint arXiv:2412.19437*, 2024a.
34. Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. *Is your code generated by chatgpt really correct? rigorous evaluation of large language models for code generation*. In *NeurIPS*, 2023.
35. Jing Liu, Ruihao Gong, Xiuying Wei, Zhiwei Dong, Jianfei Cai, and Bohan Zhuang. *Qllm: Accurate and efficient low-bitwidth quantization for large language models*. In *ICLR*, 2024b.
36. Zechun Liu, Changsheng Zhao, Igor Fedorov, Bilge Soran, Dhruv Choudhary, Raghuraman Krishnamoorthi, Vikas Chandra, Yuandong Tian, and Tijmen Blankevoort. *Spinquant: Llm quantization with learned rotations*. In *ICLR*, 2025.
37. Shuming Ma, Hongyu Wang, Lingxiao Ma, Lei Wang, Wenhui Wang, Shaohan Huang, Li Dong, Ruiping Wang, Jilong Xue, and Furu Wei. *The era of 1-bit llms: All large language models are in 1.58 bits*. *arXiv preprint arXiv:2402.17764*, 2024a.
38. Yuexiao Ma, Huixia Li, Xiawu Zheng, Feng Ling, Xuefeng Xiao, Rui Wang, Shilei Wen, Fei Chao, and Rongrong Ji. *Affinequant: Affine transformation quantization for large language models*. In *ICLR*, 2024b.
39. Eldad Meller, Alexander Finkelstein, Uri Almog, and Mark Grobman. *Same, same but different: Recovering neural network quantization error through weight factorization*. In *ICML*, 2019.
40. Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. *Pointer sentinel mixture models*. In *ICLR*, 2017.
41. Markus Nagel, Rana Ali Amjad, Mart Van Baalen, Christos Louizos, and Tijmen Blankevoort. *Up or down? adaptive rounding for post-training quantization*. In *ICML*, 2020.
42. Gunho Park, Baeseong Park, Minsub Kim, Sungjae Lee, Jeonghoon Kim, Beomseok Kwon, Se Jung Kwon, Byeongwook Kim, Youngjoo Lee, and Dongsoo Lee. *Lut-gemm: Quantized matrix multiplication based on luts for efficient inference in large-scale generative language models*. In *ICLR*, 2024.
43. Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. *Exploring the limits of transfer learning with a unified text-to-text transformer*. *JMLR*, 2020.
44. Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. *Winogrande: An adversarial winograd schema challenge at scale*. *Communications of the ACM*, 2021.
45. Wenqi Shao, Mengzhao Chen, Zhaoyang Zhang, Peng Xu, Lirui Zhao, Zhiqian Li, Kaipeng Zhang, Peng Gao, Yu Qiao, and Ping Luo. *Omniquant: Omnidirectionally calibrated quantization for large language models*. In *ICLR*, 2024.
46. Sheng Shen, Zhen Dong, Jiayu Ye, Linjian Ma, Zhewei Yao, Amir Gholami, Michael W Mahoney, and Kurt Keutzer. *Q-bert: Hessian based ultra low precision quantization of bert*. In *AAAI*, 2020.
47. Yuxuan Sun, Ruikang Liu, Haoli Bai, Han Bao, Kang Zhao, Yuening Li, Jiaxin Hu, Xianzhi Yu, Lu Hou, Chun Yuan, et al. *Flatquant: Flatness matters for llm quantization*. In *ICML*, 2025.
48. Hanlin Tang, Yifu Sun, Decheng Wu, Kai Liu, Jianchen Zhu, and Zhanhui Kang. *Easyquant: An efficient data-free quantization algorithm for llms*. In *EMNLP*, 2023.
49. Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth\'ee Lacroix, Baptiste Rozi\`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. *Llama: Open and efficient foundation language models*. *arXiv preprint arXiv:2302.13971*, 2023a.
50. Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. *Llama 2: Open foundation and fine-tuned chat models*. *arXiv preprint arXiv:2307.09288*, 2023b.
51. Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, and Christopher De Sa. *Quip\#: Even better llm quantization with hadamard incoherence and lattice codebooks*. In *ICML*, 2024.
52. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, ukasz Kaiser, and Illia Polosukhin. *Attention is all you need*. In *NIPS*, 2017.
53. Xiuying Wei, Yunchen Zhang, Xiangguo Zhang, Ruihao Gong, Shanghang Zhang, Qi Zhang, Fengwei Yu, and Xianglong Liu. *Outlier suppression: Pushing the limit of low-bit transformer language models*. In *NeurIPS*, 2022.
54. Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. *Smoothquant: Accurate and efficient post-training quantization for large language models*. In *ICML*, 2023.
55. Yuzhuang Xu, Xu Han, Zonghan Yang, Shuo Wang, Qingfu Zhu, Zhiyuan Liu, Weidong Liu, and Wanxiang Che. *Onebit: Towards extremely low-bit large language models*. In *NeurIPS*, 2024.
56. An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. *Qwen2. 5 technical report*. *arXiv preprint arXiv:2412.15115*, 2024.
57. An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. *Qwen3 technical report*. *arXiv preprint arXiv:2505.09388*, 2025.
58. Zhewei Yao, Reza Yazdani Aminabadi, Minjia Zhang, Xiaoxia Wu, Conglong Li, and Yuxiong He. *Zeroquant: Efficient and affordable post-training quantization for large-scale transformers*. In *NeurIPS*, 2022.
59. Ali Hadi Zadeh, Isak Edo, Omar Mohamed Awad, and Andreas Moshovos. *Gobo: Quantizing attention-based nlp models for low latency and energy efficient inference*. In *MICRO*, 2020.
60. Ofir Zafrir, Guy Boudoukh, Peter Izsak, and Moshe Wasserblat. *Q8bert: Quantized 8bit bert*. In *EMC2-NeurIPS*, 2019.
61. Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. *Hellaswag: Can a machine really finish your sentence?*. In *ACL*, 2019.
62. Yilong Zhao, Chien-Yu Lin, Kan Zhu, Zihao Ye, Lequn Chen, Size Zheng, Luis Ceze, Arvind Krishnamurthy, Tianqi Chen, and Baris Kasikci. *Atom: Low-bit quantization for efficient and accurate llm serving*. In *MLSys*, 2024.
63. DanDan Zheng, Yuanliu Liu, Liang Li, et al. *Leveraging inter-layer dependency for post-training quantization*. In *NeurIPS*, 2022.

</details>
---

## 附录 / Appendix

:::zh
> 附录章节号沿用原文（A–I），图表编号沿用原文（Table A–R、Figure A–R）。
:::

## 附录 A：实验所用数据集 / Appendix A: Datasets

<a id="S037"></a>
**Source:** p.15 S037

:::en
**Original:** **WikiText2** (Merity et al., 2017) is a popular language modeling benchmark consisting of over 2 million tokens from verified Wikipedia articles.
:::

:::zh
**中文:** **WikiText2**（Merity et al., 2017）是一个流行的语言建模基准，由来自经审核的维基百科文章的 200 多万个 token 构成。
:::

<a id="S038"></a>
**Source:** p.15 S038

:::en
**Original:** **C4** (Raffel et al., 2020)(Colossal Clean Crawled Corpus) is a large-scale dataset primarily used for language modeling tasks, comprising 156 billion clean tokens. It is sourced from cleaned web pages, originally from Common Crawl.
:::

:::zh
**中文:** **C4**（Raffel et al., 2020）（Colossal Clean Crawled Corpus，巨型干净爬取语料）是一个主要用于语言建模任务的大规模数据集，包含 1560 亿个干净 token。它取自清洗后的网页，最初来源于 Common Crawl。
:::

<a id="S039"></a>
**Source:** p.15 S039

:::en
**Original:** **PIQA** (Bisk et al., 2020) contains 16,000 training and 3,000 validation samples. It focuses on physical reasoning through multiple-choice questions, where models select the most appropriate solution from two options, with exactly one correct answer.
:::

:::zh
**中文:** **PIQA**（Bisk et al., 2020）包含 16,000 个训练样本与 3,000 个验证样本。它通过多项选择题考察物理常识推理：模型从两个选项中选出最合适的解决方案，其中恰有一个正确答案。
:::

<a id="S040"></a>
**Source:** p.15 S040

:::en
**Original:** **ARC** (Clark et al., 2018) is a dataset of 7,787 genuine grade-school level, multiple-choice science questions, assembled to encourage research in advanced question-answering. The dataset is partitioned into a Challenge Set and an Easy Set, where the former contains only questions answered incorrectly by both a retrieval-based algorithm and a word co-occurrence algorithm.
:::

:::zh
**中文:** **ARC**（Clark et al., 2018）是一个包含 7,787 道真实小学水平科学选择题的数据集，旨在推动高级问答研究。该数据集被划分为挑战集（Challenge Set）与简单集（Easy Set），前者只包含那些被基于检索的算法与基于词共现的算法同时答错的题目。
:::

<a id="S041"></a>
**Source:** p.15 S041

:::en
**Original:** **HellaSwag** (Zellers et al., 2019) contains 70,000 training and 10,000 validation samples. It focuses on commonsense reasoning by predicting the most plausible sentence continuation, sourced from crowdsourced captions and activity descriptions.
:::

:::zh
**中文:** **HellaSwag**（Zellers et al., 2019）包含 70,000 个训练样本与 10,000 个验证样本。它通过预测最合理的句子续写来考察常识推理，数据来源于众包图像描述与活动描述。
:::

<a id="S042"></a>
**Source:** p.15 S042

:::en
**Original:** **Winograde** (Sakaguchi et al., 2021) is a collection of 44,000 problems, which is formulated as a fill-in-a-blank task with binary options. The goal is to choose the right option for a given sentence which requires commonsense reasoning.
:::

:::zh
**中文:** **Winograde**（Sakaguchi et al., 2021；原文如此拼写，通常写作 Winogrande）是 44,000 道题目的集合，被构造为带二元选项的填空任务。其目标是为给定句子选出正确的选项，这需要常识推理。
:::

<a id="S043"></a>
**Source:** p.15 S043

:::en
**Original:** **BooIQ** (Clark et al., 2019) is a dataset comprising 15,942 naturally occurring yes/no questions paired with Wikipedia passages. Each example consists of a question, a passage, and a binary answer, aiming to evaluate reading comprehension and entailment-like reasoning.
:::

:::zh
**中文:** **BooIQ**（Clark et al., 2019；原文如此拼写，通常写作 BoolQ）是一个包含 15,942 个自然产生的“是/否”问题、并与维基百科段落配对的数据集。每个样例由一个问题、一个段落与一个二元答案组成，旨在评估阅读理解与类蕴含推理能力。
:::

<a id="S044"></a>
**Source:** p.15 S044

:::en
**Original:** **MMLU** (Hendrycks et al., 2020) is a benchmark designed to evaluate the multitask accuracy of language models across 57 diverse subjects, including elementary mathematics, U.S. history, computer science, law, and more. The dataset consists of multiple-choice questions and is intended to assess models' world knowledge and problem-solving abilities in zero-shot and few-shot settings.
:::

:::zh
**中文:** **MMLU**（Hendrycks et al., 2020）是一个旨在评估语言模型在 57 个不同学科上多任务准确率的基准，涵盖初等数学、美国历史、计算机科学、法律等。该数据集由多项选择题构成，意在评估模型在零样本与小样本设定下的世界知识与问题求解能力。
:::

<a id="S045"></a>
**Source:** p.15 S045

:::en
**Original:** **MATH-500** (Hendrycks et al., 2021) comprises 500 challenging competition-level mathematics problems sampled from the MATH dataset. These problems span various topics such as algebra, geometry, number theory, and probability, and are designed to test a model's ability to perform complex mathematical reasoning and generate step-by-step solutions.
:::

:::zh
**中文:** **MATH-500**（Hendrycks et al., 2021）由从 MATH 数据集中采样得到的 500 道有挑战性的竞赛级数学题构成。这些题目涵盖代数、几何、数论与概率等多个主题，旨在检验模型进行复杂数学推理并生成逐步解答的能力。
:::

<a id="S046"></a>
**Source:** p.15 S046

:::en
**Original:** **AIME-2024** (Jia, 2024) includes 30 problems from the 2024 American Invitational Mathematics Examination (AIME), a prestigious high school mathematics competition. The dataset serves as a benchmark for evaluating models' capabilities in solving advanced mathematical problems that require deep understanding and creative problem-solving skills.
:::

:::zh
**中文:** **AIME-2024**（Jia, 2024）包含来自 2024 年美国数学邀请赛（AIME）的 30 道题目，这是一项颇具声望的高中数学竞赛。该数据集作为基准，用于评估模型解决需要深刻理解与创造性解题技巧的高阶数学问题的能力。
:::

<a id="S047"></a>
**Source:** p.15 S047

:::en
**Original:** **GSM8K** (Cobbe et al., 2021) is a dataset of 8,792 high-quality, linguistically diverse grade school math word problems created by human problem writers. The dataset is segmented into 7,473 training problems and 1,319 test problems, each requiring multi-step reasoning and basic arithmetic operations to solve.
:::

:::zh
**中文:** **GSM8K**（Cobbe et al., 2021）是一个由人类出题者编写的、包含 8,792 道高质量、语言多样的小学数学应用题的数据集。该数据集被划分为 7,473 道训练题与 1,319 道测试题，每道题都需要多步推理与基本算术运算才能求解。
:::

<a id="S048"></a>
**Source:** p.16 S048

:::en
**Original:** **HumanEval+** (Liu et al., 2023) is an extension of the HumanEval dataset, consisting of 164 original programming problems designed to assess the functional correctness of code generated by language models. Each problem includes a function signature, a docstring specifying the intended functionality, and multiple test cases for evaluation.
:::

:::zh
**中文:** **HumanEval+**（Liu et al., 2023）是 HumanEval 数据集的扩展，包含 164 道原始编程题，旨在评估语言模型所生成代码的功能正确性。每道题包含函数签名、说明预期功能的文档字符串（docstring），以及多个用于评测的测试用例。
:::

<a id="S049"></a>
**Source:** p.16 S049

:::en
**Original:** **MBPP+** (Liu et al., 2023) is an augmented version of the Mostly Basic Programming Problems (MBPP) dataset, comprising approximately 378 crowd-sourced Python programming tasks. Each task includes a natural language description, a reference solution, and three test cases, aiming to evaluate models' abilities in basic programming and problem-solving.
:::

:::zh
**中文:** **MBPP+**（Liu et al., 2023）是 Mostly Basic Programming Problems（MBPP）数据集的增强版本，包含约 378 个众包 Python 编程任务。每个任务包含一段自然语言描述、一个参考解与三个测试用例，旨在评估模型的基础编程与问题求解能力。
:::

---

## 附录 B：SliderQuant 的实现细节 / Appendix B: Implementation Details

### B.1 量化细节 / Quantization Details

<a id="S050"></a>
**Source:** p.16 S050

:::en
**Original:** By default, we adopt a uniform quantizer for both weights and activations to maintain simplicity and efficiency, while ensuring fair comparison with other post-training quantization methods. Specifically, we apply per-channel quantization for weights and per-token quantization for activations, using the WikiText2 dataset for calibration. For particularly challenging tasks where quantization errors can have a larger impact, specifically the MoE models evaluated in Table 4 and the mathematical reasoning and code generation tasks in Table 5, we employ group-wise quantization with a group size of 128 to preserve performance, instead using the C4 dataset for calibration.
:::

:::zh
**中文:** 默认情况下，我们对权重与激活都采用均匀量化器，以保持简洁与效率，同时确保与其他训练后量化方法的公平比较。具体而言，权重采用逐通道量化、激活采用逐 token 量化，并使用 WikiText2 数据集进行校准。对于量化误差影响更大的特别困难的任务——具体指表 4 中评估的 MoE 模型与表 5 中的数学推理和代码生成任务——我们改用分组量化（组大小 128）以保持性能，并改用 C4 数据集进行校准。
:::

<a id="E003"></a>
**Source:** p.16 E003 · Eq. (A)

$$
\text{quantizer}(\mathbf{Z})=\text{clamp}\left(\left\lfloor\frac{\mathbf{Z}}{\alpha}\right\rceil-\beta,\ 0,\ 2^{b}-1\right),\qquad
\alpha=\frac{\mathbf{Z}_{\max}-\mathbf{Z}_{\min}}{2^{b}-1},\qquad
\beta=\left\lfloor\frac{\mathbf{Z}_{\min}}{\alpha}\right\rceil
$$

:::zh
**中文说明：** 这是全文唯一的量化算子定义，$b$ 为位宽，$\alpha$ 为步长、$\beta$ 为零点。注意 $\beta$ 也取了整，因此反量化时用 $(\hat{Z}+\beta)\alpha$ 即可还原到浮点尺度；`clamp` 的上下界 $[0,2^b-1]$ 正是 $b$ 比特无符号整数的范围。
:::

<a id="S051"></a>
**Source:** p.16 S051

:::en
**Original:** where clamp$(x, Q_{min}, Q_{max})$ clips $x$ to the interval $[Q_{min}, Q_{max}]$; $\left\lfloor x \right\rceil$ denotes rounding to the nearest integer; $b$ is the bit-width; and $\mathbf{Z}_{\min}, \mathbf{Z}_{\max}$ are the minimum and maximum values of $\mathbf{Z}$, respectively.
:::

:::zh
**中文:** 其中 clamp$(x,Q_{min},Q_{max})$ 把 $x$ 裁剪到区间 $[Q_{min},Q_{max}]$；$\left\lfloor x\right\rceil$ 表示取最近整数；$b$ 为位宽；$\mathbf{Z}_{\min},\mathbf{Z}_{\max}$ 分别为 $\mathbf{Z}$ 的最小值与最大值。
:::

### B.2 通道缩放 / Channel Scaling

<a id="S052"></a>
**Source:** p.16 S052

:::en
**Original:** Recall that, in our SliderQuant, we adopt popular used channel scaling and low-rank adaptation (LoRA) to effectively remove outliers in weights and activations at each layer of a pre-trained LLM. For the channel scaling, we simply follow the implementation of OmniQuant (Shao et al., 2024). For the Llama (Touvron et al., 2023a) and Llama2 (Touvron et al., 2023b) families, we introduce learnable scaling factors for the $Q_{proj}$, $K_{proj}$, $V_{proj}$, $O_{proj}$, $Up_{proj}$, and $Down_{proj}$ operators. For the Llama3 (Dubey et al., 2024) and Qwen2.5 (Yang et al., 2024) families, due to the presence of Group Query Attention, we align the dimensions of the Query and Key by replicating the scaling factor. All scaling factors are initialized to 1 and absorbed into the adjacent weights after quantization, introducing no additional inference overhead.
:::

:::zh
**中文:** 回顾一下，在 SliderQuant 中我们采用广泛使用的通道缩放与低秩适配（LoRA），以有效去除预训练 LLM 每层权重与激活中的离群值。通道缩放部分我们直接沿用 OmniQuant（Shao et al., 2024）的实现。对 Llama（Touvron et al., 2023a）与 Llama2（Touvron et al., 2023b）家族，我们为 $Q_{proj}$、$K_{proj}$、$V_{proj}$、$O_{proj}$、$Up_{proj}$ 与 $Down_{proj}$ 算子引入可学习缩放因子。对 Llama3（Dubey et al., 2024）与 Qwen2.5（Yang et al., 2024）家族，由于存在分组查询注意力（GQA），我们通过复制缩放因子来对齐 Query 与 Key 的维度。所有缩放因子初始化为 1，并在量化后被吸收进相邻权重，不引入任何额外推理开销。
:::

<a id="TA"></a>
#### Table A. <span class="lang-zh">SliderQuant 的详细超参数设置</span>

**Placed near:** p.16 S054<br>
**Source:** p.16 CA

<img src="/images/quantization/sliderquant/tables/tabA.png" alt="Table A" width="743" height="576" loading="lazy" decoding="async">

:::en
**Original caption:** The detailed hyper-parameter settings of SliderQuant.
:::

:::zh
**中文表注:** SliderQuant 的详细超参数设置。
:::

:::zh
**Reading note:** 竖线 `|` 表示“按情形二选一”。例如校准集 `WikiText2 | C4` 指默认 WikiText2、MoE 与数学/代码任务用 C4；量化组大小 `channel-wise | 128` 指默认逐通道、特难任务用 128 分组；`Epochs (W2A16 | others) 60 | 20` 指 2 比特权重需要更多轮训练；`Batch size 3 | 1` 同理。这张表是复现的关键，注意学习率有两套：缩放因子 1e-3、LoRA 1e-4，均线性衰减到 0。
:::

### B.3 低秩适配 / Low-Rank Adaptation

<a id="S053"></a>
**Source:** p.16 S053

:::en
**Original:** For every LLM tested in our experiments, we apply LoRA to all the linear layers to reduce quantization loss through learnable weight adjustments. The rank of LoRA is set to 4 (i.e., $r$=4), which introduces significantly fewer learnable parameters compared to full parameter fine-tuning. After quantization, the additional learnable parameters introduced by LoRA are absorbed into the model weights, resulting in no extra computational overhead during inference.
:::

:::zh
**中文:** 对我们实验中测试的每一个 LLM，我们都在所有线性层上应用 LoRA，通过可学习的权重调整来降低量化损失。LoRA 的秩设为 4（即 $r=4$），相比全参数微调所引入的可学习参数少得多。量化结束后，LoRA 引入的额外可学习参数被吸收进模型权重，因而推理时没有额外计算开销。
:::

### B.4 超参数设置 / Hyper-parameters

<a id="S054"></a>
**Source:** p.16 S054

:::en
**Original:** For inter-layer sliding quantization, we set both the number of shallow layers $L_{s}$ and deep layers $L_{d}$ to 4 as default. For the remaining intermediate layers, we adopt a fixed-size sliding window with $s=2$ and $i=1$. For intra-layer sliding quantization, we set the sliding ratio $\gamma$ to 0.5. All the hyper-parameters are provided in Table A.
:::

:::zh
**中文:** 对层间滑动量化，我们默认把浅层数 $L_s$ 与深层数 $L_d$ 都设为 4；对余下的中间层，采用 $s=2$、$i=1$ 的固定尺寸滑动窗口。对层内滑动量化，滑动比例 $\gamma$ 设为 0.5。全部超参数见表 A。
:::

### B.5 量化硬件与部署加速 / Hardware and Deployment

<a id="S055"></a>
**Source:** p.17 S055

:::en
**Original:** Post-training quantization experiments for models up to 32B parameters are performed on a single NVIDIA RTX A6000 (48GB) GPU, while larger models such as the 65B and 70B variants require two A6000 GPUs. Since the A6000 serves as our primary experimental platform, its deployment results are reported in Table B. To provide a broader evaluation of hardware efficiency, we additionally benchmark the quantized models on other GPUs, including a consumer-grade NVIDIA RTX 4090 (24GB) and a data-center NVIDIA A100 (40GB), with results summarized in Tables C and D.
:::

:::zh
**中文:** 参数量不超过 32B 的模型，其训练后量化实验在单块 NVIDIA RTX A6000（48GB）GPU 上完成；更大的模型（如 65B 与 70B 变体）需要两块 A6000。由于 A6000 是我们的主要实验平台，其部署结果报告在表 B 中。为对硬件效率做更广泛的评估，我们还在其他 GPU 上对量化模型进行了基准测试，包括消费级的 NVIDIA RTX 4090（24GB）与数据中心的 NVIDIA A100（40GB），结果汇总于表 C 与表 D。
:::

<a id="S056"></a>
**Source:** p.17 S056

:::en
**Original:** Importantly, our method does not rely on customized quantization kernels. All deployment tests are conducted with the widely adopted llama.cpp framework using a batch size of 1, a prompt length of 512 tokens, and a generation length of 128 tokens. While specialized implementations such as GPTQ (Frantar et al., 2023) or AWQ (Lin et al., 2024b) kernels may yield higher acceleration ratios, our results demonstrate that the proposed method can be directly integrated into mainstream frameworks without additional engineering efforts, ensuring practical applicability and fair comparability across hardware platforms.
:::

:::zh
**中文:** 重要的是，我们的方法不依赖定制的量化 kernel。所有部署测试都用广泛采用的 `llama.cpp` 框架完成，批大小为 1、提示长度 512 个 token、生成长度 128 个 token。尽管 GPTQ（Frantar et al., 2023）或 AWQ（Lin et al., 2024b）等专用实现可能带来更高的加速比，我们的结果表明：所提方法无需额外工程投入即可直接集成进主流框架，从而保证了实际可用性以及跨硬件平台的公平可比性。
:::

<a id="TB"></a>
#### Table B. <span class="lang-zh">在单块 RTX A6000 上的推理效率</span>

**Placed near:** p.17 S055<br>
**Source:** p.17 CB

<img src="/images/quantization/sliderquant/tables/tabB.png" alt="Table B" width="1660" height="234" loading="lazy" decoding="async">

:::en
**Original caption:** Inference efficiency of quantized models on one NVIDIA RTX A6000 using llama.cpp. **W_Memory**: weight storage; **R_Memory**: peak runtime memory; **Tokens/s**: generated tokens per second.
:::

:::zh
**中文表注:** 使用 `llama.cpp` 时量化模型在单块 NVIDIA RTX A6000 上的推理效率。**W_Memory**：权重存储占用；**R_Memory**：运行时峰值内存；**Tokens/s**：每秒生成 token 数。
:::

:::zh
**Reading note:** 这张表回答“压缩率是否真的兑现成显存与速度”。重点看 Llama2-70B：FP16 需要 128.48 GB 直接 OOM，W4A16/W2A16 把它压进单卡（正文提到 W2A16 仅 17.03 GB、19.62 tokens/s）。注意 `R_Memory` 普遍比 `W_Memory` 大，因为还有 KV cache 与激活的运行时开销。
:::

<a id="TC"></a>
#### Table C. <span class="lang-zh">在 RTX 4090 上的推理效率</span>

**Placed near:** p.17 S055<br>
**Source:** p.17 CC

<img src="/images/quantization/sliderquant/tables/tabC.png" alt="Table C" width="1660" height="234" loading="lazy" decoding="async">

:::en
**Original caption:** Inference efficiency of quantized models on one NVIDIA RTX 4090 using llama.cpp. **W_Memory**: weight storage; **R_Memory**: peak runtime memory; **Tokens/s**: generated tokens per second.
:::

:::zh
**中文表注:** 使用 `llama.cpp` 时量化模型在单块 NVIDIA RTX 4090 上的推理效率。**W_Memory**：权重存储占用；**R_Memory**：运行时峰值内存；**Tokens/s**：每秒生成 token 数。
:::

:::zh
**Reading note:** 消费级 24GB 卡上，Llama2-13B 的 FP16 已经 OOM（峰值 24 GB 出头），量化版可以放下并跑出速度。这类结果说明方法的现实意义主要在**“能不能装进一张卡”**，而不只是困惑度小数点后的差异。
:::

<a id="TD"></a>
#### Table D. <span class="lang-zh">在 A100-40GB 上的推理效率</span>

**Placed near:** p.17 S055<br>
**Source:** p.17 CD

<img src="/images/quantization/sliderquant/tables/tabD.png" alt="Table D" width="1660" height="234" loading="lazy" decoding="async">

:::en
**Original caption:** Inference efficiency of quantized models on one NVIDIA A100-40GB using llama.cpp. **W_Memory**: weight storage; **R_Memory**: peak runtime memory; **Tokens/s**: generated tokens per second.
:::

:::zh
**中文表注:** 使用 `llama.cpp` 时量化模型在单块 NVIDIA A100-40GB 上的推理效率。**W_Memory**：权重存储占用；**R_Memory**：运行时峰值内存；**Tokens/s**：每秒生成 token 数。
:::

:::zh
**Reading note:** 与表 B/C 对照可看出：A6000 与 A100 的 tokens/s 差异主要来自显存带宽，而非量化格式；同一模型在不同卡上的相对排序一致，说明方法的收益不是某一平台的特例。
:::

---

## 附录 C：带旋转变换的 SliderQuant 实现细节 / Appendix C: SliderQuant+ with Rotations

<a id="S057"></a>
**Source:** p.17 S057

:::en
**Original:** Recall that in the Experiments section of the main paper, we apply our quantization framework SliderQuant with and without extra inference-time costs. In the default settings, all the additional parameters introduced by SliderQuant during quantization are merged into the original weights at inference. To further demonstrate the versatility of our approach, we also design a variant named SliderQuant+ that incorporates rotation transformations, corresponding to the results shown in Table 3 of the main paper. Our implementation is consistent with the QuaRot (Ashkboos et al., 2024b) codebase. The detailed design is illustrated in Figure A. The non-absorbable Hadamard transformations are added after query projection ($Q_{proj}$), key projection ($K_{proj}$) and before the output projection $O_{proj}$ in the multi-head self-attention module. In the feed-forward network (FFN), the transformations are added before the down-projection ($Down_{proj}$).
:::

:::zh
**中文:** 回顾正文实验部分，我们在有、无额外推理时开销两种情形下都应用了量化框架 SliderQuant。默认设置下，SliderQuant 在量化过程中引入的所有额外参数在推理时都被并入原始权重。为进一步展示方法的通用性，我们还设计了一个引入旋转变换的变体 SliderQuant+，对应正文表 3 的结果。我们的实现与 QuaRot（Ashkboos et al., 2024b）代码库保持一致。详细设计如图 A 所示。不可吸收的 Hadamard 变换被加在多头自注意力模块中 query 投影（$Q_{proj}$）之后、key 投影（$K_{proj}$）之后，以及输出投影 $O_{proj}$ 之前；在前馈网络（FFN）中，变换被加在 down 投影（$Down_{proj}$）之前。
:::

<a id="FA"></a>
#### Figure A. <span class="lang-zh">SliderQuant+ 中额外旋转变换的结构示意</span>

**Placed near:** p.17 S057<br>
**Source:** p.18 CA

<img src="/images/quantization/sliderquant/figures/figA.png" alt="Figure A" width="1660" height="601" loading="lazy" decoding="async">

:::en
**Original caption:** Structural illustrations on additional rotation transformations added in SliderQuant+.
:::

:::zh
**中文图注:** SliderQuant+ 中新增的额外旋转变换的结构示意。
:::

:::zh
**Reading note:** 绿色框是原模型算子（已把 Hadamard 矩阵吸收进权重，故写成 $Q^TW_q$、$Q^TW_oQ$ 这种“两侧都带 $Q$”的形式），红色框 $Q$ 是**无法吸收**的额外旋转：它夹在非线性（softmax）前后或残差汇合处，因此必须在推理时真的算一次矩阵乘。这正是“有额外推理开销”的来源；对比 [Fig. 2](#F002) 可以看到，SliderQuant 本体不引入任何这类红框。
:::

---

## 附录 D：SliderQuant 的量化效率 / Appendix D: Quantization Efficiency

<a id="S058"></a>
**Source:** p.18 S058

:::en
**Original:** To evaluate the quantization efficiency of SliderQuant, we measure its perplexity performance under different quantization time costs. As shown in Table E, SliderQuant achieves remarkably low perplexity after only 10 epochs. We adopt 20 epochs as the default setting to strike a balance between time overhead and performance. Figure B further illustrates the detailed relationship between quantization time and perplexity, demonstrating that our method converges rapidly. OmniQuant achieves 14.26|18.02 on WikiText2|C4 with the training time of 4.75 hours. Comparatively, with the training time less than 1 hour, SliderQuant achieves significantly better performance on both benchmarks (9.5|12.29 on WikiText2|C4). The results demonstrate the high quantization efficiency of SliderQuant. Even under extremely limited time budgets, the language models quantized with SliderQuant still maintains good performance. Extending the quantization duration can further improve performance.
:::

:::zh
**中文:** 为评估 SliderQuant 的量化效率，我们测量它在不同量化时间开销下的困惑度表现。如表 E 所示，SliderQuant 仅训练 10 个 epoch 就达到相当低的困惑度。我们采用 20 个 epoch 作为默认设置，以在时间开销与性能之间取得平衡。图 B 进一步展示了量化时间与困惑度之间的详细关系，表明我们的方法收敛很快。OmniQuant 在 4.75 小时训练时间下取得 WikiText2|C4 上的 14.26|18.02；相比之下，训练时间不足 1 小时的 SliderQuant 在两个基准上都明显更好（WikiText2|C4 上 9.5|12.29）。这些结果说明 SliderQuant 具有很高的量化效率：即便在极其有限的时间预算下，用它量化的语言模型仍能保持良好性能；延长量化时长还能进一步提升性能。
:::

<a id="TE"></a>
#### Table E. <span class="lang-zh">以 GPU 小时计的量化效率</span>

**Placed near:** p.18 S058<br>
**Source:** p.18 CE

<img src="/images/quantization/sliderquant/tables/tabE.png" alt="Table E" width="680" height="201" loading="lazy" decoding="async">

:::en
**Original caption:** The quantization efficiency of SliderQuant in terms of GPU hours. Experiments are conducted with Llama2-7B under W4A4 quantization on a single NVIDIA RTX A6000. The metric is perplexity.
:::

:::zh
**中文表注:** 以 GPU 小时计的 SliderQuant 量化效率。实验在单块 NVIDIA RTX A6000 上对 Llama2-7B 做 W4A4 量化，指标为困惑度。
:::

:::zh
**Reading note:** 10 epoch 已到 8.92/11.59（3.24 GPU 小时），20 epoch 8.34/11.10（6.14 小时），30/40 epoch 只再降 0.04–0.06。收益曲线在前 10 个 epoch 就走完大半，这也是后面 [Table Q](#TQ) 中 “SliderQuant-Fast”（10 epoch）能作为实用选项的原因。
:::

<a id="FB"></a>
#### Figure B. <span class="lang-zh">量化时间与困惑度的关系</span>

**Placed near:** p.18 S058<br>
**Source:** p.18 CB2

<img src="/images/quantization/sliderquant/figures/figB.png" alt="Figure B" width="1660" height="544" loading="lazy" decoding="async">

:::en
**Original caption:** The perplexity of SliderQuant on Llama2-7B under W4A4 quantization with different quantization time costs. Experiments are conducted on a single NVIDIA RTX A6000.
:::

:::zh
**中文图注:** Llama2-7B 在 W4A4 量化下、不同量化时间开销时 SliderQuant 的困惑度。实验在单块 NVIDIA RTX A6000 上进行。
:::

:::zh
**Reading note:** 左图 WikiText2、右图 C4。橙线是 SliderQuant 的收敛轨迹，两个绿点是 OmniQuant：OmniQuant 用 4.75 小时达到 14.26/18.02，而 SliderQuant 用 0.74 小时就到 9.59/12.29——图中标注 “Faster 642%”“Reduce 32.7% PPL” 就是这个对比。曲线在 6 小时后基本走平，说明默认 20 epoch 是合理的性价比点。
:::

---

## 附录 E：更多消融研究 / Appendix E: More Ablations

<a id="S059"></a>
**Source:** p.19 S059

:::en
**Original:** **The Number of Calibration Samples.** To ensure a fair comparison with prior methods, we use the calibration set with 128 samples as default. Here, we study the effect of the number of calibration samples to SliderQuant. The results are shown in Table F. Notably, even with only 32 samples, SliderQuant achieves the perplexity of 9.39|11.69 on Wikitext2|C4, demonstrating strong performance with a much smaller number of calibration samples. As the number of samples increases, performance gradually improves, reaching the perplexity of 8.34|11.10 on Wikitext2|C4 with 128 samples. Further increasing the number of calibration samples leads to consistent performance improvement. These results highlight the efficiency and robustness of SliderQuant, which maintains competitive performance even under a small number of calibration samples.
:::

:::zh
**中文:** **校准样本数量。** 为保证与先前方法的公平比较，我们默认使用含 128 个样本的校准集。这里研究校准样本数量对 SliderQuant 的影响，结果见表 F。值得注意的是，即便只有 32 个样本，SliderQuant 在 Wikitext2|C4 上也达到 9.39|11.69 的困惑度，说明在远更少的校准样本下也有强劲表现。随着样本数增加，性能逐步提升，128 个样本时在 Wikitext2|C4 上达到 8.34|11.10。继续增加校准样本还能带来持续的性能提升。这些结果凸显了 SliderQuant 的效率与鲁棒性：即便在校准样本很少的情况下，它也能保持有竞争力的性能。
:::

<a id="TF"></a>
#### Table F. <span class="lang-zh">校准样本数量的影响</span>

**Placed near:** p.19 S059<br>
**Source:** p.19 CF

<img src="/images/quantization/sliderquant/tables/tabF.png" alt="Table F" width="835" height="234" loading="lazy" decoding="async">

:::en
**Original caption:** Effect of the number of calibration samples. We apply SliderQuant on Llama2-7B under W4A4 quantization.
:::

:::zh
**中文表注:** 校准样本数量的影响。我们在 Llama2-7B 上以 W4A4 量化应用 SliderQuant。
:::

:::zh
**Reading note:** 32→128 个样本，困惑度从 9.38/11.69 降到 8.34/11.10，时间从 1.87 h 增到 6.14 h；样本效率的拐点在 64–128 附近。这个表常被用来回答“滑动窗口是不是很吃校准数据”——答案是它比逐层方法更吃一点，但 32 个样本就已可用。
:::

<a id="S060"></a>
**Source:** p.19 S060

:::en
**Original:** **SliderQuant with Group-Wise Quantization.** To ensure inference efficiency, we use channel-wise quantization for SliderQuant in the default settings. In Table G, we provide the results of SliderQuant with group-wise quantization. Compared to channel-wise quantization, group-wise quantization leads to better performance but increased parameter storage and slower inference speed, due to the need to maintain more quantization parameters. We can find that the performance gradually improves as the group size decreases, demonstrating a trade-off between performance and efficiency. When inference cost is not a limiting factor, applying finer-grained group-wise quantization within the SliderQuant framework can lead to even better performance, showcasing its flexibility and strong quantization capability.
:::

:::zh
**中文:** **采用分组量化的 SliderQuant。** 为保证推理效率，我们在默认设置下对 SliderQuant 使用逐通道量化。表 G 给出了 SliderQuant 使用分组量化的结果。与逐通道量化相比，分组量化带来更好的性能，但由于需要维护更多量化参数，参数存储增加、推理速度变慢。可以看到，随着组大小减小，性能逐步提升，体现了性能与效率之间的折中。当推理开销不是限制因素时，在 SliderQuant 框架内采用更细粒度的分组量化可以带来更好的性能，展示了其灵活性与强大的量化能力。
:::

<a id="TG"></a>
#### Table G. <span class="lang-zh">分组量化的消融（Llama2-7B，W2A16）</span>

**Placed near:** p.19 S060<br>
**Source:** p.19 CG

<img src="/images/quantization/sliderquant/tables/tabG.png" alt="Table G" width="772" height="234" loading="lazy" decoding="async">

:::en
**Original caption:** Ablation of SliderQuant with group-wise quantization on Llama2-7B under W2A16 quantization.
:::

:::zh
**中文表注:** SliderQuant 在 Llama2-7B、W2A16 量化下采用分组量化的消融。
:::

:::zh
**Reading note:** 组大小 32 时 8.20/10.90，逐通道时 9.59/13.83——细粒度分组在这个极端位宽下换来了 1.4/2.9 的困惑度改善。注意这是**同一框架内**的粒度调节，说明 SliderQuant 不是只能配固定的量化粒度，而是可以按部署预算滑动。
:::

---

## 附录 F：跨模型家族与量化设定的更多结果 / Appendix F: More Results

<a id="S061"></a>
**Source:** p.19 S061

:::en
**Original:** **About the Results for Counterpart Methods.** To demonstrate the effectiveness of SliderQuant, we compare it with other counterpart post-training quantization methods (e.g., AWQ, GPTQ, OmniQuant). Here, we clarify the details about our reported results for the methods. For perplexity on WikiText2 and C4, we adopt the evaluation results of Llama and Llama2 models as reported in the respective papers of the counterpart methods. For models and settings not covered in prior work—such as Llama3, Qwen2.5, and certain quantization configurations (e.g., W2A16)—we evaluate the results using public code and train the models to the best of our ability. For downstream tasks, we follow a prioritized strategy: reported results in the original papers, followed by evaluation using official checkpoints, and finally, reproduction via open-source code when necessary. For a fair comparison, we implement the fixed-size sliding quantization method CBQ with the same sliding window setting as we used in SliderQuant for intermediate layers.
:::

:::zh
**中文:** **关于对比方法结果。** 为展示 SliderQuant 的有效性，我们把它与其他对比训练后量化方法（如 AWQ、GPTQ、OmniQuant）进行比较。这里澄清我们所报告结果的来源细节。对于 WikiText2 与 C4 上的困惑度，Llama 与 Llama2 模型采用各对比方法论文中报告的结果。对于先前工作未覆盖的模型与设定——例如 Llama3、Qwen2.5 以及某些量化配置（如 W2A16）——我们用公开代码评测，并尽最大努力训练模型。对下游任务，我们遵循如下优先策略：优先采用原论文报告的结果，其次使用官方 checkpoint 评测，最后在必要时通过开源代码复现。为公平比较，我们按 SliderQuant 在中间层所用的相同滑动窗口设置，实现了固定尺寸滑动量化方法 CBQ。
:::

<a id="TH"></a>
#### Table H. <span class="lang-zh">Llama 家族的困惑度（Table H）</span>

**Placed near:** p.19 S062<br>
**Source:** p.20 CH

<img src="/images/quantization/sliderquant/tables/tabH.png" alt="Table H" width="1660" height="768" loading="lazy" decoding="async">

:::en
**Original caption:** Results comparison of different quantization methods without extra inference-time costs on Llama model family for the language generation tasks. The metric is perplexity.
:::

:::zh
**中文表注:** 无额外推理时开销的不同量化方法在 Llama 模型家族、语言生成任务上的结果对比。指标为困惑度。
:::

:::zh
**Reading note:** 补充了第一代 Llama（7B/13B/65B）。W4A4 下 Llama-7B 上 SmoothQuant 已到 25.25/32.32，而 SliderQuant 是 8.01/10.58；三种规模上 SliderQuant 都最低，65B 上绝对困惑度最低（6.20/8.38）。注意 65B 上 RTN 反而崩到 $3.7\times10^4$，说明大模型并不自动“更耐受”——方法与模型规模要一起看。
:::

<a id="S062"></a>
**Source:** p.19–20 S062

:::en
**Original:** **The Results on Llama Model Family.** In the main paper, we report results on the Llama2, Llama3, Qwen2.5 model families. Here, we further provide results on the first version of Llama family, including Llama-7B, Llama-13B and Llama-65B, evaluated on both language generation and zero-shot commonsense reasoning tasks. As shown in Tables H and I, SliderQuant consistently achieves the best performance under various quantization settings, outperforming existing methods such as RTN, AWQ (Lin et al., 2024b), GPTQ (Frantar et al., 2023), SmoothQuant (Xiao et al., 2023), OmniQuant (Shao et al., 2024) and CBQ (Ding et al., 2025). Notably, in challenging low-bit configurations like W4A4 and W2A16, SliderQuant maintains strong generation quality and reasoning accuracy, demonstrating its robustness and precision-preserving capability.
:::

:::zh
**中文:** **Llama 模型家族上的结果。** 正文报告了 Llama2、Llama3、Qwen2.5 模型家族的结果。这里进一步给出第一代 Llama 家族（含 Llama-7B、Llama-13B 与 Llama-65B）在语言生成与零样本常识推理任务上的结果。如表 H 与表 I 所示，在各种量化设定下 SliderQuant 都始终取得最佳性能，优于 RTN、AWQ（Lin et al., 2024b）、GPTQ（Frantar et al., 2023）、SmoothQuant（Xiao et al., 2023）、OmniQuant（Shao et al., 2024）与 CBQ（Ding et al., 2025）等现有方法。值得注意的是，在 W4A4 与 W2A16 这类困难的低位宽配置下，SliderQuant 仍保持很强的生成质量与推理准确率，展现了其鲁棒性与保精度能力。
:::

<a id="TI"></a>
#### Table I. <span class="lang-zh">Llama 家族的零样本常识推理（Table I）</span>

**Placed near:** p.19 S062<br>
**Source:** p.20 CI

<img src="/images/quantization/sliderquant/tables/tabI.png" alt="Table I" width="1660" height="614" loading="lazy" decoding="async">

:::en
**Original caption:** Results comparison of different quantization methods without extra inference-time costs on Llama model family for the zero-shot commonsense reasoning tasks. The metric is accuracy (%).
:::

:::zh
**中文表注:** 无额外推理时开销的不同量化方法在 Llama 模型家族、零样本常识推理任务上的结果对比。指标为准确率（%）。
:::

:::zh
**Reading note:** 与表 H 配对看：困惑度与下游准确率的变化方向一致，但准确率的下降更“非线性”——例如困惑度从 6 涨到 10 时，平均准确率可能掉 10 个点以上。这也是作者坚持同时报告两类指标的原因。
:::

<a id="S063"></a>
**Source:** p.20 S063

:::en
**Original:** **More Results with Weight-Activation Quantization.** In the main paper, we provide the results of Llama2-13B and Qwen2.5-14B with weight-activation quantization on zero-shot commonsense reasoning tasks. Here, we provide the additional results of other four models with different scales, including Llama2-7B, Llama2-70B, Qwen2.5-7B and Qwen2.5-32B. As shown in Table J, our SliderQuant consistently outperforms existing methods such as SmoothQuant, OmniQuant and CBQ across all tested models and tasks under W4A4 quantization. Even for smaller models like Llama2-7B and Qwen2.5-7B, which are generally more sensitive to quantization, SliderQuant achieves the best performance on average. For instance, on Llama2-7B, it reaches the average accuracy of 59.30%, notably higher than CBQ (56.26%) and OmniQuant (54.95%). Similarly, for the larger-scale Llama2-70B, our method reaches 65.24%, outperforming CBQ (62.70%) and OmniQuant (59.93%) with clear margins.
:::

:::zh
**中文:** **权重-激活量化的更多结果。** 正文给出了 Llama2-13B 与 Qwen2.5-14B 在权重-激活量化下、零样本常识推理任务的结果。这里补充另外四个不同规模模型的结果，包括 Llama2-7B、Llama2-70B、Qwen2.5-7B 与 Qwen2.5-32B。如表 J 所示，在 W4A4 量化下，我们的 SliderQuant 在所有被测模型与任务上都始终优于 SmoothQuant、OmniQuant 与 CBQ 等现有方法。即便对 Llama2-7B、Qwen2.5-7B 这类通常对量化更敏感的小模型，SliderQuant 的平均性能也最佳。例如在 Llama2-7B 上，它达到 59.30% 的平均准确率，明显高于 CBQ（56.26%）与 OmniQuant（54.95%）；同样，对规模更大的 Llama2-70B，我们的方法达到 65.24%，以明显优势超过 CBQ（62.70%）与 OmniQuant（59.93%）。
:::

<a id="TJ"></a>
#### Table J. <span class="lang-zh">四个模型的 W4A4 零样本常识推理（Table J）</span>

**Placed near:** p.20 S063<br>
**Source:** p.21 CJ

<img src="/images/quantization/sliderquant/tables/tabJ.png" alt="Table J" width="1660" height="797" loading="lazy" decoding="async">

:::en
**Original caption:** Results comparison of different quantization methods without extra inference-time costs on the zero-shot commonsense reasoning tasks under weight-activation quantization. The metric is accuracy (%).
:::

:::zh
**中文表注:** 权重-激活量化下，无额外推理时开销的不同量化方法在零样本常识推理任务上的结果对比。指标为准确率（%）。
:::

:::zh
**Reading note:** 覆盖 7B 到 70B 的四个模型。W4A4 下 Llama2-7B：SliderQuant 59.30 / CBQ 56.26 / OmniQuant 54.95 / FP16 70.54；Llama2-70B：65.24 / 62.70 / 59.93 / FP16 77.67。相对优势（约 2.5–4 个点）在各规模上稳定，但**绝对差距**在小模型上更大——量化误差对容量小的模型伤害更重，这也是 7B 常被当作“最难”测试点的原因。
:::

<a id="S064"></a>
**Source:** p.20 S064

:::en
**Original:** **More Results with Weight-Only Quantization.** In the main paper, we provide the results with weight-only quantization on language generation tasks. In Table K, we further provide the results of SliderQuant with weight-only quantization on zero-shot commonsense reasoning tasks, showcasing the performance of SliderQuant under W4A16 and W2A16 quantization settings across five representative models selected from the Llama and Qwen model families. We can see that SliderQuant achieves nearly lossless performance under W4A16, with accuracies closely matching the full-precision (W16A16) counterparts. For example, on Qwen2.5-14B, SliderQuant in W4A16 achieves the average accuracy of 76.74%, compared to 77.40% for the FP16 counterpart. Under the more aggressive W2A16 setting, performance degradation becomes more noticeable but remains within an acceptable range.
:::

:::zh
**中文:** **仅权重量化的更多结果。** 正文给出了语言生成任务上仅权重量化的结果。表 K 进一步给出 SliderQuant 在仅权重量化下、零样本常识推理任务的结果，展示其在 W4A16 与 W2A16 设定下、于 Llama 与 Qwen 家族中选出的五个代表性模型上的表现。可以看到，在 W4A16 下 SliderQuant 接近无损，准确率与全精度（W16A16）对应结果非常接近。例如在 Qwen2.5-14B 上，W4A16 的 SliderQuant 平均准确率为 76.74%，而 FP16 对应为 77.40%。在更激进的 W2A16 设定下，性能退化更明显，但仍处于可接受范围内。
:::

<a id="TK"></a>
#### Table K. <span class="lang-zh">五个模型的仅权重量化零样本结果（Table K）</span>

**Placed near:** p.20 S064<br>
**Source:** p.21 CK

<img src="/images/quantization/sliderquant/tables/tabK.png" alt="Table K" width="1660" height="655" loading="lazy" decoding="async">

:::en
**Original caption:** Zero-shot commonsense reasoning results under weight-only quantization of SliderQuant. The evaluation metric is accuracy (%).
:::

:::zh
**中文表注:** SliderQuant 在仅权重量化下的零样本常识推理结果。评测指标为准确率（%）。
:::

:::zh
**Reading note:** 表里只有 SliderQuant 自己（因为对比方法未在这些任务上报告仅权重量化结果），所以读法是**看它与 W16A16 行的差距**：W4A16 普遍只差 0.5–1.5 个点，W2A16 差距拉大到数点。这也提示：仅权重量化（W4A16）在实践中最“安全”，权重-激活（W4A4）才是真正的难点。
:::

<a id="S065"></a>
**Source:** p.21 S065

:::en
**Original:** As an additional complement to the results in the main paper, the comprehensive experiments across different model families (Llama, Llama2, Llama3, Qwen2.5), model scales (7B, 8B, 13B, 14B, 32B, 65B, 70B), quantization setting (W4A4, W2A16, W4A16), and benchmarks (language generation and zero-shot commonsense reasoning) further highlight the stable and superior performance of SliderQuant.
:::

:::zh
**中文:** 作为对正文结果的补充，跨不同模型家族（Llama、Llama2、Llama3、Qwen2.5）、模型规模（7B、8B、13B、14B、32B、65B、70B）、量化设定（W4A4、W2A16、W4A16）与基准（语言生成与零样本常识推理）的综合实验，进一步凸显了 SliderQuant 稳定而优越的性能。
:::

---

## 附录 G：为回复审稿意见补充的实验与讨论 / Appendix G: Rebuttal Experiments

<a id="S066"></a>
**Source:** p.21 S066

:::en
**Original:** In this section, we provide more experiments and discussions conducted for the rebuttal.
:::

:::zh
**中文:** 本节我们给出为回复审稿意见而开展的更多实验与讨论。
:::

### G.1 关于四个滑动窗口调度旋钮的研究 / The Four Schedule Knobs

<a id="S067"></a>
**Source:** p.21 S067

:::en
**Original:** Our SliderQuant has four schedule knobs including a progressively expanded sliding window (PESW) for $L_s$ shallow layers, a fixed-size sliding window withe the size of $s$ for $L_i$ intermediate layers and a progressively contracted sliding window (PCSW) for $L_d$ deep layers used in inter-layer sliding quantization (Inter-S), and an incremental quantization ratio $\gamma$ for intra-layer sliding quantization (Intra-S) within each window of Inter-S. In the Subsection 4.2 of the main paper, we used Llama2-7B to study their choices from the perspectives of their individual roles and combined roles, and chose $L_s=4, s=2, L_d=4, \gamma=0.5$ as the default setting of our SliderQuant applied to all different LLMs tested in our paper, for a simple implementation. Indeed, this default setting is not optimal even for Llama2-7B, let alone other LLMs. To better compare their choices, we additionally conducted multiple sets of ablative experiments with larger and recently released Qwen2.5-14B under W4A4 quantization. Detailed results are summarized in the below Table L, Table M, Table N and Table O, from which we can see that our SliderQuant: (1) with our default setting, four schedule knobs are complementary to each other (see Table L); (2) for each individual knob, its default setting is usually not the best for both Llama2-7B and Qwen2.5-14B (see Table M and Table N); (3) combining it with PESW and PCSW is significantly better than merely using fixed-size sliding quantization to all layers, both on Llama2-7B and Qwen2.5-14B (see Table O). Although our SliderQuant with the default setting already achieves superior results to existing PTQ methods,
:::

:::zh
**中文:** 我们的 SliderQuant 有四个调度旋钮：层间滑动量化（Inter-S）中用于 $L_s$ 个浅层的渐进扩张滑动窗口（PESW）、用于 $L_i$ 个中间层、尺寸为 $s$ 的固定尺寸滑动窗口、用于 $L_d$ 个深层的渐进收缩滑动窗口（PCSW），以及 Inter-S 每个窗口内层内滑动量化（Intra-S）的增量量化比例 $\gamma$。正文 4.2 节我们用 Llama2-7B 从“单独作用”与“组合作用”两个角度研究了它们的选择，并选定 $L_s=4,s=2,L_d=4,\gamma=0.5$ 作为 SliderQuant 的默认设置，应用于本文测试的所有不同 LLM，以求实现简洁。诚然，这一默认设置即便对 Llama2-7B 也并非最优，更不用说其他 LLM。为更好地比较它们的选择，我们额外在更大、更新发布的 Qwen2.5-14B 上、以 W4A4 量化开展了多组消融实验。详细结果汇总于下面的表 L、表 M、表 N 与表 O，从中可以看出我们的 SliderQuant：(1) 在默认设置下，四个调度旋钮彼此互补（见表 L）；(2) 对每个单独旋钮而言，其默认设置通常对 Llama2-7B 与 Qwen2.5-14B 都不是最优（见表 M 与表 N）；(3) 把它与 PESW、PCSW 结合，在 Llama2-7B 与 Qwen2.5-14B 上都显著优于对所有层仅使用固定尺寸滑动量化（见表 O）。尽管默认设置下的 SliderQuant 已经取得了优于现有 PTQ 方法的结果，
:::

<a id="TL"></a>
#### Table L. <span class="lang-zh">Qwen2.5-14B 上 Inter-S 与 Intra-S 的消融</span>

**Placed near:** p.21 S067<br>
**Source:** p.22 CL

<img src="/images/quantization/sliderquant/tables/tabL.png" alt="Table L" width="789" height="322" loading="lazy" decoding="async">

:::en
**Original caption:** Ablation of inter-layer sliding quantization (Inter-S) and intra-layer sliding quantization (Intra-S) on Qwen2.5-14B under W4A4 quantization.
:::

:::zh
**中文表注:** Qwen2.5-14B 在 W4A4 量化下，层间滑动量化（Inter-S）与层内滑动量化（Intra-S）的消融。
:::

:::zh
**Reading note:** 与 [Table 8](#T008) 同构，但模型换成了 Qwen2.5-14B。基线 17.41/25.20，加上 Intra-S 后 16.01/23.71，再加 PESW+PCSW 到 12.99/18.40，全开为 11.00/16.60。趋势与 Llama2-7B 完全一致——这是“方法不是只对某个模型调参有效”的证据。
:::

<a id="TM"></a>
#### Table M. <span class="lang-zh">Qwen2.5-14B 上 $L_s$/$L_d$ 的消融</span>

**Placed near:** p.21 S067<br>
**Source:** p.22 CM

<img src="/images/quantization/sliderquant/tables/tabM.png" alt="Table M" width="797" height="264" loading="lazy" decoding="async">

:::en
**Original caption:** Ablation of inter-layer sliding quantization with different choices of $L_s$ and $L_d$ on Qwen2.5-14B under W4A4 quantization.
:::

:::zh
**中文表注:** Qwen2.5-14B 在 W4A4 量化下，不同 $L_s$ 与 $L_d$ 选择的层间滑动量化消融。
:::

:::zh
**Reading note:** 2/2 → 6/6 时从 14.95/20.30 降到 12.68/17.?（持续下降，未见明显饱和）。这与 Llama2-7B 上“4/4 后饱和”略有不同，说明**更大的模型可以从更长的窗口继续获益**，默认 4/4 是为统一实现而做的保守选择。
:::

<a id="TN"></a>
#### Table N. <span class="lang-zh">Qwen2.5-14B 上 $\gamma$ 的消融</span>

**Placed near:** p.21 S067<br>
**Source:** p.22 CN

<img src="/images/quantization/sliderquant/tables/tabN.png" alt="Table N" width="789" height="209" loading="lazy" decoding="async">

:::en
**Original caption:** Ablation of intra-layer sliding quantization with different choices of $\gamma$ on Qwen2.5-14B under W4A4 quantization.
:::

:::zh
**中文表注:** Qwen2.5-14B 在 W4A4 量化下，不同 $\gamma$ 选择的层内滑动量化消融。
:::

:::zh
**Reading note:** 与 Llama2-7B 的 [Table 10](#T010) 类似，$\gamma=1.0$（不分阶段）最差（17.41/25.20），分 2–3 阶段最好，分 4 阶段又回落。两个模型上“2 阶段”都是稳健选择。
:::

<a id="TO"></a>
#### Table O. <span class="lang-zh">Qwen2.5-14B 上窗口尺寸 $s$ 的消融</span>

**Placed near:** p.21 S067<br>
**Source:** p.22 CO

<img src="/images/quantization/sliderquant/tables/tabO.png" alt="Table O" width="797" height="225" loading="lazy" decoding="async">

:::en
**Original caption:** Ablation of fixed-size sliding quantization with different choices of window size $s$ on Qwen2.5-14B under W4A4 quantization.
:::

:::zh
**中文表注:** Qwen2.5-14B 在 W4A4 量化下，固定尺寸滑动量化不同窗口尺寸 $s$ 的消融。
:::

:::zh
**Reading note:** 与 [Table 11](#T011) 同构：把 $s$ 从 2 加到 4，基线只从 17.41 改善到 14.75；而 Inter-S（$s=2$，仅浅/深层用 4）就到 12.99。再次说明**窗口放在哪里**比**窗口多大**更重要。
:::

<a id="S068"></a>
**Source:** p.22 S068

:::en
**Original:** these ablations indicate that there is still room to get improved quantization performance by choosing better settings of these four schedule knobs for different LLMs. Underlined entries indicate our default hyperparameter settings and their corresponding experimental results.
:::

:::zh
**中文:** 这些消融表明：针对不同的 LLM 为这四个调度旋钮选择更好的设置，仍有提升量化性能的空间。带下划线的条目表示我们的默认超参数设置及其对应的实验结果。
:::

### G.2 自适应滑动窗口量化 vs. 重复优化 / Adaptive Sliding vs. Repeated Optimization

<a id="S069"></a>
**Source:** p.22 S069

:::en
**Original:** In our SliderQuant, the first and the last layers are used as the anchor layer when quantizing shallow and deep layers, which means that they are quantized more times than the other layers. Naturally, a critical question is whether performance gain comes more from repeated optimization or our adaptive sliding quantization. To explore this question, we conducted an ablation to compare the roles of the sliding-window quantization design and the repeated optimization (quantization of all layers in each window by multiple times) on Llama2-7B and Qwen2.5-14B under W4A4 quantization. Detailed results are summarized in the below Table P, from which we can see that, on both Llama2-7B and Qwen2.5-14B: (1) for fixed-size sliding quantization with the window size of $s=2$ or $s=4$ applied to all model layers, performing multiple times of quantization of all layers in each window can improve the quantization performance, but the gain against its corresponding single time baseline is marginal ($<1$ perplexity); (2) comparatively, retaining a fixed-size sliding window $s=2$ for intermediate layers, by introducing a progressively expanded sliding window (PESW) for $L_s=4$ shallow layers, a progressively contracted sliding window (PCSW) for $L_d=4$ deep layers, our SliderQuant with this default setting achieves significant perplexity reductions (1.54 to 3.60 for Llama2-7B and 1.23 to 4.42 for Qwen2.5-14B on Wikitext2, with similar reductions observed on C4) over all fixed-sized sliding quantization baselines and its repeated variants. These experimental results indicate that our adaptive sliding quantization contributes significantly more to the final improvement of quantization performance compared to merely applying fixed-size sliding window to all layers of FP16 LLMs (even with repeated optimization), which well echoes the importance of our empirical observations and method's motivation.
:::

:::zh
**中文:** 在 SliderQuant 中，量化浅层与深层时第一层与最后一层被用作锚层，这意味着它们被量化的次数多于其他层。于是自然有一个关键问题：性能增益更多来自**重复优化**，还是来自我们的**自适应滑动量化**？为探究这个问题，我们在 Llama2-7B 与 Qwen2.5-14B 上、以 W4A4 量化做了消融，比较滑动窗口量化设计与重复优化（对每个窗口内所有层量化多次）各自的作用。详细结果汇总于表 P，可以看到，在 Llama2-7B 与 Qwen2.5-14B 上都有：(1) 对于把窗口尺寸 $s=2$ 或 $s=4$ 的固定尺寸滑动量化应用于所有层的情形，对每个窗口内所有层执行多次量化可以改善量化性能，但相对其对应的单次基线增益很小（困惑度 <1）；(2) 相比之下，中间层保持 $s=2$ 的固定尺寸滑动窗口，为 $L_s=4$ 个浅层引入 PESW、为 $L_d=4$ 个深层引入 PCSW，我们这一默认设置的 SliderQuant 相对所有固定尺寸滑动量化基线及其重复变体，取得了显著的困惑度下降（WikiText2 上 Llama2-7B 降 1.54–3.60、Qwen2.5-14B 降 1.23–4.42，C4 上观察到类似幅度）。这些实验结果表明，相比仅对 FP16 LLM 的所有层施加固定尺寸滑动窗口（即便配合重复优化），我们的自适应滑动量化对最终量化性能的改善贡献要大得多，这也很好地呼应了我们实证观察的重要性与方法的动机。
:::

<a id="TP"></a>
#### Table P. <span class="lang-zh">与“重复固定尺寸滑动量化”的对比</span>

**Placed near:** p.22 S069<br>
**Source:** p.23 CP

<img src="/images/quantization/sliderquant/tables/tabP.png" alt="Table P" width="1660" height="414" loading="lazy" decoding="async">

:::en
**Original caption:** Ablation of comparing SliderQuant with repeated fixed-size sliding quantization on Llama2-7B under W4A4 quantization. The repeated baseline applies quantization to all layers in each window multiple times with window size $s$. Left: Llama2-7B. Right: Qwen2.5-14B.
:::

:::zh
**中文表注:** Llama2-7B 在 W4A4 量化下，SliderQuant 与重复固定尺寸滑动量化的对比消融。重复基线对每个窗口内的所有层以窗口尺寸 $s$ 量化多次。左：Llama2-7B；右：Qwen2.5-14B。
:::

:::zh
**Reading note:** 这是全篇最重要的“排除替代解释”实验。重复优化把 $s=2$ 从 12.73 压到 12.11（4 次重复），把 $s=4$ 从 11.13 压到 10.67——都在 1 个点以内；而同一预算下换成自适应窗口（浅/深层 $s=4$、中间 $s=2$）直接到 9.13。**“多刷几遍”远不如“刷对地方”**，这句话就是这张表的结论。
:::

### G.3 训练时间开销与显存开销 / Training Time and Memory

<a id="S070"></a>
**Source:** p.22 S070

:::en
**Original:** We also conducted an ablation to compare the training time and the memory overhead of our SliderQuant (including our default version and a fast version), the fixed-size sliding quantization baseline (CBQ, which is closely related to our SliderQuant) and OmniQuant on both Llama2-7B and Qwen2.5-14B under W4A4 quantization. In the experiments, we always use the same number of calibration samples (128), the same batch size (1) and the same number of training epochs (20) for all methods unless otherwise stated. Note GPTQ is known as an efficient PTQ method but is tailored to weight-only quantization and is less related to our method, so it is not compared here. Detailed results are summarized in the below Table Q, from which we can see that, on both Llama2-7B and Qwen2.5-14B under W4A4 quantization: (1) in terms of training time, our SliderQuant (with the default setting of $L_s=4, s=2, L_d=4, \gamma=0.5$) is $1.08\times$ to $1.55\times$ slower than OmniQuant and the fixed-sized sliding quantization (CBQ) with the window size $s=2$, but is faster than the fixed-sized sliding quantization with the window size of $s=4$ or $s=3$, and our SliderQuant-Fast (the only change is the number of training epochs is 10 instead of 20) is the fastest ($1.21\times$ to $1.78\times$ faster than OmniQuant and the fixed-sized sliding quantization (CBQ) with the window size $s=2$) and achieves slightly worse quantization results compared to SliderQuant; (2) in terms of memory overhead, our SliderQuant (with the default setting of $L_s=4, s=2, L_d=4, \gamma=0.5$), SliderQuant-Fast and the fixed-sized sliding quantization (CBQ) with the window size of $s=4$ have the same memory usage, which is around $2\times$ compared to the memory usage by the fixed-sized sliding quantization (CBQ) with the window size of $s=2$ and OmniQuant; (3) in terms of model accuracy, both SliderQuant (with the default setting of $L_s=4, s=2, L_d=4, \gamma=0.5$) and SliderQuant-Fast always achieve significantly better perplexity than other counterpart methods, showing perplexity reductions ranging from 2.21 to 23.70 on Wikitext-2 and 1.93 to 41.15 on C4. Furthermore, Figure B in this Appendix presents a comprehensive view of SliderQuant's performance-efficiency trade-offs across different training time configurations. These results demonstrate that SliderQuant can achieve even greater training efficiency, showcasing up to $6.42\times$ speedup compared to OmniQuant on Llama2-7B while maintaining better perplexity on both datasets. Summarily, these results show that the training cost of SliderQuant is decent, which guarantees the practicality of our method to quantize different LLMs, also thanks to its simplicity.
:::

:::zh
**中文:** 我们还做了消融，在 Llama2-7B 与 Qwen2.5-14B 上、以 W4A4 量化，比较我们的 SliderQuant（含默认版本与快速版本）、固定尺寸滑动量化基线（CBQ，与 SliderQuant 关系密切）与 OmniQuant 的训练时间与显存开销。实验中，除特别说明外，所有方法都使用相同的校准样本数（128）、相同批大小（1）与相同训练 epoch 数（20）。注意 GPTQ 虽是公认高效的 PTQ 方法，但专为仅权重量化设计、与我们的方法关系不大，故不在此比较。详细结果汇总于表 Q，可以看到，在 Llama2-7B 与 Qwen2.5-14B 的 W4A4 量化下：(1) 训练时间上，我们的 SliderQuant（默认设置 $L_s=4,s=2,L_d=4,\gamma=0.5$）比 OmniQuant 与窗口尺寸 $s=2$ 的固定尺寸滑动量化（CBQ）慢 $1.08\times$–$1.55\times$，但快于窗口尺寸 $s=4$ 或 $s=3$ 的固定尺寸滑动量化；我们的 SliderQuant-Fast（唯一改动是把训练 epoch 从 20 减为 10）最快（比 OmniQuant 与 $s=2$ 的 CBQ 快 $1.21\times$–$1.78\times$），量化结果略差于 SliderQuant；(2) 显存开销上，我们的 SliderQuant（默认设置）、SliderQuant-Fast 与窗口尺寸 $s=4$ 的固定尺寸滑动量化（CBQ）占用相同显存，约为 $s=2$ 的 CBQ 与 OmniQuant 的 $2\times$；(3) 模型精度上，SliderQuant（默认设置）与 SliderQuant-Fast 始终取得显著优于其他对比方法的困惑度，在 Wikitext-2 上降低 2.21–23.70、在 C4 上降低 1.93–41.15。此外，本附录图 B 给出了 SliderQuant 在不同训练时间配置下性能-效率折中的全景。这些结果表明 SliderQuant 能取得更高的训练效率：在 Llama2-7B 上相对 OmniQuant 最高有 $6.42\times$ 加速，同时在两个数据集上保持更好的困惑度。总之，这些结果显示 SliderQuant 的训练成本是合理的，加之其简洁性，保证了它用于量化不同 LLM 的实用性。
:::

<a id="TQ"></a>
#### Table Q. <span class="lang-zh">训练时间与显存开销对比</span>

**Placed near:** p.22 S070<br>
**Source:** p.23 CQ

<img src="/images/quantization/sliderquant/tables/tabQ.png" alt="Table Q" width="1660" height="526" loading="lazy" decoding="async">

:::en
**Original caption:** Comparison of the training time and the memory overhead of our SliderQuant, fixed-size sliding quantization baseline and OmniQuant on Llama2-7B and Qwen2.5-14B under W4A4 quantization. All experiments are conducted on a single NVIDIA A6000-48G GPU, and we set the number of calibration samples to 128 (a popular choice in the quantization community), the batch size to 1, and the number of training epochs to 20 for all methods except our SliderQuant-Fast with 10 training epochs. Best results are bolded.
:::

:::zh
**中文表注:** Llama2-7B 与 Qwen2.5-14B 在 W4A4 量化下，我们的 SliderQuant、固定尺寸滑动量化基线与 OmniQuant 的训练时间与显存开销对比。所有实验在单块 NVIDIA A6000-48G 上进行；校准样本数设为 128（量化社区常用选择），批大小为 1，除 SliderQuant-Fast 为 10 个 epoch 外其余方法训练 20 个 epoch。最优结果加粗。
:::

:::zh
**Reading note:** 三个维度一起读：① 时间——SliderQuant 6.14 h，OmniQuant 4.75 h，但 SliderQuant-Fast 只要 3.24 h；② 显存——SliderQuant 系列 29.80 GB，是 OmniQuant（15.04 GB）的约 2 倍，这是“窗口取 4”的代价；③ 精度——8.34/11.10 vs 14.26/18.02。要注意这张表**没有**把显存算进“最佳”的比较里：显存换精度是这一框架的明确取舍，作者也没有回避。
:::

### G.4 数据效率与校准鲁棒性 / Calibration Efficiency

<a id="S071"></a>
**Source:** p.24 S071

:::en
**Original:** In the below Table R, we provide an ablation to study the performance of our SliderQuant under a smaller number of calibration samples {32, 64} and a larger number of calibration samples 256, besides the default number of calibration samples 128, and we compare our results with the reported results in the original papers of OmniQuant and DuQuant (a top rotation-based PTQ method). We can see that: (1) when using a smaller or larger number of calibration samples, our SliderQuant and SliderQuant+ (using rotation transformations) achieve significantly better results than OmniQuant and also better results than DuQuant (using learnable rotation transformations) consistently; (2) intriguingly, even with 32 calibration samples, our SliderQuant and SliderQuant+ show superior performance than OmniQuant and DuQuant with 256 calibration samples, respectively.
:::

:::zh
**中文:** 在下面的表 R 中，我们给出消融，研究 SliderQuant 在更少校准样本 {32, 64} 与更多校准样本 256（除默认的 128 之外）下的性能，并与 OmniQuant 及 DuQuant（一种顶尖的基于旋转的 PTQ 方法）原论文报告的结果比较。可以看到：(1) 无论使用更少还是更多的校准样本，我们的 SliderQuant 与 SliderQuant+（使用旋转变换）都显著优于 OmniQuant，也始终优于 DuQuant（使用可学习旋转变换）；(2) 有趣的是，即便只用 32 个校准样本，我们的 SliderQuant 与 SliderQuant+ 的表现也分别优于使用 256 个校准样本的 OmniQuant 与 DuQuant。
:::

<a id="TR"></a>
#### Table R. <span class="lang-zh">校准数据效率对比</span>

**Placed near:** p.24 S071<br>
**Source:** p.24 CR

<img src="/images/quantization/sliderquant/tables/tabR.png" alt="Table R" width="1660" height="401" loading="lazy" decoding="async">

:::en
**Original caption:** Comparison of calibration data efficiency under W4A4 quantization. Left: Llama-7B, comparing SliderQuant and OmniQuant with different numbers of calibration samples. Right: Llama2-7B, comparing SliderQuant+ (SliderQuant with rotation) and DuQuant with different numbers of calibration samples. Underlines denote the default configuration; bold indicates the best results.
:::

:::zh
**中文表注:** W4A4 量化下校准数据效率的对比。左：Llama-7B 上不同校准样本数时 SliderQuant 与 OmniQuant 的比较；右：Llama2-7B 上不同校准样本数时 SliderQuant+（带旋转的 SliderQuant）与 DuQuant 的比较。下划线表示默认配置，加粗表示最优结果。
:::

:::zh
**Reading note:** 左侧 32 个样本时 SliderQuant 8.71/11.19，而 OmniQuant 用 256 个样本才 11.41/14.90；右侧同理（SliderQuant+ 5.83/7.85 vs DuQuant 6.23/7.88）。注意左右两表**不是同一个模型**（左 Llama-7B、右 Llama2-7B），不要跨列直接比数值。
:::

---

## 附录 H：不同层对模型精度的量化影响可视化 / Appendix H: Layer-Sensitivity Visualizations

<a id="S072"></a>
**Source:** p.24 S072

:::en
**Original:** To further validate our empirical observation that different layers in LLMs exhibit varying sensitivity to quantization, we provide additional visualizations on Llama3-8B, Qwen2.5-7B, and Qwen2.5-32B, as shown in Figure C. Consistent with the observations discussed in the main paper (illustrated in Figure 1), these results reveal several important trends. First, for all tested LLMs, intermediate layers tend to be less sensitive to quantization, incurring smaller accuracy degradation compared to shallow and deep layers. This confirms that shallow and deep layers are more difficult to quantize and require special attention. Second, the first and last layers exhibit the highest quantization sensitivity, leading to the most significant increases in perplexity when they are quantized. This highlights their critical role in maintaining model fidelity. Third, as more layers are quantized sequentially from shallow to deep, the cumulative quantization error increases gradually, further demonstrating the compounding effect of poor quantization in sensitive layers. Among all methods, SliderQuant consistently achieves the lowest perplexity in both the single-layer and cumulative-layer quantization settings. This is because it explicitly focuses on reducing quantization errors in the more vulnerable shallow and deep layers, effectively controlling overall errors propagation. These additional visualizations provide further empirical evidence for layer-wise differences in quantization sensitivity, and reinforce the design rationale of SliderQuant. They demonstrate the necessity of quantization-aware strategies that account for such sensitivity variation, especially when targeting low-bit quantization.
:::

:::zh
**中文:** 为进一步验证我们“LLM 中不同层对量化表现出不同敏感度”这一实证观察，我们在 Llama3-8B、Qwen2.5-7B 与 Qwen2.5-32B 上提供了额外的可视化，如图 C 所示。与正文讨论（图 1 所示）的观察一致，这些结果揭示了若干重要趋势。第一，对所有被测 LLM，中间层往往对量化较不敏感，相比浅层与深层造成更小的精度退化。这证实浅层与深层更难量化、需要特别关注。第二，第一层与最后一层表现出最高的量化敏感度，被量化时导致困惑度上升最为显著，这凸显了它们在维持模型保真度上的关键作用。第三，随着从浅到深顺序量化的层数增多，累积量化误差逐步增大，进一步说明了敏感层中劣质量化的复合效应。在所有方法中，SliderQuant 在单层量化与累积层量化两种设定下都始终取得最低困惑度。这是因为它显式地致力于降低更脆弱的浅层与深层中的量化误差，从而有效控制整体误差传播。这些额外的可视化为进一步的“层间量化敏感度差异”提供了实证证据，并强化了 SliderQuant 的设计依据；它们表明，考虑此类敏感度差异的量化感知策略是必要的，尤其在面向低位宽量化时。
:::

<a id="FC"></a>
#### Figure C. <span class="lang-zh">Llama3-8B / Qwen2.5-7B / Qwen2.5-32B 上的层敏感度</span>

**Placed near:** p.24 S072<br>
**Source:** p.24 CC2

<img src="/images/quantization/sliderquant/figures/figC.png" alt="Figure C" width="1364" height="762" loading="lazy" decoding="async">

:::en
**Original caption:** Illustrations on the quantization impact of different layers to model accuracy: (1) quantizing a single layer (the first row) and (2) quantizing the first $l$ layers (the second row) of Llama3-8B, Qwen2.5-7B and Qwen2.5-32B. Here, we select three representative layer-wise, block-wise and multi-block-wise quantization methods, SmoothQuant, OmniQuant and CBQ, and examine them in 4-bit weight-activation (W4A4) quantization on WikiText2.
:::

:::zh
**中文图注:** 不同层对模型精度的量化影响示意：(1) 量化单个层（第一行）；(2) 量化前 $l$ 层（第二行），模型为 Llama3-8B、Qwen2.5-7B 与 Qwen2.5-32B。这里选取三种代表性的逐层、逐块与多块量化方法 SmoothQuant、OmniQuant 与 CBQ，在 WikiText2 上以 4 比特权重-激活（W4A4）量化进行考察。
:::

:::zh
**Reading note:** 与 [Fig. 1](#F001) 对照读：三张新图重复了 U 形与单调上升的形态，但幅度不同。最明显的是 Qwen2.5-32B——累计量化到 60 多层时 SmoothQuant 冲到 300 以上，而 SliderQuant 仍贴近虚线；Llama3-8B 的首层敏感度（约 7.0）比 Llama2 更突出。这说明“首末层最敏感”不是某个模型家族的偶然。
:::

---

## 附录 I：SliderQuant 中权重与激活的可视化 / Appendix I: Weight and Activation Ranges

<a id="S073"></a>
**Source:** p.25 S073

:::en
**Original:** To better understand how SliderQuant improves the quantization process, we visualize the numerical ranges of both activations and weights before and after applying it. Specifically, we compute the range as the difference between the maximum and minimum values—per channel for weights and per token for activations. Large value ranges are known to complicate quantization, especially under low-bit settings, as they increase the risk of information loss. Therefore, reducing these ranges can significantly ease quantization and improve accuracy. Taking Llama2-7B under the W4A4 quantization setting as an example, we examine the value ranges after merging the learnable parameters from OmniQuant and SliderQuant into the original weights, without actually applying quantization. This allows us to isolate the effect of these methods on the intrinsic distribution of the weights and activations, offering a clearer view of the quantization difficulty induced by each approach. As shown in Figures D to R, SliderQuant consistently reduces the channel-wise weight ranges across shallow, middle, and deep layers, clearly outperforming OmniQuant. Similarly, as shown in Figures G to R, the token-wise activation ranges after applying SliderQuant are significantly smaller than those in the original model and OmniQuant across a wide range of representative samples and layers. By simultaneously compressing the value ranges of both activations and weights across different model depths and inputs, SliderQuant reduces quantization difficulty and enables effective low-bit quantization with minimal performance degradation. This dual-range suppression is a key factor behind SliderQuant's robustness and near-lossless performance in challenging low-bit regimes.
:::

:::zh
**中文:** 为更好地理解 SliderQuant 如何改进量化过程，我们可视化应用它前后激活与权重的数值范围。具体而言，我们把范围计算为最大值与最小值之差——权重按通道计算，激活按 token 计算。众所周知，大的数值范围会使量化变难，尤其在低位宽设定下，因为它们增加了信息损失的风险；因此，缩小这些范围能显著降低量化难度并提升精度。以 W4A4 量化设定下的 Llama2-7B 为例，我们在把 OmniQuant 与 SliderQuant 的可学习参数并入原始权重之后（并不真正执行量化）考察数值范围。这样可以把这些方法对权重与激活内在分布的影响单独隔离出来，更清楚地看到各方法造成的量化难度。如图 D 至图 R 所示，SliderQuant 在浅层、中层与深层都一致地缩小了逐通道权重范围，明显优于 OmniQuant。类似地，如图 G 至图 R 所示，在广泛代表性样本与层上，应用 SliderQuant 后的逐 token 激活范围显著小于原始模型与 OmniQuant。通过在不同模型深度与输入上同时压缩激活与权重的数值范围，SliderQuant 降低了量化难度，使有效的低位宽量化在性能退化极小的前提下成为可能。这种“双范围抑制”正是 SliderQuant 在困难低位宽体制下保持鲁棒与近无损性能的关键因素之一。
:::

:::zh
**阅读提示（读者补充）**：图 D–R 是同一套读法，先记住三件事：

1. **横轴**是通道索引（权重）或 token 索引（激活），**纵轴**是 `max−min` 范围，越小越好量化；
2. 红点 = 原始 FP16 模型，绿点 = OmniQuant，橙点 = SliderQuant；
3. 橙点整体压在红、绿之下就说明“范围被压缩了”。

图 D–F 看权重（第 1/18/31 层，即浅/中/深），图 G–R 看激活（第 1/18/31 层 × 4 个随机样本）。注意激活图里 OmniQuant（绿）常常**高于**原始模型（红）——这不是画错，而是 OmniQuant 的平滑把激活推高以换取权重更好量化；SliderQuant 则把两者同时压低，这正是正文所说 “dual-range suppression” 的含义。
:::

<a id="FD"></a>
#### Figure D. <span class="lang-zh">第 1 层逐通道权重范围</span>

**Placed near:** p.25 S073<br>
**Source:** p.25 CD

<img src="/images/quantization/sliderquant/figures/figD.png" alt="Figure D" width="1739" height="580" loading="lazy" decoding="async">

:::en
**Original caption:** Visualization of channel-wise weight ranges (max–min) in the 1st layer of Llama2-7B under W4A4 quantization.
:::

:::zh
**中文图注:** Llama2-7B 在 W4A4 量化下第 1 层逐通道权重范围（max–min）的可视化。
:::

:::zh
**Reading note:** 首层六个投影（$Q/K/V/O/Up/Gate$）的橙点都贴近 0，而红点（原始）在 $Q_{proj}$ 上有零星尖峰到 20 以上。首层正是最敏感的层，范围被压平意味着量化步长可以更小。
:::

<a id="FE"></a>
#### Figure E. <span class="lang-zh">第 18 层逐通道权重范围</span>

**Placed near:** p.25 S073<br>
**Source:** p.26 CE

<img src="/images/quantization/sliderquant/figures/figE.png" alt="Figure E" width="1739" height="597" loading="lazy" decoding="async">

:::en
**Original caption:** Visualization of channel-wise weight ranges (max–min) in the 18th layer of Llama2-7B under W4A4 quantization.
:::

:::zh
**中文图注:** Llama2-7B 在 W4A4 量化下第 18 层逐通道权重范围（max–min）的可视化。
:::

:::zh
**Reading note:** 中间层的对比最“平淡”：三色差距本来就小，SliderQuant 仍略优。这与“中间层不敏感”的结论一致——在这里花力气收益有限，方法的价值主要不在中层。
:::

<a id="FF"></a>
#### Figure F. <span class="lang-zh">第 31 层逐通道权重范围</span>

**Placed near:** p.25 S073<br>
**Source:** p.26 CF

<img src="/images/quantization/sliderquant/figures/figF.png" alt="Figure F" width="1739" height="594" loading="lazy" decoding="async">

:::en
**Original caption:** Visualization of channel-wise weight ranges (max–min) in the 31st layer of Llama2-7B under W4A4 quantization.
:::

:::zh
**中文图注:** Llama2-7B 在 W4A4 量化下第 31 层逐通道权重范围（max–min）的可视化。
:::

:::zh
**Reading note:** 末层的 $Up_{proj}$/$Gate_{proj}$（MLP 扩张层）通道数上万，红点范围最高接近 3.0；橙点把它压到 0.5 以下。末层是另一个最敏感点，这里的压缩直接对应 [Table 1](#T001) 里 W4A4 的增益。
:::

<a id="FG"></a>
#### Figure G. <span class="lang-zh">第 1 层逐 token 激活范围（样本 1）</span>

**Placed near:** p.25 S073<br>
**Source:** p.26 CG

<img src="/images/quantization/sliderquant/figures/figG.png" alt="Figure G" width="1739" height="591" loading="lazy" decoding="async">

:::en
**Original caption:** Visualization of token-wise activation ranges (max–min) in the 1st layer of Llama2-7B for the first sample of 4 samples randomly selected from Wikitext2 under W4A4 quantization.
:::

:::zh
**中文图注:** Llama2-7B 在 W4A4 量化下、从 Wikitext2 随机选取的 4 个样本中的第 1 个样本，第 1 层逐 token 激活范围（max–min）的可视化。
:::

:::zh
**Reading note:** 注意绿色（OmniQuant）在 $K_{proj}$ 上被抬到 3.0，而橙点几乎贴 0——这是“把激活的难量化转移到权重”的代价；SliderQuant 在两侧同时压低。
:::

<a id="FH"></a>
#### Figure H. <span class="lang-zh">第 1 层逐 token 激活范围（样本 2）</span>

**Placed near:** p.25 S073<br>
**Source:** p.27 CH

<img src="/images/quantization/sliderquant/figures/figH.png" alt="Figure H" width="1739" height="597" loading="lazy" decoding="async">

:::en
**Original caption:** Visualization of token-wise activation ranges (max–min) in the 1st layer of Llama2-7B for the second sample of 4 samples randomly selected from Wikitext2 under W4A4 quantization.
:::

:::zh
**中文图注:** Llama2-7B 在 W4A4 量化下、从 Wikitext2 随机选取的 4 个样本中的第 2 个样本，第 1 层逐 token 激活范围（max–min）的可视化。
:::

:::zh
**Reading note:** 换一个输入样本，形态几乎不变——说明压缩效果不是对某个特定文本的过拟合。
:::

<a id="FI"></a>
#### Figure I. <span class="lang-zh">第 1 层逐 token 激活范围（样本 3）</span>

**Placed near:** p.25 S073<br>
**Source:** p.27 CI

<img src="/images/quantization/sliderquant/figures/figI.png" alt="Figure I" width="1739" height="596" loading="lazy" decoding="async">

:::en
**Original caption:** Visualization of token-wise activation ranges (max–min) in the 1st layer of Llama2-7B for the third sample of 4 samples randomly selected from Wikitext2 under W4A4 quantization.
:::

:::zh
**中文图注:** Llama2-7B 在 W4A4 量化下、从 Wikitext2 随机选取的 4 个样本中的第 3 个样本，第 1 层逐 token 激活范围（max–min）的可视化。
:::

:::zh
**Reading note:** 首层 $Q_{proj}$ 的红点在此样本上出现极值 >1000，橙点仍被压到 80 以下；这说明方法对**偶发尖峰**同样有效。
:::

<a id="FJ"></a>
#### Figure J. <span class="lang-zh">第 1 层逐 token 激活范围（样本 4）</span>

**Placed near:** p.25 S073<br>
**Source:** p.27 CJ

<img src="/images/quantization/sliderquant/figures/figJ.png" alt="Figure J" width="1739" height="593" loading="lazy" decoding="async">

:::en
**Original caption:** Visualization of token-wise activation ranges (max–min) in the 1st layer of Llama2-7B for the fourth sample of 4 samples randomly selected from Wikitext2 under W4A4 quantization.
:::

:::zh
**中文图注:** Llama2-7B 在 W4A4 量化下、从 Wikitext2 随机选取的 4 个样本中的第 4 个样本，第 1 层逐 token 激活范围（max–min）的可视化。
:::

:::zh
**Reading note:** 四个样本的图几乎可以互换，这是作者把 4 个样本全部放进附录的用意：结论不依赖单次抽样。
:::

<a id="FK"></a>
#### Figure K. <span class="lang-zh">第 18 层逐 token 激活范围（样本 1）</span>

**Placed near:** p.25 S073<br>
**Source:** p.28 CK

<img src="/images/quantization/sliderquant/figures/figK.png" alt="Figure K" width="1739" height="597" loading="lazy" decoding="async">

:::en
**Original caption:** Visualization of token-wise activation ranges (max–min) in the 18th layer of Llama2-7B for the first sample of 4 samples randomly selected from Wikitext2 under W4A4 quantization.
:::

:::zh
**中文图注:** Llama2-7B 在 W4A4 量化下、从 Wikitext2 随机选取的 4 个样本中的第 1 个样本，第 18 层逐 token 激活范围（max–min）的可视化。
:::

:::zh
**Reading note:** 中层激活的绝对范围比首层小一个量级（纵轴 0–30 量级），三色差距也小，与“中层好量化”一致。
:::

<a id="FL"></a>
#### Figure L. <span class="lang-zh">第 18 层逐 token 激活范围（样本 2）</span>

**Placed near:** p.25 S073<br>
**Source:** p.28 CL

<img src="/images/quantization/sliderquant/figures/figL.png" alt="Figure L" width="1739" height="596" loading="lazy" decoding="async">

:::en
**Original caption:** Visualization of token-wise activation ranges (max–min) in the 18th layer of Llama2-7B for the second sample of 4 samples randomly selected from Wikitext2 under W4A4 quantization.
:::

:::zh
**中文图注:** Llama2-7B 在 W4A4 量化下、从 Wikitext2 随机选取的 4 个样本中的第 2 个样本，第 18 层逐 token 激活范围（max–min）的可视化。
:::

:::zh
**Reading note:** $O_{proj}$ 上绿点（OmniQuant）明显高于红点，而橙点最低；这是 OmniQuant 与 SliderQuant 设计差异最直观的一处。
:::

<a id="FM"></a>
#### Figure M. <span class="lang-zh">第 18 层逐 token 激活范围（样本 3）</span>

**Placed near:** p.25 S073<br>
**Source:** p.28 CM

<img src="/images/quantization/sliderquant/figures/figM.png" alt="Figure M" width="1739" height="593" loading="lazy" decoding="async">

:::en
**Original caption:** Visualization of token-wise activation ranges (max–min) in the 18th layer of Llama2-7B for the third sample of 4 samples randomly selected from Wikitext2 under W4A4 quantization.
:::

:::zh
**中文图注:** Llama2-7B 在 W4A4 量化下、从 Wikitext2 随机选取的 4 个样本中的第 3 个样本，第 18 层逐 token 激活范围（max–min）的可视化。
:::

:::zh
**Reading note:** 与图 K/L 一致；可把三张并排看作“同一层在不同输入下的稳定性检查”。
:::

<a id="FN"></a>
#### Figure N. <span class="lang-zh">第 18 层逐 token 激活范围（样本 4）</span>

**Placed near:** p.25 S073<br>
**Source:** p.29 CN

<img src="/images/quantization/sliderquant/figures/figN.png" alt="Figure N" width="1739" height="597" loading="lazy" decoding="async">

:::en
**Original caption:** Visualization of token-wise activation ranges (max–min) in the 18th layer of Llama2-7B for the fourth sample of 4 samples randomly selected from Wikitext2 under W4A4 quantization.
:::

:::zh
**中文图注:** Llama2-7B 在 W4A4 量化下、从 Wikitext2 随机选取的 4 个样本中的第 4 个样本，第 18 层逐 token 激活范围（max–min）的可视化。
:::

:::zh
**Reading note:** 第 18 层的四个样本看完，可以确认中层的收益主要来自 $O_{proj}$ 与 MLP 相关投影，注意力分数本身（softmax 之前）变化不大。
:::

<a id="FO"></a>
#### Figure O. <span class="lang-zh">第 31 层逐 token 激活范围（样本 1）</span>

**Placed near:** p.25 S073<br>
**Source:** p.29 CO

<img src="/images/quantization/sliderquant/figures/figO.png" alt="Figure O" width="1739" height="596" loading="lazy" decoding="async">

:::en
**Original caption:** Visualization of token-wise activation ranges (max–min) in the 31st layer of Llama2-7B for the first sample of 4 samples randomly selected from Wikitext2 under W4A4 quantization.
:::

:::zh
**中文图注:** Llama2-7B 在 W4A4 量化下、从 Wikitext2 随机选取的 4 个样本中的第 1 个样本，第 31 层逐 token 激活范围（max–min）的可视化。
:::

:::zh
**Reading note:** 末层激活范围回升（纵轴到 3.0），与末层敏感度高的结论吻合；橙点在这个“最贵”的位置仍然最低。
:::

<a id="FP"></a>
#### Figure P. <span class="lang-zh">第 31 层逐 token 激活范围（样本 2）</span>

**Placed near:** p.25 S073<br>
**Source:** p.29 CP

<img src="/images/quantization/sliderquant/figures/figP.png" alt="Figure P" width="1739" height="593" loading="lazy" decoding="async">

:::en
**Original caption:** Visualization of token-wise activation ranges (max–min) in the 31st layer of Llama2-7B for the second sample of 4 samples randomly selected from Wikitext2 under W4A4 quantization.
:::

:::zh
**中文图注:** Llama2-7B 在 W4A4 量化下、从 Wikitext2 随机选取的 4 个样本中的第 2 个样本，第 31 层逐 token 激活范围（max–min）的可视化。
:::

:::zh
**Reading note:** 与图 O 同层不同样本，橙/绿差距在 $V_{proj}$ 上最大；这类差异正说明“逐 token 范围”会随输入波动，单一 token 的统计不足以代表整层。
:::

<a id="FQ"></a>
#### Figure Q. <span class="lang-zh">第 31 层逐 token 激活范围（样本 3）</span>

**Placed near:** p.25 S073<br>
**Source:** p.30 CQ

<img src="/images/quantization/sliderquant/figures/figQ.png" alt="Figure Q" width="1739" height="595" loading="lazy" decoding="async">

:::en
**Original caption:** Visualization of token-wise activation ranges (max–min) in the 31st layer of Llama2-7B for the third sample of 4 samples randomly selected from Wikitext2 under W4A4 quantization.
:::

:::zh
**中文图注:** Llama2-7B 在 W4A4 量化下、从 Wikitext2 随机选取的 4 个样本中的第 3 个样本，第 31 层逐 token 激活范围（max–min）的可视化。
:::

:::zh
**Reading note:** 橙点在中后段通道上出现一条“台阶”，这是逐 token 量化把个别离群 token 的缩放拉低的结果。
:::

<a id="FR"></a>
#### Figure R. <span class="lang-zh">第 31 层逐 token 激活范围（样本 4）</span>

**Placed near:** p.25 S073<br>
**Source:** p.30 CR

<img src="/images/quantization/sliderquant/figures/figR.png" alt="Figure R" width="1739" height="595" loading="lazy" decoding="async">

:::en
**Original caption:** Visualization of token-wise activation ranges (max–min) in the 31st layer of Llama2-7B for the fourth sample of 4 samples randomly selected from Wikitext2 under W4A4 quantization.
:::

:::zh
**中文图注:** Llama2-7B 在 W4A4 量化下、从 Wikitext2 随机选取的 4 个样本中的第 4 个样本，第 31 层逐 token 激活范围（max–min）的可视化。
:::

:::zh
**Reading note:** 15 张可视化到此结束。把 D–R 连起来看的结论只有一句：**SliderQuant 在“最该压”的首末层压得最多，在“本就好压”的中层至少不添乱**。
:::

---

:::zh
## 术语表

| 英文 | 中文 | 说明 / 首次出现 |
|---|---|---|
| post-training quantization (PTQ) | 训练后量化 | 只用少量校准样本、不重训练的低精度化，[S001](#S001) |
| quantization-aware training (QAT) | 量化感知训练 | 需要重训练流程，作为对照，[S002](#S002) |
| weight-only quantization | 仅权重量化 | 只压权重，激活保持 FP16（W4A16/W2A16），[S009](#S009) |
| weight-activation quantization | 权重-激活量化 | 权重与激活都压（W4A4），[S011](#S011) |
| layer-wise / block-wise / multi-block-wise quantization | 逐层／逐块／多块量化 | 序列量化的三种粒度，[S003](#S003)–[S004](#S004) |
| sequential quantization framework | 序列量化框架 | 把模型切成片段依次量化，[S003](#S003) |
| sliding window $\{s,i\}$ | 滑动窗口 | $s$ 窗口层数、$i$ 每步移动、重叠 $s-i$，[S010](#S010) |
| shallow / intermediate / deep layers | 浅层／中间层／深层 | 按深度三分，默认各 4/其余/4，[S014](#S014) |
| inter-layer sliding quantization (Inter-S) | 层间滑动量化 | 沿层方向滑的三段式窗口，[S017](#S017) |
| intra-layer sliding quantization (Intra-S) | 层内滑动量化 | 窗口内按 $\gamma$ 分阶段联合量化，[S018](#S018) |
| PESW | 渐进扩张滑动窗口 | 浅层用，锚定首层，[S017](#S017) |
| PCSW | 渐进收缩滑动窗口 | 深层用，锚定末层，[S017](#S017) |
| FSSW | 固定尺寸滑动窗口 | 中间层用，$s=2,i=1$，[S017](#S017) |
| anchor layer | 锚层 | 每步都在窗口内的层（首层或末层） |
| calibration samples | 校准样本 | 默认 128 条 × 2048 token，[S019](#S019) |
| perplexity (PPL) | 困惑度 | 语言生成指标，越低越好 |
| zero-shot commonsense reasoning | 零样本常识推理 | PIQA/ARC/HS/WG/BoolQ/MMLU，[S022](#S022) |
| mixture of experts (MoE) | 混合专家 | Qwen3-30B-A3B，[S021](#S021) |
| chain-of-thought | 思维链 | DeepSeek-R1 蒸馏模型的能力来源，[S027](#S027) |
| rotation transformation | 旋转变换 | QuaRot/SpinQuant/FlatQuant 一类方法，[S011](#S011) |
| absorbable / non-absorbable | 可吸收／不可吸收 | 能否并入相邻权重（决定有无推理开销），[S023](#S023) |
| channel scaling (CS) | 通道缩放 | 逐通道因子 $\alpha_i$，[S019](#S019) |
| low-rank adaptation (LoRA) | 低秩适配 | 秩 $r=4$ 的加性修正，[S019](#S019) |
| outlier | 离群值 | 幅度远大于其他的权重/激活 |
| group-wise quantization | 分组量化 | 组大小 128，用于 MoE 与推理任务，[S050](#S050) |
| per-channel / per-token | 逐通道／逐 token | 权重按输出通道、激活按 token 取范围，[S050](#S050) |
| mixed-precision quantization | 混合精度量化 | 部分权重/激活留 FP16，[S028](#S028) |
| uniform quantizer | 均匀量化器 | [E003](#E003) |
| ablation study | 消融实验 | [S029](#S029) |
| counterpart methods | 对比方法 | [S023](#S023) |

## 公式索引

- [E001 · Eq. (1)](#E001) — p.4，窗口级重建目标：$\arg\min_{\hat{\mathbf{W}}}\|\mathcal{F}(\mathbf{W},\mathbf{X})-\mathcal{F}(\hat{\mathbf{W}},\mathbf{X})\|_2^2$
- [E002 · Eq. (2)](#E002) — p.6，通道缩放 + LoRA 的逐层量化流程（含 $\oslash,\odot$）
- [E003 · Eq. (A)](#E003) — p.16，均匀量化器定义（$\alpha$ 步长、$\beta$ 零点、$b$ 位宽）

## 阅读提示与批判性笔记

> 本节为读者补充，不是原文内容。目的是把“论文说了什么”与“我们该信到什么程度”分开。

### 一句话定位

如果只记一件事：**这篇论文的贡献不是新的量化算子，而是新的“量化顺序/分组调度”**。它把序列 PTQ 里那个“隐含的均匀假设”显式化，用重叠窗口 + 自适应窗口尺寸把它打破，其余部件（均匀量化、通道缩放、LoRA）都是现成的。因此它的收益是**正交于**旋转类方法（QuaRot/SpinQuant/FlatQuant）的——[Table 3](#T003) 里 SliderQuant+ 与这些方法同台，正是这个论点的证据。

### 证据强度：哪些结论可以放心用

| 结论 | 证据 | 可信度 |
|---|---|---|
| 浅/深层比中间层更敏感，首/末层最敏感 | [Fig. 1](#F001) + [Fig. C](#FC)，3 种方法 × 6 个模型 × 两种量化切法 | 高（跨模型、跨方法一致） |
| 自适应窗口优于固定窗口 | [Table 8](#T008)、[Table 11](#T011)、[Table L](#TL)、[Table O](#TO) | 高（消融闭合） |
| 收益不是来自“多刷几遍” | [Table P](#TP)：重复优化 <1 点，换窗口 >1.5 点 | 高（排除了主要替代解释） |
| 与旋转类方法可叠加 | [Table 3](#T003) 全模型平均最优 | 中高（增益 0.4–1.3 点，幅度不大） |
| 推理模型上近无损 | [Table 5](#T005)：32B W4A16 平均 82.71 vs 83.17 | 中（单一模型家族，AIME 仅 30 题，方差大） |
| 校准数据效率高 | [Table R](#TR)：32 样本胜 OmniQuant 256 样本 | 中（跨模型比较，非同表同模型） |

### 值得追问的地方

1. **成本被淡化了。** [Table Q](#TQ) 显示显存约 2×（29.80 GB vs 15.04 GB）、训练 6.14 h vs OmniQuant 4.75 h；对 70B 需要两张 A6000。论文的“高效”指的是**单位精度收益**，不是绝对开销。要做部署选型，得把这块算进去。
2. **默认超参是“统一”的，而作者自己承认它不最优。** §G.1 明确说 $L_s=4,s=2,L_d=4,\gamma=0.5$ 对 Qwen2.5-14B 也不是最优（[Table M](#TM) 中 6/6 更好）。也就是说，**每个模型都还有免费的性能没拿**——但怎么自动选，论文没给方法。
3. **收益集中在极低位宽。** W4A16 上 5.61 vs FP16 5.47（[Table 1](#T001)），差距已很小；W4A4 才是主战场。若你的场景是 W8A8 或 W4A16，收益可能不足以抵消复杂度。
4. **“为什么是这三种窗口”缺少理论。** PESW/PCSW 的对称性（首层局部→全局、末层全局→局部）很优雅，但没有误差上界或收敛性分析支撑；相比之下 GPTQ 有 OBS 框架。这属于经验方法的通病，不算硬伤，但意味着**换模型时需要重新验证**。
5. **端到端加速未充分验证。** 只用 `llama.cpp`、无定制 kernel（§B.5 明确说明），也没与 AWQ/GPTQ 的专用 kernel 比吞吐。所以“压缩率→实际加速”的链条只验证了一半。
6. **敏感度的来源没有解释。** 论文观察到首末层敏感，归因于“基础特征提取/最终特征抽象”，这是合理的直觉，但没有做表征层面的分析（例如首层误差如何传播到 logits）。

### 与已读论文的关系

- **SpinQuant / QuaRot / FlatQuant**：同属“旋转类”，通过改变**坐标基底**让离群值消失；SliderQuant 改变的是**量化顺序与分组**，不碰基底。两者正交，[Table 3](#T003) 的 SliderQuant+ 就是组合版（旋转部分直接复用 QuaRot 实现）。
- **GPTQ**：逐层 + 二阶 Hessian 近似，是“层内更聪明”；SliderQuant 是“层间更聪明”。[S015](#S015) 指出逐层/逐块都是滑动量化的特例，等于把 GPTQ 一类方法纳入了同一个坐标系。
- **OmniQuant**：逐块 + 可学习平滑/裁剪，是本文最主要的对照；两者都用 CS+LoRA，差别就在窗口调度，因此 [Table 8](#T008) 的对比相当干净。
- **CBQ / SKVQ**：固定尺寸滑动窗口的直接前身，本文的 baseline。

### 复现清单

| 项 | 设置 |
|---|---|
| 硬件 | ≤32B：1× RTX A6000 48GB；65B/70B：2× A6000 |
| 校准 | WikiText2，128 条 × 2048 token（MoE 与数学/代码任务改用 C4） |
| 量化粒度 | 权重逐通道、激活逐 token；特难任务用 group=128 |
| 方法超参 | $L_s=L_d=4$、$s=2$、$i=1$、$\gamma=0.5$、LoRA $r=4$ |
| 优化 | AdamW，缩放因子 lr $10^{-3}$、LoRA lr $10^{-4}$，线性衰减到 0；20 epoch（W2A16 为 60） |
| 部署 | `llama.cpp`，batch 1、prompt 512、generate 128 |
| 评测 | LM Evaluation Harness（常识）、OpenCompass（数学）、EvalPlus（代码） |

### 三个可以接着做的方向

1. **自动调度**：用一次前向的敏感度估计（如逐层 Fisher 或输出扰动）直接定 $L_s,L_d,s$，替代人工网格——附录 G.1 已经把“默认非最优”摆在那里，这是一个明确的空位。
2. **与 KV-cache 量化结合**：本文只动权重与激活，而长上下文部署的瓶颈常在 KV cache；滑动窗口的思想是否可以搬到 KV 的 head/token 维度？
3. **理论刻画**：把“重叠层数 → 误差上界”的关系写出来，哪怕只在线性化假设下，也能解释为什么 2 阶段最优、为什么 4/4 饱和。
:::
