---
title: "Quantization：低精度推理笔记开篇"
published: 2026-10-26
description: 新开 Quantization 分类，用来整理 FP8/INT4 等低精度格式、量化算法、KV Cache 压缩、kernel 实现与精度评测。
tags: [Quantization, 推理优化, 低精度]
category: Quantization
draft: false
---

量化不是“把权重压小一点”这么简单：它同时牵动数值格式、缩放因子、校准数据、kernel 实现和最终精度。这个分类专门收这类内容，把散落在论文、框架和实测里的结论整理成可复用的工程笔记。

## 关注范围

- **数值格式**：FP8（E4M3 / E5M2）、INT8 / INT4、MXFP4 / NVFP4，指数位与尾数位如何决定动态范围和精度。
- **量化粒度**：per-tensor、per-channel、per-group，以及 group size 对精度和反量化开销的取舍。
- **算法与流程**：PTQ 的校准、AWQ / GPTQ 一类权重-only 方法、QAT 与蒸馏。
- **KV Cache**：K/V 分开量化、per-head / per-token 粒度，以及长上下文下的误差累积。
- **Kernel 与系统**：dequant 融合、W4A16 / W8A8 GEMM、tensor core 对齐和显存带宽收益。
- **评测**：perplexity 之外，任务准确率、长尾输入和极端激活值该怎么测。

## 写作约定

每篇尽量把“理论收益”和“实测数字”分开写，并交代硬件、框架版本、校准集和评测集；没有把握的结论会标明是推断。涉及 kernel 的文章会给出可复现的命令或最小代码。

## 怎么读

分类页按发布时间倒序排列，侧栏的 Quantization 入口会随文章增加自动更新。第一篇长文已经从一份可运行的 notebook 整理出来：[量化从零开始：从仿射量化到 FP4 的工程笔记](/posts/quantization/from-scratch/)，建议从那里开始，再看具体模型上的落地案例。

想先看论文的话，[量化论文汇总：从 INT8 到 FP4 的阅读地图](/posts/quantization/papers/) 按主题整理了 70 篇左右主线工作，包括基础 PTQ/QAT、outlier、GPTQ/AWQ、旋转 W4A4、KV cache、FP8/FP4 和系统 kernel，并给出三条阅读路线与选型建议。

如果准备精读某一篇，两篇中英对照精读可以当模板：[SliderQuant](/posts/quantization/sliderquant/) 与 [SpinQuant](/posts/quantization/spinquant/)。它们逐块给出原文页码、英文原文与中文译文，图表紧裁切内嵌，页面头部可以在 **中英对照 / 中文 / English** 之间切换；SpinQuant 那篇还带了 RMSNorm / 残差流 / RoPE / Hadamard 四段读前预备和 7 处补充推导。
