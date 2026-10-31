---
title: "GPU Architecture：NVIDIA SM 演进笔记开篇"
published: 2026-10-31T10:00:00+08:00
description: 新开 GPU Architecture 分类：按 CUDA compute capability 梳理 NVIDIA 每一代 SM 的改动，从 Tesla 到 Rubin，重点写这些改动对 kernel 的影响。
tags: [CUDA, GPU, NVIDIA, 计算能力]
category: GPU Architecture
draft: false
---

写 kernel 的时候，很多“为什么”最后都会落到架构上：为什么 Hopper 的 SM 能把 shared memory 开到 228KB，Turing 只有 64KB；为什么 Volta 之后同一个 warp 里的线程可以各走各的分支；为什么 Ampere 的 TF32 能让 GEMM 快一截；为什么 Rubin 的编号是 10.7，而不是 11 或 13。

这些问题的答案不在 CUDA API 里，而在 SM（streaming multiprocessor）的迭代里。这个分类按 NVIDIA 的 compute capability 逐代拆 SM：每一代增加了什么、拿掉了什么、这些改动如何改变 occupancy、访存和指令选择。

## 关注范围

- **SM 结构**：CUDA core 数量、warp scheduler、寄存器文件、shared memory 与 L1 的容量和划分。
- **存储层级**：从 Tesla 没有统一的 L1 data cache，到 Fermi 的可配置 L1/shared，再到 Hopper 的 TMA、Blackwell 的 TMEM。
- **专用单元**：FP64、Tensor Core、RT Core、DPX、TMA，以及它们各自的精度和调用方式。
- **精度**：FP32 → FP64 → FP16 → TF32 → FP8 → FP4/FP6，以及每一代 Tensor Core 支持什么。
- **兼容性**：compute capability 的 major/minor、cubin 规则、`a` / `f` 编译目标，以及“同名架构、不同 CC”的坑。
- **代表产品**：数据中心的 B200/H100、消费级的 RTX 5090、边缘的 Jetson Thor，分别落在哪一档。

## 和 Kernel 分类的分工

[Kernel 分类](/posts/kernel/intro/)讲“怎么写”：执行模型、内存层级、GEMM/Attention 的 tiling、Triton/TileLang 的写法。这里讲“为什么硬件是这样”，以及这些设计对 kernel 的约束。比如 shared memory 的容量和 bank 数量决定了 tile 能开多大，Tensor Core 的精度决定了 MMA 指令怎么选，TMA 和 cluster 决定了 Hopper 之后的流水线怎么写。

两边的文章会互相链接：需要看具体实现时去 Kernel，需要理解某个限制的来历时回到这里。

## 写作约定

- 以 NVIDIA 官方文档（CUDA Programming Guide、whitepaper、release notes）为主，产品规格以官方 compute capability 列表为准。
- 官方没有公开的微架构细节会明确标成“未公开”或“推断”，不把猜测写成事实。
- 性能数字交代测试条件；没有实测的部分只讲结构上的收益和代价。
- 每一代尽量回答三个问题：改了什么、为什么改、写 kernel 时能感受到什么。

## 怎么读

先看 [CUDA Compute Capability 版本地图](/posts/gpu-arch/compute-capability/)，把编号、兼容性和版本表建立起来。然后按五代读：

1. [Tesla 与 Fermi](/posts/gpu-arch/tesla-fermi/)：统一着色器之后，SM 才开始像计算核心。
2. [Kepler 与 Maxwell](/posts/gpu-arch/kepler-maxwell/)：192 核 SMX 和 128 核 SMM 的路线分叉。
3. [Pascal、Volta 与 Turing](/posts/gpu-arch/pascal-volta-turing/)：FP16、Tensor Core、RT Core 依次进入 SM。
4. [Ampere、Ada 与 Hopper](/posts/gpu-arch/ampere-ada-hopper/)：TF32、稀疏化、FP8 和异步流水线。
5. [Blackwell 与 Rubin](/posts/gpu-arch/blackwell-rubin/)：双 die、FP4、TMEM，以及 10.x 家族的分裂。

## 版本口径

截至 2026-10-31，官方列表覆盖 1.0 到 12.1。Rubin 是 10.7（CUDA 13.4 已支持），Jetson Thor 是 11.0，Blackwell 消费级是 12.0/12.1。4.x 从未出现；Rubin Ultra、Feynman 还没有官方 CC，等公布后再补。

一个提醒：compute capability 的大版本不是“第几代”的市场编号。同样叫 Blackwell，B200 是 10.0，RTX 5090 是 12.0，Jetson Thor 是 11.0；同样叫 8.x，A100 是 8.0，RTX 4090 是 8.9。看 CC，不要看产品名。
