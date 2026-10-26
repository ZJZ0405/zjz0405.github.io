---
title: "Kernel：GPU 算子工程笔记开篇"
published: 2026-10-27
description: 新开 Kernel 分类，收 GPU 执行模型、内存层级、GEMM 与 Attention 等算子实现、低精度 kernel 融合，以及 profiling 和性能建模的工程笔记。
tags: [CUDA, GPU, 性能优化]
category: Kernel
draft: false
---

模型跑得快不快，最后都落在一个个 kernel 上：同一个算法，tiling、访存和流水不同，性能可以差出几倍。这个分类专门收 GPU 算子层面的内容，把架构手册、框架源码和 profiler 数字对起来，写成能复现的工程笔记。

和另外两个分类的分工：Quantization 讲量化算法、数值格式和精度，这里讲量化之后 kernel 怎么落地；LLM Engine 讲框架的调度与请求链路，这里讲单个算子内部怎么实现。

## 关注范围

- **执行模型**：warp / wavefront、SM 资源占用、occupancy、寄存器压力、指令级并行，以及这些量如何决定延迟隐藏。
- **内存层级**：合并访存、bank conflict、shared memory 布局、`cp.async` 与 TMA、L2 复用，以及在 HBM 带宽约束下于 roofline 上的位置。
- **核心算子**：GEMM 的分层 tiling 与 epilogue、FlashAttention 的 online softmax、reduction / scan、top-k 采样、MoE 路由和 KV Cache 搬运。
- **低精度与融合**：FP8 / FP4 的 dequant 融合、W4A16 / W8A8、per-group scale 布局，以及 Tensor Core / Matrix Core 的对齐约束。
- **工具链**：CUDA C++、Triton、CUTLASS / CuTe，编译选项、autotune 和边界 shape 的处理。
- **性能分析**：Nsight Compute 指标怎么读，用 roofline 判断瓶颈在带宽还是算力，而不是只看总耗时下结论。

## 写作约定

每篇尽量交代硬件型号与架构（如 `sm_90`）、CUDA / 驱动 / 框架版本、输入 shape 与 dtype、编译命令；roofline 上的理论上界和实测数字分开写，差得远就解释原因。kernel 代码给出可运行的最小版本，并说明用什么做了对拍；没有实测支撑的部分会明确标成推断。

## 怎么读

分类页按发布时间倒序排列，侧栏的 Kernel 入口会随文章增加自动更新。目前只有这篇导言，后续每篇独立成文，按主题平铺或分子目录都可以。
