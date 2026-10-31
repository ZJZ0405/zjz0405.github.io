---
title: "Ampere、Ada 与 Hopper：TF32、FP8 与异步流水线"
published: 2026-10-31T09:30:00+08:00
description: Ampere 把 8.x 拆成数据中心 GA100 和消费级 GA10x，加入 TF32/BF16、2:4 稀疏、MIG 和 cp.async；Ada 用 FP8 和 72MB L2 做 inference；Hopper 用 TMA、cluster 和 FP8 Transformer Engine 把异步流水线做进 SM。
tags: [CUDA, GPU, NVIDIA, Ampere, Ada, Hopper]
category: GPU Architecture
draft: false
---

2020 到 2022 年，NVIDIA 的 8.x 和 9.0 把“SM 里只有 FP32/INT32/Tensor Core”这件事拆得更细：Ampere 在数据中心和消费级用了两种不同的 SM，Ada 在 8.9 里加 FP8，Hopper 则把 TMA、thread block cluster 和 distributed shared memory 做进 9.0。到这一步，写 kernel 的难点已经从“怎么把 FP32 流水喂满”变成“怎么把数据搬运、矩阵计算和同步组织成一条异步流水线”。

这一代还有一个容易被忽略的点：**8.0、8.6、8.9 虽然同属 8.x，但不是同一种 SM**。A100 的 GA100 是 64 FP32 + 64 INT32、164KB shared、2048 线程；RTX 3090 的 GA102 是 128 FP32 + 64 INT32、100KB shared、1536 线程；RTX 4090 的 AD102 又是另一套 4N 工艺、72MB L2 的设计。拿一张 8.x 的 occupancy 表套所有卡，一定会出错。

## 1. Ampere：GA100 和 GA10x 是两种 SM

Ampere 先发的是数据中心 GA100（CC 8.0）：台积电 7nm、542 亿晶体管、6912 个 FP32 core、108 个 SM（满配 128 个），每个 SM 64 个 FP32 core + 64 个 INT32 core、4 个第三代 Tensor Core、164KB shared、64K 寄存器、2048 线程/SM、40MB L2。A100 配 40GB 或 80GB HBM2e，带宽 1.55–2.0TB/s，NVLink 3 到 600GB/s。

消费级 GA10x（CC 8.6）是另一种配比：GA102 有 10752 个 FP32 core、84 个 SM，每个 SM 128 个 FP32 core + 64 个 INT32 core、4 个 Tensor Core、1 个第二代 RT Core、100KB shared、1536 线程/SM、6MB L2，配 384-bit GDDR6X。RTX 3090 是 82 个 SM、10496 核、24GB、936GB/s。

| CC | 芯片 | SM 数 | FP32/SM | INT32/SM | Tensor/SM | RT/SM | Shared/SM | 线程/SM | L2 | 代表产品 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 8.0 | GA100 | 108 | 64 | 64 | 4 | 0 | 164KB | 2048 | 40MB | A100、A30 |
| 8.6 | GA102 | 84 | 128 | 64 | 4 | 1 | 100KB | 1536 | 6MB | RTX 3090、A40 |
| 8.7 | GA10B | 16 | 128 | 64 | 4 | 0 | 164KB | 1536 | — | Jetson AGX Orin |
| 8.9 | AD102 | 144 | 128 | 64 | 4 | 1 | 100KB | 1536 | 72MB | RTX 4090、L40S |
| 9.0 | GH100 | 132 | 128 | 64 | 4 | 0 | 228KB | 2048 | 50MB | H100、H200 |

Ampere 的 SM 层面有三组新东西：

**TF32 和 BF16。** TF32 用 8 位指数 + 10 位尾数，动态范围和 FP32 一样，精度介于 FP16 和 FP32 之间；Tensor Core 用 TF32 做输入、FP32 做累加，GEMM 不用改数据格式就能比 FP32 CUDA core 快一截。BF16 用 8 位指数 + 7 位尾数，范围和 FP32 一样，尾数更短，适合训练里对精度不那么敏感的矩阵乘。两者都是“用范围换精度”，但换法不同。

**2:4 结构化稀疏。** 每 4 个元素里最多 2 个非零，Tensor Core 可以跳过零，理论吞吐翻倍。它不是任意稀疏：需要把权重压成 2:4 模式，再配一份 metadata。对剪枝后的模型，这是实打实的收益；对稠密模型，得先做剪枝和重训练。

**MIG 和 `cp.async`。** MIG 把 A100 切成最多 7 个 GPU instance，每个 instance 有独立的 SM、L2 和显存切片，适合多租户。`cp.async` 则把 global → shared 的拷贝从寄存器路径里拿出来，直接异步写 shared，配合 `cp.async.commit_group` / `wait_group` 可以做多级流水线。

```cpp
#include <cuda_pipeline.h>

// 16B per thread，global -> shared，不经过寄存器
__pipeline_memcpy_async(&smem[tid], &gmem[tid], sizeof(float4));
__pipeline_commit();
// ... 先算上一 stage ...
__pipeline_wait_prior(0);
```

在 Ampere 之前，global → shared 通常要“load 到寄存器，再 store 到 shared”，中间占寄存器、也占指令槽。`cp.async` 把这两步合成一条异步拷贝，是 Ampere 之后 pipeline 写法的基础。Hopper 的 TMA 可以看成它的进化版：把多维 tensor 的地址生成也交给硬件。

## 2. Ada：8.9 不是“Ampere  refresh”

Ada 的 AD102 用台积电 4N、763 亿晶体管、18432 个 FP32 core、144 个 SM，每个 SM 128 FP32 + 64 INT32、4 个第四代 Tensor Core、1 个第三代 RT Core、100KB shared、1536 线程/SM、72MB L2。RTX 4090 是 128 个 SM、16384 核、24GB GDDR6X、1008GB/s。

Ada 相对 Ampere 消费级的三个关键变化：

- **FP8 Tensor Core**：第四代 Tensor Core 增加 FP8（E4M3/E5M2），吞吐是 FP16 的 2 倍。对推理来说，FP8 比 INT8 更容易做 scale，精度也更容易控制。
- **72MB L2**：AD102 的 L2 从 GA102 的 6MB 跳到 72MB，主要用来兜住推理和光追的重复访问。L2 大了之后，很多 kernel 的瓶颈从 DRAM 带宽转到 L2 带宽和命中率。
- **第三代 RT Core + DLSS 3**：RT Core 增加动态几何、微网格等能力；DLSS 3 的帧生成则依赖光流加速器和 Tensor Core。对 compute kernel 来说，RT Core 仍然不是通用算力，但它把光追从“能跑”推到“实时”。

Ada 的 CC 是 8.9，和 Ampere 同属 8.x：从 8.0 编译的普通 cubin 可以向上加载到 8.9，反过来不行。FP8 指令是 8.9 才有的，`__CUDA_ARCH__ >= 890` 的 guard 不能省。

## 3. Hopper：TMA、Cluster 和 FP8 Transformer Engine

GH100 是 Hopper 的大芯片：台积电 4N、800 亿晶体管、16896 个 FP32 core、132 个 SM。每个 SM 128 FP32 + 64 INT32、4 个第四代 Tensor Core、256KB 寄存器、228KB shared、2048 线程/SM、64 warp/SM。H100 配 80GB HBM3、3.35TB/s，NVLink 4 到 900GB/s，L2 50MB；H200 把显存提到 141GB HBM3e、4.8TB/s。

Hopper 的 SM 里多了三样东西：

**TMA（Tensor Memory Accelerator）。** 一个专门的拷贝单元，按 tensor 描述符把多维数据从 global 搬到 shared，地址生成、边界处理、swizzle 都由硬件做。TMA 用 mbarrier 做完成通知，可以和计算重叠；它把线程从“算地址、发 load”里解放出来，特别适合 GEMM/Attention 这种规则 tile。

**Thread Block Cluster 和 DSMEM。** 一个 cluster 里的多个 CTA 可以被一起调度到同一个 GPC，CTA 之间可以直接访问对方的 shared memory（distributed shared memory）。这让“一个 CTA 算不完的 tile”可以拆到多个 CTA，再用 cluster 级同步拼起来。cluster 的可移植上限是 8 个 CTA，部分配置可以开到 16 个。

**FP8 Transformer Engine。** Hopper 的第四代 Tensor Core 支持 FP8（E4M3/E5M2），Transformer Engine 在软件层做 per-tensor 的动态 scaling，尽量把 FP8 的精度损失压住。H100 的 FP8 峰值是 FP16 的 2 倍，训练和推理都能用；但 FP8 的数值范围窄，scale 选不好会溢出或下溢，实际项目里通常要逐层校准。

Hopper 也有 DPX 指令，专门加速动态规划（比如 Smith-Waterman、编辑距离），这类 workload 在之前的 GPU 上只能靠多条指令拼。它不是 SM 的主流水线，但说明 NVIDIA 开始把“特定算法模式”做成指令。

## 4. 这一代对写 kernel 的影响

- **先看 CC，再看架构名**。8.0 是 64 FP32 + 64 INT32、2048 线程；8.6/8.9 是 128 FP32、1536 线程；9.0 又回到 2048 线程、228KB shared。block 大小和 occupancy 不能跨代套。
- **`cp.async` 是 Ampere 的标配，TMA 是 Hopper 的标配**。Ampere 的 pipeline 还是“线程发 async copy + commit/wait”；Hopper 把整块 tile 的拷贝交给 TMA，线程只负责 mbarrier 和计算。
- **FP8 要单独做数值设计**。8.9 和 9.0 都有 FP8，但 Tensor Core 的布局、scale 方式和吞吐不一样；不要以为“同一段 FP8 代码”能同时在 Ada 和 Hopper 上跑到峰值。
- **cluster 不是免费的**。DSMEM 和 cluster sync 能减少 global 往返，但 cluster 内的 CTA 必须一起调度，occupancy 和 tail effect 都会变；小 kernel 用 cluster 往往得不偿失。

这一代也是“数据中心和消费级分叉”最明显的一代：A100/H100 走 HBM + NVLink，RTX 3090/4090 走 GDDR6X + 大 L2，Jetson Orin 走 LPDDR5 + 低功耗。同样是 8.x，GA100 和 GA102 的 SM 结构就不一样；同样是 Hopper，H100 和 H200 的算力相同、显存不同。

下一站是 Blackwell 和 Rubin：双 die、FP4/FP6、TMEM、`tcgen05`，以及 10.x/11.0/12.x 三个大版本的分裂。

## 参考

- [NVIDIA A100 Tensor Core GPU Architecture Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)
- [CUDA Programming Guide: Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
- [CUDA Programming Guide: Asynchronous Data Copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)
- [NVIDIA CUDA GPU Compute Capability](https://developer.nvidia.com/cuda/gpus)
- [NVIDIA Hopper Architecture](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/)
