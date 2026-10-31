---
title: "Tesla 与 Fermi：从统一着色器到通用计算 SM"
published: 2026-10-31T09:00:00+08:00
description: 从 G80 的统一着色器到 GF100 的 32 核 SM：Tesla 怎么把 GPU 变成可编程处理器，Fermi 又怎么补上 L1/L2、ECC、并发 kernel 和 C++，为后面的 Kepler 打底。
tags: [CUDA, GPU, NVIDIA, Tesla, Fermi]
category: GPU Architecture
draft: false
---

2006 年的 G80 和 2010 年的 GF100 常被放在一起讲，因为它们分别解决了两个不同的问题：Tesla 解决“GPU 能不能按统一的方式编程”，Fermi 解决“能编程之后，能不能像一颗正经的计算核心那样工作”。

在 Tesla 之前，顶点着色器和像素着色器是分开的硬件；在 Fermi 之前，GPU 没有真正的 L1 数据缓存、没有 ECC、也不支持并发 kernel。这两代把 CUDA 从“能跑”推到“可以拿来做通用计算”，后面 Kepler 的 192 核 SMX 才有地方生长。

## 1. Tesla：把两套着色器合并成 SP

G80 是 NVIDIA 第一颗统一着色器（unified shader）GPU。它把原来分开的顶点、像素、几何处理合并成一组标量流处理器（stream processor，SP），每个 SP 一次处理一个标量分量，靠 warp 把 32 个线程捆在一起执行。这个模型一直用到今天。

G80 的 SM 很小：每个 SM 只有 8 个 SP、2 个 SFU、16KB shared memory，没有统一的 L1 数据缓存，也没有 L2。一个 SM 只有 1 个 warp scheduler，靠指令级并行和 shared memory 来掩盖延迟。8800 GTX 用 16 个 SM 凑出 128 个 SP，核心/着色器频率分别是 575MHz / 1350MHz，配 768MB GDDR3、384-bit 位宽、86.4GB/s 带宽。

| CC | 芯片 | SM 数 | SP 总数 | SP/SM | Shared/SM | L1/L2 | FP64 | 代表产品 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1.0 | G80 | 16 | 128 | 8 | 16KB | 无 / 无 | 无 | 8800 GTX、Tesla C870 |
| 1.1 | G92 | 16 | 128 | 8 | 16KB | 无 / 无 | 无 | 8800 GT、9800 GTX |
| 1.2 | G9x | 8–16 | 64–128 | 8 | 16KB | 无 / 无 | 无 | GT 240、GeForce 210 |
| 1.3 | GT200 | 30 | 240 | 8 | 16KB | 无 / 无 | 1/8 FP32 | GTX 280、Tesla C1060 |

Tesla 这一代真正留下来的不是某个具体数字，而是 CUDA 的执行模型：

- **grid / block / thread** 三层索引，warp 是最小调度单位，32 个线程共享一条指令流。
- **shared memory** 是软件管理的片上暂存，`__syncthreads()` 做块内同步。没有 L1 的年代，tiling 不是优化选项，而是把全局内存流量压下来的主要手段。
- **warp 分歧**直接暴露：同一个 warp 里如果线程走不同分支，只能串行执行。Tesla 没有独立线程调度，这个限制一直保留到 Volta。1.x 的 block 上限也只有 512 线程，Fermi 才提到 1024。
- **全局内存没有缓存**，只有 texture / constant 这类只读路径。做矩阵转置、卷积、GEMM 时，必须靠合并访存和 shared memory 把重复读取挡住。

一个典型的 Tesla 时代写法是 shared memory 转置：每个 block 读入 32×32 的 tile，`__syncthreads()` 之后转置写回。`tile[32][33]` 多出来的一列不是为了对齐，而是为了错开 bank，避免 32 个线程同时打到同一个 bank。这个技巧从 Tesla 一直用到今天。

```cpp
__global__ void transpose(const float* in, float* out, int n) {
    __shared__ float tile[32][33];
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    if (x < n && y < n) tile[threadIdx.y][threadIdx.x] = in[y * n + x];
    __syncthreads();
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    if (x < n && y < n) out[y * n + x] = tile[threadIdx.x][threadIdx.y];
}
```

Tesla 的最后一个版本 GT200（CC 1.3）把 SP 总数从 128 提到 240，晶体管从 6.81 亿涨到 14 亿，首次支持 FP64，但吞吐只有 FP32 的 1/8（每个 SM 每个时钟 1 个 FP64 FMA）。它更像“证明 GPU 能做双精度”，而不是“双精度好用”。真正把 FP64 做成可用算力的是 Fermi 的数据中心版本。

## 2. Fermi：SM 终于有了缓存和 ECC

GF100 是 Fermi 的第一颗大芯片：40nm、30 亿晶体管、512 个 CUDA core、16 个 SM，每个 SM 32 核。和 Tesla 的 8 核 SM 相比，这不是简单的数量增加，而是整个 SM 的组织方式变了。

| 项目 | Tesla GT200（1.3） | Fermi GF100（2.0） | Fermi GF104（2.1） |
| :--- | :--- | :--- | :--- |
| 工艺 / 晶体管 | 65nm / 14 亿 | 40nm / 30 亿 | 40nm / 19.5 亿 |
| CUDA core / SM | 8 | 32 | 48 |
| Warp scheduler / SM | 1 | 2 | 4 |
| Shared + L1 / SM | 16KB shared，无 L1 | 64KB 可配置（48+16 或 16+48） | 64KB 可配置 |
| 寄存器 / SM | 16K × 32-bit（1.0/1.1 为 8K） | 32K × 32-bit | 32K × 32-bit |
| 最大线程 / SM | 1024（1.0/1.1 为 768） | 1536 | 1536 |
| 最大 block | 512 | 1024 | 1024 |
| L2 | 无 | 768KB | 512KB |
| ECC | 无 | 有（专业卡） | 有（专业卡） |
| 并发 kernel | 无 | 最多 16 个 grid | 最多 16 个 grid |
| FP64 | 1/8 FP32 | 1/2 FP32（数据中心） | 大幅削减 |

Fermi 的 SM 里有 32 个 CUDA core、4 个 SFU、16 个 LD/ST 单元、2 个 warp scheduler 和 2 个 dispatch unit。每个 scheduler 每个时钟选一个 warp，发射一条指令；两个 scheduler 可以同时发射，所以一个 SM 每时钟最多执行 2 个 warp 的指令。寄存器文件是 32K × 32-bit（128KB），和今天的 64K × 32-bit 比只有一半，但在 2010 年已经足够让每个线程拿到几十个寄存器。

更重要的是缓存层级：

- **64KB 的片上存储可以在 shared memory 和 L1 之间配置**。48KB shared + 16KB L1，或者 16KB shared + 48KB L1。这给了程序员一个显式取舍：shared 用得多的 kernel 拿大 shared，依赖全局加载复用的 kernel 拿大 L1。
- **768KB 的统一 L2** 把所有 SM 连在一起，第一次让 GPU 有了跨 SM 的片上复用。Tesla 时代全局内存读一次就是一次 DRAM 访问，Fermi 之后同一块数据可能已经在 L2 里。
- **ECC** 覆盖寄存器文件、L1/L2 和 DRAM。对 HPC 来说这比多几个 core 更重要：长时间跑的仿真不能因为一个 bit flip 重算。NVIDIA 把 ECC 做成 Fermi 数据中心的卖点，Tesla C2050 因此进了不少超算节点。
- **并发 kernel** 最多 16 个 grid 同时驻留。Tesla 一次只能跑一个 grid，小的 kernel 很难把 GPU 填满；Fermi 可以把不同 kernel 拼在一起，提高利用率。
- **64 位地址空间和 C++**。Tesla 的地址宽度是 32-bit，Fermi 扩到 64-bit；CUDA 也开始支持 C++ 的模板、类和一些运行时特性。这两点让 GPU 代码可以写得更像主机代码。

GF104（CC 2.1）是另一套配比：每个 SM 48 核、4 个 scheduler，8 个 SM 凑 384 核。48 不是 32 的整数倍，SM 内部按 4 组组织，每组 12 核；它和 GF100 的 32 核 SM 不是简单放大，而是为了在消费级 die 上平衡面积、频率和良率。GTX 460 用 7 个 SM（336 核）打市场，CC 2.1 因此成了一个独立的编译目标。

Fermi 的 FP64 也分档：GF100 的数据中心版本（Tesla C2050/C2070）保持 1/2 FP32，GeForce 版本被砍到 1/8 左右；GF104 更低。这个“同架构、不同 FP64 配比”的做法后来在 Pascal 上又出现了一次：GP100 是 1/2，GP104 是 1/32。

对写 kernel 的人来说，Fermi 带来的变化是具体的：

- **occupancy 开始有意义**。1536 线程 / 48 warp / 8 block 的上限，加上 32K 寄存器，决定了你能同时挂多少个 block。Tesla 时代更多是“一个 block 能不能跑满”，Fermi 开始要算“一个 SM 能挂几个 block”。
- **L1 让全局加载有了复用机会**。不写 shared 的 kernel 也能吃到缓存，但 L1 只有 16–48KB，和 shared 抢空间，cache config 变成调优参数。
- **ECC 和并发 kernel** 让 GPU 可以进生产集群，而不只是实验室。
- **C++ 支持**让模板和泛型进入 kernel；后来的 cuDNN、CUTLASS 能把 tile 和数据类型做成编译期参数，底层依赖的就是这一步。

## 3. 这一代留下了什么

Tesla 留下了执行模型：warp、shared memory、grid/block/thread、`__syncthreads()`。它证明了 GPU 可以用统一的标量 SP 跑通用计算，但受限于没有缓存、没有 ECC、没有并发 kernel。

Fermi 留下了 SM 的基本形状：32 个 CUDA core、双 scheduler、可配置的 L1/shared、L2、ECC、并发 kernel。它把 GPU 从“图形处理器顺便能算”变成“计算处理器也能画图”。Kepler 后面把 SM 从 32 核扩到 192 核，Maxwell 又把它收回 128 核，但“SM 是一个带缓存和调度器的计算块”这个定义，是 Fermi 定下来的。

下一站是 Kepler 和 Maxwell：一个把 SM 做成 192 核的大家伙，另一个把它拆成 4×32 的小块，用能效而不是规模继续往前走。

## 参考

- [NVIDIA Fermi Compute Architecture Whitepaper](https://www.nvidia.com/content/PDF/fermi_white_papers/NVIDIA_Fermi_Compute_Architecture_Whitepaper.pdf)
- [CUDA 8.0 Programming Guide（归档，2.x–6.x 规格）](https://docs.nvidia.com/cuda/archive/8.0/cuda-c-programming-guide/index.html)
- [NVIDIA CUDA GPU Compute Capability](https://developer.nvidia.com/cuda/gpus)
- [CUDA Programming Guide: Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
