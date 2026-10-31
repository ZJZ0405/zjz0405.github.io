---
title: "Kepler 与 Maxwell：192 核 SMX 和 128 核 SMM 的路线分叉"
published: 2026-10-31T09:10:00+08:00
description: Kepler 把 SM 扩到 192 核，用 SMX 和 Hyper-Q 冲吞吐；Maxwell 把它拆成 4×32 的 SMM，用能效继续往前走。两代放在一起，能看清 SM 设计里“规模”和“效率”的取舍。
tags: [CUDA, GPU, NVIDIA, Kepler, Maxwell]
category: GPU Architecture
draft: false
---

Fermi 把 SM 定成了“32 核 + 双调度器 + 缓存”的形状，Kepler 和 Maxwell 接着往两个相反的方向走：Kepler 把 SM 做成 192 核的 SMX，用更大的共享执行单元池冲吞吐；Maxwell 把 SM 拆成 4 个 32 核的 SMM，每个调度器只盯自己那一组，用能效换规模。

这两代经常被一句“Kepler 到 Maxwell 提升了能效”带过，但真正的变化在 SM 内部：调度器和执行单元从“共享池”变成了“私有组”，这直接改变了写 kernel 时对 ILP、block 大小和 occupancy 的选择。

## 1. Kepler：把 SM 做成 192 核的 SMX

GK104（CC 3.0）是 Kepler 的第一颗消费级芯片：28nm、35.4 亿晶体管、1536 个 CUDA core、8 个 SMX，每个 SMX 192 核。GK110（CC 3.5）是数据中心版本：71 亿晶体管、2880 核、15 个 SMX，FP64 保留到 FP32 的 1/3，配 1.5MB L2 和 384-bit GDDR5。

| CC | 芯片 | SM 类型 | SM 数 | 核/SM | 总核 | Shared/SM | 寄存器/SM | 代表产品 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 3.0 | GK104 | SMX | 8 | 192 | 1536 | 48KB | 64K | GTX 680、Tesla K10 |
| 3.2 | GK20A | SMX | 1 | 192 | 192 | 48KB | 64K | Tegra K1、Jetson TK1 |
| 3.5 | GK110 | SMX | 15 | 192 | 2880 | 48KB | 64K | K20X、K40、GTX 780 |
| 3.7 | GK210 | SMX | 2×13 | 192 | 4992 | 112KB | 128K | Tesla K80 |
| 5.0 | GM107 | SMM | 5 | 128 | 640 | 64KB | 64K | GTX 750 Ti |
| 5.2 | GM204 | SMM | 16 | 128 | 2048 | 96KB | 64K | GTX 980、Tesla M40 |
| 5.3 | GM20B | SMM | 2 | 128 | 256 | 64KB | 64K | Tegra X1、Jetson Nano |

SMX 的内部结构是“共享池”：192 个 FP32 core、32 个 SFU、32 个 LD/ST 单元，由 4 个 warp scheduler 和 8 个 dispatch unit 共同喂。每个 scheduler 每个时钟选一个 warp，但发射出去的指令可以打到共享池里的任意执行单元。好处是大块执行单元可以灵活分配给不同 warp；代价是 SMX 内部需要一套 crossbar，把 4 个 scheduler 接到 192 个 core 上，面积和功耗都不低。

Kepler 这一代真正的软件特性，大多集中在 GK110（3.5）上：

- **Dynamic Parallelism**：device 端可以直接 launch 子 kernel，不用回 host。对递归、自适应网格、图遍历这类“运行时才知道要开多少并行”的算法有用。
- **Hyper-Q**：32 个硬件工作队列，允许多个 stream、多个 MPI rank 同时往 GPU 上发 kernel，减少假依赖造成的串行化。Fermi 只有一个队列，MPI 进程多了容易互相堵。
- **GPUDirect RDMA**：网卡可以直接读写 GPU 显存，跨节点通信少一次 host 内存拷贝。
- **Warp shuffle**：`__shfl` 让同一个 warp 的线程直接交换寄存器，不再经过 shared memory。做 reduction、scan、转置时能省掉一轮 shared 读写。
- **只读数据缓存**：`__ldg` 把只读的全局数据放进 texture/read-only 路径，减轻 L1 压力。

GK104 和 GK110 的差别不只是规模：GK104 的 FP64 被砍到 1/24，GK110 保留 1/3；GK104 配 2GB GDDR5、256-bit，GK110 配 6GB、384-bit。CC 3.0 和 3.5 因此是两个不同的编译目标，虽然都叫 Kepler。

GK210（CC 3.7）是 K80 用的双芯版本：每个 GK210 有 13 个 SMX、2496 核，两个芯片封在一起凑 4992 核、24GB 显存。它把 shared memory 提到 112KB、寄存器提到 128K/SM，是 Kepler 里资源最宽的一档，但架构本身没有大改。

## 2. Maxwell：把 SMX 拆成 4×32 的 SMM

Maxwell 的第一颗芯片 GM107（CC 5.0）没有急着堆规模：640 个 CUDA core、5 个 SMM、128 核/SM，配 2MB L2 和 128-bit GDDR5。它的重点是每瓦性能，而不是绝对算力。GM204（CC 5.2）把规模拉到 2048 核、16 个 SMM，配 256-bit GDDR5 和 96KB shared；GM200 再到 3072 核、24 个 SMM、3MB L2、384-bit。

SMM 的变化是结构性的：

- **每个 scheduler 私有 32 个 FP32 core、8 个 LD/ST、8 个 SFU**。Kepler 的 4 个 scheduler 共享 192 个 core，Maxwell 的 4 个 scheduler 各管 32 个 core，不再需要 SMX 那块大 crossbar。
- **128 核 SMM 的性能大约是 192 核 SMX 的 90%，能效约为 2 倍**（NVIDIA 在 GM107 发布时的说法）。这不是靠制程，而是靠调度器和执行单元的局部化，减少了不必要的共享和唤醒。
- **shared memory 从 48KB 提到 64KB/96KB**，block 上限从 16 提到 32。同样 2048 线程/SM，Maxwell 可以挂更多小 block，调度更细。
- **原生 shared memory 原子操作**：32-bit 整数 atomic、32/64-bit CAS 在硬件里做，不再用 lock/update/unlock 的模拟路径。对 histogram、桶排序、稀疏更新这类 kernel 是实打实的提升。
- **FP64 基本退出消费级**：GM200 的 FP64 只有 FP32 的 1/32，GM107 更低。Maxwell 是明确的 FP32 架构，双精度留给 Kepler 的 K80 和后来的 P100。

Maxwell 的 5.0/5.2/5.3 也不是同一档：GM107 是 64KB shared、128-bit 显存；GM204 是 96KB、256-bit；GM20B（Tegra X1）是 64KB、64-bit LPDDR4。Jetson Nano 后来用 GM20B 的 128 核版本，CC 5.3，成了很多人接触 CUDA 的第一块板子。

## 3. 路线分叉：共享池 vs 私有组

把 SMX 和 SMM 放在一起看，差别可以归结为一句话：**Kepler 让 4 个 scheduler 共享 192 个执行单元，Maxwell 让每个 scheduler 私有 32 个。**

这带来两个直接后果：

- **Kepler 更依赖 ILP**。一个 scheduler 手里的 warp 如果不够多，或者指令之间依赖太长，共享池里总有执行单元闲着。写 Kepler kernel 时，展开循环、增加独立指令、把长依赖拆开，往往比单纯加 block 更有效。
- **Maxwell 更依赖 TLP 和 occupancy**。每个 scheduler 只盯自己的 32 个 core，warp 够多就能把私有组填满；128 线程的 block 刚好对应 4 个 scheduler，调度更自然。Maxwell 的 32 block/SM 上限也是为这种细粒度调度准备的。

这也解释了为什么同一份 GEMM 代码在 Kepler 和 Maxwell 上的最佳 tile 不一样：Kepler 的 shared 只有 48KB，但 SMX 的 192 核需要更大的 tile 才能喂饱；Maxwell 的 shared 到 96KB，但每个 scheduler 只有 32 核，tile 可以更小、block 可以更多。拿一张 occupancy 表套两代，通常会得到错误的结论。

Kepler 的 3.0 和 3.5 也要分开看：Dynamic Parallelism、Hyper-Q、GPUDirect RDMA 都是 3.5 才有的，3.0 只有 shuffle 和更大的 SMX。Maxwell 则把 3.5 的这些特性整体继承下来，所以 5.x 的编程模型比 3.0 更接近现代 CUDA。

## 4. 这一代留下了什么

Kepler 把 SM 推到 192 核，证明了“大 SM + 共享执行单元”可以做到很高的 FP32 吞吐，但也暴露了 crossbar 的功耗和调度复杂度。Maxwell 反过来把 SM 拆小，用私有组和更高的 shared/block 上限换能效，成了后面 Pascal GP104、Turing、Ampere GA10x 的模板。

4.x 没有出现，3.7 之后直接跳到 5.0。官方没有解释原因，也没有对应产品；从产品线看，Maxwell 就是 Kepler 的下一站。

下一站是 Pascal、Volta 和 Turing：Pascal 把 FP16、HBM2、NVLink 带进来，Volta 加上 Tensor Core 和独立线程调度，Turing 再把 RT Core 和 INT8/INT4 推理塞进 SM。SM 从这一代开始，不再只是“更多 FP32 核”的竞赛。

## 参考

- [NVIDIA Kepler GK110 Compute Architecture Datasheet](https://www.nvidia.cn/content/dam/en-zz/Solutions/Data-Center/documents/NV-DS-Tesla-KCompute-Arch-May-2012-LR.pdf)
- [CUDA 8.0 Programming Guide（归档，2.x–6.x 规格）](https://docs.nvidia.com/cuda/archive/8.0/cuda-c-programming-guide/index.html)
- [NVIDIA CUDA GPU Compute Capability](https://developer.nvidia.com/cuda/gpus)
- [Maxwell microarchitecture（Wikipedia）](https://en.wikipedia.org/wiki/Maxwell_(microarchitecture))
