---
title: "Blackwell 与 Rubin：双 die、FP4 与 10.x 家族的分裂"
published: 2026-10-31T09:40:00+08:00
description: Blackwell 在 CUDA 里不是一块芯片，而是 10.0、10.3、11.0、12.0、12.1 五个 CC：数据中心双 die、Thor 的边缘 SoC、消费级 GB20x 和 GB10 各走各的 SM 版本。Rubin 则留在 10.7，用 328KB shared 和 HBM4 把 10.x 家族继续往前推。
tags: [CUDA, GPU, NVIDIA, Blackwell, Rubin]
category: GPU Architecture
draft: false
---

Blackwell 让同一个市场名字横跨三个 CC 大版本：数据中心的 B200 是 10.0，B300 是 10.3，Jetson Thor 是 11.0，RTX 5090 是 12.0，DGX Spark 的 GB10 是 12.1。到了 Rubin，编号又回到 10.7，而不是继续往 11 或 13 走。

这不是命名混乱，而是 CUDA 的兼容性边界在起作用：major 决定 cubin 能不能跨卡加载，minor 决定同一家族里的增量。Blackwell 的五个 CC 对应五种 SM 版本，Rubin 的 10.7 则被放进 10.x 家族。理解这一点，比记住“Blackwell 是第几代”重要得多。

## 1. Blackwell 数据中心：10.0 和 10.3

GB100 是 Blackwell 的数据中心 die：台积电 4NP、单 die 1040 亿晶体管，两个 die 封在一起到 2080 亿，18432 个 FP32 core、144 个 SM、576 个第五代 Tensor Core、60MB L2。B200 配 192GB HBM3e、8TB/s，NVLink 5 到 1.8TB/s；B300/GB300（CC 10.3）把 HBM 提到 288GB，FP4 吞吐约 1.5 倍，FP64 则被压到 64:1。

| CC | 产品 | SM 数 | FP32/SM | Tensor/SM | Shared/SM | 线程/SM | 显存 | 代表 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 10.0 | B200、GB200 | 144 | 128 | 4 | 228KB | 2048 | HBM3e 192GB | B200 |
| 10.3 | B300、GB300 | — | 128 | 4 | 228KB | 2048 | HBM3e 288GB | B300 |
| 10.7 | Rubin | — | — | — | 328KB | 1024 | HBM4 | Vera Rubin |
| 11.0 | Jetson T5000/T4000 | — | — | — | 228KB | 1536 | LPDDR5X 128GB | Jetson Thor |
| 12.0 | RTX 5090、RTX PRO 6000 | 192（GB202） | 128 | 4 | 100KB | 1536 | GDDR7 32GB | RTX 5090 |
| 12.1 | GB10、RTX Spark N1X | 48 | 128 | 4 | 100KB | 1536 | LPDDR5X 128GB | DGX Spark |

Blackwell 数据中心 SM 的核心变化是 **TMEM（Tensor Memory）和 `tcgen05`**。Volta 到 Hopper 的 Tensor Core 用寄存器和 shared memory 做操作数与累加器；到了 Blackwell，每个 SM 多了一块 256KB 的 TMEM，`tcgen05` 的 MMA 可以把累加器放在 TMEM 里，而不是挤在寄存器文件里。这样做的好处是 tile 可以更大、MMA 可以更异步，代价是编程模型变了：你要显式分配 TMEM、管理 TMEM 的读写，不能再用传统的寄存器 accumulator 思路。

Blackwell 还支持 **CTA pair**：两个 CTA 可以配对，共享一次 MMA 的 work。它和 Hopper 的 thread block cluster 不是一回事：cluster 是调度和 DSMEM 的概念，CTA pair 更贴近 Tensor Core 的算子粒度。对写 CUTLASS/DeepGEMM 这类库的人来说，CTA pair 和 TMEM 是 Blackwell 后端真正需要重写的地方。

第五代 Tensor Core 增加 FP4/FP6 输入，配合第二代 Transformer Engine 做 MXFP4/MXFP6 的 microscaling。FP4 的动态范围很窄，必须靠 per-block scale；它适合推理和部分训练，不适合直接替换所有 FP16/BF16 计算。

## 2. Thor：为什么是 11.0

Jetson T5000/T4000（Jetson AGX Thor）用的是 Blackwell 架构 GPU：2560 个 CUDA core、96 个 Tensor Core、128GB LPDDR5X、NVIDIA 标称 2070 TFLOPS FP4。CUDA 把它放在 CC 11.0，规格是 1536 线程/SM、228KB shared、FP4/FP6/FP8 Tensor Core、没有 FP64 Tensor Core。

官方没有解释 Thor 为什么单独占一个 major。能确认的只有：

- 产品页说它是 Blackwell 架构 GPU。
- CUDA 列表把它放在 11.0，`compute_110f` 只覆盖 11.0 自己。
- 它的 1536 线程/SM + 228KB shared 既不像 10.0/10.3 的 2048 线程，也不像 12.x 的 100KB shared。

所以它不是一个“Blackwell 移动版”的简单标签，而是 CUDA 里的一个独立 SM 版本。对开发者来说，给 Thor 编译要写 `sm_110`，不能拿 `sm_100` 或 `sm_120` 的 cubin 直接跑。

## 3. 消费级 12.x：GB202 和 GB10

消费级 Blackwell 是另一个家族。GB202（CC 12.0）有 192 个 SM、24576 个 FP32 core、768 个 Tensor Core、192 个 RT Core、32GB GDDR7、1792GB/s；RTX 5090 屏蔽到 170 个 SM、21760 核。它的 SM 仍然是 128 FP32/SM、4 个 Tensor Core、1 个 RT Core，但 shared 只有 100KB、线程上限 1536，和 10.x 数据中心的 228KB/2048 是两种配比。

GB10（CC 12.1）是 DGX Spark 用的 SoC：6144 个 CUDA core、48 个 SM、192 个 Tensor Core、48 个 RT Core、128GB LPDDR5X、273GB/s，NVIDIA 标称 1 PFLOPS FP4。它的 SM 和 GB202 同属 12.x 家族，`compute_120f` 覆盖 12.0/12.1。

12.x 和 10.x 之间没有二进制兼容性：一个 `sm_100` 的 cubin 不能加载到 RTX 5090，反过来也不行。两者都叫 Blackwell，但 SM 版本不同，Tensor Core 的指令、shared 容量、线程上限都不一样。RTX PRO 6000 Blackwell Server Edition 虽然是数据中心产品，也在 12.0 这一档——这再次说明 CC 不是按市场分的。

## 4. Rubin：10.7 的回归

Rubin 是 Blackwell 的下一代数据中心架构：台积电 3nm、HBM4、NVLink 6，NVIDIA 标称 50 PFLOPS FP4（稀疏），配 Vera CPU。它的 CC 是 **10.7**，不是 11.x，也不是 13.x。

从兼容性角度看，这个选择很直接：10.7 留在 10.x 家族里，`compute_100f` 覆盖 10.0、10.3、10.7，给 Blackwell 数据中心写的 family-specific 代码可以继续在 Rubin 上跑；如果换成 11.x，它就和 10.0/10.3 分属两个 major，普通 cubin 不能互通。NVIDIA 没有解释编号选择，但这是最直接的结果。

Rubin 的 SM 规格有几个反直觉的点：

- **328KB shared/SM**，比 10.0/10.3 的 228KB 多了 100KB；统一缓存到 336KB。
- **1024 线程/SM、32 warp/SM**，比 10.0/10.3 的 2048/64 少了一半。
- **FP32:FP64 = 4:1**，比 B300 的 64:1 更重视双精度。
- 官方 Tensor Core 输入表里，10.7 列出的类型是 FP64/TF32/BF16/FP16/FP8/FP6/FP4，没有列 INT8/INT4。

“更少线程 + 更大 shared”说明 Rubin 的 SM 不是把 10.0 简单放大，而是针对 Tensor Core 密集、数据复用高的负载重新配比：shared 变大，tile 可以更大；线程变少，寄存器压力和调度开销下降。这个取舍对 GEMM/Attention 友好，对需要大量独立 warp 的通用 kernel 不一定友好。具体到某个算子，还是要看 occupancy 和 shared 的实际占用。

Rubin Ultra（2027）和 Feynman（2028）还没有官方 CC，等 NVIDIA 公布后再补进版本表。

## 5. 10.x、11.0、12.x 怎么记

把 Blackwell 和 Rubin 放在一张表里：

| 家族 | CC | 产品 | 编译目标 | 关键规格 |
| :--- | :--- | :--- | :--- | :--- |
| 10.x 数据中心 | 10.0 | B200、GB200 | `sm_100` / `sm_100f` | 228KB shared、2048 线程、TMEM |
| 10.x 数据中心 | 10.3 | B300、GB300 | `sm_103` / `sm_100f` | 228KB shared、2048 线程、FP4 1.5× |
| 10.x 数据中心 | 10.7 | Rubin | `sm_107` / `sm_100f` | 328KB shared、1024 线程、HBM4 |
| 11.0 边缘 | 11.0 | Jetson Thor | `sm_110` / `sm_110f` | 228KB shared、1536 线程 |
| 12.x 消费级 | 12.0 | RTX 5090、RTX PRO 6000 | `sm_120` / `sm_120f` | 100KB shared、1536 线程、GDDR7 |
| 12.x 消费级 | 12.1 | GB10、RTX Spark | `sm_121` / `sm_120f` | 100KB shared、1536 线程、LPDDR5X |

记忆方式：

- **10.x 是数据中心主线**，从 B200 到 Rubin，靠 `compute_100f` 串起来；10.7 是这条线的最后一个小版本（至少目前是）。
- **11.0 是 Thor 的独立分支**，目前只有它自己，`compute_110f` 也只覆盖它。
- **12.x 是消费级/SoC 分支**，GB202 和 GB10 共享 `compute_120f`，但和 10.x 不互通。

这也是为什么“为 Blackwell 编译”没有意义：B200 是 `sm_100`，Thor 是 `sm_110`，RTX 5090 是 `sm_120`。三者都叫 Blackwell，但没有一个共同的 cubin。

## 6. 这一代留下了什么

Blackwell 把 Tensor Core 从“寄存器里的 MMA”推到“TMEM + 异步 MMA”，把 FP4/FP6 做成一等精度；Rubin 则用更大的 shared、更少的线程和 HBM4，继续往 Tensor Core 密集的方向走。与此同时，CUDA 的 CC 编号彻底和市场名字脱钩：同一个 Blackwell 横跨 10/11/12，Rubin 回到 10.7。

对写 kernel 的人来说，这一代最重要的不是记住哪个产品叫什么，而是先确认三件事：目标 CC 是 10.x、11.x 还是 12.x；Tensor Core 走的是哪条指令路径（`tcgen05` 还是其他 MMA 形态）；shared/TMEM 的预算够不够放下你的 tile。这三点确定之后，再谈 tile 大小、流水线和 occupancy。

## 参考

- [NVIDIA CUDA GPU Compute Capability](https://developer.nvidia.com/cuda/gpus)
- [CUDA Programming Guide: Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
- [CUDA Toolkit 13.4 Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)
- [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
- [NVIDIA Jetson Thor](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/)
- [Blackwell microarchitecture（Wikipedia）](https://en.wikipedia.org/wiki/Blackwell_(microarchitecture))
