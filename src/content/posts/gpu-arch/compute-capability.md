---
title: "CUDA Compute Capability 版本地图：从 1.0 到 12.1"
published: 2026-10-31T10:05:00+08:00
description: 按 NVIDIA 官方列表整理 CUDA compute capability 1.0–12.1：每一档对应哪代架构、SM 规模怎么变、cubin 与 a/f 编译目标怎么兼容，以及 Blackwell 为什么横跨 10/11/12 三个大版本。
tags: [CUDA, GPU, 计算能力, 兼容性]
category: GPU Architecture
draft: false
---

Compute capability（CC）常被当成 NVIDIA 的“第几代”编号，但它更像一个 ABI 版本号：它决定 cubin 能不能加载、哪些指令可用、一个 SM 能驻留多少线程。NVIDIA 的产品名和 CC 并不一一对应——Blackwell 横跨 10.0、10.3、11.0、12.0、12.1 五档，Rubin 落在 10.7，Jetson Thor 单独占 11.0。

这篇文章把官方版本表、兼容规则和每一代的 SM 改动整理成一张地图，作为 GPU Architecture 分类的总览。具体到某一代为什么这样设计，放到后面的分代篇展开。

## 1. CC 到底是什么

CC 是 `major.minor` 两个整数。`sm_100` 表示 CC 10.0，`compute_100` 是编译时的虚拟架构，`sm_100` 是实际 cubin 目标。设备代码里用 `__CUDA_ARCH__` 判断当前编译目标，主机代码则要在运行时查设备。

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

```cpp
int major = 0, minor = 0;
cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);
```

CUDA 13.4 Programming Guide 的说法是：compute capability 对应 SM 的版本号；不同 major 是不同 SM 版本，也是二进制兼容的边界。早期文档把 major 解释为“同一代核心架构”，但产品线早就打破了这个对应关系：7.x 横跨 Volta 和 Turing，8.x 横跨 Ampere 和 Ada，10.x 横跨 Blackwell 和 Rubin。

所以更准确的理解是：

- **major** 是兼容性大类，决定 cubin 能不能跨卡加载。
- **minor** 是同一大类里的增量版本，通常对应工艺、规格或指令集的小步改进。
- **CC 不等于市场代际**，也不等于“越新编号越大”。Rubin 是 10.7，发布时间却晚于 12.1 的 GB10。

## 2. 完整版本表

截至 2026-10-31，NVIDIA 官方列表覆盖 1.0 到 12.1。下表按数字排序，不是按发布时间排序：

| CC | 架构 | 代表产品 | 首发 | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| 1.0 | Tesla（G80） | GeForce 8800 GTX、Tesla C870 | 2006 | 统一着色器，CUDA 1.0 |
| 1.1 | Tesla（G92） | GeForce 8800 GT、9800 GTX | 2007 | 65nm 版本 |
| 1.2 | Tesla（G9x） | GeForce GT 240、GeForce 210 | 2008 | 55/40nm 小改 |
| 1.3 | Tesla（GT200） | GeForce GTX 280、Tesla C1060 | 2008 | 首次支持 FP64 |
| 2.0 | Fermi（GF100/GF110） | GTX 480、GTX 580、Tesla C2050 | 2010 | 32 核/SM、ECC、并发 kernel |
| 2.1 | Fermi（GF104/GF114） | GTX 460、GTX 560 Ti | 2010 | 48 核/SM |
| 3.0 | Kepler（GK104） | GTX 680、Tesla K10 | 2012 | 192 核/SM（SMX） |
| 3.2 | Kepler（GK20A） | Tegra K1、Jetson TK1 | 2014 | 移动版 |
| 3.5 | Kepler（GK110） | GTX 780、Tesla K20X/K40 | 2012 | Dynamic Parallelism、Hyper-Q |
| 3.7 | Kepler（GK210） | Tesla K80 | 2014 | 双芯、112KB shared |
| 5.0 | Maxwell（GM107） | GTX 750 Ti | 2014 | 128 核/SM（SMM） |
| 5.2 | Maxwell（GM204/GM200） | GTX 980、GTX 980 Ti、Tesla M40 | 2014 | 96KB shared |
| 5.3 | Maxwell（GM20B） | Tegra X1、Jetson Nano | 2015 | 移动版 |
| 6.0 | Pascal（GP100） | Tesla P100 | 2016 | 64 核/SM、HBM2、NVLink |
| 6.1 | Pascal（GP104/GP102） | GTX 1080、GTX 1080 Ti、Tesla P40 | 2016 | 128 核/SM |
| 6.2 | Pascal（GP10B） | Tegra X2、Jetson TX2 | 2016 | 移动版 |
| 7.0 | Volta（GV100） | V100、Titan V | 2017 | 64 FP32 + 64 INT32、Tensor Core |
| 7.2 | Volta（GV11B） | Jetson AGX Xavier | 2018 | 移动版 |
| 7.5 | Turing（TU102） | RTX 2080 Ti、T4 | 2018 | RT Core、INT8/INT4 Tensor Core |
| 8.0 | Ampere（GA100） | A100、A30 | 2020 | 64 FP32 + 64 INT32/SM、TF32、MIG |
| 8.6 | Ampere（GA10x） | RTX 3090、A40、A10 | 2020 | 消费/工作站版 |
| 8.7 | Ampere（GA10B） | Jetson AGX Orin | 2022 | 移动版 |
| 8.9 | Ada（AD10x） | RTX 4090、L40S、L4 | 2022 | FP8、第四代 Tensor Core |
| 9.0 | Hopper（GH100） | H100、H200、GH200 | 2022 | FP8 Transformer Engine、TMA |
| 10.0 | Blackwell（GB100） | B200、GB200 | 2024 | 双 die、FP4/FP6、TMEM |
| 10.3 | Blackwell Ultra（GB300） | B300、GB300 | 2025 | 更大 HBM、FP4 提升 |
| 10.7 | Rubin | Vera Rubin | 2026 | HBM4、328KB shared |
| 11.0 | Blackwell（Thor） | Jetson T5000/T4000 | 2025 | 2560 核、96 个 Tensor Core |
| 12.0 | Blackwell（GB20x） | RTX 5090、RTX PRO 6000 | 2025 | 消费级、GDDR7 |
| 12.1 | Blackwell（GB10） | DGX Spark、RTX Spark N1X | 2025 | 128GB LPDDR5X |

几个容易被忽略的点：

- **4.x 从未出现**：官方列表里 3.7 之后直接是 5.0。没有对应产品，也没有官方解释；不要把 4.x 当成“缺失的一代”去考据。
- **3.2/5.3/6.2/7.2/8.7 是 Tegra/Jetson 的移动版**，编号跟着同家族走，发布时间往往晚于桌面/数据中心产品。3.7 不是移动版，它是 K80 的双芯 GK210。
- **10.7 是 Rubin**，CUDA 13.4 release notes 已明确写出；它属于 10.x 家族，不是 11.x。
- **11.0 是 Jetson Thor**，不是“Hopper 的下一代”。产品页说它用的是 Blackwell 架构 GPU，CUDA 把它单独放在 11.0。

## 3. 编号不是时间顺序

把上表按发布时间重排，会发现 CC 的顺序和产品节奏并不一致：

- 2024：10.0（B200）
- 2025：12.0（RTX 50）→ 10.3（B300）→ 11.0（Thor）→ 12.1（GB10）
- 2026：10.7（Rubin）

12.0 比 10.3 更早，11.0 夹在 10.3 和 12.1 之间，10.7 最后才出现。12.1 的 GB10 在 2025 年上市，10.7 的 Rubin 到 2026 年才部署；如果只看编号，会误以为 10.7 比 12.1 老。

同样，3.2（Tegra K1，2014）比 3.5（GK110，2012）编号高，但发布时间更晚；5.3（Tegra X1，2015）比 5.2（GM204，2014）晚；6.2（Tegra X2，2016）比 6.1（GP104，2016）晚；7.2（Xavier，2018）比 7.0（Volta，2017）晚。移动版通常是在同家族桌面版之后补的 minor。

所以读 CC 的正确姿势是：**先看 major 的兼容边界，再看 minor 的具体规格，不要把它当发布时间轴。**

## 4. 兼容性：cubin、PTX 和 a/f 后缀

普通 cubin 的规则是：**同 major，且设备的 minor ≥ 编译目标的 minor**。

举几个例子：

- `sm_86` 的 cubin，按规则可以加载到 8.6、8.7、8.9 上，不能加载到 8.0 上。
- `sm_100` 的 cubin，按规则可以加载到 10.0、10.3、10.7 上，不能加载到 12.0 上。
- `sm_120` 的 cubin，按规则可以加载到 12.0、12.1 上，不能加载到 10.0 上。
- `sm_90` 只有 9.0 一个成员，没有“9.1 上也能跑”这回事。

“可以加载”不等于“跑得对”。同 major 的不同架构可能共享基线指令，但性能特征、缓存行为、Tensor Core 吞吐都不一样；跨卡部署还是要跑回归。

这条规则只覆盖 baseline 特性。从 9.0 开始，NVIDIA 把一部分特性标成 architecture-specific，用 `a` 后缀锁定精确 CC；从 10.0 开始，又把多个 CC 共享的特性标成 family-specific，用 `f` 后缀覆盖一个家族：

| 编译目标 | 兼容的 CC |
| :--- | :--- |
| `compute_100f` | 10.0、10.3、10.7 |
| `compute_103f` | 10.3、10.7 |
| `compute_107f` | 10.7 |
| `compute_110f` | 11.0 |
| `compute_120f` | 12.0、12.1 |
| `compute_121f` | 12.1 |

`a` 目标则是精确锁死：`compute_100a` 只跑 10.0，`compute_103a` 只跑 10.3，`compute_107a` 只跑 10.7，`compute_120a` 只跑 12.0，`compute_121a` 只跑 12.1。`a` 是 `f` 的超集，`f` 是 baseline 的超集；想要最高性能就用 `a`，想要一个家族通吃就用 `f`，想要最大兼容就用 baseline。

Hopper 9.0 目前没有 family 目标，只有 `compute_90a` 和 baseline `compute_90`。这不是遗漏，而是 family-specific 机制从 10.0 才开始。

PTX 与 JIT 是另一条路：fatbin 里带 PTX，驱动可以在更高 CC 上 JIT。但 `a` 目标没有前向兼容，PTX 里如果用了架构专属指令，到了新卡上可能直接编译失败；“能启动”也不代表“算得对”。生产环境建议按目标 CC 分别构建、分别测试，不要把 JIT 当成兼容性保证。

一个常见误区：在 B200 上用 `-arch=native` 构建，然后把二进制拿到 RTX 5090 上跑。两者都叫 Blackwell，但一个是 10.0，一个是 12.0，普通 cubin 直接加载失败；如果只带 PTX，可能能 JIT，但性能和数值行为都要重新验证。

## 5. Blackwell 为什么有五个 CC

官方列表里，Blackwell 产品散落在五档：

| CC | 产品 | 形态 |
| :--- | :--- | :--- |
| 10.0 | B200、GB200 | 数据中心双 die |
| 10.3 | B300、GB300 | Blackwell Ultra |
| 11.0 | Jetson T5000/T4000 | 边缘 SoC（Thor） |
| 12.0 | RTX 5090、RTX PRO 6000 Blackwell | 消费/工作站，GDDR7 |
| 12.1 | GB10、RTX Spark N1X | DGX Spark，LPDDR5X |

这不是按“数据中心 vs 消费级”分的：12.0 的 Data Center 列里就有 RTX PRO 6000 Blackwell Server Edition。真正的边界是 SM 版本：不同 major 的 cubin 不能互换，10.x 和 12.x 之间两个方向都不行。

官方没有解释 Thor 为什么单独占 11.0。能确认的只有：产品页说它是 Blackwell 架构 GPU，CUDA 列表把它放在 11.0，CUDA 13.4 的规格表给它 1536 threads/SM、228KB shared、FP4/FP6/FP8 Tensor Core。从规格看，它既不像 10.0/10.3 的 2048 threads/SM，也不像 12.x 的 100KB shared，而是一个独立组合。原因可能要等 NVIDIA 后续文档，本文不编。

对开发者来说，结论很直接：**“为 Blackwell 编译”是一句没有信息量的话**。要么写 `sm_100`/`sm_103`/`sm_107`，要么写 `sm_120`/`sm_121`；Thor 又是 `sm_110`。

## 6. 每一代一句话

这张表是后面五篇分代篇的索引。每一行只写“SM 层面改了什么”和“写 kernel 时能感受到什么”，细节留到分代篇。

| CC 家族 | 架构 | SM 关键变化 | 对 kernel 的影响 |
| :--- | :--- | :--- | :--- |
| 1.x | Tesla | 8 SP/SM、16KB shared、无统一 L1、warp 执行模型 | shared memory tiling 成为主要手段；warp 分歧直接暴露 |
| 2.x | Fermi | 32/48 核/SM、2/4 个 warp scheduler、64KB 可配置 L1/shared、ECC（专业卡） | occupancy 概念成型；L1 缓存、并发 kernel、C++ |
| 3.x | Kepler | SMX 192 核/SM、4 个 scheduler、shuffle；3.5 起 Dynamic Parallelism、Hyper-Q | 更依赖 ILP；warp 内交换；3.5 起 device 端 launch、多流并发 |
| 5.x | Maxwell | SMM 128 核/SM（4×32）、64/96KB shared、2× perf/W | 128/256 线程 block 更自然；5.2 的 shared 到 96KB |
| 6.x | Pascal | GP100 64 核/SM + FP16 2× + HBM2/NVLink；GP104 128 核/SM | 6.0 的半精度和 multi-GPU 进入常规优化 |
| 7.x | Volta / Turing | 64 FP32 + 64 INT32、独立线程调度、Tensor Core；Turing 加 RT Core | 混合精度 GEMM；per-thread 分支；光追 |
| 8.x | Ampere / Ada | 8.0 是 64 FP32 + 64 INT32；8.6/8.7/8.9 是 128 FP32；TF32/BF16、2:4 稀疏、`cp.async`；8.0 有 MIG，Ada 加 FP8 | 异步拷贝、TF32/FP8、稀疏 GEMM；8.0 可切 MIG |
| 9.0 | Hopper | 128 FP32/SM、FP8 Transformer Engine、TMA、Cluster/DSMEM、228KB shared | warp-specialized 流水线、TMA、分布式 shared |
| 10.x | Blackwell / Rubin | 128 FP32/SM、FP4/FP6、TMEM、`tcgen05`；数据中心双 die；Rubin 328KB shared | `tcgen05`/TMEM 编程；FP4 推理 |
| 11.0 | Thor | Blackwell GPU、2560 核、96 个 Tensor Core、128GB LPDDR5X | 边缘 FP4 推理；注意它是独立 CC |
| 12.x | Blackwell 消费级 | 128 FP32/SM、FP4、100KB shared；12.0 用 GDDR7 | 消费级 FP4；与 10.x 不兼容 |

## 7. SM 规格速查（7.5 之后）

7.5 之前的官方规格表在 CUDA 8.0 / 11.4 的归档文档里，口径和现在不完全一致，放到分代篇细讲。下表只列 7.5 到 12.x，数据来自 CUDA 13.4 Programming Guide 的 Table 30/31/33：

| CC | Shared/SM | 常驻线程/SM | 常驻 warp/SM | FP32:FP64 | Tensor Core 输入 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 7.5 | 64KB | 1024 | 32 | 32:1 | FP16、INT8、INT4 |
| 8.0 | 164KB | 2048 | 64 | 2:1 | FP64、TF32、BF16、FP16、INT8、INT4 |
| 8.6 | 100KB | 1536 | 48 | 64:1 | TF32、BF16、FP16、INT8、INT4 |
| 8.7 | 164KB | 1536 | 48 | 64:1 | TF32、BF16、FP16、INT8、INT4 |
| 8.9 | 100KB | 1536 | 48 | 64:1 | TF32、BF16、FP16、FP8、INT8、INT4 |
| 9.0 | 228KB | 2048 | 64 | 2:1 | FP64、TF32、BF16、FP16、FP8、INT8 |
| 10.0 | 228KB | 2048 | 64 | 2:1 | FP64、TF32、BF16、FP16、FP8、FP6、FP4、INT8 |
| 10.3 | 228KB | 2048 | 64 | 64:1 | TF32、BF16、FP16、FP8、FP6、FP4、INT8 |
| 10.7 | 328KB | 1024 | 32 | 4:1 | FP64、TF32、BF16、FP16、FP8、FP6、FP4 |
| 11.0 | 228KB | 1536 | 48 | 64:1 | TF32、BF16、FP16、FP8、FP6、FP4、INT8 |
| 12.x | 100KB | 1536 | 48 | 64:1 | TF32、BF16、FP16、FP8、FP6、FP4、INT8 |

几个反直觉的点：

- **Turing 的 1024 threads/SM 比 8.0 的 2048 少一半**。Turing 的 occupancy 上限本来就低，不能拿 8.0 的 occupancy 表去套。
- **8.6/8.9 是 1536 threads，不是 2048**。消费级 Ampere/Ada 的 SM 和数据中心 8.0、边缘 8.7 不是一回事。
- **10.7 Rubin 只有 1024 threads/SM，但 shared 到 328KB**。这不是笔误，是 CUDA 13.4 表里的值；从规格看，Rubin 的 SM 设计不是把 10.0 简单放大。
- **12.x 的 shared 只有 100KB**，比 10.0/10.3 的 228KB 少。消费级 die 的面积和功耗取舍不同，不能把 RTX 5090 当成“小 B200”。

另外，10.7 的 Tensor Core 输入类型里没有列 INT8/INT4，这里按 CUDA 13.4 表原样保留；如果后续文档更新，以 NVIDIA 为准。

## 8. 对写 kernel 的实际影响

**先定目标 CC，再选编译目标。** 只支持 8.0+ 的代码不要带 7.5 的 fallback；反过来，想覆盖 8.0–8.9，baseline `sm_80` 是最低公分母。用了 `cp.async` 就至少 8.0，用了 TMA/cluster 就至少 9.0，用了 FP8 至少 8.9（Ada）或 9.0（Hopper），用了 FP4 至少 10.0、11.0 或 12.x。

**occupancy 要按目标 CC 算。** 按线程上限算，同样 256 个线程的 block，在 7.5 上最多 4 个 block/SM（1024 线程），在 8.0 上最多 8 个（2048 线程），在 10.7 上又回到 4 个。shared memory 的 per-SM 上限也从 64KB 到 328KB 不等，tile 大小不能写死。

**feature guard 要写两套。** 编译期用 `__CUDA_ARCH__` 选指令，运行期用 device attribute 或 try-launch 检测能力。只在 `sm_100a` 里编 `tcgen05` 的代码，拿到 12.0 上不能跑；`a` 目标没有前向兼容。

**CI 至少覆盖每个 major 的最低 minor**：7.0、8.0、9.0、10.0、12.0。如果用了 `a` 目标，再覆盖对应的精确 SKU；如果用了 `f` 目标，至少覆盖家族里的两个成员，比如 10.0 和 10.7。

## 9. 怎么读

这张地图解决“编号是什么”；接下来的五代解决“为什么”：

1. **[Tesla 与 Fermi](/posts/gpu-arch/tesla-fermi/)**：统一着色器之后，SM 才开始像计算核心。
2. **[Kepler 与 Maxwell](/posts/gpu-arch/kepler-maxwell/)**：192 核 SMX 和 128 核 SMM 的路线分叉。
3. **[Pascal、Volta 与 Turing](/posts/gpu-arch/pascal-volta-turing/)**：FP16、Tensor Core、RT Core 依次进入 SM。
4. **[Ampere、Ada 与 Hopper](/posts/gpu-arch/ampere-ada-hopper/)**：TF32、稀疏化、FP8 和异步流水线。
5. **[Blackwell 与 Rubin](/posts/gpu-arch/blackwell-rubin/)**：双 die、FP4、TMEM，以及 10.x 家族的分裂。

如果只想看数据中心：8.0 → 9.0 → 10.0 → 10.3 → 10.7。消费级：7.5 → 8.6 → 8.9 → 12.0。边缘：6.2 → 7.2 → 8.7 → 11.0。

具体到算子怎么写，回到 [Kernel 分类](/posts/kernel/intro/)；那里有 [CUDA 手册](/posts/kernel/cuda/)、[Triton 手册](/posts/kernel/triton/)和 [TileLang 手册](/posts/kernel/tilelang/)。

## 参考

- [NVIDIA CUDA GPU Compute Capability](https://developer.nvidia.com/cuda/gpus)
- [CUDA Programming Guide: Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
- [CUDA Toolkit 13.4 Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)
- [CUDA 11.4 Programming Guide（归档，3.5–8.0 规格）](https://docs.nvidia.com/cuda/archive/11.4.0/cuda-c-programming-guide/index.html)
- [CUDA 8.0 Programming Guide（归档，2.x–6.x 规格）](https://docs.nvidia.com/cuda/archive/8.0/cuda-c-programming-guide/index.html)
