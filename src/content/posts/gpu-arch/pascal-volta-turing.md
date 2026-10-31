---
title: "Pascal、Volta 与 Turing：FP16、Tensor Core 与 RT Core 进入 SM"
published: 2026-10-31T09:20:00+08:00
description: Pascal 把 FP16、HBM2、NVLink 带进 GPU，Volta 加上 Tensor Core 和独立线程调度，Turing 再把 RT Core 与 INT8/INT4 推理塞进 SM。三年里，SM 从 FP32 阵列变成了混合精度 + 专用单元的复合体。
tags: [CUDA, GPU, NVIDIA, Pascal, Volta, Turing]
category: GPU Architecture
draft: false
---

2016 到 2018 年是 NVIDIA GPU 从“FP32 SIMT 阵列”转向“混合精度 + 专用单元”的三年。Pascal 把 16nm、HBM2、NVLink 和 FP16 带进来，Volta 在 SM 里塞进 Tensor Core，并重做了线程调度；Turing 再加 RT Core 和 INT8/INT4 推理，把 SM 变成一个由 FP32、INT32、Tensor、RT 四类单元拼起来的复合体。

这三代也是“同一个架构名，不同 SM”最明显的阶段：Pascal 的 GP100 和 GP104 是两种完全不同的 SM，Volta 和 Turing 虽然都属于 7.x，但一个为训练设计，一个为推理和光追设计。

## 1. Pascal：64 核 SM 与 128 核 SM 的分叉

Pascal 的第一颗大芯片 GP100 用了台积电 16nm FinFET，153 亿晶体管，3840 个 FP32 core、60 个 SM（P100 屏蔽到 56 个）。每个 SM 只有 64 个 FP32 core，但配了 32 个 FP64 core，FP64:FP32 保持 1/2；同时有 64KB shared、64K 寄存器、2048 线程/SM 和 4MB L2。P100 用 16GB HBM2、4096-bit、732GB/s，配 NVLink 1（160GB/s）。

消费级的 GP104 走的是另一条路：2560 个 FP32 core、20 个 SM，每个 SM 128 个 FP32 core，但 FP64 只有 4 个（1/32），shared 96KB，配 256-bit GDDR5X。GTX 1080 的 FP16 吞吐被压到 FP32 的 1/64，只有 GP100 才有 2× FP16。这就是为什么“Pascal 支持 FP16”这句话要加主语：数据中心版和消费版不是一回事。

| CC | 芯片 | SM 数 | FP32/SM | FP64/SM | Shared/SM | 显存 | 代表产品 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 6.0 | GP100 | 60（P100 56） | 64 | 32 | 64KB | HBM2 16GB | Tesla P100 |
| 6.1 | GP104 | 20 | 128 | 4 | 96KB | GDDR5X 8GB | GTX 1080 |
| 6.1 | GP102 | 28 | 128 | 4 | 96KB | GDDR5X 11GB | GTX 1080 Ti、Titan Xp |
| 6.2 | GP10B | 2 | 128 | — | 64KB | LPDDR4 | Jetson TX2 |

GP100 的 SM 分成两个 processing block，每个 block 32 个 FP32 core、16 个 FP64 core、1 个 warp scheduler、2 个 dispatch unit、128KB 寄存器文件。它和 Maxwell 的 128 核 SMM 相比，core 数少了一半，但寄存器、shared、warp 数都没少，所以每个线程能拿到的寄存器和 shared 反而更多。P100 的 FP16 用 paired operation 做，两个 FP16 打包成一条指令，峰值 21.2 TFLOPS，是 FP32 的 2 倍。

Pascal 的另外几个变化不在 SM 里，但同样影响 kernel：

- **HBM2 + NVLink**：P100 的 732GB/s 和 160GB/s NVLink 让多卡训练第一次不用挤 PCIe。NVLink 1 是 4 条 link、每条 40GB/s，双向 160GB/s。
- **计算抢占（compute preemption）**：Pascal 支持指令级抢占，长 kernel 可以被更高优先级的任务打断，调试和 QoS 都比以前好。
- **统一内存页迁移**：Pascal 开始支持 GPU 缺页、按页迁移，CPU/GPU 共享地址空间不再需要手动 `cudaMemcpy`。这对大内存、稀疏访问的程序影响很大。

## 2. Volta：Tensor Core 与独立线程调度

GV100 是 Volta 的大芯片：12nm FFN、211 亿晶体管、5120 个 FP32 core、5120 个 INT32 core、80 个 SM。每个 SM 的结构变成了 4 个 processing block，每个 block 16 个 FP32 core、16 个 INT32 core、8 个 FP64 core、2 个 Tensor Core、1 个 warp scheduler、1 个 dispatch unit、64KB 寄存器文件。四个 block 共享 96KB shared memory。

| CC | 芯片 | SM 数 | FP32/SM | INT32/SM | FP64/SM | Tensor/SM | RT/SM | Shared/SM | 线程/SM | 代表产品 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 7.0 | GV100 | 80 | 64 | 64 | 32 | 8 | 0 | 96KB | 2048 | V100、Titan V |
| 7.2 | GV11B | 8 | 64 | 64 | 32 | 8 | 0 | 96KB | 2048 | Jetson AGX Xavier |
| 7.5 | TU102 | 72 | 64 | 64 | 4 | 8 | 1 | 64KB（96KB 统一） | 1024 | RTX 2080 Ti、T4 |

Volta 最重要的两个变化：

**第一，Tensor Core。** 每个 Tensor Core 一个时钟做 4×4×4 的矩阵乘加，输入 FP16、累加 FP32。8 个 Tensor Core 合起来，一个 SM 每时钟 512 个 FP16 FMA；80 个 SM 的 V100 峰值 125 TFLOPS。它不是替代 FP32 core，而是给 GEMM/卷积这类规则矩阵运算开了一条专用路径。写 kernel 时可以用 `wmma` API，也可以直接写 `mma.sync` PTX；前者可移植，后者能控制 fragment 布局。

**第二，独立线程调度（independent thread scheduling）。** Volta 之前，一个 warp 的 32 个线程共享一个 program counter，分支 divergence 只能串行执行；Volta 给每个线程独立的 PC 和调用栈，线程可以各自 yield、reconverge。好处是 fine-grained 的并行（比如图遍历、稀疏计算）不再被 warp 锁死；代价是过去依赖“warp 同步执行”的代码不再安全，必须在需要的地方加 `__syncwarp()`。

```cpp
// 7.0 起，warp 内通信要用带 _sync 的版本，不能再假设锁步执行
int lane = threadIdx.x & 31;
int value = /* ... */;
int broadcast = __shfl_sync(0xffffffff, value, 0);
```

Volta 的 FP64 也保持 1/2，V100 有 7.8 TFLOPS FP64、15.7 TFLOPS FP32、125 TFLOPS Tensor。它在 2017 年同时覆盖了 HPC 和深度学习两个市场，直到 Hopper 才被替换。

## 3. Turing：RT Core、INT8/INT4 和更小的 SM

TU102 是 Turing 的大芯片：12nm FFN、186 亿晶体管、4608 个 FP32 core、4608 个 INT32 core、72 个 SM。每个 SM 仍然是 4 个 processing block，每个 block 16 个 FP32 core、16 个 INT32 core、2 个 Tensor Core、1 个 warp scheduler、1 个 dispatch unit、64KB 寄存器文件；四个 block 共享 96KB 的 L1/shared，可以按 64KB shared + 32KB L1 或 32KB shared + 64KB L1 划分。

Turing 的 SM 有两个新单元：

- **RT Core**：每个 SM 一个，专门做 BVH 遍历和 ray-triangle 求交。它不替代 CUDA core，而是把光追里最不规则的那部分固定下来。RTX 2080 Ti 有 68 个 RT Core，TU102 满配 72 个。
- **第二代 Tensor Core**：在 Volta 的 FP16 基础上增加 INT8 和 INT4 模式，面向推理。INT8 的吞吐是 FP16 的 2 倍，INT4 再翻倍；对量化后的模型，这是 Turing 相对 Pascal 最大的吸引力。

Turing 的线程上限反而降了：1024 线程/SM、32 warp/SM、16 block/SM，而 Pascal/Volta 是 2048/64。原因不是退步，而是 Turing 的 SM 线程容量更小、更偏向图形和推理；它的 4 个 block 每个只有 16+16 个 core，64 个 warp 也喂不饱。对 compute kernel 来说，这意味着 occupancy 上限更低，寄存器压力更容易成为瓶颈。

T4（CC 7.5）是 Turing 的数据中心推理卡：40 个 SM、2560 个 core、16GB GDDR6、320GB/s、70W，FP16 Tensor 65 TFLOPS。它没有 RT Core 的用武之地，但 INT8/INT4 和低功耗让它在推理机房里很常见。

## 4. 三年里的三次转向

把这三代连起来看：

- **Pascal 转向“带宽和精度”**：HBM2、NVLink、FP16 2×、计算抢占。SM 本身还是 FP32 阵列，但内存和互联开始追上算力。
- **Volta 转向“专用矩阵单元 + 细粒度调度”**：Tensor Core 把 GEMM 从 CUDA core 手里接过去，独立线程调度把 warp 从“锁步”里解放出来。SM 从此不再只有一种执行单元。
- **Turing 转向“推理和光追”**：INT8/INT4 Tensor Core、RT Core、更小的 SM。它把 Tensor Core 下放到消费级，同时为推理和实时光追准备了专用硬件。

对写 kernel 的人来说，这三代的分水岭是 Volta：在此之前，优化主要是“把 FP32 流水喂满”；在此之后，要先判断这段计算该走 CUDA core、Tensor Core 还是 RT Core，再决定数据布局和同步方式。Turing 又把这个问题推到推理侧：同一个 GEMM，FP16、INT8、INT4 的 tile 和 scale 都不一样。

下一站是 Ampere、Ada 和 Hopper：Ampere 把 Tensor Core 做成 3 代、加入 TF32/BF16 和 2:4 稀疏；Ada 加 FP8 和更大的 L2；Hopper 用 FP8 Transformer Engine、TMA 和 thread block cluster 把异步流水线做进 SM。

## 参考

- [NVIDIA Tesla P100 Whitepaper](https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper.pdf)
- [NVIDIA Tesla V100 Whitepaper](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)
- [NVIDIA Turing GPU Architecture Whitepaper](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf)
- [CUDA Programming Guide: Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
- [NVIDIA CUDA GPU Compute Capability](https://developer.nvidia.com/cuda/gpus)
