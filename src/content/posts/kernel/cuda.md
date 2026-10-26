---
title: "CUDA 手册：执行模型、内存层级与 Kernel 优化"
published: 2026-10-27T00:30:00+08:00
description: 从 grid/block/warp 和内存层级讲起，给出可运行的 vector add 与 GEMM 分块骨架，再落到 nvcc 产物、Nsight 指标和常见性能坑的检查清单。
tags: [CUDA, GPU, Kernel, 性能优化]
category: Kernel
draft: false
---

CUDA 是这三篇里唯一需要自己管理线程的：grid / block / thread 三层索引、shared memory、同步和寄存器压力全部摊在明面上。代价是代码更长，收益是出问题时能定位到具体一条指令、一个 bank conflict。

这篇按“执行模型 → 内存 → 一个能跑的 kernel → 分块 GEMM → 工具链 → 检查清单”的顺序整理，代码都可以直接编译。Triton、TileLang 的抽象最终也会落到这层，把这里的概念弄清楚，另外两篇会好读很多。

## 执行模型：三层索引

一次 kernel launch 的配置是 `<<<grid, block>>>`，对应三个内建变量：

| 层级 | 内建变量 | 硬件对应 | 规模上限（典型） |
|---|---|---|---|
| grid | `gridDim.x/y/z` | 一次 launch 的所有 block | 每维 $2^{31}-1$ |
| block | `blockDim.x/y/z` | 一个 SM 上可驻留的线程块 | 1024 threads |
| thread | `threadIdx.x/y/z` | 一个 CUDA core 上的逻辑线程 | 受 block 限制 |
| warp | — | 32 个连续线程，调度的基本单位 | 32 threads |

线程的全局编号通常写成：

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

warp 是硬件调度的最小单位。同一个 warp 的 32 个线程共享一条指令流，遇到分支时两条路径会串行执行（divergence）。Volta 之后线程有独立的程序计数器，不能再假设 warp 内锁步执行；需要 warp 内同步时用 `__syncwarp()`。

一个 SM 能同时驻留多少 block，取决于四个约束里最紧的那个：

- 每个 SM 的最大线程数（如 2048）；
- 每个 SM 的寄存器数量（如 65536 个 32-bit 寄存器）；
- 每个 SM 的 shared memory（如 228 KB）；
- 每个 SM 的最大 block 数（如 32）。

occupancy 是“实际驻留 warp 数 / 最大 warp 数”。它影响的是延迟隐藏能力，不是越高越好：把 occupancy 从 50% 拉到 100% 往往要砍寄存器，反而制造 spill。先看 profiler 里的 stall reason，再决定压不压寄存器。

## 最小可运行 kernel

```cpp
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr)                                                  \
  do {                                                                    \
    cudaError_t err = (expr);                                             \
    if (err != cudaSuccess) {                                             \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,                  \
              cudaGetErrorString(err));                                   \
      std::exit(EXIT_FAILURE);                                            \
    }                                                                     \
  } while (0)

__global__ void vec_add(const float* __restrict__ a,
                        const float* __restrict__ b,
                        float* __restrict__ c, int n) {
  // grid-stride loop：grid 不用刚好覆盖 n，也能处理任意大小
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  constexpr int n = 1 << 20;
  const size_t bytes = n * sizeof(float);
  float *a, *b, *c;
  CUDA_CHECK(cudaMallocManaged(&a, bytes));
  CUDA_CHECK(cudaMallocManaged(&b, bytes));
  CUDA_CHECK(cudaMallocManaged(&c, bytes));
  for (int i = 0; i < n; ++i) { a[i] = 1.0f; b[i] = 2.0f; }

  int block = 256;
  int grid = (n + block - 1) / block;
  vec_add<<<grid, block>>>(a, b, c, n);
  CUDA_CHECK(cudaGetLastError());        // launch 本身的错误
  CUDA_CHECK(cudaDeviceSynchronize());   // 执行期的错误

  for (int i = 0; i < n; i += n / 4) printf("c[%d] = %f\n", i, c[i]);
  CUDA_CHECK(cudaFree(a));
  CUDA_CHECK(cudaFree(b));
  CUDA_CHECK(cudaFree(c));
}
```

编译与运行：

```bash
nvcc -O3 -arch=sm_90 vec_add.cu -o vec_add
./vec_add
```

几个约定：

- `-arch=sm_90` 对应 Hopper；要用 wgmma 这类架构特性时写 `sm_90a`。不知道目标卡时可以先 `nvidia-smi --query-gpu=compute_cap --format=csv` 查。
- `__restrict__` 告诉编译器指针不重叠，才有向量化和只读缓存的余地。
- `cudaMallocManaged` 让 CPU/GPU 共用一个地址，示例够用；性能敏感路径仍建议 `cudaMalloc` + 显式 `cudaMemcpy`，或直接用 `cudaMallocAsync` 配合内存池。
- 真实项目里建议把 launch 包成 `CUDA_CHECK` 宏，否则异步错误会拖到下一次 API 调用才报，现场已经丢了。

## 内存层级

| 空间 | 作用域 | 生命周期 | 典型延迟 | 怎么用 |
|---|---|---|---|---|
| register | 单线程 | kernel | 1 cycle | 局部变量、累加器 |
| shared | block | kernel | ~20–30 cycle | `__shared__`，块内复用 |
| L1 / L2 | SM / 全卡 | 按缓存策略 | L1 ~30，L2 ~200 | 硬件管理 |
| global | 全卡 | 显式分配 | ~400–800 | `cudaMalloc` / 框架 tensor |
| constant | 全卡，只读 | 显式 | 命中时 1 cycle | `__constant__`，广播访问 |
| local | 单线程 | kernel | 走 global | 寄存器不够时的 spill |

两个和性能直接相关的点：

**合并访存（coalescing）**。一个 warp 访问 32 个连续的 `float`，正好覆盖 128 B 的三个 cache line 中的一段；如果步长是 32（比如按列读行主序矩阵），每次访问都落在不同 cache line，实际带宽会掉一个数量级。写 kernel 时先问自己：warp 内这 32 个地址是不是连续的。

**bank conflict**。shared memory 被切成 32 个 bank，每个 bank 4 B。地址 `addr` 落在 `(addr / 4) % 32`。同一个 warp 内两个线程访问同一 bank 的不同地址就要串行。典型冲突来自按列访问二维 `__shared__` 数组：

```cpp
__shared__ float tile[32][32];   // 行跨度 32 个 float，正好 32 个 bank
// 线程 (ty, tx) 读 tile[k][ty]：地址 = k*32 + ty，bank = ty
```

把第二维补一列 `tile[32][33]`，行跨度变成 33，按列访问时 bank 就散开了。更系统的做法是 swizzle（CUTLASS 里的 `Swizzle<B, M, S>`），用异或把地址打散，代价是索引计算多几条指令。

## 分块 GEMM：shared memory 该干什么

朴素 GEMM 每个输出元素读 A 的一行和 B 的一列，总访存 $O(MNK)$；分块后每个 block 负责 $BM \times BN$ 的输出，K 方向每次只搬 $BM \times BK$ 和 $BK \times BN$ 两块，总访存降到 $O(MNK/BM + MNK/BN)$。棋盘越大，复用越高。

下面是最小骨架，假设 `M/N/K` 都是 32 的倍数、`blockDim = (32, 32)`、每个线程算一个输出元素：

```cpp
constexpr int BM = 32, BN = 32, BK = 32;

__global__ void gemm_tiled(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C, int M, int N, int K) {
  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];

  int row = blockIdx.y * BM + threadIdx.y;   // 输出行
  int col = blockIdx.x * BN + threadIdx.x;   // 输出列
  float acc = 0.0f;

  for (int k0 = 0; k0 < K; k0 += BK) {
    As[threadIdx.y][threadIdx.x] = A[row * K + k0 + threadIdx.x];
    Bs[threadIdx.y][threadIdx.x] = B[(k0 + threadIdx.y) * N + col];
    __syncthreads();                          // 等整块搬完
    for (int k = 0; k < BK; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    __syncthreads();                          // 等所有人算完再覆盖
  }
  C[row * N + col] = acc;
}
```

这段代码在真实尺寸下还不够快，但结构是完整的：global → shared 的协作搬运、块内 barrier、寄存器累加、写回。继续往下走通常是四步：

1. **register blocking**：每个线程算 $4\times4$ 或 $8\times8$ 个输出，摊薄 shared 读取和循环开销。
2. **向量化**：`float4` 读 global，`cp.async` / TMA 异步搬运，双缓冲。
3. **tensor core**：`wmma::`、PTX `mma.sync`、Hopper 的 `wgmma`、Blackwell 的 `tcgen05`，或者直接用 CUTLASS/CuTe 的模板。
4. **按架构调 tile**：tile 大小、stage 数、swizzle 和 epilogue 融合一起决定最终性能。

## 同步与异步

- `__syncthreads()`：block 内所有线程的 barrier，同时保证 shared/global 的写对本 block 可见。**不能放在只有部分线程会走到的分支里**，否则行为未定义。
- `__syncwarp(mask)`：warp 内同步；Volta 之后做 warp-synchronous 编程必须加。
- `cp.async`（sm_80+）：global → shared 的异步拷贝，配合 `__pipeline_commit()` / `__pipeline_wait_prior(N)` 做多级流水。
- TMA（sm_90+）：`cp.async.bulk.tensor` 一条指令搬一个 tile，带多维描述符，适合大块 2D 搬运。`cuda::memcpy_async` 或 CUTLASS 封装都能用。
- cluster（sm_90+）：多个 block 组成 cluster，可通过 DSMEM 互相读写 shared memory，`__cluster_dims__` 或 `cudaLaunchKernelEx` 配置。
- stream：不同 stream 的 kernel 可以并发；`cudaMemcpyAsync` 只有在 pinned host memory 上才真正异步。
- CUDA Graph：把一串 launch 录成图，减少小 kernel 的启动开销。

## 编译产物与工具链

`nvcc` 的链路是 `.cu → 预处理 → cicc → PTX → ptxas → SASS/cubin → fatbin`。常用的检查手段：

```bash
# 看寄存器、spill、shared memory 用量
nvcc -O3 -arch=sm_90 -Xptxas -v vec_add.cu -o vec_add

# 只生成 PTX
nvcc -ptx -arch=sm_90 vec_add.cu -o vec_add.ptx

# 反汇编 SASS
cuobjdump -sass vec_add
nvdisasm vec_add.cubin
```

- `-lineinfo` 保留源码行号给 profiler 看；`-G` 是完整 device 调试，性能会掉很多，别拿它测性能。
- `--use_fast_math` 会打开近似除法、`__expf` 等，数值会变；对精度敏感的算子不要全局开。
- `-maxrregcount=N` 强制压寄存器，可能把 spill 逼出来，通常不如调 tile。

运行时检查用 compute-sanitizer：

```bash
compute-sanitizer --tool memcheck --leak-check full ./vec_add
compute-sanitizer --tool racecheck ./vec_add      # shared memory 竞争
compute-sanitizer --tool synccheck ./vec_add      # 非法 barrier
```

性能分析用 Nsight Compute / Nsight Systems：

```bash
ncu --set full --kernel-name regex:gemm ./gemm
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed ./gemm
nsys profile --stats=true ./gemm
```

读 `ncu` 报告先看两个 throughput：SM 利用率高说明 compute-bound，DRAM 利用率高说明 memory-bound，两者都不高通常是 latency-bound（occupancy 不够、依赖链太长、同步太多）。再看 warp stall reason，常见的是 `long_scoreboard`（等 global）、`barrier`、`mio_throttle`（shared 压力）。

## 优化检查清单

1. **先判断 roofline 位置**。算术强度 $I = \text{FLOPs} / \text{Bytes}$，机器的平衡点是 $B = \text{峰值 FLOP/s} / \text{峰值带宽}$。$I < B$ 时优化访存，$I > B$ 时优化计算。以 H100 SXM 的 FP16 tensor core 为例，约 989 TFLOP/s 对 3.35 TB/s，平衡点约 295 FLOP/B；FP32 约 67 TFLOP/s，平衡点约 20 FLOP/B。
2. **访存是否合并、是否向量化**。检查 `ncu` 的 `l1tex__t_sectors_per_request` 和 global load efficiency。
3. **occupancy 与 ILP 的取舍**。延迟隐藏可以靠更多 warp，也可以靠单线程更多独立指令。
4. **shared memory 的 bank conflict 与容量**。静态 shared 超过 48 KB 要改成动态分配，并 `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)`；Hopper 单 block 最多约 227 KB。
5. **分支发散**。把 warp 内不一致的分支改成谓词或重排数据。
6. **寄存器 spill**。`-Xptxas -v` 里出现 `spill stores/loads` 就要看是不是 tile 太大。
7. **尾效应**。grid 不是 SM 数量的整数倍时，最后一波 block 会拖尾；persistent kernel 可以缓解。
8. **启动开销**。小 kernel 密集时用 CUDA Graph 或融合。

## 常见坑

- 启动配置写错但没检查：`<<<0, 0>>>`、block 超过 1024、shared 超过上限，都会在 launch 或运行期报错，用 `CUDA_CHECK` 兜住。
- 忘记边界检查：grid 向上取整后最后一截线程会越界写。
- 索引用 `int` 溢出：`row * N + col` 超过 $2^{31}$ 会回绕，大矩阵改用 `long long` 或 `size_t`。
- 把 host 指针直接传进 kernel：统一内存之外会直接非法访问。
- 以为 `cudaMemcpy` 是异步的：默认 stream 上的 `cudaMemcpy` 会同步；要重叠就用 `cudaMemcpyAsync` + pinned memory + 非默认 stream。
- 在 divergent 分支里调 `__syncthreads()`：未定义行为，可能死锁或读到脏数据。
- 用 `cudaDeviceSynchronize()` 掩盖错误位置：调试期可以，性能路径上它会串行化所有 stream。
- 统一内存不等于免费：首次触页会 fault，跨卡/跨 NUMA 更明显，必要时 `cudaMemPrefetchAsync`。

## 和 Triton、TileLang 的关系

CUDA 是基准层：Triton 和 TileLang 最终都生成 CUDA C 或 PTX，再交给 nvcc/NVRTC。选择上可以这样分：

- 需要 warp 级手工调度、自定义 PTX、特殊指令或极限压榨某个 shape：写 CUDA（或 CUTLASS/CuTe）。
- 想要接近手写 CUDA 的性能但不想管线程和 shared memory：先试 [Triton](/posts/kernel/triton/)。
- 算子是 GEMM/Attention 这类 tile 结构明显、还要跨 NVIDIA/AMD/其他后端：看 [TileLang](/posts/kernel/tilelang/)。

## 参考

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- [CUTLASS](https://github.com/NVIDIA/cutlass)
- [Nsight Compute](https://developer.nvidia.com/nsight-compute) / [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)
