---
title: "TileLang 手册：Tile 级 DSL 与 Tensor Core 算子"
published: 2026-10-27T00:40:00+08:00
description: TVM 生态的 tile 级 DSL：用 T.Kernel/T.copy/T.gemm/T.Pipelined 写 GEMM，编译到 CUDA/HIP/Metal/Ascend，附 autotune、生成源码检查、IR 排错与和 CUDA/Triton 的分工。
tags: [TileLang, GPU, Kernel, Tensor Core]
category: Kernel
draft: false
---

TileLang 站在 CUDA 和 Triton 中间：像 Triton 一样用 Python 写 tile，但把 shared memory、layout、流水线和 Tensor Core 指令的选择显式留在代码里。它的目标不是“最短的 kernel”，而是“用几十行写到接近 CUTLASS 的性能，同时还能改”。

它基于 TVM，编译目标不止 NVIDIA：CUDA、ROCm/HIP、Apple Metal、Huawei Ascend、LLVM CPU 都有路径。对 GEMM、FlashAttention、dequant GEMM、MLA 这类 tile 结构明显的算子，这是目前比较省力的写法。

## 安装与最小验证

```bash
pip install tilelang
python -c "import tilelang; print(tilelang.__version__)"
```

需要可用的 NVIDIA 驱动和 CUDA 环境；首次编译会调用 NVCC/NVRTC。官方也提供 nightly wheel：

```bash
pip install tilelang --find-links https://tile-ai.github.io/whl/nightly
```

先跑通一个 vector add，确认工具链没问题：

```python
import torch
import tilelang
import tilelang.language as T
from tilelang import jit

@jit
def add(N: int, block: int = 256, dtype: str = "float32"):
    @T.prim_func
    def kernel(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block), threads=block) as bx:
            for i in T.Parallel(block):
                gi = bx * block + i
                if gi < N:
                    C[gi] = A[gi] + B[gi]
    return kernel

N = 1 << 20
a = torch.randn(N, device="cuda", dtype=torch.float32)
b = torch.randn(N, device="cuda", dtype=torch.float32)
c = torch.empty_like(a)

kernel = add(N)
kernel(a, b, c)
torch.testing.assert_close(c, a + b)
```

`@jit` 在第一次用具体参数调用时编译并缓存；`T.Tensor((N,), dtype)` 声明形状和 dtype；`T.Kernel(grid, threads=...)` 里的代码运行在 tile 粒度，`bx` 是 block 索引。这里显式加了 `if gi < N`，虽然 TileLang 的 `LegalizeSafeMemoryAccess` 也会为可能越界的访问补 guard，但边界条件写在代码里更不容易误读。

## 编程模型

| 概念 | TileLang | 对应 CUDA |
|---|---|---|
| 执行单元 | `T.Kernel(grid..., threads=N)` | `<<<grid, block>>>` |
| block 索引 | `as (bx, by)` | `blockIdx.x/y` |
| 线程索引 | `T.get_thread_binding()`（需要时） | `threadIdx.x` |
| 共享内存 | `T.alloc_shared(shape, dtype)` | `__shared__` |
| 寄存器 tile | `T.alloc_fragment(shape, dtype)` | 每线程若干寄存器 |
| 标量 | `T.alloc_var(dtype)` | 局部变量 |
| 数据搬运 | `T.copy(src, dst)` | 循环 + `__syncthreads` |
| 矩阵乘 | `T.gemm(A, B, C)` | `mma` / `wgmma` / `tcgen05` |
| 软件流水 | `T.Pipelined(n, num_stages=k)` | `cp.async` 双缓冲手写 |
| 并行循环 | `T.Parallel(...)` | 线程/循环映射 |
| 串行/展开 | `T.serial` / `T.unroll` | `for` / `#pragma unroll` |
| 同步 | 编译器按依赖插入 | `__syncthreads()` |

关键差别是 `T.alloc_fragment`：写在代码里是 `(BM, BN)` 的二维 tile，实际由 layout 推断切成每个线程的寄存器片段。也正因为如此，`T.gemm` 知道 C 的 fragment 布局，`T.copy` 知道怎么写回 global。

TileLang 文档把抽象分成三层，写 kernel 时可以混用：

- **Level 1**：只写计算，tile 和调度交给编译器/autotuner；
- **Level 2**：显式 tile、shared memory、流水线，类似 Triton 的写法，也是这篇的重点；
- **Level 3**：线程级原语、内联 PTX、`T.CUDASourceCodeKernel` 嵌 CUDA 源码，用来处理编译器搞不定的部分。

## GEMM：完整的 TileLang 写法

```python
import tilelang
import tilelang.language as T

def matmul(M, N, K, block_M=128, block_N=128, block_K=32,
           dtype="float16", accum_dtype="float32"):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128
        ) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), dtype)
            B_s = T.alloc_shared((block_K, block_N), dtype)
            C_f = T.alloc_fragment((block_M, block_N), accum_dtype)

            # 可选：按 panel 做 rasterization，提升 L2 命中
            T.use_swizzle(panel_size=10, enable=True)

            T.clear(C_f)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_s)
                T.copy(B[ko * block_K, bx * block_N], B_s)
                T.gemm(A_s, B_s, C_f)

            T.copy(C_f, C[by * block_M, bx * block_N])
    return main

func = matmul(1024, 1024, 1024)
kernel = tilelang.compile(func, out_idx=[2], target="cuda")

import torch
a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
c = kernel(a, b)
torch.testing.assert_close(c, a @ b, rtol=1e-2, atol=1e-2)

print(kernel.get_kernel_source())          # 生成的 CUDA 源码
latency = kernel.get_profiler().do_bench()
print(f"{latency:.4f} ms")
```

逐段解释：

- `T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M))`：grid 的 x 方向对应 N、y 方向对应 M，`threads=128` 是一个 block 的线程数。后续 `T.Parallel` / `T.copy` 会映射到这 128 个线程。
- `T.alloc_shared`：A/B 的 tile 放在 shared memory，生命周期是整个 kernel。
- `T.alloc_fragment`：C 的累加器放在寄存器，形状写 `(block_M, block_N)`，layout 由 `T.gemm` 决定。
- `T.Pipelined(..., num_stages=3)`：把 global→shared 的 copy 和 `T.gemm` 排成三级流水，编译器会插入异步拷贝和 barrier。
- `T.copy(A[by * block_M, ko * block_K], A_s)`：从 global 的某个偏移开始，搬一个和 `A_s` 同形状的 tile；边界不整除时由编译器处理。
- `T.gemm(A_s, B_s, C_f)`：一个 tile 级矩阵乘，NVIDIA 上会落到 MMA/WGMMA/TCGEN5，AMD 上落到 MFMA；`transpose_A` / `transpose_B` 可以处理非标准布局。
- `T.use_swizzle(panel_size=10)`：按 10 个 block 一组重排 grid，改善 L2 复用，和 Triton 的 `GROUP_M` 是同一类优化。

`tilelang.compile` 的 `out_idx=[2]` 表示第 3 个参数是输出，调用时不用传 `C`，返回值就是它。也可以用 `@tilelang.jit(out_idx=[-1])` 装饰工厂函数，得到同样的效果。

## 流水线与 layout

`T.Pipelined` 是最常用的性能旋钮，但它的行为依赖循环体里 copy 和 compute 的依赖关系。默认会给一个自动调度；需要手工控制时，可以传 `stage` / `order` 指定每条语句所属的流水阶段和发射顺序（两者按语句在源码里的顺序对齐，不能和 `num_stages` 混用）。调参时先固定 `num_stages=2/3/4` 扫一遍，再看 shared memory 是否够用。

layout 方面有两个常用手段：

```python
from tilelang.cuda.intrinsics import make_mma_swizzle_layout

T.annotate_layout({
    A_s: make_mma_swizzle_layout(A_s),
    B_s: make_mma_swizzle_layout(B_s),
})
```

`make_mma_swizzle_layout` 把 shared memory 的地址按 MMA 的需要打散，避免 bank conflict；对不规则的访问模式，也可以自己写 `T.Layout`。`T.copy` 默认会做向量化和合并访存，但前提是源/目标的内存布局允许；如果 profiler 显示 shared 访问效率低，先检查这里。

## Autotune

TileLang 自带 autotuner，支持装饰器和编程式两种入口。装饰器写法：

```python
def matmul_configs(M, N, K):
    return [
        dict(block_M=BM, block_N=BN, block_K=BK, num_stages=S, threads=TH)
        for BM in (64, 128)
        for BN in (64, 128)
        for BK in (32, 64)
        for S in (2, 3)
        for TH in (128, 256)
    ]

@tilelang.autotune(configs=matmul_configs, warmup=25, rep=100, timeout=60)
@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M=128, block_N=128, block_K=32,
           num_stages=3, threads=128, dtype="float16"):
    @T.prim_func
    def main(A: T.Tensor((M, K), dtype),
             B: T.Tensor((K, N), dtype),
             C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M),
                      threads=threads) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), dtype)
            B_s = T.alloc_shared((block_K, block_N), dtype)
            C_f = T.alloc_fragment((block_M, block_N), "float32")
            T.clear(C_f)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_s)
                T.copy(B[ko * block_K, bx * block_N], B_s)
                T.gemm(A_s, B_s, C_f)
            T.copy(C_f, C[by * block_M, bx * block_N])
    return main
```

调优时用固定输入，避免每个配置测到不同的数据：

```python
from tilelang.autotuner import set_autotune_inputs

with set_autotune_inputs(a, b, c):
    tuned = matmul(1024, 1024, 1024)
    tuned(a, b, c)
```

两个容易踩的点：

- 配置里的 key 必须和工厂函数参数名一致；动态 shape（`T.dyn`）没法用内置数据生成器，要用 `set_autotune_inputs` 或自定义 `supply_prog`。
- 结果缓存在 `~/.tilelang/cache`（可用 `TILELANG_CACHE_DIR` 改），key 包含源码、配置和编译参数；`TILELANG_DISABLE_CACHE=1` 可以整体关掉。CI 里建议固定 cache 目录，避免每次重跑。

## 编译链路与调试

TileLang 的 lowering 大致是：

```text
Python AST → TIR → tile op lowering（LowerTileOP）
           → ThreadSync / 流水线变换 → 目标代码（CUDA/HIP/...）
           → nvcc / NVRTC / hipcc → cubin / so
```

出错时通常分三类：生成失败、结果不对、性能不及预期。对应的工具：

- **看生成的源码**：`kernel.get_kernel_source()`；需要拦截/修改生成结果时注册 `register_cuda_postproc_callback`。
- **运行时打印**：`T.print(C_f, msg="accumulator:")`，会按 block/thread 打标签输出；注意并发和同步。
- **看每一趟 pass 的 IR**：`TL_LOWER_TRACE=terminal python x.py`，或 `TL_LOWER_TRACE=html` 生成可浏览的 diff。
- **结构树**：`python -m tilelang.tools.pass_visualizer.viewer kernel.py --set M=1024 ... --out passes.html`，按 pass 展开 tile op 和 lowering 结果。
- **自动缩小复现**：AutoDD（`python -m tilelang.autodd buggy.py --err-msg "..."`）。
- **layout 可视化**：把推断出的 fragment/thread 映射画出来，排查“结果不对但语法没错”的问题。
- **性能**：`get_profiler().do_bench()` 做端到端计时，细粒度指标仍然交给 Nsight Compute / rocProf。

一个实用的顺序是：先用 PyTorch 对拍确认数值，再看 `get_kernel_source()` 里 `T.gemm` 是否落到了预期的 MMA 指令，最后才调 tile 和 pipeline。

## 和 CUDA、Triton 的分工

| | CUDA | Triton | TileLang |
|---|---|---|---|
| 抽象粒度 | thread | block / tile | tile |
| shared memory | 手动 | 编译器管理 | 显式 `alloc_shared` |
| layout 控制 | 完全 | 靠编译器推断 | 注解 + 推断 |
| 软件流水 | 手写 `cp.async` | `num_stages` | `T.Pipelined` |
| Tensor Core | 手写 MMA/WGMMA | `tl.dot` | `T.gemm` |
| 主要调优入口 | 线程/寄存器/指令 | block 形状、warp、stage | tile、stage、threads、swizzle |
| 后端 | NVIDIA | NVIDIA / AMD | NVIDIA / AMD / Metal / Ascend / CPU |
| 适合 | 极限定制、特殊指令 | 融合算子、快速迭代 | GEMM/Attention/量化算子、多后端 |
| 调试难度 | 高但工具成熟 | 中 | 中，生成源码可读 |

经验上的选择顺序：先用 Triton 写一版能对拍的实现；如果 profiling 显示瓶颈在 layout 转换或缺少 Tensor Core 流水，换 TileLang；如果 TileLang 的调度仍然达不到目标，再落到 CUDA/CUTLASS，只把最热的那个 kernel 手写。

## 限制与注意

- **版本还在 0.1.x**。API 迭代快，0.1.13 就移除过一批 legacy API；升级前先看 release notes，别在生产环境直接 `pip install -U`。
- **两套写法并存**。`@T.prim_func` + `tilelang.compile` 和较新的 eager `@tilelang.jit` / `T.const` 风格在文档里都有，混用示例容易踩坑；团队内最好统一一种。
- **编译时间**。TVM pass 链路比 Triton 长，首次编译和 autotune 的开销要算进开发循环；用 `par_compile` 或持久缓存缓解。
- **生成代码依赖工具链**。CUDA 路径需要 NVCC 或 NVRTC；目标架构不匹配时会在编译期或运行期报错。
- **多后端不等于同性能**。TileLang 能生成 HIP/Metal/Ascend 代码，但每个后端的 tile 大小、指令和调优空间不同，换卡要重新 tune。
- **Python 层的错误信息**。traceback 可能停在 AST 构建阶段，真正的类型/形状错误要通过 IR trace 或生成源码定位。

## 参考

- [TileLang 文档](https://tilelang.com/)
- [GitHub: tile-ai/tilelang](https://github.com/tile-ai/tilelang)
- [Language Basics](https://tilelang.com/programming_guides/language_basics.html)
- [Autotuning](https://tilelang.com/programming_guides/autotuning.html)
- [Debugging TileLang Programs](https://tilelang.com/tutorials/debug_tools_for_tilelang.html)
- [GEMM with Tile Library](https://tilelang.com/deeplearning_operators/matmul.html)
