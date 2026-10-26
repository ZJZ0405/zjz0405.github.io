---
title: "Triton 手册：块级编程、编译链路与 Autotune"
published: 2026-10-27T00:35:00+08:00
description: 用 Python DSL 写 GPU kernel：从 vector add、softmax 到 matmul，讲清 program instance、layout inference、TTIR→TTGIR→PTX 链路、autotune 与调试手段，以及 Triton 不擅长的地方。
tags: [Triton, GPU, Kernel, 编译器]
category: Kernel
draft: false
---

Triton 把 CUDA 里“线程怎么排、shared memory 怎么分”这两件事交给了编译器。你写的是一个 block 上的张量运算，编译器负责 layout 推断、合并访存、分配 shared memory、插入同步和流水。代价是控制力下降：warp 级调度、特殊指令、非规则流水线都不好表达。

对多数融合算子（softmax、LayerNorm、RoPE、attention、量化 dequant）这已经够用；PyTorch Inductor、vLLM、SGLang 的大量 kernel 就是 Triton 写的。这篇按编程模型、实例、编译产物、调优和限制组织，和 [CUDA 手册](/posts/kernel/cuda/) 对照着看更容易理解它在哪一层。

## 编程模型：一个 program 算一个 block

CUDA 的心智模型是“一个 thread 算一个元素”，Triton 是“一个 program instance 算一个 tile”。launch 的 grid 大小等于 tile 的数量，`tl.program_id(0)` 取当前 tile 的编号。

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)

def add(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, 1024),)
    add_kernel[grid](x, y, out, n, BLOCK=1024)
    return out
```

几个关键点：

- `tl.arange(0, BLOCK)` 产生长度为 `BLOCK` 的 1D block，`BLOCK` 必须是 2 的幂。
- `tl.constexpr` 的参数在编译期特化，同一个 kernel 对不同 `BLOCK` 会生成不同版本。
- mask 不只是防越界，`other` 还决定越界位置参与运算的值（softmax 里要填 `-inf`）。
- 指针运算 `x_ptr + offsets` 是 element-wise 的；这个例子假设 `x` 连续。带 stride 的版本要把行/列偏移显式写出来。
- program instance 不是 thread：一个 instance 会被编译成多个 warp，warp 数由 `num_warps` 控制。

## 实例：一行 softmax

比 vector add 更能说明问题的是 softmax。要求数值稳定，先减最大值再取指数：

```python
@triton.jit
def softmax_kernel(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    numer = tl.exp(x)
    denom = tl.sum(numer, axis=0)
    tl.store(y_ptr + row * n_cols + cols, numer / denom, mask=mask)

def softmax(x):
    rows, cols = x.shape
    y = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(cols)
    softmax_kernel[(rows,)](x, y, cols, BLOCK=BLOCK)
    return y
```

`tl.max` / `tl.sum` 是 block 级 reduction，编译器会把它降成 warp shuffle 加 shared memory。`BLOCK` 取 `next_power_of_2(cols)`，列数不是 2 的幂时靠 mask 补齐；列数很大时这个写法会撑爆寄存器，要改成按列分块的两趟或 online softmax。

## 实例：matmul 的核心循环

Triton 的 matmul 教程是理解 layout 和 pipelining 的最好材料。下面只保留核心，省掉 grouped ordering 和 autotune 的样板：

```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr,
                  M, N, K,
                  stride_am, stride_ak,
                  stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BN)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BK, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BK, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK * stride_ak
        b_ptrs += BK * stride_bk

    c = acc.to(tl.float16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
```

`tl.dot` 的输入要求块尺寸至少 16×16、dtype 支持 fp16/bf16/fp32/tf32 等；fp32 输入默认走 TF32，要严格 IEEE 结果得传 `input_precision="ieee"`。`acc` 用 fp32 累加是常规做法，最后再转回输出 dtype。

启动时 grid 是一维的：

```python
grid = lambda META: (triton.cdiv(M, META["BM"]) * triton.cdiv(N, META["BN"]),)
matmul_kernel[grid](a, b, c, M, N, K,
                    a.stride(0), a.stride(1),
                    b.stride(0), b.stride(1),
                    c.stride(0), c.stride(1),
                    BM=128, BN=128, BK=32)
```

真实的 matmul 还会做 L2 友好的 block 重排（`GROUP_M`），让同时运行的 tile 共享 A/B 的行列，减少重复读。教程里在 A100 上报告过 10% 以上的提升，和具体 shape 有关。

## 编译链路

Triton 的编译大致是：

```text
Python AST
  → TTIR        高层张量 IR：load/store/dot/reduce
  → TTGIR       引入 layout、shared memory、async copy、warp 划分
  → LLVM IR
  → PTX / cubin
```

最关键的阶段是 TTGIR 的 layout 推断。`tl.load` 得到的 block 有一个分布式的 layout（哪些元素在哪个 thread 的寄存器里），`tl.dot` 要求操作数满足特定 layout，不满足就插入 convert_layout，convert 往往意味着走一遍 shared memory。很多“Triton 写出来比预期慢”的案例，根源是多余的 layout 转换。

看中间产物：

```python
compiled = add_kernel.warmup(x, y, out, n, BLOCK=1024, grid=(1,))
print(compiled.asm["ttir"])
print(compiled.asm["ttgir"])
print(compiled.asm["llir"])
print(compiled.asm["ptx"])
```

`asm` 的 key 随版本可能变化，至少 `ttir` / `ttgir` / `ptx` 是稳定的。也可以设置环境变量把每次编译的产物落盘：

```bash
TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/triton-dump python bench.py
TRITON_ALWAYS_COMPILE=1 python bench.py   # 绕过缓存，确保看的是新代码
```

缓存默认在 `~/.triton/cache`，key 包含源码、constexpr、目标架构和编译选项。改了 Python 代码但结果没变，先确认不是缓存。

## Autotune

`@triton.autotune` 在第一次调用时按真实输入跑一遍候选配置，选最快的：

```python
@triton.autotune(
    configs=[
        triton.Config({"BM": 128, "BN": 128, "BK": 32}, num_warps=8, num_stages=3),
        triton.Config({"BM": 64,  "BN": 128, "BK": 32}, num_warps=4, num_stages=4),
        triton.Config({"BM": 128, "BN": 64,  "BK": 32}, num_warps=4, num_stages=4),
    ],
    key=["M", "N", "K"],          # 这些参数变化会触发重新调优
)
@triton.jit
def matmul_kernel(...):
    ...
```

几个实践细节：

- `key` 里的维度变化会重新 tune；shape 种类多的服务端进程要限制 key 的粒度，否则缓存会膨胀。
- `num_warps` 影响一个 program 用多少 warp，`num_stages` 影响软件流水的深度（主要对 K 循环里的 load 生效）。
- `prune_configs_by` 可以按 shared memory 或寄存器预算提前过滤明显非法/必慢的配置。
- autotune 的结果只在进程内缓存；跨进程启动会重新测，冷启动延迟要算进预算。
- 自己写 benchmark 时用 `triton.testing.do_bench`，注意 `warmup` / `rep` 和 L2 flush。

## 性能要点

1. **block 尺寸**。`BM/BN/BK` 决定 tile 的算术强度和 shared memory 占用；太大爆 shared/寄存器，太小重复读 global。
2. **`num_stages`**。K 循环的 load 和 `tl.dot` 要重叠，stage 太浅等带宽，太深吃 shared memory。
3. **mask 的成本**。每次 load/store 都带 mask 会多出谓词和地址计算；主循环可以去掉 mask，只在尾块处理边界，或者把输入 pad 到对齐尺寸。
4. **地址计算**。尽量用 `tl.multiple_of` / `tl.max_contiguous` 告诉编译器对齐和连续性，帮助向量化；避免在热循环里做除法和取模。
5. **L2 复用**。按 `GROUP_M` 重排 program id，让相邻 tile 共用 A 的行或 B 的列。
6. **reduction 的轴**。`tl.sum(x, axis=0)` 和 `axis=1` 的代价可能不同，因为 layout 里元素到线程的映射不一样；必要时先转置/重排。
7. **persistent kernel**。grid 固定为 SM 数，循环取 tile，能减少 launch 和 tail effect，适合 decode 这类小 shape 多的场景。
8. **TMA**。Hopper 及以上可以用 `tl.make_tensor_descriptor` 建描述符，`desc.load/store` 走 TMA，减少地址计算和寄存器压力；需要先 `triton.set_allocator` 提供全局内存分配器。

## 调试

- **解释器模式**：`TRITON_INTERPRET=1 python script.py` 用 NumPy 逐条模拟，能打断点、打印中间张量。限制是不支持 bf16 和间接访存，慢，但定位逻辑错误很快。
- **device 打印**：`tl.device_print("x", x)`，`tl.device_assert(cond)` 需要 `TRITON_DEBUG=1`；`tl.static_print` / `tl.static_assert` 在编译期生效。
- **内存检查**：`compute-sanitizer --tool memcheck python script.py`，越界和 race 都能抓。
- **可视化**：`triton-viz` 可以把 load/store 的地址模式和 layout 画出来。
- **profiling**：Proton（`triton.profiler`）做 kernel 级计时；更细的硬件指标还是 `ncu`。

常见报错和方向：

| 现象 | 大概率原因 |
|---|---|
| `Dot op does not support` | 块尺寸小于 16、dtype 不支持、layout 无法匹配 |
| 结果偶发错误 | mask 写错、race、`other` 值不合适 |
| 比预期慢很多 | 多余的 layout convert、mask 太重、occupancy 不足 |
| 编译特别慢 | 配置太多、`num_stages` 太大、shape 特化爆炸 |
| 首次调用卡住 | autotune 在跑候选配置 |

## 局限

- **没有显式 shared memory 和 warp 级原语**。想让一个 warp 专门做 producer、另一个做 consumer，或者手写 `mma` 的 fragment 布局，Triton 表达不了。Gluon 是官方给的口子：暴露 layout、shared memory、warp specialization，代价是回到更底层的写法。
- **layout 由编译器决定**。大部分时候是好事，遇到 convert_layout 导致的性能悬崖时只能绕：改块形状、改 reduction 轴、换写法。
- **动态控制流有限**。数据相关的循环/分支支持有限，复杂控制流要拆 kernel 或用 mask 模拟。
- **调试仍比 CUDA 间接**。解释器和 device print 能救急，但没有 cuda-gdb 那种逐线程视角。
- **跨后端**。NVIDIA 之外有 AMD 后端，但算子覆盖和调优成熟度不如 CUDA 路径。
- **编译开销**。JIT + autotune 在冷启动时明显，服务化时要预热或落盘缓存。

## 生态

- **PyTorch Inductor**：`torch.compile` 的默认 GPU 后端大量生成 Triton kernel。
- **推理框架**：vLLM、SGLang 的 attention、采样、量化算子。
- **算子库**：FlashAttention 的 Triton 实现、xformers、Unsloth 的融合 kernel。
- **Gluon**：Triton 官方的低层编程模型，用来写需要显式 layout / warp specialization 的 kernel。

## 参考

- [Triton 文档](https://triton-lang.org/main/index.html)
- [Matrix Multiplication 教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [Debugging Triton](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html)
- [Gluon Overview](https://triton-lang.org/main/gluon/index.html)
- [triton-viz](https://github.com/Deep-Learning-Profiling-Tools/triton-viz)
