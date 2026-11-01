---
title: "DeepGEMM 内核库：FP8/FP4 GEMM、MoE 与 Mega MoE"
published: 2026-11-01T22:10:00+08:00
description: "DeepSeek 开源的 tensor core kernel 库：dense / grouped / masked GEMM、DeepSeek-V3.2 indexer 的 MQA logits、融合 EP 通信的 Mega MoE。梳理 NT 布局、per-block scale 契约、DeepJIT 缓存、1550 TFLOPS 背后的条件和选型边界。"
tags: [DeepGEMM, FP8, FP4, GEMM, MoE, CUDA]
category: Kernel
draft: false
---

DeepGEMM 是 DeepSeek 开源的 tensor core kernel 库。2025 年 2 月刚放出来时，它只做一类事：把 DeepSeek-V3 提出的 fine-grained FP8 GEMM 在 Hopper 上跑到接近峰值，dense 和 MoE grouped 两种形态都覆盖，安装时甚至不编译 kernel。到 2026 年，范围已经扩到 FP8/FP4/BF16 dense GEMM、MoE 的 contiguous / masked 分组 GEMM、DeepSeek-V3.2 lightning indexer 的 MQA 打分，以及把 EP dispatch、两层 expert GEMM、SwiGLU 和 combine 融进一个 kernel 的 Mega MoE。

这篇按“解决什么问题 → 接口和布局契约 → 四类 kernel → 性能与边界”组织。重点不是复述 README，而是说清调用前必须准备什么、哪些约束会在运行时直接 assert、以及什么情况下不该用它。

> **五分钟版**
>
> - **定位**：不是通用 BLAS，而是面向 LLM 的 kernel 库。dense GEMM 是基本盘，MoE grouped GEMM 和 Mega MoE 才是它和 cuBLAS/CUTLASS 不一样的地方。
> - **核心约定**：`D = C + A @ B`，量化的 A/B 以 `(data, scale)` 元组传入，layout 由后缀 `nt/nn/tn/tt` 指定；SM90 只支持 NT，SM100 支持全部四种。
> - **scale 是主要门槛**：SM90 用 FP32 scale，在 K 方向每 128 个元素做一次 promotion；SM100 用 packed UE8M0，4 个 scale 塞进一个 `int32`。两代格式不通用。
> - **MoE 两种形态**：contiguous 给 prefill/training，token 按 expert 拼成连续 M；masked 给 decode + CUDA graph，只算 `masked_m` 之内的行。
> - **Mega MoE 是另一个量级**：只支持 SM100、只支持 SwiGLU、需要对称内存和多进程，把 EP 通信和两层 GEMM 叠在一起。
> - **JIT 是双刃剑**：安装不编译 kernel，首次调用按 shape 编译；缓存 `~/.dj`，线上要提前 warmup，并控制 shape 变体数量。
> - **性能别只看一个数**：官方在 H800 上报过 1550 TFLOPS，但那是大 M prefill 的峰值；decode 的 M=64/128 只有几百 TFLOPS，瓶颈在带宽和调度。

## 它解决的是哪一层问题

FP8 的尾数只有 3 位，单个 per-tensor scale 遇到 outlier 就会掉精度。DeepSeek-V3 的做法是把 scale 做细：激活按 1×128 的 token 组量化，权重按 128×128 的块量化。代价落到 GEMM 上，就是每个 K 块算完之后要先乘上 `sA · sB`，再累加：

$$
C[i,j] \approx \sum_{k_b} s_A[i,k_b] \, s_B[j_b,k_b]
\sum_{k \in k_b} A_8[i,k] \, B_8[j,k]
$$

其中 $k_b$ 是一个 128 元素的 K 块，$s_A$ 每个 token 一个，$s_B$ 每 128 行权重一个。内层求和是 tensor core 的活；外层的 scale 乘法和 promotion 要另外做。DeepGEMM 的早期实现用 CUDA core 做两级累加（promotion），既补 FP8 累加的精度，也尽量不打断 tensor core 的流水。

所以它解决的问题可以拆成三层：

- **scale 融合**：per-block / per-token scale 在 kernel 内部完成，不需要先 dequant 成 BF16 再 GEMM。
- **MoE 形状**：expert 共享 N/K、只有 M 变化，把 token 沿 M 拼起来一次 launch，而不是每个 expert 调一次 GEMM。
- **LLM 专用 shape**：decode 阶段 M=64/128、N/K 很大的矩阵乘，和训练/预填的大 M 形状，用 JIT 分别特化。

和几个常见方案的分工：

| 方案 | 强项 | 在 DeepGEMM 场景里的短板 |
| :--- | :--- | :--- |
| cuBLASLt | 通用、架构覆盖全、稳定 | per-block scale 支持有限；MoE 分组和 masked decode 要自己拼；小 M 的调度不一定针对 LLM decode |
| CUTLASS | 模板齐全、epilogue 可定制 | 编译和调参成本高；同一份代码跨 SM90/SM100 要改；模板报错难读 |
| Triton | 写起来快、融合灵活 | TMA/WGMMA/tcgen05 的细节控制有限，block scale 和 SF 布局很难做到极限 |
| DeepGEMM | LLM shape 专用、scale 融合、MoE/通信融合 | 只支持 SM90/SM100；接口约束硬；不是通用 BLAS |

这张表是分工，不是排名。DeepGEMM 自己也封装了 cuBLASLt（`cublaslt_gemm_*`），一部分 BF16 路径在带 `alpha` 时会走 cuBLASLt；它不排斥成熟库，只是把 FP8 blockwise 和 MoE 这段自己做。

## 安装与版本

| 项 | 要求 | 说明 |
| :--- | :--- | :--- |
| GPU | SM90 / SM100 | H100/H800、B200/GB200 等；A100 只有 `deep_gemm.legacy` 里的 Triton 路径 |
| CUDA | 12.9+ | 当前 README 的要求；2025-07 的重构提到 NVCC 12.9 会自动做 FFMA interleaving |
| PyTorch | 2.3+ | Mega MoE 需要 2.9+ 的 symmetric memory |
| CUTLASS | 4.0+ | 作为 submodule 拉取 |
| 编译器 | 支持 C++20 `<format>` | GCC 13+ / libstdc++ 13+ 比较稳 |
| Python | 3.8+ | |

```bash
git clone --recursive git@github.com:deepseek-ai/DeepGEMM.git
cd DeepGEMM

./develop.sh        # 开发：链接 CUTLASS、build、把 .so 软链到 deep_gemm/
# 或
./install.sh        # 构建 wheel 并 pip install

python -c "import deep_gemm; print(deep_gemm.__version__)"
```

`--recursive` 不能省。`develop.sh` 会把 `third-party/cutlass/include/{cutlass,cute}` 软链到 `deep_gemm/include`，再编译宿主侧 C++ 扩展；`install.sh` 额外打 wheel 并 `pip install --force-reinstall`。

README 说“安装时不需要编译 CUDA”，指的是具体 GEMM kernel 由 DeepJIT 在运行时生成；宿主侧的 C++ 扩展仍然要编，或者用官方预编译的 wheel。`DG_SKIP_CUDA_BUILD=1` 可以跳过 CUDA 扩展构建，适合只改 Python 层；`DG_FORCE_BUILD=1` 强制本地编译，不去下载 wheel。

截至 2026 年 11 月，主线的 `deep_gemm.__version__` 是 `2.8.2`，PyPI 上的 `deep_gemm` 还停在 2025 年 2 月的 `1.0.0`。所以不要 `pip install deep_gemm` 了事，按 README 从源码装，或者取仓库 Releases 里的预编译 wheel；API 和 SM100 支持都以仓库为准。

DeepJIT 负责编译和缓存：把 template 参数、shape、layout、dtype、block size 组合成编译单元，生成 cubin 并按编译参数缓存。缓存默认在 `$HOME/.dj`，`DG_JIT_CACHE_DIR` 可以改成 `:` 分隔的多个目录，查找从前到后，未命中就编译到第一个目录。多机/多容器部署时把这个目录挂到共享存储，能省掉重复编译。

几个常用的调试变量：

| 变量 | 作用 |
| :--- | :--- |
| `DG_JIT_DEBUG=1` | 打印编译命令、PTXAS 输出，dump PTX/SASS |
| `DG_PRINT_CONFIGS=1` | 打印每个 shape 选中的 tile / stage 配置 |
| `DG_JIT_CACHE_DIR` | 编译缓存目录，支持多路径 |
| `DG_JIT_CHECK_NO_SPILLS=1` | 编译后断言没有寄存器 spill |
| `DG_JIT_DUMP_SASS=1` | dump SASS，配合 Nsight 看指令 |

## 接口地图

| 家族 | 主要函数 | 用途 |
| :--- | :--- | :--- |
| dense GEMM | `fp8_fp4_gemm_{nt,nn,tn,tt}`、`fp8_gemm_*`、`bf16_gemm_*` | 普通矩阵乘，`D = C + A @ B` |
| MoE grouped（连续） | `m_grouped_fp8_fp4_gemm_{nt,nn}_contiguous` | prefill / training，token 按 expert 拼成连续 M |
| MoE grouped（masked） | `m_grouped_fp8_fp4_gemm_nt_masked` | decode + CUDA graph，按 mask 只算有效行 |
| K-grouped | `k_grouped_fp8_gemm_{tn,nt}_contiguous`、`k_grouped_fp4_gemm_nt_contiguous` | MoE 权重梯度，按 K 拼 group |
| Indexer | `fp8_fp4_mqa_logits`、`paged` / `sparse` 变体 | V3.2 lightning indexer 打分 |
| Mega MoE | `fp8_fp4_mega_moe`、`bf16_mega_moe` | EP dispatch + L1 + SwiGLU + L2 + combine |
| 其他 | `einsum`、`mega_gate`、`mega_mhc`、`batched_syrk` / `batched_symm` | 特定模型结构 |
| 工具 | `set_num_sms`、`set_pdl`、`transform_sf_into_required_layout`、`testing.bench_kineto` | 调优与布局 |

`fp8_gemm_nt` 现在是 `fp8_fp4_gemm_nt` 的别名，A/B 可以是 FP8 或 FP4 的任意组合；但 FP4 只在 SM100 上可用，而且部分组合要求 K-major。名字里的 `fp8` 是历史遗留，不代表只能传 FP8。

`bf16_gemm_*` 不需要 scale，接口最简单，适合拿来做布局和性能对照；但它的优化重点不是低精度，别指望它在 BF16 上比 cuBLAS 快很多。

## 布局与调用约定

所有 GEMM 都遵循 `D = C + A @ B`。A 的逻辑形状是 `[M, K]`，B 是 `[N, K]`，D 是 `[M, N]`；`C` 可选，用来累加。后缀 `n` / `t` 表示该操作数在 `A @ B` 里是否转置：`n` 按逻辑形状直接参与，`t` 转置后参与。因为 A、B 的逻辑形状都以 K 为最后一维，`nt` 就对应两个操作数都按 K-major 存，而 `nn` 对应 A 按 K-major、B 按 N-major。

| 后缀 | A 的内存布局 | B 的内存布局 | 说明 | SM90 | SM100 |
| :--- | :--- | :--- | :--- | :---: | :---: |
| `nt` | `[M, K]`，K 连续 | `[N, K]`，K 连续 | A、B 都按 K-major，权重 `[N, K]` 最自然 | ✓ | ✓ |
| `nn` | `[M, K]`，K 连续 | `[K, N]`，N 连续 | A 按 K-major，B 按 N-major | ✗ | ✓ |
| `tn` | `[K, M]`，M 连续 | `[N, K]`，K 连续 | A 按 M-major，B 按 K-major | ✗ | ✓ |
| `tt` | `[K, M]`，M 连续 | `[K, N]`，N 连续 | A、B 都按 MN-major | ✗ | ✓ |

D 必须是 N-major（`[M, N]` 行主序），C 与 D 的 dtype / shape 要一致。`c is d` 时原地累加；否则库先 `d.copy_(c)` 再算，多一次全量拷贝。k-grouped 例外：`c` 必填，推荐直接传 `c=d`。

dense FP8 的 D 只支持 BF16/FP32；grouped / masked 的 D 只支持 BF16。SM100 额外支持 `alpha` 和自定义 epilogue（比如 `deep_gemm.epilogue.BF16StochasticRounding()`），SM90 传这些会直接 assert。

先跑一个不涉及 scale 的 BF16 版本，确认安装和布局没问题：

```python
import torch
import deep_gemm
from deep_gemm.testing import calc_diff

m, n, k = 4096, 4096, 7168
a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
d = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

deep_gemm.bf16_gemm_nt(a, b, d)
print("diff:", calc_diff(d, a @ b.t()))   # 1 - 余弦相似度，越小越好
```

`calc_diff` 是仓库自带的比较函数，返回 `1 - cos` 形式的相对差异。BF16 输入下这个值通常接近 0（$10^{-7}$ 量级或更小）；换成 FP8 之后会升到 $10^{-4}$ 量级，具体取决于 K 和 scale 粒度。

## Scaling factor：真正的接口契约

DeepGEMM 不猜你的量化粒度，必须通过 `recipe` 告诉它。`recipe = (gran_m, gran_n, gran_k)` 描述 scale 的**存储**粒度，不是计算粒度：一个 scale 覆盖 A 的 `gran_m × gran_k` 块、B 的 `gran_n × gran_k` 块。

未变换的 SF 是 FP32，形状分别是：

```text
SFA: [ceil(M / gran_m), ceil(K / gran_k)]
SFB: [ceil(N / gran_n), ceil(K / gran_k)]
```

`gran_k` 在 SM90 只能是 128，在 SM100 可以是 32 或 128。SM100 还要求每个 scale 是 2 的整数次幂（UE8M0 只能存指数），cast 时用 `round_sf=True`，或者自己先做 `ceil_to_ue8m0`。

名字里的 1D/1D 和 1D/2D 说的是 scale 的维度：A 的 scale 沿 K 变化、每个 token 一个，是一维；B 的 scale 同时沿 N 和 K 分块，是二维。DeepSeek-V3 的 recipe 是 A 1×128、B 128×128，所以 SM90 上走 1D2D kernel；如果 A/B 都按 token 量化，就是 1D1D。SM100 统一成 packed UE8M0 的 1D1D 路径，128×128 的块 scale 会先广播到每行再打包。

SF 有两条路径：

| 路径 | 传入的 SF | 额外开销 | 适用场景 |
| :--- | :--- | :--- | :--- |
| A 未变换 | FP32，`[mn, k]` 粒度 | 每次 GEMM 内部跑一次 transform kernel | 原型、对拍、shape 经常变 |
| B 预变换 | packed `int32`，MN-major | 只校验布局，不再 transform | 生产；权重变换一次缓存，激活由 cast kernel 直接产出 |

SM100 上把权重 SF 预变换并缓存，大概是这样：

```python
import deep_gemm
from deep_gemm.utils import per_block_cast_to_fp8

# w_bf16: [n, k] BF16；w_sf: [n/128, k/128] FP32
w_fp8, w_sf = per_block_cast_to_fp8(w_bf16, use_ue8m0=True)

# 只做一次：广播到每行 + 打包 UE8M0 + 转成 MN-major / TMA 对齐
w_sf = deep_gemm.transform_sf_into_required_layout(
    w_sf, mn=n, k=k, recipe=(128, 128)
)

# 注意：广播后 gran_n 变成 1，GEMM 的 recipe 要跟着改
# a / sfa / d 的构造见上一节
deep_gemm.fp8_gemm_nt((a, sfa), (w_fp8, w_sf), d, recipe=(1, 1, 128))
```

这里有两个容易踩的坑：

- **预变换之后 recipe 要改**。SF 的 `gran_mn` 已经变成 1，传给 GEMM 的 `recipe` 也必须改成 `(1, 1, 128)`；如果还按 `(1, 128, 128)` 传，会按错误的 shape 去校验 `int32` SF，直接 assert。
- **不要对预变换后的 SF 做 `view` / `reshape`**。它的 stride 是 `(1, align(mn, 4))`，沿 MN 连续、沿 K 跳着走；reshape 会触发 copy，破坏布局。

按文档的变换语义粗算一下：SM100 上 128×128 的权重 scale 会被广播成每行一个 UE8M0，再沿 K 每 4 个 scale 打包进一个 `int32`。一个 128 行 × 128 K 的块，原来 4 B；按 K 方向 4 块一组打包，摊下来约 128 B，也就是 32×（K 很小时 padding 占比更高）。大 MoE 模型上这部分显存要算进预算；好在权重 scale 只变一次，可以缓存。

如果走路径 A，每次 GEMM 都会在内部启动 transform kernel；对 decode 这种单次只有几十微秒的 GEMM，transform 的开销不能忽略。所以生产路径通常是：权重 SF 预变换并缓存，激活 SF 由前一个 cast kernel 直接按 packed 格式产出。

## Dense GEMM：一个能跑的例子

下面这个例子用 per-token 的 A、per-block 的 B，对应 DeepSeek-V3 的 recipe。SM90 和 SM100 都能跑，区别只在 `use_ue8m0`：

```python
import torch
import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp8, per_block_cast_to_fp8
from deep_gemm.testing import calc_diff

m, n, k = 4096, 4096, 7168
x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)

sm100 = torch.cuda.get_device_capability()[0] >= 10
a, sfa = per_token_cast_to_fp8(x, use_ue8m0=sm100)   # [m, k/128]
b, sfb = per_block_cast_to_fp8(w, use_ue8m0=sm100)   # [n/128, k/128]

d = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
deep_gemm.fp8_gemm_nt((a, sfa), (b, sfb), d, recipe=(1, 128, 128))

print("diff:", calc_diff(d, x @ w.t()))
```

这个例子里：

- `sfa` 是 `[4096, 56]`，每个 token 56 个 scale（7168 / 128）；`sfb` 是 `[32, 56]`，每 128 行权重一个 scale。
- `recipe=(1, 128, 128)` 表示 A 按 1×128、B 按 128×128。SM90 会走 1D2D；SM100 会先把两个 SF 转成 packed UE8M0，再走 1D1D。
- `per_token_cast_to_fp8` / `per_block_cast_to_fp8` 是为可读性写的参考实现，性能不是它们的重点。生产里应该把 cast 融进前一个 kernel，或者用 `tile_kernels` 这类专门的 cast 实现。

如果 B 也按 token 量化，把 `recipe` 改成 `(1, 1, 128)` 就行。SM90 会切到 1D1D，SFB 需要转成 MN-major；SM100 仍然走同一条 packed 路径，只是 SF 的粒度不同。

要累加已有的 C：

```python
deep_gemm.fp8_gemm_nt((a, sfa), (b, sfb), d, c=d, recipe=(1, 128, 128))
```

`c=d` 是原地累加；如果 `c` 是另一个 tensor，库会先做一次 `d.copy_(c)`。

## MoE：contiguous 与 masked

MoE 的 GEMM 不是每个 expert launch 一次，而是把所有 token 沿 M 拼起来，B 做成 `[G, N, K]`，一次 kernel 按 group 切 M。DeepGEMM 只 group M 轴，N/K 必须固定；这正好匹配 expert 共享形状的 MoE。

### Contiguous layout

`grouped_layout` 有两种语义：

- **默认（非 PSUM）**：长度等于 `M_total`，存每一行属于哪个 expert，padding 行填 `-1`。每个 expert 的 token 段要按 `get_mk_alignment_for_contiguous_layout()` 对齐，padding 的 A 最好清零，否则 SFA 会读到未初始化的 scale。
- **PSUM**：长度等于 `G`，存每个 group 的累计结束位置；group `i` 的有效范围是 `[align(prev_end), grouped_layout[i])`。DeepEP 的 dispatch 会给出每个 expert 的 token 数，累加一下就是这种前缀和布局，适合直接接。

对齐值用 `set_mk_alignment_for_contiguous_layout()` 设置。不要手写 128：先 `get_theoretical_mk_alignment_for_contiguous_layout()` 取当前架构 / dtype 的最小值，再按 token 数向上取整。

接口形态（省略了量化与拼接）：

```python
# a: [M_total, K] FP8；sfa: [M_total, K/128]
# b: [G, N, K] FP8；sfb: [G, N/128, K/128]
# d: [M_total, N] BF16
# grouped_layout: [G] int32，PSUM 语义
deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
    (a, sfa), (b, sfb), d, grouped_layout,
    recipe=(1, 128, 128), use_psum_layout=True,
)
```

不用 PSUM 时，`grouped_layout` 换成 `[M_total]` 的 per-row expert id，`use_psum_layout=False`；此时长度必须等于 A 的行数，padding 行填 `-1`。

### Masked layout

decode 阶段每个 expert 的 token 数在 CPU 上不可见（CUDA graph 捕获后更不能改 shape），所以 A/D 按 `[G, M_max, ...]` 固定分配，`masked_m[i]` 告诉 kernel 第 i 个 expert 只有前多少行有效。`expected_m` 是给 JIT 的平均 M 估计，用来选 tile；它不影响正确性，但会影响性能。

```python
# a: [G, M_max, K] FP8；d: [G, M_max, N] BF16
# masked_m: [G] int32，每个 expert 的实际 token 数
deep_gemm.m_grouped_fp8_gemm_nt_masked(
    (a, sfa), (b, sfb), d, masked_m, expected_m=256,
    recipe=(1, 128, 128),
)
```

结果只保证前 `masked_m[i]` 行；无效行不要读。masked 的 A 是 3D `[G, M, K]`，不是拼平的 `[M_total, K]`；B 仍是 `[G, N, K]`。这两套 API 不能混用。

### K-grouped

训练侧还有 K-grouped：`k_grouped_fp8_gemm_tn_contiguous` / `k_grouped_fp4_gemm_nt_contiguous`，把 group 拼在 K 上、M/N 固定，用来算 expert 权重梯度。`ks_cpu` 是每组 K 的列表，`grouped_layout` 语义同上；`c` 必填，推荐 `c=d` 原地累加，传 `None` 会直接 assert。

## MQA logits：DeepSeek-V3.2 indexer 的打分 kernel

DeepSeek-V3.2 的 sparse attention 里，lightning indexer 要算每个 query 对候选 KV 的分数，再选 top-k KV。DeepGEMM 的 MQA kernel 把这一步做成 fused：

$$
o_{ij} = \sum_{h=1}^{H} \mathrm{ReLU}\left(q_i^{(h)} \cdot (kv_j \odot s_j)\right) \cdot w_{i,h}
$$

$q_i^{(h)}$ 是第 $i$ 个 token 第 $h$ 个 head 的 FP8/FP4 向量，$s_j$ 是 KV 的反量化 scale（non-MX 是每 token 一个，MX 是每 32 个元素一个），$w_{i,h}$ 是打分权重。输出被压缩成 `[seq_len, max_seqlen_k]`：第 $i$ 行从第 0 列开始放它自己的有效 KV span；span 之外的列不会被清理，调用方要自己 mask。这和常规 attention logits 的 `[seq_len, seq_len_kv]` 不一样，别直接拿去做 softmax。

```python
logits = deep_gemm.fp8_fp4_mqa_logits(
    q=(q_fp8, q_sf),          # q_fp8: [seq_len, num_heads, head_dim]
    kv=(kv_fp8, kv_sf),       # kv_fp8: [seq_len_kv, head_dim]
    weights=weights,          # [seq_len, num_heads]
    cu_seq_len_k_start=ks,
    cu_seq_len_k_end=ke,
    max_seqlen_k=int((ke - ks).max()),
)
```

`q_sf` 的有无决定模式，不能混：

| 模式 | `q_sf` | `kv_sf` | 数据 | 架构 |
| :--- | :--- | :--- | :--- | :--- |
| MX | packed `int32` | packed `int32` | FP8 或 FP4 | SM100 |
| non-MX | 不传 | FP32，每 token 一个 | 仅 FP8 | SM90 |

SM90 还限制 `num_heads` 为 32 或 64，输出 FP32；SM100 输出 BF16。paged 版本 `fp8_fp4_paged_mqa_logits` 把 KV 和 SF 放在 `uint8` cache 里（每 token 的 `head_dim` 字节后跟 4 字节 scale；FP4 的 value 部分是 `head_dim/2` 字节），配 `get_paged_mqa_logits_metadata` 用；DSA 的 sparse 变体也在同一组 API 里。

这组 kernel 的 shape 和 mask 语义比较绕，建议直接看 `tests/test_attention.py`，比文档全。

## Mega MoE：把通信和计算塞进一个 kernel

标准 MoE 推理是一条串行链：EP dispatch → L1 GEMM → SwiGLU → L2 GEMM → EP combine。每一步都要等通信或等 kernel，NVLink 带宽和 tensor core 很难同时跑满。Mega MoE 把这几步做成一个 mega-kernel，在 SM100 上把 EP 通信和两层 expert GEMM 叠起来，顺带把 SwiGLU 融进去。

代价是约束很硬：

- 只支持 SM100；SM90 没有这条路径。
- 只支持 SwiGLU，L1/L2 权重同为 FP8 或 FP4（`mma_type="fp8xfp8"` / `"fp8xfp4"`），`fp8_fp4_mega_moe` 的共享 expert 只能是 FP8。
- 需要多进程 + 对称内存（PyTorch ≥ 2.9 的 `torch.distributed._symmetric_memory`），单卡小 MoE 没必要上。
- 权重必须先过 `transform_weights_for_mega_moe`：gate/up 按 8 行交织，L1/L2 的 SF 还要做 UTCCP 转置；scale 必须是 packed `int32`，`recipe` 固定 `(1, 1, 32)`，没有 FP32 回退。

调用形态：

```python
import torch
import deep_gemm

buffer = deep_gemm.get_symm_buffer_for_mega_moe(
    group, num_experts, num_max_tokens_per_rank, num_topk,
    hidden, intermediate_hidden, mma_type="fp8xfp4",
)
l1, l2 = deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights)

buffer.x[:num_tokens].copy_(x_fp8)
buffer.x_sf[:num_tokens].copy_(x_sf)
buffer.topk_idx[:num_tokens].copy_(topk_idx)
buffer.topk_weights[:num_tokens].copy_(topk_weights)

y = torch.empty((num_tokens, hidden), device="cuda", dtype=torch.bfloat16)
deep_gemm.fp8_fp4_mega_moe(y, l1, l2, buffer)
```

`group` 是 EP 的 process group；`num_max_tokens_per_rank` 决定 buffer 大小，按峰值 token 数留余量。每次调用前要把输入 copy 进 buffer；这些 copy 可以 fuse 进前一个 kernel，但第一版通常就是显式 copy。

可选项：`shared_l1_weights` / `shared_l2_weights` 传共享 expert；`cumulative_local_expert_recv_stats` 统计每个 expert 收到的 token；`activation_clamp` 限制 SwiGLU 的幅值；`fast_math=True` 默认开。`localize()` 可以把权重按 SM locality domain 切分，减少每个 SM 的权重访问范围；它会额外复制一份按 domain 排布的权重，不需要时别开，用完调 `destroy_localizer()` 释放 allocator。

什么时候值得上：EP 规模大、NVLink 域内通信占比高、decode batch 足够大，而且你已经在用 EP 和对称内存。否则先把普通 grouped GEMM 和通信 overlap 做好，收益更确定。

## 性能：1550 TFLOPS 说明了什么

官方在 2025 年 4 月报过 H800 上 1550 TFLOPS（FP8）。H800 的 FP8 dense 峰值是 1979 TFLOPS，按这个算约 78%；这是官方挑的 shape，不是所有 shape 的保证。

回到最早的 benchmark 表，M=4096 的 prefill shape 跨度很大，从 590 到 1358 TFLOPS；只有大 K / 大 N 的 shape 才能到 1300+。M=64/128 的 decode shape 只有 200–650 TFLOPS。这不是实现退步，而是小 M 下 GEMM 变成 bandwidth / launch-bound：权重读取占主导，tensor core 吃不饱。DeepGEMM 在 decode 上的价值主要来自 grouped / masked 的组织方式和更少的 launch，而不是把 tensor core 打满。

影响性能的变量：M/N/K、layout（NT vs TN）、SF 路径（FP32 每次 transform vs int32 预打包）、SM 数、JIT 选出的 block / stage。同一个 shape 换 layout 或 SF 路径，差异可能很明显，必须自己测。

测量工具：

- `deep_gemm.testing.bench_kineto`：用 Kineto 只计 kernel 时间，避免把 Python / launch 开销算进去。
- `scripts/quick_plot_pm.py`：把 NCU 的 PM 指标从 `.ncu-rep` 画成图。
- `scripts/run_ncu_mega_moe.sh`：处理多进程 NCU，省得自己拼 `ncu` 参数。
- `DG_PRINT_CONFIGS=1`：先看 JIT 选了什么，再看 NCU，别一上来就调 tile。

调优开关：

- `set_num_sms(n)`：限制使用的 SM 数，给通信或其他 kernel 留资源。
- `set_tc_util(x)`：告诉 heuristic 近似的 tensor core 利用率，影响配置选择。
- `set_pdl(True)`：开 programmatic dependent launch，减少相邻 kernel 的 launch 间隙。
- `use_deterministic_algorithms(True)`：关掉一些非确定路径；BF16 路径会因此不走 cuBLASLt 分支，对拍时有用。

## 限制与坑

- **架构**：只支持 SM90/SM100。A100 只有 `deep_gemm.legacy` 里的 Triton kernel，不在主路径；Ascend 版本拆到 DeepGEMM-Ascend，是另一个仓库。
- **不做量化 / 转置**：输入必须已经是 FP8/FP4 + scale。转置、cast、per-token 量化都要自己在前一个 kernel 里做；`per_token_cast_to_fp8` 这类工具只适合对拍和原型。
- **SF 契约**：SM100 的 scale 最终会以 2 的整数次幂、packed `int32`、MN-major、TMA 对齐的形式交给 kernel；预变换路径要自己保证这些，float32 路径由 transform 代做。不要 reshape 预变换后的 SF；预变换后 recipe 的 `gran_mn` 要改成 1。
- **Grouped 对齐**：每个 expert 的 M 段要按 `get_mk_alignment_for_contiguous_layout()` 对齐，padding 行填 `-1` 并把 A 清零。对齐值随架构和 dtype 变，别硬编码。
- **JIT 冷启动**：首次调用要编译，线上要提前用代表性 shape warmup；多副本挂同一个 `DG_JIT_CACHE_DIR`。`compiled_dims` 决定哪些维度是编译期常量，默认 `nk`（M 动态），TN/TT 默认 `mn`；要减少变体可以调 `set_ignore_compile_dims`，但会牺牲性能。
- **API 漂移**：当前主线的 `fp8_gemm_nt` 是 `(a, b, d, ...)`，A/B 传 `(data, sf)`；2025 年早期教程里的 `fp8_gemm_nt(a, a_scale, b, b_scale, d)` 已经不能直接用。升级时先跑 `tests/` 或自己的对拍。
- **Mega MoE 复杂度**：对称内存、多进程、权重变换、UTCCP 布局，每一层都能单独出错；没有 EP 场景不要为了“融合”两个字上它。
- **确定性**：官方测试对 FP8/FP4 路径做了 20 次 bitwise 对比；但 `use_deterministic_algorithms` 会影响 BF16 的 cuBLASLt 分支。线上如果要求逐位可复现，先把这条确认清楚。

## 什么时候选它

适合：

- 硬件是 Hopper / Blackwell，量化方案是 FP8/FP4 blockwise，模型是 DeepSeek 风格或 MoE。
- 已经在 vLLM / SGLang 里用 DeepGEMM 后端，想调优或排查 kernel 选择。
- 需要 Mega MoE 这种通信-计算融合，团队能维护对称内存、多进程和自定义权重布局。

不适合：

- A100/Ampere 或非 NVIDIA 硬件。
- 需要通用 BLAS、任意 epilogue、任意转置输入；这些用 cuBLASLt / CUTLASS 更省事。
- 小模型、不规则 shape，或者团队不想维护 JIT 缓存和版本兼容。
- 只是想把一个 FP8 模型跑起来：先看 vLLM / SGLang 的现成后端；直接调 DeepGEMM 是最后一步优化，不是第一步。

和本分类的三篇手册的分工：CUDA / Triton / TileLang 讲怎么写出一个 kernel，DeepGEMM 讲一个生产库怎么把 tile、流水线、scale 和 MoE 分组固定下来。和 Quantization 分类的分工：那边讲量化算法和精度，这里讲量化后的 GEMM 怎么落地。和 LLM Engine 的分工：那边讲调度和请求链路，这里讲单个库内部的算子。

## 参考

- [DeepGEMM 仓库](https://github.com/deepseek-ai/DeepGEMM)
- [Scaling Factor Format](https://github.com/deepseek-ai/DeepGEMM/blob/main/docs/scaling-factor-format.md)：SF 的 shape、stride、transform 和 Mega MoE 的 SF 约束
- [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)：fine-grained FP8 方案的出处
- [vLLM FP8 量化文档](https://docs.vllm.ai/en/latest/features/quantization/fp8/)
- [DeepGEMM-Ascend](https://github.com/deepseek-ai/DeepGEMM-Ascend)
- 本分类的相关长文：[CUDA 手册](/posts/kernel/cuda/)、[Triton 手册](/posts/kernel/triton/)、[TileLang 手册](/posts/kernel/tilelang/)
- 量化侧背景：[量化从零开始](/posts/quantization/from-scratch/)
