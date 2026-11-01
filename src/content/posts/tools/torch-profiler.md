---
title: "PyTorch Profiler 使用手册：用 torch.profiler 定位训练与推理热点"
published: 2026-11-02T00:25:00+08:00
description: "从 Kineto / CUPTI 原理、最小采集、统计表列语义，到 schedule、Chrome trace / Perfetto / TensorBoard、record_function 标注、shape / stack、显存与分布式场景：一套用 torch.profiler 定位 PyTorch 训练与推理热点的可复现流程。"
tags: [PyTorch, torch.profiler, Profiling, CUPTI, 性能优化]
category: Tools
draft: false
---

模型慢，不等于 kernel 慢。一次 step 里，时间可能花在 Python 调度、`aten::cat` 的拷贝、动态 shape 触发的新 kernel、反向图的零碎算子、`cudaDeviceSynchronize`，或者显存分配器上。[nsys](/posts/tools/nsys/) 能看到 GPU 行之间的空白，[ncu](/posts/tools/ncu/) 能解释单个 kernel 为什么慢，但“哪个 op、哪个模块、哪种 shape 慢”需要框架层的归因。

`torch.profiler` 是 PyTorch 自带的 profiler，底层是 Kineto，CUDA 后端走 CUPTI。它把 CPU 上的 `aten::*`、CUDA runtime、GPU kernel / memcpy、可选的 Python 栈、显存分配和用户标注放进同一条 trace，同时给出按 op 聚合的统计表。它不用额外装 Nsight，知道 PyTorch 的语义，适合作为第一站。

这篇按“定位 → 采集 → 读表 → 读 trace → 加标注 → shape / 栈 / 显存 → 分布式 / 编译 / 推理 → 坑”组织。API 以 PyTorch 2.15 为基准；2.x 之间有些字段改名，文中会标出来。

> **五分钟版**
>
> - **定位**：先回答“哪个 op / 模块 / shape 占了 CPU 或 GPU 时间”，再决定要不要上 nsys 或 ncu。`torch.profiler` 是框架级入口，不是 ncu 的替代品。
> - **第一次**：`profile(activities=[CPU, CUDA])` 跑 3–5 个稳态 step，`key_averages().table(sort_by="self_cuda_time_total")`，再 `export_chrome_trace()`。先别加 `record_shapes` / `with_stack` / `profile_memory`。
> - **读表**：`Self` 是独占时间，`total` 含子事件；CPU op 和它对应的 kernel 可能各占一行，GPU 时间不能相加；百分比的分母是所有事件的 self 时间之和，不是墙钟。
> - **采稳态**：用 `schedule(wait, warmup, active)` + 每步 `prof.step()`，把初始化、warmup 和第一个请求排除掉。`export_chrome_trace` 在 context 结束后只保留最后一个 cycle。
> - **看 trace**：Chrome trace JSON 用 [Perfetto](https://ui.perfetto.dev) 打开，先看最大的 GPU 块和它前面的 CPU 行；GPU 行空白往往比 kernel 本身更值得先查。
> - **开销**：event-based tracing，每个 op 都要记；`record_shapes`、`with_stack`、`profile_memory` 会再叠一层。采集时的时间不能当生产性能。
> - **一次只开一个**：`torch.profiler`、nsys、ncu 都依赖 CUPTI，同时开容易冲突；要对比就分开跑。

## 它回答什么，不回答什么

先把几类工具放在一起看：

| 想知道的事 | 首选 | 原因 |
| :--- | :--- | :--- |
| 哪个 op / 模块 / shape 占了 CPU 或 GPU 时间 | `torch.profiler` | 知道 PyTorch 的算子名、调用关系、张量形状；一条 trace 同时有 CPU 和 GPU 行 |
| CPU 是否拖住 GPU，launch gap、stream 重叠、多进程 / IO | [nsys](/posts/tools/nsys/) | 系统级时间线，保留进程树、OS runtime、CUDA API 和通信 |
| 单个 kernel 内部为什么慢 | [ncu](/posts/tools/ncu/) | SM / DRAM / cache 硬件计数器、occupancy、warp stall、roofline |
| 端到端吞吐、P50 / P99 延迟 | benchmark + nsys | profiler 自身会改变时序，不能直接当生产数字 |
| 显存泄漏、分配栈、峰值和碎片 | memory snapshot | 分配器级快照，能看到每次 alloc / free 的调用栈 |
| 纯算子微基准、A/B 两个实现 | `torch.utils.benchmark` | 低开销、可重复，不生成整条 trace |

`torch.profiler` 是 **tracing**，不是采样：每个被 hook 的 op 都会记录开始 / 结束时间、线程、可选的 shape 和栈。因此它对“小算子很多”的模型最不友好——op 数量越多，记录开销越大；反过来，它也不会像采样 profiler 那样漏掉短 kernel。

它不读 SM / DRAM 的硬件计数器，所以不能替代 ncu 回答“这个 GEMM 离 roofline 还有多远”；它也不保留进程树、文件 IO、NCCL 网络栈的完整上下文，所以不能替代 nsys 回答“GPU 为什么整段空着”。它最擅长的是把“慢”归因到具体的 `aten::*`、`nn.Module`、shape 和调用阶段，然后决定下一步用哪个工具。

一个常见的排查顺序是：

1. `torch.profiler`：找出最贵的 op / 阶段，确认是 CPU 还是 GPU 时间。
2. `nsys`：如果 GPU 行有大段空白，或问题涉及多进程、通信、IO，切到系统时间线。
3. `ncu`：如果已经锁定某个 kernel，再对这个 kernel 采硬件计数器。

反过来，先跑 ncu 再问“这个 kernel 为什么会被调用这么多次”，通常会把时间花在错误的问题上。

## 工作原理：Kineto、CUPTI 和一条 trace

PyTorch 在算子调度路径上挂了 `RecordFunction` 回调。`torch.profiler` 打开后，Kineto 会记录这些 CPU op 事件，包括名字、时间戳、线程、可选的输入形状、调用栈和用户标注。CUDA 侧由 CUPTI 采集 CUDA runtime API、kernel、memcpy、memset；Kineto 用 correlation id 把 CPU 上的 launch 和 GPU 上的 kernel 对起来。

一条 trace 里通常混着这几类事件：

| 事件类型 | 典型名字 | 来源 |
| :--- | :--- | :--- |
| CPU op | `aten::mm`、`aten::linear`、`aten::cat`、`c10d::allreduce_` | PyTorch dispatcher |
| CUDA runtime | `cudaLaunchKernel`、`cudaMemcpyAsync`、`cudaDeviceSynchronize` | CUPTI |
| GPU kernel | `ampere_sgemm_*`、`sm90_xmma_*`、`triton_*`、`ncclDevKernel_*` | CUPTI |
| GPU memcpy / memset | `[CUDA memcpy HtoD]`、`[CUDA memset]` | CUPTI |
| 用户标注 | `prefill`、`decode`、`train/backward` | `record_function` |
| 显存事件 | `[memory]` | PyTorch caching allocator（需 `profile_memory=True`） |
| Python 栈 / 函数 | 源码文件行号 | Python tracer（需 `with_stack=True`，且 activities 含 CPU） |

统计表不是另一套测量，它只是把同一批事件按名字聚合：`key_averages()` 对每个 key 累加 self / total 时间、调用次数、显存等字段，再按你给的列排序。所以表里的 `Self CUDA` 和 trace 里对应 kernel 的宽度应该能对上；对不上时，先检查是不是把 CPU op 和 kernel 重复计算了。

实现上还有几个直接影响使用的点：

- **Kineto 是 C++ 库**，`torch.profiler` 只是它的 Python 前端。`kineto_available()` 为 `False` 的构建没有 CUPTI / Kineto 路径，只能拿到 legacy autograd profiler 的 CPU 事件，trace 会缺 GPU 行。
- **CUPTI 负责 GPU 侧**。它和 nsys / ncu 共用同一套底层，多个 profiler 同时开会互相干扰，常见结果是其中一个报错、trace 缺行，或者程序直接挂住。
- **Python 栈是可选开销**。`with_stack=True` 会启用 Python tracer；它需要 CPU activity，且不同版本的行为有差异（见“常见坑”）。
- **`record_shapes=True` 会临时持有张量引用**。官方文档明确提示：这可能阻止依赖引用计数的优化，甚至引入额外拷贝。查 shape 时才开，查完就关。
- **`profile_memory=True` 走分配器事件**，不是 CUDA 的 `cudaMalloc` 全量追踪；外部库自己分配的显存不一定在表里。

## 一次最小采集

先准备一个能重复跑的 step，warmup 放在 profiler 外面，profiler 里只跑稳态。下面用的是 `torch.profiler.profile`；`torch.autograd.profiler.profile` 是旧版入口，功能少且逐步弃用，不要在新代码里混用。

先确认构建里有没有 Kineto 和 CUDA：

```python
import torch

print(torch.__version__, torch.cuda.is_available())
print(torch.profiler.kineto_available())
print(torch.profiler.supported_activities())
```

`supported_activities()` 里只有 `CPU` 时，后面不会出现 CUDA 行；`kineto_available()` 为 `False` 时，Chrome trace 的能力也会受限。这种环境先解决 PyTorch 构建 / 驱动问题，不要继续调参。

```python
# profile_min.py
import torch
from torch.profiler import profile, ProfilerActivity

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.nn.Sequential(
    torch.nn.Linear(1024, 1024),
    torch.nn.GELU(),
    torch.nn.Linear(1024, 1024),
).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
x = torch.randn(32, 128, 1024, device=device)


def step():
    opt.zero_grad(set_to_none=True)
    loss = model(x).square().mean()
    loss.backward()
    opt.step()


for _ in range(3):
    step()

if device == "cuda":
    torch.cuda.synchronize()

activities = [ProfilerActivity.CPU]
if device == "cuda":
    activities.append(ProfilerActivity.CUDA)

with profile(activities=activities) as prof:
    for _ in range(3):
        step()

sort_key = "self_cuda_time_total" if device == "cuda" else "self_cpu_time_total"
print(prof.key_averages().table(sort_by=sort_key, row_limit=15))
prof.export_chrome_trace("trace.json")
```

几个细节：

- **warmup 必须在 profiler 外**。第一次 step 会做 cuBLAS / cuDNN 的 lazy init、allocator 首次分配、可能还有 autotune；这些事件混进来，表的第一行经常是初始化而不是模型。
- **warmup 后同步一次**。`torch.cuda.synchronize()` 保证进入 profiler 时队列已经清空，不会把 warmup 的尾部 kernel 算到采集窗口里。profiler 退出时会自己同步，里面不用再手动加。
- **3 步只是下限**。真实模型建议 5–10 步，且每步形状一致；如果每步 shape 都变，先固定输入，或者用 `group_by_input_shape` 分开看。
- **不要一上来加 `record_shapes` / `with_stack` / `profile_memory`**。第一轮只回答“哪个 op 最贵”，这三个选项都会改变开销和结果的含义，等有假设再加。
- **`activities` 显式写出来**。默认值会包含当前可用的设备，但显式写 CPU + CUDA 能避免“以为采了 GPU、其实只采了 CPU”的误判。
- **CPU-only 环境**去掉 `ProfilerActivity.CUDA`，`sort_by` 换成 `self_cpu_time_total`；表里不会出现 CUDA 列，这是正常的。

`key_averages()` 的输出结构大致如下（列会随 options 增减，这里只示意）：

```text
Name                    Self CPU %   Self CPU   CPU total %   CPU total   CPU time avg   Self CUDA   Self CUDA %   CUDA total   CUDA time avg   # of Calls
aten::mm                      ...          ...           ...          ...           ...          ...           ...          ...           ...          ...
aten::linear                  ...          ...           ...          ...           ...          ...           ...          ...           ...          ...
cudaLaunchKernel              ...          ...           ...          ...           ...          ...           ...          ...           ...          ...
...
Self CPU time total: ...
Self CUDA time total: ...
```

### 每一列是什么意思

| 列 | 含义 | 用法 |
| :--- | :--- | :--- |
| `Name` | 聚合 key，通常是 op 名、kernel 名或用户标注 | 先扫一眼有没有 `cudaDeviceSynchronize`、`aten::cat`、`aten::to` 这种“非计算”大头 |
| `Self CPU %` | 该事件自身 CPU 时间占所有事件 self CPU 总和的比例 | 找 CPU 侧热点；不含子事件 |
| `Self CPU` | 该事件自身的 CPU 时间，单位 us / ms / s | 排序看谁在 host 上最花时间 |
| `CPU total %` | 含子事件的 CPU 时间占比 | 可能超过 100%，因为子事件被重复计入 |
| `CPU total` | 含子事件的 CPU 时间 | 看一个模块 / 阶段的整体 host 成本；不要拿它求和 |
| `CPU time avg` | `CPU total / # of Calls` | 判断是“单次很贵”还是“调用次数很多” |
| `Self CUDA` | 该事件自身对应的 GPU 时间，不含子事件 | GPU 侧排名的第一列 |
| `Self CUDA %` | 占所有事件 self GPU 时间总和的比例 | 判断 GPU 时间是否集中在少数 op / kernel |
| `CUDA total` | 含子事件的 GPU 时间 | 看一个 CPU op / 标注覆盖的整体 GPU 工作 |
| `CUDA time avg` | `CUDA total / # of Calls` | 单次 kernel 的平均耗时，用于发现“很多小 kernel” |
| `CPU Mem` / `Self CPU Mem` | 含子事件 / 自身的 CPU 内存分配字节数 | 仅在 `profile_memory=True` 时出现 |
| `CUDA Mem` / `Self CUDA Mem` | 含子事件 / 自身的 GPU 内存分配字节数 | 仅在 `profile_memory=True` 时出现；释放可能是负数 |
| `# of Calls` | 聚合了多少次调用 | 和 avg 一起看：次数多、单次小，通常是 launch-bound |
| `Input Shapes` | 每次调用的输入形状 | 仅在 `record_shapes=True` 时出现 |
| `Source Location` | 调用栈 / 源码位置 | 仅在 `with_stack=True` 时出现 |
| `Total FLOPs` | 估算的浮点运算量 | 仅在 `with_flops=True` 时出现，只覆盖 matmul / 2D conv |

表头里的 `CUDA` 会随设备变成 `XPU`、`MTIA` 等；PyTorch 2.15 起部分字段也从 `*_cuda_*` 改成了 `*_device_*`，比如 `self_device_time_total`。直接访问老字段会有弃用提示，表里用 `sort_by="self_cuda_time_total"` 通常仍然兼容；写新代码时优先用 `self_device_*`。

### 四个容易读错的点

**1. `CPU total` 是包含子事件的，不能把整列相加。**

`aten::linear` 的 `CPU total` 包含它调用的 `aten::addmm`、`aten::mm`；这些子事件的 CPU 时间被重复计入。把表里所有 `CPU total` 加起来，会得到远大于墙钟的数字。要比较“谁最花时间”，优先看 `Self CPU` / `Self CUDA`；要看一个阶段的总成本，才用 `total`。

**2. CPU op 和它对应的 kernel 可能各占一行，GPU 时间会重复。**

`aten::mm` 行会带上它关联 kernel 的 GPU 时间；同一个 kernel 又会以 `ampere_sgemm_*` / `sm90_xmma_*` 这样的名字单独出现。两行都在表里时，GPU 时间看起来被算了两遍。要回答“哪个 op 贵”，看 CPU op 的 `CUDA total`；要回答“哪个 kernel 贵”，看 kernel 行的 `Self CUDA`。不要把两边相加。

**3. 百分比的分母是所有事件的 self 时间之和，不是墙钟。**

所以 `CPU total %` 可以超过 100%，多个 stream 并发时 self 时间的总和也可能大于实际墙钟。百分比适合做相对排序，不适合直接换算成“占了总时长的多少”。

**4. 表的 `Self CUDA time total` 不等于 step 的墙钟。**

它只是所有 GPU 事件 self 时间的和。多个 stream 有重叠、CPU 和 GPU 有并行、或者 profiler 只采了窗口的一部分时，它和 `time.perf_counter()` 测到的 step 时间不是一回事。要判断端到端，回到 trace 的 step 边界或外部 benchmark。

## 先看表，再看 trace

统计表适合做“排名”，trace 适合回答“为什么”。一个高效的顺序是：

1. 按 `Self CUDA` 排序，找 GPU 上最贵的 5–10 个 op / kernel。
2. 按 `Self CPU` 排序，看 host 侧是不是被调度、拷贝或同步拖住。
3. 看 `# of Calls` 和 `CPU time avg` / `CUDA time avg`：次数多且单次小，往往是 launch-bound；单次很大，才值得进 ncu。
4. 打开 trace，看这些 op 前后有没有 GPU 空白、stream 等待、`cudaDeviceSynchronize` 或通信。

```python
ka = prof.key_averages()

print(ka.table(sort_by="self_cuda_time_total", row_limit=20))
print(ka.table(sort_by="self_cpu_time_total", row_limit=20))

# 同一个 op 在不同 shape 下分开统计（需要 record_shapes=True）
print(
    prof.key_averages(group_by_input_shape=True)
    .table(sort_by="self_cuda_time_total", row_limit=20)
)

# 只看顶层事件，去掉嵌套的 aten 子调用
print(
    ka.table(
        sort_by="self_cuda_time_total",
        row_limit=20,
        top_level_events_only=True,
    )
)
```

几个典型的“表里长什么样 → 先查什么”：

| 表里的信号 | 可能的原因 | 下一步 |
| :--- | :--- | :--- |
| 按 `Self CPU` 排序时顶部是 `cudaLaunchKernel`、`cudaDeviceSynchronize` | CPU launch 开销、同步等待、代码里有 `.item()` / `.cpu()` / 频繁 sync | 看 trace 的 CPU 行和 GPU 空白；考虑 CUDA Graph / `torch.compile` |
| 顶部是 `aten::cat`、`aten::copy_`、`aten::to` | 布局转换、dtype 转换、拼接导致的额外拷贝 | 看 `Input Shapes`，合并拷贝或改内存布局 |
| 顶部是 `aten::mm` / `aten::linear`，且 kernel 行也很贵 | GEMM 本身占主导 | 用 ncu 看 SOL / occupancy / roofline，判断是否值得优化 |
| 顶部是大量 elementwise / reduction kernel | 小算子多，GPU 利用率被 launch 和访存切碎 | 看 `# of Calls`；考虑融合、`torch.compile` 或改算子 |
| `Self CUDA` 不高，但 step 墙钟很高 | CPU、IO、通信或 GPU 空转 | 切 nsys；torch.profiler 的 trace 只能看到进程内的一部分 |
| 同一个 op 在 `Input Shapes` 里出现很多不同形状 | 动态 shape 导致重复编译 / 重复选 kernel | 固定 shape，或用 `group_by_input_shape` 找最贵的那一种 |

表只能告诉你“谁排第一”，不能告诉你“第一名是否已经足够快”。`aten::mm` 排第一可能是正常的；它是否还有优化空间，要 ncu 的 SOL 和 roofline 才能回答。

## 用 schedule 只采稳态

真实训练循环里，前几步可能包含初始化、autotune、allocator 扩容和 CUDA Graph capture；直接 `with profile(): for step in range(N): ...` 会把它们全录进去。`torch.profiler.schedule` 把采集窗口切成 wait / warmup / active 三段，只在 active 段落盘。

```python
import torch
from torch.profiler import profile, ProfilerActivity, schedule

activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]


def trace_handler(p):
    p.export_chrome_trace(f"/tmp/trace_step{p.step_num}.json")
    print(
        p.key_averages()
        .table(sort_by="self_cuda_time_total", row_limit=15)
    )


with profile(
    activities=activities,
    schedule=schedule(wait=2, warmup=2, active=3, repeat=1),
    on_trace_ready=trace_handler,
) as prof:
    for step in range(20):
        train_step(step)
        prof.step()
```

在这个例子里，20 次 `prof.step()` 对应的动作是：

| 调用 `prof.step()` 的迭代 | 动作 | 是否记录 |
| :--- | :--- | :--- |
| 0–1 | `NONE` | 不准备、不记录 |
| 2–3 | `WARMUP` | 准备 profiler，但不保留事件 |
| 4–6 | `RECORD` → `RECORD_AND_SAVE` | 记录；第 6 步结束时触发 `on_trace_ready` |
| 7 之后 | `NONE` | `repeat=1`，本窗口结束 |

几个容易踩错的点：

- **`prof.step()` 每个 iteration 只能调一次，且要放在 iteration 末尾**。漏调，schedule 不前进，采集窗口会和实际 step 错位，甚至整段没有记录；多调，窗口同样对不上。
- **`wait` 不是 warmup**。`wait` 阶段完全不记录；`warmup` 阶段会准备 trace，但事件不落盘。真正的数据从 `active` 开始。
- **`warmup=0` 会有警告**，且结果容易被第一次采集的冷启动污染。至少给 1 步。
- **`repeat=1` 只采一个窗口**。想周期性采，用 `repeat=0`，在 `on_trace_ready` 里按 `p.step_num` 命名文件；不要等 context 结束后再导出多个 cycle。
- **context 结束后 `export_chrome_trace()` 只保留最后一个 cycle**。多 cycle 必须在 `on_trace_ready` 里逐个导出；或者设 `acc_events=True` 累积事件，但这会改变内存和表的口径。
- **`skip_first` 适合跳过启动阶段**，比如前 10 步做 autotune；它和 `wait` 是两段时间，别混用。

如果想直接进 TensorBoard，把 `on_trace_ready` 换成：

```python
from torch.profiler import tensorboard_trace_handler

with profile(
    activities=activities,
    schedule=schedule(wait=2, warmup=2, active=3, repeat=0),
    on_trace_ready=tensorboard_trace_handler("./log", use_gzip=True),
) as prof:
    ...
```

然后 `tensorboard --logdir ./log`。如果 TensorBoard 里没有 “PyTorch Profiler” 页签，装一下 `pip install torch-tb-profiler`；文件默认是 `<hostname>_<pid>.<timestamp>.pt.trace.json[.gz]`，分布式训练时用 `worker_name=f"rank{rank}"` 让每个 rank 的文件名可区分。

## 打开 trace：Perfetto、Chrome 和 TensorBoard

`prof.export_chrome_trace("trace.json")` 写出的是 Chrome Trace Event JSON，可以直接拖进：

- **[Perfetto UI](https://ui.perfetto.dev)**：大 trace 更稳，支持搜索、pin、flow 箭头和 SQL（trace_processor）。日常首选。
- **`chrome://tracing`**：老入口，功能少一些；大文件容易卡。
- **TensorBoard**：`tensorboard --logdir ./log`，看 Overview / Operator / Kernel / Trace / Memory 等视图；适合和训练日志放在一起对比。

打开后先认车道：

- **CPU 线程**：每个线程一行，能看到 Python 调用、`aten::*`、`cudaLaunchKernel`、锁等待。
- **GPU stream**：每个 stream 一行，kernel 是其中的块；同一个 kernel 在不同 stream 上可能并行。
- **flow 箭头**：CPU 上的 launch 指向 GPU 上对应的 kernel；没有箭头通常意味着异步 / 图捕获 / correlation 缺失。

读 trace 的顺序建议：

1. **先找最宽的 GPU 块**：它通常就是表里排名第一的 kernel。点开看 `args`，可能带 grid / block / shared memory / registers 等 launch 信息；不同版本和后端暴露的字段不一样。
2. **看它前面的 CPU 行**：是 Python 在拼 batch、在等锁、在 `cudaDeviceSynchronize`，还是在发下一个 launch。
3. **看 GPU 行的空白**：空白比 kernel 本身更常是端到端瓶颈。空白期间 CPU 在干什么，决定了是补流水、改 batch，还是减少同步。
4. **看 stream 之间的重叠**：计算和通信是否真的并行，还是 all-reduce 结束后才发下一层。
5. **看 step 边界**：用 `record_function("step")` 或 schedule 生成的 `ProfilerStep*` 对齐，别把多个 step 的噪声混在一起。

trace 也是普通 JSON，可以脚本化：

```python
import json

with open("trace.json") as f:
    trace = json.load(f)

# 找 GPU 上最长的 10 个 kernel
kernels = [
    e for e in trace["traceEvents"]
    if e.get("cat") in ("kernel", "Kernel", "gpu_op")
]
for e in sorted(kernels, key=lambda x: x.get("dur", 0), reverse=True)[:10]:
    print(f'{e["dur"] / 1000:8.3f} ms  {e["name"]}')
```

如果 trace 太大，优先做减法：缩短 `active`、关掉 `with_stack`、用 `use_gzip=True`，或者用 2.15 的 activity 过滤（例如 `activities=[{ProfilerActivity.CUDA: ["CONCURRENT_KERNEL", "GPU_MEMCPY"]}]`）只留关心的 GPU 活动。`post_processing_timeout_s` 可以给超大 trace 的后处理设上限，避免卡在解析阶段。

## 用 record_function 把 trace 变成阶段

只靠 `aten::*`，trace 是一堆算子；加上阶段标注，才能回答“prefill 和 decode 各占多少”“forward 和 backward 谁更贵”。

```python
import torch

with torch.profiler.record_function("train/data"):
    batch = next(loader)

with torch.profiler.record_function("train/forward"):
    loss = model(batch)

with torch.profiler.record_function("train/backward"):
    loss.backward()

with torch.profiler.record_function("train/optimizer"):
    opt.step()
    opt.zero_grad(set_to_none=True)
```

`record_function` 生成的是 `user_annotation` 事件；把上面的代码放进 `profile()` 打开的 `with` 块里，标注才会出现在 trace / 表里。在表里它通常表现为一行带阶段名的记录：

- `CPU total` 包含阶段内所有子 op，适合看阶段总成本；
- `Self CPU` 只是标注本身的开销，通常很小；
- `CUDA total` 汇总阶段内关联到的 GPU 时间；
- `Self CUDA` 对纯 CPU 标注通常是 0，别用它来排阶段。

命名建议：

- **稳定、低基数**：`train/forward`、`train/backward`、`prefill`、`decode`。不要把 step 号、请求 ID、shape 拼进名字，否则表会被拆成几千行。
- **嵌套要有意义**：`decode/layer_0` 这种可以保留；过深的嵌套会让 trace 难读。
- **阶段边界放在真正想比较的地方**：比如 `prefill` 包住整个 prefill 调用，而不是只包住最后一个 matmul。

`record_function` 在 profiler 关闭时是 no-op，留在代码里不会产生 trace；但每步都调用仍有函数调用成本，极热的 inner loop 里不要放。

如果你已经在用 NVTX 给 nsys 打范围，不必为了 `torch.profiler` 重写一遍：`torch.profiler` 优先用 `record_function`，NVTX 范围留给 nsys 的时间线。两者都能用，但排查时一次只开一个 profiler，别让两套标注互相干扰。

## shape、调用栈、模块和 FLOPs

这四个选项都不是默认开的，因为每一个都会改变开销或结果口径。

### `record_shapes`：看清是哪种 shape 慢

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    for _ in range(3):
        step()

print(
    prof.key_averages(group_by_input_shape=True)
    .table(sort_by="self_cuda_time_total", row_limit=20)
)
```

开了之后，表里多出 `Input Shapes` 列；`group_by_input_shape=True` 会把同一个 op 的不同输入形状拆成多行。典型用途：

- 动态 shape 的推理服务：找出哪个 batch / seq len 的组合最贵；
- 训练时不同 micro-batch 或 packing 方式导致的 shape 变化；
- `torch.compile` 反复编译：同一个 `aten::mm` 出现很多 shape，说明 specialization 在增加。

代价是 profiler 会临时持有输入张量的引用，可能阻止引用计数相关的优化，甚至增加拷贝。查完就关；不要把它常驻在训练循环里。

### `with_stack`：把时间落到源码行

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    with_stack=True,
) as prof:
    step()

print(
    prof.key_averages(group_by_stack_n=5)
    .table(sort_by="self_cuda_time_total", row_limit=20)
)
prof.export_stacks("/tmp/stacks.txt", metric="self_cuda_time_total")
```

`with_stack=True` 会记录 Python 调用栈，表里出现 `Source Location` 列；`export_stacks` 输出折叠栈，可以喂给 FlameGraph：

```bash
git clone https://github.com/brendangregg/FlameGraph
./FlameGraph/flamegraph.pl --title "CUDA time" /tmp/stacks.txt > stacks.svg
```

注意：

- **`with_stack` 需要 activities 里有 CPU**；只采 CUDA 时不会启用 Python tracer。
- **2.15 上如果栈为空**，检查是否需要 `experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)`；这是版本相关行为，先用 `export_stacks` 的文件大小验证。老版本通常直接 `with_stack=True` 即可。
- **开销高**：每个 op 都要走 Python 栈展开，短窗口采集，不要和 `record_shapes`、`profile_memory` 一起无脑开。
- **栈只回答“谁调用”**，不回答“为什么慢”；它适合把 GPU 时间归因到代码，不适合替代 ncu。

### `with_modules`：不要在新代码里用

`with_modules=True` 在 2.15 已标记弃用：它只对 TorchScript 模型有意义，eager 模式下没有效果。很多老教程还在用，是因为它曾经能把 `nn.Module` 层级写进 trace。新代码用 `with_stack=True`（eager 下会记录 module 事件）或 `record_function` 手工标注模块边界。

### `with_flops`：估算 FLOPs，但只覆盖两类算子

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    with_flops=True,
) as prof:
    step()
```

表里会多出 `Total FLOPs`，只对 matmul 和 2D convolution 用公式估算。可以拿它算一个粗略的 achieved TFLOP/s：

```text
TFLOP/s ≈ Total FLOPs / Self CUDA 秒数 / 1e12
```

但 attention、fused MLP、自定义 kernel 通常没有可靠的 FLOPs 公式，这个数只能当量级参考；真要判断 compute-bound，还是用 ncu 的 SOL。

## 显存：`profile_memory` 和 memory snapshot

`profile_memory=True` 让 profiler 记录 PyTorch caching allocator 的 alloc / free，并按 op 聚合：

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,
) as prof:
    for _ in range(3):
        step()

print(
    prof.key_averages()
    .table(sort_by="self_cuda_memory_usage", row_limit=20)
)
```

表里会出现 `CPU Mem` / `Self CPU Mem` / `CUDA Mem` / `Self CUDA Mem`。要记住：

- **这是分配量，不是峰值**。一行显示 2 GB，不代表当时驻留了 2 GB；释放会以负数出现，累计值可能抵消。
- **`CUDA Mem` 含子事件，`Self CUDA Mem` 才是该 op 自己分配的**；和 CPU total 一样，不要直接求和。
- **只覆盖 PyTorch 的 allocator**。外部库用 `cudaMalloc` 直接分配的显存、cuBLAS workspace 等，不一定会出现在这张表里。
- **开销不低**。每个 alloc / free 都要记录；只在查显存问题时开，且尽量缩短窗口。

如果要查泄漏、峰值、碎片或“到底哪一行代码分配了这块显存”，用 memory snapshot 更合适：

```python
import torch

torch.cuda.memory._record_memory_history(
    enabled="all",
    context="all",
    stacks="python",
)

# 跑可疑的几步 / 几个请求
...

torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
torch.cuda.memory._record_memory_history(enabled=None)
```

把 `.pickle` 拖到 [PyTorch Memory Viz](https://pytorch.org/memory_viz)，能看到分配时间线、峰值、每个 block 的调用栈。峰值还可以直接用：

```python
torch.cuda.reset_peak_memory_stats()
# run
print(torch.cuda.max_memory_allocated() / 1024**2, "MiB")
```

2.15 起 `prof.export_memory_timeline()` 已弃用，官方建议走上面的 `_record_memory_history` + `_dump_snapshot` 路线；老版本可以继续用 timeline，但不要把它的图和 snapshot 混着解释。

## 分布式、`torch.compile` 与推理

### 分布式训练

每个 rank 通常各跑一个 profiler，各自写文件；不要指望一个进程的 trace 自动包含所有 rank 的 GPU：

```python
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

rank = dist.get_rank()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=2, warmup=2, active=3, repeat=0),
    on_trace_ready=tensorboard_trace_handler(
        "./log", worker_name=f"rank{rank}", use_gzip=True
    ),
) as prof:
    for step in range(20):
        train_step(step)
        prof.step()
```

观察点：

- **NCCL kernel** 会以 `ncclDevKernel_*` 出现；`c10d::allreduce_` 之类的 CPU op 是 host 侧调用。看它们和计算 kernel 是否重叠，比看单次 all-reduce 的绝对时间更有意义。
- **每个 rank 的窗口要一致**：同一个 step 区间、同一个 batch 大小；否则 trace 对不齐。
- **TensorBoard 插件会按 rank 分组**；文件名里的 `worker_name` 要能区分 rank。
- **多机 / IB / NIC 的问题**，torch.profiler 看不到完整网络栈，切 nsys；NCCL 本身的行为可以用 NCCL tests 或 nsys 的 NVTX 范围确认。

单进程多卡时，先在 trace 里确认到底出现了几个 device / stream；不同版本对多卡的支持程度不一样，最稳的做法仍是每个 rank 单独 profile。

### `torch.compile` 和 CUDA Graph

`torch.compile` 之后，trace 里会出现 `triton_*` 生成的 kernel，原来的 `aten::*` 可能被融合掉。建议在编译后的调用外面加 `record_function("compiled_step")`，否则很难把生成 kernel 和 Python 代码对应起来。

CUDA Graph 会把多次 kernel launch 折叠成一次 graph launch：CPU 侧的 launch 开销在 trace 里会明显变小，但逐 kernel 的时间线和 correlation 也会变。要判断 graph 内部某个 kernel 的硬件指标，还是用 nsys / ncu；torch.profiler 更适合回答“graph 外层的 Python / 调度还剩多少”。

### LLM 推理：prefill 和 decode 分开采

prefill 和 decode 的瓶颈完全不同：prefill 偏 GEMM / attention 的吞吐，decode 偏小 kernel、KV cache 访存和 launch 开销。用 `record_function` 分开包（下面是示意，`prefill` / `decode` 按实际框架的接口替换）：

```python
with torch.profiler.record_function("prefill"):
    out = model.prefill(prompt)

with torch.profiler.record_function("decode"):
    for _ in range(n_decode):
        out = model.decode(out)
```

注意事项：

- **跳过第一个请求**：CUDA Graph capture、kernel autotune、KV cache 首次分配可能都在第一个请求里。用 schedule 或手动 warmup 跳过。
- **固定 batch / seq len**：否则 prefill 的 shape 每次都不同，表会被拆散。
- **框架内部的调度不可见**：vLLM / SGLang 的 scheduler、队列等待、continuous batching 不在 torch 的 op 时间线里；这些要看框架自己的 metrics 或 nsys。torch.profiler 能看到的是模型 op 和 kernel。
- **decode 的 Python 开销占比高**：如果表里 `cudaLaunchKernel`、`aten::*` 的 self CPU 很大而 GPU 利用率低，考虑 CUDA Graph、算子融合或减少每 token 的 Python 调用。

## 开销与常见坑

- **把 profiler 当 benchmark**：event-based tracing 会改变时序，CUPTI 也有开销。采集时的 step 时间、tokens/s 不能直接当生产数字；要对比就固定输入、固定窗口，开关 profiler 各跑一次。
- **同时开多个 profiler**：`torch.profiler`、nsys、ncu 都走 CUPTI。常见报错是 profiler 已激活、CUPTI 初始化失败或 trace 缺行；一次只开一个，切换前重启进程。
- **`record_shapes` 导致变慢 / OOM**：它会持有输入张量的引用，可能阻止优化并增加显存。只在定位 shape 问题时短窗口开。
- **`with_stack` 没有栈**：先确认 activities 含 CPU；2.15 上可能需要 `_ExperimentalConfig(verbose=True)`；`export_stacks` 文件为空就是最直接的信号。
- **`with_modules` 没效果**：2.15 已弃用，eager 下不记录；改用 `with_stack` 或 `record_function`。
- **trace 是空的 / 只有最后一个 cycle**：检查 `prof.step()` 是否每步调用、`repeat` 是否已经结束、`on_trace_ready` 是否导出；context 结束后再 `export_chrome_trace()` 只保留最后一个 cycle。
- **表里 GPU 时间被算了两遍**：CPU op 行和 kernel 行都带 GPU 时间，别相加；选一个视角。
- **顶部是 `cudaDeviceSynchronize`**：代码里可能有 `.item()`、`.cpu()`、`torch.cuda.synchronize()`、或者 dataloader 的同步；先消同步，再谈 kernel 优化。
- **看不到 CUDA 行**：确认 `torch.cuda.is_available()`、`kineto_available()`、activities 里包含 CUDA；容器 / 驱动 / CUPTI 版本不匹配也会导致 GPU 事件缺失。
- **DataLoader 子进程不在 trace 里**：worker 是独立进程，torch.profiler 只覆盖当前进程；要一起看就用 nsys，或单独 profile worker。
- **Jupyter 里 profiler 卡住**：上下文跨 cell、异常没退出、或同时开了别的 profiler；重启 kernel，确保 `with` 块完整执行。
- **版本差异**：2.15 改了 `self_device_*` 命名、弃用 `with_modules` 和 `export_memory_timeline`；老脚本升级后先看弃用警告，别直接套旧字段。

## 一个可复用的工作流

| 步骤 | 做什么 | 产出 |
| :--- | :--- | :--- |
| 1. 明确问题 | 是 op 慢、shape 慢、CPU launch、显存，还是通信 | 一个可以用“是 / 否”回答的假设 |
| 2. 固定输入 | 固定 batch / seq len / dtype，warmup 3–5 步 | 可重复的 step |
| 3. 最小采集 | CPU + CUDA，不带额外选项，采 3–5 步 | 统计表 + trace |
| 4. 表里排名 | 分别按 `Self CUDA`、`Self CPU` 排序 | top op / kernel / 阶段 |
| 5. 看 trace | 最大 GPU 块、前后 CPU、GPU 空白、stream 重叠 | 瓶颈类型：计算 / 访存 / launch / 同步 / 通信 |
| 6. 定向采集 | 按需加 `record_shapes`、`with_stack`、`profile_memory` | 能解释现象的细节 |
| 7. 改一个变量 | 只改 tiling / 融合 / batch / 同步中的一项 | 可对比的新结果 |
| 8. 复测 | 同样窗口 + 外部 benchmark / nsys | 局部指标和端到端收益 |

什么时候切工具：

| 现象 | 下一步 |
| :--- | :--- |
| 已锁定某个 kernel，想知道它离上限多远 | [ncu](/posts/tools/ncu/) 看 SOL、occupancy、warp stall、roofline |
| GPU 行大段空白、CPU 等 IO / 锁 / 通信 | [nsys](/posts/tools/nsys/) 看系统时间线和 CUDA API |
| 小算子多、launch 开销占主导 | `torch.compile` / CUDA Graph，再用 nsys 对比 |
| 显存泄漏、峰值、碎片 | memory snapshot，不是 `profile_memory` 表 |
| 只想比两个 kernel 实现的微基准 | `torch.utils.benchmark`，不要用 profiler 的绝对时间 |

## 选项速查

| 需求 | 参数 / 方法 | 主要代价 |
| :--- | :--- | :--- |
| 看 op / kernel 排名 | 默认 `key_averages()` | 低 |
| 看某个 op 在不同 shape 下的差异 | `record_shapes=True` + `group_by_input_shape=True` | 持有张量引用，可能增加拷贝 |
| 看调用栈 / 源码行 | `with_stack=True` + `export_stacks()` | Python 栈展开，开销高 |
| 看显存分配 | `profile_memory=True` | 分配器事件，开销中高 |
| 估算 matmul / conv 的 FLOPs | `with_flops=True` | 只覆盖两类算子，公式估算 |
| 只采稳态 | `schedule(wait, warmup, active)` + `prof.step()` | 需要改循环 |
| 多 cycle 汇总 | `on_trace_ready`（逐 cycle 导出）或 `acc_events=True`（事件累积，主要影响表） | 文件多 / 内存增长 |
| 导出可视化 | `export_chrome_trace()` / TensorBoard handler | trace 可能很大 |
| 看分布式通信 | 每 rank 单独 profiler + `worker_name` | 需要对齐窗口 |

最小命令汇总（GPU 环境；CPU-only 时去掉 `ProfilerActivity.CUDA`）：

```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
) as prof:
    for _ in range(3):
        step()

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
prof.export_chrome_trace("trace.json")
```

## 参考

- [torch.profiler API 文档](https://docs.pytorch.org/docs/stable/profiler.html)：`profile`、`schedule`、`tensorboard_trace_handler` 的完整参数和注意事项
- [PyTorch Profiler Recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)：官方入门示例，含 Chrome trace 和 TensorBoard 的基本用法
- [Kineto](https://github.com/pytorch/kineto)：torch.profiler 的 C++ 后端
- [torch-tb-profiler](https://pypi.org/project/torch-tb-profiler/)：TensorBoard 的 PyTorch Profiler 插件，提供 Overview / Operator / Kernel / Trace / Memory 视图
- [Perfetto UI](https://ui.perfetto.dev)：打开 Chrome trace JSON，支持大文件和 SQL 查询
- [PyTorch Memory Viz](https://pytorch.org/memory_viz)：memory snapshot 的可视化
- [CUDA Memory Management](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html)：caching allocator、峰值、碎片和 snapshot 的背景
- [CUPTI 文档](https://docs.nvidia.com/cupti/)：CUDA 侧事件采集的底层接口
- [Holistic Trace Analysis](https://github.com/facebookresearch/HolisticTraceAnalysis)：对分布式 PyTorch trace 做聚合分析
- [torch.utils.benchmark](https://docs.pytorch.org/docs/stable/benchmark_utils.html)：低开销微基准，适合验证 profiler 的结论

和 Kernel 分类的分工：[CUDA 手册](/posts/kernel/cuda/)、[Triton 手册](/posts/kernel/triton/)、[TileLang 手册](/posts/kernel/tilelang/) 讲算子内部怎么写，[DeepGEMM](/posts/kernel/deepgemm/) 讲生产库怎么组织；这篇讲怎么在 PyTorch 框架层把时间归因到 op、shape 和阶段。和 [GPU Architecture](/posts/gpu-arch/intro/) 的分工：那边解释硬件为什么这样设计，这边用框架事件决定下一步该看哪块硬件指标。

需要系统时间线时接 [nsys 手册](/posts/tools/nsys/)，需要单 kernel 的硬件计数器时接 [ncu 手册](/posts/tools/ncu/)；三者一次只开一个。
