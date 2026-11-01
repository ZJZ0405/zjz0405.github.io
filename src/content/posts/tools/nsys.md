---
title: "Nsight Systems 手册：从系统时间线定位 GPU 瓶颈"
published: 2026-11-01T23:55:00+08:00
description: "NVIDIA 系统级 profiler 的实战手册：nsys 的采集参数、NVTX 标注、Python/PyTorch 与多卡场景、stats / SQLite 报告，以及怎么从时间线判断 GPU 空转、串行化和通信重叠。"
tags: [Nsight Systems, Profiling, CUDA, GPU, 性能优化]
category: Tools
draft: false
---

模型训练或推理慢，第一反应常是某个 kernel 写得不够快。但实际时间经常花在别处：GPU 在等 CPU 发下一批 kernel，两个 stream 本可以重叠却串行，NCCL all-reduce 结束后才开始下一层计算，数据加载把 step 切成两半。单个 kernel 的指标再漂亮，也解释不了这些空白。

Nsight Systems（命令行是 `nsys`）是 NVIDIA 的系统级 profiler。它把进程树、CPU 采样、CUDA API、GPU kernel / memcpy、NVTX 范围、OS runtime、MPI/UCX 等事件放进同一条时间线，回答的是“这段时间到底花在哪”，而不是“这个 kernel 的 shared memory 为什么 bank conflict”。

这篇按“定位 → 采集 → 标注 → 读图 → 报告 → 场景”组织。命令以 Nsight Systems 2025.1.3 的 CLI 核对过；参数会随版本增删，实际以机器上的 `nsys profile --help` 为准。

和 Kernel 分类的分工：Kernel 讲算子内部怎么写、怎么用 ncu 看指标；这里讲怎么把整个系统的时间线采下来、读出来。nsys 不交付算子实现，观测对象是进程树、CPU 调度、CUDA API 和 GPU 时间线，所以归在 Tools。

> **五分钟版**
>
> - **先 nsys，后 ncu**：nsys 找时间窗口和瓶颈类型，Nsight Compute 再看单 kernel 的 occupancy、stall 和 roofline。两者别同时开。
> - **最小采集**：`nsys profile -t cuda,nvtx,osrt --stats=true -o report ./app`，得到 `.nsys-rep` 和 `.sqlite`。
> - **一定加 NVTX**：没有阶段标注，时间线只是一堆 `ampere_sgemm_*` / `elementwise_kernel`。用 request / step / forward / decode 这种粗粒度范围。
> - **只采稳态**：用 `--capture-range=nvtx` 或 `cudaProfilerApi` 跳过初始化、warmup 和第一个请求。
> - **先找空白**：GPU 行空着时，看同一时刻 CPU 在 Python、在 `cudaStreamSynchronize`，还是在等锁或 IO。空白比 kernel 本身更常是瓶颈。
> - **报告看三张**：`cuda_gpu_kern_sum` 看 kernel 总时间，`cuda_kern_exec_sum` 看 launch / queue，`nvtx_gpu_proj_sum` 把 GPU 时间映射回阶段。
> - **profiler 会改变程序**：API trace、Python sampling、GPU metrics、`--cuda-memory-usage` 都有开销；对比开关前后的数字，别把采集结果当生产性能。
> - **环境先自检**：先跑 `nsys --version` 和 `nsys status -e`；只有 CUDA API、没有 GPU 行时，先查版本 / 权限 / 容器支持，不要先改代码。

## 它回答什么，不回答什么

把三类工具放在一起看最清楚：

| 工具 | 强项 | 看不到 / 不适合 |
| :--- | :--- | :--- |
| Nsight Systems (`nsys`) | 系统级时间线、CPU-GPU 重叠、多 stream、通信、IO、启动间隔 | 单个 kernel 内部的 stall、occupancy、指令级指标 |
| Nsight Compute (`ncu`) | 单 kernel 的硬件计数器、roofline、source correlation | 长时间线的全局关系；会 replay kernel，改变时序 |
| [PyTorch Profiler](/posts/tools/torch-profiler/) | 算子级耗时、框架 op、显存、Chrome trace | 系统级上下文、跨进程 / IO / OS runtime 较少 |

选择顺序通常是：nsys 看到某个阶段 GPU 空转，或者发现 kernel 之间有大段间隔；定位到具体 kernel 或调度间隙后，如果还想知道 kernel 内部为什么慢，再对那个 kernel 跑 ncu。

三者都依赖 CUPTI（PyTorch Profiler 的 CUDA 后端也是），同时开容易互相干扰，甚至让程序直接报错。一次只开一个。

nsys 自己也能用 `--gpu-metrics-devices` 采 SM / DRAM 等采样型指标，但它按固定频率采样，适合看趋势，不是 ncu 逐 kernel 计数器的替代品。

## 先建立时间线模型

Nsight Systems 的报告可以拆成几条水平轨道：

- **CPU 线程**：每个线程一行，能看到 Python / C++ 函数、`poll`、`fread`、锁等待；开了 CPU sampling 还会有调用栈采样。
- **CUDA API**：host 侧的 `cudaLaunchKernel`、`cudaMemcpyAsync`、`cudaStreamSynchronize`、`cudaMalloc` 等调用。
- **GPU**：kernel、memcpy、memset、CUDA Graph，按 context / stream 展开；多个 stream 是并行轨道。
- **NVTX**：request / step / forward / decode 等逻辑范围，CPU 侧和 GPU projection 都会用到。
- **可选轨道**：OS runtime、MPI / UCX、NIC、GPU metrics、Python GIL。

横向是时间，宽度是持续时间。不要只看“GPU 行有没有颜色”，空白和相邻轨道的因果关系通常更有信息量：

| 现象 | 同一时刻 CPU 在做什么 | 常见原因 |
| :--- | :--- | :--- |
| GPU 长空白 | Python 函数、数据加载、调度器 | host-bound：tokenizer、dataloader、GIL、请求调度 |
| GPU 长空白 | `cudaStreamSynchronize`、`cudaMemcpy` | 显式同步、同步拷贝、错误的 stream 用法 |
| GPU 长空白 | `poll`、`fread`、`ioctl` | 文件 / 网络 IO、日志、检查点 |
| GPU 有 kernel，但多个 stream 不重叠 | 正常 launch | 同 stream 依赖、event / sync、假依赖 |
| kernel 之间只有细小空隙 | 连续 launch | launch overhead、allocator、Python dispatch |
| 最后一个 kernel 后拖一条长空白 | 等待或收尾 | 尾效应、负载不均、同步点 |
| NCCL kernel 与计算交替出现 | 正常 launch | 通信没有和计算重叠，或在等最慢的 rank |

这张表只是入口。比如“CPU 在 Python”不等于 Python 是根因，可能是它在等锁、等 IO，或者在等 GPU 的结果。要沿着调用栈和时间线继续往下看。

## 安装与版本

GUI 和 CLI 是两个使用面：GUI 用来读 `.nsys-rep`，CLI 在目标机上采集。Linux 上可以从 NVIDIA 官网装 `.deb` / `.rpm` / `.run`；CUDA Toolkit 也可能自带一份，但版本不一定和 GUI 匹配。先确认：

```bash
which nsys
nsys --version
nsys status --environment
```

`nsys status -e` 会检查 CPU 采样权限、`perf_event_paranoid`、system-wide 是否可用。容器、WSL2、MIG 或虚拟化环境的能力不一样，先看这个输出，再决定能不能采 GPU metrics / system-wide。

版本匹配也很重要：CLI 要能识别目标机的 CUDA 驱动；GUI 不要低于采集端版本。`.nsys-rep` 是二进制格式，跨大版本可能打不开，团队里最好固定一版。

远程用法很直接：在目标机跑 CLI，把 `.nsys-rep` 拷到有 GUI 的机器，用 `nsys-ui report.nsys-rep` 打开；需要交互式会话时再用 `nsys launch` / `nsys start`。容器里则要保证 GPU 访问和 perf 事件权限；GPU metrics、system-wide sampling 往往还要额外 capabilities（例如 `--cap-add=SYS_ADMIN`），具体按容器文档和集群安全策略来，不要在生产容器里默认开。

## 第一次采集

最小可用命令：

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=process-tree \
  --stats=true \
  --force-overwrite=true \
  --output=report \
  ./app
```

几个参数的含义：

- `--trace`：要采的 API。显式写 `cuda,nvtx,osrt`，避免默认里的 OpenGL 等无关轨道；只看 GPU 时用 `cuda` 就够。
- `--sample=process-tree`：CPU IP / backtrace 采样；不想付采样开销时用 `none`。
- `--stats=true`：采集后导出 `.sqlite` 并跑默认报告。
- `--output`：输出基名，不带扩展名；默认是 `report%n`。
- `--force-overwrite=true`：覆盖同名结果，脚本里常用。

采集结束会得到 `report.nsys-rep` 和 `report.sqlite`。GUI 打开前者，`nsys stats` 和 SQL 用后者。

如果程序是常驻服务，不要从启动开始录。用 NVTX 或 `cudaProfilerApi` 限定窗口：

```bash
nsys profile -t cuda,nvtx \
  -c nvtx -p decode_step \
  --capture-range-end=stop \
  -o decode python serve.py
```

`-c nvtx` 表示收到指定 NVTX range 时才开始采集，`-p` 指定 range 名。`--capture-range-end=stop` 在 range 结束时停止采集但保留进程；默认的 `stop-shutdown` 会结束会话，可能把服务一起关掉。默认只认第一个匹配的 range；要看多个窗口，用 `--capture-range-end=repeat:N`，或者分开采集。

也可以在代码里显式开关：

```python
import torch

torch.cuda.cudart().cudaProfilerStart()
# 只跑稳态的若干 step
torch.cuda.cudart().cudaProfilerStop()
```

```bash
nsys profile -t cuda,nvtx -c cudaProfilerApi \
  --capture-range-end=stop -o steady python serve.py
```

`cudaProfilerStart/Stop` 需要 `--trace` 里包含 CUDA 或 NVTX，否则 trigger 不会生效。

## 参数怎么选

| 目的 | 参数 | 备注 |
| :--- | :--- | :--- |
| 首次摸底 | `-t cuda,nvtx,osrt --stats=true` | API + kernel + OS runtime，信息最全 |
| 低开销 GPU 时间线 | `-t cuda --sample=none --cuda-event-trace=false` | 不采 CPU 采样和 event 依赖 |
| 稳态窗口 | `-c nvtx -p <range> --capture-range-end=stop` | 最推荐的服务采集方式 |
| Python 热点 | `--python-sampling=true` | 有调用栈，但会扰动时序 |
| PyTorch autograd | `--pytorch=autograd-nvtx` | 自动加 autograd NVTX 范围 |
| CUDA Graph | `--cuda-graph-trace=graph` / `node` | `graph` 低开销，`node` 看节点 |
| 通信 | `-t cuda,nvtx,mpi,ucx --mpi-impl=openmpi` | 只在实际使用 MPI / UCX 时加 |
| NIC 指标 | `--nic-metrics=true` | 支持范围有限，且增加开销 |
| GPU 采样指标 | `--gpu-metrics-devices=cuda-visible --gpu-metrics-set=<set>` | 先查设备与 set 支持 |
| fork 子进程 | `--trace-fork-before-exec=true` | 有崩溃 / 死锁风险，别默认开 |
| 长服务 | `--delay` / `--duration` / capture range | 优先用 capture range |

几个容易踩的点：

- **`--cuda-event-trace`**：默认 `auto`；打开后会追踪 CUDA Event 完成，改善 event 关联，但可能制造跨 stream 的假依赖。做重叠分析时先设 `false`；专门查 event 同步时再开。
- **`--cuda-memory-usage=true`**：显著抬高开销，只在查 allocator、OOM 或显存峰值时开。
- **`--osrt-threshold`**：默认 1 us，短调用很多时报告会很大；调到 10 us 能过滤噪音，但可能漏掉短 IO。
- **`--python-sampling=true`**：适合找 tokenizer / dataloader 热点；不要和“生产性能”混为一谈。
- **`--pytorch=autograd-nvtx`**：省手工标注，但范围会比较密，长训练步建议配合 capture range。
- **`--cuda-graph-trace=graph`**：把整个 graph 当一个活动，开销最低；要看 graph 内部节点才用 `node`，但开销明显更高。
- **`--gpu-metrics-devices=help`**：先看哪些 GPU 可采，再用 `--gpu-metrics-set=help` 看可用 set；不是所有卡都支持，采样也会增加开销。
- **`--trace-fork-before-exec=true`**：只在 launcher fork 后立即 exec 的场景需要；帮助里明确说依赖未定义行为，可能崩溃或死锁。
- **`--wait=all`**：默认等整个进程树结束；如果只关心主进程，可用 `primary`，但子进程的 GPU 工作可能漏掉。

## 用 NVTX 把时间线变成阶段

Nsight Systems 自带 NVTX 支持，不需要额外插件；采集时加 `-t nvtx` 即可。

Python 里最方便的是 `torch.cuda.nvtx.range`：

```python
import torch

@torch.no_grad()
def decode_step(model, input_ids):
    with torch.cuda.nvtx.range("decode_step"):
        with torch.cuda.nvtx.range("forward"):
            logits = model(input_ids)
        with torch.cuda.nvtx.range("sample"):
            next_token = logits[:, -1].argmax(dim=-1)
    return next_token
```

`range` 是上下文管理器，底层是 push / pop，异常时会自动配对。

C++ 里可以用 NVTX v3 的 RAII 包装：

```cpp
#include <nvtx3/nvtx3.hpp>

void decode_step(...) {
    nvtx3::scoped_range range{"decode_step"};
    // launch kernels...
}
```

编译时加上 NVTX 头文件路径：

```bash
nvcc -I"$NSYS_ROOT/target-linux-x64/nvtx/include" ...
```

`NSYS_ROOT` 是 Nsight Systems 安装目录（例如 `/opt/nvidia/nsight-systems/2025.1.3`）。NVTX v3 是 header-only；如果项目还在用 `nvToolsExt.h`，则要链接 `-lnvToolsExt`。

几个标注原则：

- **范围名用低基数**：`decode_step`、`prefill`、`all_reduce`，不要拼 request id 或 prompt。名字太散，汇总报告会变成几千行，还容易泄露输入。
- **一个 request / step 一个主范围**，下面只挂你准备行动的阶段；不要每个 token 都开范围。
- **push / pop 必须在同一线程配对**；跨线程或异步用 start / end range，但 `nvtx_gpu_proj_sum` 只统计同线程的范围。
- **范围要包住 launch**，不是只包住 Python 计算；GPU projection 靠 launch 时的 correlation 把 kernel 归到范围里。
- **用 domain 区分框架和业务范围**；范围太多时用 `--nvtx-domain-include` 只看自己的 domain。

## GUI 里先看什么

1. **先框稳态窗口**。把初始化、warmup、第一个请求排除掉，否则平均值会被一次性开销带偏。
2. **扫 GPU 行的空白**。空白出现在哪，就把同一 x 位置的 CPU 线程、CUDA API、OS runtime 对齐看。
3. **判断等待类型**。CPU 在 `cudaStreamSynchronize` / `cudaMemcpy` 是显式同步；在 `poll` / `fread` 是 IO；在 Python 函数里是 host 逻辑；线程 runnable 但没上 CPU，可能是调度或锁。
4. **看 stream 和 event**。多个 stream 本应重叠却排成一条线，检查 event record / wait、`cudaDeviceSynchronize`、默认 stream 语义和 allocator 锁。
5. **看通信**。NCCL kernel 是否和计算 kernel 重叠；all-reduce 是否在 backward 的关键路径上；多卡时是不是只有一张卡在等。
6. **用 GPU projection 汇总**。GUI 里按 NVTX 投影，或跑 `nvtx_gpu_proj_sum`，把 kernel 时间按阶段归并，比在时间线上肉眼加总可靠。
7. **单 kernel 仍然长，转 ncu**。nsys 只能告诉你这个 kernel 占了多少时间，不能告诉你为什么。

一个常见的误判：GPU 行一直有 kernel，不代表 SM 利用率高。一个 memory-bound 的小 kernel 也可以把时间线填满，但 SM 大部分时间在等访存。要看“为什么慢”，还是得回到 ncu 的 stall 和 throughput。

## 命令行报告与 SQLite

GUI 适合探索，命令行适合回归和自动化：

```bash
# 最常用的三张
nsys stats -r cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum report.nsys-rep

# launch / queue / kernel 三段
nsys stats -r cuda_kern_exec_sum report.nsys-rep

# 按 NVTX 阶段汇总 GPU 时间
nsys stats -r nvtx_gpu_proj_sum --filter-nvtx decode_step report.nsys-rep

# 导出逐次 kernel 记录，方便 pandas / awk
nsys stats -r cuda_gpu_trace -f csv -o gpu_trace report.nsys-rep
```

报告名和参数可以用 `nsys stats --help-reports` 查；`--help-reports <name>` 会解释每一列。几个常用报告的分工：

- **`cuda_gpu_kern_sum`**：按 kernel 名聚合 total / avg / median / min / max。注意 `Time` 列是该 kernel 占所有 kernel 时间的比例，不是占应用 wall time；多 stream 重叠时，kernel 时间相加会超过墙钟。
- **`cuda_api_sum`**：CUDA API 的 CPU 时间。`cudaDeviceSynchronize`、同步 `cudaMemcpy`、`cudaMalloc` 排在前面时，先看它们是不是在等 GPU 或分配器。
- **`cuda_kern_exec_sum`**：把一次 launch 拆成 API time、queue time、kernel time。queue time 大不一定是坏事：GPU 正在忙，说明有排队；每个 launch 都立即执行反而可能是 GPU 空闲。
- **`nvtx_gpu_proj_sum`**：按 NVTX 范围汇总 GPU 操作，适合回答 prefill / decode / forward 各占多少。父范围不会减去子范围，所以不要把父子范围的时间直接相加。
- **`osrt_sum`**：`poll`、`ioctl`、`fread` 等系统调用；数据加载、网络、文件读取的线索。
- **`cuda_gpu_trace`**：逐条 kernel 记录，适合自己算分位数、重叠率或画图。

需要更自由的分析时，导出 SQLite：

```bash
nsys export --type=sqlite -o report.sqlite report.nsys-rep
```

`--stats=true` 已经生成 `.sqlite` 时可以跳过。核心表是 `CUPTI_ACTIVITY_KIND_KERNEL`、`CUPTI_ACTIVITY_KIND_RUNTIME`、`NVTX_EVENTS`、`StringIds`；schema 随版本变，先看 `.schema`。查 kernel 总时间的一个例子：

```sql
SELECT COALESCE(s.value, '?') AS kernel,
       COUNT(*) AS calls,
       ROUND(SUM(k.end - k.start) / 1e6, 2) AS total_ms
FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
LEFT JOIN StringIds AS s
  ON s.id = COALESCE(k.demangledName, k.shortName)
GROUP BY 1
ORDER BY total_ms DESC
LIMIT 20;
```

`demangledName` 不一定每版都有；没有就退回 `shortName`。这也是为什么先用 `nsys stats`，需要自定义分析时再写 SQL。

`nsys analyze` 会跑一组规则，默认检查同步 memcpy / memset、同步 API、GPU gap 和 GPU 时间占比：

```bash
nsys analyze report.nsys-rep
```

它适合当体检清单，不适合当结论；规则命中后仍要回时间线看上下文。

## 实战场景

### LLM 推理：先分清 prefill 和 decode

prefill 是大 GEMM，decode 是小 M、kernel 多、launch 和访存占比高，两者不要混在一个窗口里平均。用 `prefill` / `decode_step` 两个 NVTX 范围分开采，decode 只采稳态的几十个 step。

先看 decode step 之间的 GPU 空白：如果 CPU 在 tokenizer、采样、调度器或 Python 循环里，GPU 再快也会被拖住；这类问题换 kernel 没用。TP / PP 场景再看 NCCL kernel 是否和下一层计算重叠；如果 all-reduce 后面总是紧跟一个 gap，检查通信 stream、同步点和 bucket 大小。

开了 CUDA Graph 后，默认 `--cuda-graph-trace=graph` 只会看到整个 graph；要拆节点用 `node`，但会显著增加开销，别拿它测生产吞吐。多进程服务先确认 GPU 工作进程被采到，输出名可以用 `%p` 区分，例如 `-o serve-%p`。

### 训练：一个 step 的时间去哪儿了

用 `step` / `dataloader` / `forward` / `backward` / `optimizer` / `all_reduce` 标注，先看 step 边界的 GPU 空白。dataloader 慢通常表现为每个 step 开始 GPU 空闲、CPU 在 `fread` / `poll` / Python；先解决输入，再谈 kernel。

backward 里的 all-reduce 应该和计算重叠；串行时看 NCCL 的 stream 和计算 stream 之间是不是有 event / sync 依赖，是否被 `cudaDeviceSynchronize` 打断。多机采集加 `-t cuda,nvtx,mpi,ucx` 和 `--mpi-impl=openmpi`，但每 rank 一个大报告很占空间；先采一个 representative rank，确认时钟同步后再做多机对齐。

只采 warmup 之后的一两个 step；优化器 step、梯度累计、checkpoint 都会改变形状。

### 单算子：nsys 和 ncu 怎么接力

先用 nsys 看这个 kernel 的 launch 间隔、queue time、是否和别的 stream 重叠、前后有没有同步。低开销组合是：

```bash
nsys profile -t cuda --sample=none --cuda-event-trace=false -o r ./app
```

只看 GPU 时没必要采 OS runtime 和 Python。确认这个 kernel 值得深挖后，再用 ncu 跑 `--set full` 或指定 metrics；ncu 会 replay，不适合拿来测端到端延迟。

`cuda_kern_exec_sum` 还能看出 launch API 本身是不是瓶颈：如果 queue time 很小、kernel 很短、launch 很多，先考虑融合或 CUDA Graph。

## 常见坑

- **把 kernel 时间占比当墙钟**：`cuda_gpu_kern_sum` 的百分比只占 kernel 时间，不是应用总时间；多 stream 并行时也不能直接相加。
- **把 GPU 行有颜色当高利用率**：kernel 驻留不等于 SM 忙，可能只是在等访存；occupancy 和 stall 要看 ncu。
- **忽略 profiler 开销**：API trace、Python sampling、GPU metrics、显存追踪都会改变时序；用同一套参数做前后对比，别拿采集时的吞吐当生产数据。
- **采集窗口太大**：几十分钟的常驻服务会生成巨大的报告，还会拖慢 IO；用 capture range、`--duration` 或 NVTX 过滤。
- **NVTX 太细或跨线程**：每个 token 一个范围会让报告爆炸；push / pop 跨线程会得到不完整的范围。
- **同时开多个 profiler**：nsys、ncu、PyTorch Profiler 都走 CUPTI，容易互相干扰；一次只开一个。
- **环境或权限问题**：只有 CUDA API、没有 GPU 行时，先跑 `nsys status -e`，检查驱动版本、`perf_event_paranoid`、容器 capabilities 和 WSL2 / MIG 支持。
- **版本不匹配**：采集端和 GUI 跨大版本可能打不开报告；固定一版，或者先用 `nsys stats` 在采集端导出。
- **报告泄露信息**：`.nsys-rep` 里有命令行、路径、kernel 名和 NVTX 文本；对外分享前先过滤或脱敏。

## 速查表

```bash
# 版本与环境
nsys --version
nsys status -e

# 基础采集
nsys profile -t cuda,nvtx,osrt --stats=true -o r ./app

# 低开销，只看 GPU
nsys profile -t cuda --sample=none --cuda-event-trace=false -o r ./app

# 只采某个 NVTX 窗口
nsys profile -t cuda,nvtx -c nvtx -p decode_step \
  --capture-range-end=stop -o r python app.py

# 报告
nsys stats -r cuda_gpu_kern_sum,cuda_kern_exec_sum,nvtx_gpu_proj_sum r.nsys-rep
nsys stats --filter-nvtx decode_step -r nvtx_gpu_proj_sum r.nsys-rep
nsys analyze r.nsys-rep
nsys export --type=sqlite -o r.sqlite r.nsys-rep
```

## 参考

- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [Nsight Systems Installation Guide](https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html)
- [Nsight Systems Release Notes](https://docs.nvidia.com/nsight-systems/ReleaseNotes/index.html)
- [NVTX documentation](https://docs.nvidia.com/nvtx/index.html)
- [Nsight Compute](https://developer.nvidia.com/nsight-compute)
- [CUDA 手册：性能分析与优化清单](/posts/kernel/cuda/)
- [Kernel 分类导言](/posts/kernel/intro/)
