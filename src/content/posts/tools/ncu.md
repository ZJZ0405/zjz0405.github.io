---
title: "Nsight Compute 使用手册：用 ncu 定位 Kernel 瓶颈"
published: 2026-11-02T00:10:00+08:00
description: "从权限、kernel 过滤、section/metric 采集，到 SOL、occupancy、warp stall、memory tables、source counters 与 roofline：一套能复现的 ncu profiling 流程，附官方样例报告的完整解读。"
tags: [Nsight Compute, ncu, CUDA, Profiling, 性能优化]
category: Tools
draft: false
---

`ncu` 是 NVIDIA Nsight Compute 的命令行入口。它不回答“整个程序哪里慢”，而是把某一次 kernel launch 拆开：SM 的哪个 pipeline 忙、DRAM 带宽用了多少、warp 卡在哪种 stall、shared memory 有没有 bank conflict、哪一条 SASS 指令在等数据。

[CUDA 手册](/posts/kernel/cuda/) 里已经有一节 `ncu` 命令，给的是“先跑起来”的最小用法；这篇把那一节展开成完整流程：怎么申请权限、怎么只测目标 kernel、`--set` / `--section` / `--metrics` 怎么选、报告里先看什么后看什么、以及怎么用官方样例报告练习。命令以 Nsight Compute 2025.2.1 为基准；新版本的选项以 `ncu --help` 和 Release Notes 为准。

> **五分钟版**
>
> - **定位**：`ncu` 是 kernel 级显微镜，`nsys` 是系统级时间线。先用 `nsys` 找热点，再用 `ncu` 解释这个 kernel 为什么慢。
> - **第一次**：`ncu --set basic -k <kernel> -s 2 -c 1 ./app`。不要一上来 `--set full`。
> - **读报告**：SOL → Occupancy → Scheduler / Warp State → Memory Tables → Source Counters；Roofline 用来看离上限还有多远。
> - **开销**：`ncu` 会串行化 kernel、replay、默认 flush cache 和锁频；`Duration` 不能当线上延迟用。
> - **权限**：`ERR_NVGPUCTRPERM` 是驱动设置，不是 CUDA 装坏了；要么 `sudo`，要么在宿主机打开非管理员 profiling。
> - **版本**：新 GPU 配旧 `ncu` 会提示 `Unsupported GPU` 或缺少指标；`ncu` 可以独立升级，不必换整套 CUDA Toolkit。

## 先分清 ncu 和 nsys

两个工具经常被混着叫“Nsight”，但解决的问题不一样；`nsys` 的安装、采集和报告阅读见 [Nsight Systems 手册](/posts/tools/nsys/)：

| 想知道的事 | 工具 | 原因 |
| :--- | :--- | :--- |
| 哪个 kernel 最耗时，stream 之间有没有重叠，CPU 有没有空泡 | [Nsight Systems（`nsys`）](/posts/tools/nsys/) | 系统级时间线，保留多 kernel 并发和 API 调用顺序 |
| 这个 kernel 是 compute-bound 还是 memory-bound | `ncu` | 读 SM / DRAM / cache 的硬件计数器 |
| 哪条指令、哪行源码在 stall | `ncu` + Source Counters | 按指令和源码行聚合采样 |
| 端到端 tokens/s、P50 / P99 延迟 | benchmark / `nsys` | `ncu` 的 replay 会破坏原始时序 |
| 多卡通信、NCCL 行为 | `nsys` / NCCL tests | `ncu` 会把并发 kernel 串行化 |

`ncu` 的核心代价是**为了可复现而破坏原始执行**：它默认把 kernel 串行化、在每次 replay 前清空 cache、把时钟锁到 base 频率，必要时还会保存和恢复显存。所以它能告诉你“这个 kernel 内部发生了什么”，但不能直接告诉你“线上服务能跑多快”。

它也能 attach 到已经运行的进程，但 attach 期间同样会串行化 kernel。对长时间服务，更稳的做法是把热点 kernel 抽成一个最小复现，再单独 profile。

## 安装、版本与权限

### 安装位置与版本

`ncu` 通常随 CUDA Toolkit 一起安装：

```bash
/usr/local/cuda/bin/ncu          # 常见入口，可能是指向具体版本的 wrapper
ncu --version
```

也可以单独安装 Nsight Compute，常见路径是：

```text
/opt/nvidia/nsight-compute/<version>/ncu
/usr/local/cuda-<ver>/nsight-compute-<version>/ncu
```

GUI 是 `ncu-ui`；单独安装时它不一定在 `PATH` 里，用完整路径，或在采集 / 导入报告时加 `ncu --open-in-ui`。

版本选择看两件事：

- **GPU 架构**：`ncu` 必须认识目标 GPU 的 compute capability。新卡配旧 `ncu`，常见结果是 `Unsupported GPU`，或者某些 section 的指标缺失。
- **驱动 / CUDA**：`ncu` 对驱动有最低版本要求；报告文件也尽量用同一版本的 `ncu-ui` 打开，旧版 UI 可能读不了新版报告。

`ncu` 可以独立于 CUDA Toolkit 升级，不必为了新卡重装整套 CUDA。先升级 `ncu`，再看驱动和 Toolkit 是否满足要求。

不带 GPU 采集也能先检查安装：

```bash
ncu --version
ncu --list-sets        # 有哪些 section 组合
ncu --list-sections    # 每个 section 的 identifier
ncu --set basic --list-metrics   # 这个 set 会采集哪些指标
```

### 权限：ERR_NVGPUCTRPERM

第一次跑最常见的报错是：

```text
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access
NVIDIA GPU Performance Counters on the target device 0.
```

这不是安装问题，而是 NVIDIA 驱动默认只允许管理员读取性能计数器。三种处理方式：

**1. 临时用 root 跑**

```bash
sudo ncu --set basic -k vec_add -s 2 -c 1 ./vec_add
```

适合一次性排查，不适合 CI 和日常开发。

**2. 在宿主机打开非管理员 profiling**

创建 `/etc/modprobe.d/nvidia-profiler.conf`：

```bash
sudo tee /etc/modprobe.d/nvidia-profiler.conf >/dev/null <<'EOF'
options nvidia NVreg_RestrictProfilingToAdminUsers=0
EOF
```

然后按发行版重建 initramfs 并重启：

```bash
# Debian / Ubuntu
sudo update-initramfs -u

# Fedora / RHEL
sudo dracut -f

sudo reboot
```

重启后再跑一条最小 kernel 验证。这个设置会让本机所有用户都能读 GPU 性能计数器；多租户或共享机器先问管理员，不要为了跑一次 profile 全局放开。

**3. WSL / 容器**

- WSL2 的性能计数器开关在 Windows 宿主的 NVIDIA Control Panel 里，在 Linux 客户机里改驱动参数不会生效。
- 容器里的 `ncu` 依赖宿主机的驱动设置；容器本身要能访问 GPU。共享 CI runner 上不要默认全局放开权限，先确认 runner 的隔离策略。
- Jetson / 移动平台通常需要 root 或 `sudo` 才能 profile。

## 一次最小 profile

### 准备一个可复现的 kernel

下面是最小 vector add，host 侧分配三块显存，launch 5 次，最后一次同步：

```cpp
__global__ void vec_add(const float* __restrict__ a,
                        const float* __restrict__ b,
                        float* __restrict__ c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main() {
  const int n = 1 << 26;
  const size_t bytes = (size_t)n * sizeof(float);
  float *a, *b, *c;
  cudaMalloc(&a, bytes);
  cudaMalloc(&b, bytes);
  cudaMalloc(&c, bytes);
  cudaMemset(a, 0, bytes);
  cudaMemset(b, 0, bytes);

  dim3 block(256);
  dim3 grid((n + block.x - 1) / block.x);
  for (int i = 0; i < 5; ++i) vec_add<<<grid, block>>>(a, b, c, n);
  cudaDeviceSynchronize();
  return 0;
}
```

用和线上一致的优化选项编译，只额外加 `-lineinfo`：

```bash
nvcc -O3 -lineinfo -arch=native vec_add.cu -o vec_add
# 或 -arch=sm_120 / sm_90 这类与目标卡一致的选项
```

两个常见错误：

- 用 `-G` 编译。`-G` 是完整 device 调试，会关掉大部分优化，测出来的 register / spill / stall 和真实版本不是一回事。
- 用 `-lineinfo` 之外还改了 `-O` 级别、`-maxrregcount` 或宏定义。profiling 版本必须和要优化的版本尽量一致，否则数字没有意义。

### 第一次采集

```bash
ncu --set basic \
  --kernel-name vec_add \
  --launch-skip 2 \
  --launch-count 1 \
  ./vec_add
```

- `--set basic`：只采高层吞吐、launch 配置和 occupancy，约 191 个指标（2025.2 的估算值），足够做第一轮判断。
- `--kernel-name vec_add`：只匹配这个 kernel；默认按函数名匹配，模板或命名空间复杂时加 `--kernel-name-base demangled`。
- `--launch-skip 2`：跳过前两次**匹配到的** launch，采第三次。`--launch-count 1` 只采一次。
- 跳过前两次是因为第一次 launch 往往包含 context / cache / clock 的冷启动效应；如果你只想看稳定态，warmup 后再采。

输出里 `==PROF==` 开头的是 profiler 的信息，其余是程序自己的 stdout。你会看到类似：

```text
==PROF== Connected to process 12345 (/path/vec_add)
==PROF== Profiling "vec_add" - 0: 0%....50%....100% - N passes
==PROF== Report: ...
```

`N` 是 replay pass 数。即使 `--set basic` 也可能要跑多趟，因为 GPU 一次能同时收集的硬件计数器有限；`--set full` 的 pass 数会多得多。

### 保存和读取报告

不指定 `-o` 时，`ncu` 用临时文件，终端打印完就删。要复现和对比，显式保存：

```bash
ncu --set basic -k vec_add -s 2 -c 1 \
  -o vec_add --force-overwrite ./vec_add
```

会生成 `vec_add.ncu-rep`（实际文件名看最后一行输出）。之后可以：

```bash
# 终端看 Details 页
ncu -i vec_add.ncu-rep --page details

# 导出原始指标，方便脚本处理
ncu -i vec_add.ncu-rep --page raw --csv > vec_add.csv

# 打开 GUI
ncu-ui vec_add.ncu-rep
```

`--page raw --csv` 是宽表：第一行是指标名，第二行是单位，第三行才是值。写解析脚本时不要默认第一行就是数据。

## 只测想测的 kernel

真实程序一次会 launch 成百上千个 kernel，全部 profile 既慢又难读。常用过滤方式：

| 需求 | 参数 |
| :--- | :--- |
| 按名字精确匹配 | `-k vec_add` |
| 按正则匹配 | `-k 'regex:.*gemm.*'` |
| 模板 / 命名空间复杂 | `--kernel-name-base demangled` |
| 跳过前 N 次匹配 | `-s N` |
| 只采 N 次 | `-c N` |
| 按 context / stream / 调用序号 | `--kernel-id "::vec_add:2"` |
| 跟子进程 | `--target-processes all` |
| 按 NVTX 范围 | `--nvtx --nvtx-include "forward/"` |
| 从 profiler API 开始 | `--profile-from-start off` |
| 只测某张卡 | `--devices 0` |

几个容易踩的点：

- **`--launch-skip` 只数匹配到的 kernel**，不是数所有 launch。如果程序先跑了很多别的 kernel，`-s 2` 不会把它们算进去；要按“所有 launch”计数，用 `--launch-skip-before-match`。
- **`--kernel-id` 的格式**是 `context-id:stream-id:[name-operator:]kernel-name:invocation-nr`，可以省略不关心的字段。比如 `::vec_add:2` 表示任意 context / stream 下第二次调用的 `vec_add`。
- **正则要加引号**，否则 shell 可能先展开 `*`。`-k 'regex:.*gemm.*'` 比 `-k regex:.*gemm.*` 稳。
- **Python / PyTorch 默认可能只跟主进程**。当前版本 `--target-processes` 默认是 `all`，但旧版本或某些包装脚本不一定；显式写 `--target-processes all` 最稳。

### Python 程序

```bash
ncu --set basic --target-processes all \
  --kernel-name-base demangled \
  --kernel-name 'regex:.*gemm.*' \
  python bench.py
```

PyTorch 的 cuBLAS / CUTLASS kernel 名字很长，先用 `nsys` 或只采一次（`-c 1`）看看实际 kernel 名，再写正则。Triton 生成的 kernel 通常带 `triton_` 前缀，也可以用它过滤。

如果只想 profile 某一段代码，而不是从进程启动开始：

```python
import torch

torch.cuda.profiler.start()
# 只 profile 这一段
torch.cuda.profiler.stop()
```

```bash
ncu --profile-from-start off --target-processes all python bench.py
```

C++ 里对应 `cudaProfilerStart()` / `cudaProfilerStop()`。

### NVTX 范围

框架和推理引擎通常已经打了 NVTX：

```bash
ncu --nvtx --nvtx-include "forward/" --target-processes all python run.py
```

push/pop 范围末尾要带 `/`；start/end 范围不带。`--nvtx-include` 必须配 `--nvtx`，否则不生效。范围过滤比 kernel 名更稳，因为同一个 kernel 可能在不同阶段被调用，名字一样但输入分布完全不同。

### 多进程 / MPI

`--target-processes all` 会跟到子进程，但每个 rank 都采会生成大量报告，还会互相争抢 profiling 资源。通常只测一个 rank，或者用 `--devices` / `--filter-mode per-gpu` 限制到目标 GPU。CI 里更推荐把单卡复现抽出来，而不是直接 profile 整个分布式作业。

## 采集多少：set、section、metric

`ncu` 用 **set** 决定采集多少，set 由若干 **section** 组成，每个 section 是一组逻辑相关的指标。2025.2 的几个常用 set：

| set | 主要内容 | 估算指标数 | 适合 |
| :--- | :--- | ---: | :--- |
| `basic` | LaunchStats、Occupancy、SpeedOfLight、WorkloadDistribution | 191 | 第一轮：是不是 compute / memory 瓶颈 |
| `detailed` | 加 ComputeWorkloadAnalysis、MemoryWorkloadAnalysis、SourceCounters、Roofline | 560 | 第二轮：看 pipeline、内存和 stall |
| `full` | 加 InstructionStats、Memory Tables、Scheduler、WarpState、PM Sampling、NVLink | 6581 | 存档 / 一次性深挖 |
| `roofline` | SOL + 四套 hierarchical roofline | 5912 | 专门看 roofline |
| `pmsampling` | PM Sampling 与 Warp States 时间线 | 231 | 长 kernel 的相位分析 |

数字是 `--list-sets` 给出的估算值，不同版本和架构会变，不要拿它当实际采集次数。`full` 的指标数是 `basic` 的三十多倍，pass 数和耗时也会相应增加；官方 Quickstart 里一个简单 vector add 在默认设置下就跑了 46 passes。

### 用 section 做定向采集

知道了怀疑方向之后，不要重复跑 `full`，直接选 section：

```bash
# 只看 SOL + 内存概况
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis ./app

# 看 shared memory bank conflict 等明细
ncu --section MemoryWorkloadAnalysis_Tables ./app

# 看采样 stall 与源码/SASS 关联
ncu --section SourceCounters ./app

# 只看 roofline
ncu --section SpeedOfLight_RooflineChart ./app

# 长 kernel 的时间线
ncu --section PmSampling ./app
```

`--list-sections` 会打印所有 identifier；`--set basic --list-metrics` 可以看某个 set 具体包含哪些指标。

### 用 metric 精确采集

只想验证一两个数时，`--metrics` 比整个 section 便宜：

```bash
ncu --metrics \
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__time_duration.sum \
./app
```

指标名是“基础名 + 后缀”的结构，例如：

```text
基础名   sm__throughput
rollup   .avg
suffix   .pct_of_peak_sustained_elapsed
```

- `.sum` / `.avg` / `.min` / `.max`：在多个硬件实例之间求和、平均、取最小 / 最大。
- `.pct_of_peak_sustained_elapsed`：相对峰值持续吞吐的百分比。
- `.per_second`：按秒归一化。
- `.ratio`：两个计数器的比值，例如每 request 多少 sector。

不确定某个基础名有哪些合法后缀时：

```bash
ncu --query-metrics-mode suffix --metrics sm__throughput
```

这条命令会读设备能力，同样受性能计数器权限影响，需要在能访问 GPU 的机器上跑。`--list-metrics` 只读 section 文件，不需要采集权限。

### replay 与开销

`ncu` 的默认 replay 模式是 **kernel replay**：同一个 kernel 跑多遍，每遍收集一部分指标，遍与遍之间恢复被写过的显存。指标越多，pass 越多，开销越大。

三种 replay 模式：

- `kernel`（默认）：只 replay 目标 kernel，最常用。
- `application`：整个程序重新跑，适合 kernel 与 host 有交互、kernel replay 会挂死的场景；要求程序行为确定。
- `range`：回放一段 CUDA API / kernel 范围，适合多 kernel 并发、通信与计算重叠的场景；需要用 NVTX 或 `cudaProfilerStart/Stop` 标范围。

另外两个默认行为会影响数字：

- `--cache-control all`：每次 replay 前清空 L1/L2，模拟“冷 cache”。想看应用内真实 cache 复用时改 `--cache-control none`，但结果会更依赖执行顺序。
- `--clock-control base`：把 GPU 时钟锁到 base 频率，减少冷热启动差异。如果外部已经用 `nvidia-smi` 锁频，或者要分析应用内动态频率，改 `--clock-control none`。

对比两次 profile 时，这些设置必须一致，否则差异可能来自 cache 或时钟，而不是代码。

## 报告怎么读

下面是一条比较稳的阅读顺序。不要从第一张表逐行读到最后，先建立假设，再找证据。

### 1. SOL：先判断 compute 还是 memory

`GPU Speed Of Light Throughput` 是总览：

| 指标 | 看什么 |
| :--- | :--- |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM 侧综合吞吐；高 → compute-bound 候选 |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | DRAM 带宽；高 → memory-bound 候选 |
| `l1tex__throughput...` / `lts__throughput...` | L1 / L2 路径；高但 DRAM 不高 → cache 或 shared 瓶颈 |
| `gpu__time_duration.sum` | kernel 时长；只用于同一设置下的相对比较 |

粗略的判断表：

| 现象 | 可能的方向 | 下一步 |
| :--- | :--- | :--- |
| SM 高、DRAM 低 | compute-bound | 看 ComputeWorkloadAnalysis 的 pipe 利用率、指令 mix、tensor core |
| DRAM 高、SM 低 | memory-bound | 看 Memory Tables、bytes/sector、L2 命中、访问复用 |
| 两者都不高 | latency / occupancy / launch / tail | 看 Occupancy、Scheduler、Warp State、waves |
| L1/L2 高、DRAM 不高 | cache 路径或 shared memory 压力 | 看 bank conflict、wavefronts、L1 命中率 |
| Duration 很短、SOL 很低 | launch / tail 占主导 | 看 grid 是否只有一波、是否值得融合或 persistent kernel |

NCU 的分析规则会在 SOL 超过 80% 时提示“已经接近可用上限”，但 80% 不是硬阈值。不同指令 mix 的 kernel 可能到 60% 就已经是实际屋顶；反过来，SOL 低也不一定说明有优化空间，短 kernel 的固定开销就会把百分比拉低。

### 2. Occupancy：先看限制项，再看 achieved

Occupancy 是“实际驻留 warp 数 / 最大可驻留 warp 数”。它影响延迟隐藏能力，但不是越高越好。

| 字段 | 含义 |
| :--- | :--- |
| Block Limit SM | 硬件每 SM 最多几个 block |
| Block Limit Warps | warp 总数上限 / 每 block 的 warp 数 |
| Block Limit Registers | 寄存器文件 / 每 block 的寄存器用量 |
| Block Limit Shared Mem | shared memory 总量 / 每 block 的 shared 用量 |
| Theoretical Occupancy | 理论上限 |
| Achieved Occupancy | 实际执行期间的平均值 |

四个 `Block Limit` 里最小的那个就是当前限制项。比如官方样例里：

```text
Block Limit SM           16
Block Limit Registers    10
Block Limit Shared Mem   12
Block Limit Warps         6
Theoretical Occupancy   100%
Achieved Occupancy     93.89%
```

限制项是 warps：每 block 8 warp，最多 6 个 block，合计 48 warp，正好是理论满载。achieved 93.89% 说明执行期间几乎没有掉 occupancy，这个 kernel 的问题不在“warp 不够”。

两种常见误判：

- **Theoretical 100% 但 Achieved 60%**：不是资源限制，更可能是尾效应、负载不均、barrier 或 block 调度不均。看 `launch__waves_per_multiprocessor` 是否带小数，以及 WorkloadDistribution。
- **Theoretical 很低**：先看是寄存器、shared memory 还是 block 数限制。为了拉高 occupancy 强行压寄存器，可能把 spill 逼出来，反而更慢；先看 `-Xptxas -v` 或 local load/store 指标。

### 3. Scheduler 与 warp stall：找出“等什么”

如果 SOL 不高、occupancy 也不低，下一步看 scheduler 为什么发不出指令：

| 指标 | 含义 |
| :--- | :--- |
| `smsp__warps_eligible.avg.per_cycle_active` | 每个 scheduler 每 cycle 平均有多少 eligible warp |
| `smsp__issue_active.avg.pct_of_peak_sustained_active` | issue slot 的忙碌比例 |
| `smsp__average_warps_issue_stalled_*_per_issue_active.ratio` | 每种 stall 平均占多少个 cycle / 指令 |

判断逻辑很简单：**如果 issue 已经很忙，stall 不是主要矛盾**；只有 issue 经常空着、eligible warp 又不足 1 时，top stall 才值得追。`Warp Cycles Per Issued Instruction` 是两条指令之间的平均间隔，top stall 的占比加起来大致就是这个间隔的构成。

常见 stall 原因和处理方向：

| stall | 在等什么 | 先试什么 |
| :--- | :--- | :--- |
| `long_scoreboard` | L1TEX：global / local / surface / texture 数据 | 合并访存、提高复用、异步拷贝、增加并行 |
| `short_scoreboard` | MIO：shared memory、MUFU、动态分支 | 查 bank conflict、减少 shared 操作、改用寄存器 |
| `barrier` | `__syncthreads()` 等 CTA 屏障 | 均衡各 warp 的工作量、拆小 block、减少不必要的同步 |
| `mio_throttle` | MIO 指令队列满（shared、特殊数学、分支） | 减少 shared / MUFU 指令，合并成更宽的访问 |
| `lg_throttle` | local / global 的 LSU 队列满 | 减少访存指令、向量化、检查 spill |
| `math_pipe_throttle` | 某条数学 pipeline 被占满 | 调整指令 mix，能用 tensor core 就用 tensor core |
| `not_selected` | 有 eligible warp，但 scheduler 选了别人 | 通常不是坏事；说明并行度足够，可考虑降 occupancy 换 cache 局部性 |
| `wait` | 固定延迟的指令依赖 | 展开循环、提高 ILP、换低延迟指令 |
| `drain` | kernel 结束时等未完成 store | 合并写、减少尾部集中写 |
| `branch_resolving` / `no_instructions` | 分支目标计算 / 取指 | 减少控制流发散和跳转，检查 instruction cache |
| `membar` | 内存屏障 | 去掉不必要的 fence |
| `imc_miss` | 常量内存 miss | 避免 warp 内访问不同常量地址 |
| `sleeping` | `__nanosleep` 等主动休眠 | 降低 sleep 时间或调整调度 |
| `tex_throttle` | texture / surface 队列满 | 减少 texture 指令，考虑改 global load |

top stall 是采样统计，不是因果证明。比如一个纯 streaming kernel 的 `long_scoreboard` 很高是正常的，真正要优化的是“每个字节搬了多少次”，而不是继续加 occupancy。要定位到具体指令，去 Source Counters。

### 4. Memory：先看是不是搬多了

`Memory Workload Analysis` 的几个字段容易混：

| 字段 | 含义 |
| :--- | :--- |
| Memory Throughput | 实际达到的 GB/s |
| Mem Busy | 内存单元（L1/L2/DRAM 等）的利用率；高说明某个单元忙 |
| Max Bandwidth | 内存单元之间请求通道的带宽；高说明数据传输路径接近饱和 |
| L1/TEX Hit Rate | L1 命中率 |
| L2 Hit Rate | L2 命中率 |
| Mem Pipes Busy | SM 内存指令管线的吞吐；高说明 load/store 指令太多 |

- `Max Bandwidth` 高：内存请求通道接近饱和；再看 `gpu__dram_throughput` 确认是不是 DRAM 带宽到顶。到了就先减少搬运，而不是继续优化计算。
- `Mem Busy` 高但 `Max Bandwidth` 不高：可能是 L1/L2/shared 路径忙，或者延迟/并行度不够，不一定是 DRAM 带宽。
- `Mem Pipes Busy` 高：LSU 指令太多，考虑向量化、减少重复 load/store。

#### Global coalescing

GPU 以 32B sector 为最小传输单位。一个 warp 的 32 个线程各读 4B，如果地址连续，合计 128B，正好 4 个 sector；如果地址散开，每个线程可能各碰一个 sector，实际搬运量会放大。

最直接的指标是：

```text
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio
```

理想值是 32 B/sector；如果只有 4–8 B，说明每个 sector 大部分字节被浪费。也可以看：

```text
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum
```

两者相除是“每个 request 平均碰几个 sector”。注意要结合访问宽度看：4B/线程连续访问时理想是 4；16B/线程的向量访问理想是 16，不要一律拿 4 当目标。

#### Shared memory bank conflict

shared memory 通常按 32 个 bank、4B 宽度组织。一个 warp 访问 32 个连续 float 是 conflict-free；如果 stride 正好是 32 个 float，32 个线程会落到同一个 bank，变成 32-way conflict。

指标：

```text
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum
```

`wavefronts` 是实际拆出来的访问波前；冲突越多，wavefront 越接近 request 的倍数。经典修法是给二维 shared 数组多补一列：

```cpp
__shared__ float tile[32][32];      // 转置时列访问会 32-way conflict
__shared__ float tile[32][33];      // padding 打散 bank
```

### 5. Source Counters：落到指令和源码行

Source Counters 把采样到的 stall、访存 sector、bank conflict 关联到 SASS 指令和 CUDA 源码行。要看 CUDA 源码，编译时必须有 `-lineinfo`：

```bash
ncu --set detailed --import-source on \
  -k vec_add -s 2 -c 1 -o vec_add --force-overwrite ./vec_add

ncu -i vec_add.ncu-rep --page source --print-source cuda,sass
```

`--import-source on` 会把源码复制进报告，方便换机器后还能看；不加时报告里只保留路径。

Source 页很宽，终端里建议重定向到文件再用 `less` / `grep`：

```bash
ncu -i vec_add.ncu-rep --page source --print-source sass > vec_add_source.txt
```

重点看：

- **Warp Stall Sampling**：哪条指令贡献了最多 stall。
- **Instructions Executed / Threads**：这条指令执行了多少次。
- **Sectors / Requests**：访存有没有放大。
- **Bank Conflicts / Wavefronts**：shared 访问是否冲突。
- **Stall 分解**：同一行上 `long_scoreboard`、`barrier` 等各占多少。

采样是统计性的：kernel 太短时样本可能很少，`smsp__pcsamp_sample_count` 和 `smsp__pcsamp_dropped_bytes` 可以判断数据是否够用。PyTorch / Triton 的 JIT kernel 通常没有 CUDA 源码，但 SASS 仍然可用，能看到具体是 `LDG`、`STG`、`LDS` 还是 `MMA` 在 stall。

### 6. Roofline：看离上限还有多远

Roofline 把“算力上限”和“带宽上限”画在同一张图上：

$$
P_\text{achieved} \le \min\left(P_\text{peak},\; I \cdot B_\text{peak}\right),
\qquad
I = \frac{\text{FLOPs}}{\text{Bytes}}
$$

其中 $I$ 是算术强度，$B_\text{peak}$ 是峰值带宽，$P_\text{peak}$ 是峰值算力。两条线的交点是 ridge point：

$$
I^* = \frac{P_\text{peak}}{B_\text{peak}}
$$

以 H100 SXM 的公开峰值粗算：FP16 tensor 约 989 TFLOP/s，HBM3 约 3.35 TB/s，ridge point 约 295 FLOP/B。$I < 295$ 时更可能受带宽限制，$I > 295$ 时才轮到算力。

读图：

- 点落在斜线附近：memory-bound。继续优化计算没用，先提高数据复用、减少 bytes、做融合。
- 点落在平顶附近：compute-bound。看 pipeline 利用率、指令 mix、tensor core 是否吃满。
- 点离两条线都远：既没吃满算力也没吃满带宽，通常是延迟 / occupancy / 依赖链问题。

`ncu` 有四套 hierarchical roofline，对应 FP64 / FP32 / FP16 / Tensor Core：

```text
SpeedOfLight_HierarchicalDoubleRooflineChart
SpeedOfLight_HierarchicalSingleRooflineChart
SpeedOfLight_HierarchicalHalfRooflineChart
SpeedOfLight_HierarchicalTensorRooflineChart
```

选错精度会得到错误的“屋顶”。另外，roofline 的峰值来自规格，FLOPs 来自指令计数；对融合、量化、稀疏 kernel，绝对位置可能偏，适合看方向和 headroom，不适合拿绝对百分比做对外结论。

### 7. PM Sampling：长 kernel 的时间线

静态 section 是整个 kernel 的平均值。如果 kernel 很长、行为随时间变化（persistent kernel、通信与计算重叠、阶段式流水），平均值会骗人：

```bash
ncu --section PmSampling ./app
```

PM Sampling 会周期性采样指标，给出 kernel 内部的时间线，可以看到某个阶段 DRAM 打满、另一个阶段 tensor pipe 打满。它依赖架构和驱动，先跑一条确认支持；它仍然是在单个 kernel 内采样，不能替代 `nsys` 的跨 kernel 时间线。

## 官方样例：从 88.7% DRAM 到 32-way bank conflict

Nsight Compute 安装包自带 sample report，没有 GPU 也能用 `ncu -i` 打开，非常适合练手：

```bash
ls /opt/nvidia/nsight-compute/*/extras/samples
# 或 /usr/local/cuda-*/nsight-compute-*/extras/samples

ncu -i .../sharedBankConflicts/transposeCoalesced.ncu-rep --page details
ncu-ui .../sharedBankConflicts/transposeCoalesced.ncu-rep
```

这个样例做矩阵转置，有两个版本：`transposeCoalesced` 用 `tile[32][32]`，写回时按列读 shared，产生 bank conflict；`transposeNoBankConflicts` 用 `tile[32][33]` 打散 bank。两份报告都是 `--set full` 采的，硬件是 RTX A4500（CC 8.6）。

| 指标 | 有 bank conflict | padding 后 | 变化 |
| :--- | ---: | ---: | :--- |
| Duration | 1.4208 ms | 0.9976 ms | −29.8% |
| DRAM Throughput | 61.92% | 88.69% | +26.8 pp |
| SM Throughput | 22.60% | 32.43% | +9.8 pp |
| Achieved Occupancy | 93.89% | 95.64% | 基本不变 |
| Shared load bank conflicts | 65,011,712 | 0 | −100% |
| Shared load wavefronts | 67,446,928 | 2,098,898 | −96.9% |

读数过程：

1. **SOL**：DRAM 61.9%、SM 22.6%。不是典型 compute-bound；DRAM 不低，但也没到 90%，先怀疑内存路径。
2. **Occupancy**：achieved 93.9%，theoretical 100%。warp 足够，不是“occupancy 不够”导致延迟隐藏失败。
3. **Memory Tables**：规则直接报出 `32.2-way bank conflict across all 2097152 shared load requests`，65,011,712 次冲突占 shared load wavefront 的 96.39%。瓶颈找到了。
4. **修复**：给 shared 数组补一列，冲突降到 0；shared load wavefront 从 67.45M 降到 2.10M。
5. **复测**：时间下降 29.8%，DRAM 利用率反而升到 88.7%。这不是“修完变成 DRAM 瓶颈”的矛盾，而是原来 shared 路径把流水线卡住了，DRAM 吃不满；冲突消失后，同样的全局访存能跑得更快，DRAM 才接近饱和。

这个案例的教训：

- occupancy 高不代表没有瓶颈，它只说明 warp 数量够。
- SOL 里 DRAM 没到 90% 也不代表内存没问题；要看 Memory Tables 和 wavefronts。
- 修完一个瓶颈，原来的“次要瓶颈”会浮上来。padding 之后如果还要继续优化，方向就变成减少全局 bytes 或提高 L2 复用，而不是继续调 shared。

样例是 CC 8.6 的旧报告，绝对数字不能直接套到 Hopper / Blackwell；要学的是“先看哪张表、怎么建立假设、怎么用第二个报告验证”。

## 一个可复用的工作流

| 步骤 | 做什么 | 产出 |
| :--- | :--- | :--- |
| 1. 找热点 | `nsys profile --stats=true ./app` 或 benchmark | 最耗时的 kernel、shape、占比 |
| 2. 做复现 | 固定输入 / warmup / 单独 launch | 可重复的最小 harness |
| 3. 第一轮 | `ncu --set basic -k ... -s ... -c 1` | SOL、occupancy、duration |
| 4. 提假设 | 根据 SOL 选 compute / memory / latency 方向 | 一个可验证的判断 |
| 5. 定向采集 | `--section` 或 `--metrics` | bank conflict、sector 效率、stall 指令 |
| 6. 改一个变量 | 只改 tiling / 布局 / 向量化中的一项 | 可对比的新报告 |
| 7. 复测 | 同样的 ncu 设置 + benchmark / nsys | 局部指标和端到端收益 |

对比报告有两种方式：

- **GUI**：`ncu-ui` 打开两份报告，把一份设为 baseline，能并排看指标差异和 roofline 上的两个点。适合人工分析。
- **CLI**：`ncu -i report.ncu-rep --page raw --csv > report.csv`，用脚本提取关心的指标；`--print-summary per-kernel` 可以快速看多个 kernel 的汇总。

CI 里做回归可以这样：

```bash
ncu --set basic -k 'regex:gemm' -c 1 \
  -o gemm --force-overwrite ./bench

ncu -i gemm.ncu-rep --page raw --csv > gemm.csv
```

但不要把 `gpu__time_duration.sum` 直接当性能门槛：CI runner 的时钟、邻居负载、MIG 配置都可能变。更稳的是固定时钟、固定输入，并且同时看 SOL、occupancy 和关键 stall 的变化；实在要卡门槛，先跑一段基线统计，设一个宽容的阈值。

## 常见坑

- **把 ncu 当 benchmark**：它串行化 kernel、replay、flush cache、锁频。端到端延迟和吞吐用 benchmark / `nsys`，`ncu` 只用来解释原因。
- **用 `-G` 编译**：调试版关优化，register、spill、stall 全变。用 `-O3 -lineinfo`。
- **`--set full` 一把梭**：6581 个指标、几十个 pass，大 kernel 可能跑几十分钟。先 `basic`，再按假设加 section。
- **忽略 cache / clock 设置**：默认 flush cache + 锁 base 频率；两次对比设置不一致，差异可能来自环境。看 cache 复用就显式 `--cache-control none`，并接受可复现性下降。
- **kernel 之间有通信时用默认 replay**：producer / consumer 或 NVSHMEM 场景可能挂死。用 `--replay-mode range` 配 NVTX / profiler API，或 `--replay-mode application`。
- **Python 只测到父进程**：加 `--target-processes all`；JIT kernel 名字多变，先不加 `-k` 看一次真实名字。
- **`--launch-skip` 语义记错**：它只数匹配到的 kernel；要按全部 launch 计数用 `--launch-skip-before-match`。
- **版本不匹配**：新报告用旧 `ncu-ui` 打不开；新 GPU 用旧 `ncu` 缺指标。报告、CLI、UI 尽量同版本。
- **容器 / WSL 权限**：驱动参数是宿主机级别；WSL 在 Windows 侧开关；共享机器不要随便全局放开。
- **同时开多个 profiler**：`ncu`、`nsys`、PyTorch Profiler 都走 CUPTI，容易互相干扰甚至直接报错；一次只开一个。
- **只看一个指标**：SOL 高不代表没问题，occupancy 高也不代表没问题。至少交叉看 SOL、occupancy、top stall 和 memory tables。

## 参考

- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)：sets / sections、replay、指标结构、roofline、warp stall 定义
- [Nsight Compute CLI 文档](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)：完整命令行选项、NVTX 过滤、输出页
- [Nsight Compute Release Notes](https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html)：版本、驱动、支持的 GPU
- [ERR_NVGPUCTRPERM](https://developer.nvidia.com/ERR_NVGPUCTRPERM)：性能计数器权限的官方说明
- [Nsight Compute Python Report Interface](https://docs.nvidia.com/nsight-compute/PythonReportInterface/index.html)：用 Python 批量解析 `.ncu-rep`
- [Nsight Systems](https://developer.nvidia.com/nsight-systems)：系统级时间线，和 `ncu` 配合使用
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)：访存、occupancy 和 kernel 优化的背景

和 Kernel 分类的分工：[CUDA 手册](/posts/kernel/cuda/)、[Triton 手册](/posts/kernel/triton/)、[TileLang 手册](/posts/kernel/tilelang/) 讲怎么写 kernel，[DeepGEMM](/posts/kernel/deepgemm/) 讲生产库怎么组织；这篇讲怎么量、怎么定位瓶颈。和 [GPU Architecture](/posts/gpu-arch/intro/) 的分工：那边解释硬件为什么这样设计，这边用硬件计数器验证它在具体 kernel 上的后果。
