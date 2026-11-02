---
title: "SGLang 的 Continuous Batching：RadixAttention 感知的调度器"
published: 2026-11-03T01:25:00+08:00
description: SGLang 没有把 continuous batching 做成一个开关，而是把它写进了 event loop：waiting_queue 怎么准入、running_batch 怎么合并、chunked prefill 怎么切、KV 不够时怎么 retract，以及 overlap scheduler 怎么把 CPU 调度藏到 GPU 计算后面。
tags: [SGLang, Continuous Batching, 调度, chunked prefill, RadixAttention]
category: LLM Engine
draft: false
---

先回答标题里的问题：**有，而且它不是后来加的补丁，是 SGLang 调度器的默认工作方式。** 代码里没有 `--enable-continuous-batching` 这种开关，因为 `event_loop` 从结构上就是“每跑完一次前向，重新决定下一个 batch 由谁组成”。

但它和 vLLM v1 的 continuous batching 又不是同一套写法。vLLM v1 把 prefill / decode 阶段从调度器里删掉，只保留 `num_computed_tokens` 追平；SGLang 保留了显式的 prefill batch 和 decode batch，并且让**前缀缓存参与准入和排序**。这篇拆的就是这套“RadixAttention 感知”的调度。

上篇讲 KV 怎么管：[SGLang 的 KV Cache](/posts/llm-engine/sglang/kv-cache/)。这篇讲这些槽位怎么被分给一批不断变动的请求。

> **五分钟版**
>
> - **粒度**：每跑完一次前向重排一次。完成的请求当步退出，等待的请求下一轮就能进；`waiting_queue`、`running_batch`、`chunked_req` 是三个核心状态。
> - **阶段**：SGLang 仍然分 prefill batch 和 decode batch；有 prefill 就先跑 prefill，没有才跑 decode。`--enable-mixed-chunk` 可以把 decode 行拼进 chunked prefill，默认关闭。
> - **准入**：`PrefillAdder` 同时看 prefix 命中、`max_prefill_tokens`（默认 16384）、`chunked_prefill_size`（按 GPU 档位 2K–16K）、`max_running_requests` 和剩余 KV；放不下就排队。
> - **chunked prefill**：长 prompt 切成 chunk，`chunked_req` 跨 step 续算；`--chunked-prefill-size -1` 可以关掉。
> - **回收**：decode 显存不够时 `retract_decode()` 按“短输出、长输入优先”把请求踢回 waiting，默认重算，也可以备份到 CPU/host pool 再恢复。
> - **overlap**：`event_loop_overlap` 先 launch 当前 batch，再处理上一个 batch 的结果，把 CPU 调度藏到 GPU 计算后面。
> - **缓存感知**：默认 `fcfs`，另有 `lpm` / `dfs-weight` / `hrrn` / `shortest-prefill-first`；还有 in-batch prefix caching，避免同时 prefill 多个共享同一小前缀的请求。
> - **PD 分离**：把 prefill 和 decode 拆成两个进程，各自继续 continuous batching，中间传 KV。
>
> 代码口径：SGLang `main` @ `a99c7d0`（2026-11）。入口是 `python/sglang/srt/managers/scheduler.py` 的 `event_loop_normal()` / `event_loop_overlap()` 和 `get_next_batch_to_run()`。

## 一、静态 batching 浪费的是什么

同一个 batch 里，A 要跑 20 步、B 只跑 6 步、C 要跑 30 步。静态 batching 要等最慢的 C：

```
静态 batching（等整批结束）
step: 1      6            20           30
A:    ██████████████████████████████
B:    ██████░░░░░░░░░░░░░░░░░░░░░░░░
C:    ████████████████████████████████████
D:    ...............................  (等下一轮)
```

B 在第 6 步就算完了，但位置一直占到第 20 步；D 明明可以进来，却要等整批结束。GPU 不是算不动，是没活干。

Continuous batching 把调度粒度从“一批请求”降到“一次前向”：每跑完一步就重排，谁做完谁走，谁在等谁进。这个想法在 Orca（OSDI'22）里叫 iteration-level scheduling，后来成了主流推理引擎的默认。

但它有一个前提：**显存必须能随时借还**。如果每个请求都按 `max_model_len` 预留连续显存，batch 每变一次就要重排一次，碎片和拷贝会吃掉所有收益。SGLang 的 RadixAttention + 动态槽位分配，正好提供了这个前提。

## 二、SGLang 的 event loop：三个状态，一次前向

SGLang 的调度器围绕三个状态转：

| 状态 | 含义 |
| :--- | :--- |
| `waiting_queue` | 还没被准入的请求（新请求、retract 回来的请求） |
| `running_batch` | 正在 decode 的请求 |
| `chunked_req` | 上一步没算完的 chunked prefill 请求，下一步继续 |

再加一个 `last_batch`：上一轮跑的 prefill batch。它在这一轮会被合并进 `running_batch`，然后参与 decode。

`event_loop_normal()` 的骨架非常短：

```python
while True:
    self.ingest_requests()                       # 收新请求，进 waiting_queue

    plan = self.get_next_batch_to_run(           # 决定这一轮跑谁
        running_batch=self.running_batch,
        last_batch=self.last_batch,
    )
    self.running_batch = plan.running_batch
    batch = plan.batch_to_run

    if batch:
        result = self.run_batch(batch)           # 一次前向
        self.process_batch_result(batch, result) # 采样、流式返回、释放 KV
    else:
        self.on_idle()

    self.last_batch = batch
```

`get_next_batch_to_run()` 才是重点，简化后是：

```python
# 1. 上一个 prefill batch 跑完了：把还活着的请求并进 running_batch
if last_batch and last_batch.forward_mode.is_extend():
    last_batch.filter_batch(reqs_to_exclude=...)
    if running_batch.is_empty():
        running_batch = last_batch
    else:
        running_batch.merge_batch(last_batch)

# 2. 清掉已经完成的请求
running_batch.filter_batch()

# 3. 尝试从 waiting_queue 组一个新的 prefill batch
new_batch = self.get_new_batch_prefill(running_batch)

# 4. 有 prefill 就跑 prefill；没有才跑 decode
if new_batch is not None:
    return new_batch
else:
    return self.update_running_batch(running_batch)
```

这几行决定了 SGLang 的“性格”：

- **请求是连续进出的**：`filter_batch()` 每轮清掉完成的，`get_new_batch_prefill()` 每轮尝试接纳新的；不存在“等整批结束”。
- **prefill 优先**：只要 `waiting_queue` 里能组出 batch，这一轮就跑 prefill；decode 要等没有 prefill 的轮次。这和 vLLM v1 的“先 running、后 waiting”是相反的顺序。
- **prefill 和 decode 默认不混批**：一个 step 要么是 prefill batch，要么是 decode batch；`--enable-mixed-chunk` 才把 decode 行拼进 chunked prefill。

所以“SGLang 有没有 continuous batching”的答案是：**有，而且是 prefill 优先的 continuous batching**。它不是没有阶段，而是阶段之间每步都在切换。

## 三、准入：一个请求要过几道闸

`get_new_batch_prefill()` 最终会创建一个 `PrefillAdder`，然后按 `waiting_queue` 顺序逐个 `add_one_req()`。每个请求要过四道闸：

### 第一道：前缀命中

请求在进 adder 之前会先 `init_next_round_input(tree_cache)`，里面调用 `match_kv_cache()` 查 Radix 树，得到 `prefix_len` 和 `num_matched_prefix_tokens`。这一步决定了：

- 这次 prefill 实际要算多少 token：`input_len - prefix_len`；
- 要分配多少新 KV：只有未命中的后缀；
- 如果命中到 host/L3，还要发起 load-back / prefetch。

**前缀命中不是“优化”，而是准入的一部分**：命中越多，这个请求越容易通过后面的显存检查，因为它要的新槽更少。

### 第二道：token 预算

`PrefillAdder` 维护 `rem_input_tokens` 和 `rem_chunk_tokens`：

- `rem_input_tokens` 来自 `max_prefill_tokens`，默认 **16384**；它限制一个 prefill batch 的总 token 数；
- `rem_chunk_tokens` 来自 `chunked_prefill_size`，限制单条请求一个 chunk 的大小。

`chunked_prefill_size` 不传时按 GPU 档位解析：

| GPU 档位 | 默认 chunk 大小 |
| :--- | :--- |
| T4 / 4080（< 20 GB） | 2048 |
| A10 / 4090 / 5090（20–35 GB） | 4096 |
| A100 40G / L40（35–60 GB） | 4096 |
| H100 / A100 80G（60–90 GB） | 8192 |
| H20 / H200（90–160 GB） | 8192 |
| B200 / MI300（≥ 160 GB） | 16384 |

这个表和 vLLM 的 `max_num_batched_tokens` 按设备分档是同一个思路：小卡上 chunk 小一点，避免单步激活把显存打爆；大卡上 chunk 大一点，prefill 更快。

### 第三道：KV 预算

`PrefillAdder` 不只看“空闲槽”，而是看 **可用 + 可驱逐**：

```python
remaining_total = allocator.available_size() + tree_cache.evictable_size() - total_offset
```

`total_offset` 是已经承诺但还没分配的 decode 预留。SGLang 对“新请求”和“已在跑的请求”用了两套估计：

- **新请求**：把 `max_new_tokens` 截到 `CLIP_MAX_NEW_TOKENS`（默认 4096）后全额计入，避免一个 32K 输出的请求把池子锁死；
- **已在跑的请求**：用 `new_token_ratio` 估计剩余输出，初始约 0.7，随 decode 步数衰减到约 0.1，而不是按 `max_new_tokens` 全额预留；
- `schedule_conservativeness` 乘在初始比例上，调大就更保守。

这套估计的目的是**别把“最坏情况”当成“平均情况”**：如果每个请求都按最大输出预留，并发会被压得很低；如果完全不预留，decode 时又会频繁 retract。SGLang 对新请求保留一个截断后的上限，对老请求用会衰减的中间值。

### 第四道：请求槽与 batch 上限

`max_running_requests` 限制同时在跑的请求数，`req_to_token_pool` 的行数也有限；`get_num_allocatable_reqs()` 还会扣掉 PP 的 micro-batch 预算和 beam 预留。任何一道不够，`add_one_req()` 返回 `NO_TOKEN`，请求留在队列里。

如果开了 priority scheduling，`preempt_to_schedule()` 还可以在准入时抢占低优先级请求，给高优先级请求腾位置。

### 一个请求被拒绝之后

`add_one_req()` 返回三种结果：

| 结果 | 含义 | 调度器动作 |
| :--- | :--- | :--- |
| `CONTINUE` | 接纳成功 | 继续看下一个请求 |
| `NO_TOKEN` | 显存/预算不够 | 停止本轮 prefill，`batch_is_full = True` |
| `OTHER` | 其他原因（grammar、LoRA、tile 预算等） | 跳过这个请求，继续看下一个 |

注意 `NO_TOKEN` 不是异常，也不是把请求丢掉；它只是告诉调度器“这一轮装不下了”。请求还在 `waiting_queue`，下一轮再试。

## 四、chunked prefill：长 prompt 切成几段算

一个 32K prompt 如果一次 prefill，单步激活和 KV 分配都会很重，而且会把 decode 饿死。chunked prefill 把它切成 `chunked_prefill_size` 大小的段，跨多个 step 算。

SGLang 用 `chunked_req` 记住“上一步没算完的那条请求”：

```python
# get_new_batch_prefill() 里
if self.chunked_req is not None:
    self.chunked_req.init_next_round_input()
    self.chunked_req = adder.add_chunked_req(self.chunked_req)

# add_chunked_req() 里
req.extend_end = req.prefix_len + new_len   # 这一步算到哪
self.can_run_list.append(req)
return req if truncated else None           # 没算完就继续挂在 chunked_req
```

几个细节：

- **chunk 边界按页对齐**：`extend_len` 会向下取整到 `page_size` 的倍数，避免一个 chunk 跨页导致分配器多算一页；
- **chunked 请求不重新准入**：它已经持有 `req_to_token` 行，跳过 `max_running_requests` 检查，否则会在 PP/多 microbatch 下泄漏；
- **算完最后一段才转 decode**：`add_chunked_req()` 返回 `None` 表示这条请求的 prefill 结束，下一轮它就在 `running_batch` 里 decode；
- **`--chunked-prefill-size -1` 关闭**：所有 prefill 必须一次装下，长 prompt 的 TTFT 会更差，但调度更简单。

`--enable-mixed-chunk` 是另一个维度：它把 `running_batch` 里的 decode 行拼进 chunked prefill 的同一个 batch，让一次前向里既有 prefill 又有 decode。默认关闭。关闭时，prefill 和 decode 仍然分批跑；开启后，decode 的 ITL 更平滑，但 batch 组装、attention backend、CUDA graph 都更复杂。

## 五、decode 与 retract：显存不够时牺牲谁

prefill 跑完之后，请求进入 `running_batch`，每轮 decode 一个 token（投机解码是 k+1 个）。`update_running_batch()` 每轮做三件事：

```python
# 1. 清掉已经完成的请求
batch.filter_batch()

# 2. 检查下一步 decode 的 KV 够不够
if not batch.check_decode_mem():
    # 3. 不够就 retract，直到装得下
    retracted_reqs, new_ratio, abort_reqs = batch.retract_decode()
    for req in retracted_reqs:
        self._add_request_to_queue(req, is_retracted=True)

batch.prepare_for_decode()
```

`check_decode_mem()` 不是简单看 `available_size()`，而是让 allocator 做一次 `check_decode_capacity()`：先尝试驱逐可驱逐的缓存，再判断剩下的够不够。这样 retract 循环和分配器用的是同一个判断，不会出现“检查通过、分配失败”的裂缝。

`retract_decode()` 的 victim 顺序默认是 `(output_len, -input_len)` 升序弹出：

```
先踢：输出短、输入长     → 已生成少，重算代价相对可控
后踢：输出长、输入短     → 已经跑了很多步，尽量保住
```

这很符合直觉：一个刚生成 2 个 token、prompt 有 30K 的请求，踢掉它损失的算力最少；一个已经生成 2000 个 token 的请求，踢掉它等于扔掉最多的工作。

被踢的请求有几种命运：

| 配置 | 行为 |
| :--- | :--- |
| 默认 | 回 waiting 队列，下次重新 prefill；Radix 树里还留着的部分会命中 |
| `--disaggregation-decode-retraction-backup cpu_tensor` | KV 备份到 CPU tensor，恢复时 load 回来，不重算 |
| `--disaggregation-decode-retraction-backup host_pool` | 备份到 HiCache host pool；池满则 abort |
| `none` | 不备份，直接 503，让客户端重试 |

retract 之后，`new_token_ratio` 会按当前 batch 的实际输出进度重估，通常会更保守；`--retraction-policy priority` 可以改成按优先级踢人。`num_retracted_reqs` 是要盯的指标：它持续上涨，说明 `max-running-requests`、`mem-fraction-static` 或 `schedule-conservativeness` 设得太激进。

## 六、overlap scheduler：把 CPU 调度藏到 GPU 后面

Continuous batching 每步都要做不少 CPU 工作：收请求、匹配前缀、组 batch、准备 metadata、处理上一步结果、采样、流式返回。如果这些工作串行在 GPU 前向之后，GPU 会有明显的空泡。

SGLang 的 `event_loop_overlap()` 用一个小队列把 CPU 和 GPU 重叠起来：

```python
result_queue = deque()

while True:
    batch = self.get_next_batch_to_run(...)   # 组当前 batch
    if batch:
        result = self.run_batch(batch)         # 异步 launch 到 GPU
        result_queue.append((batch.copy(), result))

    # 处理上一个 batch 的结果；GPU 正在跑当前 batch
    if self.last_batch:
        tmp_batch, tmp_result = result_queue.popleft()
        self.process_batch_result(tmp_batch, tmp_result)

    self.last_batch = batch
```

时间线大概是这样：

```
CPU:  组 batch N  处理 N-1 结果  组 batch N+1  处理 N 结果
GPU:             跑 batch N                  跑 batch N+1
```

这样 CPU 的调度、采样、流式返回都被藏到 GPU 计算后面。默认开启；`--disable-overlap-schedule` 可以关掉，方便调试或排查显存问题。

有些边界会临时关闭 overlap：

- 投机解码的 grammar 状态需要同步；
- 连续两个 prefill batch 在特定环境变量下会被强制不 overlap（`SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP`，默认关闭）；
- 某些 batch 含有必须立即处理的依赖（比如 grammar FSM 的推进）。

overlap 的代价是**在途 batch 更多**：当前 batch 在 GPU 上跑，上一个 batch 的结果还在 CPU 队列里，两者的 KV 和临时 buffer 同时存在。显存紧张时，关掉 overlap 有时反而能多跑一点并发；这是吞吐和显存之间的另一组取舍。

另外，PP 场景还有 `--enable-two-batch-overlap`，那是把 micro-batch 的 forward/backward 交叠，和这里的 CPU/GPU overlap 不是一回事。

## 七、RadixAttention 感知的调度策略

这是 SGLang 和“普通 continuous batching”差异最大的地方：**调度器不只按到达顺序排队，还会看前缀缓存**。

### 七种策略

`SchedulePolicy.calc_priority()` 在每轮 prefill 前对 `waiting_queue` 排序：

| 策略 | 排序依据 | 适合 |
| :--- | :--- | :--- |
| `fcfs`（默认） | 先到先服务 | 通用、公平 |
| `lpm` | 最长前缀命中优先 | 多轮对话、共享 system prompt |
| `dfs-weight` | 按 Radix 树的 DFS 权重 | 让同一分支的请求连续被服务，放大缓存局部性 |
| `hrrn` | 最高响应比优先，按已处理 token 数老化 | 想兼顾短请求和防饿死 |
| `shortest-prefill-first` | 未缓存 token 最少优先（实验性） | 降低短请求 TTFT |
| `lof` | 最长输出优先 | 离线吞吐 |
| `random` / `routing-key` | 随机 / 按 routing key 聚簇 | 特定路由或实验场景 |

`lpm` 的实现很直接：对每个 waiting 请求做一次 `match_kv_cache()`，按 `num_matched_prefix_tokens` 降序排。命中越多的请求越先被服务，因为它要算的新 token 最少，而且它跑完之后能把更多前缀留在树里给后面的请求用。

但 `lpm` 有饥饿风险：一个前缀很短的请求可能永远排在后面。所以 SGLang 默认用 `fcfs`，把选择权交给用户；`hrrn` 用“已等待 token 数 / 未缓存 token 数”做 aging，是在缓存局部性和公平性之间的一个折中。

### in-batch prefix caching：别同时 prefill 两个一样的前缀

假设 waiting 队列里有 10 个请求，前缀都只有 20 个 token 相同，后面各不相同。如果它们一起被 prefill，每个请求都要算这 20 个 token，但第一个算完之后，后面 9 个本来可以命中——**同时 prefill 反而浪费了缓存机会**。

SGLang 在 `_compute_prefix_matches()` 里用一棵临时的 `waiting_queue_radix_tree` 检测这种情况：

```python
# 简化：prefix_ids = origin_input_ids + output_ids
if r.prefix_len <= IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD:  # 默认 32
    match = waiting_queue_radix_tree.match_prefix(prefix_ids)
    if match.device_prefix_len >= IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD:
        temporary_deprioritized.add(r.rid)   # 这个请求先压后
    else:
        waiting_queue_radix_tree.insert(prefix_ids)  # 记下它的前缀
```

效果是：**同一小前缀的多个请求，先服务一个，让它的 KV 进树，再服务其余的**。这样后续请求能命中，而不是大家同时从零算。这个优化只在 `prefix_len <= 32` 时触发，避免为长前缀做额外匹配；长前缀的收益已经足够大，不需要靠这个启发式。

### priority scheduling

`--enable-priority-scheduling` 打开后，请求可以带 `priority`；`calc_priority()` 会先按优先级排，再按策略排。`preempt_to_schedule()` 允许高优先级请求在准入时抢占低优先级请求；`--priority-scheduling-preemption-threshold` 控制优先级差多少才值得抢。

这套机制和 continuous batching 是正交的：continuous batching 决定“每步谁能进”，priority 决定“谁先进”。两者叠在一起，就是“高优先级请求可以插队，但插队也要过 KV 预算”。

## 八、PD、DP、PP：continuous batching 的几种变体

### PD 分离

PD 分离把 prefill 和 decode 拆成两个进程：prefill 进程只算 KV，decode 进程收 KV 后连续 decode。两边各自还在跑 continuous batching，只是一个只跑 prefill batch，一个只跑 decode batch。

好处是：

- prefill 不再打断 decode 的 ITL；
- prefill 和 decode 可以按比例独立扩缩容；
- decode 侧可以开 Radix 缓存，避免重复传输相同前缀。

代价是 KV 传输的带宽和延迟，以及 decode 侧要提前预留目标槽。它适合长 prompt、高并发、prefill/decode 比例不均衡的负载，不是所有场景都比统一调度快。

### DP attention

DP attention 下每个 DP rank 有自己的 `running_batch`，但 MLP/attention 的同步要求各 rank 在同一个 step 上达成一致。SGLang 用 `maybe_prepare_mlp_sync_batch()` 做全局同步，避免一个 rank 在 prefill、另一个在 decode 导致 collective 死锁。这也是为什么 SGLang 的调度器里会有 “decode → extend” 的转换：某些 rank 为了对齐，会把 decode batch 临时当成 extend 跑。

### PP

PP 把模型按层切到多个 stage，每个 stage 跑自己的 micro-batch。`pp_max_micro_batch_size` 限制每个 stage 同时处理的请求数；`--enable-two-batch-overlap` 让两个 micro-batch 的计算/通信交叠。调度器仍然按 iteration 组织，只是每个 iteration 要照顾多个 stage 的流水。

### prefill-decode interval

prefill 优先的代价是 decode 的 ITL 会被新请求打断。`--prefill-decode-interval N` 让调度器在跑完一个 prefill 后，连续跑 N 轮 decode 再考虑下一个 prefill。默认关闭（0），只在部分经过 profile 的配置里自动开启；它对“prefill 密集、decode 尾延迟敏感”的负载很有用。

## 九、和 vLLM v1 的对照

| 维度 | SGLang | vLLM v1 |
| :--- | :--- | :--- |
| 调度粒度 | 一次前向 | 一次前向 |
| 阶段 | 显式 prefill batch / decode batch，有 prefill 先跑 | 调度器里没有阶段，只有 `num_computed_tokens` 追平 |
| 混批 | 默认不混；`--enable-mixed-chunk` 可选 | 默认把 prefill chunk 和 decode 混在一个 varlen batch |
| 前缀缓存 | Radix 树，token/页粒度，参与排序和准入 | 块哈希链，block 粒度；调度器用它跳过已算 token |
| 准入顺序 | prefill 优先 | 先 running，后 waiting |
| 抢占 | `retract_decode()`，可备份到 CPU/host pool | 踢 `running[-1]`，recompute |
| 调度策略 | `fcfs` / `lpm` / `dfs-weight` / `hrrn` / ... | 以 FCFS 为主 |
| CPU/GPU 重叠 | `event_loop_overlap` | async scheduling（视版本） |
| PD | 一等公民，独立 prefill/decode 进程 | KV connector 生态 |

这张表只对比“机制”，不代表谁绝对更快。SGLang 的 prefill 优先适合 prefill 占比高、前缀复用强的负载；vLLM 的混批适合 decode 占比高、追求 ITL 平稳的负载。两边都在快速演进，具体版本要以代码为准。

## 十、调优与指标

| 旋钮 | 默认 | 调大 / 调小 |
| :--- | :--- | :--- |
| `--max-prefill-tokens` | 16384 | 调大：prefill 吞吐高，但单步更重、更容易打断 decode |
| `--chunked-prefill-size` | 按 GPU 2K–16K | 调大：长 prompt TTFT 好，ITL 差；调小反之 |
| `--max-running-requests` | 自动 | 调大：并发高，retract 风险高；调小：尾延迟稳 |
| `--schedule-policy` | `fcfs` | 换 `lpm` 提命中率，但可能饿死短前缀请求 |
| `--enable-mixed-chunk` | 关闭 | 打开：decode ITL 更平滑，batch 组装更复杂 |
| `--disable-overlap-schedule` | 关闭 | 打开：省在途显存，但 CPU 空泡变大 |
| `--prefill-decode-interval` | 0（部分配置自动开启） | 调大：decode 连续性强，prefill 排队变长 |
| `--schedule-conservativeness` | 1.0 | 调大：少 retract，低并发；调小反之 |
| `--retraction-policy` | `length` | 换 `priority`：按优先级踢人 |

看四个指标：

- **TTFT**：排队 + prefill 长度。chunked prefill 会拉长它；`lpm` 和高命中率会缩短命中请求的 TTFT。
- **ITL / TBT**：每步 batch 的构成。prefill 优先会抬高它；`prefill-decode-interval` 和混批能压平它。
- **吞吐**：batch 填满程度和 KV 利用率。`max-running-requests`、`mem-fraction-static`、chunk 大小都会影响。
- **retract 次数 / 缓存命中率**：前者高说明显存或并发设得太激进；后者低说明前缀复用没起来，可能是 `lpm` 没开、路由把相同前缀打散、或者缓存被驱逐得太快。

## 十一、几个常见误解

**“SGLang 没有 continuous batching，因为它没有这个开关。”** 有。`event_loop` 本身就是 continuous batching；没有开关是因为它是默认架构，不是一个可选优化。

**“必须开 `--enable-mixed-chunk` 才算 continuous batching。”** 不是。continuous batching 的核心是“每步重排、请求连续进出”，混批只是把 prefill 和 decode 放进同一个前向的额外优化。SGLang 默认分批跑，仍然是 continuous batching。

**“prefill 优先没有代价。”** 有。新请求的 prefill 会打断正在 decode 的请求，抬高 ITL；`--prefill-decode-interval` 就是为这个代价准备的。

**“被 retract 的请求一定从头重算。”** 不一定。Radix 树里还留着的部分会命中；PD 模式还可以备份到 CPU/host pool，恢复时直接 load。

**“chunked prefill 越大越好。”** 不是。chunk 越大，单步激活和 KV 分配越重，越容易把 decode 挤掉；chunk 越小，TTFT 越长，调度开销也越多。按 GPU 档位和负载调，不要盲目拉满。

**“continuous batching 等于无限并发。”** 不是。并发上限由 KV 槽、请求行、`max-running-requests` 共同决定；调度器只能决定谁先跑，不能凭空变出显存。

## 十二、小结

SGLang 的 continuous batching 可以概括成四句话：

1. **每步重排**：`waiting_queue`、`running_batch`、`chunked_req` 三个状态，完成即出、等待即进。
2. **prefill 优先**：有 prefill 先跑 prefill，没有才跑 decode；混批是可选项，不是默认。
3. **准入看缓存**：先查 Radix 前缀，再算 token/KV 预算；`lpm` / `dfs-weight` / `hrrn` 让调度器按缓存局部性排序，in-batch prefix caching 避免同时 prefill 相同小前缀。
4. **压力靠 retract 释放**：decode 装不下就按“短输出、长输入优先”回退，能重算就重算，不能就备份或 503。

想顺着代码读，建议顺序：

`scheduler.py` 的 `event_loop_normal()` / `event_loop_overlap()` → `get_next_batch_to_run()` → `get_new_batch_prefill()` / `_get_new_batch_prefill_raw()` → `schedule_policy.py` 的 `PrefillAdder.add_one_req()` / `add_chunked_req()` → `SchedulePolicy.calc_priority()` / `_compute_prefix_matches()` → `schedule_batch.py` 的 `update_running_batch()` / `retract_decode()`。

下一篇可以接着看 PD 分离和 HiCache 怎么把单机调度扩到多机；KV 侧的细节在上篇：[SGLang 的 KV Cache](/posts/llm-engine/sglang/kv-cache/)。
