---
title: "Continuous Batching：vLLM v1 调度器怎么排一个 batch"
published: 2026-11-02T23:50:00+08:00
description: 静态 batching 浪费的是 GPU 的空转时间；continuous batching 把调度粒度从请求降到一次前向。拆开 vLLM v1 的 schedule()：token 预算怎么切 chunked prefill、块不够时牺牲谁、SchedulerOutput 怎么变成一次不 padding 的 varlen 前向。
tags: [vLLM, Continuous Batching, 调度, chunked prefill]
category: LLM Engine
draft: false
---

同一个 batch 里，A 要跑 20 步、B 只跑 6 步、C 要跑 30 步。静态 batching 的做法是等最慢的那个——B 算完之后占着位置空转 14 步，新来的 D 只能等下一轮。GPU 在这段时间里既不缺算力也不缺显存，缺的是“被填满的 batch”。

Continuous batching 的全部内容就是：**把调度粒度从“一批请求”降到“一次前向”**。每跑完一步就重排一次，谁做完了谁走，谁在等谁进。

上篇讲显存怎么管：[vLLM 的 KV Cache](/posts/llm-engine/vllm/kv-cache/)。这篇讲这些块怎么被分给一批不断变动的请求。

> **五分钟版**
>
> - **静态 batching 的浪费**：短请求早算完只能空转，长请求决定整批的时间，队列延迟等于一整批。
> - **v1 的抽象**：调度器里没有 prefill 阶段也没有 decode 阶段，只有 `num_computed_tokens` 和 `num_tokens`——每步给请求分配一些 token，让它追平。
> - **预算单位是 token**：`max_num_batched_tokens` 每步扣减，所以 8K 的长 prompt 会被切成 chunk，和别人的 decode 混在一批里跑。
> - **抢占只有一种**：KV 块不够时，把最晚进入的请求整条踢回等待队列，`num_computed_tokens` 归零，下次从头算（recompute）。v1 没有 v0 的 swap。
> - **输出是一张契约**：`SchedulerOutput` 里的 `num_scheduled_tokens` + `block_tables` + `slot_mapping`，让 model runner 用 varlen 方式把长短不一的 query 打包进一次前向。
> - **代价**：TTFT、ITL、吞吐三者互斥，`max_num_batched_tokens` 调大换吞吐、换更差的 ITL。
>
> 代码口径：vLLM `main` @ `fa2881ed19`（2026-11），入口是 `vllm/v1/core/sched/scheduler.py` 的 `schedule()`。

## 一、静态 batching 浪费的是什么

```
静态 batching（等整批结束）
               step: 1      6            20           30
req A            ██████████████████████████████
req B            ██████░░░░░░░░░░░░░░░░░░░░░░░░
req C            ████████████████████████████████████
req D            ...............................  (要等下一轮)
```

B 在第 6 步就算完了，位置却一直占到第 20 步：这段时间 GPU 不是算不动，是没活干。

三个直接后果：

- **短请求被长请求拖住**。整批的完成时间由最慢的序列决定，短序列提前完成也不释放资源。
- **队列延迟等于一整批**。新请求必须等到当前批次整体结束才能进 GPU。
- **batch size 被迫取小**。为了控制尾延迟，只能限制每批请求数，GPU 又填不满——两头不讨好。

问题的根源在于调度粒度：以“请求”为单位调度，就必须等一个请求完整跑完才能动下一个。

## 二、iteration-level scheduling：每一步都重排

把粒度换掉就解决了：**每做完一次模型前向，重新决定下一个 batch 由谁组成**。

```
continuous batching（每步重排）
step: 1  2  3  4  5  6  7  8 ...
A:    █  █  █  █  █  █  █  █
B:    █  █  █  █  █  █              ← 第 6 步完成，立刻退出
C:    █  █  █  █  █  █  █  █  █  █
D:          █  █  █  █  █  █  █  █   ← 第 3 步就补位进来
```

- 完成的请求当步退出，空出的槽位、显存块立即归还；
- 等待的请求下一个调度周期就能进入，排队延迟从“一整批”降到“一步”；
- batch 的组成每一步都可能不同，GPU 始终被填满。

这个想法在 2019 年就被提出来了，但真正落地要等一个前提：**显存能按需、细粒度地借还**。按序列预分配连续显存的话，batch 变一次就要重新分配一次，碎片和拷贝会吃掉所有收益。PagedAttention 提供的分页块管理，正好让“随时进、随时出”变得廉价——这就是上篇和这篇必须连在一起看的原因。

## 三、v1 的核心抽象：只有“追平”

很多资料把调度讲成两个阶段：先 prefill，再 decode，两批分开跑。vLLM v1 把这两个阶段**从调度器里删掉了**。源码里的注释是整个设计的钥匙（`scheduler.py`，`schedule()` 开头）：

```python
# NOTE(woosuk) on the scheduling algorithm:
# There's no "decoding phase" nor "prefill phase" in the scheduler.
# Each request just has the num_computed_tokens and
# num_tokens_with_spec. ...
# At each step, the scheduler tries to assign tokens to the requests
# so that each request's num_computed_tokens can catch up its
# num_tokens_with_spec.
```

每个请求只有两个数字：

- `num_computed_tokens`：KV 已经算到哪了；
- `num_tokens_with_spec`：这个请求一共有多少 token 需要被算（prompt + 已生成 + 草稿）。

调度器每步做的事，就是**给每个请求分配若干个 token，让它往前追一截**。于是三种看起来不同的行为被统一了：

| 叫法 | 实际含义 | 每步追多少 |
| :--- | :--- | :--- |
| prefill | 从头追 | prompt 长度（可能被预算截断） |
| chunked prefill | 追一部分，剩下的留给后面的 step | 预算允许的那部分 |
| decode | 追 1 个 | 1（或投机解码的 k+1） |
| 抢占后重算 | 把 `num_computed_tokens` 打回 0 再追 | 从头（前缀缓存能捞回一部分） |
| 投机解码 | 追 k 个草稿，验证后回退 | 1 + k |

统一之后，代码里就不需要“prefill 分支”和“decode 分支”了。没有阶段，就没有阶段切换的开销，也没有“prefill 批次占满 GPU、decode 饿死”这类经典问题。

## 四、一步 `schedule()` 的真实顺序

简化成伪代码，骨架是这样的：

```python
token_budget = max_num_batched_tokens      # 本步的 token 预算

# 1. 先服务正在跑的请求：保证 decode 的 ITL 不被新请求破坏
for req in running:
    n = min(req.num_tokens_with_spec - req.num_computed_tokens, token_budget)
    while kv_cache.allocate_slots(req, n) is None:   # 块不够
        preempt(最晚进入的 running 请求)              # 2. 抢占，直到能放下或放弃
    token_budget -= n

# 3. 用剩下的预算接纳等待队列里的新请求（近似 FCFS）
while token_budget > 0 and len(running) < max_num_seqs:
    req = waiting[0]
    hit = kv_cache.get_computed_blocks(req)          # 前缀缓存
    if kv_cache.allocate_slots(req, n, hit) is None:
        break                                        # 放不下就停，不强塞
    token_budget -= n

return SchedulerOutput(num_scheduled_tokens, block_tables, slot_mappings, ...)
```

几个顺序上的决定值得单独说：

**先 running，后 waiting。** 正在解码的请求几乎不花预算（1 个 token），却直接决定 ITL（token 间延迟）。先安排它们，新请求就只能用剩下的预算，突发流量不会把已在跑的请求挤到卡顿。

**预算按 token 扣，不按请求数。** 一个 8K prompt 会一次吃掉全部预算，如果按请求调度，它会独占这一步、把别的请求全部挡在后面。按 token 扣，它自然被切成 chunk。

**等待队列近似 FCFS，但会跳过放不下的。** 队首请求如果显存不够，不是死等，而是跳过它看下一个（代码注释明确说这不是严格 FCFS）。这样能减少队首阻塞，代价是可能有轻微的不公平。

**准入受两道闸限制**：`token_budget` 和 `max_num_seqs`（同时在跑的序列数上限）。此外还有一个更细的准入检查 `full_sequence_must_fit`：chunked prefill 容易只看第一块就放行，导致一条长序列跑到一半才发现装不下，所以可选地先按完整序列长度验一次。

## 五、预算单位是 token，所以有了 chunked prefill

`max_num_batched_tokens` 是这套设计的核心旋钮。它的默认值不是常数，而是按设备档位解析出来的（`arg_utils.get_batch_defaults`）：

| 设备档位 | 离线推理（`LLM`） | 在线服务（API server） |
| :--- | :--- | :--- |
| 小显存 / A100 / GB10 | 8192 | 2048 |
| H100 / H200（≥70 GiB） | 16384 | 8192 |
| ≥160 GiB（B200/B300 等） | 16384 | 16384 |

背后的判断很朴素：**离线批处理追求吞吐，越大越好；在线服务追求延迟，越小越稳**。所以同一个模型，`vllm serve` 和 `LLM(...)` 拿到的默认值可以差 8 倍。

因为预算按 token 计，长 prompt 天然被切成若干 chunk：

```
一步的 token 预算 = 2048
┌──────────────┬──────────────┬─────────────────┐
│ decode A (1) │ decode B (1) │ prefill C chunk │ ...
│              │              │  (2046 tokens)  │
└──────────────┴──────────────┴─────────────────┘
```

这就是 **chunked prefill**：把一次长 prefill 拆到多个 step 里做，中间穿插别人的 decode。它带来两个好处——长 prompt 不再阻塞 decode（ITL 平稳），GPU 每步都被填满；代价是长 prompt 自己的 TTFT 变长（要等好几个 step 才轮到它出第一个 token）。

v1 里 `enable_chunked_prefill` 默认为 `True`。即便手动关掉，调度器里仍然没有阶段概念——变化只是 prefill 不能再跨 step 拆分，必须等预算够装下整段才能被准入。

还有几个更细的节流手段，都是同一个思路的不同变体：

- `long_prefill_token_threshold`：给单个请求每步能吃的 token 数设上限，防止一条超长 prompt 把预算吃干；
- `prefill_schedule_interval`：按步数节流 prefill，让 decode 能连续跑若干步不被打断；
- DP 场景下的 prefill 均衡：多副本之间错开 prefill，避免所有 rank 同时进 prefill 造成气泡。

## 六、抢占：块不够时牺牲谁

`allocate_slots()` 返回 `None` 只有一个含义：显存块不够了。这时候调度器必须做选择：

1. **抢占一个正在跑的请求**，把它的块全部释放，凑出空间；
2. 如果连抢占都凑不出来（或者把自己也抢了），这个请求本步就不调度。

默认策略下，victim 是 `self.running[-1]`——**最晚进入 running 的那个**。道理和操作系统的调度一样：新来的请求已经拿到的进度最少，踢它的代价最小；先来的请求继续跑，保证公平性和尾延迟。

抢占的动作本身很短（`_preempt_request`）：

```python
self._free_request_blocks(request)      # 块全部还回池子
request.status = RequestStatus.PREEMPTED
request.num_computed_tokens = 0         # 进度清零
request.num_preemptions += 1
self.waiting.prepend_request(request)   # 插回等待队列队首
```

注意 **v1 只有 recompute 这一种抢占形式**：不搬 KV、不落盘，直接把进度丢掉，下次从第一个 token 重新算。被释放的块如果带着哈希，仍留在前缀缓存里——所以真正重算的往往只是最后那一小段，前缀命中的部分不用重算。

v0 曾经有 swap 模式（把 KV 拷到 CPU 再拷回来），v1 把它删了。原因很实际：PCIe 上搬 KV 的带宽远低于 GPU 重算的等效带宽，尤其是在前缀缓存已经能捞回大部分的情况下，**swap 是负优化**。真要跨层放 KV，v1 走的是 KV connector / offload 这条路，面向的是跨实例共享和容量扩展，不是单机内的临时换出。

抢占的代价是真实的：高负载下反复抢占会让 TTFT 抖动、让尾延迟变差，`num_preemptions` 涨起来通常说明 `gpu_memory_utilization` 或 `max_num_seqs` 设得太激进。

## 七、从 `SchedulerOutput` 到一次前向

调度器算完之后，交给 worker 的是 `SchedulerOutput`（`vllm/v1/core/sched/output.py`）。它是两端唯一的契约，字段可以分成四组：

| 字段 | 作用 |
| :--- | :--- |
| `scheduled_new_reqs` / `scheduled_cached_reqs` | 新请求带完整数据；老请求只发增量（worker 侧有缓存） |
| `num_scheduled_tokens` | 每个请求本步算几个 token，整个 batch 的组装依据 |
| `finished_req_ids` / `preempted_req_ids` | 通知 worker 释放状态 |
| `kv_connector_metadata` 等 | 可选的 KV 传输/编码器元数据 |

worker 侧拿到它之后做三件事：

```python
# 1. 把新的块号追加到 block table 的对应行
input_batch.block_table.append_row(new_block_ids, req_index)

# 2. 按 positions 算出每个 token 的物理槽位
input_batch.block_table.compute_slot_mapping(num_reqs, query_start_loc, positions)

# 3. 把长度不一的 query 打包：query_start_loc 是前缀和，不需要 padding
```

关键在第三点。一个 batch 里可能同时有 query 长度 1 的 decode 和长度 2000 的 prefill chunk，普通实现会 padding 到最长，浪费大量算力；vLLM 用 varlen 接口（`query_start_loc` / `cu_seqlens`、`seq_lens`、`block_table`）把 token 直接拼成一条，attention kernel 按段处理。这也是“没有 prefill/decode 阶段”能在 kernel 层成立的原因：**kernel 眼里只有 token 和段，没有阶段**。

## 八、旋钮和指标

| 旋钮 | 默认（本版本） | 调大 | 调小 |
| :--- | :--- | :--- | :--- |
| `max_num_batched_tokens` | 2048 / 8192 / 16384（按设备） | 吞吐↑，TTFT 平稳但 ITL 变差 | 延迟↓，吞吐↓，GPU 填不满 |
| `max_num_seqs` | 128（字段）/ 256–1024（解析后） | 并发↑，显存压力↑，抢占↑ | 并发↓，尾延迟更稳 |
| `gpu_memory_utilization` | 0.92 | 更多 KV 块、更少抢占 | 更容易 OOM 或频繁抢占 |
| `enable_chunked_prefill` | `True` | — | 仅在个别模型/场景关闭 |
| `long_prefill_token_threshold` | 0（不限） | 单请求少吃，公平性↑ | 长 prompt 更快追上 |

三个指标要一起看，它们互相拉扯：

- **TTFT**（首 token 延迟）：由排队 + prefill 长度决定。chunked prefill 会拉长它；
- **ITL / TBT**（token 间延迟）：由每步 batch 的构成决定。prefill chunk 塞进来就会抬高；
- **吞吐**（tokens/s）：由 batch 填满程度和显存利用率决定。

提高 `max_num_batched_tokens` 通常是“拿 ITL 换吞吐”，降低则是反过来。在线服务一般把 `max_num_batched_tokens` 压到 2048–8192，离线批处理拉到 16384 甚至更高——这也是官方默认值按档位解析的原因。

## 九、三个常见误解

**“continuous batching 就是无限并发。”** 不是。并发上限由显存块数决定，不是由调度器决定。调度器只能决定“谁先进”，不能凭空变出块；`max_num_seqs` 和 KV 块数才是硬约束，越过就会触发抢占，而抢占的代价是重算。

**“prefill 和 decode 必须分两批跑。”** v1 恰恰相反：它们混在同一个 batch 里，靠 varlen 打包，不需要 padding 对齐。分阶段是 v0 和很多其他实现的思路，不是必然。

**“被抢占的请求会从零开始重算整个 prompt。”** 通常不会。块被释放时哈希还在前缀缓存里，重新调度时会命中大部分前缀，实际重算的往往只有最后几个块。

## 十、小结

把两篇连起来看，分工非常清楚：

- **KV Cache** 决定“东西放在哪、能放多少、能不能共享”——它给调度器一个可以随时借还的池子；
- **Continuous Batching** 决定“每一步把池子分给谁、分多少”——它把这个池子的利用率推到接近上限。

缺了分页，batch 每变一次就要重排显存，continuous batching 会在碎片上撞墙；缺了 continuous batching，分页只是省了点内存，换不来吞吐。两者合起来，才是“同样的卡跑出几倍吞吐”的那套东西。

想顺着代码读一遍，建议顺序：`kv_cache_utils.py`（块与空闲队列）→ `block_pool.py`（分配/释放/哈希）→ `kv_cache_manager.py`（`allocate_slots` 的布局图）→ `sched/scheduler.py`（`schedule()` 主循环）→ `sched/output.py`（`SchedulerOutput` 契约）。
