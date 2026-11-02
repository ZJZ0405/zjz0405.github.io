---
title: "SGLang 的 KV Cache：RadixAttention、分页池与 HiCache"
published: 2026-11-03T01:20:00+08:00
description: 从显存账出发拆 SGLang 的 KV 管理：ReqToTokenPool 与 TokenToKVPoolAllocator 怎么分工、RadixAttention 的前缀树怎么 match/insert/split、lock_ref 与 LRU 怎么驱逐、HiCache 怎么把 KV 扩到主机与远端，以及 SWA/Mamba/MLA 这些新模型带来的多池结构。
tags: [SGLang, KV Cache, RadixAttention, HiCache, 显存管理]
category: LLM Engine
draft: false
---

SGLang 的 KV Cache 常被一句话概括成 RadixAttention。但 RadixAttention 只回答了“怎么复用”，没有回答“内存在哪、怎么分配、满了怎么办”。把 `python/sglang/srt/mem_cache/` 拆开看，它其实是四层：请求行、槽位分配器、物理池、前缀树；再往外还有 HiCache 的三级缓存和 PD 分离的跨机传输。

这篇只讲 KV 怎么管，不讲调度。调度在下篇 [SGLang 的 Continuous Batching](/posts/llm-engine/sglang/continuous-batching/)。

> **五分钟版**
>
> - **显存账**：`2 × 层数 × KV heads × head_dim × dtype` 每 token。Llama-3-8B fp16 是 128 KiB/token；MLA 把每层压成 576 维，DeepSeek-V3 一类模型约 34 KiB/token。KV 决定并发上限。
> - **四层分工**：`ReqToTokenPool` 存“请求的逻辑位置 → 物理 token 槽”，`TokenToKVPoolAllocator` 发号和回收，`KVCache` 持有每层 K/V 张量，`RadixCache` 只存前缀树和索引，不存 KV 数据。
> - **不按 `max_model_len` 预留**：prefill 只分配未命中的后缀，decode 每步只分配新 token；默认 `page_size=1`，每个 token 就是一个槽，`page_size>1` 时切到页分配器。
> - **前缀复用**：RadixAttention 是一棵按 token 序列分叉的基数树，`match_prefix` 找最长前缀，`insert` 把新前缀挂上去，命中停在节点中间就 `split`。`lock_ref` 保护在跑请求，`evict` 只回收没被锁的叶子。
> - **回收有三层**：驱逐缓存叶子、retract 在跑请求、准入时排队。默认 LRU，另有 LFU / SLRU / priority / TLRU；decode 显存不够时按“短输出、长输入优先”回退。
> - **扩容量**：HiCache 把 GPU 当 L1、主机内存当 L2、Mooncake/3FS/NIXL 等当 L3；L2 是实例私有的，L3 才可能跨实例共享。
> - **异构模型**：SWA 层和 full 层分池，Mamba/线性注意力用状态池，UnifiedRadixCache 把 Full / SWA / Mamba 当作同一棵树上的三个 component。
>
> 代码口径：SGLang `main` @ `a99c7d0`（2026-11）。路径都在 `python/sglang/srt/mem_cache/` 下。

## 一、先把账算清楚

自回归解码第 `t` 步只需要算第 `t` 个 token 的 Q，但 attention 要读 `0..t` 的全部 K/V。因果 mask 保证历史 K/V 不变，所以缓存住就能把每步的投影从 `O(t)` 降到 `O(1)`：

```
不缓存：第 t 步重算 0..t 的 K/V        → 总计 O(n²) 次投影
缓存：  第 t 步只算 t 的 K/V，读历史   → 总计 O(n) 次投影
```

代价是显存按 token 线性增长：

$$
\text{bytes/token} = \underbrace{2}_{K,V} \times L_{\text{layers}} \times H_{\text{kv}} \times d_{\text{head}} \times \text{sizeof(dtype)}
$$

以 Llama-3-8B 为例：32 层、8 个 KV head（GQA）、head_dim 128、fp16，每个 token `2 × 32 × 8 × 128 × 2 = 131072` 字节，正好 **128 KiB**。32K 上下文的一条序列就是 **4 GiB**。

MLA 是另一条路：它不存完整的 per-head K/V，而是把每层压成 `kv_lora_rank + qk_rope_head_dim` 的 latent。DeepSeek-V3 是 512 + 64 = 576 维、61 层、FP8，每 token 约 `576 × 61 ≈ 35 KB`。如果同样的 61 层按 128 个 KV head、128 维、BF16 的 MHA 估算，每 token 接近 4 MiB——差两个数量级。**所以“KV Cache 策略”不只包含分配器，也包含模型结构本身：MLA、GQA、SWA、Mamba 都是在改这笔账。**

| 项目 | 量级 | 说明 |
| :--- | :--- | :--- |
| 8B 模型权重（fp16） | 16 GiB | 固定开销 |
| Llama-3-8B 单条 32K 序列 | 4 GiB | 随上下文线性增长 |
| Llama-3-8B 单条 128K 序列 | 16 GiB | 长上下文按 token 吃显存 |
| DeepSeek-V3 MLA（FP8） | ≈ 34 KiB/token | 61 层、576 维 |
| H100 80G 可用于 KV | ≈ 55 GiB | 扣掉权重、激活、CUDA graph 后的粗估 |

结论很直接：**权重决定能不能装下模型，KV 决定能同时跑多少请求**。SGLang 的所有 KV 设计，都是在“少浪费、多复用、能扩出去”这三件事上做文章。

## 二、四层结构：树管索引，池管字节

先看整体。SGLang 把“请求 → token 位置 → 物理槽 → K/V 字节”拆成了四层：

```
Req（一条请求）
 │
 ├─ ReqToTokenPool       req_pool_idx → [token slot ids ...]
 │                        逻辑位置 i 的 KV 在哪个物理槽
 │
 ├─ TokenToKVPoolAllocator   发号 / 回收（page_size=1 时是 token 分配器）
 │
 ├─ KVCache             每层 K/V 张量：[slots, heads, dim]（大页时按页组织）
 │
 └─ RadixCache          前缀树；节点 value 指向上面那些 slot ids
```

| 结构 | 位置 | 职责 |
| :--- | :--- | :--- |
| `ReqToTokenPool` | `memory_pool.py` | 每个请求一行 `req_to_token`，`int32`，形状 `(size+1, max_context_len)`（`size` 是请求槽数）；行内是这条请求每个位置的物理槽号 |
| `TokenToKVPoolAllocator` | `allocator/token.py` | `page_size=1` 时的槽位分配器；`alloc` / `free` 管理空闲槽 |
| `PagedTokenToKVPoolAllocator` | `allocator/paged.py` | `page_size>1` 时的页分配器；`alloc_extend` / `alloc_decode` 按页发号 |
| `KVCache` | `memory_pool.py` | 真正持有每层 K/V 的显存；逻辑上是 `[num_slots, heads, dim]`，`page_size>1` 时物理上按页组织；按 `layer_id` 取 `k_buffer` / `v_buffer` |
| `RadixCache` | `radix_cache.py` | 前缀树；节点只存 token 片段和物理索引，不存 K/V 张量 |

这个分层的价值在于**前缀树和物理池可以各自演进**。树要回答“这段前缀有没有算过”，池要回答“哪个槽空着”；两者通过一个 `torch.Tensor` 的索引数组连接。HiCache 后来能把 L2/L3 接进来，也是因为树里的 `value` 本来就不是数据本身，而是“数据在哪”的句柄。

`ReqToTokenPool` 里有一个细节：它多留了一行 0 作为 padding。CUDA graph 的 padded batch 默认把 `req_pool_indices` 指向 0，这样 dummy 读写落到空行上，不会破坏真实请求。

## 三、分配：没有按 `max_model_len` 预留

最朴素的实现是请求进来就按 `max_model_len` 切一块连续显存。SGLang 不这么做：它只分配**这次前向真正要写的 token**。

### prefill：只分配未命中的后缀

请求进入 `PrefillAdder` 之前会先做一次前缀匹配，得到 `prefix_len`。prefill 阶段真正要算的是 `input_len - prefix_len` 个 token，分配器也只为这段后缀发号：

```python
# 简化：alloc_extend 只处理 prefix 之后的新 token
out_cache_loc = allocator.alloc_extend(
    prefix_lens,      # 每条请求命中的前缀长度
    seq_lens,         # 每条请求前向结束后的总长度
    last_loc,         # 前缀最后一个 token 的物理槽
    ...
)
```

命中部分直接复用树里已有的物理槽，不需要拷贝。`alloc_extend` 要处理三种情况：补齐前缀所在页的尾部、分配整页、分配最后一页的尾部；所以它比“一次 `alloc(n)`”复杂，但换来的是**只为新 token 付显存**。

### decode：每步只分配新 token

decode 阶段每条请求每步只多一个 token（投机解码会多几个），所以 `alloc_decode` 每步只发一个槽；只有跨页边界时才发新页。释放则按请求整行回收，或者按页段回收。

```
prefill A: prompt 5000, 命中 4096  → 只分配 904 个槽
decode A:  第 1 步                → 只分配 1 个槽
           第 2..N 步             → 每步 1 个槽
finish A:  整行释放（树保留的部分除外）
```

### `page_size`：默认 1，但不是固定 1

`--page-size` 不传时，SGLang 默认解析成 **1**（MUSA、部分模型/backend override 除外）。这不是“没有分页”，而是**页粒度等于 token**：

- 普通 MHA/MLA 路径下，`page_size=1` 时走 `TokenToKVPoolAllocator`，每个 token 一个槽，前缀可以按 token 精确共享；
- `page_size>1` 时走 `PagedTokenToKVPoolAllocator`，一次发一整页，`RadixKey` 在匹配前会先按页对齐，尾部不足一页的部分不进树；
- 某些 attention backend 会强制页大小，比如 FlashMLA / TRT-LLM MLA 要求 64，TRT-LLM MHA 支持 16/32/64/128。启动时 `_mla_backend_page_constraints` 这类 hook 会自动改并打 warning。

| `page_size` | 好处 | 代价 |
| :--- | :--- | :--- |
| 1 | 前缀命中粒度最细，尾部浪费最少 | 页表/索引更长，attention 的 gather 次数多 |
| 16 / 32 | 元数据变短，kernel 更顺 | 尾部最多浪费一页，前缀命中按页对齐 |
| 64 / 128 | 大页更适合 MLA/TRT-LLM 的 kernel | 短请求尾部浪费明显，共享粒度变粗 |

所以“SGLang 用不用 PagedAttention”要分两层回答：**物理上它一直是通过索引数组做非连续寻址；页大小是可选参数，默认 1。** 这和 vLLM 默认 `block_size=16` 是同一个设计空间里的两个点，不是有无之分。

## 四、RadixAttention：前缀树而不是哈希表

前缀缓存最直接的实现是哈希表：把整段 prompt 哈希一下，命中就复用。但哈希表只能做**整段精确匹配**，不能回答“这两个请求共享前 3000 个 token，后面分叉”的情况。RadixAttention 用一棵基数树（radix tree）解决这个问题。

### 节点存什么

```python
class TreeNode:
    children: dict          # key = 这段 token 的第一个 token（page_size>1 时是第一页）
    parent: TreeNode
    key: RadixKey           # 这个节点覆盖的 token 片段
    value: torch.Tensor     # 这段 token 在 KV 池里的物理索引
    lock_ref: int           # 被多少在跑请求锁住
    hit_count: int          # 被命中过多少次（LFU/SLRU/TLRU 用）
    last_access_time: float # LRU 用
    priority: int           # priority 驱逐策略用
```

一条从根到叶的路径拼起来，就是一条完整前缀；路径上每个节点的 `value` 拼起来，就是这条前缀的 KV 物理索引。**树里没有 K/V 数据，只有“这段 token → 这些槽”的映射。**

### match：最长前缀

`match_prefix` 从根开始，按 `key.child_key(page_size)` 找子节点，逐段比较：

```python
node = root
while key:
    child = node.children.get(key.child_key(page_size))
    if child is None:
        break
    n = child.key.match(key, page_size)
    if n < len(child.key):
        # 命中停在节点中间：把节点切开，露出精确边界
        node = split_node(child, n)
        break
    key = key[n:]
    node = child
return node
```

命中停在节点中间时，SGLang 会 `split` 出一个新节点，把“公共前缀”和“分叉后缀”分开。比如树里已有 `[1,2,3,4]`，新请求是 `[1,2,5,6]`：

```
split 前：root → [1,2,3,4]

split 后：root → [1,2] → [3,4]
                    └→ [5,6]   （新请求挂上去）
```

`split` 不拷贝 K/V，只把 `value` 张量按边界切开，然后把新节点接回父节点。之后两个请求共享 `[1,2]` 的槽，各自持有后缀。

### insert：把新前缀挂上去

`insert` 先走一遍和 `match` 类似的路径，把重叠部分复用，把剩余部分挂成新叶子；重叠段里请求自己多分配的那部分物理槽会还给分配器（因为树已经持有了同一段 KV）。这也是为什么 `RadixCache` 的 `insert` 需要和 `TokenToKVPoolAllocator` 配合：**树接管前缀的“所有权”，请求行里重复的槽要释放，否则会泄漏。**

### 三个容易忽略的字段

- **`lock_ref`**：请求命中一个节点后，会沿路径给所有祖先 `lock_ref += 1`。只要 `lock_ref > 0`，这个节点就不在 `evictable_leaves` 里，不会被驱逐。请求结束再 `dec_lock_ref`。
- **`extra_key` / `cache_salt`**：命名空间。不同 LoRA adapter、不同 sampling salt、不同租户的相同 token 序列不应该共享 KV；这两个字段把树和 L3 存储隔离开。
- **`hit_count` / `last_access_time` / `priority`**：给不同驱逐策略打分用。`lru` 只看时间，`lfu` 看命中次数，`slru` 看“probationary / protected”两段，`priority` 看请求优先级，`tlru` 看“这条会话的尾部是否已经超出 TTFT 预算”。

## 五、回收：驱逐、retract、排队

KV 池满了之后，SGLang 有三层动作，从便宜到贵：

### 第一层：驱逐缓存叶子

`RadixCache.evict(num_tokens)` 维护一个 `evictable_leaves` 集合：节点有 KV、`lock_ref == 0`、且没有仍持有 KV 的子节点。每次从堆里取优先级最低的叶子，释放它的 `value` 段，然后删掉；父节点如果因此变成叶子，再入堆。

```python
while num_evicted < num_tokens and heap:
    _, leaf = heapq.heappop(heap)
    allocator.free_segment(leaf.value, start_pos=0)
    delete_leaf(leaf)
    if parent_is_leaf(parent) and parent.lock_ref == 0:
        heapq.heappush(heap, (policy.get_priority(parent), parent))
```

在经典 `RadixCache` 里，**只驱逐叶子**是正确性的前提：内部节点的 KV 被所有子节点共享，删掉它等于让所有后代失去前缀。先删叶子，父节点才可能变成新叶子，驱逐自然沿着分支从末端往根走。UnifiedRadixCache 的 SWA/Mamba component 会用 tombstone 处理内部节点，但 Full 的驱逐仍然遵守这条规则。

| 策略 | 先驱逐谁 | 适用场景 |
| :--- | :--- | :--- |
| `lru`（默认） | 最久没被访问的 | 通用；前缀复用随时间衰减 |
| `lfu` | 命中次数最少的，再按 LRU | 有一小撮 prompt 被反复用，想让它扛住一次性流量 |
| `slru` | probationary 段优先，再按 LRU | 想给“只出现一次”的前缀设一个硬上限 |
| `priority` | 请求优先级最低的，再按 LRU | 开了 priority scheduling，希望缓存保留和准入同序 |
| `tlru` | 会话尾部超出 TTFT 预算的 token（TEL-safe），再按 LRU | 多轮 agent，尾延迟比命中率更重要 |

`fifo` / `mru` / `filo` 也在策略注册表里，但命令行不暴露，主要给实验用。策略只负责**打分**，不负责“钉住”内存；即使某个节点分数很高，只要驱逐它能凑够空间，它还是会被删。

### 第二层：retract 在跑请求

decode 时 `check_decode_mem()` 发现下一步的 KV 放不下，`update_running_batch()` 会调用 `retract_decode()`：按“最不想要”的顺序把整条请求踢回 waiting 队列，释放它的 KV 行。

默认 `retraction_policy="length"` 的 victim 顺序是 `(output_len, -input_len)` 升序弹出——**先 retract 输出短、输入长的请求**。理由是这种请求已经生成的部分少，重算代价相对可控；长输入往往还能靠 Radix 前缀命中捞回大部分。开了 `priority` 就按优先级先踢低优先级。

被 retract 的请求不是一定从头算：

- 默认走 recompute，下次重新 prefill；Radix 树里还留着的部分会命中；
- PD decode 可以配 `--disaggregation-decode-retraction-backup cpu_tensor|host_pool`，把 KV 备份到 CPU 或 host pool，恢复时直接 load 回来；
- 如果连备份都没有（`none`），请求会被 abort，客户端收到 503。

`new_token_ratio` 会随着 retract 调整：初始按 `max_new_tokens` 的 0.7 估计未来占用，随 decode 步数衰减到约 0.1；`schedule_conservativeness` 越大，初始估计越保守，越不容易 retract，但并发也越低。

### 第三层：准入时排队

`PrefillAdder` 在接纳新请求前会检查 `rem_total_tokens`（可用槽 + 可驱逐缓存 - 已承诺的 decode 预留）、`max_running_requests`、`max_prefill_tokens` 和 `chunked_prefill_size`。放不下就返回 `NO_TOKEN`，请求留在 `waiting_queue`，等下一轮。**调度器只回答“能不能”，不能就排队或 retract，不会让分配器在 forward 里抛 OOM。**

## 六、HiCache：把 KV 从显存扩到主机和远端

GPU 显存再大也有上限，而多轮对话、长 system prompt、few-shot 模板会让同一段前缀被反复使用。HiCache 的思路是 CPU 的三级缓存：**L1 = GPU，L2 = 主机内存，L3 = 分布式存储**。

```
        ┌─────────────── 单实例 ───────────────┐
        │  L1 GPU HBM     RadixTree + KV 池     │
        │        ↕ H2D/D2H                       │
        │  L2 host DRAM   HiRadixTree + host 池  │
        └───────────────┬───────────────────────┘
                        ↕
        ┌───────────────▼───────────────────────┐
        │  L3 storage: file / Mooncake / 3FS /   │
        │  NIXL / AIBrix / TensorCast / ...      │
        └───────────────────────────────────────┘
```

| 层级 | 介质 | 共享范围 | 说明 |
| :--- | :--- | :--- | :--- |
| L1 | GPU HBM | 单实例 | 正在用和刚用过的 KV |
| L2 | 主机 DRAM | **单实例私有** | 同一台机器上的另一个实例也读不到 |
| L3 | 存储后端 | 取决于后端配置 | 只有 Mooncake/3FS/NIXL 这类分布式后端（或共享挂载的 `file`）才可能跨实例 |

一个常见误解是“把 `--hicache-ratio` 调大就能把多台机器的内存拼成一个 L2”。不能。L2 是进程私有的 host 池；跨实例复用必须走 L3，并且每个实例都指向同一个 namespace。

### 读写流程

1. **本地匹配**：`match_prefix` 同时看 L1 和 L2，返回 `device_prefix_len` 和 `host_hit_length`；
2. **L3 prefetch**：本地没命中的部分，如果 L3 命中超过阈值（默认 256 token），异步预取到 L2/GPU；预取策略 `best_effort` / `wait_complete` / `timeout` 决定等多久；
3. **prefill**：只算真正没命中的后缀；host 命中的部分通过 H2D 拷回；
4. **写回**：请求结束后按 write policy 写到 L2/L3。

### 写策略

| 策略 | 什么时候写下一层 | 适合 |
| :--- | :--- | :--- |
| `write_through`（默认） | 每次访问都立即写 | 带宽足够，想要最强缓存收益 |
| `write_through_selective` | 命中次数超过阈值才写 | 只备份热数据，降低 I/O |
| `write_back` | 被上层驱逐时才写 | 存储容量有限，想尽量榨干内存 |

### 预取策略

| 策略 | 行为 | 取舍 |
| :--- | :--- | :--- |
| `best_effort` | GPU 能开始 prefill 就停 | 延迟最低，命中率可能低 |
| `wait_complete` | 等所有预取完成 | 命中率最高，TTFT 可能变长 |
| `timeout`（默认） | 超时或完成就停 | 用 `prefetch_timeout_base` / `per_ki_token` / `max` 调平衡 |

### 常用旋钮

```bash
python3 -m sglang.launch_server \
  --model-path MODEL_PATH \
  --enable-hierarchical-cache \
  --hicache-ratio 2.0 \
  --hicache-write-policy write_through \
  --hicache-storage-backend mooncake
```

| 参数 | 作用 |
| :--- | :--- |
| `--enable-hierarchical-cache` / `--enable-hicache` | 打开三级缓存 |
| `--hicache-ratio` | host 池 / device 池的比例，cache 模式默认 2.0 |
| `--hicache-size` | 直接指定 host 池 GB 数，优先级高于 ratio |
| `--hicache-write-policy` | `write_through` / `write_through_selective` / `write_back` |
| `--hicache-storage-backend` | `file` / `mooncake` / `hf3fs` / `nixl` / `aibrix` / `tensorcast` / `seaweedfs` 等 |
| `--hicache-storage-prefetch-policy` | `best_effort` / `wait_complete` / `timeout` |

HiCache 不是“把 KV 丢到磁盘就完了”。L3 命中要先 prefetch 回 L2/GPU，prefetch 的等待时间和命中率是一对矛盾；如果预取太慢，反而不如直接重算。SGLang 用 256 token 的阈值和三种预取策略来控这个平衡。

## 七、异构模型：SWA、Mamba、MLA 带来的多池结构

2025 年之后的模型很少是“每层都是 full attention”。SGLang 的 KV 管理也跟着从单池变成多池。

### SWA：full 层和 sliding-window 层分池

Gemma-2/3、Qwen3-Next 这类模型隔层用滑动窗口注意力：SWA 层只需要最近 `W` 个 token 的 KV，full 层才需要全上下文。如果按 full 层的容量给所有层分配，SWA 层会浪费大量显存。

SGLang 用 `SWAKVPool` 把两类层拆成两个池：`full_kv_pool` 存 full 层，`swa_kv_pool` 存 SWA 层，容量比例由 `--swa-full-tokens-ratio` 控制。分配器 `SWATokenToKVPoolAllocator` 负责在两类池之间做映射；SWA 池满了可以回收窗口外的 KV，而不影响 full 层。

### Mamba / 线性注意力：状态池

Mamba、GDN、KDA 这类层的“缓存”不是 per-token K/V，而是每个请求固定大小的 recurrent state（外加卷积窗口）。SGLang 用独立的 `MambaPool` 管状态，并在分支点缓存 checkpoint；`--mamba-full-memory-ratio` 控制状态池和 full KV 池的显存分配。

这类模型不能只按 token 数算容量：一条请求的状态是固定成本，并发数受状态槽限制。SGLang 的 `PrefillAdder` 会单独检查 `rem_mamba_slots`，状态槽不够时即使 KV 够也不会接纳。

### MLA / DSA：更小的 KV 和额外的 indexer

MLA 把每层 K/V 压成 latent，物理池是 `MLATokenToKVPool`；DSA（DeepSeek Sparse Attention）还多一份 indexer cache，decode 时先选 top-k 再读稀疏 KV。HiSparse 进一步把稀疏 KV 放到 host，按需换入。这些路径各有自己的池和分配器，但都复用同一套“树 + 锁 + 驱逐”的上层逻辑。

### UnifiedRadixCache：一棵树管三种 component

老实现里 Full、SWA、Mamba 各有各的树。现在 SGLang 在往 `UnifiedRadixCache` 收敛：一棵树，每个节点按 `ComponentType` 存 `FULL` / `SWA` / `MAMBA` 三份 component data，每份有自己的 `lock_ref`、LRU 和驱逐驱动。

内部节点的级联驱逐优先级是 `Full > SWA > Mamba`：驱逐 Full 会连带驱逐 SWA 和 Mamba，驱逐 SWA 会连带驱逐 Mamba，驱逐 Mamba 只影响自己。叶子节点则一次删掉所有 component。这样既保证跨 component 的一致性，又不需要在主树里写模型特判。

## 八、PD 分离：KV 变成可传输的对象

单机内的 RadixAttention 解决“同一副本内复用”。PD 分离（prefill-decode disaggregation）解决的是另一个问题：prefill 是计算密集，decode 是访存密集，混在一个引擎里会互相打断。

SGLang 的 PD 模式把两个阶段拆成独立进程：

- **prefill 进程**：算完整个 prompt 的 KV，通过 Mooncake / NIXL 等后端把 KV 传给 decode；
- **decode 进程**：收 KV，分配目标槽，然后连续 decode；
- **decode 侧可选 Radix 缓存**（`--disaggregation-decode-enable-radix-cache`）：缓存已经传过来的前缀，避免重复传输。

这不再是“缓存驱逐”问题，而是**部署层的 KV 策略**：用网络带宽换 prefill 和 decode 的独立扩缩容。代价是 KV 传输延迟和额外的显存占用；所以 PD 适合长 prompt、高并发、prefill 和 decode 比例不均衡的场景，不是所有负载都更快。

## 九、调优速查

| 旋钮 | 默认 | 调大 / 调小的效果 |
| :--- | :--- | :--- |
| `--mem-fraction-static` | 按显存自动推导 | 调大：KV 池更大、并发更高，但激活/CUDA graph 空间更小，容易 OOM |
| `--max-total-tokens` | 自动 | 手动钉死 KV 池 token 数，调试用 |
| `--page-size` | 1 | 调大：元数据少、kernel 顺，但前缀共享粒度变粗、尾部浪费多 |
| `--max-running-requests` | 自动 | 直接限制并发；调小能压尾延迟，调大更容易 retract |
| `--chunked-prefill-size` | 按 GPU 档位（2K–16K） | 调大：prefill 更快，但单步更重、更容易打断 decode |
| `--schedule-conservativeness` | 1.0 | 调大：准入更保守，retract 更少，并发更低 |
| `--radix-eviction-policy` | `lru` | 按负载换 `lfu` / `slru` / `priority` / `tlru` |
| `--disable-radix-cache` | 关闭 | 关掉前缀树，省掉 match/insert 开销，但丢掉复用 |
| `--kv-cache-dtype` | 模型 dtype | `fp8_e4m3` / `fp8_e5m2` / `nvfp4`，用精度换容量 |
| `--enable-session-radix-cache` | 关闭 | 会话引用优先于普通 LRU，适合多轮 agent |

有一条容易忽略的因果链：**前缀缓存保留的 KV 和并发请求用的是同一块显存**。命中率越高，被树“钉住”的叶子越多，能分给新请求的越少；驱逐策略、`mem-fraction-static` 和并发上限要一起调，不能只看命中率。

## 十、几个常见误解

**“RadixAttention 就是哈希前缀缓存。”** 不是。哈希只能整段精确匹配；RadixAttention 是树，能在任意 token 处分叉，命中停在节点中间还会 split。它支持按页对齐、命名空间隔离、LRU/LFU/priority 等多种驱逐，这些都不是一个哈希表能表达的。

**“`page_size=1` 说明 SGLang 没做分页。”** 分页的本质是“逻辑连续、物理离散、通过索引寻址”，SGLang 一直这么做；`page_size` 只是页的粒度。默认 1 是把共享粒度做到最细，代价是索引更长。

**“HiCache 的 L2 是跨实例共享的。”** 不是。L2 是进程私有的 host 池，同一台机器上的另一个实例也读不到；跨实例要走 L3，而且后端得配成共享 namespace。

**“驱逐策略能把热前缀钉在显存里。”** 不能。策略只打分；`lock_ref` 才能保护在跑请求，而且请求结束后就释放。真正的“保留”靠容量、命中率和策略一起调。

**“KV 池越大越好。”** 池越大并发越高，但激活、CUDA graph、通信 buffer 都要从同一张卡里出；`mem-fraction-static` 调过头会在启动或 capture 阶段 OOM。

## 十一、小结

SGLang 的 KV Cache 可以概括成四句话：

1. **不为请求预留连续显存**：按 token/页动态发号，prefill 只分配未命中的后缀，decode 每步只分配新 token。
2. **用基数树做前缀复用**：`match` / `insert` / `split` / `lock_ref`，树只存索引，数据在池里。
3. **三层回收**：驱逐缓存叶子、retract 在跑请求、准入时排队；默认 LRU，另有 LFU/SLRU/priority/TLRU。
4. **容量不够就分层**：HiCache 的 L1/L2/L3，加上 SWA/Mamba/MLA 的多池结构，把“单机显存”变成“显存 + 主机 + 存储”的容量问题。

想顺着代码读，建议顺序：

`mem_cache/README.md`（分层总览）→ `memory_pool.py`（`ReqToTokenPool` / `KVCache`）→ `allocator/token.py` 与 `allocator/paged.py`（发号）→ `radix_cache.py`（match/insert/split/evict）→ `evict_policy.py`（策略）→ `unified_radix_cache.py` 与 `unified_cache/components/README.md`（多 component）→ `docs/docs/advanced_features/hicache_design.mdx`（HiCache）。

下一篇讲这些槽位怎么被分给一批不断变动的请求：[SGLang 的 Continuous Batching](/posts/llm-engine/sglang/continuous-batching/)。
