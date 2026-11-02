---
title: "vLLM 的 KV Cache：分页、引用计数与前缀缓存"
published: 2026-11-03T00:00:00+08:00
description: 从一行显存账开始拆 vLLM v1 的 KV 管理：block_table 与 slot_mapping 怎么寻址、BlockPool 怎么用引用计数加 LRU 管块、链式哈希怎么让前缀缓存命中，以及 block_size、gpu_memory_utilization 这些旋钮各自在买什么。
tags: [vLLM, KV Cache, PagedAttention, 显存管理]
category: LLM Engine
draft: false
---

KV Cache 是推理引擎里最像操作系统的部分：本质上它就是在显存里实现一套虚拟内存——逻辑连续、物理分页、按引用计数共享、按 LRU 换出。vLLM 把这件事做得最彻底，也是它吞吐优势的主要来源。

这篇只讲 KV 怎么管，不讲调度。调度在下篇 [Continuous Batching](/posts/llm-engine/vllm/continuous-batching/)。

> **五分钟版**
>
> - **为什么要有它**：因果 mask 下前缀 K/V 永不改变，缓存住就把每条序列的 O(n²) 重复计算压成 O(n)。省的是算力，代价是显存。
> - **显存账**：`2 × 层数 × KV heads × head_dim × dtype` 每 token。Llama-3-8B fp16 是 128 KiB/token，8K 上下文一条序列就是 1 GiB，所以 KV 才是并发上限。
> - **分页**：逻辑块连续、物理块离散，`block_table` 当页表，`slot_mapping` 当写指针；默认 `block_size=16`，尾部最多浪费 15 个槽位。
> - **三个结构**：`KVCacheBlock`（引用计数 + 哈希 + 链表指针）、`FreeKVCacheBlockQueue`（O(1) 双向链表，LRU 顺序）、`BlockHashToBlockMap`（哈希→块）。
> - **前缀缓存**：块哈希是链式的，`h_i = hash(tokens_i, h_{i-1})`，所以“物理块相同”严格等价于“整段前缀相同”，不会误命中。
> - **代价**：分页让每次 attention 多一次 gather；块越小碎片越少但 gather 越多；前缀缓存要哈希、要驱逐，块被占住时还会加剧抢占。
>
> 代码口径：vLLM `main` @ `fa2881ed19`（2026-11），`block_size`、`gpu_memory_utilization` 等默认值都以该版本为准。

## 一、先把账算清楚

自回归解码到第 `t` 步时，模型真正“新”的只有第 `t` 个 token 的 Q。K/V 来自 `0..t` 全部 token，而在因果 mask 下，`0..t-1` 的 K/V 与上一步算出来的**完全一致**：

```
不缓存：第 t 步重算 0..t 的 K/V        → 总计 O(n²) 次投影
缓存：  第 t 步只算 t 的 K/V，读历史   → 总计 O(n) 次投影
```

所以 KV Cache 的第一性原理是**用显存换算力**。这笔交易在这个时代极其划算，因为 decode 是访存受限的：一步只算一个 token，GPU 的算力大量闲置，瓶颈在把权重和 KV 从 HBM 搬进 SM。

代价则按 token 线性增长：

$$
\text{bytes/token} = \underbrace{2}_{K,V} \times L_{\text{layers}} \times H_{\text{kv}} \times d_{\text{head}} \times \text{sizeof(dtype)}
$$

以 Llama-3-8B 为例：32 层、8 个 KV head（GQA，8× 压缩）、head_dim 128、fp16——每个 token `2 × 32 × 8 × 128 × 2 = 131072` 字节，正好 **128 KiB**。8K 上下文的一条序列就是 **1 GiB**。

| 项目 | 大小 | 说明 |
| :--- | :--- | :--- |
| 模型权重（8B, fp16） | 16 GiB | 固定开销，与并发无关 |
| 单条 8K 序列的 KV | 1 GiB | 随上下文线性增长 |
| 单条 32K 序列的 KV | 4 GiB | 长上下文要按块算 |
| H100 80G 可用于 KV | ≈ 55 GiB | 80×0.92 扣掉 16 GiB 权重和激活后的粗估 |

结论很直接：**权重决定能不能装下模型，KV 决定能同时跑多少请求**。而“怎么管 KV”直接等于“能并发多少”，也就等于吞吐。这也是 PagedAttention 那篇论文的出发点。

## 二、朴素分配浪费在哪

论文之前的主流做法是：请求进来时按 `max_model_len` 预分配一块**连续**显存。三种浪费同时出现：

- **预留浪费**：请求只要 200 token，却占着 8192 的位置；
- **内部碎片**：序列实际长度很少正好落在块边界，尾部空着但不能给别人用；
- **外部碎片 + 不可共享**：不同长度的连续块互相卡位，相同 system prompt 也只能各存一份。

论文实测这套做法的**有效利用率只有 20%–40%**。也就是说 80 GB 的卡实际只有 20–30 GB 装着有用的 KV，剩下全是空洞。

## 三、分页：逻辑块连续，物理块离散

PagedAttention 的思路借自操作系统：序列在**逻辑上**连续，在**物理上**按固定大小切块、可以散落各处。

```
逻辑块 (A) :  [0]   [1]   [2]   [3]
               |     |     |     |
               v     v     v     v
物理块     :   7     2     41    3

block_table(A) = [7, 2, 41, 3]   <- 就是页表
```

- **`block_table`**：每个请求一张，逻辑块号 → 物理块号。attention kernel 拿着它去 gather K/V 页，数学上等价于一块连续张量。
- **`slot_mapping`**：写路径用。新 token 要写进哪个物理槽位，由 `slot = block_table[logical_block] * block_size + offset` 算出，一次 scatter 落地。

默认 `block_size = 16`（`vllm/config/cache.py` 里的 `CacheConfig.DEFAULT_BLOCK_SIZE`）。对 Llama-3-8B 来说一块就是 `16 × 128 KiB = 2 MiB`。

块大小的取舍很直接：

| block_size | 好处 | 代价 |
| :--- | :--- | :--- |
| 小（8 / 16） | 碎片小，共享粒度细，长序列尾部浪费少 | `block_table` 更长，attention 多几次 gather，小块降低 kernel 效率 |
| 大（32 / 64） | gather 少，kernel 更顺 | 每条序列尾部浪费变多，前缀命中粒度变粗 |

16 是这个折中的默认值，不是随便定的。

### 读路径与写路径

一次前向里两件事同时发生：

- **读**：attention 需要 `seq_lens`、`block_table`，kernel 按页取 K/V。vLLM v1 用 varlen 接口把整个 batch 打包，所以不同长度请求共用一次 kernel launch。
- **写**：本轮新算出的 K/V 按 `slot_mapping` 散射进各自物理槽位。`gpu_model_runner` 里就是 `input_batch.block_table.compute_slot_mapping(...)` 算出来的（`vllm/v1/worker/block_table.py`）。

换句话说，分页对上层是透明的：模型代码看到的仍是连续的逻辑序列，只有内存布局变了。

## 四、v1 里的四个数据结构

实现在 `vllm/v1/core/` 的两个文件里：块与空闲链表在 `kv_cache_utils.py`，池与哈希表在 `block_pool.py`。

| 结构 | 位置 | 职责 |
| :--- | :--- | :--- |
| `KVCacheBlock` | `kv_cache_utils.py` | 单个块的元数据：`block_id`、`ref_cnt`、块哈希、空闲链表前后指针 |
| `FreeKVCacheBlockQueue` | `kv_cache_utils.py` | 空闲块的 O(1) 双向链表，按淘汰优先级排序（LRU） |
| `BlockHashToBlockMap` | `block_pool.py` | 块哈希 → 块，前缀缓存查找用 |
| `BlockPool` | `block_pool.py` | 把上面三个拼起来：分配、释放、缓存、驱逐 |

`KVCacheBlock` 的字段值得单独看一眼，它同时承载了三种语义：

```python
@dataclass(slots=True)
class KVCacheBlock:
    block_id: int                 # 0 .. num_gpu_blocks-1
    ref_cnt: int = 0              # 被多少请求/多少块表引用
    _block_hash: BlockHashWithGroupId | None = None   # 满块且已缓存时才有
    prev_free_block: KVCacheBlock | None = None       # 空闲链表
    next_free_block: KVCacheBlock | None = None
    is_null: bool = False         # 占位块，永不缓存
```

一个关键设计是：**已缓存但没人引用的块（`ref_cnt == 0`）仍然留在空闲队列里**。它不是“空闲”，而是“可回收”：下次分配时从队头拿，拿走时顺手把它的哈希从缓存表里摘掉，这就完成了淘汰。这样做的好处是分配和淘汰是同一次操作，不需要单独的 GC 线程或后台扫描。

`is_null` 的块是给滑动窗口之类“跳过前缀”的表项占位用的：`block_id = 0`，不维护引用计数，也永远不进缓存。

## 五、一次分配、释放、命中的完整动作

### 分配：从队头拿，顺手淘汰

```python
def get_new_blocks(self, num_blocks):
    ret = self.free_block_queue.popleft_n(num_blocks)   # 从队头取
    for block in ret:
        self._maybe_evict_cached_block(block)           # 旧哈希失效 = 被淘汰
        block.ref_cnt += 1
    return ret
```

队头的块被优先复用——所以拿队头就等于 LRU 淘汰。如果块上还挂着哈希，说明它被前缀缓存索引着，此时必须把哈希清掉，否则别的请求会命中一个内容即将被覆盖的块。

### 释放：分两档插回队列

```python
def free_blocks(self, ordered_blocks):
    for block in ordered_blocks:
        block.ref_cnt -= 1
        if block.ref_cnt == 0:
            if block.block_hash is None:
                blocks_to_evict_first.append(block)   # 无缓存价值 → 前插，LIFO 复用
            else:
                blocks_to_evict_last.append(block)    # 有缓存价值 → 后插，LRU 顺序
```

这是整个管理器里最精巧的一处：

- **没有哈希的块**（从没写满过、或缓存关闭）前插，下次分配立刻复用——LIFO 让同一块显存在短时间内被反复使用，GPU 局部性更好；
- **有哈希的块**后插，留在队列尾部等待被命中，直到真的需要新块时才淘汰——这就是 LRU。

一个 `free_blocks` 就同时实现了“快速复用”和“尽量保留缓存”，不需要两个池子。

### 命中：引用计数 +1，把块从空闲队列摘出来

```python
def touch(self, blocks):
    for block in blocks:
        if block.ref_cnt == 0:          # 还在空闲队列里
            self.free_block_queue.remove(block)
        block.ref_cnt += 1
```

前缀命中走的正是 `touch()`：块从“可回收”变回“在用”，同时 `ref_cnt` 加一。这时两个请求的 `block_table` 指向同一批物理块——共享不需要拷贝任何数据，只需要计数。

**写时复制**也是同一个机制：谁要改共享块，先检查 `ref_cnt == 1`；不满足就复制一块再写。`is_block_writable()` 就是这个判断（`ref_cnt == 1 and block_hash is None`）。

## 六、前缀缓存：链式哈希才是正确性来源

块级缓存最容易出的错是“误命中”：两个请求的第 3 块 token 相同，但前面两块不同，能共享吗？显然不能。解法是把哈希链起来：

```
h_0 = hash(tokens[0:16],        NONE_HASH)
h_1 = hash(tokens[16:32],       h_0)
h_2 = hash(tokens[32:48],       h_1)
```

`hash_block_tokens(parent_block_hash, curr_block_token_ids)` 做的就是这个。于是“物理块相同”严格等价于“从第 0 个 token 到该块结束的整段前缀相同”，误命中在数学上被排除了。

顺着这条链查找就是 `get_computed_blocks()`：

```python
computed_blocks, num_new_computed_tokens, num_uncached = (
    self.coordinator.find_longest_cache_hit(request.block_hashes, max_cache_hit_length)
)
```

三个细节值得记住：

1. **只认满块**。缓存的是完整 16 token 的块，因为不满的块内容会变，哈希不稳定。
2. **全部命中也要重算最后一个 token**。命中 N 个块只能说明前 N×16 个 token 的 K/V 有了，但 logits 必须由最后一个 token 的前向产生。所以 `max_cache_hit_length = request.num_tokens - 1`，代码注释里写得很直白。
3. **本版本还支持尾块缓存**（`BlockPool.cache_partial_block`），把不满一块的尾部也纳入哈希；代价是命中边界不再总是块对齐，需要额外处理 `num_uncached` 的分界。

前缀缓存默认开启（`CacheConfig.enable_prefix_caching = True`）。收益取决于负载：多轮对话、固定 system prompt、few-shot 模板、agent 的长工具描述，命中率能非常高；每个请求都不同的短 prompt 则基本没有收益，只剩哈希开销。

## 七、`allocate_slots()`：一次分配要算几笔账

调度器每步对每个请求调用一次 `KVCacheManager.allocate_slots(request, num_new_tokens, ...)`。它的 docstring 里有一张布局图，是理解 v1 内存管理的最好入口（下面只留主干）：

```
| < comp > | < new_comp > | < ext_comp > | < new > | < lookahead > |
```

- `comp`：已经算过的 token（`request.num_computed_tokens`）；
- `new_comp`：本次前缀命中、直接认领的 token；
- `ext_comp`：KV connector 从外部（PD 解耦的 prefill 实例、KV 池）搬来的 token；
- `new`：本轮真正要算的新 token；
- `lookahead`：投机解码的草稿槽位。

三个阶段：先释放窗口外/不需要的块并检查余量，再处理前缀 token，最后为 `new` 分配新块。**块不够就返回 `None`**——注意它不抛异常、也不自己抢占，只是把决定权交回调度器。这个边界切得很干净：KV 管理器只回答“能不能”，不能就由调度层决定牺牲谁。

返回的 `KVCacheBlocks` 会一路传到 worker，变成 block table 的一行和 `slot_mapping`。

## 八、旋钮与代价

| 旋钮 | 默认 | 调大/调小的后果 |
| :--- | :--- | :--- |
| `block_size` | 16 | 调小：碎片少、共享细，但表更长、kernel 更碎；调大反之 |
| `gpu_memory_utilization` | 0.92 | 调大能多装 KV、提高并发，但留给激活和 CUDA graph 的空间变少，容易 OOM |
| `enable_prefix_caching` | `True` | 关掉省哈希开销与驱逐逻辑，但丢掉重复前缀的收益 |
| `max_model_len` | 模型上限 | 直接决定单序列最多占多少块，间接决定并发 |
| KV offload / connector | 关闭 | 把冷 KV 换到 CPU/远端，v1 用 connector 替代了 v0 的 swap |

有一条容易忽略的因果链：**前缀缓存保留的块会挤占可用块**。高并发下，缓存命中率越高，被“钉住”的块越多，真正能分给新请求的越少，抢占就越频繁。这不是 bug，是缓存容量和并发容量在同一块显存上的零和博弈。

## 九、它和调度的接口

KV 管理器只暴露三个动作给调度器：查（`get_computed_blocks`）、要（`allocate_slots`）、还（`free`）。所有“策略”都在调度器那边：先服务谁、块不够时牺牲谁、一个请求一步算多少 token。

这也解释了为什么两个模块能各自演进：KV 侧换成分层存储、加 KV connector，调度侧加投机解码、加 PP，接口都还是那三个。

下一篇讲另一半：[Continuous Batching：vLLM v1 调度器怎么排一个 batch](/posts/llm-engine/vllm/continuous-batching/)。想看整条请求链路的时序，可以配合 [vLLM 聊天补全请求全链路](/posts/llm-engine/vllm/) 一起读。
