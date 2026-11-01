---
title: "Mooncake：KVCache 为中心的推理基础设施"
published: 2026-11-01T22:10:00+08:00
description: "Mooncake 是月之暗面为 Kimi 打造的 KVCache 中心推理架构，也开源成可独立使用的库：Transfer Engine 做跨节点零拷贝搬运，Store 把 DRAM/SSD 汇成分布式 KV 池，EP/PG 支撑弹性 MoE，vLLM/SGLang 通过 connector 接入。本文按组件、API、接入方式和部署坑梳理。"
tags: [Mooncake, KV Cache, PD 解耦, vLLM, SGLang, 推理基础设施]
category: LLM Engine
draft: false
---

Mooncake 是月之暗面（Moonshot AI）为 Kimi 做的推理基础设施，也是这套架构中可复用部分的开源实现。它经常被一句话概括成“KV Cache 库”，但真正要理解它，得先把它和推理引擎的分工拆开：vLLM、SGLang 负责模型怎么算、请求怎么调度；Mooncake 负责算出来的 KV 放在哪、跨节点怎么搬、多个实例之间怎么共享。

这个定位决定了它的形态。Mooncake 不是一个能 `serve` 模型的引擎，而是引擎下面的一层：一个进程内的传输库（Transfer Engine）、一个分布式对象池（Mooncake Store）、一组给 MoE 用的集合通信组件（PG/EP），以及一个给网关用的全局前缀索引（Conductor）。你可以只用其中一层，也可以按 PD 解耦、共享 KV 池、弹性 EP 等场景组合。

> **五分钟版**
>
> - **定位**：Kimi 的 KVCache 中心架构，Apache-2.0 开源；不替代 vLLM/SGLang，而是补上跨节点 KV 传输与共享存储。
> - **三块核心**：Transfer Engine 搬数据，Store 存 KV 和权重，PG/EP 管 MoE 通信；Conductor 负责缓存感知选路，Reshard 管权重布局迁移。
> - **两条接入路径**：PD 解耦用 `MooncakeConnector`（vLLM）/ `--disaggregation-mode`（SGLang）直接搬 KV；共享 KV 池用 `MooncakeStoreConnector` / SGLang HiCache 的 `mooncake` 后端。
> - **收益有前提**：RDMA、多网卡、足够的 KV 复用率，三者缺一，收益会明显缩水；纯 TCP 也能跑，但别拿它对比 RDMA 的数字。
> - **部署门槛**：GPUDirect RDMA、Store master 的高可用、版本配对，是落地时最容易出问题的三处。
> - **版本口径**：截至 2026-11-01，PyPI 稳定版是 `0.3.13.post1`（2026-08-31），`0.3.14` 还在预发布；API 动得快，生产环境锁版本。

## 它解决什么问题

LLM 推理的两个阶段，资源画像完全不同：

| 阶段 | 主要瓶颈 | 关键指标 | 优化方向 |
| :--- | :--- | :--- | :--- |
| Prefill | 算力（长 prompt 下尤其明显） | TTFT | 提高 GPU 利用率、复用前缀 KV、chunked prefill |
| Decode | 显存带宽与 batch 大小 | TBT / ITL | 增大 batch、减少 KV 换入换出、提高 MFU |

把 prefill 和 decode 放在同一组实例上，两边会互相抢资源：长 prompt 的 prefill 抬高 decode 的尾延迟，decode 的低利用率又浪费 prefill 的算力。PD 解耦（prefill-decode disaggregation）把两段拆到不同实例，各自扩缩容，代价是 KV 要跨节点搬。

Mooncake 的判断是：**KV Cache 应该像存储一样被池化，而不是绑在单机显存里**。GPU 集群里大量 CPU、DRAM、SSD 和网卡在推理时是闲置的，把这些资源组织成一个分布式的 KV 池，热数据靠近 GPU，冷数据下沉到 DRAM/SSD，请求来了先查全局前缀，能复用就搬过来，不能复用再算。论文标题里的“Trading More Storage for Less Computation”就是这个意思。

这条路线有三个直接后果：

- **前缀缓存从单机变成集群级**。单机 HBM/DRAM 能装下的 KV 有限，跨实例共享后，重复的系统提示、多轮对话、agent 的固定上下文都能命中。
- **调度要懂缓存**。请求该发给哪台 prefill、哪台 decode，不只看出队长度，还要看那台机器上有没有可复用的前缀。这是 Conductor 和论文里 KVCache-centric scheduler 要解决的问题。
- **传输必须够快**。KV 搬得慢，PD 解耦就白拆了。Mooncake 把大量工程投在 Transfer Engine 上：RDMA 零拷贝、多网卡聚合、拓扑感知选路、故障切换。

论文里的 Mooncake 是 Kimi 的生产架构；开源仓库是其中可复用的部分，今天的组件已经比论文多了 EP/PG、Reshard、TENT，以及 Store 的 HA、多租户和对象存储分层。两者不要混为一谈：**论文讲的是“为什么这样设计”，仓库讲的是“现在能怎么用”**。

## 组件地图

| 组件 | 形态 | 负责什么 | 典型场景 |
| :--- | :--- | :--- | :--- |
| Transfer Engine | 进程内 C++/Python 库 | 跨节点、跨设备的零拷贝批量搬运 | PD KV 传输、权重同步、checkpoint 分发 |
| Mooncake Store | Master 进程 + Client 库 | 分布式对象池，管理 KV/权重的放置、副本、淘汰 | vLLM 共享 KV 池、SGLang HiCache L3 |
| PG / EP | 进程组 + MoE 通信运行时 | 容错集合通信、DeepEP 风格 dispatch/combine | 弹性 EP、大规模 MoE 推理 |
| Conductor | 独立服务 | 消费 KV 事件，维护全局前缀索引，回答“哪台实例命中最多” | 网关/路由器选路 |
| Reshard | 规划库 | 在 TP/PP/EP/DP 布局之间搬运权重 | 模型权重缓存、扩缩容 |
| P2P Store | Go 服务 | 基于 Transfer Engine 的 P2P 分发 | checkpoint-engine 的权重更新 |

设计上反复出现的一条原则是**控制面与数据面分离**：

- Transfer Engine 的元数据服务只交换 segment、buffer、endpoint 的位置信息，不碰 payload；真正的数据在两端注册过的 buffer 之间直接走 RDMA/TCP。
- Store 的 Master 只做对象目录、空间分配、副本和租约，返回的是副本描述符；Client 拿到描述符后自己通过 Transfer Engine 搬，Master 不在数据路径上。
- Conductor 只维护索引和事件，不转发请求体；它回答的是“这个前缀在哪些实例、哪些层级上有”，由网关决定怎么用。

所以这些组件可以拆开部署，也可以只选一层。只要 PD 搬 KV，装 Transfer Engine 就够；要跨实例共享，才需要 Store；要容错 MoE，才需要 PG/EP。

## Transfer Engine：跨节点搬 KV

Transfer Engine（TE）是 Mooncake 的数据面，也是最早开源的部分。它把“一次传输”抽象成两个概念：

- **Segment**：一段可被远端读写的地址空间。RAM Segment 对应进程的 DRAM/VRAM，每个进程只有一个，名字默认取 `local_hostname`；NVMe-oF Segment 对应持久化存储。Segment 内部再划分成多个 **Buffer**，每个 Buffer 是一段连续、同设备的地址空间，注册后才能被 RDMA 访问。
- **BatchTransfer**：一组异步的读/写请求，每个请求给出本地地址、远端地址和长度，可以跨多个 segment。它像更灵活的 scatter/gather，提交后通过状态接口轮询（C++ 是 `getTransferStatus`，Python 是 `transfer_check_status`）。

传输后端按环境和编译选项选择：TCP、RDMA、AWS EFA、NVLink、HIP（AMD）、SHM（同机 DRAM）、NVMe-oF、CXL，以及昇腾等国产加速器。对上层来说接口基本一致，差异被 Transport 层吃掉。

### 路径选择、切片与容错

TE 的性能不只来自 RDMA，还来自“把多张网卡用起来”：

- **拓扑感知**：启动时生成拓扑矩阵，记录每类内存（CPU NUMA、GPU）偏好的网卡；传输时优先走同 NUMA / 同 PCIe switch 的路径，避免 UPI 或 PCIe 成为瓶颈。
- **多网卡聚合**：单个请求超过 64KB 会被切成多个 slice，不同 slice 可以走不同网卡，从而把多张 200G/400G 的带宽叠加起来。
- **Endpoint 池**：连接按需建立，用 SIEVE 算法淘汰不活跃的 endpoint，避免海量连接拖慢请求处理。
- **故障切换**：某张网卡或某条链路不可用时，自动换一条可达路径重试；失败的 endpoint 会从池里移除，下次传输再重建。

这些机制解释了一个常见疑问：为什么 Mooncake 的 benchmark 能跑到单卡带宽以上。它测的不是一张网卡，而是一组网卡加拓扑优化后的聚合带宽。

### 元数据服务

TE 需要知道对端的 segment 和 buffer 地址。元数据可以放在：

- **etcd**：适合已有 etcd 的集群，支持多节点高可用；
- **Redis**：部署轻，适合中小规模；
- **HTTP**：Mooncake 自带 HTTP metadata server，也可以由 `mooncake_master` 内嵌；
- **P2PHANDSHAKE**：不依赖独立元数据服务，连接建立时点对点交换元数据，适合快速起步和单机调试。

元数据服务只交换控制信息，不承载 KV 数据；但它仍然是关键依赖，跨主机部署时要注意监听地址、防火墙和代理变量。HTTP metadata 跨主机部署时要配 token（`MC_METADATA_HTTP_TOKEN`），别把它当成无鉴权的内部服务直接暴露。

### Python 最小示例

下面这段只演示 TE 的核心动作：初始化、注册本地 buffer、向对端已注册的 buffer 写数据。远端地址不在 TE 里查，需要由上层控制面（RPC、注册表、引擎的 connector）传给你。

```python
import numpy as np
from mooncake.engine import TransferEngine

engine = TransferEngine()
rc = engine.initialize(
    "10.0.0.1:12345",   # 本机地址，集群内唯一
    "P2PHANDSHAKE",     # 也可以 etcd://、redis://、http://
    "rdma",             # 没有 RDMA 时用 "tcp"
    "mlx5_0",           # 逗号分隔；留空则自动发现
)
if rc != 0:
    raise RuntimeError(f"initialize failed: {rc}")

buf = np.zeros(4 * 1024 * 1024, dtype=np.uint8)
local_addr = buf.ctypes.data
engine.register_memory(local_addr, buf.nbytes)

# remote_addr 由对端通过控制面告知，必须是它已注册 buffer 的起始地址
rc = engine.transfer_sync_write("10.0.0.2:12345", local_addr, remote_addr, buf.nbytes)
if rc != 0:
    raise RuntimeError(f"transfer failed: {rc}")

engine.unregister_memory(local_addr)
```

几个容易踩的点：

- `local_hostname` 必须集群内唯一，它同时也是 segment 名；两台机器取同一个值会在元数据里冲突。
- `register_memory` 注册的是地址范围，不是对象；buffer 释放前要先 `unregister_memory`，否则 RDMA 侧可能访问已释放内存。
- 同步接口适合小批量、控制路径；KV 这种大块数据用 `batch_transfer_sync_write` 或 `transfer_submit_write` + `transfer_check_status` 异步提交，才能把多张网卡和多个请求重叠起来。
- Store 的 `setup()` 支持复用已初始化的 `TransferEngine`（位置参数重载里的 `engine` 参数），可以避免重复注册和重复建连。

TE 也能直接用在 PD 之外的场景：多模态流水线的 embedding 传递、训练和推理之间的 hidden state/权重同步、checkpoint 分发。判断标准很简单：**有没有一大块数据需要跨节点、低延迟、零拷贝地搬**。有，就值得用；只是同机小对象复制，`memcpy` 更省事。

## Mooncake Store：把 KV 变成集群对象池

Store 建在 TE 之上，对外是一个分布式 key-value 对象池。它和 Redis 这类通用缓存的关键区别是：**value 写入后不可变，key 在 LLM 场景里通常由前缀/块的内容哈希得到**。这正好匹配 KV Cache 的语义——同一个前缀算出来的 KV 是确定的，缓存条目只需要读、复用和淘汰，不需要原地修改。

### Master 与 Client

Store 只有两个角色：

- **Master Service**（`mooncake_master`）：控制面。维护对象目录、segment 成员、空间分配、副本状态、租约、租户配额和淘汰策略。它返回副本描述符，但不转发对象数据。
- **Client**：既可以是发起 `Put/Get/Remove` 的客户端，也可以是贡献内存的存储节点。`global_segment_size` 是它贡献给集群的内存大小；`local_buffer_size` 是 TE 用的本地中转 buffer。两者可以独立配置：
  - `global_segment_size=0`：纯客户端，只发请求，不贡献内存；
  - `local_buffer_size=0`：纯存储节点，只提供空间，不发起请求；
  - 两者都大于 0：既当客户端又当存储节点，这是嵌入式部署的默认形态。

Client 有三种跑法：

| 形态 | 怎么配 | 适合 |
| :--- | :--- | :--- |
| 嵌入式 | 推理进程里 `setup()`，`global_segment_size > 0` | 引擎顺带贡献空闲内存，部署最简单 |
| Dummy + real | 每个 TP rank 一个 dummy，每个实例一个 real client | TP/DP 多进程，避免每个 rank 都占一大段内存 |
| 独立 store service | `python -m mooncake.mooncake_store_service` | 存储与计算分离，引擎侧 `global_segment_size=0` |

独立服务配合 dummy/real 的模式，把存储和计算拆得更彻底：引擎进程只保留网络和本地 buffer，内存池由专门的 store 节点负责，扩缩容和重启互不影响。

### 对象语义与容量管理

Store 的几个行为值得先弄清楚：

- **Put 不可变**：同一个 key 再次 `put` 会返回 `OBJECT_ALREADY_EXISTS (-705)`，要更新用 `upsert`。这避免了“读到一半被覆盖”的问题，但也意味着 key 的设计要带上版本或内容哈希。
- **Get 强一致**：`put` 成功返回后，后续 `get` 一定能读到完整数据；key 不存在时返回空 `bytes`，需要区分“空值”和“不存在”就用 `is_exist`。
- **副本与分片**：`ReplicateConfig.replica_num` 控制副本数，大对象可以按 slice 跨 segment 条带化，读的时候并行拉取，聚合多网卡带宽。
- **Pin 与淘汰**：软 pin / 硬 pin 可以保护热 KV 或模型权重不被淘汰；淘汰策略按水位线和访问历史工作，Master 负责执行。
- **分层**：内存之外可以挂 SSD/NVMe，甚至对象存储（OSS/S3）和 DFS 后端。冷 KV 下沉，热 KV 留内存，容量和成本之间做权衡。
- **多租户**：默认关闭；开启后按 tenant 做配额准入和隔离，适合一个集群服务多个业务线。

### 高可用

默认部署是单 Master，Master 挂了集群就不能继续服务。生产环境要开 HA：多个 Master 通过 etcd 选主，leader 负责处理请求，follower 待命；leader 失联后重新选举，新 leader 完成状态追赶再对外服务。客户端要用 `etcd://10.0.0.1:2379;...` 这种地址形式，才能发现当前 leader；只写单个 `IP:Port` 是连不到 HA 集群的。

HA 只解决 Master 的高可用，不解决数据持久性。Store 里的 KV 本质上是缓存，丢了可以重算；如果要保模型权重这类数据，得配合 SSD/对象存储和快照机制，别把缓存当数据库用。

### Python 最小示例

```python
from mooncake.store import MooncakeDistributedStore

store = MooncakeDistributedStore()
rc = store.setup(
    local_hostname="10.0.0.1",
    metadata_server="P2PHANDSHAKE",
    global_segment_size=512 * 1024 * 1024,  # 贡献给集群的内存
    local_buffer_size=128 * 1024 * 1024,    # TE 本地 buffer
    protocol="rdma",
    rdma_devices="",                        # 留空自动发现；注意关键字不是 device_name
    master_server_addr="10.0.0.1:50051",
)
if rc != 0:
    raise RuntimeError(f"store setup failed: {rc}")

kv_bytes = b"..."  # 实际使用中是序列化好的 KV block
store.put("kv:model-a:prefix-hash", kv_bytes)

# 同一个 key 不能重复 put，更新用 upsert
store.upsert("kv:model-a:prefix-hash", kv_bytes)

data = store.get("kv:model-a:prefix-hash")
print(store.is_exist("kv:model-a:prefix-hash"), len(data))
store.close()
```

大块 KV 建议走零拷贝接口：先 `register_buffer` 注册源/目的 buffer，再用 `put_from(key, ptr, size)` 和 `get_into(key, ptr, size)`，省掉 Python `bytes` 的拷贝和临时内存。对象越大，这个差别越明显。

配置字段有两套拼写，别混：

- Python `setup()`：`rdma_devices`、`master_server_addr`；
- JSON 配置 / `MooncakeConfig`：`device_name`、`master_server_address`。

写 connector 配置时按 JSON 那套来，写代码时按 `setup()` 那套来。

## 接到推理引擎上

Mooncake 不提供 OpenAI 兼容 API，接入要靠引擎自己的 connector。文档最完整、接入最深的是 vLLM 和 SGLang 两条线。

### vLLM：PD 解耦

vLLM V1 里用 `MooncakeConnector` 做跨节点 KV 传输。prefill 实例是 producer，decode 实例是 consumer：

```bash
# Prefill 节点（10.0.0.1）
vllm serve Qwen/Qwen3-8B \
  --port 8010 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}'

# Decode 节点（10.0.0.2）
vllm serve Qwen/Qwen3-8B \
  --port 8020 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}'

# 代理：把请求发给 prefill，再把 transfer_id 和 bootstrap 地址转给 decode
python examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py \
  --port 8000 \
  --prefill http://10.0.0.1:8010 8998 \
  --decode http://10.0.0.2:8020
```

关键参数：

- `VLLM_MOONCAKE_BOOTSTRAP_PORT`：prefill 侧的 bootstrap 端口，默认 8998。TP/DP 下每个 worker 的端口按 `base_port + dp_rank * tp_size + tp_rank` 计算，同一台机器上不能撞端口。
- `--kv-transfer-config` 里的 `num_workers`：prefill worker 里发送 KV 的线程池大小，默认 10。带宽上不去时可以调，但别超过实际并发需要。
- `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT`：请求中止后多久释放 KV，默认 480 秒。长请求多、取消频繁时要注意这个值。

**一个容易静默出错的点**：`MooncakeConnector` 要求路由器生成 `transfer_id`，并把 prefill 的 bootstrap 地址转发给 decode。vLLM 测试目录里的 toy proxy（`tests/v1/kv_connector/nixl_integration/`）不实现这个契约；照抄它时输出仍然正确，但 decode 会退化成重新计算 KV，传输根本没发生——从结果看一切正常，只有看日志或带宽才发现。自己写 router 的话，这部分要按 vLLM 的 connector 契约实现。

### vLLM：共享 KV 池

如果目标不是 PD 解耦，而是让多个 vLLM 实例共享前缀缓存、把 KV 下沉到 CPU/SSD，用 `MooncakeStoreConnector`。先写一份 Mooncake 配置：

```json
{
  "local_hostname": "10.0.0.1",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "10.0.0.1:50051",
  "global_segment_size": "32gb",
  "local_buffer_size": "2gb",
  "protocol": "rdma",
  "device_name": ""
}
```

```bash
export MOONCAKE_CONFIG_PATH=/etc/mooncake/mooncake_config.json

# 单实例：把 KV 卸载到共享池
vllm serve Qwen/Qwen3-8B \
  --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both"}'
```

`kv_both` 是单实例的读写模式；做 XpYd 解耦时，prefill/decode 分别用 `kv_producer`/`kv_consumer`，再用 `MultiConnector` 把 `MooncakeConnector`（直接搬）和 `MooncakeStoreConnector`（落池、复用）组合起来。下面只列 producer，decode 侧把 `kv_role` 和每个 connector 的 role 对称换成 `kv_consumer` 即可：

```bash
vllm serve Qwen/Qwen3-8B \
  --kv-transfer-config '{
    "kv_connector": "MultiConnector",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {
      "connectors": [
        {"kv_connector": "MooncakeConnector", "kv_role": "kv_producer"},
        {"kv_connector": "MooncakeStoreConnector", "kv_role": "kv_producer"}
      ]
    }
  }'
```

两个细节：

- `global_segment_size` 是这台实例贡献给池子的内存。如果你用独立的 `mooncake_store_service` 提供存储，这里设 `"0"`，让 vLLM 只贡献网络和本地 buffer。
- 开 DP 时设 `PYTHONHASHSEED=0`。不同 DP rank 的 Python 哈希种子不一致，同样的 prompt 可能算出不同的 block hash，跨实例前缀就命不中了。

### SGLang：PD 与 HiCache

SGLang 的 PD 解耦不经过 connector 配置，直接给实例指定角色：

```bash
# Prefill
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B \
  --disaggregation-mode prefill \
  --disaggregation-ib-device mlx5_0 \
  --host 10.0.0.1 --port 30000

# Decode
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B \
  --disaggregation-mode decode \
  --disaggregation-ib-device mlx5_0 \
  --host 10.0.0.2 --port 30001

# Router
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill "http://10.0.0.1:30000" 8998 \
  --decode "http://10.0.0.2:30001" \
  --host 0.0.0.0 --port 8000
```

`--disaggregation-ib-device` 可以省略让 SGLang 自动探测；多网卡时用逗号分隔。同机部署多个 decode 实例要错开 `--base-gpu-id`；同机部署多个 prefill 会撞 bootstrap 端口，目前不支持。

HiCache 是另一条线：把 RadixAttention 的前缀缓存扩展到 device/host/remote 三层，Mooncake Store 做 L3。启动 master 后：

```bash
mooncake_master --enable_http_metadata_server=true

MOONCAKE_MASTER=127.0.0.1:50051 \
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B \
  --page-size 64 \
  --enable-hierarchical-cache \
  --hicache-storage-backend mooncake \
  --hicache-storage-prefetch-policy timeout
```

`timeout` 是 HiCache 从远端预取时的停止策略，通常在 Mooncake 后端下比较平衡；`best_effort` 更快但可能不完整，`wait_complete` 更稳，但要把预取等完再继续。HiCache 对内存和 hugepage 有要求，官方示例里的 `56gb`、`8gb` 是特定规模下的调优值，不是默认值，按自己的 `--hicache-size` 和机器内存重新算。

### 怎么选

| 目标 | 接入方式 | 需要 Store | 典型收益 |
| :--- | :--- | :--- | :--- |
| prefill/decode 分开扩缩容 | vLLM `MooncakeConnector` / SGLang `--disaggregation-mode` | 否 | TTFT 与 ITL 解耦，长 prompt 不拖累 decode |
| 多实例共享前缀缓存 | vLLM `MooncakeStoreConnector` / SGLang HiCache | 是 | 缓存命中率、TTFT、回源计算量 |
| PD + 共享池 | `MultiConnector` / HiCache + PD | 是 | 既拆阶段又复用 KV，部署最复杂 |
| MoE 弹性容错 | SGLang `--elastic-ep-backend mooncake` | 否（用 PG/EP） | 部分 rank 失败时继续服务 |
| 训练/推理权重同步 | Transfer Engine / Store | 可选 | 千卡级权重更新压到秒级（checkpoint-engine 案例里 1T 模型约 20s） |

## 性能数字怎么读

官方文档和论文里的数字很多，放在一起很容易误读。下面这张表把条件和口径标出来：

| 场景 | 数字 | 条件 |
| :--- | :--- | :--- |
| TE 带宽 | 87 GB/s（4×200 Gb RoCE）、190 GB/s（8×400 Gb RoCE），约为 TCP 的 2.4× / 4.6× | 40 GB 数据，官方 benchmark |
| vLLM PD | 峰值 142.25 GB/s，约 8×RoCE 的 71.1%；32K prompt（4.5 GB KV）传输 31.65 ms，占 TTFT 4.2% | vLLM V1 + `MooncakeConnector` |
| vLLM 共享池 | 吞吐 3.8×；P50 TTFT 降到 1/46、E2E 降到 1/8.6；60 卡下命中率 >95% | 官方 benchmark，1P1D、12 卡（前两项） |
| SGLang PD | ITL 降低约 30%，吞吐持平 | 官方 benchmark |
| 论文（模拟） | 有效请求容量 +59%~498%；缓存命中率最高 2.36×；prefill 计算节省最高 48% | LLaMA3-70B dummy、真实 trace 回放 |
| Kimi 生产 | README/arXiv 摘要：多承接 75% 请求；FAST'25 论文：A800/H800 分别 +115% / +107% | 两处数字不同，引用时注明来源 |

读这些数字时注意三件事：

1. **带宽是聚合值**。190 GB/s 是 8 张 400G 网卡加拓扑优化的结果，单机单卡到不了；拿它和单张网卡的 TCP 对比没有意义。
2. **命中率决定收益**。KV 复用率低时，Store 只是多了一层网络和内存开销。短 prompt、低重复、单实例场景，先别上共享池。
3. **baseline 很重要**。PD 解耦的对比对象是“同实例混合 prefill/decode”，共享池的对比对象是“单机 prefix cache”或“Redis 后端”；换成别的 baseline，倍数会变。

## 版本、生态与限制

截至 2026-11-01：

- **稳定版**：`0.3.13.post1`（2026-08-31，PyPI）；`0.3.14` 处于 pre-release（`pre-3`，2026-10-30），文档里的 Docker `0.3.14` 标签已经出现。
- **Python**：3.10+。默认 wheel 覆盖 CUDA 12.1–12.9；CUDA 13 用 `mooncake-transfer-engine-cuda13`；此外还有 non-CUDA、NPU、ROCm、MUSA、EFA 等变体，一个环境只装一个。
- **许可证**：Apache-2.0。2026 年 2 月加入 PyTorch Ecosystem。
- **生态**：vLLM、SGLang、TensorRT-LLM、LMDeploy、vLLM-Ascend、LMCache、NIXL、llm-d、Dynamo、verl、checkpoint-engine、TorchSpec 等都已接入；具体支持到哪一层（TE / Store / EP）要看上游版本。

限制也要说清楚：

- **它不压缩 KV**。Mooncake 负责搬和存，省显存靠的是把 KV 卸载到 DRAM/SSD，不是压缩；KV 量化是另一条线（见 [Quantization 分类](/posts/quantization/intro/)），两者可以叠加，但别指望 Store 帮你把 KV 变小。
- **它不替代引擎的调度**。Conductor 给网关提供缓存视图，最终怎么选实例还是网关/路由器的策略；引擎内部的连续批处理、chunked prefill 仍由 vLLM/SGLang 自己管。
- **强依赖网络**。RDMA 配不好时只能退回 TCP，延迟和带宽都不是一个量级；跨机房、跨云的网络条件下，先做小规模验证。
- **API 还在快速演进**。0.3.x 内部的小版本也会有接口和配置字段变化，生产环境把 Mooncake、引擎、CUDA/驱动一起锁版本。

## 部署踩坑清单

按落地顺序排：

1. **先确认需不需要**。单机、短上下文、低重复负载，vLLM/SGLang 自带的 prefix cache 往往够了；PD 解耦和跨实例共享才有明显收益。
2. **RDMA 环境**。Mellanox OFED、GPUDirect RDMA、`memlock` 限制、容器里的 `--ipc=host` 和 `--ulimit memlock=-1`，缺一个都可能初始化失败，或者只能退回 TCP。
3. **元数据服务**。先试 `P2PHANDSHAKE`，最少依赖；多机长期运行再上 HTTP/etcd。HTTP metadata 跨主机要配 token，别裸奔。
4. **Master 高可用**。单 Master 是 SPOF；生产开 etcd HA，客户端地址用 `etcd://` 形式。
5. **字段拼写**。`setup()` 用 `rdma_devices`/`master_server_addr`，JSON 用 `device_name`/`master_server_address`；两套拼写不能混用，写错轻则被忽略、走了默认值，重则连不上 master。
6. **对象不可变**。重复 `put` 返回 `-705`，更新用 `upsert`；key 里带上模型、租户、内容哈希，避免跨模型串缓存。
7. **vLLM 路由契约**。`transfer_id` 和 bootstrap 地址要转发到位，否则 KV 传输静默失效；TP/DP 下检查 bootstrap 端口是否唯一。
8. **DP 哈希种子**。`PYTHONHASHSEED=0`，否则跨 rank 的 block hash 对不上。
9. **版本配对**。老版本 vLLM（≤0.8.4）走 V0 路径时需要 `mooncake-transfer-engine ≤0.3.3.post2`；新版本按官方集成页的对应关系来，别拿旧 wheel 配新引擎。
10. **HiCache 内存**。hugepage、`MC_MMAP_ARENA_POOL_SIZE`、`MOONCAKE_GLOBAL_SEGMENT_SIZE` 要一起算；官方示例的数值是特定机器上的调优结果，不是通用默认。

## 和本站其他文章的关系

- [vLLM 聊天补全请求全链路](/posts/llm-engine/vllm/) 和 [SGLang 请求全链路](/posts/llm-engine/sglang/) 讲的是单实例内部：请求怎么进调度器、前缀怎么查、KV 怎么分配。Mooncake 讲的是这些实例之间：KV 怎么跨节点搬、怎么共享。两篇合起来看，正好是“机内”和“机间”两层。
- [Kernel 分类](/posts/kernel/intro/) 讲算子内部实现；Mooncake 的 EP/PG 里有 MoE 通信 kernel，但主体是数据搬运和存储，不展开到指令级。
- [Quantization 分类](/posts/quantization/intro/) 讲 KV/权重的低精度表示；Mooncake 不改变 KV 的数值格式，量化后的 KV 一样可以走 TE/Store，两者是正交的优化。

## 参考资料

- 仓库：[kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake)
- 文档：[Mooncake Documentation](https://kvcache-ai.github.io/Mooncake/)
- 论文：[Mooncake: Trading More Storage for Less Computation (FAST'25)](https://www.usenix.org/system/files/fast25-qin.pdf)，[arXiv:2407.00079](https://arxiv.org/abs/2407.00079)
- vLLM 集成：[PD 解耦](https://kvcache-ai.github.io/Mooncake/deployment/integrations/vllm/disagg-prefill-decode.html)、[KV Cache 共享](https://kvcache-ai.github.io/Mooncake/deployment/integrations/vllm/kv-cache-storage.html)
- SGLang 集成：[PD 解耦](https://kvcache-ai.github.io/Mooncake/deployment/integrations/sglang/pd-disaggregation.html)、[HiCache](https://lmsys.org/blog/2025-09-10-sglang-hicache/)
- PyPI：[mooncake-transfer-engine](https://pypi.org/project/mooncake-transfer-engine/)
