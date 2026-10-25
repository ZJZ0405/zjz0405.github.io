---
title: "DeepInfer 请求全链路：事件拓扑、单请求时序与调度 KV"
published: 2026-10-26T12:01:00+08:00
description: 三张图看同一条离线推理路径：事件流拓扑给出队列与线程的整体地图，generate 时序展开从入队到解码文本的过程，调度细看图放大 prefill 水位、chunked prefill、decode 抢占和 KV 回收。
tags: [DeepInfer, 时序图, 数据流]
category: LLM Engine
draft: false
---

DeepInfer 的离线路径比在线服务短一截：没有鉴权，没有前缀缓存查询，也不落盘。三张图从不同粒度看同一套流程：先用事件流拓扑建立整体地图，再沿单请求时间轴走一遍，最后放大调度与 KV 的细节。

## 事件流拓扑：请求队列、多模态通道与 CP 广播

这张图换成数据流：请求从哪来、经过哪些队列和线程、在哪个环节被排序，最后又怎么回到调用方。适合先建立整体地图，再回头看后面的时序细节。

读图时重点看三处：

- **五段结构**：事件源 → 通道 → 有序处理 → 执行与状态 → 消费与回收，颜色区分主数据、异步通道、数据存储和数据流。
- **队列与通道**：waiting 队列是 FIFO，被抢占的请求回插队首；多模态请求走无界 `mp.Queue` 的 `input_queue` / `output_queue`；CP 广播经 `MessageQueue` 进 sharding buffer。
- **单线程调度**：`Scheduler::schedule` 是 C++ 单线程批调度，`execute_model_step` 按 TP / EP rank 执行，`sample` 做采样和 nextn 草稿，`post_process` 汇总 `RequestOutput` 并按请求排序回收。

<iframe src="/demos/llm-engine/deepinfer/deepinfer-event-topology.html?embed=1" title="DeepInfer 事件流拓扑" loading="lazy" style="box-sizing:border-box;width:100%;aspect-ratio:1068/686;min-height:380px;border:1px solid rgba(128,128,128,.25);border-radius:12px;background:transparent"></iframe>

<p style="text-align:center;margin-top:-0.25rem"><a href="/demos/llm-engine/deepinfer/deepinfer-event-topology.html" target="_blank" rel="noopener">在新标签页打开原网页 ↗</a></p>

## 单请求时序：从 generate 到解码文本

从 `generate(prompts)` 开始，到一个 `RequestOutput` 被解码成文本结束。这是单请求路径的概览版。

读图时重点看三处：

- **入队**：`add_request(ids)` 把序列推进 waiting 队列；`BlockManager` 按 `block_table` 分配 KV 块，这里不做前缀查找。
- **prefill 与 decode**：prefill 批带上 metadata 跑 `forward(prefill)`，`process_outputs` 拿到首 token 并把完成的序列转入 running；之后每轮 `forward(decode)` 采样，循环到 EOS 或长度上限。
- **返回与日志**：最终 `RequestOutput` 携带解码文本；`log_step` 走可选 daemon 日志线程，默认关闭。

<iframe src="/demos/llm-engine/deepinfer/deepinfer-request.html?embed=1" title="DeepInfer 单请求时序：从 generate 到解码文本" loading="lazy" style="box-sizing:border-box;width:100%;aspect-ratio:920/818;min-height:420px;border:1px solid rgba(128,128,128,.25);border-radius:12px;background:transparent"></iframe>

<p style="text-align:center;margin-top:-0.25rem"><a href="/demos/llm-engine/deepinfer/deepinfer-request.html" target="_blank" rel="noopener">在新标签页打开原网页 ↗</a></p>

## 调度细节：prefill 水位、抢占与 KV 回收

这是单请求路径的调度细看版：prefill 什么时候才被允许开始，decode 阶段 KV 块不够时怎么抢占，请求结束时 block 和 state 又在哪里回收。

读图时重点看三处：

- **prefill 有水位**：`add_request` → `add_sequence` → `waiting.push_back`；`schedule(prefill)` 要等水位达标才尝试，chunked prefill 可以切块。
- **decode 会抢占**：`schedule(decode)` 在块不足时触发抢占，`append_slot` 按需补一块，再组 decode 批。
- **收尾回收**：结束或中止走 `free_seq`，释放 blocks 和 state，`ref_count` 归零后才算真正回收；`log_step` 同样是可选的。

<iframe src="/demos/llm-engine/deepinfer/deepinfer-scheduler-kv.html?embed=1" title="DeepInfer 单请求时序：调度、KV 分配与解码返回" loading="lazy" style="box-sizing:border-box;width:100%;aspect-ratio:1061/1173;min-height:460px;border:1px solid rgba(128,128,128,.25);border-radius:12px;background:transparent"></iframe>

<p style="text-align:center;margin-top:-0.25rem"><a href="/demos/llm-engine/deepinfer/deepinfer-scheduler-kv.html" target="_blank" rel="noopener">在新标签页打开原网页 ↗</a></p>
