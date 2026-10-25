---
title: "DeepInfer 事件流拓扑：请求队列、多模态通道与 CP 广播"
published: 2026-10-26T12:01:00+08:00
description: 事件流拓扑：waiting 队列、多模态 input_queue、Scheduler 单线程批调度、execute_model_step / sample、post_process 与 KV 回收之间的通道和顺序。
tags: [DeepInfer, 数据流]
category: LLM Engine
draft: false
---

前几张是单请求的时间轴，这张换成数据流：请求从哪来、经过哪些队列和线程、在哪个环节被排序，最后又怎么回到调用方。适合先建立整体地图，再回头看时序细节。

## 读图重点

- **五段结构**：事件源 → 通道 → 有序处理 → 执行与状态 → 消费与回收，颜色区分主数据、异步通道、数据存储和数据流。
- **队列与通道**：waiting 队列是 FIFO，被抢占的请求回插队首；多模态请求走无界 `mp.Queue` 的 `input_queue` / `output_queue`；CP 广播经 `MessageQueue` 进 sharding buffer。
- **单线程调度**：`Scheduler::schedule` 是 C++ 单线程批调度，`execute_model_step` 按 TP / EP rank 执行，`sample` 做采样和 nextn 草稿，`post_process` 汇总 `RequestOutput` 并按请求排序回收。

<iframe src="/demos/llm-engine/deepinfer/deepinfer-event-topology.html?embed=1" title="DeepInfer 事件流拓扑" loading="lazy" style="box-sizing:border-box;width:100%;aspect-ratio:1068/686;min-height:380px;border:1px solid rgba(128,128,128,.25);border-radius:12px;background:transparent"></iframe>

<p style="text-align:center;margin-top:-0.25rem"><a href="/demos/llm-engine/deepinfer/deepinfer-event-topology.html" target="_blank" rel="noopener">在新标签页打开原网页 ↗</a></p>
