---
title: "DeepInfer 单请求时序：调度、KV 分配与解码返回"
published: 2026-10-26T12:02:00+08:00
description: 调度细节：prefill 水位、chunked prefill、decode 块不足时的抢占、append_slot 补块，以及结束或中止时 block 与 state 的回收。
tags: [DeepInfer, 时序图]
category: LLM Engine
draft: false
---

这是单请求路径的调度细看版：prefill 什么时候才被允许开始，decode 阶段 KV 块不够时怎么抢占，请求结束时 block 和 state 又在哪里回收。

## 读图重点

- **prefill 有水位**：`add_request` → `add_sequence` → `waiting.push_back`；`schedule(prefill)` 要等水位达标才尝试，chunked prefill 可以切块。
- **decode 会抢占**：`schedule(decode)` 在块不足时触发抢占，`append_slot` 按需补一块，再组 decode 批。
- **收尾回收**：结束或中止走 `free_seq`，释放 blocks 和 state，`ref_count` 归零后才算真正回收；`log_step` 同样是可选的。

<iframe src="/demos/llm-engine/deepinfer/deepinfer-scheduler-kv.html?embed=1" title="DeepInfer 单请求时序：调度、KV 分配与解码返回" loading="lazy" style="box-sizing:border-box;width:100%;aspect-ratio:1061/1173;min-height:460px;border:1px solid rgba(128,128,128,.25);border-radius:12px;background:transparent"></iframe>

<p style="text-align:center;margin-top:-0.25rem"><a href="/demos/llm-engine/deepinfer/deepinfer-scheduler-kv.html" target="_blank" rel="noopener">在新标签页打开交互图 ↗</a></p>
