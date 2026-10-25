---
title: "DeepInfer 单请求时序：从 generate 到解码文本"
published: 2026-10-26T12:03:00+08:00
description: 离线 generate 路径：add_request 入 waiting、BlockManager 分配 KV、prefill 拿首 token、逐轮 decode 到 EOS，最后 RequestOutput 解码成文本。
tags: [DeepInfer, 时序图]
category: LLM Engine
draft: false
---

DeepInfer 的离线路径比在线服务短一截：没有鉴权，没有前缀缓存查询，也不落盘。这张图从 `generate(prompts)` 开始，到一个 `RequestOutput` 被解码成文本结束。

## 读图重点

- **入队**：`add_request(ids)` 把序列推进 waiting 队列；`BlockManager` 按 `block_table` 分配 KV 块，这里不做前缀查找。
- **prefill 与 decode**：prefill 批带上 metadata 跑 `forward(prefill)`，`process_outputs` 拿到首 token 并把完成的序列转入 running；之后每轮 `forward(decode)` 采样，循环到 EOS 或长度上限。
- **返回与日志**：最终 `RequestOutput` 携带解码文本；`log_step` 走可选 daemon 日志线程，默认关闭。

<iframe src="/demos/llm-engine/deepinfer/deepinfer-request.html?embed=1" title="DeepInfer 单请求时序：从 generate 到解码文本" loading="lazy" style="box-sizing:border-box;width:100%;aspect-ratio:920/818;min-height:420px;border:1px solid rgba(128,128,128,.25);border-radius:12px;background:transparent"></iframe>

<p style="text-align:center;margin-top:-0.25rem"><a href="/demos/llm-engine/deepinfer/deepinfer-request.html" target="_blank" rel="noopener">在新标签页打开交互图 ↗</a></p>
