---
title: SGLang 模型网关 /v1/responses 请求时序
published: 2026-10-26T12:05:00+08:00
description: SMG 网关视角：Bearer 鉴权、按 previous_response_id 载入历史、cache_aware 前缀树选 worker、gRPC 流式生成，以及响应条目落库。
tags: [SGLang, 时序图]
category: LLM Engine
draft: false
---

这张图停在网关层，看 SMG（sgl-model-gateway）怎么把一条 `/v1/responses` 请求接住、选一个 worker、再把流式结果写回并落库。引擎内部的调度不在这一张里。

## 读图重点

- **鉴权与历史**：`auth_middleware` 校验 Bearer；不匹配或不是 Bearer 直接 401。带 `previous_response_id` 时向 `history_backend` 按需载入历史，读失败只 warn，会话不存在才 404。
- **cache_aware 选路**：前缀树命中就复用对应 worker，低于阈值则选最小负载的 worker，并把这次的前缀写回树里。
- **生成与落库**：通过 gRPC `generate` 拿到 token 流，网关写响应和 history 条目；`store=false` 时跳过落库，落库失败也只 warn，不挡响应。trace 和事件批次异步导出，不占响应路径。

<iframe src="/demos/llm-engine/sglang/sglang-request-lifecycle.html?embed=1" title="SGLang 模型网关 /v1/responses 请求时序" loading="lazy" style="box-sizing:border-box;width:100%;aspect-ratio:920/920;min-height:420px;border:1px solid rgba(128,128,128,.25);border-radius:12px;background:transparent"></iframe>

<p style="text-align:center;margin-top:-0.25rem"><a href="/demos/llm-engine/sglang/sglang-request-lifecycle.html" target="_blank" rel="noopener">在新标签页打开原网页 ↗</a></p>
