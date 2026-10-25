---
title: "SGLang 请求全链路：网关、调度器与 KV Cache"
published: 2026-10-26T12:04:00+08:00
description: 一条 /v1/responses 请求的两段旅程：SMG 网关做鉴权、载入历史和 cache_aware 选路；引擎侧 TokenizerManager 接单，Radix 前缀缓存命中后分配 KV，调度 prefill / decode，token 流回传并落库。
tags: [SGLang, 时序图]
category: LLM Engine
draft: false
---

SGLang 的一条 `/v1/responses` 请求可以拆成网关和引擎两段：SMG 网关负责鉴权、载入历史、选 worker 并把结果写回；引擎负责 tokenize、查前缀缓存、调度和分配 KV。两张图按请求流向拼起来，就是一次调用的完整生命周期。

## 网关层：鉴权、cache_aware 选路与响应落库

先停在网关层，看 SMG（sgl-model-gateway）怎么把一条 `/v1/responses` 请求接住、选一个 worker、再把流式结果写回并落库。引擎内部的调度不在这一张里。

读图时重点看三处：

- **鉴权与历史**：`auth_middleware` 校验 Bearer；不匹配或不是 Bearer 直接 401。带 `previous_response_id` 时向 `history_backend` 按需载入历史，读失败只 warn，会话不存在才 404。
- **cache_aware 选路**：前缀树命中就复用对应 worker，低于阈值则选最小负载的 worker，并把这次的前缀写回树里。
- **生成与落库**：通过 gRPC `generate` 拿到 token 流，网关写响应和 history 条目；`store=false` 时跳过落库，落库失败也只 warn，不挡响应。trace 和事件批次异步导出，不占响应路径。

<iframe src="/demos/llm-engine/sglang/sglang-request-lifecycle.html?embed=1" title="SGLang 模型网关 /v1/responses 请求时序" loading="lazy" style="box-sizing:border-box;width:100%;aspect-ratio:920/920;min-height:420px;border:1px solid rgba(128,128,128,.25);border-radius:12px;background:transparent"></iframe>

<p style="text-align:center;margin-top:-0.25rem"><a href="/demos/llm-engine/sglang/sglang-request-lifecycle.html" target="_blank" rel="noopener">在新标签页打开原网页 ↗</a></p>

## 引擎层：TokenizerManager、Radix 前缀缓存与 KV 分配

网关把请求交给引擎之后，镜头推进到 `TokenizerManager`：请求怎么进入调度器的 waiting 队列，Radix 前缀缓存怎么命中，KV 什么时候分配，最后 token 流又怎么回到网关。

读图时重点看三处：

- **入口**：SMG 网关通过 gRPC 把请求交给 `TokenizerManager`，随后进入调度器的 waiting 队列。
- **前缀与 KV**：Radix 缓存先查前缀，命中 L 个 token（未命中就是 0）；接着分配 KV，并把这次的新前缀写回树里。
- **回传**：token 流回到网关，网关写响应和 history 条目；trace、事件和 metrics 走异步上报，不阻塞返回。

<iframe src="/demos/llm-engine/sglang/sglang-request-engine.html?embed=1" title="SGLang 请求全链路" loading="lazy" style="box-sizing:border-box;width:100%;aspect-ratio:920/958;min-height:440px;border:1px solid rgba(128,128,128,.25);border-radius:12px;background:transparent"></iframe>

<p style="text-align:center;margin-top:-0.25rem"><a href="/demos/llm-engine/sglang/sglang-request-engine.html" target="_blank" rel="noopener">在新标签页打开原网页 ↗</a></p>

## 两层缓存怎么衔接

网关的 `cache_aware` 前缀树决定请求落到哪个 worker，引擎的 Radix 缓存决定这个 worker 上能复用多少 KV。网关命中只说明这个 worker 见过相似前缀；真正省掉多少 prefill，还要看引擎侧命中多少 token。两边都会把这次的新前缀写回各自的树。
