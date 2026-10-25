---
title: "SGLang 请求全链路：网关 → 调度器 → KV Cache"
published: 2026-10-26T12:04:00+08:00
description: 把网关之后的引擎段拆开：TokenizerManager 入口、waiting → running 调度、Radix 前缀缓存命中与 KV 分配，直到 token 流回传和响应条目落库。
tags: [SGLang, 时序图]
category: LLM Engine
draft: false
---

上一篇停在网关，这一张把镜头推进引擎：请求怎么从 `TokenizerManager` 进入调度器，Radix 前缀缓存怎么命中，KV 什么时候分配，最后 token 流又怎么回到网关。

## 读图重点

- **入口**：SMG 网关通过 gRPC 把请求交给 `TokenizerManager`，随后进入调度器的 waiting 队列。
- **前缀与 KV**：Radix 缓存先查前缀，命中 L 个 token（未命中就是 0）；接着分配 KV，并把这次的新前缀写回树里。
- **回传**：token 流回到网关，网关写响应和 history 条目；trace、事件和 metrics 走异步上报，不阻塞返回。

<iframe src="/demos/llm-engine/sglang/sglang-request-engine.html?embed=1" title="SGLang 请求全链路" loading="lazy" style="box-sizing:border-box;width:100%;aspect-ratio:920/958;min-height:440px;border:1px solid rgba(128,128,128,.25);border-radius:12px;background:transparent"></iframe>
