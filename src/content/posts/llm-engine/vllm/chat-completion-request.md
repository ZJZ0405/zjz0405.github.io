---
title: vLLM 聊天补全请求时序
published: 2026-10-26T12:06:00+08:00
description: 从 OpenAI 客户端发出 /v1/chat/completions，经鉴权、AsyncLLM 提交、KV 前缀与 offload 两级查找、EngineCore 调度，到 GPU 采样并返回 JSON 或 SSE 的完整链路。
tags: [vLLM, 时序图]
category: LLM Engine
draft: false
---

一次 `/v1/chat/completions` 要穿过 API Server、调度器、KV Cache 管理器和 GPU Worker 四层。这张图把一次请求的完整往返按时间顺序摊开，重点看 KV 的两次查找，以及流式与非流式的分叉。

## 读图重点

- **入口与鉴权**：API Server 先校验 Bearer token；启动时没配 `--api-key` 就整段跳过，校验失败直接 401。
- **两级 KV 查找**：先按 block hash 查 GPU 前缀缓存，命中 N 个 token 就能省掉这部分重算；没命中再落到 offload tier（CPU / 文件），那里才是持久化的那一层。
- **调度与返回**：EngineCore 调度 prefill / decode，GPU Worker 前向并采样；非流式聚合成 200 JSON，流式逐块走 SSE，trace 通过 OTLP 异步导出。

<iframe src="/demos/llm-engine/vllm/vllm-chat-request.html?embed=1" title="vLLM 聊天补全请求时序" loading="lazy" style="box-sizing:border-box;width:100%;aspect-ratio:920/902;min-height:420px;border:1px solid rgba(128,128,128,.25);border-radius:12px;background:transparent"></iframe>

<p style="text-align:center;margin-top:-0.25rem"><a href="/demos/llm-engine/vllm/vllm-chat-request.html" target="_blank" rel="noopener">在新标签页打开原网页 ↗</a></p>
