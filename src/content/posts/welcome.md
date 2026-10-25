---
title: 开张：用 Astro + GitHub Pages 搭一个能嵌 demo 的博客
published: 2026-10-25
description: 从零搭一个技术博客：写作流程、部署链路，以及怎么把 AI 生成的 HTML 塞进文章里。
tags: [Astro, GitHub Pages, 工程实践]
category: 工程实践
draft: false
---

这是第一篇文章，顺便验证整条链路：Markdown 渲染、代码高亮、标签归档、RSS，以及嵌入式 demo。

站点用 [Fuwari](https://github.com/saicaca/fuwari) 主题，构建交给 GitHub Actions，推送到 `main` 即发布。

## 写作流程

在 `src/content/posts/` 下新建 `.md`，frontmatter 只填几项：

```yaml
---
title: 标题
published: 2026-10-25
description: 摘要，会出现在列表和 RSS 里
tags: [CUDA, 性能优化]
category: 性能
draft: false
---
```

`draft: true` 的文章不会出现在列表、归档和 RSS 中。

## 嵌入 AI 生成的 HTML

把完整的 HTML 文件原样放进 `public/demos/`，然后在文章里用 iframe 引用。好处是样式和脚本被完全隔离——AI 生成的页面经常自带 Tailwind CDN 和全局 `body` 样式，直接内联会污染整站。

<iframe src="/demos/particles.html" title="Canvas 粒子演示" loading="lazy"
        style="width:100%;height:380px;border:1px solid #262b33;border-radius:12px;background:#0b1020"></iframe>

这里是一个 canvas 粒子系统，鼠标移动会产生排斥效果。完整源码在 `public/demos/particles.html`，只用了几十行原生 JS。

## 代码块

```cpp
__global__ void reduce(const float* __restrict__ in, float* out, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    sdata[tid] = (tid < n) ? in[tid] : 0.0f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}
```

## 下一步

- 接入 giscus 评论
- 把常写的可视化 demo 逐步收进 `public/demos/`
- 给长文补上前言和结论，让归档页更好读
