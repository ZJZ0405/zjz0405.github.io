---
title: 开张：用 Astro + GitHub Pages 搭一个能嵌 demo 的博客
description: 记录这个站点的结构、写作流程，以及怎么把 AI 生成的 HTML 塞进文章里。
date: 2026-10-25
tags: [astro, github-pages, meta]
---

这是第一篇文章，顺便验证整条链路：Markdown 渲染、代码高亮、标签归档、RSS，以及嵌入式 demo。

## 为什么是这个组合

Astro 默认输出纯静态 HTML，除文章内嵌组件外几乎不向浏览器发送 JS；GitHub Pages 只托管静态文件，两者刚好匹配。写作时只写 Markdown，构建和发布交给 GitHub Actions。

## 目录结构

```text
src/
  content/blog/        # 文章（Markdown）
  layouts/             # 页面骨架
  pages/               # 路由
  components/          # 可复用组件
public/
  demos/               # 自包含 HTML，iframe 引用
  favicon.svg
.github/workflows/
  deploy.yml           # 推送即部署
```

## 写一篇新文章

在 `src/content/blog/` 下新建 `.md` 文件，frontmatter 只有四个字段：

```yaml
---
title: 标题
description: 摘要，会出现在列表和 RSS 里
date: 2026-10-25
tags: [cuda, perf]
---
```

保存并推送，Actions 会自动重建。`tags` 里出现的词会自动生成 `/tags/<tag>/` 页面，不需要手动维护索引。

## 嵌入 AI 生成的 HTML

把完整的 HTML 文件原样放进 `public/demos/`，然后在文章里用 iframe 引用：

```html
<iframe src="/demos/particles.html" title="粒子演示"
        loading="lazy" style="width:100%;height:360px;border:0"></iframe>
```

实际效果：

<iframe src="/demos/particles.html" title="Canvas 粒子演示" loading="lazy"
        style="width:100%;height:360px;border:1px solid #262b33;border-radius:12px;background:#0b1020"></iframe>

这样做的好处是样式和脚本被完全隔离——AI 生成的页面经常自带 Tailwind CDN 和全局 `body` 样式，直接内联会污染整站。代价是 iframe 内容不被搜索引擎索引，所以正文里最好补一段文字说明。

如果只是 HTML 片段、不需要隔离，也可以把它做成 `src/components/` 下的 `.astro` 组件，在页面里直接使用；组件里的 `<style>` 默认是局部作用域，`<script>` 会被打包。

## 一个代码块示例

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

- 加 giscus 评论
- 用 Pagefind 做站内搜索
- 把常用的可视化 demo 逐步搬进 `public/demos/`
