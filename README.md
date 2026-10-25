# zjz0405.github.io

个人技术博客，Astro 静态站点，部署在 GitHub Pages。

- 站点地址：https://zjz0405.github.io/
- 文章目录：`src/content/blog/`
- 自包含 demo：`public/demos/`

## 本地开发

```bash
npm install
npm run dev      # http://localhost:4321
npm run build    # 产物在 dist/
npm run preview  # 预览构建结果
```

## 发布

推送到 `main` 分支即触发 `.github/workflows/deploy.yml`，构建后发布到 Pages。

首次使用需要在仓库 **Settings → Pages → Source** 选择 **GitHub Actions**，否则工作流会报错。

## 写文章

在 `src/content/blog/` 新建 `.md`：

```yaml
---
title: 标题
description: 摘要
date: 2026-10-25
tags: [tag1, tag2]
draft: false
---
```

`draft: true` 的文章不会出现在列表、标签页和 RSS 里。

## 嵌入 AI 生成的 HTML

完整页面放 `public/demos/foo.html`，文章里：

```html
<iframe src="/demos/foo.html" title="演示" loading="lazy"
        style="width:100%;height:420px;border:0"></iframe>
```

需要高度自适应时，在 demo 里发送 `postMessage({ type: 'demo-height', height })`，
并用 `src/components/DemoFrame.astro` 包裹（该组件已监听此消息）。

HTML 片段可以直接写成 `.astro` 组件，样式默认局部作用域，不会污染全站。

## 自定义域名

1. 在 `public/` 放一个 `CNAME` 文件，内容为域名（如 `blog.example.io`）。
2. 仓库 Settings → Pages → Custom domain 填入同一域名。
3. DNS：根域用 4 条 A 记录指向 `185.199.108.153`、`185.199.109.153`、`185.199.110.153`、`185.199.111.153`；`www` 用 CNAME 指向 `zjz0405.github.io`。
4. 勾选 Enforce HTTPS。

## 技术栈

- Astro（内容集合 + 静态输出）
- 原生 CSS（跟随系统深浅色）
- GitHub Actions + GitHub Pages
