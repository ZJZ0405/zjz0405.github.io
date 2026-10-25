# zjz0405.github.io

个人技术博客。Astro + [Fuwari](https://github.com/saicaca/fuwari) 主题，部署在 GitHub Pages。

- 站点：https://zjz0405.github.io/
- 文章：`src/content/posts/`
- 站点信息：`src/config.ts`
- 自包含 demo：`public/demos/`

## 本地开发

主题使用 pnpm：

```bash
pnpm install
pnpm dev       # http://localhost:4321
pnpm build     # 构建 + Pagefind 索引，产物在 dist/
pnpm preview   # 预览构建结果
```

## 发布

推送到 `main` 触发 `.github/workflows/deploy.yml`，用 `pnpm build` 构建后发布。

Pages 的 Source 已是 `GitHub Actions`，不需要改。

## 写文章

在 `src/content/posts/` 新建 `.md`：

```yaml
---
title: 标题
published: 2026-10-25
description: 摘要
tags: [CUDA, 性能]
category: 性能
draft: false
---
```

`draft: true` 不会出现在列表、归档和 RSS。

也可以用 `pnpm new-post <filename>` 生成骨架（来自主题的 `scripts/new-post.js`）。

## 嵌入 AI 生成的 HTML

完整页面放 `public/demos/foo.html`，文章里直接写：

```html
<iframe src="/demos/foo.html" title="演示" loading="lazy"
        style="width:100%;height:420px;border:0"></iframe>
```

样式和脚本被 iframe 隔离，不会污染主题。示例见 `public/demos/particles.html`。

## 站点配置

`src/config.ts` 里改：

- `siteConfig`：标题、副标题、语言、主题色、横幅、目录
- `navBarConfig`：导航项
- `profileConfig`：头像、昵称、简介、社交链接

## 自定义域名

1. `public/CNAME` 写入域名。
2. Settings → Pages → Custom domain 填同一域名。
3. DNS：根域 4 条 A 记录指向 `185.199.108.153` / `185.199.109.153` / `185.199.110.153` / `185.199.111.153`；`www` CNAME 指向 `zjz0405.github.io`。
4. 勾选 Enforce HTTPS。

## 技术栈

Astro 5 · Fuwari · Tailwind CSS · Svelte · Pagefind · GitHub Actions
