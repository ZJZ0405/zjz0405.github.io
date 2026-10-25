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

`.github/workflows/sync-awesome-skills.yml` 每天同步一次 Tools 说明，有变化时提交并显式触发一次部署。

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

## LLM Engine 图集

`src/content/posts/llm-engine/` 按引擎归档 Archify 生成的交互图，文章统一 `category: LLM Engine`，
正文用 iframe 引用 `public/demos/llm-engine/<engine>/` 下的自包含 HTML：

- vLLM：`vllm/chat-completion-request.md`
- SGLang：`sglang/gateway-request.md`、`sglang/engine-request.md`
- DeepInfer：`deepinfer/generate-request.md`、`deepinfer/scheduler-kv.md`、`deepinfer/event-topology.md`

HTML 更新时替换同名文件即可。iframe 加 `?embed=1` 隐藏页面外壳只留图，`?present=1` 进入演示模式。

## Quantization 分类

`src/content/posts/quantization/` 收低精度推理相关笔记，文章统一 `category: Quantization`。
首篇 `intro.md` 是分类导言，`from-scratch.md` 是量化算法长文（仿射量化、校准、PTQ/QAT、GPTQ/AWQ/SmoothQuant、FP8/FP4），配图在 `public/images/quantization/`。图表准备了明暗两套：亮色用 `<name>.png`，暗色用 `<name>-dark.png`（同一 notebook、同一 seed 重跑，只换 matplotlib 配色，两套尺寸一致）；文章里的 `<img>` 同时写 `data-light-src` 与 `data-dark-src`，`Layout.astro` 会在主题切换时更新 `src`，灯箱打开的是当前主题那张。后续按主题平铺或分子目录即可，侧栏分类计数会自动更新。

## Tools 分类与 awesome-skills 说明

`src/content/posts/tools/awesome-skills.md` 是 [awesome-skills](https://github.com/ZJZ0405/awesome-skills) 的安装与使用说明，`category: Tools`。

文章里只有两组标记块内的版本号、技能表、工具别名数和最新 release 下载按钮由脚本生成，正文手写：

```md
<!-- awesome-skills:catalog:start -->
...
<!-- awesome-skills:catalog:end -->

<!-- awesome-skills:download:start -->
...
<!-- awesome-skills:download:end -->
```

脚本 `scripts/sync-awesome-skills.mjs`（零依赖，Node 18+）拉取上游 `VERSION`、`catalog.json`、`sources.json`、`lib/agents.tsv`，再通过 GitHub API（失败时回退到 commits Atom feed）取最近提交；同时请求 GitHub Releases latest，取最新 tag 的 `.tar.gz` / `.sha256` / `.zip` 资产。它重写技能清单和下载按钮，并更新 frontmatter 的 `updated`；Releases API 失败时保留旧按钮，只更新技能清单。卡片样式在 `src/styles/markdown.css`。

本地命令：

```bash
pnpm sync:skills                                  # 手动同步
pnpm sync:skills:check                            # 只检查，过期时退出码 1
node scripts/sync-awesome-skills.mjs --dry-run    # 只打印，不写文件
```

本地需要代理时，脚本会走 Node 内置 fetch：Node 24 用 `NODE_USE_ENV_PROXY=1` 加 `HTTPS_PROXY`，老版本 Node 需要自行确认 fetch 的代理行为。GitHub Actions 里不需要代理。

`.github/workflows/sync-awesome-skills.yml` 每天北京时间 09:17 跑一次，只有上游变化才提交；提交后用 `gh workflow run deploy.yml` 显式触发部署。原因是 `GITHUB_TOKEN` 推送的 commit 不会触发 `push` 事件，不补这一步仓库更新了但网站不会重建。也可以在 awesome-skills 仓库里发 `repository_dispatch` 立即触发：

```bash
gh api repos/ZJZ0405/zjz0405.github.io/dispatches \
  -f event_type=awesome-skills-updated
```

设计上不在 Astro 构建期访问 GitHub：静态构建要保持离线、可复现，raw / API 在国内也不稳定。同步失败只会让这个 workflow 报错，站点继续发布已有内容。若仓库开了分支保护，需要把 Actions bot 加进允许直接推送的名单，或把 workflow 改成 PR 流程。

## 站点配置

`src/config.ts` 里改：

- `siteConfig`：标题、副标题、语言、主题色、横幅、目录
- `navBarConfig`：导航项
- `profileConfig`：头像、昵称、简介、社交链接

## 头像与站点图标

侧边栏头像 `public/avatar.jpg` 取自 GitHub 用户头像（`https://avatars.githubusercontent.com/u/105736491`，最大 460×460），
下载到仓库里本地引用，避免国内访问 GitHub CDN 不稳定。想自动跟随 GitHub 头像，可把 `profileConfig.avatar`
改成直链 `https://github.com/ZJZ0405.png?size=460`。

浏览器标签页图标由同一张头像裁切生成，裁切区域为原图 `(60, 30, 330, 300)`，圆角 22%：

```bash
python3 - <<'PY'
from PIL import Image, ImageDraw, ImageFilter

crop = Image.open("public/avatar.jpg").convert("RGB").crop((60, 30, 330, 300))

def rounded(img, rr=0.22):
    n = img.size[0]
    m = Image.new("L", (n * 4, n * 4), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, n * 4 - 1, n * 4 - 1], radius=int(n * 4 * rr), fill=255)
    out = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    out.paste(img.convert("RGBA"), (0, 0), m.resize((n, n), Image.LANCZOS))
    return out

def make(size, radius=True):
    im = crop.resize((size, size), Image.LANCZOS)
    im = im.filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=2))
    return rounded(im) if radius else im

def save_png(im, path):  # 量化到 255 色，压到原来的 1/4
    q = im.quantize(colors=255, method=Image.FASTOCTREE) if im.mode == "RGBA" else im.quantize(colors=255)
    q.save(path, optimize=True)

for size in (32, 128, 192):
    save_png(make(size), f"public/favicon/favicon-{size}.png")
save_png(make(180, radius=False), "public/favicon/favicon-180.png")  # iOS 自己裁圆角
make(48).save("public/favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)])
PY
```

图标列表在 `src/constants/icon.ts` 的 `defaultFavicons`；`siteConfig.favicon` 留空即用这份默认值。

## 自定义域名

1. `public/CNAME` 写入域名。
2. Settings → Pages → Custom domain 填同一域名。
3. DNS：根域 4 条 A 记录指向 `185.199.108.153` / `185.199.109.153` / `185.199.110.153` / `185.199.111.153`；`www` CNAME 指向 `zjz0405.github.io`。
4. 勾选 Enforce HTTPS。

## 技术栈

Astro 5 · Fuwari · Tailwind CSS · Svelte · Pagefind · GitHub Actions
