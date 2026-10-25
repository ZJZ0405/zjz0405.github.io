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
