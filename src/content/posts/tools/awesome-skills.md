---
title: "awesome-skills 使用说明：一个仓库带走所有 agent 技能"
published: 2026-10-26
updated: 2026-10-26
description: "从打包、校验到离线安装：把同一套 agent 技能接进 Claude Code、Codex、Cursor、DSH 等工具，并让这份说明跟着上游仓库自动更新。"
tags: [Agent Skills, 工具链, 离线安装, GitHub Actions]
category: Tools
draft: false
---

[awesome-skills](https://github.com/ZJZ0405/awesome-skills) 是一个可离线分发的 agent 技能集合，外加一个纯 POSIX shell 的安装器。它要解决的不是“再写一个 skill”，而是**技能装在哪、换机器怎么带走**。

Claude Code 读 `~/.claude/skills`，Cursor 读 `~/.cursor/skills`，Codex、DSH 和另外十几个工具共享 `~/.agents/skills`。同一个技能想在每个工具里都能用，就得在每个目录各放一份。awesome-skills 的做法是先把技能装进共享 hub，再按各工具目录建软链或复制；`--tool auto` 会扫描本机，把认得出的 agent 一次接上。

安装器本身不联网，也不需要 Node、Python 或 npm。打包好的 tar.gz 拷进内网机器就能装。

## 30 秒上手

### 联网机器：打包

```sh
git clone https://github.com/ZJZ0405/awesome-skills.git
cd awesome-skills
./scripts/pack.sh
ls dist/
# awesome-skills-0.1.1.tar.gz
# awesome-skills-0.1.1.tar.gz.sha256
```

`pack.sh` 会排除 `.git`、`dist/`、`__pycache__` 和本机安装目录；加 `--zip` 可以额外生成 zip。版本号取自仓库根目录的 `VERSION`，所以文件名会随版本变。

### 离线机器：校验、解压、安装

```sh
shasum -a 256 -c awesome-skills-0.1.1.tar.gz.sha256   # Linux 用 sha256sum -c
tar xzf awesome-skills-0.1.1.tar.gz
cd awesome-skills-0.1.1
./install.sh --tool auto
```

装完**重开 agent 会话**，很多工具只在启动时扫描 skills 目录。

> 不想打包也可以：`git clone` 后把整个目录拷过去，运行 `./install.sh --tool auto`。安装过程同样不联网。

## 常用命令

| 想做的事 | 命令 |
| :--- | :--- |
| 装到探测到的工具 | `./install.sh --tool auto` |
| 只装指定工具 | `./install.sh --tool claude-code,cursor,codex` |
| 铺满所有工具目录 | `./install.sh --tool all` |
| 只装一个技能 | `./install.sh --skill handoff` |
| 装进某个项目 | `./install.sh --project ~/code/my-app` |
| 不建软链，复制文件 | `./install.sh --copy` |
| 先看计划不落盘 | `./install.sh --dry-run` |
| 覆盖非本工具安装的路径 | `./install.sh --force` |
| 看仓库 / 已装清单 | `./bin/awesome-skills list [--available\|--installed]` |
| 看工具别名与路径 | `./bin/awesome-skills tools [--detected]` |
| 自检 | `./bin/awesome-skills doctor` |
| 卸载 | `./bin/awesome-skills uninstall --tool cursor` / `--all` |

`--force` 不是静默覆盖：旧内容会先移到备份目录。`uninstall` 也只删清单里记录过的路径。

## 技能装到哪里

全局安装的落点是：

```text
~/.agents/skills/<name>       # 共享 hub
~/.claude/skills/<name>       # 软链到 hub
~/.agents/.awesome-skills/
└── manifest.tsv              # 安装清单
```

项目安装（`--project`）把 hub 换成 `<project>/.agents/skills`，镜像放进 `<project>/.claude/skills` 等目录，不影响全局。

几个环境变量会改变落点：

| 变量 | 作用 |
| :--- | :--- |
| `AWESOME_SKILLS_HOME` | 覆盖全局 hub，默认 `~/.agents/skills` |
| `CLAUDE_CONFIG_DIR` | Claude Code 镜像改到 `$CLAUDE_CONFIG_DIR/skills` |
| `XDG_CONFIG_HOME` | crush / goose / opencode 的配置基目录 |
| `CODEX_HOME` | 只影响 Codex 的探测路径，技能仍在共享 hub |

变量的解析发生在**安装时**。改完变量后旧位置的链接不会自己搬走，先 `uninstall --all` 再重装更干净。Windows 的 MSYS / MinGW / Cygwin 环境会自动退化成复制模式。

完整路径表在仓库的 `docs/tools.md`，或者直接跑 `./bin/awesome-skills tools`。

## 仓库里有什么

每个技能就是一个目录，安装器只复制这个目录：

```text
skills/<name>/
├── SKILL.md      # 必需
├── references/   # 可选：细节
└── scripts/      # 可选：脚本
```

覆盖画图、论文阅读、实验分析、前端、MCP、测试和交接这些方向。下面的清单直接取上游 `catalog.json`，描述保留英文原文，方便和仓库里的 `SKILL.md` 对照。

<!-- awesome-skills:catalog:start -->
> 数据源：[awesome-skills](https://github.com/ZJZ0405/awesome-skills) [`v0.1.1`](https://github.com/ZJZ0405/awesome-skills/releases/tag/v0.1.1) · 最近提交 [`d9884e1`](https://github.com/ZJZ0405/awesome-skills/commit/d9884e1b0aa04dc37d3a24ecb5149acffc124806)（2026-10-25）
>
> **17** 个技能 · **80** 个工具别名 · 原创 2 个（`handoff`、`shell-review`），其余 15 个收录自上游。
>
> 本段由 `pnpm sync:skills` 自动生成，请勿手改。

| 技能 | 用途 | 许可证 |
| :--- | :--- | :--- |
| [`archify`](https://github.com/ZJZ0405/awesome-skills/tree/main/skills/archify) | Create validated architecture, workflow, sequence, data-flow, lifecycle/state, ERD… | MIT |
| [`deep-explain`](https://github.com/ZJZ0405/awesome-skills/tree/main/skills/deep-explain) | Explain a mathematical concept or theorem in exhaustive reader-friendly detail, with… | MIT |
| [`formula-derivation`](https://github.com/ZJZ0405/awesome-skills/tree/main/skills/formula-derivation) | Structures and derives research formulas when the user wants to 推导公式, build a theory… | MIT |
| [`frontend-design`](https://github.com/ZJZ0405/awesome-skills/tree/main/skills/frontend-design) | Guidance for distinctive, intentional visual design when building new UI or reshaping… | Apache-2.0 |
| [`handoff`](https://github.com/ZJZ0405/awesome-skills/tree/main/skills/handoff) | Write a compact, source-linked handoff note so another session, agent, or machine can… | MIT |
| [`html-ppt`](https://github.com/ZJZ0405/awesome-skills/tree/main/skills/html-ppt) | HTML PPT Studio — author professional static HTML presentations in many styles… | MIT |
| [`lean-formalize`](https://github.com/ZJZ0405/awesome-skills/tree/main/skills/lean-formalize) | Develop and verify a mathematical proof in Lean, continue an incomplete Lean project… | MIT |
| [`mcp-builder`](https://github.com/ZJZ0405/awesome-skills/tree/main/skills/mcp-builder) | Guide for creating high-quality MCP (Model Context Protocol) servers that enable LLMs… | Apache-2.0 |
| [`nature-reader`](https://github.com/ZJZ0405/awesome-skills/tree/main/skills/nature-reader) | Create source-grounded Chinese-English paper readers with aligned text, figures… | Apache-2.0 |
| [`nature-shared`](https://github.com/ZJZ0405/awesome-skills/tree/main/skills/nature-shared) | Internal shared-reference support package for installed Nature Skills, including… | Apache-2.0 |
| [`results-analysis`](https://github.com/ZJZ0405/awesome-skills/tree/main/skills/results-analysis) | This skill should be used when the user asks to "analyze experimental results", "run… | MIT |
| [`results-report`](https://github.com/ZJZ0405/awesome-skills/tree/main/skills/results-report) | This skill should be used when the user asks to "write an experiment report"… | MIT |
| [`shell-review`](https://github.com/ZJZ0405/awesome-skills/tree/main/skills/shell-review) | Review shell scripts for destructive commands, quoting and portability bugs, and… | MIT |
| [`skill-creator`](https://github.com/ZJZ0405/awesome-skills/tree/main/skills/skill-creator) | Create new skills, modify and improve existing skills, and measure skill performance.… | Apache-2.0 |
| [`statistical-analysis`](https://github.com/ZJZ0405/awesome-skills/tree/main/skills/statistical-analysis) | Guided statistical analysis for research data - test selection, assumption checking… | MIT |
| [`status-report`](https://github.com/ZJZ0405/awesome-skills/tree/main/skills/status-report) | Generate a status report with KPIs, risks, and action items. Use when writing a weekly… | Apache-2.0 |
| [`webapp-testing`](https://github.com/ZJZ0405/awesome-skills/tree/main/skills/webapp-testing) | Toolkit for interacting with and testing local web applications using Playwright.… | Apache-2.0 |
<!-- awesome-skills:catalog:end -->

## 加自己的技能

从模板起手：

```sh
mkdir -p skills/my-skill
cp templates/skill/SKILL.md skills/my-skill/SKILL.md
$EDITOR skills/my-skill/SKILL.md

python3 scripts/validate.py      # frontmatter、链接、目录约定
python3 scripts/catalog.py       # 重新生成 README 表格、catalog.json、docs/tools.md
sh tests/test-install.sh         # 安装器回归测试
./install.sh --tool claude-code --skill my-skill
```

收录现成目录：

```sh
python3 scripts/import.py ~/Downloads/some-skill
python3 scripts/import.py ./vendor/some-skill --name my-skill
```

`import.py` 会保留原 `LICENSE`、把来源写进 `sources.json`，并拒绝覆盖同名技能（除非 `--force`）。命名要求很硬：只允许小写字母、数字和单个连字符，`name` 必须与目录名一致，`description` 最多 1024 字符。`SKILL.md` 建议控制在 500 行以内，细节拆到 `references/`。

## 维护者：同步上游

第三方技能不手工复制，而是走白名单 + lock：

```sh
python3 scripts/sync_upstreams.py --dry-run          # 看计划
python3 scripts/sync_upstreams.py --upstream anthropic
python3 scripts/sync_upstreams.py --check            # 上游是否已前进
python3 scripts/sync_upstreams.py --prune            # 清理上游已删除的技能
```

`upstreams.json` 是唯一需要手写的白名单；`upstreams.lock.json` 记录 commit 和每个 vendored 文件的 SHA-256，必须提交。vendored 技能视为只读，要改就复制成新名字并移出 `include`；只改一行行为时可以放 `patches/<skill>/` 下的 unified diff，同步时会先打补丁再算哈希。

仓库的 GitHub Actions 每周一开一个 `chore/sync-skills` PR，**不会自动合并**。

## 安全边界

技能本质上是**给 agent 的指令**，带 `scripts/` 的技能还可能执行命令。所以：

- 安装前至少读一遍 `skills/<name>/SKILL.md`，确认没有意外的网络访问或删除操作；
- 陌生环境先跑 `./install.sh --tool auto --dry-run`，看它准备往哪写；
- 用 `--force` 前想清楚备份在哪；
- 卸载用 `uninstall`，不要手工 `rm -rf` 一堆 skills 目录。

安装器没有网络代码，这一点比“安装时顺手拉个最新版”更适合内网。

## 这份说明怎么保持更新

这一页里最容易过期的不是命令，而是**技能数量、版本和清单**。所以正文手写，只把易变事实放进一对以 `awesome-skills:catalog` 命名的 HTML 注释标记之间；脚本只替换这一段，不碰其他文字。

博客仓库里的 `scripts/sync-awesome-skills.mjs` 会拉取上游的 `VERSION`、`catalog.json`、`sources.json` 和 `lib/agents.tsv`，重写这段表格，并顺带更新 frontmatter 的 `updated`。上游 README 的技能表本来就是 `scripts/catalog.py` 的产物，所以这里不解析 README 散文，只读同一批机器可读文件。本地手动同步：

```sh
pnpm sync:skills          # 重写标记块
pnpm sync:skills:check    # 只检查；过期时退出码 1
```

GitHub Actions 每天北京时间 09:17 跑一次，只有生成内容变化才提交；提交之后显式触发一次 Pages 部署。原因很实际：`GITHUB_TOKEN` 推的 commit 不会触发其他 workflow，不补这一步，仓库更新了但网站不会重建。

刻意没有在 Astro 构建期请求 GitHub API：静态站构建要保持离线、可复现，而且 raw / API 在国内并不稳定。生成结果落盘进 Markdown，构建只读本地文件；抓取失败时同步任务报错，站点照常发布旧版本。
