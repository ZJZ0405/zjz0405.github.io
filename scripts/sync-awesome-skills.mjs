#!/usr/bin/env node
/**
 * 从 ZJZ0405/awesome-skills 同步《使用说明》里的事实块。
 *
 * 只重写标记之间的内容（版本、技能表、工具别名数、上游提交），正文保持手写。
 * 数据来源：
 *   - VERSION            仓库版本
 *   - catalog.json       技能名 / 描述 / 许可证
 *   - sources.json       原创与收录来源
 *   - lib/agents.tsv     工具别名数量
 *   - GitHub API/Atom    最近一次提交
 *
 * 用法：
 *   node scripts/sync-awesome-skills.mjs
 *   node scripts/sync-awesome-skills.mjs --check
 *   node scripts/sync-awesome-skills.mjs --dry-run
 *
 * 环境变量 GITHUB_TOKEN / GH_TOKEN 可选，用于提高 GitHub API 的速率上限。
 * 不依赖任何第三方包。
 */

import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const OWNER = "ZJZ0405";
const REPO = "awesome-skills";
const BRANCH = "main";
const REPO_URL = `https://github.com/${OWNER}/${REPO}`;
const RAW_BASE = `https://raw.githubusercontent.com/${OWNER}/${REPO}/${BRANCH}`;
const POST_PATH = fileURLToPath(
	new URL("../src/content/posts/tools/awesome-skills.md", import.meta.url),
);
const START_MARKER = "<!-- awesome-skills:catalog:start -->";
const END_MARKER = "<!-- awesome-skills:catalog:end -->";
const DESC_LIMIT = 88;

const args = new Set(process.argv.slice(2));
const checkOnly = args.has("--check");
const dryRun = args.has("--dry-run");

if (args.has("--help") || args.has("-h")) {
	console.log(`同步 awesome-skills 说明文档

用法：
  node scripts/sync-awesome-skills.mjs            重写标记块
  node scripts/sync-awesome-skills.mjs --check    只检查，过期时退出码为 1
  node scripts/sync-awesome-skills.mjs --dry-run  只打印新块，不写文件

可选环境变量：GITHUB_TOKEN / GH_TOKEN（提高 GitHub API 速率上限）。`);
	process.exit(0);
}

for (const arg of args) {
	if (!["--check", "--dry-run", "--help", "-h"].includes(arg)) {
		console.error(`未知参数：${arg}（用 --help 查看用法）`);
		process.exit(2);
	}
}

const token = process.env.GITHUB_TOKEN || process.env.GH_TOKEN || "";
const shanghaiDate = new Intl.DateTimeFormat("en-CA", {
	timeZone: "Asia/Shanghai",
	year: "numeric",
	month: "2-digit",
	day: "2-digit",
});

function ymd(date) {
	const parts = shanghaiDate.formatToParts(date);
	const get = (type) => parts.find((part) => part.type === type)?.value ?? "";
	return `${get("year")}-${get("month")}-${get("day")}`;
}

async function fetchText(url, { authenticated = false } = {}) {
	const headers = { "User-Agent": "zjz0405.github.io-sync" };
	if (authenticated) {
		headers.Accept = "application/vnd.github+json";
		headers["X-GitHub-Api-Version"] = "2022-11-28";
		if (token) headers.Authorization = `Bearer ${token}`;
	}

	let lastError;
	for (let attempt = 1; attempt <= 3; attempt += 1) {
		try {
			const response = await fetch(url, {
				headers,
				signal: AbortSignal.timeout(15_000),
			});
			if (response.ok) return await response.text();

			const body = (await response.text()).slice(0, 200).replace(/\s+/g, " ");
			lastError = new Error(
				`GET ${url} -> ${response.status} ${response.statusText}${body ? `: ${body}` : ""}`,
			);
			if ([401, 403, 404].includes(response.status)) break;
		} catch (error) {
			const cause = error?.cause?.code ?? error?.cause?.message ?? "";
			lastError = new Error(
				`GET ${url} 失败：${error?.message ?? error}${cause ? `（${cause}）` : ""}`,
			);
		}
		if (attempt < 3) {
			await new Promise((resolve) => setTimeout(resolve, attempt * 500));
		}
	}
	throw lastError;
}

function parseJson(text, label) {
	try {
		return JSON.parse(text);
	} catch (error) {
		throw new Error(`${label} 不是合法 JSON：${error.message}`);
	}
}

async function getCommit() {
	try {
		const data = parseJson(
			await fetchText(
				`https://api.github.com/repos/${OWNER}/${REPO}/commits/${BRANCH}`,
				{ authenticated: true },
			),
			"GitHub commit API",
		);
		if (!data?.sha) throw new Error("响应里没有 sha");
		const date = new Date(
			data.commit?.committer?.date ?? data.commit?.author?.date ?? "",
		);
		if (Number.isNaN(date.getTime()))
			throw new Error("响应里没有可解析的提交时间");
		return {
			sha: data.sha,
			short: data.sha.slice(0, 7),
			date,
			url: data.html_url ?? `${REPO_URL}/commit/${data.sha}`,
		};
	} catch (apiError) {
		try {
			const xml = await fetchText(`${REPO_URL}/commits/${BRANCH}.atom`);
			const sha = xml.match(/Grit::Commit\/([0-9a-f]{40})/)?.[1];
			const updated =
				xml.match(/<entry>[\s\S]*?<updated>([^<]+)<\/updated>/)?.[1] ??
				xml.match(/<updated>([^<]+)<\/updated>/)?.[1];
			if (!sha || !updated) throw new Error("Atom feed 里没有提交号或时间");

			const date = new Date(updated);
			if (Number.isNaN(date.getTime())) {
				throw new Error(`Atom 时间无法解析：${updated}`);
			}
			return {
				sha,
				short: sha.slice(0, 7),
				date,
				url: `${REPO_URL}/commit/${sha}`,
			};
		} catch (atomError) {
			throw new Error(
				`获取上游提交失败。API：${apiError.message}；Atom：${atomError.message}`,
			);
		}
	}
}

function clip(text, limit) {
	const normalized = String(text ?? "")
		.replace(/`/g, "'")
		.replace(/\s+/g, " ")
		.trim();
	const chars = Array.from(normalized);
	if (chars.length <= limit) return normalized;

	let clipped = chars.slice(0, limit).join("");
	const boundary = Math.max(
		clipped.lastIndexOf(" "),
		clipped.lastIndexOf("。"),
		clipped.lastIndexOf("，"),
		clipped.lastIndexOf("；"),
		clipped.lastIndexOf(";"),
	);
	if (boundary >= Math.floor(limit * 0.6)) clipped = clipped.slice(0, boundary);
	return `${clipped.replace(/[\s,;:，。；：、-]+$/u, "")}…`;
}

function tableCell(value) {
	return String(value ?? "")
		.replace(/\|/g, "\\|")
		.replace(/\s+/g, " ")
		.trim();
}

function buildBlock({ version, commit, skills, sources, toolCount }) {
	const ordered = [...skills].sort((a, b) =>
		a.name.localeCompare(b.name, "en"),
	);
	const originalNames = Object.entries(sources?.skills ?? {})
		.filter(([, entry]) => entry?.source === "original")
		.map(([name]) => name)
		.filter((name) => ordered.some((skill) => skill.name === name))
		.sort((a, b) => a.localeCompare(b, "en"));

	const versionLabel =
		version === "dev"
			? "`dev`"
			: `[\`v${tableCell(version)}\`](${REPO_URL}/releases/tag/v${encodeURIComponent(version)})`;
	const originalNote = originalNames.length
		? `原创 ${originalNames.length} 个（${originalNames.map((name) => `\`${name}\``).join("、")}），其余 ${ordered.length - originalNames.length} 个收录自上游`
		: "技能来源记录在 `sources.json`";

	const rows = ordered.map((skill) => {
		const skillPath = skill.path ?? `skills/${skill.name}`;
		const url = `${REPO_URL}/tree/${BRANCH}/${skillPath}`;
		const license = skill.license ? tableCell(skill.license) : "—";
		return `| [\`${tableCell(skill.name)}\`](${url}) | ${tableCell(clip(skill.description, DESC_LIMIT))} | ${license} |`;
	});

	return [
		START_MARKER,
		`> 数据源：[awesome-skills](${REPO_URL}) ${versionLabel} · 最近提交 [\`${commit.short}\`](${commit.url})（${ymd(commit.date)}）`,
		">",
		`> **${ordered.length}** 个技能 · **${toolCount}** 个工具别名 · ${originalNote}。`,
		">",
		"> 本段由 `pnpm sync:skills` 自动生成，请勿手改。",
		"",
		"| 技能 | 用途 | 许可证 |",
		"| :--- | :--- | :--- |",
		...rows,
		END_MARKER,
	].join("\n");
}

function replaceBlock(markdown, block) {
	const start = markdown.indexOf(START_MARKER);
	const end = markdown.indexOf(END_MARKER);
	if (start === -1 || end === -1 || end < start) {
		throw new Error(
			`找不到标记块，请确认 ${path.relative(process.cwd(), POST_PATH)} 里有 ${START_MARKER} 和 ${END_MARKER}`,
		);
	}
	return (
		markdown.slice(0, start) + block + markdown.slice(end + END_MARKER.length)
	);
}

function currentBlock(markdown) {
	const start = markdown.indexOf(START_MARKER);
	const end = markdown.indexOf(END_MARKER);
	if (start === -1 || end === -1 || end < start) {
		throw new Error(`找不到标记块：${START_MARKER} … ${END_MARKER}`);
	}
	return markdown.slice(start, end + END_MARKER.length);
}

function bumpUpdated(markdown, date) {
	if (/^updated:\s*.*$/m.test(markdown)) {
		return markdown.replace(/^updated:\s*.*$/m, `updated: ${date}`);
	}
	return markdown.replace(/^(published:\s*.*)$/m, `$1\nupdated: ${date}`);
}

async function main() {
	if (!token) {
		console.log("提示：未设置 GITHUB_TOKEN，GitHub API 走匿名额度。");
	}

	const markdown = await readFile(POST_PATH, "utf8");
	const [versionText, catalogText, sourcesText, agentsText, commit] =
		await Promise.all([
			fetchText(`${RAW_BASE}/VERSION`),
			fetchText(`${RAW_BASE}/catalog.json`),
			fetchText(`${RAW_BASE}/sources.json`),
			fetchText(`${RAW_BASE}/lib/agents.tsv`),
			getCommit(),
		]);

	const version = versionText.trim();
	if (!/^[0-9A-Za-z._-]+$/.test(version)) {
		throw new Error(`上游 VERSION 内容异常：${JSON.stringify(version)}`);
	}

	const catalog = parseJson(catalogText, "catalog.json");
	if (!Array.isArray(catalog?.skills)) {
		throw new Error("catalog.json 里没有 skills 数组");
	}
	const skills = catalog.skills.map((skill) => {
		if (!skill?.name || typeof skill.name !== "string") {
			throw new Error("catalog.json 有技能缺少 name");
		}
		return {
			name: skill.name,
			description: skill.description ?? "",
			path: skill.path,
			license: skill.license ?? "",
		};
	});
	if (skills.length === 0) throw new Error("catalog.json 里没有任何技能");

	const sources = parseJson(sourcesText, "sources.json");
	const toolCount = agentsText
		.split("\n")
		.map((line) => line.trim())
		.filter((line) => line && !line.startsWith("#")).length;
	if (toolCount === 0) throw new Error("lib/agents.tsv 里没有解析到工具别名");

	const block = buildBlock({
		version,
		commit,
		skills,
		sources,
		toolCount,
	});
	const existing = currentBlock(markdown);

	if (existing === block) {
		console.log(
			`说明已是最新：${skills.length} 个技能 · ${toolCount} 个工具别名（${commit.short}）。`,
		);
		return;
	}

	if (checkOnly) {
		console.error(
			`说明已过期：上游为 ${skills.length} 个技能 · ${toolCount} 个工具别名（${commit.short}），请运行 pnpm sync:skills。`,
		);
		process.exitCode = 1;
		return;
	}

	if (dryRun) {
		console.log(block);
		return;
	}

	const next = bumpUpdated(replaceBlock(markdown, block), ymd(new Date()));
	await writeFile(POST_PATH, next, "utf8");
	console.log(
		`已同步：${skills.length} 个技能 · ${toolCount} 个工具别名（${commit.short}）-> ${path.relative(process.cwd(), POST_PATH)}`,
	);
}

main().catch((error) => {
	console.error(
		`\n同步失败：${error instanceof Error ? error.message : error}`,
	);
	process.exitCode = 1;
});
