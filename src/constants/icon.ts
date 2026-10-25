import type { Favicon } from "@/types/config.ts";

// 站点图标：由 GitHub 头像裁切而来（见 README「站点图标」一节）。
// 生成的尺寸见 public/favicon/，这是默认使用的图标列表。
export const defaultFavicons: Favicon[] = [
	{
		src: "/favicon/favicon-32.png",
		sizes: "32x32",
	},
	{
		src: "/favicon/favicon-128.png",
		sizes: "128x128",
	},
	{
		src: "/favicon/favicon-180.png",
		sizes: "180x180",
	},
	{
		src: "/favicon/favicon-192.png",
		sizes: "192x192",
	},
];
