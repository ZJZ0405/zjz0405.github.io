import { h } from "hastscript";

/**
 * Render a `:::en ... :::` or `:::zh ... :::` container directive as a
 * language block that the post language switch can show or hide.
 *
 * @param {Object} properties - The properties of the component.
 * @param {import('mdast').RootContent[]} children - The children elements of the component.
 * @param {'en'|'zh'} lang - The language of the block.
 * @returns {import('hast').Element} The language block element.
 */
export function LanguageBlockComponent(properties, children, lang) {
	if (!Array.isArray(children) || children.length === 0) {
		return h("div", { class: "hidden" });
	}

	return h(
		"div",
		{
			class: `lang-block lang-${lang}`,
			"data-lang": lang,
			lang: lang === "zh" ? "zh-CN" : "en",
		},
		children,
	);
}
