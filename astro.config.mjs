import { defineConfig } from 'astro/config';

export default defineConfig({
  site: 'https://zjz0405.github.io',
  base: '/',
  markdown: {
    shikiConfig: { theme: 'github-dark' },
  },
});
