import { defineConfig } from 'astro/config';
import preact from '@astrojs/preact';
import mdx from '@astrojs/mdx';

export default defineConfig({
  site: 'https://sharathsphd.github.io',
  base: '/prayoga',
  trailingSlash: 'ignore',
  integrations: [preact({ compat: true }), mdx()],
  vite: {
    build: {
      rollupOptions: {
        output: {
          manualChunks: {
            d3: ['d3'],
            three: ['three'],
          },
        },
      },
    },
  },
});
