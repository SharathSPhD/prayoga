import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';

export default defineConfig({
  site: 'https://sharathsphd.github.io',
  base: '/prayoga',
  output: 'static',
  integrations: [tailwind()],
});
