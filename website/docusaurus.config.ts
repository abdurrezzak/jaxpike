import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// The GitHub account, matching [project.urls] in pyproject.toml. The navbar, footer and
// "edit this page" links read from here, and `organizationName`/`url` below must use the
// same account for GitHub Pages to resolve.
const GITHUB_ACCOUNT = 'abdurrezzak';
const GITHUB_URL = `https://github.com/${GITHUB_ACCOUNT}/jaxpike`;

const config: Config = {
  title: 'jaxpike',
  tagline: 'Fast, flexible spiking neural networks in JAX',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: `https://${GITHUB_ACCOUNT}.github.io`,
  baseUrl: '/jaxpike/',

  organizationName: GITHUB_ACCOUNT,
  projectName: 'jaxpike',

  onBrokenLinks: 'throw',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          routeBasePath: 'docs',
          editUrl: `${GITHUB_URL}/tree/main/website/`,
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'jaxpike',
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Docs',
        },
        {to: '/docs/guides/coming-from-snntorch', label: 'From snnTorch', position: 'left'},
        {to: '/docs/benchmarks', label: 'Benchmarks', position: 'left'},
        {
          href: GITHUB_URL,
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Learn',
          items: [
            {label: 'Getting started', to: '/docs/getting-started/installation'},
            {label: 'Train SHD end to end', to: '/docs/getting-started/training-shd'},
            {label: 'Coming from snnTorch', to: '/docs/guides/coming-from-snntorch'},
          ],
        },
        {
          title: 'Reference',
          items: [
            {label: 'Neurons', to: '/docs/reference/neurons'},
            {label: 'Execution', to: '/docs/reference/execution'},
            {label: 'Benchmarks', to: '/docs/benchmarks'},
          ],
        },
        {
          title: 'Project',
          items: [
            {label: 'GitHub', href: GITHUB_URL},
            {label: 'NIR', href: 'https://github.com/neuromorphs/NIR'},
          ],
        },
      ],
      copyright: 'jaxpike is Apache-2.0 licensed. Built with Docusaurus.',
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'bash'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
