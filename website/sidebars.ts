import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docsSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Getting started',
      collapsed: false,
      items: [
        'getting-started/installation',
        'getting-started/quickstart',
        'getting-started/training-shd',
      ],
    },
    {
      type: 'category',
      label: 'Guides',
      collapsed: false,
      items: [
        'guides/coming-from-snntorch',
        'guides/silent-networks',
        'guides/execution',
        'guides/topologies',
        'guides/convnets',
        'guides/online-learning',
        'guides/plasticity',
        'guides/visualization',
        'guides/nir',
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      collapsed: false,
      items: [
        'reference/neurons',
        'reference/surrogates',
        'reference/layers',
        'reference/execution',
        'reference/training',
        'reference/data',
      ],
    },
    'benchmarks',
  ],
};

export default sidebars;
