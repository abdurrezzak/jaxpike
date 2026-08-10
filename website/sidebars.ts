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
      label: 'Tutorials',
      collapsed: false,
      items: [
        'tutorials/first-network',
        'tutorials/custom-neurons',
        'tutorials/stdp-learning',
        'tutorials/long-sequences',
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
        'reference/graph',
        'reference/execution',
        'reference/training',
        'reference/plasticity',
        'reference/data',
      ],
    },
    'model-zoo',
    'benchmarks',
  ],
};

export default sidebars;
