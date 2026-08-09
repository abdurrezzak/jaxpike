import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import CodeBlock from '@theme/CodeBlock';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

const QUICKSTART = `import jax
import jaxpike as jp

k1, k2 = jax.random.split(jax.random.key(0))

net = jp.Sequential(
    jp.Dense(700, 256, key=k1, gain=jp.lif_gain(20.0)),
    jp.LinearLIF(tau=20.0, threshold=0.5),
    jp.Dense(256, 20, key=k2, gain=jp.lif_gain(20.0)),
    jp.LeakyIntegrator(tau=20.0),
)

membrane, state = jp.unroll_parallel(net, xs)   # (time, batch, features)
logits = jp.max_membrane_logits(membrane)`;

const RESULTS: {value: string; label: string; detail: string; to: string}[] = [
  {
    value: '31–43×',
    label: 'faster than snnTorch and Norse',
    detail: 'Identical SHD model, same GPU, same container. Within 1.35× of SpikingJelly’s fused CuPy kernel.',
    to: '/docs/benchmarks',
  },
  {
    value: '12×',
    label: 'less memory than SpikingJelly',
    detail: 'A 256-step BPTT graph in 64 MB against 792 MB, via rematerialization.',
    to: '/docs/guides/execution',
  },
  {
    value: 'flat in T',
    label: 'e-prop memory',
    detail: 'Online learning with no backward pass over time — 2671× less than BPTT at T=4000.',
    to: '/docs/guides/online-learning',
  },
  {
    value: '0.751',
    label: 'test accuracy on SHD',
    detail: 'Matching the 0.70–0.75 band published for Spyx under the same protocol.',
    to: '/docs/getting-started/training-shd',
  },
];

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link className="button button--primary button--lg" to="/docs/getting-started/quickstart">
            Get started
          </Link>
          <Link
            className="button button--secondary button--lg"
            to="/docs/guides/coming-from-snntorch">
            Coming from snnTorch
          </Link>
        </div>
      </div>
    </header>
  );
}

function Results() {
  return (
    <section className={styles.results}>
      <div className="container">
        <div className="row">
          {RESULTS.map((result) => (
            <div className="col col--3" key={result.label}>
              <Link to={result.to} className={styles.resultCard}>
                <div className={styles.resultValue}>{result.value}</div>
                <div className={styles.resultLabel}>{result.label}</div>
                <p className={styles.resultDetail}>{result.detail}</p>
              </Link>
            </div>
          ))}
        </div>
        <p className={styles.footnote}>
          Every number is measured on an NVIDIA T4 and reproducible from <code>benchmarks/</code>,{' '}
          <Link to="/docs/benchmarks">including the runs that went against us</Link>.
        </p>
      </div>
    </section>
  );
}

function Overview() {
  return (
    <section className={styles.overview}>
      <div className="container">
        <div className="row">
          <div className="col col--6">
            <Heading as="h2">Speed or flexibility is a false choice</Heading>
            <p>
              Libraries built on hand-written CUDA kernels are fast but only support the neuron
              models somebody already wrote a kernel for. Libraries in pure PyTorch or JAX let you
              write any neuron and run considerably slower.
            </p>
            <p>
              jaxpike attacks that from the algorithmic side first. Parallel-in-time execution,
              rematerialization and online learning are all pure JAX, need no kernel code, and
              work on any neuron that fits a three-method contract.
            </p>
            <p>
              Networks are ordinary pytrees, state is explicit and functional, and everything
              composes with <code>jax.jit</code>, <code>jax.grad</code> and <code>jax.vmap</code>.
            </p>
          </div>
          <div className="col col--6">
            <CodeBlock language="python">{QUICKSTART}</CodeBlock>
          </div>
        </div>
      </div>
    </section>
  );
}

const PATHS: {title: string; body: string; to: string; cta: string}[] = [
  {
    title: 'My network will not train',
    body: 'Deep SNNs go silent: activity decays with depth until nothing fires and there is no gradient anywhere. One line fixes it.',
    to: '/docs/guides/silent-networks',
    cta: 'Read the diagnosis',
  },
  {
    title: 'I am porting from snnTorch',
    body: 'Two conventions differ numerically — input normalization and reset timing — and both change results silently rather than raising.',
    to: '/docs/guides/coming-from-snntorch',
    cta: 'Migration guide',
  },
  {
    title: 'I need it faster',
    body: 'Three execution paths with identical signatures, and an honest account of which speedup applies to what.',
    to: '/docs/guides/execution',
    cta: 'Execution guide',
  },
];

function Paths() {
  return (
    <section className={styles.paths}>
      <div className="container">
        <div className="row">
          {PATHS.map((path) => (
            <div className="col col--4" key={path.title}>
              <div className={styles.pathCard}>
                <Heading as="h3">{path.title}</Heading>
                <p>{path.body}</p>
                <Link to={path.to}>{path.cta} →</Link>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  return (
    <Layout
      title="Fast, flexible spiking neural networks in JAX"
      description="jaxpike is a spiking neural network library for JAX: parallel-in-time execution, rematerialized BPTT, e-prop online learning, and NIR export.">
      <HomepageHeader />
      <main>
        <Results />
        <Overview />
        <Paths />
      </main>
    </Layout>
  );
}
