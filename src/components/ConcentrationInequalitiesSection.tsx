import { useState, useMemo } from 'react'
import katex from 'katex'
import 'katex/dist/katex.min.css'
import {
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts'
import {
  runCISimulation,
  getMean,
  getVariance,
  VALID_DISTRIBUTIONS,
  type CIDistribution,
  type CIDistributionType,
  type InequalityType,
} from '@/lib/concentrationInequalities'
import { createSeededRng } from '@/lib/random'
import styles from './MarkovChainSection.module.css'

function tex(latex: string, display = false): string {
  try {
    return katex.renderToString(latex, { displayMode: display, throwOnError: false })
  } catch {
    return latex
  }
}

const INEQUALITY_OPTIONS: {
  id: InequalityType
  label: string
  paramLabel: string
  note: string
}[] = [
  {
    id: 'markov',
    label: 'Markov',
    paramLabel: 'a',
    note: 'Requires X ≥ 0. Available for: Bernoulli, Exponential, Uniform (a ≥ 0).',
  },
  {
    id: 'chebyshev',
    label: 'Chebyshev',
    paramLabel: 'k',
    note: 'Applies to any distribution with finite variance.',
  },
  {
    id: 'hoeffding',
    label: 'Hoeffding',
    paramLabel: 't',
    note: 'Requires bounded X ∈ [a, b]. Available for: Bernoulli, Uniform.',
  },
  {
    id: 'subgaussian',
    label: 'Sub-Gaussian',
    paramLabel: 't',
    note: 'Gaussian is exactly σ-sub-Gaussian; Bernoulli and Uniform are also sub-Gaussian.',
  },
]

const DIST_OPTIONS: { id: CIDistributionType; label: string }[] = [
  { id: 'bernoulli', label: 'Bernoulli(p)' },
  { id: 'gaussian', label: 'Gaussian(μ, σ)' },
  { id: 'exponential', label: 'Exponential(λ)' },
  { id: 'uniform', label: 'Uniform(a, b)' },
]

const FORMULA_LATEX: Record<InequalityType, string> = {
  markov:
    '\\Pr(X \\geq a) \\leq \\dfrac{\\mathbb{E}[X]}{a}',
  chebyshev:
    '\\Pr\\bigl(|X - \\mu| \\geq k\\sigma\\bigr) \\leq \\dfrac{1}{k^2}',
  hoeffding:
    '\\Pr\\!\\left(\\bar{X}_n - \\mu \\geq t\\right) \\leq \\exp\\!\\left(\\dfrac{-2nt^2}{(b-a)^2}\\right)',
  subgaussian:
    '\\Pr(X - \\mu \\geq t) \\leq \\exp\\!\\left(\\dfrac{-t^2}{2\\sigma^2}\\right)',
}

const CHART_TITLES: Record<InequalityType, string> = {
  markov: 'Pr(X ≥ a) — Markov bound vs empirical',
  chebyshev: 'Pr(|X − μ| ≥ kσ) — Chebyshev bound vs empirical',
  hoeffding: 'Pr(X̄ₙ − μ ≥ t) — Hoeffding bound vs empirical',
  subgaussian: 'Pr(X − μ ≥ t) — Sub-Gaussian bound vs empirical',
}

function getDefaultDist(type: CIDistributionType): CIDistribution {
  switch (type) {
    case 'bernoulli': return { type: 'bernoulli', p: 0.3 }
    case 'gaussian': return { type: 'gaussian', mean: 0, std: 1 }
    case 'exponential': return { type: 'exponential', lambda: 1 }
    case 'uniform': return { type: 'uniform', a: 0, b: 1 }
  }
}

export function ConcentrationInequalitiesSection() {
  const [inequality, setInequality] = useState<InequalityType>('markov')
  const [distType, setDistType] = useState<CIDistributionType>('bernoulli')
  const [dist, setDist] = useState<CIDistribution>({ type: 'bernoulli', p: 0.3 })
  const [hoeffdingN, setHoeffdingN] = useState(30)
  const [nSamples, setNSamples] = useState(10000)
  const [seed, setSeed] = useState('')
  const [result, setResult] = useState<ReturnType<typeof runCISimulation> | null>(null)
  const [running, setRunning] = useState(false)

  const validDists = VALID_DISTRIBUTIONS[inequality]
  const ineqInfo = INEQUALITY_OPTIONS.find(o => o.id === inequality)!

  function handleInequalityChange(ineq: InequalityType) {
    setInequality(ineq)
    setResult(null)
    const valid = VALID_DISTRIBUTIONS[ineq]
    if (!valid.includes(distType)) {
      const newType = valid[0]
      setDistType(newType)
      setDist(getDefaultDist(newType))
    }
  }

  function handleDistTypeChange(type: CIDistributionType) {
    setDistType(type)
    setDist(getDefaultDist(type))
    setResult(null)
  }

  function handleRun() {
    setRunning(true)
    setResult(null)
    const rand =
      seed.trim() !== '' && !isNaN(Number(seed))
        ? createSeededRng(Number(seed))
        : Math.random
    setTimeout(() => {
      const res = runCISimulation(dist, inequality, nSamples, hoeffdingN, rand)
      setResult(res)
      setRunning(false)
    }, 0)
  }

  const mu = getMean(dist)
  const variance = getVariance(dist)
  const sigma = Math.sqrt(Math.max(variance, 1e-12))

  const chartData = useMemo(() => {
    if (!result) return []
    return result.points.map(p => ({
      param: parseFloat(p.param.toFixed(5)),
      bound: parseFloat(p.bound.toFixed(5)),
      empirical: parseFloat(p.empirical.toFixed(5)),
    }))
  }, [result])

  return (
    <div className={styles.section}>
      {/* Intro */}
      <div className={styles.intro}>
        <p className={styles.introText}>
          <strong>Concentration inequalities</strong> bound the probability that a
          random variable deviates far from its expected value. They are fundamental
          tools in probability and statistics: the <strong>Markov</strong> inequality
          uses only the mean; <strong>Chebyshev</strong> also uses variance;{' '}
          <strong>Hoeffding</strong> exploits boundedness for exponentially tight bounds;
          and <strong>sub-Gaussian</strong> tail bounds generalise to distributions with
          Gaussian-like tails. Run the simulation to see the bound (orange) always stay
          above the empirical probability (blue).
        </p>
      </div>

      {/* Controls */}
      <div className={styles.editorBlock}>
        <h3 className={styles.optionsTitle}>Inequality &amp; Distribution</h3>
        <div className={styles.theoreticalForm}>
          <label className={styles.fieldLabel}>
            <span>Inequality</span>
            <select
              className={styles.input}
              value={inequality}
              onChange={e => handleInequalityChange(e.target.value as InequalityType)}
            >
              {INEQUALITY_OPTIONS.map(o => (
                <option key={o.id} value={o.id}>
                  {o.label}
                </option>
              ))}
            </select>
          </label>

          <label className={styles.fieldLabel}>
            <span>Distribution</span>
            <select
              className={styles.input}
              value={distType}
              onChange={e => handleDistTypeChange(e.target.value as CIDistributionType)}
            >
              {DIST_OPTIONS.filter(o => validDists.includes(o.id)).map(o => (
                <option key={o.id} value={o.id}>
                  {o.label}
                </option>
              ))}
            </select>
          </label>

          {dist.type === 'bernoulli' && (
            <label className={styles.fieldLabel}>
              <span>p</span>
              <input
                type="number"
                min={0.01}
                max={0.99}
                step={0.01}
                value={dist.p}
                onChange={e =>
                  setDist({
                    ...dist,
                    p: Math.max(0.01, Math.min(0.99, Number(e.target.value) || 0.01)),
                  })
                }
                className={styles.input}
                style={{ width: 80 }}
              />
            </label>
          )}

          {dist.type === 'gaussian' && (
            <>
              <label className={styles.fieldLabel}>
                <span>μ (mean)</span>
                <input
                  type="number"
                  value={dist.mean}
                  onChange={e => setDist({ ...dist, mean: Number(e.target.value) || 0 })}
                  className={styles.input}
                  style={{ width: 80 }}
                />
              </label>
              <label className={styles.fieldLabel}>
                <span>σ (std)</span>
                <input
                  type="number"
                  min={0.01}
                  value={dist.std}
                  onChange={e =>
                    setDist({
                      ...dist,
                      std: Math.max(0.01, Number(e.target.value) || 0.01),
                    })
                  }
                  className={styles.input}
                  style={{ width: 80 }}
                />
              </label>
            </>
          )}

          {dist.type === 'exponential' && (
            <label className={styles.fieldLabel}>
              <span>λ</span>
              <input
                type="number"
                min={0.01}
                value={dist.lambda}
                onChange={e =>
                  setDist({
                    ...dist,
                    lambda: Math.max(0.01, Number(e.target.value) || 0.01),
                  })
                }
                className={styles.input}
                style={{ width: 80 }}
              />
            </label>
          )}

          {dist.type === 'uniform' && (
            <>
              <label className={styles.fieldLabel}>
                <span>a (min)</span>
                <input
                  type="number"
                  value={dist.a}
                  onChange={e => {
                    const a = Number(e.target.value) || 0
                    setDist({ ...dist, a, b: Math.max(a + 0.01, dist.b) })
                  }}
                  className={styles.input}
                  style={{ width: 80 }}
                />
              </label>
              <label className={styles.fieldLabel}>
                <span>b (max)</span>
                <input
                  type="number"
                  value={dist.b}
                  onChange={e => {
                    const b = Number(e.target.value) || 1
                    setDist({ ...dist, b, a: Math.min(dist.a, b - 0.01) })
                  }}
                  className={styles.input}
                  style={{ width: 80 }}
                />
              </label>
            </>
          )}
        </div>
        <p className={styles.hint} style={{ marginTop: '0.25rem' }}>
          {ineqInfo.note}
        </p>

        {inequality === 'hoeffding' && (
          <div style={{ marginTop: '0.75rem' }}>
            <label className={styles.fieldLabel}>
              <span>
                n (samples per batch for{' '}
                <span dangerouslySetInnerHTML={{ __html: tex('\\bar{X}_n') }} />)
              </span>
              <input
                type="number"
                min={2}
                max={500}
                value={hoeffdingN}
                onChange={e =>
                  setHoeffdingN(Math.max(2, Math.min(500, Number(e.target.value) || 2)))
                }
                className={styles.input}
                style={{ width: 100 }}
              />
            </label>
          </div>
        )}

        <h3 className={styles.optionsTitle} style={{ marginTop: '1rem' }}>
          Simulation
        </h3>
        <div className={styles.theoreticalForm}>
          <label className={styles.fieldLabel}>
            <span>{inequality === 'hoeffding' ? 'Trials' : 'Samples'}</span>
            <input
              type="number"
              min={1000}
              max={100000}
              value={nSamples}
              onChange={e =>
                setNSamples(
                  Math.max(1000, Math.min(100000, Number(e.target.value) || 1000))
                )
              }
              className={styles.input}
              style={{ width: 100 }}
            />
          </label>
          <label className={styles.fieldLabel}>
            <span>Seed (optional)</span>
            <input
              type="text"
              value={seed}
              onChange={e => setSeed(e.target.value)}
              placeholder="e.g. 42"
              className={styles.input}
              style={{ width: 100 }}
            />
          </label>
          <button
            type="button"
            className={styles.runBtn}
            onClick={handleRun}
            disabled={running}
          >
            {running ? 'Running…' : 'Run simulation'}
          </button>
        </div>
      </div>

      {/* Formula & description */}
      <div className={styles.matrixBlock}>
        <h3 className={styles.matrixTitle}>{ineqInfo.label} inequality</h3>
        <div
          className={styles.introFormula}
          dangerouslySetInnerHTML={{ __html: tex(FORMULA_LATEX[inequality], true) }}
          style={{ margin: '0.75rem 0' }}
        />
        <p className={styles.hint}>
          {inequality === 'markov' && (
            <>
              For any non-negative random variable{' '}
              <span dangerouslySetInnerHTML={{ __html: tex('X \\geq 0') }} /> and
              any threshold{' '}
              <span dangerouslySetInnerHTML={{ __html: tex('a > 0') }} />: the tail
              probability is at most the ratio of the mean to the threshold. Only
              the mean is needed — the bound is weak but universal.
            </>
          )}
          {inequality === 'chebyshev' && (
            <>
              For any r.v. with finite mean{' '}
              <span dangerouslySetInnerHTML={{ __html: tex('\\mu') }} /> and variance{' '}
              <span dangerouslySetInnerHTML={{ __html: tex('\\sigma^2') }} />, the
              probability of deviating more than{' '}
              <span dangerouslySetInnerHTML={{ __html: tex('k\\sigma') }} /> from the
              mean is at most{' '}
              <span dangerouslySetInnerHTML={{ __html: tex('1/k^2') }} />. The bound
              is tight for distributions that concentrate mass at{' '}
              <span dangerouslySetInnerHTML={{ __html: tex('\\pm k\\sigma') }} />.
            </>
          )}
          {inequality === 'hoeffding' && (
            <>
              For{' '}
              <span dangerouslySetInnerHTML={{ __html: tex('n') }} /> i.i.d. bounded
              r.v.s{' '}
              <span dangerouslySetInnerHTML={{ __html: tex('X_i \\in [a, b]') }} />,
              the sample mean concentrates exponentially around{' '}
              <span dangerouslySetInnerHTML={{ __html: tex('\\mu') }} />. Each
              simulated trial draws{' '}
              <span dangerouslySetInnerHTML={{ __html: tex('n') }} /> samples and
              checks whether{' '}
              <span dangerouslySetInnerHTML={{ __html: tex('\\bar{X}_n - \\mu \\geq t') }} />.
            </>
          )}
          {inequality === 'subgaussian' && (
            <>
              A random variable is{' '}
              <span dangerouslySetInnerHTML={{ __html: tex('\\sigma') }} />-sub-Gaussian if{' '}
              <span
                dangerouslySetInnerHTML={{
                  __html: tex('\\mathbb{E}[e^{\\lambda(X-\\mu)}] \\leq e^{\\lambda^2\\sigma^2/2}'),
                }}
              />{' '}
              for all{' '}
              <span dangerouslySetInnerHTML={{ __html: tex('\\lambda \\in \\mathbb{R}') }} />.
              This yields Gaussian-like one-sided tail bounds. The Gaussian is exactly
              sub-Gaussian; bounded distributions are sub-Gaussian by Hoeffding's lemma.
            </>
          )}
        </p>
        <p className={styles.hint} style={{ marginTop: '0.5rem' }}>
          <span dangerouslySetInnerHTML={{ __html: tex('\\mu') }} /> ={' '}
          <strong>{mu.toFixed(4)}</strong>
          {'  '}
          <span dangerouslySetInnerHTML={{ __html: tex('\\sigma^2') }} /> ={' '}
          <strong>{variance.toFixed(4)}</strong>
          {'  '}
          <span dangerouslySetInnerHTML={{ __html: tex('\\sigma') }} /> ={' '}
          <strong>{sigma.toFixed(4)}</strong>
        </p>
      </div>

      {/* Chart */}
      {result && chartData.length > 0 && (
        <div className={styles.graphBlock}>
          <h3 className={styles.graphTitle}>{CHART_TITLES[inequality]}</h3>
          <p className={styles.matrixHint}>
            {nSamples.toLocaleString()}
            {inequality === 'hoeffding'
              ? ` trials (n = ${hoeffdingN} per trial).`
              : ' samples.'}{' '}
            The bound (orange dashed) must always lie above the empirical proportion (blue).
          </p>
          <ResponsiveContainer width="100%" height={360}>
            <ComposedChart
              data={chartData}
              margin={{ top: 8, right: 16, bottom: 28, left: 8 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis
                dataKey="param"
                type="number"
                domain={['auto', 'auto']}
                tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                tickFormatter={v => Number(v).toFixed(3)}
                label={{
                  value: ineqInfo.paramLabel,
                  position: 'insideBottom',
                  offset: -12,
                  fill: 'var(--text-muted)',
                  fontSize: 13,
                }}
              />
              <YAxis
                type="number"
                domain={[0, 1]}
                tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                tickFormatter={v => Number(v).toFixed(2)}
                label={{
                  value: 'Probability',
                  angle: -90,
                  position: 'insideLeft',
                  offset: 12,
                  fill: 'var(--text-muted)',
                  fontSize: 12,
                }}
              />
              <Tooltip
                contentStyle={{
                  background: 'var(--bg-elevated)',
                  border: '1px solid var(--border)',
                  borderRadius: '8px',
                }}
                formatter={(value: number, name: string) => [value.toFixed(4), name]}
                labelFormatter={v =>
                  `${ineqInfo.paramLabel} = ${Number(v).toFixed(4)}`
                }
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="bound"
                name="Bound"
                stroke="#f97316"
                strokeWidth={2}
                strokeDasharray="6 3"
                dot={false}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="empirical"
                name="Empirical"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}

      {result && result.points.length === 0 && (
        <p className={styles.hint} style={{ color: 'var(--danger)' }}>
          Cannot compute: distribution parameters are invalid for this inequality (e.g.
          mean ≤ 0 for Markov, or range = 0 for Hoeffding).
        </p>
      )}

      {!result && !running && (
        <p
          className={styles.hint}
          style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}
        >
          Select an inequality and distribution, then click{' '}
          <strong>Run simulation</strong>.
        </p>
      )}
    </div>
  )
}
