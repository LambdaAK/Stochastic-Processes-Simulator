import { useState, useMemo } from 'react'
import katex from 'katex'
import 'katex/dist/katex.min.css'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'
import { runLLNSimulation, theoreticalMean } from '@/lib/lln'
import { createSeededRng } from '@/lib/random'
import type { LLNDistribution, LLNResult } from '@/types/lln'
import styles from './MarkovChainSection.module.css'

function renderLatex(latex: string, displayMode = false): string {
  try {
    return katex.renderToString(latex, { displayMode, throwOnError: false })
  } catch {
    return latex
  }
}

const DISTRIBUTION_OPTIONS: { id: LLNDistribution['type']; label: string }[] = [
  { id: 'bernoulli', label: 'Bernoulli(p)' },
  { id: 'gaussian', label: 'Gaussian(μ, σ)' },
  { id: 'uniform', label: 'Uniform(a, b)' },
  { id: 'exponential', label: 'Exponential(λ)' },
  { id: 'poisson', label: 'Poisson(λ)' },
  { id: 'beta', label: 'Beta(α, β)' },
]

function getDefaultDistribution(type: LLNDistribution['type']): LLNDistribution {
  switch (type) {
    case 'bernoulli':
      return { type: 'bernoulli', p: 0.5 }
    case 'gaussian':
      return { type: 'gaussian', mean: 0, std: 1 }
    case 'uniform':
      return { type: 'uniform', min: 0, max: 1 }
    case 'exponential':
      return { type: 'exponential', lambda: 1 }
    case 'poisson':
      return { type: 'poisson', lambda: 3 }
    case 'beta':
      return { type: 'beta', alpha: 2, beta: 5 }
    default:
      return { type: 'bernoulli', p: 0.5 }
  }
}

export function LLNSection() {
  const [distType, setDistType] = useState<LLNDistribution['type']>('bernoulli')
  const [distribution, setDistribution] = useState<LLNDistribution>(() =>
    getDefaultDistribution('bernoulli')
  )
  const [numSamples, setNumSamples] = useState(5000)
  const [seed, setSeed] = useState('')
  const [result, setResult] = useState<LLNResult | null>(null)
  const [running, setRunning] = useState(false)

  const handleDistTypeChange = (type: LLNDistribution['type']) => {
    setDistType(type)
    setDistribution(getDefaultDistribution(type))
  }

  const handleRun = () => {
    setRunning(true)
    setResult(null)
    const rand =
      seed.trim() !== '' && !Number.isNaN(Number(seed))
        ? createSeededRng(Number(seed))
        : Math.random
    setTimeout(() => {
      const res = runLLNSimulation(
        { distribution, numSamples },
        rand
      )
      setResult(res)
      setRunning(false)
    }, 0)
  }

  const chartData = useMemo(() => {
    if (!result) return []
    return result.n.map((n, i) => ({
      n,
      runningMean: result.runningMean[i],
    }))
  }, [result])

  const meanLine = result?.theoreticalMean ?? theoreticalMean(distribution)

  return (
    <div className={styles.section}>
      <div className={styles.intro}>
        <p className={styles.introText}>
          The <strong>law of large numbers</strong> says that the sample average converges to the expected value. For i.i.d. <span dangerouslySetInnerHTML={{ __html: renderLatex('X_1, X_2, \\ldots') }} /> with <span dangerouslySetInnerHTML={{ __html: renderLatex('\\mathbb{E}[X_1] = \\mu') }} />, the running average satisfies
        </p>
        <p className={styles.introFormula} dangerouslySetInnerHTML={{ __html: renderLatex('\\bar{X}_n = \\frac{1}{n}\\sum_{i=1}^n X_i \\,\\to\\, \\mu \\quad \\text{as } n \\to \\infty', true) }} />
        <p className={styles.introText}>
          (in probability or almost surely). Choose a distribution, set the number of samples, and run to see the running mean converge to the theoretical mean.
        </p>
      </div>

      <div className={styles.editorBlock}>
        <h3 className={styles.optionsTitle}>Distribution</h3>
        <div className={styles.theoreticalForm}>
          <label className={styles.fieldLabel}>
            <span>Type</span>
            <select
              className={styles.input}
              value={distType}
              onChange={(e) => handleDistTypeChange(e.target.value as LLNDistribution['type'])}
            >
              {DISTRIBUTION_OPTIONS.map((opt) => (
                <option key={opt.id} value={opt.id}>
                  {opt.label}
                </option>
              ))}
            </select>
          </label>
          {distribution.type === 'bernoulli' && (
            <label className={styles.fieldLabel}>
              <span>p (probability of 1)</span>
              <input
                type="number"
                min={0}
                max={1}
                step={0.01}
                value={distribution.p}
                onChange={(e) =>
                  setDistribution({ ...distribution, p: Number(e.target.value) || 0 })
                }
                className={styles.input}
              />
            </label>
          )}
          {distribution.type === 'gaussian' && (
            <>
              <label className={styles.fieldLabel}>
                <span>μ (mean)</span>
                <input
                  type="number"
                  value={distribution.mean}
                  onChange={(e) =>
                    setDistribution({ ...distribution, mean: Number(e.target.value) || 0 })
                  }
                  className={styles.input}
                />
              </label>
              <label className={styles.fieldLabel}>
                <span>σ (std)</span>
                <input
                  type="number"
                  min={0.01}
                  value={distribution.std}
                  onChange={(e) =>
                    setDistribution({ ...distribution, std: Number(e.target.value) || 0.01 })
                  }
                  className={styles.input}
                />
              </label>
            </>
          )}
          {distribution.type === 'uniform' && (
            <>
              <label className={styles.fieldLabel}>
                <span>a (min)</span>
                <input
                  type="number"
                  value={distribution.min}
                  onChange={(e) =>
                    setDistribution({ ...distribution, min: Number(e.target.value) || 0 })
                  }
                  className={styles.input}
                />
              </label>
              <label className={styles.fieldLabel}>
                <span>b (max)</span>
                <input
                  type="number"
                  value={distribution.max}
                  onChange={(e) =>
                    setDistribution({ ...distribution, max: Number(e.target.value) || 1 })
                  }
                  className={styles.input}
                />
              </label>
            </>
          )}
          {distribution.type === 'exponential' && (
            <label className={styles.fieldLabel}>
              <span>λ (rate)</span>
              <input
                type="number"
                min={0.01}
                step={0.1}
                value={distribution.lambda}
                onChange={(e) =>
                  setDistribution({ ...distribution, lambda: Number(e.target.value) || 0.01 })
                }
                className={styles.input}
              />
            </label>
          )}
          {distribution.type === 'poisson' && (
            <label className={styles.fieldLabel}>
              <span>λ (mean)</span>
              <input
                type="number"
                min={0.01}
                step={0.1}
                value={distribution.lambda}
                onChange={(e) =>
                  setDistribution({ ...distribution, lambda: Number(e.target.value) || 0.01 })
                }
                className={styles.input}
              />
            </label>
          )}
          {distribution.type === 'beta' && (
            <>
              <label className={styles.fieldLabel}>
                <span>α</span>
                <input
                  type="number"
                  min={1}
                  value={distribution.alpha}
                  onChange={(e) =>
                    setDistribution({ ...distribution, alpha: Math.max(1, Number(e.target.value) || 1) })
                  }
                  className={styles.input}
                />
              </label>
              <label className={styles.fieldLabel}>
                <span>β</span>
                <input
                  type="number"
                  min={1}
                  value={distribution.beta}
                  onChange={(e) =>
                    setDistribution({ ...distribution, beta: Math.max(1, Number(e.target.value) || 1) })
                  }
                  className={styles.input}
                />
              </label>
            </>
          )}
        </div>
        <p className={styles.theoreticalHint}>
          Theoretical mean <span dangerouslySetInnerHTML={{ __html: renderLatex('\\mathbb{E}[X]') }} /> = {meanLine.toFixed(4)}
        </p>
        <div className={styles.theoreticalForm}>
          <label className={styles.fieldLabel}>
            <span>Number of samples (n)</span>
            <input
              type="number"
              min={100}
              max={100000}
              value={numSamples}
              onChange={(e) => setNumSamples(Math.max(100, Number(e.target.value) || 100))}
              className={styles.input}
            />
          </label>
          <label className={styles.fieldLabel}>
            <span>Seed (optional)</span>
            <input
              type="text"
              placeholder="Leave empty for random"
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
              className={styles.input}
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

      {result && chartData.length > 0 && (
        <div className={styles.graphBlock}>
          <h3 className={styles.graphTitle}>Running mean vs n</h3>
          <p className={styles.matrixHint}>
            <span dangerouslySetInnerHTML={{ __html: renderLatex('\\bar{X}_n') }} /> (blue) and <span dangerouslySetInnerHTML={{ __html: renderLatex('\\mathbb{E}[X]') }} /> (dashed)
          </p>
          <ResponsiveContainer width="100%" height={360}>
            <LineChart data={chartData} margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis
                dataKey="n"
                type="number"
                tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                label={{
                  value: 'n (sample size)',
                  position: 'insideBottom',
                  offset: -4,
                  fill: 'var(--text-muted)',
                  fontSize: 12,
                }}
              />
              <YAxis
                type="number"
                tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                tickFormatter={(v) => Number(v).toFixed(3)}
                label={{
                  value: 'Running mean',
                  angle: -90,
                  position: 'insideLeft',
                  fill: 'var(--text-muted)',
                  fontSize: 12,
                }}
              />
              <ReferenceLine
                y={result.theoreticalMean}
                stroke="var(--text-muted)"
                strokeDasharray="4 2"
                strokeWidth={1.5}
              />
              <Tooltip
                contentStyle={{
                  background: 'var(--bg-elevated)',
                  border: '1px solid var(--border)',
                  borderRadius: '8px',
                }}
                formatter={(value: number) => [Number(value).toFixed(4), 'Running mean']}
                labelFormatter={(n) => `n = ${n}`}
              />
              <Line
                type="monotone"
                dataKey="runningMean"
                name="X̄ₙ"
                stroke="var(--accent)"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}
