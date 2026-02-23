import { useState, useMemo } from 'react'
import katex from 'katex'
import 'katex/dist/katex.min.css'
import {
  ComposedChart,
  Scatter,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
} from 'recharts'
import { fit, predictProbability, loss, type LogisticModel } from '@/lib/logisticRegression'
import { createSeededRng } from '@/lib/random'
import styles from './MarkovChainSection.module.css'

function renderLatex(latex: string, displayMode = false): string {
  try {
    return katex.renderToString(latex, { displayMode, throwOnError: false })
  } catch {
    return latex
  }
}

const DEFAULT_DATA = `1, 0
2, 0
3, 0
4, 1
5, 0
6, 1
7, 1
8, 1
9, 1
10, 1`

function parseData(text: string): { x: number[]; y: number[] } | null {
  const lines = text.trim().split(/\n/).map((l) => l.trim()).filter(Boolean)
  const x: number[] = []
  const y: number[] = []
  for (const line of lines) {
    const parts = line.split(/[\s,;]+/).map((s) => parseFloat(s.trim()))
    if (parts.length >= 2 && Number.isFinite(parts[0])) {
      const yi = Number(parts[1])
      if (yi !== 0 && yi !== 1) continue
      x.push(parts[0])
      y.push(yi)
    }
  }
  return x.length >= 2 ? { x, y } : null
}

function sigmoid(z: number): number {
  const cap = 20
  const v = Math.max(-cap, Math.min(cap, z))
  return 1 / (1 + Math.exp(-v))
}

export function LogisticRegressionSection() {
  const [dataInput, setDataInput] = useState(DEFAULT_DATA)
  const [fitResult, setFitResult] = useState<LogisticModel | null>(null)
  const [fitLoss, setFitLoss] = useState<number | null>(null)
  const [parseError, setParseError] = useState<string | null>(null)

  const [learningRate, setLearningRate] = useState(0.2)
  const [maxIter, setMaxIter] = useState(2000)

  const [genN, setGenN] = useState(80)
  const [genIntercept, setGenIntercept] = useState(-3)
  const [genSlope, setGenSlope] = useState(0.8)
  const [genXMin, setGenXMin] = useState(0)
  const [genXMax, setGenXMax] = useState(10)
  const [seed, setSeed] = useState('')

  const handleFit = () => {
    setParseError(null)
    const parsed = parseData(dataInput)
    if (!parsed) {
      setParseError('Need at least two rows of x,y. y must be 0 or 1 (comma, space, or semicolon separated).')
      setFitResult(null)
      setFitLoss(null)
      return
    }
    const X = parsed.x.map((xi) => [xi])
    const model = fit(X, parsed.y, { learningRate, maxIter })
    setFitResult(model)
    setFitLoss(loss(X, parsed.y, model))
  }

  const handleGenerate = () => {
    setParseError(null)
    const rand =
      seed.trim() !== '' && !Number.isNaN(Number(seed))
        ? createSeededRng(Number(seed))
        : Math.random
    const x: number[] = []
    const y: number[] = []
    for (let i = 0; i < genN; i++) {
      const xi = genXMin + (genXMax - genXMin) * rand()
      const p = sigmoid(genIntercept + genSlope * xi)
      x.push(xi)
      y.push(rand() < p ? 1 : 0)
    }
    setDataInput(x.map((xi, i) => `${xi.toFixed(4)}, ${y[i]}`).join('\n'))
    const X = x.map((xi) => [xi])
    const model = fit(X, y, { learningRate, maxIter })
    setFitResult(model)
    setFitLoss(loss(X, y, model))
  }

  const chartData = useMemo(() => {
    const parsed = parseData(dataInput)
    if (!parsed || !fitResult) return { points: [], curve: [] }
    const points = parsed.x.map((xi, i) => ({
      x: xi,
      y: parsed.y[i],
      label: parsed.y[i] === 1 ? 'y=1' : 'y=0',
    }))
    const xMin = Math.min(...parsed.x)
    const xMax = Math.max(...parsed.x)
    const padding = (xMax - xMin) * 0.1 || 1
    const curveX = []
    for (let i = 0; i <= 80; i++) {
      curveX.push(xMin - padding + ((xMax + padding) - (xMin - padding)) * (i / 80))
    }
    const curve = curveX.map((xi) => ({
      x: xi,
      p: predictProbability([[xi]], fitResult)[0],
    }))
    return { points, curve }
  }, [dataInput, fitResult])

  return (
    <div className={styles.section}>
      <div className={styles.intro}>
        <p className={styles.introText}>
          <strong>Logistic regression</strong> models P(y=1|x) with{' '}
          <span dangerouslySetInnerHTML={{ __html: renderLatex('P(y=1|x) = \\sigma(\\beta_0 + \\beta_1 x) = \\frac{1}{1+e^{-(\\beta_0+\\beta_1 x)}}') }} />.
          Fit by minimizing binary cross-entropy via gradient descent. Paste x,y data with y in {'{0, 1}'} or generate synthetic data, then click Fit.
        </p>
      </div>

      <div className={styles.editorBlock}>
        <label className={styles.label} htmlFor="logr-data">
          Data (x, y per line — y must be 0 or 1)
        </label>
        <p className={styles.hint}>
          One pair per line; numbers separated by comma, space, or semicolon. Example: <code>2.5, 1</code>
        </p>
        <textarea
          id="logr-data"
          className={styles.textarea}
          value={dataInput}
          onChange={(e) => setDataInput(e.target.value)}
          rows={8}
          spellCheck={false}
        />
        {parseError && <p className={styles.error}>{parseError}</p>}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px', alignItems: 'center' }}>
          <label className={styles.fieldLabel}>
            <span>Learning rate</span>
            <input
              type="number"
              min={0.01}
              max={2}
              step={0.05}
              value={learningRate}
              onChange={(e) => setLearningRate(Number(e.target.value))}
              className={styles.input}
            />
          </label>
          <label className={styles.fieldLabel}>
            <span>Max iterations</span>
            <input
              type="number"
              min={100}
              max={20000}
              step={100}
              value={maxIter}
              onChange={(e) => setMaxIter(Number(e.target.value))}
              className={styles.input}
            />
          </label>
          <button type="button" className={styles.loadBtn} onClick={handleFit}>
            Fit regression
          </button>
        </div>
      </div>

      <div className={styles.editorBlock}>
        <h3 className={styles.optionsTitle}>Generate synthetic data</h3>
        <p className={styles.hint}>
          True model: P(y=1|x) = σ(intercept + slope × x). Each y is sampled from Bernoulli with that probability.
        </p>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
            gap: '12px',
          }}
        >
          <label className={styles.fieldLabel}>
            <span>n</span>
            <input
              type="number"
              min={10}
              max={2000}
              value={genN}
              onChange={(e) => setGenN(Number(e.target.value))}
              className={styles.input}
            />
          </label>
          <label className={styles.fieldLabel}>
            <span>Intercept β₀</span>
            <input
              type="number"
              step={0.5}
              value={genIntercept}
              onChange={(e) => setGenIntercept(Number(e.target.value))}
              className={styles.input}
            />
          </label>
          <label className={styles.fieldLabel}>
            <span>Slope β₁</span>
            <input
              type="number"
              step={0.1}
              value={genSlope}
              onChange={(e) => setGenSlope(Number(e.target.value))}
              className={styles.input}
            />
          </label>
          <label className={styles.fieldLabel}>
            <span>x min</span>
            <input
              type="number"
              step={0.1}
              value={genXMin}
              onChange={(e) => setGenXMin(Number(e.target.value))}
              className={styles.input}
            />
          </label>
          <label className={styles.fieldLabel}>
            <span>x max</span>
            <input
              type="number"
              step={0.1}
              value={genXMax}
              onChange={(e) => setGenXMax(Number(e.target.value))}
              className={styles.input}
            />
          </label>
          <label className={styles.fieldLabel}>
            <span>Seed (optional)</span>
            <input
              type="text"
              placeholder="Random"
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
              className={styles.input}
            />
          </label>
        </div>
        <button type="button" className={styles.runBtn} onClick={handleGenerate}>
          Generate and fit
        </button>
      </div>

      {fitResult && (
        <>
          <div className={styles.graphBlock}>
            <h3 className={styles.graphTitle}>Data and fitted P(y=1|x)</h3>
            <ResponsiveContainer width="100%" height={360}>
              <ComposedChart margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis
                  type="number"
                  dataKey="x"
                  tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                  label={{
                    value: 'x',
                    position: 'insideBottom',
                    offset: -4,
                    fill: 'var(--text-muted)',
                    fontSize: 12,
                  }}
                />
                <YAxis
                  type="number"
                  domain={[-0.1, 1.1]}
                  tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                  label={{
                    value: 'y / P(y=1)',
                    angle: -90,
                    position: 'insideLeft',
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
                  formatter={(value: number) => [Number(value).toFixed(4), '']}
                  labelFormatter={(label) => `x = ${label}`}
                />
                <ReferenceLine y={0.5} stroke="var(--border)" strokeDasharray="2 2" />
                <Legend />
                <Scatter
                  name="y=0"
                  data={chartData.points.filter((p) => p.y === 0)}
                  dataKey="y"
                  fill="var(--text-muted)"
                  fillOpacity={0.8}
                  isAnimationActive={false}
                />
                <Scatter
                  name="y=1"
                  data={chartData.points.filter((p) => p.y === 1)}
                  dataKey="y"
                  fill="var(--accent)"
                  fillOpacity={0.8}
                  isAnimationActive={false}
                />
                <Line
                  name="P(y=1|x)"
                  data={chartData.curve}
                  type="monotone"
                  dataKey="p"
                  xAxisId={0}
                  yAxisId={0}
                  stroke="#22c55e"
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          <div className={styles.matrixBlock}>
            <h4 className={styles.matrixTitle}>Fitted model</h4>
            <div className={styles.matrixWrap}>
              <table className={styles.matrixTable}>
                <tbody>
                  <tr>
                    <th className={styles.matrixRowHeader}>Intercept β₀</th>
                    <td className={styles.matrixCell}>{fitResult.intercept.toFixed(4)}</td>
                  </tr>
                  <tr>
                    <th className={styles.matrixRowHeader}>Slope β₁</th>
                    <td className={styles.matrixCell}>{fitResult.coef[0]?.toFixed(4) ?? '—'}</td>
                  </tr>
                  <tr>
                    <th className={styles.matrixRowHeader}>Binary cross-entropy loss</th>
                    <td className={styles.matrixCell}>{fitLoss != null ? fitLoss.toFixed(4) : '—'}</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className={styles.theoreticalHint} style={{ marginTop: '8px' }}>
              P(y=1|x) = σ({fitResult.intercept.toFixed(3)} + {fitResult.coef[0]?.toFixed(3) ?? '0'} x)
            </p>
          </div>
        </>
      )}
    </div>
  )
}
