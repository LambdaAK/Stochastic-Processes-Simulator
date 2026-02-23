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
} from 'recharts'
import {
  fitLinearRegression,
  generateLinearData,
  type LinearRegressionResult,
} from '@/lib/linearRegression'
import { createSeededRng } from '@/lib/random'
import styles from './MarkovChainSection.module.css'

function renderLatex(latex: string, displayMode = false): string {
  try {
    return katex.renderToString(latex, { displayMode, throwOnError: false })
  } catch {
    return latex
  }
}

const DEFAULT_DATA = `1, 2.1
2, 3.9
3, 5.8
4, 8.2
5, 9.7
6, 11.5
7, 14.1
8, 15.8
9, 17.2
10, 19.5`

function parseData(text: string): { x: number[]; y: number[] } | null {
  const lines = text.trim().split(/\n/).map((l) => l.trim()).filter(Boolean)
  const x: number[] = []
  const y: number[] = []
  for (const line of lines) {
    const parts = line.split(/[\s,;]+/).map((s) => parseFloat(s.trim()))
    if (parts.length >= 2 && Number.isFinite(parts[0]) && Number.isFinite(parts[1])) {
      x.push(parts[0])
      y.push(parts[1])
    }
  }
  return x.length >= 2 ? { x, y } : null
}

export function LinearRegressionSection() {
  const [dataInput, setDataInput] = useState(DEFAULT_DATA)
  const [fitResult, setFitResult] = useState<LinearRegressionResult | null>(null)
  const [parseError, setParseError] = useState<string | null>(null)

  // Synthetic data controls
  const [genN, setGenN] = useState(50)
  const [genIntercept, setGenIntercept] = useState(1)
  const [genSlope, setGenSlope] = useState(2)
  const [genXMin, setGenXMin] = useState(0)
  const [genXMax, setGenXMax] = useState(10)
  const [genNoise, setGenNoise] = useState(1)
  const [seed, setSeed] = useState('')

  const handleFit = () => {
    setParseError(null)
    const parsed = parseData(dataInput)
    if (!parsed) {
      setParseError('Need at least two rows of x,y (numbers separated by comma, space, or semicolon).')
      setFitResult(null)
      return
    }
    const result = fitLinearRegression(parsed.x, parsed.y)
    setFitResult(result)
  }

  const handleGenerate = () => {
    setParseError(null)
    const rand =
      seed.trim() !== '' && !Number.isNaN(Number(seed))
        ? createSeededRng(Number(seed))
        : Math.random
    const { x, y } = generateLinearData(
      genN,
      genIntercept,
      genSlope,
      genXMin,
      genXMax,
      genNoise,
      rand
    )
    setDataInput(x.map((xi, i) => `${xi.toFixed(4)}, ${y[i].toFixed(4)}`).join('\n'))
    const result = fitLinearRegression(x, y)
    setFitResult(result)
  }

  const chartData = useMemo(() => {
    if (!fitResult) return []
    const parsed = parseData(dataInput)
    if (!parsed) return []
    const points = parsed.x.map((xi, i) => ({
      x: xi,
      y: parsed.y[i],
      fitted: fitResult.fitted[i],
    }))
    points.sort((a, b) => a.x - b.x)
    return points
  }, [fitResult, dataInput])

  return (
    <div className={styles.section}>
      <div className={styles.intro}>
        <p className={styles.introText}>
          <strong>Ordinary least squares (OLS)</strong> fits a line{' '}
          <span dangerouslySetInnerHTML={{ __html: renderLatex('y = \\beta_0 + \\beta_1 x') }} /> by minimizing the sum of squared residuals{' '}
          <span dangerouslySetInnerHTML={{ __html: renderLatex('\\sum_i (y_i - \\hat{y}_i)^2') }} />. The closed-form solution is{' '}
          <span dangerouslySetInnerHTML={{ __html: renderLatex('\\beta = (X\'X)^{-1} X\'y') }} /> with design matrix{' '}
          <span dangerouslySetInnerHTML={{ __html: renderLatex('X = [\\mathbf{1}, \\, x]') }} />. Paste x,y data below or generate synthetic data, then click Fit.
        </p>
      </div>

      <div className={styles.editorBlock}>
        <label className={styles.label} htmlFor="lr-data">
          Data (x, y per line)
        </label>
        <p className={styles.hint}>
          One pair per line; numbers separated by comma, space, or semicolon. Example: <code>1, 2.5</code> or <code>1  2.5</code>
        </p>
        <textarea
          id="lr-data"
          className={styles.textarea}
          value={dataInput}
          onChange={(e) => setDataInput(e.target.value)}
          rows={8}
          spellCheck={false}
        />
        {parseError && <p className={styles.error}>{parseError}</p>}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px', alignItems: 'center' }}>
          <button type="button" className={styles.loadBtn} onClick={handleFit}>
            Fit regression
          </button>
        </div>
      </div>

      <div className={styles.editorBlock}>
        <h3 className={styles.optionsTitle}>Generate synthetic data</h3>
        <p className={styles.hint}>
          y = intercept + slope × x + N(0, noise). Generated data replaces the textarea and is fitted automatically.
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
              min={5}
              max={2000}
              value={genN}
              onChange={(e) => setGenN(Number(e.target.value))}
              className={styles.input}
            />
          </label>
          <label className={styles.fieldLabel}>
            <span>Intercept</span>
            <input
              type="number"
              step={0.1}
              value={genIntercept}
              onChange={(e) => setGenIntercept(Number(e.target.value))}
              className={styles.input}
            />
          </label>
          <label className={styles.fieldLabel}>
            <span>Slope</span>
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
            <span>Noise σ</span>
            <input
              type="number"
              min={0}
              step={0.1}
              value={genNoise}
              onChange={(e) => setGenNoise(Number(e.target.value))}
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
            <h3 className={styles.graphTitle}>Scatter and fitted line</h3>
            <ResponsiveContainer width="100%" height={360}>
              <ComposedChart data={chartData} margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
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
                  tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                  label={{
                    value: 'y',
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
                <Legend />
                <Scatter
                  name="Data"
                  dataKey="y"
                  fill="var(--accent)"
                  fillOpacity={0.8}
                  isAnimationActive={false}
                />
                <Line
                  name="ŷ = β₀ + β₁x"
                  type="monotone"
                  dataKey="fitted"
                  stroke="#22c55e"
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          <div className={styles.matrixBlock}>
            <h4 className={styles.matrixTitle}>OLS results</h4>
            <div className={styles.matrixWrap}>
              <table className={styles.matrixTable}>
                <tbody>
                  <tr>
                    <th className={styles.matrixRowHeader}>Intercept β₀</th>
                    <td className={styles.matrixCell}>{fitResult.intercept.toFixed(4)}</td>
                  </tr>
                  <tr>
                    <th className={styles.matrixRowHeader}>Slope β₁</th>
                    <td className={styles.matrixCell}>{fitResult.slope.toFixed(4)}</td>
                  </tr>
                  <tr>
                    <th className={styles.matrixRowHeader}>R²</th>
                    <td className={styles.matrixCell}>{fitResult.rSquared.toFixed(4)}</td>
                  </tr>
                  <tr>
                    <th className={styles.matrixRowHeader}>Residual SE (σ̂)</th>
                    <td className={styles.matrixCell}>{fitResult.sigma.toFixed(4)}</td>
                  </tr>
                  <tr>
                    <th className={styles.matrixRowHeader}>n</th>
                    <td className={styles.matrixCell}>{fitResult.n}</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className={styles.theoreticalHint} style={{ marginTop: '8px' }}>
              Equation: ŷ = {fitResult.intercept.toFixed(3)} + {fitResult.slope.toFixed(3)} x
            </p>
          </div>
        </>
      )}
    </div>
  )
}
