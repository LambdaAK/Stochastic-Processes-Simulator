import { useState, useMemo } from 'react'
import katex from 'katex'
import 'katex/dist/katex.min.css'
import {
  ComposedChart,
  Scatter,
  Line,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  Cell,
} from 'recharts'
import {
  trainSVM,
  marginLine,
  generateDataset,
  predict,
  type DataPoint,
  type SVMResult,
} from '@/lib/svm'
import { createSeededRng } from '@/lib/random'
import styles from './MarkovChainSection.module.css'

function renderLatex(latex: string, displayMode = false): string {
  try {
    return katex.renderToString(latex, { displayMode, throwOnError: false })
  } catch {
    return latex
  }
}

type Preset = 'blobs' | 'wide-margin' | 'near-boundary' | 'overlapping'

const PRESETS: { id: Preset; label: string }[] = [
  { id: 'blobs', label: 'Two blobs' },
  { id: 'wide-margin', label: 'Wide margin' },
  { id: 'near-boundary', label: 'Near boundary' },
  { id: 'overlapping', label: 'Overlapping' },
]

const DEFAULT_CUSTOM = `# x1, x2, label  (+1 or -1)
 2.1,  1.5, +1
 1.3,  2.2, +1
 2.5,  0.8, +1
 1.9,  2.8, +1
-1.8, -1.2, -1
-2.3, -0.5, -1
-1.0, -2.5, -1
-2.8, -1.9, -1`

function parseCustomData(text: string): { pts: DataPoint[] } | { error: string } {
  const rows: { x: number; y: number; raw: string }[] = []
  for (const line of text.split('\n')) {
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith('#') || trimmed.startsWith('//')) continue
    const parts = trimmed.split(/[\s,;]+/).filter(Boolean)
    if (parts.length < 3) continue
    const x = parseFloat(parts[0])
    const y = parseFloat(parts[1])
    if (!isFinite(x) || !isFinite(y)) continue
    rows.push({ x, y, raw: parts[2] })
  }
  if (rows.length < 4) return { error: 'Need at least 4 valid rows (x1, x2, label).' }

  const seen: string[] = []
  for (const r of rows) if (!seen.includes(r.raw)) seen.push(r.raw)
  if (seen.length < 2) return { error: 'Need exactly 2 distinct labels.' }
  if (seen.length > 2)
    return { error: `Found ${seen.length} distinct labels — SVM is binary only.` }

  const [la, lb] = seen
  const aIs1 =
    ['+1', '1'].includes(la) && ['-1', '0'].includes(lb)
      ? true
      : ['-1', '0'].includes(la) && ['+1', '1'].includes(lb)
        ? false
        : true

  const pts: DataPoint[] = rows.map((r) => ({
    x: r.x,
    y: r.y,
    label: (r.raw === la ? (aIs1 ? 1 : -1) : aIs1 ? -1 : 1) as 1 | -1,
  }))
  return { pts }
}

const DOMAIN = { xMin: -4, xMax: 4, yMin: -4, yMax: 4 }
const COLOR_POS = 'var(--accent)'
const COLOR_NEG = '#0ea5e9'
const COLOR_BOUNDARY = '#f97316'
const COLOR_POS_MARGIN = '#fb923c'
const COLOR_NEG_MARGIN = '#38bdf8'

type IndexedPoint = DataPoint & { _idx: number }

export function SVMSection() {
  const [preset, setPreset] = useState<Preset>('blobs')
  const [nPerClass, setNPerClass] = useState(30)
  const [marginType, setMarginType] = useState<'hard' | 'soft'>('hard')
  const [C, setC] = useState(1.0)
  const [seed, setSeed] = useState('')
  const [data, setData] = useState<DataPoint[]>([])
  const [result, setResult] = useState<SVMResult | null>(null)
  const [trainError, setTrainError] = useState<string | null>(null)

  const [customInput, setCustomInput] = useState(DEFAULT_CUSTOM)
  const [customError, setCustomError] = useState<string | null>(null)

  const runWithData = (pts: DataPoint[]) => {
    setTrainError(null)
    try {
      const cValue = marginType === 'hard' ? Infinity : C
      const res = trainSVM(pts, cValue)
      setData(pts)
      setResult(res)
    } catch (e) {
      setTrainError(e instanceof Error ? e.message : 'Training failed.')
    }
  }

  const handleRun = () => {
    const rand =
      seed.trim() !== '' && !Number.isNaN(Number(seed))
        ? createSeededRng(Number(seed))
        : Math.random
    runWithData(generateDataset(preset, nPerClass, rand))
  }

  const handleCustomTrain = () => {
    setCustomError(null)
    const res = parseCustomData(customInput)
    if ('error' in res) {
      setCustomError(res.error)
      return
    }
    runWithData(res.pts)
  }

  // Partition data, tagging each point with its original index
  const { pos, neg, svSet } = useMemo(() => {
    const pos: IndexedPoint[] = []
    const neg: IndexedPoint[] = []
    data.forEach((p, i) => {
      if (p.label === 1) pos.push({ ...p, _idx: i })
      else neg.push({ ...p, _idx: i })
    })
    const svSet = new Set(result?.model.supportVectorIndices ?? [])
    return { pos, neg, svSet }
  }, [data, result])

  // Clipped line endpoints for boundary and the two margin planes
  const { boundary, posMarginPts, negMarginPts } = useMemo(() => {
    if (!result)
      return { boundary: [], posMarginPts: [], negMarginPts: [] }
    const m = result.model
    const { xMin, xMax, yMin, yMax } = DOMAIN
    return {
      boundary: marginLine(m, 0, xMin, xMax, yMin, yMax),
      posMarginPts: marginLine(m, 1, xMin, xMax, yMin, yMax),
      negMarginPts: marginLine(m, -1, xMin, xMax, yMin, yMax),
    }
  }, [result])

  const accuracy = useMemo(() => {
    if (!result || data.length === 0) return null
    const m = result.model
    const correct = data.filter((p) => predict(p.x, p.y, m) === p.label).length
    return correct / data.length
  }, [result, data])

  const wNorm = result
    ? Math.sqrt(result.model.w1 ** 2 + result.model.w2 ** 2)
    : null

  return (
    <div className={styles.section}>
      {/* Intro */}
      <div className={styles.intro}>
        <p className={styles.introText}>
          <strong>Support Vector Machine (SVM)</strong> finds the maximum-margin separating
          hyperplane. The{' '}
          <span dangerouslySetInnerHTML={{ __html: renderLatex('C') }} />
          -SVM primal is:
        </p>
        <p
          className={styles.introFormula}
          dangerouslySetInnerHTML={{
            __html: renderLatex(
              '\\min_{\\mathbf{w},b,\\boldsymbol{\\xi}}\\;\\tfrac{1}{2}\\|\\mathbf{w}\\|^2 + C\\sum_i \\xi_i \\quad \\text{s.t.}\\; y_i(\\mathbf{w}\\cdot\\mathbf{x}_i+b)\\geq 1-\\xi_i,\\;\\xi_i\\geq 0',
              true,
            ),
          }}
        />
        <p className={styles.introText}>
          <strong>Hard margin</strong> (
          <span dangerouslySetInnerHTML={{ __html: renderLatex('C=\\infty') }} />) requires
          perfect linear separation. <strong>Soft margin</strong> (finite{' '}
          <span dangerouslySetInnerHTML={{ __html: renderLatex('C') }} />) allows a few
          violations. The geometric margin is{' '}
          <span dangerouslySetInnerHTML={{ __html: renderLatex('2/\\|\\mathbf{w}\\|') }} />.
          Points on the margin bands are <em>support vectors</em> (shown hollow in the chart).
        </p>
      </div>

      {/* Controls */}
      <div className={styles.editorBlock}>
        <h3 className={styles.optionsTitle}>Dataset &amp; training</h3>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
            gap: '12px',
          }}
        >
          <label className={styles.fieldLabel}>
            <span>Preset</span>
            <select
              className={styles.input}
              value={preset}
              onChange={(e) => {
                setPreset(e.target.value as Preset)
                setResult(null)
                setData([])
              }}
            >
              {PRESETS.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.label}
                </option>
              ))}
            </select>
          </label>

          <label className={styles.fieldLabel}>
            <span>Points per class</span>
            <input
              type="number"
              min={5}
              max={150}
              value={nPerClass}
              onChange={(e) =>
                setNPerClass(Math.max(5, Math.min(150, Number(e.target.value))))
              }
              className={styles.input}
            />
          </label>

          <label className={styles.fieldLabel}>
            <span>Margin type</span>
            <select
              className={styles.input}
              value={marginType}
              onChange={(e) => setMarginType(e.target.value as 'hard' | 'soft')}
            >
              <option value="hard">Hard (C = ∞)</option>
              <option value="soft">Soft (finite C)</option>
            </select>
          </label>

          {marginType === 'soft' && (
            <label className={styles.fieldLabel}>
              <span>C (regularisation)</span>
              <input
                type="number"
                min={0.01}
                max={1000}
                step={0.1}
                value={C}
                onChange={(e) => setC(Math.max(0.001, Number(e.target.value)))}
                className={styles.input}
              />
            </label>
          )}

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

        {preset === 'overlapping' && marginType === 'hard' && (
          <p className={styles.hint} style={{ color: '#f59e0b', marginTop: '6px' }}>
            Overlapping data may not be linearly separable — switch to soft margin for better
            results.
          </p>
        )}

        {trainError && (
          <p className={styles.error} style={{ marginTop: '6px' }}>
            {trainError}
          </p>
        )}

        <button
          type="button"
          className={styles.runBtn}
          onClick={handleRun}
          style={{ marginTop: '6px', alignSelf: 'flex-start' }}
        >
          Generate &amp; train
        </button>
      </div>

      {/* Custom data */}
      <div className={styles.editorBlock}>
        <h3 className={styles.optionsTitle}>Custom data</h3>
        <p className={styles.hint}>
          One point per line: <code>x₁, x₂, label</code> — labels can be{' '}
          <code>+1</code> / <code>-1</code>, <code>1</code> / <code>0</code>, or any two
          symbols. Lines starting with <code>#</code> are ignored. Uses the margin type and C
          set above.
        </p>
        <textarea
          className={styles.textarea}
          rows={9}
          spellCheck={false}
          value={customInput}
          onChange={(e) => {
            setCustomInput(e.target.value)
            setCustomError(null)
          }}
        />
        {customError && <p className={styles.error}>{customError}</p>}
        <button
          type="button"
          className={styles.runBtn}
          onClick={handleCustomTrain}
          style={{ marginTop: '4px' }}
        >
          Train on custom data
        </button>
      </div>

      {result && data.length > 0 && (
        <>
          {/* Decision boundary + margin chart */}
          <div className={styles.graphBlock}>
            <h3 className={styles.graphTitle}>Decision boundary &amp; margin</h3>
            <p className={styles.hint}>
              Solid line: decision boundary. Dashed lines: margin planes (
              <span
                dangerouslySetInnerHTML={{
                  __html: renderLatex('\\mathbf{w}\\cdot\\mathbf{x}+b=\\pm 1'),
                }}
              />
              ). <strong>Hollow points</strong> are support vectors.
            </p>

            <ResponsiveContainer width="100%" height={420}>
              <ComposedChart margin={{ top: 16, right: 16, left: 0, bottom: 16 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis
                  type="number"
                  dataKey="x"
                  domain={[DOMAIN.xMin, DOMAIN.xMax]}
                  stroke="var(--text-muted)"
                  tick={{ fill: 'var(--text-muted)', fontSize: 12 }}
                  label={{
                    value: 'x₁',
                    position: 'insideBottom',
                    offset: -4,
                    fill: 'var(--text-muted)',
                    fontSize: 12,
                  }}
                />
                <YAxis
                  type="number"
                  dataKey="y"
                  domain={[DOMAIN.yMin, DOMAIN.yMax]}
                  stroke="var(--text-muted)"
                  tick={{ fill: 'var(--text-muted)', fontSize: 12 }}
                  label={{
                    value: 'x₂',
                    angle: -90,
                    position: 'insideLeft',
                    fill: 'var(--text-muted)',
                    fontSize: 12,
                  }}
                />
                <ZAxis range={[55, 55]} />
                <Tooltip
                  cursor={{ stroke: 'var(--border)' }}
                  contentStyle={{
                    background: 'var(--glass-bg)',
                    border: '1px solid var(--glass-border)',
                    borderRadius: 'var(--radius)',
                  }}
                  formatter={(value: number) => [value.toFixed(3), '']}
                  labelFormatter={(_label, payload) =>
                    payload?.[0]
                      ? `(${payload[0].payload?.x?.toFixed(3)}, ${payload[0].payload?.y?.toFixed(3)})`
                      : ''
                  }
                />

                {/* +1 class — hollow = support vector */}
                <Scatter name="+1" data={pos} isAnimationActive={false}>
                  {pos.map((p, i) => (
                    <Cell
                      key={i}
                      fill={svSet.has(p._idx) ? 'transparent' : COLOR_POS}
                      stroke={COLOR_POS}
                      strokeWidth={svSet.has(p._idx) ? 2.5 : 0.5}
                    />
                  ))}
                </Scatter>

                {/* −1 class — hollow = support vector */}
                <Scatter name="−1" data={neg} isAnimationActive={false}>
                  {neg.map((p, i) => (
                    <Cell
                      key={i}
                      fill={svSet.has(p._idx) ? 'transparent' : COLOR_NEG}
                      stroke={COLOR_NEG}
                      strokeWidth={svSet.has(p._idx) ? 2.5 : 0.5}
                    />
                  ))}
                </Scatter>

                {/* Decision boundary */}
                {boundary.length === 2 && (
                  <Line
                    name="Boundary"
                    data={boundary}
                    type="linear"
                    dataKey="y"
                    stroke={COLOR_BOUNDARY}
                    strokeWidth={2.5}
                    dot={false}
                    isAnimationActive={false}
                    legendType="line"
                  />
                )}

                {/* +1 margin */}
                {posMarginPts.length === 2 && (
                  <Line
                    name="+1 margin"
                    data={posMarginPts}
                    type="linear"
                    dataKey="y"
                    stroke={COLOR_POS_MARGIN}
                    strokeWidth={1.5}
                    strokeDasharray="6 4"
                    dot={false}
                    isAnimationActive={false}
                    legendType="line"
                  />
                )}

                {/* −1 margin */}
                {negMarginPts.length === 2 && (
                  <Line
                    name="−1 margin"
                    data={negMarginPts}
                    type="linear"
                    dataKey="y"
                    stroke={COLOR_NEG_MARGIN}
                    strokeWidth={1.5}
                    strokeDasharray="6 4"
                    dot={false}
                    isAnimationActive={false}
                    legendType="line"
                  />
                )}

                <Legend />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          {/* Stats table */}
          <div className={styles.matrixBlock}>
            <h4 className={styles.matrixTitle}>Model summary</h4>
            <div className={styles.matrixWrap}>
              <table className={styles.matrixTable}>
                <tbody>
                  <tr>
                    <th className={styles.matrixRowHeader}>w₁</th>
                    <td className={styles.matrixCell}>{result.model.w1.toFixed(4)}</td>
                  </tr>
                  <tr>
                    <th className={styles.matrixRowHeader}>w₂</th>
                    <td className={styles.matrixCell}>{result.model.w2.toFixed(4)}</td>
                  </tr>
                  <tr>
                    <th className={styles.matrixRowHeader}>b</th>
                    <td className={styles.matrixCell}>{result.model.b.toFixed(4)}</td>
                  </tr>
                  <tr>
                    <th className={styles.matrixRowHeader}>‖w‖</th>
                    <td className={styles.matrixCell}>{wNorm?.toFixed(4) ?? '—'}</td>
                  </tr>
                  <tr>
                    <th className={styles.matrixRowHeader}>Margin 2/‖w‖</th>
                    <td className={styles.matrixCell}>{result.model.margin.toFixed(4)}</td>
                  </tr>
                  <tr>
                    <th className={styles.matrixRowHeader}>C</th>
                    <td className={styles.matrixCell}>
                      {marginType === 'hard' ? '∞ (hard margin)' : C}
                    </td>
                  </tr>
                  <tr>
                    <th className={styles.matrixRowHeader}>Support vectors</th>
                    <td className={styles.matrixCell}>
                      {result.model.supportVectorIndices.length}
                    </td>
                  </tr>
                  <tr>
                    <th className={styles.matrixRowHeader}>Accuracy</th>
                    <td className={styles.matrixCell}>
                      {accuracy !== null ? `${(accuracy * 100).toFixed(1)}%` : '—'}
                    </td>
                  </tr>
                  <tr>
                    <th className={styles.matrixRowHeader}>Converged</th>
                    <td
                      className={styles.matrixCell}
                      style={{ color: result.converged ? '#22c55e' : '#f59e0b' }}
                    >
                      {result.converged
                        ? `Yes (${result.iterations} passes)`
                        : `No (${result.iterations} passes) — try more iterations`}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
            {!result.converged && (
              <p className={styles.theoreticalHint} style={{ marginTop: '8px' }}>
                Training did not fully converge. For non-separable data, switch to soft margin.
                For separable data, try a smaller seed or different preset.
              </p>
            )}
          </div>
        </>
      )}
    </div>
  )
}
