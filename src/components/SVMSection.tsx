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
  Customized,
} from 'recharts'
import {
  trainSVM,
  makePredictor,
  marginLine,
  generateDataset,
  type DataPoint,
  type SVMResult,
  type KernelType,
  type KernelParams,
  type PresetId,
} from '@/lib/svm'
import { createSeededRng } from '@/lib/random'
import styles from './MarkovChainSection.module.css'

// ── Helpers ──────────────────────────────────────────────────────────────────

function renderLatex(latex: string, displayMode = false): string {
  try {
    return katex.renderToString(latex, { displayMode, throwOnError: false })
  } catch {
    return latex
  }
}

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
  const [la] = seen
  const aIs1 =
    ['+1', '1'].includes(la) && ['-1', '0'].includes(seen[1])
      ? true
      : ['-1', '0'].includes(la) && ['+1', '1'].includes(seen[1])
        ? false
        : true
  const pts: DataPoint[] = rows.map((r) => ({
    x: r.x,
    y: r.y,
    label: (r.raw === la ? (aIs1 ? 1 : -1) : aIs1 ? -1 : 1) as 1 | -1,
  }))
  return { pts }
}

// ── Constants ─────────────────────────────────────────────────────────────────

const DOMAIN = { xMin: -4, xMax: 4, yMin: -4, yMax: 4 }
const GRID_SIZE = 60

const COLOR_POS = 'var(--accent)'
const COLOR_NEG = '#0ea5e9'
const COLOR_BOUNDARY = '#f97316'
const COLOR_POS_MARGIN = '#fb923c'
const COLOR_NEG_MARGIN = '#38bdf8'

const DEFAULT_CUSTOM = `# x1, x2, label  (+1 or -1)
 2.1,  1.5, +1
 1.3,  2.2, +1
 2.5,  0.8, +1
 1.9,  2.8, +1
-1.8, -1.2, -1
-2.3, -0.5, -1
-1.0, -2.5, -1
-2.8, -1.9, -1`

// Presets that benefit from a non-linear kernel
const NONLINEAR_PRESETS: PresetId[] = ['circles', 'xor', 'moons']

// ── Decision-region heatmap layer (rendered inside Recharts SVG) ──────────────

function HeatmapLayer(props: Record<string, unknown>) {
  const xAxisMap = props.xAxisMap as
    | Record<number, { scale: (v: number) => number }>
    | undefined
  const yAxisMap = props.yAxisMap as
    | Record<number, { scale: (v: number) => number }>
    | undefined
  const grid = props.grid as Int8Array | null

  if (!grid) return null
  const xScale = xAxisMap?.[0]?.scale
  const yScale = yAxisMap?.[0]?.scale
  if (!xScale || !yScale) return null

  const x0 = xScale(DOMAIN.xMin)
  const x1 = xScale(DOMAIN.xMax)
  const y0 = yScale(DOMAIN.yMax) // top of plot area (smaller pixel y)
  const y1 = yScale(DOMAIN.yMin) // bottom of plot area (larger pixel y)
  const cellW = (x1 - x0) / GRID_SIZE + 0.6 // slight overlap prevents gap lines
  const cellH = (y1 - y0) / GRID_SIZE + 0.6

  return (
    <g>
      {Array.from({ length: GRID_SIZE * GRID_SIZE }, (_, idx) => {
        const row = Math.floor(idx / GRID_SIZE)
        const col = idx % GRID_SIZE
        return (
          <rect
            key={idx}
            x={x0 + col * ((x1 - x0) / GRID_SIZE)}
            y={y0 + row * ((y1 - y0) / GRID_SIZE)}
            width={cellW}
            height={cellH}
            fill={grid[idx] === 1 ? COLOR_POS : COLOR_NEG}
            opacity={0.13}
          />
        )
      })}
    </g>
  )
}

// ── Component ─────────────────────────────────────────────────────────────────

type IndexedPoint = DataPoint & { _idx: number }

export function SVMSection() {
  // Dataset controls
  const [preset, setPreset] = useState<PresetId>('blobs')
  const [nPerClass, setNPerClass] = useState(30)
  const [seed, setSeed] = useState('')

  // Margin controls
  const [marginType, setMarginType] = useState<'hard' | 'soft'>('hard')
  const [C, setC] = useState(1.0)

  // Kernel controls
  const [kernelType, setKernelType] = useState<KernelType>('linear')
  const [gamma, setGamma] = useState(0.5)
  const [degree, setDegree] = useState(3)
  const [coef0, setCoef0] = useState(1.0)

  // Results
  const [data, setData] = useState<DataPoint[]>([])
  const [result, setResult] = useState<SVMResult | null>(null)
  const [trainError, setTrainError] = useState<string | null>(null)

  // Custom data
  const [customInput, setCustomInput] = useState(DEFAULT_CUSTOM)
  const [customError, setCustomError] = useState<string | null>(null)

  function buildKernelParams(): KernelParams {
    if (kernelType === 'rbf') return { type: 'rbf', gamma }
    if (kernelType === 'poly') return { type: 'poly', degree, coef0 }
    return { type: 'linear' }
  }

  function runWithData(pts: DataPoint[]) {
    setTrainError(null)
    try {
      const kp = buildKernelParams()
      const cValue = marginType === 'hard' ? Infinity : C
      const res = trainSVM(pts, cValue, kp)
      setData(pts)
      setResult(res)
    } catch (e) {
      setTrainError(e instanceof Error ? e.message : 'Training failed.')
    }
  }

  function handleRun() {
    const rand =
      seed.trim() !== '' && !Number.isNaN(Number(seed))
        ? createSeededRng(Number(seed))
        : Math.random
    runWithData(generateDataset(preset, nPerClass, rand))
  }

  function handleCustomTrain() {
    setCustomError(null)
    const res = parseCustomData(customInput)
    if ('error' in res) {
      setCustomError(res.error)
      return
    }
    runWithData(res.pts)
  }

  // Partition data by class, tagging each with its original index
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

  // Fast predictor closure
  const predictor = useMemo(
    () => (result ? makePredictor(data, result.model) : null),
    [data, result],
  )

  // 60×60 decision-region grid (row 0 = top = yMax)
  const decisionGrid = useMemo((): Int8Array | null => {
    if (!predictor) return null
    const grid = new Int8Array(GRID_SIZE * GRID_SIZE)
    const dx = (DOMAIN.xMax - DOMAIN.xMin) / GRID_SIZE
    const dy = (DOMAIN.yMax - DOMAIN.yMin) / GRID_SIZE
    for (let row = 0; row < GRID_SIZE; row++) {
      for (let col = 0; col < GRID_SIZE; col++) {
        const x = DOMAIN.xMin + (col + 0.5) * dx
        const y = DOMAIN.yMax - (row + 0.5) * dy
        grid[row * GRID_SIZE + col] = predictor(x, y)
      }
    }
    return grid
  }, [predictor])

  // Margin lines (linear kernel only)
  const { boundary, posMarginPts, negMarginPts } = useMemo(() => {
    if (!result || result.model.kernel.type !== 'linear')
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
    if (!predictor || data.length === 0) return null
    return data.filter((p) => predictor(p.x, p.y) === p.label).length / data.length
  }, [predictor, data])

  const wNorm = result
    ? Math.sqrt(result.model.w1 ** 2 + result.model.w2 ** 2)
    : null

  const isLinear = kernelType === 'linear'
  const isNonLinearPreset = NONLINEAR_PRESETS.includes(preset)

  // Kernel formula string for intro
  const kernelFormulaLatex =
    kernelType === 'rbf'
      ? `K(\\mathbf{x}_i,\\mathbf{x}_j) = \\exp(-${gamma}\\|\\mathbf{x}_i-\\mathbf{x}_j\\|^2)`
      : kernelType === 'poly'
        ? `K(\\mathbf{x}_i,\\mathbf{x}_j) = (\\mathbf{x}_i\\cdot\\mathbf{x}_j + ${coef0})^{${degree}}`
        : `K(\\mathbf{x}_i,\\mathbf{x}_j) = \\mathbf{x}_i\\cdot\\mathbf{x}_j`

  return (
    <div className={styles.section}>
      {/* ── Intro ── */}
      <div className={styles.intro}>
        <p className={styles.introText}>
          <strong>Support Vector Machine (SVM)</strong> maximises the margin between two classes.
          The kernel trick maps inputs into a feature space where linear separation is possible,
          enabling non-linear boundaries in the original space. Current kernel:
        </p>
        <p
          className={styles.introFormula}
          dangerouslySetInnerHTML={{ __html: renderLatex(kernelFormulaLatex, true) }}
        />
        <p className={styles.introText}>
          The soft-margin objective (
          <span dangerouslySetInnerHTML={{ __html: renderLatex('C\\text{-SVM}') }} />) is:
        </p>
        <p
          className={styles.introFormula}
          dangerouslySetInnerHTML={{
            __html: renderLatex(
              '\\min_{\\mathbf{w},b,\\boldsymbol{\\xi}}\\;\\tfrac{1}{2}\\|\\mathbf{w}\\|^2 + C\\sum_i \\xi_i \\quad \\text{s.t.}\\; y_i(\\mathbf{w}\\cdot\\phi(\\mathbf{x}_i)+b)\\geq 1-\\xi_i,\\;\\xi_i\\geq 0',
              true,
            ),
          }}
        />
      </div>

      {/* ── Controls ── */}
      <div className={styles.editorBlock}>
        <h3 className={styles.optionsTitle}>Dataset &amp; training</h3>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
            gap: '12px',
          }}
        >
          {/* Preset */}
          <label className={styles.fieldLabel}>
            <span>Preset</span>
            <select
              className={styles.input}
              value={preset}
              onChange={(e) => {
                setPreset(e.target.value as PresetId)
                setResult(null)
                setData([])
              }}
            >
              <optgroup label="Linearly separable">
                <option value="blobs">Two blobs</option>
                <option value="wide-margin">Wide margin</option>
                <option value="near-boundary">Near boundary</option>
                <option value="overlapping">Overlapping</option>
              </optgroup>
              <optgroup label="Non-linear (kernel SVM)">
                <option value="circles">Concentric circles</option>
                <option value="xor">XOR pattern</option>
                <option value="moons">Two moons</option>
              </optgroup>
            </select>
          </label>

          {/* Points per class */}
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

          {/* Margin type */}
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

          {/* C (soft only) */}
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

          {/* Kernel type */}
          <label className={styles.fieldLabel}>
            <span>Kernel</span>
            <select
              className={styles.input}
              value={kernelType}
              onChange={(e) => setKernelType(e.target.value as KernelType)}
            >
              <option value="linear">Linear</option>
              <option value="rbf">RBF (Gaussian)</option>
              <option value="poly">Polynomial</option>
            </select>
          </label>

          {/* γ (RBF) */}
          {kernelType === 'rbf' && (
            <label className={styles.fieldLabel}>
              <span>γ (gamma)</span>
              <input
                type="number"
                min={0.01}
                max={20}
                step={0.05}
                value={gamma}
                onChange={(e) => setGamma(Math.max(0.01, Number(e.target.value)))}
                className={styles.input}
              />
            </label>
          )}

          {/* degree + coef₀ (poly) */}
          {kernelType === 'poly' && (
            <>
              <label className={styles.fieldLabel}>
                <span>Degree</span>
                <input
                  type="number"
                  min={2}
                  max={7}
                  step={1}
                  value={degree}
                  onChange={(e) =>
                    setDegree(Math.max(2, Math.min(7, Math.round(Number(e.target.value)))))
                  }
                  className={styles.input}
                />
              </label>
              <label className={styles.fieldLabel}>
                <span>coef₀</span>
                <input
                  type="number"
                  min={0}
                  max={10}
                  step={0.5}
                  value={coef0}
                  onChange={(e) => setCoef0(Math.max(0, Number(e.target.value)))}
                  className={styles.input}
                />
              </label>
            </>
          )}

          {/* Seed */}
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

        {/* Hints */}
        {isNonLinearPreset && isLinear && (
          <p className={styles.hint} style={{ color: '#f59e0b', marginTop: '6px' }}>
            This preset is non-linearly separable — try the RBF kernel for a curved boundary.
          </p>
        )}
        {preset === 'overlapping' && marginType === 'hard' && (
          <p className={styles.hint} style={{ color: '#f59e0b', marginTop: '6px' }}>
            Overlapping data may not be separable — switch to soft margin.
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

      {/* ── Custom data ── */}
      <div className={styles.editorBlock}>
        <h3 className={styles.optionsTitle}>Custom data</h3>
        <p className={styles.hint}>
          One point per line: <code>x₁, x₂, label</code>. Labels: <code>+1</code> /{' '}
          <code>-1</code>. Lines starting with <code>#</code> are ignored.
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

      {/* ── Results ── */}
      {result && data.length > 0 && (
        <>
          {/* Chart */}
          <div className={styles.graphBlock}>
            <h3 className={styles.graphTitle}>Decision boundary &amp; margin</h3>
            <p className={styles.hint}>
              Shaded regions show the predicted class. <strong>Hollow points</strong> are support
              vectors.
              {isLinear && ' Dashed lines are margin planes (w·x + b = ±1).'}
            </p>

            <ResponsiveContainer width="100%" height={440}>
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
                  formatter={(v: number) => [v.toFixed(3), '']}
                  labelFormatter={(_l, payload) =>
                    payload?.[0]
                      ? `(${payload[0].payload?.x?.toFixed(3)}, ${payload[0].payload?.y?.toFixed(3)})`
                      : ''
                  }
                />

                {/* Decision-region heatmap — rendered first so it sits behind data points */}
                <Customized component={HeatmapLayer} grid={decisionGrid} />

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

                {/* −1 class */}
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

                {/* Linear kernel: decision boundary + margin lines */}
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

          {/* Stats */}
          <div className={styles.matrixBlock}>
            <h4 className={styles.matrixTitle}>Model summary</h4>
            <div className={styles.matrixWrap}>
              <table className={styles.matrixTable}>
                <tbody>
                  <tr>
                    <th className={styles.matrixRowHeader}>Kernel</th>
                    <td className={styles.matrixCell}>
                      {kernelType === 'linear'
                        ? 'Linear'
                        : kernelType === 'rbf'
                          ? `RBF  (γ = ${gamma})`
                          : `Polynomial  (d = ${degree}, c₀ = ${coef0})`}
                    </td>
                  </tr>
                  <tr>
                    <th className={styles.matrixRowHeader}>C</th>
                    <td className={styles.matrixCell}>
                      {marginType === 'hard' ? '∞ (hard margin)' : C}
                    </td>
                  </tr>
                  {isLinear && (
                    <>
                      <tr>
                        <th className={styles.matrixRowHeader}>w₁</th>
                        <td className={styles.matrixCell}>{result.model.w1.toFixed(4)}</td>
                      </tr>
                      <tr>
                        <th className={styles.matrixRowHeader}>w₂</th>
                        <td className={styles.matrixCell}>{result.model.w2.toFixed(4)}</td>
                      </tr>
                      <tr>
                        <th className={styles.matrixRowHeader}>‖w‖</th>
                        <td className={styles.matrixCell}>{wNorm?.toFixed(4) ?? '—'}</td>
                      </tr>
                      <tr>
                        <th className={styles.matrixRowHeader}>Margin 2/‖w‖</th>
                        <td className={styles.matrixCell}>
                          {result.model.margin.toFixed(4)}
                        </td>
                      </tr>
                    </>
                  )}
                  <tr>
                    <th className={styles.matrixRowHeader}>b</th>
                    <td className={styles.matrixCell}>{result.model.b.toFixed(4)}</td>
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
                        : `No (${result.iterations} passes)`}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
            {!result.converged && (
              <p className={styles.theoreticalHint} style={{ marginTop: '8px' }}>
                Training did not fully converge. For non-separable data use soft margin; for
                non-linear data try RBF or polynomial kernel.
              </p>
            )}
          </div>
        </>
      )}
    </div>
  )
}
