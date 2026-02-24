import { useState, useMemo, useEffect } from 'react'
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
  LineChart,
  ReferenceLine,
  Cell,
} from 'recharts'
import {
  train,
  boundaryLine,
  generateDataset,
  type DataPoint,
  type TrainResult,
} from '@/lib/perceptron'
import { createSeededRng } from '@/lib/random'
import styles from './MarkovChainSection.module.css'

function renderLatex(latex: string, displayMode = false): string {
  try {
    return katex.renderToString(latex, { displayMode, throwOnError: false })
  } catch {
    return latex
  }
}

const DEFAULT_CUSTOM =
  `# x1, x2, label  (+1 or -1)
 1.2,  1.8, +1
 2.1,  0.7, +1
 0.5,  2.3, +1
 1.8,  2.5, +1
-1.4, -0.8, -1
-0.8, -2.1, -1
 0.2, -1.7, -1
-2.0, -0.4, -1`

/**
 * Parse "x1, x2, label" text into DataPoint[].
 * Labels +1/1 → +1 and -1/0 → -1. Any two other distinct strings:
 * first seen → +1, second → -1.
 * Returns null with an error message on failure.
 */
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
  if (seen.length > 2) return { error: `Found ${seen.length} distinct labels — perceptron is binary only.` }

  const [la, lb] = seen
  // Honour common +1/-1 and 1/0 conventions
  let aIs1 =
    (['+1', '1'].includes(la) && ['-1', '0'].includes(lb)) ? true
    : (['-1', '0'].includes(la) && ['+1', '1'].includes(lb)) ? false
    : true  // arbitrary: first seen = +1

  const pts: DataPoint[] = rows.map((r) => ({
    x: r.x,
    y: r.y,
    label: (r.raw === la ? (aIs1 ? 1 : -1) : (aIs1 ? -1 : 1)) as 1 | -1,
  }))
  return { pts }
}

type Preset = 'blobs' | 'diagonal' | 'vertical' | 'wide-margin'

const PRESETS: { id: Preset; label: string }[] = [
  { id: 'blobs', label: 'Two blobs' },
  { id: 'diagonal', label: 'Diagonal boundary' },
  { id: 'vertical', label: 'Vertical split' },
  { id: 'wide-margin', label: 'Wide margin' },
]

const SPEEDS = { slow: 900, medium: 350, fast: 100 } as const
type Speed = keyof typeof SPEEDS

const DOMAIN = { xMin: -4, xMax: 4, yMin: -4, yMax: 4 }
const COLOR_POS = 'var(--accent)'
const COLOR_NEG = '#0ea5e9'
const COLOR_BOUNDARY = '#f97316'
const COLOR_WRONG = '#ef4444'

const btnBase: React.CSSProperties = {
  padding: '0.3rem 0.6rem',
  fontSize: '1.05rem',
  lineHeight: 1,
  background: 'var(--glass-bg)',
  color: 'var(--text)',
  border: '1px solid var(--glass-border)',
  borderRadius: 'var(--radius)',
  cursor: 'pointer',
}

export function PerceptronSection() {
  const [preset, setPreset] = useState<Preset>('blobs')
  const [nPerClass, setNPerClass] = useState(40)
  const [learningRate, setLearningRate] = useState(1)
  const [maxEpochs, setMaxEpochs] = useState(50)
  const [seed, setSeed] = useState('')
  const [data, setData] = useState<DataPoint[]>([])
  const [result, setResult] = useState<TrainResult | null>(null)

  const [customInput, setCustomInput] = useState(DEFAULT_CUSTOM)
  const [customError, setCustomError] = useState<string | null>(null)

  const [animEpoch, setAnimEpoch] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [speed, setSpeed] = useState<Speed>('medium')

  const runWithData = (pts: DataPoint[]) => {
    const res = train(pts, { learningRate, maxEpochs })
    setData(pts)
    setResult(res)
    setAnimEpoch(0)
    setIsPlaying(true)
  }

  const handleRun = () => {
    const rand =
      seed.trim() !== '' && !Number.isNaN(Number(seed))
        ? createSeededRng(Number(seed))
        : Math.random
    const pts = generateDataset(preset, nPerClass, rand)
    runWithData(pts)
  }

  const handleCustomTrain = () => {
    setCustomError(null)
    const result = parseCustomData(customInput)
    if ('error' in result) {
      setCustomError(result.error)
      return
    }
    runWithData(result.pts)
  }

  // Animation loop
  useEffect(() => {
    if (!isPlaying || !result) return
    const timer = setInterval(() => {
      setAnimEpoch((prev) => {
        if (prev >= result.epochs) {
          setIsPlaying(false)
          return prev
        }
        return prev + 1
      })
    }, SPEEDS[speed])
    return () => clearInterval(timer)
  }, [isPlaying, result, speed])

  const pos = useMemo(() => data.filter((p) => p.label === 1), [data])
  const neg = useMemo(() => data.filter((p) => p.label === -1), [data])

  const currentModel = useMemo(
    () => result?.snapshots[animEpoch] ?? null,
    [result, animEpoch]
  )

  // Which points in pos/neg are misclassified under the current model
  const misclassifiedPos = useMemo(() => {
    if (!currentModel) return new Set<number>()
    const s = new Set<number>()
    pos.forEach((pt, i) => {
      if (currentModel.w1 * pt.x + currentModel.w2 * pt.y + currentModel.b < 0) s.add(i)
    })
    return s
  }, [currentModel, pos])

  const misclassifiedNeg = useMemo(() => {
    if (!currentModel) return new Set<number>()
    const s = new Set<number>()
    neg.forEach((pt, i) => {
      if (currentModel.w1 * pt.x + currentModel.w2 * pt.y + currentModel.b >= 0) s.add(i)
    })
    return s
  }, [currentModel, neg])

  const currentErrors = misclassifiedPos.size + misclassifiedNeg.size

  const boundary = useMemo(() => {
    if (!currentModel) return []
    return boundaryLine(currentModel, DOMAIN.xMin, DOMAIN.xMax, DOMAIN.yMin, DOMAIN.yMax)
  }, [currentModel])

  const handlePlayPause = () => {
    if (!result) return
    if (!isPlaying && animEpoch >= result.epochs) {
      setAnimEpoch(0)
      setIsPlaying(true)
    } else {
      setIsPlaying((p) => !p)
    }
  }

  const epochStatusText =
    animEpoch === 0
      ? 'Epoch 0 — untrained'
      : `Epoch ${animEpoch} / ${result?.epochs ?? '?'} — ${currentErrors} misclassified`

  return (
    <div className={styles.section}>
      {/* Intro */}
      <div className={styles.intro}>
        <p className={styles.introText}>
          <strong>Perceptron</strong> is the simplest linear binary classifier. It learns a
          hyperplane{' '}
          <span
            dangerouslySetInnerHTML={{
              __html: renderLatex('\\hat{y} = \\operatorname{sign}(w_1 x_1 + w_2 x_2 + b)'),
            }}
          />{' '}
          by updating weights whenever a point is misclassified:
        </p>
        <p
          className={styles.introFormula}
          dangerouslySetInnerHTML={{
            __html: renderLatex(
              '\\mathbf{w} \\leftarrow \\mathbf{w} + \\eta\\, y_i\\, \\mathbf{x}_i, \\quad b \\leftarrow b + \\eta\\, y_i',
              true
            ),
          }}
        />
        <p className={styles.introText}>
          By Rosenblatt's convergence theorem, training is guaranteed to converge in finite steps
          if (and only if) the data is linearly separable.
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
              max={200}
              value={nPerClass}
              onChange={(e) => setNPerClass(Math.max(5, Math.min(200, Number(e.target.value))))}
              className={styles.input}
            />
          </label>
          <label className={styles.fieldLabel}>
            <span>Learning rate η</span>
            <input
              type="number"
              min={0.01}
              max={10}
              step={0.1}
              value={learningRate}
              onChange={(e) => setLearningRate(Number(e.target.value))}
              className={styles.input}
            />
          </label>
          <label className={styles.fieldLabel}>
            <span>Max epochs</span>
            <input
              type="number"
              min={1}
              max={1000}
              step={10}
              value={maxEpochs}
              onChange={(e) =>
                setMaxEpochs(Math.max(1, Math.min(1000, Number(e.target.value))))
              }
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
        <button
          type="button"
          className={styles.runBtn}
          onClick={handleRun}
          style={{ marginTop: '4px' }}
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
          symbols. Lines starting with <code>#</code> are ignored. Uses the learning rate
          and max epochs above.
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
          {/* Decision boundary chart */}
          <div className={styles.graphBlock}>
            {/* Title row */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                flexWrap: 'wrap',
                gap: '8px',
              }}
            >
              <h3 className={styles.graphTitle}>Decision boundary</h3>
              <span
                className={styles.hint}
                style={{ fontVariantNumeric: 'tabular-nums', color: animEpoch === 0 ? 'var(--text-muted)' : currentErrors === 0 ? '#22c55e' : COLOR_WRONG }}
              >
                {epochStatusText}
              </span>
            </div>

            {/* Animation controls */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                flexWrap: 'wrap',
              }}
            >
              {/* Step buttons */}
              <div style={{ display: 'flex', gap: '4px' }}>
                <button
                  type="button"
                  style={btnBase}
                  onClick={() => {
                    setIsPlaying(false)
                    setAnimEpoch(0)
                  }}
                  title="First epoch"
                >
                  ⏮
                </button>
                <button
                  type="button"
                  style={btnBase}
                  onClick={() => {
                    setIsPlaying(false)
                    setAnimEpoch((p) => Math.max(0, p - 1))
                  }}
                  title="Previous epoch"
                >
                  ⏪
                </button>
                <button
                  type="button"
                  className={styles.runBtn}
                  style={{ padding: '0.3rem 0.85rem', fontSize: '1.1rem', lineHeight: 1, minWidth: '2.5rem' }}
                  onClick={handlePlayPause}
                  title={isPlaying ? 'Pause' : 'Play'}
                >
                  {isPlaying ? '⏸' : '▶'}
                </button>
                <button
                  type="button"
                  style={btnBase}
                  onClick={() => {
                    setIsPlaying(false)
                    setAnimEpoch((p) => Math.min(result.epochs, p + 1))
                  }}
                  title="Next epoch"
                >
                  ⏩
                </button>
                <button
                  type="button"
                  style={btnBase}
                  onClick={() => {
                    setIsPlaying(false)
                    setAnimEpoch(result.epochs)
                  }}
                  title="Last epoch"
                >
                  ⏭
                </button>
              </div>

              {/* Scrubber */}
              <input
                type="range"
                min={0}
                max={result.epochs}
                value={animEpoch}
                onChange={(e) => {
                  setIsPlaying(false)
                  setAnimEpoch(Number(e.target.value))
                }}
                style={{ flex: '1', minWidth: '100px' }}
              />

              {/* Speed */}
              <label
                className={styles.fieldLabel}
                style={{ flexDirection: 'row', alignItems: 'center', gap: '6px' }}
              >
                <span>Speed</span>
                <select
                  className={styles.input}
                  style={{ minWidth: '80px' }}
                  value={speed}
                  onChange={(e) => setSpeed(e.target.value as Speed)}
                >
                  <option value="slow">Slow</option>
                  <option value="medium">Medium</option>
                  <option value="fast">Fast</option>
                </select>
              </label>
            </div>

            {/* Scatter + boundary */}
            <ResponsiveContainer width="100%" height={400}>
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
                {/* +1 class — misclassified shown as hollow red circles */}
                <Scatter name="+1" data={pos} isAnimationActive={false}>
                  {pos.map((_, i) => (
                    <Cell
                      key={i}
                      fill={misclassifiedPos.has(i) ? 'transparent' : COLOR_POS}
                      stroke={misclassifiedPos.has(i) ? COLOR_WRONG : COLOR_POS}
                      strokeWidth={misclassifiedPos.has(i) ? 2 : 1}
                    />
                  ))}
                </Scatter>
                {/* −1 class — misclassified shown as hollow red circles */}
                <Scatter name="−1" data={neg} isAnimationActive={false}>
                  {neg.map((_, i) => (
                    <Cell
                      key={i}
                      fill={misclassifiedNeg.has(i) ? 'transparent' : COLOR_NEG}
                      stroke={misclassifiedNeg.has(i) ? COLOR_WRONG : COLOR_NEG}
                      strokeWidth={misclassifiedNeg.has(i) ? 2 : 1}
                    />
                  ))}
                </Scatter>
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
                <Legend />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          {/* Weights table */}
          <div className={styles.matrixBlock}>
            <h4 className={styles.matrixTitle}>
              Model weights — {animEpoch === 0 ? 'initial (untrained)' : `epoch ${animEpoch}`}
            </h4>
            <div className={styles.matrixWrap}>
              <table className={styles.matrixTable}>
                <tbody>
                  <tr>
                    <th className={styles.matrixRowHeader}>w₁</th>
                    <td className={styles.matrixCell}>
                      {currentModel?.w1.toFixed(4) ?? '—'}
                    </td>
                  </tr>
                  <tr>
                    <th className={styles.matrixRowHeader}>w₂</th>
                    <td className={styles.matrixCell}>
                      {currentModel?.w2.toFixed(4) ?? '—'}
                    </td>
                  </tr>
                  <tr>
                    <th className={styles.matrixRowHeader}>bias b</th>
                    <td className={styles.matrixCell}>
                      {currentModel?.b.toFixed(4) ?? '—'}
                    </td>
                  </tr>
                  <tr>
                    <th className={styles.matrixRowHeader}>Misclassified</th>
                    <td className={styles.matrixCell}>
                      {currentErrors} / {data.length}
                    </td>
                  </tr>
                  <tr>
                    <th className={styles.matrixRowHeader}>Accuracy</th>
                    <td className={styles.matrixCell}>
                      {data.length > 0
                        ? `${(((data.length - currentErrors) / data.length) * 100).toFixed(1)}%`
                        : '—'}
                    </td>
                  </tr>
                  <tr>
                    <th className={styles.matrixRowHeader}>Converged</th>
                    <td
                      className={styles.matrixCell}
                      style={{ color: result.converged ? '#22c55e' : 'var(--text-muted)' }}
                    >
                      {result.converged ? `Yes (epoch ${result.epochs})` : 'No'}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
            {!result.converged && (
              <p className={styles.theoreticalHint} style={{ marginTop: '8px' }}>
                Increase max epochs to allow full convergence.
              </p>
            )}
          </div>

          {/* Error-per-epoch chart */}
          {result.history.length > 1 && (
            <div className={styles.graphBlock}>
              <h3 className={styles.graphTitle}>Misclassifications per epoch</h3>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart
                  data={result.history}
                  margin={{ top: 8, right: 16, left: 0, bottom: 8 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis
                    dataKey="epoch"
                    stroke="var(--text-muted)"
                    tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                    label={{
                      value: 'Epoch',
                      position: 'insideBottom',
                      offset: -4,
                      fill: 'var(--text-muted)',
                      fontSize: 12,
                    }}
                  />
                  <YAxis
                    stroke="var(--text-muted)"
                    tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                    allowDecimals={false}
                    label={{
                      value: 'Errors',
                      angle: -90,
                      position: 'insideLeft',
                      fill: 'var(--text-muted)',
                      fontSize: 12,
                    }}
                  />
                  <Tooltip
                    contentStyle={{
                      background: 'var(--glass-bg)',
                      border: '1px solid var(--glass-border)',
                      borderRadius: 'var(--radius)',
                    }}
                    formatter={(v: number) => [v, 'errors']}
                    labelFormatter={(l) => `Epoch ${l}`}
                  />
                  <ReferenceLine y={0} stroke="var(--border)" strokeDasharray="2 2" />
                  {/* Cursor tracking current animation epoch */}
                  {animEpoch > 0 && (
                    <ReferenceLine
                      x={animEpoch}
                      stroke="var(--accent)"
                      strokeWidth={2}
                      label={{
                        value: `${animEpoch}`,
                        position: 'top',
                        fill: 'var(--accent)',
                        fontSize: 11,
                      }}
                    />
                  )}
                  <Line
                    type="monotone"
                    dataKey="errors"
                    stroke={COLOR_BOUNDARY}
                    strokeWidth={2}
                    dot={result.history.length <= 30}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </>
      )}
    </div>
  )
}
