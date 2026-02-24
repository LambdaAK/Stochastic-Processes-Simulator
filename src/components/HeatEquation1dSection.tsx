import { useState, useMemo, useCallback, useRef, useEffect } from 'react'
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
} from 'recharts'
import {
  solveHeatEquation1d,
  gridX,
  type HeatConfig1d,
  type HeatResult1d,
  type InitialCondition1d,
} from '@/lib/heatEquation1d'
import styles from './MarkovChainSection.module.css'

function renderLatex(latex: string, displayMode = false): string {
  try {
    return katex.renderToString(latex, { displayMode, throwOnError: false })
  } catch {
    return latex
  }
}

const INITIAL_OPTIONS: { id: InitialCondition1d; label: string }[] = [
  { id: 'point', label: 'Point source (center)' },
  { id: 'half', label: 'Half hot / half cold' },
  { id: 'bump', label: 'Gaussian bump' },
  { id: 'two-humps', label: 'Two humps' },
]

export function HeatEquation1dSection() {
  const [N, setN] = useState(80)
  const [alpha, setAlpha] = useState(0.2)
  const [dt, setDt] = useState(0.0005)
  const [T, setT] = useState(1)
  const [initial, setInitial] = useState<InitialCondition1d>('point')
  const [result, setResult] = useState<HeatResult1d | null>(null)
  const [snapshotIndex, setSnapshotIndex] = useState(0)
  const [playing, setPlaying] = useState(false)
  const rafRef = useRef<number | null>(null)
  const lastTickRef = useRef(0)
  const accumulatedRef = useRef(0)
  const STEPS_PER_SECOND = 20

  const runSimulation = useCallback(() => {
    const config: HeatConfig1d = { N, alpha, dt, T, initial }
    const res = solveHeatEquation1d(config)
    setResult(res)
    setSnapshotIndex(0)
    setPlaying(false)
  }, [N, alpha, dt, T, initial])

  const numSnapshots = result?.snapshots.length ?? 0
  const currentIndex = result ? Math.max(0, Math.min(snapshotIndex, numSnapshots - 1)) : 0
  const currentTime = result?.times[currentIndex] ?? 0

  const chartData = useMemo(() => {
    if (!result || currentIndex < 0 || currentIndex >= result.snapshots.length) return []
    const snap = result.snapshots[currentIndex]
    return Array.from({ length: result.nPoints }, (_, i) => ({
      x: gridX(result, i),
      u: snap[i],
    }))
  }, [result, currentIndex])

  const yDomain = useMemo(() => {
    if (!result?.snapshots[0]) return [0, 1]
    const initial = result.snapshots[0]
    let max = 0
    for (let i = 0; i < initial.length; i++) {
      const v = initial[i]
      if (isFinite(v) && v > max) max = v
    }
    if (max <= 0) max = 1
    return [0, max] as [number, number]
  }, [result])

  useEffect(() => {
    if (!result || !playing || numSnapshots <= 1) return
    accumulatedRef.current = 0

    const animate = (now: number) => {
      const elapsed = (now - lastTickRef.current) / 1000
      lastTickRef.current = now
      accumulatedRef.current += elapsed
      const timePerStep = 1 / STEPS_PER_SECOND
      let steps = 0
      while (accumulatedRef.current >= timePerStep && steps < numSnapshots) {
        accumulatedRef.current -= timePerStep
        steps += 1
      }
      if (steps > 0) {
        setSnapshotIndex((prev) => {
          const next = Math.min(prev + steps, numSnapshots - 1)
          if (next >= numSnapshots - 1) setPlaying(false)
          return next
        })
      }
      rafRef.current = requestAnimationFrame(animate)
    }
    lastTickRef.current = performance.now()
    rafRef.current = requestAnimationFrame(animate)
    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current)
    }
  }, [playing, result, numSnapshots])

  const stabilityHint = useMemo(() => {
    const h = 1 / (N + 1)
    const maxDt = (h * h) / (2 * alpha)
    return { maxDt, stable: dt <= maxDt * 1.01 }
  }, [N, alpha, dt])

  return (
    <div className={styles.section}>
      <div className={styles.intro}>
        <p className={styles.introText}>
          The <strong>1D heat equation</strong> is{' '}
          <span
            dangerouslySetInnerHTML={{
              __html: renderLatex('\\frac{\\partial u}{\\partial t} = \\alpha \\frac{\\partial^2 u}{\\partial x^2}', true),
            }}
          />
          {' '}on the interval <span dangerouslySetInnerHTML={{ __html: renderLatex('x\\in[0,1]') }} /> with{' '}
          <span dangerouslySetInnerHTML={{ __html: renderLatex('u(0,t)=u(1,t)=0') }} />. Solved with finite differences (forward Euler + 3-point second derivative).
        </p>
      </div>

      <div className={styles.editorBlock}>
        <h3 className={styles.optionsTitle}>Analytical solution</h3>
        <p className={styles.introText}>
          With <span dangerouslySetInnerHTML={{ __html: renderLatex('u(x,0)=f(x)') }} />, separation of variables gives{' '}
          <span dangerouslySetInnerHTML={{ __html: renderLatex('u(x,t) = \\sum_{m=1}^\\infty A_m \\sin(m\\pi x)\\,e^{-\\alpha\\pi^2 m^2 t}') }} />,
          where <span dangerouslySetInnerHTML={{ __html: renderLatex('A_m = 2\\int_0^1 f(x)\\,\\sin(m\\pi x)\\,dx') }} />. As{' '}
          <span dangerouslySetInnerHTML={{ __html: renderLatex('t\\to\\infty') }} />, <span dangerouslySetInnerHTML={{ __html: renderLatex('u\\to 0') }} /> everywhere (uniform temperature).
        </p>
      </div>

      <div className={styles.editorBlock}>
        <h3 className={styles.optionsTitle}>Parameters</h3>
        <div className={styles.theoreticalForm}>
          <label className={styles.fieldLabel}>
            <span>Grid points N</span>
            <input
              type="number"
              min={20}
              max={200}
              value={N}
              onChange={(e) => setN(Math.max(20, Math.min(200, Number(e.target.value) || 50)))}
              className={styles.input}
            />
          </label>
          <label className={styles.fieldLabel}>
            <span>Diffusivity α</span>
            <input
              type="number"
              min={0.01}
              step={0.05}
              value={alpha}
              onChange={(e) => setAlpha(Math.max(0.01, Number(e.target.value) || 0.2))}
              className={styles.input}
            />
          </label>
          <label className={styles.fieldLabel}>
            <span>Time step dt</span>
            <input
              type="number"
              min={0.0001}
              step={0.0005}
              value={dt}
              onChange={(e) => setDt(Math.max(0.0001, Number(e.target.value) || 0.001))}
              className={styles.input}
            />
          </label>
          <label className={styles.fieldLabel}>
            <span>Total time T</span>
            <input
              type="number"
              min={0.1}
              step={0.5}
              value={T}
              onChange={(e) => setT(Math.max(0.1, Number(e.target.value) || 1))}
              className={styles.input}
            />
          </label>
          <label className={styles.fieldLabel}>
            <span>Initial condition</span>
            <select
              value={initial}
              onChange={(e) => setInitial(e.target.value as InitialCondition1d)}
              className={styles.input}
            >
              {INITIAL_OPTIONS.map((opt) => (
                <option key={opt.id} value={opt.id}>
                  {opt.label}
                </option>
              ))}
            </select>
          </label>
          <button type="button" className={styles.runBtn} onClick={runSimulation}>
            Run
          </button>
        </div>
        <p className={styles.hint}>
          Stability: dt ≤ h²/(2α) ≈ {stabilityHint.maxDt.toExponential(3)}.
          {!stabilityHint.stable && (
            <span style={{ color: 'var(--danger)' }}> Solver will use a smaller step if needed.</span>
          )}
        </p>
      </div>

      {result && (
        <div className={styles.graphBlock}>
          <h3 className={styles.graphTitle}>Solution u(x, t)</h3>
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={chartData} margin={{ top: 8, right: 8, left: 8, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis
                type="number"
                dataKey="x"
                domain={[0, 1]}
                stroke="var(--text-muted)"
                tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
              />
              <YAxis
                type="number"
                stroke="var(--text-muted)"
                tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                domain={yDomain}
              />
              <Tooltip
                contentStyle={{
                  background: 'var(--glass-bg)',
                  border: '1px solid var(--glass-border)',
                  borderRadius: 'var(--radius)',
                }}
                formatter={(value: number) => [value.toFixed(4), 'u']}
                labelFormatter={(x) => `x = ${Number(x).toFixed(3)}`}
              />
              <Line
                type="monotone"
                dataKey="u"
                stroke="var(--accent)"
                dot={false}
                strokeWidth={2}
                name="u"
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
          <div className={styles.theoreticalForm} style={{ flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center', gap: '1rem', marginTop: '0.75rem' }}>
            <label className={styles.fieldLabel}>
              <span>Time t = {currentTime.toFixed(3)}</span>
              <input
                type="range"
                min={0}
                max={numSnapshots - 1}
                value={currentIndex}
                onChange={(e) => setSnapshotIndex(Number(e.target.value))}
                className={styles.input}
                style={{ width: 220 }}
              />
            </label>
            <button type="button" className={styles.runBtn} onClick={() => setPlaying((p) => !p)}>
              {playing ? 'Pause' : 'Play'}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
