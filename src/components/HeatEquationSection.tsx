import { useState, useMemo, useCallback, useRef, useEffect } from 'react'
import katex from 'katex'
import 'katex/dist/katex.min.css'
import {
  solveHeatEquation,
  getCell,
  type HeatConfig,
  type HeatResult,
  type InitialCondition,
} from '@/lib/heatEquation'
import styles from './MarkovChainSection.module.css'

function renderLatex(latex: string, displayMode = false): string {
  try {
    return katex.renderToString(latex, { displayMode, throwOnError: false })
  } catch {
    return latex
  }
}

const INITIAL_OPTIONS: { id: InitialCondition; label: string }[] = [
  { id: 'point', label: 'Point source (center)' },
  { id: 'half', label: 'Half hot / half cold' },
  { id: 'corner', label: 'Hot corner' },
  { id: 'two-spots', label: 'Two spots' },
]

/** Map value in [0,1] to CSS color (blue -> cyan -> white -> yellow -> red). */
function tempToColor(t: number): string {
  if (!isFinite(t) || t <= 0) return 'rgb(32, 32, 128)'
  if (t >= 1) return 'rgb(180, 0, 0)'
  const r = t <= 0.5 ? Math.round(2 * t * 255) : 255
  const g = t <= 0.5 ? Math.round(100 + 155 * 2 * t) : Math.round(255 - 255 * 2 * (t - 0.5))
  const b = t <= 0.5 ? 255 : Math.round(255 - 255 * 2 * (t - 0.5))
  return `rgb(${r},${g},${b})`
}

const CANVAS_SIZE = 400

function HeatmapCanvas({
  result,
  snapshotIndex,
}: {
  result: HeatResult
  snapshotIndex: number
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const gridSize = result.gridSize

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || snapshotIndex < 0 || snapshotIndex >= result.snapshots.length) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    const w = CANVAS_SIZE
    const h = CANVAS_SIZE
    canvas.width = w * dpr
    canvas.height = h * dpr
    ctx.scale(dpr, dpr)

    // Fixed scale: use max from initial condition so decay to zero shows as cooling to blue
    let scaleMax = 0
    const initial = result.snapshots[0]
    for (let i = 0; i < initial.length; i++) {
      const v = initial[i]
      if (isFinite(v) && v > scaleMax) scaleMax = v
    }
    if (scaleMax <= 0) scaleMax = 1

    const cellW = w / gridSize
    const cellH = h / gridSize

    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        const v = getCell(result, snapshotIndex, i, j)
        const t = scaleMax > 0 ? Math.min(1, Math.max(0, v / scaleMax)) : 0
        ctx.fillStyle = tempToColor(t)
        ctx.fillRect(j * cellW, i * cellH, cellW + 0.5, cellH + 0.5)
      }
    }
  }, [result, snapshotIndex])

  return (
    <canvas
      ref={canvasRef}
      width={CANVAS_SIZE}
      height={CANVAS_SIZE}
      style={{ width: CANVAS_SIZE, height: CANVAS_SIZE, display: 'block', borderRadius: 'var(--radius)' }}
      aria-label="2D temperature field"
    />
  )
}

export function HeatEquationSection() {
  const [N, setN] = useState(40)
  const [alpha, setAlpha] = useState(0.2)
  const [dt, setDt] = useState(0.001)
  const [T, setT] = useState(1)
  const [initial, setInitial] = useState<InitialCondition>('point')
  const [result, setResult] = useState<HeatResult | null>(null)
  const [snapshotIndex, setSnapshotIndex] = useState(0)
  const [playing, setPlaying] = useState(false)
  const rafRef = useRef<number | null>(null)
  const lastTickRef = useRef(0)
  const accumulatedRef = useRef(0)
  const STEPS_PER_SECOND = 20

  const runSimulation = useCallback(() => {
    const config: HeatConfig = { N, alpha, dt, T, initial }
    const res = solveHeatEquation(config)
    setResult(res)
    setSnapshotIndex(0)
    setPlaying(false)
  }, [N, alpha, dt, T, initial])

  const numSnapshots = result?.snapshots.length ?? 0
  const currentIndex = result
    ? Math.max(0, Math.min(snapshotIndex, numSnapshots - 1))
    : 0
  const currentTime = result?.times[currentIndex] ?? 0

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
    const maxDt = (h * h) / (4 * alpha)
    return { maxDt, stable: dt <= maxDt * 1.01 }
  }, [N, alpha, dt])

  return (
    <div className={styles.section}>
      <div className={styles.intro}>
        <p className={styles.introText}>
          The <strong>2D heat equation</strong> is{' '}
          <span
            dangerouslySetInnerHTML={{
              __html: renderLatex('\\frac{\\partial u}{\\partial t} = \\alpha \\nabla^2 u', true),
            }}
          />
          {' '}where <span dangerouslySetInnerHTML={{ __html: renderLatex('u(x,y,t)') }} /> is temperature,{' '}
          <span dangerouslySetInnerHTML={{ __html: renderLatex('\\alpha') }} /> is thermal diffusivity, and{' '}
          <span dangerouslySetInnerHTML={{ __html: renderLatex('\\nabla^2 u = \\frac{\\partial^2 u}{\\partial x^2} + \\frac{\\partial^2 u}{\\partial y^2}') }} />.
          We solve it on the unit square with zero boundary conditions using finite differences (forward Euler + 5-point Laplacian).
        </p>
      </div>

      <div className={styles.editorBlock}>
        <h3 className={styles.optionsTitle}>Analytical solution</h3>
        <p className={styles.introText}>
          On the unit square with <span dangerouslySetInnerHTML={{ __html: renderLatex('u=0') }} /> on the boundary and initial condition{' '}
          <span dangerouslySetInnerHTML={{ __html: renderLatex('u(x,y,0) = f(x,y)') }} />, separation of variables gives the Fourier series solution:
        </p>
        <div className={styles.introFormula} style={{ marginTop: '0.5rem', marginBottom: '0.5rem' }}>
          <span
            dangerouslySetInnerHTML={{
              __html: renderLatex('u(x,y,t) = \\sum_{m=1}^\\infty \\sum_{n=1}^\\infty A_{mn} \\sin(m\\pi x)\\,\\sin(n\\pi y)\\,e^{-\\alpha\\pi^2(m^2+n^2)t}', true),
            }}
          />
        </div>
        <p className={styles.hint}>
          The coefficients are the double Fourier sine coefficients of the initial data:{' '}
          <span dangerouslySetInnerHTML={{ __html: renderLatex('A_{mn} = 4\\int_0^1\\!\\!\\int_0^1 f(x,y)\\,\\sin(m\\pi x)\\,\\sin(n\\pi y)\\,dx\\,dy') }} />.
          Each mode decays exponentially at rate <span dangerouslySetInnerHTML={{ __html: renderLatex('\\alpha\\pi^2(m^2+n^2)') }} />; higher modes decay faster, so the solution smooths out over time.
          With zero boundary conditions, <span dangerouslySetInnerHTML={{ __html: renderLatex('u\\to 0') }} /> everywhere as <span dangerouslySetInnerHTML={{ __html: renderLatex('t\\to\\infty') }} />—the temperature tends to a <strong>uniform</strong> value (the boundary temperature).
        </p>
      </div>

      <div className={styles.editorBlock}>
        <h3 className={styles.optionsTitle}>Parameters</h3>
        <div className={styles.theoreticalForm}>
          <label className={styles.fieldLabel}>
            <span>Grid size N</span>
            <input
              type="number"
              min={10}
              max={80}
              value={N}
              onChange={(e) => setN(Math.max(10, Math.min(80, Number(e.target.value) || 20)))}
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
              onChange={(e) => setInitial(e.target.value as InitialCondition)}
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
          Stability: dt ≤ h²/(4α) ≈ {stabilityHint.maxDt.toExponential(3)}.
          {!stabilityHint.stable && (
            <span style={{ color: 'var(--danger)' }}> Current dt may be unstable; solver will use a smaller step.</span>
          )}
        </p>
      </div>

      {result && (
        <div className={styles.graphBlock}>
          <h3 className={styles.graphTitle}>Temperature u(x, y, t)</h3>
          <HeatmapCanvas result={result} snapshotIndex={currentIndex} />
          <div className={styles.heatKey}>
            <span className={styles.heatKeyLabel}>Cold (low u)</span>
            <div
              className={styles.heatKeyBar}
              style={{
                background: 'linear-gradient(to right, rgb(32,32,128), rgb(128,200,255), rgb(255,255,255), rgb(255,220,100), rgb(180,0,0))',
              }}
            />
            <span className={styles.heatKeyLabel}>Hot (high u)</span>
          </div>
          <p className={styles.hint} style={{ marginTop: '0.35rem' }}>
            Scale is fixed to the initial max so you see the solution decay to uniform (cold) as <span dangerouslySetInnerHTML={{ __html: renderLatex('t\\to\\infty') }} />.
          </p>
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
            <button
              type="button"
              className={styles.runBtn}
              onClick={() => setPlaying((p) => !p)}
            >
              {playing ? 'Pause' : 'Play'}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
