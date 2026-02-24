import { useState, useMemo } from 'react'
import katex from 'katex'
import 'katex/dist/katex.min.css'
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'
import { solveQP, type QPProblem } from '@/lib/qp'
import { feasibleVertices } from '@/lib/simplex'
import styles from './MarkovChainSection.module.css'

// ─── Helpers ──────────────────────────────────────────────────────────────────

function latex(src: string, display = false): string {
  try {
    return katex.renderToString(src, { displayMode: display, throwOnError: false })
  } catch {
    return src
  }
}

function fmt(v: number): string {
  if (!isFinite(v)) return String(v)
  if (Math.abs(v) < 1e-8) return '0'
  const s = v.toPrecision(6)
  return String(parseFloat(s))
}

// ─── Presets ──────────────────────────────────────────────────────────────────

type Preset = {
  label: string
  desc: string
  problem: QPProblem
}

const PRESETS: Preset[] = [
  {
    label: 'Nearest to (1, 1)',
    desc: 'Unconstrained: min (x₁−1)² + (x₂−1)². Optimal at x* = (1, 1), f* = 0.',
    problem: {
      Q: [[2, 0], [0, 2]],
      c: [-2, -2],
      A: [],
      b: [],
      nonNeg: false,
    },
  },
  {
    label: 'Box-constrained bowl',
    desc: 'min (x₁−3)² + (x₂−2)² s.t. x₁ ≤ 2, x₁,x₂ ≥ 0. Optimal x* = (2, 2), f* = 1.',
    problem: {
      Q: [[2, 0], [0, 2]],
      c: [-6, -4],
      A: [[1, 0]],
      b: [2],
      nonNeg: true,
    },
  },
  {
    label: 'SVM-like QP',
    desc: 'min ½‖x‖² − x₁ − x₂ s.t. x₁+x₂ ≤ 1, x ≥ 0. Optimal x* = (½, ½), f* = −¾.',
    problem: {
      Q: [[1, 0], [0, 1]],
      c: [-1, -1],
      A: [[1, 1]],
      b: [1],
      nonNeg: true,
    },
  },
  {
    label: 'Correlated Q',
    desc: 'min ½(4x₁²+4x₁x₂+3x₂²) s.t. x₁+x₂ ≤ 4, x ≥ 0.',
    problem: {
      Q: [[4, 2], [2, 3]],
      c: [0, 0],
      A: [[1, 1]],
      b: [4],
      nonNeg: true,
    },
  },
]

// ─── State types ──────────────────────────────────────────────────────────────

type ProblemState = {
  nVars: number
  nCons: number
  Q: string[][]
  c: string[]
  A: string[][]
  b: string[]
  nonNeg: boolean
}

function defaultState(p: QPProblem): ProblemState {
  const n = p.c.length
  const m = p.b.length
  return {
    nVars: n,
    nCons: m,
    Q: p.Q.map((row) => row.map(String)),
    c: p.c.map(String),
    A: p.A.map((row) => row.map(String)),
    b: p.b.map(String),
    nonNeg: p.nonNeg,
  }
}

function stateToNumbers(s: ProblemState): QPProblem | null {
  const n = s.nVars
  const m = s.nCons
  const Q = s.Q.slice(0, n).map((row) => row.slice(0, n).map(Number))
  const c = s.c.slice(0, n).map(Number)
  const A = s.A.slice(0, m).map((row) => row.slice(0, n).map(Number))
  const b = s.b.slice(0, m).map(Number)
  if ([...Q.flat(), ...c, ...A.flat(), ...b].some(isNaN)) return null
  return { Q, c, A, b, nonNeg: s.nonNeg }
}

// ─── Component ────────────────────────────────────────────────────────────────

export function QPSection() {
  const [state, setState] = useState<ProblemState>(defaultState(PRESETS[0].problem))
  const [result, setResult] = useState<ReturnType<typeof solveQP> | null>(null)
  const [error, setError] = useState<string | null>(null)

  // ── Formula preview ───────────────────────────────────────────────────────

  const formulaHtml = useMemo(() => {
    const n = state.nVars
    const m = state.nCons

    // Build quadratic objective string
    const terms: string[] = []
    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) {
        const q = parseFloat(state.Q[i]?.[j] ?? '0')
        if (Math.abs(q) < 1e-12) continue
        const coeff = i === j ? q / 2 : q // ½Qᵢⱼ + ½Qⱼᵢ = Qᵢⱼ for off-diag (symmetric Q)
        const label = i === j ? `x_{${i + 1}}^2` : `x_{${i + 1}}x_{${j + 1}}`
        terms.push(`${fmt(coeff)}${label}`)
      }
    }
    for (let j = 0; j < n; j++) {
      const cj = parseFloat(state.c[j] ?? '0')
      if (Math.abs(cj) < 1e-12) continue
      terms.push(`${fmt(cj)}x_{${j + 1}}`)
    }
    const objStr = terms.length > 0 ? terms.join(' + ') : '0'

    const constraintLines: string[] = []
    for (let i = 0; i < m; i++) {
      const rowTerms = Array.from({ length: n }, (_, j) => {
        const a = parseFloat(state.A[i]?.[j] ?? '0')
        return Math.abs(a) < 1e-12 ? null : `${fmt(a)}x_{${j + 1}}`
      })
        .filter(Boolean)
        .join(' + ') || '0'
      constraintLines.push(`${rowTerms} \\le ${state.b[i] ?? 0}`)
    }
    if (state.nonNeg) constraintLines.push('x \\ge 0')

    const stStr =
      constraintLines.length > 0
        ? `\\text{s.t.} \\quad \\begin{cases} ${constraintLines.join(' \\\\ ')} \\end{cases}`
        : '\\text{(no constraints)}'

    return latex(`\\min_{x} \\quad ${objStr} \\\\ ${stStr}`, true)
  }, [state])

  // ── Handlers ──────────────────────────────────────────────────────────────

  function loadPreset(idx: number) {
    setState(defaultState(PRESETS[idx].problem))
    setResult(null)
    setError(null)
  }

  function setNVars(n: number) {
    const nv = Math.max(1, Math.min(4, n))
    setState((s) => {
      const c = [...s.c]
      while (c.length < nv) c.push('0')
      const Q = s.Q.map((row) => {
        const r = [...row]
        while (r.length < nv) r.push('0')
        return r
      })
      while (Q.length < nv) {
        const row = new Array(nv).fill('0')
        row[Q.length] = '1' // default diagonal = 1
        Q.push(row)
      }
      const A = s.A.map((row) => {
        const r = [...row]
        while (r.length < nv) r.push('0')
        return r
      })
      return { ...s, nVars: nv, c, Q, A }
    })
    setResult(null)
  }

  function setNCons(n: number) {
    const nc = Math.max(0, Math.min(6, n))
    setState((s) => {
      const b = [...s.b]
      const A = [...s.A]
      while (b.length < nc) b.push('1')
      while (A.length < nc) A.push(new Array(s.nVars).fill('0'))
      return { ...s, nCons: nc, b, A }
    })
    setResult(null)
  }

  function setQ(i: number, j: number, val: string) {
    setState((s) => {
      const Q = s.Q.map((r) => [...r])
      Q[i][j] = val
      Q[j][i] = val // enforce symmetry
      return { ...s, Q }
    })
    setResult(null)
  }

  function setC(j: number, val: string) {
    setState((s) => {
      const c = [...s.c]
      c[j] = val
      return { ...s, c }
    })
    setResult(null)
  }

  function setA(i: number, j: number, val: string) {
    setState((s) => {
      const A = s.A.map((r) => [...r])
      A[i][j] = val
      return { ...s, A }
    })
    setResult(null)
  }

  function setB(i: number, val: string) {
    setState((s) => {
      const b = [...s.b]
      b[i] = val
      return { ...s, b }
    })
    setResult(null)
  }

  function solve() {
    setError(null)
    const prob = stateToNumbers(state)
    if (!prob) {
      setError('All coefficients must be valid numbers.')
      return
    }
    // Basic PSD check: all diagonal entries of Q ≥ 0
    for (let i = 0; i < prob.Q.length; i++) {
      if (prob.Q[i][i] < -1e-9) {
        setError(`Q[${i + 1}][${i + 1}] = ${prob.Q[i][i]} is negative — Q must be positive semidefinite.`)
        return
      }
    }
    const res = solveQP(prob)
    setResult(res)
    if (res.message) setError(res.message)
  }

  // ── 2-D feasible region chart (n = 2 only) ────────────────────────────────

  const chartData = useMemo(() => {
    if (state.nVars !== 2) return null
    const prob = stateToNumbers(state)
    if (!prob) return null
    if (prob.A.length === 0 && !prob.nonNeg) return null

    // Build full inequality set for visualization (include non-negativity)
    const visA = [...prob.A]
    const visB = [...prob.b]
    const visSigns = prob.A.map(() => '<=' as const)

    if (prob.nonNeg) {
      visA.push([-1, 0])
      visB.push(0)
      visSigns.push('<=')
      visA.push([0, -1])
      visB.push(0)
      visSigns.push('<=')
    }

    const verts = feasibleVertices(visA, visB, visSigns)
    if (verts.length < 2) return null

    const polygon = [...verts, verts[0]]
    const optPt =
      result?.status === 'optimal' && result.x
        ? [{ x: result.x[0], y: result.x[1] }]
        : []

    // Unconstrained minimum (for reference)
    const uncon = solveQP({ ...prob, A: [], b: [], nonNeg: false })
    const unconPt = uncon.status === 'optimal' && uncon.x ? [{ x: uncon.x[0], y: uncon.x[1] }] : []

    return { polygon, verts, optPt, unconPt, visA, visB, prob }
  }, [state, result])

  const xMax = useMemo(() => {
    if (!chartData) return 5
    const xs = chartData.verts.map((v) => v.x)
    const optXs = chartData.optPt.map((p) => p.x)
    const uXs = chartData.unconPt.map((p) => p.x)
    return Math.ceil(Math.max(...xs, ...optXs, ...uXs, 1) * 1.3)
  }, [chartData])

  const yMax = useMemo(() => {
    if (!chartData) return 5
    const ys = chartData.verts.map((v) => v.y)
    const optYs = chartData.optPt.map((p) => p.y)
    const uYs = chartData.unconPt.map((p) => p.y)
    return Math.ceil(Math.max(...ys, ...optYs, ...uYs, 1) * 1.3)
  }, [chartData])

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className={styles.section}>
      {/* ── Intro ─────────────────────────────────────────────────────────── */}
      <div className={styles.intro}>
        <p className={styles.introText}>
          <strong>Quadratic Programming</strong> (QP) — minimize a convex quadratic objective
          subject to linear constraints. QPs appear throughout machine learning (SVMs, least-squares
          with penalties, model predictive control) and operations research.
        </p>
        <p
          className={styles.introFormula}
          dangerouslySetInnerHTML={{
            __html: latex(
              '\\min_{x \\ge 0} \\; \\tfrac{1}{2} x^{\\!\\top} Q x + c^{\\!\\top} x \\quad \\text{s.t.} \\quad Ax \\le b',
              true
            ),
          }}
        />
        <p className={styles.introText}>
          <strong>Q</strong> must be symmetric positive semidefinite (PSD) — this guarantees a
          convex objective and a unique global minimum. The solver uses the{' '}
          <strong>active-set method</strong>: it maintains a <em>working set</em> W of active
          (tight) inequality constraints and solves an equality-constrained QP (via KKT
          conditions) at each step.
        </p>
        <p
          className={styles.introFormula}
          dangerouslySetInnerHTML={{
            __html: latex(
              '\\underbrace{\\begin{bmatrix} Q & A_W^\\top \\\\ A_W & 0 \\end{bmatrix}}_{\\text{KKT matrix}} \\begin{bmatrix} d \\\\ \\lambda \\end{bmatrix} = \\begin{bmatrix} -g \\\\ 0 \\end{bmatrix}, \\quad g = Qx + c',
              true
            ),
          }}
        />
        <p className={styles.introText}>
          If the search direction{' '}
          <span dangerouslySetInnerHTML={{ __html: latex('d = 0') }} /> and all KKT multipliers{' '}
          <span dangerouslySetInnerHTML={{ __html: latex('\\lambda \\ge 0') }} />, the{' '}
          <strong>KKT conditions</strong> are satisfied and{' '}
          <span dangerouslySetInnerHTML={{ __html: latex('x') }} /> is optimal. Otherwise the
          working set is updated and the process repeats.
        </p>
      </div>

      {/* ── Presets ────────────────────────────────────────────────────────── */}
      <div className={styles.editorBlock}>
        <h3 className={styles.optionsTitle}>Presets</h3>
        <div className={styles.theoreticalForm}>
          {PRESETS.map((p, i) => (
            <button
              key={i}
              type="button"
              className={styles.runBtn}
              onClick={() => loadPreset(i)}
              style={{ fontWeight: 'normal' }}
            >
              {p.label}
            </button>
          ))}
        </div>
        {PRESETS.map((p, i) => {
          const active = state.nVars === p.problem.c.length &&
            JSON.stringify(stateToNumbers(state)) === JSON.stringify(p.problem)
          return active ? (
            <p key={i} className={styles.hint} style={{ marginTop: '0.25rem' }}>
              {p.desc}
            </p>
          ) : null
        })}
      </div>

      {/* ── Problem editor ────────────────────────────────────────────────── */}
      <div className={styles.editorBlock} style={{ maxWidth: '640px' }}>
        <h3 className={styles.optionsTitle}>Problem</h3>

        {/* Dimensions */}
        <div className={styles.theoreticalForm}>
          <label className={styles.fieldLabel}>
            <span>Variables (n)</span>
            <input
              type="number"
              min={1}
              max={4}
              value={state.nVars}
              onChange={(e) => setNVars(Number(e.target.value))}
              className={styles.input}
              style={{ maxWidth: 80 }}
            />
          </label>
          <label className={styles.fieldLabel}>
            <span>Constraints (m)</span>
            <input
              type="number"
              min={0}
              max={6}
              value={state.nCons}
              onChange={(e) => setNCons(Number(e.target.value))}
              className={styles.input}
              style={{ maxWidth: 80 }}
            />
          </label>
          <label className={styles.fieldLabel} style={{ flexDirection: 'row', alignItems: 'center', gap: '0.4rem' }}>
            <input
              type="checkbox"
              checked={state.nonNeg}
              onChange={(e) => {
                setState((s) => ({ ...s, nonNeg: e.target.checked }))
                setResult(null)
              }}
            />
            <span dangerouslySetInnerHTML={{ __html: latex('x \\ge 0') }} />
          </label>
        </div>

        <p className={styles.hint} style={{ marginTop: '0.25rem' }}>
          {state.nVars === 2
            ? 'With 2 variables, the feasible region and optimal point are plotted below.'
            : 'Set n = 2 to visualize the feasible region and solution on a chart.'}
        </p>

        {/* Q matrix */}
        <div style={{ marginTop: '0.75rem' }}>
          <p
            className={styles.hint}
            style={{ marginBottom: '0.4rem' }}
            dangerouslySetInnerHTML={{
              __html: latex('\\text{Hessian matrix } Q \\; (\\text{symmetric, enter upper triangle — lower auto-fills})'),
            }}
          />
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
            {Array.from({ length: state.nVars }, (_, i) => (
              <div key={i} style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
                {Array.from({ length: state.nVars }, (_, j) => (
                  <label key={j} className={styles.fieldLabel}>
                    <span dangerouslySetInnerHTML={{ __html: latex(`Q_{${i + 1}${j + 1}}`) }} />
                    <input
                      type="number"
                      step="any"
                      value={state.Q[i]?.[j] ?? '0'}
                      onChange={(e) => setQ(i, j, e.target.value)}
                      className={styles.input}
                      style={{ maxWidth: 72, opacity: j < i ? 0.5 : 1 }}
                      readOnly={j < i}
                    />
                  </label>
                ))}
              </div>
            ))}
          </div>
          <p className={styles.hint} style={{ marginTop: '0.3rem' }}>
            Lower-triangle entries mirror upper triangle (symmetry). Diagonal entries must be ≥ 0 (PSD).
          </p>
        </div>

        {/* c vector */}
        <div style={{ marginTop: '0.75rem' }}>
          <p
            className={styles.hint}
            style={{ marginBottom: '0.4rem' }}
            dangerouslySetInnerHTML={{ __html: latex('\\text{Linear cost vector } c') }}
          />
          <div className={styles.theoreticalForm}>
            {Array.from({ length: state.nVars }, (_, j) => (
              <label key={j} className={styles.fieldLabel}>
                <span dangerouslySetInnerHTML={{ __html: latex(`c_{${j + 1}}`) }} />
                <input
                  type="number"
                  step="any"
                  value={state.c[j] ?? '0'}
                  onChange={(e) => setC(j, e.target.value)}
                  className={styles.input}
                  style={{ maxWidth: 80 }}
                />
              </label>
            ))}
          </div>
        </div>

        {/* Inequality constraints */}
        {state.nCons > 0 && (
          <div style={{ marginTop: '0.75rem' }}>
            <p className={styles.hint} style={{ marginBottom: '0.4rem' }}>
              Inequality constraints (Ax ≤ b)
            </p>
            {Array.from({ length: state.nCons }, (_, i) => (
              <div
                key={i}
                style={{
                  display: 'flex',
                  alignItems: 'flex-end',
                  gap: '0.5rem',
                  flexWrap: 'wrap',
                  marginBottom: '0.5rem',
                }}
              >
                {Array.from({ length: state.nVars }, (_, j) => (
                  <label key={j} className={styles.fieldLabel} style={{ minWidth: 0 }}>
                    <span dangerouslySetInnerHTML={{ __html: latex(`a_{${i + 1}${j + 1}}`) }} />
                    <input
                      type="number"
                      step="any"
                      value={state.A[i]?.[j] ?? '0'}
                      onChange={(e) => setA(i, j, e.target.value)}
                      className={styles.input}
                      style={{ maxWidth: 70 }}
                    />
                  </label>
                ))}
                <label className={styles.fieldLabel}>
                  <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>≤</span>
                  <span dangerouslySetInnerHTML={{ __html: latex(`b_{${i + 1}}`) }} />
                  <input
                    type="number"
                    step="any"
                    value={state.b[i] ?? '1'}
                    onChange={(e) => setB(i, e.target.value)}
                    className={styles.input}
                    style={{ maxWidth: 80 }}
                  />
                </label>
              </div>
            ))}
          </div>
        )}

        {/* Formula preview */}
        <div
          className={styles.introFormula}
          style={{ marginTop: '0.5rem' }}
          dangerouslySetInnerHTML={{ __html: formulaHtml }}
        />

        <button
          type="button"
          className={styles.runBtn}
          onClick={solve}
          style={{ marginTop: '0.5rem', alignSelf: 'flex-start' }}
        >
          Solve
        </button>
        {error && (
          <p className={styles.hint} style={{ color: 'var(--danger, #ef4444)', marginTop: '0.4rem' }}>
            {error}
          </p>
        )}
      </div>

      {/* ── Result ────────────────────────────────────────────────────────── */}
      {result && (
        <div className={styles.graphBlock}>
          <h3 className={styles.graphTitle}>
            Result:{' '}
            <span
              style={{
                color: result.status === 'optimal' ? 'var(--accent)' : 'var(--danger, #ef4444)',
              }}
            >
              {result.status.toUpperCase().replace('_', ' ')}
            </span>
          </h3>

          {result.status === 'optimal' && result.x && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
              <p className={styles.hint}>
                Optimal value:{' '}
                <strong style={{ color: 'var(--text)' }}>{fmt(result.objectiveValue!)}</strong>
              </p>
              <p className={styles.hint}>
                Solution:{' '}
                {result.x.map((v, j) => (
                  <span key={j} style={{ marginRight: '1rem' }}>
                    <span dangerouslySetInnerHTML={{ __html: latex(`x_{${j + 1}}^*`) }} />
                    {' = '}
                    <strong style={{ color: 'var(--text)' }}>{fmt(v)}</strong>
                  </span>
                ))}
              </p>
              {result.iterations > 0 && (
                <p className={styles.hint}>Active-set iterations: {result.iterations}</p>
              )}
              {/* Constraint verification */}
              {(() => {
                const prob = stateToNumbers(state)
                if (!prob || !result.x) return null
                const violations = prob.A.map((row, i) => {
                  const lhs = row.reduce((s, a, j) => s + a * result.x![j], 0)
                  const slack = prob.b[i] - lhs
                  return { i, lhs, rhs: prob.b[i], slack, active: Math.abs(slack) < 1e-6 }
                })
                if (violations.length === 0) return null
                return (
                  <div style={{ marginTop: '0.3rem' }}>
                    <p className={styles.hint} style={{ marginBottom: '0.2rem' }}>Constraint check (Ax ≤ b):</p>
                    {violations.map(({ i, lhs, rhs, slack, active }) => (
                      <p key={i} className={styles.hint} style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>
                        {'  '}constraint {i + 1}: {fmt(lhs)} ≤ {fmt(rhs)}{' '}
                        <span style={{ color: active ? 'var(--accent)' : 'var(--text-muted)' }}>
                          (slack = {fmt(slack)}{active ? ', active' : ''})
                        </span>
                      </p>
                    ))}
                  </div>
                )
              })()}
            </div>
          )}

          {result.status === 'infeasible' && (
            <p className={styles.hint}>
              The feasible region is empty — no x satisfies all constraints simultaneously.
            </p>
          )}

          {result.status === 'max_iter' && result.x && (
            <p className={styles.hint}>
              Approximate solution after {result.iterations} iterations:{' '}
              {result.x.map((v, j) => (
                <span key={j} style={{ marginRight: '0.75rem' }}>
                  <span dangerouslySetInnerHTML={{ __html: latex(`x_{${j + 1}}`) }} />
                  {' ≈ '}
                  <strong>{fmt(v)}</strong>
                </span>
              ))}
            </p>
          )}
        </div>
      )}

      {/* ── 2-D chart (n = 2 only) ────────────────────────────────────────── */}
      {state.nVars === 2 && chartData && chartData.verts.length >= 2 && (
        <div className={styles.graphBlock}>
          <h3 className={styles.graphTitle}>Feasible region (2-variable)</h3>
          <ResponsiveContainer width="100%" height={400}>
            <ScatterChart margin={{ top: 16, right: 24, left: 16, bottom: 16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis
                type="number"
                dataKey="x"
                domain={[0, xMax]}
                name="x₁"
                label={{ value: 'x₁', position: 'insideBottomRight', offset: -8, fill: 'var(--text-muted)', fontSize: 12 }}
                stroke="var(--text-muted)"
                tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
              />
              <YAxis
                type="number"
                dataKey="y"
                domain={[0, yMax]}
                name="x₂"
                label={{ value: 'x₂', angle: -90, position: 'insideLeft', offset: 8, fill: 'var(--text-muted)', fontSize: 12 }}
                stroke="var(--text-muted)"
                tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
              />
              <Tooltip
                contentStyle={{
                  background: 'var(--glass-bg)',
                  border: '1px solid var(--glass-border)',
                  borderRadius: 'var(--radius)',
                }}
                formatter={(v: number) => [fmt(v)]}
                labelFormatter={(_l, payload) =>
                  payload?.[0]
                    ? `(${fmt(payload[0].payload?.x)}, ${fmt(payload[0].payload?.y)})`
                    : ''
                }
              />
              {/* Constraint boundary lines */}
              {chartData.prob.A.map((row, i) => {
                const a0 = row[0]
                const a1 = row[1]
                const bi = chartData.prob.b[i]
                if (Math.abs(a1) < 1e-10) {
                  const xv = a0 !== 0 ? bi / a0 : 0
                  return (
                    <ReferenceLine
                      key={i}
                      x={xv}
                      stroke="var(--accent)"
                      strokeDasharray="4 3"
                      strokeOpacity={0.7}
                      label={{ value: `c${i + 1}`, fill: 'var(--accent)', fontSize: 10 }}
                    />
                  )
                }
                const x0 = 0
                const y0 = (bi - a0 * x0) / a1
                const x1 = xMax
                const y1 = (bi - a0 * x1) / a1
                return (
                  <ReferenceLine
                    key={i}
                    segment={[{ x: x0, y: y0 }, { x: x1, y: y1 }]}
                    stroke="var(--accent)"
                    strokeDasharray="4 3"
                    strokeOpacity={0.7}
                  />
                )
              })}
              {/* Feasible polygon */}
              <Scatter
                name="Feasible region"
                data={chartData.polygon}
                fill="var(--accent)"
                fillOpacity={0.15}
                line={{ stroke: 'var(--accent)', strokeWidth: 1.5 }}
                lineType="joint"
                shape="circle"
                r={3}
                isAnimationActive={false}
              />
              {/* Unconstrained minimum */}
              {chartData.unconPt.length > 0 && (
                <Scatter
                  name="Unconstrained min"
                  data={chartData.unconPt}
                  fill="var(--text-muted)"
                  shape="diamond"
                  r={5}
                  isAnimationActive={false}
                />
              )}
              {/* Optimal point */}
              {chartData.optPt.length > 0 && (
                <Scatter
                  name="Optimal"
                  data={chartData.optPt}
                  fill="#22c55e"
                  shape="star"
                  r={7}
                  isAnimationActive={false}
                />
              )}
            </ScatterChart>
          </ResponsiveContainer>
          <p className={styles.hint}>
            Dashed lines: constraint boundaries. Shaded region: feasible set vertices.
            {chartData.unconPt.length > 0 && ' Diamond: unconstrained minimum.'}
            {chartData.optPt.length > 0 && ' Green star: optimal point x*.'}
          </p>
          <p className={styles.hint} style={{ marginTop: '0.3rem' }}>
            <em>
              Contour curves of the quadratic objective are ellipses centred at the
              unconstrained minimum. The optimal x* is where the smallest ellipse touches the
              feasible region.
            </em>
          </p>
        </div>
      )}

      {/* ── KKT explainer ─────────────────────────────────────────────────── */}
      <div className={styles.intro} style={{ maxWidth: '640px' }}>
        <h3 className={styles.graphTitle} style={{ marginBottom: '0.5rem' }}>KKT Optimality Conditions</h3>
        <p className={styles.introText}>
          At an optimal point x* the <strong>Karush–Kuhn–Tucker (KKT)</strong> conditions hold:
        </p>
        <p
          className={styles.introFormula}
          dangerouslySetInnerHTML={{
            __html: latex(
              '\\nabla f(x^*) + \\sum_i \\lambda_i \\nabla g_i(x^*) = 0 \\quad (\\text{stationarity})',
              true
            ),
          }}
        />
        <p
          className={styles.introFormula}
          dangerouslySetInnerHTML={{
            __html: latex(
              '\\lambda_i \\ge 0, \\quad g_i(x^*) \\le 0, \\quad \\lambda_i g_i(x^*) = 0 \\quad (\\text{complementarity})',
              true
            ),
          }}
        />
        <p className={styles.introText}>
          Complementarity means each constraint is either <em>active</em> (tight, λ &gt; 0) or
          the corresponding multiplier is zero. For convex QPs these conditions are also{' '}
          <em>sufficient</em> — any KKT point is a global minimum.
        </p>
      </div>
    </div>
  )
}
