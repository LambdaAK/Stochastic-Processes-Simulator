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
import { solveSimplex, feasibleVertices, type LPProblem, type ConstraintSign } from '@/lib/simplex'
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
  problem: LPProblem
}

const BLANK_PROBLEM: LPProblem = {
  direction: 'max',
  c: [0, 0],
  A: [[0, 0], [0, 0]],
  b: [0, 0],
  signs: ['<=', '<='],
}

const PRESETS: Preset[] = [
  {
    label: 'Max profit (2 vars)',
    desc: 'Classic 2-variable production problem',
    problem: {
      direction: 'max',
      c: [3, 5],
      A: [[1, 0], [0, 2], [3, 5]],
      b: [4, 12, 25],
      signs: ['<=', '<=', '<='],
    },
  },
  {
    label: 'Min cost (3 vars)',
    desc: 'Simple 3-variable minimisation',
    problem: {
      direction: 'min',
      c: [2, 3, 1],
      A: [[1, 1, 1], [2, 1, 0], [0, 1, 3]],
      b: [6, 8, 9],
      signs: ['<=', '<=', '<='],
    },
  },
  {
    label: 'Mixed signs (2 vars)',
    desc: '≥ constraint — Big-M artificial variable',
    problem: {
      direction: 'min',
      c: [1, 2],
      A: [[1, 1], [2, 1]],
      b: [4, 6],
      signs: ['>=', '>='],
    },
  },
]

// ─── Types ────────────────────────────────────────────────────────────────────

type ProblemState = {
  direction: 'min' | 'max'
  nVars: number
  nCons: number
  c: string[]
  A: string[][]
  b: string[]
  signs: ConstraintSign[]
}

function defaultState(p: LPProblem): ProblemState {
  return {
    direction: p.direction,
    nVars: p.c.length,
    nCons: p.b.length,
    c: p.c.map(String),
    A: p.A.map((row) => row.map(String)),
    b: p.b.map(String),
    signs: [...p.signs],
  }
}

function stateToNumbers(s: ProblemState): LPProblem | null {
  const c = s.c.slice(0, s.nVars).map(Number)
  const b = s.b.slice(0, s.nCons).map(Number)
  const A = s.A.slice(0, s.nCons).map((row) => row.slice(0, s.nVars).map(Number))
  if ([...c, ...b, ...A.flat()].some(isNaN)) return null
  return { direction: s.direction, c, A, b, signs: s.signs.slice(0, s.nCons) }
}

// ─── Component ────────────────────────────────────────────────────────────────

export function SimplexSection() {
  const [state, setState] = useState<ProblemState>(defaultState(PRESETS[0].problem))
  const [result, setResult] = useState<ReturnType<typeof solveSimplex> | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [showTableau, setShowTableau] = useState(false)

  // ── Derived LP formula ────────────────────────────────────────────────────

  const formulaHtml = useMemo(() => {
    const terms = state.c
      .slice(0, state.nVars)
      .map((ci, j) => `${ci ?? 0}x_{${j + 1}}`)
      .join(' + ')
    const dir = state.direction === 'max' ? '\\text{maximize}' : '\\text{minimize}'
    const constraintLines = state.A.slice(0, state.nCons)
      .map((row, i) => {
        const rowTerms = row
          .slice(0, state.nVars)
          .map((aij, j) => `${aij ?? 0}x_{${j + 1}}`)
          .join(' + ')
        const sign = state.signs[i] === '<=' ? '\\le' : state.signs[i] === '>=' ? '\\ge' : '='
        return `${rowTerms} ${sign} ${state.b[i] ?? 0}`
      })
      .join(' \\\\ ')
    const src = `${dir} \\quad ${terms} \\\\ \\text{s.t.} \\quad \\begin{cases} ${constraintLines} \\\\ x_j \\ge 0 \\end{cases}`
    return latex(src, true)
  }, [state])

  // ── Handlers ──────────────────────────────────────────────────────────────

  function loadPreset(idx: number) {
    setState(defaultState(PRESETS[idx].problem))
    setResult(null)
    setError(null)
  }

  function setNVars(n: number) {
    const nv = Math.max(1, Math.min(6, n))
    setState((s) => {
      const c = [...s.c]
      while (c.length < nv) c.push('0')
      const A = s.A.map((row) => {
        const r = [...row]
        while (r.length < nv) r.push('0')
        return r
      })
      return { ...s, nVars: nv, c, A }
    })
    setResult(null)
  }

  function setNCons(n: number) {
    const nc = Math.max(1, Math.min(8, n))
    setState((s) => {
      const b = [...s.b]
      const A = [...s.A]
      const signs = [...s.signs]
      while (b.length < nc) b.push('0')
      while (A.length < nc) A.push(new Array(s.nVars).fill('0'))
      while (signs.length < nc) signs.push('<=')
      return { ...s, nCons: nc, b, A, signs }
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

  function setSign(i: number, val: ConstraintSign) {
    setState((s) => {
      const signs = [...s.signs]
      signs[i] = val
      return { ...s, signs }
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
    const res = solveSimplex(prob)
    setResult(res)
  }

  // ── 2-D feasibility chart data ────────────────────────────────────────────

  const chartData = useMemo(() => {
    if (state.nVars !== 2) return null
    const prob = stateToNumbers(state)
    if (!prob) return null

    const verts = feasibleVertices(prob.A, prob.b, prob.signs)
    if (verts.length < 3) return null

    // Close polygon
    const polygon = [...verts, verts[0]]

    // Optimal point
    const optPt =
      result?.status === 'optimal' && result.variables
        ? [{ x: result.variables[0], y: result.variables[1] }]
        : []

    return { polygon, optPt, verts }
  }, [state, result])

  const xMax = useMemo(() => {
    if (!chartData) return 10
    const xs = chartData.verts.map((v) => v.x)
    return Math.ceil(Math.max(...xs, 1) * 1.2)
  }, [chartData])

  const yMax = useMemo(() => {
    if (!chartData) return 10
    const ys = chartData.verts.map((v) => v.y)
    return Math.ceil(Math.max(...ys, 1) * 1.2)
  }, [chartData])

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className={styles.section}>
      {/* Intro */}
      <div className={styles.intro}>
        <p className={styles.introText}>
          <strong>Linear Programming</strong> — optimize a linear objective subject to linear
          constraints. By the fundamental theorem of LP, if an optimum exists it is attained at
          a <em>vertex</em> (basic feasible solution) of the feasible polytope. The{' '}
          <strong>Simplex method</strong> moves between adjacent vertices, improving the objective
          at every step.
        </p>
        <p
          className={styles.introFormula}
          dangerouslySetInnerHTML={{
            __html: latex(
              '\\min_{x}\\; c^{\\!\\top} x \\quad \\text{s.t.}\\quad Ax = b,\\; x \\ge 0',
              true
            ),
          }}
        />
        <p className={styles.introText}>
          Adding slack variables <em>s</em> converts inequalities to equalities. At each
          iteration, the algorithm selects an <strong>entering variable</strong> with negative
          reduced cost and a <strong>leaving variable</strong> via the min-ratio test:
        </p>
        <p
          className={styles.introFormula}
          dangerouslySetInnerHTML={{
            __html: latex(
              '\\bar{c}_j = c_j - c_B^{\\!\\top} B^{-1} a_j < 0 \\quad \\text{(enter)} \\qquad r = \\arg\\min_i \\left\\{\\frac{\\bar{b}_i}{\\bar{a}_{ij}}\\;:\\;\\bar{a}_{ij}>0\\right\\} \\quad \\text{(leave)}',
              true
            ),
          }}
        />
        <p className={styles.introText}>
          Optimality is reached when{' '}
          <span dangerouslySetInnerHTML={{ __html: latex('\\bar{c}_j \\ge 0') }} /> for all
          non-basic <span dangerouslySetInnerHTML={{ __html: latex('j') }} />. This solver uses
          the <strong>Big-M method</strong> — artificial variables with large penalty{' '}
          <span dangerouslySetInnerHTML={{ __html: latex('M') }} /> — to find an initial basic
          feasible solution for ≥ and = constraints.
        </p>
      </div>

      {/* Presets */}
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
          <button
            type="button"
            className={styles.runBtn}
            onClick={() => {
              setState(defaultState(BLANK_PROBLEM))
              setResult(null)
              setError(null)
            }}
            style={{ fontWeight: 'normal', background: 'var(--bg-input)', color: 'var(--text)', border: '1px solid var(--border)' }}
          >
            Custom (blank)
          </button>
        </div>
      </div>

      {/* Problem definition */}
      <div className={styles.editorBlock}>
        <h3 className={styles.optionsTitle}>Problem</h3>

        {/* Direction + dimensions */}
        <div className={styles.theoreticalForm}>
          <label className={styles.fieldLabel}>
            <span>Direction</span>
            <select
              className={styles.input}
              value={state.direction}
              onChange={(e) => {
                setState((s) => ({ ...s, direction: e.target.value as 'min' | 'max' }))
                setResult(null)
              }}
            >
              <option value="min">Minimize</option>
              <option value="max">Maximize</option>
            </select>
          </label>
          <label className={styles.fieldLabel}>
            <span>Variables (n)</span>
            <input
              type="number"
              min={1}
              max={6}
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
              min={1}
              max={8}
              value={state.nCons}
              onChange={(e) => setNCons(Number(e.target.value))}
              className={styles.input}
              style={{ maxWidth: 80 }}
            />
          </label>
        </div>

        {/* 2-variable graph hint */}
        <p className={styles.hint} style={{ marginTop: '0.5rem' }}>
          {state.nVars === 2
            ? 'With 2 variables, the feasible region and optimal vertex are plotted automatically below.'
            : 'Set n\u202f=\u202f2 to visualize the feasible region and optimal vertex on a graph.'}
        </p>

        {/* Objective coefficients */}
        <div style={{ marginTop: '0.75rem' }}>
          <p
            className={styles.hint}
            style={{ marginBottom: '0.4rem' }}
            dangerouslySetInnerHTML={{
              __html: latex(
                `\\text{Objective: } ${state.direction === 'max' ? '\\text{maximize}' : '\\text{minimize}'} \\; c^\\top x`
              ),
            }}
          />
          <div className={styles.theoreticalForm}>
            {Array.from({ length: state.nVars }, (_, j) => (
              <label key={j} className={styles.fieldLabel}>
                <span
                  dangerouslySetInnerHTML={{ __html: latex(`c_{${j + 1}}`) }}
                />
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

        {/* Constraints */}
        <div style={{ marginTop: '0.75rem' }}>
          <p className={styles.hint} style={{ marginBottom: '0.4rem' }}>
            Constraints &nbsp;(A, sign, b)
          </p>
          {Array.from({ length: state.nCons }, (_, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'flex-end', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.5rem' }}>
              {Array.from({ length: state.nVars }, (_, j) => (
                <label key={j} className={styles.fieldLabel} style={{ minWidth: 0 }}>
                  <span
                    dangerouslySetInnerHTML={{ __html: latex(`a_{${i + 1}${j + 1}}`) }}
                  />
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
                <span>Sign</span>
                <select
                  className={styles.input}
                  value={state.signs[i] ?? '<='}
                  onChange={(e) => setSign(i, e.target.value as ConstraintSign)}
                  style={{ maxWidth: 70 }}
                >
                  <option value="<=">≤</option>
                  <option value=">=">&ge;</option>
                  <option value="=">=</option>
                </select>
              </label>
              <label className={styles.fieldLabel}>
                <span
                  dangerouslySetInnerHTML={{ __html: latex(`b_{${i + 1}}`) }}
                />
                <input
                  type="number"
                  step="any"
                  value={state.b[i] ?? '0'}
                  onChange={(e) => setB(i, e.target.value)}
                  className={styles.input}
                  style={{ maxWidth: 80 }}
                />
              </label>
            </div>
          ))}
        </div>

        {/* LP formula preview */}
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
        {error && <p className={styles.error}>{error}</p>}
      </div>

      {/* Result */}
      {result && (
        <div className={styles.graphBlock}>
          <h3 className={styles.graphTitle}>
            Result:{' '}
            <span
              style={{
                color:
                  result.status === 'optimal'
                    ? 'var(--accent)'
                    : result.status === 'unbounded'
                      ? 'var(--danger)'
                      : 'var(--danger)',
              }}
            >
              {result.status.toUpperCase()}
            </span>
          </h3>

          {result.status === 'optimal' && result.variables && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
              <p className={styles.hint}>
                Optimal value:{' '}
                <strong style={{ color: 'var(--text)' }}>{fmt(result.optimalValue!)}</strong>
              </p>
              <p className={styles.hint}>
                Solution:{' '}
                {result.variables.map((v, j) => (
                  <span key={j} style={{ marginRight: '1rem' }}>
                    <span
                      dangerouslySetInnerHTML={{ __html: latex(`x_{${j + 1}}`) }}
                    />
                    {' = '}
                    <strong style={{ color: 'var(--text)' }}>{fmt(v)}</strong>
                  </span>
                ))}
              </p>
              <p className={styles.hint}>
                Pivots performed: {result.pivotHistory.length}
              </p>
            </div>
          )}

          {result.status === 'infeasible' && (
            <p className={styles.hint}>
              The feasible region is empty — no solution satisfies all constraints simultaneously.
            </p>
          )}

          {result.status === 'unbounded' && (
            <p className={styles.hint}>
              The objective is unbounded — it can be improved indefinitely along some feasible
              direction.
            </p>
          )}
        </div>
      )}

      {/* Final tableau */}
      {result?.status === 'optimal' && result.tableau && (
        <div className={styles.graphBlock}>
          <h3 className={styles.graphTitle} style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            Final simplex tableau
            <button
              type="button"
              onClick={() => setShowTableau((v) => !v)}
              style={{
                fontSize: '0.8rem',
                padding: '0.2rem 0.6rem',
                background: 'var(--bg-input)',
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius)',
                cursor: 'pointer',
                color: 'var(--text-muted)',
              }}
            >
              {showTableau ? 'Hide' : 'Show'}
            </button>
          </h3>

          {showTableau && (
            <div style={{ overflowX: 'auto' }}>
              <table className={styles.matrixTable}>
                <thead>
                  <tr>
                    <th className={styles.matrixCorner}>Basis</th>
                    {result.tableau.varNames.map((name, j) => (
                      <th key={j} className={styles.matrixHeader}>
                        <span dangerouslySetInnerHTML={{ __html: latex(name.replace(/([xsa])(\d+)/, '$1_{$2}')) }} />
                      </th>
                    ))}
                    <th className={styles.matrixHeader}>RHS</th>
                  </tr>
                </thead>
                <tbody>
                  {result.tableau.data.map((row, i) => (
                    <tr key={i}>
                      <td className={styles.matrixRowHeader}>
                        {i === 0 ? (
                          <span dangerouslySetInnerHTML={{ __html: latex('z') }} />
                        ) : (
                          <span
                            dangerouslySetInnerHTML={{
                              __html: latex(
                                result.tableau!.varNames[result.tableau!.basis[i - 1]]?.replace(
                                  /([xsa])(\d+)/,
                                  '$1_{$2}'
                                ) ?? '?'
                              ),
                            }}
                          />
                        )}
                      </td>
                      {row.map((val, j) => (
                        <td key={j} className={styles.matrixCell}>
                          {fmt(val)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className={styles.matrixHint} style={{ marginTop: '0.4rem' }}>
                Row 0 is the objective row (reduced costs). Last column is the RHS.
              </p>
            </div>
          )}
        </div>
      )}

      {/* 2-D feasible region chart */}
      {state.nVars === 2 && chartData && chartData.verts.length >= 2 && (
        <div className={styles.graphBlock}>
          <h3 className={styles.graphTitle}>Feasible region (2-variable)</h3>
          <ResponsiveContainer width="100%" height={380}>
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
              {/* Constraint boundary reference lines */}
              {stateToNumbers(state)?.A.map((row, i) => {
                // Draw line: row[0]*x + row[1]*y = b[i]
                const a0 = row[0]
                const a1 = row[1]
                const bi = stateToNumbers(state)!.b[i]
                if (Math.abs(a1) < 1e-10) {
                  // Vertical line x = bi/a0
                  const xv = a0 !== 0 ? bi / a0 : 0
                  return <ReferenceLine key={i} x={xv} stroke="var(--accent)" strokeDasharray="4 3" strokeOpacity={0.6} />
                }
                // y = (bi - a0*x) / a1 — two points
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
                    strokeOpacity={0.6}
                  />
                )
              })}
              {/* Feasible polygon vertices */}
              <Scatter
                name="Feasible vertices"
                data={chartData.verts}
                fill="var(--accent)"
                fillOpacity={0.18}
                line={{ stroke: 'var(--accent)', strokeWidth: 1.5 }}
                lineType="joint"
                shape="circle"
                isAnimationActive={false}
              />
              {/* Optimal point */}
              {chartData.optPt.length > 0 && (
                <Scatter
                  name="Optimal"
                  data={chartData.optPt}
                  fill="#22c55e"
                  shape="star"
                  isAnimationActive={false}
                />
              )}
            </ScatterChart>
          </ResponsiveContainer>
          <p className={styles.hint}>
            Dashed lines: constraint boundaries. Shaded polygon: feasible region vertices.
            {chartData.optPt.length > 0 && ' Green star: optimal point.'}
          </p>
        </div>
      )}
    </div>
  )
}
