import { useState, useMemo } from 'react'
import katex from 'katex'
import 'katex/dist/katex.min.css'
import { computeLPDual, type LPPrimal } from '@/lib/duality'
import styles from './MarkovChainSection.module.css'

function tex(src: string, display = false): string {
  try {
    return katex.renderToString(src, { displayMode: display, throwOnError: false })
  } catch {
    return src
  }
}

function fmt(v: number): string {
  if (!isFinite(v)) return String(v)
  if (Math.abs(v) < 1e-8) return '0'
  return String(parseFloat(v.toPrecision(5)))
}

function formatLPPrimal(p: LPPrimal): string {
  const cStr = p.c.map((v, j) => `${fmt(v)}x_{${j + 1}}`).join(' + ') || '0'
  const rows = p.A.map((row, i) => {
    const lhs = row.map((a, j) => `${fmt(a)}x_{${j + 1}}`).join(' + ') || '0'
    return `${lhs} \\ge ${fmt(p.b[i])}`
  })
  return `\\min \\; ${cStr} \\quad \\text{s.t.} \\quad ${rows.join(' \\\\ ')}`
}

function formatLPDual(dual: ReturnType<typeof computeLPDual>): string {
  const objStr = dual.obj.map((v, i) => `${fmt(v)}\\lambda_{${i + 1}}`).join(' + ') || '0'
  const rows = dual.eqA.map((row, j) => {
    const lhs = row.map((a, i) => `${fmt(a)}\\lambda_{${i + 1}}`).join(' + ') || '0'
    return `${lhs} = ${fmt(dual.eqB[j])}`
  })
  return `\\max \\; ${objStr} \\quad \\text{s.t.} \\quad ${rows.join(' \\\\ ')} \\quad \\lambda \\ge 0`
}

/** Parse "1, 2, 3" or "1 2 3" into numbers; null if invalid */
function parseRow(s: string): number[] | null {
  const parts = s.split(/[\s,]+/).filter(Boolean)
  const nums = parts.map((p) => Number(p.trim()))
  return nums.every((n) => !isNaN(n)) ? nums : null
}

const PRESETS: { label: string; problem: LPPrimal }[] = [
  { label: '2×2', problem: { c: [2, 3], A: [[1, 1], [2, 1]], b: [4, 6] } },
  { label: 'Min norm', problem: { c: [0, 0], A: [[1, 1]], b: [1] } },
  { label: '3 constraints', problem: { c: [1, 1], A: [[1, 0], [0, 1], [1, 1]], b: [0, 0, 1] } },
]

function problemFromInputs(cStr: string, aStr: string, bStr: string): LPPrimal | null {
  const c = parseRow(cStr)
  const b = parseRow(bStr)
  const rawRows = aStr.trim().split('\n').map((line) => parseRow(line))
  const rows = rawRows.filter((r): r is number[] => r !== null && r.length > 0)
  if (!c || !b || rows.length === 0 || rows.length !== b.length) return null
  const n = c.length
  if (rows.some((r) => r.length !== n)) return null
  return { c, A: rows, b }
}

export function DualitySection() {
  const [preset, setPreset] = useState(0)
  const [cStr, setCStr] = useState(PRESETS[0].problem.c.join(', '))
  const [aStr, setAStr] = useState(PRESETS[0].problem.A.map((r) => r.join(', ')).join('\n'))
  const [bStr, setBStr] = useState(PRESETS[0].problem.b.join(', '))

  const problem = useMemo(
    () => problemFromInputs(cStr, aStr, bStr),
    [cStr, aStr, bStr]
  )
  const dual = useMemo(() => (problem ? computeLPDual(problem) : null), [problem])

  function loadPreset(i: number) {
    setPreset(i)
    const p = PRESETS[i].problem
    setCStr(p.c.join(', '))
    setAStr(p.A.map((r) => r.join(', ')).join('\n'))
    setBStr(p.b.join(', '))
  }

  return (
    <div className={styles.section}>
      <p className={styles.introText}>
        Primal: <strong>min c′x</strong> s.t. <strong>Ax ≥ b</strong>. Dual: <strong>max b′λ</strong> s.t. <strong>A′λ = c</strong>, λ ≥ 0.
      </p>

      <div className={styles.editorBlock}>
        <label className={styles.label}>Preset</label>
        <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
          {PRESETS.map((p, i) => (
            <button
              key={i}
              type="button"
              className={styles.runBtn}
              onClick={() => loadPreset(i)}
              style={{ fontWeight: preset === i ? 600 : 'normal' }}
            >
              {p.label}
            </button>
          ))}
        </div>
        <label className={styles.label}>c (objective, comma-separated)</label>
        <input
          type="text"
          className={styles.textarea}
          value={cStr}
          onChange={(e) => setCStr(e.target.value)}
          placeholder="e.g. 2, 3"
          style={{ minHeight: 'auto', padding: '0.5rem' }}
        />
        <label className={styles.label}>A (one row per line, comma-separated)</label>
        <textarea
          className={styles.textarea}
          value={aStr}
          onChange={(e) => setAStr(e.target.value)}
          placeholder={'1, 1\n2, 1'}
          rows={3}
        />
        <label className={styles.label}>b (RHS, comma-separated)</label>
        <input
          type="text"
          className={styles.textarea}
          value={bStr}
          onChange={(e) => setBStr(e.target.value)}
          placeholder="e.g. 4, 6"
          style={{ minHeight: 'auto', padding: '0.5rem' }}
        />
      </div>

      {problem && dual && (
        <>
          <div className={styles.graphBlock}>
            <h3 className={styles.graphTitle}>Primal</h3>
            <p className={styles.introFormula} dangerouslySetInnerHTML={{ __html: tex(formatLPPrimal(problem), true) }} />
          </div>
          <div className={styles.graphBlock}>
            <h3 className={styles.graphTitle}>Dual</h3>
            <p className={styles.introFormula} dangerouslySetInnerHTML={{ __html: tex(formatLPDual(dual), true) }} />
          </div>
        </>
      )}
      {!problem && (cStr.trim() || aStr.trim() || bStr.trim()) && (
        <p className={styles.hint}>Enter valid numbers: c and b comma-separated; A one row per line, comma-separated. Row lengths must match c.</p>
      )}
    </div>
  )
}
