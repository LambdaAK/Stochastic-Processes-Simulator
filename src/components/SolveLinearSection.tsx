import { useState, useMemo } from 'react'
import katex from 'katex'
import 'katex/dist/katex.min.css'
import { lu, solveLU } from '@/lib/matrixFactorizations'
import styles from './MarkovChainSection.module.css'

function tex(latex: string, displayMode = false): string {
  try {
    return katex.renderToString(latex, { displayMode, throwOnError: false })
  } catch {
    return latex
  }
}

function fmt(v: number): string {
  if (!Number.isFinite(v)) return String(v)
  if (Math.abs(v) < 1e-10) return '0'
  const s = v.toPrecision(5)
  return String(parseFloat(s))
}

function parseMatrix(text: string): number[][] | string {
  const lines = text
    .trim()
    .split(/\n/)
    .map((l) => l.trim())
    .filter(Boolean)
  if (lines.length === 0) return 'Enter at least one row.'
  const rows: number[][] = []
  let cols = -1
  for (let i = 0; i < lines.length; i++) {
    const parts = lines[i]
      .split(/[\s,;]+/)
      .map((s) => parseFloat(s))
      .filter((v) => Number.isFinite(v))
    if (parts.length === 0) return `Row ${i + 1}: no numbers found.`
    if (cols >= 0 && parts.length !== cols) return `Row ${i + 1}: expected ${cols} numbers, got ${parts.length}.`
    cols = parts.length
    rows.push(parts)
  }
  return rows
}

function parseVector(text: string, expectedLen: number): number[] | string {
  const parts = text
    .trim()
    .split(/[\s,;]+/)
    .map((s) => parseFloat(s))
    .filter((v) => Number.isFinite(v))
  if (parts.length !== expectedLen) return `Vector b must have exactly ${expectedLen} entries (one per row).`
  return parts
}

/** ‖Ax − b‖₂ */
function residual(A: number[][], x: number[], b: number[]): number {
  let sum = 0
  for (let i = 0; i < A.length; i++) {
    let Ax_i = 0
    for (let j = 0; j < x.length; j++) Ax_i += A[i][j] * x[j]
    sum += (Ax_i - b[i]) ** 2
  }
  return Math.sqrt(sum)
}

const DEFAULT_A = `2  1
1  -1`
const DEFAULT_B = `5
1`

const SVG_SIZE = 400
const PAD = 60

/** For 2×2: line a*x + b*y = c. Return two points for drawing (in data coords). */
function lineSegment(
  a: number,
  b: number,
  c: number,
  extent: number
): [[number, number], [number, number]] {
  const tol = 1e-12
  if (Math.abs(b) > tol) {
    // y = (c - a*x)/b; use x = -extent, x = +extent
    return [
      [-extent, (c + a * extent) / b],
      [extent, (c - a * extent) / b],
    ]
  }
  if (Math.abs(a) > tol) {
    // x = c/a (vertical line); use y = -extent, y = +extent
    const x0 = c / a
    return [
      [x0, -extent],
      [x0, extent],
    ]
  }
  return [[0, 0], [0, 0]]
}

function AxEqualsBViz({
  A,
  b,
  x,
}: {
  A: number[][]
  b: number[]
  x: number[]
}) {
  const extent = useMemo(() => {
    const xs = [x[0], x[0]]
    const ys = [x[1], x[1]]
    for (let i = 0; i < 2; i++) {
      const [[x1, y1], [x2, y2]] = lineSegment(A[i][0], A[i][1], b[i], 10)
      xs.push(x1, x2)
      ys.push(y1, y2)
    }
    const maxAbs = Math.max(
      Math.max(...xs.map(Math.abs)),
      Math.max(...ys.map(Math.abs)),
      1.2
    )
    return { maxAbs }
  }, [A, b, x])

  const scale = (SVG_SIZE - 2 * PAD) / (2 * extent.maxAbs)
  const cx = SVG_SIZE / 2
  const cy = SVG_SIZE / 2
  function toSvg(px: number, py: number): [number, number] {
    return [cx + scale * px, cy - scale * py]
  }

  const line1 = lineSegment(A[0][0], A[0][1], b[0], extent.maxAbs + 1)
  const line2 = lineSegment(A[1][0], A[1][1], b[1], extent.maxAbs + 1)
  const [x1a, y1a] = toSvg(line1[0][0], line1[0][1])
  const [x1b, y1b] = toSvg(line1[1][0], line1[1][1])
  const [x2a, y2a] = toSvg(line2[0][0], line2[0][1])
  const [x2b, y2b] = toSvg(line2[1][0], line2[1][1])
  const [sx, sy] = toSvg(x[0], x[1])

  return (
    <svg
      width="100%"
      height={SVG_SIZE}
      viewBox={`0 0 ${SVG_SIZE} ${SVG_SIZE}`}
      style={{ maxWidth: 420 }}
    >
      <line x1={0} y1={cy} x2={SVG_SIZE} y2={cy} stroke="var(--border)" strokeWidth={1} strokeDasharray="4 2" />
      <line x1={cx} y1={0} x2={cx} y2={SVG_SIZE} stroke="var(--border)" strokeWidth={1} strokeDasharray="4 2" />
      <line x1={x1a} y1={y1a} x2={x1b} y2={y1b} stroke="#0ea5e9" strokeWidth={2} />
      <line x1={x2a} y1={y2a} x2={x2b} y2={y2b} stroke="#10b981" strokeWidth={2} />
      <circle cx={sx} cy={sy} r={6} fill="var(--accent)" stroke="var(--text)" strokeWidth={1.5} />
    </svg>
  )
}

export function SolveLinearSection() {
  const [matrixText, setMatrixText] = useState(DEFAULT_A)
  const [bText, setBText] = useState(DEFAULT_B)

  const parseA = useMemo(() => {
    const p = parseMatrix(matrixText)
    return typeof p === 'string' ? { matrix: null, error: p } : { matrix: p, error: null }
  }, [matrixText])

  const parseB = useMemo((): { vec: number[]; error: null } | { vec: null; error: string } | null => {
    if (!parseA.matrix) return null
    const n = parseA.matrix.length
    const p = parseVector(bText, n)
    return typeof p === 'string' ? { vec: null, error: p } : { vec: p, error: null }
  }, [bText, parseA.matrix])

  const result = useMemo(() => {
    if (!parseA.matrix || !parseB?.vec) return null
    const A = parseA.matrix
    const b = parseB.vec
    if (A.length !== (A[0]?.length ?? 0)) return { error: 'Matrix must be square to solve Ax = b (use LU).' }
    try {
      const luResult = lu(A)
      const x = solveLU(luResult, b)
      const res = residual(A, x, b)
      return { x, residual: res, error: null }
    } catch (e) {
      return { error: e instanceof Error ? e.message : String(e) }
    }
  }, [parseA.matrix, parseB])

  const solution: { x: number[]; residual: number } | null =
    result && 'x' in result && result.x != null && result.residual != null
      ? { x: result.x, residual: result.residual }
      : null
  const is2x2 = parseA.matrix?.length === 2 && (parseA.matrix[0]?.length ?? 0) === 2

  return (
    <div className={styles.section}>
      <div className={styles.intro}>
        <p className={styles.introText}>
          Solve the linear system <span dangerouslySetInnerHTML={{ __html: tex('A\\mathbf{x} = \\mathbf{b}') }} />. Enter square matrix <span dangerouslySetInnerHTML={{ __html: tex('A') }} /> and vector <span dangerouslySetInnerHTML={{ __html: tex('\\mathbf{b}') }} />. The solver uses <strong>LU factorization with partial pivoting</strong>; solution <span dangerouslySetInnerHTML={{ __html: tex('\\mathbf{x}') }} /> and residual <span dangerouslySetInnerHTML={{ __html: tex('\\|A\\mathbf{x} - \\mathbf{b}\\|_2') }} /> are shown. For 2×2 systems you get a plot of the two lines and their intersection.
        </p>
      </div>

      <div className={styles.editorBlock}>
        <label className={styles.label}>Matrix A (square; one row per line)</label>
        <textarea
          className={styles.textarea}
          value={matrixText}
          onChange={(e) => setMatrixText(e.target.value)}
          rows={4}
          spellCheck={false}
        />
        <label className={styles.label} style={{ marginTop: '0.75rem' }}>Vector b (one entry per row of A; space or comma separated)</label>
        <input
          type="text"
          className={styles.input}
          value={bText}
          onChange={(e) => setBText(e.target.value)}
          style={{ width: '100%', maxWidth: '320px' }}
        />
      </div>

      {parseA.error && <p className={styles.error}>Matrix: {parseA.error}</p>}
      {parseB?.error && !parseA.error && <p className={styles.error}>{parseB.error}</p>}
      {result?.error && !parseA.error && parseB && !parseB.error && (
        <p className={styles.error}>{result.error}</p>
      )}

      {solution && parseA.matrix && parseB?.vec && (
        <>
          <div className={styles.matrixBlock}>
            <h4 className={styles.matrixTitle}>Solution x</h4>
            <p className={styles.matrixHint}>
              x = [{solution.x.map((v) => fmt(v)).join(', ')}]
            </p>
            <p className={styles.matrixHint}>
              Residual <span dangerouslySetInnerHTML={{ __html: tex('\\|A\\mathbf{x} - \\mathbf{b}\\|_2') }} /> = {solution.residual.toExponential(6)}
            </p>
          </div>
          {is2x2 && parseB.vec && (
            <div className={styles.matrixBlock}>
              <h4 className={styles.matrixTitle}>2D view</h4>
              <p className={styles.matrixHint}>
                Row 1: <span dangerouslySetInnerHTML={{ __html: tex('A_{1,1}x_1 + A_{1,2}x_2 = b_1') }} /> (blue). Row 2 (green). Dot: solution.
              </p>
              <AxEqualsBViz A={parseA.matrix} b={parseB.vec} x={solution.x} />
            </div>
          )}
        </>
      )}
    </div>
  )
}
