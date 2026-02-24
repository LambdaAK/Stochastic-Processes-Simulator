import { useState, useMemo } from 'react'
import katex from 'katex'
import 'katex/dist/katex.min.css'
import { eigen2x2, apply2x2, unitCirclePoints } from '@/lib/eigenvalues'
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

const DEFAULT_MATRIX = `2  1
1  2`

function parse2x2(text: string): number[][] | string {
  const lines = text
    .trim()
    .split(/\n/)
    .map((l) => l.trim())
    .filter(Boolean)
  if (lines.length !== 2) return 'Enter exactly 2 rows.'
  const rows: number[][] = []
  for (let i = 0; i < 2; i++) {
    const parts = lines[i].split(/[\s,;]+/).map((s) => parseFloat(s)).filter((v) => Number.isFinite(v))
    if (parts.length !== 2) return `Row ${i + 1}: need 2 numbers.`
    rows.push([parts[0], parts[1]])
  }
  return rows
}

const SVG_SIZE = 400
const PAD = 60

function EigenViz({
  A: rawA,
  result,
}: {
  A: number[][]
  result: { values: [number, number]; vectors: number[][]; complex: boolean }
}) {
  const circlePts = useMemo(() => unitCirclePoints(80), [])
  const ellipsePts = useMemo(
    () => circlePts.map((p) => apply2x2(rawA, p as [number, number])),
    [rawA, circlePts]
  )
  const allX = useMemo(() => {
    const xs = [...circlePts.map((p) => p[0]), ...ellipsePts.map((p) => p[0])]
    return xs
  }, [circlePts, ellipsePts])
  const allY = useMemo(() => {
    const ys = [...circlePts.map((p) => p[1]), ...ellipsePts.map((p) => p[1])]
    return ys
  }, [circlePts, ellipsePts])
  const maxAbs = useMemo(() => {
    const mx = Math.max(...allX.map(Math.abs), 1)
    const my = Math.max(...allY.map(Math.abs), 1)
    return Math.max(mx, my, 1.2)
  }, [allX, allY])
  const scale = (SVG_SIZE - 2 * PAD) / (2 * maxAbs)
  const cx = SVG_SIZE / 2
  const cy = SVG_SIZE / 2
  function toSvg(x: number, y: number) {
    return `${cx + scale * x},${cy - scale * y}`
  }
  const circlePath = circlePts.map((p) => toSvg(p[0], p[1])).join(' L ')
  const ellipsePath = ellipsePts.map((p) => toSvg(p[0], p[1])).join(' L ')
  const arrowLen = 0.9
  return (
    <svg width="100%" height={SVG_SIZE} viewBox={`0 0 ${SVG_SIZE} ${SVG_SIZE}`} style={{ maxWidth: 420 }}>
      <line x1={0} y1={cy} x2={SVG_SIZE} y2={cy} stroke="var(--border)" strokeWidth={1} strokeDasharray="4 2" />
      <line x1={cx} y1={0} x2={cx} y2={SVG_SIZE} stroke="var(--border)" strokeWidth={1} strokeDasharray="4 2" />
      <path d={`M ${circlePath} Z`} fill="none" stroke="var(--text-muted)" strokeWidth={1.5} opacity={0.8} />
      <path d={`M ${ellipsePath} Z`} fill="none" stroke="var(--accent)" strokeWidth={2} />
      <defs>
        <marker id="arrowhead-eigen1" markerWidth={10} markerHeight={10} refX={9} refY={3} orient="auto">
          <path d="M0,0 L0,6 L9,3 z" fill="#0ea5e9" />
        </marker>
        <marker id="arrowhead-eigen2" markerWidth={10} markerHeight={10} refX={9} refY={3} orient="auto">
          <path d="M0,0 L0,6 L9,3 z" fill="#10b981" />
        </marker>
      </defs>
      {!result.complex &&
        result.vectors.length >= 2 &&
        result.vectors.map((v, i) => {
          const x = v[0] * arrowLen
          const y = v[1] * arrowLen
          const color = i === 0 ? '#0ea5e9' : '#10b981'
          const markerId = i === 0 ? 'arrowhead-eigen1' : 'arrowhead-eigen2'
          return (
            <line
              key={i}
              x1={cx}
              y1={cy}
              x2={cx + scale * x}
              y2={cy - scale * y}
              stroke={color}
              strokeWidth={2.5}
              markerEnd={`url(#${markerId})`}
            />
          )
        })}
      <circle cx={cx} cy={cy} r={4} fill="var(--text)" />
    </svg>
  )
}

export function EigenvaluesSection() {
  const [matrixText, setMatrixText] = useState(DEFAULT_MATRIX)

  const parseResult = useMemo(() => {
    const p = parse2x2(matrixText)
    return typeof p === 'string' ? { matrix: null, error: p } : { matrix: p, error: null }
  }, [matrixText])

  const result = useMemo(() => {
    if (!parseResult.matrix) return null
    try {
      return eigen2x2(parseResult.matrix)
    } catch (e) {
      return { error: e instanceof Error ? e.message : String(e) }
    }
  }, [parseResult.matrix])

  const eigenResult = result && !('error' in result) ? result : null

  return (
    <div className={styles.section}>
      <div className={styles.intro}>
        <p className={styles.introText}>
          For a <strong>2×2 matrix A</strong>, eigenvalues <span dangerouslySetInnerHTML={{ __html: tex('\\lambda') }} /> satisfy{' '}
          <span dangerouslySetInnerHTML={{ __html: tex('A\\mathbf{v} = \\lambda\\mathbf{v}') }} />. Enter <span dangerouslySetInnerHTML={{ __html: tex('A') }} /> below to get{' '}
          <span dangerouslySetInnerHTML={{ __html: tex('\\lambda_1, \\lambda_2') }} /> and eigenvectors. The unit circle and its image under <span dangerouslySetInnerHTML={{ __html: tex('A') }} /> show how the matrix stretches and rotates the plane; eigenvectors point along the stretch directions.
        </p>
      </div>

      <div className={styles.editorBlock}>
        <label className={styles.label}>Matrix A (2×2, one row per line)</label>
        <textarea
          className={styles.textarea}
          value={matrixText}
          onChange={(e) => setMatrixText(e.target.value)}
          rows={3}
          spellCheck={false}
        />
      </div>

      {parseResult.error && <p className={styles.error}>Matrix: {parseResult.error}</p>}
      {result && 'error' in result && !parseResult.error && <p className={styles.error}>{result.error}</p>}

      {eigenResult && parseResult.matrix && (
        <>
          <div className={styles.matrixBlock}>
            <h4 className={styles.matrixTitle}>Eigenvalues</h4>
            <div className={styles.matrixWrap}>
              <table className={styles.matrixTable}>
                <tbody>
                  {eigenResult.complex ? (
                    <>
                      <tr>
                        <th className={styles.matrixRowHeader}>λ₁</th>
                        <td className={styles.matrixCell}>
                          {fmt(eigenResult.values[0])} + {fmt(Math.abs(eigenResult.values[1]))}i
                        </td>
                      </tr>
                      <tr>
                        <th className={styles.matrixRowHeader}>λ₂</th>
                        <td className={styles.matrixCell}>
                          {fmt(eigenResult.values[0])} − {fmt(Math.abs(eigenResult.values[1]))}i
                        </td>
                      </tr>
                    </>
                  ) : (
                    <>
                      <tr>
                        <th className={styles.matrixRowHeader}>λ₁</th>
                        <td className={styles.matrixCell}>{fmt(eigenResult.values[0])}</td>
                      </tr>
                      <tr>
                        <th className={styles.matrixRowHeader}>λ₂</th>
                        <td className={styles.matrixCell}>{fmt(eigenResult.values[1])}</td>
                      </tr>
                    </>
                  )}
                </tbody>
              </table>
            </div>
            {eigenResult.complex && (
              <div className={styles.matrixHint} style={{ marginTop: '0.5rem' }}>
                <strong>About complex eigenvalues</strong>
                <p style={{ margin: '0.35rem 0 0' }}>
                  When <span dangerouslySetInnerHTML={{ __html: tex('\\lambda = re \\pm im\\cdot i') }} />, there are no real eigenvectors: the equation <span dangerouslySetInnerHTML={{ __html: tex('A\\mathbf{v} = \\lambda\\mathbf{v}') }} /> has only complex solutions. In the plane, such a matrix acts as a <strong>rotation combined with scaling</strong>. The magnitude <span dangerouslySetInnerHTML={{ __html: tex('|\\lambda| = \\sqrt{re^2 + im^2}') }} /> is the scaling factor (here <span dangerouslySetInnerHTML={{ __html: tex('|\\lambda|') }} /> = {fmt(Math.hypot(eigenResult.values[0], eigenResult.values[1]))}); the ellipse in the plot is the rotated and scaled image of the unit circle. No single real line is left invariant, so no eigenvector arrows are drawn.
                </p>
              </div>
            )}
          </div>

          {!eigenResult.complex && eigenResult.vectors.length >= 2 && (
            <div className={styles.matrixBlock}>
              <h4 className={styles.matrixTitle}>Eigenvectors (unit)</h4>
              <p className={styles.matrixHint}>
                v₁ for λ₁, v₂ for λ₂ (each normalized). <span dangerouslySetInnerHTML={{ __html: tex('A\\mathbf{v}_i = \\lambda_i \\mathbf{v}_i') }} />.
              </p>
              <div className={styles.matrixWrap}>
                <table className={styles.matrixTable}>
                  <tbody>
                    <tr>
                      <th className={styles.matrixRowHeader}>v₁</th>
                      <td className={styles.matrixCell}>
                        ({fmt(eigenResult.vectors[0][0])}, {fmt(eigenResult.vectors[0][1])})
                      </td>
                    </tr>
                    <tr>
                      <th className={styles.matrixRowHeader}>v₂</th>
                      <td className={styles.matrixCell}>
                        ({fmt(eigenResult.vectors[1][0])}, {fmt(eigenResult.vectors[1][1])})
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <div className={styles.matrixBlock}>
            <h4 className={styles.matrixTitle}>Visualization</h4>
            <p className={styles.matrixHint}>
              Gray: unit circle. Accent: image under A (ellipse). Arrows: eigenvector directions (real case only).
            </p>
            <EigenViz A={parseResult.matrix} result={eigenResult} />
          </div>
        </>
      )}
    </div>
  )
}
