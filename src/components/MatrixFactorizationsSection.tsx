import { useState, useMemo, type ReactNode } from 'react'
import katex from 'katex'
import 'katex/dist/katex.min.css'
import {
  lu,
  qr,
  cholesky,
  svd,
  spectralDecomposition,
  solveLU,
  solveCholesky,
  svdReconstruct,
  luReconstructionError,
  qrReconstructionError,
  choleskyReconstructionError,
  spectralReconstructionError,
  type LUResult,
  type QRResult,
  type CholeskyResult,
  type SVDResult,
  type SpectralResult,
} from '@/lib/matrixFactorizations'
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

type FactorizationType = 'lu' | 'qr' | 'cholesky' | 'svd' | 'spectral'

const DEFAULT_MATRIX = '2  -1   0\n-1  2  -1\n0  -1   2'

const FACTORIZATION_DESCRIPTIONS: Record<
  FactorizationType,
  { title: string; body: ReactNode }
> = {
  lu: {
    title: 'LU (with partial pivoting)',
    body: (
      <>
        <p className={styles.matrixHint} style={{ margin: 0 }}>
          For a square matrix <span dangerouslySetInnerHTML={{ __html: tex('A') }} />, LU with partial pivoting gives{' '}
          <span dangerouslySetInnerHTML={{ __html: tex('PA = LU') }} />, where <span dangerouslySetInnerHTML={{ __html: tex('P') }} /> is a permutation matrix,{' '}
          <span dangerouslySetInnerHTML={{ __html: tex('L') }} /> is unit lower triangular, and <span dangerouslySetInnerHTML={{ __html: tex('U') }} /> is upper triangular.
          Pivoting improves numerical stability. Once computed, you can solve <span dangerouslySetInnerHTML={{ __html: tex('A\\mathbf{x} = \\mathbf{b}') }} /> via forward/back substitution.
        </p>
      </>
    ),
  },
  qr: {
    title: 'QR',
    body: (
      <>
        <p className={styles.matrixHint} style={{ margin: 0 }}>
          For any <span dangerouslySetInnerHTML={{ __html: tex('m \\times n') }} /> matrix <span dangerouslySetInnerHTML={{ __html: tex('A') }} />,{' '}
          <span dangerouslySetInnerHTML={{ __html: tex('A = QR') }} /> where <span dangerouslySetInnerHTML={{ __html: tex('Q') }} /> has orthonormal columns (computed here via modified Gram–Schmidt) and{' '}
          <span dangerouslySetInnerHTML={{ __html: tex('R') }} /> is upper triangular. Used for least squares, eigenvalues (QR algorithm), and orthogonalization.
        </p>
      </>
    ),
  },
  cholesky: {
    title: 'Cholesky',
    body: (
      <>
        <p className={styles.matrixHint} style={{ margin: 0 }}>
          For a <strong>symmetric positive definite</strong> matrix <span dangerouslySetInnerHTML={{ __html: tex('A') }} />,{' '}
          <span dangerouslySetInnerHTML={{ __html: tex('A = LL^\\top') }} /> where <span dangerouslySetInnerHTML={{ __html: tex('L') }} /> is lower triangular with positive diagonal.
          Requires <span dangerouslySetInnerHTML={{ __html: tex('A = A^\\top') }} /> and <span dangerouslySetInnerHTML={{ __html: tex('\\mathbf{x}^\\top A\\mathbf{x} > 0') }} /> for all nonzero <span dangerouslySetInnerHTML={{ __html: tex('\\mathbf{x}') }} />.
          Faster and more stable than LU for SPD systems; useful in optimization and sampling.
        </p>
      </>
    ),
  },
  svd: {
    title: 'SVD (singular value decomposition)',
    body: (
      <>
        <p className={styles.matrixHint} style={{ margin: 0 }}>
          For any <span dangerouslySetInnerHTML={{ __html: tex('m \\times n') }} /> matrix <span dangerouslySetInnerHTML={{ __html: tex('A') }} />,{' '}
          <span dangerouslySetInnerHTML={{ __html: tex('A = U\\Sigma V^\\top') }} /> where <span dangerouslySetInnerHTML={{ __html: tex('U') }} /> and <span dangerouslySetInnerHTML={{ __html: tex('V') }} /> are orthogonal
          and <span dangerouslySetInnerHTML={{ __html: tex('\\Sigma') }} /> is diagonal with non‑negative singular values. Reveals rank, range, null space, and is the basis for PCA, low-rank approximation, and many applications in ML and data science.
        </p>
      </>
    ),
  },
  spectral: {
    title: 'Spectral decomposition',
    body: (
      <>
        <p className={styles.matrixHint} style={{ margin: 0 }}>
          For a <strong>symmetric</strong> square matrix <span dangerouslySetInnerHTML={{ __html: tex('A') }} />,{' '}
          <span dangerouslySetInnerHTML={{ __html: tex('A = Q\\Lambda Q^\\top') }} /> where <span dangerouslySetInnerHTML={{ __html: tex('Q') }} /> is orthogonal (columns are eigenvectors) and{' '}
          <span dangerouslySetInnerHTML={{ __html: tex('\\Lambda') }} /> is diagonal (eigenvalues). All eigenvalues are real. Used in quadratic forms, PCA (covariance matrix), and understanding the geometry of <span dangerouslySetInnerHTML={{ __html: tex('A') }} />.
        </p>
      </>
    ),
  },
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

export function MatrixFactorizationsSection() {
  const [matrixText, setMatrixText] = useState(DEFAULT_MATRIX)
  const [factorizationType, setFactorizationType] = useState<FactorizationType>('lu')
  const [bText, setBText] = useState('1\n0\n1')

  const parseResult = useMemo(() => {
    const p = parseMatrix(matrixText)
    return typeof p === 'string' ? { matrix: null, error: p } : { matrix: p, error: null }
  }, [matrixText])
  const parsed = parseResult.matrix

  const result = useMemo(() => {
    if (!parsed || parsed.length === 0) return null
    const rows = parsed.length
    const cols = parsed[0].length
    try {
      if (factorizationType === 'lu') {
        if (rows !== cols) return { error: 'LU requires a square matrix.' }
        const r = lu(parsed)
        return { type: 'lu' as const, data: r, error: null }
      }
      if (factorizationType === 'qr') {
        const r = qr(parsed)
        return { type: 'qr' as const, data: r, error: null }
      }
      if (factorizationType === 'cholesky') {
        if (rows !== cols) return { error: 'Cholesky requires a square matrix.' }
        const r = cholesky(parsed)
        return { type: 'cholesky' as const, data: r, error: null }
      }
      if (factorizationType === 'svd') {
        const r = svd(parsed)
        return { type: 'svd' as const, data: r, error: null }
      }
      if (factorizationType === 'spectral') {
        if (rows !== cols) return { error: 'Spectral decomposition requires a square matrix.' }
        const r = spectralDecomposition(parsed)
        return { type: 'spectral' as const, data: r, error: null }
      }
    } catch (e) {
      return { error: e instanceof Error ? e.message : String(e) }
    }
    return null
  }, [parsed, factorizationType])

  const solveResult = useMemo(() => {
    if (!parsed || !result || result.error) return null
    if (factorizationType !== 'lu' && factorizationType !== 'cholesky') return null
    const bParts = bText.trim().split(/[\s,;]+/).map((s) => parseFloat(s)).filter((v) => Number.isFinite(v))
    if (bParts.length !== parsed.length) return null
    try {
      if (factorizationType === 'lu' && result.type === 'lu') return solveLU(result.data as LUResult, bParts)
      if (factorizationType === 'cholesky' && result.type === 'cholesky') return solveCholesky(result.data as CholeskyResult, bParts)
    } catch {
      return null
    }
    return null
  }, [parsed, result, factorizationType, bText])

  const reconError = useMemo(() => {
    if (!parsed || !result || result.error) return null
    try {
      if (result.type === 'lu') return luReconstructionError(parsed, result.data as LUResult)
      if (result.type === 'qr') return qrReconstructionError(parsed, result.data as QRResult)
      if (result.type === 'cholesky') return choleskyReconstructionError(parsed, result.data as CholeskyResult)
      if (result.type === 'spectral') return spectralReconstructionError(parsed, result.data as SpectralResult)
      if (result.type === 'svd') {
        const recon = svdReconstruct(result.data as SVDResult)
        let s = 0
        for (let i = 0; i < parsed.length; i++)
          for (let j = 0; j < (parsed[0]?.length ?? 0); j++)
            s += (parsed[i][j] - recon[i][j]) ** 2
        return Math.sqrt(s)
      }
    } catch {
      return null
    }
    return null
  }, [parsed, result])

  function renderMatrix(M: number[][], title: string) {
    return (
      <div className={styles.matrixBlock}>
        <h4 className={styles.matrixTitle}>{title}</h4>
        <div className={styles.matrixWrap}>
          <table className={styles.matrixTable}>
            <tbody>
              {M.map((row, i) => (
                <tr key={i}>
                  {row.map((v, j) => (
                    <td key={j} className={styles.matrixCell}>
                      {fmt(v)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    )
  }

  return (
    <div className={styles.section}>
      <div className={styles.intro}>
        <p className={styles.introText}>
          Decompose a matrix into simpler factors: <strong>LU</strong>, <strong>QR</strong>, <strong>Cholesky</strong>, <strong>SVD</strong>, or <strong>spectral</strong>. Enter <span dangerouslySetInnerHTML={{ __html: tex('A') }} /> below and choose a factorization.
        </p>
      </div>

      <div className={styles.editorBlock}>
        <label className={styles.label}>Matrix A (one row per line; numbers separated by spaces or commas)</label>
        <textarea
          className={styles.textarea}
          value={matrixText}
          onChange={(e) => setMatrixText(e.target.value)}
          rows={5}
          spellCheck={false}
        />
        <div className={styles.simulateForm} style={{ flexWrap: 'wrap', gap: '1rem', alignItems: 'center' }}>
          <div className={styles.fieldLabel}>
            <span>Factorization</span>
            <select
              className={styles.select}
              value={factorizationType}
              onChange={(e) => setFactorizationType(e.target.value as FactorizationType)}
            >
              <option value="lu">LU (P, L, U)</option>
              <option value="qr">QR (Q, R)</option>
              <option value="cholesky">Cholesky (L)</option>
              <option value="svd">SVD (U, Σ, V)</option>
              <option value="spectral">Spectral (Q, Λ)</option>
            </select>
          </div>
        </div>
        {(factorizationType === 'lu' || factorizationType === 'cholesky') && (
          <div className={styles.fieldLabel} style={{ marginTop: '0.5rem' }}>
            <span>Vector b (for solve Ax = b; space/comma separated)</span>
            <input
              type="text"
              className={styles.input}
              value={bText}
              onChange={(e) => setBText(e.target.value)}
              style={{ width: '100%', maxWidth: '320px' }}
            />
          </div>
        )}
      </div>

      <div className={styles.matrixBlock}>
        <h3 className={styles.matrixTitle}>{FACTORIZATION_DESCRIPTIONS[factorizationType].title}</h3>
        {FACTORIZATION_DESCRIPTIONS[factorizationType].body}
      </div>

      {parseResult.error && (
        <p className={styles.error}>Matrix: {parseResult.error}</p>
      )}
      {result?.error && !parseResult.error && (
        <p className={styles.error}>{result.error}</p>
      )}

      {result && !result.error && result.type === 'lu' && (
        <>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', alignItems: 'flex-start' }}>
            {renderMatrix(result.data.P, 'P (permutation)')}
            {renderMatrix(result.data.L, 'L (unit lower)')}
            {renderMatrix(result.data.U, 'U (upper)')}
          </div>
          {reconError != null && (
            <p className={styles.matrixHint}>
              Reconstruction error <span dangerouslySetInnerHTML={{ __html: tex('\\|PA - LU\\|_F') }} /> = {reconError.toExponential(4)}
            </p>
          )}
          {solveResult != null && (
            <div className={styles.matrixBlock}>
              <h4 className={styles.matrixTitle}>Solution x (Ax = b)</h4>
              <p className={styles.matrixHint}>x = [{solveResult.map((v) => fmt(v)).join(', ')}]</p>
            </div>
          )}
        </>
      )}

      {result && !result.error && result.type === 'qr' && (
        <>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', alignItems: 'flex-start' }}>
            {renderMatrix(result.data.Q, 'Q (orthogonal)')}
            {renderMatrix(result.data.R, 'R (upper triangular)')}
          </div>
          {reconError != null && (
            <p className={styles.matrixHint}>
              Reconstruction error <span dangerouslySetInnerHTML={{ __html: tex('\\|A - QR\\|_F') }} /> = {reconError.toExponential(4)}
            </p>
          )}
        </>
      )}

      {result && !result.error && result.type === 'cholesky' && (
        <>
          {renderMatrix(result.data.L, 'L (lower triangular)')}
          {reconError != null && (
            <p className={styles.matrixHint}>
              Reconstruction error <span dangerouslySetInnerHTML={{ __html: tex('\\|A - LL^\\top\\|_F') }} /> = {reconError.toExponential(4)}
            </p>
          )}
          {solveResult != null && (
            <div className={styles.matrixBlock}>
              <h4 className={styles.matrixTitle}>Solution x (Ax = b)</h4>
              <p className={styles.matrixHint}>x = [{solveResult.map((v) => fmt(v)).join(', ')}]</p>
            </div>
          )}
        </>
      )}

      {result && !result.error && result.type === 'svd' && (
        <>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', alignItems: 'flex-start' }}>
            {renderMatrix(result.data.U, 'U (left singular vectors)')}
            {renderMatrix(
              result.data.S.map((s, i) =>
                result.data.S.map((_, j) => (i === j ? s : 0))
              ),
              'Σ (diagonal matrix of singular values)'
            )}
            {renderMatrix(result.data.V, 'V (right singular vectors)')}
          </div>
          {reconError != null && (
            <p className={styles.matrixHint}>
              Reconstruction error <span dangerouslySetInnerHTML={{ __html: tex('\\|A - U\\Sigma V^\\top\\|_F') }} /> = {reconError.toExponential(4)}
            </p>
          )}
        </>
      )}

      {result && !result.error && result.type === 'spectral' && (
        <>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', alignItems: 'flex-start' }}>
            {renderMatrix(result.data.Q, 'Q (eigenvectors as columns)')}
            {renderMatrix(
              result.data.eigenvalues.map((v, i) =>
                result.data.eigenvalues.map((_, j) => (i === j ? v : 0))
              ),
              'Λ (diagonal matrix of eigenvalues)'
            )}
          </div>
          {reconError != null && (
            <p className={styles.matrixHint}>
              Reconstruction error <span dangerouslySetInnerHTML={{ __html: tex('\\|A - Q\\Lambda Q^\\top\\|_F') }} /> = {reconError.toExponential(4)}
            </p>
          )}
        </>
      )}
    </div>
  )
}
