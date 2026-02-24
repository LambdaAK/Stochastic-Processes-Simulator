/**
 * Matrix factorizations: LU (with partial pivoting), QR, Cholesky, SVD.
 * All operations in-place or on copies; no mutation of input matrices.
 */

const EPS = 1e-12

/** Deep clone of a matrix. */
function clone(M: number[][]): number[][] {
  return M.map((row) => [...row])
}

/** Matrix transpose (m×n → n×m). */
function transpose(M: number[][]): number[][] {
  const m = M.length
  const n = M[0]?.length ?? 0
  const T: number[][] = Array.from({ length: n }, () => new Array(m).fill(0))
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) T[j][i] = M[i][j]
  return T
}

/** Matrix product C = A * B (A m×p, B p×n). */
function matMul(A: number[][], B: number[][]): number[][] {
  const m = A.length
  const p = A[0]?.length ?? 0
  const n = B[0]?.length ?? 0
  const C: number[][] = Array.from({ length: m }, () => new Array(n).fill(0))
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++)
      for (let k = 0; k < p; k++) C[i][j] += A[i][k] * B[k][j]
  return C
}

/** Frobenius norm of matrix. */
function frobeniusNorm(M: number[][]): number {
  let s = 0
  for (const row of M) for (const v of row) s += v * v
  return Math.sqrt(s)
}

/** Permutation matrix from permutation array p: P[i] = row that has 1 in column i. */
function permMatrix(p: number[]): number[][] {
  const n = p.length
  const P: number[][] = Array.from({ length: n }, () => new Array(n).fill(0))
  for (let j = 0; j < n; j++) if (p[j] >= 0) P[p[j]][j] = 1
  return P
}

// ─── LU with partial pivoting ─────────────────────────────────────────────────
// P A = L U: P permutation, L unit lower, U upper.

export interface LUResult {
  P: number[][]  // permutation matrix (P*A = L*U)
  L: number[][]
  U: number[][]
  pivot: number[] // pivot indices (0-based)
}

export function lu(A: number[][]): LUResult {
  const M = clone(A)
  const n = M.length
  const pivot: number[] = Array.from({ length: n }, (_, i) => i)

  for (let k = 0; k < n; k++) {
    let maxIdx = k
    let maxVal = Math.abs(M[k][k])
    for (let i = k + 1; i < n; i++) {
      const v = Math.abs(M[i][k])
      if (v > maxVal) {
        maxVal = v
        maxIdx = i
      }
    }
    if (maxVal < EPS) continue // singular or nearly
    // swap rows k and maxIdx in M and in pivot
    ;[M[k], M[maxIdx]] = [M[maxIdx], M[k]]
    ;[pivot[k], pivot[maxIdx]] = [pivot[maxIdx], pivot[k]]

    for (let i = k + 1; i < n; i++) {
      M[i][k] /= M[k][k]
      for (let j = k + 1; j < n; j++) M[i][j] -= M[i][k] * M[k][j]
    }
  }

  const L: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (__, j) => (j < i ? M[i][j] : j === i ? 1 : 0))
  )
  const U: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (__, j) => (j >= i ? M[i][j] : 0))
  )
  const P = permMatrix(pivot)

  return { P, L, U, pivot }
}

/** Solve Ax = b using LU: solve L y = P b then U x = y. */
export function solveLU(luResult: LUResult, b: number[]): number[] {
  const { P, L, U } = luResult
  const n = P.length
  const Pb = new Array(n).fill(0)
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) Pb[i] += P[i][j] * b[j]
  const y = new Array(n).fill(0)
  for (let i = 0; i < n; i++) {
    y[i] = Pb[i]
    for (let j = 0; j < i; j++) y[i] -= L[i][j] * y[j]
  }
  const x = new Array(n).fill(0)
  for (let i = n - 1; i >= 0; i--) {
    x[i] = y[i]
    for (let j = i + 1; j < n; j++) x[i] -= U[i][j] * x[j]
    x[i] /= U[i][i]
  }
  return x
}

// ─── QR (modified Gram–Schmidt) ───────────────────────────────────────────────

export interface QRResult {
  Q: number[][]  // orthogonal (columns are orthonormal)
  R: number[][]  // upper triangular
}

export function qr(A: number[][]): QRResult {
  const m = A.length
  const n = A[0]?.length ?? 0
  const Q: number[][] = Array.from({ length: m }, (_, i) =>
    Array.from({ length: n }, (__, j) => A[i][j])
  )
  const R: number[][] = Array.from({ length: n }, () => new Array(n).fill(0))

  for (let j = 0; j < n; j++) {
    for (let k = 0; k < j; k++) {
      let dot = 0
      for (let i = 0; i < m; i++) dot += Q[i][k] * Q[i][j]
      R[k][j] = dot
      for (let i = 0; i < m; i++) Q[i][j] -= dot * Q[i][k]
    }
    let norm = 0
    for (let i = 0; i < m; i++) norm += Q[i][j] * Q[i][j]
    norm = Math.sqrt(norm)
    if (norm < EPS) norm = 1
    R[j][j] = norm
    for (let i = 0; i < m; i++) Q[i][j] /= norm
  }

  return { Q, R }
}

// ─── Cholesky A = L Lᵀ (A symmetric positive definite) ───────────────────────────

export interface CholeskyResult {
  L: number[][]  // lower triangular with positive diagonal
}

export function cholesky(A: number[][]): CholeskyResult {
  const n = A.length
  const L: number[][] = Array.from({ length: n }, () => new Array(n).fill(0))

  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i][j]
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k]
      if (i === j) {
        if (s <= 0) throw new Error('Matrix is not positive definite')
        L[i][j] = Math.sqrt(s)
      } else {
        if (Math.abs(L[j][j]) < EPS) throw new Error('Matrix is not positive definite')
        L[i][j] = s / L[j][j]
      }
    }
  }
  return { L }
}

/** Solve Ax = b using Cholesky: L y = b, Lᵀ x = y. */
export function solveCholesky(chol: CholeskyResult, b: number[]): number[] {
  const L = chol.L
  const n = L.length
  const y = new Array(n).fill(0)
  for (let i = 0; i < n; i++) {
    y[i] = b[i]
    for (let j = 0; j < i; j++) y[i] -= L[i][j] * y[j]
    y[i] /= L[i][i]
  }
  const x = new Array(n).fill(0)
  for (let i = n - 1; i >= 0; i--) {
    x[i] = y[i]
    for (let j = i + 1; j < n; j++) x[i] -= L[j][i] * x[j]
    x[i] /= L[i][i]
  }
  return x
}

// ─── SVD A = U Σ Vᵀ (via eigendecomposition of AᵀA) ──────────────────────────

export interface SVDResult {
  U: number[][]   // m×r (r = rank, or m×n if full)
  S: number[]     // singular values (descending, length r)
  V: number[][]   // n×n (right singular vectors as columns)
}

/** Symmetric Jacobi eigendecomposition (returns eigenvalues and eigenvectors as columns of V). */
function jacobiEig(S: number[][]): { values: number[]; vectors: number[][] } {
  const n = S.length
  const A: number[][] = S.map((row) => [...row])
  let V: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (__, j) => (i === j ? 1 : 0))
  )

  const maxIter = 200 * n * n
  for (let iter = 0; iter < maxIter; iter++) {
    let p = 0,
      q = 1,
      maxVal = 0
    for (let i = 0; i < n - 1; i++)
      for (let j = i + 1; j < n; j++)
        if (Math.abs(A[i][j]) > maxVal) {
          maxVal = Math.abs(A[i][j])
          p = i
          q = j
        }
    if (maxVal < 1e-14) break

    const theta =
      Math.abs(A[p][p] - A[q][q]) < 1e-14
        ? Math.PI / 4
        : 0.5 * Math.atan2(2 * A[p][q], A[p][p] - A[q][q])
    const c = Math.cos(theta)
    const s = Math.sin(theta)

    const newA: number[][] = A.map((row) => [...row])
    for (let j = 0; j < n; j++) {
      newA[p][j] = c * A[p][j] + s * A[q][j]
      newA[q][j] = -s * A[p][j] + c * A[q][j]
    }
    const tmp: number[][] = newA.map((row) => [...row])
    for (let i = 0; i < n; i++) {
      tmp[i][p] = c * newA[i][p] + s * newA[i][q]
      tmp[i][q] = -s * newA[i][p] + c * newA[i][q]
    }
    for (let i = 0; i < n; i++) A[i] = tmp[i]

    const newV: number[][] = V.map((row) => [...row])
    for (let i = 0; i < n; i++) {
      newV[i][p] = c * V[i][p] + s * V[i][q]
      newV[i][q] = -s * V[i][p] + c * V[i][q]
    }
    V = newV
  }

  const values = A.map((_, i) => A[i][i])
  // Sort by eigenvalue descending
  const order = values
    .map((v, i) => ({ v, i }))
    .sort((a, b) => b.v - a.v)
    .map((x) => x.i)
  const sortedValues = order.map((i) => values[i])
  const sortedVectors = order.map((j) => V.map((row) => row[j]))
  const Vsorted: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (__, j) => sortedVectors[j][i])
  )
  return { values: sortedValues, vectors: Vsorted }
}

export function svd(A: number[][]): SVDResult {
  const m = A.length
  const n = A[0]?.length ?? 0
  const At = transpose(A)
  const AtA = matMul(At, A)

  const { values: eigenvalues, vectors: V } = jacobiEig(AtA)
  const S: number[] = eigenvalues.map((lam) => (lam > 0 ? Math.sqrt(lam) : 0))

  // U = A V Σ⁻¹ (columns of U)
  const r = S.filter((s) => s > EPS).length
  const U: number[][] = Array.from({ length: m }, () => new Array(r).fill(0))
  for (let j = 0; j < r; j++) {
    const sigma = S[j]
    if (sigma < EPS) continue
    for (let i = 0; i < m; i++) {
      let sum = 0
      for (let k = 0; k < n; k++) sum += A[i][k] * V[k][j]
      U[i][j] = sum / sigma
    }
  }

  return { U, S: S.slice(0, r), V }
}

/** Reconstruct A from SVD: A ≈ U * diag(S) * Vᵀ (using first r columns of V). */
export function svdReconstruct(svdResult: SVDResult): number[][] {
  const { U, S, V } = svdResult
  const r = S.length
  const US: number[][] = U.map((row) => row.slice(0, r).map((v, j) => v * S[j]))
  const Vr = V.map((row) => row.slice(0, r))
  return matMul(US, transpose(Vr))
}

// ─── Spectral decomposition (symmetric A = Q Λ Qᵀ) ───────────────────────────

export interface SpectralResult {
  Q: number[][]       // orthogonal matrix (eigenvectors as columns)
  eigenvalues: number[]  // diagonal of Λ (descending)
}

export function spectralDecomposition(A: number[][]): SpectralResult {
  const n = A.length
  if (n !== (A[0]?.length ?? 0)) throw new Error('Spectral decomposition requires a square matrix.')
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++)
      if (Math.abs(A[i][j] - A[j][i]) > EPS) throw new Error('Matrix must be symmetric.')
  const { values: eigenvalues, vectors: Q } = jacobiEig(A)
  return { Q, eigenvalues }
}

export function spectralReconstructionError(A: number[][], result: SpectralResult): number {
  const QL = result.Q.map((row) => row.map((v, j) => v * result.eigenvalues[j]))
  const QLQT = matMul(QL, transpose(result.Q))
  const diff = A.map((row, i) => row.map((v, j) => v - QLQT[i][j]))
  return frobeniusNorm(diff)
}

// ─── Reconstruction error ─────────────────────────────────────────────────────

export function luReconstructionError(A: number[][], result: LUResult): number {
  const PA = matMul(result.P, A)
  const LU = matMul(result.L, result.U)
  const diff = PA.map((row, i) => row.map((v, j) => v - LU[i][j]))
  return frobeniusNorm(diff)
}

export function qrReconstructionError(A: number[][], result: QRResult): number {
  const QR = matMul(result.Q, result.R)
  const diff = A.map((row, i) => row.map((v, j) => v - QR[i][j]))
  return frobeniusNorm(diff)
}

export function choleskyReconstructionError(A: number[][], result: CholeskyResult): number {
  const LLT = matMul(result.L, transpose(result.L))
  const diff = A.map((row, i) => row.map((v, j) => v - LLT[i][j]))
  return frobeniusNorm(diff)
}
