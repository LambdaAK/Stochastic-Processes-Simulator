/**
 * PCA: reduce ℝⁿ → ℝ² and reconstruct back to ℝⁿ.
 *
 * Steps:
 *  1. Center the data (subtract column means).
 *  2. Compute sample covariance matrix Σ = (1/(m-1)) Xᵀ X  (X already centered).
 *  3. Eigendecompose Σ via Jacobi iteration.
 *  4. Sort eigenvectors by eigenvalue descending.
 *  5. Project onto top-k eigenvectors.
 *  6. Reconstruct from projection and compute MSE.
 */

/** Compute column means for an m×n matrix (array of m row-vectors of length n). */
function colMeans(data: number[][]): number[] {
  const m = data.length
  const n = data[0].length
  const means = new Array<number>(n).fill(0)
  for (const row of data) for (let j = 0; j < n; j++) means[j] += row[j]
  return means.map((s) => s / m)
}

/** Center data by subtracting column means. Returns centered matrix and means. */
function center(data: number[][]): { centered: number[][]; means: number[] } {
  const means = colMeans(data)
  const centered = data.map((row) => row.map((v, j) => v - means[j]))
  return { centered, means }
}

/** Compute sample covariance matrix (n×n) from centered m×n matrix. */
function covarianceMatrix(X: number[][]): number[][] {
  const m = X.length
  const n = X[0].length
  const C: number[][] = Array.from({ length: n }, () => new Array(n).fill(0))
  for (const row of X) {
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        C[i][j] += row[i] * row[j]
      }
    }
  }
  const denom = Math.max(1, m - 1)
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) C[i][j] /= denom
  return C
}

/**
 * Symmetric Jacobi eigendecomposition of a real symmetric n×n matrix.
 * Returns eigenvalues and corresponding eigenvectors (columns of V).
 */
function jacobiEig(S: number[][]): { values: number[]; vectors: number[][] } {
  const n = S.length
  // Work on a copy
  const A: number[][] = S.map((row) => [...row])
  // V accumulates rotations (starts as identity)
  let V: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (__, j) => (i === j ? 1 : 0))
  )

  const MAX_ITER = 200 * n * n
  for (let iter = 0; iter < MAX_ITER; iter++) {
    // Find off-diagonal element with largest absolute value
    let p = 0,
      q = 1,
      maxVal = 0
    for (let i = 0; i < n - 1; i++) {
      for (let j = i + 1; j < n; j++) {
        if (Math.abs(A[i][j]) > maxVal) {
          maxVal = Math.abs(A[i][j])
          p = i
          q = j
        }
      }
    }
    if (maxVal < 1e-12) break

    // Compute rotation angle
    const theta =
      Math.abs(A[p][p] - A[q][q]) < 1e-14
        ? Math.PI / 4
        : 0.5 * Math.atan2(2 * A[p][q], A[p][p] - A[q][q])
    const c = Math.cos(theta)
    const s = Math.sin(theta)

    // Update A = Jᵀ A J
    const newA: number[][] = A.map((row) => [...row])
    // Update rows p and q
    for (let j = 0; j < n; j++) {
      newA[p][j] = c * A[p][j] + s * A[q][j]
      newA[q][j] = -s * A[p][j] + c * A[q][j]
    }
    // Update cols p and q
    const tmp: number[][] = newA.map((row) => [...row])
    for (let i = 0; i < n; i++) {
      tmp[i][p] = c * newA[i][p] + s * newA[i][q]
      tmp[i][q] = -s * newA[i][p] + c * newA[i][q]
    }
    for (let i = 0; i < n; i++) {
      A[i] = tmp[i]
    }

    // Update eigenvector matrix V
    const newV: number[][] = V.map((row) => [...row])
    for (let i = 0; i < n; i++) {
      newV[i][p] = c * V[i][p] + s * V[i][q]
      newV[i][q] = -s * V[i][p] + c * V[i][q]
    }
    V = newV
  }

  // Extract eigenvalues (diagonal of A) and eigenvectors (columns of V)
  const values = Array.from({ length: n }, (_, i) => A[i][i])
  // V columns are eigenvectors; convert to array of column vectors
  const vectors: number[][] = Array.from({ length: n }, (_, j) =>
    Array.from({ length: n }, (__, i) => V[i][j])
  )
  return { values, vectors }
}

export interface PCAResult {
  /** Column means of original data */
  means: number[]
  /** Top eigenvalues (variance explained by each PC) */
  eigenvalues: number[]
  /** Top eigenvectors [pc1, pc2, ...] each of length n */
  eigenvectors: number[][]
  /** All eigenvalues sorted descending */
  allEigenvalues: number[]
  /** Explained variance ratio per component */
  explainedVarianceRatio: number[]
  /** Projected points in ℝ² (m×2) */
  projected: number[][]
  /** Reconstructed points in ℝⁿ (m×n) */
  reconstructed: number[][]
  /** Mean squared reconstruction error */
  mse: number
  /** Covariance matrix */
  covariance: number[][]
}

/**
 * Run PCA on dataset (m points × n features), projecting onto top `k` PCs.
 */
export function runPCA(data: number[][], k = 2): PCAResult {
  const { centered, means } = center(data)
  const cov = covarianceMatrix(centered)
  const { values, vectors } = jacobiEig(cov)

  // Sort by eigenvalue descending
  const order = values
    .map((v, i) => ({ v, i }))
    .sort((a, b) => b.v - a.v)
    .map((x) => x.i)

  const allEigenvalues = order.map((i) => Math.max(0, values[i]))
  const totalVar = allEigenvalues.reduce((a, b) => a + b, 0) || 1

  const topK = order.slice(0, k)
  const eigenvalues = topK.map((i) => Math.max(0, values[i]))
  const eigenvectors = topK.map((i) => vectors[i]) // each length n

  const explainedVarianceRatio = allEigenvalues.map((v) => v / totalVar)

  // Project: for each point x (centered), z_j = eigenvectors[j] · x
  const m = data.length
  const n = data[0].length
  const projected: number[][] = centered.map((row) =>
    eigenvectors.map((ev) => ev.reduce((s, vi, i) => s + vi * row[i], 0))
  )

  // Reconstruct: x̂ = Σ_j z_j * eigenvectors[j] + means
  const reconstructed: number[][] = projected.map((z) => {
    const rec = new Array<number>(n).fill(0)
    for (let j = 0; j < k; j++)
      for (let i = 0; i < n; i++) rec[i] += z[j] * eigenvectors[j][i]
    for (let i = 0; i < n; i++) rec[i] += means[i]
    return rec
  })

  // MSE over all m points and n dimensions
  let mse = 0
  for (let r = 0; r < m; r++)
    for (let i = 0; i < n; i++)
      mse += (data[r][i] - reconstructed[r][i]) ** 2
  mse /= m * n

  return {
    means,
    eigenvalues,
    eigenvectors,
    allEigenvalues,
    explainedVarianceRatio,
    projected,
    reconstructed,
    mse,
    covariance: cov,
  }
}

// ─── Dataset generators ──────────────────────────────────────────────────────

/** Gaussian sample via Box-Muller. */
function randn(rand: () => number): number {
  const u = 1 - rand()
  const v = rand()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}

export type DataPreset = 'correlated' | 'blob' | 'random' | 'diagonal'

/**
 * Generate a synthetic dataset in ℝⁿ.
 * @param preset data shape
 * @param m number of points
 * @param n dimension
 * @param rand RNG
 */
export function generatePCADataset(
  preset: DataPreset,
  m: number,
  n: number,
  rand: () => number
): number[][] {
  if (preset === 'blob') {
    // Isotropic Gaussian
    return Array.from({ length: m }, () =>
      Array.from({ length: n }, () => randn(rand))
    )
  }
  if (preset === 'correlated') {
    // Strong first-PC: x₁ dominant, others are x₁ + small noise
    return Array.from({ length: m }, () => {
      const z = randn(rand)
      return Array.from({ length: n }, (_, i) =>
        i === 0 ? z * 3 : z * 1.5 + randn(rand) * 0.5
      )
    })
  }
  if (preset === 'diagonal') {
    // Different variance per dimension (σ_i = n - i + 1 for i=1..n)
    return Array.from({ length: m }, () =>
      Array.from({ length: n }, (_, i) => randn(rand) * (n - i))
    )
  }
  // 'random': uniform noise — little structure
  return Array.from({ length: m }, () =>
    Array.from({ length: n }, () => (rand() * 2 - 1) * 4)
  )
}
