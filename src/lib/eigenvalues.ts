/**
 * Eigenvalues and eigenvectors for 2×2 matrices.
 * Closed-form: λ² − (trace)λ + (det) = 0; eigenvectors from (A − λI)v = 0.
 */

export interface Eigen2x2Result {
  /** Eigenvalues [λ₁, λ₂] when real; when complex, [re, im] for λ = re ± im·i */
  values: [number, number]
  /** Eigenvectors as [v₁, v₂] (each length 2); only when both eigenvalues are real */
  vectors: number[][]
  /** True when eigenvalues are complex conjugates (no real eigenvector basis) */
  complex: boolean
}

const EPS = 1e-12

function normalize2(v: [number, number]): [number, number] {
  const n = Math.hypot(v[0], v[1])
  if (n < EPS) return v
  return [v[0] / n, v[1] / n]
}

/**
 * Compute eigenvalues and eigenvectors of a 2×2 matrix A = [[a,b],[c,d]].
 * Characteristic equation: λ² − (a+d)λ + (ad−bc) = 0.
 */
export function eigen2x2(A: number[][]): Eigen2x2Result {
  if (A.length !== 2 || (A[0]?.length ?? 0) !== 2 || (A[1]?.length ?? 0) !== 2) {
    throw new Error('eigen2x2 requires a 2×2 matrix.')
  }
  const a = A[0][0],
    b = A[0][1],
    c = A[1][0],
    d = A[1][1]
  const tr = a + d
  const det = a * d - b * c
  const disc = tr * tr - 4 * det

  if (disc >= 0) {
    const sqrtD = Math.sqrt(disc)
    const l1 = (tr + sqrtD) / 2
    const l2 = (tr - sqrtD) / 2
    const values: [number, number] = [l1, l2]

    function eigenvector(lambda: number): [number, number] {
      if (Math.abs(b) > EPS) {
        return normalize2([b, lambda - a])
      }
      if (Math.abs(c) > EPS) {
        return normalize2([lambda - d, c])
      }
      // Diagonal: A = diag(a, d); use (1,0) and (0,1)
      if (Math.abs(lambda - a) < EPS) return [1, 0]
      return [0, 1]
    }

    const v1 = eigenvector(l1)
    const v2 = eigenvector(l2)
    return { values, vectors: [v1, v2], complex: false }
  }

  const re = tr / 2
  const im = Math.sqrt(-disc) / 2
  return {
    values: [re, im],
    vectors: [],
    complex: true,
  }
}

/**
 * Apply 2×2 matrix A to vector v: A * v.
 */
export function apply2x2(A: number[][], v: [number, number]): [number, number] {
  return [A[0][0] * v[0] + A[0][1] * v[1], A[1][0] * v[0] + A[1][1] * v[1]]
}

/**
 * Sample points on the unit circle (for visualization).
 */
export function unitCirclePoints(n: number): [number, number][] {
  const pts: [number, number][] = []
  for (let i = 0; i <= n; i++) {
    const t = (2 * Math.PI * i) / n
    pts.push([Math.cos(t), Math.sin(t)])
  }
  return pts
}
