/**
 * Convex Quadratic Program solver — Active-Set Method.
 *
 * Solves:
 *   min  ½xᵀQx + cᵀx
 *   s.t. Ax ≤ b        (inequality constraints)
 *        x ≥ 0         (non-negativity, optional)
 *
 * Q must be symmetric positive (semi)definite.
 * Uses the active-set method (Nocedal & Wright, Ch. 16).
 * KKT sub-problems are solved by Gaussian elimination.
 */

const EPS = 1e-9
const MAX_ITER = 300

// ─── Linear algebra helpers ───────────────────────────────────────────────────

/**
 * Solve Ax = b via Gaussian elimination with partial pivoting.
 * Returns x or null if A is singular.
 */
function gaussElim(A: number[][], b: number[]): number[] | null {
  const n = b.length
  // Build augmented matrix [A | b]
  const M: number[][] = A.map((row, i) => [...row, b[i]])

  for (let col = 0; col < n; col++) {
    // Find pivot (partial pivoting)
    let maxRow = col
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(M[row][col]) > Math.abs(M[maxRow][col])) maxRow = row
    }
    ;[M[col], M[maxRow]] = [M[maxRow], M[col]]

    if (Math.abs(M[col][col]) < EPS) return null // singular

    // Eliminate
    for (let row = 0; row < n; row++) {
      if (row === col) continue
      const factor = M[row][col] / M[col][col]
      for (let j = col; j <= n; j++) {
        M[row][j] -= factor * M[col][j]
      }
    }
  }

  return M.map((row, i) => row[n] / row[i])
}

function matVec(A: number[][], x: number[]): number[] {
  return A.map((row) => row.reduce((s, a, j) => s + a * x[j], 0))
}

function matTranspose(A: number[][]): number[][] {
  if (A.length === 0) return []
  const m = A.length
  const n = A[0].length
  const T: number[][] = Array.from({ length: n }, () => new Array(m).fill(0))
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) T[j][i] = A[i][j]
  return T
}

function vecAdd(a: number[], b: number[]): number[] {
  return a.map((v, i) => v + b[i])
}

function vecScale(a: number[], s: number): number[] {
  return a.map((v) => v * s)
}

function vecNorm(a: number[]): number {
  return Math.sqrt(a.reduce((s, v) => s + v * v, 0))
}

// ─── Equality-constrained QP via KKT ─────────────────────────────────────────

/**
 * Solve the equality-constrained QP:
 *   min ½xᵀQx + gᵀx   s.t.  Cw x = rhs
 *
 * KKT system:
 *   [Q   Cwᵀ] [x]   [-g ]
 *   [Cw   0 ] [λ] = [rhs]
 *
 * Returns {x, lambda} or null if the KKT matrix is singular.
 */
function solveEqualityQP(
  Q: number[][],
  g: number[],
  Cw: number[][],
  rhs: number[]
): { x: number[]; lambda: number[] } | null {
  const n = g.length
  const k = rhs.length

  if (k === 0) {
    // Unconstrained: Qx = -g
    const x = gaussElim(Q, g.map((v) => -v))
    return x ? { x, lambda: [] } : null
  }

  const Cwt = matTranspose(Cw)
  const size = n + k

  // Build KKT matrix
  const KKT: number[][] = Array.from({ length: size }, () => new Array(size).fill(0))
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) KKT[i][j] = Q[i][j]
  for (let i = 0; i < n; i++)
    for (let j = 0; j < k; j++) KKT[i][n + j] = Cwt[i][j]
  for (let i = 0; i < k; i++)
    for (let j = 0; j < n; j++) KKT[n + i][j] = Cw[i][j]

  const rhsFull: number[] = [...g.map((v) => -v), ...rhs]
  const sol = gaussElim(KKT, rhsFull)
  if (!sol) return null

  return { x: sol.slice(0, n), lambda: sol.slice(n) }
}

// ─── Types ────────────────────────────────────────────────────────────────────

export type QPStatus = 'optimal' | 'infeasible' | 'max_iter' | 'error'

export type QPProblem = {
  /** n×n symmetric positive (semi)definite matrix */
  Q: number[][]
  /** Linear cost term, length n */
  c: number[]
  /** Inequality constraint matrix, m×n (Ax ≤ b) */
  A: number[][]
  /** Inequality RHS, length m */
  b: number[]
  /** Whether to enforce x ≥ 0 */
  nonNeg: boolean
}

export type QPResult = {
  status: QPStatus
  x?: number[]
  objectiveValue?: number
  iterations: number
  message?: string
}

// ─── Objective evaluation ─────────────────────────────────────────────────────

export function qpObjective(Q: number[][], c: number[], x: number[]): number {
  const Qx = matVec(Q, x)
  return 0.5 * x.reduce((s, v, i) => s + v * Qx[i], 0) + c.reduce((s, v, i) => s + v * x[i], 0)
}

// ─── Main solver ──────────────────────────────────────────────────────────────

/**
 * Solve: min ½xᵀQx + cᵀx  s.t.  Ax ≤ b,  x ≥ 0 (if nonNeg)
 *
 * Active-set method (Nocedal & Wright, Ch. 16):
 * 1. Starting point x = 0 (feasible when b ≥ 0).
 * 2. Maintain working set W of active inequality constraints.
 * 3. Each iteration: solve equality QP on working set → direction d.
 *    - d ≈ 0 & all λ ≥ 0 → optimal.
 *    - d ≈ 0 & some λ < 0 → remove most-negative multiplier constraint from W.
 *    - d ≠ 0 → line search for step length, add blocking constraint to W.
 */
export function solveQP(problem: QPProblem): QPResult {
  const { Q, c, nonNeg } = problem
  const n = c.length

  // ── Build full inequality constraint set ──────────────────────────────────
  const ineqA: number[][] = problem.A.map((row) => [...row])
  const ineqB: number[] = [...problem.b]

  if (nonNeg) {
    for (let j = 0; j < n; j++) {
      const row = new Array(n).fill(0)
      row[j] = -1 // -xⱼ ≤ 0  →  xⱼ ≥ 0
      ineqA.push(row)
      ineqB.push(0)
    }
  }

  const m = ineqA.length

  // ── Unconstrained case ────────────────────────────────────────────────────
  if (m === 0) {
    const x = gaussElim(Q, c.map((v) => -v))
    if (!x) return { status: 'error', iterations: 0, message: 'Q is singular — unconstrained QP has no unique solution.' }
    return { status: 'optimal', x, objectiveValue: qpObjective(Q, c, x), iterations: 0 }
  }

  // ── Phase 1: feasibility at x = 0 ────────────────────────────────────────
  let x = new Array(n).fill(0)
  const feasible0 = ineqA.every(
    (row, i) => row.reduce((s, a, j) => s + a * x[j], 0) <= ineqB[i] + EPS
  )

  if (!feasible0) {
    return {
      status: 'infeasible',
      iterations: 0,
      message:
        'Problem is infeasible at x = 0. For problems with x ≥ 0, ensure all inequality RHS values (b) are ≥ 0.',
    }
  }

  // ── Initialize working set ────────────────────────────────────────────────
  // Start with all constraints that are active (tight) at x = 0
  const W = new Set<number>()
  for (let i = 0; i < m; i++) {
    const lhs = ineqA[i].reduce((s, a, j) => s + a * x[j], 0)
    if (Math.abs(lhs - ineqB[i]) < EPS * 100) W.add(i)
  }

  // ── Active-set iterations ─────────────────────────────────────────────────
  for (let iter = 0; iter < MAX_ITER; iter++) {
    // Gradient g = Qx + c
    const g = vecAdd(matVec(Q, x), c)

    const Warray = Array.from(W)
    const Cw = Warray.map((i) => ineqA[i])
    const rhs = new Array(Warray.length).fill(0)

    // Solve EQP for search direction d:
    //   [Q  Cwᵀ][d]   [-g]
    //   [Cw  0 ][λ] = [ 0]
    const eqp = solveEqualityQP(Q, g, Cw, rhs)

    if (!eqp) {
      // KKT system singular — drop last inequality from W and retry
      const lastIneq = [...Warray].reverse().find((i) => i < m)
      if (lastIneq !== undefined) {
        W.delete(lastIneq)
        continue
      }
      return { status: 'error', iterations: iter, message: 'KKT system is singular.' }
    }

    const d = eqp.x
    const lambda = eqp.lambda
    const dNorm = vecNorm(d)

    if (dNorm < EPS * 1e5) {
      // d ≈ 0 — check KKT multipliers for inequality constraints in W
      let minLambda = Infinity
      let minIdx = -1
      for (let k = 0; k < Warray.length; k++) {
        if (lambda[k] < minLambda) {
          minLambda = lambda[k]
          minIdx = Warray[k]
        }
      }

      if (minLambda >= -EPS || W.size === 0) {
        // All multipliers ≥ 0 → KKT satisfied → OPTIMAL
        return {
          status: 'optimal',
          x,
          objectiveValue: qpObjective(Q, c, x),
          iterations: iter,
        }
      }

      // Remove constraint with most negative multiplier
      W.delete(minIdx)
      continue
    }

    // d ≠ 0 — compute step length α via min-ratio test over inactive constraints
    let alpha = 1.0
    let blockingConstraint = -1

    for (let i = 0; i < m; i++) {
      if (W.has(i)) continue
      const aId = ineqA[i].reduce((s, a, j) => s + a * d[j], 0)
      if (aId > EPS) {
        const slack = ineqB[i] - ineqA[i].reduce((s, a, j) => s + a * x[j], 0)
        const ratio = slack / aId
        if (ratio < alpha - EPS) {
          alpha = Math.max(0, ratio)
          blockingConstraint = i
        }
      }
    }

    x = vecAdd(x, vecScale(d, alpha))

    if (blockingConstraint >= 0) {
      W.add(blockingConstraint)
    }
  }

  return {
    status: 'max_iter',
    x,
    objectiveValue: qpObjective(Q, c, x),
    iterations: MAX_ITER,
    message: `Maximum iterations (${MAX_ITER}) reached.`,
  }
}
