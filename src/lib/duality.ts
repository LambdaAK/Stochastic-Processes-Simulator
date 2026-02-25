/**
 * Primal–dual visualization for a 2D convex QP with one linear inequality.
 *
 * Primal:  min  ½xᵀQx + cᵀx   s.t.  aᵀx ≥ b
 * Dual:    max  q(λ)   s.t.  λ ≥ 0
 * where   q(λ) = inf_x L(x,λ) = inf_x [ ½xᵀQx + cᵀx + λ(b − aᵀx) ].
 *
 * For convex Q and linear constraint, strong duality holds when Slater holds
 * (e.g. strict feasibility aᵀx > b for some x). We compute primal optimum via
 * KKT and dual function q(λ) in closed form for 2×2 Q.
 */

const EPS = 1e-9

/** 2×2 symmetric positive definite Q, 2-vec c, constraint aᵀx ≥ b */
export type DualityProblem = {
  /** 2×2 objective Hessian (symmetric PD) */
  Q: [[number, number], [number, number]]
  /** Linear cost, length 2 */
  c: [number, number]
  /** Constraint: aᵀx ≥ b */
  a: [number, number]
  b: number
}

export type DualityResult = {
  /** Primal optimal x */
  xStar: [number, number]
  /** Primal optimal value ½xᵀQx + cᵀx */
  primalValue: number
  /** Dual optimal λ */
  lambdaStar: number
  /** Dual optimal value q(λ*) */
  dualValue: number
  /** Constraint active at optimum (aᵀx* = b) */
  constraintActive: boolean
  /** Whether unconstrained minimizer is feasible (then λ* = 0) */
  unconstrainedFeasible: boolean
  /** Sample points for plotting q(λ): { λ, q } */
  dualCurve: { lambda: number; q: number }[]
}

function inv2x2(Q: [[number, number], [number, number]]): [[number, number], [number, number]] {
  const [[a, b], [c, d]] = Q
  const det = a * d - b * c
  if (Math.abs(det) < EPS) return [[0, 0], [0, 0]]
  return [
    [d / det, -b / det],
    [-c / det, a / det],
  ]
}

function matVec2(Q: [[number, number], [number, number]], x: [number, number]): [number, number] {
  return [
    Q[0][0] * x[0] + Q[0][1] * x[1],
    Q[1][0] * x[0] + Q[1][1] * x[1],
  ]
}

function dot(a: [number, number], b: [number, number]): number {
  return a[0] * b[0] + a[1] * b[1]
}

/** Objective ½xᵀQx + cᵀx */
function objective(Q: DualityProblem['Q'], c: DualityProblem['c'], x: [number, number]): number {
  const Qx = matVec2(Q, x)
  return 0.5 * dot(x, Qx) + dot(c, x)
}

/**
 * Solve primal and dual, and sample q(λ) for plotting.
 * Primal: min ½xᵀQx + cᵀx  s.t.  aᵀx ≥ b.
 * KKT: Qx + c = λa,  λ ≥ 0,  λ(aᵀx − b) = 0,  aᵀx ≥ b.
 * Unconstrained minimizer: x_u = −Q⁻¹c. If aᵀx_u ≥ b then x* = x_u, λ* = 0.
 * Else constraint active: aᵀx = b and Qx + c = λa ⇒ x = Q⁻¹(λa − c) ⇒
 * aᵀQ⁻¹(λa − c) = b ⇒ λ = (b + aᵀQ⁻¹c) / (aᵀQ⁻¹a). Then x* = Q⁻¹(λ*a − c).
 *
 * Dual: q(λ) = λb − ½(λa − c)ᵀQ⁻¹(λa − c). Max over λ ≥ 0.
 */
export function solveDuality(problem: DualityProblem, lambdaSamples = 80): DualityResult {
  const { Q, c, a, b } = problem
  const Qinv = inv2x2(Q)

  // Unconstrained minimizer: Q x_u + c = 0 => x_u = -Q^{-1} c
  const xU: [number, number] = [
    -(Qinv[0][0] * c[0] + Qinv[0][1] * c[1]),
    -(Qinv[1][0] * c[0] + Qinv[1][1] * c[1]),
  ]
  const aTxU = dot(a, xU)
  const unconstrainedFeasible = aTxU >= b - EPS

  let xStar: [number, number]
  let primalValue: number
  let lambdaStar: number
  let constraintActive: boolean

  if (unconstrainedFeasible) {
    xStar = [xU[0], xU[1]]
    primalValue = objective(Q, c, xStar)
    lambdaStar = 0
    constraintActive = false
  } else {
    // aᵀQ⁻¹a and aᵀQ⁻¹c
    const QinvA: [number, number] = matVec2(Qinv, a)
    const aQinvA = dot(a, QinvA)
    const aQinvC = dot(a, [Qinv[0][0] * c[0] + Qinv[0][1] * c[1], Qinv[1][0] * c[0] + Qinv[1][1] * c[1]])
    if (Math.abs(aQinvA) < EPS) {
      // Degenerate: constraint parallel to null space; take x on boundary
      lambdaStar = 0
      constraintActive = true
      // Placeholder: use projection of xU onto line a'x = b
      const t = (b - aTxU) / (dot(a, a) + EPS)
      xStar = [xU[0] + t * a[0], xU[1] + t * a[1]]
    } else {
      lambdaStar = (b + aQinvC) / aQinvA
      if (lambdaStar < 0) lambdaStar = 0
      const lamA_minus_c: [number, number] = [lambdaStar * a[0] - c[0], lambdaStar * a[1] - c[1]]
      xStar = matVec2(Qinv, lamA_minus_c)
      constraintActive = true
    }
    primalValue = objective(Q, c, xStar)
  }

  // Dual value q(λ*) = λ*b − ½(λ*a − c)ᵀQ⁻¹(λ*a − c)
  const lamA_minus_c: [number, number] = [lambdaStar * a[0] - c[0], lambdaStar * a[1] - c[1]]
  const Qinv_d = matVec2(Qinv, lamA_minus_c)
  const dualValue = lambdaStar * b - 0.5 * dot(lamA_minus_c, Qinv_d)

  // Sample q(λ) for λ in [0, λ_max]
  const lambdaMax = Math.max(lambdaStar * 1.5, 1, 2)
  const dualCurve: { lambda: number; q: number }[] = []
  for (let i = 0; i <= lambdaSamples; i++) {
    const lam = (i / lambdaSamples) * lambdaMax
    const d: [number, number] = [lam * a[0] - c[0], lam * a[1] - c[1]]
    const QinvD = matVec2(Qinv, d)
    const q = lam * b - 0.5 * dot(d, QinvD)
    dualCurve.push({ lambda: lam, q })
  }

  return {
    xStar,
    primalValue,
    lambdaStar,
    dualValue,
    constraintActive,
    unconstrainedFeasible,
    dualCurve,
  }
}

/** Evaluate primal objective at a point (for contour plotting) */
export function evalPrimalObjective(
  problem: DualityProblem,
  x: number,
  y: number
): number {
  const { Q, c } = problem
  return objective(Q, c, [x, y])
}

/** Check if (x,y) is feasible: aᵀx ≥ b */
export function isFeasible(problem: DualityProblem, x: number, y: number): boolean {
  const { a, b } = problem
  return a[0] * x + a[1] * y >= b - EPS
}

// ─── LP dual: primal min c'x s.t. Ax ≥ b  →  dual max b'λ s.t. A'λ = c, λ ≥ 0 ───

export type LPPrimal = {
  /** Objective: min c'x, length n */
  c: number[]
  /** Constraint matrix: Ax ≥ b, m×n */
  A: number[][]
  /** RHS, length m */
  b: number[]
}

/** Dual of min c'x s.t. Ax ≥ b: max b'λ s.t. A'λ = c, λ ≥ 0. Returns data for the dual. */
export function computeLPDual(primal: LPPrimal): {
  /** Dual objective coeffs: max b'λ */
  obj: number[]
  /** Dual equality constraints: (A')λ = c; matrix is n×m (n rows, m cols) */
  eqA: number[][]
  /** RHS of A'λ = c, length n */
  eqB: number[]
  /** Dual vars λ ≥ 0 */
  lambdaNonNeg: true
} {
  const { c, A, b } = primal
  const n = c.length
  const eqA: number[][] = []
  for (let j = 0; j < n; j++) {
    eqA.push(A.map((row) => row[j]))
  }
  return { obj: [...b], eqA, eqB: [...c], lambdaNonNeg: true }
}
