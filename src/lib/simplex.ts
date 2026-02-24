/**
 * Simplex LP solver using the Big-M method.
 *
 * Solves:
 *   min/max  cᵀx
 *   s.t.     A x  ⋈  b,   x ≥ 0
 *
 * Constraint signs: '<=' | '>=' | '='
 * All b_i must be finite; negative RHS rows are normalised by sign-flip.
 */

export type ConstraintSign = '<=' | '>=' | '='

export type LPProblem = {
  /** Objective coefficients (length n) */
  c: number[]
  /** Constraint matrix (m × n) */
  A: number[][]
  /** RHS vector (length m) */
  b: number[]
  /** One sign per constraint */
  signs: ConstraintSign[]
  /** 'min' | 'max' */
  direction: 'min' | 'max'
}

export type LPStatus = 'optimal' | 'infeasible' | 'unbounded'

export type SimplexTableauSnapshot = {
  data: number[][]   // (m+1) rows × (totalVars+1) cols; last col = RHS; row 0 = objective
  basis: number[]    // basic variable index for each constraint row (length m)
  varNames: string[]
}

export type LPResult = {
  status: LPStatus
  /** Objective value at optimum (undefined when not optimal) */
  optimalValue?: number
  /** Values of original decision variables x_1…x_n (undefined when not optimal) */
  variables?: number[]
  /** Final tableau snapshot */
  tableau?: SimplexTableauSnapshot
  /** Pivot history: [pivotRow, pivotCol] per iteration */
  pivotHistory: [number, number][]
}

const BIG_M = 1e6
const MAX_ITER = 500
const EPS = 1e-8

// ─── Internal helpers ────────────────────────────────────────────────────────

function pivotOn(tab: number[][], pivotRow: number, pivotCol: number): void {
  const m1 = tab.length      // m + 1 rows
  const n1 = tab[0].length   // totalVars + 1 cols
  const scale = tab[pivotRow][pivotCol]

  // Normalise pivot row
  for (let j = 0; j < n1; j++) {
    tab[pivotRow][j] /= scale
  }

  // Eliminate pivot column from every other row
  for (let i = 0; i < m1; i++) {
    if (i === pivotRow) continue
    const factor = tab[i][pivotCol]
    if (Math.abs(factor) < EPS) continue
    for (let j = 0; j < n1; j++) {
      tab[i][j] -= factor * tab[pivotRow][j]
    }
  }
}

// ─── Main solver ─────────────────────────────────────────────────────────────

export function solveSimplex(problem: LPProblem): LPResult {
  const { c: cOrig, A: AOrig, b: bOrig, signs, direction } = problem
  const n = cOrig.length   // number of decision variables
  const m = bOrig.length   // number of constraints

  // --- 1. Convert max → min by negating c ---
  const cMin = direction === 'max' ? cOrig.map((v) => -v) : [...cOrig]

  // --- 2. Normalise rows so b_i ≥ 0 ---
  const A: number[][] = AOrig.map((row) => [...row])
  const b: number[] = [...bOrig]
  const normSigns: ConstraintSign[] = [...signs]

  for (let i = 0; i < m; i++) {
    if (b[i] < 0) {
      // Flip row sign and invert inequality
      b[i] = -b[i]
      A[i] = A[i].map((v) => -v)
      if (normSigns[i] === '<=') normSigns[i] = '>='
      else if (normSigns[i] === '>=') normSigns[i] = '<='
      // '=' stays '='
    }
  }

  // --- 3. Count auxiliary variables ---
  let numSlack = 0
  let numArtificial = 0
  // For each constraint, record which auxiliary columns it gets
  type AuxInfo = { slackIdx?: number; artificialIdx?: number }
  const auxInfo: AuxInfo[] = []

  for (let i = 0; i < m; i++) {
    const info: AuxInfo = {}
    if (normSigns[i] === '<=') {
      info.slackIdx = n + numSlack++
    } else if (normSigns[i] === '>=') {
      info.slackIdx = n + numSlack++       // surplus (coefficient −1)
      info.artificialIdx = n + numSlack + numArtificial++
    } else {
      // '='
      info.artificialIdx = n + numSlack + numArtificial++
    }
    auxInfo.push(info)
  }

  // Total variables: decision + slack/surplus + artificials
  const totalVars = n + numSlack + numArtificial
  const numCols = totalVars + 1  // +1 for RHS

  // --- 4. Build variable names ---
  const varNames: string[] = []
  for (let j = 0; j < n; j++) varNames.push(`x${j + 1}`)
  for (let j = 0; j < numSlack; j++) varNames.push(`s${j + 1}`)
  for (let j = 0; j < numArtificial; j++) varNames.push(`a${j + 1}`)

  // --- 5. Build initial tableau ---
  // Row 0: objective row (reduced costs), rows 1..m: constraints
  const tab: number[][] = Array.from({ length: m + 1 }, () => new Array(numCols).fill(0))

  // Objective row: minimise cMin·x + M·Σ(artificials)
  for (let j = 0; j < n; j++) tab[0][j] = cMin[j]
  for (let k = 0; k < numArtificial; k++) {
    const col = n + numSlack + k
    tab[0][col] = BIG_M
  }

  // Constraint rows
  const basis: number[] = new Array(m).fill(-1)

  for (let i = 0; i < m; i++) {
    const row = i + 1
    // Decision variable coefficients
    for (let j = 0; j < n; j++) tab[row][j] = A[i][j]
    // RHS
    tab[row][numCols - 1] = b[i]

    const info = auxInfo[i]
    if (normSigns[i] === '<=') {
      // Slack: coefficient +1
      tab[row][info.slackIdx!] = 1
      basis[i] = info.slackIdx!
    } else if (normSigns[i] === '>=') {
      // Surplus: coefficient −1
      tab[row][info.slackIdx!] = -1
      // Artificial: coefficient +1
      tab[row][info.artificialIdx!] = 1
      basis[i] = info.artificialIdx!
    } else {
      // '=': artificial only
      tab[row][info.artificialIdx!] = 1
      basis[i] = info.artificialIdx!
    }
  }

  // --- 6. Subtract Big-M rows from objective for initial artificials ---
  // When artificial is basic, its reduced cost must be 0 in the objective row;
  // the current obj row has +M for each artificial column, so subtract M × constraint row.
  for (let i = 0; i < m; i++) {
    if (auxInfo[i].artificialIdx !== undefined) {
      const row = i + 1
      const M = BIG_M
      for (let j = 0; j < numCols; j++) {
        tab[0][j] -= M * tab[row][j]
      }
    }
  }

  // --- 7. Simplex pivot loop ---
  const pivotHistory: [number, number][] = []

  for (let iter = 0; iter < MAX_ITER; iter++) {
    // Find entering variable: most negative reduced cost in objective row
    let pivotCol = -1
    let minCost = -EPS
    for (let j = 0; j < totalVars; j++) {
      if (tab[0][j] < minCost) {
        minCost = tab[0][j]
        pivotCol = j
      }
    }

    if (pivotCol === -1) break  // optimal

    // Find leaving variable: minimum ratio test
    let pivotRow = -1
    let minRatio = Infinity
    for (let i = 1; i <= m; i++) {
      const coeff = tab[i][pivotCol]
      if (coeff > EPS) {
        const ratio = tab[i][numCols - 1] / coeff
        if (ratio < minRatio - EPS) {
          minRatio = ratio
          pivotRow = i
        }
      }
    }

    if (pivotRow === -1) {
      // Unbounded
      return { status: 'unbounded', pivotHistory }
    }

    pivotHistory.push([pivotRow, pivotCol])
    pivotOn(tab, pivotRow, pivotCol)
    basis[pivotRow - 1] = pivotCol
  }

  // --- 8. Check feasibility: any artificial still basic with value > EPS? ---
  const artificialStart = n + numSlack
  for (let i = 0; i < m; i++) {
    const basisVar = basis[i]
    if (basisVar >= artificialStart) {
      const val = tab[i + 1][numCols - 1]
      if (Math.abs(val) > EPS) {
        return { status: 'infeasible', pivotHistory }
      }
    }
  }

  // --- 9. Extract solution ---
  const xValues = new Array(n).fill(0)
  for (let i = 0; i < m; i++) {
    const basisVar = basis[i]
    if (basisVar < n) {
      xValues[basisVar] = tab[i + 1][numCols - 1]
    }
  }

  // Objective value in original direction
  let objVal = -tab[0][numCols - 1]  // tab[0][rhs] stores −z
  if (direction === 'max') objVal = -objVal

  // Final tableau snapshot (copy)
  const snapshot: SimplexTableauSnapshot = {
    data: tab.map((row) => [...row]),
    basis: [...basis],
    varNames,
  }

  return {
    status: 'optimal',
    optimalValue: objVal,
    variables: xValues,
    tableau: snapshot,
    pivotHistory,
  }
}

// ─── 2-D feasibility geometry ─────────────────────────────────────────────────

/**
 * For a 2-variable LP with constraints A x ≤ b (all ≤, already normalised),
 * return all vertices of the feasible polytope by intersecting every pair
 * of boundary lines (including the two non-negativity axes).
 */
export function feasibleVertices(
  A: number[][],
  b: number[],
  signs: ConstraintSign[]
): { x: number; y: number }[] {
  const m = b.length
  // Build list of lines: a[0]*x + a[1]*y = b
  type Line = { a: [number, number]; rhs: number }
  const lines: Line[] = []
  for (let i = 0; i < m; i++) {
    lines.push({ a: [A[i][0], A[i][1]], rhs: b[i] })
  }
  // Non-negativity boundaries: x=0, y=0
  lines.push({ a: [-1, 0], rhs: 0 })  // x ≥ 0 → -x ≤ 0
  lines.push({ a: [0, -1], rhs: 0 })  // y ≥ 0 → -y ≤ 0

  const vertices: { x: number; y: number }[] = []
  const L = lines.length

  for (let i = 0; i < L; i++) {
    for (let j = i + 1; j < L; j++) {
      const [a1, b1] = lines[i].a
      const [a2, b2] = lines[j].a
      const r1 = lines[i].rhs
      const r2 = lines[j].rhs
      const det = a1 * b2 - a2 * b1
      if (Math.abs(det) < EPS) continue
      const x = (r1 * b2 - r2 * b1) / det
      const y = (a1 * r2 - a2 * r1) / det

      // Check all constraints (including non-negativity)
      let feasible = true
      for (let k = 0; k < m; k++) {
        const lhs = A[k][0] * x + A[k][1] * y
        if (signs[k] === '<=') {
          if (lhs > b[k] + EPS) { feasible = false; break }
        } else if (signs[k] === '>=') {
          if (lhs < b[k] - EPS) { feasible = false; break }
        } else {
          if (Math.abs(lhs - b[k]) > EPS) { feasible = false; break }
        }
      }
      if (feasible && x >= -EPS && y >= -EPS) {
        vertices.push({ x: Math.max(0, x), y: Math.max(0, y) })
      }
    }
  }

  // Deduplicate
  const unique: { x: number; y: number }[] = []
  for (const v of vertices) {
    const dup = unique.some((u) => Math.abs(u.x - v.x) < EPS && Math.abs(u.y - v.y) < EPS)
    if (!dup) unique.push(v)
  }

  // Sort vertices by angle (convex hull order)
  const cx = unique.reduce((s, v) => s + v.x, 0) / (unique.length || 1)
  const cy = unique.reduce((s, v) => s + v.y, 0) / (unique.length || 1)
  unique.sort((a, b) => Math.atan2(a.y - cy, a.x - cx) - Math.atan2(b.y - cy, b.x - cx))

  return unique
}
