/**
 * Linear SVM via Sequential Minimal Optimization (SMO, Platt 1998).
 *
 * Hard-margin SVM:  min  ½‖w‖²          s.t.  yᵢ(w·xᵢ + b) ≥ 1   ∀i
 * Soft-margin SVM:  min  ½‖w‖² + C Σξᵢ  s.t.  yᵢ(w·xᵢ + b) ≥ 1 − ξᵢ, ξᵢ ≥ 0
 *
 * Dual (linear kernel K(xᵢ,xⱼ) = xᵢ · xⱼ):
 *   max  Σᵢ αᵢ − ½ Σᵢⱼ αᵢαⱼ yᵢyⱼ (xᵢ·xⱼ)
 *   s.t. 0 ≤ αᵢ ≤ C,   Σᵢ αᵢ yᵢ = 0
 *
 * w = Σᵢ αᵢ yᵢ xᵢ.  Support vectors: αᵢ > 0.
 */

export type DataPoint = { x: number; y: number; label: 1 | -1 }

export type SVMModel = {
  w1: number
  w2: number
  b: number
  alphas: number[]
  supportVectorIndices: number[]
  margin: number // geometric margin: 2 / ‖w‖
}

export type SVMResult = {
  model: SVMModel
  converged: boolean
  iterations: number
}

/**
 * Train a linear SVM using the SMO algorithm.
 *
 * @param data    Training points with labels +1 / −1.
 * @param C       Regularisation parameter.  Infinity → hard-margin SVM.
 * @param maxIter Maximum SMO outer passes.
 * @param tol     KKT violation tolerance.
 */
export function trainSVM(
  data: DataPoint[],
  C: number,
  maxIter = 1000,
  tol = 1e-4,
): SVMResult {
  const n = data.length
  if (n === 0) throw new Error('No data points.')

  // Hard-margin: use large finite C to avoid arithmetic overflow
  const Ceff = isFinite(C) ? C : 1e6

  const X = data.map((p) => [p.x, p.y])
  const Y = data.map((p) => p.label as number)

  // Precompute linear kernel matrix: K[i][j] = xᵢ · xⱼ
  const K: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => X[i][0] * X[j][0] + X[i][1] * X[j][1]),
  )

  const alpha = new Array<number>(n).fill(0)
  let b = 0

  // Error cache: E[i] = f(xᵢ) − yᵢ.  Initially f = 0, so E[i] = −Y[i].
  const E = Y.map((y) => -y)

  let iter = 0
  for (iter = 0; iter < maxIter; iter++) {
    let changed = 0

    for (let i = 0; i < n; i++) {
      const Ei = E[i]
      const ai = alpha[i]

      // KKT: yᵢ · f(xᵢ) = 1 + yᵢ · Eᵢ
      // Violation: (yᵢ·Eᵢ < −tol && αᵢ < C) OR (yᵢ·Eᵢ > tol && αᵢ > 0)
      const yiEi = Y[i] * Ei
      if (!((yiEi < -tol && ai < Ceff) || (yiEi > tol && ai > 0))) continue

      // Pick j: heuristic — maximise |Eᵢ − Eⱼ|
      let j = -1
      let maxDE = 0
      for (let k = 0; k < n; k++) {
        if (k === i) continue
        const de = Math.abs(Ei - E[k])
        if (de > maxDE) {
          maxDE = de
          j = k
        }
      }
      if (j === -1) continue

      const Ej = E[j]
      const ai_old = alpha[i]
      const aj_old = alpha[j]

      // Box bounds for αⱼ
      let L: number, H: number
      if (Y[i] !== Y[j]) {
        L = Math.max(0, aj_old - ai_old)
        H = Math.min(Ceff, Ceff + aj_old - ai_old)
      } else {
        L = Math.max(0, ai_old + aj_old - Ceff)
        H = Math.min(Ceff, ai_old + aj_old)
      }
      if (H - L < 1e-12) continue

      // 2nd-order step: η = ‖xᵢ − xⱼ‖² (linear kernel)
      const eta = K[i][i] + K[j][j] - 2 * K[i][j]
      if (eta <= 1e-12) continue

      let aj_new = aj_old + Y[j] * (Ei - Ej) / eta
      aj_new = Math.max(L, Math.min(H, aj_new))
      if (Math.abs(aj_new - aj_old) < 1e-8) continue

      // Update αᵢ (enforces Σ αᵢ yᵢ = 0)
      const ai_new = ai_old + Y[i] * Y[j] * (aj_old - aj_new)

      // Update b
      const b1 =
        b -
        Ei -
        Y[i] * (ai_new - ai_old) * K[i][i] -
        Y[j] * (aj_new - aj_old) * K[i][j]
      const b2 =
        b -
        Ej -
        Y[i] * (ai_new - ai_old) * K[i][j] -
        Y[j] * (aj_new - aj_old) * K[j][j]

      const bOld = b
      if (ai_new > 0 && ai_new < Ceff) b = b1
      else if (aj_new > 0 && aj_new < Ceff) b = b2
      else b = (b1 + b2) / 2

      const db = b - bOld
      const dai = ai_new - ai_old
      const daj = aj_new - aj_old

      alpha[i] = ai_new
      alpha[j] = aj_new

      // Update error cache: ΔEₖ = Δαᵢ·yᵢ·Kᵢₖ + Δαⱼ·yⱼ·Kⱼₖ + Δb
      for (let k = 0; k < n; k++) {
        E[k] += dai * Y[i] * K[i][k] + daj * Y[j] * K[j][k] + db
      }

      changed++
    }

    if (changed === 0) break
  }

  // Recompute b from margin support vectors for numerical accuracy
  const svTol = tol * 10
  let bSum = 0
  let bCount = 0
  for (let i = 0; i < n; i++) {
    if (alpha[i] > svTol && alpha[i] < Ceff - svTol) {
      let fi = 0
      for (let k = 0; k < n; k++) fi += alpha[k] * Y[k] * K[k][i]
      bSum += Y[i] - fi
      bCount++
    }
  }
  if (bCount > 0) b = bSum / bCount

  // Primal: w = Σᵢ αᵢ yᵢ xᵢ
  let w1 = 0
  let w2 = 0
  for (let i = 0; i < n; i++) {
    w1 += alpha[i] * Y[i] * X[i][0]
    w2 += alpha[i] * Y[i] * X[i][1]
  }

  // Detect support vectors using both conditions:
  //   1. α_i > threshold  (meaningful dual variable)
  //   2. y_i·(w·x_i + b) ≤ 1 + geomTol  (point lies on or inside the margin slab)
  //
  // Condition 2 filters out points whose alpha drifted above zero due to
  // floating-point accumulation in the error cache but are geometrically
  // well outside the margin and should have α = 0.
  const geomTol = 0.1
  const supportVectorIndices: number[] = []
  for (let i = 0; i < n; i++) {
    if (alpha[i] <= svTol) continue
    const fi = w1 * X[i][0] + w2 * X[i][1] + b  // primal form, exact for linear kernel
    if (Y[i] * fi <= 1 + geomTol) supportVectorIndices.push(i)
  }

  const wNorm = Math.sqrt(w1 * w1 + w2 * w2)
  const margin = wNorm > 1e-10 ? 2 / wNorm : 0

  return {
    model: { w1, w2, b, alphas: alpha, supportVectorIndices, margin },
    converged: iter < maxIter,
    iterations: iter,
  }
}

/** Predict label (+1 or −1) for a 2D point. */
export function predict(
  x: number,
  y: number,
  model: Pick<SVMModel, 'w1' | 'w2' | 'b'>,
): 1 | -1 {
  return model.w1 * x + model.w2 * y + model.b >= 0 ? 1 : -1
}

/**
 * Returns the two clip-box endpoints of the line  w·x + b = targetValue
 * within [xMin,xMax] × [yMin,yMax].
 *
 * targetValue = 0  → decision boundary
 * targetValue = +1 → positive-class margin
 * targetValue = −1 → negative-class margin
 */
export function marginLine(
  model: Pick<SVMModel, 'w1' | 'w2' | 'b'>,
  targetValue: number,
  xMin: number,
  xMax: number,
  yMin: number,
  yMax: number,
): { x: number; y: number }[] {
  const { w1, w2 } = model
  // Rewrite as w·x + bEff = 0 so existing clipping logic works
  const bEff = model.b - targetValue
  if (Math.abs(w1) < 1e-10 && Math.abs(w2) < 1e-10) return []

  const pts: { x: number; y: number }[] = []

  if (Math.abs(w2) > 1e-10) {
    const yAt = (x: number) => -(w1 * x + bEff) / w2
    const yL = yAt(xMin)
    const yR = yAt(xMax)
    if (yL >= yMin && yL <= yMax) pts.push({ x: xMin, y: yL })
    if (yR >= yMin && yR <= yMax) pts.push({ x: xMax, y: yR })
  }

  if (Math.abs(w1) > 1e-10) {
    const xAt = (y: number) => -(w2 * y + bEff) / w1
    const xB = xAt(yMin)
    const xT = xAt(yMax)
    if (xB > xMin && xB < xMax) pts.push({ x: xB, y: yMin })
    if (xT > xMin && xT < xMax) pts.push({ x: xT, y: yMax })
  }

  if (pts.length < 2) return []
  pts.sort((a, c) => a.x - c.x)
  return [pts[0], pts[pts.length - 1]]
}

/** Generate preset 2D binary datasets for the SVM demo. */
export function generateDataset(
  preset: 'blobs' | 'wide-margin' | 'near-boundary' | 'overlapping',
  nPerClass: number,
  rand: () => number,
): DataPoint[] {
  const gaussian = () =>
    Math.sqrt(-2 * Math.log(Math.max(rand(), 1e-15))) * Math.cos(2 * Math.PI * rand())

  const out: DataPoint[] = []

  if (preset === 'blobs') {
    for (let i = 0; i < nPerClass; i++) {
      out.push({ x: 1.8 + 0.7 * gaussian(), y: 1.8 + 0.7 * gaussian(), label: 1 })
      out.push({ x: -1.8 + 0.7 * gaussian(), y: -1.8 + 0.7 * gaussian(), label: -1 })
    }
  } else if (preset === 'wide-margin') {
    for (let i = 0; i < nPerClass; i++) {
      out.push({ x: 0.6 * gaussian(), y: 2.5 + 0.4 * gaussian(), label: 1 })
      out.push({ x: 0.6 * gaussian(), y: -2.5 + 0.4 * gaussian(), label: -1 })
    }
  } else if (preset === 'near-boundary') {
    for (let i = 0; i < nPerClass; i++) {
      out.push({ x: 0.4 + 1.0 * Math.abs(gaussian()), y: 1.2 * gaussian(), label: 1 })
      out.push({ x: -0.4 - 1.0 * Math.abs(gaussian()), y: 1.2 * gaussian(), label: -1 })
    }
  } else {
    // overlapping — ideal for soft-margin demo
    for (let i = 0; i < nPerClass; i++) {
      out.push({ x: 1.0 + gaussian(), y: gaussian(), label: 1 })
      out.push({ x: -1.0 + gaussian(), y: gaussian(), label: -1 })
    }
  }

  return out
}
