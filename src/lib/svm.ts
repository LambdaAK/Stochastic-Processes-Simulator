/**
 * SVM via Sequential Minimal Optimization (SMO, Platt 1998).
 *
 * Supports linear, RBF, and polynomial kernels.
 * Hard margin (C = ∞) and soft margin (finite C).
 *
 * Dual problem:
 *   max  Σᵢ αᵢ − ½ Σᵢⱼ αᵢαⱼ yᵢyⱼ K(xᵢ,xⱼ)
 *   s.t. 0 ≤ αᵢ ≤ C,   Σᵢ αᵢ yᵢ = 0
 */

// ── Types ───────────────────────────────────────────────────────────────────

export type KernelType = 'linear' | 'rbf' | 'poly'

export type KernelParams = {
  type: KernelType
  /** RBF: K(xᵢ,xⱼ) = exp(−γ‖xᵢ−xⱼ‖²).  Default 0.5. */
  gamma?: number
  /** Poly: K(xᵢ,xⱼ) = (xᵢ·xⱼ + coef0)^degree.  Default 3. */
  degree?: number
  /** Poly constant offset.  Default 1. */
  coef0?: number
}

export type DataPoint = { x: number; y: number; label: 1 | -1 }

export type SVMModel = {
  alphas: number[]
  b: number
  supportVectorIndices: number[]
  kernel: KernelParams
  /** Primal weights — only valid for linear kernel (0 otherwise). */
  w1: number
  w2: number
  /** 2/‖w‖ for linear kernel; 0 for non-linear. */
  margin: number
}

export type SVMResult = {
  model: SVMModel
  converged: boolean
  iterations: number
}

export type PresetId =
  | 'blobs'
  | 'wide-margin'
  | 'near-boundary'
  | 'overlapping'
  | 'circles'
  | 'xor'
  | 'moons'

// ── Kernel functions ─────────────────────────────────────────────────────────

export function makeKernel(
  params: KernelParams,
): (xi: [number, number], xj: [number, number]) => number {
  if (params.type === 'rbf') {
    const gamma = params.gamma ?? 0.5
    return (xi, xj) => {
      const dx = xi[0] - xj[0]
      const dy = xi[1] - xj[1]
      return Math.exp(-gamma * (dx * dx + dy * dy))
    }
  }
  if (params.type === 'poly') {
    const degree = params.degree ?? 3
    const coef0 = params.coef0 ?? 1
    return (xi, xj) => Math.pow(xi[0] * xj[0] + xi[1] * xj[1] + coef0, degree)
  }
  // linear
  return (xi, xj) => xi[0] * xj[0] + xi[1] * xj[1]
}

// ── Training ─────────────────────────────────────────────────────────────────

/**
 * Train a kernelised SVM with SMO.
 *
 * @param data         Training points (labels +1 / −1).
 * @param C            Regularisation. Infinity → hard-margin.
 * @param kernelParams Kernel choice and hyperparameters.
 * @param maxIter      Max SMO passes.
 * @param tol          KKT violation tolerance.
 */
export function trainSVM(
  data: DataPoint[],
  C: number,
  kernelParams: KernelParams = { type: 'linear' },
  maxIter = 1000,
  tol = 1e-4,
): SVMResult {
  const n = data.length
  if (n === 0) throw new Error('No data points.')

  const Ceff = isFinite(C) ? C : 1e6
  const X = data.map((p) => [p.x, p.y] as [number, number])
  const Y = data.map((p) => p.label as number)

  const kernelFn = makeKernel(kernelParams)

  // Precompute kernel matrix
  const K: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => kernelFn(X[i], X[j])),
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
      const yiEi = Y[i] * Ei
      if (!((yiEi < -tol && ai < Ceff) || (yiEi > tol && ai > 0))) continue

      // Pick j: maximise |Eᵢ − Eⱼ|
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

      let L: number, H: number
      if (Y[i] !== Y[j]) {
        L = Math.max(0, aj_old - ai_old)
        H = Math.min(Ceff, Ceff + aj_old - ai_old)
      } else {
        L = Math.max(0, ai_old + aj_old - Ceff)
        H = Math.min(Ceff, ai_old + aj_old)
      }
      if (H - L < 1e-12) continue

      const eta = K[i][i] + K[j][j] - 2 * K[i][j]
      if (eta <= 1e-12) continue

      let aj_new = aj_old + Y[j] * (Ei - Ej) / eta
      aj_new = Math.max(L, Math.min(H, aj_new))
      if (Math.abs(aj_new - aj_old) < 1e-8) continue

      const ai_new = ai_old + Y[i] * Y[j] * (aj_old - aj_new)

      const b1 =
        b - Ei - Y[i] * (ai_new - ai_old) * K[i][i] - Y[j] * (aj_new - aj_old) * K[i][j]
      const b2 =
        b - Ej - Y[i] * (ai_new - ai_old) * K[i][j] - Y[j] * (aj_new - aj_old) * K[j][j]

      const bOld = b
      if (ai_new > 0 && ai_new < Ceff) b = b1
      else if (aj_new > 0 && aj_new < Ceff) b = b2
      else b = (b1 + b2) / 2

      const db = b - bOld
      const dai = ai_new - ai_old
      const daj = aj_new - aj_old

      alpha[i] = ai_new
      alpha[j] = aj_new

      for (let k = 0; k < n; k++) {
        E[k] += dai * Y[i] * K[i][k] + daj * Y[j] * K[j][k] + db
      }

      changed++
    }

    if (changed === 0) break
  }

  // Refine b from margin support vectors
  const svTol = tol * 10
  let bSum = 0
  let bCount = 0
  for (let i = 0; i < n; i++) {
    if (alpha[i] > svTol && alpha[i] < Ceff - svTol) {
      let fi_no_b = 0
      for (let k = 0; k < n; k++) fi_no_b += alpha[k] * Y[k] * K[k][i]
      bSum += Y[i] - fi_no_b
      bCount++
    }
  }
  if (bCount > 0) b = bSum / bCount

  // SV detection: alpha threshold + geometric condition (kernel-agnostic)
  const geomTol = 0.1
  const supportVectorIndices: number[] = []
  for (let i = 0; i < n; i++) {
    if (alpha[i] <= svTol) continue
    let fi = b
    for (let k = 0; k < n; k++) fi += alpha[k] * Y[k] * K[k][i]
    if (Y[i] * fi <= 1 + geomTol) supportVectorIndices.push(i)
  }

  // Primal weights — only meaningful for linear kernel
  let w1 = 0
  let w2 = 0
  if (kernelParams.type === 'linear') {
    for (let i = 0; i < n; i++) {
      w1 += alpha[i] * Y[i] * X[i][0]
      w2 += alpha[i] * Y[i] * X[i][1]
    }
  }
  const wNorm = Math.sqrt(w1 * w1 + w2 * w2)
  const margin = wNorm > 1e-10 ? 2 / wNorm : 0

  return {
    model: { alphas: alpha, b, supportVectorIndices, kernel: kernelParams, w1, w2, margin },
    converged: iter < maxIter,
    iterations: iter,
  }
}

// ── Predictor factory ─────────────────────────────────────────────────────────

/**
 * Returns a fast prediction closure.
 * For linear kernel uses the primal O(1) form.
 * For non-linear kernels sums over dual SVs.
 */
export function makePredictor(
  data: DataPoint[],
  model: SVMModel,
): (x: number, y: number) => 1 | -1 {
  if (model.kernel.type === 'linear') {
    const { w1, w2, b } = model
    return (x, y) => (w1 * x + w2 * y + b >= 0 ? 1 : -1)
  }

  const kernelFn = makeKernel(model.kernel)
  const b = model.b
  // Pre-extract significant dual variables
  const activeSVs = model.alphas
    .map((a, i) => ({ a, i }))
    .filter(({ a }) => a > 1e-6)
    .map(({ a, i }) => ({
      xy: [data[i].x, data[i].y] as [number, number],
      ay: a * (data[i].label as number),
    }))

  return (x, y) => {
    const xp: [number, number] = [x, y]
    let decision = b
    for (const sv of activeSVs) decision += sv.ay * kernelFn(sv.xy, xp)
    return decision >= 0 ? 1 : -1
  }
}

// ── Margin lines (linear kernel only) ────────────────────────────────────────

/**
 * Clip-box endpoints of the line w·x + b = targetValue within the domain.
 * targetValue = 0 → boundary, ±1 → margin planes.
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

// ── Dataset generation ────────────────────────────────────────────────────────

export function generateDataset(
  preset: PresetId,
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
  } else if (preset === 'overlapping') {
    for (let i = 0; i < nPerClass; i++) {
      out.push({ x: 1.0 + gaussian(), y: gaussian(), label: 1 })
      out.push({ x: -1.0 + gaussian(), y: gaussian(), label: -1 })
    }
  } else if (preset === 'circles') {
    // Concentric circles — showcases RBF kernel
    for (let i = 0; i < nPerClass; i++) {
      const a1 = rand() * 2 * Math.PI
      const r1 = 0.9 + 0.2 * (rand() - 0.5)
      out.push({ x: r1 * Math.cos(a1), y: r1 * Math.sin(a1), label: 1 })
      const a2 = rand() * 2 * Math.PI
      const r2 = 2.2 + 0.25 * (rand() - 0.5)
      out.push({ x: r2 * Math.cos(a2), y: r2 * Math.sin(a2), label: -1 })
    }
  } else if (preset === 'xor') {
    // XOR pattern — showcases polynomial / RBF kernel
    const half = Math.ceil(nPerClass / 2)
    for (let i = 0; i < half; i++) {
      out.push({ x: 1.5 + 0.45 * gaussian(), y: 1.5 + 0.45 * gaussian(), label: 1 })
      out.push({ x: -1.5 + 0.45 * gaussian(), y: -1.5 + 0.45 * gaussian(), label: 1 })
      out.push({ x: -1.5 + 0.45 * gaussian(), y: 1.5 + 0.45 * gaussian(), label: -1 })
      out.push({ x: 1.5 + 0.45 * gaussian(), y: -1.5 + 0.45 * gaussian(), label: -1 })
    }
  } else {
    // moons — two interleaved half-circles
    for (let i = 0; i < nPerClass; i++) {
      const t1 = rand() * Math.PI
      out.push({
        x: 1.5 * Math.cos(t1) + 0.22 * gaussian(),
        y: 1.5 * Math.sin(t1) + 0.3 + 0.22 * gaussian(),
        label: 1,
      })
      const t2 = rand() * Math.PI
      out.push({
        x: 1.5 - 1.5 * Math.cos(t2) + 0.22 * gaussian(),
        y: -1.5 * Math.sin(t2) - 0.3 + 0.22 * gaussian(),
        label: -1,
      })
    }
  }

  return out
}
