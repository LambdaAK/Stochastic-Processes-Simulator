/**
 * Logistic regression for binary classification.
 * Model: P(y=1|x) = 1 / (1 + exp(-(β₀ + β'x)))
 * Fitted via batch gradient descent.
 */

export type LogisticModel = {
  /** Intercept (β₀). */
  intercept: number
  /** Coefficients for each feature (β). */
  coef: number[]
}

export type FitOptions = {
  /** Learning rate for gradient descent. */
  learningRate?: number
  /** Maximum number of iterations. */
  maxIter?: number
  /** Stop when average absolute gradient change is below this. */
  tol?: number
  /** Optional seed for reproducible initialization. */
  seed?: number
}

const defaultOptions: Required<FitOptions> = {
  learningRate: 0.1,
  maxIter: 1000,
  tol: 1e-6,
  seed: 0,
}

/**
 * Sigmoid: σ(z) = 1 / (1 + exp(-z)).
 * Clamped to avoid overflow.
 */
function sigmoid(z: number): number {
  const cap = 20
  const v = Math.max(-cap, Math.min(cap, z))
  return 1 / (1 + Math.exp(-v))
}

/**
 * Fit logistic regression by batch gradient descent.
 * @param X - Feature matrix; each row is one sample, each column one feature.
 * @param y - Binary labels (0 or 1), one per sample.
 * @param options - Learning rate, max iterations, tolerance.
 * @returns Fitted model { intercept, coef }.
 */
export function fit(
  X: number[][],
  y: number[],
  options: FitOptions = {}
): LogisticModel {
  const opts = { ...defaultOptions, ...options }
  const n = X.length
  if (n === 0) return { intercept: 0, coef: [] }
  if (n !== y.length) throw new Error('X and y must have the same length')

  const d = X[0].length
  for (let i = 0; i < n; i++) {
    if (X[i].length !== d) throw new Error('All rows of X must have the same length')
  }

  // Initialize: simple deterministic init (or use seed for RNG if we add random init)
  let intercept = 0
  const coef = new Array<number>(d).fill(0)

  const lr = opts.learningRate
  const maxIter = opts.maxIter
  const tol = opts.tol

  for (let iter = 0; iter < maxIter; iter++) {
    let sumInterceptGrad = 0
    const sumCoefGrad = new Array<number>(d).fill(0)

    for (let i = 0; i < n; i++) {
      const xi = X[i]
      let z = intercept
      for (let j = 0; j < d; j++) z += coef[j] * xi[j]
      const p = sigmoid(z)
      const err = p - y[i]
      sumInterceptGrad += err
      for (let j = 0; j < d; j++) sumCoefGrad[j] += err * xi[j]
    }

    const scale = 1 / n
    const newIntercept = intercept - lr * sumInterceptGrad * scale
    const newCoef = coef.map((c, j) => c - lr * sumCoefGrad[j] * scale)

    const deltaIntercept = Math.abs(newIntercept - intercept)
    let maxDeltaCoef = 0
    for (let j = 0; j < d; j++) {
      const dj = Math.abs(newCoef[j] - coef[j])
      if (dj > maxDeltaCoef) maxDeltaCoef = dj
    }
    intercept = newIntercept
    for (let j = 0; j < d; j++) coef[j] = newCoef[j]

    if (deltaIntercept < tol && maxDeltaCoef < tol) break
  }

  return { intercept, coef }
}

/**
 * Predict P(y=1|x) for each row in X.
 */
export function predictProbability(X: number[][], model: LogisticModel): number[] {
  const { intercept, coef } = model
  const d = coef.length
  return X.map((xi) => {
    let z = intercept
    for (let j = 0; j < d; j++) z += coef[j] * xi[j]
    return sigmoid(z)
  })
}

/**
 * Predict class labels (0 or 1) using threshold 0.5.
 */
export function predict(X: number[][], model: LogisticModel): number[] {
  return predictProbability(X, model).map((p) => (p >= 0.5 ? 1 : 0))
}

/**
 * Binary cross-entropy loss (negative log likelihood) for monitoring.
 * L = - (1/n) Σ [ y_i log(p_i) + (1-y_i) log(1-p_i) ].
 */
export function loss(X: number[][], y: number[], model: LogisticModel): number {
  const probs = predictProbability(X, model)
  const eps = 1e-15
  let sum = 0
  for (let i = 0; i < y.length; i++) {
    const p = Math.max(eps, Math.min(1 - eps, probs[i]))
    sum += y[i] * Math.log(p) + (1 - y[i]) * Math.log(1 - p)
  }
  return -sum / y.length
}
