/**
 * Ordinary least squares (OLS) linear regression: y = β₀ + β₁ x + ε.
 * Fit by solving the normal equations β = (X'X)⁻¹ X'y with X = [1, x].
 */

export type LinearRegressionResult = {
  /** Intercept β₀ */
  intercept: number
  /** Slope β₁ */
  slope: number
  /** Fitted values ŷ = β₀ + β₁ x */
  fitted: number[]
  /** Residuals e = y - ŷ */
  residuals: number[]
  /** Residual standard error (RSE): sqrt(RSS / (n-2)) */
  sigma: number
  /** R² = 1 - SS_res / SS_tot */
  rSquared: number
  /** Number of points */
  n: number
}

/**
 * Fit y = β₀ + β₁ x via OLS.
 * Returns null if n < 2 or variance of x is zero.
 */
export function fitLinearRegression(x: number[], y: number[]): LinearRegressionResult | null {
  const n = x.length
  if (n !== y.length || n < 2) return null

  const sumX = x.reduce((a, b) => a + b, 0)
  const sumY = y.reduce((a, b) => a + b, 0)
  const sumXX = x.reduce((a, v) => a + v * v, 0)
  const sumXY = x.reduce((a, v, i) => a + v * y[i], 0)

  // X'X = [[n, sumX], [sumX, sumXX]], X'y = [sumY, sumXY]
  const det = n * sumXX - sumX * sumX
  if (Math.abs(det) < 1e-14) return null // singular (constant x)

  const intercept = (sumXX * sumY - sumX * sumXY) / det
  const slope = (n * sumXY - sumX * sumY) / det

  const fitted: number[] = []
  const residuals: number[] = []
  let ssRes = 0
  const meanY = sumY / n
  let ssTot = 0

  for (let i = 0; i < n; i++) {
    const hat = intercept + slope * x[i]
    fitted.push(hat)
    const e = y[i] - hat
    residuals.push(e)
    ssRes += e * e
    const d = y[i] - meanY
    ssTot += d * d
  }

  const sigma = n > 2 ? Math.sqrt(ssRes / (n - 2)) : 0
  const rSquared = ssTot > 0 ? 1 - ssRes / ssTot : 0

  return {
    intercept,
    slope,
    fitted,
    residuals,
    sigma,
    rSquared,
    n,
  }
}

/**
 * Predict ŷ for a single x or an array of x values.
 */
export function predict(
  fit: { intercept: number; slope: number },
  x: number | number[]
): number | number[] {
  if (typeof x === 'number') {
    return fit.intercept + fit.slope * x
  }
  return x.map((v) => fit.intercept + fit.slope * v)
}

/**
 * Generate synthetic data: y = trueIntercept + trueSlope * x + N(0, noiseStd).
 */
export function generateLinearData(
  n: number,
  trueIntercept: number,
  trueSlope: number,
  xMin: number,
  xMax: number,
  noiseStd: number,
  rand: () => number = Math.random
): { x: number[]; y: number[] } {
  const x: number[] = []
  const y: number[] = []
  for (let i = 0; i < n; i++) {
    const xi = xMin + (xMax - xMin) * rand()
    const eps = normal(rand) * noiseStd
    x.push(xi)
    y.push(trueIntercept + trueSlope * xi + eps)
  }
  return { x, y }
}

function normal(rand: () => number): number {
  const u1 = rand()
  const u2 = rand()
  const sqrt = Math.sqrt(-2 * Math.log(u1))
  return sqrt * Math.cos(2 * Math.PI * u2)
}
