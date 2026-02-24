import { normal } from '@/lib/random'

export type CIDistributionType = 'bernoulli' | 'gaussian' | 'exponential' | 'uniform'
export type InequalityType = 'markov' | 'chebyshev' | 'hoeffding' | 'subgaussian'

export type CIDistribution =
  | { type: 'bernoulli'; p: number }
  | { type: 'gaussian'; mean: number; std: number }
  | { type: 'exponential'; lambda: number }
  | { type: 'uniform'; a: number; b: number }

export interface SimPoint {
  param: number
  bound: number
  empirical: number
}

export interface CIResult {
  points: SimPoint[]
  mu: number
  sigma: number
  variance: number
  nSamples: number
  inequality: InequalityType
  hoeffdingN: number
}

/** Which distributions are valid for each inequality */
export const VALID_DISTRIBUTIONS: Record<InequalityType, CIDistributionType[]> = {
  markov: ['bernoulli', 'exponential', 'uniform'],
  chebyshev: ['bernoulli', 'gaussian', 'exponential', 'uniform'],
  hoeffding: ['bernoulli', 'uniform'],
  subgaussian: ['bernoulli', 'gaussian', 'uniform'],
}

export function getMean(dist: CIDistribution): number {
  switch (dist.type) {
    case 'bernoulli': return dist.p
    case 'gaussian': return dist.mean
    case 'exponential': return 1 / dist.lambda
    case 'uniform': return (dist.a + dist.b) / 2
  }
}

export function getVariance(dist: CIDistribution): number {
  switch (dist.type) {
    case 'bernoulli': return dist.p * (1 - dist.p)
    case 'gaussian': return dist.std * dist.std
    case 'exponential': return 1 / (dist.lambda * dist.lambda)
    case 'uniform': return (dist.b - dist.a) ** 2 / 12
  }
}

function sampleDist(dist: CIDistribution, rand: () => number): number {
  switch (dist.type) {
    case 'bernoulli': return rand() < dist.p ? 1 : 0
    case 'gaussian': return dist.mean + dist.std * normal(rand)
    case 'exponential': {
      let u = rand()
      while (u <= 0) u = rand()
      return -Math.log(u) / dist.lambda
    }
    case 'uniform': return dist.a + (dist.b - dist.a) * rand()
  }
}

/** Range (b − a) for bounded distributions used in Hoeffding bound */
function getBoundedRange(dist: CIDistribution): number {
  switch (dist.type) {
    case 'bernoulli': return 1      // X ∈ [0, 1]
    case 'uniform': return dist.b - dist.a
    default: return 1
  }
}

const N_STEPS = 60

export function runCISimulation(
  dist: CIDistribution,
  inequality: InequalityType,
  nSamples: number,
  hoeffdingN: number,
  rand: () => number
): CIResult {
  const mu = getMean(dist)
  const variance = getVariance(dist)
  const sigma = Math.sqrt(Math.max(variance, 1e-12))
  const points: SimPoint[] = []

  if (inequality === 'markov') {
    // P(X ≥ a) ≤ E[X] / a  (requires X ≥ 0, mu > 0)
    if (mu <= 0) {
      return { points: [], mu, sigma, variance, nSamples, inequality, hoeffdingN }
    }
    const samples = Array.from({ length: nSamples }, () => sampleDist(dist, rand))
    const aMin = mu * 0.3
    const aMax = mu * 6
    for (let i = 1; i <= N_STEPS; i++) {
      const a = aMin + (aMax - aMin) * (i / N_STEPS)
      const bound = Math.min(1, mu / a)
      const empirical = samples.filter(x => x >= a).length / nSamples
      points.push({ param: a, bound, empirical })
    }

  } else if (inequality === 'chebyshev') {
    // P(|X − μ| ≥ kσ) ≤ 1/k²
    const samples = Array.from({ length: nSamples }, () => sampleDist(dist, rand))
    for (let i = 1; i <= N_STEPS; i++) {
      const k = 0.2 + 4.8 * (i / N_STEPS)
      const bound = Math.min(1, 1 / (k * k))
      const empirical = samples.filter(x => Math.abs(x - mu) >= k * sigma).length / nSamples
      points.push({ param: k, bound, empirical })
    }

  } else if (inequality === 'hoeffding') {
    // P(X̄ₙ − μ ≥ t) ≤ exp(−2nt²/(b−a)²)
    // Each "trial" draws hoeffdingN samples and records their mean
    const range = getBoundedRange(dist)
    const trialMeans: number[] = []
    for (let i = 0; i < nSamples; i++) {
      let s = 0
      for (let j = 0; j < hoeffdingN; j++) s += sampleDist(dist, rand)
      trialMeans.push(s / hoeffdingN)
    }
    // t range: meaningful deviations of the sample mean
    const sigmaOfMean = sigma / Math.sqrt(hoeffdingN)
    const tMax = Math.min(range * 0.45, sigmaOfMean * 4)
    for (let i = 1; i <= N_STEPS; i++) {
      const t = tMax * (i / N_STEPS)
      const bound = Math.min(1, Math.exp(-2 * hoeffdingN * t * t / (range * range)))
      const empirical = trialMeans.filter(m => m - mu >= t).length / nSamples
      points.push({ param: t, bound, empirical })
    }

  } else {
    // subgaussian: P(X − μ ≥ t) ≤ exp(−t²/(2σ²))
    const samples = Array.from({ length: nSamples }, () => sampleDist(dist, rand))
    const tMax = sigma * 4
    for (let i = 1; i <= N_STEPS; i++) {
      const t = tMax * (i / N_STEPS)
      const bound = Math.min(1, Math.exp(-t * t / (2 * sigma * sigma)))
      const empirical = samples.filter(x => x - mu >= t).length / nSamples
      points.push({ param: t, bound, empirical })
    }
  }

  return { points, mu, sigma, variance, nSamples, inequality, hoeffdingN }
}
