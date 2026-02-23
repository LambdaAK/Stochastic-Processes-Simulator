import { normal } from '@/lib/random'
import type { LLNDistribution, LLNConfig, LLNResult } from '@/types/lln'

/** Sample one value from the distribution. */
export function sampleOne(
  dist: LLNDistribution,
  rand: () => number
): number {
  switch (dist.type) {
    case 'bernoulli':
      return rand() < dist.p ? 1 : 0
    case 'gaussian':
      return dist.mean + dist.std * normal(rand)
    case 'uniform':
      return dist.min + (dist.max - dist.min) * rand()
    case 'exponential': {
      const u = rand()
      return u <= 0 || u >= 1 ? sampleOne(dist, rand) : -Math.log(u) / dist.lambda
    }
    case 'poisson': {
      const L = Math.exp(-dist.lambda)
      let k = 0
      let p = 1
      do {
        k++
        p *= rand()
      } while (p > L)
      return k - 1
    }
    case 'beta': {
      const a = Math.floor(dist.alpha)
      const b = Math.floor(dist.beta)
      if (a < 1 || b < 1) return 0
      let sumA = 0
      let sumB = 0
      for (let i = 0; i < a; i++) {
        let u = rand()
        while (u <= 0 || u >= 1) u = rand()
        sumA += -Math.log(u)
      }
      for (let i = 0; i < b; i++) {
        let u = rand()
        while (u <= 0 || u >= 1) u = rand()
        sumB += -Math.log(u)
      }
      return sumA / (sumA + sumB)
    }
    default:
      return 0
  }
}

/** Theoretical mean E[X]. */
export function theoreticalMean(dist: LLNDistribution): number {
  switch (dist.type) {
    case 'bernoulli':
      return dist.p
    case 'gaussian':
      return dist.mean
    case 'uniform':
      return (dist.min + dist.max) / 2
    case 'exponential':
      return 1 / dist.lambda
    case 'poisson':
      return dist.lambda
    case 'beta': {
      const a = Math.max(1, Math.floor(dist.alpha))
      const b = Math.max(1, Math.floor(dist.beta))
      return a / (a + b)
    }
    default:
      return 0
  }
}

/** Run one sequence of n samples and return running mean at each step. */
export function runLLNSimulation(
  config: LLNConfig,
  rand: () => number = Math.random
): LLNResult {
  const { distribution, numSamples } = config
  const n: number[] = []
  const runningMean: number[] = []
  let sum = 0
  for (let i = 1; i <= numSamples; i++) {
    sum += sampleOne(distribution, rand)
    n.push(i)
    runningMean.push(sum / i)
  }
  return {
    n,
    runningMean,
    theoreticalMean: theoreticalMean(distribution),
  }
}
