import type {
  BanditProblem,
  BanditParseResult,
  RewardDistribution,
  AlgorithmConfig,
  PullResult,
  SimulationResult,
} from '@/types/bandit'

/**
 * Multi-Armed Bandit utilities.
 *
 * Parse a bandit problem DSL:
 *
 * Arms: Arm1, Arm2, Arm3
 * Arm1: Bernoulli(0.3)
 * Arm2: Gaussian(5, 2)
 * Arm3: Uniform(0, 10)
 *
 * Supported distributions:
 * - Bernoulli(p): returns 1 with probability p, 0 otherwise
 * - Gaussian(mean, std): normal distribution
 * - Uniform(min, max): uniform distribution
 */
export function parseBanditDSL(text: string): BanditParseResult {
  const trimmed = text.trim()
  if (!trimmed) {
    return { ok: false, error: 'Definition is empty.' }
  }

  const lines = trimmed.split(/\n/)
  const armNames: string[] = []
  const armDistributions = new Map<string, RewardDistribution>()

  let foundArmsLine = false

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim()
    if (!line) continue

    // Parse "Arms: A, B, C"
    if (/^Arms:\s*/i.test(line)) {
      foundArmsLine = true
      const rest = line.replace(/^Arms:\s*/i, '').trim()
      const names = rest.split(',').map((s) => s.trim()).filter(Boolean)
      if (names.length === 0) {
        return { ok: false, error: 'Arms list is empty.' }
      }
      const seen = new Set<string>()
      for (const name of names) {
        if (seen.has(name)) {
          return { ok: false, error: `Duplicate arm name: ${name}` }
        }
        seen.add(name)
        armNames.push(name)
      }
      continue
    }

    // Parse "ArmName: Distribution(params)"
    const distMatch = line.match(/^(\w+)\s*:\s*(\w+)\s*\(([^)]+)\)\s*$/i)
    if (distMatch) {
      const armName = distMatch[1].trim()
      const distType = distMatch[2].trim().toLowerCase()
      const paramsStr = distMatch[3].trim()

      if (!armNames.includes(armName)) {
        return { ok: false, error: `Unknown arm "${armName}" in distribution definition.` }
      }

      const params = paramsStr.split(',').map((p) => parseFloat(p.trim()))

      let distribution: RewardDistribution | null = null

      if (distType === 'bernoulli') {
        if (params.length !== 1) {
          return { ok: false, error: `Bernoulli requires 1 parameter (p), got ${params.length}.` }
        }
        const p = params[0]
        if (Number.isNaN(p) || p < 0 || p > 1) {
          return { ok: false, error: `Bernoulli parameter p must be in [0,1], got ${paramsStr}.` }
        }
        distribution = { type: 'bernoulli', p }
      } else if (distType === 'gaussian' || distType === 'normal') {
        if (params.length !== 2) {
          return { ok: false, error: `Gaussian requires 2 parameters (mean, std), got ${params.length}.` }
        }
        const [mean, std] = params
        if (Number.isNaN(mean) || Number.isNaN(std) || std <= 0) {
          return { ok: false, error: `Gaussian parameters invalid: mean=${mean}, std=${std} (std must be > 0).` }
        }
        distribution = { type: 'gaussian', mean, std }
      } else if (distType === 'uniform') {
        if (params.length !== 2) {
          return { ok: false, error: `Uniform requires 2 parameters (min, max), got ${params.length}.` }
        }
        const [min, max] = params
        if (Number.isNaN(min) || Number.isNaN(max) || min >= max) {
          return { ok: false, error: `Uniform parameters invalid: min=${min}, max=${max} (min must be < max).` }
        }
        distribution = { type: 'uniform', min, max }
      } else {
        return { ok: false, error: `Unknown distribution type: ${distType}` }
      }

      armDistributions.set(armName, distribution)
      continue
    }

    // If we get here, line didn't match any pattern
    if (foundArmsLine) {
      return { ok: false, error: `Invalid line: "${line}". Expected format: ArmName: Distribution(params)` }
    } else {
      return { ok: false, error: 'First line must be "Arms: A, B, C, ...".' }
    }
  }

  if (armNames.length === 0) {
    return { ok: false, error: 'No arms defined.' }
  }

  // Check that all arms have distributions
  for (const name of armNames) {
    if (!armDistributions.has(name)) {
      return { ok: false, error: `Arm "${name}" has no distribution defined.` }
    }
  }

  const arms = armNames.map((name) => ({
    name,
    distribution: armDistributions.get(name)!,
  }))

  return { ok: true, problem: { arms } }
}

/**
 * Sample a reward from a distribution.
 */
export function sampleReward(distribution: RewardDistribution, rand: () => number): number {
  switch (distribution.type) {
    case 'bernoulli':
      return rand() < distribution.p ? 1 : 0
    case 'gaussian': {
      // Box-Muller transform
      const u1 = rand()
      const u2 = rand()
      const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
      return distribution.mean + distribution.std * z
    }
    case 'uniform':
      return distribution.min + (distribution.max - distribution.min) * rand()
  }
}

/**
 * Compute the expected reward (mean) of a distribution.
 */
export function expectedReward(distribution: RewardDistribution): number {
  switch (distribution.type) {
    case 'bernoulli':
      return distribution.p
    case 'gaussian':
      return distribution.mean
    case 'uniform':
      return (distribution.min + distribution.max) / 2
  }
}

/**
 * Find the optimal arm (highest expected reward).
 */
export function getOptimalArm(problem: BanditProblem): number {
  let bestIdx = 0
  let bestReward = expectedReward(problem.arms[0].distribution)
  for (let i = 1; i < problem.arms.length; i++) {
    const reward = expectedReward(problem.arms[i].distribution)
    if (reward > bestReward) {
      bestReward = reward
      bestIdx = i
    }
  }
  return bestIdx
}

/**
 * Bandit algorithm interface.
 */
export interface BanditAlgorithm {
  selectArm(): number
  update(armIndex: number, reward: number): void
  getName(): string
}

/**
 * Random algorithm: select arms uniformly at random.
 */
export class RandomAlgorithm implements BanditAlgorithm {
  constructor(
    private numArms: number,
    private rand: () => number
  ) {}

  selectArm(): number {
    return Math.floor(this.rand() * this.numArms)
  }

  update(_armIndex: number, _reward: number): void {
    // No learning
  }

  getName(): string {
    return 'Random'
  }
}

/**
 * Greedy algorithm: always select the arm with highest estimated mean.
 */
export class GreedyAlgorithm implements BanditAlgorithm {
  private counts: number[]
  private values: number[]

  constructor(
    private numArms: number
  ) {
    this.counts = new Array(numArms).fill(0)
    this.values = new Array(numArms).fill(0)
  }

  selectArm(): number {
    // If any arm has not been tried, try it
    for (let i = 0; i < this.numArms; i++) {
      if (this.counts[i] === 0) return i
    }

    // Otherwise, pick the arm with highest estimated value
    let best = 0
    for (let i = 1; i < this.numArms; i++) {
      if (this.values[i] > this.values[best]) {
        best = i
      }
    }
    return best
  }

  update(armIndex: number, reward: number): void {
    this.counts[armIndex]++
    const n = this.counts[armIndex]
    const value = this.values[armIndex]
    this.values[armIndex] = ((n - 1) / n) * value + (1 / n) * reward
  }

  getName(): string {
    return 'Greedy'
  }
}

/**
 * Epsilon-greedy algorithm: explore with probability epsilon.
 */
export class EpsilonGreedyAlgorithm implements BanditAlgorithm {
  private counts: number[]
  private values: number[]

  constructor(
    private numArms: number,
    private epsilon: number,
    private rand: () => number
  ) {
    this.counts = new Array(numArms).fill(0)
    this.values = new Array(numArms).fill(0)
  }

  selectArm(): number {
    if (this.rand() < this.epsilon) {
      // Explore: random arm
      return Math.floor(this.rand() * this.numArms)
    } else {
      // Exploit: best arm
      // If any arm has not been tried, try it
      for (let i = 0; i < this.numArms; i++) {
        if (this.counts[i] === 0) return i
      }

      let best = 0
      for (let i = 1; i < this.numArms; i++) {
        if (this.values[i] > this.values[best]) {
          best = i
        }
      }
      return best
    }
  }

  update(armIndex: number, reward: number): void {
    this.counts[armIndex]++
    const n = this.counts[armIndex]
    const value = this.values[armIndex]
    this.values[armIndex] = ((n - 1) / n) * value + (1 / n) * reward
  }

  getName(): string {
    return `ε-greedy (ε=${this.epsilon})`
  }
}

/**
 * UCB (Upper Confidence Bound) algorithm.
 */
export class UCBAlgorithm implements BanditAlgorithm {
  private counts: number[]
  private values: number[]
  private totalCount: number = 0

  constructor(
    private numArms: number,
    private c: number
  ) {
    this.counts = new Array(numArms).fill(0)
    this.values = new Array(numArms).fill(0)
  }

  selectArm(): number {
    // Try each arm once first
    for (let i = 0; i < this.numArms; i++) {
      if (this.counts[i] === 0) return i
    }

    // Compute UCB values
    const ucbValues = this.values.map((value, i) => {
      const bonus = this.c * Math.sqrt((2 * Math.log(this.totalCount)) / this.counts[i])
      return value + bonus
    })

    // Select arm with highest UCB
    let best = 0
    for (let i = 1; i < this.numArms; i++) {
      if (ucbValues[i] > ucbValues[best]) {
        best = i
      }
    }
    return best
  }

  update(armIndex: number, reward: number): void {
    this.counts[armIndex]++
    this.totalCount++
    const n = this.counts[armIndex]
    const value = this.values[armIndex]
    this.values[armIndex] = ((n - 1) / n) * value + (1 / n) * reward
  }

  getName(): string {
    return `UCB (c=${this.c})`
  }
}

/**
 * Thompson Sampling algorithm (Bayesian).
 * For Bernoulli rewards, uses Beta priors.
 * For Gaussian/Uniform rewards, uses Gaussian priors with known variance.
 */
export class ThompsonSamplingAlgorithm implements BanditAlgorithm {
  private counts: number[]
  private sumRewards: number[]
  private distributions: RewardDistribution[]

  constructor(
    private numArms: number,
    distributions: RewardDistribution[],
    private rand: () => number
  ) {
    this.counts = new Array(numArms).fill(0)
    this.sumRewards = new Array(numArms).fill(0)
    this.distributions = distributions
  }

  selectArm(): number {
    // Sample from posterior for each arm
    const samples = this.distributions.map((dist, i) => {
      if (dist.type === 'bernoulli') {
        // Beta posterior: Beta(1 + successes, 1 + failures)
        const successes = this.sumRewards[i]
        const failures = this.counts[i] - successes
        return this.sampleBeta(1 + successes, 1 + failures)
      } else {
        // Gaussian posterior (assuming known variance)
        // For simplicity, use the empirical mean with some added noise
        if (this.counts[i] === 0) {
          // Prior: use the true mean with high variance
          const trueMean = expectedReward(dist)
          return trueMean + this.sampleGaussian(0, 10)
        }
        const empiricalMean = this.sumRewards[i] / this.counts[i]
        const uncertainty = 1 / Math.sqrt(this.counts[i] + 1)
        return empiricalMean + this.sampleGaussian(0, uncertainty)
      }
    })

    // Select arm with highest sample
    let best = 0
    for (let i = 1; i < this.numArms; i++) {
      if (samples[i] > samples[best]) {
        best = i
      }
    }
    return best
  }

  update(armIndex: number, reward: number): void {
    this.counts[armIndex]++
    this.sumRewards[armIndex] += reward
  }

  getName(): string {
    return 'Thompson Sampling'
  }

  private sampleBeta(alpha: number, beta: number): number {
    // Sample from Beta(alpha, beta) using Gamma samples
    const x = this.sampleGamma(alpha, 1)
    const y = this.sampleGamma(beta, 1)
    return x / (x + y)
  }

  private sampleGamma(shape: number, scale: number): number {
    // Simple rejection sampling for Gamma (works well for shape >= 1)
    if (shape < 1) {
      return this.sampleGamma(shape + 1, scale) * Math.pow(this.rand(), 1 / shape)
    }

    const d = shape - 1 / 3
    const c = 1 / Math.sqrt(9 * d)

    while (true) {
      let x = this.sampleGaussian(0, 1)
      const v = 1 + c * x
      if (v <= 0) continue

      const v3 = v * v * v
      const u = this.rand()
      if (u < 1 - 0.0331 * x * x * x * x) {
        return d * v3 * scale
      }
      if (Math.log(u) < 0.5 * x * x + d * (1 - v3 + Math.log(v3))) {
        return d * v3 * scale
      }
    }
  }

  private sampleGaussian(mean: number, std: number): number {
    // Box-Muller transform
    const u1 = this.rand()
    const u2 = this.rand()
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
    return mean + std * z
  }
}

/**
 * Create a bandit algorithm instance.
 */
export function createAlgorithm(
  config: AlgorithmConfig,
  numArms: number,
  distributions: RewardDistribution[],
  rand: () => number
): BanditAlgorithm {
  switch (config.type) {
    case 'random':
      return new RandomAlgorithm(numArms, rand)
    case 'greedy':
      return new GreedyAlgorithm(numArms)
    case 'epsilon-greedy':
      return new EpsilonGreedyAlgorithm(numArms, config.epsilon ?? 0.1, rand)
    case 'ucb':
      return new UCBAlgorithm(numArms, config.c ?? 2)
    case 'thompson-sampling':
      return new ThompsonSamplingAlgorithm(numArms, distributions, rand)
  }
}

/**
 * Run a bandit simulation.
 */
export function runBanditSimulation(
  problem: BanditProblem,
  config: AlgorithmConfig,
  numPulls: number,
  rand: () => number = Math.random
): SimulationResult {
  const algorithm = createAlgorithm(
    config,
    problem.arms.length,
    problem.arms.map((a) => a.distribution),
    rand
  )

  const optimalArmIdx = getOptimalArm(problem)
  const optimalReward = expectedReward(problem.arms[optimalArmIdx].distribution)

  const pulls: PullResult[] = []
  const cumulativeReward: number[] = []
  const regret: number[] = []
  const armCounts = new Array(problem.arms.length).fill(0)

  let totalReward = 0
  let totalRegret = 0

  for (let t = 0; t < numPulls; t++) {
    const armIdx = algorithm.selectArm()
    const reward = sampleReward(problem.arms[armIdx].distribution, rand)

    algorithm.update(armIdx, reward)
    armCounts[armIdx]++

    totalReward += reward
    totalRegret += optimalReward - expectedReward(problem.arms[armIdx].distribution)

    pulls.push({ armIndex: armIdx, reward })
    cumulativeReward.push(totalReward)
    regret.push(totalRegret)
  }

  return {
    algorithm: algorithm.getName(),
    pulls,
    cumulativeReward,
    regret,
    armCounts,
  }
}
