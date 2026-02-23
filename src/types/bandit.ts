export type RewardDistribution =
  | { type: 'bernoulli'; p: number }
  | { type: 'gaussian'; mean: number; std: number }
  | { type: 'uniform'; min: number; max: number }

export type BanditArm = {
  name: string
  distribution: RewardDistribution
}

export type BanditProblem = {
  arms: BanditArm[]
}

export type BanditParseResult =
  | { ok: true; problem: BanditProblem }
  | { ok: false; error: string }

export type AlgorithmType = 'random' | 'greedy' | 'epsilon-greedy' | 'ucb' | 'thompson-sampling'

export type AlgorithmConfig = {
  type: AlgorithmType
  epsilon?: number  // for epsilon-greedy
  c?: number        // for UCB (exploration constant)
}

export type PullResult = {
  armIndex: number
  reward: number
}

export type SimulationResult = {
  algorithm: string
  pulls: PullResult[]
  cumulativeReward: number[]
  regret: number[]
  armCounts: number[]
}
