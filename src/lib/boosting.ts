import {
  buildDecisionTree,
  predictTree,
  type TrainingPoint,
  type TreeNode,
} from '@/lib/decisionTree'
import { createSeededRng } from '@/lib/random'

export type BoostingConfig = {
  nEstimators: number
  maxDepth: number
  /** Optional seed for reproducible bootstrap sampling */
  seed?: number
}

export type BoostedTreeModel = {
  trees: TreeNode[]
  alphas: number[]
}

/**
 * Simple AdaBoost-style boosting of shallow decision trees.
 *
 * - Works with arbitrary string labels (binary or multi-class).
 * - Uses weighted bootstrap sampling to approximate training with sample weights.
 */
export function trainBoostedTrees(
  points: TrainingPoint[],
  config: BoostingConfig
): BoostedTreeModel {
  const n = points.length
  if (n === 0 || config.nEstimators <= 0) {
    return { trees: [], alphas: [] }
  }

  const trees: TreeNode[] = []
  const alphas: number[] = []

  // Sample weights w_i, normalized to sum to 1
  const weights = new Array<number>(n).fill(1 / n)

  const rand =
    typeof config.seed === 'number' && !Number.isNaN(config.seed)
      ? createSeededRng(config.seed)
      : Math.random

  for (let m = 0; m < config.nEstimators; m++) {
    // Weighted bootstrap sample
    const sampled: TrainingPoint[] = []
    for (let i = 0; i < n; i++) {
      const idx = sampleIndex(weights, rand)
      sampled.push(points[idx])
    }

    const tree = buildDecisionTree(sampled, { maxDepth: config.maxDepth, minSamplesSplit: 2 })
    if (!tree) break

    // Weighted error of this tree on the original dataset
    let error = 0
    for (let i = 0; i < n; i++) {
      const pred = predictTree(tree, points[i])
      if (pred !== points[i].label) {
        error += weights[i]
      }
    }

    // If error is too high or zero, stop early (no useful learner)
    if (error <= 0 || error >= 0.5) {
      // Keep a strictly better-than-random learner; otherwise break
      if (error <= 0 || trees.length === 0) {
        break
      }
      continue
    }

    const alpha = 0.5 * Math.log((1 - error) / Math.max(error, 1e-12))
    trees.push(tree)
    alphas.push(alpha)

    // Update weights: increase misclassified, decrease correctly classified
    let weightSum = 0
    for (let i = 0; i < n; i++) {
      const pred = predictTree(tree, points[i])
      const correct = pred === points[i].label
      const factor = correct ? Math.exp(-alpha) : Math.exp(alpha)
      const w = weights[i] * factor
      weights[i] = w
      weightSum += w
    }

    // Renormalize
    if (weightSum > 0) {
      for (let i = 0; i < n; i++) {
        weights[i] /= weightSum
      }
    }
  }

  return { trees, alphas }
}

/**
 * Predict a label via weighted vote of boosted trees.
 */
export function predictBoosted(model: BoostedTreeModel, point: { x: number; y: number }): string {
  const { trees, alphas } = model
  if (trees.length === 0) return ''

  const scores: Record<string, number> = {}
  for (let i = 0; i < trees.length; i++) {
    const tree = trees[i]
    const alpha = alphas[i] ?? 1
    const label = predictTree(tree, point)
    scores[label] = (scores[label] ?? 0) + alpha
  }

  let bestLabel = ''
  let bestScore = -Infinity
  for (const [label, score] of Object.entries(scores)) {
    if (score > bestScore) {
      bestScore = score
      bestLabel = label
    }
  }
  return bestLabel
}

/**
 * Decision boundary grid for a boosted ensemble.
 */
export function boostedDecisionGrid(
  model: BoostedTreeModel,
  xMin: number,
  xMax: number,
  yMin: number,
  yMax: number,
  nX: number,
  nY: number
): { x: number; y: number; label: string }[] {
  const { trees } = model
  if (!trees.length) return []

  const dx = (xMax - xMin) / Math.max(1, nX - 1)
  const dy = (yMax - yMin) / Math.max(1, nY - 1)
  const out: { x: number; y: number; label: string }[] = []
  for (let iy = 0; iy < nY; iy++) {
    for (let ix = 0; ix < nX; ix++) {
      const x = xMin + ix * dx
      const y = yMin + iy * dy
      out.push({ x, y, label: predictBoosted(model, { x, y }) })
    }
  }
  return out
}

function sampleIndex(weights: number[], rand: () => number): number {
  const r = rand()
  let acc = 0
  for (let i = 0; i < weights.length; i++) {
    acc += weights[i]
    if (r <= acc) return i
  }
  return weights.length - 1
}

