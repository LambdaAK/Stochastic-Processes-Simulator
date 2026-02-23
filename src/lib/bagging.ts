import type { Point2D, TrainingPoint, TreeNode } from '@/lib/decisionTree'
import { buildDecisionTree, predictTree } from '@/lib/decisionTree'

export type BaggedTreeModel = {
  trees: TreeNode[]
}

export function fitBaggedTrees(
  training: TrainingPoint[],
  nTrees: number,
  maxDepth: number
): BaggedTreeModel | null {
  const n = training.length
  const numTrees = Math.max(1, Math.floor(nTrees))
  if (n === 0 || numTrees <= 0) {
    return null
  }

  const trees: TreeNode[] = []

  for (let t = 0; t < numTrees; t++) {
    const sample: TrainingPoint[] = []
    for (let i = 0; i < n; i++) {
      const idx = Math.floor(Math.random() * n)
      sample.push(training[idx])
    }
    const tree = buildDecisionTree(sample, { maxDepth })
    if (tree) {
      trees.push(tree)
    }
  }

  if (trees.length === 0) {
    return null
  }

  return { trees }
}

export function predictBagged(model: BaggedTreeModel | null, point: Point2D): string {
  if (!model || model.trees.length === 0) return ''
  const counts: Record<string, number> = {}
  for (const tree of model.trees) {
    const label = predictTree(tree, point)
    if (!label) continue
    counts[label] = (counts[label] ?? 0) + 1
  }
  let bestLabel = ''
  let bestCount = 0
  for (const [label, count] of Object.entries(counts)) {
    if (count > bestCount) {
      bestCount = count
      bestLabel = label
    }
  }
  return bestLabel
}

export function baggingDecisionGrid(
  model: BaggedTreeModel | null,
  xMin: number,
  xMax: number,
  yMin: number,
  yMax: number,
  nX: number,
  nY: number
): { x: number; y: number; label: string }[] {
  if (!model || model.trees.length === 0) return []
  const dx = (xMax - xMin) / Math.max(1, nX - 1)
  const dy = (yMax - yMin) / Math.max(1, nY - 1)
  const out: { x: number; y: number; label: string }[] = []
  for (let iy = 0; iy < nY; iy++) {
    for (let ix = 0; ix < nX; ix++) {
      const x = xMin + ix * dx
      const y = yMin + iy * dy
      const label = predictBagged(model, { x, y })
      out.push({ x, y, label })
    }
  }
  return out
}

