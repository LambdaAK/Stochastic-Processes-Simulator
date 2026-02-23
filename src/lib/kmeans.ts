/**
 * K-Means clustering: partition n points into k clusters by minimizing
 * within-cluster sum of squared distances (inertia).
 */

export type KMeansConfig = {
  /** Rows = points, columns = dimensions. */
  points: number[][]
  /** Number of clusters. */
  k: number
  /** Maximum iterations. */
  maxIterations?: number
  /** Initialization: 'random' or 'kmeans++' (default). */
  init?: 'random' | 'kmeans++'
  /** Optional RNG for reproducibility. */
  rand?: () => number
}

export type KMeansResult = {
  /** Cluster index (0..k-1) for each point. */
  labels: number[]
  /** Centroids, one per cluster. */
  centroids: number[][]
  /** Sum of squared distances from each point to its assigned centroid. */
  inertia: number
  /** Number of iterations performed. */
  nIterations: number
}

const DEFAULT_MAX_ITERATIONS = 300

function sqEuclidean(a: number[], b: number[]): number {
  let sum = 0
  for (let d = 0; d < a.length; d++) {
    const d_ = a[d] - b[d]
    sum += d_ * d_
  }
  return sum
}

function copyPoint(p: number[]): number[] {
  return p.slice()
}

/**
 * K-means++ initialization: first centroid random; then each new centroid
 * chosen with probability proportional to D(x)^2 (squared distance to nearest existing centroid).
 */
function initKMeansPlusPlus(
  points: number[][],
  k: number,
  rand: () => number
): number[][] {
  const n = points.length
  const centroids: number[][] = []
  const indices = new Set<number>()

  let idx = Math.floor(rand() * n)
  indices.add(idx)
  centroids.push(copyPoint(points[idx]))

  const dSq = new Float64Array(n)

  for (let c = 1; c < k; c++) {
    let total = 0
    for (let i = 0; i < n; i++) {
      let minDsq = Infinity
      for (let j = 0; j < centroids.length; j++) {
        const dsq = sqEuclidean(points[i], centroids[j])
        if (dsq < minDsq) minDsq = dsq
      }
      dSq[i] = minDsq
      total += minDsq
    }
    if (total <= 0) {
      let fallback = 0
      while (indices.has(fallback) && fallback < n) fallback++
      if (fallback < n) {
        indices.add(fallback)
        centroids.push(copyPoint(points[fallback]))
      }
      continue
    }
    const target = rand() * total
    let cum = 0
    let chosen = 0
    for (let i = 0; i < n; i++) {
      cum += dSq[i]
      if (cum >= target) {
        chosen = i
        break
      }
    }
    indices.add(chosen)
    centroids.push(copyPoint(points[chosen]))
  }

  return centroids
}

/**
 * Random initialization: k distinct points chosen uniformly at random.
 */
function initRandom(
  points: number[][],
  k: number,
  rand: () => number
): number[][] {
  const n = points.length
  const indices: number[] = []
  const used = new Set<number>()
  while (indices.length < Math.min(k, n)) {
    const i = Math.floor(rand() * n)
    if (!used.has(i)) {
      used.add(i)
      indices.push(i)
    }
  }
  return indices.map((i) => copyPoint(points[i]))
}

/**
 * Assign each point to the nearest centroid; return labels and new inertia.
 */
function assign(
  points: number[][],
  centroids: number[][],
  labels: number[]
): number {
  let inertia = 0
  for (let i = 0; i < points.length; i++) {
    const p = points[i]
    let bestJ = 0
    let bestDsq = sqEuclidean(p, centroids[0])
    for (let j = 1; j < centroids.length; j++) {
      const dsq = sqEuclidean(p, centroids[j])
      if (dsq < bestDsq) {
        bestDsq = dsq
        bestJ = j
      }
    }
    labels[i] = bestJ
    inertia += bestDsq
  }
  return inertia
}

/**
 * Update centroids to the mean of assigned points per cluster.
 */
function updateCentroids(
  points: number[][],
  labels: number[],
  k: number,
  centroids: number[][]
): void {
  const dim = points[0].length
  const sums = centroids.map(() => new Array(dim).fill(0))
  const counts = new Array(k).fill(0)

  for (let i = 0; i < points.length; i++) {
    const j = labels[i]
    counts[j]++
    for (let d = 0; d < dim; d++) sums[j][d] += points[i][d]
  }

  for (let j = 0; j < k; j++) {
    const n = counts[j] || 1
    for (let d = 0; d < dim; d++) centroids[j][d] = sums[j][d] / n
  }
}

/**
 * Run K-Means clustering. Returns labels, centroids, inertia, and iteration count.
 */
export function kmeans(config: KMeansConfig): KMeansResult {
  const {
    points,
    k,
    maxIterations = DEFAULT_MAX_ITERATIONS,
    init = 'kmeans++',
    rand = Math.random,
  } = config

  const n = points.length
  const dim = points.length > 0 ? points[0].length : 0

  if (n === 0 || k <= 0 || dim === 0) {
    return {
      labels: [],
      centroids: [],
      inertia: 0,
      nIterations: 0,
    }
  }

  const actualK = Math.min(k, n)
  let centroids: number[][] =
    init === 'kmeans++'
      ? initKMeansPlusPlus(points, actualK, rand)
      : initRandom(points, actualK, rand)

  const labels = new Array(n).fill(0)
  let inertia = assign(points, centroids, labels)
  let prevInertia = Infinity
  let iter = 0

  while (iter < maxIterations && inertia < prevInertia) {
    prevInertia = inertia
    updateCentroids(points, labels, actualK, centroids)
    inertia = assign(points, centroids, labels)
    iter++
  }

  return {
    labels,
    centroids,
    inertia,
    nIterations: iter,
  }
}
