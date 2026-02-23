/**
 * DBSCAN: Density-Based Spatial Clustering of Applications with Noise.
 * Points within eps of each other are neighbors; core points have >= minPts
 * neighbors. Clusters are maximal sets of density-reachable points; rest is noise.
 */

export type DBSCANConfig = {
  /** Rows = points, columns = dimensions. */
  points: number[][]
  /** Max distance for two points to be neighbors. */
  eps: number
  /** Min points in eps-ball (including self) to be a core point. */
  minPts: number
}

export type DBSCANResult = {
  /** Cluster index (0, 1, ...) for each point, or -1 for noise. */
  labels: number[]
  /** Number of clusters (excluding noise). */
  nClusters: number
}

const UNVISITED = -2

function sqEuclidean(a: number[], b: number[]): number {
  let sum = 0
  for (let d = 0; d < a.length; d++) {
    const d_ = a[d] - b[d]
    sum += d_ * d_
  }
  return sum
}

/**
 * For each point, compute indices of points within distance eps (including self).
 */
function neighborhoodIndices(points: number[][], eps: number): number[][] {
  const n = points.length
  const epsSq = eps * eps
  const out: number[][] = []
  for (let i = 0; i < n; i++) {
    const row: number[] = []
    for (let j = 0; j < n; j++) {
      if (sqEuclidean(points[i], points[j]) <= epsSq) row.push(j)
    }
    out.push(row)
  }
  return out
}

/**
 * Run DBSCAN. Returns labels (cluster index or -1 for noise) and number of clusters.
 */
export function dbscan(config: DBSCANConfig): DBSCANResult {
  const { points, eps, minPts } = config
  const n = points.length
  const dim = points.length > 0 ? points[0].length : 0

  if (n === 0 || dim === 0 || eps <= 0 || minPts < 1) {
    return { labels: [], nClusters: 0 }
  }

  const labels = new Array<number>(n).fill(UNVISITED)
  const neighbors = neighborhoodIndices(points, eps)
  let clusterId = 0

  for (let i = 0; i < n; i++) {
    if (labels[i] !== UNVISITED) continue
    const ni = neighbors[i]
    if (ni.length < minPts) {
      labels[i] = -1
      continue
    }
    labels[i] = clusterId
    const stack: number[] = [...ni]
    let head = 0
    while (head < stack.length) {
      const j = stack[head]
      head++
      if (labels[j] === -1) labels[j] = clusterId
      if (labels[j] !== UNVISITED) continue
      labels[j] = clusterId
      if (neighbors[j].length >= minPts) {
        for (const q of neighbors[j]) {
          if (labels[q] === UNVISITED) stack.push(q)
        }
      }
    }
    clusterId++
  }

  return { labels, nClusters: clusterId }
}
