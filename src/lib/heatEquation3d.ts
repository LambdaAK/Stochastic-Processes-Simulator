/**
 * 3D heat equation: ∂u/∂t = α ∇²u
 * On the unit cube [0,1]³ with u = 0 on the boundary.
 * Finite differences: 7-point Laplacian, forward Euler.
 * ∇²u ≈ (sum of 6 neighbors - 6u) / h²
 */

export type InitialCondition3d = 'point' | 'corner' | 'half' | 'two-spots'

export interface HeatConfig3d {
  /** N per dimension; interior (1..N) plus boundary; grid (N+2)³ */
  N: number
  alpha: number
  dt: number
  T: number
  initial: InitialCondition3d
}

function idx(N: number, i: number, j: number, k: number): number {
  const S = N + 2
  return i * S * S + j * S + k
}

function initialGrid3d(N: number, preset: InitialCondition3d): Float64Array {
  const S = N + 2
  const size = S * S * S
  const u = new Float64Array(size)
  const h = 1 / (N + 1)

  for (let i = 1; i <= N; i++) {
    for (let j = 1; j <= N; j++) {
      for (let k = 1; k <= N; k++) {
        const x = i * h
        const y = j * h
        const z = k * h
        switch (preset) {
          case 'point':
            if (
              (x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2 < 0.04
            ) {
              u[idx(N, i, j, k)] = 1
            }
            break
          case 'corner':
            if (x < 0.35 && y < 0.35 && z < 0.35) u[idx(N, i, j, k)] = 1
            break
          case 'half':
            u[idx(N, i, j, k)] = x < 0.5 ? 1 : 0
            break
          case 'two-spots':
            if (
              (x - 0.25) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2 < 0.03 ||
              (x - 0.75) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2 < 0.03
            ) {
              u[idx(N, i, j, k)] = 1
            }
            break
        }
      }
    }
  }
  return u
}

function step3d(
  u: Float64Array,
  N: number,
  alpha: number,
  dt: number,
  h: number
): void {
  const lap = (i: number, j: number, k: number) =>
    (u[idx(N, i + 1, j, k)] +
      u[idx(N, i - 1, j, k)] +
      u[idx(N, i, j + 1, k)] +
      u[idx(N, i, j - 1, k)] +
      u[idx(N, i, j, k + 1)] +
      u[idx(N, i, j, k - 1)] -
      6 * u[idx(N, i, j, k)]) /
    (h * h)

  const next = new Float64Array(u.length)
  next.set(u)
  for (let i = 1; i <= N; i++) {
    for (let j = 1; j <= N; j++) {
      for (let k = 1; k <= N; k++) {
        next[idx(N, i, j, k)] = u[idx(N, i, j, k)] + dt * alpha * lap(i, j, k)
      }
    }
  }
  u.set(next)
}

export interface HeatResult3d {
  gridSize: number
  h: number
  steps: number
  dt: number
  snapshots: Float64Array[]
  times: number[]
}

export function solveHeatEquation3d(config: HeatConfig3d): HeatResult3d {
  const { N, alpha, dt, T, initial } = config
  const h = 1 / (N + 1)
  const maxStableDt = (h * h) / (6 * alpha)
  const safeDt = Math.min(dt, 0.95 * maxStableDt)
  const numSteps = Math.max(1, Math.floor(T / safeDt))
  const actualDt = T / numSteps

  const u = initialGrid3d(N, initial)
  const snapshots: Float64Array[] = []
  const times: number[] = []

  const maxSnapshots = 101
  const stride = Math.max(1, Math.floor(numSteps / (maxSnapshots - 1)))

  snapshots.push(new Float64Array(u))
  times.push(0)

  for (let n = 1; n <= numSteps; n++) {
    step3d(u, N, alpha, actualDt, h)
    if (n % stride === 0 || n === numSteps) {
      snapshots.push(new Float64Array(u))
      times.push(n * actualDt)
    }
  }

  return {
    gridSize: N + 2,
    h,
    steps: numSteps,
    dt: actualDt,
    snapshots,
    times,
  }
}

export function getCell3d(
  result: HeatResult3d,
  snapshotIndex: number,
  i: number,
  j: number,
  k: number
): number {
  const S = result.gridSize
  if (
    snapshotIndex < 0 ||
    snapshotIndex >= result.snapshots.length ||
    i < 0 ||
    i >= S ||
    j < 0 ||
    j >= S ||
    k < 0 ||
    k >= S
  )
    return 0
  return result.snapshots[snapshotIndex][i * S * S + j * S + k]
}
