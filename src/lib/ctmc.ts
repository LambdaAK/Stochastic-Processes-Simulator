import type { CtmcDef, CtmcParseResult, CtmcTransition } from '@/types/ctmc'

/**
 * Continuous-Time Markov Chain (CTMC) utilities.
 *
 * Parse a CTMC DSL:
 *
 * Section 1: States: A, B, C, ...
 * Section 2: Initial distribution: A : 0.5, B : 0.3, C : 0.2   (or "uniform")
 * Section 3: Rates: A -> B : 2.5, B -> C : 1.0, ...
 *
 * Rates are positive numbers representing transition rates (not probabilities).
 * The rate matrix Q has Q(i,j) = rate for i≠j, and Q(i,i) = -sum of outgoing rates.
 */
export function parseCtmcDSL(text: string): CtmcParseResult {
  const trimmed = text.trim()
  if (!trimmed) {
    return { ok: false, error: 'Definition is empty.' }
  }

  const lines = trimmed.split(/\n/)
  let states: string[] = []
  let initialDistribution: Record<string, number> = {}
  const transitions: CtmcTransition[] = []
  let section: 'states' | 'initial' | 'rates' | null = null
  let stateLines: string[] = []
  let initialLines: string[] = []
  let rateLines: string[] = []

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const trimmedLine = line.trim()
    if (!trimmedLine) continue

    if (/^States:\s*/i.test(trimmedLine)) {
      section = 'states'
      const part = trimmedLine.replace(/^States:\s*/i, '').trim()
      if (part) stateLines.push(part)
      continue
    }

    if (/^Initial distribution:\s*/i.test(trimmedLine)) {
      section = 'initial'
      const rest = trimmedLine.replace(/^Initial distribution:\s*/i, '').trim()
      if (rest) initialLines.push(rest)
      continue
    }

    if (/^Rates?:\s*/i.test(trimmedLine)) {
      section = 'rates'
      const rest = trimmedLine.replace(/^Rates?:\s*/i, '').trim()
      if (rest) rateLines.push(rest)
      continue
    }

    if (section === 'states') {
      stateLines.push(trimmedLine)
      continue
    }

    if (section === 'initial') {
      initialLines.push(trimmedLine)
      continue
    }

    if (section === 'rates') {
      rateLines.push(trimmedLine)
      continue
    }

    return { ok: false, error: 'First section must be "States: A, B, C, ...".' }
  }

  const stateText = stateLines.join(' ')
  const names = stateText.split(',').map((s) => s.trim()).filter(Boolean)
  if (names.length === 0) {
    return { ok: false, error: 'States list is empty.' }
  }
  const seenState = new Set<string>()
  for (const n of names) {
    if (seenState.has(n)) {
      return { ok: false, error: `Duplicate state name: ${n}` }
    }
    seenState.add(n)
    states.push(n)
  }

  if (states.length === 0) {
    return { ok: false, error: 'Missing "States: A, B, C, ..." section.' }
  }

  if (initialLines.length === 0) {
    return { ok: false, error: 'Missing "Initial distribution:" section.' }
  }

  const stateSet = new Set(states)
  const initialText = initialLines.join(' ').trim()
  if (initialText.toLowerCase() === 'uniform') {
    const p = 1 / states.length
    for (const s of states) {
      initialDistribution[s] = p
    }
  } else {
    const probRegex = /(\S+)\s*:\s*([\d.]+)/g
    let m: RegExpExecArray | null
    while ((m = probRegex.exec(initialText)) !== null) {
      const stateName = m[1].trim()
      const prob = parseFloat(m[2])
      if (!stateSet.has(stateName)) {
        return { ok: false, error: `Unknown state "${stateName}" in initial distribution.` }
      }
      if (Number.isNaN(prob) || prob < 0 || prob > 1) {
        return { ok: false, error: `Invalid probability "${m[2]}" in initial distribution.` }
      }
      initialDistribution[stateName] = (initialDistribution[stateName] ?? 0) + prob
    }
    for (const s of states) {
      if (!(s in initialDistribution)) initialDistribution[s] = 0
    }
    const sum = states.reduce((acc, s) => acc + initialDistribution[s], 0)
    if (Math.abs(sum - 1) > 1e-6) {
      return { ok: false, error: `Initial distribution must sum to 1 (got ${sum.toFixed(4)}).` }
    }
  }

  const rateRegex = /^\s*(\S+)\s*->\s*(\S+)\s*:\s*([\d.]+)\s*$/
  for (const line of rateLines) {
    const parts = line.split(',')
    for (const part of parts) {
      const m = part.trim().match(rateRegex)
      if (m) {
        const from = m[1].trim()
        const to = m[2].trim()
        const rate = parseFloat(m[3])
        if (!stateSet.has(from)) {
          return { ok: false, error: `Unknown state "${from}" in rate specification.` }
        }
        if (!stateSet.has(to)) {
          return { ok: false, error: `Unknown state "${to}" in rate specification.` }
        }
        if (from === to) {
          return { ok: false, error: `Self-loops not allowed in CTMC (from "${from}" to "${to}").` }
        }
        if (Number.isNaN(rate) || rate <= 0) {
          return { ok: false, error: `Invalid rate "${m[3]}" (must be > 0).` }
        }
        transitions.push({ from, to, rate })
      }
    }
  }

  if (section !== 'rates') {
    return { ok: false, error: 'Missing "Rates:" section.' }
  }

  return { ok: true, chain: { states, initialDistribution, transitions } }
}

/**
 * Build the rate matrix Q: Q(i,j) = rate for i≠j, Q(i,i) = -sum of outgoing rates.
 */
export function buildRateMatrix(chain: CtmcDef): Record<string, Record<string, number>> {
  const Q: Record<string, Record<string, number>> = {}
  for (const from of chain.states) {
    Q[from] = {}
    for (const to of chain.states) {
      Q[from][to] = 0
    }
  }
  for (const t of chain.transitions) {
    Q[t.from][t.to] = (Q[t.from][t.to] ?? 0) + t.rate
  }
  // Set diagonal entries: Q(i,i) = -sum of outgoing rates
  for (const s of chain.states) {
    let sum = 0
    for (const s2 of chain.states) {
      if (s !== s2) sum += Q[s][s2]
    }
    Q[s][s] = -sum
  }
  return Q
}

/**
 * Check if the chain is irreducible (every state can reach every other state).
 */
export function isIrreducible(chain: CtmcDef): boolean {
  const n = chain.states.length
  if (n <= 1) return true
  const Q = buildRateMatrix(chain)
  const successors = (s: string): string[] =>
    chain.states.filter((j) => s !== j && Q[s][j] > 0)
  for (const start of chain.states) {
    const visited = new Set<string>()
    const queue: string[] = [start]
    visited.add(start)
    while (queue.length > 0) {
      const u = queue.shift()!
      for (const v of successors(u)) {
        if (!visited.has(v)) {
          visited.add(v)
          queue.push(v)
        }
      }
    }
    if (visited.size !== n) return false
  }
  return true
}

/**
 * Compute stationary distribution π satisfying πQ = 0, Σπ = 1.
 * Returns null if the chain is not irreducible.
 */
export function getStationaryDistribution(chain: CtmcDef): Record<string, number> | null {
  if (!isIrreducible(chain)) return null
  const states = chain.states
  const n = states.length
  if (n === 0) return {}
  const Q = buildRateMatrix(chain)
  // πQ = 0  =>  (Q^T) π^T = 0. Replace last row by sum(π)=1.
  const M: number[][] = states.map((_, i) =>
    states.map((_, j) =>
      i === n - 1 ? 1 : Q[states[j]][states[i]]
    )
  )
  const b: number[] = states.map((_, i) => (i === n - 1 ? 1 : 0))
  const eps = 1e-10
  for (let col = 0; col < n; col++) {
    let pivot = -1
    for (let row = col; row < n; row++) {
      if (Math.abs(M[row][col]) > eps) {
        pivot = row
        break
      }
    }
    if (pivot === -1) continue
    ;[M[col], M[pivot]] = [M[pivot], M[col]]
    ;[b[col], b[pivot]] = [b[pivot], b[col]]
    const scale = M[col][col]
    for (let j = 0; j < n; j++) M[col][j] /= scale
    b[col] /= scale
    for (let row = 0; row < n; row++) {
      if (row === col || Math.abs(M[row][col]) < eps) continue
      const f = M[row][col]
      for (let j = 0; j < n; j++) M[row][j] -= f * M[col][j]
      b[row] -= f * b[col]
    }
  }
  const pi: Record<string, number> = {}
  for (let i = 0; i < n; i++) {
    pi[states[i]] = Math.max(0, b[i]) // Clamp to avoid negative due to numerical errors
  }
  return pi
}

/**
 * Matrix exponential exp(Qt) using scaling and squaring with Padé approximation.
 * Returns P(t) where P(t)[i][j] = Pr(X(t) = j | X(0) = i).
 */
export function matrixExponential(
  Q: Record<string, Record<string, number>>,
  states: string[],
  t: number
): Record<string, Record<string, number>> {
  const n = states.length
  if (n === 0) return {}

  // Convert to array form for easier manipulation
  const Qt: number[][] = states.map((i) => states.map((j) => Q[i][j] * t))

  // Use Padé approximation of order 6
  const I = states.map((_, i) => states.map((_, j) => (i === j ? 1 : 0)))

  // Scaling: find m such that ||Qt/2^m|| < 1
  let norm = 0
  for (let i = 0; i < n; i++) {
    let rowSum = 0
    for (let j = 0; j < n; j++) {
      rowSum += Math.abs(Qt[i][j])
    }
    norm = Math.max(norm, rowSum)
  }
  const m = Math.max(0, Math.ceil(Math.log2(norm)))

  // Scale: Qt = Qt / 2^m
  const scale = Math.pow(2, m)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      Qt[i][j] /= scale
    }
  }

  // Padé approximation: exp(Qt) ≈ (I + Qt/2 + Qt^2/12)/(I - Qt/2 + Qt^2/12)
  // Simplified order-2 Padé for stability
  const matMult = (A: number[][], B: number[][]): number[][] => {
    const C: number[][] = states.map(() => states.map(() => 0))
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        for (let k = 0; k < n; k++) {
          C[i][j] += A[i][k] * B[k][j]
        }
      }
    }
    return C
  }

  const matAdd = (A: number[][], B: number[][], coefA = 1, coefB = 1): number[][] => {
    return states.map((_, i) => states.map((_, j) => coefA * A[i][j] + coefB * B[i][j]))
  }

  const Qt2 = matMult(Qt, Qt)

  // Numerator: I + Qt/2 + Qt^2/12
  const num = matAdd(matAdd(I, Qt, 1, 0.5), Qt2, 1, 1/12)

  // Denominator: I - Qt/2 + Qt^2/12
  const den = matAdd(matAdd(I, Qt, 1, -0.5), Qt2, 1, 1/12)

  // Solve den * X = num using Gaussian elimination (X = den^-1 * num)
  const invDen = gaussJordanInverse(den)
  let result = matMult(invDen, num)

  // Square m times to get exp(Qt) from exp(Qt/2^m)
  for (let i = 0; i < m; i++) {
    result = matMult(result, result)
  }

  // Convert back to record form
  const P: Record<string, Record<string, number>> = {}
  for (let i = 0; i < n; i++) {
    P[states[i]] = {}
    for (let j = 0; j < n; j++) {
      P[states[i]][states[j]] = result[i][j]
    }
  }

  return P
}

function gaussJordanInverse(M: number[][]): number[][] {
  const n = M.length
  const A = M.map((row) => [...row])
  const I = M.map((_, i) => M.map((_, j) => (i === j ? 1 : 0)))

  for (let col = 0; col < n; col++) {
    // Find pivot
    let pivot = col
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(A[row][col]) > Math.abs(A[pivot][col])) {
        pivot = row
      }
    }

    if (Math.abs(A[pivot][col]) < 1e-10) continue

    // Swap rows
    ;[A[col], A[pivot]] = [A[pivot], A[col]]
    ;[I[col], I[pivot]] = [I[pivot], I[col]]

    // Scale pivot row
    const scale = A[col][col]
    for (let j = 0; j < n; j++) {
      A[col][j] /= scale
      I[col][j] /= scale
    }

    // Eliminate column
    for (let row = 0; row < n; row++) {
      if (row === col) continue
      const f = A[row][col]
      for (let j = 0; j < n; j++) {
        A[row][j] -= f * A[col][j]
        I[row][j] -= f * I[col][j]
      }
    }
  }

  return I
}

/**
 * Total variation distance: (1/2) Σ_s |p(s) - q(s)|.
 */
export function totalVariationDistance(
  p: Record<string, number>,
  q: Record<string, number>,
  states: string[]
): number {
  let sum = 0
  for (const s of states) {
    sum += Math.abs((p[s] ?? 0) - (q[s] ?? 0))
  }
  return 0.5 * sum
}

export type DistributionOverTimeResult = {
  t: number[]
  distributions: Record<string, number[]>
}

/**
 * Compute P(X(t) = s) for each state s at given time points.
 * Uses initial distribution μ and matrix exponential: distribution at time t is exp(Qt) μ.
 */
export function computeDistributionOverTime(
  chain: CtmcDef,
  timePoints: number[]
): DistributionOverTimeResult {
  const states = chain.states
  const distributions: Record<string, number[]> = {}
  for (const s of states) {
    distributions[s] = new Array(timePoints.length).fill(0)
  }

  const Q = buildRateMatrix(chain)

  for (let idx = 0; idx < timePoints.length; idx++) {
    const t = timePoints[idx]
    const P = matrixExponential(Q, states, t)

    // Compute distribution at time t: sum over initial states
    for (const to of states) {
      let prob = 0
      for (const from of states) {
        prob += (chain.initialDistribution[from] ?? 0) * P[from][to]
      }
      distributions[to][idx] = prob
    }
  }

  return { t: timePoints, distributions }
}

/**
 * From each state, build cumulative probabilities for jump destinations.
 */
export function buildJumpMap(chain: CtmcDef): Map<string, {
  totalRate: number
  to: string[]
  cumul: number[]
}> {
  const map = new Map<string, { totalRate: number; to: string[]; cumul: number[] }>()
  const byFrom = new Map<string, { to: string; rate: number }[]>()

  for (const t of chain.transitions) {
    if (!byFrom.has(t.from)) byFrom.set(t.from, [])
    byFrom.get(t.from)!.push({ to: t.to, rate: t.rate })
  }

  for (const state of chain.states) {
    const list = byFrom.get(state) ?? []
    const to: string[] = []
    const cumul: number[] = []
    let totalRate = 0

    for (const { to: next, rate } of list) {
      to.push(next)
      totalRate += rate
      cumul.push(totalRate)
    }

    map.set(state, { totalRate, to, cumul })
  }

  return map
}

/**
 * Sample next jump: from state `from`, return (nextState, holdingTime).
 */
export function sampleJump(
  from: string,
  jumpMap: Map<string, { totalRate: number; to: string[]; cumul: number[] }>,
  rand: () => number
): { nextState: string; holdingTime: number } {
  const row = jumpMap.get(from)
  if (!row || row.totalRate === 0) {
    // Absorbing state: stay forever
    return { nextState: from, holdingTime: Infinity }
  }

  // Sample holding time from Exponential(totalRate)
  const holdingTime = -Math.log(rand()) / row.totalRate

  // Sample next state proportional to rates
  const u = rand() * row.totalRate
  let nextState = from
  for (let i = 0; i < row.cumul.length; i++) {
    if (u < row.cumul[i]) {
      nextState = row.to[i]
      break
    }
  }

  return { nextState, holdingTime }
}

/**
 * Sample one state from a distribution.
 */
function sampleFromDistribution(
  dist: Record<string, number>,
  states: string[],
  rand: () => number
): string {
  const u = rand()
  let cumul = 0
  for (const s of states) {
    cumul += dist[s] ?? 0
    if (u < cumul) return s
  }
  return states[states.length - 1]
}

export type TrajectoryPoint = {
  t: number
  state: string
}

/**
 * Sample one trajectory up to maxTime.
 * Returns array of (time, state) pairs representing the jump times.
 */
export function sampleTrajectory(
  chain: CtmcDef,
  maxTime: number,
  rand: () => number
): TrajectoryPoint[] {
  const jumpMap = buildJumpMap(chain)
  const start = sampleFromDistribution(chain.initialDistribution, chain.states, rand)

  const trajectory: TrajectoryPoint[] = [{ t: 0, state: start }]
  let currentState = start
  let currentTime = 0

  while (currentTime < maxTime) {
    const { nextState, holdingTime } = sampleJump(currentState, jumpMap, rand)

    if (!isFinite(holdingTime) || currentTime + holdingTime >= maxTime) {
      break
    }

    currentTime += holdingTime
    trajectory.push({ t: currentTime, state: nextState })
    currentState = nextState
  }

  return trajectory
}

export type SimulateConfig = {
  M: number
  maxTime: number
  numTimePoints: number
  seed?: number
}

export type SimulateResult = {
  t: number[]
  proportions: Record<string, number[]>
}

/**
 * Run M trajectories up to maxTime; return proportions in each state at numTimePoints evenly spaced times.
 */
export function runSimulation(
  chain: CtmcDef,
  config: SimulateConfig,
  rand: () => number = Math.random
): SimulateResult {
  const { M, maxTime, numTimePoints } = config
  const timePoints = Array.from({ length: numTimePoints }, (_, i) => (i / (numTimePoints - 1)) * maxTime)

  const proportions: Record<string, number[]> = {}
  for (const s of chain.states) {
    proportions[s] = new Array(numTimePoints).fill(0)
  }

  for (let m = 0; m < M; m++) {
    const trajectory = sampleTrajectory(chain, maxTime, rand)

    // For each time point, find the state at that time
    for (let i = 0; i < numTimePoints; i++) {
      const t = timePoints[i]

      // Find the last jump before or at time t
      let state = trajectory[0].state
      for (let j = 1; j < trajectory.length; j++) {
        if (trajectory[j].t <= t) {
          state = trajectory[j].state
        } else {
          break
        }
      }

      proportions[state][i] += 1
    }
  }

  // Normalize to proportions
  for (const s of chain.states) {
    for (let i = 0; i < numTimePoints; i++) {
      proportions[s][i] /= M
    }
  }

  return { t: timePoints, proportions }
}
