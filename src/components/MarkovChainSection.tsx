import { useState, useMemo } from 'react'
import katex from 'katex'
import 'katex/dist/katex.min.css'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts'
import {
  parseMarkovDSL,
  runSimulation,
  getStationaryDistribution,
  totalVariationDistance,
  computeDistributionOverTime,
  type SimulateResult,
} from '@/lib/markovChain'
import { createSeededRng } from '@/lib/random'
import type { MarkovChainDef } from '@/types/markov'
import { MarkovChainGraph } from '@/components/MarkovChainGraph'
import styles from './MarkovChainSection.module.css'

function renderLatex(latex: string, displayMode = false): string {
  try {
    return katex.renderToString(latex, { displayMode, throwOnError: false })
  } catch {
    return latex
  }
}

const DEFAULT_DSL = `States: A, B, C
Initial distribution: A : 0.5, B : 0.3, C : 0.2
Transitions: A -> B : 0.5, A -> C : 0.5, B -> A : 1, C -> C : 1`

const COLORS = ['var(--accent)', '#0ea5e9', '#22c55e', '#a855f7', '#f59e0b']

export function MarkovChainSection() {
  const [dsl, setDsl] = useState(DEFAULT_DSL)
  const [chain, setChain] = useState<MarkovChainDef | null>(null)
  const [error, setError] = useState<string | null>(null)

  const [numTrajectories, setNumTrajectories] = useState(500)
  const [trajectoryLength, setTrajectoryLength] = useState(50)
  const [theoreticalSteps, setTheoreticalSteps] = useState(50)
  const [seed, setSeed] = useState<string>('')
  const [simResult, setSimResult] = useState<SimulateResult | null>(null)
  const [simRunning, setSimRunning] = useState(false)

  const handleLoad = () => {
    setError(null)
    setSimResult(null)
    const result = parseMarkovDSL(dsl)
    if (result.ok) {
      setChain(result.chain)
    } else {
      setError(result.error)
      setChain(null)
    }
  }

  const handleRunSimulation = () => {
    if (!chain) return
    setSimRunning(true)
    setSimResult(null)
    const rand =
      seed.trim() !== '' && !Number.isNaN(Number(seed))
        ? createSeededRng(Number(seed))
        : Math.random
    setTimeout(() => {
      const result = runSimulation(
        chain,
        {
          M: numTrajectories,
          N: trajectoryLength,
          seed: seed.trim() ? Number(seed) : undefined,
        },
        rand
      )
      setSimResult(result)
      setSimRunning(false)
    }, 0)
  }

  const transitionMatrix = useMemo(() => {
    if (!chain) return null
    const P: Record<string, Record<string, number>> = {}
    for (const from of chain.states) {
      P[from] = {}
      for (const to of chain.states) P[from][to] = 0
    }
    for (const t of chain.transitions) {
      P[t.from][t.to] = (P[t.from][t.to] ?? 0) + t.p
    }
    return P
  }, [chain])

  const stationaryDist = useMemo(
    () => (chain ? getStationaryDistribution(chain) : null),
    [chain]
  )

  const chartData = useMemo(() => {
    if (!simResult || !chain) return []
    return simResult.t.map((t, i) => {
      const row: Record<string, number | string> = { t }
      for (const s of chain.states) {
        row[s] = simResult.proportions[s][i]
      }
      return row
    })
  }, [simResult, chain])

  const finalTvDistance = useMemo(() => {
    if (!simResult || !chain || !stationaryDist) return null
    const last = simResult.t.length - 1
    const empirical: Record<string, number> = {}
    for (const s of chain.states) empirical[s] = simResult.proportions[s][last]
    return totalVariationDistance(empirical, stationaryDist, chain.states)
  }, [simResult, chain, stationaryDist])

  const theoreticalResult = useMemo(
    () => (chain && theoreticalSteps > 0 ? computeDistributionOverTime(chain, theoreticalSteps) : null),
    [chain, theoreticalSteps]
  )

  const theoreticalChartData = useMemo(() => {
    if (!theoreticalResult || !chain) return []
    return theoreticalResult.t.map((t, i) => {
      const row: Record<string, number | string> = { t }
      for (const s of chain.states) {
        row[s] = theoreticalResult.distributions[s][i]
      }
      return row
    })
  }, [theoreticalResult, chain])

  return (
    <div className={styles.section}>
      <div className={styles.intro}>
        <p className={styles.introText}>
          A <strong>Markov chain</strong> is a discrete-time stochastic process over a finite set of states. At each step the process moves from its current state to another (or stays put) according to fixed transition probabilities. The key property is the <strong>Markov property</strong>: the probability of the next state depends only on the current state, not on how the process got there:
        </p>
        <p className={styles.introFormula} dangerouslySetInnerHTML={{ __html: renderLatex('\\mathbb{P}(X_{t+1} = j \\mid X_0, \\ldots, X_t) = \\mathbb{P}(X_{t+1} = j \\mid X_t) = P(X_t, j)', true) }} />
        <p className={styles.introText}>
          The transition matrix <span dangerouslySetInnerHTML={{ __html: renderLatex('P') }} /> has entries <span dangerouslySetInnerHTML={{ __html: renderLatex('P(i,j) = \\mathbb{P}(X_{t+1}\\!=\\!j \\mid X_t\\!=\\!i)') }} />, with rows summing to 1: <span dangerouslySetInnerHTML={{ __html: renderLatex('\\sum_j P(i,j) = 1') }} /> for each <span dangerouslySetInnerHTML={{ __html: renderLatex('i') }} />. You specify the states, an initial distribution <span dangerouslySetInnerHTML={{ __html: renderLatex('\\mu') }} />, and the transition probabilities; below you can simulate paths, view the graph and matrix, and compute the distribution over time.
        </p>
      </div>
      <div className={styles.editorBlock}>
        <label className={styles.label} htmlFor="markov-dsl">
          Markov chain definition
        </label>
        <p className={styles.hint}>
          Three sections (each with a header): <code>States: A, B, C, ...</code> — <code>Initial distribution: A : 0.5, B : 0.3, ...</code> (or <code>uniform</code>) — <code>Transitions: A -&gt; B : 0.5, ...</code>
        </p>
        <textarea
          id="markov-dsl"
          className={styles.textarea}
          value={dsl}
          onChange={(e) => setDsl(e.target.value)}
          rows={8}
          spellCheck={false}
        />
        {error && <p className={styles.error}>{error}</p>}
        <button type="button" className={styles.loadBtn} onClick={handleLoad}>
          Load Markov chain
        </button>
      </div>

      {chain && (
        <>
          <div className={styles.graphBlock}>
            <h3 className={styles.graphTitle}>Transition graph</h3>
            <MarkovChainGraph chain={chain} />
          </div>

          {transitionMatrix && (
            <div className={styles.matrixBlock}>
              <h3 className={styles.matrixTitle}>Transition matrix P</h3>
              <p className={styles.matrixHint}>
                <span dangerouslySetInnerHTML={{ __html: renderLatex('P(i,j) = \\mathbb{P}(X_{t+1} = j \\mid X_t = i)') }} />
                {' — each row sums to '}
                <span dangerouslySetInnerHTML={{ __html: renderLatex('\\sum_{j} P(i,j) = 1') }} />
              </p>
              <div className={styles.matrixWrap}>
                <table className={styles.matrixTable}>
                  <thead>
                    <tr>
                      <th className={styles.matrixCorner}></th>
                      {chain.states.map((s) => (
                        <th key={s} className={styles.matrixHeader}>
                          {s}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {chain.states.map((from) => (
                      <tr key={from}>
                        <th className={styles.matrixRowHeader}>{from}</th>
                        {chain.states.map((to) => (
                          <td key={to} className={styles.matrixCell}>
                            {transitionMatrix[from][to] === 0
                              ? '0'
                              : transitionMatrix[from][to] === 1
                                ? '1'
                                : transitionMatrix[from][to].toFixed(3)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <div className={styles.optionsBlock}>
            <h3 className={styles.optionsTitle}>What to do</h3>

            <div className={styles.theoreticalBlock}>
              <h4 className={styles.simulateTitle}>Probability over time (theoretical)</h4>
              <p className={styles.theoreticalHint}>
                Let <span dangerouslySetInnerHTML={{ __html: renderLatex('\\mu') }} /> be the initial distribution (column vector). The distribution at time <span dangerouslySetInnerHTML={{ __html: renderLatex('t') }} /> is{' '}
                <span dangerouslySetInnerHTML={{ __html: renderLatex('\\mu_t = P^t \\mu') }} />
                , so <span dangerouslySetInnerHTML={{ __html: renderLatex('P(X_t = s) = (P^t \\mu)_s') }} />.
              </p>
              <div className={styles.theoreticalForm}>
                <label className={styles.fieldLabel}>
                  <span>Steps</span>
                  <input
                    type="number"
                    min={1}
                    max={500}
                    value={theoreticalSteps}
                    onChange={(e) => setTheoreticalSteps(Math.max(1, Number(e.target.value) || 1))}
                    className={styles.input}
                  />
                </label>
              </div>
              {theoreticalChartData.length > 0 && (
                <div className={styles.chartBlock}>
                  <ResponsiveContainer width="100%" height={280}>
                    <LineChart
                      data={theoreticalChartData}
                      margin={{ top: 8, right: 8, bottom: 8, left: 8 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis
                        dataKey="t"
                        type="number"
                        tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                        label={{
                          value: 'Time step (t)',
                          position: 'insideBottom',
                          offset: -4,
                          fill: 'var(--text-muted)',
                          fontSize: 12,
                        }}
                      />
                      <YAxis
                        type="number"
                        domain={[0, 1]}
                        tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                        tickFormatter={(v) => Number(v).toFixed(2)}
                        label={{
                          value: 'P(X_t = s)',
                          angle: -90,
                          position: 'insideLeft',
                          fill: 'var(--text-muted)',
                          fontSize: 12,
                        }}
                      />
                      <Tooltip
                        contentStyle={{
                          background: 'var(--bg-elevated)',
                          border: '1px solid var(--border)',
                          borderRadius: '8px',
                        }}
                        formatter={(value: number, name: string) => [Number(value).toFixed(4), name]}
                        labelFormatter={(t) => `t = ${t}`}
                      />
                      <Legend />
                      {chain.states.map((s, i) => (
                        <Line
                          key={s}
                          type="monotone"
                          dataKey={s}
                          name={s}
                          stroke={COLORS[i % COLORS.length]}
                          strokeWidth={2}
                          dot={false}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>

            <div className={styles.simulateBlock}>
              <h4 className={styles.simulateTitle}>Simulate</h4>
              <p className={styles.simulateHint}>
                Sample <span dangerouslySetInnerHTML={{ __html: renderLatex('M') }} /> trajectories of length <span dangerouslySetInnerHTML={{ __html: renderLatex('N') }} />. The proportion of paths in state <span dangerouslySetInnerHTML={{ __html: renderLatex('s') }} /> at time <span dangerouslySetInnerHTML={{ __html: renderLatex('t') }} /> estimates <span dangerouslySetInnerHTML={{ __html: renderLatex('P(X_t = s)') }} />; as <span dangerouslySetInnerHTML={{ __html: renderLatex('M \\to \\infty') }} /> it converges to the theoretical value.
              </p>
              <div className={styles.simulateForm}>
                <label className={styles.fieldLabel}>
                  <span>Number of trajectories</span>
                  <input
                    type="number"
                    min={1}
                    max={100000}
                    value={numTrajectories}
                    onChange={(e) => setNumTrajectories(Number(e.target.value) || 1)}
                    className={styles.input}
                  />
                </label>
                <label className={styles.fieldLabel}>
                  <span>Length of each trajectory (steps)</span>
                  <input
                    type="number"
                    min={1}
                    max={1000}
                    value={trajectoryLength}
                    onChange={(e) => setTrajectoryLength(Number(e.target.value) || 1)}
                    className={styles.input}
                  />
                </label>
                <label className={styles.fieldLabel}>
                  <span>Seed (optional)</span>
                  <input
                    type="text"
                    placeholder="Leave empty for random"
                    value={seed}
                    onChange={(e) => setSeed(e.target.value)}
                    className={styles.input}
                  />
                </label>
                <button
                  type="button"
                  className={styles.runBtn}
                  onClick={handleRunSimulation}
                  disabled={simRunning}
                >
                  {simRunning ? 'Running…' : 'Run simulation'}
                </button>
              </div>
            </div>

            {simResult && chartData.length > 0 && (
              <div className={styles.chartBlock}>
                <h4 className={styles.chartTitle}>Proportion in each state over time</h4>
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart data={chartData} margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis
                      dataKey="t"
                      type="number"
                      tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                      label={{
                        value: 'Time step (t)',
                        position: 'insideBottom',
                        offset: -4,
                        fill: 'var(--text-muted)',
                        fontSize: 12,
                      }}
                    />
                    <YAxis
                      type="number"
                      domain={[0, 1]}
                      tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                      tickFormatter={(v) => Number(v).toFixed(2)}
                      label={{
                        value: 'Proportion',
                        angle: -90,
                        position: 'insideLeft',
                        fill: 'var(--text-muted)',
                        fontSize: 12,
                      }}
                    />
                    <Tooltip
                      contentStyle={{
                        background: 'var(--bg-elevated)',
                        border: '1px solid var(--border)',
                        borderRadius: '8px',
                      }}
                      formatter={(value: number, name: string) => [Number(value).toFixed(4), name]}
                      labelFormatter={(t) => `t = ${t}`}
                    />
                    <Legend />
                    {chain.states.map((s, i) => (
                      <Line
                        key={s}
                        type="monotone"
                        dataKey={s}
                        name={s}
                        stroke={COLORS[i % COLORS.length]}
                        strokeWidth={2}
                        dot={false}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
                {stationaryDist != null && finalTvDistance != null && (
                  <div className={styles.convergenceHint}>
                    <p className={styles.convergenceFormula}>
                      <span dangerouslySetInnerHTML={{ __html: renderLatex('d_{\\mathrm{TV}}(\\hat{p}, \\pi) = \\frac{1}{2}\\sum_s |\\hat{p}(s) - \\pi(s)|') }} />
                    </p>
                    <p className={styles.convergenceValue}>
                      Total variation (final proportions vs stationary <span dangerouslySetInnerHTML={{ __html: renderLatex('\\pi') }} />): {finalTvDistance.toFixed(4)}
                      {finalTvDistance < 0.05 && ' (close to stationary)'}
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}
