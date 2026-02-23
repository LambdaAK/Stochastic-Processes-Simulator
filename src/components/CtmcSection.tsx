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
  parseCtmcDSL,
  runSimulation,
  getStationaryDistribution,
  totalVariationDistance,
  computeDistributionOverTime,
  buildRateMatrix,
  type SimulateResult,
} from '@/lib/ctmc'
import { createSeededRng } from '@/lib/random'
import type { CtmcDef } from '@/types/ctmc'
import { CtmcGraph } from '@/components/CtmcGraph'
import styles from './MarkovChainSection.module.css'

function renderLatex(latex: string, displayMode = false): string {
  try {
    return katex.renderToString(latex, { displayMode, throwOnError: false })
  } catch {
    return latex
  }
}

const DEFAULT_DSL = `States: A, B, C
Initial distribution: A : 1.0
Rates: A -> B : 2.0, B -> C : 1.5, C -> A : 0.5`

const COLORS = ['var(--accent)', '#0ea5e9', '#22c55e', '#a855f7', '#f59e0b']

export function CtmcSection() {
  const [dsl, setDsl] = useState(DEFAULT_DSL)
  const [chain, setChain] = useState<CtmcDef | null>(null)
  const [error, setError] = useState<string | null>(null)

  const [numTrajectories, setNumTrajectories] = useState(500)
  const [maxTime, setMaxTime] = useState(10)
  const [numTimePoints, setNumTimePoints] = useState(100)
  const [theoreticalMaxTime, setTheoreticalMaxTime] = useState(10)
  const [theoreticalNumPoints, setTheoreticalNumPoints] = useState(100)
  const [seed, setSeed] = useState<string>('')
  const [simResult, setSimResult] = useState<SimulateResult | null>(null)
  const [simRunning, setSimRunning] = useState(false)

  const handleLoad = () => {
    setError(null)
    setSimResult(null)
    const result = parseCtmcDSL(dsl)
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
          maxTime,
          numTimePoints,
          seed: seed.trim() ? Number(seed) : undefined,
        },
        rand
      )
      setSimResult(result)
      setSimRunning(false)
    }, 0)
  }

  const rateMatrix = useMemo(() => {
    if (!chain) return null
    return buildRateMatrix(chain)
  }, [chain])

  const stationaryDist = useMemo(
    () => (chain ? getStationaryDistribution(chain) : null),
    [chain]
  )

  const chartData = useMemo(() => {
    if (!simResult || !chain) return []
    return simResult.t.map((t, i) => {
      const row: Record<string, number | string> = { t: Number(t.toFixed(3)) }
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

  const theoreticalResult = useMemo(() => {
    if (!chain || theoreticalNumPoints <= 0) return null
    const timePoints = Array.from(
      { length: theoreticalNumPoints },
      (_, i) => (i / (theoreticalNumPoints - 1)) * theoreticalMaxTime
    )
    return computeDistributionOverTime(chain, timePoints)
  }, [chain, theoreticalMaxTime, theoreticalNumPoints])

  const theoreticalChartData = useMemo(() => {
    if (!theoreticalResult || !chain) return []
    return theoreticalResult.t.map((t, i) => {
      const row: Record<string, number | string> = { t: Number(t.toFixed(3)) }
      for (const s of chain.states) {
        row[s] = theoreticalResult.distributions[s][i]
      }
      return row
    })
  }, [theoreticalResult, chain])

  return (
    <div className={styles.section}>
      <div className={styles.editorBlock}>
        <label className={styles.label} htmlFor="ctmc-dsl">
          Continuous-time Markov chain definition
        </label>
        <p className={styles.hint}>
          Three sections: <code>States: A, B, C, ...</code> — <code>Initial distribution: A : 1.0, ...</code> (or <code>uniform</code>) — <code>Rates: A -&gt; B : 2.0, ...</code> (rates must be &gt; 0)
        </p>
        <textarea
          id="ctmc-dsl"
          className={styles.textarea}
          value={dsl}
          onChange={(e) => setDsl(e.target.value)}
          rows={8}
          spellCheck={false}
        />
        {error && <p className={styles.error}>{error}</p>}
        <button type="button" className={styles.loadBtn} onClick={handleLoad}>
          Load CTMC
        </button>
      </div>

      {chain && (
        <>
          <div className={styles.graphBlock}>
            <h3 className={styles.graphTitle}>Rate graph</h3>
            <CtmcGraph chain={chain} />
          </div>

          {rateMatrix && (
            <div className={styles.matrixBlock}>
              <h3 className={styles.matrixTitle}>Rate matrix Q</h3>
              <p className={styles.matrixHint}>
                Q(i, j) = rate of transition from i to j (i≠j); Q(i, i) = −sum of outgoing rates
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
                        {chain.states.map((to) => {
                          const val = rateMatrix[from][to]
                          return (
                            <td key={to} className={styles.matrixCell}>
                              {val === 0
                                ? '0'
                                : val === Math.round(val)
                                  ? val
                                  : val.toFixed(2)}
                            </td>
                          )
                        })}
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
                <span dangerouslySetInnerHTML={{ __html: renderLatex('P(X(t) = s)') }} />
                {' from initial distribution '}
                <span dangerouslySetInnerHTML={{ __html: renderLatex('\\mu') }} />
                {': distribution at time '}
                <span dangerouslySetInnerHTML={{ __html: renderLatex('t') }} />
                {' is '}
                <span dangerouslySetInnerHTML={{ __html: renderLatex('e^{Qt} \\mu') }} />
                .
              </p>
              <div className={styles.theoreticalForm}>
                <label className={styles.fieldLabel}>
                  <span>Max time</span>
                  <input
                    type="number"
                    min={0.1}
                    step={0.1}
                    value={theoreticalMaxTime}
                    onChange={(e) => setTheoreticalMaxTime(Math.max(0.1, Number(e.target.value) || 0.1))}
                    className={styles.input}
                  />
                </label>
                <label className={styles.fieldLabel}>
                  <span>Number of time points</span>
                  <input
                    type="number"
                    min={10}
                    max={1000}
                    value={theoreticalNumPoints}
                    onChange={(e) => setTheoreticalNumPoints(Math.max(10, Number(e.target.value) || 10))}
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
                          value: 'Time (t)',
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
                          value: 'P(X(t) = s)',
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
                Sample trajectories using exponential holding times and see the proportion in each state over time.
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
                  <span>Max time</span>
                  <input
                    type="number"
                    min={0.1}
                    step={0.1}
                    value={maxTime}
                    onChange={(e) => setMaxTime(Number(e.target.value) || 0.1)}
                    className={styles.input}
                  />
                </label>
                <label className={styles.fieldLabel}>
                  <span>Number of time points</span>
                  <input
                    type="number"
                    min={10}
                    max={1000}
                    value={numTimePoints}
                    onChange={(e) => setNumTimePoints(Number(e.target.value) || 10)}
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
                        value: 'Time (t)',
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
                  <p className={styles.convergenceHint}>
                    Total variation distance from final proportions to π: {finalTvDistance.toFixed(4)}
                    {finalTvDistance < 0.05 && ' (close to stationary)'}
                  </p>
                )}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}
