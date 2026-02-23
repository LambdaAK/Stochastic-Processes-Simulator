import { useState, useMemo } from 'react'
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts'
import {
  parseBanditDSL,
  runBanditSimulation,
  expectedReward,
  getOptimalArm,
} from '@/lib/bandit'
import { createSeededRng } from '@/lib/random'
import type { BanditProblem, AlgorithmConfig, SimulationResult } from '@/types/bandit'
import styles from './MarkovChainSection.module.css'

const DEFAULT_DSL = `Arms: Slot1, Slot2, Slot3
Slot1: Bernoulli(0.3)
Slot2: Bernoulli(0.5)
Slot3: Bernoulli(0.7)`

const COLORS = ['var(--accent)', '#0ea5e9', '#22c55e', '#a855f7', '#f59e0b', '#ec4899']

export function BanditSection() {
  const [dsl, setDsl] = useState(DEFAULT_DSL)
  const [problem, setProblem] = useState<BanditProblem | null>(null)
  const [error, setError] = useState<string | null>(null)

  const [numPulls, setNumPulls] = useState(1000)
  const [seed, setSeed] = useState<string>('')

  const [selectedAlgorithms, setSelectedAlgorithms] = useState<Set<string>>(
    new Set(['epsilon-greedy', 'ucb', 'thompson-sampling'])
  )
  const [epsilon, setEpsilon] = useState(0.1)
  const [ucbC, setUcbC] = useState(2)

  const [results, setResults] = useState<SimulationResult[]>([])
  const [simRunning, setSimRunning] = useState(false)

  const handleLoad = () => {
    setError(null)
    setResults([])
    const result = parseBanditDSL(dsl)
    if (result.ok) {
      setProblem(result.problem)
    } else {
      setError(result.error)
      setProblem(null)
    }
  }

  const handleRunSimulation = () => {
    if (!problem) return
    setSimRunning(true)
    setResults([])

    const rand =
      seed.trim() !== '' && !Number.isNaN(Number(seed))
        ? createSeededRng(Number(seed))
        : Math.random

    setTimeout(() => {
      const newResults: SimulationResult[] = []

      const algorithms: AlgorithmConfig[] = []
      if (selectedAlgorithms.has('random')) {
        algorithms.push({ type: 'random' })
      }
      if (selectedAlgorithms.has('greedy')) {
        algorithms.push({ type: 'greedy' })
      }
      if (selectedAlgorithms.has('epsilon-greedy')) {
        algorithms.push({ type: 'epsilon-greedy', epsilon })
      }
      if (selectedAlgorithms.has('ucb')) {
        algorithms.push({ type: 'ucb', c: ucbC })
      }
      if (selectedAlgorithms.has('thompson-sampling')) {
        algorithms.push({ type: 'thompson-sampling' })
      }

      for (const config of algorithms) {
        const result = runBanditSimulation(problem, config, numPulls, rand)
        newResults.push(result)
      }

      setResults(newResults)
      setSimRunning(false)
    }, 0)
  }

  const toggleAlgorithm = (alg: string) => {
    const newSet = new Set(selectedAlgorithms)
    if (newSet.has(alg)) {
      newSet.delete(alg)
    } else {
      newSet.add(alg)
    }
    setSelectedAlgorithms(newSet)
  }

  const optimalArmInfo = useMemo(() => {
    if (!problem) return null
    const optIdx = getOptimalArm(problem)
    const optReward = expectedReward(problem.arms[optIdx].distribution)
    return { index: optIdx, name: problem.arms[optIdx].name, reward: optReward }
  }, [problem])

  const regretChartData = useMemo(() => {
    if (results.length === 0) return []
    const numPoints = Math.min(numPulls, 500) // Downsample for performance
    const step = Math.max(1, Math.floor(numPulls / numPoints))

    const data = []
    for (let i = 0; i < numPulls; i += step) {
      const point: Record<string, number> = { t: i + 1 }
      for (const result of results) {
        point[result.algorithm] = result.regret[i]
      }
      data.push(point)
    }
    return data
  }, [results, numPulls])

  const cumulativeRewardChartData = useMemo(() => {
    if (results.length === 0) return []
    const numPoints = Math.min(numPulls, 500)
    const step = Math.max(1, Math.floor(numPulls / numPoints))

    const data = []
    for (let i = 0; i < numPulls; i += step) {
      const point: Record<string, number> = { t: i + 1 }
      for (const result of results) {
        point[result.algorithm] = result.cumulativeReward[i]
      }
      // Add optimal baseline
      if (optimalArmInfo) {
        point['Optimal'] = (i + 1) * optimalArmInfo.reward
      }
      data.push(point)
    }
    return data
  }, [results, numPulls, optimalArmInfo])

  const armCountsData = useMemo(() => {
    if (results.length === 0 || !problem) return []

    return problem.arms.map((arm, i) => {
      const row: Record<string, number | string> = { arm: arm.name }
      for (const result of results) {
        row[result.algorithm] = result.armCounts[i]
      }
      return row
    })
  }, [results, problem])

  return (
    <div className={styles.section}>
      <div className={styles.editorBlock}>
        <label className={styles.label} htmlFor="bandit-dsl">
          Multi-Armed Bandit Problem
        </label>
        <p className={styles.hint}>
          Define arms and their reward distributions. Supported: <code>Bernoulli(p)</code>, <code>Gaussian(mean, std)</code>, <code>Uniform(min, max)</code>
        </p>
        <textarea
          id="bandit-dsl"
          className={styles.textarea}
          value={dsl}
          onChange={(e) => setDsl(e.target.value)}
          rows={8}
          spellCheck={false}
        />
        {error && <p className={styles.error}>{error}</p>}
        <button type="button" className={styles.loadBtn} onClick={handleLoad}>
          Load Problem
        </button>
      </div>

      {problem && (
        <>
          <div className={styles.matrixBlock}>
            <h3 className={styles.matrixTitle}>Arm Definitions</h3>
            <div className={styles.matrixWrap}>
              <table className={styles.matrixTable}>
                <thead>
                  <tr>
                    <th className={styles.matrixHeader}>Arm</th>
                    <th className={styles.matrixHeader}>Distribution</th>
                    <th className={styles.matrixHeader}>Expected Reward</th>
                  </tr>
                </thead>
                <tbody>
                  {problem.arms.map((arm, i) => {
                    const exp = expectedReward(arm.distribution)
                    const isOptimal = optimalArmInfo?.index === i
                    return (
                      <tr key={arm.name}>
                        <th className={styles.matrixRowHeader}>
                          {arm.name} {isOptimal && '⭐'}
                        </th>
                        <td className={styles.matrixCell}>
                          {arm.distribution.type === 'bernoulli' && `Bernoulli(${arm.distribution.p})`}
                          {arm.distribution.type === 'gaussian' &&
                            `Gaussian(${arm.distribution.mean}, ${arm.distribution.std})`}
                          {arm.distribution.type === 'uniform' &&
                            `Uniform(${arm.distribution.min}, ${arm.distribution.max})`}
                        </td>
                        <td className={styles.matrixCell}>{exp.toFixed(3)}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            {optimalArmInfo && (
              <p className={styles.theoreticalHint}>
                Optimal arm: <strong>{optimalArmInfo.name}</strong> (expected reward: {optimalArmInfo.reward.toFixed(3)})
              </p>
            )}
          </div>

          <div className={styles.optionsBlock}>
            <h3 className={styles.optionsTitle}>Algorithm Selection</h3>
            <div className={styles.simulateForm}>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px', marginBottom: '16px' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <input
                    type="checkbox"
                    checked={selectedAlgorithms.has('random')}
                    onChange={() => toggleAlgorithm('random')}
                  />
                  <span>Random (baseline)</span>
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <input
                    type="checkbox"
                    checked={selectedAlgorithms.has('greedy')}
                    onChange={() => toggleAlgorithm('greedy')}
                  />
                  <span>Greedy</span>
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <input
                    type="checkbox"
                    checked={selectedAlgorithms.has('epsilon-greedy')}
                    onChange={() => toggleAlgorithm('epsilon-greedy')}
                  />
                  <span>ε-greedy</span>
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <input
                    type="checkbox"
                    checked={selectedAlgorithms.has('ucb')}
                    onChange={() => toggleAlgorithm('ucb')}
                  />
                  <span>UCB</span>
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <input
                    type="checkbox"
                    checked={selectedAlgorithms.has('thompson-sampling')}
                    onChange={() => toggleAlgorithm('thompson-sampling')}
                  />
                  <span>Thompson Sampling</span>
                </label>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px' }}>
                {selectedAlgorithms.has('epsilon-greedy') && (
                  <label className={styles.fieldLabel}>
                    <span>ε (epsilon)</span>
                    <input
                      type="number"
                      min={0}
                      max={1}
                      step={0.01}
                      value={epsilon}
                      onChange={(e) => setEpsilon(Number(e.target.value))}
                      className={styles.input}
                    />
                  </label>
                )}
                {selectedAlgorithms.has('ucb') && (
                  <label className={styles.fieldLabel}>
                    <span>UCB constant (c)</span>
                    <input
                      type="number"
                      min={0}
                      step={0.1}
                      value={ucbC}
                      onChange={(e) => setUcbC(Number(e.target.value))}
                      className={styles.input}
                    />
                  </label>
                )}
              </div>
            </div>

            <h3 className={styles.optionsTitle} style={{ marginTop: '24px' }}>
              Simulation Settings
            </h3>
            <div className={styles.simulateForm}>
              <label className={styles.fieldLabel}>
                <span>Number of pulls</span>
                <input
                  type="number"
                  min={10}
                  max={100000}
                  value={numPulls}
                  onChange={(e) => setNumPulls(Number(e.target.value) || 10)}
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
                disabled={simRunning || selectedAlgorithms.size === 0}
              >
                {simRunning ? 'Running…' : 'Run Simulation'}
              </button>
            </div>

            {results.length > 0 && (
              <>
                <div className={styles.chartBlock} style={{ marginTop: '32px' }}>
                  <h4 className={styles.chartTitle}>Cumulative Reward Over Time</h4>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={cumulativeRewardChartData} margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis
                        dataKey="t"
                        type="number"
                        tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                        label={{
                          value: 'Pull number',
                          position: 'insideBottom',
                          offset: -4,
                          fill: 'var(--text-muted)',
                          fontSize: 12,
                        }}
                      />
                      <YAxis
                        type="number"
                        tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                        label={{
                          value: 'Cumulative Reward',
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
                        formatter={(value: number, name: string) => [Number(value).toFixed(2), name]}
                        labelFormatter={(t) => `Pull ${t}`}
                      />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="Optimal"
                        stroke="#888"
                        strokeWidth={2}
                        strokeDasharray="5 5"
                        dot={false}
                      />
                      {results.map((result, i) => (
                        <Line
                          key={result.algorithm}
                          type="monotone"
                          dataKey={result.algorithm}
                          stroke={COLORS[i % COLORS.length]}
                          strokeWidth={2}
                          dot={false}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                <div className={styles.chartBlock}>
                  <h4 className={styles.chartTitle}>Regret Over Time</h4>
                  <p className={styles.theoreticalHint}>
                    Regret = cumulative difference between optimal reward and actual reward
                  </p>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={regretChartData} margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis
                        dataKey="t"
                        type="number"
                        tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                        label={{
                          value: 'Pull number',
                          position: 'insideBottom',
                          offset: -4,
                          fill: 'var(--text-muted)',
                          fontSize: 12,
                        }}
                      />
                      <YAxis
                        type="number"
                        tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                        label={{
                          value: 'Cumulative Regret',
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
                        formatter={(value: number, name: string) => [Number(value).toFixed(2), name]}
                        labelFormatter={(t) => `Pull ${t}`}
                      />
                      <Legend />
                      {results.map((result, i) => (
                        <Line
                          key={result.algorithm}
                          type="monotone"
                          dataKey={result.algorithm}
                          stroke={COLORS[i % COLORS.length]}
                          strokeWidth={2}
                          dot={false}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                <div className={styles.chartBlock}>
                  <h4 className={styles.chartTitle}>Arm Selection Counts</h4>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={armCountsData} margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis
                        dataKey="arm"
                        tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                        label={{
                          value: 'Arm',
                          position: 'insideBottom',
                          offset: -4,
                          fill: 'var(--text-muted)',
                          fontSize: 12,
                        }}
                      />
                      <YAxis
                        tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                        label={{
                          value: 'Number of pulls',
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
                      />
                      <Legend />
                      {results.map((result, i) => (
                        <Bar
                          key={result.algorithm}
                          dataKey={result.algorithm}
                          fill={COLORS[i % COLORS.length]}
                        />
                      ))}
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                <div className={styles.matrixBlock}>
                  <h4 className={styles.matrixTitle}>Final Results</h4>
                  <div className={styles.matrixWrap}>
                    <table className={styles.matrixTable}>
                      <thead>
                        <tr>
                          <th className={styles.matrixHeader}>Algorithm</th>
                          <th className={styles.matrixHeader}>Total Reward</th>
                          <th className={styles.matrixHeader}>Final Regret</th>
                          <th className={styles.matrixHeader}>Avg Reward/Pull</th>
                        </tr>
                      </thead>
                      <tbody>
                        {results.map((result) => {
                          const totalReward = result.cumulativeReward[result.cumulativeReward.length - 1]
                          const finalRegret = result.regret[result.regret.length - 1]
                          const avgReward = totalReward / numPulls
                          return (
                            <tr key={result.algorithm}>
                              <th className={styles.matrixRowHeader}>{result.algorithm}</th>
                              <td className={styles.matrixCell}>{totalReward.toFixed(2)}</td>
                              <td className={styles.matrixCell}>{finalRegret.toFixed(2)}</td>
                              <td className={styles.matrixCell}>{avgReward.toFixed(3)}</td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              </>
            )}
          </div>
        </>
      )}
    </div>
  )
}
