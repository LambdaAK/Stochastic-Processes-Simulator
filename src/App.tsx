import { useState, useMemo, useCallback } from 'react'
import type { ProcessDef, CustomProcessInput } from '@/types/process'
import type { SimConfig, SimResult } from '@/types/simulation'
import { ProcessPicker } from '@/components/ProcessPicker'
import { SimConfigForm } from '@/components/SimConfigForm'
import { PathsPlot } from '@/components/PathsPlot'
import { StatsPlot } from '@/components/StatsPlot'
import { eulerMaruyama } from '@/lib/sde'
import { computeStats } from '@/lib/stats'
import { compileCustomProcess } from '@/lib/customProcess'
import { getBuiltInProcesses } from '@/lib/processes'
import styles from './App.module.css'

const defaultCustomInput: CustomProcessInput = {
  driftExpr: 'theta * (mu - x)',
  diffusionExpr: 'sigma',
  params: [
    { id: 'theta', name: 'θ', default: 1, min: 0.01, max: 10 },
    { id: 'mu', name: 'μ', default: 0, min: -20, max: 20 },
    { id: 'sigma', name: 'σ', default: 1, min: 0.01, max: 5 },
  ],
}

function getDefaultParams(process: ProcessDef): Record<string, number> {
  const out: Record<string, number> = {}
  process.params.forEach((p) => (out[p.id] = p.default))
  return out
}

export default function App() {
  const [mode, setMode] = useState<'built-in' | 'custom'>('built-in')
  const [selectedId, setSelectedId] = useState<string | null>('ornstein-uhlenbeck')
  const [customInput, setCustomInput] = useState<CustomProcessInput>(defaultCustomInput)
  const [params, setParams] = useState<Record<string, number>>(() => {
    const ou = getBuiltInProcesses().find((p) => p.id === 'ornstein-uhlenbeck')!
    return getDefaultParams(ou)
  })
  const [x0, setX0] = useState(0)
  const [config, setConfig] = useState<SimConfig>({
    t0: 0,
    T: 5,
    dt: 0.01,
    M: 200,
    x0: 0,
  })
  const [result, setResult] = useState<SimResult | null>(null)
  const [running, setRunning] = useState(false)

  const compiledCustom = useMemo(() => {
    if (mode !== 'custom') return null
    const out = compileCustomProcess(customInput)
    return 'error' in out ? null : out
  }, [mode, customInput])

  const customError = useMemo(() => {
    if (mode !== 'custom') return null
    const out = compileCustomProcess(customInput)
    return 'error' in out ? out.error : null
  }, [mode, customInput])

  const currentProcess = mode === 'built-in'
    ? getBuiltInProcesses().find((p) => p.id === selectedId) ?? null
    : compiledCustom

  const handleSelectBuiltIn = useCallback((p: ProcessDef) => {
    setSelectedId(p.id)
    setParams(getDefaultParams(p))
  }, [])

  const handleRun = useCallback(() => {
    if (!currentProcess) return
    setRunning(true)
    const simConfig: SimConfig = { ...config, x0 }
    const allParams = { ...params }
    currentProcess.params.forEach((p) => {
      if (!(p.id in allParams)) allParams[p.id] = p.default
    })
    setTimeout(() => {
      const paths = eulerMaruyama(simConfig, currentProcess, allParams)
      setResult({ paths, config: simConfig })
      setRunning(false)
    }, 0)
  }, [currentProcess, config, x0, params])

  const stats = result ? computeStats(result.paths) : null

  return (
    <div className={styles.app}>
      <header className={styles.header}>
        <h1 className={styles.title}>Stochastic Processes Simulator</h1>
        <p className={styles.tagline}>
          Stochastic differential equations — simulate paths, view statistics
        </p>
      </header>

      <main className={styles.main}>
        <div className={styles.controls}>
          <ProcessPicker
            mode={mode}
            onModeChange={setMode}
            selectedId={selectedId}
            onSelectBuiltIn={handleSelectBuiltIn}
            customInput={customInput}
            onCustomInputChange={setCustomInput}
            compiledCustom={compiledCustom}
            customError={customError}
            params={params}
            onParamsChange={setParams}
            x0={x0}
            onX0Change={setX0}
          />
          <SimConfigForm
            config={config}
            onChange={setConfig}
            onRun={handleRun}
            running={running}
          />
        </div>

        <div className={styles.results}>
          <section className={styles.resultSection}>
            <h2 className={styles.resultHeading}>Paths</h2>
            <PathsPlot paths={result?.paths ?? []} x0={x0} />
          </section>
          <section className={styles.resultSection}>
            <h2 className={styles.resultHeading}>Statistics</h2>
            <StatsPlot stats={stats} x0={x0} />
          </section>
        </div>
      </main>

      <footer className={styles.footer}>
        Euler–Maruyama · No backend · All simulation in the browser
      </footer>
    </div>
  )
}
