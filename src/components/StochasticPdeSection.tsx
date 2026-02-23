import { useState, useMemo, useCallback, useRef } from 'react'
import katex from 'katex'
import 'katex/dist/katex.min.css'
import type { ProcessDef, CustomProcessInput } from '@/types/process'
import type { SimConfig, SimResult } from '@/types/simulation'
import { ProcessPicker } from '@/components/ProcessPicker'
import { SimConfigForm } from '@/components/SimConfigForm'
import { PathsPlot } from '@/components/PathsPlot'
import { StatsPlot } from '@/components/StatsPlot'
import { SolutionsPanel } from '@/components/SolutionsPanel'
import { DensityPanel } from '@/components/DensityPanel'
import { eulerMaruyama } from '@/lib/sde'
import { computeStats } from '@/lib/stats'
import { compileCustomProcess } from '@/lib/customProcess'
import { getBuiltInProcesses } from '@/lib/processes'
import {
  pathsToCsv,
  statsToCsv,
  downloadBlob,
  exportChartSvg,
  exportChartPng,
} from '@/lib/export'
import styles from './StochasticPdeSection.module.css'

function renderLatex(latex: string, displayMode = false): string {
  try {
    return katex.renderToString(latex, { displayMode, throwOnError: false })
  } catch {
    return latex
  }
}

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

export function StochasticPdeSection() {
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
  const [resultTab, setResultTab] = useState<'paths' | 'statistics' | 'solutions' | 'density'>('paths')
  const chartContainerRef = useRef<HTMLDivElement>(null)

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

  const currentProcess =
    mode === 'built-in'
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

  const handleExportPathsCsv = useCallback(() => {
    if (!result?.paths.length) return
    const csv = pathsToCsv(result.paths)
    downloadBlob(csv, 'paths.csv', 'text/csv;charset=utf-8')
  }, [result])

  const handleExportStatsCsv = useCallback(() => {
    if (!stats) return
    const csv = statsToCsv(stats)
    downloadBlob(csv, 'statistics.csv', 'text/csv;charset=utf-8')
  }, [stats])

  const handleExportChartSvg = useCallback(() => {
    const name = `${resultTab}-chart.svg`
    exportChartSvg(chartContainerRef.current, name)
  }, [resultTab])

  const handleExportChartPng = useCallback(() => {
    const name = `${resultTab}-chart.png`
    exportChartPng(chartContainerRef.current, name)
  }, [resultTab])

  return (
    <div className={styles.section}>
      <div className={styles.intro}>
        <p className={styles.introText}>
          A <strong>stochastic differential equation (SDE)</strong> has the form{' '}
          <span dangerouslySetInnerHTML={{ __html: renderLatex('dX_t = f(X_t)\\,dt + g(X_t)\\,dW_t') }} />
          , where <span dangerouslySetInnerHTML={{ __html: renderLatex('W_t') }} /> is a Wiener process. Paths are simulated with the <strong>Euler–Maruyama</strong> scheme:{' '}
          <span dangerouslySetInnerHTML={{ __html: renderLatex('X_{t+\\Delta t} = X_t + f(X_t)\\Delta t + g(X_t)\\sqrt{\\Delta t}\\,Z') }} />
          with <span dangerouslySetInnerHTML={{ __html: renderLatex('Z \\sim \\mathcal{N}(0,1)') }} />. The probability density <span dangerouslySetInnerHTML={{ __html: renderLatex('p(x,t)') }} /> evolves according to the <strong>Fokker–Planck equation</strong>{' '}
          <span dangerouslySetInnerHTML={{ __html: renderLatex('\\frac{\\partial p}{\\partial t} = -\\frac{\\partial}{\\partial x}[f\\,p] + \\frac{1}{2}\\frac{\\partial^2}{\\partial x^2}[g^2\\,p]') }} />.
        </p>
      </div>
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
        <div className={styles.resultTabs}>
          <button
            type="button"
            className={resultTab === 'paths' ? styles.resultTabActive : styles.resultTab}
            onClick={() => setResultTab('paths')}
          >
            Paths
          </button>
          <button
            type="button"
            className={resultTab === 'statistics' ? styles.resultTabActive : styles.resultTab}
            onClick={() => setResultTab('statistics')}
          >
            Statistics
          </button>
          <button
            type="button"
            className={resultTab === 'solutions' ? styles.resultTabActive : styles.resultTab}
            onClick={() => setResultTab('solutions')}
          >
            Solutions
          </button>
          <button
            type="button"
            className={resultTab === 'density' ? styles.resultTabActive : styles.resultTab}
            onClick={() => setResultTab('density')}
          >
            Density p(x,t)
          </button>
        </div>
        <div className={styles.exportRow}>
          <span className={styles.exportLabel}>Export:</span>
          <button
            type="button"
            className={styles.exportBtn}
            onClick={handleExportPathsCsv}
            disabled={!result?.paths.length}
            title="Download paths as CSV (t, x1, x2, …)"
          >
            Paths CSV
          </button>
          <button
            type="button"
            className={styles.exportBtn}
            onClick={handleExportStatsCsv}
            disabled={!stats}
            title="Download statistics as CSV"
          >
            Stats CSV
          </button>
          <button
            type="button"
            className={styles.exportBtn}
            onClick={handleExportChartSvg}
            title="Download current chart as SVG"
          >
            Chart SVG
          </button>
          <button
            type="button"
            className={styles.exportBtn}
            onClick={handleExportChartPng}
            title="Download current chart as PNG"
          >
            Chart PNG
          </button>
        </div>
        {resultTab === 'paths' && (
          <section className={styles.resultSection}>
            <h2 className={styles.resultHeading}>Paths</h2>
            <p className={styles.graphDesc}>
              Sample trajectories of the SDE: each curve is one path <span dangerouslySetInnerHTML={{ __html: renderLatex('X_t') }} /> simulated with Euler–Maruyama from <span dangerouslySetInnerHTML={{ __html: renderLatex('t_0') }} /> to <span dangerouslySetInnerHTML={{ __html: renderLatex('T') }} />. Up to 150 paths are drawn for clarity.
            </p>
            <PathsPlot
              paths={result?.paths ?? []}
              x0={x0}
              chartRef={chartContainerRef}
            />
          </section>
        )}
        {resultTab === 'statistics' && (
          <section className={styles.resultSection}>
            <h2 className={styles.resultHeading}>Statistics</h2>
            <p className={styles.graphDesc}>
              Empirical mean and mean ± 2 standard deviations across all paths at each time. The band shows how the distribution of <span dangerouslySetInnerHTML={{ __html: renderLatex('X_t') }} /> spreads as <span dangerouslySetInnerHTML={{ __html: renderLatex('t') }} /> increases.
            </p>
            <StatsPlot
              stats={stats}
              x0={x0}
              chartRef={chartContainerRef}
            />
          </section>
        )}
        {resultTab === 'solutions' && (
          <section className={styles.resultSection}>
            <h2 className={styles.resultHeading}>Solutions</h2>
            <p className={styles.graphDesc}>
              Compares the simulated mean (and ±2σ band) from the paths with the theoretical mean and standard deviation for this process, when a closed-form solution exists. Theory is shown as dashed lines; simulation as solid.
            </p>
            <SolutionsPanel
              processId={currentProcess?.id ?? ''}
              params={params}
              x0={x0}
              config={config}
              result={result}
              stats={stats ?? null}
              chartRef={chartContainerRef}
            />
          </section>
        )}
        {resultTab === 'density' && (
          <section className={styles.resultSection}>
            <h2 className={styles.resultHeading}>Density p(x, t)</h2>
            <p className={styles.graphDesc}>
              Solution of the Fokker–Planck equation for the probability density <span dangerouslySetInnerHTML={{ __html: renderLatex('p(x,t)') }} />. The curve shows <span dangerouslySetInnerHTML={{ __html: renderLatex('p(x,t)') }} /> at the time selected by the slider; move it to see how the density evolves from the initial condition.
            </p>
            {currentProcess ? (
              <DensityPanel
                process={currentProcess}
                params={params}
                x0={x0}
                config={config}
                chartRef={chartContainerRef}
              />
            ) : (
              <div className={styles.emptyState}>
                Select a process to solve the Fokker-Planck equation for p(x, t).
              </div>
            )}
          </section>
        )}
      </div>
    </div>
  )
}
