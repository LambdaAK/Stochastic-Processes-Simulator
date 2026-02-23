import { useState, useMemo, useCallback } from 'react'
import katex from 'katex'
import 'katex/dist/katex.min.css'
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ZAxis,
  Cell,
} from 'recharts'
import {
  knnPredict,
  knnDecisionGrid,
  generatePresetDataset,
  getKNearest,
} from '@/lib/knn'
import { createSeededRng } from '@/lib/random'
import type { TrainingPoint } from '@/lib/knn'
import type { KNNDatasetPreset } from '@/types/knn'
import styles from './MarkovChainSection.module.css'

function renderLatex(latex: string, displayMode = false): string {
  try {
    return katex.renderToString(latex, { displayMode, throwOnError: false })
  } catch {
    return latex
  }
}

const PRESETS: { id: KNNDatasetPreset; label: string }[] = [
  { id: 'blobs', label: 'Two blobs' },
  { id: 'xor', label: 'XOR' },
  { id: 'circles', label: 'Circles' },
]

const LABEL_COLORS: Record<string, string> = {
  A: 'var(--accent)',
  B: '#0ea5e9',
  '0': '#22c55e',
  '1': '#a855f7',
}

function getColor(label: string): string {
  return LABEL_COLORS[label] ?? '#94a3b8'
}

const N_PER_CLASS = 50
const DOMAIN = { xMin: -4, xMax: 4, yMin: -4, yMax: 4 }

export function KNNSection() {
  const [preset, setPreset] = useState<KNNDatasetPreset>('blobs')
  const [training, setTraining] = useState<TrainingPoint[]>(() => [])
  const [k, setK] = useState(5)
  const [query, setQuery] = useState<{ x: number; y: number } | null>(null)
  const [predictedLabel, setPredictedLabel] = useState<string | null>(null)
  const [showDecisionBoundary, setShowDecisionBoundary] = useState(false)
  const [gridRes, setGridRes] = useState(24)
  const [seed, setSeed] = useState('')

  const generateData = useCallback(() => {
    const rand =
      seed.trim() !== '' && !Number.isNaN(Number(seed))
        ? createSeededRng(Number(seed))
        : Math.random
    setTraining(generatePresetDataset(preset, N_PER_CLASS, rand))
    setQuery(null)
    setPredictedLabel(null)
  }, [preset, seed])

  // Initial load and when preset changes
  const handlePresetChange = (p: KNNDatasetPreset) => {
    setPreset(p)
    setTraining([])
    setQuery(null)
    setPredictedLabel(null)
  }

  const handlePredict = useCallback(() => {
    if (training.length === 0) return
    const q = query ?? { x: 0, y: 0 }
    const label = knnPredict(q, training, Math.min(k, training.length))
    setQuery(q)
    setPredictedLabel(label)
  }, [training, k, query])

  const scatterByLabel = useMemo(() => {
    const byLabel: Record<string, { x: number; y: number; label: string }[]> = {}
    for (const p of training) {
      if (!byLabel[p.label]) byLabel[p.label] = []
      byLabel[p.label].push({ x: p.x, y: p.y, label: p.label })
    }
    return Object.entries(byLabel)
  }, [training])

  const decisionGridData = useMemo(() => {
    if (!showDecisionBoundary || training.length === 0) return []
    return knnDecisionGrid(
      training,
      Math.min(k, training.length),
      DOMAIN.xMin,
      DOMAIN.xMax,
      DOMAIN.yMin,
      DOMAIN.yMax,
      gridRes,
      gridRes
    )
  }, [training, k, showDecisionBoundary, gridRes])

  const kNearest = useMemo(() => {
    if (!query || training.length === 0) return []
    return getKNearest(query, training, Math.min(k, training.length))
  }, [query, training, k])

  return (
    <div className={styles.section}>
      <div className={styles.intro}>
        <p className={styles.introText}>
          <strong>K-Nearest Neighbors (KNN)</strong> is a non-parametric classifier. For a query point{' '}
          <span dangerouslySetInnerHTML={{ __html: renderLatex('\\mathbf{x}') }} />, find the{' '}
          <span dangerouslySetInnerHTML={{ __html: renderLatex('k') }} /> nearest training points (by Euclidean distance) and predict the majority class.
        </p>
        <p className={styles.introFormula} dangerouslySetInnerHTML={{ __html: renderLatex('\\hat{y} = \\mathrm{mode}\\{y_i : i \\in \\mathcal{N}_k(\\mathbf{x})\\}', true) }} />
        <p className={styles.introText}>
          Choose a preset dataset, set <em>k</em>, then enter a query (x, y) and click Predict to see the classification.
        </p>
      </div>

      <div className={styles.editorBlock}>
        <h3 className={styles.optionsTitle}>Dataset</h3>
        <div className={styles.theoreticalForm}>
          <label className={styles.fieldLabel}>
            <span>Preset</span>
            <select
              className={styles.input}
              value={preset}
              onChange={(e) => handlePresetChange(e.target.value as KNNDatasetPreset)}
            >
              {PRESETS.map((opt) => (
                <option key={opt.id} value={opt.id}>
                  {opt.label}
                </option>
              ))}
            </select>
          </label>
          <label className={styles.fieldLabel}>
            <span>Seed (optional)</span>
            <input
              type="text"
              className={styles.input}
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
              placeholder="e.g. 42"
            />
          </label>
          <button type="button" className={styles.runBtn} onClick={generateData}>
            Generate data
          </button>
        </div>
      </div>

      <div className={styles.editorBlock}>
        <h3 className={styles.optionsTitle}>KNN</h3>
        <div className={styles.theoreticalForm}>
          <label className={styles.fieldLabel}>
            <span>k (neighbors)</span>
            <input
              type="number"
              min={1}
              max={training.length || 99}
              value={k}
              onChange={(e) => setK(Math.max(1, Math.min(training.length || 99, Number(e.target.value) || 1)))}
              className={styles.input}
            />
          </label>
          <label className={styles.fieldLabel}>
            <input
              type="checkbox"
              checked={showDecisionBoundary}
              onChange={(e) => setShowDecisionBoundary(e.target.checked)}
            />
            <span>Show decision boundary</span>
          </label>
          {showDecisionBoundary && (
            <label className={styles.fieldLabel}>
              <span>Grid resolution</span>
              <input
                type="number"
                min={10}
                max={50}
                value={gridRes}
                onChange={(e) => setGridRes(Math.max(10, Math.min(50, Number(e.target.value) || 24)))}
                className={styles.input}
              />
            </label>
          )}
        </div>
      </div>

      <div className={styles.editorBlock}>
        <h3 className={styles.optionsTitle}>Query point</h3>
        <div className={styles.theoreticalForm}>
          <label className={styles.fieldLabel}>
            <span>x</span>
            <input
              type="number"
              step={0.1}
              value={query?.x ?? ''}
              onChange={(e) =>
                setQuery((q) => ({
                  x: Number(e.target.value) || 0,
                  y: q?.y ?? 0,
                }))
              }
              className={styles.input}
              placeholder="0"
            />
          </label>
          <label className={styles.fieldLabel}>
            <span>y</span>
            <input
              type="number"
              step={0.1}
              value={query?.y ?? ''}
              onChange={(e) =>
                setQuery((q) => ({
                  x: q?.x ?? 0,
                  y: Number(e.target.value) || 0,
                }))
              }
              className={styles.input}
              placeholder="0"
            />
          </label>
          <button
            type="button"
            className={styles.runBtn}
            onClick={handlePredict}
            disabled={training.length === 0}
          >
            Predict
          </button>
          {predictedLabel !== null && (
            <span className={styles.introText} style={{ alignSelf: 'center' }}>
              Predicted: <strong style={{ color: getColor(predictedLabel) }}>{predictedLabel}</strong>
            </span>
          )}
        </div>
      </div>

      {training.length > 0 && (
        <div className={styles.graphBlock}>
          <h3 className={styles.graphTitle}>Training data and query</h3>
          <ResponsiveContainer width="100%" height={420}>
            <ScatterChart
              margin={{ top: 16, right: 16, left: 16, bottom: 16 }}
              data={training}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis
                type="number"
                dataKey="x"
                domain={[DOMAIN.xMin, DOMAIN.xMax]}
                stroke="var(--text-muted)"
                tick={{ fill: 'var(--text-muted)', fontSize: 12 }}
              />
              <YAxis
                type="number"
                dataKey="y"
                domain={[DOMAIN.yMin, DOMAIN.yMax]}
                stroke="var(--text-muted)"
                tick={{ fill: 'var(--text-muted)', fontSize: 12 }}
              />
              <ZAxis range={[50, 400]} />
              <Tooltip
                cursor={{ stroke: 'var(--border)' }}
                contentStyle={{
                  background: 'var(--glass-bg)',
                  border: '1px solid var(--glass-border)',
                  borderRadius: 'var(--radius)',
                }}
                formatter={(value: number) => [value.toFixed(3), '']}
                labelFormatter={(_label, payload) =>
                  payload?.[0] ? `(${payload[0].payload?.x?.toFixed(3)}, ${payload[0].payload?.y?.toFixed(3)})` : ''
                }
              />
              {showDecisionBoundary && decisionGridData.length > 0 && (
                <Scatter
                  name="Boundary"
                  data={decisionGridData.map((d) => ({ ...d, z: 30 }))}
                  fillOpacity={0.25}
                  isAnimationActive={false}
                >
                  {decisionGridData.map((entry, i) => (
                    <Cell key={i} fill={getColor(entry.label)} fillOpacity={0.25} />
                  ))}
                </Scatter>
              )}
              {scatterByLabel.map(([label, points]) => (
                <Scatter
                  key={label}
                  name={`Class ${label}`}
                  data={points}
                  fill={getColor(label)}
                  shape="circle"
                  isAnimationActive={false}
                >
                  {points.map((_, i) => (
                    <Cell key={i} fill={getColor(label)} stroke="var(--text)" strokeWidth={1} />
                  ))}
                </Scatter>
              ))}
              {query !== null && (
                <Scatter
                  name="Query"
                  data={[{ ...query, label: predictedLabel ?? '?' }]}
                  fill="none"
                  shape="cross"
                  line={{ stroke: 'var(--text)', strokeWidth: 2 }}
                  isAnimationActive={false}
                >
                  <Cell stroke="var(--text)" strokeWidth={2} />
                </Scatter>
              )}
              <Legend />
            </ScatterChart>
          </ResponsiveContainer>
          {kNearest.length > 0 && (
            <p className={styles.hint} style={{ marginTop: '0.5rem' }}>
              K-nearest distances: {kNearest.map((n) => n.dist.toFixed(3)).join(', ')}
            </p>
          )}
        </div>
      )}
    </div>
  )
}
