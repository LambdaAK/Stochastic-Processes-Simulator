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
import { generatePresetDataset } from '@/lib/knn'
import type { KNNDatasetPreset } from '@/types/knn'
import type { TrainingPoint } from '@/lib/decisionTree'
import { createSeededRng } from '@/lib/random'
import { fitBaggedTrees, baggingDecisionGrid, predictBagged } from '@/lib/bagging'
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
  { id: 'moons', label: 'Moons' },
  { id: 'three-blobs', label: 'Three blobs' },
  { id: 'stripes', label: 'Stripes' },
  { id: 'nested', label: 'Nested rectangle' },
]

const PALETTE = [
  'var(--accent)',
  '#0ea5e9',
  '#22c55e',
  '#a855f7',
  '#eab308',
  '#ef4444',
  '#06b6d4',
  '#f97316',
]

const N_PER_CLASS = 50
const DOMAIN = { xMin: -4, xMax: 4, yMin: -4, yMax: 4 }

function getColorForLabel(label: string, allLabels: string[]): string {
  const index = allLabels.indexOf(label)
  if (index === -1) return '#94a3b8'
  return PALETTE[index % PALETTE.length] ?? '#94a3b8'
}

export function BaggingSection() {
  const [preset, setPreset] = useState<KNNDatasetPreset>('blobs')
  const [training, setTraining] = useState<TrainingPoint[]>([])
  const [seed, setSeed] = useState('')
  const [nTrees, setNTrees] = useState(25)
  const [maxDepth, setMaxDepth] = useState(5)
  const [showDecisionBoundary, setShowDecisionBoundary] = useState(true)
  const [gridRes, setGridRes] = useState(32)
  const [query, setQuery] = useState<{ x: number; y: number } | null>(null)
  const [predictedLabel, setPredictedLabel] = useState<string | null>(null)

  const generateData = useCallback(() => {
    const rand =
      seed.trim() !== '' && !Number.isNaN(Number(seed))
        ? createSeededRng(Number(seed))
        : Math.random
    const points = generatePresetDataset(preset, N_PER_CLASS, rand)
    setTraining(points)
    setQuery(null)
    setPredictedLabel(null)
  }, [preset, seed])

  const handlePresetChange = (p: KNNDatasetPreset) => {
    setPreset(p)
    setTraining([])
    setQuery(null)
    setPredictedLabel(null)
  }

  const model = useMemo(
    () => fitBaggedTrees(training, nTrees, maxDepth),
    [training, nTrees, maxDepth]
  )

  const allLabels = useMemo(
    () => Array.from(new Set(training.map((p) => p.label))).sort(),
    [training]
  )

  const scatterByLabel = useMemo(() => {
    const byLabel: Record<string, TrainingPoint[]> = {}
    for (const p of training) {
      if (!byLabel[p.label]) byLabel[p.label] = []
      byLabel[p.label].push(p)
    }
    return Object.entries(byLabel)
  }, [training])

  const decisionGridData = useMemo(() => {
    if (!showDecisionBoundary || training.length === 0 || !model) return []
    return baggingDecisionGrid(
      model,
      DOMAIN.xMin,
      DOMAIN.xMax,
      DOMAIN.yMin,
      DOMAIN.yMax,
      gridRes,
      gridRes
    )
  }, [model, showDecisionBoundary, gridRes, training.length])

  const handlePredict = useCallback(() => {
    if (!model || training.length === 0) return
    const q = query ?? { x: 0, y: 0 }
    setQuery(q)
    setPredictedLabel(predictBagged(model, q))
  }, [model, training.length, query])

  const trainingAccuracy = useMemo(() => {
    if (!model || training.length === 0) return null
    let correct = 0
    for (const p of training) {
      const pred = predictBagged(model, { x: p.x, y: p.y })
      if (pred === p.label) correct += 1
    }
    return correct / training.length
  }, [model, training])

  return (
    <div className={styles.section}>
      <div className={styles.intro}>
        <p className={styles.introText}>
          <strong>Bagging (Bootstrap Aggregating)</strong> builds many decision trees on bootstrap
          samples of the training data and averages their predictions. Compared to a single,
          high-variance tree, bagging <em>reduces variance</em> by voting across independently
          trained models.
        </p>
        <p className={styles.introText}>
          In this sandbox each base learner is a CART tree on 2D data (x, y, label). Use the same
          synthetic datasets as in KNN and single decision trees, then vary the number of trees and
          tree depth to see how the ensemble&apos;s decision boundary changes.
        </p>
        <p
          className={styles.introFormula}
          dangerouslySetInnerHTML={{
            __html: renderLatex(
              '\\hat{y}_{\\text{bag}}(x) = \\mathrm{mode}\\{\\hat{y}^{(b)}(x) : b = 1, \\dots, B\\}',
              true
            ),
          }}
        />
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
        <h3 className={styles.optionsTitle}>Bagging configuration</h3>
        <div className={styles.theoreticalForm}>
          <label className={styles.fieldLabel}>
            <span>Number of trees (B)</span>
            <input
              type="number"
              min={1}
              max={200}
              value={nTrees}
              onChange={(e) =>
                setNTrees(Math.max(1, Math.min(200, Number(e.target.value) || 1)))
              }
              className={styles.input}
            />
          </label>
          <label className={styles.fieldLabel}>
            <span>Max depth per tree</span>
            <input
              type="number"
              min={1}
              max={15}
              value={maxDepth}
              onChange={(e) =>
                setMaxDepth(Math.max(1, Math.min(15, Number(e.target.value) || 1)))
              }
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
                min={12}
                max={60}
                value={gridRes}
                onChange={(e) =>
                  setGridRes(Math.max(12, Math.min(60, Number(e.target.value) || 32)))
                }
                className={styles.input}
              />
            </label>
          )}
        </div>
        {trainingAccuracy !== null && (
          <p className={styles.hint} style={{ marginTop: '0.5rem', marginBottom: 0 }}>
            Training accuracy of ensemble: {(trainingAccuracy * 100).toFixed(1)}%
          </p>
        )}
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
            disabled={!model || training.length === 0}
          >
            Predict (bagged)
          </button>
          {predictedLabel !== null && (
            <span className={styles.introText} style={{ alignSelf: 'center' }}>
              Predicted: <strong>{predictedLabel}</strong>
            </span>
          )}
        </div>
      </div>

      {training.length > 0 && (
        <div className={styles.graphBlock}>
          <h3 className={styles.graphTitle}>
            Training data {showDecisionBoundary && 'and bagged decision boundary'}
          </h3>
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
                  payload?.[0]
                    ? `(${payload[0].payload?.x?.toFixed(3)}, ${payload[0].payload?.y?.toFixed(
                        3
                      )})`
                    : ''
                }
              />
              {showDecisionBoundary && decisionGridData.length > 0 && (
                <Scatter
                  name="Bagged boundary"
                  data={decisionGridData.map((d) => ({ ...d, z: 30 }))}
                  fillOpacity={0.25}
                  isAnimationActive={false}
                >
                  {decisionGridData.map((entry, i) => (
                    <Cell
                      key={i}
                      fill={getColorForLabel(entry.label, allLabels)}
                      fillOpacity={0.25}
                    />
                  ))}
                </Scatter>
              )}
              {scatterByLabel.map(([label, points]) => (
                <Scatter
                  key={label}
                  name={`Class ${label}`}
                  data={points}
                  fill={getColorForLabel(label, allLabels)}
                  shape="circle"
                  isAnimationActive={false}
                >
                  {points.map((_, i) => (
                    <Cell
                      key={i}
                      fill={getColorForLabel(label, allLabels)}
                      stroke="var(--text)"
                      strokeWidth={1}
                    />
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
        </div>
      )}
    </div>
  )
}

