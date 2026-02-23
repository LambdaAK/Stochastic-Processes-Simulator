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
  trainBoostedTrees,
  boostedDecisionGrid,
  predictBoosted,
  type BoostedTreeModel,
} from '@/lib/boosting'
import { generatePresetDataset } from '@/lib/knn'
import type { KNNDatasetPreset } from '@/types/knn'
import { createSeededRng } from '@/lib/random'
import type { TrainingPoint } from '@/lib/decisionTree'
import { TreeVisualization } from '@/components/TreeVisualization'
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

function getLabelOrder(points: TrainingPoint[]): string[] {
  const seen = new Set<string>()
  for (const p of points) {
    if (!seen.has(p.label)) seen.add(p.label)
  }
  return Array.from(seen)
}

export function BoostingSection() {
  const [preset, setPreset] = useState<KNNDatasetPreset>('blobs')
  const [training, setTraining] = useState<TrainingPoint[]>([])
  const [nEstimators, setNEstimators] = useState(20)
  const [maxDepth, setMaxDepth] = useState(2)
  const [seed, setSeed] = useState('')
  const [boostSeed, setBoostSeed] = useState('')
  const [model, setModel] = useState<BoostedTreeModel>({ trees: [], alphas: [] })
  const [showDecisionBoundary, setShowDecisionBoundary] = useState(true)
  const [gridRes, setGridRes] = useState(32)
  const [query, setQuery] = useState<{ x: number; y: number } | null>(null)
  const [predictedLabel, setPredictedLabel] = useState<string | null>(null)

  const labelOrder = useMemo(() => getLabelOrder(training), [training])

  const getColor = useCallback(
    (label: string) => {
      const idx = labelOrder.indexOf(label)
      return PALETTE[idx >= 0 ? idx % PALETTE.length : 0] ?? '#94a3b8'
    },
    [labelOrder]
  )

  const generateData = useCallback(() => {
    const rand =
      seed.trim() !== '' && !Number.isNaN(Number(seed))
        ? createSeededRng(Number(seed))
        : Math.random
    const points = generatePresetDataset(preset, N_PER_CLASS, rand)
    setTraining(points)
    setModel({ trees: [], alphas: [] })
    setQuery(null)
    setPredictedLabel(null)
  }, [preset, seed])

  const handlePresetChange = (p: KNNDatasetPreset) => {
    setPreset(p)
    setTraining([])
    setModel({ trees: [], alphas: [] })
    setQuery(null)
    setPredictedLabel(null)
  }

  const trainModel = useCallback(() => {
    if (training.length === 0) return
    const cfgSeed =
      boostSeed.trim() !== '' && !Number.isNaN(Number(boostSeed))
        ? Number(boostSeed)
        : undefined
    const m = trainBoostedTrees(training, {
      nEstimators: Math.max(1, Math.min(100, nEstimators)),
      maxDepth: Math.max(1, Math.min(8, maxDepth)),
      seed: cfgSeed,
    })
    setModel(m)
    setPredictedLabel(null)
  }, [training, nEstimators, maxDepth, boostSeed])

  const handlePredict = useCallback(() => {
    if (!model.trees.length || training.length === 0) return
    const q = query ?? { x: 0, y: 0 }
    setQuery(q)
    const label = predictBoosted(model, q)
    setPredictedLabel(label || null)
  }, [model, query, training.length])

  const scatterByLabel = useMemo(() => {
    const byLabel: Record<string, { x: number; y: number; label: string }[]> = {}
    for (const p of training) {
      if (!byLabel[p.label]) byLabel[p.label] = []
      byLabel[p.label].push({ x: p.x, y: p.y, label: p.label })
    }
    return Object.entries(byLabel)
  }, [training])

  const decisionGridData = useMemo(() => {
    if (!showDecisionBoundary || training.length === 0 || !model.trees.length) return []
    return boostedDecisionGrid(
      model,
      DOMAIN.xMin,
      DOMAIN.xMax,
      DOMAIN.yMin,
      DOMAIN.yMax,
      gridRes,
      gridRes
    )
  }, [model, showDecisionBoundary, gridRes, training.length])

  return (
    <div className={styles.section}>
      <div className={styles.intro}>
        <p className={styles.introText}>
          <strong>Boosting</strong> combines many weak learners (here: shallow decision trees){' '}
          trained <em>sequentially</em>. Each tree focuses more on points that previous trees
          misclassified. The final prediction is a weighted vote of all trees.
        </p>
        <p
          className={styles.introFormula}
          dangerouslySetInnerHTML={{
            __html: renderLatex(
              '\\hat{f}(x) = \\sum_{m=1}^M \\alpha_m h_m(x),\\quad \\hat{y}(x) = \\arg\\max_k \\sum_{m: h_m(x)=k} \\alpha_m',
              true
            ),
          }}
        />
        <p className={styles.introText}>
          Compared to bagging (parallel averaging), boosting is <em>sequential</em> and tends to
          reduce bias. Increase the number of trees to see how the decision boundary sharpens.
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
            <span>Seed (data)</span>
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
        <h3 className={styles.optionsTitle}>Boosted trees</h3>
        <div className={styles.theoreticalForm}>
          <label className={styles.fieldLabel}>
            <span>Number of trees (M)</span>
            <input
              type="number"
              min={1}
              max={100}
              value={nEstimators}
              onChange={(e) =>
                setNEstimators(Math.max(1, Math.min(100, Number(e.target.value) || 1)))
              }
              className={styles.input}
            />
          </label>
          <label className={styles.fieldLabel}>
            <span>Max depth per tree</span>
            <input
              type="number"
              min={1}
              max={8}
              value={maxDepth}
              onChange={(e) =>
                setMaxDepth(Math.max(1, Math.min(8, Number(e.target.value) || 1)))
              }
              className={styles.input}
            />
          </label>
          <label className={styles.fieldLabel}>
            <span>Seed (boosting, optional)</span>
            <input
              type="text"
              className={styles.input}
              value={boostSeed}
              onChange={(e) => setBoostSeed(e.target.value)}
              placeholder="e.g. 7"
            />
          </label>
          <button
            type="button"
            className={styles.runBtn}
            onClick={trainModel}
            disabled={training.length === 0}
          >
            Train ensemble
          </button>
        </div>
        {model.trees.length > 0 && (
          <p className={styles.hint}>
            Trained {model.trees.length} trees. Weights{' '}
            <span className={styles.mono}>α_m</span> are learned from the weighted errors of each
            tree (AdaBoost-style).
          </p>
        )}
      </div>

      <div className={styles.editorBlock}>
        <h3 className={styles.optionsTitle}>Visualization</h3>
        <div className={styles.theoreticalForm}>
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
            disabled={training.length === 0 || !model.trees.length}
          >
            Predict
          </button>
          {predictedLabel !== null && (
            <span className={styles.introText} style={{ alignSelf: 'center' }}>
              Predicted:{' '}
              <strong style={{ color: getColor(predictedLabel) }}>{predictedLabel}</strong>
            </span>
          )}
        </div>
      </div>

      {training.length > 0 && (
        <div className={styles.graphBlock}>
          <h3 className={styles.graphTitle}>
            Training data {showDecisionBoundary && model.trees.length > 0 && 'and boosted boundary'}
          </h3>
          <ResponsiveContainer width="100%" height={420}>
            <ScatterChart margin={{ top: 16, right: 16, left: 16, bottom: 16 }} data={training}>
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
                    ? `(${payload[0].payload?.x?.toFixed(3)}, ${payload[0].payload?.y?.toFixed(3)})`
                    : ''
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
        </div>
      )}

      {model.trees.length > 0 && (
        <div className={styles.graphBlock}>
          <h3 className={styles.graphTitle}>Example weak learner (last tree)</h3>
          <TreeVisualization tree={model.trees[model.trees.length - 1]} getColor={getColor} />
        </div>
      )}
    </div>
  )
}

