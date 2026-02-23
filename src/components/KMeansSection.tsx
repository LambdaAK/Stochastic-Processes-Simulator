import { useState, useCallback } from 'react'
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
import { kmeans } from '@/lib/kmeans'
import { generatePresetDataset } from '@/lib/knn'
import { createSeededRng } from '@/lib/random'
import styles from './MarkovChainSection.module.css'

type DataPreset = 'blobs' | 'random' | 'circles' | 'xor'

const PRESETS: { id: DataPreset; label: string }[] = [
  { id: 'blobs', label: 'Blobs (2 Gaussians)' },
  { id: 'random', label: 'Random (uniform)' },
  { id: 'circles', label: 'Circles (ring + center)' },
  { id: 'xor', label: 'XOR (4 blobs)' },
]

const N_POINTS = 150
const DOMAIN = { xMin: -4, xMax: 4, yMin: -4, yMax: 4 }

const CLUSTER_COLORS = [
  'var(--accent)',
  '#0ea5e9',
  '#22c55e',
  '#a855f7',
  '#f59e0b',
  '#ef4444',
  '#ec4899',
  '#14b8a6',
]

function getClusterColor(idx: number): string {
  return CLUSTER_COLORS[idx % CLUSTER_COLORS.length]
}

function generatePoints(
  preset: DataPreset,
  n: number,
  rand: () => number
): number[][] {
  if (preset === 'random') {
    const points: number[][] = []
    const spanX = DOMAIN.xMax - DOMAIN.xMin
    const spanY = DOMAIN.yMax - DOMAIN.yMin
    for (let i = 0; i < n; i++) {
      points.push([
        DOMAIN.xMin + rand() * spanX,
        DOMAIN.yMin + rand() * spanY,
      ])
    }
    return points
  }
  const nPerClass = Math.max(1, Math.floor(n / (preset === 'blobs' ? 2 : preset === 'xor' ? 4 : 2)))
  const training = generatePresetDataset(
    preset === 'blobs' ? 'blobs' : preset === 'xor' ? 'xor' : 'circles',
    nPerClass,
    rand
  )
  return training.map((p) => [p.x, p.y])
}

export function KMeansSection() {
  const [preset, setPreset] = useState<DataPreset>('blobs')
  const [points, setPoints] = useState<number[][]>([])
  const [k, setK] = useState(3)
  const [init, setInit] = useState<'kmeans++' | 'random'>('kmeans++')
  const [seed, setSeed] = useState('')
  const [result, setResult] = useState<{
    labels: number[]
    centroids: number[][]
    inertia: number
    nIterations: number
  } | null>(null)

  const generateData = useCallback(() => {
    const rand =
      seed.trim() !== '' && !Number.isNaN(Number(seed))
        ? createSeededRng(Number(seed))
        : Math.random
    setPoints(generatePoints(preset, N_POINTS, rand))
    setResult(null)
  }, [preset, seed])

  const runClustering = useCallback(() => {
    if (points.length === 0) return
    const rand =
      seed.trim() !== '' && !Number.isNaN(Number(seed))
        ? createSeededRng(Number(seed))
        : Math.random
    const res = kmeans({
      points,
      k: Math.min(k, points.length),
      init,
      rand,
    })
    setResult(res)
  }, [points, k, init, seed])

  const scatterDataByCluster = result
    ? (() => {
        const byCluster: { x: number; y: number; cluster: number }[][] = []
        for (let j = 0; j < (result?.centroids.length ?? 0); j++) byCluster.push([])
        points.forEach((p, i) => {
          const c = result.labels[i]
          if (byCluster[c]) byCluster[c].push({ x: p[0], y: p[1], cluster: c })
        })
        return byCluster
      })()
    : null

  const centroidData =
    result?.centroids.map((c, j) => ({ x: c[0], y: c[1], cluster: j })) ?? []

  return (
    <div className={styles.section}>
      <div className={styles.intro}>
        <p className={styles.introText}>
          <strong>K-Means</strong> clustering partitions 2D points into{' '}
          <em>k</em> clusters by minimizing within-cluster sum of squared
          distances (inertia). Choose a data preset, set <em>k</em> and
          initialization, then generate data and run clustering to see labels and
          centroids.
        </p>
      </div>

      <div className={styles.editorBlock}>
        <h3 className={styles.optionsTitle}>Data</h3>
        <div className={styles.theoreticalForm}>
          <label className={styles.fieldLabel}>
            <span>Preset</span>
            <select
              className={styles.input}
              value={preset}
              onChange={(e) => {
                setPreset(e.target.value as DataPreset)
                setPoints([])
                setResult(null)
              }}
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
        <h3 className={styles.optionsTitle}>K-Means</h3>
        <div className={styles.theoreticalForm}>
          <label className={styles.fieldLabel}>
            <span>k (clusters)</span>
            <input
              type="number"
              min={1}
              max={Math.max(1, points.length)}
              value={k}
              onChange={(e) =>
                setK(
                  Math.max(
                    1,
                    Math.min(points.length || 99, Number(e.target.value) || 1)
                  )
                )
              }
              className={styles.input}
            />
          </label>
          <label className={styles.fieldLabel}>
            <span>Init</span>
            <select
              className={styles.input}
              value={init}
              onChange={(e) =>
                setInit(e.target.value as 'kmeans++' | 'random')
              }
            >
              <option value="kmeans++">k-means++</option>
              <option value="random">Random</option>
            </select>
          </label>
          <button
            type="button"
            className={styles.runBtn}
            onClick={runClustering}
            disabled={points.length === 0}
          >
            Run K-Means
          </button>
          {result !== null && (
            <p className={styles.hint} style={{ marginTop: '0.5rem' }}>
              Inertia = {result.inertia.toFixed(2)}, iterations ={' '}
              {result.nIterations}
            </p>
          )}
        </div>
      </div>

      {points.length > 0 && (
        <div className={styles.graphBlock}>
          <h3 className={styles.graphTitle}>
            {result !== null
              ? 'Clusters and centroids'
              : 'Points (run K-Means to cluster)'}
          </h3>
          <ResponsiveContainer width="100%" height={420}>
            <ScatterChart
              margin={{ top: 16, right: 16, left: 16, bottom: 16 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="var(--border)"
              />
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
              {scatterDataByCluster?.map((clusterPoints, j) => (
                <Scatter
                  key={j}
                  name={`Cluster ${j}`}
                  data={clusterPoints}
                  fill={getClusterColor(j)}
                  shape="circle"
                  isAnimationActive={false}
                >
                  {clusterPoints.map((_, i) => (
                    <Cell
                      key={i}
                      fill={getClusterColor(j)}
                      stroke="var(--text)"
                      strokeWidth={1}
                    />
                  ))}
                </Scatter>
              ))}
              {result === null && (
                <Scatter
                  name="Points"
                  data={points.map((p) => ({ x: p[0], y: p[1] }))}
                  fill="var(--accent)"
                  shape="circle"
                  isAnimationActive={false}
                >
                  {points.map((_, i) => (
                    <Cell
                      key={i}
                      fill="var(--accent)"
                      stroke="var(--text)"
                      strokeWidth={1}
                    />
                  ))}
                </Scatter>
              )}
              {centroidData.length > 0 && (
                <Scatter
                  name="Centroids"
                  data={centroidData}
                  fill="none"
                  shape="diamond"
                  isAnimationActive={false}
                >
                  {centroidData.map((_, i) => (
                    <Cell
                      key={i}
                      stroke={getClusterColor(i)}
                      strokeWidth={3}
                      fill="var(--bg-elevated)"
                    />
                  ))}
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
