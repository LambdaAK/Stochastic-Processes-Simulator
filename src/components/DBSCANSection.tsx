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
import { dbscan } from '@/lib/dbscan'
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
const NOISE_COLOR = '#64748b'

function getClusterColor(idx: number): string {
  return idx === -1 ? NOISE_COLOR : CLUSTER_COLORS[idx % CLUSTER_COLORS.length]
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
  const nPerClass = Math.max(
    1,
    Math.floor(n / (preset === 'blobs' ? 2 : preset === 'xor' ? 4 : 2))
  )
  const training = generatePresetDataset(
    preset === 'blobs' ? 'blobs' : preset === 'xor' ? 'xor' : 'circles',
    nPerClass,
    rand
  )
  return training.map((p) => [p.x, p.y])
}

export function DBSCANSection() {
  const [preset, setPreset] = useState<DataPreset>('blobs')
  const [points, setPoints] = useState<number[][]>([])
  const [eps, setEps] = useState(0.8)
  const [minPts, setMinPts] = useState(5)
  const [seed, setSeed] = useState('')
  const [result, setResult] = useState<{
    labels: number[]
    nClusters: number
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
    const res = dbscan({
      points,
      eps: Math.max(1e-9, eps),
      minPts: Math.max(1, Math.floor(minPts)),
    })
    setResult(res)
  }, [points, eps, minPts])

  const scatterDataByLabel = result
    ? (() => {
        const byLabel: { x: number; y: number; label: number }[][] = []
        const noise: { x: number; y: number; label: number }[] = []
        points.forEach((p, i) => {
          const L = result.labels[i]
          const entry = { x: p[0], y: p[1], label: L }
          if (L === -1) noise.push(entry)
          else {
            while (byLabel.length <= L) byLabel.push([])
            byLabel[L].push(entry)
          }
        })
        return { byLabel, noise }
      })()
    : null

  return (
    <div className={styles.section}>
      <div className={styles.intro}>
        <p className={styles.introText}>
          <strong>DBSCAN</strong> (Density-Based Spatial Clustering) finds
          clusters from dense regions: points within <em>eps</em> are neighbors;
          a point with at least <em>minPts</em> neighbors is a core point.
          Clusters are maximal sets of density-reachable points; points that
          don’t belong to any cluster are labeled as <strong>noise</strong>.
          No need to choose <em>k</em>—the number of clusters is determined by
          the data.
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
        <h3 className={styles.optionsTitle}>DBSCAN</h3>
        <div className={styles.theoreticalForm}>
          <label className={styles.fieldLabel}>
            <span>ε (eps)</span>
            <input
              type="number"
              min={0.01}
              step={0.1}
              value={eps}
              onChange={(e) =>
                setEps(Math.max(0.01, Number(e.target.value) || 0.01))
              }
              className={styles.input}
            />
          </label>
          <label className={styles.fieldLabel}>
            <span>minPts</span>
            <input
              type="number"
              min={1}
              max={Math.max(1, points.length)}
              value={minPts}
              onChange={(e) =>
                setMinPts(
                  Math.max(
                    1,
                    Math.min(points.length || 99, Number(e.target.value) || 1)
                  )
                )
              }
              className={styles.input}
            />
          </label>
          <button
            type="button"
            className={styles.runBtn}
            onClick={runClustering}
            disabled={points.length === 0}
          >
            Run DBSCAN
          </button>
          {result !== null && (
            <p className={styles.hint} style={{ marginTop: '0.5rem' }}>
              Clusters: {result.nClusters}, noise:{' '}
              {result.labels.filter((l) => l === -1).length}
            </p>
          )}
        </div>
      </div>

      {points.length > 0 && (
        <div className={styles.graphBlock}>
          <h3 className={styles.graphTitle}>
            {result !== null
              ? 'Clusters and noise'
              : 'Points (run DBSCAN to cluster)'}
          </h3>
          <ResponsiveContainer width="100%" height={420}>
            <ScatterChart margin={{ top: 16, right: 16, left: 16, bottom: 16 }}>
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
              {scatterDataByLabel?.byLabel.map((clusterPoints, j) => (
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
              {scatterDataByLabel?.noise.length ? (
                <Scatter
                  name="Noise"
                  data={scatterDataByLabel.noise}
                  fill={NOISE_COLOR}
                  shape="circle"
                  isAnimationActive={false}
                >
                  {scatterDataByLabel.noise.map((_, i) => (
                    <Cell
                      key={i}
                      fill={NOISE_COLOR}
                      stroke="var(--text)"
                      strokeWidth={1}
                    />
                  ))}
                </Scatter>
              ) : null}
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
              <Legend />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}
