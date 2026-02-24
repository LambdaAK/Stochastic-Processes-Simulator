import { useState, useMemo } from 'react'
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
  Cell,
} from 'recharts'
import { runPCA, generatePCADataset, type DataPreset } from '@/lib/pca'
import { createSeededRng } from '@/lib/random'
import styles from './MarkovChainSection.module.css'

function tex(latex: string, displayMode = false): string {
  try {
    return katex.renderToString(latex, { displayMode, throwOnError: false })
  } catch {
    return latex
  }
}

const PRESETS: { id: DataPreset; label: string; desc: string }[] = [
  {
    id: 'correlated',
    label: 'Correlated',
    desc: 'Strong first principal component; all dims correlated',
  },
  {
    id: 'blob',
    label: 'Isotropic blob',
    desc: 'Spherical Gaussian — all PCs roughly equal',
  },
  {
    id: 'diagonal',
    label: 'Diagonal (decreasing σ)',
    desc: 'Each dimension has decreasing variance',
  },
  {
    id: 'random',
    label: 'Uniform random',
    desc: 'Unstructured — low variance explained by top PCs',
  },
]

const N_POINTS = 120

function matFmt(v: number): string {
  return v.toFixed(3)
}

function parseCustomData(text: string, n: number): number[][] | string {
  const lines = text
    .trim()
    .split(/\n/)
    .map((l) => l.trim())
    .filter(Boolean)
  if (lines.length < 3) return 'Need at least 3 data rows.'
  const rows: number[][] = []
  for (let i = 0; i < lines.length; i++) {
    const parts = lines[i]
      .split(/[\s,;]+/)
      .map((s) => parseFloat(s))
      .filter((v) => Number.isFinite(v))
    if (parts.length !== n)
      return `Row ${i + 1}: expected ${n} numbers, got ${parts.length}.`
    rows.push(parts)
  }
  return rows
}

export function PCASection() {
  const [dim, setDim] = useState(3)
  const [preset, setPreset] = useState<DataPreset>('correlated')
  const [seed, setSeed] = useState('')
  const [dataText, setDataText] = useState('')
  const [useCustom, setUseCustom] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [data, setData] = useState<number[][] | null>(null)
  const [selectedIdx, setSelectedIdx] = useState(0)

  const result = useMemo(() => {
    if (!data || data.length < 3) return null
    try {
      return runPCA(data, 2)
    } catch {
      return null
    }
  }, [data])

  function handleGenerate() {
    setError(null)
    setUseCustom(false)
    const rand =
      seed.trim() !== '' && !Number.isNaN(Number(seed))
        ? createSeededRng(Number(seed))
        : Math.random
    const generated = generatePCADataset(preset, N_POINTS, dim, rand)
    setData(generated)
    // Show generated data in textarea
    setDataText(
      generated
        .map((row) => row.map((v) => v.toFixed(4)).join(', '))
        .join('\n')
    )
  }

  function handleRunCustom() {
    setError(null)
    const parsed = parseCustomData(dataText, dim)
    if (typeof parsed === 'string') {
      setError(parsed)
      return
    }
    setData(parsed)
    setUseCustom(true)
  }

  // Scatter data for 2D projection
  const projectedScatter = useMemo(() => {
    if (!result) return []
    return result.projected.map((p, i) => ({ x: p[0], y: p[1], idx: i }))
  }, [result])

  // For n==3: original and reconstructed in first 2 dims for overlay
  const overlay3d = useMemo(() => {
    if (!result || dim !== 3 || !data) return null
    const orig = data.map((p) => ({ x: p[0], y: p[1] }))
    const rec = result.reconstructed.map((p) => ({ x: p[0], y: p[1] }))
    return { orig, rec }
  }, [result, dim, data])


  return (
    <div className={styles.section}>
      {/* Intro */}
      <div className={styles.intro}>
        <p className={styles.introText}>
          <strong>Principal Component Analysis (PCA)</strong> finds the
          directions of maximum variance in high-dimensional data. Given{' '}
          <span dangerouslySetInnerHTML={{ __html: tex('m') }} /> points in{' '}
          <span dangerouslySetInnerHTML={{ __html: tex('\\mathbb{R}^n') }} />,
          PCA projects them onto the top{' '}
          <span dangerouslySetInnerHTML={{ __html: tex('k') }} /> eigenvectors
          of the sample covariance matrix. Here{' '}
          <span dangerouslySetInnerHTML={{ __html: tex('k=2') }} /> so you can
          visualise any{' '}
          <span dangerouslySetInnerHTML={{ __html: tex('\\mathbb{R}^n') }} />{' '}
          dataset in a 2D scatter plot, and reconstruct back to{' '}
          <span dangerouslySetInnerHTML={{ __html: tex('\\mathbb{R}^n') }} />.
        </p>
      </div>

      {/* Math block */}
      <div className={styles.matrixBlock}>
        <h3 className={styles.matrixTitle}>Algorithm</h3>
        <ol style={{ margin: '0.5rem 0 0', paddingLeft: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          <li>
            <strong>Center:</strong>{' '}
            <span
              dangerouslySetInnerHTML={{
                __html: tex('\\tilde{X} = X - \\mathbf{1}\\mu^\\top,\\quad \\mu_j = \\tfrac{1}{m}\\sum_i x_{ij}'),
              }}
            />
          </li>
          <li>
            <strong>Covariance:</strong>{' '}
            <span
              dangerouslySetInnerHTML={{
                __html: tex(
                  '\\Sigma = \\dfrac{1}{m-1}\\,\\tilde{X}^\\top \\tilde{X} \\in \\mathbb{R}^{n \\times n}'
                ),
              }}
            />
          </li>
          <li>
            <strong>Eigendecompose:</strong>{' '}
            <span
              dangerouslySetInnerHTML={{
                __html: tex('\\Sigma = V \\Lambda V^\\top,\\quad \\lambda_1 \\ge \\lambda_2 \\ge \\cdots'),
              }}
            />
          </li>
          <li>
            <strong>Project:</strong>{' '}
            <span
              dangerouslySetInnerHTML={{
                __html: tex(
                  'Z = \\tilde{X} V_k \\in \\mathbb{R}^{m \\times k},\\quad V_k = [v_1,\\,v_2]'
                ),
              }}
            />
          </li>
          <li>
            <strong>Reconstruct:</strong>{' '}
            <span
              dangerouslySetInnerHTML={{
                __html: tex('\\hat{X} = Z\\,V_k^\\top + \\mathbf{1}\\mu^\\top'),
              }}
            />
          </li>
        </ol>
      </div>

      {/* Controls */}
      <div className={styles.editorBlock}>
        <h3 className={styles.optionsTitle}>Data</h3>
        <div className={styles.theoreticalForm}>
          <label className={styles.fieldLabel}>
            <span>Dimension n</span>
            <input
              type="number"
              min={2}
              max={10}
              value={dim}
              onChange={(e) => {
                const v = Math.max(2, Math.min(10, Number(e.target.value) || 2))
                setDim(v)
                setData(null)
                setError(null)
              }}
              className={styles.input}
              style={{ width: 80 }}
            />
          </label>
          <label className={styles.fieldLabel}>
            <span>Preset</span>
            <select
              className={styles.input}
              value={preset}
              onChange={(e) => setPreset(e.target.value as DataPreset)}
            >
              {PRESETS.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.label}
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
              style={{ width: 100 }}
            />
          </label>
          <button type="button" className={styles.runBtn} onClick={handleGenerate}>
            Generate & run PCA
          </button>
        </div>
        {PRESETS.find((p) => p.id === preset) && (
          <p className={styles.hint} style={{ marginTop: '0.25rem' }}>
            {PRESETS.find((p) => p.id === preset)!.desc}
          </p>
        )}
      </div>

      {/* Custom data */}
      <div className={styles.editorBlock}>
        <h3 className={styles.optionsTitle}>
          Custom data{' '}
          <span style={{ fontWeight: 400, fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            (or edit generated data above)
          </span>
        </h3>
        <p className={styles.hint}>
          Paste rows of {dim} comma-separated numbers, one point per line.
        </p>
        <textarea
          className={styles.textarea}
          rows={6}
          value={dataText}
          onChange={(e) => setDataText(e.target.value)}
          spellCheck={false}
          placeholder={`e.g. (${dim} values per row)\n1.2, 0.5${dim > 2 ? ', 3.1' : ''}${dim > 3 ? ', ...' : ''}\n...`}
        />
        {error && <p className={styles.error}>{error}</p>}
        <button
          type="button"
          className={styles.runBtn}
          onClick={handleRunCustom}
          disabled={dataText.trim() === ''}
        >
          Run PCA on custom data
        </button>
      </div>

      {/* Results */}
      {result && data && (
        <>
          {/* Eigenvalues / explained variance */}
          <div className={styles.matrixBlock}>
            <h3 className={styles.matrixTitle}>Results</h3>
            <p className={styles.matrixHint}>
              {data.length} points · n = {dim} · k = 2 principal components
              {useCustom ? ' (custom data)' : ''}
            </p>

            {/* Explained variance table */}
            <div style={{ marginTop: '0.75rem' }}>
              <p className={styles.hint} style={{ marginBottom: '0.4rem' }}>
                <strong>Eigenvalues &amp; explained variance:</strong>
              </p>
              <div className={styles.matrixWrap}>
                <table className={styles.matrixTable}>
                  <thead>
                    <tr>
                      <th className={styles.matrixHeader}>PC</th>
                      <th className={styles.matrixHeader}>
                        <span
                          dangerouslySetInnerHTML={{ __html: tex('\\lambda_i') }}
                        />
                      </th>
                      <th className={styles.matrixHeader}>Var. explained</th>
                      <th className={styles.matrixHeader}>Cumulative</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.allEigenvalues.map((ev, i) => {
                      const cumulative = result.explainedVarianceRatio
                        .slice(0, i + 1)
                        .reduce((a, b) => a + b, 0)
                      return (
                        <tr key={i}>
                          <td className={styles.matrixCell}>
                            PC{i + 1}
                            {i < 2 && (
                              <span
                                style={{
                                  marginLeft: 4,
                                  fontSize: '0.75rem',
                                  color: 'var(--accent)',
                                  fontWeight: 600,
                                }}
                              >
                                ★
                              </span>
                            )}
                          </td>
                          <td className={styles.matrixCell}>{ev.toFixed(4)}</td>
                          <td className={styles.matrixCell}>
                            {(result.explainedVarianceRatio[i] * 100).toFixed(1)}%
                          </td>
                          <td className={styles.matrixCell}>
                            {(cumulative * 100).toFixed(1)}%
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Top eigenvectors */}
            <div style={{ marginTop: '1rem' }}>
              <p className={styles.hint} style={{ marginBottom: '0.4rem' }}>
                <strong>Top 2 eigenvectors</strong>{' '}
                <span
                  dangerouslySetInnerHTML={{ __html: tex('v_1, v_2 \\in \\mathbb{R}^n') }}
                />{' '}
                (principal directions):
              </p>
              <div className={styles.matrixWrap}>
                <table className={styles.matrixTable}>
                  <thead>
                    <tr>
                      <th className={styles.matrixHeader}></th>
                      {result.eigenvectors[0].map((_, j) => (
                        <th key={j} className={styles.matrixHeader}>
                          dim {j + 1}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {result.eigenvectors.map((ev, i) => (
                      <tr key={i}>
                        <td className={styles.matrixRowHeader}>
                          <span
                            dangerouslySetInnerHTML={{
                              __html: tex(`v_${i + 1}`),
                            }}
                          />
                        </td>
                        {ev.map((v, j) => (
                          <td key={j} className={styles.matrixCell}>
                            {matFmt(v)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Projection walkthrough */}
            <div style={{ marginTop: '1rem' }}>
              <p className={styles.hint} style={{ marginBottom: '0.5rem' }}>
                <strong>Projection walkthrough — point by point</strong>
              </p>
              <p className={styles.hint} style={{ marginBottom: '0.6rem' }}>
                Each centered point{' '}
                <span dangerouslySetInnerHTML={{ __html: tex('\\tilde{x}_i = x_i - \\mu') }} />{' '}
                is projected by taking its dot product with each principal direction:{' '}
                <span dangerouslySetInnerHTML={{ __html: tex('z_{i,k} = v_k \\cdot \\tilde{x}_i = \\sum_{j=1}^{n} v_{kj}\\,\\tilde{x}_{ij}') }} />
              </p>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.75rem', flexWrap: 'wrap' }}>
                <label className={styles.fieldLabel} style={{ flexDirection: 'row', alignItems: 'center', gap: '0.5rem' }}>
                  <span>Point</span>
                  <input
                    type="number"
                    min={1}
                    max={data.length}
                    value={selectedIdx + 1}
                    onChange={(e) => {
                      const v = Math.max(1, Math.min(data.length, Number(e.target.value) || 1))
                      setSelectedIdx(v - 1)
                    }}
                    className={styles.input}
                    style={{ width: 72 }}
                  />
                  <span style={{ color: 'var(--text-muted)' }}>/ {data.length}</span>
                </label>
                <div style={{ display: 'flex', gap: '0.35rem' }}>
                  <button
                    type="button"
                    className={styles.runBtn}
                    style={{ padding: '0.3rem 0.7rem', fontSize: '0.85rem' }}
                    onClick={() => setSelectedIdx((i) => Math.max(0, i - 1))}
                    disabled={selectedIdx === 0}
                  >‹ prev</button>
                  <button
                    type="button"
                    className={styles.runBtn}
                    style={{ padding: '0.3rem 0.7rem', fontSize: '0.85rem' }}
                    onClick={() => setSelectedIdx((i) => Math.min(data.length - 1, i + 1))}
                    disabled={selectedIdx === data.length - 1}
                  >next ›</button>
                </div>
              </div>
              {(() => {
                const xi = data[selectedIdx]
                const xc = result.eigenvectors[0].map((_, j) => xi[j] - result.means[j])
                const z1 = result.eigenvectors[0].reduce((s, v, j) => s + v * xc[j], 0)
                const z2 = result.eigenvectors[1].reduce((s, v, j) => s + v * xc[j], 0)
                return (
                  <div className={styles.matrixWrap}>
                    <table className={styles.matrixTable}>
                      <thead>
                        <tr>
                          <th className={styles.matrixHeader}>dim j</th>
                          <th className={styles.matrixHeader}>
                            <span dangerouslySetInnerHTML={{ __html: tex('x_{ij}') }} />
                            <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}> (orig)</span>
                          </th>
                          <th className={styles.matrixHeader}>
                            <span dangerouslySetInnerHTML={{ __html: tex('\\mu_j') }} />
                          </th>
                          <th className={styles.matrixHeader}>
                            <span dangerouslySetInnerHTML={{ __html: tex('\\tilde{x}_{ij}') }} />
                            <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}> (centered)</span>
                          </th>
                          <th className={styles.matrixHeader}>
                            <span dangerouslySetInnerHTML={{ __html: tex('v_{1j}') }} />
                          </th>
                          <th className={styles.matrixHeader}>
                            <span dangerouslySetInnerHTML={{ __html: tex('v_{1j} \\cdot \\tilde{x}_{ij}') }} />
                          </th>
                          <th className={styles.matrixHeader}>
                            <span dangerouslySetInnerHTML={{ __html: tex('v_{2j}') }} />
                          </th>
                          <th className={styles.matrixHeader}>
                            <span dangerouslySetInnerHTML={{ __html: tex('v_{2j} \\cdot \\tilde{x}_{ij}') }} />
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        {xi.map((xij, j) => (
                          <tr key={j}>
                            <td className={styles.matrixCell}>{j + 1}</td>
                            <td className={styles.matrixCell}>{xij.toFixed(4)}</td>
                            <td className={styles.matrixCell}>{result.means[j].toFixed(4)}</td>
                            <td className={styles.matrixCell}>{xc[j].toFixed(4)}</td>
                            <td className={styles.matrixCell}>{matFmt(result.eigenvectors[0][j])}</td>
                            <td className={styles.matrixCell}>{(result.eigenvectors[0][j] * xc[j]).toFixed(4)}</td>
                            <td className={styles.matrixCell}>{matFmt(result.eigenvectors[1][j])}</td>
                            <td className={styles.matrixCell}>{(result.eigenvectors[1][j] * xc[j]).toFixed(4)}</td>
                          </tr>
                        ))}
                        <tr style={{ fontWeight: 600, borderTop: '2px solid var(--border)' }}>
                          <td className={styles.matrixCell} colSpan={5} style={{ textAlign: 'right', color: 'var(--text-muted)', fontWeight: 400, fontStyle: 'italic' }}>
                            sum →
                          </td>
                          <td className={styles.matrixCell} style={{ color: 'var(--accent)', fontWeight: 700 }}>
                            {z1.toFixed(4)}
                          </td>
                          <td className={styles.matrixCell} style={{ color: 'var(--text-muted)', fontWeight: 400, fontStyle: 'italic' }}>
                          </td>
                          <td className={styles.matrixCell} style={{ color: 'var(--accent)', fontWeight: 700 }}>
                            {z2.toFixed(4)}
                          </td>
                        </tr>
                      </tbody>
                    </table>
                    <p className={styles.hint} style={{ marginTop: '0.5rem' }}>
                      Point {selectedIdx + 1} maps to 2D coordinates{' '}
                      <span dangerouslySetInnerHTML={{ __html: tex(`(z_1,\\,z_2) = (${z1.toFixed(3)},\\;${z2.toFixed(3)})`) }} />
                    </p>
                  </div>
                )
              })()}
            </div>

            {/* Covariance matrix */}
            <div style={{ marginTop: '1rem' }}>
                <p className={styles.hint} style={{ marginBottom: '0.4rem' }}>
                  <strong>Covariance matrix</strong>{' '}
                  <span
                    dangerouslySetInnerHTML={{
                      __html: tex(
                        '\\Sigma \\in \\mathbb{R}^{n \\times n}'
                      ),
                    }}
                  />
                  :
                </p>
                <div className={styles.matrixWrap}>
                  <table className={styles.matrixTable}>
                    <thead>
                      <tr>
                        <th className={styles.matrixCorner}></th>
                        {result.covariance[0].map((_, j) => (
                          <th key={j} className={styles.matrixHeader}>
                            {j + 1}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {result.covariance.map((row, i) => (
                        <tr key={i}>
                          <td className={styles.matrixRowHeader}>{i + 1}</td>
                          {row.map((v, j) => (
                            <td key={j} className={styles.matrixCell}>
                              {matFmt(v)}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
            </div>

            {/* Reconstruction error */}
            <div style={{ marginTop: '1rem' }}>
              <p className={styles.hint}>
                <strong>Reconstruction MSE:</strong>{' '}
                <span
                  dangerouslySetInnerHTML={{
                    __html: tex(
                      '\\frac{1}{mn}\\|X - \\hat{X}\\|_F^2'
                    ),
                  }}
                />{' '}
                ={' '}
                <span style={{ color: 'var(--accent)', fontWeight: 600 }}>
                  {result.mse.toFixed(6)}
                </span>
              </p>
              <p className={styles.hint} style={{ marginTop: '0.25rem' }}>
                <strong>Variance captured by top 2 PCs:</strong>{' '}
                <span style={{ color: 'var(--accent)', fontWeight: 600 }}>
                  {(
                    (result.explainedVarianceRatio[0] +
                      (result.explainedVarianceRatio[1] ?? 0)) *
                    100
                  ).toFixed(1)}
                  %
                </span>
              </p>
            </div>
          </div>

          {/* 2D scatter of projected points */}
          <div className={styles.graphBlock}>
            <h3 className={styles.graphTitle}>
              2D projection onto PC1 × PC2
            </h3>
            <p className={styles.hint} style={{ marginBottom: '0.5rem' }}>
              Each point{' '}
              <span dangerouslySetInnerHTML={{ __html: tex('x_i \\in \\mathbb{R}^n') }} />{' '}
              projected as{' '}
              <span dangerouslySetInnerHTML={{ __html: tex('z_i = V_k^\\top \\tilde{x}_i') }} />
            </p>
            <ResponsiveContainer width="100%" height={360}>
              <ScatterChart margin={{ top: 16, right: 24, left: 0, bottom: 16 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis
                  type="number"
                  dataKey="x"
                  name="PC1"
                  stroke="var(--text-muted)"
                  tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                  label={{ value: 'PC 1', position: 'insideBottom', offset: -8, fill: 'var(--text-muted)', fontSize: 12 }}
                />
                <YAxis
                  type="number"
                  dataKey="y"
                  name="PC2"
                  stroke="var(--text-muted)"
                  tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                  label={{ value: 'PC 2', angle: -90, position: 'insideLeft', offset: 12, fill: 'var(--text-muted)', fontSize: 12 }}
                />
                <Tooltip
                  cursor={{ stroke: 'var(--border)' }}
                  contentStyle={{
                    background: 'var(--glass-bg)',
                    border: '1px solid var(--glass-border)',
                    borderRadius: 'var(--radius)',
                  }}
                  formatter={(value: number) => [value.toFixed(4), '']}
                  labelFormatter={(_l, payload) =>
                    payload?.[0]
                      ? `PC1=${payload[0].payload.x?.toFixed(3)}, PC2=${payload[0].payload.y?.toFixed(3)}`
                      : ''
                  }
                />
                <Scatter
                  name="Projected"
                  data={projectedScatter}
                  isAnimationActive={false}
                >
                  {projectedScatter.map((_, i) => (
                    <Cell
                      key={i}
                      fill="var(--accent)"
                      stroke="var(--text)"
                      strokeWidth={0.5}
                      fillOpacity={0.7}
                    />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>

          {/* n=3: original vs reconstructed in x–y plane */}
          {overlay3d && (
            <div className={styles.graphBlock}>
              <h3 className={styles.graphTitle}>
                Original vs. reconstructed (n = 3, x–y plane)
              </h3>
              <p className={styles.hint} style={{ marginBottom: '0.5rem' }}>
                Blue = original points, orange = reconstructed{' '}
                <span dangerouslySetInnerHTML={{ __html: tex('\\hat{x}_i') }} />{' '}
                (projected back to ℝ³), shown in the x₁–x₂ plane.
              </p>
              <ResponsiveContainer width="100%" height={320}>
                <ScatterChart margin={{ top: 16, right: 24, left: 0, bottom: 16 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis
                    type="number"
                    dataKey="x"
                    name="x₁"
                    stroke="var(--text-muted)"
                    tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                    label={{ value: 'x₁', position: 'insideBottom', offset: -8, fill: 'var(--text-muted)', fontSize: 12 }}
                  />
                  <YAxis
                    type="number"
                    dataKey="y"
                    name="x₂"
                    stroke="var(--text-muted)"
                    tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                    label={{ value: 'x₂', angle: -90, position: 'insideLeft', offset: 12, fill: 'var(--text-muted)', fontSize: 12 }}
                  />
                  <Tooltip
                    cursor={{ stroke: 'var(--border)' }}
                    contentStyle={{
                      background: 'var(--glass-bg)',
                      border: '1px solid var(--glass-border)',
                      borderRadius: 'var(--radius)',
                    }}
                    formatter={(value: number) => [value.toFixed(4), '']}
                  />
                  <Scatter
                    name="Original"
                    data={overlay3d.orig}
                    fill="#3b82f6"
                    isAnimationActive={false}
                  >
                    {overlay3d.orig.map((_, i) => (
                      <Cell
                        key={i}
                        fill="#3b82f6"
                        stroke="#1d4ed8"
                        strokeWidth={0.5}
                        fillOpacity={0.6}
                      />
                    ))}
                  </Scatter>
                  <Scatter
                    name="Reconstructed"
                    data={overlay3d.rec}
                    fill="#f97316"
                    isAnimationActive={false}
                  >
                    {overlay3d.rec.map((_, i) => (
                      <Cell
                        key={i}
                        fill="#f97316"
                        stroke="#c2410c"
                        strokeWidth={0.5}
                        fillOpacity={0.6}
                      />
                    ))}
                  </Scatter>
                  <Legend />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          )}
        </>
      )}

      {!result && !data && (
        <p className={styles.hint} style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>
          Choose a preset and dimension, then click <strong>Generate &amp; run PCA</strong>.
        </p>
      )}
    </div>
  )
}
