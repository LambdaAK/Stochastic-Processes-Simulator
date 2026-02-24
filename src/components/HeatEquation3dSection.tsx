import { useState, useMemo, useCallback, useRef, useEffect } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import katex from 'katex'
import 'katex/dist/katex.min.css'
import {
  solveHeatEquation3d,
  getCell3d,
  type HeatConfig3d,
  type HeatResult3d,
  type InitialCondition3d,
} from '@/lib/heatEquation3d'
import styles from './MarkovChainSection.module.css'

function renderLatex(latex: string, displayMode = false): string {
  try {
    return katex.renderToString(latex, { displayMode, throwOnError: false })
  } catch {
    return latex
  }
}

const INITIAL_OPTIONS: { id: InitialCondition3d; label: string }[] = [
  { id: 'point', label: 'Point source (center)' },
  { id: 'corner', label: 'Hot corner' },
  { id: 'half', label: 'Half space (x < 0.5)' },
  { id: 'two-spots', label: 'Two spots' },
]

function tempToRgb(t: number): [number, number, number] {
  if (!isFinite(t) || t <= 0) return [32 / 255, 32 / 255, 128 / 255]
  if (t >= 1) return [180 / 255, 0, 0]
  const r = t <= 0.5 ? 2 * t : 1
  const g = t <= 0.5 ? (100 + 155 * 2 * t) / 255 : (255 - 255 * 2 * (t - 0.5)) / 255
  const b = t <= 0.5 ? 1 : (255 - 255 * 2 * (t - 0.5)) / 255
  return [r, g, b]
}

const VIEW_SIZE = 400

function HeatScene3d({
  result,
  snapshotIndex,
  containerRef,
}: {
  result: HeatResult3d
  snapshotIndex: number
  containerRef: React.RefObject<HTMLDivElement | null>
}) {
  const sceneRef = useRef<THREE.Scene | null>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const controlsRef = useRef<OrbitControls | null>(null)
  const pointsRef = useRef<THREE.Points | null>(null)
  const frameRef = useRef<number | null>(null)

  const gridSize = result.gridSize
  const scaleMax = useMemo(() => {
    const initial = result.snapshots[0]
    let max = 0
    for (let i = 0; i < initial.length; i++) {
      const v = initial[i]
      if (isFinite(v) && v > max) max = v
    }
    return max <= 0 ? 1 : max
  }, [result])

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x0f172a)
    sceneRef.current = scene

    const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100)
    camera.position.set(1.8, 1.5, 1.8)
    camera.lookAt(0.5, 0.5, 0.5)
    cameraRef.current = camera

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    renderer.setSize(VIEW_SIZE, VIEW_SIZE)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    container.appendChild(renderer.domElement)
    rendererRef.current = renderer

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.target.set(0.5, 0.5, 0.5)
    controls.enableDamping = true
    controls.dampingFactor = 0.05
    controlsRef.current = controls

    const n = gridSize * gridSize * gridSize
    const positions = new Float32Array(n * 3)
    const colors = new Float32Array(n * 3)
    let idx = 0
    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        for (let k = 0; k < gridSize; k++) {
          positions[idx * 3] = i * result.h
          positions[idx * 3 + 1] = j * result.h
          positions[idx * 3 + 2] = k * result.h
          idx++
        }
      }
    }

    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    const material = new THREE.PointsMaterial({
      size: (result.h * 1.2),
      vertexColors: true,
      sizeAttenuation: true,
    })
    const points = new THREE.Points(geometry, material)
    scene.add(points)
    pointsRef.current = points

    function animate() {
      frameRef.current = requestAnimationFrame(animate)
      controls.update()
      renderer.render(scene, camera)
    }
    animate()

    return () => {
      if (frameRef.current != null) cancelAnimationFrame(frameRef.current)
      controls.dispose()
      renderer.dispose()
      geometry.dispose()
      material.dispose()
      if (container && renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement)
      }
      sceneRef.current = null
      cameraRef.current = null
      rendererRef.current = null
      controlsRef.current = null
      pointsRef.current = null
    }
  }, [result, containerRef, gridSize])

  useEffect(() => {
    const points = pointsRef.current
    if (!points || snapshotIndex < 0 || snapshotIndex >= result.snapshots.length) return
    const colorAttr = points.geometry.getAttribute('color') as THREE.BufferAttribute
    const colors = colorAttr.array as Float32Array
    let i = 0
    for (let xi = 0; xi < gridSize; xi++) {
      for (let xj = 0; xj < gridSize; xj++) {
        for (let xk = 0; xk < gridSize; xk++) {
          const v = getCell3d(result, snapshotIndex, xi, xj, xk)
          const t = scaleMax > 0 ? Math.min(1, Math.max(0, v / scaleMax)) : 0
          const [r, g, b] = tempToRgb(t)
          colors[i * 3] = r
          colors[i * 3 + 1] = g
          colors[i * 3 + 2] = b
          i++
        }
      }
    }
    colorAttr.needsUpdate = true
  }, [result, snapshotIndex, gridSize, scaleMax])

  return null
}

export function HeatEquation3dSection() {
  const [N, setN] = useState(14)
  const [alpha, setAlpha] = useState(0.15)
  const [dt, setDt] = useState(0.0003)
  const [T, setT] = useState(0.5)
  const [initial, setInitial] = useState<InitialCondition3d>('point')
  const [result, setResult] = useState<HeatResult3d | null>(null)
  const [snapshotIndex, setSnapshotIndex] = useState(0)
  const [playing, setPlaying] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)
  const rafRef = useRef<number | null>(null)
  const lastTickRef = useRef(0)

  const runSimulation = useCallback(() => {
    const config: HeatConfig3d = { N, alpha, dt, T, initial }
    const res = solveHeatEquation3d(config)
    setResult(res)
    setSnapshotIndex(0)
    setPlaying(false)
  }, [N, alpha, dt, T, initial])

  const numSnapshots = result?.snapshots.length ?? 0
  const currentIndex = result ? Math.max(0, Math.min(snapshotIndex, numSnapshots - 1)) : 0
  const currentTime = result?.times[currentIndex] ?? 0

  useEffect(() => {
    if (!result || !playing || numSnapshots <= 1) return
    const animate = (now: number) => {
      const elapsed = (now - lastTickRef.current) / 1000
      lastTickRef.current = now
      setSnapshotIndex((prev) => {
        const step = Math.max(1, Math.floor(elapsed * 12))
        const next = Math.min(prev + step, numSnapshots - 1)
        if (next >= numSnapshots - 1) setPlaying(false)
        return next
      })
      rafRef.current = requestAnimationFrame(animate)
    }
    lastTickRef.current = performance.now()
    rafRef.current = requestAnimationFrame(animate)
    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current)
    }
  }, [playing, result, numSnapshots])

  const stabilityHint = useMemo(() => {
    const h = 1 / (N + 1)
    const maxDt = (h * h) / (6 * alpha)
    return { maxDt, stable: dt <= maxDt * 1.01 }
  }, [N, alpha, dt])

  return (
    <div className={styles.section}>
      <div className={styles.intro}>
        <p className={styles.introText}>
          The <strong>3D heat equation</strong> is{' '}
          <span
            dangerouslySetInnerHTML={{
              __html: renderLatex('\\frac{\\partial u}{\\partial t} = \\alpha \\nabla^2 u', true),
            }}
          />
          {' '}on the unit cube with <span dangerouslySetInnerHTML={{ __html: renderLatex('u=0') }} /> on the boundary. Solved with finite differences (forward Euler + 7-point Laplacian). Drag to rotate the view; scroll to zoom.
        </p>
      </div>

      <div className={styles.editorBlock}>
        <h3 className={styles.optionsTitle}>Analytical solution</h3>
        <p className={styles.introText}>
          The solution is a 3D Fourier series; each mode <span dangerouslySetInnerHTML={{ __html: renderLatex('\\sin(m\\pi x)\\sin(n\\pi y)\\sin(p\\pi z)') }} /> decays like{' '}
          <span dangerouslySetInnerHTML={{ __html: renderLatex('e^{-\\alpha\\pi^2(m^2+n^2+p^2)t}') }} />. As{' '}
          <span dangerouslySetInnerHTML={{ __html: renderLatex('t\\to\\infty') }} />, <span dangerouslySetInnerHTML={{ __html: renderLatex('u\\to 0') }} /> everywhere.
        </p>
      </div>

      <div className={styles.editorBlock}>
        <h3 className={styles.optionsTitle}>Parameters</h3>
        <div className={styles.theoreticalForm}>
          <label className={styles.fieldLabel}>
            <span>Grid N (each side)</span>
            <input
              type="number"
              min={8}
              max={24}
              value={N}
              onChange={(e) => setN(Math.max(8, Math.min(24, Number(e.target.value) || 12)))}
              className={styles.input}
            />
          </label>
          <label className={styles.fieldLabel}>
            <span>Diffusivity α</span>
            <input
              type="number"
              min={0.05}
              step={0.05}
              value={alpha}
              onChange={(e) => setAlpha(Math.max(0.05, Number(e.target.value) || 0.15))}
              className={styles.input}
            />
          </label>
          <label className={styles.fieldLabel}>
            <span>Time step dt</span>
            <input
              type="number"
              min={0.0001}
              step={0.0001}
              value={dt}
              onChange={(e) => setDt(Math.max(0.0001, Number(e.target.value) || 0.0003))}
              className={styles.input}
            />
          </label>
          <label className={styles.fieldLabel}>
            <span>Total time T</span>
            <input
              type="number"
              min={0.1}
              step={0.1}
              value={T}
              onChange={(e) => setT(Math.max(0.1, Number(e.target.value) || 0.5))}
              className={styles.input}
            />
          </label>
          <label className={styles.fieldLabel}>
            <span>Initial condition</span>
            <select
              value={initial}
              onChange={(e) => setInitial(e.target.value as InitialCondition3d)}
              className={styles.input}
            >
              {INITIAL_OPTIONS.map((opt) => (
                <option key={opt.id} value={opt.id}>
                  {opt.label}
                </option>
              ))}
            </select>
          </label>
          <button type="button" className={styles.runBtn} onClick={runSimulation}>
            Run
          </button>
        </div>
        <p className={styles.hint}>
          Stability: dt ≤ h²/(6α) ≈ {stabilityHint.maxDt.toExponential(3)}. Grid (N+2)³; keep N ≤ 24 for performance.
          {!stabilityHint.stable && (
            <span style={{ color: 'var(--danger)' }}> Solver will use a smaller step if needed.</span>
          )}
        </p>
      </div>

      {result && (
        <div className={styles.graphBlock}>
          <h3 className={styles.graphTitle}>Temperature u(x, y, z, t) — drag to rotate, scroll to zoom</h3>
          <div
            ref={containerRef}
            style={{
              width: VIEW_SIZE,
              height: VIEW_SIZE,
              borderRadius: 'var(--radius)',
              overflow: 'hidden',
            }}
          >
            <HeatScene3d result={result} snapshotIndex={currentIndex} containerRef={containerRef} />
          </div>
          <div className={styles.heatKey} style={{ marginTop: '0.5rem' }}>
            <span className={styles.heatKeyLabel}>Cold (low u)</span>
            <div
              className={styles.heatKeyBar}
              style={{
                background: 'linear-gradient(to right, rgb(32,32,128), rgb(128,200,255), rgb(255,255,255), rgb(255,220,100), rgb(180,0,0))',
              }}
            />
            <span className={styles.heatKeyLabel}>Hot (high u)</span>
          </div>
          <div className={styles.theoreticalForm} style={{ flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center', gap: '1rem', marginTop: '0.75rem' }}>
            <label className={styles.fieldLabel}>
              <span>Time t = {currentTime.toFixed(3)}</span>
              <input
                type="range"
                min={0}
                max={numSnapshots - 1}
                value={currentIndex}
                onChange={(e) => setSnapshotIndex(Number(e.target.value))}
                className={styles.input}
                style={{ width: 220 }}
              />
            </label>
            <button type="button" className={styles.runBtn} onClick={() => setPlaying((p) => !p)}>
              {playing ? 'Pause' : 'Play'}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
