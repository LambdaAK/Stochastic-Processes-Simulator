import { useMemo, useState, useRef, useEffect } from 'react'
import type { MarkovChainDef } from '@/types/markov'
import styles from './MarkovChainGraph.module.css'

type Props = {
  chain: MarkovChainDef
  width?: number
  height?: number
}

const DEFAULT_SIZE = 420
const NODE_R = 24
const LABEL_OFFSET = 12
const ARROW_SIZE = 10
/** When both A->B and B->A exist, curve offset (perpendicular distance) as fraction of edge length. */
const CURVE_OFFSET = 0.25

/** Merge transitions with same (from, to) by summing probabilities. */
function mergeTransitions(transitions: { from: string; to: string; p: number }[]) {
  const map = new Map<string, number>()
  for (const t of transitions) {
    const key = `${t.from}\t${t.to}`
    map.set(key, (map.get(key) ?? 0) + t.p)
  }
  return Array.from(map.entries()).map(([key, p]) => {
    const [from, to] = key.split('\t')
    return { from, to, p }
  })
}

function getInitialLayout(states: string[], width: number, height: number): Map<string, { x: number; y: number }> {
  const n = states.length
  const cx = width / 2
  const cy = height / 2
  const radius = Math.min(width, height) / 2 - NODE_R - LABEL_OFFSET - 20
  const angleStep = (2 * Math.PI) / Math.max(n, 1)
  const startAngle = -Math.PI / 2
  const nodePos = new Map<string, { x: number; y: number }>()
  states.forEach((name, i) => {
    const angle = startAngle + i * angleStep
    nodePos.set(name, {
      x: cx + radius * Math.cos(angle),
      y: cy + radius * Math.sin(angle),
    })
  })
  return nodePos
}

export function MarkovChainGraph({ chain, width = DEFAULT_SIZE, height = DEFAULT_SIZE }: Props) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [positions, setPositions] = useState<Record<string, { x: number; y: number }> | null>(null)
  const [dragging, setDragging] = useState<{
    node: string
    startPos: { x: number; y: number }
    startPointer: { x: number; y: number }
  } | null>(null)

  const statesKey = chain.states.join(',')
  const initialLayout = useMemo(
    () => getInitialLayout(chain.states, width, height),
    [statesKey, width, height]
  )

  useEffect(() => {
    setPositions(null)
  }, [statesKey])

  useEffect(() => {
    if (!dragging || !svgRef.current) return
    const svg = svgRef.current
    const getSvgPoint = (evt: PointerEvent) => {
      const pt = new DOMPoint(evt.clientX, evt.clientY)
      const ctm = svg.getScreenCTM()
      return ctm ? pt.matrixTransform(ctm.inverse()) : null
    }
    const onMove = (evt: PointerEvent) => {
      evt.preventDefault()
      const pt = getSvgPoint(evt)
      if (!pt) return
      const newX = dragging.startPos.x + (pt.x - dragging.startPointer.x)
      const newY = dragging.startPos.y + (pt.y - dragging.startPointer.y)
      setPositions((prev) => {
        const base = prev ?? Object.fromEntries(initialLayout.entries())
        return { ...base, [dragging.node]: { x: newX, y: newY } }
      })
    }
    const onUp = () => setDragging(null)
    window.addEventListener('pointermove', onMove, { capture: true })
    window.addEventListener('pointerup', onUp, { capture: true })
    window.addEventListener('pointercancel', onUp, { capture: true })
    return () => {
      window.removeEventListener('pointermove', onMove, { capture: true })
      window.removeEventListener('pointerup', onUp, { capture: true })
      window.removeEventListener('pointercancel', onUp, { capture: true })
    }
  }, [dragging, initialLayout])

  const nodePos = useMemo(() => {
    const map = new Map<string, { x: number; y: number }>()
    chain.states.forEach((name) => {
      const p = positions?.[name] ?? initialLayout.get(name)
      if (p) map.set(name, p)
    })
    return map
  }, [chain.states, positions, initialLayout])

  const { nodes, edges, selfLoops } = useMemo(() => {
    const { states, transitions } = chain
    const merged = mergeTransitions(transitions)
    const edgeKeys = new Set(merged.map((t) => `${t.from}\t${t.to}`))
    const hasReverse = (from: string, to: string) => from !== to && edgeKeys.has(`${to}\t${from}`)

    type EdgeItem = {
      from: string
      to: string
      p: number
      x1: number
      y1: number
      x2: number
      y2: number
      curve: number
      cpx?: number
      cpy?: number
      labelX?: number
      labelY?: number
    }
    const edgesList: EdgeItem[] = []
    const selfLoopsList: { state: string; p: number; x: number; y: number }[] = []

    for (const t of merged) {
      const fromPos = nodePos.get(t.from)
      const toPos = nodePos.get(t.to)
      if (!fromPos || !toPos) continue
      if (t.from === t.to) {
        selfLoopsList.push({ state: t.from, p: t.p, x: fromPos.x, y: fromPos.y })
      } else {
        const dx = toPos.x - fromPos.x
        const dy = toPos.y - fromPos.y
        const len = Math.hypot(dx, dy) || 1
        const ux = dx / len
        const uy = dy / len
        const x1 = fromPos.x + NODE_R * ux
        const y1 = fromPos.y + NODE_R * uy
        const x2 = toPos.x - (NODE_R + ARROW_SIZE) * ux
        const y2 = toPos.y - (NODE_R + ARROW_SIZE) * uy
        const midX = (x1 + x2) / 2
        const midY = (y1 + y2) / 2
        const edgeLen = Math.hypot(x2 - x1, y2 - y1)
        const bidirectional = hasReverse(t.from, t.to)
        let cpx: number | undefined
        let cpy: number | undefined
        if (bidirectional) {
          const low = t.from < t.to ? t.from : t.to
          const high = t.from < t.to ? t.to : t.from
          const lowPos = nodePos.get(low)!
          const highPos = nodePos.get(high)!
          const segDx = highPos.x - lowPos.x
          const segDy = highPos.y - lowPos.y
          const segLen = Math.hypot(segDx, segDy) || 1
          const nx = -segDy / segLen
          const ny = segDx / segLen
          const offset = CURVE_OFFSET * edgeLen
          const side = t.from < t.to ? 1 : -1
          cpx = midX + side * nx * offset
          cpy = midY + side * ny * offset
        }
        const labelX = cpx != null ? 0.25 * x1 + 0.5 * cpx + 0.25 * x2 : midX
        const labelY = cpy != null ? 0.25 * y1 + 0.5 * cpy + 0.25 * y2 : midY
        const curve = bidirectional ? (t.from < t.to ? 1 : -1) : 0
        edgesList.push({
          from: t.from,
          to: t.to,
          p: t.p,
          x1,
          y1,
          x2,
          y2,
          curve,
          cpx,
          cpy,
          labelX,
          labelY,
        })
      }
    }

    return {
      nodes: states.map((name) => ({ name, ...nodePos.get(name)! })),
      edges: edgesList,
      selfLoops: selfLoopsList,
    }
  }, [chain, nodePos, width, height])

  const handlePointerDown = (nodeName: string, evt: React.PointerEvent) => {
    evt.preventDefault()
    const svg = svgRef.current
    if (!svg) return
    const pt = new DOMPoint(evt.clientX, evt.clientY)
    const ctm = svg.getScreenCTM()
    const svgPt = ctm ? pt.matrixTransform(ctm.inverse()) : null
    if (!svgPt) return
    const pos = nodePos.get(nodeName)
    if (!pos) return
    setDragging({ node: nodeName, startPos: pos, startPointer: { x: svgPt.x, y: svgPt.y } })
  }

  return (
    <div className={styles.wrapper}>
      <svg
        ref={svgRef}
        className={styles.svg}
        viewBox={`0 0 ${width} ${height}`}
        width="100%"
        height="100%"
        preserveAspectRatio="xMidYMid meet"
      >
        <defs>
          <marker
            id="arrow"
            markerWidth={ARROW_SIZE}
            markerHeight={ARROW_SIZE}
            refX={ARROW_SIZE}
            refY={ARROW_SIZE / 2}
            orient="auto"
          >
            <path
              d={`M ${ARROW_SIZE} ${ARROW_SIZE / 2} L 0 0 L 0 ${ARROW_SIZE} Z`}
              fill="var(--text-muted)"
            />
          </marker>
        </defs>
        {/* Edges */}
        {edges.map((e, i) => {
          const labelX = e.labelX ?? (e.x1 + e.x2) / 2
          const labelY = e.labelY ?? (e.y1 + e.y2) / 2
          const pathD =
            e.curve !== 0 && e.cpx != null && e.cpy != null
              ? `M ${e.x1} ${e.y1} Q ${e.cpx} ${e.cpy} ${e.x2} ${e.y2}`
              : null
          return (
            <g key={`${e.from}-${e.to}-${i}`}>
              {pathD != null ? (
                <path
                  d={pathD}
                  fill="none"
                  className={styles.edge}
                  strokeWidth={1.5}
                  markerEnd="url(#arrow)"
                />
              ) : (
                <line
                  x1={e.x1}
                  y1={e.y1}
                  x2={e.x2}
                  y2={e.y2}
                  className={styles.edge}
                  strokeWidth={1.5}
                  markerEnd="url(#arrow)"
                />
              )}
              <text x={labelX} y={labelY} className={styles.edgeLabel} textAnchor="middle" dominantBaseline="middle">
                {e.p === Math.round(e.p) ? e.p : e.p.toFixed(2)}
              </text>
            </g>
          )
        })}
        {/* Self-loops */}
        {selfLoops.map((s, i) => {
          const r = NODE_R + 14
          return (
            <g key={`loop-${s.state}-${i}`}>
              <path
                d={`M ${s.x} ${s.y - NODE_R} A ${r} ${r} 0 1 1 ${s.x + 0.01} ${s.y - NODE_R}`}
                fill="none"
                className={styles.edge}
                strokeWidth={1.5}
                markerEnd="url(#arrow)"
              />
              <text
                x={s.x}
                y={s.y - NODE_R - r - 4}
                className={styles.edgeLabel}
                textAnchor="middle"
                dominantBaseline="middle"
              >
                {s.p === Math.round(s.p) ? s.p : s.p.toFixed(2)}
              </text>
            </g>
          )
        })}
        {/* Nodes */}
        {nodes.map((node) => (
          <g key={node.name}>
            <circle
              cx={node.x}
              cy={node.y}
              r={NODE_R}
              className={styles.node}
              style={{ cursor: dragging?.node === node.name ? 'grabbing' : 'grab' }}
              onPointerDown={(e) => handlePointerDown(node.name, e)}
            />
            <text
              x={node.x}
              y={node.y}
              className={styles.nodeLabel}
              textAnchor="middle"
              dominantBaseline="middle"
              pointerEvents="none"
            >
              {node.name}
            </text>
          </g>
        ))}
      </svg>
    </div>
  )
}
