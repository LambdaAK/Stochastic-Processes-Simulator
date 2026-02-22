import {
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  ComposedChart,
  ReferenceLine,
} from 'recharts'
import type { Stats as StatsType } from '@/types/simulation'
import styles from './StatsPlot.module.css'

type Props = {
  stats: StatsType | null
  x0: number
}

export function StatsPlot({ stats, x0: _x0 }: Props) {
  if (!stats || stats.t.length === 0) {
    return (
      <div className={styles.empty}>
        Run a simulation to see mean ± 2σ.
      </div>
    )
  }

  const meanPlus2 = stats.mean.map((m, i) => m + 2 * stats.std[i])
  const meanMinus2 = stats.mean.map((m, i) => m - 2 * stats.std[i])

  const data = stats.t.map((t, i) => ({
    t,
    mean: stats.mean[i],
    meanMinus2: meanMinus2[i],
    bandHeight: meanPlus2[i] - meanMinus2[i],
  }))

  return (
    <div className={styles.wrapper}>
      <p className={styles.hint}>
        Mean and mean ± 2σ across paths
      </p>
      <ResponsiveContainer width="100%" height={320}>
        <ComposedChart
          data={data}
          margin={{ top: 8, right: 8, bottom: 8, left: 8 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis
            dataKey="t"
            type="number"
            domain={['auto', 'auto']}
            tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
            tickFormatter={(v) => Number(v).toFixed(2)}
          />
          <YAxis
            type="number"
            domain={['auto', 'auto']}
            tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
            tickFormatter={(v) => Number(v).toFixed(2)}
          />
          <Tooltip
            contentStyle={{
              background: 'var(--bg-elevated)',
              border: '1px solid var(--border)',
              borderRadius: '8px',
            }}
            labelStyle={{ color: 'var(--text)' }}
            formatter={(value: number) => [Number(value).toFixed(4), '']}
            labelFormatter={(t) => `t = ${Number(t).toFixed(4)}`}
          />
          <ReferenceLine x={stats.t[0]} stroke="var(--accent)" strokeDasharray="2 2" />
          <Area
            type="monotone"
            dataKey="meanMinus2"
            stackId="band"
            baseValue={0}
            stroke="none"
            fill="var(--bg)"
          />
          <Area
            type="monotone"
            dataKey="bandHeight"
            stackId="band"
            stroke="none"
            fill="var(--accent)"
            fillOpacity={0.18}
          />
          <Line
            type="monotone"
            dataKey="mean"
            stroke="var(--accent)"
            strokeWidth={2}
            dot={false}
            name="Mean"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
