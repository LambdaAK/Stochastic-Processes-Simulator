import styles from './TitlePage.module.css'

type Props = {
  onSelect: (section: 'stochastic-pde' | 'markov-chain') => void
}

export function TitlePage({ onSelect }: Props) {
  return (
    <div className={styles.page}>
      <span className={styles.logoWrap}>
        <img src="/logo.png" alt="dX logo" className={styles.logo} />
      </span>
      <h1 className={styles.title}>Stochastic Processes Simulator</h1>
      <p className={styles.tagline}>
        Simulate and explore stochastic differential equations and Markov chains
      </p>
      <div className={styles.choices}>
        <button
          type="button"
          className={styles.card}
          onClick={() => onSelect('stochastic-pde')}
        >
          <span className={styles.cardTitle}>Stochastic PDE</span>
          <span className={styles.cardDesc}>
            Simulate SDEs, view paths, statistics, and the Fokker–Planck density p(x, t)
          </span>
        </button>
        <button
          type="button"
          className={styles.card}
          onClick={() => onSelect('markov-chain')}
        >
          <span className={styles.cardTitle}>Markov Chain</span>
          <span className={styles.cardDesc}>
            Discrete-time Markov chains and stationary distributions (coming soon)
          </span>
        </button>
      </div>
    </div>
  )
}
