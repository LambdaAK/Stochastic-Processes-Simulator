import styles from './TitlePage.module.css'

type Props = {
  onSelect: (
    section:
      | 'stochastic-pde'
      | 'markov-chain'
      | 'ctmc'
      | 'bandit'
      | 'lln'
      | 'clt'
      | 'rl'
      | 'pendulum'
      | 'linear-regression'
      | 'logistic-regression'
      | 'kmeans'
      | 'dbscan'
      | 'knn'
      | 'decision-tree'
      | 'bagging'
      | 'boosting'
  ) => void
}

export function TitlePage({ onSelect }: Props) {
  return (
    <div className={styles.page}>
      <span className={styles.logoWrap}>
        <img src="/logo.png" alt="dX logo" className={styles.logo} />
      </span>
      <h1 className={styles.title}>Math & ML Lab</h1>
      <div className={styles.choices}>
        <section className={styles.category}>
          <h2 className={styles.categoryTitle}>Math</h2>
          <div className={styles.categoryCards}>
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
                Discrete-time Markov chains with empirical trials and theoretical distributions
              </span>
            </button>
            <button
              type="button"
              className={styles.card}
              onClick={() => onSelect('ctmc')}
            >
              <span className={styles.cardTitle}>CTMC</span>
              <span className={styles.cardDesc}>
                Continuous-time Markov chains with exponential holding times and rate matrices
              </span>
            </button>
            <button
              type="button"
              className={styles.card}
              onClick={() => onSelect('bandit')}
            >
              <span className={styles.cardTitle}>Multi-Armed Bandits</span>
              <span className={styles.cardDesc}>
                Compare exploration strategies: ε-greedy, UCB, Thompson Sampling. Visualize regret over time
              </span>
            </button>
            <button
              type="button"
              className={styles.card}
              onClick={() => onSelect('lln')}
            >
              <span className={styles.cardTitle}>Law of Large Numbers</span>
              <span className={styles.cardDesc}>
                See the sample average converge to the expected value for Bernoulli, Gaussian, or Uniform
              </span>
            </button>
            <button
              type="button"
              className={styles.card}
              onClick={() => onSelect('clt')}
            >
              <span className={styles.cardTitle}>Central Limit Theorem</span>
              <span className={styles.cardDesc}>
                Histogram of sample means: see the distribution become normal as n grows
              </span>
            </button>
            <button
              type="button"
              className={styles.card}
              onClick={() => onSelect('rl')}
            >
              <span className={styles.cardTitle}>Reinforcement Learning</span>
              <span className={styles.cardDesc}>
                MDPs and grid worlds: Value Iteration, Q-Learning, SARSA. Visualize policies and learning curves
              </span>
            </button>
          </div>
        </section>
        <section className={styles.category}>
          <h2 className={styles.categoryTitle}>ML</h2>
          <div className={styles.categoryCards}>
            <button
              type="button"
              className={styles.card}
              onClick={() => onSelect('linear-regression')}
            >
              <span className={styles.cardTitle}>Linear regression</span>
              <span className={styles.cardDesc}>
                OLS fit y = β₀ + β₁x. Paste data or generate synthetic; see scatter, fitted line, R² and residual SE
              </span>
            </button>
            <button
              type="button"
              className={styles.card}
              onClick={() => onSelect('logistic-regression')}
            >
              <span className={styles.cardTitle}>Logistic regression</span>
              <span className={styles.cardDesc}>
                Binary classification: P(y=1|x) = σ(β₀ + β₁x). Paste x,y with y in {'{0,1}'} or generate synthetic; see sigmoid fit and loss
              </span>
            </button>
            <button
              type="button"
              className={styles.card}
              onClick={() => onSelect('kmeans')}
            >
              <span className={styles.cardTitle}>K-Means</span>
              <span className={styles.cardDesc}>
                Cluster 2D points with K-Means. Generate blobs or random data, choose k, view clusters and centroids
              </span>
            </button>
            <button
              type="button"
              className={styles.card}
              onClick={() => onSelect('dbscan')}
            >
              <span className={styles.cardTitle}>DBSCAN</span>
              <span className={styles.cardDesc}>
                Density-based clustering: set eps and minPts, no k needed. Finds clusters and labels outliers as noise
              </span>
            </button>
            <button
              type="button"
              className={styles.card}
              onClick={() => onSelect('knn')}
            >
              <span className={styles.cardTitle}>K-Nearest Neighbors</span>
              <span className={styles.cardDesc}>
                Classify points by majority vote among k nearest neighbors. Try blobs, XOR, and circles
              </span>
            </button>
            <button
              type="button"
              className={styles.card}
              onClick={() => onSelect('decision-tree')}
            >
              <span className={styles.cardTitle}>Decision tree</span>
              <span className={styles.cardDesc}>
                CART trees (Gini split). Add or paste x,y,label data; view the tree and decision boundary
              </span>
            </button>
            <button
              type="button"
              className={styles.card}
              onClick={() => onSelect('bagging')}
            >
              <span className={styles.cardTitle}>Bagging (trees)</span>
              <span className={styles.cardDesc}>
                Ensemble of decision trees trained on bootstrap samples. Compare the bagged decision boundary to a single tree
              </span>
            </button>
            <button
              type="button"
              className={styles.card}
              onClick={() => onSelect('boosting')}
            >
              <span className={styles.cardTitle}>Boosting (trees)</span>
              <span className={styles.cardDesc}>
                Sequential ensemble of decision trees; each tree focuses on previous errors. AdaBoost-style weighted vote and decision boundary
              </span>
            </button>
          </div>
        </section>
        <section className={styles.category}>
          <h2 className={styles.categoryTitle}>Physics</h2>
          <div className={styles.categoryCards}>
            <button
              type="button"
              className={styles.card}
              onClick={() => onSelect('pendulum')}
            >
              <span className={styles.cardTitle}>Pendulum</span>
              <span className={styles.cardDesc}>
                Simple and damped pendulum: θ″ = −(g/L) sin θ − b θ′. Phase portrait, time series, and animation
              </span>
            </button>
          </div>
        </section>
      </div>
    </div>
  )
}
