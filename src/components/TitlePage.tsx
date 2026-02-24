import { useState, useMemo } from 'react'
import styles from './TitlePage.module.css'

export type SectionId =
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
  | 'pca'
  | 'concentration-inequalities'
  | 'simplex'
  | 'perceptron'
  | 'qp'
  | 'svm'
  | 'heat-equation'
  | 'heat-equation-1d'
  | 'heat-equation-3d'
  | 'matrix-factorizations'

type Lab = { id: SectionId; category: string; title: string; description: string }

const LABS: Lab[] = [
  { id: 'stochastic-pde', category: 'Stochastic processes', title: 'Stochastic PDE', description: 'SDEs, sample paths, and Fokker–Planck density' },
  { id: 'markov-chain', category: 'Probability', title: 'Markov Chain', description: 'Discrete-time chains; empirical trials and stationary distribution' },
  { id: 'ctmc', category: 'Probability', title: 'CTMC', description: 'Continuous-time chains; rate matrix and holding times' },
  { id: 'bandit', category: 'Reinforcement learning', title: 'Multi-Armed Bandits', description: 'ε-greedy, UCB, Thompson Sampling; regret over time' },
  { id: 'lln', category: 'Probability', title: 'Law of Large Numbers', description: 'Sample mean → expectation (Bernoulli, Gaussian, Uniform)' },
  { id: 'clt', category: 'Probability', title: 'Central Limit Theorem', description: 'Distribution of sample means → normal as n grows' },
  { id: 'rl', category: 'Reinforcement learning', title: 'Markov Decision Process', description: 'Value Iteration, Q-Learning, SARSA; policies and value functions' },
  { id: 'linear-regression', category: 'Regression & classification', title: 'Linear regression', description: 'OLS fit; scatter, fitted line, R²' },
  { id: 'logistic-regression', category: 'Regression & classification', title: 'Logistic regression', description: 'Binary classification with sigmoid; fit and loss' },
  { id: 'kmeans', category: 'Clustering & dimensionality', title: 'K-Means', description: 'Cluster 2D points; choose k, view centroids' },
  { id: 'dbscan', category: 'Clustering & dimensionality', title: 'DBSCAN', description: 'Density-based clustering; eps, minPts; no k needed' },
  { id: 'knn', category: 'Regression & classification', title: 'K-Nearest Neighbors', description: 'Classify by majority vote of k nearest neighbors' },
  { id: 'decision-tree', category: 'Regression & classification', title: 'Decision tree', description: 'CART (Gini); tree and decision boundary' },
  { id: 'bagging', category: 'Regression & classification', title: 'Bagging (trees)', description: 'Bootstrap ensemble of trees; compare to single tree' },
  { id: 'boosting', category: 'Regression & classification', title: 'Boosting (trees)', description: 'Sequential trees on errors; AdaBoost-style boundary' },
  { id: 'perceptron', category: 'Regression & classification', title: 'Perceptron', description: 'Linear classifier ŷ = sign(w·x+b); 2D boundary' },
  { id: 'svm', category: 'Regression & classification', title: 'SVM', description: 'Max-margin linear classifier; hard & soft margin; support vectors' },
  { id: 'pca', category: 'Clustering & dimensionality', title: 'PCA', description: 'ℝⁿ→ℝ² via eigenvectors; projection and reconstruction' },
  { id: 'concentration-inequalities', category: 'Probability', title: 'Concentration Inequalities', description: 'Markov, Chebyshev, Hoeffding; bounds vs empirical tails' },
  { id: 'simplex', category: 'Optimization', title: 'Linear Program Solver', description: 'Min cᵀx s.t. Ax≤b; Big-M; 2D feasible polytope' },
  { id: 'qp', category: 'Optimization', title: 'Quadratic Program Solver', description: 'Min ½xᵀQx+cᵀx s.t. Ax≤b; active-set; 2D region' },
  { id: 'matrix-factorizations', category: 'Math', title: 'Matrix factorizations', description: 'LU, QR, Cholesky, SVD; factors and solve Ax = b' },
  { id: 'pendulum', category: 'Physics', title: 'Pendulum', description: 'Phase portrait, time series, and animation' },
  { id: 'heat-equation-1d', category: 'Physics', title: '1D Heat Equation', description: '∂u/∂t = α∂²u/∂x²; graph u(x) over time' },
  { id: 'heat-equation', category: 'Physics', title: '2D Heat Equation', description: '∂u/∂t = α∇²u; finite-difference solver and heat map' },
  { id: 'heat-equation-3d', category: 'Physics', title: '3D Heat Equation', description: '∂u/∂t = α∇²u; 3D view, rotate and zoom' },
]

type Props = {
  onSelect: (section: SectionId) => void
}

function matchQuery(lab: Lab, q: string): boolean {
  if (!q.trim()) return true
  const lower = q.toLowerCase().trim()
  const text = `${lab.category} ${lab.title} ${lab.description}`.toLowerCase()
  return text.includes(lower)
}

export function TitlePage({ onSelect }: Props) {
  const [query, setQuery] = useState('')
  const [categoryOpen, setCategoryOpen] = useState<Record<string, boolean>>(() =>
    Object.fromEntries([...new Set(LABS.map((l) => l.category))].map((c) => [c, true]))
  )

  const byCategory = useMemo(() => {
    const filtered = query.trim() ? LABS.filter((l) => matchQuery(l, query)) : LABS
    const map = new Map<string, Lab[]>()
    for (const lab of filtered) {
      const list = map.get(lab.category) ?? []
      list.push(lab)
      map.set(lab.category, list)
    }
    const order = ['Math', 'Probability', 'Stochastic processes', 'Regression & classification', 'Clustering & dimensionality', 'Reinforcement learning', 'Optimization', 'Physics']
    return order.filter((c) => map.has(c)).map((c) => ({ category: c, labs: map.get(c)! }))
  }, [query])

  const toggleCategory = (category: string) => {
    setCategoryOpen((prev) => ({ ...prev, [category]: !prev[category] }))
  }

  return (
    <div className={styles.page}>
      <span className={styles.logoWrap}>
        <img src="/logo.png" alt="dX logo" className={styles.logo} />
      </span>
      <h1 className={styles.title}>Math & ML Lab</h1>
      <div className={styles.searchWrap}>
        <input
          type="search"
          className={styles.search}
          placeholder="Search labs…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          aria-label="Search labs"
        />
      </div>
      <div className={styles.choices}>
        {byCategory.map(({ category, labs }) => (
          <section key={category} className={styles.category}>
            <button
              type="button"
              className={styles.categoryHeader}
              onClick={() => toggleCategory(category)}
              aria-expanded={categoryOpen[category]}
            >
              <h2 className={styles.categoryTitle}>{category}</h2>
              <span className={styles.categoryChevron} aria-hidden>
                {categoryOpen[category] ? '▼' : '▶'}
              </span>
            </button>
            {categoryOpen[category] && (
              <div className={styles.categoryCards}>
                {labs.map((lab) => (
                  <button
                    key={lab.id}
                    type="button"
                    className={styles.card}
                    onClick={() => onSelect(lab.id)}
                  >
                    <span className={styles.cardTitle}>{lab.title}</span>
                    <span className={styles.cardDesc}>{lab.description}</span>
                  </button>
                ))}
              </div>
            )}
          </section>
        ))}
      </div>
      {byCategory.length === 0 && (
        <p className={styles.noResults}>No labs match “{query}”. Try a different search.</p>
      )}
    </div>
  )
}
