import { useState } from 'react'
import { TitlePage } from '@/components/TitlePage'
import { StochasticPdeSection } from '@/components/StochasticPdeSection'
import { MarkovChainSection } from '@/components/MarkovChainSection'
import { CtmcSection } from '@/components/CtmcSection'
import { BanditSection } from '@/components/BanditSection'
import { LLNSection } from '@/components/LLNSection'
import { CLTSection } from '@/components/CLTSection'
import { RLSection } from '@/components/RLSection'
import { LinearRegressionSection } from '@/components/LinearRegressionSection'
import { LogisticRegressionSection } from '@/components/LogisticRegressionSection'
import { KMeansSection } from '@/components/KMeansSection'
import { DBSCANSection } from '@/components/DBSCANSection'
import { PendulumSection } from '@/components/PendulumSection'
import { HeatEquationSection } from '@/components/HeatEquationSection'
import { HeatEquation1dSection } from '@/components/HeatEquation1dSection'
import { HeatEquation3dSection } from '@/components/HeatEquation3dSection'
import { KNNSection } from '@/components/KNNSection'
import { DecisionTreeSection } from '@/components/DecisionTreeSection'
import { BaggingSection } from '@/components/BaggingSection'
import { BoostingSection } from '@/components/BoostingSection'
import { PCASection } from '@/components/PCASection'
import { ConcentrationInequalitiesSection } from '@/components/ConcentrationInequalitiesSection'
import { SimplexSection } from '@/components/SimplexSection'
import { PerceptronSection } from '@/components/PerceptronSection'
import { QPSection } from '@/components/QPSection'
import { SVMSection } from '@/components/SVMSection'
import { MatrixFactorizationsSection } from '@/components/MatrixFactorizationsSection'
import { EigenvaluesSection } from '@/components/EigenvaluesSection'
import { SolveLinearSection } from '@/components/SolveLinearSection'
import styles from './App.module.css'

function getInitialTheme(): 'light' | 'dark' {
  return document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light'
}

export type AppPage =
  | 'home'
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
  | 'eigenvalues'
  | 'solve-ax-b'

export default function App() {
  const [page, setPage] = useState<AppPage>('home')
  const [theme, setTheme] = useState<'light' | 'dark'>(getInitialTheme)

  function toggleTheme() {
    const next = theme === 'dark' ? 'light' : 'dark'
    setTheme(next)
    if (next === 'dark') {
      document.documentElement.setAttribute('data-theme', 'dark')
    } else {
      document.documentElement.removeAttribute('data-theme')
    }
    localStorage.setItem('theme', next)
  }

  return (
    <div className={styles.app}>
      {page === 'home' ? (
        <>
          <button
            type="button"
            className={styles.themeToggleFixed}
            onClick={toggleTheme}
            aria-label="Toggle dark mode"
          >
            {theme === 'dark' ? '☀' : '🌙'}
          </button>
          <main className={styles.mainHome}>
            <TitlePage onSelect={setPage} />
          </main>
        </>
      ) : (
        <>
          <header className={styles.headerMinimal}>
            <button
              type="button"
              className={styles.titleLink}
              onClick={() => setPage('home')}
              aria-label="Home"
            >
              <img src="/logo.png" alt="dX" className={styles.headerLogo} />
            </button>
            <button
              type="button"
              className={styles.themeToggle}
              onClick={toggleTheme}
              aria-label="Toggle dark mode"
            >
              {theme === 'dark' ? '☀' : '🌙'}
            </button>
          </header>
          <main className={styles.main}>
            {page === 'stochastic-pde' && <StochasticPdeSection />}
            {page === 'markov-chain' && <MarkovChainSection />}
            {page === 'ctmc' && <CtmcSection />}
            {page === 'bandit' && <BanditSection />}
            {page === 'lln' && <LLNSection />}
            {page === 'clt' && <CLTSection />}
            {page === 'rl' && <RLSection />}
            {page === 'pendulum' && <PendulumSection />}
            {page === 'linear-regression' && <LinearRegressionSection />}
            {page === 'logistic-regression' && <LogisticRegressionSection />}
            {page === 'kmeans' && <KMeansSection />}
            {page === 'dbscan' && <DBSCANSection />}
            {page === 'knn' && <KNNSection />}
            {page === 'decision-tree' && <DecisionTreeSection />}
            {page === 'bagging' && <BaggingSection />}
            {page === 'boosting' && <BoostingSection />}
            {page === 'pca' && <PCASection />}
            {page === 'concentration-inequalities' && <ConcentrationInequalitiesSection />}
            {page === 'simplex' && <SimplexSection />}
            {page === 'perceptron' && <PerceptronSection />}
            {page === 'qp' && <QPSection />}
            {page === 'svm' && <SVMSection />}
            {page === 'heat-equation' && <HeatEquationSection />}
            {page === 'heat-equation-1d' && <HeatEquation1dSection />}
            {page === 'heat-equation-3d' && <HeatEquation3dSection />}
            {page === 'matrix-factorizations' && <MatrixFactorizationsSection />}
            {page === 'eigenvalues' && <EigenvaluesSection />}
            {page === 'solve-ax-b' && <SolveLinearSection />}
          </main>
        </>
      )}

    </div>
  )
}
