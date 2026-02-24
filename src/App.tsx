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
import { KNNSection } from '@/components/KNNSection'
import { DecisionTreeSection } from '@/components/DecisionTreeSection'
import { BaggingSection } from '@/components/BaggingSection'
import { BoostingSection } from '@/components/BoostingSection'
import { PCASection } from '@/components/PCASection'
import { SimplexSection } from '@/components/SimplexSection'
import { PerceptronSection } from '@/components/PerceptronSection'
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
  | 'simplex'
  | 'perceptron'

export default function App() {
  const [page, setPage] = useState<AppPage>('home')
  const [theme, setTheme] = useState<'light' | 'dark'>(getInitialTheme)
  const [navOpen, setNavOpen] = useState(false)

  function navigate(p: AppPage) {
    setPage(p)
    setNavOpen(false)
  }

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
          {navOpen && (
            <div
              className={styles.navOverlay}
              onClick={() => setNavOpen(false)}
              aria-hidden="true"
            />
          )}
          <header className={styles.header}>
            <button
              type="button"
              className={styles.titleLink}
              onClick={() => { setPage('home'); setNavOpen(false) }}
              aria-label="Home"
            >
              <img src="/logo.png" alt="dX" className={styles.headerLogo} />
            </button>
            <button
              type="button"
              className={styles.hamburger}
              onClick={() => setNavOpen(o => !o)}
              aria-label={navOpen ? 'Close navigation' : 'Open navigation'}
              aria-expanded={navOpen}
            >
              <span className={navOpen ? styles.barTop : ''} />
              <span className={navOpen ? styles.barMid : ''} />
              <span className={navOpen ? styles.barBot : ''} />
            </button>
            <nav className={`${styles.nav} ${navOpen ? styles.navOpen : ''}`} aria-label="Main">
              <button
                type="button"
                className={page === 'stochastic-pde' ? styles.navBtnActive : styles.navBtn}
                onClick={() => navigate('stochastic-pde')}
              >
                Stochastic PDE
              </button>
              <button
                type="button"
                className={page === 'markov-chain' ? styles.navBtnActive : styles.navBtn}
                onClick={() => navigate('markov-chain')}
              >
                Markov Chain
              </button>
              <button
                type="button"
                className={page === 'ctmc' ? styles.navBtnActive : styles.navBtn}
                onClick={() => navigate('ctmc')}
              >
                CTMC
              </button>
              <button
                type="button"
                className={page === 'bandit' ? styles.navBtnActive : styles.navBtn}
                onClick={() => navigate('bandit')}
              >
                Bandits
              </button>
              <button
                type="button"
                className={page === 'lln' ? styles.navBtnActive : styles.navBtn}
                onClick={() => navigate('lln')}
              >
                LLN
              </button>
              <button
                type="button"
                className={page === 'clt' ? styles.navBtnActive : styles.navBtn}
                onClick={() => navigate('clt')}
              >
                CLT
              </button>
              <button
                type="button"
                className={page === 'rl' ? styles.navBtnActive : styles.navBtn}
                onClick={() => navigate('rl')}
              >
                RL
              </button>
              <button
                type="button"
                className={page === 'pendulum' ? styles.navBtnActive : styles.navBtn}
                onClick={() => navigate('pendulum')}
              >
                Pendulum
              </button>
              <button
                type="button"
                className={page === 'linear-regression' ? styles.navBtnActive : styles.navBtn}
                onClick={() => navigate('linear-regression')}
              >
                Linear regression
              </button>
              <button
                type="button"
                className={page === 'logistic-regression' ? styles.navBtnActive : styles.navBtn}
                onClick={() => navigate('logistic-regression')}
              >
                Logistic regression
              </button>
              <button
                type="button"
                className={page === 'kmeans' ? styles.navBtnActive : styles.navBtn}
                onClick={() => navigate('kmeans')}
              >
                K-Means
              </button>
              <button
                type="button"
                className={page === 'dbscan' ? styles.navBtnActive : styles.navBtn}
                onClick={() => navigate('dbscan')}
              >
                DBSCAN
              </button>
              <button
                type="button"
                className={page === 'knn' ? styles.navBtnActive : styles.navBtn}
                onClick={() => navigate('knn')}
              >
                KNN
              </button>
              <button
                type="button"
                className={page === 'decision-tree' ? styles.navBtnActive : styles.navBtn}
                onClick={() => navigate('decision-tree')}
              >
                Decision tree
              </button>
              <button
                type="button"
                className={page === 'bagging' ? styles.navBtnActive : styles.navBtn}
                onClick={() => navigate('bagging')}
              >
                Bagging
              </button>
              <button
                type="button"
                className={page === 'boosting' ? styles.navBtnActive : styles.navBtn}
                onClick={() => navigate('boosting')}
              >
                Boosting
              </button>
              <button
                type="button"
                className={page === 'pca' ? styles.navBtnActive : styles.navBtn}
                onClick={() => navigate('pca')}
              >
                PCA
              </button>
              <button
                type="button"
                className={page === 'simplex' ? styles.navBtnActive : styles.navBtn}
                onClick={() => navigate('simplex')}
              >
                Linear Program Solver
              </button>
              <button
                type="button"
                className={page === 'perceptron' ? styles.navBtnActive : styles.navBtn}
                onClick={() => navigate('perceptron')}
              >
                Perceptron
              </button>
            </nav>
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
            {page === 'simplex' && <SimplexSection />}
            {page === 'perceptron' && <PerceptronSection />}
          </main>
        </>
      )}

      <footer className={styles.footer}>
        Euler–Maruyama · No backend · All simulation in the browser
      </footer>
    </div>
  )
}
