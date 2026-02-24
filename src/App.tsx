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
import { SimplexSection } from '@/components/SimplexSection'
import { PerceptronSection } from '@/components/PerceptronSection'
import { QPSection } from '@/components/QPSection'
import styles from './App.module.css'

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
  | 'simplex'
  | 'perceptron'
  | 'qp'

export default function App() {
  const [page, setPage] = useState<AppPage>('home')

  return (
    <div className={styles.app}>
      {page === 'home' ? (
        <>
          <main className={styles.mainHome}>
            <TitlePage onSelect={setPage} />
          </main>
        </>
      ) : (
        <>
          <header className={styles.header}>
            <button
              type="button"
              className={styles.titleLink}
              onClick={() => setPage('home')}
              aria-label="Home"
            >
              <img src="/logo.png" alt="dX" className={styles.headerLogo} />
            </button>
            <nav className={styles.nav} aria-label="Main">
              <button
                type="button"
                className={page === 'stochastic-pde' ? styles.navBtnActive : styles.navBtn}
                onClick={() => setPage('stochastic-pde')}
              >
                Stochastic PDE
              </button>
              <button
                type="button"
                className={page === 'markov-chain' ? styles.navBtnActive : styles.navBtn}
                onClick={() => setPage('markov-chain')}
              >
                Markov Chain
              </button>
              <button
                type="button"
                className={page === 'ctmc' ? styles.navBtnActive : styles.navBtn}
                onClick={() => setPage('ctmc')}
              >
                CTMC
              </button>
              <button
                type="button"
                className={page === 'bandit' ? styles.navBtnActive : styles.navBtn}
                onClick={() => setPage('bandit')}
              >
                Bandits
              </button>
              <button
                type="button"
                className={page === 'lln' ? styles.navBtnActive : styles.navBtn}
                onClick={() => setPage('lln')}
              >
                LLN
              </button>
              <button
                type="button"
                className={page === 'clt' ? styles.navBtnActive : styles.navBtn}
                onClick={() => setPage('clt')}
              >
                CLT
              </button>
              <button
                type="button"
                className={page === 'rl' ? styles.navBtnActive : styles.navBtn}
                onClick={() => setPage('rl')}
              >
                RL
              </button>
              <button
                type="button"
                className={page === 'pendulum' ? styles.navBtnActive : styles.navBtn}
                onClick={() => setPage('pendulum')}
              >
                Pendulum
              </button>
              <button
                type="button"
                className={page === 'linear-regression' ? styles.navBtnActive : styles.navBtn}
                onClick={() => setPage('linear-regression')}
              >
                Linear regression
              </button>
              <button
                type="button"
                className={page === 'logistic-regression' ? styles.navBtnActive : styles.navBtn}
                onClick={() => setPage('logistic-regression')}
              >
                Logistic regression
              </button>
              <button
                type="button"
                className={page === 'kmeans' ? styles.navBtnActive : styles.navBtn}
                onClick={() => setPage('kmeans')}
              >
                K-Means
              </button>
              <button
                type="button"
                className={page === 'dbscan' ? styles.navBtnActive : styles.navBtn}
                onClick={() => setPage('dbscan')}
              >
                DBSCAN
              </button>
              <button
                type="button"
                className={page === 'knn' ? styles.navBtnActive : styles.navBtn}
                onClick={() => setPage('knn')}
              >
                KNN
              </button>
              <button
                type="button"
                className={page === 'decision-tree' ? styles.navBtnActive : styles.navBtn}
                onClick={() => setPage('decision-tree')}
              >
                Decision tree
              </button>
              <button
                type="button"
                className={page === 'bagging' ? styles.navBtnActive : styles.navBtn}
                onClick={() => setPage('bagging')}
              >
                Bagging
              </button>
              <button
                type="button"
                className={page === 'boosting' ? styles.navBtnActive : styles.navBtn}
                onClick={() => setPage('boosting')}
              >
                Boosting
              </button>
              <button
                type="button"
                className={page === 'simplex' ? styles.navBtnActive : styles.navBtn}
                onClick={() => setPage('simplex')}
              >
                Linear Program Solver
              </button>
              <button
                type="button"
                className={page === 'perceptron' ? styles.navBtnActive : styles.navBtn}
                onClick={() => setPage('perceptron')}
              >
                Perceptron
              </button>
              <button
                type="button"
                className={page === 'qp' ? styles.navBtnActive : styles.navBtn}
                onClick={() => setPage('qp')}
              >
                Quadratic Program
              </button>
            </nav>
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
            {page === 'simplex' && <SimplexSection />}
            {page === 'perceptron' && <PerceptronSection />}
            {page === 'qp' && <QPSection />}
          </main>
        </>
      )}

      <footer className={styles.footer}>
        Euler–Maruyama · No backend · All simulation in the browser
      </footer>
    </div>
  )
}
