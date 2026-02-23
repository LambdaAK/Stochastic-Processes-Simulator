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
import { KNNSection } from '@/components/KNNSection'
import styles from './App.module.css'

export type AppPage = 'home' | 'stochastic-pde' | 'markov-chain' | 'ctmc' | 'bandit' | 'lln' | 'clt' | 'rl' | 'linear-regression' | 'logistic-regression' | 'kmeans' | 'dbscan' | 'knn'

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
            {page === 'linear-regression' && <LinearRegressionSection />}
            {page === 'logistic-regression' && <LogisticRegressionSection />}
            {page === 'kmeans' && <KMeansSection />}
            {page === 'dbscan' && <DBSCANSection />}
            {page === 'knn' && <KNNSection />}
          </main>
        </>
      )}

      <footer className={styles.footer}>
        Euler–Maruyama · No backend · All simulation in the browser
      </footer>
    </div>
  )
}
