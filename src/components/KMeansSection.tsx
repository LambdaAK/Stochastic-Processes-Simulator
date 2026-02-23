import styles from './MarkovChainSection.module.css'

export function KMeansSection() {
  return (
    <div className={styles.section}>
      <div className={styles.intro}>
        <p className={styles.introText}>
          <strong>K-Means</strong> clustering for 2D points. The algorithm lives in <code>@/lib/kmeans</code>; this UI section is a stub.
        </p>
      </div>
    </div>
  )
}
