
import styles from './Header.module.css'

function Introduction() {
  return (
    <div className={styles.introduction}>
      <div className={styles.container}>
        <div className={styles.content}>
          <div className={styles.contentContainer}>
            <p className={styles.title}>Báo cáo khóa luận</p>
            <p className={styles.des}>I developed this project while participating in the AI-Challenge competition in 2023, where I utilized cutting-edge AI technologies, including Zero-Shot learning with CLIP model and cosine similarity calculation, to retrieve event-specific videos from visual data based on Textual KIS queries.</p>
          </div>
        </div>
      </div>
    </div>
  )
}


export default Introduction
