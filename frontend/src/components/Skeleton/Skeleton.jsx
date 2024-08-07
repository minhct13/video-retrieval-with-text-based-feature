
import styles from './Skeleton.module.css'

function Skeleton() {
  return (
   <>
    <span className={styles.skeleton}>
      &zwnj;
    </span>
    <span className={styles.skeleton}>
      &zwnj;
    </span>
    <span className={styles.skeleton}>
      &zwnj;
    </span>
    <span className={styles.skeleton}>
      &zwnj;
    </span>
    <span className={styles.skeleton}>
      &zwnj;
    </span>
 
   </>
  )
}


export default Skeleton
