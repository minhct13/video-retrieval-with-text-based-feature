
import styles from './Header.module.css'

function Introduction() {
  return (
    <div className={styles.introduction}>
      <div className={styles.container}>
        <div className={styles.content}>
          <div className={styles.contentContainer}>
            <p className={styles.title}>Facial Vỉdeo Retrieval System</p>
            <div className={styles.info}>
              <div>
                <p className={styles.des}>The Facial Video Retrieval System is a cutting-edge solution designed to efficiently retrieve and analyze facial data from video sources. This system utilizes advanced algorithms and techniques to extract, process, and index facial information from video streams, enabling users to search and retrieve specific facial features or identities quickly and accurately.</p>
              </div>
              <div className={styles.infoStudent}>
                <div>
                  <p><span className={styles.label}>HV:</span> Trần Công Minh - CH210101011</p>
                  <p><span className={styles.label}>GVHD:</span> TS.Mai Tiến Dũng</p>
                  <p><span className={styles.label}>GVPB 1:</span> TS.Dương Việt Hằng</p>
                  <p><span className={styles.label}>GVPB 2:</span> Nguyễn Ngọc Thảo</p>
                </div>

              </div>
            </div>


          </div>
        </div>
      </div>
    </div>
  )
}

export default Introduction
