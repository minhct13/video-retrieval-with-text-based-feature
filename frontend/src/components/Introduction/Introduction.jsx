
import styles from './Header.module.css'

function Introduction() {
  return (
    <div className={styles.introduction}>
      <div className={styles.container}>
        <div className={styles.content}>
          <div className={styles.contentContainer}>
            <p className={styles.title}>MASTER OF COMPUTER SCIENCE THESIS
              CONTENT-BASED TEXT-TO-VIDEO RETRIEVAL WITH SEMANTIC TEXT FEATURES</p>
            <p className={styles.des}>This work aims to research and develop an application that facilitates the retrieval of videos from a large video corpus using text queries. The proposed system leverages semantic text features to enhance the accuracy and relevance of the search results. By taking advantage of large language models (LLMs) such as GPT-4 to generate and interpret descriptive text features, the application can understand and interpret the context and meaning of the input text, thereby providing more precise video matches. The system is designed to handle diverse video content, making it applicable to various domains such as education, entertainment, and information retrieval. Experimental results show promising improvements in video retrieval performance compared to traditional keyframe-based methods. This thesis contributes to the field of content-based video retrieval by offering a novel solution that bridges the gap between textual information and visual content.</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Introduction
