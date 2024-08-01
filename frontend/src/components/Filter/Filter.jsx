import { FaArrowUp, FaArrowDown } from 'react-icons/fa'
import styles from './Filter.module.css'

function Filter() {
    return (
        <div className={styles.filter}>
            <div className={styles.container}>
                <p>Similarity</p>
                <FaArrowUp />
            </div>
        </div>
    )
}


export default Filter
