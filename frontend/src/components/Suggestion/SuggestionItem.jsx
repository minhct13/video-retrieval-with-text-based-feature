/* eslint-disable react/prop-types */
import styles from './Suggestion.module.css'

function SuggestionItem(props) {
    const { id, name } = props
    return (
        <div className={styles.item}>
            <div className={styles.itemContainer}>
                <p>{name}</p>
            </div>
        </div>
    )
}


export default SuggestionItem
