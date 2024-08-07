import { useSelector } from 'react-redux'
import SuggestionItem from "./SuggestionItem"
import styles from './Suggestion.module.css'

function ListSuggestion() {
    const { suggesstions, keySearch, videos } = useSelector((state) => state.queryVideoSlice)

    return (
        <div className={styles.listSugest}>
            {
                !keySearch && videos.length === 0 && suggesstions.map((el, index) => (
                    <div key={index} className={styles.sugestItem}>
                        <SuggestionItem
                            id={el.id}
                            name={el.name}
                        />
                    </div>
                ))
            }
        </div>
    )
}


export default ListSuggestion
