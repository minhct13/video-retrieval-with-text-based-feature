import SuggestionItem from "./SuggestionItem"
import styles from './Suggestion.module.css'

function ListSuggestion() {
    const fakeData = [
        {
            id: 1,
            name: "SUGGESTION | English meaning - Cambridge Dictionary"
        },
        {
            id: 2,
            name: "Gợi ý 2222 2222"
        },
        {
            id: 3,
            name: "Gợi ý 33333 3333"
        },
        {
            id: 4,
            name: "Gợi ý 4444 4444"
        },
    ]
    return (
        <div className={styles.listSugest}>
            {
                fakeData.map((el, index) => (
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
