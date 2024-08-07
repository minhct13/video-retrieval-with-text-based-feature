/* eslint-disable react/prop-types */
import { useDispatch } from 'react-redux'
import styles from './Suggestion.module.css'
import { getVideoAction } from '../../Redux/Actions/QueryVideoActions'

function SuggestionItem(props) {
    const dispatch = useDispatch()
    const { name } = props

    const handleQuery = () => {
        dispatch(getVideoAction({
            query: name
        }))
    }

    return (
        <div className={styles.item} onClick={handleQuery}>
            <div className={styles.itemContainer}>
                <p>{name}</p>
            </div>
        </div>
    )
}


export default SuggestionItem
