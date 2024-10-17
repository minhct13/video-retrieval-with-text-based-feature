/* eslint-disable react/prop-types */
import { useDispatch } from 'react-redux'
import styles from './Suggestion.module.css'
import { getVideoAction } from '../../Redux/Actions/QueryVideoActions'

function SuggestionItem(props) {
    // const { mode } = useSelector((state) => state.queryVideoSlice);
    const dispatch = useDispatch()
    const { name } = props

    const handleQuery = () => {
        const formData = new FormData()
        formData.append('query', name)
        formData.append('image', null)
        dispatch(getVideoAction({
            query: "",
            formData
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
