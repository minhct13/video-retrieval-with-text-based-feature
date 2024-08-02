import { useSelector, useDispatch } from 'react-redux'
import icon from '../../assets/arrowUp.svg'
import styles from './SearchBar.module.css'
import { setKeySearch } from '../../Redux/slices/QueryVideoSlice'

function SearchBar() {
  const dispatch = useDispatch()
  const { keySearch } = useSelector((state) => state.queryVideoSlice)

  const onChangeKeySearch = (e) => {
    dispatch(setKeySearch(e.target.value))
  }

  return (
    <div className={styles.searchBar}>
      <div className={styles.container}>
        <input
          className={styles.inputQuerry}
          placeholder='Chat'
          value={keySearch}
          onChange={onChangeKeySearch}
        />
        <div className={styles.queryBtn}>
          <img
            className={styles.iconQuery}
            src={icon}
            alt="icon"
          />
        </div>
      </div>
    </div>
  )
}

export default SearchBar
