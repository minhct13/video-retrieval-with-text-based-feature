import { useState } from 'react'
import { useSelector, useDispatch } from 'react-redux'
import icon from '../../assets/arrowUp.svg'
import styles from './SearchBar.module.css'
import { setKeySearch } from '../../Redux/slices/QueryVideoSlice'
import { getVideoAction } from '../../Redux/Actions/QueryVideoActions'
import ModelSelect from '../ModelSelect/ModelSelect'

function SearchBar() {
  const dispatch = useDispatch()
  const { keySearch } = useSelector((state) => state.queryVideoSlice)
  const [isOpenSelect, setOpenSelect] = useState(false)
  const onChangeKeySearch = (e) => {
    dispatch(setKeySearch(e.target.value))
  }

  const handleQuery = () => {
    dispatch(getVideoAction({
      query: keySearch
    }))
  }

  const handleKeyDown = (event) => {
    if (event.key === 'Enter') handleQuery()
  }
  
  return (
    <div className={styles.searchBar}>
      <ModelSelect 
      isOpen = {isOpenSelect}
      setOpenSelect = {setOpenSelect}
      />
      <div className={styles.container}>
        <input
          className={styles.inputQuerry}
          placeholder='Search'
          value={keySearch}
          onChange={onChangeKeySearch}
          onKeyDown={handleKeyDown}
        />
        <div className={styles.queryBtn} onClick={handleQuery}>
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
