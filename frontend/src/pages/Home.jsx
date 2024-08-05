import { useEffect } from 'react'
import { useSelector, useDispatch } from 'react-redux'
import { ToastContainer } from 'react-toastify'
import 'react-toastify/dist/ReactToastify.css'
import Introduction from '../components/Introduction/Introduction'
import ListVideo from '../components/ListVideo/ListVideo'
import SearchBar from '../components/SearchBar/SearchBar'
import ListSuggestion from '../components/Suggestion/ListSuggestion'
import styles from './Home.module.css'
import { getVideoAction, getSuggestion } from '../Redux/Actions/QueryVideoActions'

function Home() {
    const dispatch = useDispatch()
    useEffect(() => {
        dispatch(getVideoAction({
            query:''
        }))
        dispatch(getSuggestion())
    }, [])
    const { videos } = useSelector((state) => state.queryVideoSlice)
    
    return (
        <div className={styles.home}>
            <Introduction />
            <ListVideo
                videos={videos}
            />
            <ListSuggestion />
            <SearchBar />
            <ToastContainer />
        </div>
    )
}

export default Home
