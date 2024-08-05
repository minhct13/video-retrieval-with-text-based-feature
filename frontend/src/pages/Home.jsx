import { useSelector, useDispatch } from 'react-redux'
import Filter from '../components/Filter/Filter'
import Introduction from '../components/Introduction/Introduction'
import ListVideo from '../components/ListVideo/ListVideo'
import SearchBar from '../components/SearchBar/SearchBar'
import ListSuggestion from '../components/Suggestion/ListSuggestion'
import styles from './Home.module.css'

function Home() {
    const { videos } = useSelector((state) => state.queryVideoSlice)
    const fakeData = [
        {
            id: 1,
            video_name: "video1.mp4",
            video_path: "https://res.cloudinary.com/dvvi0pivw/video/upload/v1722612622/20230910_170501_llm8bh.mp4",
            similarity: 0.8
        },
        {
            id: 2,
            video_name: "video1.mp4",
            video_path: "https://res.cloudinary.com/dvvi0pivw/video/upload/v1722612622/20230910_170501_llm8bh.mp4",
            similarity: 0.8
        },
        {
            id: 3,
            video_name: "video1.mp4",
            video_path: "https://res.cloudinary.com/dvvi0pivw/video/upload/v1722612622/20230910_170501_llm8bh.mp4",
            similarity: 0.8
        },
    ]
    return (
        <div className={styles.home}>
            <Introduction />
            <ListVideo
                videos={fakeData}
            />
            <ListSuggestion />
            <SearchBar />
        </div>
    )
}

export default Home
