import Introduction from '../components/Introduction/Introduction'
import ListVideo from '../components/ListVideo/ListVideo'
import styles from './Home.module.css'

function Home() {
    const fakeData = [
        {
            id: 1,
            link: "https://www.youtube.com/watch?v=oUFJJNQGwhk'",
            title: "video 1"
        },
        {
            id: 2,
            link: "https://www.youtube.com/watch?v=oUFJJNQGwhk'",
            title: "video 2"
        },
        {
            id: 3,
            link: "https://www.youtube.com/watch?v=oUFJJNQGwhk'",
            title: "video 3"
        },
        {
            id: 4,
            link: "https://www.youtube.com/watch?v=oUFJJNQGwhk'",
            title: "video 4"
        },
        {
            id:5,
            link: "https://www.youtube.com/watch?v=oUFJJNQGwhk'",
            title: "video 5"
        },
        {
            id: 6,
            link: "https://www.youtube.com/watch?v=oUFJJNQGwhk'",
            title: "video 6"
        },
        {
            id: 7,
            link: "https://www.youtube.com/watch?v=oUFJJNQGwhk'",
            title: "video 7"
        },
    ]
    return (
        <div className={styles.home}>
            <Introduction />
            <ListVideo
                videos={fakeData}
            />
        </div>
    )
}

export default Home
