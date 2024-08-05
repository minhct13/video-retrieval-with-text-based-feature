/* eslint-disable react/prop-types */
import VideoItem from '../VideoItem/VideoItem'
import styles from './ListVideo.module.css'

function ListVideo(props) {
    const { videos } = props
    return (
        <div className={styles.listVideo}>
            <div className={styles.container}>
                {
                    videos.map((el, index) => (
                        <div className={styles.item} key={index}>
                            <VideoItem
                                link={el.video_name}
                                title = {el.video_name}
                            />
                        </div>
                    ))
                }
            </div>
        </div>
    )
}

export default ListVideo
