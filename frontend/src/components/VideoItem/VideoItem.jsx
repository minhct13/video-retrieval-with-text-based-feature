/* eslint-disable react/prop-types */
import ReactPlayer from 'react-player'
import styles from './VideoItem.module.css'

function VideoItem(props) {
    const { link } = props
    return (
        <div className={styles.videoItem}>
            <ReactPlayer
                url={link}
                width="100%"
                height="auto"
                playing={false}
                controls={true}
            />
        </div>
    )
}


export default VideoItem
