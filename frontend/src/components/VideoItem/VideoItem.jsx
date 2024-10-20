/* eslint-disable react/prop-types */
import { useState } from 'react';
import ReactPlayer from 'react-player'
import styles from './VideoItem.module.css'
import PopupVideo from './PopupVideo'

function VideoItem(props) {
    const { link, title, similarity } = props
    const [isOpenPopup, setIsOpenPopup] = useState(false)
    const onSetOpenPopup = (value) => setIsOpenPopup(value)
    return (
        <div className={styles.videoItem}>
            <div className={styles.videoPlayer}>
                <div className={styles.videoMedia}>
                    <ReactPlayer
                        url={link}
                        width="100%"
                        height="100%"
                        playing={false}
                        controls={true}
                        className={styles.reactPlayer}
                    />
                </div>
                <div className={styles.topLayer} onClick={() => onSetOpenPopup(true)}>
                </div>
            </div>
            <div className={styles.videoInfo}>
                <p className={styles.title}>{title}</p>
                <p className={styles.similarity}>{similarity.toFixed(2)}</p>
            </div>
            <PopupVideo
                isOpen={isOpenPopup}
                link={link}
                onSetOpenPopup={onSetOpenPopup}
            />
        </div>
    )
}

export default VideoItem
