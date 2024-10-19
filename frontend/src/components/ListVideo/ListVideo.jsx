/* eslint-disable react/prop-types */
import { useSelector } from 'react-redux'
import VideoItem from '../VideoItem/VideoItem'
import styles from './ListVideo.module.css'
import Skeleton from '../../components/Skeleton/Skeleton'

function ListVideo(props) {
    const { videos } = props
    const { isLoading, countQuery, query, queryImg, file } = useSelector((state) => state.queryVideoSlice)
    
    return (
        <div className={styles.listVideo}>
            <div>
                {
                    countQuery > 0 && <>
                        <p className={styles.resultQuery}>Results for: {query} {queryImg}</p>
                        {
                            file && <div className={styles.queryResultImg}>
                                <img
                                    className={styles.previewImg}
                                    src={file ? URL.createObjectURL(file) : null}
                                    alt='image preview'
                                />
                            </div>
                        }
                    </>
                }
            </div>

            <div className={styles.container}>
                {
                    isLoading ? (
                        <Skeleton />
                    ) :
                        <>
                            {
                                videos.length > 0 && (
                                    <>
                                        {
                                            videos.map((el, index) => (
                                                <div className={styles.item} key={index}>
                                                    <VideoItem
                                                        link={el.video_path}
                                                        title={el.video_name}
                                                        similarity={el.similarity}
                                                    />
                                                </div>
                                            ))
                                        }
                                    </>
                                )
                            }
                        </>
                }
            </div>
        </div>
    )
}

export default ListVideo
