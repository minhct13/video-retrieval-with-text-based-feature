/* eslint-disable react/prop-types */
import { MdClose } from 'react-icons/md'
import 'react-responsive-modal/styles.css'
import { Modal } from 'react-responsive-modal'
import ReactPlayer from 'react-player'
import styles from './VideoItem.module.css'
import { VITE_API_URL } from '../../config'

function PopupVideo(props) {
    const { isOpen, link, onSetOpenPopup } = props
    return (
        <>
            <Modal
                open={isOpen}
                onClose={() => onSetOpenPopup(false)}
                closeIcon={<MdClose className={styles.closeIcon} />}
                classNames={{
                    modal: styles.customModal,
                }}
                center
            >
                <div className={styles.showVideo}>
                    <ReactPlayer
                        url={`${VITE_API_URL}${link}`}
                        width="100%"
                        height="100%"
                        playing={true}
                        controls={true}
                    />
                </div>
            </Modal>
        </>
    )
}

export default PopupVideo
