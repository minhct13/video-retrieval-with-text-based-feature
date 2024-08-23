import React from 'react';
import { Modal } from 'react-responsive-modal'
import { useSelector, useDispatch } from 'react-redux'
import { MdClose } from 'react-icons/md'
import styles from './SearchBar.module.css'
import { sendImg } from '../../Redux/Actions/QueryVideoActions'

const PopupConfirmImage = (props) => {
    const { isOpen, setOpen, file, setFile } = props
    const dispatch = useDispatch()
    const handleClose = () => {
        setOpen(false)
        setFile(null)
    }
    const onSend = () => {
        const payload = {

        }
        dispatch(sendImg(payload))
    }
    return (
        <Modal
            open={isOpen}
            onClose={handleClose}
            closeIcon={<MdClose className={styles.closeIcon} />}
            classNames={{
                modal: styles.customModal,
            }}
            center
        >
            <div className={styles.div_previewImg}>
                <img
                    className={styles.previewImg}
                    src={file ? URL.createObjectURL(file) : null}
                    alt='image preview'
                />
            </div>
            <div className={styles.listPreviewBtn}>
                <div className={styles.cancelBtn} onClick={handleClose}>
                    <p>Cancel</p>
                </div>
                <div className={styles.sendBtn} onClick={onSend}>
                    <p>Send</p>
                </div>
            </div>
        </Modal>
    );
};


export default PopupConfirmImage;
