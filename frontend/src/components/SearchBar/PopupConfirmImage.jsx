/* eslint-disable react/prop-types */
import { useState, useEffect } from 'react';
import { Modal } from 'react-responsive-modal'
import { useDispatch } from 'react-redux'
import { MdClose } from 'react-icons/md'
import styles from './SearchBar.module.css'
// import { sendImg } from '../../Redux/Actions/QueryVideoActions'
import { getVideoAction } from '../../Redux/Actions/QueryVideoActions'

const PopupConfirmImage = (props) => {
    const { isOpen, setOpen, file, setFile } = props
    const dispatch = useDispatch()
    const [query, setQuery] = useState('')

    useEffect(() => {
        if (!isOpen) {
            setQuery('')
            setFile(null)
        }
    }, [isOpen])

    const handleClose = () => {
        setOpen(false)
        setFile(null)
    }

    const onChangeQuery = (e) => {
        setQuery(e.target.value)
    }

    const onSend = () => {
        const formData = new FormData()
        formData.append('query', query)
        formData.append('image', file)
        dispatch(getVideoAction({
            query: "",
            formData
        }))
        setOpen(false)
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
            <div>
                <input
                    className={styles.imgSearch}
                    placeholder='query'
                    alt='image query'
                    value={query}
                    onChange={onChangeQuery}
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
