/* eslint-disable react-hooks/exhaustive-deps */
/* eslint-disable react/prop-types */
import { useEffect } from 'react';
import { Modal } from 'react-responsive-modal'
import { useDispatch, useSelector } from 'react-redux'
import { MdClose } from 'react-icons/md'
import styles from './SearchBar.module.css'
// import { sendImg } from '../../Redux/Actions/QueryVideoActions'
import { getVideoAction } from '../../Redux/Actions/QueryVideoActions'
import { setQueryImg, setKeySearch } from '../../Redux/slices/QueryVideoSlice'

const PopupConfirmImage = (props) => {
    const { isOpen, setOpen, file, setFile } = props
    const dispatch = useDispatch()
    const { queryImg } = useSelector((state) => state.queryVideoSlice)
    
    useEffect(() => {
        if (!isOpen) {
            dispatch(setQueryImg(''))
            setFile(null)
        }
    }, [isOpen])

    const handleClose = () => {
        setOpen(false)
        setFile(null)
    }

    const onChangeQuery = (e) => {
        dispatch(setQueryImg(e.target.value))
    }

    const onSend = () => {
        const formData = new FormData()
        formData.append('query', queryImg)
        formData.append('image', file)
        dispatch(setKeySearch(''))
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
                    value={queryImg}
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
