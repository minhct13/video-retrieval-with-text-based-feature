/* eslint-disable react-hooks/exhaustive-deps */
/* eslint-disable react/prop-types */
import { useEffect, useState } from 'react';
import { Modal } from 'react-responsive-modal'
import { useDispatch, useSelector } from 'react-redux'
import { MdClose } from 'react-icons/md'
import styles from './SearchBar.module.css'
// import { sendImg } from '../../Redux/Actions/QueryVideoActions'
import { getVideoAction } from '../../Redux/Actions/QueryVideoActions'
import { setQueryImg, setKeySearch, setFile, setQuery } from '../../Redux/slices/QueryVideoSlice'
import { setQueryImg, setKeySearch, setFile, setQuery } from '../../Redux/slices/QueryVideoSlice'

const PopupConfirmImage = (props) => {
    const { isOpen, setOpen, fileState, setFileState } = props
    const dispatch = useDispatch()
    const { file } = useSelector((state) => state.queryVideoSlice)
    const [query, setQueryState] = useState('')
    const [query, setQueryState] = useState('')
    useEffect(() => {
        if (!isOpen) {
            setQueryState('')
            setQueryState('')
            setFileState(null)
        }
    }, [isOpen])

    const handleClose = () => {
        setOpen(false)
        setQueryState('')
        setQueryState('')
        setFileState(null)
    }

    const onChangeQuery = (e) => {
        setQueryState(e.target.value)
        setQueryState(e.target.value)
    }

    const onSend = () => {
        const formData = new FormData()
        formData.append('query', query)
        formData.append('image', file)
        dispatch(setKeySearch(''))
        dispatch(setQuery(''))
        dispatch(setQuery(''))
        dispatch(setQueryImg(query))
        dispatch(setFile(fileState))
        dispatch(getVideoAction({
            query: query,
            formData:formData,
            isSetQueryImage:true
            formData:formData,
            isSetQueryImage:true
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
                    src={fileState ? URL.createObjectURL(fileState) : null}
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
