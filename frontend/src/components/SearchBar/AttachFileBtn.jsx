import React, { useState } from 'react'
import { GrAttachment } from "react-icons/gr"
import styles from './SearchBar.module.css'
import PopupConfirmImage from './PopupConfirmImage'

const AttachFileBtn = () => {
    const [isOpenConfirm, setIsOpenConfirm] = useState(false)
    const [file, setFile] = useState(null)
    const handleImageChange = (event) => {
        const file = event.target.files[0]
        const { target = {} } = event || {};
        target.value = "";
        if (file && file.type.includes('image')) {
            setFile(file)
            setIsOpenConfirm(true)
        }
    }
    return (
        <>
            <label htmlFor='btnAttach'>
                <GrAttachment className={styles.attachIcon} />
            </label>
            <input
                hidden
                type='file'
                id='btnAttach'
                onInput={handleImageChange}
            />
            <PopupConfirmImage
                isOpen={isOpenConfirm}
                setOpen={setIsOpenConfirm}
                file={file}
                setFile={setFile}
            />
        </>
    );
};

export default AttachFileBtn;
