import { useState } from 'react'
// import { useDispatch } from 'react-redux'
import { toast } from 'react-toastify'
import { GrAttachment } from "react-icons/gr"
import styles from './SearchBar.module.css'
import PopupConfirmImage from './PopupConfirmImage'
// import { setFile} from '../../Redux/slices/QueryVideoSlice'

const AttachFileBtn = () => {
    // const dispatch = useDispatch()
    const [isOpenConfirm, setIsOpenConfirm] = useState(false)
    const [file, setFile] = useState(null)
    // const { file } = useSelector((state) => state.queryVideoSlice)

    const handleImageChange = (event) => {
        const file = event.target.files[0]
        if (file.size > 512000) {
            toast('Please upload images smaller than 500KB')
        } else {
            const { target = {} } = event || {};
            target.value = "";
            if (file && file.type.includes('image')) {
                setFile(file)
                setIsOpenConfirm(true)
            }
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
                fileState={file}
                setFileState={setFile}
            />
        </>
    );
};

export default AttachFileBtn;
