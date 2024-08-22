import React from 'react'
import { GrAttachment } from "react-icons/gr";
import styles from './SearchBar.module.css'

const AttachFileBtn = (props) => {
    return (
        <>
            <label htmlFor='btnAttach'>
                <GrAttachment className={styles.attachIcon} />
            </label>
            <input
                hidden
                type='file'
                id='btnAttach'
            />
        </>
    );
};



export default AttachFileBtn;
