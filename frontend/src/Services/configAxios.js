import axios from 'axios';
import { VITE_API_URL } from '../config'

export const instance = axios.create({
    method: 'post',
    baseURL: VITE_API_URL,
    withCredentials: true,
    headers: {
        'Content-Type': 'application/json',
        "ngrok-skip-browser-warning": true
    },
})
instance.interceptors.request.use(config => {
    return config
}, null)


export const instanceForm = axios.create({
    method: 'post',
    baseURL: VITE_API_URL,
    withCredentials: true,
    headers: {
        'Content-Type': 'multipart/form-data',
        "ngrok-skip-browser-warning": true
    },
});
instanceForm.interceptors.request.use(config => {
    return config
}, null)