import axios from 'axios';
import { VITE_API_URL } from '../config'
// import { getLocalStorage } from '../common/FunctionCommon/FunctionCommon'
// import { TOKEN_IN_LOCALSTORAGE } from '../common/ParamsCommon/ParamsCommon'

export const instance = axios.create({
    method: 'post',
    baseURL: VITE_API_URL,
    // timeout: VITE_TIMEOUT,
    withCredentials: true,
    headers: {
        'Content-Type': 'application/json',
        // "Authorization": token ? `Bearer ${token}` : null
    },
})
instance.interceptors.request.use(config => {
    // config.headers.Authorization = "Bearer " + getLocalStorage(TOKEN_IN_LOCALSTORAGE)
    return config
}, null)
