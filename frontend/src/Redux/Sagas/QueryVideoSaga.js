import { all, put, call, takeEvery } from 'redux-saga/effects'
import { v4 as uuidv4 } from 'uuid'
import { getVideoAction, getSuggestion } from '../Actions/QueryVideoActions'
import { setLoading, setListVideo, setSuggesstion } from '../slices/QueryVideoSlice'
import { Service } from '../../Services/QueryVideoServices'
import { toast } from 'react-toastify'

function* handleGetVideosApi(action) {
    yield put(setLoading(true))
    try {
        const res = yield call(Service.getVideosApi, action.payload)
        if (res.data.isError === false) {
            let data = [
                {
                    id: 1,
                    video_name: "video1.mp4",
                    video_path: "https://res.cloudinary.com/dvvi0pivw/video/upload/v1722612622/20230910_170501_llm8bh.mp4",
                    similarity: 0.8
                },
                {
                    id: 2,
                    video_name: "video1.mp4",
                    video_path: "https://res.cloudinary.com/dvvi0pivw/video/upload/v1722612622/20230910_170501_llm8bh.mp4",
                    similarity: 0.8
                },
                {
                    id: 3,
                    video_name: "video1.mp4",
                    video_path: "https://res.cloudinary.com/dvvi0pivw/video/upload/v1722612622/20230910_170501_llm8bh.mp4",
                    similarity: 0.8
                },
            ]
            data.forEach(el => {
                el.id = uuidv4()
            })
            yield put(setListVideo(data))
        }
        yield put(setLoading(false))
    } catch (error) {
        yield put(setLoading(false))
        toast('Lỗi hệ thống:', error)
    }
}
function* handleGetSuggestionApi() {
    try {
        const res = yield call(Service.getSuggestionsApi)
        if (res.data.isError === false) {
            const convertData = []
            let data = [
                "What are the main actions or activities happening in the video?",
                "Who are the main characters or subjects appearing in the video?",
                "What is the setting or location where the video takes place?",
                "What objects or items are prominently featured in the video?",
                "What is the overall mood or atmosphere of the video?"
            ]
            data.forEach(el => {
                convertData.push({
                    id: uuidv4(),
                    name: el
                })
            })
            yield put(setSuggesstion(data))
        }
    } catch (error) {
        toast('Lỗi hệ thống:', error)
    }
}
function* getVideosSaga() {
    yield takeEvery(getVideoAction, handleGetVideosApi)
}
function* getSuggestionSaga() {
    yield takeEvery(getSuggestion, handleGetSuggestionApi)
}


export function* queryVideosSagaList() {
    yield all([
        getVideosSaga(),
        getSuggestionSaga()
    ])
}
