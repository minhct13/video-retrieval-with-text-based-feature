import { all, put, call, takeEvery, select } from 'redux-saga/effects'
import { v4 as uuidv4 } from 'uuid'
import { getVideoAction, getSuggestion } from '../Actions/QueryVideoActions'
import { setLoading, setListVideo, setSuggesstion, setCountQuery, setQuery } from '../slices/QueryVideoSlice'
import { Service } from '../../Services/QueryVideoServices'
import { toast } from 'react-toastify'

function* handleGetVideosApi(action) {
    yield put(setLoading(true))
    try {
        const res = yield call(Service.getVideosApi, action.payload)
        if (res.data && res.data.data) {
            let { countQuery } = yield select(state => state.queryVideoSlice)
            countQuery = countQuery + 1
            res.data.data.forEach(el => {
                el.id = uuidv4()
            })
            yield put(setListVideo(res.data.data))
            yield put(setCountQuery(countQuery))
            yield put(setQuery(action.payload.query))
        }
        yield put(setLoading(false))
    } catch (error) {
        yield put(setLoading(false))
        toast('System error:', error.message)
    }
}
function* handleGetSuggestionApi() {
    try {
        const res = yield call(Service.getSuggestionsApi)
        if (res.data && res.data.data) {
            const convertData = []
            res.data.data.forEach(el => {
                if (el) {
                    convertData.push({
                        id: uuidv4(),
                        name: el
                    })
                }
            })
            yield put(setSuggesstion(convertData))
        }
    } catch (error) {
        toast('System error:', error.message)
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
