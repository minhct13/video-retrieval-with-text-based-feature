import { createSlice } from '@reduxjs/toolkit'
const initialState = {
    keySearch: '',
    videos: [],
    suggesstions: [],
    isLoading: false,
    countQuery:0,
    query:''
}

const queryVideoSlice = createSlice({
    name: 'aboutMeSlice',
    initialState,
    reducers: {
        setKeySearch: (state, action) => {
            let newState = { ...state }
            newState.keySearch = action.payload
            return newState
        },
        setListVideo: (state, action) => {
            let newState = { ...state }
            newState.videos = action.payload
            return newState
        },
        setSuggesstion: (state, action) => {
            let newState = { ...state }
            newState.suggesstions = action.payload
            return newState
        },
        setLoading: (state, action) => {
            let newState = { ...state }
            newState.isLoading = action.payload
            return newState
        },
        setCountQuery: (state, action) => {
            let newState = { ...state }
            newState.countQuery = action.payload
            return newState
        },
        setQuery: (state, action) => {
            let newState = { ...state }
            newState.query = action.payload
            return newState
        },
    },
});
const { reducer } = queryVideoSlice;
export const { setKeySearch, setListVideo, setSuggesstion, setLoading, setCountQuery, setQuery } = queryVideoSlice.actions;
export default reducer;