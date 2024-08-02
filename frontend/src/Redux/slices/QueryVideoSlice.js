import { createSlice } from '@reduxjs/toolkit'
const initialState = {
    keySearch: '',
    videos: [],
    suggesstions: [],
    isLoading: false,
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
            return newState
        },
        setSuggesstion: (state, action) => {
            let newState = { ...state }
            return newState
        },
        setLoading: (state, action) => {
            let newState = { ...state }
            return newState
        },
    },
});
const { reducer } = queryVideoSlice;
export const { setKeySearch, setListVideo, setSuggesstion, setLoading } = queryVideoSlice.actions;
export default reducer;