import { configureStore } from '@reduxjs/toolkit'
import createSagaMiddleware from 'redux-saga'
import queryVideoSlice from './slices/QueryVideoSlice'
import rootSaga from './Sagas/RootSaga'
import { isRejectedWithValue } from '@reduxjs/toolkit'

export const rtkQueryErrorLogger = () => (next) => (action) => {
  // RTK Query uses `createAsyncThunk` from redux-toolkit under the hood, so we're able to utilize these matchers!
  if (isRejectedWithValue(action)) {
    console.warn('We got a rejected action!')
  }
  return next(action)
}

let sagaMiddleware = createSagaMiddleware();
const allReducer = {
  queryVideoSlice
}
const store = configureStore({
  reducer: {
    ...allReducer,
  },
  middleware: (getDefaultMiddleware) => [...getDefaultMiddleware({ thunk: false, serializableCheck: false, }), sagaMiddleware],
})
sagaMiddleware.run(rootSaga)
export default store