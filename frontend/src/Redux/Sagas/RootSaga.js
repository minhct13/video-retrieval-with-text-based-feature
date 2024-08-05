import { all } from 'redux-saga/effects'
import { queryVideosSagaList } from './QueryVideoSaga'
const sagasList = [
  queryVideosSagaList()
]

export default function* () {
  yield all(sagasList)
}
