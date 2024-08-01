import { all } from 'redux-saga/effects'

const sagasList = [
]

export default function* () {
  yield all(sagasList)
}
