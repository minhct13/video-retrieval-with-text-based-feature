import { instance } from './configAxios'
export const Service = {
    getSuggestionsApi,
    getVideosApi,
}
const servicePattern = {
    getSuggestion: 'suggest',
    getVideos: 'query',
}
function getSuggestionsApi() {
    return instance.get(`${servicePattern.getSuggestion}`)
}
function getVideosApi(data) {
    return instance.post(servicePattern.getVideos, data)
}