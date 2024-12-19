import { instance , instanceForm } from './configAxios'
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
    return instanceForm.post(servicePattern.getVideos, data)
}