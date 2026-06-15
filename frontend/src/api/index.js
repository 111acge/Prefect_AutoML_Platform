import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
})

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.data?.detail) {
      error.message = error.response.data.detail
    }
    return Promise.reject(error)
  }
)

// 数据集 API
export const datasetApi = {
  list: () => api.get('/datasets'),
  upload: (formData) => api.post('/datasets/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }),
  get: (id) => api.get(`/datasets/${id}`),
  preview: (id) => api.get(`/datasets/${id}/preview`),
  quality: (id) => api.get(`/datasets/${id}/quality`),
  delete: (id) => api.delete(`/datasets/${id}`),
}

// 训练任务 API
export const runApi = {
  list: () => api.get('/runs'),
  create: (data) => api.post('/runs', data),
  get: (id) => api.get(`/runs/${id}`),
  getResults: (id) => api.get(`/runs/${id}/results`),
  predict: (id, data) => api.post(`/runs/${id}/predict`, data),
  delete: (id) => api.delete(`/runs/${id}`),
}

export default api
