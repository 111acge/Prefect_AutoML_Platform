// Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
// This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
// See LICENSE for details.

import axios from 'axios'
import { getLocale } from '@/i18n'

// 默认走前端服务器（vite dev proxy / nginx / 同端口网关）的 /api 路径
// 如需独立域名部署，可通过 VITE_API_BASE_URL 覆盖
const baseURL = import.meta.env.VITE_API_BASE_URL || '/api'

const api = axios.create({
  baseURL,
  timeout: 30000,
})

api.interceptors.request.use((config) => {
  config.headers['Accept-Language'] = getLocale()
  return config
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
  create: (data) => api.post('/runs', data, { timeout: 120000 }),
  get: (id) => api.get(`/runs/${id}`),
  getResults: (id) => api.get(`/runs/${id}/results`),
  getSteps: (id) => api.get(`/runs/${id}/steps`),
  executeStep: (id, stepName) => api.post(`/runs/${id}/steps/${stepName}`),
  continue: (id, stepName = null) => api.post(`/runs/${id}/continue`, { step_name: stepName }),
  getArtifact: (id, name) => api.get(`/runs/${id}/artifacts/${name}`),
  getArtifactBlob: (id, name) =>
    api.get(`/runs/${id}/artifacts/${name}`, { responseType: 'blob' }),
  previewArtifact: (id, name, n = 20) =>
    api.get(`/runs/${id}/artifacts/${name}/preview`, { params: { n } }),
  getLogs: (id) => api.get(`/runs/${id}/logs`),
  predict: (id, data) => api.post(`/runs/${id}/predict`, data),
  delete: (id) => api.delete(`/runs/${id}`),
  compare: (data) => api.post('/runs/compare', data),
  regenerateInterpretation: (id, apiKey) =>
    api.post(`/runs/${id}/interpretation/regenerate`, { api_key: apiKey || undefined }, { timeout: 0 }),
}

// LLM 配置 API
export const llmSettingsApi = {
  get: () => api.get('/settings/llm'),
  save: (data) => api.post('/settings/llm', data),
  listProviders: () => api.get('/settings/llm/providers'),
}

export default api
