// Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
// This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
// See LICENSE for details.

import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useAppStore = defineStore('app', () => {
  const loading = ref(false)
  const error = ref(null)

  const setLoading = (value) => {
    loading.value = value
  }

  const setError = (value) => {
    error.value = value
  }

  return {
    loading,
    error,
    setLoading,
    setError,
  }
})
