import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { runApi } from '@/api'

export const usePipelineStore = defineStore('pipeline', () => {
  const runId = ref(null)
  const run = ref(null)
  const steps = ref([])
  const logs = ref('')
  const loading = ref(false)
  const error = ref(null)

  const stepMap = computed(() => {
    const map = {}
    steps.value.forEach((s) => {
      map[s.step_name] = s
    })
    return map
  })

  const currentStep = computed(() => {
    return steps.value.find((s) => s.status === 'running') || null
  })

  const nextPendingStep = computed(() => {
    return steps.value.find((s) => s.status === 'pending' || s.status === 'failed') || null
  })

  const allCompleted = computed(() => {
    return steps.value.length > 0 && steps.value.every((s) => s.status === 'completed')
  })

  function setRunId(id) {
    runId.value = id
  }

  async function loadRun() {
    if (!runId.value) return
    const res = await runApi.get(runId.value)
    run.value = res.data
  }

  async function loadSteps() {
    if (!runId.value) return
    const res = await runApi.getSteps(runId.value)
    steps.value = res.data
  }

  async function loadLogs() {
    if (!runId.value) return
    try {
      const res = await runApi.getLogs(runId.value)
      logs.value = res.data
    } catch (e) {
      logs.value = ''
    }
  }

  async function refresh() {
    await Promise.all([loadRun(), loadSteps(), loadLogs()])
  }

  async function executeStep(stepName) {
    if (!runId.value) return
    loading.value = true
    error.value = null
    try {
      await runApi.executeStep(runId.value, stepName)
    } catch (e) {
      error.value = e.message || '执行步骤失败'
      throw e
    } finally {
      loading.value = false
    }
  }

  async function continueRun(stepName = null) {
    if (!runId.value) return
    loading.value = true
    error.value = null
    try {
      await runApi.continue(runId.value, stepName)
    } catch (e) {
      error.value = e.message || '继续执行失败'
      throw e
    } finally {
      loading.value = false
    }
  }

  async function loadArtifact(name) {
    if (!runId.value) return null
    const res = await runApi.getArtifact(runId.value, name)
    return res.data
  }

  return {
    runId,
    run,
    steps,
    logs,
    loading,
    error,
    stepMap,
    currentStep,
    nextPendingStep,
    allCompleted,
    setRunId,
    loadRun,
    loadSteps,
    loadLogs,
    refresh,
    executeStep,
    continueRun,
    loadArtifact,
  }
})
