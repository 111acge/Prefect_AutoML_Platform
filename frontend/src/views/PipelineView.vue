<!--
Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
See LICENSE for details.
-->

<template>
  <div class="pipeline-view">
    <el-page-header @back="goBack" :title="$t('pipeline.title')" />

    <el-card class="run-info" v-if="run">
      <el-descriptions :column="3" border>
        <el-descriptions-item :label="$t('pipeline.info.runId')">{{ run.id }}</el-descriptions-item>
        <el-descriptions-item :label="$t('pipeline.info.status')">
          <el-tag :type="statusType(run.status)">{{ run.status }}</el-tag>
        </el-descriptions-item>
        <el-descriptions-item :label="$t('pipeline.info.dataset')">{{ run.dataset_id }}</el-descriptions-item>
        <el-descriptions-item :label="$t('pipeline.info.targetColumn')">{{ run.config?.target_column }}</el-descriptions-item>
        <el-descriptions-item :label="$t('pipeline.info.taskType')">{{ taskTypeLabel(run.config?.task_type) }}</el-descriptions-item>
        <el-descriptions-item :label="$t('pipeline.info.timeBudget')">{{ run.time_budget_minutes ?? $t('common.unlimited') }} min</el-descriptions-item>
      </el-descriptions>
    </el-card>

    <div class="pipeline-body">
      <div class="step-list">
        <el-steps direction="vertical" :active="activeIndex" finish-status="success">
          <el-step
            v-for="(step, index) in stepDisplayList"
            :key="step.step_name"
            :title="stepTitle(step)"
            :description="stepDescription(step)"
            :status="stepStatus(step)"
            @click="selectStep(index)"
          />
        </el-steps>
      </div>

      <div class="step-detail">
        <el-card v-if="selectedStep">
          <template #header>
            <div class="detail-header">
              <span>{{ stepTitle(selectedStep) }}</span>
              <div class="actions">
                <el-button
                  v-if="canRun(selectedStep)"
                  type="primary"
                  size="small"
                  :loading="pipeline.loading"
                  @click="runStep(selectedStep.step_name)"
                >
                  {{ $t('pipeline.executeStep') }}
                </el-button>
                <el-button
                  v-if="selectedStep.status === 'completed'"
                  size="small"
                  :loading="artifactLoading"
                  @click="loadArtifacts(selectedStep.step_name)"
                >
                  {{ $t('pipeline.refreshArtifact') }}
                </el-button>
              </div>
            </div>
          </template>

          <div class="step-meta">
            <p><strong>{{ $t('pipeline.stepMeta.status') }}：</strong>{{ statusLabel(selectedStep.status) }}</p>
            <p v-if="selectedStep.started_at">
              <strong>{{ $t('pipeline.stepMeta.started') }}：</strong>{{ formatTime(selectedStep.started_at) }}
            </p>
            <p v-if="selectedStep.completed_at">
              <strong>{{ $t('pipeline.stepMeta.completed') }}：</strong>{{ formatTime(selectedStep.completed_at) }}
            </p>
            <p v-if="selectedStep.error_message" class="error-message">
              <strong>{{ $t('pipeline.stepMeta.error') }}：</strong>{{ selectedStep.error_message }}
            </p>
          </div>

          <div v-if="artifactPreview?.type" class="artifact-section">
            <div class="artifact-header">
              <h4>{{ artifactTitle }}</h4>
              <el-text v-if="artifactPreview.total !== undefined" type="info" size="small">
                {{ $t('pipeline.artifact.rowsTotal', { total: artifactPreview.total }) }}{{ artifactPreview.truncated ? $t('pipeline.artifact.rowsTruncated') : '' }}
              </el-text>
            </div>

            <!-- JSON -->
            <div v-if="artifactPreview.type === 'json'" class="artifact-json">
              <pre>{{ JSON.stringify(artifactPreview.data, null, 2) }}</pre>
            </div>

            <!-- 表格（CSV / Parquet） -->
            <div v-else-if="artifactPreview.type === 'table'" class="artifact-table">
              <el-table :data="artifactPreview.rows" style="width: 100%" max-height="420px" border>
                <el-table-column
                  v-for="col in artifactPreview.columns"
                  :key="col.name"
                  :prop="col.name"
                  :label="col.name"
                  min-width="120"
                  show-overflow-tooltip
                />
              </el-table>
            </div>

            <!-- HTML 报告 -->
            <div v-else-if="artifactPreview.type === 'html'" class="artifact-html">
              <iframe :src="artifactPreview.url" class="artifact-iframe"></iframe>
            </div>

            <!-- 可下载二进制 -->
            <div v-else-if="artifactPreview.type === 'download'" class="artifact-download">
              <el-alert
                :title="$t('pipeline.artifact.binary')"
                type="info"
                :closable="false"
                show-icon
              />
              <el-button
                type="primary"
                :href="artifactPreview.url"
                :download="artifactPreview.filename"
                tag="a"
                style="margin-top: 12px"
              >
                {{ $t('pipeline.artifact.download', { filename: artifactPreview.filename }) }}
              </el-button>
            </div>
          </div>

          <div v-else-if="selectedStep.status === 'completed'" class="artifact-hint">
            <el-text type="info">{{ $t('pipeline.artifact.empty') }}</el-text>
          </div>
        </el-card>

        <el-card class="global-actions">
          <el-button
            type="primary"
            :loading="pipeline.loading"
            :disabled="pipeline.allCompleted"
            @click="continueRun"
          >
            {{ $t('pipeline.continueNext') }}
          </el-button>
          <el-button
            type="success"
            :loading="pipeline.loading"
            :disabled="pipeline.allCompleted"
            @click="runAll"
          >
            {{ $t('pipeline.runAllRemaining') }}
          </el-button>
          <el-button @click="refresh">{{ $t('pipeline.refreshStatus') }}</el-button>
          <el-button v-if="run?.status === 'completed'" @click="goResults">
            {{ $t('pipeline.viewFullResult') }}
          </el-button>
        </el-card>

        <el-card class="logs-card">
          <template #header>
            <span>{{ $t('pipeline.logs') }}</span>
          </template>
          <el-input
            v-model="pipeline.logs"
            type="textarea"
            :rows="12"
            readonly
            class="log-textarea"
          />
        </el-card>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { usePipelineStore } from '@/stores/pipeline'
import { ElMessage } from 'element-plus'

const route = useRoute()
const router = useRouter()
const { t } = useI18n()
const pipeline = usePipelineStore()

const runId = computed(() => route.params.id)
const run = computed(() => pipeline.run)
const steps = computed(() => pipeline.steps)
const selectedIndex = ref(0)
const selectedStep = computed(() => steps.value[selectedIndex.value] || null)
const artifactPreview = ref(null)
const artifactLoading = ref(false)
const pollTimer = ref(null)

const stepDisplayList = computed(() => steps.value)

const activeIndex = computed(() => {
  const running = steps.value.findIndex((s) => s.status === 'running')
  if (running >= 0) return running
  const lastCompleted = steps.value.map((s) => s.status).lastIndexOf('completed')
  return lastCompleted + 1
})

const stepTitles = {
  ingest: t('pipeline.steps.ingest'),
  analyze: t('pipeline.steps.analyze'),
  quality: t('pipeline.steps.quality'),
  strategy: t('pipeline.steps.strategy'),
  split: t('pipeline.steps.split'),
  cross_validate: t('pipeline.steps.cross_validate'),
  fit_preprocessor: t('pipeline.steps.fit_preprocessor'),
  transform: t('pipeline.steps.transform'),
  sample: t('pipeline.steps.sample'),
  train: t('pipeline.steps.train'),
  evaluate: t('pipeline.steps.evaluate'),
  interpret: t('pipeline.steps.interpret'),
  report: t('pipeline.steps.report'),
}

const taskTypeLabels = {
  binary_classification: t('trainForm.binary'),
  multiclass_classification: t('trainForm.multiclass'),
  regression: t('trainForm.regression'),
}

// 步骤 -> 产物名称 & 展示类型
const artifactMeta = {
  ingest: { name: 'raw', type: 'parquet' },
  analyze: { name: 'metadata', type: 'json' },
  quality: { name: 'quality_report', type: 'json' },
  strategy: { name: 'strategy', type: 'json' },
  split: { name: 'train_raw', type: 'parquet' },
  cross_validate: { name: 'cv_results', type: 'json' },
  fit_preprocessor: { name: 'feature_columns', type: 'json' },
  transform: { name: 'train_transformed', type: 'parquet' },
  sample: { name: 'sampled_train', type: 'parquet' },
  train: { name: 'leaderboard', type: 'csv' },
  evaluate: { name: 'metrics', type: 'json' },
  interpret: { name: 'interpretation', type: 'json' },
  report: { name: 'report', type: 'html' },
}

function stepTitle(step) {
  return stepTitles[step.step_name] || step.step_name
}

function taskTypeLabel(type) {
  return taskTypeLabels[type] || type || '-'
}

function stepDescription(step) {
  if (step.status === 'running') return t('pipeline.stepDescriptions.running')
  if (step.status === 'completed') return t('pipeline.stepDescriptions.completed')
  if (step.status === 'failed') return step.error_message || t('pipeline.stepDescriptions.failed')
  return t('pipeline.stepDescriptions.pending')
}

function statusLabel(status) {
  const map = {
    pending: t('pipeline.stepDescriptions.pending'),
    running: t('pipeline.stepDescriptions.running'),
    completed: t('pipeline.stepDescriptions.completed'),
    failed: t('pipeline.stepDescriptions.failed'),
  }
  return map[status] || status
}

function stepStatus(step) {
  if (step.status === 'running') return 'process'
  if (step.status === 'completed') return 'success'
  if (step.status === 'failed') return 'error'
  return 'wait'
}

function statusType(status) {
  const map = {
    pending: 'info',
    running: 'warning',
    completed: 'success',
    failed: 'danger',
  }
  return map[status] || 'info'
}

function canRun(step) {
  return step.status !== 'running' && step.status !== 'completed'
}

let currentObjectUrl = null

function revokeCurrentUrl() {
  if (currentObjectUrl) {
    URL.revokeObjectURL(currentObjectUrl)
    currentObjectUrl = null
  }
}

function selectStep(index) {
  selectedIndex.value = index
  revokeCurrentUrl()
  artifactPreview.value = null
}

function formatTime(iso) {
  if (!iso) return ''
  const d = new Date(iso)
  return d.toLocaleString()
}

async function refresh() {
  await pipeline.refresh()
}

async function runStep(stepName) {
  try {
    await pipeline.executeStep(stepName)
    ElMessage.success(t('pipeline.messages.stepSubmitted', { name: stepName }))
    startPolling()
  } catch (e) {
    ElMessage.error(e.message || t('pipeline.messages.stepExecuteFailed'))
  }
}

async function continueRun() {
  try {
    await pipeline.continueRun()
    ElMessage.success(t('pipeline.messages.nextSubmitted'))
    startPolling()
  } catch (e) {
    ElMessage.error(e.message || t('pipeline.messages.continueFailed'))
  }
}

async function runAll() {
  const pending = steps.value.filter((s) => s.status === 'pending' || s.status === 'failed')
  if (pending.length === 0) {
    ElMessage.info(t('pipeline.messages.noPending'))
    return
  }
  try {
    await pipeline.continueRun()
    ElMessage.success(t('pipeline.messages.runAllStarted'))
    startPolling()
  } catch (e) {
    ElMessage.error(e.message || t('pipeline.messages.stepExecuteFailed'))
  }
}

const artifactTitle = computed(() => {
  if (!selectedStep.value) return ''
  const meta = artifactMeta[selectedStep.value.step_name]
  if (!meta) return ''
  const titles = {
    json: t('pipeline.artifact.json'),
    csv: t('pipeline.artifact.csv'),
    parquet: t('pipeline.artifact.parquet'),
    html: t('pipeline.artifact.html'),
    download: t('pipeline.artifact.download', { filename: meta.name }),
  }
  return titles[meta.type] || meta.name
})

async function loadArtifacts(stepName) {
  revokeCurrentUrl()
  artifactPreview.value = null
  artifactLoading.value = true

  const meta = artifactMeta[stepName]
  if (!meta) {
    artifactLoading.value = false
    return
  }

  try {
    if (meta.type === 'json') {
      const data = await pipeline.loadArtifact(meta.name)
      artifactPreview.value = { type: 'json', data }
      return
    }

    if (meta.type === 'csv' || meta.type === 'parquet') {
      const data = await pipeline.previewArtifact(meta.name, 20)
      artifactPreview.value = {
        type: 'table',
        columns: data.columns || [],
        rows: data.rows || [],
        total: data.total,
        truncated: data.truncated,
      }
      return
    }

    if (meta.type === 'html') {
      const { blob, filename } = await pipeline.downloadArtifact(meta.name)
      currentObjectUrl = URL.createObjectURL(blob)
      artifactPreview.value = { type: 'html', url: currentObjectUrl, filename }
      return
    }

    // joblib 等二进制
    const { blob, filename } = await pipeline.downloadArtifact(meta.name)
    currentObjectUrl = URL.createObjectURL(blob)
    artifactPreview.value = { type: 'download', url: currentObjectUrl, filename }
  } catch (e) {
    ElMessage.warning(t('pipeline.messages.artifactUnavailable', { msg: e.message }))
  } finally {
    artifactLoading.value = false
  }
}

function goBack() {
  router.push('/runs')
}

function goResults() {
  router.push(`/runs/${runId.value}`)
}

function startPolling() {
  stopPolling()
  pollTimer.value = setInterval(() => {
    pipeline.refresh()
  }, 3000)
}

function stopPolling() {
  if (pollTimer.value) {
    clearInterval(pollTimer.value)
    pollTimer.value = null
  }
}

watch(
  () => pipeline.currentStep,
  (current) => {
    if (current) {
      const idx = steps.value.findIndex((s) => s.step_name === current.step_name)
      if (idx >= 0) selectedIndex.value = idx
    }
  }
)

onMounted(async () => {
  pipeline.setRunId(runId.value)
  await pipeline.refresh()
  if (steps.value.length > 0 && !selectedStep.value) {
    selectedIndex.value = 0
  }
  startPolling()
})

onUnmounted(() => {
  stopPolling()
  revokeCurrentUrl()
})
</script>

<style scoped>
.pipeline-view {
  padding: 20px;
}

.run-info {
  margin-top: 16px;
  margin-bottom: 16px;
}

.pipeline-body {
  display: flex;
  gap: 20px;
  align-items: flex-start;
}

.step-list {
  width: 300px;
  flex-shrink: 0;
  max-height: calc(100vh - 240px);
  overflow-y: auto;
}

.step-detail {
  flex: 1;
  min-width: 0;
}

.detail-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.actions {
  display: flex;
  gap: 8px;
}

.step-meta {
  margin-bottom: 16px;
}

.error-message {
  color: #f56c6c;
}

.artifact-section {
  margin-top: 16px;
}

.artifact-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}

.artifact-header h4 {
  margin: 0;
}

.artifact-json {
  background: #f5f7fa;
  padding: 12px;
  border-radius: 4px;
  overflow-x: auto;
}

.artifact-json pre {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
}

.artifact-table {
  border-radius: 4px;
  overflow: hidden;
}

.artifact-html {
  border: 1px solid #ebeef5;
  border-radius: 4px;
  overflow: hidden;
}

.artifact-iframe {
  width: 100%;
  height: 480px;
  border: none;
}

.artifact-download {
  padding: 12px;
  background: #f5f7fa;
  border-radius: 4px;
}

.artifact-hint {
  margin-top: 12px;
}

.global-actions {
  margin-top: 16px;
  margin-bottom: 16px;
}

.logs-card {
  margin-top: 16px;
}

.log-textarea :deep(textarea) {
  font-family: 'Courier New', monospace;
  font-size: 12px;
}
</style>
