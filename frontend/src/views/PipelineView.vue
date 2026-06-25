<!--
Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
See LICENSE for details.
-->

<template>
  <div class="pipeline-view">
    <el-page-header @back="goBack" title="Pipeline 流水线" />

    <el-card class="run-info" v-if="run">
      <el-descriptions :column="3" border>
        <el-descriptions-item label="Run ID">{{ run.id }}</el-descriptions-item>
        <el-descriptions-item label="状态">
          <el-tag :type="statusType(run.status)">{{ run.status }}</el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="数据集">{{ run.dataset_id }}</el-descriptions-item>
        <el-descriptions-item label="目标列">{{ run.config?.target_column }}</el-descriptions-item>
        <el-descriptions-item label="任务类型">{{ taskTypeLabel(run.config?.task_type) }}</el-descriptions-item>
        <el-descriptions-item label="时间预算">{{ run.time_budget_minutes ?? '无限制' }} min</el-descriptions-item>
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
                  执行此步骤
                </el-button>
                <el-button
                  v-if="selectedStep.status === 'completed'"
                  size="small"
                  :loading="artifactLoading"
                  @click="loadArtifacts(selectedStep.step_name)"
                >
                  刷新产物
                </el-button>
              </div>
            </div>
          </template>

          <div class="step-meta">
            <p><strong>状态：</strong>{{ statusLabel(selectedStep.status) }}</p>
            <p v-if="selectedStep.started_at">
              <strong>开始：</strong>{{ formatTime(selectedStep.started_at) }}
            </p>
            <p v-if="selectedStep.completed_at">
              <strong>完成：</strong>{{ formatTime(selectedStep.completed_at) }}
            </p>
            <p v-if="selectedStep.error_message" class="error-message">
              <strong>错误：</strong>{{ selectedStep.error_message }}
            </p>
          </div>

          <div v-if="artifactPreview?.type" class="artifact-section">
            <div class="artifact-header">
              <h4>{{ artifactTitle }}</h4>
              <el-text v-if="artifactPreview.total !== undefined" type="info" size="small">
                共 {{ artifactPreview.total }} 行{{ artifactPreview.truncated ? '，仅展示前 20 行' : '' }}
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
                title="该产物为二进制文件，无法直接预览"
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
                下载 {{ artifactPreview.filename }}
              </el-button>
            </div>
          </div>

          <div v-else-if="selectedStep.status === 'completed'" class="artifact-hint">
            <el-text type="info">点击“刷新产物”查看此步骤输出</el-text>
          </div>
        </el-card>

        <el-card class="global-actions">
          <el-button
            type="primary"
            :loading="pipeline.loading"
            :disabled="pipeline.allCompleted"
            @click="continueRun"
          >
            继续下一步
          </el-button>
          <el-button
            type="success"
            :loading="pipeline.loading"
            :disabled="pipeline.allCompleted"
            @click="runAll"
          >
            执行全部剩余步骤
          </el-button>
          <el-button @click="refresh">刷新状态</el-button>
          <el-button v-if="run?.status === 'completed'" @click="goResults">
            查看完整结果
          </el-button>
        </el-card>

        <el-card class="logs-card">
          <template #header>
            <span>训练日志</span>
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
import { usePipelineStore } from '@/stores/pipeline'
import { ElMessage } from 'element-plus'

const route = useRoute()
const router = useRouter()
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
  ingest: '数据加载与 Schema 校验',
  analyze: '元数据分析',
  quality: '数据质量评估',
  strategy: '训练策略构建',
  split: '训练/验证/测试划分',
  cross_validate: '交叉验证',
  fit_preprocessor: '拟合预处理器',
  transform: '数据转换',
  sample: '条件采样',
  train: '模型训练',
  evaluate: '模型评估',
  interpret: '业务解读',
  report: 'HTML 报告',
}

const taskTypeLabels = {
  binary_classification: '二分类',
  multiclass_classification: '多分类',
  regression: '回归',
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
  if (step.status === 'running') return '执行中...'
  if (step.status === 'completed') return '已完成'
  if (step.status === 'failed') return step.error_message || '失败'
  return '等待执行'
}

function statusLabel(status) {
  const map = {
    pending: '等待执行',
    running: '执行中',
    completed: '已完成',
    failed: '失败',
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
    ElMessage.success(`已提交步骤：${stepName}`)
    startPolling()
  } catch (e) {
    ElMessage.error(e.message || '执行失败')
  }
}

async function continueRun() {
  try {
    await pipeline.continueRun()
    ElMessage.success('已提交下一步')
    startPolling()
  } catch (e) {
    ElMessage.error(e.message || '继续执行失败')
  }
}

async function runAll() {
  const pending = steps.value.filter((s) => s.status === 'pending' || s.status === 'failed')
  if (pending.length === 0) {
    ElMessage.info('没有待执行的步骤')
    return
  }
  try {
    await pipeline.continueRun()
    ElMessage.success('已开始执行后续步骤')
    startPolling()
  } catch (e) {
    ElMessage.error(e.message || '执行失败')
  }
}

const artifactTitle = computed(() => {
  if (!selectedStep.value) return ''
  const meta = artifactMeta[selectedStep.value.step_name]
  if (!meta) return ''
  const titles = {
    json: 'JSON 产物',
    csv: 'CSV 产物预览',
    parquet: 'Parquet 产物预览',
    html: 'HTML 报告',
    download: '产物下载',
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
    ElMessage.warning(`产物暂不可用：${e.message}`)
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
