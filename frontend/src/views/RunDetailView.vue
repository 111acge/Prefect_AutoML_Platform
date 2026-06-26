<!--
Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
See LICENSE for details.
-->

<template>
  <div class="run-detail">
    <el-page-header @back="$router.push('/runs')" :title="$t('runDetail.title')">
      <template #extra>
        <el-button size="small" @click="$router.push(`/runs/${run.id}/pipeline`)">
          {{ $t('runDetail.viewPipeline') }}
        </el-button>
      </template>
    </el-page-header>

    <el-card v-loading="loading" class="info-card">
      <template #header>
        <div class="card-header">
          <span>{{ $t('runDetail.info.title') }}</span>
          <el-tag :type="statusType(run.status)">{{ statusLabel(run.status) }}</el-tag>
        </div>
      </template>

      <el-descriptions :column="2" border>
        <el-descriptions-item :label="$t('runDetail.info.runId')">{{ run.id }}</el-descriptions-item>
        <el-descriptions-item :label="$t('runDetail.info.dataset')">{{ run.dataset_id }}</el-descriptions-item>
        <el-descriptions-item :label="$t('runDetail.info.targetColumn')">{{ run.config?.target_column || '-' }}</el-descriptions-item>
        <el-descriptions-item :label="$t('runDetail.info.taskType')">{{ taskTypeLabel(run.config?.task_type) }}</el-descriptions-item>
        <el-descriptions-item :label="$t('runDetail.info.metric')">{{ run.primary_metric || '-' }}</el-descriptions-item>
        <el-descriptions-item :label="$t('runDetail.info.timeBudget')">{{ run.time_budget_minutes ?? $t('common.unlimited') }} min</el-descriptions-item>
        <el-descriptions-item :label="$t('runDetail.info.seed')">{{ run.config?.seed ?? '-' }}</el-descriptions-item>
        <el-descriptions-item :label="$t('runDetail.info.createdAt')">{{ formatDate(run.created_at) }}</el-descriptions-item>
        <el-descriptions-item :label="$t('runDetail.info.completedAt')" v-if="run.completed_at">{{ formatDate(run.completed_at) }}</el-descriptions-item>
        <el-descriptions-item :label="$t('runDetail.info.error')" v-if="run.error_message">
          <span class="error-text">{{ run.error_message }}</span>
        </el-descriptions-item>
        <el-descriptions-item :label="$t('runDetail.info.errorType')" v-if="results?.error_details?.error_type">
          <span class="error-text">{{ results.error_details.error_type }}</span>
        </el-descriptions-item>
      </el-descriptions>
    </el-card>

    <el-card v-if="!isTerminal(run.status) || run.status === 'failed'" class="progress-card">
      <template #header>
        <div class="card-header">
          <span>{{ $t('runDetail.progress') }}</span>
          <el-tag :type="statusType(run.status)">{{ statusLabel(run.status) }}</el-tag>
        </div>
      </template>

      <el-progress
        :percentage="progressPercentage"
        :status="run.status === 'failed' ? 'exception' : ''"
        :stroke-width="18"
        striped
        striped-flow
      />

      <div class="progress-steps" v-if="steps.length">
        <div
          v-for="step in steps"
          :key="step.step_name"
          class="progress-step"
          :class="step.status"
        >
          <el-icon v-if="step.status === 'completed'" class="step-icon success"><component :is="Check" /></el-icon>
          <el-icon v-else-if="step.status === 'running'" class="step-icon running"><component :is="Loading" /></el-icon>
          <el-icon v-else-if="step.status === 'failed'" class="step-icon failed"><component :is="CloseBold" /></el-icon>
          <el-icon v-else class="step-icon pending"><component :is="Minus" /></el-icon>
          <span class="step-name">{{ step.step_name }}</span>
          <el-tag size="small" :type="statusType(step.status)">{{ step.status }}</el-tag>
        </div>
      </div>

      <h3 class="section-title">{{ $t('runDetail.logs.title') }}</h3>
      <el-input
        ref="logTextareaRef"
        v-model="logs"
        type="textarea"
        :rows="12"
        readonly
        :placeholder="$t('runDetail.logs.loading')"
        class="log-textarea"
      />
    </el-card>

    <el-card v-if="results" class="result-card">
      <el-tabs v-model="activeTab" type="border-card">
        <el-tab-pane :label="$t('runDetail.tabs.overview')" name="overview">
          <div class="tab-content">
            <h3 class="section-title">{{ $t('runDetail.overview.testMetrics') }}</h3>
            <el-row :gutter="16" class="metric-row">
              <el-col :xs="12" :sm="8" :md="6" :lg="4" v-for="(value, key) in results.metrics" :key="key">
                <el-statistic :title="key" :value="typeof value === 'number' ? value : NaN" :precision="4" />
              </el-col>
            </el-row>

            <template v-if="results.train_metrics && Object.keys(results.train_metrics).length > 0">
              <h3 class="section-title">{{ $t('runDetail.overview.trainMetrics') }}</h3>
              <el-row :gutter="16" class="metric-row">
                <el-col :xs="12" :sm="8" :md="6" :lg="4" v-for="(value, key) in results.train_metrics" :key="key">
                  <el-statistic :title="key" :value="typeof value === 'number' ? value : NaN" :precision="4" />
                </el-col>
              </el-row>
            </template>

            <template v-if="results.cv_results && results.cv_results.cv_scores">
              <h3 class="section-title">{{ $t('runDetail.overview.cv.title') }}</h3>
              <el-row :gutter="16" class="metric-row">
                <el-col :xs="12" :sm="6">
                  <div class="metric-item">
                    <div class="metric-label">{{ $t('runDetail.overview.cv.type') }}</div>
                    <div class="metric-value">{{ results.cv_results.cv_type }}</div>
                  </div>
                </el-col>
                <el-col :xs="12" :sm="6">
                  <el-statistic :title="$t('runDetail.overview.cv.folds')" :value="results.cv_results.n_folds" />
                </el-col>
                <el-col :xs="12" :sm="6">
                  <el-statistic :title="$t('runDetail.overview.cv.mean')" :value="results.cv_results.cv_mean" :precision="4" />
                </el-col>
                <el-col :xs="12" :sm="6">
                  <el-statistic :title="$t('runDetail.overview.cv.std')" :value="results.cv_results.cv_std" :precision="4" />
                </el-col>
              </el-row>
              <el-alert
                v-if="results.cv_results.cv_error"
                :title="$t('runDetail.overview.cv.failed', { msg: results.cv_results.cv_error })"
                type="warning"
                :closable="false"
              />
            </template>

            <template v-if="extendedMetricItems.length > 0">
              <h3 class="section-title">{{ $t('runDetail.overview.extendedMetrics') }}</h3>
              <el-row :gutter="16" class="metric-row">
                <el-col :xs="12" :sm="8" :md="6" :lg="4" v-for="item in extendedMetricItems" :key="item.key">
                  <el-statistic :title="item.label" :value="Number(item.value)" :precision="4" />
                </el-col>
              </el-row>
            </template>

            <template v-if="results.business_interpretation">
              <h3 class="section-title">{{ $t('runDetail.overview.interpretation.title') }}</h3>
              <el-card shadow="never" class="interpretation-card">
                <el-alert
                  v-if="results.business_interpretation.provider === 'rule_template'"
                  :title="$t('runDetail.overview.interpretation.ruleBased')"
                  type="info"
                  :closable="false"
                  style="margin-bottom: 12px;"
                >
                  <template #default>
                    <div>
                      {{ $t('runDetail.overview.interpretation.ruleBasedDesc') }}
                      <el-button
                        v-if="llmConfigured"
                        link
                        type="primary"
                        :loading="regeneratingInterpretation"
                        @click="openDisclaimer"
                      >
                        {{ $t('runDetail.overview.interpretation.generateLLM') }}
                      </el-button>
                      <el-button
                        v-else
                        link
                        type="primary"
                        @click="llmDialogVisible = true"
                      >
                        {{ $t('runDetail.overview.interpretation.configureLLM') }}
                      </el-button>
                    </div>
                  </template>
                </el-alert>
                <p class="interpretation-summary">
                  {{ results.business_interpretation.business_summary }}
                </p>
                <div v-if="results.business_interpretation.key_insights?.length" class="interpretation-section">
                  <h4>{{ $t('runDetail.overview.interpretation.keyFindings') }}</h4>
                  <ul>
                    <li v-for="(item, idx) in results.business_interpretation.key_insights" :key="'insight-' + idx">
                      {{ item }}
                    </li>
                  </ul>
                </div>
                <div v-if="results.business_interpretation.feature_interpretations?.length" class="interpretation-section">
                  <h4>{{ $t('runDetail.overview.interpretation.featureMeanings') }}</h4>
                  <ul>
                    <li v-for="(item, idx) in results.business_interpretation.feature_interpretations" :key="'feature-' + idx">
                      <strong>{{ item.feature || item }}</strong>
                      <span v-if="item.interpretation">：{{ item.interpretation }}</span>
                    </li>
                  </ul>
                </div>
                <div v-if="results.business_interpretation.caveats?.length" class="interpretation-section">
                  <h4>{{ $t('runDetail.overview.interpretation.caveats') }}</h4>
                  <ul>
                    <li v-for="(item, idx) in results.business_interpretation.caveats" :key="'caveat-' + idx">
                      {{ item }}
                    </li>
                  </ul>
                </div>
                <div v-if="results.business_interpretation.recommendations?.length" class="interpretation-section">
                  <h4>{{ $t('runDetail.overview.interpretation.recommendations') }}</h4>
                  <ul>
                    <li v-for="(item, idx) in results.business_interpretation.recommendations" :key="'rec-' + idx">
                      {{ item }}
                    </li>
                  </ul>
                </div>
              </el-card>
            </template>

            <template v-if="results?.error_details?.traceback">
              <h3 class="section-title">{{ $t('runDetail.overview.interpretation.errorStack') }}</h3>
              <el-input v-model="results.error_details.traceback" type="textarea" :rows="10" readonly />
            </template>
          </div>
        </el-tab-pane>

        <el-tab-pane :label="$t('runDetail.tabs.leaderboard')" name="leaderboard">
          <div class="tab-content">
            <h3 class="section-title">{{ $t('runDetail.leaderboard.title') }}</h3>
            <div class="leaderboard-controls">
              <el-select v-model="familyFilter" :placeholder="$t('runDetail.leaderboard.modelFamily')" style="width: 140px">
                <el-option :label="$t('runDetail.leaderboard.all')" value="all" />
                <el-option v-for="f in families" :key="f" :label="f" :value="f" />
              </el-select>
              <el-select v-model="sortBy" :placeholder="$t('runDetail.leaderboard.sortBy')" style="width: 160px; margin-left: 10px">
                <el-option v-for="col in sortableColumns" :key="col" :label="col" :value="col" />
              </el-select>
            </div>
            <el-table :data="processedLeaderboard" style="width: 100%" max-height="400px" border>
              <el-table-column
                v-for="col in leaderboardColumns"
                :key="col"
                :prop="col"
                :label="col"
                show-overflow-tooltip
              />
            </el-table>

            <h3 class="section-title">{{ $t('runDetail.leaderboard.featureImportanceTop10') }}</h3>
            <el-table :data="results.feature_importance.slice(0, 10)" style="width: 100%" max-height="360px" border>
              <el-table-column
                v-for="col in importanceColumns"
                :key="col"
                :prop="col"
                :label="col"
                show-overflow-tooltip
              />
            </el-table>
          </div>
        </el-tab-pane>

        <el-tab-pane :label="$t('runDetail.tabs.features')" name="features">
          <div class="tab-content">
            <h3 class="section-title">{{ $t('runDetail.features.title') }}</h3>
            <EChart :option="featureImportanceOption" height="420px" />

            <template v-if="results.permutation_importance && results.permutation_importance.length">
              <h3 class="section-title">{{ $t('runDetail.features.permutationTitle') }}</h3>
              <EChart :option="permutationImportanceOption" height="420px" />
            </template>
          </div>
        </el-tab-pane>

        <el-tab-pane :label="$t('runDetail.tabs.confusion')" name="confusion" v-if="confusionMatrix">
          <div class="tab-content">
            <h3 class="section-title">{{ $t('runDetail.confusionMatrix.title') }}</h3>
            <el-row :gutter="20">
              <el-col :xs="24" :md="12">
                <EChart :option="confusionMatrixOption" height="420px" />
              </el-col>
              <el-col :xs="24" :md="12">
                <el-table :data="confusionMatrixRows" border style="width: 100%;">
                  <el-table-column :label="$t('runDetail.confusionMatrix.trueVsPred')" prop="label" width="120" />
                  <el-table-column
                    v-for="(label, idx) in confusionMatrixLabels"
                    :key="idx"
                    :label="label"
                    :prop="'pred_' + idx"
                  />
                </el-table>
              </el-col>
            </el-row>
          </div>
        </el-tab-pane>

        <el-tab-pane :label="$t('runDetail.tabs.report')" name="report">
          <div class="tab-content">
            <div class="action-buttons">
              <el-button type="primary" @click="downloadReport">{{ $t('runDetail.report.downloadReport') }}</el-button>
              <el-button v-if="run.status === 'completed'" type="success" @click="downloadModel">{{ $t('runDetail.report.downloadModel') }}</el-button>
            </div>
            <iframe v-if="results.report_path" :src="`/api/runs/${runId}/report`" class="report-iframe"></iframe>
            <el-empty v-else :description="$t('runDetail.report.notReady')" />
          </div>
        </el-tab-pane>

        <el-tab-pane :label="$t('runDetail.tabs.predict')" name="predict">
          <div class="tab-content">
            <el-row :gutter="24">
              <el-col :xs="24" :md="12">
                <el-card shadow="never">
                  <template #header>
                    <span>{{ $t('runDetail.predict.jsonTitle') }}</span>
                  </template>
                  <el-alert
                    :title="$t('runDetail.predict.jsonHint')"
                    type="info"
                    :description="$t('runDetail.predict.jsonExample')"
                    show-icon
                    :closable="false"
                  />
                  <el-input
                    v-model="predictInput"
                    type="textarea"
                    :rows="6"
                    :placeholder="$t('runDetail.predict.jsonPlaceholder')"
                    style="margin-top: 15px;"
                  />
                  <div v-if="predictResult" class="predict-result">
                    <h4>{{ $t('runDetail.predict.result') }}</h4>
                    <pre>{{ JSON.stringify(predictResult, null, 2) }}</pre>
                  </div>
                  <div style="margin-top: 12px; text-align: right;">
                    <el-button type="primary" @click="submitPredict" :loading="predicting">{{ $t('runDetail.predict.predict') }}</el-button>
                  </div>
                </el-card>
              </el-col>
              <el-col :xs="24" :md="12">
                <el-card shadow="never">
                  <template #header>
                    <span>{{ $t('runDetail.predict.csvTitle') }}</span>
                  </template>
                  <el-upload
                    drag
                    :auto-upload="false"
                    :limit="1"
                    accept=".csv"
                    @change="handleBatchFileChange"
                  >
                    <el-icon class="el-icon--upload"><upload-icon /></el-icon>
                    <div class="el-upload__text">{{ $t('runDetail.predict.csvDrag') }}<em>{{ $t('runDetail.predict.csvClick') }}</em></div>
                  </el-upload>
                  <div v-if="batchResultUrl" class="batch-result">
                    <el-alert :title="$t('runDetail.predict.completed')" type="success" :closable="false" />
                    <el-button type="primary" :href="batchResultUrl" download="predictions.csv" tag="a" style="margin-top: 12px;">
                      {{ $t('runDetail.predict.downloadResult') }}
                    </el-button>
                  </div>
                  <div style="margin-top: 12px; text-align: right;">
                    <el-button type="success" @click="submitBatchPredict" :loading="batchPredicting" :disabled="!batchFile">
                      {{ $t('runDetail.predict.submitBatch') }}
                    </el-button>
                  </div>
                </el-card>
              </el-col>
            </el-row>
          </div>
        </el-tab-pane>

        <el-tab-pane :label="$t('runDetail.tabs.logs')" name="logs">
          <div class="tab-content">
            <el-input
              ref="logTextareaRef"
              v-model="logs"
              type="textarea"
              :rows="20"
              readonly
              :placeholder="$t('runDetail.logs.loading')"
              class="log-textarea"
            />
          </div>
        </el-tab-pane>
      </el-tabs>
    </el-card>

    <el-card v-else class="result-card">
      <el-empty :description="run.status === 'running' ? $t('runDetail.noResults') : $t('runDetail.noResultsEmpty')" />
    </el-card>
  </div>

  <LLMSettingsDialog v-model:visible="llmDialogVisible" @saved="onLLMSettingsSaved" />

  <el-dialog
    v-model="disclaimerVisible"
    :title="$t('runDetail.interpretationDialog.title')"
    width="480px"
    :close-on-click-modal="false"
  >
    <el-alert
      type="warning"
      :closable="false"
      show-icon
    >
      <template #title>
        <span>{{ $t('runDetail.interpretationDialog.alert') }}</span>
      </template>
      <p>{{ $t('runDetail.interpretationDialog.content') }}</p>
    </el-alert>
    <p style="margin-top: 16px;">{{ $t('runDetail.interpretationDialog.confirm') }}</p>
    <template #footer>
      <el-button @click="disclaimerVisible = false">{{ $t('runDetail.interpretationDialog.cancel') }}</el-button>
      <el-button type="primary" :loading="regeneratingInterpretation" @click="confirmLLMInterpretation">
        {{ $t('runDetail.interpretationDialog.confirmGenerate') }}
      </el-button>
    </template>
  </el-dialog>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { Upload as UploadIcon, Check, Loading, CloseBold, Minus } from '@element-plus/icons-vue'
import api, { runApi, llmSettingsApi } from '@/api'
import EChart from '@/components/EChart.vue'
import { useI18n } from 'vue-i18n'
import LLMSettingsDialog from '@/components/LLMSettingsDialog.vue'

const route = useRoute()
const runId = route.params.id
const { t } = useI18n()

const run = ref({})
const results = ref(null)
const loading = ref(false)
const activeTab = ref('overview')
const showPredictDialog = ref(false)
const predictInput = ref('')
const predictResult = ref(null)
const predicting = ref(false)
const batchFile = ref(null)
const batchPredicting = ref(false)
const batchResultUrl = ref(null)
const logs = ref('')
const logTextareaRef = ref(null)
const steps = ref([])
const llmConfigured = ref(false)
const llmDialogVisible = ref(false)
const disclaimerVisible = ref(false)
const regeneratingInterpretation = ref(false)
let logTimer = null
let evtSource = null

const familyFilter = ref('all')
const sortBy = ref('score_val')

const taskTypeLabels = {
  binary_classification: t('trainForm.binary'),
  multiclass_classification: t('trainForm.multiclass'),
  regression: t('trainForm.regression'),
}

function taskTypeLabel(type) {
  return taskTypeLabels[type] || type || '-'
}

function statusLabel(status) {
  const map = {
    pending: t('taskStatus.pending'),
    running: t('taskStatus.running'),
    completed: t('taskStatus.completed'),
    failed: t('taskStatus.failed'),
  }
  return map[status] || status
}

const statusType = (status) => {
  const map = {
    pending: 'info',
    running: 'warning',
    completed: 'success',
    failed: 'danger',
  }
  return map[status] || 'info'
}

const leaderboardColumns = computed(() => {
  const rows = processedLeaderboard.value
  if (!rows.length) return []
  return Object.keys(rows[0])
})

const getModelFamily = (name) => {
  const n = (name || '').toLowerCase()
  if (n.includes('lightgbm') || n.includes('catboost') || n.includes('xgboost')) return t('runDetail.modelFamilies.gbdt')
  if (n.includes('randomforest') || n.includes('extratrees') || n.includes('xt_')) return t('runDetail.modelFamilies.treeEnsemble')
  if (n.includes('neuralnet') || n.includes('torch') || n.includes('mlp') || n.includes('fastai')) return t('runDetail.modelFamilies.neuralNetwork')
  if (n.includes('kneighbors') || n.includes('knn')) return t('runDetail.modelFamilies.knn')
  if (n.includes('linearmodel') || n.includes('ridge') || n.includes('elastic') || n.includes('logistic')) return t('runDetail.modelFamilies.linear')
  return t('runDetail.modelFamilies.other')
}

const processedLeaderboard = computed(() => {
  if (!results.value || !results.value.leaderboard.length) return []
  let rows = results.value.leaderboard.map((row) => ({ ...row, family: getModelFamily(row.model) }))
  if (familyFilter.value && familyFilter.value !== 'all') {
    rows = rows.filter((row) => row.family === familyFilter.value)
  }
  const key = sortBy.value || 'score_val'
  rows = [...rows].sort((a, b) => {
    const av = a[key]
    const bv = b[key]
    if (typeof av === 'number' && typeof bv === 'number') return bv - av
    return String(av || '').localeCompare(String(bv || ''))
  })
  return rows
})

const families = computed(() => {
  if (!results.value || !results.value.leaderboard.length) return []
  return Array.from(new Set(results.value.leaderboard.map((row) => getModelFamily(row.model)))).sort()
})

const sortableColumns = computed(() => leaderboardColumns.value)

const importanceColumns = computed(() => {
  if (!results.value || !results.value.feature_importance.length) return []
  return Object.keys(results.value.feature_importance[0])
})

const metricLabelMap = {
  precision_macro: t('metric.precisionMacro'),
  precision_weighted: t('metric.precisionWeighted'),
  precision_micro: t('metric.precisionMicro'),
  recall_macro: t('metric.recallMacro'),
  recall_weighted: t('metric.recallWeighted'),
  recall_micro: t('metric.recallMicro'),
  f1_macro: t('metric.f1Macro'),
  f1_weighted: t('metric.f1Weighted'),
  f1_micro: t('metric.f1Micro'),
  mcc: t('metric.mcc'),
  cohens_kappa: t('metric.cohensKappa'),
  balanced_accuracy: t('metric.balancedAccuracy'),
  auc_roc: t('metric.aucRoc'),
  auc_pr: t('metric.aucPr'),
  mae: t('metric.mae'),
  mse: t('metric.mse'),
  rmse: t('metric.rmse'),
  r2: t('metric.r2'),
  mape: t('metric.mape'),
  smape: t('metric.smape'),
  residual_mean: t('metric.residualMean'),
  residual_std: t('metric.residualStd'),
}

const extendedMetricItems = computed(() => {
  if (!results.value || !results.value.extended_metrics) return []
  const extended = results.value.extended_metrics
  const items = []
  for (const [key, value] of Object.entries(extended)) {
    if (key === 'confusion_matrix' || key === 'labels' || key === 'roc_curve' || key === 'pr_curve') continue
    if (typeof value !== 'number') continue
    items.push({
      key,
      label: metricLabelMap[key] || key,
      value: value.toFixed(4),
    })
  }
  return items
})

const confusionMatrix = computed(() => {
  return results.value?.extended_metrics?.confusion_matrix || null
})

const confusionMatrixLabels = computed(() => {
  return results.value?.extended_metrics?.labels || []
})

const confusionMatrixRows = computed(() => {
  if (!confusionMatrix.value) return []
  return confusionMatrix.value.map((row, idx) => {
    const obj = { label: confusionMatrixLabels.value[idx] || idx }
    row.forEach((val, colIdx) => {
      obj['pred_' + colIdx] = val
    })
    return obj
  })
})

const confusionMatrixOption = computed(() => {
  const matrix = confusionMatrix.value
  const labels = confusionMatrixLabels.value
  if (!matrix || !labels.length) return {}

  const data = []
  matrix.forEach((row, i) => {
    row.forEach((val, j) => data.push([j, i, val]))
  })

  return {
    tooltip: { position: 'top' },
    grid: { height: '70%', top: '10%' },
    xAxis: { type: 'category', data: labels, name: t('runDetail.confusionMatrix.predicted') },
    yAxis: { type: 'category', data: labels, name: t('runDetail.confusionMatrix.actual') },
    visualMap: {
      min: 0,
      max: Math.max(...matrix.flat()),
      calculable: true,
      orient: 'horizontal',
      left: 'center',
      bottom: '0%',
    },
    series: [
      {
        type: 'heatmap',
        data,
        label: { show: true },
        emphasis: { itemStyle: { shadowBlur: 10 } },
      },
    ],
  }
})

const featureImportanceOption = computed(() => {
  const data = results.value?.feature_importance?.slice(0, 15) || []
  if (!data.length) return {}

  const sorted = [...data].sort((a, b) => a.importance - b.importance)
  const names = sorted.map((row) => row.feature || row['Unnamed: 0'] || row[''] || t('common.unknown'))
  const values = sorted.map((row) => row.importance)

  return {
    tooltip: { trigger: 'axis' },
    grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
    xAxis: { type: 'value', name: t('runDetail.features.importanceAxisName') },
    yAxis: { type: 'category', data: names },
    series: [
      {
        type: 'bar',
        data: values,
        itemStyle: { color: '#409eff' },
      },
    ],
  }
})

const permutationImportanceOption = computed(() => {
  const data = results.value?.permutation_importance?.slice(0, 15) || []
  if (!data.length) return {}

  const sorted = [...data].sort((a, b) => a.importance - b.importance)
  const names = sorted.map((row) => row.feature || row['Unnamed: 0'] || t('common.unknown'))
  const values = sorted.map((row) => row.importance)

  return {
    tooltip: { trigger: 'axis' },
    grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
    xAxis: { type: 'value', name: t('runDetail.features.permutationAxisName') },
    yAxis: { type: 'category', data: names },
    series: [
      {
        type: 'bar',
        data: values,
        itemStyle: { color: '#67c23a' },
      },
    ],
  }
})

const fetchRun = async () => {
  try {
    const res = await runApi.get(runId)
    run.value = res.data
  } catch (error) {
    ElMessage.error(t('runDetail.errors.fetchFailed', { msg: error.message }))
  }
}

const fetchResults = async () => {
  if (run.value.status !== 'completed') return
  try {
    const res = await runApi.getResults(runId)
    results.value = res.data
  } catch (error) {
    console.error(t('runDetail.errors.fetchResultsFailed', { msg: error.message || t('common.unknown') }), error)
  }
}

const checkLLMConfig = async () => {
  try {
    const res = await llmSettingsApi.get()
    llmConfigured.value = !!(res.data?.provider && res.data?.api_key_masked)
  } catch (error) {
    llmConfigured.value = false
    console.error(t('runDetail.errors.checkLLMFailed', { msg: error.message || t('common.unknown') }), error)
  }
}

const regenerateInterpretation = async () => {
  regeneratingInterpretation.value = true
  try {
    const res = await runApi.regenerateInterpretation(runId)
    results.value = { ...results.value, business_interpretation: res.data }
    ElMessage.success(t('runDetail.interpretationRegenerated'))
  } catch (error) {
    ElMessage.error(t('runDetail.errors.regenerateFailed', { msg: error.message || t('common.unknown') }))
  } finally {
    regeneratingInterpretation.value = false
  }
}

const openDisclaimer = () => {
  if (!llmConfigured.value) {
    llmDialogVisible.value = true
    return
  }
  disclaimerVisible.value = true
}

const confirmLLMInterpretation = async () => {
  disclaimerVisible.value = false
  await regenerateInterpretation()
}

const onLLMSettingsSaved = async () => {
  await checkLLMConfig()
}

const fetchLogs = async () => {
  try {
    const res = await api.get(`/runs/${runId}/logs`)
    logs.value = res.data
    scrollLogToBottom()
  } catch (error) {
    console.error(t('runDetail.errors.fetchLogsFailed', { msg: error.message || t('common.unknown') }), error)
  }
}

const fetchSteps = async () => {
  try {
    const res = await runApi.getSteps(runId)
    steps.value = res.data
  } catch (error) {
    console.error(t('runDetail.errors.fetchStepsFailed', { msg: error.message || t('common.unknown') }), error)
  }
}

const progressPercentage = computed(() => {
  if (!steps.value.length) return 0
  const completed = steps.value.filter((s) => s.status === 'completed').length
  return Math.round((completed / steps.value.length) * 100)
})

const scrollLogToBottom = () => {
  nextTick(() => {
    const textarea = logTextareaRef.value?.$el?.querySelector('textarea')
    if (textarea) {
      textarea.scrollTop = textarea.scrollHeight
    }
  })
}

const loadData = async () => {
  loading.value = true
  await fetchRun()
  await fetchResults()
  loading.value = false
}

const submitPredict = async () => {
  if (!predictInput.value) {
    ElMessage.warning(t('runDetail.predict.validation.dataRequired'))
    return
  }

  predicting.value = true
  try {
    const data = JSON.parse(predictInput.value)
    const res = await runApi.predict(runId, { data })
    predictResult.value = res.data
  } catch (error) {
    ElMessage.error(t('runDetail.predict.errors.predictFailed', { msg: error.message }))
  } finally {
    predicting.value = false
  }
}

const handleBatchFileChange = (file) => {
  batchFile.value = file.raw
  if (batchResultUrl.value) {
    URL.revokeObjectURL(batchResultUrl.value)
    batchResultUrl.value = null
  }
}

const submitBatchPredict = async () => {
  if (!batchFile.value) {
    ElMessage.warning(t('runDetail.predict.validation.fileRequired'))
    return
  }
  batchPredicting.value = true
  try {
    const formData = new FormData()
    formData.append('file', batchFile.value)
    const res = await api.post(`/runs/${runId}/predict/batch`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      responseType: 'blob',
    })
    batchResultUrl.value = URL.createObjectURL(res.data)
    ElMessage.success(t('runDetail.predict.completed'))
  } catch (error) {
    ElMessage.error(t('runDetail.predict.errors.batchFailed', { msg: error.message || t('common.unknown') }))
  } finally {
    batchPredicting.value = false
  }
}

const downloadReport = () => {
  window.open(`/api/runs/${runId}/report?download=1`, '_blank')
}

const downloadModel = () => {
  window.open(`/api/runs/${runId}/model`, '_blank')
}

const formatDate = (date) => {
  if (!date) return null
  return new Date(date).toLocaleString()
}

const eventBase = import.meta.env.VITE_API_BASE_URL || '/api'
const eventsUrl = eventBase.startsWith('http')
  ? `${eventBase}/runs/${runId}/events`
  : `${window.location.origin}${eventBase}/runs/${runId}/events`

const isTerminal = (status) => status === 'completed' || status === 'failed'

const handleStatusEvent = (payload) => {
  if (payload.status !== undefined) {
    run.value.status = payload.status
  }
  if (payload.error_message !== undefined) {
    run.value.error_message = payload.error_message
  }
  fetchLogs()
  fetchSteps()
  if (run.value.status === 'completed' && !results.value) {
    fetchResults()
  }
  if (isTerminal(run.value.status)) {
    closeEventSource()
  }
}

const connectEvents = () => {
  closeEventSource()
  try {
    evtSource = new EventSource(eventsUrl)
    evtSource.onmessage = (e) => {
      try {
        const payload = JSON.parse(e.data)
        handleStatusEvent(payload)
      } catch (err) {
        console.error(t('runDetail.errors.sseParseFailed', { msg: err.message || t('common.unknown') }), err)
      }
    }
    evtSource.onerror = (err) => {
      console.error(t('runDetail.errors.sseError', { msg: err.message || t('common.unknown') }), err)
      closeEventSource()
      if (!isTerminal(run.value.status)) {
        setTimeout(connectEvents, 3000)
      }
    }
  } catch (err) {
    console.error(t('runDetail.errors.sseCreateFailed', { msg: err.message || t('common.unknown') }), err)
  }
}

const closeEventSource = () => {
  if (evtSource) {
    evtSource.close()
    evtSource = null
  }
}

watch(logs, scrollLogToBottom)

onMounted(async () => {
  await loadData()
  await fetchSteps()
  await fetchLogs()
  await checkLLMConfig()
  if (!isTerminal(run.value.status)) {
    connectEvents()
  }
  logTimer = setInterval(() => {
    if (!isTerminal(run.value.status)) {
      fetchLogs()
      fetchSteps()
    }
  }, 5000)
})

onUnmounted(() => {
  closeEventSource()
  if (logTimer) clearInterval(logTimer)
  if (batchResultUrl.value) URL.revokeObjectURL(batchResultUrl.value)
})
</script>

<style scoped>
.run-detail {
  padding: 20px;
}

.info-card,
.result-card,
.progress-card {
  margin-top: 20px;
}

.progress-steps {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin: 16px 0;
}

.progress-step {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: 4px;
  background-color: #f5f7fa;
  border: 1px solid #ebeef5;
}

.progress-step.running {
  background-color: #fdf6ec;
  border-color: #f5dab1;
}

.progress-step.failed {
  background-color: #fef0f0;
  border-color: #fde2e2;
}

.progress-step.completed {
  background-color: #f0f9eb;
  border-color: #d1edc4;
}

.step-icon {
  font-size: 16px;
}

.step-icon.success {
  color: #67c23a;
}

.step-icon.running {
  color: #e6a23c;
}

.step-icon.failed {
  color: #f56c6c;
}

.step-icon.pending {
  color: #c0c4cc;
}

.step-name {
  font-size: 13px;
  color: #606266;
  text-transform: capitalize;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.error-text {
  color: #f56c6c;
}

.tab-content {
  padding: 12px 4px;
}

.section-title {
  margin: 20px 0 14px;
  color: #303133;
  border-left: 4px solid #409eff;
  padding-left: 10px;
  font-size: 16px;
}

.metric-row {
  margin-bottom: 8px;
}

.leaderboard-controls {
  margin-bottom: 12px;
}

.action-buttons {
  margin-bottom: 16px;
}

.report-iframe {
  width: 100%;
  height: 600px;
  border: 1px solid #ebeef5;
  border-radius: 4px;
  background-color: #fff;
}

.predict-result {
  margin-top: 15px;
  padding: 15px;
  background-color: #f5f7fa;
  border-radius: 4px;
}

.predict-result pre {
  margin: 0;
  white-space: pre-wrap;
}

.batch-result {
  margin-top: 16px;
}

.interpretation-card {
  background-color: #f7f9fc;
}

.interpretation-summary {
  font-size: 15px;
  line-height: 1.8;
  color: #303133;
  margin-bottom: 16px;
}

.interpretation-section {
  margin-top: 16px;
}

.interpretation-section h4 {
  margin: 12px 0 8px;
  color: #409eff;
  font-size: 14px;
}

.interpretation-section ul {
  margin: 0;
  padding-left: 20px;
}

.interpretation-section li {
  margin-bottom: 6px;
  line-height: 1.6;
  color: #606266;
}

.metric-item {
  text-align: center;
  padding: 15px;
  background-color: #f5f7fa;
  border-radius: 4px;
}

.metric-label {
  font-size: 12px;
  color: #909399;
  margin-bottom: 8px;
}

.metric-value {
  font-size: 20px;
  font-weight: bold;
  color: #409eff;
}

.log-textarea :deep(textarea) {
  font-family: 'Courier New', monospace;
  font-size: 12px;
}
</style>
