<template>
  <div class="run-detail">
    <el-page-header @back="$router.push('/runs')" title="任务详情">
      <template #extra>
        <el-button size="small" @click="$router.push(`/runs/${run.id}/pipeline`)">
          查看 Pipeline
        </el-button>
      </template>
    </el-page-header>

    <el-card v-loading="loading" class="info-card">
      <template #header>
        <div class="card-header">
          <span>任务信息</span>
          <el-tag :type="statusType(run.status)">{{ statusLabel(run.status) }}</el-tag>
        </div>
      </template>

      <el-descriptions :column="2" border>
        <el-descriptions-item label="任务ID">{{ run.id }}</el-descriptions-item>
        <el-descriptions-item label="数据集">{{ run.dataset_id }}</el-descriptions-item>
        <el-descriptions-item label="目标列">{{ run.config?.target_column || '-' }}</el-descriptions-item>
        <el-descriptions-item label="任务类型">{{ taskTypeLabel(run.config?.task_type) }}</el-descriptions-item>
        <el-descriptions-item label="评估指标">{{ run.primary_metric || '-' }}</el-descriptions-item>
        <el-descriptions-item label="时间预算">{{ run.time_budget_minutes ?? '无限制' }} min</el-descriptions-item>
        <el-descriptions-item label="随机种子">{{ run.config?.seed ?? '-' }}</el-descriptions-item>
        <el-descriptions-item label="创建时间">{{ formatDate(run.created_at) }}</el-descriptions-item>
        <el-descriptions-item label="完成时间" v-if="run.completed_at">{{ formatDate(run.completed_at) }}</el-descriptions-item>
        <el-descriptions-item label="错误信息" v-if="run.error_message">
          <span class="error-text">{{ run.error_message }}</span>
        </el-descriptions-item>
        <el-descriptions-item label="错误类型" v-if="results?.error_details?.error_type">
          <span class="error-text">{{ results.error_details.error_type }}</span>
        </el-descriptions-item>
      </el-descriptions>
    </el-card>

    <el-card v-if="!isTerminal(run.status) || run.status === 'failed'" class="progress-card">
      <template #header>
        <div class="card-header">
          <span>训练进度</span>
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

      <h3 class="section-title">实时日志</h3>
      <el-input
        ref="logTextareaRef"
        v-model="logs"
        type="textarea"
        :rows="12"
        readonly
        placeholder="日志加载中..."
        class="log-textarea"
      />
    </el-card>

    <el-card v-if="results" class="result-card">
      <el-tabs v-model="activeTab" type="border-card">
        <el-tab-pane label="概览" name="overview">
          <div class="tab-content">
            <h3 class="section-title">测试集评估指标</h3>
            <el-row :gutter="16" class="metric-row">
              <el-col :xs="12" :sm="8" :md="6" :lg="4" v-for="(value, key) in results.metrics" :key="key">
                <el-statistic :title="key" :value="typeof value === 'number' ? value : NaN" :precision="4" />
              </el-col>
            </el-row>

            <template v-if="results.train_metrics && Object.keys(results.train_metrics).length > 0">
              <h3 class="section-title">训练集参考指标</h3>
              <el-row :gutter="16" class="metric-row">
                <el-col :xs="12" :sm="8" :md="6" :lg="4" v-for="(value, key) in results.train_metrics" :key="key">
                  <el-statistic :title="key" :value="typeof value === 'number' ? value : NaN" :precision="4" />
                </el-col>
              </el-row>
            </template>

            <template v-if="results.cv_results && results.cv_results.cv_scores">
              <h3 class="section-title">交叉验证</h3>
              <el-row :gutter="16" class="metric-row">
                <el-col :xs="12" :sm="6">
                  <div class="metric-item">
                    <div class="metric-label">CV 类型</div>
                    <div class="metric-value">{{ results.cv_results.cv_type }}</div>
                  </div>
                </el-col>
                <el-col :xs="12" :sm="6">
                  <el-statistic title="折数" :value="results.cv_results.n_folds" />
                </el-col>
                <el-col :xs="12" :sm="6">
                  <el-statistic title="CV 均值" :value="results.cv_results.cv_mean" :precision="4" />
                </el-col>
                <el-col :xs="12" :sm="6">
                  <el-statistic title="CV 标准差" :value="results.cv_results.cv_std" :precision="4" />
                </el-col>
              </el-row>
              <el-alert
                v-if="results.cv_results.cv_error"
                :title="'CV 计算失败: ' + results.cv_results.cv_error"
                type="warning"
                :closable="false"
              />
            </template>

            <template v-if="extendedMetricItems.length > 0">
              <h3 class="section-title">扩展评估指标</h3>
              <el-row :gutter="16" class="metric-row">
                <el-col :xs="12" :sm="8" :md="6" :lg="4" v-for="item in extendedMetricItems" :key="item.key">
                  <el-statistic :title="item.label" :value="Number(item.value)" :precision="4" />
                </el-col>
              </el-row>
            </template>

            <template v-if="results.business_interpretation">
              <h3 class="section-title">业务解读摘要</h3>
              <el-card shadow="never" class="interpretation-card">
                <el-alert
                  v-if="results.business_interpretation.provider === 'rule_template'"
                  title="当前为规则模板生成的兜底解读"
                  type="info"
                  :closable="false"
                  style="margin-bottom: 12px;"
                >
                  <template #default>
                    <div>
                      当前为规则模板生成的兜底解读。若需 LLM 智能解读，请主动点击生成。
                      <el-button
                        v-if="llmConfigured"
                        link
                        type="primary"
                        :loading="regeneratingInterpretation"
                        @click="openDisclaimer"
                      >
                        生成 LLM 业务解读
                      </el-button>
                      <el-button
                        v-else
                        link
                        type="primary"
                        @click="llmDialogVisible = true"
                      >
                        配置 LLM API Key
                      </el-button>
                    </div>
                  </template>
                </el-alert>
                <p class="interpretation-summary">
                  {{ results.business_interpretation.business_summary }}
                </p>
                <div v-if="results.business_interpretation.key_insights?.length" class="interpretation-section">
                  <h4>关键发现</h4>
                  <ul>
                    <li v-for="(item, idx) in results.business_interpretation.key_insights" :key="'insight-' + idx">
                      {{ item }}
                    </li>
                  </ul>
                </div>
                <div v-if="results.business_interpretation.feature_interpretations?.length" class="interpretation-section">
                  <h4>Top 特征业务含义</h4>
                  <ul>
                    <li v-for="(item, idx) in results.business_interpretation.feature_interpretations" :key="'feature-' + idx">
                      <strong>{{ item.feature || item }}</strong>
                      <span v-if="item.interpretation">：{{ item.interpretation }}</span>
                    </li>
                  </ul>
                </div>
                <div v-if="results.business_interpretation.caveats?.length" class="interpretation-section">
                  <h4>使用注意</h4>
                  <ul>
                    <li v-for="(item, idx) in results.business_interpretation.caveats" :key="'caveat-' + idx">
                      {{ item }}
                    </li>
                  </ul>
                </div>
                <div v-if="results.business_interpretation.recommendations?.length" class="interpretation-section">
                  <h4>下一步建议</h4>
                  <ul>
                    <li v-for="(item, idx) in results.business_interpretation.recommendations" :key="'rec-' + idx">
                      {{ item }}
                    </li>
                  </ul>
                </div>
              </el-card>
            </template>

            <template v-if="results?.error_details?.traceback">
              <h3 class="section-title">错误堆栈</h3>
              <el-input v-model="results.error_details.traceback" type="textarea" :rows="10" readonly />
            </template>
          </div>
        </el-tab-pane>

        <el-tab-pane label="排行榜" name="leaderboard">
          <div class="tab-content">
            <h3 class="section-title">模型排行榜</h3>
            <div class="leaderboard-controls">
              <el-select v-model="familyFilter" placeholder="模型族" style="width: 140px">
                <el-option label="全部" value="all" />
                <el-option v-for="f in families" :key="f" :label="f" :value="f" />
              </el-select>
              <el-select v-model="sortBy" placeholder="排序依据" style="width: 160px; margin-left: 10px">
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

            <h3 class="section-title">特征重要性 Top 10</h3>
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

        <el-tab-pane label="特征重要性" name="features">
          <div class="tab-content">
            <h3 class="section-title">特征重要性 Top 15</h3>
            <EChart :option="featureImportanceOption" height="420px" />

            <template v-if="results.permutation_importance && results.permutation_importance.length">
              <h3 class="section-title">Permutation Importance Top 15</h3>
              <EChart :option="permutationImportanceOption" height="420px" />
            </template>
          </div>
        </el-tab-pane>

        <el-tab-pane label="混淆矩阵" name="confusion" v-if="confusionMatrix">
          <div class="tab-content">
            <h3 class="section-title">混淆矩阵</h3>
            <el-row :gutter="20">
              <el-col :xs="24" :md="12">
                <EChart :option="confusionMatrixOption" height="420px" />
              </el-col>
              <el-col :xs="24" :md="12">
                <el-table :data="confusionMatrixRows" border style="width: 100%;">
                  <el-table-column label="真实 \ 预测" prop="label" width="120" />
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

        <el-tab-pane label="报告" name="report">
          <div class="tab-content">
            <div class="action-buttons">
              <el-button type="primary" @click="downloadReport">下载报告</el-button>
              <el-button v-if="run.status === 'completed'" type="success" @click="downloadModel">下载模型</el-button>
            </div>
            <iframe v-if="results.report_path" :src="`/api/runs/${runId}/report`" class="report-iframe"></iframe>
            <el-empty v-else description="报告尚未生成" />
          </div>
        </el-tab-pane>

        <el-tab-pane label="预测" name="predict">
          <div class="tab-content">
            <el-row :gutter="24">
              <el-col :xs="24" :md="12">
                <el-card shadow="never">
                  <template #header>
                    <span>单条 / 批量 JSON 预测</span>
                  </template>
                  <el-alert
                    title="请输入 JSON 格式的数据数组"
                    type="info"
                    description='例如：[{"feature1": 1, "feature2": 2}]'
                    show-icon
                    :closable="false"
                  />
                  <el-input
                    v-model="predictInput"
                    type="textarea"
                    :rows="6"
                    placeholder='[{"feature1": 1, "feature2": 2}]'
                    style="margin-top: 15px;"
                  />
                  <div v-if="predictResult" class="predict-result">
                    <h4>预测结果：</h4>
                    <pre>{{ JSON.stringify(predictResult, null, 2) }}</pre>
                  </div>
                  <div style="margin-top: 12px; text-align: right;">
                    <el-button type="primary" @click="submitPredict" :loading="predicting">预测</el-button>
                  </div>
                </el-card>
              </el-col>
              <el-col :xs="24" :md="12">
                <el-card shadow="never">
                  <template #header>
                    <span>批量 CSV 预测</span>
                  </template>
                  <el-upload
                    drag
                    :auto-upload="false"
                    :limit="1"
                    accept=".csv"
                    @change="handleBatchFileChange"
                  >
                    <el-icon class="el-icon--upload"><upload-icon /></el-icon>
                    <div class="el-upload__text">拖拽 CSV 文件到此处，或 <em>点击上传</em></div>
                  </el-upload>
                  <div v-if="batchResultUrl" class="batch-result">
                    <el-alert title="批量预测完成" type="success" :closable="false" />
                    <el-button type="primary" :href="batchResultUrl" download="predictions.csv" tag="a" style="margin-top: 12px;">
                      下载预测结果
                    </el-button>
                  </div>
                  <div style="margin-top: 12px; text-align: right;">
                    <el-button type="success" @click="submitBatchPredict" :loading="batchPredicting" :disabled="!batchFile">
                      提交批量预测
                    </el-button>
                  </div>
                </el-card>
              </el-col>
            </el-row>
          </div>
        </el-tab-pane>

        <el-tab-pane label="日志" name="logs">
          <div class="tab-content">
            <el-input
              ref="logTextareaRef"
              v-model="logs"
              type="textarea"
              :rows="20"
              readonly
              placeholder="日志加载中..."
              class="log-textarea"
            />
          </div>
        </el-tab-pane>
      </el-tabs>
    </el-card>

    <el-card v-else class="result-card">
      <el-empty :description="run.status === 'running' ? '训练进行中，结果将在完成后自动加载' : '暂无训练结果'" />
    </el-card>
  </div>

  <LLMSettingsDialog v-model:visible="llmDialogVisible" @saved="onLLMSettingsSaved" />

  <el-dialog
    v-model="disclaimerVisible"
    title="数据外传风险提示"
    width="480px"
    :close-on-click-modal="false"
  >
    <el-alert
      type="warning"
      :closable="false"
      show-icon
    >
      <template #title>
        <span>生成 LLM 业务解读将把训练摘要数据发送到第三方 LLM 服务</span>
      </template>
      <p>发送内容主要包括：任务类型、评估指标、Top 特征名称及重要性、数据质量摘要等。原始数据行不会被发送，但特征名称、指标数值等仍可能包含业务敏感信息，存在泄露风险。</p>
    </el-alert>
    <p style="margin-top: 16px;">请确认是否继续生成？</p>
    <template #footer>
      <el-button @click="disclaimerVisible = false">取消</el-button>
      <el-button type="primary" :loading="regeneratingInterpretation" @click="confirmLLMInterpretation">
        确认生成
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
import LLMSettingsDialog from '@/components/LLMSettingsDialog.vue'

const route = useRoute()
const runId = route.params.id

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
  binary_classification: '二分类',
  multiclass_classification: '多分类',
  regression: '回归',
}

function taskTypeLabel(type) {
  return taskTypeLabels[type] || type || '-'
}

function statusLabel(status) {
  const map = {
    pending: '等待中',
    running: '运行中',
    completed: '已完成',
    failed: '失败',
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
  if (n.includes('lightgbm') || n.includes('catboost') || n.includes('xgboost')) return 'GBDT'
  if (n.includes('randomforest') || n.includes('extratrees') || n.includes('xt_')) return '树集成'
  if (n.includes('neuralnet') || n.includes('torch') || n.includes('mlp') || n.includes('fastai')) return '神经网络'
  if (n.includes('kneighbors') || n.includes('knn')) return 'KNN'
  if (n.includes('linearmodel') || n.includes('ridge') || n.includes('elastic') || n.includes('logistic')) return '线性模型'
  return '其他'
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
  precision_macro: 'Precision (Macro)',
  precision_weighted: 'Precision (Weighted)',
  precision_micro: 'Precision (Micro)',
  recall_macro: 'Recall (Macro)',
  recall_weighted: 'Recall (Weighted)',
  recall_micro: 'Recall (Micro)',
  f1_macro: 'F1 (Macro)',
  f1_weighted: 'F1 (Weighted)',
  f1_micro: 'F1 (Micro)',
  mcc: 'MCC',
  cohens_kappa: "Cohen's Kappa",
  balanced_accuracy: 'Balanced Accuracy',
  auc_roc: 'AUC-ROC',
  auc_pr: 'AUC-PR',
  mae: 'MAE',
  mse: 'MSE',
  rmse: 'RMSE',
  r2: 'R²',
  mape: 'MAPE (%)',
  smape: 'SMAPE (%)',
  residual_mean: '残差均值',
  residual_std: '残差标准差',
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
    xAxis: { type: 'category', data: labels, name: '预测' },
    yAxis: { type: 'category', data: labels, name: '真实' },
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
  const names = sorted.map((row) => row.feature || row['Unnamed: 0'] || row[''] || '未知')
  const values = sorted.map((row) => row.importance)

  return {
    tooltip: { trigger: 'axis' },
    grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
    xAxis: { type: 'value', name: '重要性' },
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
  const names = sorted.map((row) => row.feature || row['Unnamed: 0'] || '未知')
  const values = sorted.map((row) => row.importance)

  return {
    tooltip: { trigger: 'axis' },
    grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
    xAxis: { type: 'value', name: 'Permutation Importance' },
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
    ElMessage.error('获取任务失败: ' + error.message)
  }
}

const fetchResults = async () => {
  if (run.value.status !== 'completed') return
  try {
    const res = await runApi.getResults(runId)
    results.value = res.data
  } catch (error) {
    console.error('获取结果失败:', error)
  }
}

const checkLLMConfig = async () => {
  try {
    const res = await llmSettingsApi.get()
    llmConfigured.value = !!(res.data?.provider && res.data?.api_key_masked)
  } catch (error) {
    llmConfigured.value = false
    console.error('检查 LLM 配置失败:', error)
  }
}

const regenerateInterpretation = async () => {
  regeneratingInterpretation.value = true
  try {
    const res = await runApi.regenerateInterpretation(runId)
    results.value = { ...results.value, business_interpretation: res.data }
    ElMessage.success('业务解读已重新生成')
  } catch (error) {
    ElMessage.error('重新生成失败: ' + (error.message || '未知错误'))
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
    console.error('获取日志失败:', error)
  }
}

const fetchSteps = async () => {
  try {
    const res = await runApi.getSteps(runId)
    steps.value = res.data
  } catch (error) {
    console.error('获取步骤状态失败:', error)
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
    ElMessage.warning('请输入预测数据')
    return
  }

  predicting.value = true
  try {
    const data = JSON.parse(predictInput.value)
    const res = await runApi.predict(runId, { data })
    predictResult.value = res.data
  } catch (error) {
    ElMessage.error('预测失败: ' + error.message)
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
    ElMessage.warning('请选择 CSV 文件')
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
    ElMessage.success('批量预测完成')
  } catch (error) {
    ElMessage.error('批量预测失败: ' + (error.message || '未知错误'))
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
        console.error('解析 SSE 消息失败:', err)
      }
    }
    evtSource.onerror = (err) => {
      console.error('SSE 连接错误，3 秒后重连:', err)
      closeEventSource()
      if (!isTerminal(run.value.status)) {
        setTimeout(connectEvents, 3000)
      }
    }
  } catch (err) {
    console.error('创建 EventSource 失败:', err)
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
