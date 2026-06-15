<template>
  <div class="run-detail">
    <el-page-header @back="$router.push('/runs')" title="任务详情" />

    <el-card v-loading="loading" class="info-card">
      <template #header>
        <div class="card-header">
          <span>任务信息</span>
          <el-tag :type="statusType(run.status)">{{ run.status }}</el-tag>
        </div>
      </template>

      <el-descriptions :column="2" border>
        <el-descriptions-item label="任务ID">{{ run.id }}</el-descriptions-item>
        <el-descriptions-item label="数据集ID">{{ run.dataset_id }}</el-descriptions-item>
        <el-descriptions-item label="状态">{{ run.status }}</el-descriptions-item>
        <el-descriptions-item label="时间预算">{{ run.time_budget_minutes }} 分钟</el-descriptions-item>
        <el-descriptions-item label="随机种子">{{ run.config?.seed ?? '-' }}</el-descriptions-item>
        <el-descriptions-item label="评估指标">{{ run.primary_metric || '-' }}</el-descriptions-item>
        <el-descriptions-item label="创建时间">{{ formatDate(run.created_at) }}</el-descriptions-item>
        <el-descriptions-item label="完成时间">{{ formatDate(run.completed_at) || '-' }}</el-descriptions-item>
        <el-descriptions-item label="错误信息" v-if="run.error_message">
          <span style="color: #f56c6c;">{{ run.error_message }}</span>
        </el-descriptions-item>
      </el-descriptions>
    </el-card>

    <el-card v-if="results" class="result-card">
      <template #header>
        <span>训练结果</span>
      </template>

      <h3>评估指标</h3>
      <el-row :gutter="20" class="metric-row">
        <el-col :span="4" v-for="(value, key) in results.metrics" :key="key">
          <div class="metric-item">
            <div class="metric-label">{{ key }}</div>
            <div class="metric-value">{{ typeof value === 'number' ? value.toFixed(4) : value }}</div>
          </div>
        </el-col>
      </el-row>

      <template v-if="results.extended_metrics && extendedMetricItems.length > 0">
        <h3>扩展评估指标</h3>
        <el-row :gutter="20" class="metric-row">
          <el-col :span="4" v-for="item in extendedMetricItems" :key="item.key">
            <div class="metric-item">
              <div class="metric-label">{{ item.label }}</div>
              <div class="metric-value">{{ item.value }}</div>
            </div>
          </el-col>
        </el-row>

        <template v-if="confusionMatrix">
          <h3>混淆矩阵</h3>
          <el-row :gutter="20">
            <el-col :xs="24" :md="12">
              <EChart :option="confusionMatrixOption" height="360px" />
            </el-col>
            <el-col :xs="24" :md="12">
              <el-table :data="confusionMatrixRows" border style="width: 100%;">
                <el-table-column label="真实 \\ 预测" prop="label" width="120" />
                <el-table-column
                  v-for="(label, idx) in confusionMatrixLabels"
                  :key="idx"
                  :label="label"
                  :prop="'pred_' + idx"
                />
              </el-table>
            </el-col>
          </el-row>
        </template>
      </template>

      <h3>特征重要性 Top 15</h3>
      <EChart :option="featureImportanceOption" height="360px" />

      <h3>模型排行榜</h3>
      <el-table :data="results.leaderboard" style="width: 100%" max-height="300px">
        <el-table-column
          v-for="col in leaderboardColumns"
          :key="col"
          :prop="col"
          :label="col"
        />
      </el-table>

      <h3>特征重要性 Top 10</h3>
      <el-table :data="results.feature_importance.slice(0, 10)" style="width: 100%" max-height="300px">
        <el-table-column
          v-for="col in importanceColumns"
          :key="col"
          :prop="col"
          :label="col"
        />
      </el-table>

      <div class="action-buttons">
        <el-button type="primary" @click="showPredictDialog = true">使用模型预测</el-button>
        <el-button v-if="results.report_path" @click="downloadReport">下载报告</el-button>
        <el-button v-if="run.status === 'completed'" type="success" @click="downloadModel">
          下载模型
        </el-button>
      </div>

      <template v-if="results.report_path">
        <h3>报告预览</h3>
        <iframe :src="`/api/runs/${runId}/report`" class="report-iframe"></iframe>
      </template>

      <h3>训练日志</h3>
      <el-input
        v-model="logs"
        type="textarea"
        :rows="10"
        readonly
        placeholder="日志加载中..."
      />
    </el-card>

    <!-- 预测对话框 -->
    <el-dialog v-model="showPredictDialog" title="模型预测" width="600px">
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
      <template #footer>
        <el-button @click="showPredictDialog = false">关闭</el-button>
        <el-button type="primary" @click="submitPredict" :loading="predicting">预测</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import api, { runApi } from '@/api'
import EChart from '@/components/EChart.vue'

const route = useRoute()
const runId = route.params.id

const run = ref({})
const results = ref(null)
const loading = ref(false)
const showPredictDialog = ref(false)
const predictInput = ref('')
const predictResult = ref(null)
const predicting = ref(false)
const logs = ref('')
let timer = null

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
  if (!results.value || !results.value.leaderboard.length) return []
  return Object.keys(results.value.leaderboard[0])
})

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
  const names = sorted.map((row) => row[''] || row.feature || '未知')
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

const fetchLogs = async () => {
  try {
    const res = await api.get(`/runs/${runId}/logs`)
    logs.value = res.data
  } catch (error) {
    console.error('获取日志失败:', error)
  }
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

const downloadReport = () => {
  window.open(`/api/runs/${runId}/report`, '_blank')
}

const downloadModel = () => {
  window.open(`/api/runs/${runId}/model`, '_blank')
}

const formatDate = (date) => {
  if (!date) return null
  return new Date(date).toLocaleString()
}

onMounted(() => {
  loadData()
  fetchLogs()
  timer = setInterval(async () => {
    await fetchRun()
    await fetchLogs()
    if (run.value.status === 'completed' && !results.value) {
      await fetchResults()
    }
  }, 3000)
})

onUnmounted(() => {
  if (timer) clearInterval(timer)
})
</script>

<style scoped>
.run-detail {
  padding: 20px 0;
}

.info-card,
.result-card {
  margin-top: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.metric-row {
  margin: 20px 0;
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

.result-card h3 {
  margin-top: 30px;
  margin-bottom: 15px;
  color: #303133;
  border-left: 4px solid #409eff;
  padding-left: 10px;
}

.action-buttons {
  margin-top: 30px;
  text-align: center;
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

.report-iframe {
  width: 100%;
  height: 600px;
  border: 1px solid #ebeef5;
  border-radius: 4px;
  background-color: #fff;
}
</style>
