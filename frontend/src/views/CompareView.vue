<!--
Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
See LICENSE for details.
-->

<template>
  <div class="compare-page">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>{{ $t('compare.title') }}</span>
          <el-button type="primary" @click="runCompare" :loading="loading" :disabled="selectedRuns.length < 2">
            {{ $t('compare.start') }}
          </el-button>
        </div>
      </template>

      <el-alert
        :title="$t('compare.selectHint')"
        type="info"
        :closable="false"
        style="margin-bottom: 16px;"
      />

      <div class="table-toolbar">
        <el-input
          v-model="searchQuery"
          :placeholder="$t('compare.searchPlaceholder')"
          style="width: 300px"
          clearable
        />
      </div>

      <el-table
        :data="pagedRuns"
        @selection-change="handleSelectionChange"
        v-loading="runsLoading"
        style="width: 100%"
        max-height="360px"
      >
        <el-table-column type="selection" width="55" />
        <el-table-column prop="id" :label="$t('compare.columns.runId')" width="220" sortable />
        <el-table-column prop="dataset_name" :label="$t('compare.columns.dataset')" sortable />
        <el-table-column prop="status" :label="$t('compare.columns.status')" sortable>
          <template #default="{ row }">
            <el-tag :type="statusType(row.status)">{{ row.status }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="primary_metric" :label="$t('compare.columns.primaryMetric')" />
        <el-table-column :label="$t('common.actions')" width="120">
          <template #default="{ row }">
            <el-button size="small" @click="viewDetail(row)">{{ $t('compare.detail') }}</el-button>
          </template>
        </el-table-column>
        <template #empty>
          <el-empty :description="$t('compare.empty')" />
        </template>
      </el-table>

      <el-pagination
        v-model:current-page="currentPage"
        v-model:page-size="pageSize"
        :page-sizes="[10, 20, 50]"
        :total="filteredRuns.length"
        layout="total, sizes, prev, pager, next"
        style="margin-top: 16px; justify-content: flex-end"
      />
    </el-card>

    <el-card v-if="comparison" class="result-card">
      <template #header>
        <span>{{ $t('compare.result.title') }}</span>
      </template>

      <el-alert
        v-if="comparison.best_run_id"
        :title="$t('compare.result.bestRun', { runId: comparison.best_run_id }) + (comparison.metric_name ? $t('compare.result.byMetric', { metric: comparison.metric_name }) : '')"
        type="success"
        :closable="false"
        style="margin-bottom: 16px;"
      />
      <el-alert
        v-else
        :title="$t('compare.result.noBestRun')"
        type="warning"
        :closable="false"
        style="margin-bottom: 16px;"
      />

      <h3>{{ $t('compare.result.metricComparison') }}</h3>
      <el-table :data="comparison.runs" style="width: 100%" border>
        <el-table-column prop="run_id" :label="$t('compare.result.runId')" width="220" />
        <el-table-column prop="dataset_name" :label="$t('compare.result.dataset')" />
        <el-table-column prop="best_model" :label="$t('compare.result.bestModel')" min-width="160" show-overflow-tooltip>
          <template #default="{ row }">
            {{ row.best_model || $t('compare.result.noModelData') }}
          </template>
        </el-table-column>
        <el-table-column prop="best_model_score" :label="$t('compare.result.modelScore')" min-width="120">
          <template #default="{ row }">
            {{ row.best_model_score !== null && row.best_model_score !== undefined ? row.best_model_score.toFixed(4) : '-' }}
          </template>
        </el-table-column>
        <el-table-column prop="feature_count" :label="$t('compare.result.featureCount')" />
        <el-table-column v-for="key in metricKeys" :key="key" :prop="`metrics.${key}`" :label="key">
          <template #default="{ row }">
            {{ row.metrics[key]?.toFixed(4) ?? '-' }}
          </template>
        </el-table-column>
      </el-table>

      <h3>{{ $t('compare.result.visualization') }}</h3>
      <EChart v-if="chartOption" :option="chartOption" height="360px" />
      <el-empty v-else-if="comparison" :description="$t('compare.result.noCommonMetric')" />
    </el-card>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { ElMessage } from 'element-plus'
import { runApi } from '@/api'
import EChart from '@/components/EChart.vue'

const router = useRouter()
const { t } = useI18n()
const runs = ref([])
const runsLoading = ref(false)
const selectedRuns = ref([])
const loading = ref(false)
const comparison = ref(null)
const searchQuery = ref('')
const currentPage = ref(1)
const pageSize = ref(10)

const statusType = (status) => {
  const map = { pending: 'info', running: 'warning', completed: 'success', failed: 'danger' }
  return map[status] || 'info'
}

const filteredRuns = computed(() => {
  const q = searchQuery.value.trim().toLowerCase()
  if (!q) return runs.value
  return runs.value.filter((r) => r.id.toLowerCase().includes(q) || r.dataset_name.toLowerCase().includes(q))
})

const pagedRuns = computed(() => {
  const start = (currentPage.value - 1) * pageSize.value
  return filteredRuns.value.slice(start, start + pageSize.value)
})

const metricKeys = computed(() => {
  if (!comparison.value?.runs.length) return []
  const keys = new Set()
  comparison.value.runs.forEach((row) => {
    Object.keys(row.metrics).forEach((k) => keys.add(k))
  })
  return Array.from(keys)
})

const chartOption = computed(() => {
  if (!comparison.value?.runs.length) return null
  const metric = comparison.value.metric_name
  if (!metric) {
    // 无共同指标时给出占位提示
    return null
  }
  const data = comparison.value.runs.map((row) => ({
    name: row.run_id.slice(0, 8),
    value: row.metrics[metric] ?? 0,
  }))
  return {
    tooltip: { trigger: 'axis' },
    grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
    xAxis: { type: 'category', data: data.map((d) => d.name) },
    yAxis: { type: 'value', name: metric },
    series: [{ data: data.map((d) => d.value), type: 'bar', itemStyle: { color: '#409eff' } }],
  }
})

const fetchRuns = async () => {
  runsLoading.value = true
  try {
    const res = await runApi.list()
    runs.value = res.data
      .filter((run) => run.status === 'completed')
      .map((run) => ({
        ...run,
        dataset_name: run.config?.snapshot?.dataset_name || run.dataset_id,
      }))
  } catch (error) {
    ElMessage.error(t('compare.errors.fetchFailed', { msg: error.message }))
  } finally {
    runsLoading.value = false
  }
}

const handleSelectionChange = (rows) => {
  selectedRuns.value = rows
}

const runCompare = async () => {
  if (selectedRuns.value.length < 2) {
    ElMessage.warning(t('compare.validation.minTwo'))
    return
  }
  loading.value = true
  try {
    const res = await runApi.compare({ run_ids: selectedRuns.value.map((r) => r.id) })
    comparison.value = res.data
  } catch (error) {
    ElMessage.error(t('compare.errors.compareFailed', { msg: error.message }))
  } finally {
    loading.value = false
  }
}

const viewDetail = (row) => {
  router.push(`/runs/${row.id}`)
}

onMounted(fetchRuns)
</script>

<style scoped>
.compare-page {
  padding: 20px 0;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.table-toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.result-card {
  margin-top: 20px;
}

.result-card h3 {
  margin-top: 24px;
  margin-bottom: 12px;
  color: #303133;
  border-left: 4px solid #409eff;
  padding-left: 10px;
}
</style>
