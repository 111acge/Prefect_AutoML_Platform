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
          <span>跨 Run 模型对比</span>
          <el-button type="primary" @click="runCompare" :loading="loading" :disabled="selectedRuns.length < 2">
            开始对比
          </el-button>
        </div>
      </template>

      <el-alert
        title="请选择 2~10 个已完成的训练任务进行对比（仅显示已完成任务）"
        type="info"
        :closable="false"
        style="margin-bottom: 16px;"
      />

      <div class="table-toolbar">
        <el-input
          v-model="searchQuery"
          placeholder="搜索任务 ID 或数据集"
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
        <el-table-column prop="id" label="Run ID" width="220" sortable />
        <el-table-column prop="dataset_name" label="数据集" sortable />
        <el-table-column prop="status" label="状态" sortable>
          <template #default="{ row }">
            <el-tag :type="statusType(row.status)">{{ row.status }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="primary_metric" label="主指标" />
        <el-table-column label="操作" width="120">
          <template #default="{ row }">
            <el-button size="small" @click="viewDetail(row)">详情</el-button>
          </template>
        </el-table-column>
        <template #empty>
          <el-empty description="暂无已完成的训练任务，请先去训练任务页完成训练" />
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
        <span>对比结果</span>
      </template>

      <el-alert
        v-if="comparison.best_run_id"
        :title="`最佳 Run: ${comparison.best_run_id}${comparison.metric_name ? '（按 ' + comparison.metric_name + '）' : ''}`"
        type="success"
        :closable="false"
        style="margin-bottom: 16px;"
      />
      <el-alert
        v-else
        title="无法判定最佳 Run：缺少可比较的指标或模型评分"
        type="warning"
        :closable="false"
        style="margin-bottom: 16px;"
      />

      <h3>指标对比</h3>
      <el-table :data="comparison.runs" style="width: 100%" border>
        <el-table-column prop="run_id" label="Run ID" width="220" />
        <el-table-column prop="dataset_name" label="数据集" />
        <el-table-column prop="best_model" label="最佳模型" min-width="160" show-overflow-tooltip>
          <template #default="{ row }">
            {{ row.best_model || '无模型数据' }}
          </template>
        </el-table-column>
        <el-table-column prop="best_model_score" label="模型评分" min-width="120">
          <template #default="{ row }">
            {{ row.best_model_score !== null && row.best_model_score !== undefined ? row.best_model_score.toFixed(4) : '-' }}
          </template>
        </el-table-column>
        <el-table-column prop="feature_count" label="特征数" />
        <el-table-column v-for="key in metricKeys" :key="key" :prop="`metrics.${key}`" :label="key">
          <template #default="{ row }">
            {{ row.metrics[key]?.toFixed(4) ?? '-' }}
          </template>
        </el-table-column>
      </el-table>

      <h3>主指标可视化</h3>
      <EChart v-if="chartOption" :option="chartOption" height="360px" />
      <el-empty v-else-if="comparison" description="缺少共同主指标，无法绘制对比图表" />
    </el-card>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { runApi } from '@/api'
import EChart from '@/components/EChart.vue'

const router = useRouter()
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
    ElMessage.error('获取任务列表失败: ' + error.message)
  } finally {
    runsLoading.value = false
  }
}

const handleSelectionChange = (rows) => {
  selectedRuns.value = rows
}

const runCompare = async () => {
  if (selectedRuns.value.length < 2) {
    ElMessage.warning('请至少选择 2 个任务')
    return
  }
  loading.value = true
  try {
    const res = await runApi.compare({ run_ids: selectedRuns.value.map((r) => r.id) })
    comparison.value = res.data
  } catch (error) {
    ElMessage.error('对比失败: ' + error.message)
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
