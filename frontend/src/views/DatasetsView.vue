<!--
Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
See LICENSE for details.
-->

<template>
  <div class="datasets">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>数据集管理</span>
          <div>
            <el-button type="success" @click="trainDefaultDataset" :loading="trainingDefault">
              <el-icon><component :is="Cpu" /></el-icon> 使用默认数据集训练
            </el-button>
            <el-button type="primary" @click="showUploadDialog = true">
              <el-icon><component :is="Upload" /></el-icon> 上传数据集
            </el-button>
          </div>
        </div>
      </template>

      <div class="table-toolbar">
        <el-input
          v-model="searchQuery"
          placeholder="搜索数据集名称"
          style="width: 260px"
          clearable
        />
        <el-button
          type="danger"
          :disabled="selectedDatasets.length === 0"
          @click="batchDeleteDatasets"
        >
          批量删除 ({{ selectedDatasets.length }})
        </el-button>
      </div>

      <el-table
        :data="pagedDatasets"
        v-loading="loading"
        style="width: 100%"
        @selection-change="handleSelectionChange"
      >
        <el-table-column type="selection" width="55" />
        <el-table-column prop="name" label="名称" sortable />
        <el-table-column prop="row_count" label="行数" sortable />
        <el-table-column prop="column_count" label="列数" sortable />
        <el-table-column prop="file_size_bytes" label="大小" sortable>
          <template #default="{ row }">
            {{ formatSize(row.file_size_bytes) }}
          </template>
        </el-table-column>
        <el-table-column prop="created_at" label="创建时间" sortable>
          <template #default="{ row }">
            {{ formatDate(row.created_at) }}
          </template>
        </el-table-column>
        <el-table-column label="操作" width="320">
          <template #default="{ row }">
            <el-button size="small" @click="previewDataset(row)">预览</el-button>
            <el-tooltip content="数据质量检测报告">
              <el-button size="small" type="info" @click="showQuality(row)">质量报告</el-button>
            </el-tooltip>
            <el-button size="small" type="primary" @click="startTraining(row)">训练</el-button>
            <el-button size="small" type="danger" @click="deleteDataset(row)">删除</el-button>
          </template>
        </el-table-column>
        <template #empty>
          <el-empty description="暂无数据集，请先上传" />
        </template>
      </el-table>

      <el-pagination
        v-model:current-page="currentPage"
        v-model:page-size="pageSize"
        :page-sizes="[10, 20, 50]"
        :total="filteredDatasets.length"
        layout="total, sizes, prev, pager, next"
        style="margin-top: 16px; justify-content: flex-end"
      />
    </el-card>

    <!-- 上传对话框 -->
    <el-dialog v-model="showUploadDialog" title="上传数据集" width="500px">
      <el-form :model="uploadForm" label-width="80px">
        <el-form-item label="名称">
          <el-input v-model="uploadForm.name" placeholder="数据集名称" />
        </el-form-item>
        <el-form-item label="文件">
          <el-upload
            ref="uploadRef"
            :auto-upload="false"
            :on-change="handleFileChange"
            :limit="1"
            accept=".csv,.xlsx,.xls,.parquet,.jsonl,.json"
          >
            <el-button type="primary">选择文件</el-button>
          </el-upload>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showUploadDialog = false">取消</el-button>
        <el-button type="primary" @click="submitUpload" :loading="uploading">上传</el-button>
      </template>
    </el-dialog>

    <!-- 训练对话框 -->
    <el-dialog v-model="showTrainDialog" title="启动训练任务" width="520px">
      <TrainConfigForm
        :dataset="currentDataset"
        :loading="training"
        @submit="submitTrain"
        @cancel="showTrainDialog = false"
      />
    </el-dialog>

    <!-- 预览对话框 -->
    <el-dialog v-model="showPreviewDialog" title="数据预览" width="800px">
      <el-table :data="previewRows" style="width: 100%" max-height="400px">
        <el-table-column
          v-for="(col, index) in previewData.columns"
          :key="col"
          :prop="`col_${index}`"
          :label="col"
        />
      </el-table>
    </el-dialog>

    <!-- 质量报告对话框 -->
    <el-dialog v-model="showQualityDialog" title="数据质量报告" width="900px">
      <div v-if="qualityData" v-loading="qualityLoading">
        <el-row :gutter="20" class="quality-summary">
          <el-col :span="6">
            <div class="quality-stat">综合得分</div>
            <div class="quality-stat-value">{{ (qualityData.overall_score * 100).toFixed(2) }}%</div>
          </el-col>
          <el-col :span="6">
            <div class="quality-stat">样本数</div>
            <div class="quality-stat-value">{{ qualityData.n_rows }}</div>
          </el-col>
          <el-col :span="6">
            <div class="quality-stat">特征数</div>
            <div class="quality-stat-value">{{ qualityData.n_features }}</div>
          </el-col>
          <el-col :span="6">
            <div class="quality-stat">含缺失值行数</div>
            <div class="quality-stat-value">{{ qualityData.completeness?.rows_with_missing || 0 }}</div>
          </el-col>
        </el-row>

        <h3 class="quality-title">质量维度得分</h3>
        <el-row :gutter="20">
          <el-col :span="4" v-for="dim in dimensionList" :key="dim.key">
            <el-card shadow="hover" class="dim-card">
              <div class="quality-stat-value">{{ (dim.score * 100).toFixed(2) }}%</div>
              <div class="quality-stat">{{ dim.label }}</div>
            </el-card>
          </el-col>
        </el-row>

        <el-alert
          v-for="(warning, idx) in qualityData.warnings"
          :key="idx"
          :title="warning"
          type="warning"
          :closable="false"
          style="margin-top: 10px;"
        />

        <h3 class="quality-title">缺失率分布</h3>
        <EChart :option="missingRateOption" height="280px" />

        <h3 class="quality-title">目标列分布</h3>
        <EChart v-if="targetDistributionOption" :option="targetDistributionOption" height="300px" />
        <el-empty v-else description="目标列非类别类型，未返回分布" />

        <h3 class="quality-title">异常值概览</h3>
        <EChart :option="outlierOption" height="280px" />

        <el-row :gutter="20" style="margin-top: 20px;">
          <el-col :span="8">
            <el-card shadow="hover">
              <div class="quality-stat">常量/零方差列</div>
              <div class="quality-stat-value">
                {{ (qualityData.consistency?.constant_columns || []).length }}
              </div>
            </el-card>
          </el-col>
          <el-col :span="8">
            <el-card shadow="hover">
              <div class="quality-stat">疑似 ID 列</div>
              <div class="quality-stat-value">
                {{ (qualityData.uniqueness?.id_like_columns || []).length }}
              </div>
            </el-card>
          </el-col>
          <el-col :span="8">
            <el-card shadow="hover">
              <div class="quality-stat">重复行</div>
              <div class="quality-stat-value">
                {{ qualityData.uniqueness?.duplicated_rows || 0 }}
              </div>
            </el-card>
          </el-col>
        </el-row>
      </div>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { Upload, Cpu } from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { datasetApi, runApi } from '@/api'
import EChart from '@/components/EChart.vue'
import TrainConfigForm from '@/components/TrainConfigForm.vue'

const router = useRouter()
const datasets = ref([])
const loading = ref(false)
const showUploadDialog = ref(false)
const showTrainDialog = ref(false)
const showPreviewDialog = ref(false)
const showQualityDialog = ref(false)
const qualityLoading = ref(false)
const qualityData = ref(null)
const uploading = ref(false)
const training = ref(false)
const trainingDefault = ref(false)
const selectedFile = ref(null)
const currentDataset = ref(null)
const previewData = ref({ columns: [], rows: [] })
const searchQuery = ref('')
const currentPage = ref(1)
const pageSize = ref(10)
const selectedDatasets = ref([])

const filteredDatasets = computed(() => {
  const q = searchQuery.value.trim().toLowerCase()
  if (!q) return datasets.value
  return datasets.value.filter((d) => d.name.toLowerCase().includes(q))
})

const pagedDatasets = computed(() => {
  const start = (currentPage.value - 1) * pageSize.value
  return filteredDatasets.value.slice(start, start + pageSize.value)
})

const previewRows = computed(() => {
  return previewData.value.rows.map((row) => {
    const obj = {}
    previewData.value.columns.forEach((col, index) => {
      obj[`col_${index}`] = row[index]
    })
    return obj
  })
})

const uploadForm = ref({
  name: '',
})


const fetchDatasets = async () => {
  loading.value = true
  try {
    const res = await datasetApi.list()
    datasets.value = res.data
  } catch (error) {
    ElMessage.error('获取数据集失败: ' + error.message)
  } finally {
    loading.value = false
  }
}

const handleFileChange = (file) => {
  selectedFile.value = file.raw
}

const submitUpload = async () => {
  if (!uploadForm.value.name || !selectedFile.value) {
    ElMessage.warning('请填写名称并选择文件')
    return
  }

  uploading.value = true
  const formData = new FormData()
  formData.append('name', uploadForm.value.name)
  formData.append('file', selectedFile.value)

  try {
    await datasetApi.upload(formData)
    ElMessage.success('上传成功')
    showUploadDialog.value = false
    uploadForm.value.name = ''
    selectedFile.value = null
    await fetchDatasets()
  } catch (error) {
    ElMessage.error('上传失败: ' + error.message)
  } finally {
    uploading.value = false
  }
}

const previewDataset = async (row) => {
  try {
    const res = await datasetApi.preview(row.id)
    previewData.value = res.data
    showPreviewDialog.value = true
  } catch (error) {
    ElMessage.error('预览失败: ' + error.message)
  }
}

const showQuality = async (row) => {
  qualityLoading.value = true
  showQualityDialog.value = true
  qualityData.value = null
  try {
    const res = await datasetApi.quality(row.id)
    qualityData.value = res.data
  } catch (error) {
    ElMessage.error('获取质量报告失败: ' + error.message)
    showQualityDialog.value = false
  } finally {
    qualityLoading.value = false
  }
}

const missingRateOption = computed(() => {
  const missingRates = qualityData.value?.completeness?.missing_rates
  if (!missingRates) return {}
  const entries = Object.entries(missingRates)
    .filter(([_, rate]) => rate > 0)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 20)
  return {
    tooltip: { trigger: 'axis', formatter: '{b}: {c}%' },
    grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
    xAxis: { type: 'category', data: entries.map(([name]) => name) },
    yAxis: { type: 'value', axisLabel: { formatter: '{value}%' } },
    series: [{ data: entries.map(([_, rate]) => (rate * 100).toFixed(2)), type: 'bar', itemStyle: { color: '#e6a23c' } }],
  }
})

const targetDistributionOption = computed(() => {
  const dist = qualityData.value?.target_info?.class_distribution
  if (!dist) return null
  const data = Object.entries(dist).map(([name, value]) => ({ name, value }))
  return {
    tooltip: { trigger: 'item' },
    legend: { top: '5%', left: 'center' },
    series: [
      {
        type: 'pie',
        radius: ['40%', '70%'],
        avoidLabelOverlap: false,
        itemStyle: { borderRadius: 10, borderColor: '#fff', borderWidth: 2 },
        label: { show: false, position: 'center' },
        emphasis: { label: { show: true, fontSize: 18, fontWeight: 'bold' } },
        data,
      },
    ],
  }
})

const outlierOption = computed(() => {
  const outlierSummary = qualityData.value?.accuracy?.outlier_summary
  if (!outlierSummary) return {}
  const entries = Object.entries(outlierSummary)
    .map(([col, info]) => ({ col, count: info.outlier_count }))
    .sort((a, b) => b.count - a.count)
  return {
    tooltip: { trigger: 'axis' },
    grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
    xAxis: { type: 'category', data: entries.map((e) => e.col) },
    yAxis: { type: 'value' },
    series: [{ data: entries.map((e) => e.count), type: 'bar', itemStyle: { color: '#f56c6c' } }],
  }
})

const dimensionList = computed(() => {
  if (!qualityData.value) return []
  const dims = [
    { key: 'completeness', label: '完整性' },
    { key: 'consistency', label: '一致性' },
    { key: 'accuracy', label: '准确性' },
    { key: 'timeliness', label: '时效性' },
    { key: 'uniqueness', label: '唯一性' },
    { key: 'validity', label: '有效性' },
  ]
  return dims.map((d) => ({
    ...d,
    score: qualityData.value[d.key]?.score ?? 0,
  }))
})

const startTraining = (row) => {
  currentDataset.value = row
  showTrainDialog.value = true
}

const submitTrain = async (payload) => {
  training.value = true
  try {
    const res = await runApi.create(payload)
    ElMessage.success(payload.mode === 'step' ? 'Pipeline 草稿已创建' : '训练任务已启动')
    showTrainDialog.value = false
    if (payload.mode === 'step') {
      router.push(`/runs/${res.data.id}/pipeline`)
    } else {
      router.push(`/runs/${res.data.id}`)
    }
  } catch (error) {
    ElMessage.error('启动训练失败: ' + error.message)
  } finally {
    training.value = false
  }
}

const handleSelectionChange = (rows) => {
  selectedDatasets.value = rows
}

const deleteDataset = async (row) => {
  try {
    await ElMessageBox.confirm('确定删除该数据集吗？', '提示', { type: 'warning', confirmButtonText: '确定', cancelButtonText: '取消' })
    await datasetApi.delete(row.id)
    ElMessage.success('删除成功')
    await fetchDatasets()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('删除失败: ' + error.message)
    }
  }
}

const batchDeleteDatasets = async () => {
  const ids = selectedDatasets.value.map((d) => d.id)
  if (!ids.length) return
  try {
    await ElMessageBox.confirm(`确定删除选中的 ${ids.length} 个数据集吗？`, '提示', { type: 'warning', confirmButtonText: '确定', cancelButtonText: '取消' })
    await Promise.all(ids.map((id) => datasetApi.delete(id)))
    ElMessage.success('批量删除成功')
    selectedDatasets.value = []
    await fetchDatasets()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('批量删除失败: ' + error.message)
    }
  }
}

const formatSize = (bytes) => {
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB'
  return (bytes / 1024 / 1024).toFixed(2) + ' MB'
}

const formatDate = (date) => {
  return new Date(date).toLocaleString()
}

const trainDefaultDataset = async () => {
  try {
    const res = await datasetApi.list()
    const defaultDataset = res.data.find((d) => d.name === 'iris')
    if (!defaultDataset) {
      ElMessage.warning('默认数据集尚未加载，请刷新页面')
      return
    }

    trainingDefault.value = true
    ElMessage.info('正在使用默认数据集启动训练，请稍候...')
    const runRes = await runApi.create({
      dataset_id: defaultDataset.id,
      target_column: 'target',
      task_type: 'multiclass_classification',
      time_budget_minutes: 5,
      preset: 'auto',
    })
    ElMessage.success('默认数据集训练任务已启动')
    router.push(`/runs/${runRes.data.id}`)
  } catch (error) {
    ElMessage.error('启动默认训练失败: ' + error.message)
  } finally {
    trainingDefault.value = false
  }
}

onMounted(fetchDatasets)
</script>

<style scoped>
.datasets {
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

.quality-title {
  margin-top: 25px;
  margin-bottom: 10px;
  color: #303133;
  border-left: 4px solid #67c23a;
  padding-left: 10px;
}

.quality-stat {
  font-size: 12px;
  color: #909399;
  text-align: center;
}

.quality-stat-value {
  font-size: 28px;
  font-weight: bold;
  color: #409eff;
  text-align: center;
  margin-top: 8px;
}

.dim-card {
  margin-bottom: 10px;
}

.dim-card .quality-stat-value {
  font-size: 20px;
}

.quality-summary {
  margin-bottom: 10px;
}
</style>
