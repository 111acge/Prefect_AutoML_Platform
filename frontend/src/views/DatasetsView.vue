<template>
  <div class="datasets">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>数据集管理</span>
          <div>
            <el-button type="success" @click="trainDefaultDataset" :loading="trainingDefault">
              <el-icon><Cpu /></el-icon> 使用默认数据集训练
            </el-button>
            <el-button type="primary" @click="showUploadDialog = true">
              <el-icon><Upload /></el-icon> 上传数据集
            </el-button>
          </div>
        </div>
      </template>

      <el-table :data="datasets" v-loading="loading" style="width: 100%">
        <el-table-column prop="name" label="名称" />
        <el-table-column prop="row_count" label="行数" />
        <el-table-column prop="column_count" label="列数" />
        <el-table-column prop="file_size_bytes" label="大小">
          <template #default="{ row }">
            {{ formatSize(row.file_size_bytes) }}
          </template>
        </el-table-column>
        <el-table-column prop="created_at" label="创建时间">
          <template #default="{ row }">
            {{ formatDate(row.created_at) }}
          </template>
        </el-table-column>
        <el-table-column label="操作" width="320">
          <template #default="{ row }">
            <el-button size="small" @click="previewDataset(row)">预览</el-button>
            <el-button size="small" type="info" @click="showQuality(row)">质量报告</el-button>
            <el-button size="small" type="primary" @click="startTraining(row)">训练</el-button>
            <el-button size="small" type="danger" @click="deleteDataset(row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
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
    <el-dialog v-model="showTrainDialog" title="启动训练任务" width="500px">
      <el-form :model="trainForm" label-width="120px">
        <el-form-item label="目标列">
          <el-select v-model="trainForm.target_column" style="width: 100%" placeholder="请选择目标列">
            <el-option
              v-for="col in datasetColumns"
              :key="col"
              :label="col"
              :value="col"
            />
          </el-select>
        </el-form-item>
        <el-form-item label="任务类型">
          <el-select v-model="trainForm.task_type" style="width: 100%">
            <el-option label="二分类" value="binary_classification" />
            <el-option label="多分类" value="multiclass_classification" />
            <el-option label="回归" value="regression" />
          </el-select>
        </el-form-item>
        <el-form-item label="时间预算(分钟)">
          <el-slider v-model="trainForm.time_budget_minutes" :min="1" :max="60" show-input />
        </el-form-item>
        <el-form-item label="Preset">
          <el-select v-model="trainForm.preset" style="width: 100%">
            <el-option label="自动选择（推荐）" value="auto" />
            <el-option label="medium_quality" value="medium_quality" />
            <el-option label="best_quality" value="best_quality" />
          </el-select>
        </el-form-item>
        <el-form-item label="随机种子">
          <el-input-number v-model="trainForm.seed" :min="0" :controls="false" style="width: 100%" placeholder="留空则不固定" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showTrainDialog = false">取消</el-button>
        <el-button type="primary" @click="submitTrain" :loading="training">开始训练</el-button>
      </template>
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
        <el-descriptions :column="3" border>
          <el-descriptions-item label="行数">{{ qualityData.n_rows }}</el-descriptions-item>
          <el-descriptions-item label="列数">{{ qualityData.n_columns }}</el-descriptions-item>
          <el-descriptions-item label="特征数">{{ qualityData.n_features }}</el-descriptions-item>
          <el-descriptions-item label="含缺失值行数">{{ qualityData.rows_with_missing }}</el-descriptions-item>
        </el-descriptions>

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
                {{ (qualityData.constant_columns || []).length }}
              </div>
            </el-card>
          </el-col>
          <el-col :span="8">
            <el-card shadow="hover">
              <div class="quality-stat">疑似 ID 列</div>
              <div class="quality-stat-value">
                {{ (qualityData.id_like_columns || []).length }}
              </div>
            </el-card>
          </el-col>
          <el-col :span="8">
            <el-card shadow="hover">
              <div class="quality-stat">高基数类别列</div>
              <div class="quality-stat-value">
                {{ (qualityData.high_cardinality_columns || []).length }}
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
const datasetColumns = ref([])
const previewData = ref({ columns: [], rows: [] })

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

const trainForm = ref({
  dataset_id: '',
  target_column: '',
  task_type: 'binary_classification',
  time_budget_minutes: 10,
  preset: 'auto',
  seed: null,
})

const resetTrainForm = () => {
  trainForm.value = {
    dataset_id: '',
    target_column: '',
    task_type: 'binary_classification',
    time_budget_minutes: 10,
    preset: 'medium_quality',
    seed: null,
  }
}

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
  if (!qualityData.value?.missing_rates) return {}
  const entries = Object.entries(qualityData.value.missing_rates)
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
  if (!qualityData.value?.outlier_summary) return {}
  const entries = Object.entries(qualityData.value.outlier_summary)
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

const startTraining = (row) => {
  currentDataset.value = row
  trainForm.value.dataset_id = row.id
  trainForm.value.target_column = ''
  datasetColumns.value = Object.keys(row.schema_info?.field_types || {})
  showTrainDialog.value = true
}

const submitTrain = async () => {
  if (!trainForm.value.target_column) {
    ElMessage.warning('请选择目标列')
    return
  }

  training.value = true
  try {
    const res = await runApi.create({
      dataset_id: currentDataset.value.id,
      ...trainForm.value,
    })
    ElMessage.success('训练任务已启动')
    showTrainDialog.value = false
    resetTrainForm()
    router.push(`/runs/${res.data.id}`)
  } catch (error) {
    ElMessage.error('启动训练失败: ' + error.message)
  } finally {
    training.value = false
  }
}

const deleteDataset = async (row) => {
  try {
    await ElMessageBox.confirm('确定删除该数据集吗？', '提示', { type: 'warning' })
    await datasetApi.delete(row.id)
    ElMessage.success('删除成功')
    await fetchDatasets()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('删除失败: ' + error.message)
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
    const runRes = await runApi.create({
      dataset_id: defaultDataset.id,
      target_column: 'target',
      task_type: 'multiclass_classification',
      time_budget_minutes: 2,
      preset: 'medium_quality',
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
</style>
