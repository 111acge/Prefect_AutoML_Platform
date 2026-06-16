<template>
  <div class="runs">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>训练任务</span>
          <div>
            <el-button type="primary" @click="showCreateDialog = true">
              <el-icon><Plus /></el-icon> 新建训练任务
            </el-button>
            <el-button @click="fetchRuns">刷新</el-button>
          </div>
        </div>
      </template>

      <el-table :data="runs" v-loading="loading" style="width: 100%">
        <el-table-column prop="id" label="ID" width="220">
          <template #default="{ row }">
            <el-link type="primary" @click="viewDetail(row)">{{ row.id }}</el-link>
          </template>
        </el-table-column>
        <el-table-column prop="dataset_id" label="数据集ID" width="220" />
        <el-table-column prop="status" label="状态" width="110">
          <template #default="{ row }">
            <el-tag :type="statusType(row.status)">{{ row.status }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="失败原因" min-width="220" show-overflow-tooltip>
          <template #default="{ row }">
            <span v-if="row.error_message" class="error-text">{{ row.error_message }}</span>
            <span v-else-if="row.status === 'failed'" class="error-text">-</span>
            <span v-else>-</span>
          </template>
        </el-table-column>
        <el-table-column prop="time_budget_minutes" label="时间预算(分钟)" />
        <el-table-column label="随机种子">
          <template #default="{ row }">
            {{ row.config?.seed ?? '-' }}
          </template>
        </el-table-column>
        <el-table-column prop="primary_metric" label="评估指标" />
        <el-table-column prop="created_at" label="创建时间">
          <template #default="{ row }">
            {{ formatDate(row.created_at) }}
          </template>
        </el-table-column>
        <el-table-column label="操作" width="150">
          <template #default="{ row }">
            <el-button size="small" type="danger" @click="deleteRun(row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- 新建训练任务对话框 -->
    <el-dialog v-model="showCreateDialog" title="新建训练任务" width="500px">
      <el-form :model="createForm" label-width="120px">
        <el-form-item label="数据集">
          <el-select v-model="createForm.dataset_id" style="width: 100%" placeholder="请选择数据集">
            <el-option
              v-for="dataset in datasets"
              :key="dataset.id"
              :label="`${dataset.name} (${dataset.row_count}行 x ${dataset.column_count}列)`"
              :value="dataset.id"
            />
          </el-select>
        </el-form-item>
        <el-form-item label="目标列">
          <el-select v-model="createForm.target_column" style="width: 100%" placeholder="请选择目标列">
            <el-option
              v-for="col in datasetColumns"
              :key="col"
              :label="col"
              :value="col"
            />
          </el-select>
        </el-form-item>
        <el-form-item label="任务类型">
          <el-select v-model="createForm.task_type" style="width: 100%">
            <el-option label="二分类" value="binary_classification" />
            <el-option label="多分类" value="multiclass_classification" />
            <el-option label="回归" value="regression" />
          </el-select>
        </el-form-item>
        <el-form-item label="时间预算(分钟)">
          <el-slider v-model="createForm.time_budget_minutes" :min="1" :max="1440" show-input />
        </el-form-item>
        <el-form-item label="Preset">
          <el-select v-model="createForm.preset" style="width: 100%">
            <el-option label="自动选择（推荐）" value="auto" />
            <el-option label="medium_quality" value="medium_quality" />
            <el-option label="best_quality" value="best_quality" />
          </el-select>
        </el-form-item>
        <el-form-item label="随机种子">
          <el-input-number v-model="createForm.seed" :min="0" :controls="false" style="width: 100%" placeholder="留空则不固定" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showCreateDialog = false">取消</el-button>
        <el-button type="primary" @click="submitCreate" :loading="creating">创建</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import { Plus } from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { runApi, datasetApi } from '@/api'

const router = useRouter()
const runs = ref([])
const datasets = ref([])
const loading = ref(false)
const showCreateDialog = ref(false)
const creating = ref(false)
let timer = null

const datasetColumns = ref([])

const createForm = ref({
  dataset_id: '',
  target_column: '',
  task_type: 'binary_classification',
  time_budget_minutes: 10,
  preset: 'auto',
  seed: null,
})

const statusType = (status) => {
  const map = {
    pending: 'info',
    running: 'warning',
    completed: 'success',
    failed: 'danger',
  }
  return map[status] || 'info'
}

const fetchRuns = async () => {
  loading.value = true
  try {
    const res = await runApi.list()
    runs.value = res.data
  } catch (error) {
    ElMessage.error('获取任务失败: ' + error.message)
  } finally {
    loading.value = false
  }
}

const fetchDatasets = async () => {
  try {
    const res = await datasetApi.list()
    datasets.value = res.data
  } catch (error) {
    ElMessage.error('获取数据集失败: ' + error.message)
  }
}

const submitCreate = async () => {
  if (!createForm.value.dataset_id || !createForm.value.target_column) {
    ElMessage.warning('请选择数据集和目标列')
    return
  }

  creating.value = true
  try {
    const res = await runApi.create(createForm.value)
    ElMessage.success('训练任务已创建')
    showCreateDialog.value = false
    resetCreateForm()
    router.push(`/runs/${res.data.id}`)
  } catch (error) {
    ElMessage.error('创建训练任务失败: ' + error.message)
  } finally {
    creating.value = false
  }
}

const resetCreateForm = () => {
  createForm.value = {
    dataset_id: '',
    target_column: '',
    task_type: 'binary_classification',
    time_budget_minutes: 10,
    preset: 'medium_quality',
    seed: null,
  }
  datasetColumns.value = []
}

watch(
  () => createForm.value.dataset_id,
  (datasetId) => {
    if (!datasetId) {
      datasetColumns.value = []
      return
    }
    const dataset = datasets.value.find((d) => d.id === datasetId)
    datasetColumns.value = dataset ? Object.keys(dataset.schema_info?.field_types || {}) : []
  }
)

const viewDetail = (row) => {
  router.push(`/runs/${row.id}`)
}

const deleteRun = async (row) => {
  try {
    await ElMessageBox.confirm('确定删除该训练任务吗？', '提示', { type: 'warning' })
    await runApi.delete(row.id)
    ElMessage.success('删除成功')
    await fetchRuns()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('删除失败: ' + error.message)
    }
  }
}

const formatDate = (date) => {
  return new Date(date).toLocaleString()
}

onMounted(() => {
  fetchRuns()
  fetchDatasets()
  timer = setInterval(fetchRuns, 5000)
})

onUnmounted(() => {
  if (timer) clearInterval(timer)
})
</script>

<style scoped>
.runs {
  padding: 20px 0;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.error-text {
  color: #f56c6c;
}
</style>
