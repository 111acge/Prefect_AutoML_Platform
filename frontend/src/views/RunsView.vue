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
        <el-table-column label="时间预算(分钟)">
          <template #default="{ row }">
            {{ row.time_budget_minutes ?? '无限制' }}
          </template>
        </el-table-column>
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
        <template #empty>
          <el-empty description="暂无训练任务" />
        </template>
      </el-table>
    </el-card>

    <!-- 新建训练任务对话框 -->
    <el-dialog v-model="showCreateDialog" title="新建训练任务" width="520px">
      <el-form label-width="80px" style="margin-bottom: 16px;">
        <el-form-item label="数据集">
          <el-select v-model="selectedDatasetId" style="width: 100%" placeholder="请选择数据集">
            <el-option
              v-for="dataset in datasets"
              :key="dataset.id"
              :label="`${dataset.name} (${dataset.row_count}行 x ${dataset.column_count}列)`"
              :value="dataset.id"
            />
          </el-select>
        </el-form-item>
      </el-form>
      <TrainConfigForm
        :dataset="selectedDataset"
        :loading="creating"
        @submit="submitCreate"
        @cancel="showCreateDialog = false"
      />
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { Plus } from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { runApi, datasetApi } from '@/api'
import TrainConfigForm from '@/components/TrainConfigForm.vue'

const router = useRouter()
const runs = ref([])
const datasets = ref([])
const loading = ref(false)
const showCreateDialog = ref(false)
const creating = ref(false)
let timer = null

const selectedDatasetId = ref('')
const selectedDataset = computed(() => {
  return datasets.value.find((d) => d.id === selectedDatasetId.value) || null
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

const submitCreate = async (payload) => {
  if (!payload.dataset_id || !payload.target_column) {
    ElMessage.warning('请选择数据集和目标列')
    return
  }

  creating.value = true
  try {
    const res = await runApi.create(payload)
    ElMessage.success(payload.mode === 'step' ? 'Pipeline 草稿已创建' : '训练任务已创建')
    showCreateDialog.value = false
    selectedDatasetId.value = ''
    if (payload.mode === 'step') {
      router.push(`/runs/${res.data.id}/pipeline`)
    } else {
      router.push(`/runs/${res.data.id}`)
    }
  } catch (error) {
    ElMessage.error('创建训练任务失败: ' + error.message)
  } finally {
    creating.value = false
  }
}

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
