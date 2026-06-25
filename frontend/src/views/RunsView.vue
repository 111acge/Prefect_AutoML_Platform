<template>
  <div class="runs">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>训练任务</span>
          <div>
            <el-button type="primary" @click="showCreateDialog = true">
              <el-icon><component :is="Plus" /></el-icon> 新建训练任务
            </el-button>
            <el-button @click="fetchRuns">刷新</el-button>
          </div>
        </div>
      </template>

      <div class="table-toolbar">
        <el-input
          v-model="searchQuery"
          placeholder="搜索任务 ID 或数据集"
          style="width: 300px"
          clearable
        />
      </div>

      <el-table :data="pagedRuns" v-loading="loading" style="width: 100%">
        <el-table-column prop="id" label="ID" width="220" sortable>
          <template #default="{ row }">
            <el-link type="primary" @click="viewDetail(row)">{{ row.id }}</el-link>
          </template>
        </el-table-column>
        <el-table-column prop="dataset_id" label="数据集ID" width="220" sortable />
        <el-table-column prop="status" label="状态" width="140" sortable>
          <template #default="{ row }">
            <el-tooltip v-if="row.status === 'failed' && row.error_message" :content="row.error_message" placement="top">
              <el-tag :type="statusType(row.status)">{{ row.status }}</el-tag>
            </el-tooltip>
            <el-tag v-else :type="statusType(row.status)">{{ row.status }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="时间预算(分钟)" sortable>
          <template #default="{ row }">
            {{ row.time_budget_minutes ?? '无限制' }}
          </template>
        </el-table-column>
        <el-table-column label="随机种子" sortable>
          <template #default="{ row }">
            {{ row.config?.seed ?? '-' }}
          </template>
        </el-table-column>
        <el-table-column prop="primary_metric" label="评估指标" sortable />
        <el-table-column prop="created_at" label="创建时间" sortable>
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

      <el-pagination
        v-model:current-page="currentPage"
        v-model:page-size="pageSize"
        :page-sizes="[10, 20, 50]"
        :total="filteredRuns.length"
        layout="total, sizes, prev, pager, next"
        style="margin-top: 16px; justify-content: flex-end"
      />
    </el-card>

    <!-- 新建训练任务对话框 -->
    <el-dialog v-model="showCreateDialog" title="新建训练任务" width="520px">
      <el-form ref="datasetFormRef" :model="datasetForm" :rules="datasetRules" label-width="80px" style="margin-bottom: 16px;">
        <el-form-item label="数据集" prop="dataset_id">
          <el-select v-model="datasetForm.dataset_id" style="width: 100%" placeholder="请选择数据集" @change="onDatasetChange">
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

const datasetFormRef = ref(null)
const datasetForm = ref({ dataset_id: '' })
const datasetRules = {
  dataset_id: [{ required: true, message: '请选择数据集', trigger: 'change' }],
}
const selectedDataset = computed(() => {
  return datasets.value.find((d) => d.id === datasetForm.value.dataset_id) || null
})

const searchQuery = ref('')
const currentPage = ref(1)
const pageSize = ref(10)

const filteredRuns = computed(() => {
  const q = searchQuery.value.trim().toLowerCase()
  if (!q) return runs.value
  return runs.value.filter((r) => {
    const datasetName = r.config?.snapshot?.dataset_name || ''
    return r.id.toLowerCase().includes(q) || datasetName.toLowerCase().includes(q) || r.status.toLowerCase().includes(q)
  })
})

const pagedRuns = computed(() => {
  const start = (currentPage.value - 1) * pageSize.value
  return filteredRuns.value.slice(start, start + pageSize.value)
})

const onDatasetChange = () => {
  datasetFormRef.value?.clearValidate('dataset_id')
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
  const valid = await datasetFormRef.value?.validate().catch(() => false)
  if (!valid) return
  if (!payload.target_column) {
    ElMessage.warning('请选择目标列')
    return
  }

  creating.value = true
  try {
    const res = await runApi.create(payload)
    ElMessage.success(payload.mode === 'step' ? 'Pipeline 草稿已创建' : '训练任务已创建')
    showCreateDialog.value = false
    datasetForm.value.dataset_id = ''
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
    await ElMessageBox.confirm('确定删除该训练任务吗？', '提示', { type: 'warning', confirmButtonText: '确定', cancelButtonText: '取消' })
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

.table-toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.error-text {
  color: #f56c6c;
}
</style>
