<!--
Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
See LICENSE for details.
-->

<template>
  <el-form ref="formRef" :model="form" :rules="rules" label-width="120px">
    <el-form-item label="目标列" prop="target_column">
      <el-select v-model="form.target_column" style="width: 100%" placeholder="请选择目标列">
        <el-option
          v-for="col in columns"
          :key="col"
          :label="col"
          :value="col"
        />
      </el-select>
    </el-form-item>

    <el-form-item label="任务类型" prop="task_type">
      <el-select v-model="form.task_type" style="width: 100%">
        <el-option label="二分类" value="binary_classification" />
        <el-option label="多分类" value="multiclass_classification" />
        <el-option label="回归" value="regression" />
      </el-select>
    </el-form-item>

    <el-form-item label="快速模式">
      <el-radio-group v-model="quickMode" @change="onQuickModeChange">
        <el-radio-button value="quick">快速体验</el-radio-button>
        <el-radio-button value="standard">标准</el-radio-button>
        <el-radio-button value="deep">深度</el-radio-button>
        <el-radio-button value="unlimited">不限制</el-radio-button>
      </el-radio-group>
    </el-form-item>

    <el-form-item label="时间预算(分钟)">
      <div style="display: flex; align-items: center; gap: 12px;">
        <el-input-number
          v-model="form.time_budget_minutes"
          :min="0.1"
          :max="1440"
          :disabled="unlimitedTime"
          :controls="false"
          style="width: 160px"
        />
        <el-checkbox v-model="unlimitedTime">无限制</el-checkbox>
      </div>
    </el-form-item>

    <el-form-item label="Preset">
      <el-select v-model="form.preset" style="width: 100%">
        <el-option label="自动选择（推荐）" value="auto" />
        <el-option label="good_quality" value="good_quality" />
        <el-option label="medium_quality" value="medium_quality" />
        <el-option label="best_quality" value="best_quality" />
      </el-select>
    </el-form-item>

    <el-form-item label="随机种子">
      <el-input-number
        v-model="form.seed"
        :min="0"
        :controls="false"
        style="width: 100%"
        placeholder="留空则不固定"
      />
    </el-form-item>

    <el-form-item label="特征工程">
      <el-switch
        v-model="form.feature_engineering_enabled"
        active-text="启用高级特征工程"
        inactive-text="仅基础清洗"
      />
    </el-form-item>

    <el-form-item label="执行模式">
      <el-radio-group v-model="form.mode">
        <el-radio-button value="auto">一键训练</el-radio-button>
        <el-radio-button value="step">分步 Pipeline</el-radio-button>
      </el-radio-group>
    </el-form-item>
  </el-form>

  <div class="form-footer">
    <el-button @click="$emit('cancel')">取消</el-button>
    <el-button type="primary" :loading="loading" @click="submit">
      {{ form.mode === 'step' ? '创建 Pipeline' : '开始训练' }}
    </el-button>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'
import { ElMessage } from 'element-plus'

const props = defineProps({
  dataset: { type: Object, default: null },
  loading: { type: Boolean, default: false },
})

const emit = defineEmits(['submit', 'cancel'])

const formRef = ref(null)
const columns = ref([])
const unlimitedTime = ref(false)
const quickMode = ref('standard')

const defaultForm = () => ({
  target_column: '',
  task_type: 'binary_classification',
  time_budget_minutes: 10,
  preset: 'auto',
  seed: null,
  feature_engineering_enabled: true,
  mode: 'auto',
})

const form = ref(defaultForm())

const QUICK_MODES = {
  quick: { preset: 'good_quality', time_budget_minutes: 1 },
  standard: { preset: 'auto', time_budget_minutes: 10 },
  deep: { preset: 'best_quality', time_budget_minutes: 30 },
  unlimited: { preset: 'best_quality', time_budget_minutes: null },
}

function inferTaskType(targetInfo) {
  if (!targetInfo) return 'binary_classification'
  const type = targetInfo.type
  const unique = targetInfo.unique_values
  if (type === 'numeric') return 'regression'
  if (unique === 2) return 'binary_classification'
  return 'multiclass_classification'
}

function applyDataset(dataset) {
  if (!dataset) {
    columns.value = []
    form.value = defaultForm()
    quickMode.value = 'standard'
    unlimitedTime.value = false
    return
  }
  const fieldTypes = dataset.schema_info?.field_types || {}
  columns.value = Object.keys(fieldTypes)

  const target = dataset.target_column
  const task = dataset.task_type
  const inferredTask = task || inferTaskType(dataset.schema_info?.target_info)

  form.value = {
    ...defaultForm(),
    target_column: target || '',
    task_type: inferredTask,
  }
}

watch(() => props.dataset, applyDataset, { immediate: true })

function onQuickModeChange(mode) {
  const cfg = QUICK_MODES[mode]
  if (!cfg) return
  form.value.preset = cfg.preset
  form.value.time_budget_minutes = cfg.time_budget_minutes ?? 10
  unlimitedTime.value = cfg.time_budget_minutes === null
}

const rules = {
  target_column: [{ required: true, message: '请选择目标列', trigger: 'change' }],
  task_type: [{ required: true, message: '请选择任务类型', trigger: 'change' }],
}

async function submit() {
  if (!formRef.value) return
  const valid = await formRef.value.validate().catch(() => false)
  if (!valid) return
  const payload = { ...form.value }
  if (unlimitedTime.value) {
    payload.time_budget_minutes = null
  }
  if (props.dataset) {
    payload.dataset_id = props.dataset.id
  }
  emit('submit', payload)
}
</script>

<style scoped>
.form-footer {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
  margin-top: 8px;
}
</style>
