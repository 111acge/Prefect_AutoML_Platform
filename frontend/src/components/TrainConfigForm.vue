<!--
Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
See LICENSE for details.
-->

<template>
  <el-form ref="formRef" :model="form" :rules="rules" label-width="120px">
    <el-form-item :label="$t('trainForm.targetColumn')" prop="target_column">
      <el-select v-model="form.target_column" style="width: 100%" :placeholder="$t('trainForm.targetPlaceholder')">
        <el-option
          v-for="col in columns"
          :key="col"
          :label="col"
          :value="col"
        />
      </el-select>
    </el-form-item>

    <el-form-item :label="$t('trainForm.taskType')" prop="task_type">
      <el-select v-model="form.task_type" style="width: 100%">
        <el-option :label="$t('trainForm.binary')" value="binary_classification" />
        <el-option :label="$t('trainForm.multiclass')" value="multiclass_classification" />
        <el-option :label="$t('trainForm.regression')" value="regression" />
      </el-select>
    </el-form-item>

    <el-form-item :label="$t('trainForm.quickMode')">
      <el-radio-group v-model="quickMode" @change="onQuickModeChange">
        <el-radio-button value="quick">{{ $t('trainForm.quickModes.quick') }}</el-radio-button>
        <el-radio-button value="standard">{{ $t('trainForm.quickModes.standard') }}</el-radio-button>
        <el-radio-button value="deep">{{ $t('trainForm.quickModes.deep') }}</el-radio-button>
        <el-radio-button value="unlimited">{{ $t('trainForm.quickModes.unlimited') }}</el-radio-button>
      </el-radio-group>
    </el-form-item>

    <el-form-item :label="$t('trainForm.timeBudget')">
      <div style="display: flex; align-items: center; gap: 12px;">
        <el-input-number
          v-model="form.time_budget_minutes"
          :min="0.1"
          :max="1440"
          :disabled="unlimitedTime"
          :controls="false"
          style="width: 160px"
        />
        <el-checkbox v-model="unlimitedTime">{{ $t('trainForm.unlimited') }}</el-checkbox>
      </div>
    </el-form-item>

    <el-form-item :label="$t('trainForm.preset')">
      <el-select v-model="form.preset" style="width: 100%">
        <el-option :label="$t('trainForm.presets.auto')" value="auto" />
        <el-option :label="$t('trainForm.presets.good')" value="good_quality" />
        <el-option :label="$t('trainForm.presets.medium')" value="medium_quality" />
        <el-option :label="$t('trainForm.presets.best')" value="best_quality" />
      </el-select>
    </el-form-item>

    <el-form-item :label="$t('trainForm.seed')">
      <el-input-number
        v-model="form.seed"
        :min="0"
        :controls="false"
        style="width: 100%"
        :placeholder="$t('trainForm.seedPlaceholder')"
      />
    </el-form-item>

    <el-form-item :label="$t('trainForm.featureEngineering')">
      <el-switch
        v-model="form.feature_engineering_enabled"
        :active-text="$t('trainForm.featureEngineeringOn')"
        :inactive-text="$t('trainForm.featureEngineeringOff')"
      />
    </el-form-item>

    <el-form-item :label="$t('trainForm.mode')">
      <el-radio-group v-model="form.mode">
        <el-radio-button value="auto">{{ $t('trainForm.modes.auto') }}</el-radio-button>
        <el-radio-button value="step">{{ $t('trainForm.modes.step') }}</el-radio-button>
      </el-radio-group>
    </el-form-item>
  </el-form>

  <div class="form-footer">
    <el-button @click="$emit('cancel')">{{ $t('common.cancel') }}</el-button>
    <el-button type="primary" :loading="loading" @click="submit">
      {{ form.mode === 'step' ? $t('trainForm.createPipeline') : $t('trainForm.startTrain') }}
    </el-button>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { ElMessage } from 'element-plus'

const props = defineProps({
  dataset: { type: Object, default: null },
  loading: { type: Boolean, default: false },
})

const emit = defineEmits(['submit', 'cancel'])

const { t } = useI18n()
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
  target_column: [{ required: true, message: t('trainForm.validation.targetRequired'), trigger: 'change' }],
  task_type: [{ required: true, message: t('trainForm.validation.taskTypeRequired'), trigger: 'change' }],
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
