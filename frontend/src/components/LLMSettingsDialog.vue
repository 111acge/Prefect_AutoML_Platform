<!--
Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
See LICENSE for details.
-->

<template>
  <el-dialog
    v-model="visible"
    title="LLM API 配置"
    width="520px"
    :close-on-click-modal="false"
    @open="loadConfig"
  >
    <el-form
      ref="formRef"
      :model="form"
      :rules="rules"
      label-position="top"
      @submit.prevent
    >
      <el-alert
        type="info"
        :closable="false"
        style="margin-bottom: 16px;"
      >
        配置 LLM API Key 后，平台将使用大模型生成业务解读、意图解析等智能内容。未配置时将自动使用规则兜底。
      </el-alert>

      <el-form-item label="提供商" prop="provider">
        <el-select v-model="form.provider" placeholder="请选择 LLM 提供商" style="width: 100%">
          <el-option
            v-for="p in supportedProviders"
            :key="p.key"
            :label="p.label"
            :value="p.key"
          />
        </el-select>
      </el-form-item>

      <el-form-item label="API Key" prop="apiKey">
        <el-input
          v-model="form.apiKey"
          type="password"
          show-password
          placeholder="请输入所选提供商的 API Key"
          clearable
        />
        <div v-if="config.api_key_masked" class="key-hint">
          当前已保存：{{ config.api_key_masked }}
        </div>
      </el-form-item>

      <el-form-item label="模型（可选）" prop="model">
        <el-input
          v-model="form.model"
          placeholder="留空使用默认模型"
          clearable
        />
        <div class="model-hint">
          默认模型：{{ defaultModel }}
        </div>
        <div v-if="modelHint" class="model-hint deprecation-hint">
          {{ modelHint }}
        </div>
      </el-form-item>
    </el-form>

    <template #footer>
      <el-button @click="visible = false">取消</el-button>
      <el-button type="primary" :loading="saving" @click="handleSave">保存配置</el-button>
    </template>
  </el-dialog>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { llmSettingsApi } from '@/api'

const visible = defineModel('visible', { type: Boolean, default: false })
const emit = defineEmits(['saved'])

const formRef = ref(null)
const saving = ref(false)
const config = ref({
  provider: null,
  model: null,
  api_key_masked: null,
  supported_providers: [],
})

const providerOptions = {
  kimi: { label: 'KIMI（Moonshot）', defaultModel: 'moonshot-v1-8k' },
  deepseek: { label: 'DeepSeek', defaultModel: 'deepseek-v4-flash' },
  minimax: { label: 'MiniMax', defaultModel: 'MiniMax-M3' },
  glm: { label: '智谱 GLM', defaultModel: 'glm-4-flash' },
}

const form = ref({
  provider: 'deepseek',
  apiKey: '',
  model: '',
})

const supportedProviders = computed(() =>
  (config.value.supported_providers || []).map((key) => ({
    key,
    label: providerOptions[key]?.label || key,
  }))
)

const defaultModel = computed(() => {
  const key = form.value.provider
  return providerOptions[key]?.defaultModel || '-'
})

const modelHint = computed(() => {
  const key = form.value.provider
  if (key === 'deepseek') {
    return '推荐：deepseek-v4-flash / deepseek-v4-pro；deepseek-chat、deepseek-reasoner 将于 2026/07/24 弃用'
  }
  return ''
})

const rules = {
  provider: [{ required: true, message: '请选择提供商', trigger: 'change' }],
  apiKey: [{ required: true, message: '请输入 API Key', trigger: 'blur' }],
}

watch(
  () => form.value.provider,
  () => {
    if (!form.value.model) {
      form.value.model = ''
    }
  }
)

const loadConfig = async () => {
  try {
    const res = await llmSettingsApi.get()
    config.value = res.data || {}
    if (config.value.provider) {
      form.value.provider = config.value.provider
      form.value.model = config.value.model || ''
    } else {
      form.value.provider = 'deepseek'
      form.value.model = ''
    }
    form.value.apiKey = ''
  } catch (error) {
    ElMessage.error(error.message || '加载配置失败')
  }
}

const handleSave = async () => {
  const valid = await formRef.value.validate().catch(() => false)
  if (!valid) return

  saving.value = true
  try {
    const payload = {
      provider: form.value.provider,
      api_key: form.value.apiKey,
      model: form.value.model.trim() || undefined,
    }
    const res = await llmSettingsApi.save(payload)
    config.value = res.data || {}
    ElMessage.success('LLM 配置已保存')
    emit('saved')
    visible.value = false
  } catch (error) {
    ElMessage.error(error.message || '保存失败')
  } finally {
    saving.value = false
  }
}
</script>

<style scoped>
.key-hint,
.model-hint {
  margin-top: 6px;
  font-size: 12px;
  color: #909399;
  line-height: 1.4;
}

.deprecation-hint {
  color: #e6a23c;
}
</style>
