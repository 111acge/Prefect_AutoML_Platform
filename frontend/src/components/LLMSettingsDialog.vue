<!--
Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
See LICENSE for details.
-->

<template>
  <el-dialog
    v-model="visible"
    :title="$t('llmSettings.title')"
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
        {{ $t('llmSettings.description') }}
      </el-alert>

      <el-form-item :label="$t('llmSettings.provider')" prop="provider">
        <el-select v-model="form.provider" :placeholder="$t('llmSettings.providerPlaceholder')" style="width: 100%">
          <el-option
            v-for="p in supportedProviders"
            :key="p.key"
            :label="p.label"
            :value="p.key"
          />
        </el-select>
      </el-form-item>

      <el-form-item :label="$t('llmSettings.apiKey')" prop="apiKey">
        <el-input
          v-model="form.apiKey"
          type="password"
          show-password
          :placeholder="$t('llmSettings.apiKeyPlaceholder')"
          clearable
        />
        <div v-if="config.api_key_masked" class="key-hint">
          {{ $t('llmSettings.saved', { masked: config.api_key_masked }) }}
        </div>
      </el-form-item>

      <el-form-item :label="$t('llmSettings.model')" prop="model">
        <el-input
          v-model="form.model"
          :placeholder="$t('llmSettings.modelPlaceholder')"
          clearable
        />
        <div class="model-hint">
          {{ $t('llmSettings.defaultModel', { model: defaultModel }) }}
        </div>
        <div v-if="modelHint" class="model-hint deprecation-hint">
          {{ modelHint }}
        </div>
      </el-form-item>
    </el-form>

    <template #footer>
      <el-button @click="visible = false">{{ $t('llmSettings.cancel') }}</el-button>
      <el-button type="primary" :loading="saving" @click="handleSave">{{ $t('llmSettings.save') }}</el-button>
    </template>
  </el-dialog>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { ElMessage } from 'element-plus'
import { llmSettingsApi } from '@/api'

const visible = defineModel('visible', { type: Boolean, default: false })
const emit = defineEmits(['saved'])

const { t } = useI18n()
const formRef = ref(null)
const saving = ref(false)
const config = ref({
  provider: null,
  model: null,
  api_key_masked: null,
  supported_providers: [],
})

const providerOptions = {
  kimi: { defaultModel: 'moonshot-v1-8k' },
  deepseek: { defaultModel: 'deepseek-v4-flash' },
  minimax: { defaultModel: 'MiniMax-M3' },
  glm: { defaultModel: 'glm-4-flash' },
}

const form = ref({
  provider: 'deepseek',
  apiKey: '',
  model: '',
})

const supportedProviders = computed(() =>
  (config.value.supported_providers || []).map((key) => ({
    key,
    label: t(`llmSettings.providers.${key}`) || key,
  }))
)

const defaultModel = computed(() => {
  const key = form.value.provider
  return providerOptions[key]?.defaultModel || '-'
})

const modelHint = computed(() => {
  const key = form.value.provider
  if (key === 'deepseek') {
    return t('llmSettings.recommendation')
  }
  return ''
})

const rules = {
  provider: [{ required: true, message: t('llmSettings.validation.providerRequired'), trigger: 'change' }],
  apiKey: [{ required: true, message: t('llmSettings.validation.apiKeyRequired'), trigger: 'blur' }],
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
    ElMessage.error(error.message || t('llmSettings.errors.loadFailed'))
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
    ElMessage.success(t('llmSettings.saveSuccess'))
    emit('saved')
    visible.value = false
  } catch (error) {
    ElMessage.error(error.message || t('llmSettings.errors.saveFailed'))
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
