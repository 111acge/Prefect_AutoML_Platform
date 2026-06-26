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

      <el-alert
        v-if="envKeyHint"
        type="warning"
        :closable="false"
        style="margin-bottom: 16px;"
      >
        {{ envKeyHint }}
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
        <div v-if="config.api_key_configured" class="key-hint success-hint">
          {{ $t('llmSettings.apiKeyConfigured') }}
        </div>
        <div v-else class="key-hint">
          {{ $t('llmSettings.apiKeyNotConfigured') }}
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
  api_key_configured: false,
  supported_providers: [],
})

const providerOptions = {
  kimi: { defaultModel: 'moonshot-v1-8k', envKey: 'KIMI_API_KEY' },
  deepseek: { defaultModel: 'deepseek-v4-flash', envKey: 'DEEPSEEK_API_KEY' },
  minimax: { defaultModel: 'MiniMax-M3', envKey: 'MINIMAX_API_KEY' },
  glm: { defaultModel: 'glm-4-flash', envKey: 'GLM_API_KEY' },
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

const envKeyHint = computed(() => {
  const key = form.value.provider
  const envKey = providerOptions[key]?.envKey
  if (!envKey) return ''
  return t('llmSettings.envKeyHint', { envKey })
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
  apiKey: [{ required: false }],
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
    // 临时 API Key 仅保存在浏览器 sessionStorage 中，页面关闭即失，不会发送到服务器保存
    if (form.value.apiKey) {
      sessionStorage.setItem('llm_api_key', form.value.apiKey.trim())
    }
    const payload = {
      provider: form.value.provider,
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

.success-hint {
  color: #67c23a;
}

.deprecation-hint {
  color: #e6a23c;
}
</style>
