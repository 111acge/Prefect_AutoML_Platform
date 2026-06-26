<!--
Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
See LICENSE for details.
-->

<template>
  <div class="home">
    <el-row :gutter="20">
      <el-col :span="24">
        <el-card class="welcome-card">
          <h1>Prefect AutoML Platform</h1>
          <p class="subtitle">{{ $t('home.subtitle') }}</p>
          <p>{{ $t('home.description') }}</p>
          <div class="welcome-actions">
            <el-button type="primary" size="large" @click="$router.push('/datasets')">
              {{ $t('home.uploadData') }}
            </el-button>
            <el-button size="large" @click="$router.push('/datasets')">{{ $t('home.viewDatasets') }}</el-button>
            <el-button size="large" @click="$router.push('/runs')">{{ $t('home.viewRuns') }}</el-button>
            <el-button size="large" @click="$router.push('/compare')">{{ $t('home.compareModels') }}</el-button>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="20" class="feature-row">
      <el-col :span="8">
        <el-card>
          <template #header>
            <div class="card-header">
              <el-icon><component :is="Upload" /></el-icon>
              <span>{{ $t('home.dataAccess.title') }}</span>
            </div>
          </template>
          <p>{{ $t('home.dataAccess.desc') }}</p>
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card>
          <template #header>
            <div class="card-header">
              <el-icon><component :is="Cpu" /></el-icon>
              <span>{{ $t('home.autoTrain.title') }}</span>
            </div>
          </template>
          <p>{{ $t('home.autoTrain.desc') }}</p>
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card>
          <template #header>
            <div class="card-header">
              <el-icon><component :is="TrendCharts" /></el-icon>
              <span>{{ $t('home.resultAnalysis.title') }}</span>
            </div>
          </template>
          <p>{{ $t('home.resultAnalysis.desc') }}</p>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="20" class="guide-row">
      <el-col :span="24">
        <h2 class="guide-title">{{ $t('home.quickStart') }}</h2>
      </el-col>
      <el-col :span="8">
        <el-card>
          <template #header>
            <div class="card-header">
              <el-icon><component :is="Upload" /></el-icon>
              <span>{{ $t('home.step1') }}</span>
            </div>
          </template>
          <p>{{ $t('home.step1Desc') }}</p>
          <el-button type="primary" text @click="$router.push('/datasets')">{{ $t('home.goUpload') }}</el-button>
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card>
          <template #header>
            <div class="card-header">
              <el-icon><component :is="Cpu" /></el-icon>
              <span>{{ $t('home.step2') }}</span>
            </div>
          </template>
          <p>{{ $t('home.step2Desc') }}</p>
          <el-button type="primary" text @click="$router.push('/runs')">{{ $t('home.goTrain') }}</el-button>
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card>
          <template #header>
            <div class="card-header">
              <el-icon><component :is="TrendCharts" /></el-icon>
              <span>{{ $t('home.step3') }}</span>
            </div>
          </template>
          <p>{{ $t('home.step3Desc') }}</p>
          <el-button type="primary" text @click="$router.push('/compare')">{{ $t('home.goCompare') }}</el-button>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { useI18n } from 'vue-i18n'
import { Upload, Cpu, TrendCharts } from '@element-plus/icons-vue'

const { t } = useI18n()
</script>

<style scoped>
.home {
  padding: 20px 0;
}

.welcome-card {
  text-align: center;
  padding: 40px 20px;
  background: linear-gradient(135deg, #f5f7fa 0%, #e4e7ed 100%);
}

.welcome-card h1 {
  margin: 0 0 10px 0;
  color: #303133;
}

.subtitle {
  font-size: 18px;
  color: #606266;
  margin-bottom: 20px;
}

.welcome-actions {
  margin-top: 24px;
  display: flex;
  justify-content: center;
  gap: 12px;
  flex-wrap: wrap;
}

.feature-row {
  margin-top: 20px;
}

.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: bold;
}

.guide-row {
  margin-top: 20px;
}

.guide-title {
  color: #303133;
  margin-bottom: 8px;
  padding-left: 8px;
  border-left: 4px solid #409eff;
}

.guide-row ul {
  padding-left: 20px;
  line-height: 1.8;
}

.guide-row pre {
  background: #f5f7fa;
  padding: 12px;
  border-radius: 4px;
  overflow-x: auto;
}
</style>
