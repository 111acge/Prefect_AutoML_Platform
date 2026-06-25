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
          <p class="subtitle">端到端全自动机器学习平台</p>
          <p>基于 Prefect 工作流编排 + AutoGluon 模型内核，实现数据上传、自动训练、评估与预测。</p>
          <div class="welcome-actions">
            <el-button type="primary" size="large" @click="$router.push('/datasets')">
              开始上传数据
            </el-button>
            <el-button size="large" @click="$router.push('/datasets')">查看数据集</el-button>
            <el-button size="large" @click="$router.push('/runs')">查看训练任务</el-button>
            <el-button size="large" @click="$router.push('/compare')">模型对比</el-button>
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
              <span>数据接入</span>
            </div>
          </template>
          <p>支持 CSV / Excel / Parquet 文件上传，自动推断 Schema 与字段类型。</p>
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card>
          <template #header>
            <div class="card-header">
              <el-icon><component :is="Cpu" /></el-icon>
              <span>自动训练</span>
            </div>
          </template>
          <p>Prefect 编排全流程，AutoGluon 自动搜索最优模型并集成。</p>
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card>
          <template #header>
            <div class="card-header">
              <el-icon><component :is="TrendCharts" /></el-icon>
              <span>结果分析</span>
            </div>
          </template>
          <p>查看评估指标、模型排行榜、特征重要性与 SHAP 可解释性报告。</p>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="20" class="guide-row">
      <el-col :span="24">
        <h2 class="guide-title">快速开始</h2>
      </el-col>
      <el-col :span="8">
        <el-card>
          <template #header>
            <div class="card-header">
              <el-icon><component :is="Upload" /></el-icon>
              <span>1. 上传数据</span>
            </div>
          </template>
          <p>进入数据集页面，上传 CSV / Excel / Parquet 文件，系统自动推断 Schema 和目标列。</p>
          <el-button type="primary" text @click="$router.push('/datasets')">去上传</el-button>
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card>
          <template #header>
            <div class="card-header">
              <el-icon><component :is="Cpu" /></el-icon>
              <span>2. 启动训练</span>
            </div>
          </template>
          <p>选择数据集和目标列，设置时间预算，一键启动 AutoML 训练，Prefect 编排全流程。</p>
          <el-button type="primary" text @click="$router.push('/runs')">去训练</el-button>
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card>
          <template #header>
            <div class="card-header">
              <el-icon><component :is="TrendCharts" /></el-icon>
              <span>3. 查看结果</span>
            </div>
          </template>
          <p>训练完成后查看指标、排行榜、特征重要性和报告，并下载模型进行预测。</p>
          <el-button type="primary" text @click="$router.push('/compare')">去对比</el-button>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { Upload, Cpu, TrendCharts } from '@element-plus/icons-vue'
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
