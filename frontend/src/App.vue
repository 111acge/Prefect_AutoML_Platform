<template>
  <el-container class="app-container">
    <el-header class="app-header">
      <div class="logo">
        <el-icon size="24"><component :is="DataLine" /></el-icon>
        <span>AutoML</span>
      </div>
      <el-menu
        :default-active="activeIndex"
        class="app-menu"
        mode="horizontal"
        :ellipsis="false"
        router
      >
        <el-menu-item index="/">首页</el-menu-item>
        <el-menu-item index="/datasets">数据集</el-menu-item>
        <el-menu-item index="/runs">
          训练任务
          <el-badge v-if="runningCount > 0" :value="runningCount" class="running-badge" />
        </el-menu-item>
        <el-menu-item index="/compare">模型对比</el-menu-item>
      </el-menu>
    </el-header>
    <el-main>
      <router-view />
    </el-main>
    <el-footer class="app-footer">
      <div class="footer-content">
        <span>Prefect AutoML Platform v0.1.0</span>
        <span class="footer-divider">|</span>
        <span>Powered by Prefect + AutoGluon + FastAPI + Vue 3</span>
        <span class="footer-divider">|</span>
        <el-link type="primary" href="https://github.com" target="_blank">文档</el-link>
        <span class="footer-divider">|</span>
        <el-link type="primary" href="https://github.com" target="_blank">问题反馈</el-link>
      </div>
    </el-footer>
  </el-container>
</template>

<script setup>
import { computed, ref, onMounted, onUnmounted } from 'vue'
import { useRoute } from 'vue-router'
import { DataLine } from '@element-plus/icons-vue'
import { runApi } from '@/api'

const route = useRoute()
const activeIndex = computed(() => route.path)
const runningCount = ref(0)
let timer = null

const checkRunning = async () => {
  try {
    const res = await runApi.list()
    runningCount.value = res.data.filter((r) => r.status === 'running').length
  } catch {
    runningCount.value = 0
  }
}

onMounted(() => {
  checkRunning()
  timer = setInterval(checkRunning, 5000)
})

onUnmounted(() => {
  if (timer) clearInterval(timer)
})
</script>

<style scoped>
.app-container {
  min-height: 100vh;
}

.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background-color: #fff;
  border-bottom: 1px solid #e4e7ed;
}

.logo {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 20px;
  font-weight: bold;
  color: #409eff;
}

.app-menu {
  border-bottom: none;
  flex-shrink: 0;
  white-space: nowrap;
}

.app-footer {
  display: flex;
  align-items: center;
  justify-content: center;
  color: #909399;
  font-size: 14px;
}

.footer-content {
  display: flex;
  align-items: center;
  gap: 12px;
}

.footer-divider {
  color: #dcdfe6;
}

.running-badge {
  margin-left: 6px;
  vertical-align: middle;
}

:deep(.running-badge .el-badge__content) {
  border: none;
}
</style>
