<template>
  <el-container class="app-container">
    <el-header class="app-header">
      <div class="logo">
        <el-icon size="24"><DataLine /></el-icon>
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
        <el-menu-item index="/runs">训练任务</el-menu-item>
        <el-menu-item index="/compare">模型对比</el-menu-item>
      </el-menu>
    </el-header>
    <el-main>
      <router-view />
    </el-main>
    <el-footer class="app-footer">
      <span>Powered by Prefect + AutoGluon + FastAPI + Vue 3</span>
    </el-footer>
  </el-container>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { DataLine } from '@element-plus/icons-vue'

const route = useRoute()
const activeIndex = computed(() => route.path)
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
</style>
