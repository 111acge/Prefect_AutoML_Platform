<template>
  <div ref="chartRef" class="echart-container" :style="{ height: height }"></div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch, nextTick } from 'vue'
import * as echarts from 'echarts'

const props = defineProps({
  option: {
    type: Object,
    required: true,
  },
  height: {
    type: String,
    default: '300px',
  },
})

const chartRef = ref(null)
let chartInstance = null
let resizeObserver = null

const hasValidSize = (el) => {
  const rect = el.getBoundingClientRect()
  return rect.width > 0 && rect.height > 0
}

const initChart = () => {
  if (!chartRef.value) return
  // 容器未获得实际尺寸时（如在隐藏 tab 中）延迟初始化，避免 ECharts 报 0 width/height
  if (!hasValidSize(chartRef.value)) return

  if (chartInstance) {
    chartInstance.dispose()
  }
  chartInstance = echarts.init(chartRef.value)
  chartInstance.setOption(props.option, true)
}

const handleResize = () => {
  chartInstance?.resize()
}

onMounted(() => {
  nextTick(initChart)
  window.addEventListener('resize', handleResize)

  // 使用 ResizeObserver 在容器从隐藏变为可见时自动初始化/重绘
  if (typeof ResizeObserver !== 'undefined' && chartRef.value) {
    resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect
        if (width > 0 && height > 0) {
          if (!chartInstance) {
            initChart()
          } else {
            chartInstance.resize()
          }
        }
      }
    })
    resizeObserver.observe(chartRef.value)
  }
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
  if (resizeObserver && chartRef.value) {
    resizeObserver.unobserve(chartRef.value)
    resizeObserver.disconnect()
    resizeObserver = null
  }
  chartInstance?.dispose()
  chartInstance = null
})

watch(
  () => props.option,
  () => {
    if (chartInstance) {
      chartInstance.setOption(props.option, true)
    } else if (chartRef.value && hasValidSize(chartRef.value)) {
      initChart()
    }
  },
  { deep: true }
)
</script>

<style scoped>
.echart-container {
  width: 100%;
  min-width: 0;
}
</style>
