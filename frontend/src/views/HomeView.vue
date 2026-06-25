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
        <el-card>
          <template #header>
            <div class="card-header">
              <el-icon><component :is="Document" /></el-icon>
              <span>如何解读训练报告</span>
            </div>
          </template>

          <el-collapse>
            <el-collapse-item title="1. 任务概览" name="overview">
              <p>报告顶部会展示任务 ID、状态、Preset、主要指标、随机种子、样本数、特征数等基本信息。</p>
              <ul>
                <li><b>Preset</b>：训练质量预设。选择「自动选择」时，系统会根据数据规模、特征数、内存等自动选择；也可以手动指定 medium_quality 或 best_quality。</li>
                <li><b>随机种子</b>：固定后可使数据划分和模型初始化更可复现；留空则由系统随机决定。</li>
                <li><b>样本数 / 特征数</b>：训练数据的总行数与总列数（含目标列）。</li>
              </ul>
            </el-collapse-item>

            <el-collapse-item title="2. 评估指标" name="metrics">
              <p>评估指标使用测试集（默认 20%）计算，不同任务类型关注的指标不同：</p>
              <ul>
                <li><b>accuracy</b>：准确率，分类正确样本占总样本比例。</li>
                <li><b>log_loss</b>：对数损失，越小越好，反映模型预测概率的校准程度。</li>
                <li><b>f1 / mcc / balanced_accuracy</b>：适用于类别不平衡场景。</li>
                <li><b>root_mean_squared_error (RMSE)</b>：回归任务常用，单位与目标列一致，越小越好。</li>
              </ul>
              <p>注意：某些指标可能为负值（如 log_loss、RMSE 在 AutoGluon 内部优化时取负），比较时看绝对值大小即可。</p>
            </el-collapse-item>

            <el-collapse-item title="3. 模型排行榜" name="leaderboard">
              <p>排行榜列出 AutoGluon 尝试的所有基模型及集成模型，通常按验证集分数排序：</p>
              <ul>
                <li><b>model</b>：模型名称，如 LightGBMXT、WeightedEnsemble_L2。</li>
                <li><b>score_val</b>：验证集分数，对于损失类指标越接近 0 越好。</li>
                <li><b>pred_time_val</b>：验证集预测耗时，反映模型推理速度。</li>
                <li><b>fit_time</b>：训练耗时。</li>
                <li><b>stack_level</b>：模型堆叠层级，层级越高表示参与集成的程度越深。</li>
              </ul>
            </el-collapse-item>

            <el-collapse-item title="4. 特征重要性" name="importance">
              <p>特征重要性帮助理解模型更依赖哪些输入字段，常见列含义：</p>
              <ul>
                <li><b>feature</b>：特征名称。</li>
                <li><b>importance</b>：重要性数值，通常越大表示该特征对预测贡献越大。</li>
                <li><b>stddev / p_value</b>：重要性的统计稳定性，p_value 越小越可信。</li>
              </ul>
              <p>建议优先关注和业务相关、重要性高的特征，必要时可对低重要性特征进行降维或剔除。</p>
            </el-collapse-item>

            <el-collapse-item title="5. 使用模型预测" name="predict">
              <p>训练完成后，可在任务详情页点击“使用模型预测”，输入 JSON 格式样本数组。例如：</p>
              <pre>[{"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2}]</pre>
              <p>分类任务会返回预测类别和各类别概率；回归任务只返回预测值。</p>
            </el-collapse-item>
          </el-collapse>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { Upload, Cpu, TrendCharts, Document } from '@element-plus/icons-vue'
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
