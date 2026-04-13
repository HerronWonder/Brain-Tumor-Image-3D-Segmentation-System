<template>
  <div class="upload-panel">
    <h2>医疗影像 3D 分割系统</h2>
    <p class="subtitle">多模态分析引擎</p>

    <div class="upload-box" @drop.prevent="handleDrop" @dragover.prevent v-if="!inferenceDone">
      <input type="file" multiple @change="handleFileSelect" ref="fileInput" accept=".nii,.nii.gz" hidden />
      <button class="btn-primary" @click="$refs.fileInput.click()">
        📂 请上传 4 个模态文件
      </button>
      <p class="drag-text">T1, T1ce, T2, FLAIR</p>
    </div>

    <ul class="file-list" v-if="files.length > 0 && !inferenceDone">
      <li v-for="(file, index) in files" :key="index">📄 {{ file.name }}</li>
    </ul>

    <button 
      class="btn-submit" 
      v-if="!inferenceDone"
      :disabled="files.length !== 4 || isUploading" 
      @click="submitForInference"
    >
      <span v-if="isUploading" class="spinner">⚙️ 推理中，请耐心等待...</span>
      <span v-else>🚀 开始 3D 肿瘤分割</span>
    </button>

    <div v-if="errorMsg" class="error-msg">❌ {{ errorMsg }}</div>

    <div class="clinical-report" v-if="inferenceDone && clinicalMetrics">
      <h3>📊 自动化临床分析报告</h3>
      <div class="metric-card">
        <div class="metric-item">
          <span class="dot necrotic"></span>
          <span class="label">坏死与非增强核心 (NCR)</span>
          <span class="value">{{ clinicalMetrics.necrotic_cm3 }} cm³</span>
        </div>
        <div class="metric-item">
          <span class="dot edema"></span>
          <span class="label">瘤周水肿区 (ED)</span>
          <span class="value">{{ clinicalMetrics.edema_cm3 }} cm³</span>
        </div>
        <div class="metric-item">
          <span class="dot enhancing"></span>
          <span class="label">增强肿瘤区 (ET)</span>
          <span class="value">{{ clinicalMetrics.enhancing_cm3 }} cm³</span>
        </div>
        <div class="divider"></div>
        <div class="metric-item total">
          <span class="label">🔥 异常组织总体积 (WT)</span>
          <span class="value">{{ clinicalMetrics.total_cm3 }} cm³</span>
        </div>
      </div>
      <button class="btn-reset" @click="resetPanel">🔄 分析下一个病例</button>
    </div>

  </div>
</template>

<script setup>
import { ref } from 'vue';
import axios from 'axios';

const files = ref([]);
const isUploading = ref(false);
const errorMsg = ref('');
const inferenceDone = ref(false);
const clinicalMetrics = ref(null);

const emit = defineEmits(['inferenceComplete', 'resetView']);

const handleFileSelect = (event) => { files.value = Array.from(event.target.files); };
const handleDrop = (event) => { files.value = Array.from(event.dataTransfer.files); };

const submitForInference = async () => {
  if (files.value.length !== 4) {
    errorMsg.value = "请严格上传 4 个对应模态的 NIfTI 文件！";
    return;
  }
  
  isUploading.value = true;
  errorMsg.value = '';
  
  const formData = new FormData();
  files.value.forEach(file => formData.append('files', file));

  try {
    // 实验室服务器的IP
    const SERVER_IP = "localhost"; 
    
    const response = await axios.post(`http://${SERVER_IP}:5000/api/predict`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    
    // 获取后端传来的临床体积指标
    clinicalMetrics.value = response.data.metrics;
    inferenceDone.value = true;

    emit('inferenceComplete', {
      maskUrl: `http://${SERVER_IP}:5000${response.data.mask_url}`,
      originalFiles: files.value 
    });
  } catch (error) {
    errorMsg.value = error.response?.data?.error || "服务器响应超时或网络错误";
  } finally {
    isUploading.value = false;
  }
};

const resetPanel = () => {
  files.value = [];
  inferenceDone.value = false;
  clinicalMetrics.value = null;
  errorMsg.value = '';
  emit('resetView'); // 告诉主视图清空画布
};
</script>

<style scoped>
.upload-panel { padding: 2rem; background: #ffffff; border-radius: 12px; height: 100%; box-shadow: 0 4px 12px rgba(0,0,0,0.05); display: flex; flex-direction: column;}
.subtitle { color: #64748b; margin-bottom: 2rem; font-size: 0.9rem;}
.upload-box { border: 2px dashed #cbd5e1; padding: 2.5rem 1rem; text-align: center; border-radius: 8px; transition: 0.3s; background: #f8fafc;}
.upload-box:hover { border-color: #3b82f6; background: #eff6ff; }
.btn-primary, .btn-submit, .btn-reset { color: white; border: none; padding: 12px 20px; border-radius: 6px; cursor: pointer; font-weight: bold; width: 100%; transition: 0.2s; font-size: 1rem;}
.btn-primary { background: #3b82f6; }
.btn-primary:hover { background: #2563eb; }
.btn-submit { margin-top: 1.5rem; background: #10b981; }
.btn-submit:disabled { background: #9ca3af; cursor: not-allowed; }
.btn-submit:hover:not(:disabled) { background: #059669; }
.btn-reset { margin-top: 2rem; background: #64748b; }
.btn-reset:hover { background: #475569; }
.file-list { margin-top: 1rem; list-style: none; padding: 0; font-size: 0.85rem; color: #475569; }
.drag-text { margin-top: 0.8rem; color: #94a3b8; font-size: 0.85rem; }
.error-msg { margin-top: 1rem; color: #ef4444; font-size: 0.9rem; font-weight: bold; text-align: center;}

/* 临床报告卡片样式 */
.clinical-report h3 { color: #0f172a; margin-bottom: 1rem; font-size: 1.2rem; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px;}
.metric-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.5rem; display: flex; flex-direction: column; gap: 12px; }
.metric-item { display: flex; align-items: center; justify-content: space-between; font-size: 0.95rem; color: #334155;}
.dot { width: 12px; height: 12px; border-radius: 50%; margin-right: 10px; display: inline-block;}
.dot.necrotic { background: #ff3333; }
.dot.edema { background: #33ff33; }
.dot.enhancing { background: #ffcc00; }
.label { flex: 1; font-weight: 500;}
.value { font-weight: bold; font-family: monospace; font-size: 1.1rem; color: #0f172a;}
.divider { height: 1px; background: #cbd5e1; margin: 8px 0; }
.metric-item.total .label { font-weight: bold; color: #e11d48; }
.metric-item.total .value { color: #e11d48; font-size: 1.2rem; }
.spinner { animation: pulse 1.5s infinite; }
@keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
</style>