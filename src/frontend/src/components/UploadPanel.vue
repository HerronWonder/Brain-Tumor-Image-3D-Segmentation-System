<template>
  <div class="upload-panel">
    <h2>Medical Imaging 3D Segmentation</h2>
    <p class="subtitle">Multimodal Analysis Engine</p>

    <div class="upload-box" @drop.prevent="handleDrop" @dragover.prevent v-if="!inferenceDone">
      <input type="file" multiple @change="handleFileSelect" ref="fileInput" accept=".nii,.nii.gz" hidden />
      <button class="btn-primary" @click="$refs.fileInput.click()">
        📂 Upload 4 Modality Files
      </button>
      <p class="drag-text">T1, T1ce, T2, FLAIR</p>
    </div>

    <ul class="file-list" v-if="files.length > 0 && !inferenceDone">
      <li v-for="(file, index) in files" :key="index">📄 {{ file.name }}</li>
    </ul>

    <div class="model-select" v-if="!inferenceDone">
      <label for="modelType">Model</label>
      <select id="modelType" v-model="selectedModel">
        <option value="unet">3D U-Net (Baseline)</option>
        <option value="mamba">Mamba3D (Hybrid)</option>
      </select>
    </div>

    <button 
      class="btn-submit" 
      v-if="!inferenceDone"
      :disabled="files.length !== 4 || isUploading" 
      @click="submitForInference"
    >
      <span v-if="isUploading" class="spinner">⚙️ Task Running, Please Wait...</span>
      <span v-else>🚀 Start 3D Tumor Segmentation</span>
    </button>

    <div class="task-progress" v-if="isUploading">
      <div class="progress-header">
        <span>Task ID: {{ activeTaskId || 'Pending' }}</span>
        <span>{{ taskProgress }}%</span>
      </div>
      <div class="progress-track">
        <div class="progress-fill" :style="{ width: `${taskProgress}%` }"></div>
      </div>
      <p class="progress-text">{{ taskStatusText || 'Initializing task...' }}</p>
    </div>

    <div v-if="errorMsg" class="error-msg">❌ {{ errorMsg }}</div>

    <div class="clinical-report" v-if="inferenceDone && clinicalMetrics">
      <h3>📊 Automated Clinical Analysis Report</h3>
      <div class="metric-card">
        <div class="metric-item">
          <span class="dot necrotic"></span>
          <span class="label">Necrotic / Non-Enhancing Core (NCR)</span>
          <span class="value">{{ clinicalMetrics.necrotic_cm3 }} cm³</span>
        </div>
        <div class="metric-item">
          <span class="dot edema"></span>
          <span class="label">Peritumoral Edema (ED)</span>
          <span class="value">{{ clinicalMetrics.edema_cm3 }} cm³</span>
        </div>
        <div class="metric-item">
          <span class="dot enhancing"></span>
          <span class="label">Enhancing Tumor (ET)</span>
          <span class="value">{{ clinicalMetrics.enhancing_cm3 }} cm³</span>
        </div>
        <div class="divider"></div>
        <div class="metric-item total">
          <span class="label">🔥 Total Abnormal Tissue Volume (WT)</span>
          <span class="value">{{ clinicalMetrics.total_cm3 }} cm³</span>
        </div>
      </div>
      <div class="export-actions" v-if="maskDownloadUrl || reportDownloadUrl">
        <a v-if="maskDownloadUrl" :href="maskDownloadUrl" class="btn-export" download>⬇️ Export 3D Mask</a>
        <a v-if="reportDownloadUrl" :href="reportDownloadUrl" class="btn-export secondary" download>⬇️ Export Structured Report</a>
      </div>
      <button class="btn-reset" @click="resetPanel">🔄 Analyze Next Case</button>
    </div>

  </div>
</template>

<script setup>
import { onBeforeUnmount, ref } from 'vue';
import { apiClient, toApiUrl } from '@/services/api';

const files = ref([]);
const isUploading = ref(false);
const errorMsg = ref('');
const inferenceDone = ref(false);
const clinicalMetrics = ref(null);
const selectedModel = ref('unet');
const activeTaskId = ref('');
const taskProgress = ref(0);
const taskStatusText = ref('');
const pollIntervalMs = ref(1000);
const maskDownloadUrl = ref('');
const reportDownloadUrl = ref('');

let pollTimer = null;

const emit = defineEmits(['inferenceComplete', 'resetView']);

const handleFileSelect = (event) => { files.value = Array.from(event.target.files); };
const handleDrop = (event) => { files.value = Array.from(event.dataTransfer.files); };

const clearPollTimer = () => {
  if (pollTimer) {
    clearTimeout(pollTimer);
    pollTimer = null;
  }
};

const schedulePoll = (statusUrl) => {
  clearPollTimer();
  pollTimer = setTimeout(() => pollTaskStatus(statusUrl), pollIntervalMs.value);
};

const completeTask = (statusData) => {
  clinicalMetrics.value = statusData.metrics || null;
  inferenceDone.value = true;
  isUploading.value = false;
  taskProgress.value = 100;
  taskStatusText.value = statusData.message || 'Task completed';

  maskDownloadUrl.value = statusData.mask_url ? toApiUrl(statusData.mask_url) : '';
  reportDownloadUrl.value = statusData.report_url ? toApiUrl(statusData.report_url) : '';

  if (maskDownloadUrl.value) {
    emit('inferenceComplete', {
      maskUrl: maskDownloadUrl.value,
      originalFiles: files.value
    });
  }
};

const failTask = (statusData) => {
  isUploading.value = false;
  errorMsg.value = statusData?.error || statusData?.message || 'Inference task failed';
};

const pollTaskStatus = async (statusUrl) => {
  try {
    const response = await apiClient.get(statusUrl);
    const statusData = response.data;

    taskProgress.value = Number(statusData.progress || 0);
    taskStatusText.value = statusData.message || '';

    if (statusData.status === 'COMPLETED') {
      clearPollTimer();
      completeTask(statusData);
      return;
    }

    if (statusData.status === 'FAILED') {
      clearPollTimer();
      failTask(statusData);
      return;
    }

    if (isUploading.value) {
      schedulePoll(statusUrl);
    }
  } catch (error) {
    clearPollTimer();
    isUploading.value = false;
    errorMsg.value = error.response?.data?.error || 'Failed to fetch task status. Please retry.';
  }
};

const submitForInference = async () => {
  if (files.value.length !== 4) {
    errorMsg.value = 'Please upload exactly 4 NIfTI files (T1, T1ce, T2, FLAIR).';
    return;
  }
  
  isUploading.value = true;
  errorMsg.value = '';
  taskProgress.value = 0;
  taskStatusText.value = 'Task submitted. Waiting for gateway scheduling...';
  clinicalMetrics.value = null;
  maskDownloadUrl.value = '';
  reportDownloadUrl.value = '';
  clearPollTimer();
  
  const formData = new FormData();
  files.value.forEach(file => formData.append('files', file));
  formData.append('model', selectedModel.value);

  try {
    const response = await apiClient.post('/api/predict', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });

    activeTaskId.value = response.data.task_id || '';
    taskStatusText.value = response.data.message || 'Task queued';
    pollIntervalMs.value = Number(response.data.poll_interval_ms || 1000);

    const statusUrl = response.data.status_url || `/api/tasks/${activeTaskId.value}`;
    await pollTaskStatus(statusUrl);
  } catch (error) {
    clearPollTimer();
    errorMsg.value = error.response?.data?.error || 'Server timeout or network error.';
    isUploading.value = false;
  }
};

const resetPanel = () => {
  clearPollTimer();
  files.value = [];
  activeTaskId.value = '';
  taskProgress.value = 0;
  taskStatusText.value = '';
  inferenceDone.value = false;
  clinicalMetrics.value = null;
  maskDownloadUrl.value = '';
  reportDownloadUrl.value = '';
  errorMsg.value = '';
  emit('resetView');
};

onBeforeUnmount(() => {
  clearPollTimer();
});
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
.task-progress { margin-top: 1rem; background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; }
.progress-header { display: flex; justify-content: space-between; font-size: 0.82rem; color: #334155; margin-bottom: 8px; }
.progress-track { width: 100%; height: 10px; border-radius: 999px; background: #dbeafe; overflow: hidden; }
.progress-fill { height: 100%; background: linear-gradient(90deg, #2563eb, #10b981); transition: width 0.35s ease; }
.progress-text { margin: 8px 0 0; color: #334155; font-size: 0.83rem; }
.model-select { margin-top: 1rem; display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.model-select label { color: #334155; font-size: 0.9rem; font-weight: 600; }
.model-select select { flex: 1; border: 1px solid #cbd5e1; background: #ffffff; border-radius: 6px; padding: 8px 10px; color: #0f172a; }

/* Clinical report card */
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
.export-actions { margin-top: 1rem; display: grid; grid-template-columns: 1fr; gap: 10px; }
.btn-export { display: block; text-align: center; text-decoration: none; border-radius: 6px; background: #0ea5e9; color: #ffffff; font-size: 0.92rem; font-weight: 700; padding: 10px 12px; }
.btn-export.secondary { background: #0f766e; }
.spinner { animation: pulse 1.5s infinite; }
@keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
</style>