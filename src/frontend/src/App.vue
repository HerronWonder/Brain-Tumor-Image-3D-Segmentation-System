<template>
  <div class="app-container">
    <!-- 左侧：控制台 (解耦) -->
    <aside class="sidebar">
      <UploadPanel @inferenceComplete="handleRender" @resetView="renderData = null" />
    </aside>

    <!-- 右侧：3D MPR 渲染画布 -->
    <main class="viewer-container">
      <div v-if="!renderData" class="placeholder">
        <h3 style="font-size: 1.5rem; margin-bottom: 10px;">🧠 等待影像接入...</h3>
        <p>请在左侧上传 T1, T1ce, T2, FLAIR 序列以启动 3D 渲染引擎</p>
      </div>
      
      <!-- 这里完美嵌入了 VtkViewer 引擎！ -->
      <div v-else class="vtk-mount-point">
        <VtkViewer 
          :maskUrl="renderData.maskUrl" 
          :originalFiles="renderData.originalFiles" 
        />
      </div>
    </main>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import UploadPanel from './components/UploadPanel.vue';
import VtkViewer from './components/VtkViewer.vue';

// 存储后端返回的渲染数据
const renderData = ref(null);

const handleRender = (data) => {
  renderData.value = data;
  console.log("准备渲染 3D 掩码，下载地址：", data.maskUrl);
};
</script>

<style>
body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; overflow: hidden; }
.app-container { display: flex; height: 100vh; background: #0b0f19; }
.sidebar { width: 420px; padding: 20px; background: #ffffff; z-index: 10; box-shadow: 2px 0 15px rgba(0,0,0,0.1); }
.viewer-container { flex: 1; display: flex; align-items: center; justify-content: center; padding: 20px; }
.placeholder { text-align: center; color: #475569; padding: 40px; border: 2px dashed #334155; border-radius: 12px; }
.vtk-mount-point { width: 100%; height: 100%; }
</style>