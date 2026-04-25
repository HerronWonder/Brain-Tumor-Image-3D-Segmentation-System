<template>
  <div class="app-container">
    <!-- Left panel: workflow controls -->
    <aside class="sidebar">
      <UploadPanel @inferenceComplete="handleRender" @resetView="renderData = null" />
    </aside>

    <!-- Right panel: 3D MPR rendering canvas -->
    <main class="viewer-container">
      <div v-if="!renderData" class="placeholder">
        <h3 style="font-size: 1.5rem; margin-bottom: 10px;">🧠 Waiting For Imaging Data...</h3>
        <p>Upload T1, T1ce, T2, and FLAIR volumes in the left panel to start 3D rendering.</p>
      </div>
      
      <!-- VtkViewer rendering engine mount point -->
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

// Rendering payload returned by backend.
const renderData = ref(null);

const handleRender = (data) => {
  renderData.value = data;
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