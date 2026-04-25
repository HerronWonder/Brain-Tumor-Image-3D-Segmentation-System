<template>
  <div class="vtk-viewer">
    <div v-if="loading" class="loading-overlay">
      <h2>🔄 Loading And Parsing 3D Medical Data...</h2>
      <p>This may take a few seconds while the 3D volume is being prepared...</p>
    </div>

    <div class="toolbar" v-if="!loading">
      
      <div class="mode-switch">
        <button :class="{ active: viewMode === '2D' }" @click="switchMode('2D')">🎞️ 2D Orthogonal Slices</button>
        <button :class="{ active: viewMode === '3D' }" @click="switchMode('3D')">🧊 3D Volume View</button>
      </div>
      <div class="divider-vertical"></div>

      <template v-if="viewMode === '2D'">
        <div class="control-group">
          <label>🧭 View: </label>
          <select v-model="sliceMode" @change="updateSliceMode">
            <option :value="2">Axial</option>
            <option :value="1">Coronal</option>
            <option :value="0">Sagittal</option>
          </select>
        </div>
        
        <div class="control-group slider-group">
          <label>🔪 Slice Depth ({{ currentSlice }}/{{ maxSlice }}):</label>
          <input type="range" min="0" :max="maxSlice" v-model.number="currentSlice" @input="updateSlice">
        </div>
      </template>

      <template v-if="viewMode === '3D'">
        <div class="control-group">
          <span style="color: #10b981; font-weight: bold;">🖱️ Tip: Left drag rotate, Ctrl+drag roll, wheel zoom/pan</span>
        </div>
      </template>

      <div class="divider-vertical"></div>

      <div class="control-group">
        <label>👁️ Tumor Opacity:</label>
        <input type="range" min="0" max="1" step="0.1" v-model.number="maskOpacity" @input="updateOpacity">
      </div>
    </div>

    <div ref="vtkContainer" class="vtk-canvas" v-show="!loading"></div>
    <div v-if="renderError" class="render-error">{{ renderError }}</div>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, nextTick } from 'vue';
import * as nifti from 'nifti-reader-js';
import { apiClient } from '@/services/api';

import '@kitware/vtk.js/Rendering/Profiles/All';
import vtkGenericRenderWindow from '@kitware/vtk.js/Rendering/Misc/GenericRenderWindow';
import vtkImageMapper from '@kitware/vtk.js/Rendering/Core/ImageMapper';
import vtkImageSlice from '@kitware/vtk.js/Rendering/Core/ImageSlice';
import vtkImageData from '@kitware/vtk.js/Common/DataModel/ImageData';
import vtkDataArray from '@kitware/vtk.js/Common/Core/DataArray';
import vtkColorTransferFunction from '@kitware/vtk.js/Rendering/Core/ColorTransferFunction';
import vtkPiecewiseFunction from '@kitware/vtk.js/Common/DataModel/PiecewiseFunction';

import vtkVolume from '@kitware/vtk.js/Rendering/Core/Volume';
import vtkVolumeMapper from '@kitware/vtk.js/Rendering/Core/VolumeMapper';
import vtkInteractorStyleImage from '@kitware/vtk.js/Interaction/Style/InteractorStyleImage';
import vtkInteractorStyleTrackballCamera from '@kitware/vtk.js/Interaction/Style/InteractorStyleTrackballCamera';

const props = defineProps({ maskUrl: String, originalFiles: Array });

const loading = ref(true);
const vtkContainer = ref(null);
const viewMode = ref('2D'); 
const sliceMode = ref(2); 
const currentSlice = ref(0);
const maxSlice = ref(100);
const maskOpacity = ref(0.7);
const renderError = ref('');

let renderWindow, renderer, renderWindowContainer, interactor;
let style2D, style3D;
let bgMapper, bgSlice, maskMapper, maskSlice;
let bgVolMapper, bgVolActor, maskVolMapper, maskVolActor;
let bgVtkImg, maskVtkImg; 

// Keep a ResizeObserver reference for cleanup on unmount.
let resizeObserver = null;

const initVtk = () => {
  renderWindowContainer = vtkGenericRenderWindow.newInstance({
    background: [0.05, 0.05, 0.08], 
  });
  renderWindowContainer.setContainer(vtkContainer.value);
  renderWindow = renderWindowContainer.getRenderWindow();
  renderer = renderWindowContainer.getRenderer();
  interactor = renderWindowContainer.getInteractor();

  style2D = vtkInteractorStyleImage.newInstance();
  style2D.setInteractionMode('IMAGE2D');
  style3D = vtkInteractorStyleTrackballCamera.newInstance();
};

const parseNiftiToVtk = (arrayBuffer) => {
  if (nifti.isCompressed(arrayBuffer)) {
    arrayBuffer = nifti.decompress(arrayBuffer);
  }
  if (!nifti.isNIFTI(arrayBuffer)) throw new Error('Invalid NIfTI format');
  
  const header = nifti.readHeader(arrayBuffer);
  const image = nifti.readImage(header, arrayBuffer);
  
  let typedData;
  if (header.datatypeCode === nifti.NIFTI1.TYPE_UINT8) typedData = new Uint8Array(image);
  else if (header.datatypeCode === nifti.NIFTI1.TYPE_INT16) typedData = new Int16Array(image);
  else if (header.datatypeCode === nifti.NIFTI1.TYPE_INT32) typedData = new Int32Array(image);
  else if (header.datatypeCode === nifti.NIFTI1.TYPE_FLOAT32) typedData = new Float32Array(image);
  else typedData = new Float32Array(image);

  const vtkImg = vtkImageData.newInstance({
    spacing: [header.pixDims[1], header.pixDims[2], header.pixDims[3]],
    extent: [0, header.dims[1] - 1, 0, header.dims[2] - 1, 0, header.dims[3] - 1],
  });
  
  vtkImg.getPointData().setScalars(vtkDataArray.newInstance({ values: typedData, name: 'scalars' }));
  return vtkImg;
};

const loadData = async () => {
  try {
    renderError.value = '';
    const t1ceFile = props.originalFiles.find(f => f.name.toLowerCase().includes('t1ce')) || props.originalFiles[0];
    const bgBuffer = await t1ceFile.arrayBuffer();
    bgVtkImg = parseNiftiToVtk(bgBuffer);

    const maskRes = await apiClient.get(props.maskUrl, { responseType: 'arraybuffer' });
    maskVtkImg = parseNiftiToVtk(maskRes.data);

    setup2DPipeline();
    setup3DPipeline();

    // Add all actors once and only toggle visibility afterwards.
    renderer.removeAllActors(); 
    renderer.addActor(bgSlice);
    renderer.addActor(maskSlice);
    renderer.addActor(bgVolActor);
    renderer.addActor(maskVolActor);

    switchMode('2D');
    loading.value = false;

    nextTick(() => {
      if (renderWindowContainer) {
        renderWindowContainer.resize();
        adjustCameraFor2D(); 
      }
    });

  } catch (err) {
    renderError.value = err?.message || '3D rendering failed. Please verify input data and network connectivity.';
    loading.value = false;
  }
};

const setup2DPipeline = () => {
  bgMapper = vtkImageMapper.newInstance();
  bgMapper.setInputData(bgVtkImg);
  bgSlice = vtkImageSlice.newInstance();
  bgSlice.setMapper(bgMapper);
  const dataRange = bgVtkImg.getPointData().getScalars().getRange();
  bgSlice.getProperty().setColorWindow(dataRange[1] - dataRange[0]);
  bgSlice.getProperty().setColorLevel((dataRange[1] + dataRange[0]) / 2.0);

  maskMapper = vtkImageMapper.newInstance();
  maskMapper.setInputData(maskVtkImg);
  maskSlice = vtkImageSlice.newInstance();
  maskSlice.setMapper(maskMapper);

  const ctf = vtkColorTransferFunction.newInstance();
  ctf.addRGBPoint(0, 0.0, 0.0, 0.0); // Explicitly map background to transparent black.
  ctf.addRGBPoint(1, 1.0, 0.2, 0.2); 
  ctf.addRGBPoint(2, 0.2, 1.0, 0.2); 
  ctf.addRGBPoint(4, 1.0, 0.8, 0.0); 

  const pwf = vtkPiecewiseFunction.newInstance();
  pwf.addPoint(0, 0.0); 
  pwf.addPoint(1, maskOpacity.value);
  pwf.addPoint(2, maskOpacity.value);
  pwf.addPoint(4, maskOpacity.value);

  maskSlice.getProperty().setRGBTransferFunction(ctf);
  maskSlice.getProperty().setPiecewiseFunction(pwf);
  maskSlice.getProperty().setInterpolationTypeToNearest(); // Disable interpolation to avoid halo artifacts.

  // Force 2D mask rendering to use its own value range.
  const maskDataRange = maskVtkImg.getPointData().getScalars().getRange();
  maskSlice.getProperty().setColorWindow(maskDataRange[1] - maskDataRange[0]);
  maskSlice.getProperty().setColorLevel((maskDataRange[1] + maskDataRange[0]) / 2.0);
  // Make mask non-pickable so interactions target the base image layer.
  maskSlice.setPickable(false);
};

const setup3DPipeline = () => {
  bgVolMapper = vtkVolumeMapper.newInstance();
  bgVolMapper.setInputData(bgVtkImg);
  bgVolActor = vtkVolume.newInstance();
  bgVolActor.setMapper(bgVolMapper);
  
  const bgDataRange = bgVtkImg.getPointData().getScalars().getRange();
  const bgCtf = vtkColorTransferFunction.newInstance();
  bgCtf.addRGBPoint(bgDataRange[0], 0.0, 0.0, 0.0);
  bgCtf.addRGBPoint(bgDataRange[1], 0.8, 0.8, 0.9);
  
  const bgPwf = vtkPiecewiseFunction.newInstance();
  bgPwf.addPoint(bgDataRange[0], 0.0);
  bgPwf.addPoint(bgDataRange[0] + (bgDataRange[1]-bgDataRange[0])*0.2, 0.0); 
  bgPwf.addPoint(bgDataRange[1], 0.05); 
  
  bgVolActor.getProperty().setRGBTransferFunction(0, bgCtf);
  bgVolActor.getProperty().setScalarOpacity(0, bgPwf);

  maskVolMapper = vtkVolumeMapper.newInstance();
  maskVolMapper.setInputData(maskVtkImg);
  maskVolActor = vtkVolume.newInstance();
  maskVolActor.setMapper(maskVolMapper);
  
  // Use dedicated transfer functions for 3D to avoid cross-mode artifacts.
  const maskVolCtf = vtkColorTransferFunction.newInstance();
  maskVolCtf.addRGBPoint(0, 0.0, 0.0, 0.0); 
  maskVolCtf.addRGBPoint(1, 1.0, 0.2, 0.2); 
  maskVolCtf.addRGBPoint(2, 0.2, 1.0, 0.2); 
  maskVolCtf.addRGBPoint(4, 1.0, 0.8, 0.0); 

  const maskVolPwf = vtkPiecewiseFunction.newInstance();
  maskVolPwf.addPoint(0, 0.0); 
  maskVolPwf.addPoint(1, maskOpacity.value);
  maskVolPwf.addPoint(2, maskOpacity.value);
  maskVolPwf.addPoint(4, maskOpacity.value);

  maskVolActor.getProperty().setRGBTransferFunction(0, maskVolCtf);
  maskVolActor.getProperty().setScalarOpacity(0, maskVolPwf);
  maskVolActor.getProperty().setInterpolationTypeToNearest(); 
};

const switchMode = (mode) => {
  viewMode.value = mode;

  if (mode === '2D') {
    // Show 2D slice actors and hide 3D volumes.
    if(bgVolActor) bgVolActor.setVisibility(false);
    if(maskVolActor) maskVolActor.setVisibility(false);
    if(bgSlice) bgSlice.setVisibility(true);
    if(maskSlice) maskSlice.setVisibility(true);
    
    interactor.setInteractorStyle(style2D);
    renderer.getActiveCamera().setParallelProjection(true);
    updateSliceMode(); 
  } else {
    // Show 3D volumes and hide 2D slice actors.
    if(bgSlice) bgSlice.setVisibility(false);
    if(maskSlice) maskSlice.setVisibility(false);
    if(bgVolActor) bgVolActor.setVisibility(true);
    if(maskVolActor) maskVolActor.setVisibility(true);
    
    interactor.setInteractorStyle(style3D);
    renderer.getActiveCamera().setParallelProjection(false);
    renderer.resetCamera(); 
    renderWindow.render();
  }
};


const updateSliceMode = () => {
  bgMapper.setSlicingMode(sliceMode.value);
  maskMapper.setSlicingMode(sliceMode.value);
  updateSliceBounds();
  currentSlice.value = Math.floor(maxSlice.value / 2);
  
  updateSlice(); // Use the guarded slice update path.
  adjustCameraFor2D();
};


const adjustCameraFor2D = () => {
  const camera = renderer.getActiveCamera();
  const bounds = bgMapper.getInputData().getBounds();
  
  const center = [
    (bounds[0] + bounds[1]) / 2.0,
    (bounds[2] + bounds[3]) / 2.0,
    (bounds[4] + bounds[5]) / 2.0,
  ];
  
  camera.setFocalPoint(center[0], center[1], center[2]);

  const maxDim = Math.max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]);
  camera.setParallelProjection(true);
  
  // Add a safe margin to prevent edge clipping.
  camera.setParallelScale(maxDim / 1.8); 

  const dist = maxDim * 2; 

  if (sliceMode.value === 2) { 
    camera.setPosition(center[0], center[1], center[2] + dist);
    camera.setViewUp(0, 1, 0); 
  } else if (sliceMode.value === 1) { 
    camera.setPosition(center[0], center[1] - dist, center[2]);
    camera.setViewUp(0, 0, 1);
  } else if (sliceMode.value === 0) { 
    camera.setPosition(center[0] + dist, center[1], center[2]);
    camera.setViewUp(0, 0, 1);
  }
  
  renderer.resetCameraClippingRange();
  renderWindow.render();
};

const updateSliceBounds = () => {
  const bounds = bgMapper.getInputData().getExtent(); 
  if (sliceMode.value === 2) maxSlice.value = bounds[5];
  else if (sliceMode.value === 1) maxSlice.value = bounds[3];
  else maxSlice.value = bounds[1];
};

const updateSlice = () => {
  bgMapper.setSlice(currentSlice.value);

  // Hide mask when current slice is outside mask extent.
  const maskExtent = maskMapper.getInputData().getExtent();
  let maskMin = 0, maskMax = 0;

  if (sliceMode.value === 2) { maskMin = maskExtent[4]; maskMax = maskExtent[5]; } // Axial
  else if (sliceMode.value === 1) { maskMin = maskExtent[2]; maskMax = maskExtent[3]; } // Coronal
  else { maskMin = maskExtent[0]; maskMax = maskExtent[1]; } // Sagittal

  if (currentSlice.value >= maskMin && currentSlice.value <= maskMax) {
    if(viewMode.value === '2D') maskSlice.setVisibility(true);
    maskMapper.setSlice(currentSlice.value);
  } else {
    maskSlice.setVisibility(false);
  }

  renderWindow.render();
};


const updateOpacity = () => {
  // Keep 2D and 3D opacity mappings in sync.
  const pwf2D = vtkPiecewiseFunction.newInstance();
  pwf2D.addPoint(0, 0.0);
  pwf2D.addPoint(1, maskOpacity.value);
  pwf2D.addPoint(2, maskOpacity.value);
  pwf2D.addPoint(4, maskOpacity.value);
  if(maskSlice) maskSlice.getProperty().setPiecewiseFunction(pwf2D);

  const pwf3D = vtkPiecewiseFunction.newInstance();
  pwf3D.addPoint(0, 0.0);
  pwf3D.addPoint(1, maskOpacity.value);
  pwf3D.addPoint(2, maskOpacity.value);
  pwf3D.addPoint(4, maskOpacity.value);
  if(maskVolActor) maskVolActor.getProperty().setScalarOpacity(0, pwf3D); 
  
  renderWindow.render();
};


onMounted(() => { 
  initVtk(); 
  loadData(); 

  // Track container size changes and trigger VTK resize/render.
  resizeObserver = new ResizeObserver(() => {
    if (renderWindowContainer) {
      renderWindowContainer.resize();
      renderWindow.render();
    }
  });
  if (vtkContainer.value) {
    resizeObserver.observe(vtkContainer.value);
  }
});

onBeforeUnmount(() => { 
  if (resizeObserver && vtkContainer.value) {
    resizeObserver.unobserve(vtkContainer.value);
  }
  if (renderWindowContainer) {
    renderWindowContainer.delete(); 
  }
});
</script>


<style scoped>
.vtk-viewer {
  width: 100%;
  height: 100%;
  /* Relative positioning anchors the canvas and overlays. */
  position: relative; 
  display: flex;
  flex-direction: column;
  background: #0f172a;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 10px 25px rgba(0,0,0,0.5);
}
.loading-overlay {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: #38bdf8;
}
.render-error {
  position: absolute;
  left: 16px;
  bottom: 16px;
  background: rgba(127, 29, 29, 0.9);
  color: #fee2e2;
  border: 1px solid #ef4444;
  border-radius: 8px;
  padding: 10px 12px;
  max-width: 70%;
  font-size: 0.85rem;
}
.toolbar {
  /* Keep toolbar above the canvas layer. */
  z-index: 10;
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 15px;
  padding: 12px 20px;
  background: #1e293b;
  border-bottom: 1px solid #334155;
  color: #e2e8f0;
  font-size: 0.85rem;
}
.mode-switch {
  display: flex;
  background: #0f172a;
  border-radius: 6px;
  overflow: hidden;
  border: 1px solid #475569;
}
.mode-switch button {
  background: transparent;
  border: none;
  color: #94a3b8;
  padding: 6px 15px;
  cursor: pointer;
  transition: 0.2s;
  font-weight: bold;
}
.mode-switch button.active {
  background: #3b82f6;
  color: white;
}
.divider-vertical {
  width: 1px;
  height: 24px;
  background: #475569;
}
.control-group {
  display: flex;
  align-items: center;
  gap: 8px;
}
.slider-group {
  flex: 1;
  min-width: 200px;
}
select, input[type="range"] {
  cursor: pointer;
  accent-color: #38bdf8;
}
select {
  background: #0f172a;
  color: white;
  border: 1px solid #475569;
  padding: 4px 8px;
  border-radius: 4px;
  outline: none;
}
.vtk-canvas {
  position: absolute;
  top: 60px; /* Approximately toolbar height */
  bottom: 0;
  left: 0;
  right: 0;
  width: 100%;
  height: calc(100% - 60px); 
}
/* Hide the default focus outline from vtk.js canvas. */
.vtk-canvas:focus {
    outline: none;
}
</style>