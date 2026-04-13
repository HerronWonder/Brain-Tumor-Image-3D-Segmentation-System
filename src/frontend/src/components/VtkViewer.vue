<template>
  <div class="vtk-viewer">
    <div v-if="loading" class="loading-overlay">
      <h2>🔄 正在加载与解析 3D 医疗数据...</h2>
      <p>这可能需要几秒钟时间，构建 3D 实体模型中...</p>
    </div>

    <div class="toolbar" v-if="!loading">
      
      <div class="mode-switch">
        <button :class="{ active: viewMode === '2D' }" @click="switchMode('2D')">🎞️ 2D 正交切片</button>
        <button :class="{ active: viewMode === '3D' }" @click="switchMode('3D')">🧊 3D 全息实体</button>
      </div>
      <div class="divider-vertical"></div>

      <template v-if="viewMode === '2D'">
        <div class="control-group">
          <label>🧭 视角: </label>
          <select v-model="sliceMode" @change="updateSliceMode">
            <option :value="2">Axial (横断面)</option>
            <option :value="1">Coronal (冠状面)</option>
            <option :value="0">Sagittal (矢状面)</option>
          </select>
        </div>
        
        <div class="control-group slider-group">
          <label>🔪 深度调节 ({{ currentSlice }}/{{ maxSlice }}):</label>
          <input type="range" min="0" :max="maxSlice" v-model.number="currentSlice" @input="updateSlice">
        </div>
      </template>

      <template v-if="viewMode === '3D'">
        <div class="control-group">
          <span style="color: #10b981; font-weight: bold;">🖱️ 提示: 左键旋转，ctrl+左键旋转，滚轮平移</span>
        </div>
      </template>

      <div class="divider-vertical"></div>

      <div class="control-group">
        <label>👁️ 肿瘤透明度:</label>
        <input type="range" min="0" max="1" step="0.1" v-model.number="maskOpacity" @input="updateOpacity">
      </div>
    </div>

    <div ref="vtkContainer" class="vtk-canvas" v-show="!loading"></div>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, nextTick } from 'vue';
import axios from 'axios';
import * as nifti from 'nifti-reader-js';

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

let renderWindow, renderer, renderWindowContainer, interactor;
let style2D, style3D;
let bgMapper, bgSlice, maskMapper, maskSlice;
let bgVolMapper, bgVolActor, maskVolMapper, maskVolActor;
let bgVtkImg, maskVtkImg; 

// 🌟 新增：存储 ResizeObserver 以便后续清理
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
  if (!nifti.isNIFTI(arrayBuffer)) throw new Error("无效的 NIfTI 格式");
  
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
    const t1ceFile = props.originalFiles.find(f => f.name.toLowerCase().includes('t1ce')) || props.originalFiles[0];
    const bgBuffer = await t1ceFile.arrayBuffer();
    bgVtkImg = parseNiftiToVtk(bgBuffer);

    const maskRes = await axios.get(props.maskUrl, { responseType: 'arraybuffer' });
    maskVtkImg = parseNiftiToVtk(maskRes.data);

    setup2DPipeline();
    setup3DPipeline();

    // 🌟 一次性添加所有 Actor 到渲染器，后续只通过可见性(Visibility)来控制显示
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
    console.error("VTK 渲染失败:", err);
    alert("3D 数据渲染失败，请查看控制台报错详情。");
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
  ctf.addRGBPoint(0, 0.0, 0.0, 0.0); // 🌟 必须显式定义背景0为全透明/黑色
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
  maskSlice.getProperty().setInterpolationTypeToNearest(); // 🌟 关闭插值防绿圈边缘

  // 🌟 强制 2D 掩码使用自身的数据范围，修复全红问题
  const maskDataRange = maskVtkImg.getPointData().getScalars().getRange();
  maskSlice.getProperty().setColorWindow(maskDataRange[1] - maskDataRange[0]);
  maskSlice.getProperty().setColorLevel((maskDataRange[1] + maskDataRange[0]) / 2.0);
  // 🌟 让掩码失去“可拾取性”，鼠标的窗宽窗位调节就会无视它，直接穿透到底图上
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
  
  // 🌟 3D 使用完全独立的映射函数，彻底解决切回 2D 出现拉伸残影的问题
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
    // 🌟 显示 2D 切片，隐藏 3D 实体
    if(bgVolActor) bgVolActor.setVisibility(false);
    if(maskVolActor) maskVolActor.setVisibility(false);
    if(bgSlice) bgSlice.setVisibility(true);
    if(maskSlice) maskSlice.setVisibility(true);
    
    interactor.setInteractorStyle(style2D);
    renderer.getActiveCamera().setParallelProjection(true);
    updateSliceMode(); 
  } else {
    // 🌟 隐藏 2D 切片，显示 3D 实体
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
  
  updateSlice(); // 🌟 统一调用安全更新函数
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
  
  // 💥 放宽相机的边缘视野，防止任何哪怕 1 像素的裁切
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

  // 🌟 边缘越界保护：如果掩码厚度不对齐，超出部分直接隐藏，防止边缘拉长
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
  // 🌟 解耦后，调整透明度时需要同步更新 2D 和 3D 两个材质对象
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

  // 🌟 核心修复 2：监听浏览器窗口变化，实时通知 VTK 重绘，彻底拒绝白边和错位！
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
  /* 🌟 使用相对定位，作为画布的锚点 */
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
.toolbar {
  /* 🌟 给工具栏一个固定的 Z 轴高度，防止被后面的画布挡住 */
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
  top: 60px; /* 大约是工具栏的高度 */
  bottom: 0;
  left: 0;
  right: 0;
  width: 100%;
  height: calc(100% - 60px); 
}
/* 隐藏 VTK 自带的不美观焦点环 */
.vtk-canvas:focus {
    outline: none;
}
</style>