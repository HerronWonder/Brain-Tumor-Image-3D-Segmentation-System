import os
import sys
from threading import Lock
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from data.pipelines import build_infer_transforms
from models.mamba3d import build_mamba3d
from models.unet3d import build_unet3d
from services.modality import detect_modality


class SegmentationInferenceService:
    def __init__(
        self,
        model_weights: Dict[str, str],
        roi_size: Tuple[int, int, int] = (128, 128, 128),
        sw_batch_size: int = 4,
        default_device: str = "auto",
    ):
        self.model_weights = {key.lower(): value for key, value in model_weights.items()}
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.default_device = default_device

        self._model = None
        self._model_key = None
        self._lock = Lock()
        self._infer_transforms = build_infer_transforms(has_label=False)

    def run(self, image_paths: List[str], output_dir: str, model_name: str = "unet", device: str = None):
        normalized_model_name = model_name.lower()
        device_str = self._resolve_device(device)
        model = self._get_or_load_model(normalized_model_name, device_str)

        patient_dict = {"image": image_paths}
        test_data = self._infer_transforms(patient_dict)
        input_tensor = test_data["image"].unsqueeze(0).to(device_str)

        bg_mask = (test_data["image"] != 0).any(dim=0).cpu().numpy()

        with torch.no_grad():
            outputs = sliding_window_inference(
                inputs=input_tensor,
                roi_size=self.roi_size,
                sw_batch_size=self.sw_batch_size,
                predictor=model,
            )
            pred_vol = torch.argmax(outputs, dim=1).cpu().numpy()[0]

        pred_vol[~bg_mask] = 0

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "pred_mask.nii.gz")
        affine_ref = self._pick_affine_reference(image_paths)
        reference_nii = nib.load(affine_ref)
        voxel_vol_cm3 = self._compute_voxel_volume_cm3(reference_nii)
        metrics = self._compute_metrics(pred_vol, voxel_vol_cm3)

        mask_nii = nib.Nifti1Image(pred_vol.astype(np.uint8), reference_nii.affine, reference_nii.header)
        nib.save(mask_nii, output_path)

        return output_path, metrics

    def _resolve_device(self, override: str = None) -> str:
        if override:
            return override
        if self.default_device != "auto":
            return self.default_device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _get_or_load_model(self, model_name: str, device_str: str):
        builder_map = {
            "unet": build_unet3d,
            "mamba": build_mamba3d,
        }

        if model_name not in builder_map:
            raise ValueError(f"Unsupported model type: {model_name}. Supported values: unet, mamba.")
        if model_name not in self.model_weights:
            raise ValueError(f"No weights configured for model type: {model_name}.")

        weights_path = self.model_weights[model_name]
        model_key = (model_name, device_str, weights_path)

        with self._lock:
            if self._model is not None and self._model_key == model_key:
                return self._model

            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Model weights missing at: {weights_path}")

            device = torch.device(device_str)
            builder = builder_map[model_name]
            model = builder(in_channels=4, out_channels=5, device=device)
            try:
                state_dict = torch.load(weights_path, map_location=device, weights_only=True)
            except TypeError:
                state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

            self._model = model
            self._model_key = model_key
            return self._model

    def _pick_affine_reference(self, image_paths: List[str]) -> str:
        for path in image_paths:
            if detect_modality(path) == "t1ce":
                return path
        if len(image_paths) > 1:
            return image_paths[1]
        return image_paths[0]

    @staticmethod
    def _compute_voxel_volume_cm3(reference_nii: nib.Nifti1Image) -> float:
        zooms = reference_nii.header.get_zooms()[:3]
        if len(zooms) != 3:
            return 0.001
        return float(np.prod(zooms) / 1000.0)

    @staticmethod
    def _compute_metrics(pred_vol: np.ndarray, voxel_vol_cm3: float) -> Dict[str, float]:

        vol_necrotic = float(np.sum(pred_vol == 1)) * voxel_vol_cm3
        vol_edema = float(np.sum(pred_vol == 2)) * voxel_vol_cm3
        vol_enhancing = float(np.sum(pred_vol == 4)) * voxel_vol_cm3
        total_vol = vol_necrotic + vol_edema + vol_enhancing

        return {
            "necrotic_cm3": round(vol_necrotic, 2),
            "edema_cm3": round(vol_edema, 2),
            "enhancing_cm3": round(vol_enhancing, 2),
            "total_cm3": round(total_vol, 2),
        }