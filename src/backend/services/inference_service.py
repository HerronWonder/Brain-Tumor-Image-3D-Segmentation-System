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
        self.model_weights = model_weights
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.default_device = default_device

        self._model_cache: Dict[Tuple[str, str], torch.nn.Module] = {}
        self._lock = Lock()
        self._infer_transforms = build_infer_transforms(has_label=False)

    def run(
        self,
        image_paths: List[str],
        output_dir: str,
        model_name: str = "unet",
        device: str = None,
    ):
        device_str = self._resolve_device(device)
        normalized_model_name = (model_name or "unet").strip().lower()
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
                predictor=lambda batch: self._extract_segmentation_logits(model(batch)),
            )
            pred_vol = torch.argmax(outputs, dim=1).cpu().numpy()[0]

        pred_vol[~bg_mask] = 0
        affine_ref = self._pick_affine_reference(image_paths)
        original_nii = nib.load(affine_ref)
        voxel_volume_cm3 = self._resolve_voxel_volume_cm3(original_nii)
        metrics = self._compute_metrics(pred_vol, voxel_volume_cm3)

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "pred_mask.nii.gz")
        mask_nii = nib.Nifti1Image(pred_vol.astype(np.uint8), original_nii.affine)
        nib.save(mask_nii, output_path)

        return output_path, metrics

    def _resolve_device(self, override: str = None) -> str:
        if override:
            return override
        if self.default_device != "auto":
            return self.default_device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _get_or_load_model(self, model_name: str, device_str: str):
        model_key = (model_name, device_str)

        with self._lock:
            if model_key in self._model_cache:
                return self._model_cache[model_key]

            weights_path = self._resolve_weights_path(model_name)
            if not os.path.exists(weights_path):
                raise FileNotFoundError(
                    f"Model weights for '{model_name}' are missing at: {weights_path}"
                )

            device = torch.device(device_str)
            model = self._build_model(model_name, device)
            state_dict = self._safe_load_state_dict(weights_path, device)
            model.load_state_dict(state_dict, strict=False)
            model.to(device)
            model.eval()

            self._model_cache[model_key] = model
            return model

    def _resolve_weights_path(self, model_name: str) -> str:
        if model_name in self.model_weights:
            return self.model_weights[model_name]
        raise ValueError(f"Unsupported model '{model_name}'. Available: {', '.join(sorted(self.model_weights.keys()))}")

    @staticmethod
    def _build_model(model_name: str, device: torch.device):
        if model_name == "unet":
            return build_unet3d(in_channels=4, out_channels=5, device=device)
        if model_name == "mamba":
            return build_mamba3d(in_channels=4, out_channels=5, device=device)
        raise ValueError(f"Unsupported model '{model_name}'. Use one of: unet, mamba")

    @staticmethod
    def _safe_load_state_dict(weights_path: str, device: torch.device):
        try:
            checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(weights_path, map_location=device)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return checkpoint["state_dict"]
        return checkpoint

    @staticmethod
    def _extract_segmentation_logits(model_output: torch.Tensor):
        if isinstance(model_output, (tuple, list)):
            return model_output[0]
        return model_output

    def _pick_affine_reference(self, image_paths: List[str]) -> str:
        for path in image_paths:
            if detect_modality(path) == "t1ce":
                return path
        if len(image_paths) > 1:
            return image_paths[1]
        return image_paths[0]

    @staticmethod
    def _resolve_voxel_volume_cm3(reference_nii: nib.Nifti1Image) -> float:
        spacing = reference_nii.header.get_zooms()[:3]
        voxel_vol_mm3 = float(np.prod(spacing))
        return voxel_vol_mm3 / 1000.0

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