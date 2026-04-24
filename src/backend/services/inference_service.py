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
from models.unet3d import build_unet3d
from services.modality import detect_modality


class SegmentationInferenceService:
    def __init__(
        self,
        weights_path: str,
        roi_size: Tuple[int, int, int] = (128, 128, 128),
        sw_batch_size: int = 4,
        default_device: str = "auto",
    ):
        self.weights_path = weights_path
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.default_device = default_device

        self._model = None
        self._model_device = None
        self._lock = Lock()
        self._infer_transforms = build_infer_transforms(has_label=False)

    def run(self, image_paths: List[str], output_dir: str, device: str = None):
        device_str = self._resolve_device(device)
        model = self._get_or_load_model(device_str)

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
        metrics = self._compute_metrics(pred_vol)

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "pred_mask.nii.gz")
        affine_ref = self._pick_affine_reference(image_paths)

        original_nii = nib.load(affine_ref)
        mask_nii = nib.Nifti1Image(pred_vol.astype(np.uint8), original_nii.affine)
        nib.save(mask_nii, output_path)

        return output_path, metrics

    def _resolve_device(self, override: str = None) -> str:
        if override:
            return override
        if self.default_device != "auto":
            return self.default_device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _get_or_load_model(self, device_str: str):
        with self._lock:
            if self._model is not None and self._model_device == device_str:
                return self._model

            if not os.path.exists(self.weights_path):
                raise FileNotFoundError(f"Model weights missing at: {self.weights_path}")

            device = torch.device(device_str)
            model = build_unet3d(in_channels=4, out_channels=5, device=device)
            model.load_state_dict(torch.load(self.weights_path, map_location=device))
            model.to(device)
            model.eval()

            self._model = model
            self._model_device = device_str
            return self._model

    def _pick_affine_reference(self, image_paths: List[str]) -> str:
        for path in image_paths:
            if detect_modality(path) == "t1ce":
                return path
        if len(image_paths) > 1:
            return image_paths[1]
        return image_paths[0]

    @staticmethod
    def _compute_metrics(pred_vol: np.ndarray) -> Dict[str, float]:
        voxel_vol_cm3 = 0.001

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