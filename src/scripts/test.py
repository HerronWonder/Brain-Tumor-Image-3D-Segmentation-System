import os
import sys
import time
import json
import torch
import warnings
import argparse
import numpy as np

# 动态将项目根目录加入环境变量
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import build_train_loader
from monai.inferers import SlidingWindowInferer
from monai.metrics import compute_dice, compute_hausdorff_distance

warnings.filterwarnings("ignore")

# ==========================================
# 默认配置区
# ==========================================
DATA_DIRECTORY = "../../../dataset/All/"
JSON_PATH = "../../../dataset/dataset.json"

REGION_LABELS = {
    "WT": [1, 2, 4],
    "TC": [1, 4],
    "ET": [4],
}

def get_model(model_name, device):
    """根据参数实例化不同的模型架构"""
    if model_name == "unet":
        from models.unet3d import build_unet3d
        return build_unet3d(in_channels=4, out_channels=5, device=device)
    elif model_name == "mamba":
        from models.mamba3d import build_mamba3d
        return build_mamba3d(in_channels=4, out_channels=5, device=device)
    else:
        raise ValueError(f"不支持的模型架构: {model_name}")

def parse_args():
    parser = argparse.ArgumentParser(description="3D Medical Image Inference & Testing")
    parser.add_argument("--data_dir", type=str, default=DATA_DIRECTORY)
    parser.add_argument("--json_path", type=str, default=JSON_PATH)
    
    # 核心参数：指定测试哪个模型，以及加载哪个权重
    parser.add_argument("--model", type=str, choices=["unet", "mamba"], required=True, 
                        help="选择要测试的模型架构 (unet 或 mamba)")
    parser.add_argument("--weight_path", type=str, required=True, 
                        help="训练好的 .pth 权重文件路径")
    parser.add_argument("--report_path", type=str, default="outputs/eval_report.json",
                        help="评估报告输出路径")
    return parser.parse_args()


def _region_mask(segmentation: np.ndarray, region_name: str) -> np.ndarray:
    return np.isin(segmentation, REGION_LABELS[region_name]).astype(np.float32)


def _optional_float(value: float):
    if value is None:
        return None
    if not np.isfinite(value):
        return None
    return float(value)


def _mean_or_none(values: list[float]):
    if not values:
        return None
    arr = np.array(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(arr.mean())

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader, test_count = build_train_loader(
        data_dir=args.data_dir, 
        json_path=args.json_path, 
        split="test", 
        batch_size=1,
        mini_dataset_size=None
    )

    model = get_model(args.model, device)
    
    if os.path.exists(args.weight_path):
        model.load_state_dict(torch.load(args.weight_path, map_location=device))
        print(f"[MODEL]  Successfully loaded weights from: {args.weight_path}")
    else:
        raise FileNotFoundError(f"找不到权重文件: {args.weight_path}")
    
    model.eval()

    inferer = SlidingWindowInferer(roi_size=(128, 128, 128), sw_batch_size=4, overlap=0.5)

    region_scores = {
        region: {"dice": [], "hd95_mm": []}
        for region in REGION_LABELS
    }

    total_start_time = time.time()

    with torch.no_grad():
        for test_data in test_loader:
            test_inputs = test_data["image"].to(device)
            test_labels = test_data["label"].to(device)

            test_outputs = inferer(test_inputs, model)
            pred_labels = torch.argmax(test_outputs, dim=1).cpu().numpy()
            gt_labels = test_labels.squeeze(1).cpu().numpy()

            for batch_idx in range(pred_labels.shape[0]):
                pred_np = pred_labels[batch_idx]
                gt_np = gt_labels[batch_idx]

                for region_name in REGION_LABELS:
                    pred_region = _region_mask(pred_np, region_name)
                    gt_region = _region_mask(gt_np, region_name)

                    pred_tensor = torch.from_numpy(pred_region).unsqueeze(0).unsqueeze(0)
                    gt_tensor = torch.from_numpy(gt_region).unsqueeze(0).unsqueeze(0)

                    dice_value = compute_dice(pred_tensor, gt_tensor, include_background=True).item()
                    hd95_value = compute_hausdorff_distance(
                        pred_tensor,
                        gt_tensor,
                        include_background=True,
                        percentile=95,
                    ).item()

                    dice_value = _optional_float(dice_value)
                    hd95_value = _optional_float(hd95_value)

                    if dice_value is not None:
                        region_scores[region_name]["dice"].append(dice_value)
                    if hd95_value is not None:
                        region_scores[region_name]["hd95_mm"].append(hd95_value)

    total_duration = time.time() - total_start_time

    regions_report = {}
    dice_means = []
    hd95_means = []
    for region_name, values in region_scores.items():
        dice_mean = _mean_or_none(values["dice"])
        hd95_mean = _mean_or_none(values["hd95_mm"])
        if dice_mean is not None:
            dice_means.append(dice_mean)
        if hd95_mean is not None:
            hd95_means.append(hd95_mean)

        regions_report[region_name] = {
            "dice": {
                "mean": dice_mean,
                "sample_count": len(values["dice"]),
            },
            "hd95_mm": {
                "mean": hd95_mean,
                "sample_count": len(values["hd95_mm"]),
            },
        }

    report = {
        "schema_version": "1.1",
        "task": {
            "mode": "offline_evaluation",
            "model": args.model,
            "weight_path": args.weight_path,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "dataset": {
            "split": "test",
            "data_dir": args.data_dir,
            "json_path": args.json_path,
            "case_count": int(test_count),
        },
        "evaluation_metrics": {
            "regions": regions_report,
            "summary": {
                "mean_dice": _mean_or_none(dice_means),
                "mean_hd95_mm": _mean_or_none(hd95_means),
            },
        },
        "runtime": {
            "device": str(device),
            "duration_seconds": round(float(total_duration), 3),
        },
    }

    report_path = args.report_path
    report_dir = os.path.dirname(report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)

    print(f"Evaluation report written to: {report_path}")

if __name__ == "__main__":
    main()