import os
from typing import Dict


def resolve_weight_path(env_name: str, candidates: list[str]) -> str:
    env_path = os.getenv(env_name)
    if env_path:
        return env_path

    for path in candidates:
        if os.path.exists(path):
            return path

    return candidates[0]


def resolve_model_weights(project_root: str) -> Dict[str, str]:
    unet_candidates = [
        os.path.join(project_root, "weights", "baseline_unet_best.pth"),
        os.path.join(project_root, "scripts", "weights", "baseline_unet_best.pth"),
    ]
    mamba_candidates = [
        os.path.join(project_root, "weights", "mamba_unet_best.pth"),
        os.path.join(project_root, "scripts", "weights", "mamba_unet_best.pth"),
    ]

    return {
        "unet": resolve_weight_path("MODEL_UNET_WEIGHTS_PATH", unet_candidates),
        "mamba": resolve_weight_path("MODEL_MAMBA_WEIGHTS_PATH", mamba_candidates),
    }
