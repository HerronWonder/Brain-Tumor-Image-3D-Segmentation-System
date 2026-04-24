import os
import re
from typing import Dict, List, Optional

MODALITY_ORDER = ("t1", "t1ce", "t2", "flair")


def detect_modality(path: str) -> Optional[str]:
    name = os.path.basename(path).lower()

    if "t1ce" in name or "t1c" in name:
        return "t1ce"
    if "flair" in name:
        return "flair"
    if re.search(r"(^|[_\-.])t2([_\-.]|$)", name):
        return "t2"
    if re.search(r"(^|[_\-.])t1([_\-.]|$)", name):
        return "t1"

    return None


def order_modality_paths(paths: List[str]) -> List[str]:
    if len(paths) != 4:
        raise ValueError(f"Expected 4 modality files, got {len(paths)}.")

    resolved: Dict[str, str] = {}
    unresolved: List[str] = []
    for path in paths:
        modality = detect_modality(path)
        if modality is None:
            unresolved.append(os.path.basename(path))
            continue
        if modality in resolved:
            raise ValueError(
                f"Duplicate modality detected for '{modality}'. "
                f"Please upload exactly one file for each modality: {', '.join(MODALITY_ORDER)}."
            )
        resolved[modality] = path

    if unresolved:
        raise ValueError(
            "Unable to infer modality from file names: "
            f"{', '.join(unresolved)}. Expected names containing t1, t1ce, t2, flair."
        )

    missing = [modality for modality in MODALITY_ORDER if modality not in resolved]
    if missing:
        raise ValueError(
            f"Missing modality files: {', '.join(missing)}. "
            f"Expected complete set: {', '.join(MODALITY_ORDER)}."
        )

    return [resolved[modality] for modality in MODALITY_ORDER]