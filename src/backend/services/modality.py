import os
import re
from typing import List, Optional

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

    resolved = {}
    for path in paths:
        modality = detect_modality(path)
        if modality is None:
            return sorted(paths)
        if modality in resolved:
            return sorted(paths)
        resolved[modality] = path

    if not all(modality in resolved for modality in MODALITY_ORDER):
        return sorted(paths)

    return [resolved[modality] for modality in MODALITY_ORDER]