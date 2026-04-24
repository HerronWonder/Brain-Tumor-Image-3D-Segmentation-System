import os
import shutil
import uuid
from dataclasses import dataclass
from typing import List, Optional

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from .modality import order_modality_paths


@dataclass(frozen=True)
class TaskContext:
    task_id: str
    upload_dir: str
    output_dir: str
    image_paths: List[str]


class TaskStorageService:
    def __init__(self, upload_root: str, output_root: str):
        self.upload_root = upload_root
        self.output_root = output_root
        os.makedirs(self.upload_root, exist_ok=True)
        os.makedirs(self.output_root, exist_ok=True)

    def create_task(self, files: List[FileStorage]) -> TaskContext:
        if not files:
            raise ValueError("No files part in the request.")
        if len(files) != 4:
            raise ValueError(f"Expected 4 NIfTI modalities, but got {len(files)}.")

        task_id = uuid.uuid4().hex[:8]
        upload_dir = os.path.join(self.upload_root, task_id)
        output_dir = os.path.join(self.output_root, task_id)
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        saved_paths: List[str] = []
        try:
            for file in files:
                filename = secure_filename(file.filename or "")
                if not filename:
                    continue

                lower_name = filename.lower()
                if not (lower_name.endswith(".nii") or lower_name.endswith(".nii.gz")):
                    raise ValueError(f"Invalid file type: {filename}. Only .nii or .nii.gz files are supported.")

                file_path = os.path.join(upload_dir, filename)
                file.save(file_path)
                saved_paths.append(file_path)

            if len(saved_paths) != 4:
                raise ValueError("Exactly 4 valid files must be provided.")

            ordered_paths = order_modality_paths(saved_paths)
        except Exception:
            self._cleanup_task_dirs(upload_dir=upload_dir, output_dir=output_dir)
            raise

        return TaskContext(
            task_id=task_id,
            upload_dir=upload_dir,
            output_dir=output_dir,
            image_paths=ordered_paths,
        )

    @staticmethod
    def _cleanup_task_dirs(upload_dir: str, output_dir: str) -> None:
        shutil.rmtree(upload_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)

    def resolve_output_file(self, task_id: str, filename: str) -> Optional[str]:
        safe_filename = secure_filename(filename)
        if not safe_filename:
            return None

        file_path = os.path.join(self.output_root, task_id, safe_filename)
        if os.path.exists(file_path):
            return file_path
        return None