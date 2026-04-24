import os
import sys

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from services.inference_service import SegmentationInferenceService
from services.task_storage import TaskStorageService


def _resolve_weights_path() -> str:
    env_path = os.getenv("MODEL_WEIGHTS_PATH")
    if env_path:
        return env_path

    candidates = [
        os.path.join(PROJECT_ROOT, "weights", "baseline_unet_best.pth"),
        os.path.join(PROJECT_ROOT, "scripts", "weights", "baseline_unet_best.pth"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return candidates[0]


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    upload_root = os.path.join(BASE_DIR, "workspace", "uploads")
    output_root = os.path.join(BASE_DIR, "workspace", "outputs")

    storage = TaskStorageService(upload_root=upload_root, output_root=output_root)
    infer_service = SegmentationInferenceService(weights_path=_resolve_weights_path())

    @app.route("/api/health", methods=["GET"])
    def health_check():
        return jsonify({"status": "running", "service": "Brain Tumor Segmentation API"}), 200

    @app.route("/api/predict", methods=["POST"])
    def predict():
        files = request.files.getlist("files")

        try:
            task = storage.create_task(files)
            _, metrics = infer_service.run(image_paths=task.image_paths, output_dir=task.output_dir)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 500
        except Exception as exc:
            return jsonify({"error": f"Inference failed: {exc}"}), 500

        return jsonify(
            {
                "message": "Inference successful",
                "task_id": task.task_id,
                "mask_url": f"/api/download/{task.task_id}/pred_mask.nii.gz",
                "metrics": metrics,
            }
        ), 200

    @app.route("/api/download/<task_id>/<filename>", methods=["GET"])
    def download_file(task_id: str, filename: str):
        file_path = storage.resolve_output_file(task_id=task_id, filename=filename)
        if file_path is None:
            return jsonify({"error": "File not found"}), 404
        return send_file(file_path, as_attachment=True)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)