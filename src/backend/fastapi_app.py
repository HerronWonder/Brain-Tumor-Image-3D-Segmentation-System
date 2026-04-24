import os
import sys
import uuid

from fastapi import FastAPI, HTTPException, Request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from api.config import resolve_model_weights
from api.schemas import InferRequest, InferResponse
from services.inference_service import SegmentationInferenceService
from services.modality import order_modality_paths


def create_app() -> FastAPI:
    app = FastAPI(title="FinalDesign Inference Service", version="1.0.0")
    infer_service = SegmentationInferenceService(model_weights=resolve_model_weights(PROJECT_ROOT))
    internal_api_token = os.getenv("INTERNAL_API_TOKEN", "").strip()

    @app.middleware("http")
    async def attach_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.get("/internal/health")
    def health_check() -> dict[str, str]:
        return {"status": "running", "service": "Brain Tumor Segmentation Inference"}

    @app.post("/internal/infer", response_model=InferResponse)
    def infer(request: InferRequest, http_request: Request) -> InferResponse:
        model_name = request.model.strip().lower()

        if internal_api_token:
            provided_token = http_request.headers.get("X-Internal-Token", "")
            if provided_token != internal_api_token:
                raise HTTPException(status_code=401, detail="Unauthorized internal request.")

        try:
            ordered_paths = order_modality_paths(request.image_paths)
            output_path, metrics = infer_service.run(
                image_paths=ordered_paths,
                output_dir=request.output_dir,
                model_name=model_name,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

        return InferResponse(
            mask_filename=os.path.basename(output_path),
            model=model_name,
            metrics=metrics,
        )

    return app


app = create_app()
