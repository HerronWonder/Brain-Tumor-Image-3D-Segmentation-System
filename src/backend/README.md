# FastAPI Inference Service

This module is the inference runtime service for model execution.

## Responsibilities

- Load segmentation models and weights.
- Execute inference for ordered modality paths.
- Save output mask to task output directory.
- Return metrics and mask filename.

## Internal API

- GET /internal/health
- POST /internal/infer

## Security And Traceability

- Optional internal token check:
	- If INTERNAL_API_TOKEN is set, /internal/infer requires header X-Internal-Token with the same value.
- Request correlation:
	- Service returns X-Request-ID for each request.
	- If request already contains X-Request-ID, that value is preserved.

Request body fields for /internal/infer:

- image_paths: 4 ordered NIfTI file paths
- output_dir: task output directory
- model: unet or mamba

## Non-Responsibilities

- No frontend-facing upload/download API.
- No task orchestration logic.

Those are handled by Spring Boot gateway.
