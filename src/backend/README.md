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

## How to use
进入项目并执行一键启动
cd /data/ssd2/wangheran/FinalDesign/src
./start.sh

启动后访问前端
http://localhost:5173

端口分工
FastAPI 推理服务: 8000
Spring Boot 网关: 8080
前端 Vite: 5173