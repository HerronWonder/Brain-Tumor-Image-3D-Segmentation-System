# FinalDesign Backend Architecture

## Service Split

- Spring Boot Gateway
  - Role: external API gateway, request validation, file upload/download, task storage and orchestration.
  - External API: /api/health, /api/predict, /api/download/{taskId}/{filename}
  - Location: src/gateway-springboot
  - Port: 8080 (default)

- FastAPI Inference Service
  - Role: model loading and inference execution only.
  - Internal API: /internal/health, /internal/infer
  - Location: src/backend/fastapi_app.py and src/backend/api
  - Port: 8000 (default)

- Vue Frontend
  - Role: upload and visualization UI.
  - Frontend only calls gateway API (/api/*).
  - Vite dev proxy defaults to http://127.0.0.1:8080.

## Why This Split

- Spring Boot is better for enterprise API governance, layered code structure, and integration concerns.
- FastAPI is better for Python ML runtime and GPU inference stack.
- The split keeps model runtime isolated from web orchestration logic.

## Request Flow

1. Frontend uploads 4 modality files to gateway /api/predict.
2. Gateway validates modalities, stores files, allocates task output directory.
3. Gateway immediately returns task_id and status_url (non-blocking UI).
4. Frontend enters polling state via /api/tasks/{taskId} and displays progress.
5. Gateway calls FastAPI /internal/infer with ordered paths + output dir + model.
6. Gateway and FastAPI propagate X-Request-ID for end-to-end traceability.
7. Optional: Gateway sends X-Internal-Token when INFERENCE_INTERNAL_TOKEN is configured.
8. FastAPI validates X-Internal-Token if INTERNAL_API_TOKEN is configured.
9. FastAPI runs inference and writes pred_mask.nii.gz.
10. Gateway stores structured report.json and exposes mask/report download URLs.
11. Frontend fetches mask from gateway and enables mask/report export.

## Development Startup

Use src/start.sh. It starts:

- FastAPI inference service on 8000
- Spring Boot gateway on 8080
- Vue frontend dev server

## Migration Notes

- Legacy Flask external API has been removed.
- New backend features should be implemented in gateway-springboot (workflow/API) or FastAPI (inference).
