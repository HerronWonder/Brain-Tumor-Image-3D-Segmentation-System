# Spring Boot Gateway

This service is the external API gateway for the project.

## Responsibilities

- Accept frontend multipart upload requests.
- Validate files and modality completeness.
- Create task-level storage directories.
- Call FastAPI inference service.
- Expose download endpoint for generated mask.

## External API

- GET /api/health
- POST /api/predict
- GET /api/download/{taskId}/{filename}

## Config

Configured by src/gateway-springboot/src/main/resources/application.yml.

Key environment variables:

- INFERENCE_BASE_URL (default: http://127.0.0.1:8000)
- INFERENCE_INFER_PATH (default: /internal/infer)
- INFERENCE_TIMEOUT_SECONDS (default: 600)
- INFERENCE_INTERNAL_TOKEN (optional, forwarded as X-Internal-Token)
- STORAGE_UPLOAD_ROOT
- STORAGE_OUTPUT_ROOT

## Traceability

- Gateway accepts incoming X-Request-ID and forwards it to FastAPI.
- If the client does not provide X-Request-ID, gateway generates one.
- Gateway returns X-Request-ID in /api/predict responses.

## Run

Recommended: use src/start.sh to start the full stack.
