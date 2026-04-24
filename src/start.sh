#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- 1. Python 推理服务配置 ---
INFERENCE_DIR="$SCRIPT_DIR/backend"
CONDA_ENV_NAME="FinalDesign"
INFERENCE_PORT="8000"
INFERENCE_INTERNAL_TOKEN="${INFERENCE_INTERNAL_TOKEN:-}"

# --- 2. Spring Boot 网关配置 ---
GATEWAY_DIR="$SCRIPT_DIR/gateway-springboot"
GATEWAY_PORT="8080"
LOCAL_MAVEN_BIN="$PROJECT_ROOT/env/apache-maven-3.9.15/bin/mvn"

# --- 3. 前端配置 ---
FRONTEND_DIR="$SCRIPT_DIR/frontend"

echo "Starting full-stack workspace"

# --- 4. 启动 FastAPI 推理服务 ---
(
    cd "$INFERENCE_DIR" || exit
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME"
    if [ -n "$INFERENCE_INTERNAL_TOKEN" ]; then
        export INTERNAL_API_TOKEN="$INFERENCE_INTERNAL_TOKEN"
    fi
    uvicorn fastapi_app:app --host 0.0.0.0 --port "$INFERENCE_PORT"
) &

# --- 5. 启动 Spring Boot 网关 ---
(
    cd "$GATEWAY_DIR" || exit
    export INFERENCE_BASE_URL="http://127.0.0.1:${INFERENCE_PORT}"
    if [ -n "$INFERENCE_INTERNAL_TOKEN" ]; then
        export INFERENCE_INTERNAL_TOKEN="$INFERENCE_INTERNAL_TOKEN"
    fi
    if [ -x "./mvnw" ]; then
        ./mvnw spring-boot:run
    elif [ -x "$LOCAL_MAVEN_BIN" ]; then
        "$LOCAL_MAVEN_BIN" spring-boot:run
    elif command -v mvn >/dev/null 2>&1; then
        mvn spring-boot:run
    else
        echo "Maven not found. Install mvn globally or place it at $LOCAL_MAVEN_BIN"
        exit 1
    fi
) &

# --- 6. 启动前端开发服务 ---
(
    cd "$FRONTEND_DIR" || exit
    export VITE_DEV_API_PROXY_TARGET="http://127.0.0.1:${GATEWAY_PORT}"
    npm run dev
) &

# --- 7. 保持脚本运行并监听退出 ---
echo "Services started: inference=${INFERENCE_PORT}, gateway=${GATEWAY_PORT}"
echo "Press Ctrl+C to stop all services"

trap "kill 0" EXIT
wait