#!/bin/bash

# --- 1. 后端配置 ---
BACKEND_DIR="./backend"
CONDA_ENV_NAME="FinalDesign"

# --- 2. 前端配置 ---
FRONTEND_DIR="./frontend"

echo "🚀 正在启动全栈开发环境..."

# --- 3. 启动后端 (在后台运行) ---
echo "📡 启动后端服务器 (Conda: $CONDA_ENV_NAME)..."
# 使用 bash -i (交互模式) 确保能识别 conda 命令，或者直接 source conda.sh
# 这里采用最稳妥的交互式执行方式
(
    cd $BACKEND_DIR || exit
    # 初始化 shell 里的 conda 环境
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate $CONDA_ENV_NAME
    python app.py
) &

# --- 4. 启动前端 (在后台运行) ---
echo "🎨 启动前端开发环境 (npm)..."
(
    cd $FRONTEND_DIR || exit
    npm run dev
) &

# --- 5. 保持脚本运行并监听退出 ---
echo "✅ 全部服务已在后台启动。"
echo "按 [Ctrl+C] 停止所有服务"

# 捕获退出信号，关闭所有子进程
trap "kill 0" EXIT

# 保持前台运行，否则脚本结束会导致子进程被回收
wait