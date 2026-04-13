import os
import uuid
import torch  # 修复了这里：导入 torch 库以检测 GPU
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from engine import run_medical_inference

app = Flask(__name__)
# 允许 Vue 前端跨域请求，这是前后端分离的必须配置
CORS(app)

# 配置工作区目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'workspace', 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'workspace', 'outputs')
WEIGHTS_PATH = os.path.join(os.path.dirname(BASE_DIR), 'scripts', 'weights', 'baseline_unet.pth')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "running", "service": "Brain Tumor Segmentation API"}), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    """接收前端上传的 4 个模态文件，进行推理，返回生成的掩码下载地址"""
    if 'files' not in request.files:
        return jsonify({"error": "No files part in the request"}), 400
    
    files = request.files.getlist('files')
    
    # BraTS 规范：需要 4 个模态
    if len(files) != 4:
        return jsonify({"error": f"Expected 4 NIfTI modalities, but got {len(files)}."}), 400

    # 生成本次任务的唯一 ID，防止多用户并发覆盖文件
    task_id = str(uuid.uuid4())[:8]
    task_upload_dir = os.path.join(UPLOAD_FOLDER, task_id)
    task_output_dir = os.path.join(OUTPUT_FOLDER, task_id)
    os.makedirs(task_upload_dir, exist_ok=True)

    saved_paths = []
    # 保存前端传来的文件
    for file in files:
        if file.filename == '':
            continue
        filename = secure_filename(file.filename)
        filepath = os.path.join(task_upload_dir, filename)
        file.save(filepath)
        saved_paths.append(filepath)

    # 排序以确保模态顺序严格对齐 (T1, T1ce, T2, FLAIR)
    saved_paths.sort()
    patient_dict = {"image": saved_paths}

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # ⚠️ 接收 metrics 返回值
        mask_path, metrics = run_medical_inference(patient_dict, WEIGHTS_PATH, task_output_dir, device)
        
        download_url = f"/api/download/{task_id}/pred_mask.nii.gz"
        
        return jsonify({
            "message": "Inference successful",
            "task_id": task_id,
            "mask_url": download_url,
            "metrics": metrics  # ⚠️ 将体积数据打包进 JSON 发给 Vue
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500
        

@app.route('/api/download/<task_id>/<filename>', methods=['GET'])
def download_file(task_id, filename):
    """前端 vtk.js 请求下载 3D NIfTI 文件的接口"""
    file_path = os.path.join(OUTPUT_FOLDER, task_id, secure_filename(filename))
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({"error": "File not found"}), 404

if __name__ == '__main__':
    # 绑定 0.0.0.0 方便后续云端公网访问调试
    app.run(host='0.0.0.0', port=5000, debug=True)