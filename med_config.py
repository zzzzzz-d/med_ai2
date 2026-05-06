import os
import datetime
import random
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(os.getenv("MED_AI_PROJECT_ROOT", Path(__file__).resolve().parent))
MODEL_PATH = Path(os.getenv("MED_AI_MODEL_PATH", PROJECT_ROOT / "outputs" / "checkpoints" / "eca_resnet50_v2_best.pth"))
MODEL_BEST_PATH = Path(os.getenv("MED_AI_MODEL_BEST_PATH", PROJECT_ROOT / "outputs" / "checkpoints" / "eca_resnet50_best.pth"))
MODEL_V2_BEST_PATH = Path(os.getenv("MED_AI_MODEL_V2_BEST_PATH", PROJECT_ROOT / "outputs" / "checkpoints" / "eca_resnet50_v2_best.pth"))
MODEL_V2_FINETUNED_PATH = Path(os.getenv("MED_AI_MODEL_V2_FINETUNED_PATH", PROJECT_ROOT / "outputs" / "checkpoints" / "eca_resnet50_v2_finetuned.pth"))
MODEL_V8_BEST_PATH = Path(os.getenv("MED_AI_MODEL_V8_BEST_PATH", PROJECT_ROOT / "outputs" / "checkpoints" / "eca_resnet50_v8_best_acc87_97.pth"))
FEEDBACK_DIR = Path(os.getenv("MED_AI_FEEDBACK_DIR", PROJECT_ROOT / "feedback_data1"))
GLOBAL_SEED = int(os.getenv("MED_AI_SEED", "42"))
FEEDBACK_SCHEMA_VERSION = "v3"
FEEDBACK_HEADERS = ["Schema版本", "病例ID", "录入时间", "病例图片路径", "AI预测结果", "当前决策模式", "AI置信度", "医生评价", "最终修正标签", "审核状态", "审核时间", "审核备注"]
CASE_HISTORY_SCHEMA_VERSION = "v1"
CASE_HISTORY_HEADERS = ["Schema版本", "患者ID", "病例ID", "录入时间", "病例图片路径", "当前决策模式", "阶段一结论", "恶性风险概率", "临床风险分层", "预测疾病类别", "AI校准后置信度", "不确定性指数", "前二类别置信差", "医生评价", "最终修正标签", "随访建议"]
TEMPERATURE_DEFAULT = float(os.getenv("MED_AI_TEMPERATURE", "1.2"))
TEMPERATURE_DEFAULT = min(3.0, max(0.5, TEMPERATURE_DEFAULT))
MALIGNANT_CLASS_INDICES = [1, 3, 4]
STAGE1_HIGH_RECALL_THRESHOLD = float(os.getenv("MED_AI_STAGE1_HIGH_RECALL_THRESHOLD", "0.15"))
STAGE1_BALANCED_THRESHOLD = float(os.getenv("MED_AI_STAGE1_BALANCED_THRESHOLD", "0.30"))
STAGE1_ROBUST_THRESHOLD = float(os.getenv("MED_AI_STAGE1_ROBUST_THRESHOLD", "0.40"))
HIGH_RISK_THRESHOLD = float(os.getenv("MED_AI_HIGH_RISK_THRESHOLD", "0.65"))
MEDIUM_RISK_THRESHOLD = float(os.getenv("MED_AI_MEDIUM_RISK_THRESHOLD", "0.35"))


def resolve_torch_device(requested_device=None):
    """
    统一设备策略：
    - 默认读取 MED_AI_DEVICE，支持 auto/cpu/cuda/cuda:0
    - auto: 有 CUDA 则用 CUDA，否则用 CPU
    - 指定 CUDA 但不可用时自动回退到 CPU
    """
    raw_value = os.getenv("MED_AI_DEVICE", "auto") if requested_device is None else requested_device
    request = str(raw_value).strip().lower()

    if request in ("", "auto"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device, "auto"

    if request == "cpu":
        return torch.device("cpu"), "cpu"

    if request.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(request), request
        return torch.device("cpu"), f"{request}->cpu"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, f"invalid:{request}"


def set_global_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model_registry():
    """
    当前推理模型保留强化训练最优模型和反馈增量模型。
    """
    return {
        "v8": {
            "label": "v8（强化训练最优）",
            "path": MODEL_V8_BEST_PATH,
            "best_metric": os.getenv("MED_AI_MODEL_BEST_ACC_V8", "87.97%"),
        },
        "finetuned": {
            "label": "v2_finetuned（反馈增量）",
            "path": MODEL_V2_FINETUNED_PATH,
            "best_metric": os.getenv("MED_AI_MODEL_BEST_ACC_FINETUNED", "88.27%"),
        },
    }


def format_model_time(path_obj):
    if not path_obj.exists():
        return "文件不存在"
    return datetime.datetime.fromtimestamp(path_obj.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
