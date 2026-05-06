import math

import torch

from med_config import HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD


# 对模型输出 logits 执行温度缩放并转换为概率分布
def apply_temperature_scaling(logits, temperature):
    # 避免温度为 0 导致除零错误
    safe_temperature = max(float(temperature), 1e-6)
    return torch.softmax(logits / safe_temperature, dim=0)


# 计算不确定性指标：归一化熵 + Top-2 概率差
def compute_uncertainty_metrics(probabilities):
    # 迁移到 CPU 侧做数值统计，避免设备差异影响
    probs = probabilities.detach().cpu()
    entropy = float((-probs * torch.log(probs + 1e-12)).sum().item())
    max_entropy = math.log(len(probs)) if len(probs) > 0 else 1.0
    uncertainty_index = entropy / max_entropy if max_entropy > 0 else 0.0
    top_k = torch.topk(probs, k=min(2, len(probs))).values
    top2_margin = float((top_k[0] - top_k[1]).item()) if len(top_k) > 1 else 1.0
    return uncertainty_index, top2_margin


# 将百分比文本（如 "87.5%"）解析为浮点数
def parse_percent_text(percent_text):
    text = str(percent_text).strip().replace("%", "")
    if text == "":
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


# 根据决策模式、恶性概率与不确定性综合输出风险分层与随访建议
def build_risk_stratification(decision_mode, stage1_is_malignant, malignant_prob, uncertainty_index, top2_margin):
    if stage1_is_malignant and malignant_prob >= HIGH_RISK_THRESHOLD:
        risk_level = "高风险"
        risk_reason = f"阶段一恶性概率 {malignant_prob*100:.1f}% 已达到高风险阈值"
    elif stage1_is_malignant or malignant_prob >= MEDIUM_RISK_THRESHOLD or uncertainty_index >= 0.60 or top2_margin <= 0.08:
        risk_level = "中风险"
        risk_reason = "存在恶性倾向或模型不确定性偏高"
    else:
        risk_level = "低风险"
        risk_reason = "阶段一显示良性倾向，且模型不确定性处于可控范围"
    mode_key = "高召回模式" if "高召回模式" in decision_mode else ("稳健模式" if "稳健模式" in decision_mode else "平衡模式")
    templates = {
        "高召回模式": {
            "高风险": "建议 24-72 小时内完成皮肤镜复查并优先安排病理活检，同时进入重点随访清单。",
            "中风险": "建议 1-2 周内复诊并进行皮肤镜对比，如病灶形态变化应升级为病理评估。",
            "低风险": "建议 3 个月内随访复查，保持病灶影像留存用于纵向对比。"
        },
        "平衡模式": {
            "高风险": "建议尽快转入专科进一步评估，优先考虑病理检查确认。",
            "中风险": "建议 2-4 周内复查并结合病史决定是否追加病理或其他检查。",
            "低风险": "建议常规随访观察，6 个月内复查一次并记录病灶变化。"
        },
        "稳健模式": {
            "高风险": "建议立即人工复核并尽快安排病理检查，以降低漏诊风险。",
            "中风险": "建议在短期内完成人工复核，必要时追加皮肤镜与病理联合判断。",
            "低风险": "建议常规门诊随访，并在复诊时复核影像与临床体征一致性。"
        }
    }
    followup_advice = templates[mode_key][risk_level]
    return risk_level, risk_reason, followup_advice
