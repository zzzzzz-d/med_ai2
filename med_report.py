import base64
from io import BytesIO

import numpy as np
from PIL import Image


def img_to_base64(img_data):
    if isinstance(img_data, np.ndarray):
        img_data = Image.fromarray(img_data)
    buffered = BytesIO()
    img_data.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()


def build_html_report(
    current_time,
    selected_model_info,
    selected_model_path,
    format_model_time,
    stage1_result,
    malignant_prob,
    stage1_threshold,
    risk_level,
    risk_reason,
    followup_advice,
    result_name,
    decision_mode,
    confidence,
    uncertainty_index,
    top2_margin,
    temperature_scale,
    topk_prob_text,
    why_not_text,
    orig_b64,
    cam_b64,
    topk_card_1_b64,
    topk_card_2_b64
):
    html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>AI 医疗影像辅助诊断报告</title>
        </head>
        <body style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 40px; max-width: 900px; margin: 0 auto; background-color: #f9f9f9;">
            <div style="background-color: white; padding: 40px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                <h1 style="text-align: center; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 20px;">医疗影像 AI 辅助诊断报告</h1>
                
                <table style="width: 100%; margin-top: 20px; font-size: 16px;">
                    <tr>
                        <td><strong>生成时间：</strong> {current_time}</td>
                        <td style="text-align: right;"><strong>诊断模型：</strong> {selected_model_info['label']}</td>
                    </tr>
                    <tr>
                        <td><strong>模型文件：</strong> {selected_model_path.name}</td>
                        <td style="text-align: right;"><strong>训练日期：</strong> {format_model_time(selected_model_path)} | <strong>最佳指标：</strong> {selected_model_info['best_metric']}</td>
                    </tr>
                </table>
                
                <h2 style="color: #34495e; margin-top: 30px;">1. 影像分析结果</h2>
                <div style="background-color: #e8f4f8; padding: 15px; border-left: 5px solid #3498db; border-radius: 5px;">
                    <p style="font-size: 16px; margin: 5px 0;"><strong>阶段一筛查结论：</strong> {stage1_result}</p>
                    <p style="font-size: 16px; margin: 5px 0;"><strong>恶性风险概率：</strong> {malignant_prob*100:.2f}%（阈值 {stage1_threshold*100:.2f}%）</p>
                    <p style="font-size: 16px; margin: 5px 0;"><strong>临床风险分层：</strong> {risk_level}</p>
                    <p style="font-size: 16px; margin: 5px 0;"><strong>分层依据：</strong> {risk_reason}</p>
                    <p style="font-size: 16px; margin: 5px 0;"><strong>建议随访方案：</strong> {followup_advice}</p>
                    <p style="font-size: 18px; margin: 5px 0;"><strong>预测疾病类别：</strong> <span style="color: #e74c3c; font-weight: bold; font-size: 22px;">{result_name}</span></p>
                    <p style="font-size: 16px; margin: 5px 0;"><strong>当前决策模式：</strong> {decision_mode}</p>
                    <p style="font-size: 16px; margin: 5px 0;"><strong>AI 校准后置信度：</strong> {confidence:.2f}%</p>
                    <p style="font-size: 16px; margin: 5px 0;"><strong>不确定性指数：</strong> {uncertainty_index*100:.2f}%</p>
                    <p style="font-size: 16px; margin: 5px 0;"><strong>前二类别置信差：</strong> {top2_margin*100:.2f}%</p>
                    <p style="font-size: 16px; margin: 5px 0;"><strong>温度缩放系数：</strong> T={temperature_scale:.1f}</p>
                    <p style="font-size: 16px; margin: 5px 0;"><strong>Top-K 概率对比：</strong> {topk_prob_text}</p>
                    <p style="font-size: 16px; margin: 5px 0;"><strong>为什么是 A 而不是 B：</strong> {why_not_text}</p>
                </div>
                
                <h2 style="color: #34495e; margin-top: 30px;">2. 影像记录与可解释性分析</h2>
                <div style="display: flex; justify-content: space-between; margin-top: 20px;">
                    <div style="width: 48%; text-align: center;">
                        <h4 style="color: #555;">原始上传影像</h4>
                        <img src="data:image/jpeg;base64,{orig_b64}" style="width: 100%; border-radius: 8px; border: 1px solid #ddd;"/>
                    </div>
                    <div style="width: 48%; text-align: center;">
                        <h4 style="color: #555;">Grad-CAM AI 关注区域</h4>
                        <img src="data:image/jpeg;base64,{cam_b64}" style="width: 100%; border-radius: 8px; border: 1px solid #ddd;"/>
                    </div>
                </div>
                <p style="margin-top: 15px; color: #7f8c8d; font-size: 14px;"><em>注：右侧热力图中越红的区域，代表 AI 提取到的关键病灶特征越强烈。</em></p>
                <h2 style="color: #34495e; margin-top: 30px;">3. Top-2 类别对比解释</h2>
                <div style="display: flex; justify-content: space-between; margin-top: 20px;">
                    <div style="width: 48%; text-align: center;">
                        <h4 style="color: #555;">A 类（系统最终倾向）</h4>
                        <img src="data:image/jpeg;base64,{topk_card_1_b64}" style="width: 100%; border-radius: 8px; border: 1px solid #ddd;"/>
                    </div>
                    <div style="width: 48%; text-align: center;">
                        <h4 style="color: #555;">B 类（主要竞争类别）</h4>
                        <img src="data:image/jpeg;base64,{topk_card_2_b64}" style="width: 100%; border-radius: 8px; border: 1px solid #ddd;"/>
                    </div>
                </div>
                <p style="margin-top: 15px; color: #2c3e50; font-size: 14px;"><strong>解释：</strong>{why_not_text}</p>
                
                <div style="background-color: #fdf2e9; padding: 15px; border-left: 5px solid #e67e22; margin-top: 40px; border-radius: 5px;">
                    <strong style="color: #d35400;">⚠️ 伦理与免责声明：</strong><br>
                    <span style="font-size: 14px; color: #555;">本系统含有动态阈值防漏诊机制，结论仅供科研展示与初步筛查参考，<strong>绝不可替代专业医生的最终诊断决策</strong>。</span>
                </div>
            </div>
        </body>
        </html>
    """
    return html_content
