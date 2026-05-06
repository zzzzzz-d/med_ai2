import streamlit as st
import datetime
import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from med_config import (
    GLOBAL_SEED,
    TEMPERATURE_DEFAULT,
    MALIGNANT_CLASS_INDICES,
    STAGE1_HIGH_RECALL_THRESHOLD,
    STAGE1_BALANCED_THRESHOLD,
    STAGE1_ROBUST_THRESHOLD,
    set_global_seed,
    get_model_registry,
    format_model_time
)
from med_inference import (
    apply_temperature_scaling,
    compute_uncertainty_metrics,
    build_risk_stratification
)
from med_models import load_model_and_cam
from med_report import img_to_base64, build_html_report
from med_clinical_ui import render_clinical_workbench
from med_metric_store import get_stored_best_metric, merge_metric_text

set_global_seed(GLOBAL_SEED)

# ==========================================
# 0. 页面基础配置 
# ==========================================
st.set_page_config(page_title="AI 医疗影像辅助诊断", layout="wide", page_icon="🏥")
#样式
st.markdown("""
<style>
    /* 1. 全局背景设为极浅的高级灰，凸显内容的纯白质感 */
    [data-testid="stAppViewContainer"] {
        background-color: #F8F9FA;
        color: #2C3E50;
    }
    .block-container {
        padding-bottom: 72px;
    }
    
    /* 2. 侧边栏纯白化 + 微阴影分离 */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        box-shadow: 2px 0 12px rgba(0,0,0,0.03);
        border-right: none;
    }
    
    /* 3. 按钮的苹果风优化：白底、细边框、圆角、悬浮发光 */
    div.stButton > button {
        background-color: #FFFFFF;
        color: #2C3E50;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s cubic-bezier(.25,.8,.25,1);
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    div.stButton > button:hover {
        border-color: #3498DB;
        color: #3498DB;
        box-shadow: 0 4px 12px rgba(52,152,219,0.15);
        transform: translateY(-1px);
    }
    
    /* 4. 文件上传拖拽区的拟物化微调 */
    [data-testid="stFileUploadDropzone"] {
        background-color: #FFFFFF;
        border: 2px dashed #CBD5E1;
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #3498DB;
        background-color: #F0F8FF;
    }

    /* 5. 
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;} /* 隐藏右上角可能出现的 Deploy 按钮 */
    
    /* 6. 统一各种卡片和提示框的圆角 */
    div.stAlert {
        border-radius: 8px;
        border: none;
    }
    div[data-testid="stDownloadButton"] {
        position: fixed;
        right: 18px;
        bottom: 16px;
        z-index: 9999;
        width: auto;
    }
    div[data-testid="stDownloadButton"] > button {
        width: auto;
        border-radius: 18px;
        border: 1px solid rgba(148, 163, 184, 0.45);
        background: rgba(255, 255, 255, 0.58);
        color: rgba(31, 41, 55, 0.78);
        font-weight: 500;
        min-height: 34px;
        padding: 0 14px;
        backdrop-filter: blur(4px);
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08);
        transition: all 0.18s ease;
    }
    div[data-testid="stDownloadButton"] > button:hover {
        background: rgba(255, 255, 255, 0.96);
        border-color: #93C5FD;
        color: #1D4ED8;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.18);
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)
# 左侧边栏
with st.sidebar:
    st.header("📋 系统支持的诊断类型")
    st.markdown("本系统基于 HAM10000 数据集，支持自动识别以下 **7 种** 常见皮肤病变 (dx)：")
    st.markdown("""
    - **nv**: 痣 (Melanocytic nevi)
    - **mel**: 黑色素瘤 (Melanoma) ⚠️*高危*
    - **bkl**: 良性角化病 (Benign keratosis-like)
    - **bcc**: 基底细胞癌 (Basal cell carcinoma) ⚠️*高危*
    - **akiec**: 光化性角化病 (Actinic keratoses)
    - **vasc**: 血管病变 (Vascular lesions)
    - **df**: 皮肤纤维瘤 (Dermatofibroma)
    """)
    st.info("💡 提示：系统集成了 Grad-CAM 热力图，红色区域代表 AI 关注的重点病灶部位。")
    st.success("🎯 当前系统内核：ECA-ResNet50 + GradCAM")
    st.markdown("---")
    st.subheader("🧠 模型版本管理面板")
    model_registry = get_model_registry()
    for model_key, info in model_registry.items():
        persisted_metric = get_stored_best_metric(model_key)
        info["best_metric"] = merge_metric_text(info.get("best_metric", ""), persisted_metric)
    model_option_keys = list(model_registry.keys())
    default_model_key = "v8" if "v8" in model_registry else model_option_keys[0]
    model_key = st.selectbox(
        "选择当前推理模型",
        options=model_option_keys,
        index=model_option_keys.index(default_model_key),
        format_func=lambda key: model_registry[key]["label"] if model_registry[key]["path"].exists() else f"{model_registry[key]['label']}（权重缺失）"
    )
    selected_model_info = model_registry[model_key]
    selected_model_path = selected_model_info["path"]
    if not selected_model_path.exists():
        fallback_key = "v8" if ("v8" in model_registry and model_registry["v8"]["path"].exists()) else None
        if fallback_key is None:
            fallback_key = next((key for key, info in model_registry.items() if info["path"].exists()), None)
        if fallback_key and model_key != fallback_key:
            st.warning(f"所选模型权重不存在：{selected_model_path.name}，已自动回退到 {model_registry[fallback_key]['label']}")
            model_key = fallback_key
            selected_model_info = model_registry[model_key]
            selected_model_path = selected_model_info["path"]
        else:
            st.error(f"模型权重不存在：{selected_model_path}")
            st.stop()
    st.caption(f"当前模型文件：{selected_model_path.name}")
    st.caption(f"训练日期：{format_model_time(selected_model_path)}")
    st.caption(f"最佳指标：{selected_model_info['best_metric']}")
    
    # ==========================================
    # 🌟 新增：临床决策模式选择器
    # ==========================================
    st.markdown("---")
    st.subheader("⚙️ 临床辅助决策模式")
    decision_mode = st.radio(
        "请根据科室场景选择系统灵敏度：",
        ("⚖️ 平衡模式 (常规诊断)", "🔍 高召回模式 (体检筛查/防漏诊)", "🛡️ 稳健模式 (专家确诊/防误诊)"),
        help="平衡模式：综合准确率最高。\n高召回模式：对恶性肿瘤极度敏感，恶性概率>15%即越级报警，宁可误报绝不漏诊。\n稳健模式：最高置信度>85%才给出确定结论，否则建议人工复查。"
    )
    st.subheader("📏 置信度校准")
    temperature_scale = st.slider(
        "温度缩放系数 T",
        min_value=0.5,
        max_value=3.0,
        value=float(TEMPERATURE_DEFAULT),
        step=0.1,
        help="T>1 会降低过高置信度，T<1 会提高置信度，默认建议 1.2。"
    )

# 主标题
st.title("🏥 基于深度学习的医疗影像辅助诊断系统")
st.markdown("**核心算法：ECA-Net 轻量化注意力机制 + 多模式决策引擎 + Grad-CAM 热力图**")
st.caption(f"当前加载模型：{selected_model_info['label']} | 文件：{selected_model_path.name} | 训练日期：{format_model_time(selected_model_path)} | 最佳指标：{selected_model_info['best_metric']}")
st.warning("⚠️ 伦理声明：本系统仅为科研演示辅助工具，分析结果不可替代临床诊断决策。")
st.markdown("---")

# ==========================================
# 1. 加载模型与热力图生成器 
# ==========================================
model, cam, device, device_policy = load_model_and_cam(
    str(selected_model_path),
    requested_device=os.getenv("MED_AI_DEVICE", "auto"),
    model_version_token=selected_model_path.stat().st_mtime_ns if selected_model_path.exists() else 0,
)
st.caption(f"推理设备：`{device}`（device policy: `{device_policy}`）")

class_names = {
    0: '痣 (nv)', 1: '黑色素瘤 (mel)', 2: '良性角化病 (bkl)', 3: '基底细胞癌 (bcc)', 
    4: '光化性角化病 (akiec)', 5: '血管病变 (vasc)', 6: '皮肤纤维瘤 (df)'
}

# ==========================================
# 3. Streamlit 主逻辑
# ==========================================
uploaded_file = st.file_uploader("请上传皮肤病理影像 (支持 JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert('RGB')
    bg_image_np = np.array(image_pil.resize((224, 224))) / 255.0
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image_pil).unsqueeze(0)
    img_tensor = img_tensor.to(device)

    st.divider() 

    with st.spinner('AI 大脑正在高速运转，生成诊断与热力图中...'):
        # 纯推理阶段关闭梯度，减少内存占用与计算开销
        with torch.no_grad():
            outputs = model(img_tensor)
        probabilities = apply_temperature_scaling(outputs[0], temperature_scale).detach().cpu()
        uncertainty_index, top2_margin = compute_uncertainty_metrics(probabilities)
        malignant_prob = float(sum(probabilities[idx].item() for idx in MALIGNANT_CLASS_INDICES))
        benign_prob = max(0.0, 1.0 - malignant_prob)
        if "高召回模式" in decision_mode:
            stage1_threshold = STAGE1_HIGH_RECALL_THRESHOLD
        elif "稳健模式" in decision_mode:
            stage1_threshold = STAGE1_ROBUST_THRESHOLD
        else:
            stage1_threshold = STAGE1_BALANCED_THRESHOLD
        stage1_is_malignant = malignant_prob >= stage1_threshold
        stage1_result = "恶性高风险 (建议优先排查)" if stage1_is_malignant else "良性倾向 (常规随访)"
        stage1_confidence = malignant_prob if stage1_is_malignant else benign_prob
        risk_level, risk_reason, followup_advice = build_risk_stratification(
            decision_mode=decision_mode,
            stage1_is_malignant=stage1_is_malignant,
            malignant_prob=malignant_prob,
            uncertainty_index=uncertainty_index,
            top2_margin=top2_margin
        )

        max_prob, predicted_idx = torch.max(probabilities, 0)
        final_idx = predicted_idx.item()
        final_confidence = max_prob.item()

        alert_message = ""

        if stage1_is_malignant and final_idx not in MALIGNANT_CLASS_INDICES:
            stage2_malignant_idx = max(MALIGNANT_CLASS_INDICES, key=lambda idx: probabilities[idx].item())
            final_idx = stage2_malignant_idx
            final_confidence = probabilities[stage2_malignant_idx].item()
            alert_message = f"⚠️ 【双阶段筛查触发】第一阶段判定恶性风险 {malignant_prob*100:.1f}%，第二阶段已优先输出恶性细分结论。"
        elif "高召回模式" in decision_mode and stage1_is_malignant:
            alert_message = f"⚠️ 【高召回防线触发】第一阶段恶性风险 {malignant_prob*100:.1f}% 已超过阈值 {stage1_threshold*100:.1f}%，防漏诊机制启动。"

        if "稳健模式" in decision_mode:
            if final_confidence < 0.85:
                final_idx = -1 
                alert_message = "🛡️ 【稳健模式拦截】当前最高诊断置信度未达 85% 确诊红线。系统已拒绝对此病例下最终定论，建议人工活检复查。"

        cam_target_idx = final_idx if final_idx != -1 else predicted_idx.item()
        targets = [ClassifierOutputTarget(cam_target_idx)]
        
        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(bg_image_np, grayscale_cam, use_rgb=True)
        top_k = min(3, len(probabilities))
        top_k_values, top_k_indices = torch.topk(probabilities, k=top_k)
        if final_idx != -1:
            primary_idx = final_idx
            primary_prob = probabilities[primary_idx].item()
        else:
            primary_idx = top_k_indices[0].item()
            primary_prob = top_k_values[0].item()
        competitor_indices = [idx.item() for idx in top_k_indices if idx.item() != primary_idx]
        if len(competitor_indices) == 0:
            competitor_indices = [top_k_indices[0].item()]
        secondary_idx = competitor_indices[0]
        secondary_prob = probabilities[secondary_idx].item()
        decision_margin = primary_prob - secondary_prob
        if decision_margin >= 0.05:
            why_not_text = f"系统判定“{class_names[primary_idx]}”而非“{class_names[secondary_idx]}”，主要因为校准概率领先 {decision_margin*100:.2f}%（{primary_prob*100:.2f}% vs {secondary_prob*100:.2f}%）。"
        else:
            why_not_text = f"系统当前更倾向“{class_names[primary_idx]}”而非“{class_names[secondary_idx]}”，但两者仅相差 {decision_margin*100:.2f}%（{primary_prob*100:.2f}% vs {secondary_prob*100:.2f}%），建议结合临床信息复核。"
        compare_indices = []
        for idx in [primary_idx, secondary_idx] + [idx.item() for idx in top_k_indices]:
            if idx not in compare_indices:
                compare_indices.append(idx)
        compare_indices = compare_indices[:3]
        topk_cam_cards = []
        for idx in compare_indices:
            card_targets = [ClassifierOutputTarget(idx)]
            card_cam = cam(input_tensor=img_tensor, targets=card_targets)[0, :]
            card_vis = show_cam_on_image(bg_image_np, card_cam, use_rgb=True)
            topk_cam_cards.append({
                "idx": idx,
                "label": class_names[idx],
                "prob": probabilities[idx].item(),
                "image": card_vis
            })
        topk_prob_lines = [f"{rank+1}. {class_names[idx.item()]}：{value.item()*100:.2f}%" for rank, (value, idx) in enumerate(zip(top_k_values, top_k_indices))]
        topk_prob_text = " | ".join(topk_prob_lines)
    if final_idx == -1:
        result_name = "诊断不明确 (需人工介入)"
        confidence = final_confidence * 100
    else:
        result_name = class_names[final_idx]
        confidence = final_confidence * 100

    tab_overview, tab_explain = st.tabs(["🩺 诊断总览", "🔥 可解释性对比"])

    with tab_overview:
        left_col, right_col = st.columns([1, 1.2])
        with left_col:
            st.subheader("原始输入")
            st.image(image_pil, use_container_width=True, caption="患者上传的原图")
        with right_col:
            st.subheader("诊断结论")
            if alert_message:
                if "高召回" in decision_mode:
                    st.error(alert_message)
                else:
                    st.warning(alert_message)
            if final_idx == -1:
                st.info(f"**AI 建议：** \n### {result_name}")
            elif "高危" in result_name or "高召回" in decision_mode and alert_message:
                st.error(f"**预测类别：** \n### {result_name}")
            elif confidence > 90:
                st.success(f"**预测类别：** \n### {result_name}")
            else:
                st.warning(f"**预测类别：** \n### {result_name}")
            st.info(f"**阶段一（良恶性筛查）：** {stage1_result}")
            if risk_level == "高风险":
                st.error(f"**临床风险分层：** {risk_level}")
            elif risk_level == "中风险":
                st.warning(f"**临床风险分层：** {risk_level}")
            else:
                st.success(f"**临床风险分层：** {risk_level}")
            st.info(f"**建议随访方案：** {followup_advice}")
            st.info(f"**AI 校准后置信度：** {confidence:.2f}%")
            st.progress(min(final_confidence, 1.0))

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("恶性风险", f"{malignant_prob*100:.2f}%")
        with m2:
            st.metric("筛查阈值", f"{stage1_threshold*100:.2f}%")
        with m3:
            st.metric("不确定性指数", f"{uncertainty_index*100:.2f}%")
        with m4:
            st.metric("前二类别置信差", f"{top2_margin*100:.2f}%")
        st.caption(f"分层依据：{risk_reason} | 温度系数 T={temperature_scale:.1f}")
        st.caption(f"Top-K 概率对比：{topk_prob_text}")
        st.info(f"**为什么是 A 而不是 B：** {why_not_text}")
        if uncertainty_index >= 0.60 or top2_margin <= 0.08:
            st.warning("⚠️ 当前样本不确定性较高，建议结合病史、皮镜及病理结果进行人工复核。")

    with tab_explain:
        main_col, compare_col = st.columns([1.3, 1])
        with main_col:
            st.subheader("Grad-CAM 主热力图")
            st.image(visualization, use_container_width=True, caption="越红区域代表 AI 做出最终诊断的核心依据")
        with compare_col:
            st.subheader("Top-K 概率")
            for rank, (value, idx) in enumerate(zip(top_k_values, top_k_indices), start=1):
                st.caption(f"{rank}. {class_names[idx.item()]}：{value.item()*100:.2f}%")
            st.info(why_not_text)

        st.subheader("Top-K 类别对比热力图")
        topk_cols = st.columns(len(topk_cam_cards))
        for col, card in zip(topk_cols, topk_cam_cards):
            with col:
                st.image(card["image"], use_container_width=True, caption=f"{card['label']} | {card['prob']*100:.2f}%")
                st.caption(f"关注类别：{card['label']}")

    orig_b64 = img_to_base64(image_pil)
    cam_b64 = img_to_base64(visualization)
    topk_card_1_b64 = img_to_base64(topk_cam_cards[0]["image"]) if len(topk_cam_cards) > 0 else cam_b64
    topk_card_2_b64 = img_to_base64(topk_cam_cards[1]["image"]) if len(topk_cam_cards) > 1 else cam_b64
    
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    html_content = build_html_report(
        current_time=current_time,
        selected_model_info=selected_model_info,
        selected_model_path=selected_model_path,
        format_model_time=format_model_time,
        stage1_result=stage1_result,
        malignant_prob=malignant_prob,
        stage1_threshold=stage1_threshold,
        risk_level=risk_level,
        risk_reason=risk_reason,
        followup_advice=followup_advice,
        result_name=result_name,
        decision_mode=decision_mode,
        confidence=confidence,
        uncertainty_index=uncertainty_index,
        top2_margin=top2_margin,
        temperature_scale=temperature_scale,
        topk_prob_text=topk_prob_text,
        why_not_text=why_not_text,
        orig_b64=orig_b64,
        cam_b64=cam_b64,
        topk_card_1_b64=topk_card_1_b64,
        topk_card_2_b64=topk_card_2_b64
    )
    
    st.download_button(
        label="📄 导出报告",
        data=html_content,
        file_name=f"AI诊断报告_{file_time}.html",
        mime="text/html"
    )

    render_clinical_workbench(
        image_pil=image_pil,
        file_time=file_time,
        current_time=current_time,
        class_names=class_names,
        result_name=result_name,
        decision_mode=decision_mode,
        confidence=confidence,
        stage1_result=stage1_result,
        malignant_prob=malignant_prob,
        risk_level=risk_level,
        uncertainty_index=uncertainty_index,
        top2_margin=top2_margin,
        followup_advice=followup_advice
    )
