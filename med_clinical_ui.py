import csv
import datetime
import time
import uuid
from pathlib import Path

import streamlit as st

from med_config import (
    FEEDBACK_DIR,
    FEEDBACK_SCHEMA_VERSION,
    CASE_HISTORY_SCHEMA_VERSION,
    FEEDBACK_HEADERS,
    CASE_HISTORY_HEADERS,
)
from med_feedback import (
    normalize_feedback_csv,
    normalize_case_history_csv,
    build_error_case_analysis_from_rows,
    load_feedback_rows,
    export_finetune_candidates,
    save_feedback_rows,
    load_case_history_rows,
    multi_csv_file_lock,
)
from med_inference import parse_percent_text


def _generate_case_id():
    return f"case_{time.time_ns()}_{uuid.uuid4().hex[:8]}"


def render_clinical_workbench(
    image_pil,
    file_time,
    current_time,
    class_names,
    result_name,
    decision_mode,
    confidence,
    stage1_result,
    malignant_prob,
    risk_level,
    uncertainty_index,
    top2_margin,
    followup_advice,
):
    del file_time

    fb_case_id = FEEDBACK_HEADERS[1]
    fb_record_time = FEEDBACK_HEADERS[2]
    fb_image_path = FEEDBACK_HEADERS[3]
    fb_ai_result = FEEDBACK_HEADERS[4]
    fb_decision_mode = FEEDBACK_HEADERS[5]
    fb_ai_conf = FEEDBACK_HEADERS[6]
    fb_doctor_feedback = FEEDBACK_HEADERS[7]
    fb_final_label = FEEDBACK_HEADERS[8]
    fb_review_status = FEEDBACK_HEADERS[9]
    fb_review_time = FEEDBACK_HEADERS[10]
    fb_review_note = FEEDBACK_HEADERS[11]

    ch_patient_id = CASE_HISTORY_HEADERS[1]
    ch_case_id = CASE_HISTORY_HEADERS[2]
    ch_record_time = CASE_HISTORY_HEADERS[3]
    ch_stage1_result = CASE_HISTORY_HEADERS[6]
    ch_malignant_prob = CASE_HISTORY_HEADERS[7]
    ch_risk_level = CASE_HISTORY_HEADERS[8]
    ch_predicted_class = CASE_HISTORY_HEADERS[9]
    ch_calibrated_conf = CASE_HISTORY_HEADERS[10]
    ch_doctor_feedback = CASE_HISTORY_HEADERS[13]
    ch_final_label = CASE_HISTORY_HEADERS[14]
    ch_followup_advice = CASE_HISTORY_HEADERS[15]

    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    feedback_csv_file = FEEDBACK_DIR / "clinical_feedback_log.csv"
    history_csv_file = FEEDBACK_DIR / "case_history_log.csv"
    finetune_csv_file = FEEDBACK_DIR / "finetune_candidates.csv"

    st.markdown("---")
    st.subheader("临床闭环工作台")
    st.caption("以下模块默认折叠，可按需展开。")

    with st.expander("5. 临床专家反馈（模型迭代闭环）", expanded=False):
        st.markdown("医生可在此修正 AI 诊断结果，系统将自动沉淀错判样本用于后续微调。")
        with st.form("feedback_form"):
            st.write("**请评价本次 AI 诊断结果**")
            patient_id_input = st.text_input(
                "患者ID（用于病例级历史追踪）",
                value="",
                placeholder="建议填写真实患者ID；留空将自动生成",
            )
            feedback_status = st.radio(
                "AI 诊断结论是否与临床结果一致？",
                ("未评价", "✅ 诊断准确 (常规归档)", "❌ 诊断有误 (加入错题本)"),
            )
            actual_disease = st.selectbox(
                "若诊断有误，请选择实际疾病（若准确可忽略）",
                ["请选择实际疾病..."] + list(class_names.values()),
            )
            submit_btn = st.form_submit_button("提交临床反馈并入库")
            if submit_btn:
                if feedback_status == "未评价":
                    st.warning("请先选择评价状态。")
                elif feedback_status == "❌ 诊断有误 (加入错题本)" and actual_disease == "请选择实际疾病...":
                    st.warning("已选择诊断有误，请补充实际疾病标签。")
                else:
                    case_id = _generate_case_id()
                    patient_id = patient_id_input.strip() or f"patient_{time.time_ns()}"
                    img_save_path = FEEDBACK_DIR / f"{case_id}.jpg"
                    image_pil.save(img_save_path)
                    final_label = result_name if "准确" in feedback_status else actual_disease

                    with multi_csv_file_lock(feedback_csv_file, history_csv_file):
                        normalize_feedback_csv(feedback_csv_file)
                        normalize_case_history_csv(history_csv_file)

                        with open(feedback_csv_file, mode="a", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            writer.writerow(
                                [
                                    FEEDBACK_SCHEMA_VERSION,
                                    case_id,
                                    current_time,
                                    str(img_save_path.resolve()),
                                    result_name,
                                    decision_mode,
                                    f"{confidence:.2f}%",
                                    feedback_status,
                                    final_label,
                                    "待审核",
                                    "",
                                    "",
                                ]
                            )
                        with open(history_csv_file, mode="a", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            writer.writerow(
                                [
                                    CASE_HISTORY_SCHEMA_VERSION,
                                    patient_id,
                                    case_id,
                                    current_time,
                                    str(img_save_path.resolve()),
                                    decision_mode,
                                    stage1_result,
                                    f"{malignant_prob*100:.2f}%",
                                    risk_level,
                                    result_name,
                                    f"{confidence:.2f}%",
                                    f"{uncertainty_index*100:.2f}%",
                                    f"{top2_margin*100:.2f}%",
                                    feedback_status,
                                    final_label,
                                    followup_advice,
                                ]
                            )
                    st.success("反馈提交成功，已写入反馈与病例历史记录。")

    with st.expander("6. 错误案例分析模块", expanded=False):
        all_feedback_rows = load_feedback_rows(feedback_csv_file)
        summary_rows, pair_samples, total_error_count = build_error_case_analysis_from_rows(all_feedback_rows, class_names)
        if total_error_count == 0:
            st.info("当前暂无已审核通过的错判样本。")
        else:
            st.write(f"累计错判样本：{total_error_count} 条")
            st.dataframe(
                [
                    {
                        "误判对": row["pair_label"],
                        "AI预测": row["ai_label"],
                        "最终标签": row["final_label"],
                        "出现次数": row["count"],
                    }
                    for row in summary_rows
                ],
                use_container_width=True,
                hide_index=True,
            )
            max_show = min(5, len(summary_rows))
            if max_show <= 1:
                show_count = max_show
            else:
                show_count = st.slider("代表误判样例展示数量", min_value=1, max_value=max_show, value=min(3, max_show), step=1)
            for row in summary_rows[:show_count]:
                pair_key = (row["ai_code"], row["final_code"])
                sample = pair_samples.get(pair_key, {})
                sample_path = Path(str(sample.get(fb_image_path, "")).strip())
                sample_time = sample.get(fb_record_time, "未知时间")
                sample_mode = sample.get(fb_decision_mode, "未知模式")
                sample_confidence = sample.get(fb_ai_conf, "未知")
                st.markdown(f"**代表样例：{row['pair_label']}（累计 {row['count']} 条）**")
                info_col, image_col = st.columns([2, 3])
                with info_col:
                    st.write(f"录入时间：{sample_time}")
                    st.write(f"决策模式：{sample_mode}")
                    st.write(f"AI置信度：{sample_confidence}")
                    st.write(f"医生修正：{sample.get(fb_final_label, row['final_label'])}")
                with image_col:
                    if sample_path.exists():
                        st.image(str(sample_path), use_container_width=True, caption=f"误判样例：{sample_path.name}")
                    else:
                        st.warning(f"样例图片不存在：{sample_path}")

    with st.expander("7. 反馈样本审核", expanded=False):
        if not feedback_csv_file.exists():
            st.info("暂无待审核样本。")
        else:
            all_feedback_rows = load_feedback_rows(feedback_csv_file)
            candidate_count = export_finetune_candidates(all_feedback_rows, finetune_csv_file)
            st.caption(f"当前微调候选集：{finetune_csv_file.name}（{candidate_count} 条）")
            pending_rows = [row for row in all_feedback_rows if str(row.get(fb_review_status, "")).strip() in ("", "待审核")]
            if len(pending_rows) == 0:
                st.success("当前没有待审核样本。")
            else:
                st.write(f"待审核样本：{len(pending_rows)} 条")
                st.dataframe(
                    [
                        {
                            "病例ID": row.get(fb_case_id, ""),
                            "录入时间": row.get(fb_record_time, ""),
                            "AI预测结果": row.get(fb_ai_result, ""),
                            "最终修正标签": row.get(fb_final_label, ""),
                            "医生评价": row.get(fb_doctor_feedback, ""),
                            "审核状态": row.get(fb_review_status, "待审核"),
                        }
                        for row in pending_rows
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
                with st.form("review_feedback_form"):
                    review_case_id = st.selectbox("选择待审核病例ID", [row.get(fb_case_id, "") for row in pending_rows])
                    review_action = st.radio("审核结论", ("通过", "驳回"))
                    review_note = st.text_input("审核备注")
                    submit_review = st.form_submit_button("提交审核")
                    if submit_review:
                        review_status = "已审核通过" if review_action == "通过" else "已审核驳回"
                        review_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        with multi_csv_file_lock(feedback_csv_file, finetune_csv_file):
                            latest_feedback_rows = load_feedback_rows(feedback_csv_file)
                            for row in latest_feedback_rows:
                                if row.get(fb_case_id, "") == review_case_id:
                                    row[fb_review_status] = review_status
                                    row[fb_review_time] = review_time
                                    row[fb_review_note] = review_note.strip()
                                    break
                            save_feedback_rows(feedback_csv_file, latest_feedback_rows)
                            export_finetune_candidates(latest_feedback_rows, finetune_csv_file)
                        st.success(f"病例 {review_case_id} 已更新为：{review_status}")
                        st.rerun()

    with st.expander("8. 病例级历史追踪", expanded=False):
        if not history_csv_file.exists():
            st.info("暂无病例历史记录。请先提交至少一条临床反馈。")
        else:
            history_rows = load_case_history_rows(history_csv_file)
            if len(history_rows) == 0:
                st.info("病例历史记录为空。")
            else:
                patient_ids = sorted({str(row.get(ch_patient_id, "")).strip() for row in history_rows if str(row.get(ch_patient_id, "")).strip()})
                if not patient_ids:
                    st.info("历史记录中暂无有效患者ID。")
                else:
                    selected_patient_id = st.selectbox("选择患者ID", patient_ids)
                    patient_rows = [row for row in history_rows if str(row.get(ch_patient_id, "")).strip() == selected_patient_id]
                    patient_rows = sorted(patient_rows, key=lambda row: str(row.get(ch_record_time, "")))
                    st.write(f"患者 {selected_patient_id} 的历史记录：{len(patient_rows)} 条")
                    case_ids = sorted({str(row.get(ch_case_id, "")).strip() for row in patient_rows if str(row.get(ch_case_id, "")).strip()})
                    selected_case_id = st.selectbox("选择病例ID（可选）", ["全部病例"] + case_ids)
                    display_rows = patient_rows if selected_case_id == "全部病例" else [row for row in patient_rows if str(row.get(ch_case_id, "")).strip() == selected_case_id]
                    display_rows = sorted(display_rows, key=lambda row: str(row.get(ch_record_time, "")))
                    if len(display_rows) == 0:
                        st.warning("当前筛选条件下无记录。")
                    else:
                        first_row = display_rows[0]
                        last_row = display_rows[-1]
                        delta_malignant = parse_percent_text(last_row.get(ch_malignant_prob, "0%")) - parse_percent_text(first_row.get(ch_malignant_prob, "0%"))
                        delta_confidence = parse_percent_text(last_row.get(ch_calibrated_conf, "0%")) - parse_percent_text(first_row.get(ch_calibrated_conf, "0%"))
                        stat_col1, stat_col2, stat_col3 = st.columns(3)
                        with stat_col1:
                            st.metric("最新风险分层", last_row.get(ch_risk_level, "未知"))
                        with stat_col2:
                            st.metric("恶性风险趋势变化", f"{delta_malignant:+.2f}%")
                        with stat_col3:
                            st.metric("置信度趋势变化", f"{delta_confidence:+.2f}%")
                        st.dataframe(
                            [
                                {
                                    "录入时间": row.get(ch_record_time, ""),
                                    "患者ID": row.get(ch_patient_id, ""),
                                    "病例ID": row.get(ch_case_id, ""),
                                    "阶段一结论": row.get(ch_stage1_result, ""),
                                    "临床风险分层": row.get(ch_risk_level, ""),
                                    "恶性风险概率": row.get(ch_malignant_prob, ""),
                                    "预测疾病类别": row.get(ch_predicted_class, ""),
                                    "AI校准后置信度": row.get(ch_calibrated_conf, ""),
                                    "医生评价": row.get(ch_doctor_feedback, ""),
                                    "最终修正标签": row.get(ch_final_label, ""),
                                    "随访建议": row.get(ch_followup_advice, ""),
                                }
                                for row in display_rows
                            ],
                            use_container_width=True,
                            hide_index=True,
                        )
