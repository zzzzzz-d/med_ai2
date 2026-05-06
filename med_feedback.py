import csv
import datetime
import builtins
import os
import threading
import time
from collections import Counter
from contextlib import ExitStack, contextmanager
from pathlib import Path

from med_config import FEEDBACK_SCHEMA_VERSION, FEEDBACK_HEADERS, CASE_HISTORY_SCHEMA_VERSION, CASE_HISTORY_HEADERS


CSV_READ_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk", "cp936")
CSV_LOCK_TIMEOUT_SECONDS = float(os.getenv("MED_AI_CSV_LOCK_TIMEOUT_SECONDS", "10"))
CSV_LOCK_POLL_INTERVAL_SECONDS = float(os.getenv("MED_AI_CSV_LOCK_POLL_INTERVAL_SECONDS", "0.1"))
CSV_LOCK_STALE_SECONDS = float(os.getenv("MED_AI_CSV_LOCK_STALE_SECONDS", "300"))
_THREAD_LOCK_STATE = threading.local()


def _get_thread_lock_state():
    if not hasattr(_THREAD_LOCK_STATE, "depth"):
        _THREAD_LOCK_STATE.depth = {}
    if not hasattr(_THREAD_LOCK_STATE, "fd"):
        _THREAD_LOCK_STATE.fd = {}
    return _THREAD_LOCK_STATE.depth, _THREAD_LOCK_STATE.fd


@contextmanager
def csv_file_lock(csv_file, timeout_seconds=CSV_LOCK_TIMEOUT_SECONDS, poll_interval_seconds=CSV_LOCK_POLL_INTERVAL_SECONDS):
    """
    Process-safe lock for CSV read-modify-write sections.
    Uses sidecar lock file: <csv>.lock
    """
    lock_file = Path(f"{csv_file}.lock")
    lock_key = str(lock_file.resolve())
    depth_map, fd_map = _get_thread_lock_state()
    current_depth = int(depth_map.get(lock_key, 0))
    if current_depth > 0:
        depth_map[lock_key] = current_depth + 1
        try:
            yield
        finally:
            depth_map[lock_key] = depth_map[lock_key] - 1
        return

    lock_fd = None
    deadline = time.monotonic() + max(float(timeout_seconds), 0.1)
    poll_interval = max(float(poll_interval_seconds), 0.01)
    stale_seconds = max(float(CSV_LOCK_STALE_SECONDS), 0.0)

    while True:
        try:
            lock_fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            break
        except FileExistsError:
            if stale_seconds > 0:
                try:
                    age = time.time() - lock_file.stat().st_mtime
                    if age > stale_seconds:
                        try:
                            lock_file.unlink()
                        except OSError:
                            pass
                        continue
                except FileNotFoundError:
                    continue

            if time.monotonic() >= deadline:
                raise TimeoutError(f"CSV lock timeout: {lock_file}")
            time.sleep(poll_interval)

    depth_map[lock_key] = 1
    fd_map[lock_key] = lock_fd
    try:
        yield
    finally:
        depth = int(depth_map.get(lock_key, 1)) - 1
        if depth > 0:
            depth_map[lock_key] = depth
            return

        depth_map.pop(lock_key, None)
        fd = fd_map.pop(lock_key, None)
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            lock_file.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def multi_csv_file_lock(*csv_files):
    """
    Acquire multiple CSV locks in stable order to avoid deadlocks.
    """
    files = sorted({str(Path(item)) for item in csv_files if item is not None})
    with ExitStack() as stack:
        for item in files:
            stack.enter_context(csv_file_lock(item))
        yield


def _read_csv_with_fallback(csv_file, as_dict=False, return_encoding=False):
    """
    兼容历史 CSV 编码（UTF-8/GBK 等），避免 UnicodeDecodeError。
    读取失败时最终使用 utf-8 + errors=replace 保底。
    """
    for encoding in CSV_READ_ENCODINGS:
        try:
            with builtins.open(csv_file, mode="r", newline="", encoding=encoding) as f:
                reader = csv.DictReader(f) if as_dict else csv.reader(f)
                rows = [row for row in reader]
                if return_encoding:
                    return rows, encoding
                return rows
        except UnicodeDecodeError:
            continue
    with builtins.open(csv_file, mode="r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f) if as_dict else csv.reader(f)
        rows = [row for row in reader]
        if return_encoding:
            return rows, "utf-8-replace"
        return rows


# 基于时间与图片路径生成病例ID，兼容历史空值场景
def _make_case_id(record_time, image_path):
    image_name = Path(str(image_path).strip()).stem
    if image_name:
        return image_name
    safe_time = str(record_time).replace(" ", "_").replace(":", "").replace("-", "")
    return f"legacy_{safe_time}" if safe_time else f"legacy_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"


# 判断反馈记录是否处于“可用于训练/分析”的审核通过状态
def is_approved_review(review_status):
    status = str(review_status).strip()
    return status in ("已审核通过", "已审核通过(历史迁移)")


# 将单行反馈记录规范化为当前统一Schema，兼容多种历史列格式
def _normalize_feedback_row(row):
    normalized = [str(item).strip() for item in row]
    if not normalized or not any(normalized):
        return None
    first_col = normalized[0]
    if first_col in ("Schema版本", "病例ID", "录入时间"):
        return None
    if len(normalized) >= 12:
        if first_col.lower().startswith("v"):
            payload = normalized[1:12]
            return [FEEDBACK_SCHEMA_VERSION] + payload
        payload = normalized[:11]
        return [FEEDBACK_SCHEMA_VERSION] + payload
    if len(normalized) == 11:
        return [FEEDBACK_SCHEMA_VERSION] + normalized
    if len(normalized) == 8:
        if first_col.lower().startswith("v"):
            record_time, image_path, ai_result, decision_mode, ai_conf, doctor_feedback, final_label = normalized[1:8]
        else:
            record_time, image_path, ai_result, decision_mode, ai_conf, doctor_feedback, final_label = normalized[0:7]
        case_id = _make_case_id(record_time, image_path)
        return [FEEDBACK_SCHEMA_VERSION, case_id, record_time, image_path, ai_result, decision_mode, ai_conf, doctor_feedback, final_label, "已审核通过(历史迁移)", "", "历史数据自动迁移"]
    if len(normalized) == 7:
        record_time, image_path, ai_result, decision_mode, ai_conf, doctor_feedback, final_label = normalized
        case_id = _make_case_id(record_time, image_path)
        return [FEEDBACK_SCHEMA_VERSION, case_id, record_time, image_path, ai_result, decision_mode, ai_conf, doctor_feedback, final_label, "已审核通过(历史迁移)", "", "历史数据自动迁移"]
    if len(normalized) == 6:
        record_time, image_path, ai_result, ai_conf, doctor_feedback, final_label = normalized
        case_id = _make_case_id(record_time, image_path)
        return [FEEDBACK_SCHEMA_VERSION, case_id, record_time, image_path, ai_result, "未知(历史数据)", ai_conf, doctor_feedback, final_label, "已审核通过(历史迁移)", "", "历史数据自动迁移"]
    return None


# 读取并标准化反馈CSV，重写为统一表头与字段顺序
def normalize_feedback_csv(csv_file):
    with csv_file_lock(csv_file):
        normalized_rows = []
        existing_rows = []
        source_encoding = "utf-8"
        if csv_file.exists():
            existing_rows, source_encoding = _read_csv_with_fallback(csv_file, as_dict=False, return_encoding=True)
            for raw_row in existing_rows:
                normalized_row = _normalize_feedback_row(raw_row)
                if normalized_row:
                    normalized_rows.append(normalized_row)
        expected_rows = [FEEDBACK_HEADERS] + normalized_rows
        needs_rewrite = (
            (not csv_file.exists())
            or (existing_rows != expected_rows)
            or (source_encoding not in ("utf-8", "utf-8-sig"))
        )
        if needs_rewrite:
            with builtins.open(csv_file, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(FEEDBACK_HEADERS)
                writer.writerows(normalized_rows)
        return needs_rewrite


# 加载反馈记录，并在加载前自动执行格式标准化
def load_feedback_rows(csv_file):
    normalize_feedback_csv(csv_file)
    return _read_csv_with_fallback(csv_file, as_dict=True)


# 按统一字段将反馈记录整体写回CSV
def save_feedback_rows(csv_file, rows):
    with csv_file_lock(csv_file):
        with builtins.open(csv_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FEEDBACK_HEADERS)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in FEEDBACK_HEADERS})


# 将单行病例历史记录规范化为当前统一Schema
def _normalize_case_history_row(row):
    normalized = [str(item).strip() for item in row]
    if not normalized or not any(normalized):
        return None
    first_col = normalized[0]
    if first_col in ("Schema版本", "患者ID", "病例ID", "录入时间"):
        return None
    if len(normalized) >= len(CASE_HISTORY_HEADERS):
        if first_col.lower().startswith("v"):
            payload = normalized[1:len(CASE_HISTORY_HEADERS)]
            return [CASE_HISTORY_SCHEMA_VERSION] + payload
        payload = normalized[:len(CASE_HISTORY_HEADERS) - 1]
        return [CASE_HISTORY_SCHEMA_VERSION] + payload
    if len(normalized) == len(CASE_HISTORY_HEADERS) - 1:
        return [CASE_HISTORY_SCHEMA_VERSION] + normalized
    return None


# 读取并标准化病例历史CSV，重写为统一表头与字段顺序
def normalize_case_history_csv(csv_file):
    with csv_file_lock(csv_file):
        normalized_rows = []
        existing_rows = []
        source_encoding = "utf-8"
        if csv_file.exists():
            existing_rows, source_encoding = _read_csv_with_fallback(csv_file, as_dict=False, return_encoding=True)
            for raw_row in existing_rows:
                normalized_row = _normalize_case_history_row(raw_row)
                if normalized_row:
                    normalized_rows.append(normalized_row)
        expected_rows = [CASE_HISTORY_HEADERS] + normalized_rows
        needs_rewrite = (
            (not csv_file.exists())
            or (existing_rows != expected_rows)
            or (source_encoding not in ("utf-8", "utf-8-sig"))
        )
        if needs_rewrite:
            with builtins.open(csv_file, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CASE_HISTORY_HEADERS)
                writer.writerows(normalized_rows)
        return needs_rewrite


# 加载病例历史记录，并在加载前自动执行格式标准化
def load_case_history_rows(csv_file):
    normalize_case_history_csv(csv_file)
    return _read_csv_with_fallback(csv_file, as_dict=True)


# 导出“已审核且诊断有误”的微调候选样本集
def export_finetune_candidates(rows, output_file):
    candidate_headers = ["病例ID", "病例图片路径", "最终修正标签", "录入时间", "审核时间", "审核备注", "AI预测结果", "当前决策模式", "AI置信度"]
    approved_error_rows = []
    for row in rows:
        doctor_feedback = str(row.get("医生评价", "")).strip()
        if "有误" not in doctor_feedback:
            continue
        if not is_approved_review(row.get("审核状态", "")):
            continue
        approved_error_rows.append({
            "病例ID": row.get("病例ID", ""),
            "病例图片路径": row.get("病例图片路径", ""),
            "最终修正标签": row.get("最终修正标签", ""),
            "录入时间": row.get("录入时间", ""),
            "审核时间": row.get("审核时间", ""),
            "审核备注": row.get("审核备注", ""),
            "AI预测结果": row.get("AI预测结果", ""),
            "当前决策模式": row.get("当前决策模式", ""),
            "AI置信度": row.get("AI置信度", "")
        })
    with csv_file_lock(output_file):
        with builtins.open(output_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=candidate_headers)
            writer.writeheader()
            writer.writerows(approved_error_rows)
    return len(approved_error_rows)


# 从“中文名(缩写)”格式中提取疾病缩写编码
def extract_dx_code(label_text):
    text = str(label_text).strip()
    if "(" in text and ")" in text:
        return text.rsplit("(", 1)[-1].replace(")", "").strip()
    return text


# 汇总高频误判对，并为每个误判对选取代表样本用于界面展示
def build_error_case_analysis(csv_file, class_names_map):
    if not csv_file.exists():
        return [], {}, 0
    normalize_feedback_csv(csv_file)
    code_to_name = {}
    for _, class_label in class_names_map.items():
        class_code = extract_dx_code(class_label)
        code_to_name[class_code] = class_label
    pair_counter = Counter()
    pair_samples = {}
    total_error_count = 0
    for row in _read_csv_with_fallback(csv_file, as_dict=True):
            doctor_feedback = str(row.get("医生评价", "")).strip()
            if "有误" not in doctor_feedback:
                continue
            if not is_approved_review(row.get("审核状态", "")):
                continue
            ai_code = extract_dx_code(row.get("AI预测结果", ""))
            final_code = extract_dx_code(row.get("最终修正标签", ""))
            if not ai_code or not final_code:
                continue
            total_error_count += 1
            pair_key = (ai_code, final_code)
            pair_counter[pair_key] += 1
            old_sample = pair_samples.get(pair_key)
            current_time = str(row.get("录入时间", ""))
            old_time = str(old_sample.get("录入时间", "")) if old_sample else ""
            if old_sample is None or current_time >= old_time:
                pair_samples[pair_key] = row
    summary_rows = []
    for (ai_code, final_code), count in pair_counter.most_common():
        summary_rows.append({
            "误判对": f"{ai_code} → {final_code}",
            "AI预测": code_to_name.get(ai_code, ai_code),
            "最终标签": code_to_name.get(final_code, final_code),
            "出现次数": count,
            "AI预测简写": ai_code,
            "最终标签简写": final_code
        })
    return summary_rows, pair_samples, total_error_count


def build_error_case_analysis_from_rows(feedback_rows, class_names_map):
    if not feedback_rows:
        return [], {}, 0

    doctor_feedback_key = FEEDBACK_HEADERS[7]
    review_status_key = FEEDBACK_HEADERS[9]
    ai_result_key = FEEDBACK_HEADERS[4]
    final_label_key = FEEDBACK_HEADERS[8]
    record_time_key = FEEDBACK_HEADERS[2]

    code_to_name = {}
    for _, class_label in class_names_map.items():
        class_code = extract_dx_code(class_label)
        code_to_name[class_code] = class_label

    pair_counter = Counter()
    pair_samples = {}
    total_error_count = 0

    for row in feedback_rows:
        doctor_feedback = str(row.get(doctor_feedback_key, "")).strip()
        if "有误" not in doctor_feedback:
            continue
        if not is_approved_review(row.get(review_status_key, "")):
            continue
        ai_code = extract_dx_code(row.get(ai_result_key, ""))
        final_code = extract_dx_code(row.get(final_label_key, ""))
        if not ai_code or not final_code:
            continue

        total_error_count += 1
        pair_key = (ai_code, final_code)
        pair_counter[pair_key] += 1

        old_sample = pair_samples.get(pair_key)
        current_time = str(row.get(record_time_key, ""))
        old_time = str(old_sample.get(record_time_key, "")) if old_sample else ""
        if old_sample is None or current_time >= old_time:
            pair_samples[pair_key] = row

    summary_rows = []
    for (ai_code, final_code), count in pair_counter.most_common():
        summary_rows.append(
            {
                "pair_label": f"{ai_code} -> {final_code}",
                "ai_label": code_to_name.get(ai_code, ai_code),
                "final_label": code_to_name.get(final_code, final_code),
                "count": count,
                "ai_code": ai_code,
                "final_code": final_code,
            }
        )

    return summary_rows, pair_samples, total_error_count
