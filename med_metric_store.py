import json
import os
import re
from pathlib import Path


METRIC_STORE_PATH = Path(
    os.getenv("MED_AI_METRIC_STORE_PATH", Path(__file__).resolve().parent / "model_metrics.json")
)


def _parse_metric_value(metric_text):
    text = str(metric_text).strip()
    if not text:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text.replace(",", "."))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _read_store():
    if not METRIC_STORE_PATH.exists():
        return {}
    try:
        data = json.loads(METRIC_STORE_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _write_store(data):
    METRIC_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRIC_STORE_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def merge_metric_text(current_metric, persisted_metric):
    """
    返回更可信的指标文本：
    - 二者都可解析为数值时，取更大值对应文本
    - 仅一方可解析时，取可解析的一方
    - 都不可解析时，优先 current_metric（避免覆盖现有展示）
    """
    current_text = str(current_metric).strip()
    persisted_text = str(persisted_metric).strip()

    current_val = _parse_metric_value(current_text)
    persisted_val = _parse_metric_value(persisted_text)

    if current_val is not None and persisted_val is not None:
        return persisted_text if persisted_val > current_val else current_text
    if persisted_val is not None:
        return persisted_text
    if current_val is not None:
        return current_text
    return current_text or persisted_text


def get_stored_best_metric(model_key):
    store = _read_store()
    return str(store.get(str(model_key), "")).strip()


def update_best_metric(model_key, metric_text):
    """
    仅当新值更高时更新；返回 (updated, effective_metric_text)。
    """
    key = str(model_key).strip()
    new_text = str(metric_text).strip()
    new_val = _parse_metric_value(new_text)
    if not key:
        return False, new_text

    store = _read_store()
    old_text = str(store.get(key, "")).strip()
    old_val = _parse_metric_value(old_text)

    should_update = False
    if new_val is not None:
        if old_val is None or new_val > old_val:
            should_update = True
    else:
        if not old_text:
            should_update = True

    if should_update:
        store[key] = new_text
        _write_store(store)
        return True, new_text

    return False, old_text or new_text
