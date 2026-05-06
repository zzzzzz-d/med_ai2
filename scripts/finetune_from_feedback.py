import argparse
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parent
if PROJECT_ROOT_FOR_IMPORT.name == "scripts":
    PROJECT_ROOT_FOR_IMPORT = PROJECT_ROOT_FOR_IMPORT.parent
sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms

from med_config import resolve_torch_device
from med_model_arch import ECAResNet50, load_state_dict_compatible
from med_metric_store import update_best_metric


def get_project_root():
    project_root = Path(os.getenv("MED_AI_PROJECT_ROOT", PROJECT_ROOT_FOR_IMPORT))
    if not (project_root / "dataset").exists():
        project_root = Path.cwd()
    return project_root


PROJECT_ROOT = get_project_root()
DATASET_DIR = PROJECT_ROOT / "dataset"
METADATA_CSV = DATASET_DIR / "HAM10000_metadata.csv"
IMAGES_DIR = DATASET_DIR / "images"
FEEDBACK_DIR = Path(os.getenv("MED_AI_FEEDBACK_DIR", PROJECT_ROOT / "feedback_data1"))
FINETUNE_CANDIDATES_CSV = FEEDBACK_DIR / "finetune_candidates.csv"
FEEDBACK_CSV = FEEDBACK_DIR / "clinical_feedback_log.csv"
MODEL_V2_BEST_PATH = Path(os.getenv("MED_AI_MODEL_V2_BEST_PATH", PROJECT_ROOT / "outputs" / "checkpoints" / "eca_resnet50_v2_best.pth"))
FINETUNE_BASE_MODEL_PATH = Path(os.getenv("MED_AI_FINETUNE_BASE_MODEL_PATH", MODEL_V2_BEST_PATH))
MODEL_FINETUNED_PATH = Path(os.getenv("MED_AI_MODEL_V2_FINETUNED_PATH", PROJECT_ROOT / "outputs" / "checkpoints" / "eca_resnet50_v2_finetuned.pth"))
GLOBAL_SEED = int(os.getenv("MED_AI_SEED", "42"))
FINE_TUNE_EPOCHS = int(os.getenv("MED_AI_FINETUNE_EPOCHS", "6"))
FINE_TUNE_LR = float(os.getenv("MED_AI_FINETUNE_LR", "1e-5"))
BATCH_SIZE = int(os.getenv("MED_AI_FINETUNE_BATCH_SIZE", "8"))
EVAL_ONLY_DEFAULT = str(os.getenv("MED_AI_EVAL_ONLY", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}

# Anti-forgetting strategy: mixed sampling + early stop + regression gate
BASE_TO_HARD_RATIO = float(os.getenv("MED_AI_BASE_TO_HARD_RATIO", "2.0"))
MIN_BASE_SAMPLES = int(os.getenv("MED_AI_MIN_BASE_SAMPLES", "64"))
MAX_BASE_SAMPLES = int(os.getenv("MED_AI_MAX_BASE_SAMPLES", "2000"))
EARLY_STOP_PATIENCE = int(os.getenv("MED_AI_FINETUNE_PATIENCE", "2"))
EARLY_STOP_MIN_DELTA = float(os.getenv("MED_AI_FINETUNE_MIN_DELTA", "0.05"))
MAX_ALLOWED_VAL_DROP = float(os.getenv("MED_AI_MAX_ALLOWED_VAL_DROP", "-1.0"))


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_dx_code(label_text):
    text = str(label_text).strip()
    if "(" in text and ")" in text:
        return text.rsplit("(", 1)[-1].replace(")", "").strip()
    return text


def _pick_column(df, candidates, fallback_index, role_name):
    for name in candidates:
        if name in df.columns:
            return name
    if 0 <= fallback_index < len(df.columns):
        return df.columns[fallback_index]
    raise KeyError(f"Cannot identify column for: {role_name}")


def _snapshot_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


class HAMImageDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.image_dir / f"{row['image_id']}.jpg"
        image = Image.open(image_path).convert("RGB")
        label = int(row["label_idx"])
        return self.transform(image), label


class FeedbackDataset(Dataset):
    def __init__(self, dataframe, transform, disease_classes):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.disease_classes = disease_classes

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        label = int(self.disease_classes[row["dx_code"]])
        return self.transform(image), label


def evaluate_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / max(total, 1)


def prepare_hard_examples(feedback_df, source_name, disease_classes):
    if source_name == "finetune_candidates.csv":
        image_col = _pick_column(feedback_df, ["病例图片路径", "image_path"], 1, "image_path")
        final_label_col = _pick_column(feedback_df, ["最终修正标签", "final_label"], 2, "final_label")
        hard_examples = feedback_df.copy()
    else:
        doctor_feedback_col = _pick_column(feedback_df, ["医生评价", "doctor_feedback"], 7, "doctor_feedback")
        review_status_col = _pick_column(feedback_df, ["审核状态", "review_status"], 9, "review_status")
        image_col = _pick_column(feedback_df, ["病例图片路径", "image_path"], 3, "image_path")
        final_label_col = _pick_column(feedback_df, ["最终修正标签", "final_label"], 8, "final_label")
        hard_examples = feedback_df[
            feedback_df[doctor_feedback_col].astype(str).str.contains("有误|error|wrong", case=False, na=False)
            & feedback_df[review_status_col].astype(str).str.contains("通过|approved|迁移", case=False, na=False)
        ].copy()

    hard_examples["image_path"] = hard_examples[image_col].astype(str).map(lambda p: str(Path(p)))
    hard_examples = hard_examples[hard_examples["image_path"].map(lambda p: Path(p).exists())].copy()
    hard_examples["dx_code"] = hard_examples[final_label_col].map(extract_dx_code)
    hard_examples = hard_examples[hard_examples["dx_code"].isin(disease_classes.keys())].copy()
    return hard_examples


def choose_base_sample_count(hard_count, available_train_count):
    if available_train_count <= 0:
        return 0
    ratio_target = int(max(hard_count * BASE_TO_HARD_RATIO, hard_count))
    target = max(ratio_target, MIN_BASE_SAMPLES)
    if MAX_BASE_SAMPLES > 0:
        target = min(target, MAX_BASE_SAMPLES)
    return min(target, available_train_count)


def main():
    parser = argparse.ArgumentParser(description="Incremental fine-tuning with anti-forgetting strategy")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate, do not train/save")
    args = parser.parse_args()

    eval_only = args.eval_only or EVAL_ONLY_DEFAULT
    set_global_seed(GLOBAL_SEED)
    device, device_policy = resolve_torch_device(os.getenv("MED_AI_DEVICE", "auto"))
    disease_classes = {"nv": 0, "mel": 1, "bkl": 2, "bcc": 3, "akiec": 4, "vasc": 5, "df": 6}

    source_csv = FINETUNE_CANDIDATES_CSV if FINETUNE_CANDIDATES_CSV.exists() else FEEDBACK_CSV
    if not source_csv.exists():
        print(f"Feedback file not found: {source_csv}")
        return
    if not METADATA_CSV.exists():
        print(f"Metadata file not found: {METADATA_CSV}")
        return
    if not FINETUNE_BASE_MODEL_PATH.exists():
        print(f"Base model weight not found: {FINETUNE_BASE_MODEL_PATH}")
        return

    feedback_df = pd.read_csv(source_csv)
    hard_examples = prepare_hard_examples(feedback_df, source_csv.name, disease_classes)
    if hard_examples.empty:
        print("No approved hard examples found. Skip fine-tuning.")
        return

    metadata_df = pd.read_csv(METADATA_CSV)
    metadata_df["label_idx"] = metadata_df["dx"].map(disease_classes)
    metadata_df = metadata_df[metadata_df["label_idx"].notna()].copy()
    train_df, val_df = train_test_split(
        metadata_df,
        test_size=0.2,
        random_state=GLOBAL_SEED,
        stratify=metadata_df["label_idx"],
    )

    normalize_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    augment_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    base_sample_count = choose_base_sample_count(len(hard_examples), len(train_df))
    base_samples_df = (
        train_df.sample(n=base_sample_count, random_state=GLOBAL_SEED, replace=False).copy()
        if base_sample_count > 0
        else train_df.iloc[0:0].copy()
    )

    val_loader = DataLoader(HAMImageDataset(val_df, IMAGES_DIR, normalize_transform), batch_size=32, shuffle=False)
    hard_eval_loader = DataLoader(FeedbackDataset(hard_examples, normalize_transform, disease_classes), batch_size=32, shuffle=False)

    mixed_datasets = [FeedbackDataset(hard_examples, augment_transform, disease_classes)]
    if len(base_samples_df) > 0:
        mixed_datasets.append(HAMImageDataset(base_samples_df, IMAGES_DIR, augment_transform))
    train_dataset = ConcatDataset(mixed_datasets) if len(mixed_datasets) > 1 else mixed_datasets[0]
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ECAResNet50(num_classes=7).to(device)
    checkpoint_payload = torch.load(str(FINETUNE_BASE_MODEL_PATH), map_location=device)
    load_info = load_state_dict_compatible(model, checkpoint_payload, strict=False)
    missing_keys = load_info.get("missing_keys", [])
    unexpected_keys = load_info.get("unexpected_keys", [])
    if missing_keys or unexpected_keys:
        print("Checkpoint/model mismatch. Please use a compatible base model.")
        if missing_keys:
            print(f"Missing keys ({len(missing_keys)}): {missing_keys[:10]}")
        if unexpected_keys:
            print(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:10]}")
        return
    if load_info["legacy_mapped"] > 0:
        print(f"Legacy checkpoint keys mapped: {load_info['legacy_mapped']}")

    baseline_val_acc = evaluate_accuracy(model, val_loader, device)
    baseline_hard_acc = evaluate_accuracy(model, hard_eval_loader, device)
    print(f"Baseline val acc: {baseline_val_acc:.2f}% | device: {device} ({device_policy})")
    print(f"Baseline hard acc: {baseline_hard_acc:.2f}%")
    print(
        f"Train mix -> hard: {len(hard_examples)} | base: {len(base_samples_df)} | "
        f"total: {len(train_dataset)} | source: {source_csv}"
    )

    if eval_only:
        print("Eval-only mode enabled. Skip training and saving.")
        return

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=FINE_TUNE_LR)

    best_state_dict = _snapshot_state_dict(model)
    best_val_acc = baseline_val_acc
    best_hard_acc = baseline_hard_acc
    stale_epochs = 0

    for epoch in range(FINE_TUNE_EPOCHS):
        model.train()
        total_loss = 0.0
        step_count = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            step_count += 1

        avg_loss = total_loss / max(step_count, 1)
        val_acc = evaluate_accuracy(model, val_loader, device)
        hard_acc = evaluate_accuracy(model, hard_eval_loader, device)
        improved = val_acc > (best_val_acc + EARLY_STOP_MIN_DELTA)
        if improved:
            best_val_acc = val_acc
            best_hard_acc = hard_acc
            best_state_dict = _snapshot_state_dict(model)
            stale_epochs = 0
            improve_tag = "improved"
        else:
            stale_epochs += 1
            improve_tag = f"no_improve({stale_epochs}/{EARLY_STOP_PATIENCE})"

        print(
            f"Epoch {epoch + 1}/{FINE_TUNE_EPOCHS} | loss={avg_loss:.4f} | "
            f"val={val_acc:.2f}% | hard={hard_acc:.2f}% | {improve_tag}"
        )

        if stale_epochs >= EARLY_STOP_PATIENCE:
            print("Early stopping triggered.")
            break

    model.load_state_dict(best_state_dict)
    post_val_acc = evaluate_accuracy(model, val_loader, device)
    post_hard_acc = evaluate_accuracy(model, hard_eval_loader, device)
    val_delta = post_val_acc - baseline_val_acc
    hard_delta = post_hard_acc - baseline_hard_acc

    print(
        f"Best checkpoint -> val={post_val_acc:.2f}% ({val_delta:+.2f}%) | "
        f"hard={post_hard_acc:.2f}% ({hard_delta:+.2f}%)"
    )
    print(f"Best in training -> val={best_val_acc:.2f}% | hard={best_hard_acc:.2f}%")

    if val_delta < MAX_ALLOWED_VAL_DROP:
        print(
            f"Regression gate triggered: val drop {val_delta:.2f}% < "
            f"{MAX_ALLOWED_VAL_DROP:.2f}%. Skip saving."
        )
        return

    MODEL_FINETUNED_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(MODEL_FINETUNED_PATH))
    updated, effective_metric = update_best_metric("finetuned", f"{post_val_acc:.2f}%")
    if updated:
        print(f"finetuned best metric updated: {effective_metric}")
    else:
        print(f"finetuned best metric kept: {effective_metric}")
    print(f"Fine-tuned checkpoint saved: {MODEL_FINETUNED_PATH}")


if __name__ == "__main__":
    main()
