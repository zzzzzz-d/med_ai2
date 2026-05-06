import argparse
import datetime
import sys
from pathlib import Path

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parent
if PROJECT_ROOT_FOR_IMPORT.name == "scripts":
    PROJECT_ROOT_FOR_IMPORT = PROJECT_ROOT_FOR_IMPORT.parent
sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from med_model_arch import create_model, load_state_dict_compatible


DISEASE_CLASSES = {
    "nv": 0,
    "mel": 1,
    "bkl": 2,
    "bcc": 3,
    "akiec": 4,
    "vasc": 5,
    "df": 6,
}

CLASS_NAMES = {
    "nv": "痣 (nv)",
    "mel": "黑色素瘤 (mel)",
    "bkl": "良性角化病 (bkl)",
    "bcc": "基底细胞癌 (bcc)",
    "akiec": "光化性角化病 (akiec)",
    "vasc": "血管性病变 (vasc)",
    "df": "皮肤纤维瘤 (df)",
}

CSV_COLUMNS = [
    "病例ID",
    "病例图片路径",
    "最终修正标签",
    "录入时间",
    "审核时间",
    "审核备注",
    "AI预测结果",
    "当前决策模式",
    "AI置信度",
]


class HAMImageDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.image_dir / f"{row['image_id']}.jpg"
        image = Image.open(image_path).convert("RGB")
        label = int(row["label_idx"])
        return self.transform(image), label, str(row["image_id"]), str(image_path), str(row["dx"])


def read_csv_fallback(path):
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk", "cp936"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="utf-8", errors="replace")


def normalize_candidate_columns(df):
    df = df.copy()
    if list(df.columns[: len(CSV_COLUMNS)]) == CSV_COLUMNS:
        return df[CSV_COLUMNS].copy()
    if len(df.columns) < len(CSV_COLUMNS):
        raise ValueError("finetune candidates CSV columns are incomplete")
    rename_map = {old: new for old, new in zip(df.columns[: len(CSV_COLUMNS)], CSV_COLUMNS)}
    df = df.rename(columns=rename_map)
    return df[CSV_COLUMNS].copy()


def extract_dx_code(label_text):
    text = str(label_text).strip()
    if "(" in text and ")" in text:
        return text.rsplit("(", 1)[-1].replace(")", "").strip()
    return text


def make_augmented_images(feedback_df, output_dir, variants_per_image):
    output_dir.mkdir(parents=True, exist_ok=True)
    augmented_rows = []
    variant_names = ["hflip", "vflip", "rot_m10", "rot_p10", "bright", "contrast"]

    for _, row in feedback_df.iterrows():
        src_path = Path(str(row["病例图片路径"]))
        if not src_path.exists():
            continue
        try:
            image = Image.open(src_path).convert("RGB")
        except OSError:
            continue

        base_case_id = str(row["病例ID"])
        variants = [
            ImageOps.mirror(image),
            ImageOps.flip(image),
            image.rotate(-10, resample=Image.BILINEAR, expand=False),
            image.rotate(10, resample=Image.BILINEAR, expand=False),
            ImageEnhance.Brightness(image).enhance(1.10),
            ImageEnhance.Contrast(image).enhance(1.12),
        ]
        for index, aug_image in enumerate(variants[: max(0, variants_per_image)]):
            aug_id = f"{base_case_id}_aug_{variant_names[index]}"
            aug_path = output_dir / f"{aug_id}.jpg"
            aug_image.save(aug_path, format="JPEG", quality=95)

            aug_row = row.copy()
            aug_row["病例ID"] = aug_id
            aug_row["病例图片路径"] = str(aug_path)
            aug_row["审核备注"] = f"反馈错题数据增强：{variant_names[index]}"
            augmented_rows.append(aug_row)

    if not augmented_rows:
        return pd.DataFrame(columns=CSV_COLUMNS)
    return pd.DataFrame(augmented_rows)[CSV_COLUMNS]


def build_train_split(metadata_csv, seed):
    df = read_csv_fallback(metadata_csv)
    df["label_idx"] = df["dx"].map(DISEASE_CLASSES)
    df = df[df["label_idx"].notna()].copy()
    train_df, _ = train_test_split(
        df,
        test_size=0.2,
        random_state=seed,
        stratify=df["label_idx"],
    )
    return train_df


def load_model(weight_path, device):
    model = create_model("eca_resnet50", num_classes=7).to(device)
    payload = torch.load(str(weight_path), map_location=device)
    load_info = load_state_dict_compatible(model, payload, strict=False)
    if load_info["missing_keys"] or load_info["unexpected_keys"]:
        raise RuntimeError(
            f"Checkpoint incompatible: {weight_path}\n"
            f"missing_keys={load_info['missing_keys'][:10]}\n"
            f"unexpected_keys={load_info['unexpected_keys'][:10]}"
        )
    model.eval()
    return model


def mine_hard_examples(model, train_df, image_dir, device, batch_size, max_count, low_confidence_threshold):
    idx_to_code = {idx: code for code, idx in DISEASE_CLASSES.items()}
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32),
        ]
    )
    loader = DataLoader(HAMImageDataset(train_df, image_dir, transform), batch_size=batch_size, shuffle=False)
    mined = []
    now_text = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with torch.no_grad():
        for images, labels, image_ids, image_paths, dx_codes in loader:
            images = images.to(device)
            probs = torch.softmax(model(images), dim=1).cpu()
            preds = probs.argmax(dim=1)
            confs = probs.max(dim=1).values

            for label, pred, conf, image_id, image_path, dx_code in zip(labels, preds, confs, image_ids, image_paths, dx_codes):
                true_idx = int(label)
                pred_idx = int(pred)
                confidence = float(conf)
                is_mistake = pred_idx != true_idx
                is_low_confidence = confidence < low_confidence_threshold
                if not (is_mistake or is_low_confidence):
                    continue

                true_code = str(dx_code)
                pred_code = idx_to_code[pred_idx]
                mined.append(
                    {
                        "病例ID": f"mined_{image_id}",
                        "病例图片路径": image_path,
                        "最终修正标签": CLASS_NAMES[true_code],
                        "录入时间": now_text,
                        "审核时间": now_text,
                        "审核备注": "训练集模型错题挖掘" if is_mistake else "训练集低置信难例挖掘",
                        "AI预测结果": CLASS_NAMES[pred_code],
                        "当前决策模式": "训练集难例挖掘",
                        "AI置信度": f"{confidence * 100:.2f}%",
                        "_priority": 0 if is_mistake else 1,
                        "_confidence": confidence,
                    }
                )

    if not mined:
        return pd.DataFrame(columns=CSV_COLUMNS)

    mined_df = pd.DataFrame(mined).sort_values(["_priority", "_confidence"], ascending=[True, True])
    mined_df = mined_df.head(max_count)
    return mined_df[CSV_COLUMNS].copy()


def main():
    parser = argparse.ArgumentParser(description="Build expanded feedback fine-tuning pool.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT_FOR_IMPORT)
    parser.add_argument("--feedback-dir", type=Path, default=None)
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--augmented-dir", type=Path, default=None)
    parser.add_argument("--augment-per-image", type=int, default=4)
    parser.add_argument("--mine-max-count", type=int, default=300)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    feedback_dir = args.feedback_dir or (project_root / "feedback_data1")
    metadata_csv = args.metadata_csv or (project_root / "dataset" / "HAM10000_metadata.csv")
    images_dir = args.images_dir or (project_root / "dataset" / "images")
    model_path = args.model_path or (project_root / "outputs" / "checkpoints" / "eca_resnet50_v8_best_acc87_97.pth")
    output_csv = args.output_csv or (feedback_dir / "finetune_candidates_expanded.csv")
    augmented_dir = args.augmented_dir or (feedback_dir / "augmented_cases")
    source_csv = feedback_dir / "finetune_candidates.csv"

    if not source_csv.exists():
        raise FileNotFoundError(f"Feedback candidates not found: {source_csv}")
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model weight not found: {model_path}")

    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    feedback_df = normalize_candidate_columns(read_csv_fallback(source_csv))
    feedback_df = feedback_df[feedback_df["最终修正标签"].map(extract_dx_code).isin(DISEASE_CLASSES)].copy()

    augmented_df = make_augmented_images(feedback_df, augmented_dir, args.augment_per_image)
    train_df = build_train_split(metadata_csv, args.seed)
    model = load_model(model_path, device)
    mined_df = mine_hard_examples(
        model=model,
        train_df=train_df,
        image_dir=images_dir,
        device=device,
        batch_size=args.batch_size,
        max_count=args.mine_max_count,
        low_confidence_threshold=args.low_confidence_threshold,
    )

    expanded_df = pd.concat([feedback_df, augmented_df, mined_df], ignore_index=True)
    expanded_df = expanded_df.drop_duplicates(subset=["病例ID"], keep="first")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    expanded_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"Source feedback rows: {len(feedback_df)}")
    print(f"Augmented feedback rows: {len(augmented_df)}")
    print(f"Mined hard-example rows: {len(mined_df)}")
    print(f"Expanded rows: {len(expanded_df)}")
    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    main()
