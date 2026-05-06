import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parent
if PROJECT_ROOT_FOR_IMPORT.name == "scripts":
    PROJECT_ROOT_FOR_IMPORT = PROJECT_ROOT_FOR_IMPORT.parent
sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from med_model_arch import ECAResNet50, load_state_dict_compatible


DISEASE_CLASSES = {
    "nv": 0,
    "mel": 1,
    "bkl": 2,
    "bcc": 3,
    "akiec": 4,
    "vasc": 5,
    "df": 6,
}
CLASS_NAMES = list(DISEASE_CLASSES.keys())


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


def find_latest_optimized_model(project_root):
    pattern = re.compile(r"eca_resnet50_v(\d+)_best(?:_acc\d+_\d+)?\.pth$")
    candidates = []
    search_dirs = [
        project_root / "outputs" / "checkpoints",
        project_root,
    ]
    for model_dir in search_dirs:
        if not model_dir.exists():
            continue
        for p in model_dir.glob("eca_resnet50_v*_best*.pth"):
            m = pattern.match(p.name)
            if m:
                candidates.append((int(m.group(1)), p))
    if not candidates:
        return project_root / "outputs" / "checkpoints" / "eca_resnet50_v2_best.pth"
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def build_val_loader(metadata_csv, images_dir, seed, batch_size, preprocess_mode):
    df = pd.read_csv(metadata_csv)
    df["label_idx"] = df["dx"].map(DISEASE_CLASSES)
    df = df[df["label_idx"].notna()].copy()

    _, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=seed,
        stratify=df["label_idx"],
    )

    if preprocess_mode == "imagenet_norm":
        val_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        # Match notebook training pipeline in 01_data_process.ipynb
        val_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float32),
            ]
        )
    val_dataset = HAMImageDataset(val_df, images_dir, val_transform)
    return DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


def predict_labels(model_path, loader, device):
    model = ECAResNet50(num_classes=7).to(device)
    payload = torch.load(str(model_path), map_location=device)
    load_info = load_state_dict_compatible(model, payload, strict=False)
    if load_info["missing_keys"] or load_info["unexpected_keys"]:
        raise RuntimeError(
            f"Incompatible checkpoint: {model_path}\n"
            f"missing_keys={load_info['missing_keys'][:10]}\n"
            f"unexpected_keys={load_info['unexpected_keys'][:10]}"
        )

    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())
    return np.array(y_true), np.array(y_pred)


def save_confusion_comparison(cm_a, cm_b, output_path, title_a, title_b):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=180)
    cm_list = [cm_a, cm_b]
    title_list = [title_a, title_b]

    for ax, cm, title in zip(axes, cm_list, title_list):
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(title)
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                text = f"{val:.2f}"
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_recall_bar_compare(recall_a, recall_b, output_path, label_a, label_b):
    x = np.arange(len(CLASS_NAMES))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 6), dpi=180)
    ax.bar(x - width / 2, recall_a, width, label=label_a)
    ax.bar(x + width / 2, recall_b, width, label=label_b)

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Recall")
    ax.set_xlabel("Class")
    ax.set_title("Recall Comparison by Class")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for idx, val in enumerate(recall_a):
        ax.text(idx - width / 2, val + 0.01, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    for idx, val in enumerate(recall_b):
        ax.text(idx + width / 2, val + 0.01, f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate confusion matrix & recall comparison figures.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT_FOR_IMPORT)
    parser.add_argument("--original", type=Path, default=None, help="Original model path")
    parser.add_argument("--optimized", type=Path, default=None, help="Optimized model path")
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--preprocess",
        type=str,
        default="notebook",
        choices=["notebook", "imagenet_norm"],
        help="notebook matches 01_data_process.ipynb pipeline; imagenet_norm uses Normalize(mean,std).",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    original_model = args.original or (project_root / "outputs" / "checkpoints" / "eca_resnet50_best.pth")
    optimized_model = args.optimized or find_latest_optimized_model(project_root)
    metadata_csv = args.metadata_csv or (project_root / "dataset" / "HAM10000_metadata.csv")
    images_dir = args.images_dir or (project_root / "dataset" / "images")
    output_dir = args.output_dir or (project_root / "outputs" / "figures" / "thesis_figures")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    for path_obj in [original_model, optimized_model, metadata_csv, images_dir]:
        if not Path(path_obj).exists():
            raise FileNotFoundError(f"Not found: {path_obj}")

    loader = build_val_loader(metadata_csv, images_dir, args.seed, args.batch_size, args.preprocess)
    y_true_a, y_pred_a = predict_labels(original_model, loader, device)
    y_true_b, y_pred_b = predict_labels(optimized_model, loader, device)

    if not np.array_equal(y_true_a, y_true_b):
        raise RuntimeError("Validation label order mismatch between model evaluations.")

    cm_a = confusion_matrix(y_true_a, y_pred_a, labels=list(range(7)), normalize="true")
    cm_b = confusion_matrix(y_true_b, y_pred_b, labels=list(range(7)), normalize="true")

    recall_a = np.diag(cm_a)
    recall_b = np.diag(cm_b)

    confusion_path = output_dir / "confusion_matrix_comparison.png"
    recall_path = output_dir / "recall_bar_comparison.png"
    recall_csv_path = output_dir / "recall_by_class_comparison.csv"

    save_confusion_comparison(
        cm_a,
        cm_b,
        confusion_path,
        f"Original ({original_model.name})",
        f"Optimized ({optimized_model.name})",
    )
    save_recall_bar_compare(
        recall_a,
        recall_b,
        recall_path,
        "Original",
        "Optimized",
    )

    recall_df = pd.DataFrame(
        {
            "Class": CLASS_NAMES,
            "Recall_Original": np.round(recall_a, 4),
            "Recall_Optimized": np.round(recall_b, 4),
            "Delta_Optimized_minus_Original": np.round(recall_b - recall_a, 4),
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    recall_df.to_csv(recall_csv_path, index=False, encoding="utf-8-sig")

    print(f"Device: {device}")
    print(f"Preprocess: {args.preprocess}")
    print(f"Original model: {original_model}")
    print(f"Optimized model: {optimized_model}")
    print(f"Saved figure: {confusion_path}")
    print(f"Saved figure: {recall_path}")
    print(f"Saved csv:    {recall_csv_path}")


if __name__ == "__main__":
    main()
