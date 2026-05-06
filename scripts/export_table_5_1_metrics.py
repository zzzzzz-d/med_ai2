import argparse
import sys
from pathlib import Path

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parent
if PROJECT_ROOT_FOR_IMPORT.name == "scripts":
    PROJECT_ROOT_FOR_IMPORT = PROJECT_ROOT_FOR_IMPORT.parent
sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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


def parse_model_arg(model_arg):
    if "=" not in model_arg:
        raise ValueError(f"Invalid --model value: {model_arg}. Use format name=path")
    name, path_str = model_arg.split("=", 1)
    name = name.strip()
    path = Path(path_str.strip())
    if not name:
        raise ValueError(f"Invalid model name in --model value: {model_arg}")
    return name, path


def parse_model_arch_arg(model_arch_arg):
    if "=" not in model_arch_arg:
        raise ValueError(f"Invalid --model-arch value: {model_arch_arg}. Use format name=arch")
    name, arch = model_arch_arg.split("=", 1)
    name = name.strip()
    arch = arch.strip()
    if not name or not arch:
        raise ValueError(f"Invalid --model-arch value: {model_arch_arg}")
    return name, arch


def default_model_list(project_root):
    checkpoint_dir = project_root / "outputs" / "checkpoints"
    candidates = [
        ("baseline", checkpoint_dir / "baseline_resnet50_best.pth", "baseline_resnet50"),
        ("baseline+eca", checkpoint_dir / "eca_resnet50_epoch3.pth", "eca_resnet50"),
        ("ours", checkpoint_dir / "eca_resnet50_v8_best_acc87_97.pth", "eca_resnet50"),
    ]
    return [(n, p, a) for n, p, a in candidates if p.exists()]


def load_val_loader(metadata_csv, images_dir, seed, batch_size, preprocess_mode):
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
        val_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float32),
            ]
        )
    val_dataset = HAMImageDataset(val_df, images_dir, val_transform)
    return DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


def evaluate_model(weight_path, arch_name, val_loader, device):
    model = create_model(arch_name, num_classes=7).to(device)
    payload = torch.load(str(weight_path), map_location=device)
    load_info = load_state_dict_compatible(model, payload, strict=False)
    if load_info["missing_keys"] or load_info["unexpected_keys"]:
        raise RuntimeError(
            f"Checkpoint incompatible: {weight_path}\n"
            f"arch={arch_name}\n"
            f"missing_keys={load_info['missing_keys'][:10]}\n"
            f"unexpected_keys={load_info['unexpected_keys'][:10]}"
        )

    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.tolist())

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def build_model_specs(project_root, model_args, model_arch_args, default_arch):
    arch_override = dict(parse_model_arch_arg(item) for item in model_arch_args)

    if model_args:
        specs = []
        for item in model_args:
            name, path = parse_model_arg(item)
            arch = arch_override.get(name, default_arch)
            specs.append((name, path, arch))
        return specs

    default_specs = default_model_list(project_root)
    adjusted = []
    for name, path, arch in default_specs:
        adjusted.append((name, path, arch_override.get(name, arch)))
    return adjusted


def main():
    parser = argparse.ArgumentParser(description="Export Table 5-1 metrics for Baseline / ECA / Ours models.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT_FOR_IMPORT)
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
        help="notebook matches 01_data_process.ipynb; imagenet_norm uses ToTensor+Normalize.",
    )
    parser.add_argument("--default-arch", type=str, default="eca_resnet50")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model row in format name=path. Repeat this arg for each row.",
    )
    parser.add_argument(
        "--model-arch",
        action="append",
        default=[],
        help="Optional arch mapping in format name=arch, e.g. baseline=baseline_resnet50",
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    metadata_csv = args.metadata_csv or (project_root / "dataset" / "HAM10000_metadata.csv")
    images_dir = args.images_dir or (project_root / "dataset" / "images")
    output_csv = args.output_csv or (project_root / "outputs" / "metrics" / "table_5_1_metrics.csv")

    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    models = build_model_specs(project_root, args.model, args.model_arch, args.default_arch)
    if len(models) == 0:
        raise RuntimeError(
            "No models to evaluate. Use --model name=path, or place default comparison weights in outputs/checkpoints."
        )

    val_loader = load_val_loader(metadata_csv, images_dir, args.seed, args.batch_size, args.preprocess)

    rows = []
    for model_name, model_path, model_arch in models:
        if not model_path.exists():
            raise FileNotFoundError(f"Model weight not found: {model_path}")
        metrics = evaluate_model(model_path, model_arch, val_loader, device)
        rows.append(
            {
                "Model": model_name,
                "Arch": model_arch,
                "Accuracy": round(metrics["Accuracy"], 4),
                "Precision": round(metrics["Precision"], 4),
                "Recall": round(metrics["Recall"], 4),
                "F1": round(metrics["F1"], 4),
                "WeightPath": str(model_path.resolve()),
            }
        )

    result_df = pd.DataFrame(rows, columns=["Model", "Arch", "Accuracy", "Precision", "Recall", "F1", "WeightPath"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(result_df[["Model", "Arch", "Accuracy", "Precision", "Recall", "F1"]].to_string(index=False))
    print(f"\nSaved: {output_csv}")


if __name__ == "__main__":
    main()

