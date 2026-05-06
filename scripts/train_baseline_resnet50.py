import argparse
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
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from med_model_arch import create_model


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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(metadata_csv, images_dir, seed, batch_size, num_workers):
    df = pd.read_csv(metadata_csv)
    df["label_idx"] = df["dx"].map(DISEASE_CLASSES)
    df = df[df["label_idx"].notna()].copy()

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=seed,
        stratify=df["label_idx"],
    )

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = HAMImageDataset(train_df, images_dir, train_transform)
    val_dataset = HAMImageDataset(val_df, images_dir, val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def evaluate(model, val_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    acc = 100.0 * correct / max(total, 1)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return acc, precision, recall, f1


def save_checkpoint(state_dict, out_dir, best_acc):
    out_dir.mkdir(parents=True, exist_ok=True)
    acc_tag = f"{best_acc:.2f}".replace(".", "_")
    versioned = out_dir / f"baseline_resnet50_best_acc{acc_tag}.pth"
    stable = out_dir / "baseline_resnet50_best.pth"
    torch.save(state_dict, versioned)
    torch.save(state_dict, stable)
    return versioned, stable


def main():
    parser = argparse.ArgumentParser(description="Train Baseline ResNet50 with standard cross-entropy loss.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT_FOR_IMPORT)
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    metadata_csv = args.metadata_csv or (project_root / "dataset" / "HAM10000_metadata.csv")
    images_dir = args.images_dir or (project_root / "dataset" / "images")
    output_dir = args.output_dir or (project_root / "outputs" / "checkpoints")

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

    set_seed(args.seed)

    train_loader, val_loader = build_dataloaders(
        metadata_csv=metadata_csv,
        images_dir=images_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = create_model("baseline_resnet50", num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_acc = -1.0
    best_state = None

    print(f"Device: {device}")
    print("Training model: BaselineResNet50 + CrossEntropyLoss")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(len(train_loader), 1)
        val_acc, val_precision, val_recall, val_f1 = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch + 1:02d}/{args.epochs} | "
            f"train_loss={avg_loss:.4f} | "
            f"val_acc={val_acc:.2f}% | "
            f"precision={val_precision:.4f} | "
            f"recall={val_recall:.4f} | "
            f"f1={val_f1:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training finished but no best checkpoint was captured.")

    versioned_path, stable_path = save_checkpoint(best_state, output_dir, best_acc)
    print(f"\nBest val accuracy: {best_acc:.2f}%")
    print(f"Saved checkpoint: {versioned_path}")
    print(f"Updated checkpoint: {stable_path}")


if __name__ == "__main__":
    main()
