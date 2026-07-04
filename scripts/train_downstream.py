from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ssl_breakhis.data import (
    INDEX_TO_BINARY,
    INDEX_TO_SUBTYPE,
    BreakHisDataset,
    class_weights,
    patient_level_split,
    scan_breakhis,
    seed_everything,
)
from ssl_breakhis.models import ResNetClassifier, load_byol_encoder
from ssl_breakhis.transforms import downstream_train_transform, eval_transform


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate BreakHis classifier.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", default=None, help="Optional BYOL encoder checkpoint.")
    parser.add_argument("--out", default="checkpoints/best_downstream.pth")
    parser.add_argument("--task", choices=["subtype", "binary"], default="subtype")
    parser.add_argument("--magnification", nargs="*", default=None, help="Optional filters: 40X 100X 200X 400X")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pretrained-imagenet", action="store_true")
    return parser.parse_args()


def run_epoch(model, loader, criterion, optimizer, device):
    training = optimizer is not None
    model.train(training)
    total_loss, preds, labels = 0.0, [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * images.size(0)
        preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
        labels.extend(targets.detach().cpu().tolist())
    avg_loss = total_loss / max(len(loader.dataset), 1)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    accuracy = sum(int(p == y) for p, y in zip(preds, labels)) / max(len(labels), 1)
    return avg_loss, accuracy, macro_f1, labels, preds


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = scan_breakhis(args.data_root, magnifications=args.magnification)
    train_samples, val_samples, test_samples = patient_level_split(samples, task=args.task, seed=args.seed)
    train_ds = BreakHisDataset(train_samples, downstream_train_transform(), task=args.task)
    val_ds = BreakHisDataset(val_samples, eval_transform(), task=args.task)
    test_ds = BreakHisDataset(test_samples, eval_transform(), task=args.task)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    num_classes = 8 if args.task == "subtype" else 2
    model = ResNetClassifier(num_classes=num_classes, pretrained=args.pretrained_imagenet).to(device)
    if args.checkpoint:
        load_byol_encoder(model, args.checkpoint, map_location=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights(train_samples, args.task, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = -1.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1, _, _ = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _, _ = run_epoch(model, val_loader, criterion, None, device)
        scheduler.step()
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_f1": val_f1}, out_path)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} best_val_f1={best_f1:.4f}"
        )

    best = torch.load(out_path, map_location=device)
    model.load_state_dict(best["model"])
    test_loss, test_acc, test_f1, y_true, y_pred = run_epoch(model, test_loader, criterion, None, device)
    names = INDEX_TO_SUBTYPE if args.task == "subtype" else INDEX_TO_BINARY
    target_names = [names[i] for i in range(num_classes)]
    print(f"\nTEST loss={test_loss:.4f} acc={test_acc:.4f} macro_f1={test_f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
