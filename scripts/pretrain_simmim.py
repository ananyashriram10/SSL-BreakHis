from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ssl_breakhis.data import BreakHisDataset, patient_level_split, scan_breakhis, seed_everything
from ssl_breakhis.models import SimMIMResNet, masked_l1_loss
from ssl_breakhis.transforms import simmim_transform


def parse_args():
    parser = argparse.ArgumentParser(description="Patient-safe SimMIM pretraining on BreakHis.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", default="checkpoints/best_simmim_resnet18.pth")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mask-ratio", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def run_epoch(model, loader, optimizer, device):
    training = optimizer is not None
    model.train(training)
    total_loss, total_batches = 0.0, 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            pred, target, mask = model(images)
            loss = masked_l1_loss(pred, target, mask)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += loss.item()
        total_batches += 1
    return total_loss / max(total_batches, 1)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = scan_breakhis(args.data_root)
    train_samples, val_samples, _ = patient_level_split(samples, task="subtype", seed=args.seed)
    train_ds = BreakHisDataset(train_samples, transform=simmim_transform(), task="ssl")
    val_ds = BreakHisDataset(val_samples, transform=simmim_transform(), task="ssl")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    model = SimMIMResNet(mask_ratio=args.mask_ratio).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device)
        val_loss = run_epoch(model, val_loader, None, device)
        scheduler.step()
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, out_path)
        print(f"epoch={epoch:03d} train_loss={train_loss:.5f} val_loss={val_loss:.5f} best_val={best_val:.5f}")


if __name__ == "__main__":
    main()
