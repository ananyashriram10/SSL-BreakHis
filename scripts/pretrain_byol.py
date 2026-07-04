from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ssl_breakhis.data import BreakHisDataset, patient_level_split, scan_breakhis, seed_everything
from ssl_breakhis.models import BYOLModel, ResNetEncoder, byol_loss, load_simmim_encoder
from ssl_breakhis.transforms import TwoCropsTransform


def parse_args():
    parser = argparse.ArgumentParser(description="BYOL continuation pretraining on BreakHis.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--simmim-checkpoint", default=None)
    parser.add_argument("--out", default="checkpoints/best_byol_resnet18.pth")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def cosine_momentum(epoch: int, max_epochs: int, base: float = 0.996) -> float:
    return 1.0 - (1.0 - base) * (math.cos(math.pi * epoch / max_epochs) + 1.0) / 2.0


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = scan_breakhis(args.data_root)
    train_samples, _, _ = patient_level_split(samples, task="subtype", seed=args.seed)
    train_ds = BreakHisDataset(train_samples, transform=TwoCropsTransform(), task="ssl")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True
    )

    encoder = ResNetEncoder(include_pool=True)
    if args.simmim_checkpoint:
        load_simmim_encoder(encoder, args.simmim_checkpoint, map_location=device)
    model = BYOLModel(encoder).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float("inf")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_batches = 0.0, 0
        momentum = cosine_momentum(epoch - 1, args.epochs)
        for (view_0, view_1), _ in train_loader:
            view_0 = view_0.to(device, non_blocking=True)
            view_1 = view_1.to(device, non_blocking=True)

            pred_0 = model.forward_online(view_0)
            pred_1 = model.forward_online(view_1)
            target_0 = model.forward_target(view_0)
            target_1 = model.forward_target(view_1)
            loss = 0.5 * (byol_loss(pred_0, target_1) + byol_loss(pred_1, target_0))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model.update_target(momentum)
            total_loss += loss.item()
            total_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(total_batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "loss": avg_loss}, out_path)
        print(f"epoch={epoch:03d} byol_loss={avg_loss:.5f} best_loss={best_loss:.5f} momentum={momentum:.5f}")


if __name__ == "__main__":
    main()
