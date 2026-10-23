"""Train the Space-Time Encoder on CHELSA.

  python train.py --numpy-dir /path/to/chelsa_numpy --in-memory
"""
import argparse
import os

import torch
import torch.nn.functional as F

from ste.chelsa import chelsa_loaders
from ste.model import SpaceTimeEncoder, orthonormality_penalty

LAMBDA_ORTH = 0.01


def evaluate(model, val_loader, device):
    model.eval()
    se, n = 0.0, 0
    with torch.no_grad():
        for latlon, month, target in val_loader:
            latlon, month, target = latlon.to(device), month.to(device), target.to(device)
            pred = model(latlon, month)
            se += F.mse_loss(pred, target, reduction="sum").item()
            n += target.numel()
    model.train()
    return se / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--numpy-dir", required=True, help="output dir of scripts/chelsa_to_numpy.py")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--in-memory", action="store_true", help="load all 12 months into RAM (~60 GB)")
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--early-stop-patience", type=int, default=15)
    ap.add_argument("--val-check-interval", type=int, default=2000)
    ap.add_argument("--out", default="checkpoints/ste.pt")
    a = ap.parse_args()
    out = a.out

    train_loader, val_loader = chelsa_loaders(a.numpy_dir, batch_size=8192, num_workers=8,
                                              val_frac=0.02, split_seed=42, in_memory=a.in_memory)

    model = SpaceTimeEncoder().to(a.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SpaceTimeEncoder params: {n_params} ({n_params/1e6:.2f}M) -> {out}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    best_val, bad_checks, step = float("inf"), 0, 0
    model.train()
    for epoch in range(a.max_epochs):
        for latlon, month, target in train_loader:
            latlon, month, target = latlon.to(a.device), month.to(a.device), target.to(a.device)
            opt.zero_grad()
            emb, pred = model.forward_with_embedding(latlon, month)
            loss = F.mse_loss(pred, target) + LAMBDA_ORTH * orthonormality_penalty(emb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1

            if step % a.val_check_interval == 0:
                val_mse = evaluate(model, val_loader, a.device)
                sched.step(val_mse)
                print(f"step {step} epoch {epoch} train_loss {loss.item():.4f} val_mse {val_mse:.4f}",
                      flush=True)
                if val_mse < best_val:
                    best_val, bad_checks = val_mse, 0
                    torch.save(model.state_dict(), out)
                    print(f"  -> saved best (val_mse={best_val:.4f}) to {out}", flush=True)
                else:
                    bad_checks += 1
                    if bad_checks >= a.early_stop_patience:
                        print(f"early stopping at step {step} "
                              f"(no val improvement in {a.early_stop_patience} checks)", flush=True)
                        return
    print("max_epochs reached, done", flush=True)


if __name__ == "__main__":
    main()
