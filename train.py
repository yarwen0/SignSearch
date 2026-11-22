"""Select models and stopping epochs on validation only. Never opens test landmarks."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
from pathlib import Path
import argparse
import csv
import hashlib
import json
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from signsearch.data import LandmarkDataset
from signsearch.model import make_model


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


@torch.inference_mode()
def validation(model, loader, device):
    model.eval()
    correct, count, loss = 0, 0, 0.
    for x, mask, labels in loader:
        x, mask, labels = x.to(device), mask.to(device), labels.to(device)
        logits = model(x, mask)
        correct += int((logits.argmax(-1) == labels).sum().item())
        count += len(labels)
        loss += float(torch.nn.functional.cross_entropy(logits, labels, reduction="sum").item())
    return correct / count, loss / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[.0003, .001])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if (root / "results/test_started.json").exists():
        raise RuntimeError("Test evaluation has started; model selection is frozen")
    torch.set_num_threads(4)
    train = LandmarkDataset(root, "train", training=True)
    val = LandmarkDataset(root, "val")
    (root / "checkpoints").mkdir(exist_ok=True)
    results = root / "results"
    results.mkdir(exist_ok=True)
    config = {**vars(args), "seed": 0, "optimizer": "AdamW", "weight_decay": .01, "label_smoothing": .1,
              "schedule": "CosineAnnealingLR", "selection": "validation top-1 only; strict improvements; earliest tie",
              "input_features": 101, "frames": 64, "classes": 250, "dim": 192, "layers": 4, "heads": 4,
              "dropout": .1, "augmentations": {"scale": [.9, 1.1], "rotation_radians": [-.15, .15], "shift": [-.1,.1], "speed": [.8,1.25], "horizontal_flip_probability": .5}}
    (results / "training_config.json").write_text(json.dumps(config, indent=2) + "\n")
    selected = {}
    for kind in ("baseline", "transformer"):
        best_overall = -1.
        for learning_rate in args.learning_rates:
            seed_all(0)
            generator = torch.Generator().manual_seed(0)
            training_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0)
            val_loader = DataLoader(val, batch_size=args.batch_size, num_workers=0)
            model = make_model(kind).to(args.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            criterion = torch.nn.CrossEntropyLoss(label_smoothing=.1)
            best, stale = -1., 0
            run_id = f"{kind}_lr{learning_rate}"
            checkpoint = root / "checkpoints" / f"{run_id}.pt"
            curve = results / f"{run_id}_curve.csv"
            with curve.open("w", newline="", buffering=1) as f:
                writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "validation_loss", "validation_top1", "learning_rate", "seconds"])
                writer.writeheader()
                for epoch in range(1, args.epochs + 1):
                    train.epoch = epoch
                    started = time.perf_counter()
                    model.train()
                    loss_sum, count = 0., 0
                    for x, mask, labels in training_loader:
                        x, mask, labels = x.to(args.device), mask.to(args.device), labels.to(args.device)
                        optimizer.zero_grad(set_to_none=True)
                        loss = criterion(model(x, mask), labels)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        optimizer.step()
                        loss_sum += float(loss.item()) * len(labels)
                        count += len(labels)
                    top1, val_loss = validation(model, val_loader, args.device)
                    record = {"epoch": epoch, "train_loss": loss_sum / count, "validation_loss": val_loss,
                              "validation_top1": top1, "learning_rate": optimizer.param_groups[0]["lr"], "seconds": time.perf_counter() - started}
                    writer.writerow(record)
                    print(json.dumps({"run": run_id, **record}), flush=True)
                    if top1 > best:
                        best, stale = top1, 0
                        torch.save({"kind": kind, "epoch": epoch, "validation_top1": top1,
                                    "config": config, "learning_rate": learning_rate,
                                    "state_dict": {k:v.detach().cpu() for k,v in model.state_dict().items()}}, checkpoint)
                    else:
                        stale += 1
                    scheduler.step()
                    if stale >= args.patience:
                        break
            if best > best_overall:
                best_overall = best
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                selected[kind] = {"checkpoint": str(checkpoint.relative_to(root)), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                                  "validation_top1": best, "epoch": saved["epoch"], "learning_rate": learning_rate}
            del model
            if args.device == "mps":
                torch.mps.empty_cache()
        (results / "frozen_selection.json").write_text(json.dumps(selected, indent=2) + "\n")
    print(json.dumps(selected, indent=2), flush=True)


if __name__ == "__main__":
    main()
