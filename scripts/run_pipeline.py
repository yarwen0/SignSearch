"""Finish the experiment, starting training once train and validation extraction finish."""
from pathlib import Path
import json
import subprocess
import sys
import time


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = root / "data/selected.json"
    while not manifest.exists():
        time.sleep(5)
    splits = json.loads(manifest.read_text())
    required = [row for split in ("train", "val") for row in splits[split]]
    while True:
        missing = sum(not (root / row["keypoints"]).exists() for row in required)
        print(json.dumps({"waiting_for_train_val_clips": missing}), flush=True)
        if not missing:
            break
        time.sleep(30)
    if not (root / "results/frozen_selection.json").exists():
        with (root / "results/train.log").open("w", buffering=1) as log:
            subprocess.run([sys.executable, "-u", "train.py"], cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
    all_rows = [r for rows in splits.values() for r in rows]
    while any(not (root / row["keypoints"]).exists() for row in all_rows):
        print("Waiting for remaining test extraction", flush=True)
        time.sleep(30)
    with (root / "results/evaluate.log").open("w", buffering=1) as log:
        subprocess.run([sys.executable,"-u","evaluate.py"],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run([sys.executable,"scripts/update_readme.py"],cwd=root,check=True)
    print((root / "results/metrics.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
