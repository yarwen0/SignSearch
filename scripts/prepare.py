"""Official signer-disjoint subset, bounded downloads, and eight extraction processes."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
import csv
import io
import json
import multiprocessing as multiprocessing
import random
import threading
import time
import traceback
import urllib.request
from remotezip import RemoteZip
from scripts.extract import extract_clip

URL = "https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip"
MODELS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}


def select(root):
    data = root / "data"
    data.mkdir(exist_ok=True)
    with RemoteZip(URL, timeout=120) as archive:
        names = archive.namelist()
        (data / "zip_members.json").write_text(json.dumps(names))
        splits = {}
        for split in ("train", "val", "test"):
            member = next(n for n in names if n.endswith(f"/splits/{split}.csv"))
            payload = archive.read(member)
            (data / f"{split}.csv").write_bytes(payload)
            reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
            print(json.dumps({"split": split, "columns": reader.fieldnames}), flush=True)
            splits[split] = list(reader)
        license_file = next(n for n in names if n.endswith("/use.txt"))
        (data / "use.txt").write_bytes(archive.read(license_file))
    common = set.intersection(*(set(r["Gloss"] for r in rows) for rows in splits.values()))
    glosses = sorted(random.Random(0).sample(sorted(common), 250))
    (data / "glosses_250.txt").write_text("\n".join(glosses) + "\n")
    labels = {g: i for i, g in enumerate(glosses)}
    video_members = {Path(n).name: n for n in names if n.lower().endswith(".mp4")}
    selected = {}
    for split, rows in splits.items():
        selected[split] = [
            {"video": r["Video file"], "signer": r["Participant ID"], "gloss": r["Gloss"],
             "label": labels[r["Gloss"]], "split": split, "member": video_members[r["Video file"]],
             "video_path": f"data/videos/{split}/{r['Video file']}",
             "keypoints": f"keypoints/{split}/{Path(r['Video file']).stem}.npz"}
            for r in rows if r["Gloss"] in labels]
    signers = {s: set(r["signer"] for r in rows) for s, rows in selected.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if signers[a] & signers[b]:
            raise RuntimeError(f"Signer leakage: {a}, {b}")
    (data / "selected.json").write_text(json.dumps(selected, indent=2))
    summary = {"seed": 0, "classes": len(glosses), "selection": "random.Random(0).sample(sorted(common_glosses), 250)",
               "splits": {s: {"clips": len(rows), "signers": len(signers[s])} for s, rows in selected.items()}}
    (root / "results").mkdir(exist_ok=True)
    (root / "results/dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--download-threads", type=int, default=12)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root = args.root.resolve()
    (root / "models").mkdir(exist_ok=True)
    for name, url in MODELS.items():
        target = root / "models" / name
        if not target.exists():
            print(f"Downloading MediaPipe model {name}", flush=True)
            urllib.request.urlretrieve(url, target.with_suffix(".partial"))
            target.with_suffix(".partial").replace(target)
    selected_file = root / "data/selected.json"
    selected = json.loads(selected_file.read_text()) if selected_file.exists() else select(root)
    rows = [r for rows in selected.values() for r in rows]
    pending = [r for r in rows if not (root / r["keypoints"]).exists()]
    print(json.dumps({"selected": len(rows), "remaining": len(pending), "split_counts": {s:len(v) for s,v in selected.items()}}), flush=True)
    capacity = threading.Semaphore(args.workers * 3)
    local = threading.local()
    archives, lock = [], threading.Lock()

    def download(row):
        capacity.acquire()
        try:
            target = root / row["video_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                return row
            for attempt in range(5):
                try:
                    if not hasattr(local, "archive"):
                        local.archive = RemoteZip(URL, timeout=120)
                        with lock:
                            archives.append(local.archive)
                    payload = local.archive.read(row["member"])
                    temporary = target.with_suffix(".partial")
                    temporary.write_bytes(payload)
                    temporary.replace(target)
                    return row
                except Exception:
                    if hasattr(local, "archive"):
                        local.archive.close()
                        del local.archive
                    if attempt == 4:
                        raise
                    time.sleep(2 ** attempt)
        except Exception:
            capacity.release()
            raise

    completed, errors = len(rows) - len(pending), 0
    started = time.monotonic()
    with (root / "data/extraction.jsonl").open("a", buffering=1) as log, (root / "data/failures.jsonl").open("a", buffering=1) as failures:
        def finish(future, row):
            nonlocal completed, errors
            try:
                log.write(json.dumps(future.result()) + "\n")
                completed += 1
                if completed % 25 == 0 or completed == len(rows):
                    print(json.dumps({"extracted": completed, "total": len(rows), "failures": errors, "elapsed_seconds": time.monotonic() - started}), flush=True)
            except Exception as exc:
                errors += 1
                failures.write(json.dumps({"video": row["video"], "split": row["split"], "error": str(exc), "traceback": traceback.format_exc()}) + "\n")
                print(f"FAILED {row['video']}: {exc}", flush=True)
            finally:
                capacity.release()

        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as extraction, ThreadPoolExecutor(max_workers=args.download_threads) as downloads:
            futures = {downloads.submit(download, row): row for row in pending}
            for future in as_completed(futures):
                row = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    errors += 1
                    failures.write(json.dumps({"video": row["video"], "split": row["split"], "error": str(exc)}) + "\n")
                    continue
                task = extraction.submit(extract_clip, str(root), row)
                task.add_done_callback(lambda f, r=row: finish(f, r))
    for archive in archives:
        archive.close()
    missing = [r["video"] for r in rows if not (root / r["keypoints"]).exists()]
    if missing:
        raise RuntimeError(f"Incomplete extraction: {len(missing)} clips; inspect data/failures.jsonl and rerun")
    print("All selected clips extracted; downloaded videos deleted.", flush=True)


if __name__ == "__main__":
    main()
