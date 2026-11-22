"""One frozen final evaluation, caching raw model outputs for exact reproduction."""
from pathlib import Path
import hashlib
import json
import sys
from datetime import datetime, timezone
import numpy as np
import torch
from torch.utils.data import DataLoader
from signsearch.data import LandmarkDataset
from signsearch.model import make_model
from signsearch.metrics import retrieval_metrics
from export import export


@torch.inference_mode()
def predict(model, dataset):
    model.eval()
    return np.concatenate([model(x,mask).numpy() for x,mask,_ in DataLoader(dataset,batch_size=64)])


def main():
    root = Path(__file__).resolve().parent
    results = root / "results"
    if (results / "metrics.json").exists():
        print((results / "metrics.json").read_text())
        return
    selection = json.loads((results / "frozen_selection.json").read_text())
    if set(selection) != {"baseline", "transformer"}:
        raise RuntimeError("Finish validation selection first")
    torch.set_num_threads(4)
    # Export errors must be resolved before crossing the test boundary.
    sessions, export_result = export(root)
    marker = results / "test_started.json"
    if marker.exists():
        raise RuntimeError("Test run already started. Reconstruct from cached outputs; do not rerun test inference.")
    marker.write_text(json.dumps({"started_at":datetime.now(timezone.utc).isoformat(),"selection":selection},indent=2)+"\n")
    report = {"dataset":json.loads((results / "dataset_summary.json").read_text()),
              "training":json.loads((results / "training_config.json").read_text()),"selection":selection,**export_result,"models":{}}
    # Compute all model outputs in this one final pass, without selecting on test.
    for split in ("val","test"):
        data = LandmarkDataset(root, split)
        labels = np.asarray([r["label"] for r in data.rows])
        signers = [r["signer"] for r in data.rows]
        for kind in ("baseline","transformer"):
            checkpoint = root / selection[kind]["checkpoint"]
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != selection[kind]["sha256"]:
                raise RuntimeError("Frozen checkpoint changed")
            model = make_model(kind)
            model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=False)["state_dict"])
            logits = predict(model,data)
            np.savez_compressed(results / f"raw_{kind}_{split}.npz",logits=logits,labels=labels,signers=np.asarray(signers))
            report["models"].setdefault(kind,{})[split] = retrieval_metrics(logits,labels,signers)
        for name,session in sessions.items():
            logits = np.concatenate([session.run(None,{"features":x.numpy(),"valid":mask.numpy()})[0]
                                     for x,mask,_ in DataLoader(data,batch_size=64)])
            np.savez_compressed(results / f"raw_onnx_{name}_{split}.npz",logits=logits,labels=labels,signers=np.asarray(signers))
            report["models"].setdefault(f"onnx_{name}",{})[split] = retrieval_metrics(logits,labels,signers)
    all_ms = []
    for split,rows in json.loads((root / "data/selected.json").read_text()).items():
        for row in rows:
            with np.load(root / row["keypoints"]) as data:
                all_ms.append(data["frame_ms"])
    timings = np.concatenate(all_ms)
    np.save(results / "extraction_frame_ms.npy",timings)
    report["keypoint_extraction"] = {"median_ms_per_frame":float(np.median(timings)),"p95_ms_per_frame":float(np.quantile(timings,.95)),
                                   "mean_ms_per_frame":float(timings.mean()),"frames":len(timings),"workers":8,
                                   "scope":"HandLandmarker and PoseLandmarker detect_for_video calls, CPU, concurrent extraction; excludes model startup, video decoding and file IO"}
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    (results / "metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
