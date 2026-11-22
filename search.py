"""Rank dictionary glosses for a user's isolated sign clip."""
from pathlib import Path
import argparse
import json
import shutil
import uuid
import numpy as np
import onnxruntime as ort
from signsearch.data import resample
from scripts.extract import extract_clip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--precision", choices=["fp32", "int8"], default="int8")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    glosses = (root / "data/glosses_250.txt").read_text().splitlines()
    if not 1 <= args.top_k <= len(glosses):
        parser.error("top-k must be positive and no larger than the dictionary")
    identifier = uuid.uuid4().hex
    row = {"video":args.video.name,"split":"query","video_path":f"data/queries/{identifier}.mp4", "keypoints":f"keypoints/queries/{identifier}.npz"}
    copy = root / row["video_path"]
    copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.video, copy)
    try:
        extract_clip(root,row)
        with np.load(root / row["keypoints"]) as extracted:
            x,valid = resample(extracted["features"])
        session = ort.InferenceSession(str(root / f"models/transformer_{args.precision}.onnx"), providers=["CPUExecutionProvider"])
        logits = session.run(None,{"features":x[None],"valid":valid[None]})[0][0]
        probabilities = np.exp(logits-logits.max())
        probabilities /= probabilities.sum()
        ranked = np.argsort(-logits,kind="stable")[:args.top_k]
        print(json.dumps({"results":[{"rank":rank+1,"gloss":glosses[index],"model_probability":float(probabilities[index])} for rank,index in enumerate(ranked)]},indent=2))
    finally:
        # Delete this command's temporary copy, never the user's original file.
        copy.unlink(missing_ok=True)
        (root / row["keypoints"]).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
