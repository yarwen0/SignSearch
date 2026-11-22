"""Export frozen validation-selected transformer and measure batch-one CPU latency."""
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import time
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import torch
from signsearch.model import make_model
from signsearch.data import FEATURES


def export(root):
    selection = json.loads((root / "results/frozen_selection.json").read_text())["transformer"]
    checkpoint = root / selection["checkpoint"]
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != selection["sha256"]:
        raise RuntimeError("Frozen checkpoint changed")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = make_model("transformer").eval()
    model.load_state_dict(saved["state_dict"])
    destination = root / "models/transformer_fp32.onnx"
    torch.onnx.export(model, (torch.zeros(1, 64, FEATURES), torch.ones(1, 64, dtype=torch.bool)), str(destination),
                      input_names=["features", "valid"], output_names=["logits"],
                      dynamic_axes={"features": {0:"batch"}, "valid": {0:"batch"}, "logits": {0:"batch"}},
                      opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(destination))
    quantized = root / "models/transformer_int8.onnx"
    quantize_dynamic(str(destination), str(quantized), weight_type=QuantType.QInt8)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    sessions = {"fp32": ort.InferenceSession(str(destination), sess_options=options, providers=["CPUExecutionProvider"]),
                "int8": ort.InferenceSession(str(quantized), sess_options=options, providers=["CPUExecutionProvider"])}
    rng = np.random.default_rng(0)
    x = rng.normal(size=(1,64,FEATURES)).astype(np.float32)
    mask = np.ones((1,64), bool)
    # Numerical parity uses synthetic input and validation only, never test feedback.
    with torch.inference_mode():
        reference = model(torch.from_numpy(x), torch.from_numpy(mask)).numpy()
    fp32 = sessions["fp32"].run(None, {"features":x,"valid":mask})[0]
    np.testing.assert_allclose(fp32, reference, rtol=1e-4, atol=1e-4)
    latencies = {}
    for name, session in sessions.items():
        feed = {"features": x, "valid": mask}
        for _ in range(50):
            session.run(None, feed)
        elapsed = []
        for _ in range(500):
            start = time.perf_counter_ns()
            session.run(None, feed)
            elapsed.append((time.perf_counter_ns() - start) / 1e6)
        np.save(root / f"results/{name}_latency_ms.npy", np.asarray(elapsed))
        latencies[name] = {"median_ms": float(np.median(elapsed)), "p95_ms": float(np.quantile(elapsed,.95)),
                           "runs": len(elapsed), "warmup_runs": 50, "batch":1, "frames":64, "intra_op_threads":4, "inter_op_threads":1}
    try:
        machine = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        memory = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        machine, memory = platform.processor(), None
    result = {"cpu_latency":latencies,"machine":{"cpu":machine,"memory_bytes":memory,"platform":platform.platform(),
                "python":platform.python_version(),"onnxruntime":ort.__version__,"torch":torch.__version__},
              "export": {"fp32_sha256":hashlib.sha256(destination.read_bytes()).hexdigest(),"int8_sha256":hashlib.sha256(quantized.read_bytes()).hexdigest(),
                         "synthetic_fp32_max_abs_error":float(np.max(np.abs(reference-fp32)))}}
    (root / "results/export_metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    return sessions, result


if __name__ == "__main__":
    _, result = export(Path(__file__).resolve().parent)
    print(json.dumps(result, indent=2))
