"""Synthetic training, masking, and quantized export checks before model selection."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import tempfile
import time
import numpy as np
import torch
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from signsearch.model import SignTransformer


def main():
    torch.manual_seed(0)
    torch.set_num_threads(4)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SignTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003)
    times = []
    for _ in range(3):
        x = torch.randn(64,64,101,device=device)
        mask = torch.ones(64,64,dtype=torch.bool,device=device)
        mask[:,48:] = False
        start = time.perf_counter()
        loss = torch.nn.functional.cross_entropy(model(x,mask),torch.arange(64,device=device)%250)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if device == "mps":
            torch.mps.synchronize()
        times.append(time.perf_counter()-start)
    model = model.cpu().eval()
    x = torch.randn(2,64,101)
    mask = torch.ones(2,64,dtype=torch.bool)
    mask[:,48:] = False
    changed = x.clone()
    changed[:,48:] = 1000
    with torch.inference_mode():
        reference = model(x,mask).numpy()
        np.testing.assert_allclose(reference,model(changed,mask).numpy(),atol=1e-6)
    with tempfile.TemporaryDirectory(prefix="signsearch-smoke-") as directory:
        fp32 = str(Path(directory)/"model.onnx")
        int8 = str(Path(directory)/"int8.onnx")
        torch.onnx.export(model,(x[:1],mask[:1]),fp32,input_names=["features","valid"],output_names=["logits"],
                          dynamic_axes={"features":{0:"batch"},"valid":{0:"batch"},"logits":{0:"batch"}},opset_version=17,dynamo=False)
        quantize_dynamic(fp32,int8,weight_type=QuantType.QInt8)
        sessions = {name:ort.InferenceSession(path,providers=["CPUExecutionProvider"]) for name,path in (("fp32",fp32),("int8",int8))}
        outputs = {name:session.run(None,{"features":x.numpy(),"valid":mask.numpy()})[0] for name,session in sessions.items()}
        np.testing.assert_allclose(reference,outputs["fp32"],rtol=1e-4,atol=1e-4)
        assert np.isfinite(outputs["int8"]).all()
    print(json.dumps({"device":device,"batch_training_seconds":times,"mask_invariance":"passed", "onnx_fp32_max_abs_error":float(np.max(np.abs(reference-outputs['fp32']))),"int8_inference":"passed"},indent=2),flush=True)


if __name__ == "__main__":
    main()
