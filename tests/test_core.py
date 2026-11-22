import json
from pathlib import Path
import numpy as np
import pytest
from signsearch.data import normalize, resample, augment
from signsearch.metrics import retrieval_metrics


def test_normalization_is_translation_and_scale_invariant():
    rng = np.random.default_rng(4)
    points = rng.normal(size=(5,49,2)).astype(np.float32)
    masks = np.ones((5,3),bool)
    masks[2,1] = False
    points[2,7:28] = 0
    a = normalize(points,masks)
    b = normalize(points*3+100,masks)
    np.testing.assert_allclose(a,b,atol=2e-5)
    np.testing.assert_array_equal(a[2,14:56],0)
    np.testing.assert_allclose(a[:,2:6].reshape(-1,2,2).mean(1),0,atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(a[:,2:4]-a[:,4:6],axis=-1),1,atol=1e-6)


def test_missing_pose_anchor_interpolation_and_failure():
    points = np.ones((3,49,2),np.float32)
    points[:,1] = [0,0]
    points[:,2] = [2,0]
    masks = np.ones((3,3),bool)
    masks[1,0] = False
    result = normalize(points,masks)
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(result[1,:14],0)
    with pytest.raises(ValueError,match="anchor"):
        normalize(points,np.zeros((3,3),bool))


def test_resampling_and_padding_mask():
    x = np.arange(8,dtype=np.float32).reshape(4,2)
    padded,mask = resample(x,6)
    np.testing.assert_array_equal(padded[:4],x)
    np.testing.assert_array_equal(mask,[True,True,True,True,False,False])
    sampled,mask = resample(x,2)
    np.testing.assert_array_equal(sampled,x[[0,3]])
    assert mask.all()
    with pytest.raises(ValueError):
        resample(x[:0])


def test_mrr_and_signer_summary():
    scores = np.array([[3,2,1],[3,1,2],[3,2,1]])
    labels = np.array([0,2,2])
    result = retrieval_metrics(scores,labels,["A","B","B"])
    assert result["mrr"] == pytest.approx((1+1/2+1/3)/3)
    assert result["top1"] == pytest.approx(1/3)
    assert result["per_signer"]["min"] == 0
    assert result["per_signer"]["max"] == 1


def test_official_splits_have_no_signer_or_video_overlap():
    path = Path(__file__).resolve().parents[1] / "data/selected.json"
    if not path.exists():
        pytest.skip("Run scripts/prepare.py to obtain the official split manifests")
    splits = json.loads(path.read_text())
    for a,b in (("train","val"),("train","test"),("val","test")):
        assert not {r["signer"] for r in splits[a]} & {r["signer"] for r in splits[b]}
        assert not {r["video"] for r in splits[a]} & {r["video"] for r in splits[b]}
    assert all(len({r["label"] for r in rows}) == 250 for rows in splits.values())


def test_augmentation_preserves_absent_groups():
    x = np.zeros((10,101),np.float32)
    x[:,:14] = 1
    x[:,98] = 1
    for seed in range(10):
        output = augment(x,np.random.default_rng(seed))
        np.testing.assert_array_equal(output[:,14:98],0)


def test_transformer_ignores_padded_frames():
    import torch
    from signsearch.model import SignTransformer
    torch.manual_seed(0)
    model = SignTransformer().eval()
    x = torch.randn(2,64,101)
    valid = torch.ones(2,64,dtype=torch.bool)
    valid[:,16:] = False
    changed = x.clone()
    changed[:,16:] = 1000
    with torch.inference_mode():
        torch.testing.assert_close(model(x,valid),model(changed,valid))
