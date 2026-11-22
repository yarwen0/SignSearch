from pathlib import Path
import json
import numpy as np

POSE = (0, 11, 12, 13, 14, 15, 16)
N_POINTS = 49
FEATURES = N_POINTS * 2 + 3


def normalize(points, present):
    """Center and scale in pixel coordinates; interpolate missing pose anchors.

    points: [T,49,2], present: [T,3] (pose, left hand, right hand).
    Missing groups remain exact zeros after normalization.
    """
    points = np.asarray(points, dtype=np.float32)
    present = np.asarray(present, dtype=bool)
    shoulders = points[:, 1:3]
    widths = np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=-1)
    good = present[:, 0] & np.isfinite(widths) & (widths > 1e-6)
    if not good.any():
        raise ValueError("No valid shoulder anchor in clip")
    centers = shoulders.mean(axis=1)
    t = np.arange(len(points))
    center = np.stack([np.interp(t, t[good], centers[good, axis]) for axis in (0, 1)], -1)
    scale = np.interp(t, t[good], widths[good])
    normalized = (points - center[:, None]) / scale[:, None, None]
    valid_points = np.repeat(present, (7, 21, 21), axis=1)
    normalized[~valid_points] = 0
    if not np.isfinite(normalized).all():
        raise ValueError("Non-finite landmarks")
    return np.concatenate([normalized.reshape(len(points), -1), present.astype(np.float32)], axis=-1)


def resample(features, frames=64):
    """Subsample long clips and pad short ones, preserving a pooling mask."""
    if len(features) == 0:
        raise ValueError("Cannot resample an empty clip")
    if len(features) >= frames:
        indices = np.linspace(0, len(features) - 1, frames).round().astype(int)
        return features[indices].astype(np.float32), np.ones(frames, dtype=bool)
    out = np.zeros((frames, features.shape[1]), np.float32)
    mask = np.zeros(frames, dtype=bool)
    out[:len(features)] = features
    mask[:len(features)] = True
    return out, mask


def augment(features, rng):
    features = features.astype(np.float32).copy()
    # Speed jitter changes the temporal sampling before fixed-length padding.
    speed = rng.uniform(.8, 1.25)
    count = max(1, int(len(features) / speed))
    features = features[np.linspace(0, len(features) - 1, count).round().astype(int)]
    points = features[:, :98].reshape(-1, 49, 2)
    masks = features[:, 98:]
    if rng.random() < .5:
        points[:, :, 0] *= -1
        points[:] = points[:, [0, 2, 1, 4, 3, 6, 5, *range(28, 49), *range(7, 28)]]
        masks[:, [1, 2]] = masks[:, [2, 1]]
    angle = rng.uniform(-.15, .15)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32)
    points[:] = points @ rotation * rng.uniform(.9, 1.1) + rng.uniform(-.1, .1, (1, 1, 2))
    present = np.repeat(masks.astype(bool), (7, 21, 21), axis=1)
    points[~present] = 0
    return features


class LandmarkDataset:
    def __init__(self, root, split, training=False, seed=0):
        self.root = Path(root)
        self.rows = json.loads((self.root / "data/selected.json").read_text())[split]
        self.training = training
        self.seed = seed
        self.epoch = 0
        # These compact arrays fit in memory and avoid thousands of ZIP reads per epoch.
        self.arrays = [np.load(self.root / r["keypoints"])["features"].astype(np.float32) for r in self.rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        x = self.arrays[index]
        if self.training:
            x = augment(x, np.random.default_rng(self.seed + self.epoch * len(self) + index))
        x, mask = resample(x)
        return x, mask, self.rows[index]["label"]
