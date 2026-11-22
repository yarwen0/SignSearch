import torch
from torch import nn
from signsearch.data import FEATURES


class EncoderBlock(nn.Module):
    def __init__(self, dim=192, heads=4, dropout=.1):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.projection = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.feedforward = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, valid):
        batch, frames, dim = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(batch, frames, 3, self.heads, dim // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(-2, -1)) * (dim // self.heads) ** -.5
        scores = scores.masked_fill(~valid[:, None, None, :], -10000.)
        attended = (self.dropout(scores.softmax(dim=-1)) @ v).transpose(1, 2).reshape(batch, frames, dim)
        x = x + self.dropout(self.projection(attended))
        return x + self.dropout(self.feedforward(self.norm2(x)))


class SignTransformer(nn.Module):
    def __init__(self, classes=250, dim=192, layers=4, heads=4, frames=64, dropout=.1):
        super().__init__()
        self.project = nn.Linear(FEATURES, dim)
        self.positions = nn.Parameter(torch.empty(1, frames, dim))
        nn.init.normal_(self.positions, std=.02)
        self.blocks = nn.ModuleList([EncoderBlock(dim, heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, classes)

    def forward(self, features, valid):
        x = self.project(features) + self.positions[:, :features.shape[1]]
        for block in self.blocks:
            x = block(x, valid)
        weights = valid.unsqueeze(-1).to(x.dtype)
        pooled = (self.norm(x) * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
        return self.head(pooled)


class PooledMLP(nn.Module):
    def __init__(self, classes=250):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(FEATURES * 2, 256), nn.GELU(), nn.Dropout(.2), nn.Linear(256, classes))

    def forward(self, features, valid):
        weights = valid.unsqueeze(-1).to(features.dtype)
        count = weights.sum(dim=1).clamp(min=1)
        mean = (features * weights).sum(dim=1) / count
        variance = ((features - mean[:, None]) ** 2 * weights).sum(dim=1) / count
        return self.network(torch.cat([mean, (variance + 1e-6).sqrt()], dim=-1))


def make_model(kind):
    return SignTransformer() if kind == "transformer" else PooledMLP()
