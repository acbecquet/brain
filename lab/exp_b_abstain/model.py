"""A tiny transformer that answers a 3-token question, can say "I don't know", and
reports a confidence.

Output: logits over M answers + 1 extra IDK option, and a confidence logit from a
separate head (used only by the Brier-calibrated regime).
"""
import torch
from torch import nn


class QAModel(nn.Module):
    def __init__(self, m: int, d: int = 128, layers: int = 2, heads: int = 4):
        super().__init__()
        self.m = m
        self.tok = nn.Embedding(m + 2, d)   # values 0..m-1 for a/b, plus two task tokens
        self.pos = nn.Parameter(torch.randn(3, d) * 0.02)
        block = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.answer = nn.Linear(d, m + 1)    # index m = IDK
        self.conf = nn.Linear(d, 1)

    @property
    def idk(self) -> int:
        return self.m

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # task token ids live after the value ids so they don't collide
        tokens = torch.stack([x[:, 0] + self.m, x[:, 1], x[:, 2]], dim=1)
        h = self.norm(self.body(self.tok(tokens) + self.pos))
        pooled = h[:, -1]
        return self.answer(pooled), self.conf(pooled).squeeze(-1)
