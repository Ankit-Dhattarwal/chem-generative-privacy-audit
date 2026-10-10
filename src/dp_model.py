"""Opacus-compatible causal GPT used only for the DP-SGD arm."""
import math
import torch
from torch import nn

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head=4, dropout=0.1):
        super().__init__()
        if d_model % n_head:
            raise ValueError('d_model must be divisible by n_head')
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, length, width = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
        weights = self.attn_dropout(torch.softmax(scores, dim=-1))
        out = (weights @ v).transpose(1, 2).contiguous().view(batch, length, width)
        return self.output_dropout(self.proj(out))

class DPTransformerBlock(nn.Module):
    def __init__(self, d_model=128, n_head=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head=n_head, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, 4 * d_model)
        self.linear2 = nn.Linear(4 * d_model, d_model)
        self.activation_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        feed_forward = self.linear1(self.norm2(x))
        feed_forward = self.activation_dropout(torch.relu(feed_forward))
        return x + self.output_dropout(self.linear2(feed_forward))

class DPCausalGPT(nn.Module):
    def __init__(self, vocab, n_layer=2, d_model=128, n_head=4, context=128, dropout=0.1):
        super().__init__()
        self.context = context
        self.emb = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(context, d_model)
        self.blocks = nn.ModuleList([
            DPTransformerBlock(d_model=d_model, n_head=n_head, dropout=dropout)
            for _ in range(n_layer)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, x):
        batch, length = x.shape
        positions = torch.arange(length, device=x.device).unsqueeze(0).expand(batch, length)
        h = self.emb(x) + self.pos(positions)
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h))