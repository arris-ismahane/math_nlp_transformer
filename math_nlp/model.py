"""
This file defines the standard nanoGPT-style transformer used for training, 
plus a thin Model wrapper that run.py calls to load a trained checkpoint 
(chosen via an environment variable rather than a CLI flag, 
so other scripts can point at different checkpoints without touching run.py) 
and generate predictions.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from random import Random

import torch
import torch.nn as nn
from torch.nn import functional as F

from .tokenizer import Tokenizer


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    pad_id: int = 0


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight  # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        n = sum(p.numel() for p in self.parameters())
        n -= self.pos_emb.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.config.block_size, f"sequence length {T} > block_size {self.config.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                ignore_index=self.config.pad_id,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, eos_id: int | None = None) -> torch.Tensor:
        """Greedy (argmax) decoding -- deterministic, appropriate for a
        task with a single correct answer. No KV-cache (recomputes the
        full growing sequence each step); stops as soon as every sequence
        in the batch has emitted eos_id rather than always running the
        full max_new_tokens."""
        self.eval()
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device) if eos_id is not None else None
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            idx = torch.cat([idx, next_id], dim=1)
            if eos_id is not None:
                finished |= next_id.squeeze(-1) == eos_id
                if finished.all():
                    break
        return idx


class Model:
    def __init__(self, seed: int = 0):
        self._rng = Random(seed)
        torch.manual_seed(seed)

        ckpt_dir = Path(os.environ.get("MATH_NLP_CKPT_DIR", "checkpoints/solution"))
        cfg_path = ckpt_dir / "config.json"
        ckpt_path = ckpt_dir / "best.pt"
        if not ckpt_path.exists():
            ckpt_path = ckpt_dir / "latest.pt"
        if not cfg_path.exists() or not ckpt_path.exists():
            raise FileNotFoundError(
                f"no trained checkpoint found in {ckpt_dir} (expected config.json + "
                f"best.pt/latest.pt) -- run train.py first"
            )

        with open(cfg_path) as f:
            model_cfg_dict = json.load(f)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = Tokenizer()
        self.gpt_config = GPTConfig(**model_cfg_dict)
        self.gpt = GPT(self.gpt_config).to(self.device)
        state = torch.load(ckpt_path, map_location=self.device)
        self.gpt.load_state_dict(state["model_state"])
        self.gpt.eval()

    def predict(self, tokens):
        idx = torch.tensor([list(tokens)], dtype=torch.long, device=self.device)
        prompt_len = idx.size(1)
        max_new_tokens = self.gpt_config.block_size - prompt_len
        out = self.gpt.generate(idx, max_new_tokens=max_new_tokens, eos_id=self.tokenizer.eos_id)
        return out[0, prompt_len:].tolist()
