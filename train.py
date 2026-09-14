"""Train the multiplication model.

Not part of the given interface (generate_dataset.py / run.py are fixed) --
this script only needs to produce the checkpoints that math_nlp.model.Model
loads. Two variants:

    python train.py --variant baseline --train data/train.jsonl --test data/test.jsonl
    python train.py --variant solution --train data/train.jsonl --test data/test.jsonl

``baseline`` trains on the plain "a*b=" -> "<product>" completion (the
known-to-fail format, kept only as a documented comparison point -- see
WRITEUP.md). ``solution`` trains on the detailed long-multiplication
scratchpad (math_nlp.transform.render_target) and is what
math_nlp.model.Model loads by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from pathlib import Path

import torch

from math_nlp import exact_match_accuracy, format_completion, load_jsonl
from math_nlp.model import GPT, GPTConfig
from math_nlp.tokenizer import Tokenizer
from math_nlp.transform import Transform, render_target

PROMPT_RE = re.compile(r"^(\d+)x(\d+)=$")

# model / optimization hyperparameters -- standard small-transformer
# defaults, not tuned specially for this task
N_LAYER = 6
N_HEAD = 6
N_EMBD = 384
BLOCK_SIZE = 384  # covers worst case: 9999x9999 detailed scratchpad, ~350 tokens + BOS/EOS
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
WARMUP_STEPS = 100
MAX_STEPS = 6000
EVAL_INTERVAL = 250
EVAL_EXAMPLES = 300
EARLY_STOP_ACC = 0.99
EARLY_STOP_PATIENCE = 3
SEED = 0


def parse_prompt(prompt: str) -> tuple[int, int]:
    m = PROMPT_RE.match(prompt)
    if not m:
        raise ValueError(f"unexpected prompt shape: {prompt!r}")
    return int(m.group(1)), int(m.group(2))


def render_row(row: dict, variant: str) -> tuple[str, str]:
    a, b = parse_prompt(row["prompt"])
    target = format_completion(a, b) if variant == "baseline" else render_target(a, b)
    return row["prompt"], target


def build_tensor_dataset(rows: list[dict], variant: str, tokenizer: Tokenizer, block_size: int) -> torch.Tensor:
    pad_id = tokenizer.pad_id
    data = torch.full((len(rows), block_size), pad_id, dtype=torch.long)
    for i, row in enumerate(rows):
        prompt, target = render_row(row, variant)
        ids = tokenizer.encode(prompt + target, add_bos=True, add_eos=True)
        if len(ids) > block_size:
            raise ValueError(f"row {row!r} encodes to {len(ids)} tokens > block_size {block_size}")
        data[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return data


def lr_at(step: int) -> float:
    if step < WARMUP_STEPS:
        return LEARNING_RATE * (step + 1) / WARMUP_STEPS
    progress = min((step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS), 1.0)
    return LEARNING_RATE * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def get_batch(train_data: torch.Tensor, batch_size: int, device: str):
    ix = torch.randint(0, train_data.size(0), (batch_size,), device=train_data.device)
    seq = train_data[ix]
    return seq[:, :-1].to(device, non_blocking=True), seq[:, 1:].to(device, non_blocking=True)


@torch.no_grad()
def evaluate(model: GPT, tokenizer: Tokenizer, transform: Transform, rows: list[dict], variant: str, device: str) -> float:
    model.eval()
    preds, targets = [], []
    prompt_len = len(tokenizer.encode(rows[0]["prompt"], add_eos=False))
    max_new_tokens = model.config.block_size - prompt_len
    batch_size = 64
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        prompt_ids = [tokenizer.encode(r["prompt"], add_eos=False) for r in chunk]
        idx = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        out = model.generate(idx, max_new_tokens=max_new_tokens, eos_id=tokenizer.eos_id)
        for r, full_ids in zip(chunk, out):
            full_text = tokenizer.decode(full_ids.tolist(), stop_at_eos=True)
            preds.append(transform.postprocess(full_text))
            targets.append(r["completion"])
    return exact_match_accuracy(preds, targets)


def train(variant: str, train_path: Path, test_path: Path, ckpt_dir: Path):
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(train_path)
    test_rows = load_jsonl(test_path)

    tokenizer = Tokenizer()
    transform = Transform()

    print(f"[data] tokenizing {len(train_rows)} training examples (variant={variant!r})")
    train_data = build_tensor_dataset(train_rows, variant, tokenizer, BLOCK_SIZE).to(device)

    model_cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size, block_size=BLOCK_SIZE,
        n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD, dropout=0.0, pad_id=tokenizer.pad_id,
    )
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(vars(model_cfg), f, indent=2)

    model = GPT(model_cfg).to(device)
    print(f"[model] {model.num_params():,} parameters, block_size={BLOCK_SIZE}, device={device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))

    log_path = ckpt_dir / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["step", "train_loss", "lr", "test_exact_match"])

    best_acc = -1.0
    consecutive_at_target = 0
    running_loss = 0.0
    t0 = time.time()

    for step in range(MAX_STEPS):
        lr = lr_at(step)
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = get_batch(train_data, BATCH_SIZE, device)
        model.train()
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        running_loss += loss.item()

        do_eval = (step + 1) % EVAL_INTERVAL == 0 or step == MAX_STEPS - 1
        if do_eval:
            steps_since = EVAL_INTERVAL if (step + 1) % EVAL_INTERVAL == 0 else (step % EVAL_INTERVAL) + 1
            avg_loss = running_loss / steps_since
            running_loss = 0.0

            acc = evaluate(model, tokenizer, transform, test_rows[:EVAL_EXAMPLES], variant, device)
            elapsed = time.time() - t0
            print(f"[step {step+1}/{MAX_STEPS}] loss={avg_loss:.4f} lr={lr:.2e} test_acc={acc:.4f} ({elapsed:.0f}s elapsed)")

            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([step + 1, avg_loss, lr, acc])

            ckpt = dict(model_state=model.state_dict(), step=step + 1)
            torch.save(ckpt, ckpt_dir / "latest.pt")
            if acc > best_acc:
                best_acc = acc
                torch.save(ckpt, ckpt_dir / "best.pt")

            if acc >= EARLY_STOP_ACC:
                consecutive_at_target += 1
                if consecutive_at_target >= EARLY_STOP_PATIENCE:
                    print(f"[stop] test exact-match >= {EARLY_STOP_ACC} for {EARLY_STOP_PATIENCE} consecutive evals, stopping early.")
                    break
            else:
                consecutive_at_target = 0

    print("[final] evaluating on full test set")
    final_acc = evaluate(model, tokenizer, transform, test_rows, variant, device)
    print(f"final test exact-match accuracy: {final_acc:.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=["baseline", "solution"], required=True)
    p.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    p.add_argument("--test", type=Path, default=Path("data/test.jsonl"))
    p.add_argument("--ckpt_dir", type=Path, default=None)
    args = p.parse_args()
    ckpt_dir = args.ckpt_dir or Path("checkpoints") / args.variant
    train(args.variant, args.train, args.test, ckpt_dir)


if __name__ == "__main__":
    main()
