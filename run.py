"""Skeleton inference + evaluation pipeline.

Two modes (mutually exclusive):

    python run.py --data data/test.jsonl
    python run.py --prompt 3344x1119

The pipeline is::

    text     = format_prompt(a, b)             # canonical data side
    pre      = transform.preprocess(text)      # placeholder: identity
    tokens   = tokenizer.encode(pre)           # placeholder: identity
    pred     = model.predict(tokens)           # placeholder: random integer
    detoked  = tokenizer.decode(pred)          # placeholder: identity
    out      = transform.postprocess(detoked)  # placeholder: identity

Replace any of ``Transform`` / ``Tokenizer`` / ``Model`` with real
implementations to turn this skeleton into a working LLM-based
multiplier.
"""

import argparse
import re
from pathlib import Path

from math_nlp import (
    Model,
    Tokenizer,
    Transform,
    exact_match_accuracy,
    format_prompt,
    load_jsonl,
)


PROMPT_RE = re.compile(r"^(\d+)x(\d+)=?$")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--data", type=Path, help="JSONL file to evaluate on")
    g.add_argument(
        "--prompt", type=str,
        help="single inference, format: AxB or AxB=",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def infer(
    transform: Transform,
    tokenizer: Tokenizer,
    model: Model,
    prompt_text: str,
) -> str:
    pre_text = transform.preprocess(prompt_text)
    tokens = tokenizer.encode(pre_text)
    pred_tokens = model.predict(tokens)
    pred_text = tokenizer.decode(pred_tokens)
    return transform.postprocess(pred_text)


def run_eval(
    data_path: Path,
    transform: Transform,
    tokenizer: Tokenizer,
    model: Model,
) -> None:
    rows = load_jsonl(data_path)
    preds = [infer(transform, tokenizer, model, row["prompt"]) for row in rows]
    targets = [row["completion"] for row in rows]

    acc = exact_match_accuracy(preds, targets)
    correct = sum(p == t for p, t in zip(preds, targets))
    print(f"accuracy: {acc:.4f}  ({correct}/{len(rows)})")

    wrong = [
        (row["prompt"], pred, row["completion"])
        for row, pred in zip(rows, preds)
        if pred != row["completion"]
    ]
    if wrong:
        print("sample errors:")
        for prompt, pred, target in wrong[:5]:
            print(f"  {prompt}{pred}   (expected {target})")


def run_prompt(
    prompt: str,
    transform: Transform,
    tokenizer: Tokenizer,
    model: Model,
) -> None:
    m = PROMPT_RE.match(prompt)
    if not m:
        raise SystemExit(f"--prompt must look like 'AxB' or 'AxB=', got: {prompt!r}")
    a, b = int(m.group(1)), int(m.group(2))
    prompt_text = format_prompt(a, b)
    pred = infer(transform, tokenizer, model, prompt_text)
    print(f"prompt:     {prompt_text}")
    print(f"prediction: {pred}")


def main():
    args = parse_args()
    transform = Transform()
    tokenizer = Tokenizer()
    model = Model(seed=args.seed)

    if args.data is not None:
        run_eval(args.data, transform, tokenizer, model)
    else:
        run_prompt(args.prompt, transform, tokenizer, model)


if __name__ == "__main__":
    main()
