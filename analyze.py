"""Failure analysis for the write-up.

Not part of the given interface -- runs the same Transform/Tokenizer/Model
pipeline run.py uses (so numbers here should agree with run.py's), but adds
two things the required interface doesn't report:

  - a carry-density breakdown (does accuracy degrade on "harder"
    multiplications, i.e. ones with more carry events in the standard
    long-multiplication algorithm?)
  - a handful of qualitative correct / wrong-but-parsed / unparseable
    examples to look at directly.

Reads MATH_NLP_CKPT_DIR the same way math_nlp.model.Model does, so it can
report on either checkpoint without any extra flag:

    python analyze.py --data data/test.jsonl
    MATH_NLP_CKPT_DIR=checkpoints/baseline python analyze.py --data data/test.jsonl
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from math_nlp import Model, Tokenizer, Transform, load_jsonl
from math_nlp.transform import carry_density

PROMPT_RE = re.compile(r"^(\d+)x(\d+)=$")


def parse_prompt(prompt: str) -> tuple[int, int]:
    m = PROMPT_RE.match(prompt)
    return int(m.group(1)), int(m.group(2))


def carry_terciles(densities: list[int]) -> tuple[int, int]:
    d = sorted(densities)
    n = len(d)
    return d[n // 3], d[2 * n // 3]


def bucket(density: int, lo_cut: int, hi_cut: int) -> str:
    if density <= lo_cut:
        return "low"
    if density <= hi_cut:
        return "medium"
    return "high"


def run(rows: list[dict], model: Model, tokenizer: Tokenizer, transform: Transform) -> list[dict]:
    results = []
    for row in rows:
        a, b = parse_prompt(row["prompt"])
        pre_text = transform.preprocess(row["prompt"])
        tokens = tokenizer.encode(pre_text)
        pred_tokens = model.predict(tokens)
        raw = tokenizer.decode(pred_tokens)
        pred = transform.postprocess(raw)
        target = row["completion"]
        results.append(dict(
            a=a, b=b, prompt=row["prompt"], target=target, prediction=pred,
            raw=raw, carry_density=carry_density(a, b),
            exact_match=(pred == target), parse_failed=(pred == ""),
        ))
    return results


def report(results: list[dict]) -> None:
    n = len(results)
    parsed = [r for r in results if not r["parse_failed"]]
    correct = [r for r in results if r["exact_match"]]
    wrong_but_parsed = [r for r in parsed if not r["exact_match"]]

    print(f"n={n}")
    print(f"exact-match accuracy:  {len(correct) / n:.4f}")
    print(f"parse-failure rate:    {(n - len(parsed)) / n:.4f}")
    print(f"wrong-but-parsed rate: {len(wrong_but_parsed) / n:.4f}")

    lo_cut, hi_cut = carry_terciles([r["carry_density"] for r in results])
    print(f"\nby carry-density bucket (cuts: low<={lo_cut}, medium<={hi_cut}, high>{hi_cut}):")
    for b in ("low", "medium", "high"):
        bucket_rs = [r for r in results if bucket(r["carry_density"], lo_cut, hi_cut) == b]
        if not bucket_rs:
            print(f"  {b:6s}: n=0")
            continue
        acc = sum(r["exact_match"] for r in bucket_rs) / len(bucket_rs)
        pf = sum(r["parse_failed"] for r in bucket_rs) / len(bucket_rs)
        print(f"  {b:6s}: n={len(bucket_rs):5d}  exact_match={acc:.4f}  parse_failure={pf:.4f}")

    for label, subset in (
        ("correct", correct),
        ("wrong_but_parsed", wrong_but_parsed),
        ("unparseable", [r for r in results if r["parse_failed"]]),
    ):
        print(f"\n-- {label} (showing up to 3) --")
        for r in subset[:3]:
            print(f"  {r['prompt']}  target={r['target']}  predicted={r['prediction']!r}  raw={r['raw']!r}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/test.jsonl"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="only analyze the first N rows (run.py's own unbatched inference is slow for verbose formats)")
    args = p.parse_args()

    rows = load_jsonl(args.data)
    if args.limit is not None:
        rows = rows[: args.limit]
    transform = Transform()
    tokenizer = Tokenizer()
    model = Model(seed=args.seed)

    results = run(rows, model, tokenizer, transform)
    report(results)


if __name__ == "__main__":
    main()
