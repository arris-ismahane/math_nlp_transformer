"""Generate train / test JSONL datasets of 4-digit by 4-digit multiplication.

Pairs are sampled without replacement, so train and test are guaranteed
disjoint.

Usage
-----
    python generate_dataset.py --n 10000 --test-frac 0.1 --seed 0 --out data/

Each line of the output looks like::

    {"prompt": "1222x3399=", "completion": "4153578"}
"""

import argparse
from pathlib import Path
from random import Random

from math_nlp import (
    format_completion,
    format_prompt,
    sample_pair,
    write_jsonl,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--n", type=int, default=10_000,
        help="total number of unique examples (default: 10000)",
    )
    p.add_argument(
        "--test-frac", type=float, default=0.1,
        help="fraction of examples in the test split (default: 0.1)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("data"))
    return p.parse_args()


def main():
    args = parse_args()
    rng = Random(args.seed)

    max_unique = (9999 - 1000 + 1) ** 2  # 81_000_000
    if args.n > max_unique:
        raise SystemExit(f"--n={args.n} exceeds the {max_unique} unique 4x4 pairs")

    seen: set[tuple[int, int]] = set()
    while len(seen) < args.n:
        seen.add(sample_pair(rng))

    pairs = list(seen)
    rng.shuffle(pairs)

    n_test = int(round(args.n * args.test_frac))
    test_pairs = pairs[:n_test]
    train_pairs = pairs[n_test:]

    def to_rows(items):
        return [
            {"prompt": format_prompt(a, b), "completion": format_completion(a, b)}
            for a, b in items
        ]

    train_rows = to_rows(train_pairs)
    test_rows = to_rows(test_pairs)

    train_path = args.out / "train.jsonl"
    test_path = args.out / "test.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(test_path, test_rows)

    print(f"wrote {len(train_rows)} examples to {train_path}")
    print(f"wrote {len(test_rows)} examples to {test_path}")
    if train_rows:
        print(f"  train[0]: {train_rows[0]}")
    if test_rows:
        print(f"  test[0]:  {test_rows[0]}")


if __name__ == "__main__":
    main()
