"""Data utilities for the multiplication test.

Conventions
-----------
- Operands are 4-digit integers in [1000, 9999] (inclusive).
- One example is a JSON object with two string fields:
      {"prompt": "1222x3399=", "completion": "4153578"}
- Datasets are stored as JSON Lines: one such object per line.
"""

import json
from pathlib import Path
from random import Random


def sample_pair(rng: Random) -> tuple[int, int]:
    return rng.randint(1000, 9999), rng.randint(1000, 9999)


def format_prompt(a: int, b: int) -> str:
    return f"{a}x{b}="


def format_completion(a: int, b: int) -> str:
    return str(a * b)


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
