"""Evaluation utilities."""


def exact_match_accuracy(preds: list[str], targets: list[str]) -> float:
    if not preds:
        return 0.0
    if len(preds) != len(targets):
        raise ValueError(
            f"length mismatch: {len(preds)} preds vs {len(targets)} targets"
        )
    return sum(p == t for p, t in zip(preds, targets)) / len(preds)
