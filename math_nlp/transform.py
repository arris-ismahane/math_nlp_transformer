"""
Data conversion and scratchpad target formatting for the math_nlp task.
"""

from __future__ import annotations

EOS = "<EOS>"


def _digits_lsb_first(n: int, width: int) -> list[int]:
    s = str(n)
    if len(s) > width:
        raise ValueError(f"{n} has more than {width} digits")
    s = s.zfill(width)
    return [int(c) for c in reversed(s)]


def long_multiply_trace(a: int, b: int) -> dict:
    """Simulate grade-school long multiplication of a*b digit by digit.

    Deterministic re-implementation of the standard algorithm (not just
    `a * b`), so it can be (a) verified against `a * b`, (b) serialized
    into the scratchpad target text, and (c) used to define "carry
    density" for stratifying test examples by difficulty.
    """
    a_digits = _digits_lsb_first(a, 4)
    b_digits = _digits_lsb_first(b, 4)

    rows = []
    total_carries = 0
    for i, bd in enumerate(b_digits):
        carry_in = 0
        digit_steps = []
        for aj in a_digits:
            p = aj * bd + carry_in
            out_digit = p % 10
            carry_out = p // 10
            digit_steps.append(
                dict(a_digit=aj, b_digit=bd, carry_in=carry_in,
                     out_digit=out_digit, carry_out=carry_out)
            )
            if carry_out > 0:
                total_carries += 1
            carry_in = carry_out
        if carry_in > 0:
            digit_steps.append(
                dict(a_digit=0, b_digit=bd, carry_in=carry_in,
                     out_digit=carry_in, carry_out=0)
            )
        row_digits = [s["out_digit"] for s in digit_steps]  # lsb first
        row_value = int("".join(str(d) for d in reversed(row_digits)))
        rows.append(dict(index=i, multiplier_digit=bd, digit_steps=digit_steps,
                          row_digits=row_digits, row_value=row_value, shift=i))

    max_len = max(len(r["row_digits"]) + r["shift"] for r in rows)
    col_steps = []
    carry_in = 0
    for col in range(max_len):
        addends = []
        total = carry_in
        for r in rows:
            pos = col - r["shift"]
            if 0 <= pos < len(r["row_digits"]):
                d = r["row_digits"][pos]
                addends.append(d)
                total += d
        out_digit = total % 10
        carry_out = total // 10
        if carry_out > 0:
            total_carries += 1
        col_steps.append(dict(col=col, addends=addends, carry_in=carry_in,
                               out_digit=out_digit, carry_out=carry_out))
        carry_in = carry_out
    if carry_in > 0:
        col_steps.append(dict(col=max_len, addends=[], carry_in=carry_in,
                               out_digit=carry_in, carry_out=0))

    product_digits = [s["out_digit"] for s in col_steps]  # lsb first
    product = int("".join(str(d) for d in reversed(product_digits)))
    assert product == a * b, f"trace bug: {a}*{b} traced to {product}"

    return dict(a=a, b=b, rows=rows, col_steps=col_steps,
                total_carries=total_carries, product=product)


def carry_density(a: int, b: int) -> int:
    """Total number of carry events in the standard long-multiplication
    algorithm for a*b. Used only by analyze.py to bucket test examples
    into low/medium/high difficulty -- independent of training format."""
    return long_multiply_trace(a, b)["total_carries"]


def render_target(a: int, b: int) -> str:
    """The detailed_scratchpad training target: every single-digit
    multiply and every carry spelled out, both per-row and in the final
    column-wise addition, e.g.
        R0:2x9+0=8c0,2x9+1=9c1,...->9822|A:8+0+c0=8c0,...|=4153578
    """
    trace = long_multiply_trace(a, b)
    row_blocks = []
    for r in trace["rows"]:
        step_strs = [
            f"{s['a_digit']}x{s['b_digit']}+{s['carry_in']}={s['out_digit']}c{s['carry_out']}"
            for s in r["digit_steps"]
        ]
        row_blocks.append(f"R{r['index']}:" + ",".join(step_strs) + f"->{r['row_value']}")
    rows_block = ";".join(row_blocks)

    col_strs = []
    for c in trace["col_steps"]:
        addend_str = "+".join(str(x) for x in c["addends"])
        lhs = f"{addend_str}+c{c['carry_in']}" if addend_str else f"c{c['carry_in']}"
        col_strs.append(f"{lhs}={c['out_digit']}c{c['carry_out']}")
    add_block = "A:" + ",".join(col_strs)

    return rows_block + "|" + add_block + f"|={trace['product']}"


def parse_completion(generated: str) -> str:
    """
    Extracts the final answer from the generated text.
    It splits and returns on the last "=" if present. 
    On failure instead of raising, it uses the whole completion so it shows up as a mismatch.
    """
    generated = generated.split(EOS)[0]
    idx = generated.rfind("=")
    remainder = generated[idx + 1:].strip() if idx != -1 else generated.strip()
    if not remainder or not remainder.isdigit():
        return ""
    return remainder


class Transform:
    def preprocess(self, text: str) -> str:
        return text

    def postprocess(self, text: str) -> str:
        return parse_completion(text)
