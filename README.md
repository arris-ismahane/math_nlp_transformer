# Project map

This is a longer, more descriptive companion to `README.md` (the original
assignment brief). It walks through what every file actually does, how a
prompt travels through the model, and why the two training formats exist.
For results and failure analysis, see `WRITEUP.md` / `writeup_done.md`.

## What each file does

**`generate_dataset.py`** — makes the data. Samples random 4-digit x 4-digit
pairs without replacement (so train and test never overlap), splits off a
test fraction, and writes `data/train.jsonl` / `data/test.jsonl`. Given,
untouched.

**`math_nlp/data.py`** — the low-level building blocks: `sample_pair` (pick
one `a, b`), `format_prompt` (`"1222x3399="`), `format_completion`
(`str(a*b)`, the plain answer), and jsonl read/write. Given, untouched.

**`math_nlp/transform.py`** — the part that actually matters. Holds:
  - `long_multiply_trace(a, b)` — a real re-implementation of grade-school
    long multiplication, digit by digit, that records every single-digit
    multiply and every carry (row-wise, then column-wise during the final
    addition).
  - `render_target(a, b)` — turns that trace into the text the model is
    trained to produce, e.g. `R0:2x9+0=8c1,...->10998;...|A:8+c0=8c0,...|=4153578`.
  - `carry_density(a, b)` — counts total carries in the trace. Used only by
    `analyze.py`, to bucket test examples by difficulty.
  - `parse_completion(generated)` — pulls the final answer out after the
    last `=` (falls back to the whole string if there isn't one).
  - `Transform` — the `preprocess`/`postprocess` hooks `run.py` calls.

**`math_nlp/tokenizer.py`** — digit-level tokenizer. Every digit `0-9` and
every symbol the scratchpad format needs (`x = + , ; | : - > A R c`) gets
its own token, plus `PAD`/`BOS`/`EOS`. 25 symbols total, so `"3399"` is
always exactly 4 tokens, always in the same order.

**`math_nlp/model.py`** — the actual transformer: a small nanoGPT-style
`GPT` class, plus a `Model` wrapper that `run.py` imports. `Model` loads
whichever checkpoint `MATH_NLP_CKPT_DIR` points at (`best.pt` if it exists,
else `latest.pt`) and exposes `.predict(tokens)`.

**`math_nlp/evaluate.py`** — one function, `exact_match_accuracy`. Given.

**`train.py`** — not part of the graded interface, this is what produces
the checkpoints `Model` loads. Two variants:
```
python train.py --variant baseline --train data/train.jsonl --test data/test.jsonl
python train.py --variant solution --train data/train.jsonl --test data/test.jsonl
```

**`run.py`** — the fixed interface, not meant to be edited. Wires
`Transform` / `Tokenizer` / `Model` together and either evaluates a jsonl
file or answers one `--prompt`.

**`analyze.py`** — see its own section below.

## How training actually runs, step by step

`train.py` is the one you run — something like
`python train.py --variant solution`. It's the conductor. It doesn't do
any of the real work itself, it just calls the other files in order.

It asks `math_nlp/data.py` for training examples — thousands of random
4-digit x 4-digit pairs (like `1222` and `3399`), stored as plain numbers,
no text yet.

For each pair, it asks `math_nlp/transform.py` to turn it into the actual
text the model will read. For the `solution` variant that's the full
"do it the way a human would on paper" trace — every digit times every
digit, every carry, then the final column addition, ending in `=4153578`.
(For `baseline` it's just the plain `4153578`, no steps.)

That text goes to `math_nlp/tokenizer.py`, which chops it into single
characters and turns each one into a number, a token id, because the
model only works with numbers.

Those numbers go into `math_nlp/model.py`, a small transformer. Training
is just: show it thousands of these sequences, and after each one nudge
its weights slightly so it predicts the next token a bit better. Repeat
thousands of times, and periodically save progress to a checkpoint file
(`checkpoints/<variant>/best.pt` and `latest.pt`).

That's training — one direction, prompt-plus-answer in, loss out, over
and over on batches of examples.

Inference (`run.py`) runs the same stack in reverse and one-shot: it only
ever gets the prompt (`"1222x3399="`), and has to generate the rest of the
sequence token by token, feeding each guess back in as input for the next,
until it emits `EOS`. `Transform.postprocess` then reads the final answer
back out of whatever text came out.

## One prompt's journey through the model itself

```
"1222x3399="  (your text)
      │
      ▼
┌──────────────────────────┐
│ token embedding            │  each character → a list of 384 numbers
│ + position embedding       │  each position (1st char, 2nd char...) → its own 384 numbers, added in
└──────────────────────────┘
      │
      ▼
┌──────────────────────────┐
│ Block 1: attention + MLP  │
└──────────────────────────┘
      │
      ▼
      ⋮   (Block 2, 3, 4, 5 — same shape, different learned weights)
      │
      ▼
┌──────────────────────────┐
│ Block 6: attention + MLP  │
└──────────────────────────┘
      │
      ▼
┌──────────────────────────┐
│ final layer norm           │
└──────────────────────────┘
      │
      ▼
┌──────────────────────────┐
│ output head                │  → a guess for the next character
└──────────────────────────┘
```

That last step only produces one character. To get the full answer,
`GPT.generate` runs this whole stack again for every new character, each
time with the sequence one token longer, until it emits `<EOS>` (or runs
out of room). That's also why the scratchpad format is slow to run at
inference time compared to plain — it's ~350 characters to generate one
at a time instead of ~8.

## Plain vs. detailed scratchpad, and the paper behind it

Two formats get trained and compared here:

- **`baseline`** (`--variant baseline`): `1222x3399=` straight to
  `4153578`, one shot, no steps shown. This is the format that fails —
  0% exact match, but 0% parse failure too, so it's not garbling output,
  it's just never doing the arithmetic right.
- **`solution`** (`--variant solution`): the detailed long-multiplication
  scratchpad, every digit-multiply and every carry spelled out, matching
  how a person actually does long multiplication by hand. This is what
  `run.py` loads by default, and it gets ~99.6% exact match.

This idea, that spelling out every intermediate step as text
dramatically helps a transformer do multi-step arithmetic it otherwise
can't, is exactly what's studied in
[Lee, Sreenivasan, Lee, Lee & Papailiopoulos, "Teaching Arithmetic to
Small Transformers" (arXiv:2307.03381)](https://arxiv.org/abs/2307.03381).
They train small from-scratch transformers on addition and multiplication
under several formats (plain, reversed, simplified scratchpad, detailed
scratchpad) and find detailed scratchpad reaching ~100% on multiplication
while plain format essentially fails, which matches what this project
found independently. See `WRITEUP.md` §3 for why the other two format
variants they study (`reversed`, `simplified_scratchpad`) weren't retrained
inside this package.

## `analyze.py`, a quality check rather than part of the pipeline

`analyze.py` doesn't feed into training or into `run.py` at all — nothing
imports it, and removing it would not break the graded pipeline. Think of
it more like an assertion-style quality check on top of a trained
checkpoint than a pipeline component: it re-runs the same
Transform/Tokenizer/Model stack `run.py` uses (so its numbers should
agree), then adds two checks the fixed interface doesn't give you:

- a **carry-density breakdown**, asserting (informally, by eye) that
  accuracy doesn't quietly collapse on harder, higher-carry examples;
- a handful of **correct / wrong-but-parsed / unparseable examples**
  printed out, so failures can actually be read and understood rather than
  just trusted as a percentage.

Run it with `python analyze.py --data data/test.jsonl`, or point it at a
specific checkpoint the same way `run.py` is pointed:
`MATH_NLP_CKPT_DIR=checkpoints/baseline python analyze.py --data data/test.jsonl`.

## Makefile

A `Makefile` is included with the basic commands wired up
(`make data`, `make train-solution`, `make eval`, `make prompt PROMPT=3344x1119`,
`make analyze`, ...). Run `make help` (or just `make`) to list them.
