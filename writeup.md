# Write-up: 4-digit x 4-digit multiplication

## 0. How do humans do multiplication?

When I multiply two 4-digit numbers by hand, I don't just know the answer. I break it into single-digit multiplications first, one row at a time, and I carry whenever a digit product goes over 9. Then I add up the shifted rows, and I carry again during that addition. Nobody memorizes what 1222 times 3399 is. It's always small steps plus bookkeeping. That's why I made the model write out those same steps, instead of asking it to jump straight from `1222x3399=` to `4153578` in one guess.

## 1. Modeling strategy and why

Before picking anything, I spent time reading about why LLMs actually struggle with multiplication in the first place, instead of just guessing at a fix. The answer that kept coming up wasn't that transformers can't learn arithmetic at all, it was tokenization. A BPE tokenizer chops a number into chunks based on what's frequent in its training corpus, not based on place value, so the same digit can end up in a different token depending on what's around it. Once I understood that this was the actual root cause, I went looking for work that tackled it head on, and found Lee, Sreenivasan, Lee, Lee and Papailiopoulos, "Teaching Arithmetic to Small Transformers" ([arXiv:2307.03381](https://arxiv.org/abs/2307.03381)). I followed their approach fairly closely, digit-level tokenization plus a detailed scratchpad, and adapted it to this specific task. The rest of this section and section 2 is that decision worked out in detail.

I trained a small transformer from scratch instead of fine-tuning or prompting a pretrained model. Here's why.

A pretrained model comes with a BPE tokenizer, and BPE chops numbers up inconsistently depending on what's around them. Sometimes a 4-digit number is one token, sometimes it's split two and two, sometimes three and one. That breaks the link between where a digit sits in the string and what place value it actually has, which is exactly the thing this task is meant to probe. Building my own tokenizer let me guarantee every digit is always one token, always in the same relative spot.

Training from scratch is also cheap to iterate on, about 25 minutes per run on one RTX 4090. That let me actually train and compare the plain format against the scratchpad format instead of guessing which one would work.

The model itself is small on purpose. 6 layers, 6 heads, 384 dimensions, about 10.6 million parameters, block size 384. I wanted any accuracy gap between the two formats to come from the format itself, not from throwing more capacity at the problem.

## 2. Tokenizer and why

I used a digit-level tokenizer instead of BPE. Every digit 0 through 9 gets its own token, plus a handful of symbols the scratchpad format needs (`x = + , ; | : - > A R c`), plus PAD, BOS and EOS. 25 symbols total. So `"3399"` is always exactly 4 tokens, one per digit, always in the same order. This part was never the bottleneck, the vocabulary is small enough that it just wasn't where the difficulty lived.

## 3. Data transformation and why

This is the part that actually decides whether the model works. I trained and compared two formats.

The **plain** format goes straight from `1222x3399=` to `4153578`. One shot, no steps shown.

The **scratchpad** format goes from `1222x3399=` to a full trace of long multiplication, written out as text. Every single-digit multiply with its carry, row by row, then the column-by-column addition with its own carries, ending in the real answer. Something like:
```
R0:2x9+0=8c1,2x9+1=9c1,...->10998;R1:...;R2:...;R3:...|A:8+c0=8c0,...|=4153578
```
This is the format `run.py` actually runs.

I didn't train the "reversed" or "simplified scratchpad" variants that show up elsewhere in this line of work (see Lee, Sreenivasan, Lee, Lee and Papailiopoulos, "Teaching Arithmetic to Small Transformers," [arXiv:2307.03381](https://arxiv.org/abs/2307.03381), which studies exactly this set of format choices for small from-scratch transformers). I'd looked at those two informally before this project, on a bigger dataset, outside this package. Reversed came out barely different from plain, around 0.1 percent exact match. Simplified scratchpad, which only writes the partial products without the explicit carries, landed around 92 percent on a random split and dropped to 53 percent on a harder held-out split. Both clearly behind detailed scratchpad's 99 percent plus on both splits. Given the time box, I picked the two formats that tell the real story. Plain shows the known failure, detailed scratchpad shows the fix. I didn't spend the budget retraining the middle ground.

## 4. Results

Both numbers below come from the actual `run.py` pipeline, not an internal shortcut. I just pointed `MATH_NLP_CKPT_DIR` at each checkpoint and ran `python run.py --data data/test.jsonl`.

| | exact-match | parse-failure | wrong-but-parsed |
|---|---|---|---|
| `baseline` (plain) | 0.00% (0/1000) | 0.00% | 100.00% |
| `solution` (detailed scratchpad) | **99.60% (996/1000)** | 0.00% | 0.40% |

Plain's 0 percent isn't a formatting collapse. It reliably produces well-formed digit strings, it just never gets the arithmetic right. And it's not random either. `1222x3399=` turns into `4088888`, and `3344x1119=` turns into `4088886`, both far from their real (and very different) answers of `4153578` and `3741936`, but suspiciously similar to each other. It looks like the model learned roughly what an answer looks like, how many digits, roughly what size, without learning any actual procedure.

I also broke the scratchpad model's accuracy down by carry density, on a 300-example sample from the same checkpoint. Low carry count, 100 percent. Medium, 98.95 percent. High, 98.94 percent. It barely moves. That's really the point of writing every carry out explicitly. Each one becomes its own small, easy next-token prediction, instead of something that has to be gotten right silently inside one hidden state.

### A caveat about which checkpoint these numbers reflect

One more thing worth flagging. `train.py`'s own final printed line says `0.9890` accuracy, but that's measured at the last training step, 3500, where early stopping kicked in. `math_nlp/model.py`'s `Model` actually prefers `best.pt` over `latest.pt`, and `best.pt` got saved earlier, at step 3250, the first of two evals that tied exactly at 298 out of 300 on the small periodic-eval subset (`checkpoints/solution/train_log.csv` shows both step 3250 and step 3500 at 0.99333). Since the save rule only overwrites on a strict improvement, the tie at step 3500 never replaced it. It turns out the step-3250 weights generalize slightly better to the full 1,000-example test set, 99.60 percent, than the step-3500 weights do, 98.90 percent. A small but genuine reminder that a small proxy subset can rank two checkpoints differently than the full test set does, even when they're tied on the proxy. The number that actually matters, the one `run.py` loads and reports, is 99.60 percent.

## 5. Where it fails

I traced one `wrong_but_parsed` example by hand. `3649x6381` predicted `23184269`, true answer `23284269`. Every one of the four per-digit row-multiplications came out correct. The mistake was in the column-wise addition. One column summed to 28 (`9+9+9+carry_in(1)`), and the model got the last digit right, 8, but picked the wrong carry, `c1` instead of the correct `c2`. A genuine carry-value mistake, not a structural or formatting one, and specifically on a carry bigger than 1, which shows up less often in training since most carries are just 0 or 1. That one wrong carry shifted exactly one downstream digit.

There's a second, different failure mode too, from a hand-picked stress case rather than the random test set. `1000x1000` predicted `9000000`, true answer `1000000`. Again the whole scratchpad structure came out flawless, but one single digit-multiplication step that should have read `1x1+0=1c0` instead read `9x1+0=9c0`. A digit-recall slip, on an input that's unusually sparse, three leading zeros and one lone nonzero digit, probably underrepresented in a uniformly sampled training set.

Both failure modes share a pattern worth calling out directly. The model has clearly learned the procedure itself, the scratchpad's structure and grammar comes out essentially perfect, 0 percent parse-failure across both baseline's and solution's evaluations once measured correctly (see the bug below). What's left, the roughly 0.4 percent error rate, lives entirely in occasional wrong digit or carry values inside an otherwise correctly executed procedure, and it clusters on the parts of arithmetic that are naturally rarer in a uniformly sampled training distribution, large multi-way carries and sparse, edge-case operands.

### A bug I found and fixed along the way

`run.py`'s given `infer()` only ever passes `Transform.postprocess` the model's generated completion, never the original prompt. The `solution` format's target text conveniently ends with its own `"=<product>"`, so parsing it (find the last `"="`, take what follows) works fine unmodified. But `baseline`'s target (`format_completion`) is just the digits, no `"="` of its own. So the first version of `parse_completion` reported 100 percent parse-failure for baseline, even though the model was reliably emitting plausible-looking, if wrong, digit strings. I fixed it by falling back to treating the whole completion as the answer when there's no `"="` present (`math_nlp/transform.py:parse_completion`). This didn't change baseline's headline accuracy, still 0 percent, I confirmed that independently through `train.py`'s own full prompt-plus-completion eval, but it did turn the parse-failure and wrong-but-parsed breakdown from meaningless into accurate. Training itself was never affected, since teacher-forced loss doesn't go through this code path at all. This was purely an evaluation bug, the same kind documented in the earlier informal exploration this project builds on.

## 6. Known limitations of this package

`run.py --data` is genuinely slow for the solution format. The given `run_eval` is unbatched, one example at a time, with no KV-cache during generation, so `model.generate` recomputes the whole growing sequence at every single step. I left this alone on purpose, since `run.py` isn't meant to be modified, but it means a full 1,000-example pass takes several minutes per hundred examples for the roughly 350-token scratchpad completions, versus seconds for the roughly 8-token baseline. Not a correctness problem, just a real cost of a fixed, simple interface combined with a deliberately verbose target format.

Only two of the plausible format variants got actually trained inside this package (see section 3). `reversed` and `simplified_scratchpad` come from an earlier, separate, informal exploration, not retrained here, given the time box.

The held-out evaluation is a random split only, following the given `generate_dataset.py`, sampling without replacement. There's no harder, structurally held-out split, for instance training only on a restricted operand range and testing outside it. So the reported accuracy is really an interpolation-style number. It says less about whether the model learned a fully leading-digit-invariant procedure than a harder split would.

## 7. What I'd do with another day

Batch `Model.predict` internally, or add a KV-cache. `run.py`'s call signature can't change, but nothing stops `Model` from batching multiple `predict` calls together, or caching attention keys and values across generation steps on its own side. That would directly fix the slow full-dataset pass above without touching the given interface at all.

Try a real structural held-out split. Pull out one whole band of operands, say every `a` with leading digit 4, train without it, and test only on that band. That would tell me whether 99.6 percent reflects a procedure that's genuinely leading-digit-invariant, or partly just benefits from the random split looking a lot like the training distribution.

Try curriculum-stripping the scratchpad at inference time. Train with the full trace visible, like I did here, then during training progressively remove trailing portions of the target, first the final addition block, then more of the digit-multiplication rows, so the model gets pushed to internalize the computation those tokens used to carry out, while inference-time cost drops back toward baseline's roughly 8-token generations. That tests whether the scratchpad's benefit can be kept separate from its inference-time cost.

Retrain `reversed` and `simplified_scratchpad` inside this package itself, rather than leaning on the earlier informal exploration's numbers, if a clean, fully reproducible ablation across every format variant is wanted for this specific codebase.
