# Timed coding drills

Senior LLM loops often ask for a from-scratch implementation in 30-45 minutes.
Knowing the concept is not enough; you must produce it fluently, state shapes
out loud, and test it. Run each drill only **after** its lesson is done.

## Rules
- Blank file, no notes, no copy-paste, PyTorch/NumPy only.
- Set a timer. State the shapes before writing code.
- Write one test that would catch the most likely bug.
- Afterwards: what did you fumble? Add it to `WEAK_SPOTS.md`.
- Repeat a drill after 1-2 weeks; the second run is the real score.

| # | Drill | After | Target | Test that must pass |
| ---: | --- | --- | ---: | --- |
| 1 | Softmax + log-softmax, numerically stable | 01.15 | 10 min | Large logits do not overflow |
| 2 | Cross-entropy from logits | 01.14 | 15 min | Matches `F.cross_entropy` |
| 3 | Single-head causal self-attention | 03.6 | 25 min | Output at t unchanged if tokens after t change |
| 4 | Multi-head attention (split, merge, W_O) | 03.8 | 30 min | Matches `nn.MultiheadAttention` |
| 5 | RoPE applied to Q and K | 03.12 | 30 min | Score depends only on relative position |
| 6 | LayerNorm and RMSNorm | 03.15 | 15 min | Matches PyTorch |
| 7 | BPE train + encode + decode | 02.5 | 40 min | Round-trip on non-ASCII text |
| 8 | Top-k, top-p and temperature sampling | 5.6 | 20 min | Never samples outside the kept set |
| 9 | KV-cache generation loop | 5.10 | 30 min | Cached output equals uncached, token for token |
| 10 | KV-cache memory calculator | 5.12 | 10 min | Matches a worked example by hand |
| 11 | LoRA linear layer | 7.7 | 20 min | Zero-init: output equals base at step 0 |
| 12 | Flat vector search + recall@k | 08.3 | 20 min | Recall = 1.0 versus brute force |
| 13 | Reciprocal rank fusion | 08.11 | 10 min | Hand-computed example |
| 14 | Token-bucket rate limiter for an LLM gateway | 5.23 | 20 min | Bursts and refill behave; concurrency-safe |

Drill 14 is deliberately in your home territory: write it in Java too.

## Score log

| Date | Drill | Minutes | Passed test | What went wrong |
| --- | ---: | ---: | --- | --- |
