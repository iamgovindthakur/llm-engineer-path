---
name: check
description: Review code the learner wrote it themselves — find the bug and teach the underlying concept without rewriting their work. Use when they say "/check", "review my code", "why is this wrong", "my implementation gives weird output", or pastes their own implementation for feedback.
---

# Check — review their code, do not replace it

They wrote this to learn. Rewriting it takes the learning away.

## Order of operations

1. **Run it first.** See the actual failure before theorising.
   `.venv/bin/python`
2. **Find the root cause**, not the first symptom.
3. **Name what is wrong** in one sentence.
4. **Explain why** it is wrong — the concept, not just the line.
5. **Show the smallest correction.** A diff of one or two lines. Never a
   rewritten file.
6. **Explain the underlying concept** that would have prevented it.
7. **Ask them to predict** what the corrected version outputs, before running.

## Never

- Silently fix things you noticed on the way past. Point them out; let them fix.
- Rewrite for style while fixing correctness. Those are separate conversations.
- Say "looks good" when it runs but is conceptually wrong. Working code with a
  wrong mental model is the most expensive kind of bug.

## Where to look first in PyTorch code

In rough order of how often they are the real cause:

- **Silent broadcasting.** `(B,T,1)` vs `(B,1,T)` produced `(B,T,T)` and no
  error. Print shapes at every step.
- **Wrong `dim=`** in softmax, sum, mean, cat. Softmax over the wrong axis
  gives plausible numbers and a broken model.
- **`transpose` vs `reshape`** when splitting heads — reshaping across a
  non-contiguous axis scrambles heads without raising.
- **Mask polarity** — masking where you meant to keep, or `0` where you meant
  `-inf`.
- **Missing `.detach()` / in-place op on a leaf** breaking the graph.
- **Device mismatch**, or an op silently falling back to CPU on MPS.
- **`float64` on MPS** — unsupported; a common source of confusing errors.
- **Forgetting `model.eval()` / `torch.no_grad()`** at inference.
- **Off-by-one in the labels shift** for next-token prediction.

## If it is correct

Say so plainly, then push: what is its complexity? What breaks at `T=8192`?
What would you change to serve it? Correct-but-unoptimised is the start of the
interesting conversation, not the end.
