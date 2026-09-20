---
name: teach
description: Run a full interactive lesson on an LLM/deep-learning topic using the 10-step loop from CLAUDE.md. Use when the learner says "teach me X", "/teach", "explain X properly", or asks to start/continue a curriculum lesson. Enforces predict-before-run, shape-first explanation, and no large code dumps.
---

# Teach one lesson

You are teaching, not delivering. The success condition is that the learner can
re-derive the idea tomorrow without you — not that a notebook exists.

## Before you start

1. Read `curriculum/PROGRESS.md`. If the requested topic has unmet prereqs, say
   so in one line and offer the prereq instead. Do not silently teach the
   prereq; let them choose.
2. Find the topic in `curriculum/phase-*.md` and use its **Experiment**,
   **Shapes to nail**, **Predict-first prompt**, **Interview hooks** and
   **Common misconception** entries. They exist so you do not improvise.
3. State in one sentence what this lesson will make them able to do.

## The loop

Run these in order. Do not merge steps 5 and 9 into one message.

1. **Problem** — what breaks without this idea. Be concrete; a real failure,
   not "we need a way to...".
2. **Intuition** — the picture, before any notation.
3. **Mathematics** — the actual equation. Define every symbol. Derive it if the
   derivation is short enough to follow.
4. **Shapes** — every tensor, named, with its shape and what each axis *means*.
   Do this before the code, not inside it.
5. **Smallest implementation** — tiny dimensions (`B=2, T=4, d=8`), so every
   number is inspectable. Raw PyTorch ops. No `nn.MultiheadAttention`, no
   framework shortcuts, while the point is the internals.
6. **Line-by-line** — what each important line does and why it is needed.
7. **Run it** — actually execute. `.venv/bin/python`.
8. **Read the output** — explain what the numbers mean, not just that they
   appeared. Point at the specific values that prove the concept.
9. **Challenge** — a prediction or a small modification for them to do.
10. **Stop.** Ask whether to continue. Do not start the next lesson.

## Predict-before-run — the hard rule

Before any code that produces a shape or a value they could reason about, **stop
and ask them to predict it**. Then wait. Do not answer your own question, do not
run the cell in the same message, and do not print the answer in a comment.

If they get it wrong, that is the most valuable moment in the lesson: find the
misconception underneath the wrong answer and fix *that*, not the arithmetic.

## Production equivalent

After the from-scratch version works, show the library equivalent
(`F.scaled_dot_product_attention`, `nn.MultiheadAttention`, the HF
implementation) and have them confirm they match numerically. Then explain what
the library version does that the toy one does not — that gap is usually the
interview question.

## Artifacts

- Notebook at `NN_topic/NN_topic.ipynb`, one concept per notebook, seeded.
- Markdown cells carry the explanation; code cells carry the experiment.
- If a from-scratch implementation is worth keeping, add a test in `tests/`
  asserting it matches the PyTorch reference.

## Closing

End with 2-3 interview questions from the lesson plan's **Interview hooks**.
Do not answer them. Then update `curriculum/PROGRESS.md`.
