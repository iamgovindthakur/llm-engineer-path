# Progress

The source of truth for what has actually been learned. Updated by `/progress`.

A lesson counts as done only when: the experiment ran, the predict-first
question was attempted, the shapes can be stated from memory, the interview
hooks were attempted, and `NOTES.md` exists in the lesson directory written by
the learner — not by Claude.

**Current position:** Phase 0 — next lesson is **00.3 (vectors: norm, dot product, cosine)**.
**Agreed path (trimmed):** 00.3 → Phase 1 broadcasting/shapes lessons only → resume `02_attention/` (the paused `sqrt(d_k)` lesson) → multi-head attention. Autograd/backprop later, before building a GPT. Skip Phase 0 lessons 00.5-00.8 for now.

---

## Done

| Date | Phase | Lesson | Directory | What I can now do |
| --- | ---: | --- | --- | --- |
| 2025-09-14 | 2 | Token IDs as addresses | `01_embeddings/` | Explain that an ID is a row index, not a number to compute with |
| 2025-09-14 | 2 | `nn.Embedding` | `01_embeddings/` | Say what the layer owns and why it is not a matmul |
| 2025-09-15 | 2 | Batched lookup | `01_embeddings/` | Predict `(B,T) -> (B,T,d)` and say what each axis means |
| 2025-09-15 | 2 | Lookup rule proved | `01_embeddings/` | Show `E[b,t,d] = W[T[b,t],d]` and verify it |
| 2026-09-20 | 0 | 00.1 Environment check | — | Skipped: setup already done and verified (venv, torch 2.10, MPS) |
| 2026-09-20 | 0 | 00.2 Devices | `00_setup_math/` | Explain async GPU timing (always `synchronize()`), GPU launch overhead vs small matmul, no float64 on MPS |
| 2026-09-20 | 0 | FLOP counting | `00_setup_math/` | Count matmul FLOPs: outputs `m*n`, each `k` mults + `k-1` adds, ~`2mkn`; FLOPs/token ~ 2 x params |
| 2025-09-15 | 2 | Toy tokenizer | `01_embeddings/` | Map text to IDs through a dict vocabulary |

Also carried in from before this workspace: dot product vs matrix
multiplication.

## In progress

| Phase | Lesson | Directory | Where it stopped |
| ---: | --- | --- | --- |
| 3 | Single-head causal self-attention (PAUSED) | `02_attention/` | Paused at the `sqrt(d_k)` predict-first question (std of q.k for d_k=4,64,1024; softmax of one score 20). Q/K/V, scores, scaling, causal mask and softmax are written. Missing: `NOTES.md`, the dropout/numerics discussion, and the interview hooks were not attempted. |

## Next

1. **00.3 Vectors** (leads directly into why `sqrt(d_k)` exists).
2. Phase 1 broadcasting/shapes lessons only.
3. Resume `02_attention/`, finish `NOTES.md` and interview hooks.
4. Multi-head attention.

Open small items: 00.2 `NOTES.md` still blank (the learner writes it); optional 70B and TFLOP/s exercises unanswered.

---

## The backfill note — read this before continuing

Phases 0-1 (broadcasting, matmul intuition, autograd, backprop, cross-entropy,
a full training loop) were never covered. The embeddings and attention work so
far is *lookup and forward-pass only*, which is why it has been possible to get
this far without them.

Multi-head attention is where that stops working. The head-splitting
reshape/transpose sequence is pure broadcasting-and-contiguity reasoning, and
the standard experience is getting a tensor of the right shape with the heads
scrambled — which raises no error and produces plausible numbers.

Recommendation: do Phase 1 next, in full. It is 4-6 weeks and it is the
difference between deriving the rest of the curriculum and memorising it.

Alternative if that feels like a detour: take the broadcasting, reshape-vs-view,
and matmul lessons from Phase 1 only (about a week), continue into multi-head
attention, and do autograd/backprop/training-loop before Phase 4. This is the
compromise, not the better path.

---

## Open questions

Questions raised and deliberately deferred. Surfaced again when a later lesson
makes them answerable.

| Raised | Question | Answerable after |
| --- | --- | --- |
| 2026-09-20 | Interviewer-level detail: exact FLOPs `mn(2k-1)` vs rounded `2mkn` | Phase 5 (serving math) |

---

## Session log

| Date | Minutes | What happened |
| --- | ---: | --- |
| 2026-09-20 | ~150 | Finished 00.2 + FLOP counting; agreed trimmed path | 
| 2026-09-20 | — | Workspace rebuilt for Claude Code: CLAUDE.md, skills, 12-phase curriculum, interview bank, portfolio track |
