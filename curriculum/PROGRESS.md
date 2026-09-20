# Progress

The source of truth for what has actually been learned. Updated by `/progress`.

A lesson counts as done only when: the experiment ran, the predict-first
question was attempted, the shapes can be stated from memory, the interview
hooks were attempted, and `NOTES.md` exists in the lesson directory written by
the learner — not by Claude.

**Current position:** Phase 0 taught through 00.4. Next lesson is **00.5 (transpose)** — see the prerequisite audit below for why it is no longer skipped.
**Agreed path (revised 2026-09-20 after a prerequisite audit):**
`00.5` transpose → `01.2` indexing/slicing → `01.3` broadcasting → `01.4` bug drill →
`01.5` reshape vs view → `01.6` batched matmul → `01.7` einsum → `01.8` reductions →
**`00.9` mean/variance/sqrt(n)** → resume `02_attention/` (the paused `sqrt(d_k)` lesson)
→ multi-head attention. Autograd/backprop later, before building a GPT.
Still skipped: `00.6`, `00.7` (calculus — they return with backprop) and `00.8` (floats).

### Prerequisite audit — why three lessons were added back

| Lesson | Was | Why it had to come back |
| --- | --- | --- |
| **00.5** transpose | skipped | Its stated question is literally *"why does attention compute `Q @ K^T`?"* — the lesson we are paused on. It also owns the `(4,8) @ (8,4) -> (4,4)` score matrix, and its note that transpose returns a **view with permuted strides** is the stated prerequisite for `01.5` (reshape vs view), which was already on the path. Doing `01.5` without it was backwards. |
| **00.9** mean/variance | did not exist | `03.2` lists *"variance of a sum of independent variables"* as a prerequisite; so do `03.15` (RMSNorm) and `01.20` (initialization). Nothing in the curriculum taught it. Written 2026-09-20. |
| **01.2** indexing/slicing | not in the trimmed list | Teaches view-vs-copy, which `01.5` assumes. Its predict-first is last-timestep logits from `(B,T,V)` — the exact indexing the Phase 5 generation loop needs. |

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
| 2026-09-20 | 0 | 00.3 Vectors: norm, dot, cosine | `00_setup_math/01_vectors.ipynb` | Derive `\|a\| = sqrt(a.a)` from Pythagoras; state the algebraic vs geometric form of the dot product; explain the two reasons a dot product is big; rearrange to cosine similarity and say why it is bounded in [-1,1]; say that matmul is a grid of dot products |
| 2026-09-20 | 0 | 00.4 Matmul + shape algebra | `00_setup_math/02_matmul_shapes.ipynb` | Say in one second whether `(a,b) @ (c,d)` is legal and what comes out: inner must match, inner vanishes, outer survives. Explain a matmul as a grid of dot products (verified against a double loop). Read a shape error. Distinguish 1-D from row/column vectors. Explain a leading dim as a batch — same weights, different data — and why the output's last dim comes from the weight |

Also carried in from before this workspace: dot product vs matrix
multiplication.

## In progress

| Phase | Lesson | Directory | Where it stopped |
| ---: | --- | --- | --- |
| 3 | Single-head causal self-attention (PAUSED) | `02_attention/` | Paused at the `sqrt(d_k)` predict-first question (std of q.k for d_k=4,64,1024; softmax of one score 20). Q/K/V, scores, scaling, causal mask and softmax are written. Missing: `NOTES.md`, the dropout/numerics discussion, and the interview hooks were not attempted. |

## Next

Restructured 2026-09-20. Six-month goal: be a credible candidate for senior
LLM-infra roles (see `ROADMAP.md`). Study scope for that window: **Phases 0-5 +
the serving-lane efficiency block, shipping P1, P2, P3, P5.** Phases 8-11 are
months 7-11.

1. **00.5 Transpose** — unblocks both `Q @ K^T` and `01.5`.
2. Then 01.2 indexing, 01.3 broadcasting, 01.4 bug drill, 01.5 reshape vs view, 01.6 batched matmul, 01.7 einsum, 01.8 reductions.
3. **00.9 Mean/variance/sqrt(n)** immediately before resuming attention, so it is fresh for `03.2`.
3. Resume `02_attention/`, finish `NOTES.md` and interview hooks.
4. Multi-head attention (Phase 3).
5. Phase 1 remainder: autograd/backprop/training loop -> **P1**.
6. Phase 2 tokenization (timeboxed, 2 weeks) -> **P2**.
7. Phase 4 small GPT -> **P3** [flagship]. Then Phase 5 inference -> **P5** [flagship].

Running in parallel, not after (see `ROADMAP.md` Tracks 2 and 3):
- **Work:** find the LLM-shaped problem at the current job this month.
- **Market:** rewrite the resume this month; interview for real at **month 4**.

Open small items: 00.2, 00.3 and 00.4 `NOTES.md` sections still blank (the learner writes them);
optional 70B and TFLOP/s exercises unanswered; 00.3 notebook challenge (`a · d`, `cos(a,d)`)
not yet attempted; 00.4 notebook challenge (4 shape cases + exact FLOP count) not yet attempted.

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
| 2026-09-20 | ~60 | Taught 00.3 vectors: norm, dot product, cosine. Notebook saved, NOTES.md prompts added |
| 2026-09-20 | ~50 | Taught 00.4 matmul + shape algebra. Notebook saved (7 experiments, all executed). Phase 0 taught through 00.4 |
| 2026-09-20 | — | Roadmap review: 6-month scope cut to Phases 0-5 + P1/P2/P3/P5; work + job-search tracks added; quant/roofline pulled forward; phase numbering unified across ROADMAP/POSITIONING/PORTFOLIO |
| 2026-09-20 | — | Workspace rebuilt for Claude Code: CLAUDE.md, skills, 12-phase curriculum, interview bank, portfolio track |
