# Roadmap — from SDE-3 backend to senior LLM engineer

Twelve phases, run as three **parallel tracks** — study, work, market.
Each phase has its own lesson-level plan file; this page is the map, the
ordering argument, and the honest timeline.

`curriculum/PROGRESS.md` is the source of truth for what is actually done.

---

## The six-month goal

**Be a credible candidate for senior LLM-infra roles.** Not "finish the
curriculum" — the curriculum is the means. Credible means: you can implement the
internals, you can reason about serving cost and latency, and you have public
artifacts and one production LLM thing at work to point at.

`career/POSITIONING.md` is the market-facing companion. Its key sentence:
technical depth "gets you into the room and gets you levelled correctly. **Scope
and demonstrated impact**" are what decide the rest.

---

## Three tracks, run in parallel

The study track is the one that feels like progress. The other two matter at
least as much. Do not run them in sequence.

### Track 1 — Study (this curriculum)
Covered below.

### Track 2 — Work: get one LLM thing into production
`POSITIONING.md` §4 lists production LLM ownership as **"High severity, and
unfixable by side projects."** §8: "Six months of side projects is worth less
than one quarter of *I own the inference path for X*."

| Month | Action |
| ---: | --- |
| 1 | Find the LLM-shaped problem that already exists at work (internal search, ticket triage, doc Q&A). Identify who owns it. Ask what is blocking it — do not ask for a transfer. |
| 2 | Propose a scoped two-week piece with a measurable outcome. Frame it in reliability/cost terms. Get it on a roadmap. |
| 3-4 | Ship it. Write it up internally. |
| 5-6 | Turn it into a resume bullet with production scope. |

**The work problem will almost certainly be RAG-shaped, and that is fine.**
You do not need Phase 8 to ship v1 — an embedding API, a vector store and a
reranker are two to four weeks of ordinary engineering, and a framework is an
acceptable choice here. Phase 8 is what lets you *measure and debug* it, which
is what turns "I shipped a chatbot" into a bullet with numbers.

So pull four lessons forward out of Phase 8 to run beside the work project
(~10-12 h). Everything else in Phase 8 stays at months 7-11, where it becomes
`ragprod`:

| Lesson | Why it is needed now |
| --- | --- |
| **08.1** Retrieval vs long context vs fine-tuning | Pick the right approach at work instead of defaulting to RAG |
| **08.3** The golden set and recall@k | Build the eval set *before* the feature; this habit makes everything else measurable |
| **08.7** Chunking | `PORTFOLIO.md` calls naive fixed-size chunking "the #1 production failure" — it bites in week two |
| **08.20** Retrieval metrics vs generation metrics | Lets you say "retrieval was the bottleneck, not the prompt", which is the most senior sentence available in an applied-LLM interview |

### Track 3 — Market: interview early, on both lanes
`POSITIONING.md` §7 notes that level, not the AI label, is what employers sort
on. A backend/distributed-systems move stands on skills you already have and
does not depend on this curriculum finishing — so run the search in parallel
rather than after.

| Month | Action |
| ---: | --- |
| 1 | Rewrite resume in `POSITIONING.md` §6 format. Set up the GitHub profile README with P1 pinned. |
| 2-3 | Start conversations with 3-5 people doing this work in Bengaluru / Hyderabad. Ask what their team screens for. |
| **4** | **Interview for real — before you feel ready.** Treat the first three loops as free diagnostics, not as attempts. |
| 5-6 | Interview properly, on both the backend lane and the LLM-infra lane. |

---

## The six-month study plan

At 8-12 focused hours a week, six months is **208-312 hours**. That does not fit
both the core depth and the applied spine. It fits one lane, done properly.

Your lane is serving and infra (`POSITIONING.md` §3), so:

| Step | Content | Weeks | Output |
| ---: | --- | ---: | --- |
| 1 | Phase 0-1 trimmed: 00.5 transpose, 01.2 indexing, 01.3-01.8 shapes, then 00.9 mean/variance | 3-4 | — |
| 2 | Resume `02_attention/`; multi-head attention, RoPE, norms, the block (Phase 3) | 4-5 | Drills 3-6 |
| 3 | Phase 1 remainder: autograd, backprop, cross-entropy, training loop, optimizers | 3-4 | **P1 micrograd-tensor** |
| 4 | Phase 2 tokenization — **timeboxed to 2 weeks**, do not perfect it | 2 | **P2 bpe-lab** (a weekend) |
| 5 | Phase 4 — small GPT, trained. Keep the run small; do not chase quality | 3-4 | **P3 micro-gpt** [FLAGSHIP] |
| 6 | Phase 5 — inference and serving | 4-5 | **P5 jvm-llm-gateway** [FLAGSHIP] |
| 7 | Serving-lane efficiency block (see below) | 2-3 | Mock interview #1 |

**Deferred to months 7-11, deliberately:** Phase 8 (RAG), Phase 9 (agents),
Phase 10 (evals/security), Phase 11 (system design), and projects P4, P6, P9.

**Dropped from the six-month window entirely:** encoders/encoder-decoder,
multimodal, SSM/Mamba, the classical-ML refresher, and frameworks. These are
month 9+ material. They were on the previous version of this plan and did not
survive the hour budget.

### The serving-lane efficiency block (step 7)

Pulled forward out of Phase 6 because `POSITIONING.md` §3 lists quantization
tradeoffs and GPU memory/bandwidth among what inference-lane interviews screen
on. Take these lessons from `phase-06-07-efficiency-and-training.md` next to
Phase 5, and leave the rest of Phase 6-7 for later:

- 6.1-6.3 — bit layouts FP32/FP16/BF16, why BF16 won, FP8/INT8/INT4/NF4
- 6.4-6.5 — roofline and arithmetic intensity; why decode is bandwidth-bound
- 6.10 — quantization theory: affine vs symmetric, granularity
- 6.12 — GPTQ and AWQ
- 6.14-6.15 — weight-only vs weight+activation, PTQ vs QAT, measuring quality loss honestly
- 6.16 — KV-cache quantization

The matching project (**P6 quantbench**) comes in months 7-11. The lessons are
needed for interviews before the project is needed for the portfolio.

### Months 7-11 — the applied spine

Phase 8 RAG (less the four lessons pulled forward into Track 2) → Phase 9 agents
→ Phase 10 evals and security → Phase 11 system design, with **P9 ragprod** as
the third flagship. `PORTFOLIO.md` calls P9 "the
most commercially legible project in the Indian market… what 70% of applied-LLM
job descriptions describe," which is exactly why deferring it is a *bet* on the
serving lane rather than a free choice. Revisit at month 6: if your interviews
are coming back applied-AI-shaped rather than infra-shaped, pull P9 forward.

### Months 12+ — read-level depth

Remaining Phase 6 (FlashAttention, MoE, long context, distillation, pruning) and
Phase 7 (post-training, distributed). **Explain and reason; do not implement**,
except: LoRA from scratch (7.7), DPO (7.13), the memory-budget exercise (7.28).
Most of the rest needs NVIDIA hardware anyway.

---

## Cadence (every phase)

1. Timed coding drill from `interview/CODING_DRILLS.md`.
2. `/quiz` — 5 questions.
3. `NOTES.md` in your own words.
4. Update `interview/WEAK_SPOTS.md`.
5. `/recap` at the start of any session after a gap of several days.

Mock interviews after Phase 5, Phase 9 and Phase 11.

Supporting tracks:

- [interview/QUESTION_BANK.md](../interview/QUESTION_BANK.md) — drill with `/quiz`
- [interview/CODING_DRILLS.md](../interview/CODING_DRILLS.md) — timed from-scratch drills
- [interview/SYSTEM_DESIGN.md](../interview/SYSTEM_DESIGN.md) — `/sysdesign` from Phase 5 onward
- [projects/PORTFOLIO.md](../projects/PORTFOLIO.md) — build the project a phase unlocks before moving on
- [career/POSITIONING.md](../career/POSITIONING.md) — read early, revisit quarterly

---

## The phase map

Phase numbers here are canonical. `POSITIONING.md` and `PORTFOLIO.md` use this
same numbering.

| Phase | Title | Plan file | When |
| ---: | --- | --- | --- |
| 0 | Workspace & math warm-up | [phase-00-01-foundations.md](phase-00-01-foundations.md) | months 1-2 |
| 1 | Tensors, autograd, a net from scratch | [phase-00-01-foundations.md](phase-00-01-foundations.md) | months 1-3 |
| 2 | Tokenization & embeddings | [phase-02-03-transformer.md](phase-02-03-transformer.md) | month 3 (timeboxed) |
| 3 | Attention & the Transformer block | [phase-02-03-transformer.md](phase-02-03-transformer.md) | month 2-3 |
| 4 | Build a GPT & train it | [phase-04-05-gpt-and-inference.md](phase-04-05-gpt-and-inference.md) | month 4 |
| 5 | Inference & serving | [phase-04-05-gpt-and-inference.md](phase-04-05-gpt-and-inference.md) | months 5-6 |
| 6 | Efficiency & optimization | [phase-06-07-efficiency-and-training.md](phase-06-07-efficiency-and-training.md) | serving block at month 6; rest month 12+ |
| 7 | Post-training & distributed | [phase-06-07-efficiency-and-training.md](phase-06-07-efficiency-and-training.md) | month 12+, read-level |
| 8 | Retrieval & context engineering | [phase-08-09-rag-and-agents.md](phase-08-09-rag-and-agents.md) | months 7-9 |
| 9 | Agents & tool use | [phase-08-09-rag-and-agents.md](phase-08-09-rag-and-agents.md) | months 9-10 |
| 10 | Evaluation, reliability, security | [phase-10-11-evals-security-sysdesign.md](phase-10-11-evals-security-sysdesign.md) | month 10-11 |
| 11 | System design & career | [phase-10-11-evals-security-sysdesign.md](phase-10-11-evals-security-sysdesign.md) | month 11 |

---

## Why this order

**Foundations stay, but trimmed.** Autograd and a training loop still come
before Phase 4: every architecture choice from Phase 3 onward is a gradient-flow
argument. Broadcasting and shapes come before multi-head attention because the
head-splitting reshape is where a wrong answer raises no error.

**Inference (Phase 5) is the destination of the first six months**, not a
waypoint. It runs on the M3 and it is closest to what you already do: queueing,
batching, caching, admission control, tail latency. It is the fastest route to
being genuinely senior in *something* rather than a beginner in everything.

**P3 and P5 are the portfolio.** `PORTFOLIO.md`: "One says *I understand the
model*, the other says *I am the person who can put it in production*, and
together they are a positioning nobody else in your interview loop has."

**RAG is deferred, not dismissed.** It is the most over-represented topic and the
least differentiating, because everyone has done it. When it comes, it is done
in production terms with retrieval and generation metrics separated.

**Security and evaluation are their own phase.** "How do you know it works" and
"how does it get abused" separate people who have shipped from people who have
demoed.

---

## Hardware reality

Phases 0-5 and 8-11 run essentially entirely on the M3 with MPS.

These genuinely need NVIDIA hardware. Batch them into one or two rented sessions:

- bitsandbytes / QLoRA / NF4 (Phase 7)
- FlashAttention kernels — the *concept* is taught on CPU via online softmax
- vLLM / real PagedAttention (Phase 5)
- FP8, Triton, multi-GPU parallelism (Phases 6-7)

A T4 on Colab covers most of it free. Rent an A100 hour once, deliberately, so
that "I have run on CUDA" is true. Whole-portfolio budget: roughly $100-200.

---

## Timeline, honestly

**You cannot learn all of this in six months.** Lessons alone for Phases 0-5 and
8-11 exceed 208-312 hours before a single project is built, and `PORTFOLIO.md`
independently estimates 12-15 months for the full project list.

What six months buys, on the plan above:

- Phases 0-5 complete, plus the serving-lane efficiency block
- P1, P2, P3, P5 shipped and public
- One LLM thing shipped at work
- A resume and GitHub a stranger reads correctly in 30 seconds
- Real interview data from month 4 onward

That is a credible senior LLM-infra candidate.

Three failure modes, in order of likelihood:

- **Studying for six months without interviewing.** You optimise for feeling
  ready instead of being calibrated. Interview at month 4 regardless.
- **Collecting phases without building.** A phase "finished" with no project is
  nothing you can show. The projects are the deliverable.
- **Depth without scope → downlevelled to mid.** `POSITIONING.md` §3 names this
  exactly: "your LLM evidence is all small projects and none of it shows scope."
  Track 2 exists to fix it.

## How to run a session

1. `/progress` — where am I
2. `/recap` — 10 minutes on what is decaying (skip if you worked yesterday)
3. `/teach <topic>` — the lesson
4. `/quiz` — 5 questions on it
5. Write `NOTES.md` in the lesson directory, in your own words. Not optional.
