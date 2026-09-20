# Roadmap — from SDE-3 backend to LLM engineer

Twelve phases. Each has its own lesson-level plan file; this page is the map,
the ordering argument, and the honest timeline.

`curriculum/PROGRESS.md` is the source of truth for what is actually done.

---

## The shape of it

| Phase | Title | Plan file | Unlocks |
| ---: | --- | --- | --- |
| 0 | Workspace & math warm-up | [phase-00-01-foundations.md](phase-00-01-foundations.md) | Reading any PyTorch code without panic |
| 1 | Tensors, autograd, a net from scratch | [phase-00-01-foundations.md](phase-00-01-foundations.md) | Training loops; every later phase |
| 2 | Tokenization & embeddings | [phase-02-03-transformer.md](phase-02-03-transformer.md) | BPE from scratch; vocab tradeoffs |
| 3 | Attention & the Transformer block | [phase-02-03-transformer.md](phase-02-03-transformer.md) | MHA/GQA, RoPE, RMSNorm, SwiGLU |
| 4 | Build a GPT & train it | [phase-04-05-gpt-and-inference.md](phase-04-05-gpt-and-inference.md) | **Milestone.** Reading real model code |
| 5 | Inference & serving | [phase-04-05-gpt-and-inference.md](phase-04-05-gpt-and-inference.md) | KV cache, batching, vLLM, latency math |
| 6 | Efficiency & optimization | [phase-06-07-efficiency-and-training.md](phase-06-07-efficiency-and-training.md) | Quantization, FlashAttention, MoE |
| 7 | Post-training & distributed | [phase-06-07-efficiency-and-training.md](phase-06-07-efficiency-and-training.md) | SFT, LoRA, DPO/GRPO, FSDP |
| 8 | Retrieval & context engineering | [phase-08-09-rag-and-agents.md](phase-08-09-rag-and-agents.md) | Production RAG, index internals |
| 9 | Agents & tool use | [phase-08-09-rag-and-agents.md](phase-08-09-rag-and-agents.md) | Agent loops, MCP, multi-agent |
| 10 | Evaluation, reliability, security | [phase-10-11-evals-security-sysdesign.md](phase-10-11-evals-security-sysdesign.md) | Evals, LLM-as-judge, threat models |
| 11 | System design & career | [phase-10-11-evals-security-sysdesign.md](phase-10-11-evals-security-sysdesign.md) | The interview itself |

Supporting tracks, worked in parallel rather than at the end:

- [interview/QUESTION_BANK.md](../interview/QUESTION_BANK.md) — drill with `/quiz` after each phase
- [interview/SYSTEM_DESIGN.md](../interview/SYSTEM_DESIGN.md) — drill with `/sysdesign` from Phase 5 onward
- [projects/PORTFOLIO.md](../projects/PORTFOLIO.md) — build the project a phase unlocks *before* moving on
- [career/POSITIONING.md](../career/POSITIONING.md) — read early, revisit quarterly

---

## Why this order

This is not the order the topics are usually listed in, and the differences are
deliberate.

**Autograd and a full training loop come before Transformers** (Phase 1, not
Phase 5). The original plan put cross-entropy and backprop in the fine-tuning
phase. That is too late: attention is easier to understand once you have
watched gradients flow through a two-layer net you built yourself, and every
architectural choice from Phase 3 onward (residuals, pre-norm, initialisation)
is a *gradient-flow* argument. Without Phase 1 those choices have to be
memorised instead of derived.

**Build the whole GPT at Phase 4, before optimizing anything.** You cannot
reason about KV caches, quantization or LoRA until the thing they modify is
concrete. Phase 4 is the hinge of the curriculum — it is the point where the
rest stops being vocabulary.

**Inference before training-at-scale** (Phase 5 before Phase 7). Two reasons:
it runs on the M3, and it is the part of the stack closest to what you already
do. Serving is queueing, batching, caching, admission control and tail latency
— you have shipped systems like that. It is the fastest route to being
genuinely senior in *something* rather than a beginner in everything.

**RAG is Phase 8, not Phase 2.** Retrieval is the most over-represented topic
in LLM content and the least differentiating in interviews, because everyone
has done it. It is here, done properly and in production terms, and it is not
allowed to dominate.

**Security and evaluation are their own phase, not an appendix.** At staff
level these are frequently where the interview actually gets decided — "how do
you know it works" and "how does it get abused" separate people who have
shipped from people who have demoed.

---

## Hardware reality

Phases 0-5 and 8-11 run essentially entirely on the M3 with MPS.

The following genuinely need NVIDIA hardware. Batch them into one or two rented
sessions rather than renting per lesson:

- bitsandbytes / QLoRA / NF4 (Phase 7)
- FlashAttention kernels — the *concept* is taught on CPU via online softmax, the
  kernel is not (Phase 6)
- vLLM / real PagedAttention (Phase 5)
- FP8, Triton, multi-GPU parallelism (Phases 6-7)

A T4 on Colab covers most of it free. For LoRA on a 7-8B model, an A100 or
4090 hour on a rental is a few hundred rupees. Budget two or three rented
sessions across the whole curriculum.

---

## Timeline, honestly

Alongside a full-time SDE-3 job at 8-12 focused hours a week:

| Phases | Realistic elapsed |
| --- | --- |
| 0-1 (foundations) | 4-6 weeks |
| 2-3 (transformer) | 5-7 weeks |
| 4 (build a GPT) | 3-4 weeks |
| 5 (inference) | 4-5 weeks |
| 6-7 (efficiency, post-training) | 8-10 weeks |
| 8-9 (RAG, agents) | 5-7 weeks |
| 10-11 (evals, security, design) | 4-6 weeks |

That is roughly **8-11 months** to the end, with portfolio projects built along
the way rather than after.

Two failure modes to watch for, both common:

- **Rushing Phases 0-1** because they feel like they are not "real LLM work."
  Everyone who does this hits a wall at multi-head attention and has to come
  back. The shape gymnastics in Phase 3 are only hard if broadcasting is shaky.
- **Collecting phases without building.** A learner who has "finished" Phase 7
  with no project has nothing to show an interviewer. The projects are the
  deliverable; the phases are how you become able to build them.

## How to run a session

1. `/progress` — where am I
2. `/recap` — 10 minutes on what is decaying (skip if you worked yesterday)
3. `/teach <topic>` — the lesson
4. `/quiz` — 5 questions on it
5. Write `NOTES.md` in the lesson directory, in your own words. Not optional —
   this is the step that converts a lesson into something you can still explain
   in an interview six months from now.
