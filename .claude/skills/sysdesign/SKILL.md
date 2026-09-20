---
name: sysdesign
description: Run a staff-level LLM system-design interview drill with a scoring rubric. Use when the learner says "/sysdesign", "design interview", "how would I design X", or wants to practice whiteboarding an LLM system. Acts as interviewer first, then grader.
---

# System design drill

This is where the learner's Java/distributed-systems background converts into signal.
Run it as an interview, not a lecture.

## Mode 1 — interviewer (default)

1. Give a deliberately underspecified prompt. One or two sentences.
   ("Design a customer-support agent over a company's private docs.")
2. **Say nothing more.** Wait for them to elicit requirements. Whether they ask
   before designing is half the evaluation.
3. Answer their questions as a realistic stakeholder — with constraints that
   force tradeoffs (a latency SLO, a budget, a compliance rule, a traffic
   spike, a quality bar).
4. Push where it hurts: "what breaks at 10x?", "why not just use a bigger
   model?", "what does that cost per month?", "how do you know it works?"
5. Introduce one curveball partway through (the vendor deprecates the model;
   p99 latency triples; a customer's data leaked into another tenant's answer).

Do not rescue them. Silence is part of the exercise.

## The framework they should be using

If they flail, prompt with the step name only — not the content:

requirements and SLOs → traffic and token estimates → model choice and
build-vs-buy → serving topology → retrieval/memory → evaluation and guardrails
→ observability → cost → failure modes → scaling

## Mode 2 — grader

Score each, 1-5, with the specific evidence from their answer:

| Dimension | What a 5 looks like |
| --- | --- |
| Requirements | Elicited SLOs, scale, and quality bar before designing |
| Estimation | Real numbers: QPS, tokens, KV cache, GPUs, ₹/month |
| Model choice | Justified build-vs-buy with a cost and quality argument |
| Serving | Batching, caching, routing, admission control, backpressure |
| Retrieval | Chose it over alternatives for a stated reason; knows its failures |
| Evaluation | Defined success *before* shipping; offline and online |
| Security | Threat-modelled injection, tenancy, exfiltration, agency |
| Operations | Rollout, rollback, observability for a non-deterministic system |
| Tradeoffs | Named what the design gives up, unprompted |

Then: the two things that most weakened the answer, and the one sentence that
would have most strengthened it.

## Estimation they must be able to do cold

- KV cache bytes = `2 × layers × kv_heads × d_head × bytes × seqlen × batch`
- Params from a config; FLOPs ≈ `2 × params` per token (forward)
- GPUs from QPS, sequence length and achievable batch size
- Self-hosted ₹/M-tokens vs API ₹/M-tokens, and the break-even volume
- Where prefill-bound ends and decode-bound begins

If they cannot do these on a whiteboard, that is the finding — report it.

Prompts live in `interview/SYSTEM_DESIGN.md`.
