# LLM system design — the drill file

Used by `/sysdesign`. Three parts: the framework, the numbers you must know
cold, and the prompts.

This is the phase of the interview where your SDE-3 background is worth the
most. Most candidates for LLM roles can describe a RAG pipeline and cannot
size it, operate it, or say what it costs. You can. Lead with that.

---

## 1. The framework

Nine steps. Say the step names out loud as you go — interviewers grade
structure, and narrating it is free signal.

1. **Requirements and SLOs.** Who uses it, for what, how often. Then the
   numbers: p95 latency target, quality bar, budget, data sensitivity. *Do not
   design before this.* Candidates who start drawing boxes lose here.
2. **Traffic and token estimates.** QPS, peak:average ratio, input tokens per
   request, output tokens per request, concurrent sessions. Write them on the
   board. Every later decision refers back to these.
3. **Model choice and build-vs-buy.** API vs self-hosted vs fine-tuned.
   Justify with cost-per-million-tokens at *your* volume, latency, data
   residency, and how fast you need to move — not with "open source is better."
4. **Serving topology.** Gateway, queue, batching strategy, replica count,
   autoscaling, prefix cache, routing/cascade. This is your home turf.
5. **Retrieval and memory.** Only if the requirements demanded it. Index type,
   chunking, hybrid, reranking, freshness, and access control.
6. **Evaluation and guardrails.** How you know it works before and after
   shipping. Offline golden set, online signals, the thing you would roll back on.
7. **Observability.** Traces per request, token and cost attribution, quality
   sampling, alerting on a non-deterministic system.
8. **Cost.** ₹ or $ per request and per month, at the stated traffic. Then the
   lever you would pull first to halve it.
9. **Failure modes and scaling.** What breaks at 10x. Vendor outage, model
   deprecation, poisoned document, tenant leak, cost spike, quality regression.

**The closing move:** state unprompted what your design gives up. "This trades
p99 latency for throughput, and I would revisit it if the SLO tightened below
2 seconds." That sentence is worth more than another box on the diagram.

---

## 2. Numbers you must produce without notes

### KV cache size

```
bytes = 2 × n_layers × n_kv_heads × d_head × bytes_per_elem × seq_len × batch
        ^
        one for K, one for V
```

Note `n_kv_heads`, not `n_heads` — that distinction *is* what GQA buys, and
interviewers check whether you use the right one.

Worked: Llama-3-8B (32 layers, 8 KV heads, d_head 128), BF16, 8k context,
batch 32:

```
2 × 32 × 8 × 128 × 2 × 8192 × 32 ≈ 34 GB
```

The weights are ~16 GB. **The cache is bigger than the model.** That single
fact motivates GQA, paged attention, and KV quantization — if you can derive
it live, you have answered the next three questions before they are asked.

### Parameters and FLOPs

- Params ≈ `12 × n_layers × d_model²` for a standard decoder (attention +
  MLP), plus `vocab × d_model` for embeddings.
- Forward FLOPs ≈ `2 × params` per token. Training ≈ `6 × params` per token
  (forward + backward).
- Weight memory = `params × bytes_per_elem`. BF16 → 2 bytes. INT4 → 0.5.

### Prefill vs decode

- **Prefill** processes all prompt tokens in parallel: compute-bound, high
  arithmetic intensity, scales with prompt length. Determines **TTFT**.
- **Decode** produces one token at a time: memory-bandwidth-bound, terrible
  arithmetic intensity, reads the whole weight matrix per token. Determines
  **ITL** (inter-token latency).
- Therefore: batching helps decode enormously and prefill much less; a bigger
  batch raises throughput and raises per-request latency. That trade is the
  single most-asked serving question.

### Capacity

```
GPUs ≈ (QPS × avg_output_tokens) / (tokens_per_sec_per_GPU at your batch size)
```

Then check memory separately: weights + KV cache at target batch and context
must fit, or the batch size you assumed is fiction.

### Cost

Know the break-even shape, not the exact vendor prices (they change):
self-hosting wins above a volume threshold set by GPU-hour cost ÷ utilisation.
Below ~continuous utilisation of a GPU, APIs are usually cheaper *and* cheaper
to operate. Say the threshold, then say you would verify current prices.

---

## 3. The rubric

`/sysdesign` grades these 1-5 with evidence. Know what a 5 is.

| Dimension | A 5 looks like |
| --- | --- |
| Requirements | Elicited SLOs, scale and quality bar before designing |
| Estimation | Real numbers on the board: QPS, tokens, KV cache, GPUs, cost |
| Model choice | Build-vs-buy defended on cost and quality at the stated volume |
| Serving | Batching, caching, routing, admission control, backpressure |
| Retrieval | Chosen over alternatives for a reason; failure modes named |
| Evaluation | Success defined before shipping; offline and online |
| Security | Threat-modelled injection, tenancy, exfiltration, agency |
| Operations | Canary, rollback, observability for a non-deterministic system |
| Tradeoffs | Stated what the design gives up, unprompted |

---

## 4. Prompts

Each is deliberately underspecified. The elicitation is the first test.

### D1 — Support agent over private docs
10k enterprise seats, each tenant's documents strictly isolated.
*Carries the interview:* tenancy and access control in retrieval; the quality
bar and how it is measured; escalation to a human; cost per conversation.
*Trap:* designing the RAG pipeline before asking about tenancy, then having to
retrofit access control into the index.

### D2 — Multi-tenant inference gateway
One platform team serving many product teams on shared GPUs.
*Carries the interview:* fair scheduling and noisy neighbours; quotas and rate
limits; prefix-cache sharing across tenants and whether that leaks; routing
across model tiers; chargeback.
*Trap:* ignoring that a shared prefix cache is a side channel between tenants.

### D3 — Code review bot on every PR
*Carries the interview:* context selection from a large repo under a token
budget; precision over recall (a noisy bot gets muted and is then worthless);
latency versus PR workflow; evaluating "is this comment useful".
*Trap:* optimising recall. The product failure mode is annoyance, not misses.

### D4 — Document extraction at 1M docs/day
Structured fields out of messy PDFs.
*Carries the interview:* batch versus online; it is a throughput problem, not
a latency one; constrained decoding for schema validity; confidence scoring and
human review routing; reprocessing when the schema changes.
*Trap:* treating it as an online serving problem and over-provisioning.

### D5 — Real-time voice agent
*Carries the interview:* the end-to-end latency budget across STT → LLM → TTS;
streaming and speculative response; barge-in and turn-taking; TTFT dominating
perceived quality; what degrades gracefully when you miss the budget.
*Trap:* not producing an actual millisecond budget per hop.

### D6 — Eval platform for the whole org
*Carries the interview:* dataset versioning and provenance; judge calibration
against humans; statistical significance; making it fast enough for CI; who
owns the golden set.
*Trap:* proposing LLM-as-a-judge without calibration or a bias discussion.

### D7 — Migrate from a vendor API to self-hosted
*Carries the interview:* the break-even calculation; the quality regression
risk and how you would detect it; shadow traffic and gradual cutover; on-call
and operational burden you are taking on; the rollback path.
*Trap:* arguing from cost alone and ignoring the operating cost of owning GPUs.

### D8 — Semantic search over 500M documents
*Carries the interview:* index choice (HNSW memory versus IVF-PQ recall);
sharding; incremental updates and deletes; hybrid with BM25; reranking budget;
recall measurement.
*Trap:* quoting a vector DB brand instead of reasoning about the index.

### D9 — Autonomous agent with write access
It can file tickets, send email, and change records.
*Carries the interview:* blast radius and least privilege; approval gates;
idempotency and retries; indirect prompt injection from the content it reads;
audit trail; the per-step reliability math over a long trajectory.
*Trap:* treating prompt injection as a prompting problem rather than an
architecture problem.

### D10 — Cut inference cost by 70% without losing users
*Carries the interview:* the ordered list of levers — caching, routing and
cascades, quantization, batching, shorter prompts, a smaller fine-tune — each
with its quality risk; how you would measure the quality cost of each; what you
would do first and why.
*Trap:* jumping to quantization. Caching and routing are usually cheaper wins
with less quality risk.

---

## 5. How to answer at staff level

- **Lead with the tradeoff**, not the component. "I would batch aggressively
  here, which costs me p99 but triples throughput — acceptable because the SLO
  is 5 seconds."
- **Quantify.** An unquantified design is an opinion.
- **Name the failure mode** before the interviewer does.
- **Say what you would measure.** Every claim you make should come with the
  metric that would prove or disprove it.
- **Say "it depends" only with the dependency.** "It depends on whether this is
  latency- or throughput-bound, and here is how I would find out."
