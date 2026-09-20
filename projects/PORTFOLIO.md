# projects/PORTFOLIO.md — the evidence track

This is the half of the workspace that a hiring manager can actually read. The
lessons produce depth; this file produces **proof of depth**.

**Phase numbers refer to `curriculum/ROADMAP.md` (Phases 0-11)**, which is
canonical. `career/POSITIONING.md` uses the same numbering. Phase 0-1
foundations · 2 tokenization · 3 attention · 4 build a GPT · 5 inference ·
6 efficiency · 7 post-training · 8 RAG · 9 agents · 10 evals & security ·
11 system design.

**In the first six months, build P1, P2, P3 and P5 only.** The rest are
months 7+. See the six-month plan in `ROADMAP.md`.

**The rule that makes these credible:** none of these projects is novel. Every
one of them has a thousand GitHub clones. The differentiator is never the code —
it is (a) a number you measured yourself, (b) a tradeoff you can defend, (c) a
writeup that admits what did *not* work. A repo with a benchmark table and an
honest "what I got wrong" section beats a repo with twice the features.

**The three flagships** are P3 (micro-GPT), P5 (JVM inference gateway) and P9
(production RAG). Those get real polish. The rest are proof artifacts: ship
them, write 400 words, move on.

---

## What you can claim after this phase

- "I implemented reverse-mode autodiff, a BPE tokenizer, a decoder-only
  Transformer, a KV-cached inference loop and a continuous-batching scheduler
  from scratch, and I can show you the tests that pin them against PyTorch."
- "I can put a number on the latency, throughput, memory and quality cost of a
  serving decision, and tell you which one I'd trade away for which SLO."
- "I have run a real fine-tune with a real eval and can tell you when *not* to
  fine-tune, because I measured the case where it lost to prompting."
- "I built the Java-side serving layer for an LLM backend with admission
  control and per-tenant token budgets — the part most LLM engineers hand-wave."
- "I can separate retrieval quality from generation quality in a RAG system and
  show which one is actually failing."
- "I can red-team my own agent and show the trajectory-level eval that catches
  the regression."

---

## P1 — micrograd-tensor: a reverse-mode autodiff engine

- **Unlocked after:** Phase 1
- **One-line pitch:** A ~400-line autodiff engine over n-dimensional arrays,
  with broadcasting-correct gradients, that trains an MLP and matches PyTorch's
  gradients to 1e-6 on a property-based test suite.
- **What it proves:** You understand backprop as a *program transformation*, not
  as `loss.backward()`. This is the single cheapest way to kill the "backend guy
  who did a course" suspicion in the first ten minutes of an interview.
- **Scope — build this:**
  - A `Tensor` class wrapping NumPy with `.grad`, a parent list, and a local
    backward closure per op.
  - Ops: `add`, `mul`, `matmul`, `sum`, `mean`, `relu`, `exp`, `log`, `max`,
    plus softmax + cross-entropy as a *fused, numerically stable* op.
  - Correct gradient reduction through broadcasting (the part everyone gets
    wrong — sum over broadcast axes, then reshape).
  - Topological sort for the backward pass; an explicit test that a diamond-
    shaped graph accumulates rather than overwrites.
  - `tests/test_autograd.py`: random-shape property tests comparing against
    `torch.autograd.grad` and against central finite differences.
  - Train a 2-layer MLP on a 2-D toy classification set; plot the loss curve.
- **Explicitly out of scope:** GPU, convolutions, RNNs, a `nn.Module` system, an
  optimizer zoo (SGD + one momentum variant is enough), graph visualisation.
- **Hardware:** M3/MPS (actually CPU/NumPy — that is the point).
- **Measure and report:**
  - Max absolute gradient error vs PyTorch, per op, as a table.
  - Max relative error of the fused softmax-CE vs a naive `log(softmax(x))`
    implementation at logits of magnitude 1e3 — show the naive one produce
    `nan`/`inf` and yours not.
  - Wall-clock for one train step vs the same model in PyTorch (you will be
    10-50x slower; say so and say why).
- **The writeup:** (1) Why the backward of a broadcast is a sum, derived once
  with shapes. (2) Why softmax and cross-entropy are fused in every real
  framework, with the overflow demonstration. (3) Why the topological sort is
  required and what breaks without it. (4) What your engine cannot do that
  PyTorch can (in-place ops, kernel fusion, `no_grad`) and the cost of each.
- **Interview story:** "I didn't trust that I understood backprop, so I wrote the
  engine. The thing that actually taught me something was broadcasting — the
  forward pass silently expands a `(3,)` bias to `(32,3)` and the backward pass
  has to sum those 32 contributions back down, and if you don't, your gradients
  are 32x too large and your loss diverges in a way that looks like a bad
  learning rate." **Be ready for:** "derive the gradient of matmul for both
  operands" and "why is the softmax Jacobian not needed explicitly when you fuse
  it with cross-entropy?"
- **Time estimate:** 12-16 h.

---

## P2 — bpe-lab: a BPE tokenizer and a fertility/pathology report

- **Unlocked after:** Phase 2 (tokenization)
- **One-line pitch:** A from-scratch byte-level BPE trainer and encoder, plus a
  measurement report on what production tokenizers do to Indian-language and
  source-code text — cost, context and correctness consequences included.
- **What it proves:** You treat the tokenizer as a systems component with a cost
  model, not as a preprocessing detail. Almost nobody does the fertility
  analysis, and it is directly commercially relevant to any India-market product.
- **Scope — build this:**
  - Byte-level BPE training (merge-frequency loop over a byte alphabet, so there
    is no UNK and no unicode-normalisation trap) on a small corpus, vocab 4k-8k.
  - An encoder with a pre-tokenisation regex split, plus a decoder; a round-trip
    fuzz test over random unicode including emoji, ZWJ sequences and Devanagari
    conjuncts.
  - A naive O(n) merge loop *and* a priority-queue/linked-list version; benchmark
    both on the same input.
  - A **fertility harness**: tokens-per-word and tokens-per-character for a fixed
    parallel text across English, Hindi, Tamil/Telugu (pick two), JSON, and
    Python source, across your tokenizer + 3 production ones
    (`transformers` `AutoTokenizer`: a Llama-family, a Qwen-family, and `tiktoken`
    `o200k_base`).
  - A **pathology section**: number-splitting behaviour (tokenize `1234`, `12345`,
    `£1,234.56`), leading-whitespace duplicates, and the tokens where
    `decode(encode(x)) != x` if any survive.
- **Explicitly out of scope:** SentencePiece unigram LM, training a model on your
  vocab (that is P3), tokenizer-free byte models, morphology-aware merges,
  a Rust rewrite.
- **Hardware:** M3/MPS — CPU only.
- **Measure and report:**
  - Fertility (tokens/word) per language per tokenizer, as one table. Anchor:
    published work puts English near ~1.2-1.4 tokens/word, Hindi near ~1.4, and
    Tamil/Telugu/Kannada/Malayalam in the ~2.1-2.9 range for common BPE
    tokenizers — reproduce or contradict that with *your* corpus and say which.
  - The derived cost claim: at a fixed per-token price, serving Tamil costs
    N times what English costs for the same semantic content. Compute N.
  - The derived context claim: effective context window in *words* per language.
  - Train time and encode throughput (chars/s) for naive vs indexed merge loop.
- **The writeup:** (1) Fertility is a latency, cost *and* context-length tax, and
  here is the multiplier for each language. (2) Byte-level BPE removes UNK but
  does not remove the tax — explain why vocabulary allocation, not coverage, is
  the mechanism. (3) Number tokenisation is a correctness bug, not an efficiency
  bug; show a case. (4) When you would pay to extend a vocabulary and when the
  embedding-table and softmax cost makes it a bad trade.
- **Interview story:** "I measured what a standard tokenizer costs in Indian
  languages. Same paragraph, Tamil takes roughly 2x the tokens of English, which
  means 2x the price, 2x the prefill FLOPs and half the usable context. That
  changes the model-selection decision for an India-facing product, and it's
  invisible unless you measure it." **Be ready for:** "so extend the vocab — what
  breaks?" (embedding + LM-head params grow, softmax cost grows, the new tokens
  are untrained, and you now need continued pretraining) and "what is the
  complexity of BPE training and encoding?"
- **Time estimate:** 18-24 h.

---

## P3 — micro-gpt: a decoder-only Transformer, trained, with ablations  **[FLAGSHIP]**

- **Unlocked after:** Phase 4
- **One-line pitch:** A ~15M-parameter decoder-only LM written from scratch —
  RoPE, RMSNorm, SwiGLU, GQA — trained on TinyStories on an M3, with an ablation
  table and a FLOPs/memory accounting that matches the measured throughput.
- **What it proves:** You can build the architecture, *and* you can reason about
  it quantitatively. The ablations are what separate this from the ten thousand
  nanoGPT forks.
- **Scope — build this:**
  - The model: token embedding, RoPE applied to Q and K only, pre-norm RMSNorm,
    causal GQA, SwiGLU MLP, weight-tied LM head. No `nn.MultiheadAttention`.
  - A correctness harness: your attention vs
    `F.scaled_dot_product_attention` to 1e-5; your RMSNorm vs a reference;
    a test that the causal mask actually blocks (perturb token t, assert
    logits at positions < t are bit-identical).
  - A training loop with cosine schedule + warmup, grad clipping, AdamW, mixed
    precision, and a deterministic seed. Train on TinyStories with the P2
    tokenizer.
  - A **param and FLOPs counter** you wrote: predicted params vs
    `sum(p.numel())`, and predicted training FLOPs via the ~6·N·D rule vs
    measured tokens/s.
  - **Four ablations**, each a single run with the same token budget:
    RoPE vs learned absolute positions; RMSNorm vs LayerNorm; pre-norm vs
    post-norm; MHA vs GQA(g=2) vs MQA.
  - Generation: greedy, temperature, top-k, top-p, with sample output for each.
- **Explicitly out of scope:** matching any published benchmark, >100M params,
  multi-GPU, FlashAttention kernels, MoE, distributed data loading, a "chat"
  fine-tune (that is P7).
- **Hardware:** M3/MPS for everything up to ~15-20M params. One optional
  confirmation run at ~50M params on a rented A100 (~8 h, ~$10-15 at community
  A100 PCIe rates around $1.19/GPU-hr) if you want a second point on the scaling
  line. MPS caveat: expect silent CPU fallback on some ops and different
  numerics from CUDA — log `tensor.device` in the training loop, not just at
  setup.
- **Measure and report:**
  - Final val loss and val perplexity per ablation, at equal tokens seen.
  - Tokens/s and MFU-style utilisation on M3, with your FLOPs estimate shown.
  - Peak memory, broken into params / gradients / optimizer state / activations,
    predicted *before* measuring and then compared.
  - Wall-clock hours and, for the rented run, dollars.
  - Post-norm divergence: show the run that blew up, with the loss curve.
- **The writeup:** (1) The memory breakdown of AdamW training is ~16 bytes/param
  in mixed precision — derive it and show your measurement agreeing. (2) Pre-norm
  is not a style choice; here is the post-norm run that diverged and the residual-
  stream argument for why. (3) GQA buys KV-cache memory at a measurable quality
  cost — here is mine at this scale, and here is why the trade gets *better* at
  larger scale. (4) The honest one: what a 15M model on TinyStories does and does
  not tell you about a 7B model.
- **Interview story:** "I built the whole decoder from scratch and then ablated
  it. The result I didn't expect was how small the RMSNorm-vs-LayerNorm quality
  gap is — it's essentially a throughput optimisation that costs nothing, not an
  accuracy win. The one that mattered was pre-norm: post-norm at this depth
  without careful init just diverged." **Be ready for:** "walk me through the
  shapes from `(B,T)` token IDs to logits"; "why is RoPE applied to Q and K but
  not V?"; "your GQA saves KV memory — quantify it for a 70B model at 8k
  context."
- **Time estimate:** 40-55 h. This is the one worth polishing.

---

## P4 — kvlab: a KV cache and continuous-batching scheduler from scratch

- **Unlocked after:** Phase 5
- **One-line pitch:** A single-process inference engine for the P3 model with a
  paged KV cache and a continuous-batching scheduler, benchmarked against static
  batching and against vLLM, with a latency/throughput frontier plot.
- **What it proves:** You understand LLM serving as a *scheduling and memory*
  problem. This is the project where your distributed-systems background stops
  being a story and starts being visible in the code.
- **Scope — build this:**
  - Prefill/decode split with a correct KV cache; a test asserting cached and
    uncached generation produce identical token IDs.
  - A **block-paged** KV allocator (fixed-size blocks, a free list, a per-
    sequence block table) — i.e. PagedAttention's memory model, without the
    fused CUDA kernel. Report internal fragmentation vs a contiguous allocator.
  - A scheduler: an arrival queue, admission control against a KV-block budget,
    per-step batch formation that admits new requests as others finish, and
    preemption (recompute-or-swap) when the cache is full. Make the policy
    pluggable: FCFS vs shortest-remaining-first.
  - A load generator with Poisson arrivals and a realistic prompt/output length
    distribution, reporting p50/p95/p99 — not means.
  - Prefix caching for a shared system prompt, with the hit-rate measured.
- **Explicitly out of scope:** custom CUDA/Triton kernels, multi-GPU, disaggregated
  prefill/decode, speculative decoding (stretch only), beating vLLM.
- **Hardware:** M3/MPS for the engine itself and for all scheduling experiments —
  the scheduling story is hardware-independent and that is the point. One rented
  A100 session (~6-10 h, ~$10-15) to run the vLLM baseline comparison on
  identical request traces. Say explicitly in the README which numbers came from
  which device.
- **Measure and report:**
  - Throughput (output tok/s) vs arrival rate, for static batching / your
    continuous batching / vLLM, on one plot.
  - TTFT and TPOT at p50/p95/p99 per configuration.
  - KV memory used vs theoretical minimum → fragmentation %, contiguous vs paged.
  - Queueing delay vs service time decomposition at the knee of the curve.
  - Prefix-cache hit rate and the TTFT reduction it buys.
  - Your KV-cache size formula
    (`2 · layers · kv_heads · d_head · seq · batch · bytes`) evaluated for a 7B
    and a 70B model at 8k and 128k context, next to the measured number for yours.
- **The writeup:** (1) Static batching wastes GPU because sequences finish at
  different times; here is the utilisation gap I measured. (2) Paging trades a
  block table lookup for the elimination of external fragmentation — here is
  the fragmentation number both ways. (3) Decode is memory-bandwidth-bound, not
  compute-bound; here is the arithmetic-intensity argument and the measurement
  that supports it. (4) Head-of-line blocking: what FCFS does to p99 when one
  4k-token prompt arrives, and what the alternative policy costs in fairness.
- **Interview story:** "Continuous batching is admission control plus a per-step
  scheduler — it's the same problem as a connection pool with variable-duration
  work, except the resource being scheduled is KV-cache blocks rather than
  threads. I built both and measured the throughput gap, then ran the same
  request trace through vLLM to see how far off I was." **Be ready for:**
  "why don't you cache Q?"; "what happens to p99 when you raise max batch size?";
  "you're out of KV memory mid-decode — preempt or swap, and why?"
- **Time estimate:** 35-45 h.

---

## P5 — jvm-llm-gateway: a Java serving tier with real SLOs  **[FLAGSHIP — your differentiator]**

- **Unlocked after:** Phase 5 (buildable in parallel with P4; it consumes P4 or a
  vLLM backend)
- **One-line pitch:** A Spring Boot / reactive gateway in front of an LLM backend
  that does token-budgeted rate limiting, SLO-aware admission and queueing,
  streaming with correct cancellation propagation, and prefix-cache-aware
  routing across replicas — with a load test proving the SLO holds under
  overload.
- **What it proves:** That you are not an LLM hobbyist who also knows Java. You
  are a distributed-systems engineer who understands what is *different* about
  an LLM backend: unbounded, variable-duration, streaming, memory-constrained
  requests where the cost unit is a token and not a call. Very few candidates in
  the India market can hold both halves of that sentence. **Make this repo as
  good as P3.**
- **Scope — build this:**
  - A Spring Boot (WebFlux or virtual threads) gateway exposing an
    OpenAI-compatible `/v1/chat/completions` with SSE streaming, backed by P4 or
    vLLM. Use Spring AI or LangChain4j for the client layer *only* if it earns
    its place — note that Spring AI 2.0 (GA June 2026) requires Spring Boot 4 /
    Framework 7, which is a platform decision, not a dependency bump; LangChain4j
    is the framework-agnostic alternative. Say which you chose and why.
  - **Token-bucket rate limiting denominated in tokens, not requests**, with the
    hard part made explicit: you must *estimate* output tokens at admission time
    and reconcile against actual usage at completion. Show the estimator's error
    distribution and what over/under-estimation does to fairness.
  - **Admission control + a bounded queue with deadline-aware shedding**: reject
    fast with a `Retry-After` rather than queueing a request whose SLO is already
    unachievable. Little's Law sizing for the queue, shown in the README.
  - **Cancellation propagation**: client disconnects mid-stream → the gateway
    must actually abort the backend generation and free its KV blocks. Prove it
    with a metric (backend tokens generated after client disconnect ≈ 0).
  - **Prefix-cache-aware routing**: hash the system-prompt prefix and route to
    the replica likely to have it warm; measure the TTFT delta vs round-robin.
  - Resilience4j circuit breaker + bulkhead + a hedged-request policy for the
    tail, with the hedge's extra cost quantified. Micrometer → Prometheus →
    a Grafana board showing TTFT/TPOT/queue depth/tokens-per-tenant.
- **Explicitly out of scope:** writing your own inference kernel in Java, a
  Kubernetes operator, auth/billing beyond a tenant header, multi-region,
  a web UI, gRPC as well as HTTP (pick one).
- **Hardware:** M3/MPS for the gateway and for all the load tests (point it at
  P4's engine or a small `llama.cpp`/MLX backend — the queueing behaviour is what
  you are measuring, and it reproduces at any token rate). One rented A100
  session (~10-15 h, ~$15-25) if you want the multi-replica prefix-routing
  numbers on real hardware.
- **Measure and report:**
  - TTFT / TPOT / end-to-end latency at p50/p95/p99, under load below, at, and
    above capacity.
  - The overload curve: goodput vs offered load, with and without admission
    control. The claim to earn: goodput stays flat past saturation instead of
    collapsing.
  - Token-estimation error: predicted vs actual output tokens, distribution and
    p95 absolute error, and the resulting rate-limit unfairness across tenants.
  - Wasted backend tokens after client cancellation: before vs after your fix.
  - Prefix-aware vs round-robin routing: TTFT p50/p95 delta and cache hit rate.
  - Gateway overhead: added latency and allocation rate per request (and, if you
    use WebFlux, what happens to it at 10k concurrent streams).
- **The writeup:** (1) An LLM backend is not an HTTP backend: request cost is
  unknown at admission time and varies by 100x, which breaks every request-count
  rate limiter — here is the token-denominated design and its estimation error.
  (2) Under overload, queueing is worse than shedding, because a queued LLM
  request holds KV memory; here is the goodput curve both ways. (3) Streaming
  makes cancellation a correctness issue, not a hygiene issue — an unaborted
  stream burns GPU for a client that left; here is the measured waste. (4) What
  the JVM specifically costs and buys you here (GC pauses vs p99, virtual threads
  vs reactive for long-lived streams) — measured, not asserted.
- **Interview story:** "Most LLM gateways rate-limit by requests per minute,
  which is meaningless when one request can cost 200x another. I built the
  gateway to meter tokens, which forces you to estimate output length at
  admission — so I measured the estimator's error and showed what the error does
  to per-tenant fairness. The second thing nobody handles is cancellation: when
  a client disconnects mid-stream the backend keeps generating and keeps holding
  KV blocks. I measured the waste and fixed the propagation." **Be ready for:**
  "how do you size the queue?" (Little's Law, from the TPOT distribution);
  "why not just autoscale?" (cold start on model weights, and GPU supply);
  "virtual threads or reactive here, and why?"
- **Time estimate:** 45-60 h. Your existing Java depth makes this the highest
  return-per-hour project in the list.

---

## P6 — quantbench: an honest quality-vs-latency-vs-memory table

- **Unlocked after:** Phase 6
- **One-line pitch:** A reproducible harness that quantizes one model several
  ways and reports the full tradeoff surface — including the configurations that
  were not worth it.
- **What it proves:** Measurement discipline, and the judgement to publish a
  negative result. This is the project that demonstrates you will not ship a 4-bit
  model into production because a blog said it was "nearly lossless".
- **Scope — build this:**
  - One base model (a 1-3B instruct model) across: FP16/BF16 baseline,
    weight-only INT8, weight-only INT4 (GPTQ **and** AWQ), and a GGUF Q4_K_M
    from `llama.cpp`. Same eval, same prompts, same seeds, every time.
  - A quality panel that is *not* just perplexity: perplexity on held-out text,
    one multiple-choice benchmark via `lm-evaluation-harness`, and a generative
    task scored against references. Report all three, because they disagree.
  - Per-layer sensitivity: quantize one layer group at a time and plot which
    parts of the network actually degrade (attention out-proj and the first/last
    blocks are the usual suspects — confirm or refute on your model).
  - An outlier study: histogram activation magnitudes per layer and connect the
    long tail to why per-channel scaling / SmoothQuant-style migration exists.
  - A one-page decision table: "at this SLO and this memory budget, pick X."
- **Explicitly out of scope:** writing quantization kernels, FP8, QAT,
  quantizing a 70B model, KV-cache quantization (stretch only), beating any
  published number.
- **Hardware:** **Mostly needs a rented GPU.** GPTQ/AWQ toolchains and
  `bitsandbytes` are CUDA-only — `bitsandbytes` does not build for MPS at all,
  and the MPS path produces NaN losses in several PEFT/quant stacks. Budget
  ~12-20 GPU-hours on an A100 PCIe (~$1.19/GPU-hr community tier, so roughly
  **$20-40** including false starts). The M3-native path that still teaches the
  idea: `llama.cpp` GGUF quantization and MLX 4-bit, both first-class on Metal —
  run the latency/memory half locally and the GPTQ/AWQ half rented. Report the
  two device families in separate columns; never average across them.
- **Measure and report:**
  - For each config: model size on disk, peak GPU/unified memory, TTFT, TPOT,
    throughput at batch 1 and batch 16, perplexity, benchmark accuracy,
    generative score.
  - Quality *delta from baseline with an uncertainty estimate* — run the eval 3x
    with different seeds and report the spread, so a 0.3-point "win" is visibly
    inside the noise.
  - Cost-per-million-tokens for each config at the rented-GPU hourly rate.
  - The per-layer sensitivity plot.
- **The writeup:** (1) Perplexity is a bad proxy for the thing you care about —
  here is a config where perplexity moved 1% and the generative score moved 8%.
  (2) INT4 is a memory-bandwidth win before it is a memory-capacity win, which is
  why it helps decode more than prefill — show the batch-1 vs batch-16 split.
  (3) Activation outliers are the mechanism behind quantization damage, not
  "bits are lossy"; here is the histogram. (4) The honest recommendation, with
  the configuration you would *not* ship and why.
- **Interview story:** "I quantized the same model six ways and measured quality
  three ways, and the three quality metrics ranked the configurations
  differently. Perplexity said INT4 was fine; the generative eval said it wasn't.
  That's the whole project — if you only measure perplexity you will ship a
  regression." **Be ready for:** "why is INT4 weight-only fine but INT4
  activations not?"; "when does quantization *not* speed you up?" (compute-bound
  prefill at large batch, where you pay dequantization for nothing).
- **Time estimate:** 25-35 h.

---

## P7 — adapt-lab: LoRA SFT + DPO with an eval that can say "this didn't work"

- **Unlocked after:** Phase 7
- **One-line pitch:** A narrow capability taught to a small instruct model via
  LoRA SFT and then DPO, evaluated against a prompt-only baseline on a held-out
  set — with the rank/target-module ablation and the cost accounting.
- **What it proves:** That you can run the full alignment pipeline *and* that you
  know fine-tuning is a cost decision with a frequently-negative return. The
  "when not to fine-tune" judgement is what senior interviewers probe for.
- **Scope — build this:**
  - Pick one narrow, checkable task (structured extraction to a JSON schema, or a
    domain-specific rewrite) with a programmatically gradable eval — do not pick
    "helpfulness".
  - Build ~500-2000 SFT examples and ~300-800 preference pairs. Document how you
    built them, including the contamination check against your eval set.
  - LoRA SFT: ablate rank {4,16,64} and target modules {attn-only, attn+MLP},
    holding token budget fixed. Report parameter count trained each time.
  - DPO on top of the SFT checkpoint. Log the implicit reward margin and the
    KL from the reference — and show what happens when beta is too low
    (reward hacking / degenerate outputs).
  - **Three baselines you must beat:** zero-shot, few-shot prompting, and
    few-shot + a constrained-decoding/JSON-schema enforcer. If the fine-tune does
    not beat baseline 3 on quality-per-rupee, *say so in the README*.
  - Merge the adapter, re-run the eval post-merge (catch the merge bug), and
    measure the inference-time cost of adapter-vs-merged serving.
- **Explicitly out of scope:** full-parameter fine-tuning, PPO/GRPO, RLHF with a
  learned reward model, 7B+ models, multi-node, chasing a leaderboard.
- **Hardware:** **Rented GPU for the main runs.** QLoRA on MPS is not viable
  (`bitsandbytes` is CUDA-only); plain LoRA on MPS works but is bandwidth-bound
  and slow. Budget ~20-40 GPU-hours on an A100 (~**$25-60** at ~$1.19-1.49/GPU-hr),
  or an H100 at ~$1.99-2.89/hr if you want the runs to finish overnight. The
  M3-native path that still teaches the idea: `mlx-lm lora` fine-tunes a 1B model
  on unified memory in minutes and is genuinely first-class on Apple Silicon —
  use it to debug the data pipeline and the eval before you spend a rupee on
  rented compute.
- **Measure and report:**
  - Task metric for all baselines + every fine-tune config, on a held-out set,
    with a confidence interval.
  - Trainable params, peak memory and wall-clock per config.
  - **Total rupees spent per point of eval improvement**, versus the rupees the
    few-shot baseline costs per 1000 requests at inference. This is the number
    that makes the project senior-level.
  - A regression check: does the fine-tune degrade a general-capability probe?
    (Measure it. It usually does.)
  - DPO: win rate vs the SFT checkpoint, plus the beta sweep showing degeneration.
- **The writeup:** (1) LoRA rank past a point buys nothing on a narrow task —
  here is the curve and where it flattens. (2) DPO optimises a *preference*, not
  a capability; here is what it improved and what it quietly broke. (3) The
  economics: at my volume the fine-tune pays back after N requests; below that,
  prompting plus constrained decoding wins. (4) The data was the bottleneck, not
  the method — here is the contamination check and what it caught.
- **Interview story:** "I ran SFT then DPO on a narrow extraction task and
  benchmarked it against few-shot prompting with schema-constrained decoding.
  The fine-tune won on quality, but it only pays back above roughly N requests a
  month, and it cost me a measurable regression on general instruction-following.
  I'd ship the constrained-decoding baseline first." **Be ready for:** "why does
  DPO not need a reward model?"; "what does LoRA's alpha actually scale and why
  do people set alpha = 2·rank?"; "how did you check your eval set wasn't in your
  training data?"
- **Time estimate:** 40-50 h (most of it data, not training).

---

## P8 — agentlab: an agent harness, an MCP server, and a trajectory eval

- **Unlocked after:** Phase 9
- **One-line pitch:** A tool-using agent with a typed tool layer exposed over
  MCP, plus an evaluation that scores the *trajectory* — tool choice, argument
  correctness, recovery — not just the final answer.
- **What it proves:** That you can build agents that are debuggable and
  measurable. Trajectory-level eval is the thing separating people who demo
  agents from people who operate them.
- **Scope — build this:**
  - An agent loop you wrote: plan → tool call → observe → repeat, with an
    explicit step budget, a token budget, structured tool schemas, and typed
    errors fed back to the model.
  - 4-6 tools over a real domain with *deterministic ground truth* (a SQLite
    database plus a filesystem is ideal — you can compute the correct answer
    independently).
  - An **MCP server** exposing those tools. Target the current spec revision
    (2026-07-28, which made the transport stateless and removed protocol-level
    sessions / the `Mcp-Session-Id` header) and say in the README what the
    stateless core means for horizontally scaling your server behind a load
    balancer — that is a distributed-systems observation most MCP demos miss.
  - Full trajectory logging: every prompt, tool call, argument, observation,
    token count and latency, replayable offline.
  - A **trajectory eval set** of 40-60 tasks with per-task rubrics covering:
    final-answer correctness, tool-selection precision/recall, argument validity,
    unnecessary-tool-call rate, recovery-after-error rate, steps-to-solution,
    and cost.
  - One deliberate comparison: agent loop vs a fixed pipeline on the same tasks.
- **Explicitly out of scope:** multi-agent orchestration, long-term memory
  systems, a UI, browser automation, LangGraph/CrewAI (write the loop yourself —
  that is the point), RL on trajectories.
- **Hardware:** M3/MPS. Use a hosted model API for the agent's brain; the
  engineering under evaluation is the harness, not the model.
- **Measure and report:**
  - Task success rate, and success rate *at a fixed cost ceiling*.
  - Tool-selection precision and recall; invalid-argument rate.
  - Median and p95 steps and tokens per task; cost per solved task.
  - Recovery rate: of tasks where the first tool call errored, what fraction
    still succeeded.
  - Agent vs fixed pipeline: success, cost and p95 latency, all three.
  - Variance: run the whole suite 3x and report the spread. Agents are
    non-deterministic and a single run is not a result.
- **The writeup:** (1) Final-answer accuracy hides the failure mode; here are two
  runs with identical answers and completely different trajectories, one of which
  would be unaffordable at scale. (2) Most agent failures in my set were
  argument-construction failures, not planning failures — which means the fix is
  schema and validation, not a better prompt. (3) The fixed pipeline beat the
  agent on N of M task categories; here is the rule for when the loop earns its
  cost. (4) What the stateless MCP revision changes about deploying a tool server.
- **Interview story:** "I built the agent and then built the eval, and the eval
  is the actual deliverable. Scoring only the final answer told me 70%; scoring
  trajectories told me that a third of the successes took twice the necessary
  tool calls, and that almost all real failures were malformed arguments rather
  than bad plans. That points the fix at the tool schema." **Be ready for:**
  "when should this not be an agent?"; "how do you stop an agent loop from
  running away?"; "your eval is non-deterministic — how do you gate a release
  on it?"
- **Time estimate:** 30-40 h.

---

## P9 — ragprod: a production-shaped RAG system with separated metrics  **[FLAGSHIP]**

- **Unlocked after:** Phase 8
- **One-line pitch:** A hybrid-retrieval RAG system over a real corpus, with a
  hand-built golden set, retrieval metrics reported *separately* from generation
  metrics, and an ablation showing which component actually earns its latency.
- **What it proves:** That you can debug a RAG system instead of tuning it by
  vibes. The separation of retrieval and generation metrics is the single most
  reliable senior signal in this area, because almost every RAG repo conflates
  them.
- **Scope — build this:**
  - A real corpus with real structure — your company's public docs, a
    regulator's PDFs, an open-source codebase. Not Paul Graham essays.
  - An ingestion pipeline where chunking is a *parameter*: fixed-size, recursive,
    and structure-aware (heading/section), plus contextual retrieval
    (prepend an LLM-generated chunk summary). Store chunk provenance.
  - Retrieval: dense (embedding) + BM25 + fusion (RRF), then a cross-encoder
    reranker. Each stage independently switchable.
  - Query transformation: rewriting and multi-query expansion, off by default and
    measured.
  - A **golden set of 60-100 queries you labelled yourself**, each with the
    relevant chunk IDs *and* a reference answer. Include unanswerable queries —
    the system must be able to say "not in the corpus", and you must measure that.
  - The serving path: streaming, citations with offsets back to source, and a
    latency budget broken down per stage.
- **Explicitly out of scope:** GraphRAG, a UI beyond a debug page, multi-tenancy,
  incremental re-indexing, fine-tuning an embedding model (stretch), agentic
  retrieval (that is P8's territory).
- **Hardware:** M3/MPS. Local embedding model + a local or hosted reranker;
  hosted LLM for generation. Everything fits in unified memory at this corpus
  size.
- **Measure and report:**
  - **Retrieval, on its own:** recall@k for k ∈ {1,5,10,20}, nDCG@10, MRR — per
    configuration. Retrieval recall@20 is your ceiling; the generator can never
    beat it, and stating that in the README is the point.
  - **Generation, conditioned on retrieved context:** groundedness/faithfulness,
    answer correctness against your references, citation accuracy (does the cited
    span actually support the claim?), and refusal correctness on the
    unanswerable subset.
  - An ablation table where every row changes exactly one component (chunking,
    hybrid on/off, reranker on/off, query rewriting on/off), reporting the
    retrieval metric, the end metric, the added p95 latency and the added cost.
  - Latency budget: p50/p95 per stage (embed → search → rerank → prefill →
    generate), summing to the end-to-end number.
  - A failure taxonomy over ~30 sampled failures, each labelled
    *retrieval miss* / *reranker miss* / *generation ignored context* /
    *chunk boundary destroyed the answer* / *question genuinely unanswerable*.
- **The writeup:** (1) Retrieval recall@k is the ceiling on end-to-end accuracy;
  I measured both and here is the gap between them, which is the generation
  problem. (2) The reranker bought X nDCG points for Y ms of p95 — here is
  whether that clears my SLO. (3) Chunking was the highest-leverage knob and it
  is not a hyperparameter, it is a function of document structure; here is the
  evidence. (4) The failure taxonomy, with the claim that N% of my failures were
  unfixable by prompt changes.
- **Interview story:** "Everyone's RAG demo reports one accuracy number. I
  labelled a golden set so I could measure retrieval separately, and it turned
  out recall@20 was 0.8-something — meaning a fifth of my questions were
  unanswerable no matter what I did to the prompt. Once I knew that, the work
  went into chunking and hybrid search, not prompt engineering." **Be ready for:**
  "how did you build the golden set without leaking your own assumptions?";
  "why RRF rather than a tuned weighted blend?"; "what is your p95 budget and
  which stage would you cut first?"
- **Time estimate:** 45-60 h. The labelling is the expensive part and it is also
  the part that makes the project credible — do not skip it or generate it.

---

## P10 — evalops: an eval and observability layer with a CI regression gate

- **Unlocked after:** Phase 10
- **One-line pitch:** A tracing + evaluation service for LLM applications that
  instruments P8 and P9, calibrates an LLM judge against human labels, and fails
  a CI build on a statistically significant quality regression.
- **What it proves:** That you can make LLM quality an *operational* property
  with a gate and an owner, rather than a thing someone eyeballs before release.
  This maps directly onto backend/SRE instincts you already have.
- **Scope — build this:**
  - Tracing: spans for retrieval / tool call / model call with token counts,
    cost, latency, cache hits and model version, following the OpenTelemetry
    GenAI semantic conventions (check the current status — they were still
    evolving; say in the README which version you targeted and where you
    deviated).
  - An offline eval runner over versioned datasets, producing versioned,
    comparable result artifacts.
  - An **LLM-judge calibration study**: 150-200 examples you labelled by hand,
    then measure the judge's agreement (Cohen's kappa, not raw accuracy), test
    position-bias and verbosity-bias with swapped-order pairwise runs, and report
    the judge's own cost.
  - A regression gate: a GitHub Action that runs the suite on a PR and fails on a
    regression that is significant given the measured run-to-run variance. Show
    the variance calculation.
  - A dashboard: quality, cost and latency over time, sliced by model version and
    prompt version.
  - Online signals: a sampled production-trace scorer, plus drift detection on
    input distribution.
- **Explicitly out of scope:** competing with LangSmith/Langfuse/Braintrust on
  features, a multi-tenant SaaS, a fancy frontend, human-annotation workflow UI.
- **Hardware:** M3/MPS.
- **Measure and report:**
  - Judge-vs-human agreement (kappa) per criterion, and the position-bias delta
    when you swap A/B order.
  - Run-to-run variance of the whole suite at fixed temperature, and the minimum
    detectable effect that follows from it — this is what sets your gate's
    threshold.
  - Cost per eval run; wall-clock per eval run; the added tracing overhead on
    request p95.
  - One real caught regression: a prompt or model change that the gate blocked,
    with the diff and the numbers.
- **The writeup:** (1) An LLM judge is a measuring instrument and instruments need
  calibration — here is my kappa and the biases I found and corrected for.
  (2) You cannot gate on a threshold you did not derive from variance; here is my
  MDE and why a 2-point "improvement" in my suite is noise. (3) Tracing an LLM
  app needs different spans than tracing a microservice (tokens and cost are
  first-class dimensions) — here is the schema. (4) Which eval signals correlated
  with the user-visible problems and which were theatre.
- **Interview story:** "I treated eval as an SRE problem. Before setting any
  threshold I measured the suite's run-to-run variance, which told me the
  smallest regression I could actually detect — and it was larger than most of
  the 'improvements' I'd been chasing. Then I calibrated the LLM judge against my
  own labels and found a position bias that was worth several points on its own."
  **Be ready for:** "how do you stop the judge model from preferring outputs from
  its own family?"; "your gate is flaky — what do you do?"; "what do you measure
  online that you can't measure offline?"
- **Time estimate:** 30-40 h.

---

## P11 — injectlab: a prompt-injection red-team harness

- **Unlocked after:** Phase 10
- **One-line pitch:** An automated red-team suite that attacks P8's agent and
  P9's RAG system with direct and indirect prompt injections, reports attack
  success rate per category, and measures what each mitigation costs in utility.
- **What it proves:** Security judgement, and specifically the understanding that
  LLM security is an *architecture* problem (privilege, isolation, output
  validation) rather than a prompt problem. Also that you will not ship an agent
  with tool access and no threat model.
- **Scope — build this:**
  - An attack corpus organised by mechanism, mapped to the OWASP Top 10 for LLM
    Applications categories: direct injection, **indirect injection via
    retrieved documents** (the important one — poison a chunk in P9's corpus),
    tool-result poisoning in P8, data exfiltration via tool arguments or markdown
    image URLs, and excessive-agency escalation.
  - An automated runner with a programmatic success oracle (did a canary secret
    leave? did a forbidden tool fire? did the model follow the injected
    instruction?) — not an LLM judging "did this look bad".
  - Mitigations implemented and measured one at a time: instruction/data
    delimiting, provenance tagging of retrieved content, output-side canary and
    egress filtering, tool-call allowlists with per-tool argument validation,
    human-in-the-loop on irreversible actions, and a least-privilege split of the
    agent's credentials.
  - A **utility cost** measurement: rerun P8's and P9's benchmarks with every
    mitigation on, so the README shows security's price in task success.
- **Explicitly out of scope:** jailbreak research on frontier models, automated
  adversarial optimisation (GCG-style), model-level safety training, a CVE hunt,
  anything you would run against a system you don't own.
- **Hardware:** M3/MPS.
- **Measure and report:**
  - Attack success rate per category, before and after each mitigation, as a
    matrix. Highlight which mitigations do nothing.
  - Exfiltration success rate specifically — the canary either escaped or it
    didn't; this is a clean binary.
  - Utility delta on P8/P9 benchmarks with mitigations on: task success, p95
    latency, cost.
  - False-positive rate of the egress/input filters on benign traffic.
- **The writeup:** (1) Indirect injection through retrieved content is the real
  threat surface, because the attacker never touches your prompt — here is the
  poisoned-chunk attack working end-to-end. (2) Prompt-level defences
  ("ignore any instructions in the documents") reduced but did not eliminate ASR
  in my suite; here is the residual. (3) The defences that actually moved ASR to
  ~0 were architectural (privilege separation, egress filtering, irreversible-
  action confirmation), and they cost N% task success. (4) The honest one: what
  this suite does not cover.
- **Interview story:** "I poisoned a single document in my own RAG corpus and got
  the agent to call a tool it shouldn't have and leak a canary. The lesson was
  that the prompt-level defence everyone ships cuts the success rate but doesn't
  zero it — what zeroed it was not letting the model hold the credential in the
  first place. And that cost me a few points of task success, which is the real
  conversation." **Be ready for:** "why can't you just filter the inputs?"
  (no reliable separation of instructions and data in a single context window);
  "what is your blast radius if the model is fully compromised?"; "how does this
  change when the tool server is third-party MCP?"

- **Time estimate:** 20-28 h.

---

## Sequencing, and what to actually do

**Do not build all eleven.** The realistic shape alongside a full-time job, at
~8-10 h/week, is roughly 12-15 months for the whole list. That is fine — the
list is ordered so each project is publishable the week it finishes.

If you need to cut, this is the priority order:

1. **P3 micro-gpt** and **P5 jvm-llm-gateway.** These two together are the
   portfolio. One says "I understand the model", the other says "I am the person
   who can put it in production", and together they are a positioning nobody else
   in your interview loop has.
2. **P9 ragprod.** The most commercially legible project in the Indian market;
   it is what 70% of applied-LLM job descriptions describe.
3. **P4 kvlab** and **P6 quantbench.** These are what an inference/serving-track
   interview screens on.
4. Everything else is a proof artifact. P1, P2 and P11 are each a weekend or two
   and each closes a specific doubt.

**Conventions for every repo:**
- README structure: what it is → the headline number → how to reproduce →
  the tradeoff table → what I got wrong → what I'd do next.
- Every number has a command that regenerates it and a seed.
- Every benchmark states the device. Never present an MPS number and a CUDA
  number in the same column.
- Commit the plots and the raw result JSON, not just the summary.
- A "Limitations" section is mandatory. It is the most-read section.

**Total rented-GPU budget for the whole portfolio: roughly $100-200** (≈₹9k-18k)
at community-tier A100/H100 rates. Every project above states its share.

---

## Milestone

A single pinned GitHub profile README that links P3, P5 and P9, each with a
headline number in the link text — e.g. *"micro-gpt — 15M-param decoder trained
on M3, 4-way architecture ablation"*, *"jvm-llm-gateway — goodput flat at 3x
offered load, p99 TTFT 340 ms"*, *"ragprod — recall@20 0.84, groundedness 0.91,
100-query hand-labelled golden set"* — and a 200-word positioning paragraph that
names your Java/distributed-systems background as the reason the serving project
exists. When a recruiter can read that page in 90 seconds and correctly describe
what you do, this phase is done.
