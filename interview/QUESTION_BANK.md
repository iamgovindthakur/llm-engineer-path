# Senior / Staff LLM Engineering — Question Bank

Scope: the questions that actually decide senior and staff LLM-engineering loops in the
Indian market (product companies, GPU-serving teams, applied-AI platform teams).
Organised by curriculum phase. Every entry is a *skeleton*, not an essay — the bullets
are the load-bearing points an interviewer is listening for. If you can only produce the
bullets as prose you are not ready; if you can produce the bullets plus a number plus a
failure mode, you are.

Drill this with `/quiz`. Do not read the skeletons before attempting the question.

---

## What you can claim after this phase

- I can derive the standard results on demand (scaling by `sqrt(d_k)`, why K/V are cached
  and Q is not, KV-cache size in bytes, why decode is bandwidth-bound, why BF16 beat FP16
  for training) rather than reciting them.
- I can size a deployment on a whiteboard: params from a config, KV cache per token,
  tokens/s ceiling from HBM bandwidth, GPU count for a QPS target, cost per million tokens.
- I can name the *failure mode* of any technique I propose — what breaks first, at what
  scale, and what metric would catch it.
- I can defend a quantization, PEFT, RAG or agent architecture under adversarial follow-up
  instead of collapsing to "we'd benchmark it".
- I can run the interview from the other side: ask questions that reveal whether the team
  has real serving depth or is gluing APIs together.

---

## How to answer at staff level

Four moves. Use them in this order, every time. Most rejections at this band are not
knowledge failures, they are *shape of answer* failures.

**1. Lead with the tradeoff, not the definition.**
Senior candidates define the thing. Staff candidates name what it costs.
Bad: "LoRA is a low-rank adapter." Good: "LoRA trades expressivity for a ~3x drop in
optimizer memory and the ability to hot-swap adapters per tenant; you pay for it on tasks
that need to move the model's distribution, not just steer it."

**2. Quantify, even roughly, and say your assumptions out loud.**
Every architecture answer should contain at least one number with a unit. "KV cache is
320 KiB/token for Llama-3-70B at FP16 — 2.5 GiB per 8K sequence — so at 64 concurrent
sequences you are spending 160 GiB just on cache, which is more than one H100." An
order-of-magnitude estimate stated with assumptions beats a precise number you cannot derive.

**3. Name the failure mode and the regime where it bites.**
Nothing is universally good. Say *when* it breaks: "speculative decoding is a win only
while you are memory-bound; at high batch size you are compute-bound and the draft
verification is pure overhead."

**4. Say what you would measure, and what would change your mind.**
End on an observable. Not "we'd evaluate it" — name the metric, the slice, and the
threshold: "I'd gate on p99 time-between-tokens under a 90th-percentile prompt-length mix,
and on groundedness on the 200-example adversarial slice; a >2% groundedness drop reverts it."

**Calibration rule:** if you do not know, say the boundary of what you know and then
reason forward from first principles out loud. "I haven't benchmarked FP8 KV cache on
Hopper; what I'd expect from the numerics is..." is a strong answer. Confident invention
is the single most common cause of a staff-level reject.

**Anti-patterns that get people dropped:** quoting a benchmark without its baseline;
saying "it depends" and stopping; naming a library instead of a mechanism ("we'd use
vLLM") when asked how something works; answering a capacity question with no arithmetic.

---

## 1. Foundations and PyTorch

**Q.** Explain broadcasting, and show how you would add a per-head bias of shape `(n_heads, 1, 1)` to attention scores of shape `(B, n_heads, T, T)`.
- *Level:* senior
- *What they are really testing:* whether you actually reason in shapes or pattern-match code.
- *Answer skeleton:*
  - Align shapes right to left; a dim of size 1 is stretched, a missing dim is treated as 1.
  - `(n_heads,1,1)` vs `(B,n_heads,T,T)` aligns as `(1,n_heads,1,1)` → broadcasts over B and both T axes.
  - No memory is copied; stride 0 is used on broadcast dims.
  - Ambiguity is impossible — mismatched non-1 dims error rather than guess.
- *Follow-ups:* What if the bias were `(n_heads, T)`? (aligns to the last two dims — wrong axis, silently valid). How do you make a broadcasting bug loud instead of silent?
- *Red flag answer:* "It just repeats the tensor" — implies a materialised copy and misses that wrong-but-legal broadcasts are the actual production bug.

**Q.** What is the difference between `.view()`, `.reshape()`, `.permute()` and `.contiguous()`, and where does this bite in multi-head attention?
- *Level:* senior
- *What they are really testing:* understanding of strides and memory layout, not API trivia.
- *Answer skeleton:*
  - A tensor is a pointer + shape + strides; `view` only relabels shape/strides and requires compatible (contiguous-enough) layout.
  - `permute`/`transpose` produce non-contiguous views; a following `view` fails, `reshape` silently copies.
  - In MHA you do `(B,T,d) -> (B,T,H,dh) -> transpose(1,2) -> (B,H,T,dh)`; the transpose is what forces a `contiguous()` before flattening heads back.
  - The copy is real memory traffic — it matters in the inner loop, not in a toy.
- *Follow-ups:* Why does `reshape` never fail but `view` does? Which of these ops allocate?
- *Red flag answer:* "`reshape` is just the safe version of `view`, always use it" — true operationally, but hides that you have no model of strides.

**Q.** Walk me through autograd: what exactly is stored during the forward pass, and why does activation memory dominate training memory?
- *Level:* senior
- *What they are really testing:* whether you know where training memory actually goes.
- *Answer skeleton:*
  - Forward builds a DAG of grad_fn nodes; each node saves the *inputs it needs* to compute its own local gradient (e.g. matmul saves both operands, softmax saves its output).
  - Memory = params + gradients + optimizer state + **activations**, and activations scale with batch x seq x d_model x layers.
  - Adam: 4 bytes/param master + 4 + 4 for m and v ≈ 12-16 bytes/param before activations — that is the fixed floor.
  - Activation checkpointing trades ~30% extra compute for storing only layer boundaries.
- *Follow-ups:* Why does `torch.no_grad()` change memory but not correctness at inference? Where does `retain_graph` actually get used?
- *Red flag answer:* "Autograd stores the gradients during forward" — it stores activations; gradients don't exist until backward.

**Q.** Why is `loss.backward()` roughly twice the cost of the forward pass, giving the 6N-FLOPs-per-token training rule?
- *Level:* senior
- *What they are really testing:* the FLOP accounting you need for every training estimate.
- *Answer skeleton:*
  - Forward ≈ 2 FLOPs per parameter per token (one multiply + one add).
  - Backward computes two products per weight matrix: grad wrt input (to propagate) and grad wrt weight → ≈ 4 FLOPs/param/token.
  - Total ≈ 6N FLOPs per token where N = non-embedding params; this is the number behind every "how long to train" estimate.
  - Attention's `O(T^2)` term is excluded and must be added separately at long context.
- *Follow-ups:* At what sequence length does the attention term stop being negligible for a 7B model? How does activation checkpointing change the constant?
- *Red flag answer:* Quoting 6N as a law without being able to decompose it into 2 + 4.

**Q.** You benchmark an op on MPS and get a suspiciously fast number. What do you check?
- *Level:* senior
- *What they are really testing:* benchmarking hygiene on accelerators — transfers directly to CUDA.
- *Answer skeleton:*
  - GPU launches are async: you must synchronize (`torch.mps.synchronize()` / `torch.cuda.synchronize()`) before reading the clock, or you time the launch not the work.
  - Warm up — first call includes kernel compilation/allocator growth.
  - Check the op didn't silently fall back to CPU (MPS) or hit a different kernel than production.
  - Report a distribution, not a mean; and check you aren't measuring an H2D/D2H copy.
- *Follow-ups:* How would you attribute a slowdown to bandwidth vs compute vs launch overhead? What does a CUDA-graph capture fix?
- *Red flag answer:* "I used `time.time()` around the call" with no sync.

**Q.** Why does `float32` sometimes give a *different* answer than `float64` for the same reduction, and when should you care in an LLM?
- *Level:* senior
- *What they are really testing:* numerical literacy, the root of half the "my loss is NaN" tickets.
- *Answer skeleton:*
  - Floating point addition is not associative; reduction order changes the result, and parallel reductions reorder non-deterministically.
  - Catastrophic cancellation and accumulation error grow with the number of terms — long sequences and large hidden dims.
  - Where it matters: softmax (use max-subtraction), LayerNorm/RMSNorm variance, loss accumulation, logits at FP16.
  - Standard practice: compute in low precision, accumulate in FP32.
- *Follow-ups:* Why is the max-subtraction in softmax mathematically a no-op but numerically essential? Why does MPS's incomplete float64 support not actually block LLM work?
- *Red flag answer:* "Use float64 to be safe" — halves throughput, and on MPS/TensorCores it isn't even available.

**Q.** What is a "kernel" and why do fused kernels matter more for LLMs than for CNNs?
- *Level:* senior
- *What they are really testing:* whether you think in terms of memory traffic.
- *Answer skeleton:*
  - A kernel is one GPU program launch; each unfused op reads its input from HBM and writes its output back.
  - Elementwise chains (bias + activation + dropout + residual) are bandwidth-bound: the arithmetic is trivial, the HBM round trips are not.
  - Transformers are full of large elementwise/normalisation ops on `(B,T,d)` tensors, so fusion removes a large fraction of total traffic.
  - This is the same argument that makes FlashAttention a win — it is a fusion/tiling result, not a new math result.
- *Follow-ups:* What does `torch.compile` actually do here? Why can't you fuse across a matmul as easily?
- *Red flag answer:* "Fusion makes it faster because fewer function calls" — CPU-side overhead is not the dominant term.

**Q.** Explain the roofline model and place LLM prefill and LLM decode on it.
- *Level:* staff
- *What they are really testing:* the single mental model that makes every serving answer coherent.
- *Answer skeleton:*
  - Arithmetic intensity = FLOPs performed per byte moved from HBM. The machine's ridge point = peak FLOP/s ÷ peak bandwidth.
  - H100 SXM: ~989 TFLOP/s dense BF16 ÷ ~3.35 TB/s ≈ **~295 FLOP/byte** ridge point.
  - Prefill processes T tokens against each weight → high intensity → compute-bound.
  - Decode processes 1 token per sequence against every weight → intensity ≈ batch size → memory-bound until batch approaches the ridge point.
  - Everything in serving (batching, speculation, quantization) is a move along this curve.
- *Follow-ups:* Why does weight-only INT4 help decode but barely help prefill? What happens to the ridge point on an A100 (~1.5x lower) or with FP8?
- *Red flag answer:* Quoting "1,979 TFLOPS BF16" from a marketing page — that is the *with-sparsity* figure; dense is ~989.

---

## 2. Tokenization

**Q.** Explain BPE training and BPE encoding as two distinct algorithms.
- *Level:* senior
- *What they are really testing:* whether you have implemented it or only used it.
- *Answer skeleton:*
  - Training: start from bytes/characters, count adjacent pair frequencies over the corpus, merge the most frequent pair, append to an ordered merge list, repeat to vocab size.
  - Encoding: split text by a pre-tokenizer regex, then apply the learned merges **in training order** — it is deterministic replay, not a search.
  - Byte-level BPE (GPT-2 onward) starts from 256 byte values, so there is no UNK token ever.
  - The merge list is the model; the vocab is derived from it.
- *Follow-ups:* Why is the pre-tokenization regex (splitting off whitespace, digits, contractions) load-bearing? How does WordPiece's likelihood-based merge criterion differ?
- *Red flag answer:* "It merges the most frequent pairs at encoding time" — confuses the two phases; encoding does not look at frequencies.

**Q.** Why does a tokenizer built mostly on English make Hindi or Tamil serving 3-5x more expensive?
- *Level:* senior
- *What they are really testing:* cost reasoning + relevance to the Indian market.
- *Answer skeleton:*
  - Fertility = tokens per word. Under-represented scripts fall back to byte-level fragments, so fertility explodes.
  - Cost, context window, and latency are all per *token*, so a 4x fertility is a 4x cost and an effective 4x context shrink.
  - It also degrades quality: fewer semantic units per attention span, and rare sequences are poorly modelled.
  - Fixes: vocabulary extension + embedding re-initialisation (mean of sub-token embeddings) + continued pretraining, or pick a model with a balanced tokenizer.
- *Follow-ups:* How do you initialise the new embedding rows so you don't destroy the model? What is the risk of extending the vocab without touching the LM head?
- *Red flag answer:* "Just train a new tokenizer" — you cannot swap a tokenizer on a trained model without retraining the embedding table and head.

**Q.** What is the "SolidGoldMagikarp" / glitch-token class of bug, and what causes it?
- *Level:* staff
- *What they are really testing:* understanding that tokenizer and model are trained on different data.
- *Answer skeleton:*
  - Tokenizer is trained on one corpus; the model on a filtered/different one. Tokens that were frequent for the tokenizer but near-absent in model training keep their random-ish initial embeddings.
  - Those embeddings sit in untrained regions of the space → wildly out-of-distribution behaviour, evasion, nonsense.
  - Detection: histogram token frequency in the actual training mix; flag tokens with near-zero counts; inspect embedding norms (untrained rows are outliers).
  - Mitigation: prune or re-initialise, and keep tokenizer corpus consistent with the model corpus.
- *Follow-ups:* Why do untrained token embeddings often have anomalous L2 norms? Could this be a security issue?
- *Red flag answer:* Treating it as a curiosity rather than a data-pipeline consistency failure.

**Q.** Why does tokenization explain why LLMs fail at character counting and arithmetic?
- *Level:* senior
- *What they are really testing:* whether you diagnose failures at the right layer.
- *Answer skeleton:*
  - The model never sees characters; "strawberry" may be 2-3 tokens, so letter counts are not represented.
  - Numbers historically tokenized inconsistently (right-to-left grouping vs arbitrary chunks), so digit alignment for addition is not available; modern tokenizers deliberately split digits into fixed groups to fix this.
  - The fix is representational (tokenizer design, or tool use), not more training.
  - Interview point: correct diagnosis = "this is a tokenizer property", not "the model is bad at math".
- *Follow-ups:* Why does chain-of-thought partially rescue arithmetic? Why is a calculator tool the correct engineering answer?
- *Red flag answer:* "It needs more math data."

**Q.** What is the difference between a tokenizer's special tokens, a chat template, and the model's actual training format — and what breaks when they disagree?
- *Level:* senior
- *What they are really testing:* the most common real-world fine-tuning bug.
- *Answer skeleton:*
  - Special tokens (BOS/EOS/pad) are vocab entries; a chat template is a Jinja-style string spec mapping role dicts → the exact token sequence the model saw in post-training.
  - Mismatch (wrong BOS, missing role header, wrong EOS) shifts the model off-distribution: rambling, failure to stop, degraded instruction-following.
  - Double-BOS from applying a template *and* letting the tokenizer add BOS is a classic silent bug.
  - Always assert on the decoded token ID sequence of one training example before launching a run.
- *Follow-ups:* Why must the loss be masked on the prompt/role tokens for SFT? What happens at inference if you train with a different EOS than you generate with?
- *Red flag answer:* "The library handles it."

**Q.** How do you decide the chunking strategy for a RAG corpus, and why is "512 tokens with 50 overlap" not an answer?
- *Level:* senior
- *What they are really testing:* whether you optimise retrieval for the downstream question distribution.
- *Answer skeleton:*
  - Chunk size trades retrieval precision (small chunks embed a single claim cleanly) against answer sufficiency (the generator needs enough context).
  - Respect document structure first (headings, sections, table boundaries); fixed windows destroy tables and code.
  - Decouple retrieval unit from generation unit: embed small, return parent/expanded window.
  - Choose empirically against recall@k on a labelled query set drawn from real traffic — the metric, not the folklore, sets the number.
- *Follow-ups:* Where does semantic chunking actually beat structural chunking? How does contextual retrieval (prepending a doc-level summary to each chunk) change the tradeoff?
- *Red flag answer:* Naming a fixed size with no reference to the query distribution or a recall measurement.

**Q.** Estimate the token count of a 40-page PDF and the cost of embedding a 2M-document corpus. Show the arithmetic.
- *Level:* senior
- *What they are really testing:* comfort doing estimation out loud.
- *Answer skeleton:*
  - Rule of thumb for English: ~0.75 words/token → ~1.3 tokens/word; ~500 words/page → ~650 tokens/page → ~26K tokens for 40 pages.
  - State the fertility assumption and adjust for non-English or code (code is denser in tokens).
  - 2M docs x ~1K tokens = 2B tokens; at an embedding price of ~$0.02–0.10 per 1M tokens that is ~$40–200 — usually *not* the dominant cost.
  - The dominant costs are re-embedding on model change, vector-store RAM, and the reranker at query time.
- *Follow-ups:* What is the storage cost of 2M vectors at 1024 dims FP32, and how does that change your index choice? What forces a full re-embed?
- *Red flag answer:* Giving a number with no stated tokens-per-word assumption.

**Q.** Why can two tokenizers with the same vocab size produce very different model quality at fixed compute?
- *Level:* staff
- *What they are really testing:* that tokenizer choice is a compute-efficiency decision.
- *Answer skeleton:*
  - Fertility determines how many tokens a fixed corpus becomes — better compression means more *content* per training step at fixed FLOPs.
  - But over-merging creates rare, under-trained tokens and hurts compositional generalisation; there is an optimum, not a monotone win.
  - Vocab size also shifts parameter budget into the embedding/softmax (`V x d`), which is a real fraction at small scale.
  - Domain match matters more than size: a code-heavy or Indic-heavy mix needs representative merge training.
- *Follow-ups:* How does vocab size interact with the compute-optimal token budget? Why does the embedding matrix cost matter for a 1B model but not a 70B?
- *Red flag answer:* "Bigger vocab is better because fewer tokens."

---

## 3. Attention and architecture

**Q.** Derive why attention scores are divided by `sqrt(d_k)`.
- *Level:* senior
- *What they are really testing:* the single most-asked derivation in the field.
- *Answer skeleton:*
  - Treat q and k components as independent, zero-mean, unit-variance. `q·k = sum of d_k` such products.
  - Variance of a sum of `d_k` independent unit-variance terms is `d_k`; standard deviation is `sqrt(d_k)`.
  - Without scaling, logit magnitude grows with `sqrt(d_k)` → softmax saturates toward one-hot → Jacobian ≈ 0 → vanishing gradients early in training.
  - Dividing by `sqrt(d_k)` restores unit-variance logits, keeping softmax in its responsive regime.
  - Note it is `d_k` (head dim), not `d_model` — this is the detail that separates recall from understanding.
- *Follow-ups:* Why doesn't LayerNorm on the inputs remove the need? What breaks first if you scale by `d_k` instead of `sqrt(d_k)`? How does this connect to attention-logit-softcapping and QK-norm in recent models?
- *Red flag answer:* "To normalise the values so they sum to 1" — that is softmax's job, not the scaling's.

**Q.** Why do we cache K and V but not Q?
- *Level:* senior
- *What they are really testing:* whether you understand the decode dataflow, not just the term "KV cache".
- *Answer skeleton:*
  - At decode step t you compute exactly one new query, `q_t`, and it is used once — in this step's attention — then never again.
  - K and V for *all previous positions* are re-read every step, and (with a causal mask) they never change once computed.
  - So K/V are reused across steps (cache), Q is consumed immediately (no reuse, nothing to cache).
  - Causality is what makes past K/V immutable; a bidirectional model could not cache this way.
- *Follow-ups:* What in the architecture would have to change for past K/V to become invalid? (any non-causal mixing, or position encodings applied post-hoc in an absolute way). Why does this argument survive RoPE?
- *Red flag answer:* "Q is smaller so it's not worth caching" — it is a reuse argument, not a size argument.

**Q.** Write the KV cache size formula and compute it for Llama-3.1-70B at 8K context.
- *Level:* senior
- *What they are really testing:* can you produce serving arithmetic under pressure.
- *Answer skeleton:*
  - `bytes = 2 (K and V) x n_layers x n_kv_heads x head_dim x dtype_bytes x seq_len x batch`.
  - Llama-3.1-70B: 80 layers, 8 KV heads (GQA), head_dim 128, FP16 → `2 x 80 x 8 x 128 x 2` = **327,680 B = 320 KiB per token**.
  - 8K tokens → ~2.5 GiB per sequence; 64 concurrent sequences → ~160 GiB, more than a single 80 GiB H100.
  - Without GQA (64 KV heads) it would be 2.5 MiB/token — 8x — which is precisely why GQA exists.
  - KV cache, not weights, is what caps concurrency at long context.
- *Follow-ups:* Now do it for Llama-3.1-8B (32 layers, 8 KV heads, 128 head_dim → 128 KiB/token). What does FP8 KV cache buy, and what does it cost? How does MLA change the formula?
- *Red flag answer:* Forgetting the factor of 2 for K and V, or using `n_heads` instead of `n_kv_heads` on a GQA model.

**Q.** MHA vs MQA vs GQA: what exactly is shared, and what is the quality/memory tradeoff?
- *Level:* senior
- *What they are really testing:* precision about which tensor gets reduced.
- *Answer skeleton:*
  - Query heads are always full count. MHA: one K/V head per query head. MQA: **one** K/V head shared by all queries. GQA: `g` KV heads, each shared by `n_heads/g` queries.
  - KV cache and KV-projection params shrink by `n_heads / n_kv_heads`; compute (the `QK^T` and `AV` matmuls) is essentially unchanged — the KV heads are broadcast.
  - MQA's quality loss is measurable and training-instability-prone; GQA (e.g. 8 groups) recovers near-MHA quality at ~8x cache reduction — the practical default.
  - GQA models are typically produced by mean-pooling MHA KV heads within a group and uptraining on a small fraction of tokens.
- *Follow-ups:* Why does GQA help decode more than prefill? What does DeepSeek's MLA do differently (low-rank latent KV, cache the latent not the heads)?
- *Red flag answer:* "GQA reduces the number of attention heads" — it reduces KV heads only.

**Q.** Why is decode memory-bandwidth-bound while prefill is compute-bound? Give the numbers.
- *Level:* staff
- *What they are really testing:* the economic core of LLM serving.
- *Answer skeleton:*
  - Prefill: T tokens are multiplied against each weight matrix once loaded → arithmetic intensity scales with T → compute-bound, near peak FLOPs.
  - Decode: per step you must stream **every weight** from HBM to produce one token per sequence → intensity ≈ batch size (FLOPs/byte ≈ B for a 2-byte dtype).
  - Ridge point on H100 ≈ 295 FLOP/byte, so you need batch in the low hundreds before decode is compute-bound.
  - Lower bound on decode latency = model bytes ÷ bandwidth: 70B FP16 = 140 GB; on 2xH100 (TP=2, ~3.35 TB/s each) ≈ 21 ms/token ≈ 48 tok/s per stream, independent of how clever your kernels are.
  - Consequences: batching is nearly free until the ridge point; weight quantization directly buys decode speed; FLOP-side tricks do not.
- *Follow-ups:* Your users demand 100 tok/s single-stream from a 70B — what are your actual options? (more TP, quantize weights, speculative decoding, a smaller model — in that order of honesty.) Why does TP help latency but hurt efficiency?
- *Red flag answer:* "Decode is slow because it's sequential" — true but shallow; the question is *what resource* is saturated.

**Q.** Compute the parameter count of a Llama-style model from its config.
- *Level:* senior
- *What they are really testing:* whether "70B" is a number you can reconstruct.
- *Answer skeleton:*
  - Per layer attention: `d x d` (Q) + `d x n_kv x d_head` (K) + same (V) + `d x d` (O).
  - Per layer FFN with SwiGLU: **three** matrices — gate, up, down — so `3 x d x d_ff` (not 2; this is the trap).
  - Norms are negligible; embeddings `V x d`, plus an untied LM head `V x d`.
  - Llama-3.1-70B: d=8192, n_kv=8, head_dim=128, d_ff=28672, L=80, V=128256 → attn 151M + FFN 705M = ~856M/layer x 80 = 68.4B, + ~2.1B embeddings/head ≈ **70.5B**. Check.
  - The classic `12 x L x d^2` shortcut assumes MHA and `d_ff = 4d` — it under-counts SwiGLU models.
- *Follow-ups:* Why is `d_ff` 28672 and not 32768? (SwiGLU uses 3 matrices, so the hidden size is scaled by ~2/3 to keep FLOPs comparable to a 4d 2-matrix FFN.) How much of a 1B model is embeddings?
- *Red flag answer:* Using `2 x d x d_ff` for a SwiGLU FFN.

**Q.** What does FlashAttention actually change — the math, the complexity, or the memory traffic?
- *Level:* staff
- *What they are really testing:* whether you know it is exact, and why it is faster.
- *Answer skeleton:*
  - The math is **exact** — identical output to standard attention, not an approximation.
  - Compute stays `O(T^2 d)`. What changes is that the `T x T` attention matrix is never materialised in HBM.
  - Tiling + online (streaming) softmax with running max and sum rescaling lets each block be processed in SRAM; activation memory for attention drops from `O(T^2)` to `O(T)`.
  - The speedup is IO: fewer HBM round trips. Backward recomputes attention blocks from the saved statistics rather than storing them.
  - Practical consequence: long context becomes feasible, and the attention term stops being the memory bottleneck.
- *Follow-ups:* Why can't you get the same win by just using `torch.compile`? What does FlashAttention need from the hardware, and why is MPS support so limited? (PyTorch SDPA exposes a fused path; the hand-tuned CUDA kernels target NVIDIA SM architectures.)
- *Red flag answer:* "It's linear attention" or "it approximates the softmax" — it is exact and still quadratic in compute.

**Q.** Explain RoPE. Why is it relative, and why does it work with the KV cache?
- *Level:* senior
- *What they are really testing:* the dominant position-encoding scheme, from the math up.
- *Answer skeleton:*
  - Split each head's Q and K into 2D pairs; rotate pair `i` by angle `m x theta_i` where `m` is the absolute position and `theta_i = base^(-2i/d_head)`.
  - The dot product of a rotated q at position m and rotated k at position n depends only on `(m - n)` — absolute rotation in, relative bias out.
  - Applied to Q and K only (not V), before the scores; so cached K already carries its rotation and stays valid forever.
  - Low-frequency dims (small theta) encode long-range position, high-frequency dims encode local ordering.
- *Follow-ups:* Why does naive extrapolation past the trained context collapse, and how do position interpolation, NTK-aware scaling and YaRN differ? Why did Llama 3 raise the RoPE base from 10,000 to 500,000?
- *Red flag answer:* "RoPE is added to the embeddings like sinusoidal encodings" — it is a multiplicative rotation applied inside attention at every layer.

**Q.** Pre-norm vs post-norm: why did everyone move to pre-norm, and what did it cost?
- *Level:* senior
- *What they are really testing:* training-stability intuition.
- *Answer skeleton:*
  - Post-norm (original Transformer) puts LayerNorm after the residual add; gradient must pass through the norm, so deep stacks need a learning-rate warmup and are unstable.
  - Pre-norm normalises the branch input and leaves a clean identity path from output to input → well-conditioned gradients, trains deep models without exotic warmup.
  - Cost: the residual stream's magnitude grows with depth, effectively down-weighting later layers; models often add a final norm and sometimes residual scaling to compensate. Some evidence pre-norm gives slightly worse final quality at equal size.
  - Hybrids exist (sandwich norm, QK-norm) precisely because pre-norm's stability is not free.
- *Follow-ups:* Why does RMSNorm replace LayerNorm in most modern models? (no mean subtraction, no bias — cheaper, empirically equivalent.) What exactly does QK-norm stabilise?
- *Red flag answer:* "Pre-norm is just more stable" with no mechanism.

**Q.** Why SwiGLU instead of ReLU or GELU, and why does it use three weight matrices?
- *Level:* senior
- *What they are really testing:* that you read architectures rather than assume them.
- *Answer skeleton:*
  - GLU family: the FFN output is `(Swish(x W_gate)) ⊙ (x W_up)` then `W_down` — a multiplicative gate, so one branch modulates the other.
  - Gating gives a data-dependent, multiplicative interaction that a single pointwise nonlinearity cannot express.
  - Three matrices, so `d_ff` is reduced to ~`8/3 d` to keep parameter and FLOP count comparable to a 2-matrix 4d FFN.
  - Empirically a consistent small perplexity win at equal compute — the honest justification is empirical, and the original paper says so.
- *Follow-ups:* What does this do to your parameter-count formula? Does the gate change the memory profile of activation checkpointing?
- *Red flag answer:* Claiming a strong theoretical justification. The literature's own framing is "we offer no explanation for its success".

**Q.** What is attention sinking / the massive-activations phenomenon, and why does it matter for streaming and quantization?
- *Level:* staff
- *What they are really testing:* awareness of empirical Transformer pathologies that break naive optimisations.
- *Answer skeleton:*
  - Models dump large attention mass on the first few tokens (often BOS) as a no-op "sink" when no token is informative; softmax must sum to 1 somewhere.
  - Evicting those initial tokens from the KV cache (naive sliding window) collapses generation quality — StreamingLLM's fix is to always retain the first few tokens plus a recent window.
  - The same phenomenon produces massive-magnitude activation outliers in specific channels, which is exactly what breaks per-tensor activation quantization (motivating LLM.int8, SmoothQuant, AWQ).
  - Practical rule: any KV-eviction policy must be validated against long-context quality, not just memory saved.
- *Follow-ups:* Why does softmax's sum-to-one constraint create the sink? How would you detect a sink-eviction regression in production?
- *Red flag answer:* Proposing plain sliding-window KV eviction as an obviously safe memory optimisation.

**Q.** Mixture-of-Experts: what is the difference between total and active parameters, and what does MoE actually cost you in serving?
- *Level:* staff
- *What they are really testing:* the currently-dominant scaling architecture, and its serving pain.
- *Answer skeleton:*
  - Replace the FFN with `E` experts + a router; each token is routed to top-`k` (commonly 1–8). FLOPs scale with active params; memory scales with total params.
  - Mixtral 8x7B: ~46.7B total / ~12.9B active. DeepSeek-V3: ~671B total / ~37B active. You pay HBM for all of it.
  - Serving pain: every expert must be resident (or paged) even though most are idle per token; expert-parallel all-to-all communication; load imbalance makes latency spiky and batch-dependent.
  - Training pain: router load balancing (auxiliary loss or aux-loss-free bias tuning), expert collapse, capacity-factor token dropping.
  - It buys quality-per-FLOP, not quality-per-GB — the right win when you are compute-bound, the wrong one when you are memory-bound.
- *Follow-ups:* Why is MoE decode latency less predictable than dense? How does batch composition change which experts are hot?
- *Red flag answer:* "MoE is cheaper because only some parameters are used" — cheaper in FLOPs, not in memory or in serving complexity.

**Q.** How would you implement causal masking, and why is `-inf` (or a large negative) used rather than zeroing the weights after softmax?
- *Level:* senior
- *What they are really testing:* that you understand softmax normalisation.
- *Answer skeleton:*
  - Mask is applied to the *logits* pre-softmax: masked positions set to `-inf` → `exp(-inf)=0` → they contribute nothing and, crucially, nothing to the denominator.
  - Zeroing post-softmax leaves the denominator polluted by future tokens, so the remaining weights no longer sum to 1 and information leaks through the normalisation.
  - Use a large negative finite value (or the dtype's min) rather than literal `-inf` in low precision to avoid NaNs when an entire row is masked.
  - Shape: mask broadcasts as `(1,1,T,T)` over `(B,H,T,T)`.
- *Follow-ups:* What happens to a fully-masked row (e.g. left padding) and how do you avoid the NaN? Why do you not need an explicit `T x T` mask in the decode path?
- *Red flag answer:* "Multiply the attention weights by a 0/1 mask after softmax."

**Q.** Left padding vs right padding for batched generation — which, and why?
- *Level:* senior
- *What they are really testing:* a bug that has broken every practitioner at least once.
- *Answer skeleton:*
  - Decoding reads the logits at the **last position**. With right padding, the last position is a pad token, so you generate from garbage.
  - So batched generation requires **left** padding; training/SFT uses right padding with loss masking.
  - Position IDs must be computed from the attention mask, not from `arange`, or left-padded sequences get wrong positions (and wrong RoPE rotations).
  - The failure is silent — outputs are plausible but degraded, which is why it survives to production.
- *Follow-ups:* How does this interact with RoPE specifically? Why do continuous-batching servers make the question mostly disappear?
- *Red flag answer:* "Padding side doesn't matter because of the attention mask."

**Q.** Why do residual connections make deep Transformers trainable? Answer in terms of gradients, not vibes.
- *Level:* senior
- *What they are really testing:* backprop fluency.
- *Answer skeleton:*
  - `y = x + F(x)` → `dy/dx = I + dF/dx`; the identity term guarantees a gradient path that does not vanish through the block.
  - Without it, gradient through L layers is a product of L Jacobians — exponential decay or blowup.
  - Interpretation: the residual stream is a shared bus that each block reads from and writes an increment to; that is what makes layer-wise interpretability and layer skipping coherent at all.
  - Combined with pre-norm you get a completely clean identity path from loss to embedding.
- *Follow-ups:* Why does the residual stream's variance grow with depth, and what do people do about it? What does this imply for layer pruning?
- *Red flag answer:* "It helps information flow" without the `I + dF/dx` term.

**Q.** Sliding-window, dilated and global-local hybrid attention: when is sub-quadratic attention actually the right call?
- *Level:* staff
- *What they are really testing:* judgement about when to break exactness.
- *Answer skeleton:*
  - Quadratic cost only dominates at long context; below a few thousand tokens the FFN dominates and sparsifying attention buys nothing.
  - Sliding window (e.g. Mistral-style) gives receptive field `window x layers`, so information still propagates — but retrieval of a specific distant token degrades sharply.
  - Hybrids (interleaved local layers with a few full-attention layers) are the practical compromise and the current industry default for long context.
  - The real win at long context is often KV-cache size, not FLOPs — a window bounds the cache, which is the actual serving constraint.
  - Validate on needle-in-a-haystack *and* multi-needle/aggregation tasks; single-needle tests hide the damage.
- *Follow-ups:* Why is single-needle retrieval a weak benchmark? Where do SSM/Mamba hybrids fit, and what do they give up? (constant-size state, so no exact recall of arbitrary past tokens.)
- *Red flag answer:* "Linear attention makes long context cheap" as a blanket claim.

---
## 4. Training

**Q.** Derive the cross-entropy loss for next-token prediction and explain what perplexity means operationally.
- *Level:* senior
- *What they are really testing:* the objective everything else is built on.
- *Answer skeleton:*
  - Model outputs logits `(B, T, V)`; softmax gives `p(token | prefix)`; loss = `-log p(target)` averaged over positions.
  - Only the log-prob of the single correct token enters the loss; the rest is pushed down implicitly by normalisation.
  - Perplexity = `exp(mean NLL)` = the effective number of equally-likely choices the model is deciding between per token.
  - Perplexities are only comparable across models with the **same tokenizer and same evaluation data** — this is the follow-up trap.
  - Shapes: logits flattened to `(B*T, V)`, targets to `(B*T,)`; the shift-by-one is where bugs live.
- *Follow-ups:* Why is a perplexity comparison across tokenizers meaningless, and what do you use instead (bits per byte)? What does label smoothing do to the objective and why is it usually skipped for LLMs?
- *Red flag answer:* Comparing perplexity numbers across differently-tokenized models.

**Q.** What are the Chinchilla scaling laws, and why do production models deliberately violate them?
- *Level:* staff
- *What they are really testing:* whether you distinguish training-optimal from deployment-optimal.
- *Answer skeleton:*
  - Chinchilla: at a fixed training compute budget, params and tokens should scale roughly together — ~20 tokens per parameter is the compute-optimal ratio.
  - That optimises *training* cost only. Inference cost is paid for the life of the model and scales with params.
  - So production models are deliberately "over-trained": far more tokens per param (Llama 3 8B saw ~15T tokens, ~1875 tokens/param) to get a small, cheap-to-serve model with big-model quality.
  - The right objective is total cost of ownership = training + expected inference volume; high-traffic products should over-train aggressively.
  - Distillation from a larger teacher is the other lever in the same direction.
- *Follow-ups:* At what inference volume does over-training pay back? How do scaling laws change under data repetition / data scarcity?
- *Red flag answer:* "Llama 3 8B violates scaling laws so it's inefficient" — it is inefficient in training FLOPs and optimal in TCO.

**Q.** Why does mixed-precision training use BF16 rather than FP16, and what is the master-weight copy for?
- *Level:* senior
- *What they are really testing:* the classic numerics question.
- *Answer skeleton:*
  - FP16: 1 sign / 5 exponent / 10 mantissa, max ~65504 — gradients underflow to zero and activations overflow, requiring dynamic loss scaling machinery that can itself destabilise a run.
  - BF16: 1 / **8** / 7 — the same exponent range as FP32, so no overflow/underflow and **no loss scaling needed**; you trade mantissa precision for range.
  - For deep learning, dynamic range matters far more than mantissa bits — this is the whole argument.
  - Master weights are kept in FP32 because a BF16 weight cannot represent a tiny update: `w + lr*g` rounds back to `w` (stale-weight problem). Accumulate in FP32, cast for the matmul.
  - Hardware: BF16 requires Ampere or later; that is why older stacks are stuck on FP16.
- *Follow-ups:* Why is FP8 training split into E4M3 for forward and E5M2 for gradients? Why do you still keep FP32 accumulation inside the tensor core matmul?
- *Red flag answer:* "BF16 is more accurate than FP16" — it is *less* precise (7 vs 10 mantissa bits) and more *robust*.

**Q.** You are fine-tuning and the loss goes to NaN at step 400. Walk me through your debugging.
- *Level:* senior
- *What they are really testing:* systematic debugging, your actual strength as a backend engineer.
- *Answer skeleton:*
  - Reproduce deterministically: fix the seed, and find the exact batch — bad data (empty sequence, all-masked labels → division by zero in the mean) is the most common cause.
  - Instrument: log grad-norm and per-layer activation max per step; NaN is preceded by a spike you can see.
  - Check precision path: FP16 without loss scaling, an un-scaled attention logit, a norm epsilon that is too small, or a softmax row that is fully masked.
  - Check LR schedule: no warmup, or LR too high after a resume that lost the scheduler state.
  - Mitigations in order: fix the data, add warmup, lower LR, clip grad norm, switch FP16→BF16, raise eps.
- *Follow-ups:* Why does a fully-masked attention row produce NaN specifically? How do you distinguish a data bug from a numerics bug in one experiment?
- *Red flag answer:* "Lower the learning rate" as the first and only move.

**Q.** Explain gradient accumulation and why it is not exactly equivalent to a larger batch.
- *Level:* senior
- *What they are really testing:* precision about an everyday technique.
- *Answer skeleton:*
  - Run k micro-batches, sum (or average) gradients, step once — emulates batch `k*micro` at `1/k` the activation memory.
  - Not exact: BatchNorm statistics differ (irrelevant for Transformers), but crucially the **loss normalisation** must divide by total tokens, not average of per-micro-batch means, or variable-length micro-batches get silently mis-weighted.
  - Also not exact when the optimizer has batch-dependent state or when dropout/RNG streams differ.
  - Cost: no FLOP saving, k× the optimizer-step latency amortised, and with DDP you must suppress the all-reduce on all but the final micro-step (`no_sync`) or you pay k× communication.
- *Follow-ups:* Why is "divide by number of micro-batches" wrong for packed variable-length data? What does this cost you in a ZeRO-3 setup?
- *Red flag answer:* "It's exactly the same as a bigger batch."

**Q.** DDP vs FSDP/ZeRO vs tensor parallel vs pipeline parallel: what does each shard, and when do you reach for each?
- *Level:* staff
- *What they are really testing:* distributed-systems transfer — your home turf.
- *Answer skeleton:*
  - DDP: replicate the full model per GPU, shard the *data*; one all-reduce of gradients per step. Only works if the model + optimizer state fits on one GPU.
  - ZeRO/FSDP: shard optimizer state (stage 1), + gradients (2), + parameters (3) across data-parallel ranks; parameters are all-gathered just-in-time per layer. Trades communication for memory.
  - Tensor parallel: split individual matmuls across GPUs (column/row partitioning); needs an all-reduce *twice per layer* → very high bandwidth, so keep it inside one NVLink node.
  - Pipeline parallel: split layers across GPUs; cheap comms, but introduces the pipeline bubble, mitigated by micro-batching / interleaved schedules.
  - Real recipe at scale: TP within a node, PP across nodes, DP/FSDP on top (3D parallelism).
- *Follow-ups:* Why does TP across nodes kill throughput? What is the bubble fraction for a p-stage pipeline with m micro-batches (`(p-1)/(m+p-1)`)? How does sequence/context parallelism fit in for long context?
- *Red flag answer:* Treating FSDP and tensor parallelism as interchangeable ways to "split the model".

**Q.** What does the learning-rate schedule (warmup + cosine decay) actually do, and why does warmup matter more for Transformers?
- *Level:* senior
- *What they are really testing:* optimizer intuition.
- *Answer skeleton:*
  - Warmup: Adam's second-moment estimate `v` is unreliable in the first steps, so the effective step size is erratic; a large early LR blows up the attention logits before anything has organised.
  - Decay: anneal to a small LR so the trajectory settles into a basin; cosine to ~10% of peak is the common default.
  - Batch size and LR are coupled (roughly square-root or linear scaling regimes) — changing one without the other invalidates the schedule.
  - Warmup-stable-decay (WSD) schedules exist so you can branch checkpoints without re-planning the whole cosine — relevant when you do not know the token budget upfront.
- *Follow-ups:* Why is a cosine schedule awkward if you might continue training? What is the interaction between weight decay and Adam (AdamW's decoupling)?
- *Red flag answer:* "Warmup is just a convention."

**Q.** What is the difference between continued pretraining, SFT, and instruction tuning, and when do you need each?
- *Level:* senior
- *What they are really testing:* whether you pick the cheapest intervention that solves the problem.
- *Answer skeleton:*
  - Continued pretraining: raw domain text, next-token loss on everything. Use when the model lacks *knowledge or vocabulary* (new language, new codebase, medical corpus). Expensive, risks catastrophic forgetting — mix in replay data.
  - SFT / instruction tuning: prompt→response pairs, loss masked to the response. Use when the model has the knowledge but the wrong *behaviour, format or style*.
  - Neither is the right tool for facts that change — that is retrieval.
  - Decision order in practice: prompt → few-shot → RAG → SFT/LoRA → continued pretraining. Move down only when the cheaper layer is measurably exhausted.
- *Follow-ups:* Why does fine-tuning on new facts tend to increase hallucination? (you teach the model to assert confidently in a format, not to know.) How do you detect catastrophic forgetting?
- *Red flag answer:* "Fine-tune it on our documents" as the answer to a knowledge problem.

**Q.** Why must SFT mask the loss on prompt tokens, and what happens if you do not?
- *Level:* senior
- *What they are really testing:* a detail that separates people who wrote the training loop from people who ran a script.
- *Answer skeleton:*
  - Labels are set to `-100` (ignore index) over prompt/system/role tokens so only response tokens contribute to the loss.
  - Without masking, the model spends capacity learning to *generate prompts* — it drifts toward imitating user turns and instruction-following degrades.
  - The effect is worse when prompts are long relative to responses (RAG-style SFT with big contexts), where most of the loss would be prompt tokens.
  - Counterpoint worth knowing: for very short responses, some recipes do train on the full sequence as a regulariser — know the tradeoff, don't dogmatise.
- *Follow-ups:* How does packing multiple examples into one sequence interact with masking (need block-diagonal attention or cross-contamination occurs)? What does a per-example loss-token count tell you in a debug pass?
- *Red flag answer:* Not knowing the loss is masked at all.

**Q.** How do you build an SFT dataset that actually improves the model rather than making it worse?
- *Level:* staff
- *What they are really testing:* data judgement, which is where most fine-tuning projects die.
- *Answer skeleton:*
  - Quality and diversity beat volume: a few thousand carefully-curated examples routinely beat hundreds of thousands of scraped ones (LIMA-style result).
  - Response style must match what you want at inference — length, format, refusal behaviour. The model copies surface form aggressively.
  - Deduplicate and decontaminate against your eval set; near-duplicate leakage produces beautiful, fake metrics.
  - Hold out a real eval slice before you look at anything; measure both target-task gain and general-capability regression.
  - Prefer editing model outputs (model-in-the-loop) over writing from scratch — cheaper, and keeps the distribution close.
- *Follow-ups:* How would you detect that your SFT set taught the model to be confidently wrong? What is the "alignment tax" and how would you measure it?
- *Red flag answer:* "We generated 100K examples with GPT-4" with no filtering, dedup or contamination story.

**Q.** What is catastrophic forgetting, and how do you detect it in a fine-tuned model before it reaches production?
- *Level:* senior
- *What they are really testing:* regression discipline applied to models.
- *Answer skeleton:*
  - Narrow fine-tuning shifts the weights off the pretrained distribution; capabilities not represented in the fine-tune data degrade — including safety behaviour and instruction-following.
  - Detect with a fixed capability regression suite run every time: general benchmarks + your own held-out behaviours + a safety slice, not just target-task accuracy.
  - Mitigate: lower LR, fewer epochs (1-3 is usually right), replay/mix a fraction of general data, or use LoRA at low rank so the update is constrained.
  - Serving mitigation: keep the adapter separable so you can revert per-tenant without a redeploy.
- *Follow-ups:* Why does LoRA reduce but not eliminate forgetting? Why does fine-tuning sometimes degrade *safety* alignment specifically, even on benign data?
- *Red flag answer:* Measuring only target-task accuracy and declaring success.

**Q.** Estimate the cost and wall-clock to fine-tune a 7B model on 50M tokens. Show the arithmetic.
- *Level:* staff
- *What they are really testing:* Fermi estimation with the 6N rule.
- *Answer skeleton:*
  - FLOPs ≈ `6 x N x tokens` = `6 x 7e9 x 5e7` ≈ **2.1e18 FLOPs** for full fine-tuning.
  - Effective throughput: assume ~40% MFU on an A100 (~312 TFLOP/s dense BF16) ≈ 1.25e14 FLOP/s → ~1.7e4 s ≈ **~4.7 GPU-hours**; at ~$2/hr ≈ $10. On H100 (~989 TFLOP/s dense) it is well under an hour.
  - State assumptions: MFU, dense BF16, no pipeline bubble, data loading not the bottleneck.
  - Memory, not FLOPs, is the real constraint: full fine-tuning 7B needs ~14 GB weights + 14 GB grads + ~56 GB Adam FP32 state + activations → a single 80 GB card is tight; LoRA removes the optimizer-state term almost entirely.
  - Conclusion an interviewer wants: *compute is cheap at this scale, memory and data quality are the binding constraints.*
- *Follow-ups:* Recompute for LoRA — what changes and what doesn't? (FLOPs barely change; optimizer memory collapses.) What is MFU and what pulls it down?
- *Red flag answer:* Producing a number with no MFU assumption stated.

---

## 5. Inference and serving

**Q.** Define TTFT, TPOT/ITL, and throughput, and explain why optimising one hurts another.
- *Level:* senior
- *What they are really testing:* whether you have real SLO vocabulary.
- *Answer skeleton:*
  - TTFT = time to first token, dominated by prefill (scales with prompt length) and queueing. TPOT/ITL = inter-token latency during decode. End-to-end = TTFT + TPOT x output_tokens.
  - Throughput = total tokens/s across all requests, which is what determines cost per token.
  - The conflict: larger batches raise throughput (decode is memory-bound, so extra sequences are nearly free) but raise queueing delay and TPOT; prioritising prefill for TTFT stalls in-flight decodes and spikes p99 ITL.
  - Chat products care about TTFT and smooth ITL; batch/offline pipelines care only about throughput and cost. You cannot tune one server for both — use separate pools.
- *Follow-ups:* Why does p99 ITL matter more than mean for perceived quality? How does chunked prefill change the TTFT/ITL tradeoff?
- *Red flag answer:* Reporting a single "latency" number for an LLM service.

**Q.** What is continuous (in-flight) batching, and why is it a bigger win than static batching?
- *Level:* senior
- *What they are really testing:* the core serving innovation.
- *Answer skeleton:*
  - Static batching: the whole batch waits for its longest sequence to finish; GPUs idle on finished slots and new requests wait for the next batch.
  - Continuous batching (Orca-style) schedules at **iteration** granularity: a finished sequence is evicted and a queued one joins on the very next decode step.
  - Win comes from the fact that output lengths are wildly variable — utilisation in static batching collapses to roughly mean/max length.
  - Reported gains are large (multiple x) on realistic mixed-length traffic, larger the more variable your lengths are.
  - It requires per-sequence KV state management, which is why PagedAttention and continuous batching arrived together.
- *Follow-ups:* What admission-control policy do you use when the KV pool is nearly full — reject, queue, or preempt? What is the fairness failure mode of always-admit?
- *Red flag answer:* Describing it as "dynamic batch size".

**Q.** What problem does PagedAttention solve, and what is the OS analogy?
- *Level:* senior
- *What they are really testing:* a concept that maps exactly onto your systems background.
- *Answer skeleton:*
  - Naive serving pre-allocates a contiguous KV buffer per sequence sized to `max_len` → internal fragmentation (most sequences finish early) plus external fragmentation across slots; effective utilisation can be far below 50%.
  - PagedAttention stores KV in fixed-size blocks with a per-sequence block table — virtual memory paging for the KV cache. Blocks need not be contiguous.
  - Near-zero waste → far more concurrent sequences → higher throughput at the same HBM.
  - Bonus: copy-on-write block sharing gives free prefix sharing for a shared system prompt and cheap parallel sampling (n>1) from one prompt.
  - Cost: a custom attention kernel that gathers via the block table, and a block allocator on the critical path.
- *Follow-ups:* How does prefix caching change TTFT for a 2K-token system prompt across 1000 requests? What is the eviction policy when the block pool is exhausted, and what does preemption-by-recompute vs swap-to-CPU cost?
- *Red flag answer:* "It compresses the KV cache" — it does not compress anything; it removes fragmentation.

**Q.** What is chunked prefill and which SLO does it protect?
- *Level:* staff
- *What they are really testing:* awareness of the prefill/decode interference problem.
- *Answer skeleton:*
  - A long prefill occupies the GPU for many milliseconds; every in-flight decode stalls behind it → ITL spikes and visible stutter.
  - Chunked prefill splits the prompt into fixed token-budget chunks and co-schedules each chunk with decode tokens in the same iteration (Sarathi-Serve), so the batch always has decode work.
  - Effect: much tighter p99 time-between-tokens, at some cost to raw prefill throughput (the prompt's attention is recomputed in pieces / less efficient tiling).
  - The alternative is disaggregated prefill/decode: separate GPU pools, KV transferred over the interconnect — better isolation, more infrastructure and a KV transfer cost.
- *Follow-ups:* When would you choose disaggregation over chunked prefill? What does the KV handoff cost across nodes, and how does that bound how small a decode batch is worth it?
- *Red flag answer:* Not knowing that prefill and decode contend at all.

**Q.** Explain speculative decoding, including why the output distribution is preserved.
- *Level:* staff
- *What they are really testing:* a favourite staff question — mechanism plus the correctness argument.
- *Answer skeleton:*
  - A cheap draft model proposes `k` tokens; the target model verifies all `k+1` positions **in a single forward pass** (parallel, like prefill).
  - Accept token `i` with probability `min(1, p_target/p_draft)`; on rejection, resample from the normalised positive part of `(p_target - p_draft)` and stop. This modified rejection sampling makes the output distribution **exactly** the target model's.
  - It is a win only because decode is memory-bound: verifying k tokens costs nearly the same wall-clock as verifying one, so you convert spare FLOPs into tokens.
  - Speedup ≈ expected accepted tokens per cycle ÷ (1 + draft cost); it collapses when acceptance is low (domain mismatch) or when batch size is high enough that you are already compute-bound.
  - Draft-free variants: n-gram/prompt-lookup (great for summarisation and code edit where output copies input), Medusa heads, EAGLE (feature-level autoregression).
- *Follow-ups:* Why does speculation *hurt* at large batch sizes? How do you pick `k` adaptively? Does temperature 0 change the acceptance rule?
- *Red flag answer:* "It's approximate but usually close enough" — with the correct acceptance rule it is exact, and saying otherwise shows you have not read it.

**Q.** How many H100s do you need to serve 100 QPS with 2000-token prompts and 300-token outputs at p95 TTFT < 1s?
- *Level:* staff
- *What they are really testing:* end-to-end capacity estimation — the question most candidates cannot start.
- *Answer skeleton:*
  - Pick a model and state it. Say 8B dense, BF16 (16 GB weights) on H100 80 GB.
  - Prefill FLOPs: `2 x N x prompt_tokens` = `2 x 8e9 x 2000` = 3.2e13 per request; x100 QPS = **3.2e15 FLOP/s**. At ~989 TFLOP/s peak and ~40% MFU ≈ 4e14 FLOP/s per GPU → **~8 GPUs for prefill alone**.
  - Decode: 100 QPS x 300 tokens = 30,000 output tok/s. Per-GPU decode throughput is bandwidth-limited: 3.35 TB/s ÷ 16 GB per pass ≈ ~209 full passes/s, x batch size. At batch 64 with ~50% efficiency that is order 6-7K tok/s → **~5 GPUs**, plus KV headroom.
  - KV: 8B is 128 KiB/token; 2300 tokens ≈ 288 MiB/sequence; 64 concurrent ≈ 18 GB — fits alongside 16 GB of weights on one card.
  - Answer: order **12-16 H100s** plus headroom for p95 and failure domains; call it 20. Then say which assumption you would measure first (MFU and achieved decode batch).
  - Always finish with: "these are order-of-magnitude; I'd validate with a load test at the real prompt-length distribution."
- *Follow-ups:* Redo it for a 70B. What changes if prompts share a 1500-token system prefix (prefix caching removes most prefill)? Where does the estimate break down?
- *Red flag answer:* "It depends on the model" and stopping, or producing a GPU count without a single FLOP or byte computed.

**Q.** Estimate cost per million output tokens for a 70B model on rented H100s.
- *Level:* staff
- *What they are really testing:* the unit economics every LLM platform team lives by.
- *Answer skeleton:*
  - 70B BF16 = 140 GB → needs 2xH100 (TP=2). Per-token weight read per GPU = 70 GB → 3.35 TB/s ÷ 70 GB ≈ 48 passes/s; at ~50% achieved bandwidth efficiency, ~24 iterations/s.
  - With batch 64, ~24 x 64 ≈ **~1500 output tok/s** from the 2-GPU unit (before attention/KV overhead, which grows with context).
  - At ~$3/GPU-hr, the unit costs $6/hr = $0.00167/s → 1M tokens takes ~667 s → **~$1.1 per million output tokens** at full utilisation.
  - Multiply by 1/utilisation (real fleets run 30-60%), add prefill cost, and add a replica factor for availability — the honest number is often 2-4x the idealised one.
  - Levers in order of impact: batch size (free until the roofline ridge), weight quantization (INT8/FP8 halves the bytes read → nearly doubles decode), a smaller or distilled model, prefix caching for repeated system prompts.
- *Follow-ups:* Why is *input* token cost so much lower than output cost per token? At what utilisation does self-hosting beat an API?
- *Red flag answer:* Quoting an API price as if it were your cost structure.

**Q.** How does prefix caching work, and what invalidates it?
- *Level:* senior
- *What they are really testing:* the highest-ROI production optimisation for chat and agent workloads.
- *Answer skeleton:*
  - KV blocks for a shared prefix (system prompt, few-shot block, tool schemas, a document) are computed once and reused across requests via hashed block lookup.
  - Saves prefill FLOPs and TTFT proportional to the shared prefix length — often the dominant cost in agent loops where the whole history is resent each turn.
  - Invalidated by **any** change to earlier tokens: a timestamp in the system prompt, reordered tool definitions, a per-user name at the top. Cache hit requires an exact token-prefix match from position 0.
  - Design rule: put everything static at the front, everything variable at the back. This is worth real money.
  - Multi-tenant caveat: cross-user prefix sharing is a side-channel risk (timing reveals what was cached); scope the cache per tenant if prompts can contain secrets.
- *Follow-ups:* Why does an agent loop with growing history benefit especially? How would you measure cache hit rate and attribute TTFT savings to it?
- *Red flag answer:* Thinking it caches *responses* rather than KV state.

**Q.** Temperature, top-k, top-p, min-p, repetition penalty: what does each do to the distribution, and in what order are they applied?
- *Level:* senior
- *What they are really testing:* precision about the sampling stack.
- *Answer skeleton:*
  - Temperature divides logits before softmax: `T<1` sharpens, `T>1` flattens, `T→0` is argmax. It rescales, it does not truncate.
  - Top-k truncates to the k highest-probability tokens; top-p (nucleus) truncates to the smallest set with cumulative probability ≥ p — adaptive to how peaked the distribution is; min-p truncates relative to the top token's probability.
  - Order matters: penalties → temperature → truncation → renormalise → sample. Applying temperature after truncation changes which tokens survive.
  - Repetition/frequency/presence penalties modify logits of already-seen tokens; they are blunt and can suppress legitimately necessary tokens (code syntax, names).
  - For structured output and evaluation, use greedy or very low temperature; note that greedy is still not bit-deterministic across batch sizes because of floating-point reduction order.
- *Follow-ups:* Why is top-p better than top-k for a distribution that is sometimes peaked and sometimes flat? Why can't you guarantee reproducible outputs even at temperature 0?
- *Red flag answer:* "Temperature 0 makes the model deterministic" without the batching/numerics caveat.

**Q.** How does constrained / structured decoding (JSON schema, grammars) work, and what does it cost?
- *Level:* staff
- *What they are really testing:* whether you know the guarantee is mechanical, not prompted.
- *Answer skeleton:*
  - Compile the schema/grammar into a state machine; at each step compute the set of token IDs that can legally continue, and mask all other logits to `-inf`.
  - This makes syntactic validity a **hard guarantee** — retries for malformed JSON disappear.
  - Cost: computing the allowed-token mask per step (mitigated by precomputed/cached FSM transitions, e.g. Outlines-style index building); and constrained decoding can *reduce* semantic quality by forcing the model off its preferred trajectory.
  - Semantic correctness is not guaranteed — well-formed JSON with wrong values still needs validation.
  - Practical: let the model reason in free text first, then emit the structured block, so the constraint applies only where it belongs.
- *Follow-ups:* Why can constrained decoding hurt accuracy? How does the tokenizer complicate grammar masking (a single token can straddle grammar boundaries)?
- *Red flag answer:* "We ask it nicely in the prompt and retry on parse failure."

**Q.** A user reports that responses get slower as a chat conversation grows. Diagnose.
- *Level:* senior
- *What they are really testing:* connecting a product symptom to the mechanism.
- *Answer skeleton:*
  - Two effects: prefill grows linearly with resent history (unless prefix caching hits), and attention cost per decode step grows linearly in context length, so TPOT drifts up.
  - KV cache per sequence grows linearly → fewer concurrent sequences → the scheduler batches less → everyone's throughput drops. This is the dominant effect at scale.
  - At the pool limit the server starts preempting/recomputing, which shows up as latency cliffs, not gradual drift.
  - Fixes: prefix caching, history summarisation/truncation, KV quantization, sliding window, admission control with a per-session context budget.
- *Follow-ups:* Which of these would you instrument first? What does the latency-vs-context curve look like, and where is the knee?
- *Red flag answer:* "Long context is just slower" with no decomposition into prefill, attention, and concurrency effects.

**Q.** vLLM vs TensorRT-LLM vs SGLang vs llama.cpp: how do you choose?
- *Level:* staff
- *What they are really testing:* pragmatic tool judgement without fanboyism.
- *Answer skeleton:*
  - vLLM: PagedAttention + continuous batching, broad model coverage, fast-moving, the default for general throughput serving on NVIDIA.
  - TensorRT-LLM: ahead-of-time compiled engines, typically the best raw latency/throughput on NVIDIA, at the cost of a build step per model/shape/GPU and weaker flexibility.
  - SGLang: RadixAttention prefix sharing and a structured front-end language — strongest where many requests share prefixes or follow programmatic control flow (agents, multi-turn, tree search).
  - llama.cpp / MLX: single-user local and Apple-silicon inference, GGUF quantization; the right choice on your M3, the wrong choice for a fleet.
  - Choose on: model coverage, your batch/prefix-sharing profile, whether you can afford a compile step, and quantization format support. Benchmark on *your* prompt-length distribution, never on a vendor's.
- *Follow-ups:* What exactly does RadixAttention add over block-hash prefix caching? What forces you off vLLM onto a custom stack?
- *Red flag answer:* "vLLM is the fastest" as a context-free claim.

**Q.** Why is dynamic batching in a classic ML server not enough for LLMs?
- *Level:* senior
- *What they are really testing:* that you understand autoregression breaks the request/response model.
- *Answer skeleton:*
  - Classic serving assumes one request = one forward pass of known cost. An LLM request is hundreds of dependent forward passes of unknown length.
  - So batch composition must change *mid-request*, which no request-level batcher supports.
  - Per-request state (KV cache) must live across iterations, so the server is stateful — it is closer to a scheduler over long-lived sessions than to an RPC handler.
  - Output length is unknown a priori, so you cannot do cost-based admission without a length predictor; this is why preemption exists.
- *Follow-ups:* How would you do autoscaling when per-request cost is unknown and unbounded? What signal do you scale on — QPS, tokens/s, or KV-pool utilisation?
- *Red flag answer:* Treating an LLM endpoint as a stateless microservice for capacity planning.

**Q.** What is the streaming (SSE) design you would build, and what are the failure modes?
- *Level:* senior
- *What they are really testing:* production engineering around the model, where your backend depth should shine.
- *Answer skeleton:*
  - SSE or chunked HTTP from the inference server; token IDs decoded incrementally — beware multi-byte UTF-8 and emoji split across tokens; buffer until a valid boundary.
  - Backpressure: a slow client must not pin a GPU slot; apply a client-side timeout and abort the generation server-side (cancellation must actually free the KV blocks).
  - Mid-stream failure: you have already sent a partial answer, so you need an in-band error event and an idempotent retry/resume contract with the client.
  - Observability: emit TTFT, per-token timestamps, finish reason, and tokens generated; p99 ITL is the metric users feel.
  - Safety filters that need the whole output conflict with streaming — decide between buffered moderation and incremental filtering with retraction.
- *Follow-ups:* How do you enforce a per-tenant token budget mid-stream? What does cancellation cost the scheduler?
- *Red flag answer:* No mention of cancellation freeing GPU state.

**Q.** How do you load-test an LLM service so the numbers mean something?
- *Level:* staff
- *What they are really testing:* benchmarking rigour.
- *Answer skeleton:*
  - Replay the real joint distribution of prompt length and output length — synthetic fixed-length prompts overstate throughput badly.
  - Sweep concurrency and plot throughput vs p99 TTFT / p99 ITL; find the knee, then set the SLO-respecting max concurrency and use it for admission control.
  - Include realistic prefix sharing, otherwise you measure a cache-hit rate you will never see (or miss one you will).
  - Report tokens/s **and** the SLO attainment, never one alone; and report cost per million tokens at the chosen operating point.
  - Warm up (CUDA graphs, compiled engines, allocator) and measure over a steady state, not the first 30 seconds.
- *Follow-ups:* Why does a fixed-length benchmark flatter continuous batching *less* than it flatters static batching? What would you monitor in production to detect drift from your load-test assumptions?
- *Red flag answer:* Quoting a tokens/s number with no latency constraint attached.

**Q.** When should you *not* self-host, and how do you make that call quantitatively?
- *Level:* staff
- *What they are really testing:* business judgement, which staff-level loops explicitly probe.
- *Answer skeleton:*
  - Compute break-even: self-hosted cost/1M tokens ≈ (GPU-hr rate x GPUs) ÷ (achieved tok/s x 3600) ÷ utilisation. Compare to API list price at your volume.
  - Self-hosting wins at high, steady volume with predictable prompts; loses at spiky or low volume, where you pay for idle GPUs and an on-call rota.
  - Non-cost drivers that override the arithmetic: data residency (relevant for Indian regulated sectors), custom weights/fine-tunes, latency floor, vendor lock-in, and frontier-quality requirements you cannot match with open weights.
  - Honest staff answer names the hidden costs: engineering headcount, eval infrastructure, upgrade treadmill, and capacity risk.
- *Follow-ups:* At what utilisation does your break-even flip? How would you hedge with a hybrid (self-host the p50 path, burst to an API)?
- *Red flag answer:* "Self-hosting is always cheaper."

---

## 6. Efficiency and quantization

**Q.** What actually gets quantized, and why is weight-only INT4 a big win for decode but not for prefill?
- *Level:* staff
- *What they are really testing:* connecting quantization back to the roofline.
- *Answer skeleton:*
  - Three separately-quantizable things: weights, activations, and the KV cache. They have different constraints and different payoffs.
  - Decode is bandwidth-bound: bytes moved ≈ weight bytes, so INT4 weights → ~4x less HBM traffic → close to 4x fewer bytes per token. Compute is unchanged (usually dequantized to BF16 for the matmul).
  - Prefill is compute-bound: you are already saturating tensor cores, so reducing weight bytes buys little — unless you also quantize activations to hit a faster INT8/FP8 tensor-core path.
  - So W4A16 is a *decode/latency/memory* optimization; W8A8 (SmoothQuant, FP8) is a *throughput* optimization.
  - KV-cache quantization is a third axis and is what buys you concurrency at long context.
- *Follow-ups:* Why does dequantize-then-BF16-matmul still help at all? What does FP8 on Hopper give that INT8 does not (native tensor-core support with better dynamic range and no calibration for activations)?
- *Red flag answer:* "Quantization makes the model faster" with no distinction between the two regimes.

**Q.** Why are activations harder to quantize than weights?
- *Level:* staff
- *What they are really testing:* the outlier problem, the single key fact in this area.
- *Answer skeleton:*
  - Weights are static, bounded and roughly Gaussian per channel — you can calibrate offline and use per-channel scales.
  - Activations are input-dependent and contain **systematic outlier channels** with magnitudes orders of magnitude above the rest; these emerge at scale (visible beyond ~6-7B) and are tied to attention-sink/massive-activation behaviour.
  - A single per-tensor scale then wastes almost the entire INT8 range on a handful of channels → catastrophic error for everything else.
  - Solutions: LLM.int8() decomposes outlier channels into an FP16 path; SmoothQuant migrates activation scale into the weights (`X/s` and `sW`, mathematically equivalent, makes both quantizable); AWQ scales up salient weight channels based on activation statistics.
  - Rule: granularity (per-tensor → per-channel → per-group) is the primary defence, and per-channel is not available on the activation side of a matmul for free.
- *Follow-ups:* Why can't you just use per-channel activation scales in a standard GEMM? Why does FP8's exponent make it more outlier-tolerant than INT8?
- *Red flag answer:* "Just use per-tensor scaling with a good calibration set."

**Q.** GPTQ vs AWQ vs SmoothQuant vs bitsandbytes NF4: what is the mechanism of each, and when do you use which?
- *Level:* staff
- *What they are really testing:* precision about a family people routinely blur together.
- *Answer skeleton:*
  - **GPTQ**: layer-wise post-training quantization using approximate second-order (Hessian) information from calibration data; quantizes column by column and compensates the induced error in the remaining weights. Weight-only (typically 4-bit, group size 128). Needs calibration, takes minutes-to-hours.
  - **AWQ**: activation-aware — identifies salient weight channels by *activation* magnitude and scales them before rounding, protecting ~1% of weights. No backprop, faster to produce, often better at low bit-width and more robust to calibration-set mismatch.
  - **SmoothQuant**: not weight-only — a W8A8 enabler that shifts quantization difficulty from activations to weights via a per-channel scale, enabling INT8 tensor-core throughput.
  - **NF4 (bitsandbytes/QLoRA)**: information-theoretically motivated 4-bit datatype for normally-distributed weights, plus double quantization of the scales; designed for *fine-tuning* memory, not for peak inference speed.
  - Choice: serving on NVIDIA at 4-bit → AWQ or GPTQ (benchmark both on your eval); need throughput → FP8/W8A8; need to fine-tune a big model on one GPU → QLoRA/NF4; local Mac → GGUF k-quants via llama.cpp/MLX.
- *Follow-ups:* Why does group size trade accuracy against memory and kernel efficiency? Why does AWQ tend to generalise better across calibration sets than GPTQ?
- *Red flag answer:* Calling SmoothQuant a 4-bit weight method, or treating GPTQ and AWQ as the same algorithm with different names.

**Q.** How do you evaluate whether a quantized model is acceptable?
- *Level:* staff
- *What they are really testing:* whether you know perplexity hides the damage.
- *Answer skeleton:*
  - Perplexity on wikitext is the standard report and is nearly useless on its own — it is dominated by easy tokens and moves very little while real capability degrades.
  - Evaluate on capability slices that stress precision: multi-step reasoning, code generation with execution tests, long-context retrieval, structured-output validity, non-English.
  - Measure behavioural drift, not just accuracy: KL divergence between FP16 and quantized output distributions on a fixed prompt set catches subtle degradation early and cheaply.
  - Check the tails: quantization damage concentrates in rare/hard inputs, so compare worst-decile performance, and check refusal/safety behaviour separately.
  - Decide against an SLO: "≤1% drop on the task suite, no regression on the safety slice, ≥1.8x decode throughput" — then it ships.
- *Follow-ups:* Why does a 0.1 perplexity delta not bound task degradation? What would you do if INT4 passed your suite but users complained?
- *Red flag answer:* "Perplexity only went up 0.05, so it's fine."

**Q.** What is QLoRA, and which of its three ingredients does what?
- *Level:* senior
- *What they are really testing:* precision about a technique everyone name-drops.
- *Answer skeleton:*
  - Base weights frozen in **NF4** 4-bit; LoRA adapters trained in BF16 on top; gradients flow *through* the dequantized base but only update the adapters.
  - Three ingredients: (1) NF4, a quantile-based datatype matched to normally-distributed weights; (2) **double quantization**, quantizing the per-block quantization constants themselves, saving ~0.4-0.5 bits/param; (3) **paged optimizers**, using unified memory to survive gradient-checkpointing memory spikes instead of OOMing.
  - Result: fine-tuning a 65B-class model on a single 48 GB GPU with quality close to 16-bit LoRA.
  - Costs: slower per step than BF16 LoRA (dequantization on every forward), and merging the adapter back into a 4-bit base is lossy — merge into the FP16 base instead.
- *Follow-ups:* Why is the dequantization overhead acceptable in training but annoying in serving? Why does QLoRA not reduce activation memory?
- *Red flag answer:* "QLoRA is LoRA on a quantized model" — true but misses double quantization and paged optimizers, which is exactly what the follow-up probes.

**Q.** How would you quantize the KV cache, and what breaks?
- *Level:* staff
- *What they are really testing:* the less-discussed third axis, where senior candidates run out of material.
- *Answer skeleton:*
  - Store K and V in INT8 or FP8 with per-token or per-head scales; dequantize inside the attention kernel. Halves cache bytes → roughly doubles concurrency at fixed HBM.
  - Asymmetry: **keys are more sensitive than values**, because key error perturbs the pre-softmax logits and is then exponentially amplified; values are averaged, so error partially cancels. Common recipe: keys at higher precision or per-channel scaling, values more aggressively quantized.
  - Error accumulates over sequence length — damage shows up at long context (retrieval from deep history) and not on short benchmarks.
  - Evaluate on long-context retrieval and multi-hop aggregation, not perplexity.
- *Follow-ups:* Why does per-channel scaling help keys specifically? How does this interact with RoPE (quantize after rotation)? What does FP8 KV buy over INT8 KV?
- *Red flag answer:* Applying the same scheme to K and V without noticing the asymmetry.

**Q.** What is knowledge distillation, and when does it beat quantizing the big model?
- *Level:* staff
- *What they are really testing:* a comparison of the two main "make it cheaper" strategies.
- *Answer skeleton:*
  - Distillation trains a small student to match the teacher: on the teacher's output distribution (soft targets carry more information per example than hard labels), or on generated (sequence-level) data.
  - Beats quantization when you need a genuine step change in cost (10x, not 2x), when the task is narrow, or when you also want lower latency floor (fewer params → fewer bytes → faster decode, and it composes with quantization).
  - Loses when you need general capability, or when you lack teacher access/licence for the outputs — a real constraint with commercial APIs.
  - On-policy/sequence-level distillation (train on the student's own sampled outputs scored by the teacher) fixes the exposure-mismatch problem that plain output-matching has.
  - Key framing: quantization preserves the model's behaviour approximately; distillation replaces it with a cheaper one that is measurably different — so it needs a far stronger eval.
- *Follow-ups:* Why is distilling on the student's own rollouts better than on the teacher's? How would you distill a reasoning model's chain of thought, and what goes wrong?
- *Red flag answer:* "Distillation is just fine-tuning on the big model's outputs" without distinguishing logit-level from sequence-level, or ignoring licence constraints.

**Q.** What is pruning (structured vs unstructured), and why is it less used than quantization for LLMs?
- *Level:* senior
- *What they are really testing:* whether you know the hardware reality check.
- *Answer skeleton:*
  - Unstructured pruning zeroes individual weights: great compression on paper, but dense hardware gets no speedup from scattered zeros — you save nothing unless the sparsity is structured.
  - 2:4 semi-structured sparsity is the exception: Ampere+ tensor cores support it natively for ~2x on the matmul, at a real quality cost that usually needs retraining.
  - Structured pruning (whole heads, FFN channels, layers) gives a genuinely smaller dense model and real speedup, but typically needs healing/continued pretraining.
  - Quantization wins in practice because it is training-free, hardware-supported, and gives predictable memory-bandwidth savings.
  - Depth pruning + distillation (e.g. Minitron-style recipes) is the version that actually ships.
- *Follow-ups:* Why does unstructured sparsity not speed up a GEMM? Which is safer to prune, layers or FFN width, and why?
- *Red flag answer:* "We pruned 50% of the weights so it's 2x faster."

**Q.** `torch.compile`, CUDA graphs and kernel fusion — which problem does each solve?
- *Level:* staff
- *What they are really testing:* knowing that "make it faster" has distinct failure modes.
- *Answer skeleton:*
  - `torch.compile` (Inductor): graph capture + fusion of elementwise/reduction chains → fewer HBM round trips; big win for the norm/activation/residual glue, little for the big GEMMs.
  - CUDA graphs: eliminate per-kernel **launch overhead** by replaying a captured graph. This matters precisely at small batch decode, where kernels are tiny and the CPU cannot enqueue fast enough — a CPU-bound problem, not a GPU one.
  - Handwritten fused kernels (FlashAttention, fused RMSNorm+residual, paged attention) solve what the compiler cannot infer: tiling strategies and algorithm-level restructuring.
  - Diagnose before applying: a profile showing GPU idle gaps → launch overhead (CUDA graphs); high HBM traffic on small ops → fusion; a hot custom op → write the kernel.
  - Cost: recompilation on shape changes (mark dynamic dims), and CUDA graphs require static shapes and fixed memory addresses, which conflicts with dynamic batching unless you bucket.
- *Follow-ups:* Why do dynamic shapes hurt both `torch.compile` and CUDA graphs, and how do serving stacks work around it? What does none of this fix?
- *Red flag answer:* Proposing `torch.compile` as a generic fix without a profile.

**Q.** What is MFU, what is a realistic value, and what pulls it down?
- *Level:* staff
- *What they are really testing:* the standard efficiency metric for training.
- *Answer skeleton:*
  - MFU = achieved FLOP/s ÷ hardware peak dense FLOP/s, using the `6N x tokens` model-FLOPs convention (HFU additionally counts recomputation).
  - Realistic large-scale training MFU is roughly 35-55%; above that is excellent, below ~30% means something is wrong.
  - Pulled down by: communication that does not overlap with compute, pipeline bubbles, small per-GPU batch, data loading stalls, excessive activation recomputation, unfused elementwise ops, and short sequences making GEMMs skinny.
  - Use it to convert a FLOP estimate into wall-clock honestly — an estimate at 100% peak is a lie by roughly 2-3x.
- *Follow-ups:* Why is inference MFU during *decode* hopeless as a metric? (you are bandwidth-bound; use bandwidth utilisation instead.) How does MFU change with sequence length?
- *Red flag answer:* Estimating training time at peak TFLOPs.

**Q.** What is tensor parallelism doing to latency and why does it stop scaling?
- *Level:* staff
- *What they are really testing:* Amdahl's law applied to a GPU interconnect — familiar territory for you.
- *Answer skeleton:*
  - TP splits each weight matrix across N GPUs, so each GPU reads `1/N` of the weights → decode latency drops roughly `1/N` while the model is bandwidth-bound.
  - But each Transformer layer needs **two all-reduces** (after attention output projection, after FFN down projection). That cost is fixed per layer and does not shrink with N.
  - So beyond the point where all-reduce time matches the per-GPU compute time, adding GPUs increases latency. Within an NVLink node this is typically N=4-8; across PCIe or Ethernet it can be N=2.
  - TP also reduces efficiency (more GPUs for the same throughput), so it is a *latency* purchase, not a throughput one. Use TP for latency SLO or to fit the model, DP replicas for throughput.
- *Follow-ups:* How do sequence parallelism and TP combine to cut activation memory? Why is TP=8 across two nodes usually a mistake?
- *Red flag answer:* "More GPUs is always faster."

**Q.** A 70B model must serve on 2xA100 80GB with p50 latency under 80 ms/token. Is it possible? Show your reasoning.
- *Level:* staff
- *What they are really testing:* the ability to say "no, and here is the arithmetic" — the highest-value answer in an interview.
- *Answer skeleton:*
  - 70B BF16 = 140 GB; 2xA100 80GB = 160 GB total — it fits, but leaves ~20 GB for KV cache and activations across both cards. Tight.
  - Bandwidth: A100 80GB SXM ≈ 2.04 TB/s each. With TP=2 each GPU reads 70 GB per token → 70/2039 ≈ **34 ms** theoretical floor; at a realistic 60-70% achieved bandwidth ≈ **50-57 ms**, plus all-reduce and kernel overhead.
  - Verdict: 80 ms is achievable at small batch, but batching to improve throughput will push TPOT up, and KV headroom limits you to a handful of concurrent long sequences.
  - Options if the SLO tightens: INT8/FP8 weights (halves bytes → ~25-30 ms), TP=4 on a 4-GPU node, or a smaller/distilled model. Say which you would try first and why.
  - The move that scores: derive the floor before answering, then answer.
- *Follow-ups:* Redo on 2xH100. How much concurrency do you actually get with 20 GB of KV headroom? (320 KiB/token → ~65K tokens total → e.g. 8 sequences at 8K.)
- *Red flag answer:* "Yes, with vLLM" — no arithmetic, no bandwidth floor.

---
## 7. Post-training and PEFT

**Q.** Explain LoRA precisely: the equation, the initialisation, and what rank controls.
- *Level:* senior
- *What they are really testing:* the most-asked PEFT question, and whether you know what `r` means.
- *Answer skeleton:*
  - Freeze `W (d_out x d_in)`; learn `B (d_out x r)` and `A (r x d_in)`; effective weight `W + (alpha/r) B A`. Only `A` and `B` receive gradients.
  - Init: `A` random (Gaussian/Kaiming), `B` **zero**, so `BA = 0` at step 0 and training starts exactly at the pretrained model — no warm-up shock.
  - `r` controls the **dimension of the subspace** in which the update can live, i.e. how many independent directions of change you can express — not "how much it learns" and not model capacity.
  - `alpha/r` is a scaling factor that decouples the effective learning rate from `r`; that is why people hold `alpha = 2r` and tune `r` alone.
  - Trainable params ≈ `r x (d_in + d_out)` per adapted matrix — typically 0.1-1% of the model; the dominant saving is **optimizer state**, not the weights themselves.
- *Follow-ups:* Why is `B` zeroed rather than both? What is the gradient memory saving versus full fine-tuning, exactly? Why does LoRA barely reduce forward FLOPs?
- *Red flag answer:* "Higher rank means the model learns more" — rank bounds the update's expressible directions; too high mostly wastes memory and starts to overfit.

**Q.** Which modules do you apply LoRA to, and why did the original attention-only recommendation change?
- *Level:* staff
- *What they are really testing:* whether you have actually tuned this rather than copied a config.
- *Answer skeleton:*
  - Original paper adapted only `W_q` and `W_v` for a strong parameter/performance ratio at the time.
  - Current practice targets **all linear layers** — q, k, v, o *and* the MLP gate/up/down — because the FFN holds most parameters and most task-specific knowledge; all-linear consistently beats attention-only at equal trainable-parameter budget.
  - Embeddings and the LM head are usually excluded (or handled separately) — adapting them is needed only for vocabulary changes.
  - Norms are sometimes unfrozen entirely (they are tiny) for a cheap extra degree of freedom.
  - Rule of thumb: prefer *more target modules at lower rank* over *fewer at high rank* for the same budget.
- *Follow-ups:* Why is the FFN the right place for factual adaptation? How does this choice interact with adapter merging and serving?
- *Red flag answer:* "q and v, as in the paper" with no awareness that the recommendation has moved.

**Q.** Can LoRA adapters be merged, and what do you lose by merging?
- *Level:* senior
- *What they are really testing:* the serving implication of PEFT.
- *Answer skeleton:*
  - Yes: `W' = W + (alpha/r) B A` folds into the base weight → **zero** inference overhead, identical latency to the base model.
  - What you lose: the ability to serve many adapters from one base, and the ability to swap per request. Once merged you have a new full-size checkpoint.
  - Merging into a **quantized** base is lossy (dequantize → merge → requantize drifts); merge into the FP16 base and requantize from there.
  - Unmerged serving (multi-LoRA, e.g. S-LoRA/punica-style batched adapter kernels) keeps one base in HBM and swaps small adapters — the right architecture for per-tenant customisation, at a modest throughput cost.
  - Merging multiple adapters by addition is not composition: they interfere, and quality is not additive.
- *Follow-ups:* Design a multi-tenant service with 500 fine-tunes — merged or unmerged, and why? How would you batch requests across different adapters in one forward pass?
- *Red flag answer:* "Merge all the adapters together to combine skills."

**Q.** When does LoRA lose to full fine-tuning?
- *Level:* staff
- *What they are really testing:* honesty about the limits of the convenient answer.
- *Answer skeleton:*
  - LoRA approximately matches full fine-tuning for *style, format and task adaptation* on modest data.
  - It falls behind when you need to add substantial new knowledge or a new language/domain — the low-rank constraint cannot move the distribution far enough, and the gap grows with dataset size (measured in the tens-of-millions-of-tokens range and up).
  - It is also weaker on long-horizon reasoning improvements and on tasks where the pretrained model is genuinely bad, not merely mis-steered.
  - Counterbalance: LoRA *forgets less*, which is sometimes the deciding factor.
  - The honest framing: LoRA is a regulariser with a memory benefit; treat "full FT is better" as a hypothesis to test on a small slice, not a given.
- *Follow-ups:* How would you design a cheap experiment to decide LoRA vs full FT for a given project? What do DoRA and rsLoRA change about this tradeoff?
- *Red flag answer:* "LoRA is always as good as full fine-tuning" — the literature says otherwise for knowledge-heavy adaptation.

**Q.** Walk through the RLHF pipeline and count the models in GPU memory during PPO.
- *Level:* staff
- *What they are really testing:* why RLHF is operationally painful.
- *Answer skeleton:*
  - Stages: pretrain → SFT → reward model (trained on pairwise preferences with a Bradley-Terry log-sigmoid loss) → RL against the reward with a KL penalty to the reference.
  - PPO holds **four** models: policy (training), reference (frozen, for the KL term), reward model, and value/critic model. That is the memory story.
  - The KL penalty to the reference is what stops reward hacking and mode collapse; `beta` sets how far the policy may drift.
  - Failure modes: reward over-optimisation (Goodhart — reward goes up, human preference goes down), length bias, and brittle training dynamics needing careful hyperparameters.
  - Practical consequence: most teams do not run PPO; they run DPO or a GRPO variant.
- *Follow-ups:* Why does the reward model degrade as the policy moves away from its training distribution? How does GRPO remove the critic (group-relative advantage from multiple sampled completions instead of a learned value function)?
- *Red flag answer:* "RLHF trains the model on human feedback" without naming the reward model, the reference KL, or the four-model memory cost.

**Q.** Why does DPO remove the reward model? Give the derivation sketch.
- *Level:* staff
- *What they are really testing:* the single most-asked post-training derivation right now.
- *Answer skeleton:*
  - The KL-regularised RLHF objective has a **closed-form optimum**: `pi*(y|x) ∝ pi_ref(y|x) exp(r(x,y)/beta)`.
  - Invert it: `r(x,y) = beta log(pi*(y|x)/pi_ref(y|x)) + beta log Z(x)`. The reward is *implicitly represented by the policy itself*.
  - Substitute into the Bradley-Terry pairwise preference likelihood; the intractable partition function `Z(x)` is identical for the chosen and rejected response and **cancels in the difference**.
  - You are left with a simple classification loss: `-log sigmoid( beta[ (log pi/pi_ref)(y_w) - (log pi/pi_ref)(y_l) ] )` — supervised, stable, two models in memory (policy + frozen reference), no sampling loop.
  - Cost: it is off-policy, trained only on the fixed preference set, so it cannot explore; and it is more sensitive to preference-data coverage than PPO.
- *Follow-ups:* Why can DPO *decrease* the likelihood of the chosen response as well as the rejected one, and why is that a problem? Why do you SFT on the preferred responses first? What does `beta` do as it goes to 0?
- *Red flag answer:* "DPO directly optimises preferences so it doesn't need a reward model" — correct as a slogan, but without the closed-form optimum and the `Z(x)` cancellation there is no answer.

**Q.** What are the known failure modes of DPO in practice?
- *Level:* staff
- *What they are really testing:* depth past the headline result.
- *Answer skeleton:*
  - The loss only cares about the *margin* between chosen and rejected, so both log-probs can fall — the model degrades on the good response while still "improving" the objective.
  - Length bias: if preferred responses are longer, DPO learns verbosity as a proxy for quality (motivating length-normalised variants like SimPO).
  - Off-policy/distribution shift: preference pairs generated by another model teach ranking over responses this policy would never produce; on-policy pair generation (iterative DPO) helps a lot.
  - Overfitting to preference data with no KL anchor left: `beta` too low → the model drifts and loses capabilities.
  - Mitigations worth naming: reference-free or length-normalised objectives, KTO when you only have binary good/bad labels, iterative/online DPO with fresh on-policy pairs.
- *Follow-ups:* How would you detect chosen-response log-prob collapse during training? When would you prefer KTO over DPO?
- *Red flag answer:* "DPO is strictly better than PPO."

**Q.** What is GRPO and why did it become the default for reasoning training?
- *Level:* staff
- *What they are really testing:* currency with post-training practice.
- *Answer skeleton:*
  - Sample a **group** of G completions for the same prompt, score each with a (often programmatic/verifiable) reward, and use the group-normalised reward as the advantage — no learned value network.
  - Removes the critic → roughly a quarter less memory and a large source of instability gone; the group mean is the baseline.
  - Fits verifiable-reward domains (math with a checker, code with unit tests) where the reward is cheap, exact and not gameable by a learned reward model.
  - Still keeps a KL term to a reference to prevent drift; still needs sampling infrastructure (a fast inference engine co-located with training) — this is the real operational cost.
  - Known issues: length and difficulty biases in the normalisation, and entropy collapse if the group loses diversity.
- *Follow-ups:* What happens when all G completions get the same reward (zero advantage — wasted compute) and how do you avoid it? Why is a verifiable reward fundamentally safer than a learned reward model?
- *Red flag answer:* Describing GRPO as "PPO without the KL term."

**Q.** How do you choose between prompting, RAG, LoRA, and full fine-tuning for a new requirement?
- *Level:* senior
- *What they are really testing:* the decision framework, which staff candidates are expected to own.
- *Answer skeleton:*
  - Diagnose first: is it a *knowledge* gap, a *behaviour/format* gap, or a *capability* gap? They have different fixes.
  - Knowledge that changes → retrieval, always. Fine-tuning facts makes the model confidently assert stale answers.
  - Behaviour/format/tone/domain-jargon → SFT/LoRA on a few hundred to a few thousand curated examples.
  - Capability the base model lacks entirely → a bigger/better base model, or continued pretraining; fine-tuning will not manufacture a capability.
  - Escalate in cost order and stop as soon as the eval passes; every rung adds an eval and a maintenance burden.
- *Follow-ups:* Give me a case where RAG and fine-tuning are complementary rather than alternatives. What is the maintenance cost of a fine-tune when the base model is upgraded?
- *Red flag answer:* Defaulting to fine-tuning for every requirement.

**Q.** What does `alpha` do in LoRA, and why do people set `alpha = 2r`?
- *Level:* senior
- *What they are really testing:* a detail that exposes copy-paste configs.
- *Answer skeleton:*
  - The update is scaled by `alpha/r`, so `alpha` sets the magnitude of the adapter's contribution independent of the rank.
  - Without it, doubling `r` would double the effective update scale and force you to retune the LR; the ratio keeps the effective step comparable across ranks.
  - `alpha = 2r` is a convention giving a scale of 2; `alpha = r` gives scale 1. Neither is theoretically privileged — they are LR-equivalent reparameterisations.
  - rsLoRA argues the correct scaling is `alpha/sqrt(r)` for stability at high rank, which is why plain `alpha/r` degrades as `r` grows.
- *Follow-ups:* If `alpha/r` is LR-equivalent, why does changing it change results at fixed LR? What does LoRA dropout regularise?
- *Red flag answer:* "Alpha is the learning rate."

**Q.** How would you serve 200 customer-specific LoRA adapters behind one endpoint?
- *Level:* staff
- *What they are really testing:* the multi-tenant design that makes PEFT commercially interesting.
- *Answer skeleton:*
  - One base model resident in HBM; adapters (a few MB each at low rank) held in host memory with an LRU cache in GPU memory — 200 x ~20 MB is a few GB, entirely feasible.
  - Batch across adapters in a single forward: the base GEMM is shared, the LoRA term is computed with a gathered/segmented kernel per adapter (S-LoRA / punica approach).
  - Routing: adapter ID from the auth token, never from the request body — this is a tenant-isolation boundary.
  - Cost: some throughput loss versus a merged model, and a more complex scheduler; the win is one GPU pool instead of 200.
  - Operational: adapter versioning, canary per tenant, and a fallback to base on adapter load failure.
- *Follow-ups:* What is the throughput penalty and where does it come from? How would you handle one tenant with 100x the traffic?
- *Red flag answer:* "Deploy each fine-tune as its own service" — 200 GPUs to serve what one can.

**Q.** Your fine-tuned model scores better on your eval but users say it got worse. What happened?
- *Level:* staff
- *What they are really testing:* eval-reality gap, a genuine staff-level concern.
- *Answer skeleton:*
  - Eval/train contamination or near-duplicates → the score is memorisation.
  - The eval measures the target task only; the regression is in unmeasured behaviours (tone, refusals, format stability, multi-turn coherence, other languages).
  - Distribution mismatch: eval prompts are clean and curated; real traffic is messy, multi-turn, and longer.
  - Style change perceived as quality change — e.g. a terser model reads as unhelpful even at equal accuracy.
  - Process fix: build the eval from sampled production traffic, keep a capability regression suite, and ship behind an A/B with user-facing metrics (thumbs, edit rate, task completion), not offline metrics alone.
- *Follow-ups:* How do you build an eval set from production traffic without leaking PII? What online metric would have caught this within a day?
- *Red flag answer:* "The users are wrong, the metrics went up."

---

## 8. RAG and retrieval

**Q.** Why can cosine similarity between embeddings mislead you, even with a good embedding model?
- *Level:* staff
- *What they are really testing:* the deepest question in this area, and the one that separates RAG users from RAG engineers.
- *Answer skeleton:*
  - Cosine similarity measures whatever the training objective made "similar" — usually *topical relatedness*, not *answers-this-question*. "What is the refund window?" and "Our refund policy is generous" can score below an unrelated but lexically similar chunk.
  - It is blind to negation, numbers, dates, entities and directionality: "X causes Y" and "X does not cause Y" embed almost identically. Identifiers and rare tokens are exactly what dense retrieval loses.
  - Geometry problems: anisotropy (embeddings occupy a narrow cone, so raw cosine values are compressed and thresholds are meaningless) and hubness (a few vectors are near-neighbours of everything).
  - Query/document asymmetry: a short question and a long passage are different lengths and registers; symmetric models handle this badly unless trained with asymmetric prefixes.
  - Fixes: hybrid with BM25 (fusion by RRF), cross-encoder reranking on the top ~50, metadata filters for hard constraints, and calibration — never a raw cosine threshold.
- *Follow-ups:* Why is a fixed cosine cutoff (e.g. 0.8) a bad relevance filter? Why does a cross-encoder beat a bi-encoder, and why can't you use it for the whole corpus? (it scores query and document jointly — `O(corpus)` per query, no precomputation.)
- *Red flag answer:* "Use a better embedding model and raise the similarity threshold."

**Q.** Explain HNSW and the knobs that trade recall against latency.
- *Level:* senior
- *What they are really testing:* whether the vector DB is a black box to you.
- *Answer skeleton:*
  - A multi-layer navigable small-world graph: sparse long-range links at the top layers for coarse navigation, dense short-range links at layer 0 for fine search. Greedy best-first descent through layers.
  - Build knobs: `M` (edges per node — memory and recall) and `efConstruction` (candidate list during build — build time and graph quality).
  - Query knob: `efSearch` (candidate list size) — the runtime recall/latency dial; raise it to recover recall without rebuilding.
  - Costs: memory is `vectors + graph` (roughly `M x 2` links per node) and is RAM-resident; deletes are tombstones requiring periodic rebuild.
  - Alternatives: IVF-PQ for far lower memory at lower recall; flat/exact for corpora up to a few hundred thousand vectors, where ANN is premature optimisation.
- *Follow-ups:* How do you measure recall@k for an ANN index when you don't have ground truth? (compare against exact brute force on a sample.) When would you choose IVF-PQ over HNSW?
- *Red flag answer:* Treating recall as 100% because "vector search returns the nearest neighbours."

**Q.** What is hybrid retrieval and how do you fuse the two ranked lists?
- *Level:* senior
- *What they are really testing:* the standard production retrieval architecture.
- *Answer skeleton:*
  - BM25 (lexical, exact-term, handles IDs, codes, rare names, negation-by-term-presence) plus dense (semantic, paraphrase-robust). Their errors are genuinely complementary.
  - Scores are not comparable across systems, so fuse on **rank**: Reciprocal Rank Fusion, `score = sum over systems of 1/(k + rank)` with `k ≈ 60`. It needs no calibration and is hard to beat.
  - Score-based fusion (normalise then weight) can beat RRF *if* you tune the weights on a labelled set — a real option, more maintenance.
  - Then rerank the fused top-50 with a cross-encoder and pass the top-5 to the generator.
  - The consistent finding: hybrid + rerank beats either leg alone, and the rerank contributes more than the fusion method.
- *Follow-ups:* Why is rank fusion more robust than score normalisation? Where does BM25 beat a strong embedding model outright? (exact identifiers, part numbers, legal citations, low-resource languages.)
- *Red flag answer:* "BM25 is legacy; dense retrieval replaced it."

**Q.** Your RAG system returns the right document but the model still answers wrong. Where do you look?
- *Level:* staff
- *What they are really testing:* the ability to decompose a RAG pipeline into separately-measurable stages.
- *Answer skeleton:*
  - Separate the stages and measure each: retrieval recall@k, reranker precision@k, **context utilisation**, and generation groundedness. A single end-to-end score hides which stage failed.
  - Position effects: the answer may be in the middle of a long context where attention is weakest (lost-in-the-middle). Fewer, better chunks beat more chunks.
  - Distractors: near-miss chunks actively degrade the answer — precision matters more than recall past a point; measure the effect of k directly.
  - Conflict: retrieved context contradicts the model's parametric knowledge, and the model may side with its prior. Instruct explicitly to ground and to say "not in context".
  - Chunk truncation: the right document, wrong chunk boundary, so the answer is half-present.
- *Follow-ups:* Design an experiment that isolates "retrieval found it" from "generator used it". Why does adding more context sometimes lower accuracy?
- *Red flag answer:* "Increase top-k."

**Q.** How do you evaluate a RAG system end to end?
- *Level:* staff
- *What they are really testing:* measurement discipline.
- *Answer skeleton:*
  - Retrieval, with labels: recall@k, MRR/nDCG on a query set built from real traffic. Recall@k is the hard ceiling on end-to-end accuracy — measure it first, always.
  - Generation, reference-free: **groundedness/faithfulness** (every claim supported by retrieved context, checked by claim-decomposition + NLI or a judge) and **answer relevance**.
  - Generation, reference-based: correctness against gold answers on a curated set.
  - Negative and adversarial slices: questions the corpus cannot answer (does it abstain?), conflicting-document cases, stale-document cases.
  - Operational: p95 latency per stage, cost per query, index freshness/staleness lag. And a regression gate in CI on a frozen query set.
- *Follow-ups:* How do you build the labelled query set cheaply? (synthesise questions from chunks, then human-verify a sample; note and correct the bias this introduces.) What single metric would you page on?
- *Red flag answer:* Evaluating only the final answer with an LLM judge and no retrieval metrics.

**Q.** What is contextual retrieval and what problem does it solve?
- *Level:* staff
- *What they are really testing:* currency with the chunking state of the art.
- *Answer skeleton:*
  - Problem: a chunk stripped from its document loses its referents — "the policy applies for 30 days" with no indication of which policy, which product, which year.
  - Fix: before embedding, prepend a short LLM-generated, document-aware context sentence to each chunk ("This chunk is from the 2024 India refund policy, section 3..."), then embed and BM25-index the augmented chunk.
  - Reported effect is a substantial reduction in retrieval failure rate, especially combined with BM25 and reranking.
  - Cost: one LLM call per chunk at index time — made affordable by prompt caching of the full document across its chunks, which is the enabling trick.
  - Cheaper approximations: prepend the heading path, or store a document summary in the metadata and do parent expansion at query time.
- *Follow-ups:* Why does prompt caching make this economically viable? What breaks when the source document changes?
- *Red flag answer:* Confusing it with simple chunk overlap.

**Q.** When is query rewriting worth the latency, and what forms does it take?
- *Level:* senior
- *What they are really testing:* judgement about adding an LLM hop.
- *Answer skeleton:*
  - Forms: conversational de-referencing (resolve "it"/"that one" against history — nearly mandatory for multi-turn), decomposition of multi-hop questions, multi-query expansion, and HyDE (embed a hypothetical answer rather than the question, to fix query/document asymmetry).
  - Cost: an extra LLM call on the critical path (100-500 ms) plus its own failure mode — a rewrite can destroy a query that was fine.
  - Worth it when queries are short, conversational, ambiguous, or multi-hop; not worth it for long, specific, single-shot queries.
  - Mitigate risk by retrieving with both the original and the rewritten query and fusing, so a bad rewrite cannot make things worse.
  - Route adaptively: a cheap classifier decides whether to rewrite, retrieve directly, or skip retrieval entirely.
- *Follow-ups:* How do you detect that rewriting is hurting a query class? Why does HyDE work at all if the hypothetical answer is factually wrong?
- *Red flag answer:* Always rewriting, with no measurement of harm and no fallback to the original query.

**Q.** With 1M-token context windows available, is RAG obsolete?
- *Level:* staff
- *What they are really testing:* resisting a fashionable argument with numbers.
- *Answer skeleton:*
  - Cost and latency: prefill is `O(context)` in FLOPs, so stuffing 1M tokens per query is orders of magnitude more expensive than retrieving 5K — and KV cache for 1M tokens is enormous (at 320 KiB/token that is ~320 GB for a 70B-class model).
  - Quality: effective context is well below advertised context; multi-needle and aggregation tasks degrade long before the limit, and distractors hurt.
  - Scale: corpora are terabytes; no context window ends that argument.
  - Also: freshness, access control (per-user filtering is a retrieval-layer feature), citation/provenance, and auditability.
  - Where long context genuinely wins: a single large document per query, and as a *replacement for chunk-level assembly* — retrieve whole documents instead of fragments. That is RAG with a bigger unit, not the absence of RAG.
- *Follow-ups:* At what corpus size and QPS does long-context-stuffing beat retrieval? How does prefix caching change the economics for a fixed corpus?
- *Red flag answer:* "Yes, just put everything in the context."

**Q.** How do you enforce per-user access control in a RAG system?
- *Level:* staff
- *What they are really testing:* security thinking in retrieval — commonly missed, heavily weighted in enterprise interviews.
- *Answer skeleton:*
  - Filter **before or during** retrieval using the caller's identity, not after generation: ACL metadata on every chunk, pushed into the vector search as a pre-filter.
  - Post-filtering is broken — the model has already seen the restricted content and can leak it; and it silently destroys your top-k.
  - Pre-filtering interacts badly with ANN indexes (filtering can empty the candidate graph neighbourhood) → use partitioned indexes per tenant/permission class, or an index that supports filtered search natively.
  - Derive ACLs from the source system at index time and re-sync on permission changes; stale ACLs are a real leak vector, so track propagation lag as an SLO.
  - Never put identity in the prompt and ask the model to respect it.
  - Also scope caches (embedding, prefix, answer) per permission class, or the cache becomes the leak.
- *Follow-ups:* How do you handle a document whose permissions change after indexing? What is the recall cost of heavy filtering on an HNSW index?
- *Red flag answer:* "We tell the model not to reveal documents the user can't see."

**Q.** How do you keep a vector index fresh with a corpus that changes constantly?
- *Level:* senior
- *What they are really testing:* the data-engineering half of RAG.
- *Answer skeleton:*
  - CDC from the source of truth → chunk → embed → upsert, keyed by a stable chunk ID derived from document ID + position/content hash so updates replace rather than duplicate.
  - Deletes must be real deletes in the serving path; HNSW tombstones mean scheduled compaction.
  - Content-hash comparison to avoid re-embedding unchanged chunks — the main cost control.
  - Changing the embedding model forces a **full re-index**: build to a new index version and swap atomically behind an alias; never mix embedding spaces in one index.
  - Track staleness lag as an SLO and expose document timestamps to the generator so it can qualify stale answers.
- *Follow-ups:* How do you run a zero-downtime embedding-model migration? What does mixing two embedding models in one index do to the score distribution?
- *Red flag answer:* "We re-index nightly" with no incremental path and no versioned swap.

**Q.** What is reranking, what does a cross-encoder actually compute, and where does it sit?
- *Level:* senior
- *What they are really testing:* understanding of the retrieve-then-rerank architecture.
- *Answer skeleton:*
  - Bi-encoder: query and document embedded **independently**, so documents are precomputable and search is ANN over vectors — fast, but the two never interact.
  - Cross-encoder: query and document are concatenated and passed through the model together, giving full token-level attention between them → far better relevance, but `O(candidates)` model calls per query and nothing can be precomputed.
  - Hence the cascade: cheap retrieval to ~50-100 candidates, expensive reranking to top ~5.
  - The reranker is usually the single highest-ROI addition to a naive RAG pipeline, and it is where precision@k comes from.
  - Cost: 50-200 ms and real GPU; late-interaction models (ColBERT-style) sit in between — token-level interaction with precomputable document representations, at a large storage cost.
- *Follow-ups:* Why can't you rerank the whole corpus? How do you choose the candidate count, and how does reranker latency scale with it?
- *Red flag answer:* Calling the reranker "another embedding model."

**Q.** Design the retrieval layer for a multilingual Indian-language support corpus (English + Hindi + 4 regional languages).
- *Level:* staff
- *What they are really testing:* applied design in a locally-relevant setting.
- *Answer skeleton:*
  - Use a genuinely multilingual embedding model with cross-lingual alignment so a Hindi query retrieves an English document; verify alignment empirically rather than trusting the model card.
  - BM25 needs per-language analysers (tokenisation, stemming) — a single English analyser silently destroys the lexical leg for Devanagari/Dravidian scripts.
  - Handle code-mixing and romanised Hindi (Hinglish) explicitly — a large fraction of real Indian queries are transliterated; add transliteration normalisation or train on romanised data.
  - Language detection at query time to route analysers and to set a language filter or boost; store document language as metadata.
  - Evaluate per-language, not in aggregate — a good average hides one language being broken. Also watch tokenizer fertility for cost per query.
- *Follow-ups:* What if the corpus is only in English but users write in Hindi? (cross-lingual retrieval or query translation — compare both.) How do you build eval sets for languages your team doesn't read?
- *Red flag answer:* "Embeddings are multilingual so it just works."

---

## 9. Agents and tool use

**Q.** An agent step is 95% reliable. What is the success rate over a 10-step task, and what do you do about it?
- *Level:* staff
- *What they are really testing:* the compounding-error argument — the central fact of agent engineering.
- *Answer skeleton:*
  - `0.95^10 ≈ 0.60`. Twenty steps ≈ 0.36. Independent per-step reliability compounds multiplicatively, which is why demos work and products don't.
  - So the levers are: fewer steps (collapse multi-step sequences into one deterministic tool), higher per-step reliability (constrained outputs, better tool design), and **error recovery** so a failed step is not a failed task.
  - Recovery beats prevention: feed the tool error back as an observation and let the model retry — this converts a multiplicative failure into a mostly-recoverable one.
  - Add verification at checkpoints rather than at the end, and make irreversible actions require confirmation.
  - Measure per-step success and step-count distribution in production; a rising step count is the leading indicator of a quality regression.
- *Follow-ups:* Which steps deserve verification, given verification also costs latency? How do you make retries safe when tools have side effects?
- *Red flag answer:* "Use a better model" as the only lever.

**Q.** When should you *not* build an agent?
- *Level:* staff
- *What they are really testing:* judgement, which is the most-weighted agent question.
- *Answer skeleton:*
  - If the workflow is known in advance, write the workflow. A deterministic pipeline with LLM calls at specific nodes is cheaper, faster, debuggable and testable.
  - Agents earn their cost only when the path genuinely cannot be enumerated: open-ended research, debugging, exploration where the next action depends on unpredictable observations.
  - Costs you take on: non-deterministic latency and spend (unbounded loops), hard evaluation (trajectories, not answers), hard debugging, and a much larger security surface.
  - Middle ground: a fixed DAG with one agentic node, or a router that dispatches to deterministic handlers.
  - Staff-level framing: "agent" is an architectural choice with an operational bill, not a capability tier.
- *Follow-ups:* Convert this agentic spec into a workflow — what do you lose? How would you cap spend per task?
- *Red flag answer:* Treating agentic as automatically more advanced and therefore better.

**Q.** How do you design a tool so the model actually calls it correctly?
- *Level:* senior
- *What they are really testing:* that tool definitions are a prompt-engineering surface with real failure modes.
- *Answer skeleton:*
  - Few tools, clearly distinct. Overlapping tools produce wrong selection — the dominant error, worse than argument errors.
  - Descriptions written for a competent new engineer: what it does, when to use it, when *not* to, and what it returns. Name and description are what the model conditions on.
  - Schemas with enums and tight types rather than free strings; use constrained decoding so argument syntax is guaranteed.
  - Return errors the model can act on: "date must be YYYY-MM-DD, got 'next Tuesday'" not "400 Bad Request".
  - Design the *return payload* for a context window: truncate, summarise, paginate. A tool returning 50K tokens of JSON will wreck the run.
  - Make tools idempotent where possible, so retries are safe.
- *Follow-ups:* How do you handle a catalogue of 200 tools? (retrieve the relevant tool subset per turn; or a hierarchy of routers.) How do you evaluate tool selection separately from tool arguments?
- *Red flag answer:* Auto-generating tool definitions from OpenAPI with no curation.

**Q.** How do you manage context in a long-running agent?
- *Level:* staff
- *What they are really testing:* the practical bottleneck of every agent system.
- *Answer skeleton:*
  - Cost and latency grow with history, and quality degrades well before the context limit (distraction, lost-in-the-middle, stale instructions).
  - Techniques: summarise/compact older turns at a threshold, keep full fidelity for recent turns; externalise state to a file or scratchpad and re-read on demand; store tool outputs by reference and let the agent fetch what it needs.
  - Keep the static prefix (system prompt, tool schemas) byte-identical and at the front so prefix caching hits — in an agent loop this is often the largest single cost saving.
  - Sub-agents with their own contexts that return only a compact result is the main way to keep the orchestrator's context small.
  - Measure: tokens per task, cache hit rate, and quality as a function of turn index — the degradation curve is the thing to watch.
- *Follow-ups:* What do you lose when you compact, and how do you decide what survives? How do you keep prefix caching working when tool definitions are dynamic?
- *Red flag answer:* "Use a bigger context window."

**Q.** How do you evaluate an agent, given the same task can be solved by different valid trajectories?
- *Level:* staff
- *What they are really testing:* evaluation design under non-determinism.
- *Answer skeleton:*
  - Primary metric: **final-state correctness** against a checkable end state (DB row, file contents, API effect) — trajectory-independent and not gameable.
  - Secondary: efficiency (steps, tokens, cost, wall-clock) and safety (no forbidden tool calls, no out-of-scope writes).
  - Trajectory metrics only where a specific step is mandatory (e.g. "must have called the auth check") — encode as assertions, not as a similarity score to a golden path.
  - Run n trials per task and report pass@1 and pass^k (consistency), since variance is the actual product problem.
  - Infrastructure is the hard part: sandboxed, resettable environments with deterministic fixtures. Budget for this; it is most of the work.
- *Follow-ups:* How do you build a regression suite from production failures without it going stale? How do you evaluate an agent whose environment you cannot reset?
- *Red flag answer:* Using an LLM judge to score similarity to a golden trajectory.

**Q.** What is the ReAct loop, and where does it break down?
- *Level:* senior
- *What they are really testing:* knowing the baseline pattern and its limits.
- *Answer skeleton:*
  - Interleave Thought → Action (tool call) → Observation, repeatedly, with the growing transcript as state. Reasoning conditions the next action; observations ground the reasoning.
  - Breaks down on: loops (calling the same failing tool repeatedly), no global plan on long-horizon tasks, context growth, and error cascades where a bad early observation poisons everything downstream.
  - Guards: a hard step/token/cost budget, loop detection on repeated (tool, args) pairs, explicit re-planning at checkpoints, and a terminal "give up and report" path.
  - Plan-then-execute variants front-load a plan (better for known-structure tasks, worse at adapting); reflection adds a critique step at extra cost with mixed evidence of benefit.
- *Follow-ups:* How do you detect a loop that is not literally identical calls? What is the right thing to do when the budget is exhausted mid-task?
- *Red flag answer:* Presenting ReAct as a solved architecture with no budget or termination discussion.

**Q.** Design the reliability layer for an agent that takes real actions (refunds, emails, DB writes).
- *Level:* staff
- *What they are really testing:* exactly the distributed-systems instincts you already have.
- *Answer skeleton:*
  - Classify tools by reversibility and blast radius; irreversible/high-blast actions require human confirmation or a policy check outside the model.
  - Idempotency keys on every side-effecting tool so a retry after a timeout cannot double-refund. The model *will* retry.
  - Authorisation enforced in the tool layer against the caller's identity — never trust an agent-supplied user ID or amount without server-side validation against policy limits.
  - Full audit trail: the prompt, the tool call, the arguments, the result, the decision; enough to reconstruct why an action happened.
  - Circuit breakers and rate limits per tool and per session; a compromised or looping agent must be bounded by infrastructure, not by its prompt.
  - Dry-run/simulation mode and a staged rollout by blast radius.
- *Follow-ups:* How do you roll back a multi-step agent action (saga/compensating transactions)? Where exactly does human approval sit so it cannot be prompted away?
- *Red flag answer:* Putting the authorisation rules in the system prompt.

**Q.** Single agent with many tools, or a multi-agent system? Defend your choice.
- *Level:* staff
- *What they are really testing:* resistance to architectural fashion.
- *Answer skeleton:*
  - Default to a single agent. Multi-agent adds inter-agent communication loss, duplicated context, more failure modes, higher cost, and much harder debugging.
  - Multi-agent is justified when subtasks are genuinely **parallelisable and independent** (e.g. searching five sources at once), or when context isolation is the point (a sub-agent burns 100K tokens and returns a 500-token summary).
  - It is not justified as role-play ("architect agent talks to coder agent") — the persona split buys nothing that sections of one prompt cannot.
  - If you do it: one orchestrator owning state, sub-agents as typed function calls with schemas, no free-form agent-to-agent chat.
  - Cost reality: parallel sub-agents multiply token spend; justify it against the latency or quality gain.
- *Follow-ups:* How do you handle partial failure of one sub-agent? Where does shared state live, and who is allowed to write it?
- *Red flag answer:* Proposing a committee of personas as an obvious quality improvement.

**Q.** How do you control latency in an agent loop when each step is an LLM call?
- *Level:* senior
- *What they are really testing:* practical engineering on the product-facing constraint.
- *Answer skeleton:*
  - Parallelise independent tool calls in a single turn rather than serialising them; this is usually the biggest single win.
  - Use a smaller/faster model for routing, classification and extraction; reserve the big model for the steps that need it.
  - Prefix caching on the static header; keep it stable so it actually hits.
  - Stream intermediate progress to the user so perceived latency drops even when total latency does not.
  - Bound it: step budget, per-step timeout, and a degraded path that returns partial results rather than hanging.
- *Follow-ups:* What is the p99 story when one tool is slow? How do you decide the model-size split per step empirically?
- *Red flag answer:* Optimising token counts while leaving three independent tool calls running sequentially.

---

## 10. Evaluation

**Q.** How do you build an evaluation suite for an LLM feature from scratch?
- *Level:* staff
- *What they are really testing:* whether you can create ground truth where none exists — the real job.
- *Answer skeleton:*
  - Start from real or realistic inputs (production traffic, sampled and stratified), not invented ones. 50-100 well-chosen examples beat 10,000 synthetic ones.
  - Define per-example success criteria with a human before writing any judge. If you cannot state what correct means, you cannot evaluate it.
  - Layer the methods by cost: deterministic checks (schema, regex, execution, exact match) → reference-based metrics → LLM judge → human review. Use the cheapest method that is valid for the criterion.
  - Split: a frozen regression set gated in CI, plus a rotating fresh set to fight overfitting to your own eval.
  - Close the loop: every production failure becomes a new eval case. The suite's value comes from this, not from its initial size.
  - Report with confidence intervals; a 2-point move on 100 examples is noise.
- *Follow-ups:* How do you avoid overfitting to the eval set? How many examples do you need to detect a 5-point difference? (roughly hundreds for that effect size at typical variance — say you would compute it, don't guess.)
- *Red flag answer:* "We use MMLU and HumanEval" for a product feature.

**Q.** What are the known biases of LLM-as-a-judge and how do you control each?
- *Level:* staff
- *What they are really testing:* whether you use judges critically.
- *Answer skeleton:*
  - **Position bias**: prefers the first (or a consistent) position in pairwise comparison → run both orders and require agreement, or average.
  - **Verbosity bias**: prefers longer answers → control for length, or include length in the rubric explicitly.
  - **Self-preference**: models favour their own generations → use a different family as judge than the one generating, when the stakes matter.
  - **Sycophancy / anchoring**: leaking the "expected" answer or a prior score into the prompt biases it.
  - **Poor calibration** on absolute scales: 1-10 scores cluster and drift → prefer pairwise or a small ordinal rubric with concrete anchors.
  - Validate the judge like a model: measure agreement with human labels (Cohen's kappa) on a held-out set, and re-validate whenever you change the judge model or prompt. An unvalidated judge is a random number generator with a good UI.
- *Follow-ups:* What kappa would you accept before trusting a judge? How do you detect judge drift after a provider model update?
- *Red flag answer:* Using GPT-class judging with a 1-10 rubric and no human agreement measurement.

**Q.** Define groundedness, faithfulness, answer relevance and correctness, and say which one you would gate a RAG release on.
- *Level:* senior
- *What they are really testing:* metric precision; these get used interchangeably and wrongly.
- *Answer skeleton:*
  - **Groundedness/faithfulness**: every claim in the answer is supported by the retrieved context. Reference-free — needs no gold answer.
  - **Answer relevance**: the answer addresses the question asked (an answer can be perfectly grounded and useless).
  - **Context relevance/precision**: the retrieved chunks are actually about the question — a retrieval metric.
  - **Correctness**: the answer matches ground truth, independent of what was retrieved. Needs labels.
  - Gate on groundedness (it catches the hallucination class, needs no labels, and can run on production traffic) plus a smaller labelled correctness set. Note the trap: a model that always says "I don't know" scores perfectly on groundedness — pair it with relevance and abstention rate.
- *Follow-ups:* How do you implement a groundedness check cheaply? (decompose into atomic claims, NLI-entail each against the context.) Why is correctness alone insufficient for RAG debugging?
- *Red flag answer:* Treating groundedness and correctness as the same metric.

**Q.** What is benchmark contamination and how do you detect it?
- *Level:* staff
- *What they are really testing:* healthy scepticism about published numbers.
- *Answer skeleton:*
  - Public benchmarks leak into pretraining corpora, so the score measures memorisation rather than capability; the effect is largest on the benchmarks people cite most.
  - Detection signals: n-gram overlap between the benchmark and the training set (when you own the data); anomalously low perplexity on the test items; performance collapse on a perturbed/rephrased variant; an ordering effect where the model can complete a test item from its prefix.
  - Defences: private held-out sets, freshly-generated variants, time-sliced evaluations (data created after the model's cutoff), and dynamic/live benchmarks.
  - Practical stance: never make a model-selection decision on a public leaderboard alone; build a private eval on your own distribution.
- *Follow-ups:* How would you build a contamination-resistant eval for a task with a fixed answer set? Why does fine-tuning make contamination worse and harder to see?
- *Red flag answer:* Picking a model purely on leaderboard rank.

**Q.** How do you detect hallucination in production, where you have no reference answers?
- *Level:* staff
- *What they are really testing:* reference-free monitoring design.
- *Answer skeleton:*
  - For grounded (RAG) answers: automated claim-level entailment against the retrieved context on a sample of traffic — this is the highest-signal check and it is cheap enough to sample continuously.
  - Self-consistency: sample n responses at temperature > 0 and measure agreement; high disagreement correlates with fabrication (SelfCheckGPT-style). Costs n× tokens, so sample.
  - Uncertainty signals: token log-probs / sequence entropy at the span level — weak alone, useful as a cheap prefilter for a more expensive check.
  - Product signals: citation click-through, user edits, thumbs-down, escalation rate, and "regenerate" clicks — noisy but free and real.
  - Route low-confidence outputs to abstention or human review rather than trying to eliminate hallucination outright.
- *Follow-ups:* Why are raw log-probs a poor confidence signal for a post-trained model? (RLHF damages calibration.) What is the cost/coverage tradeoff of sampled checking?
- *Red flag answer:* "We ask the model if it is sure."

**Q.** Offline eval says the new prompt is better; how do you decide to ship?
- *Level:* staff
- *What they are really testing:* experimentation discipline transferred from backend work.
- *Answer skeleton:*
  - Offline is a filter, not a decision. Ship behind a flag with an online A/B on user-facing metrics: task completion, edit/regenerate rate, escalation to human, retention — plus cost and latency as guardrails.
  - Power the test: LLM output metrics are high-variance; compute the sample size for the effect you care about before you start, or you will read noise.
  - Watch guardrails and segments, not just the mean — a prompt that helps the median and destroys one language or one customer segment is a rollback.
  - Have an automatic rollback trigger and a kill switch; prompt changes are code changes and need the same release discipline (versioning, review, rollback).
  - Stage: internal → 1% → 10% → 100%, with a fixed soak time at each step.
- *Follow-ups:* What if the online metric is neutral but cost drops 30%? What do you do when you have too little traffic to power an A/B?
- *Red flag answer:* Shipping on an offline win alone.

**Q.** Why is pass@k a misleading metric for a production coding assistant?
- *Level:* senior
- *What they are really testing:* metric/product alignment.
- *Answer skeleton:*
  - pass@k measures "at least one of k samples passes" — meaningful only when you can *verify* and pick the passing one automatically (unit tests present).
  - Users get one answer. Without a verifier, pass@1 (or pass^k, the probability *all* k pass) is the honest metric; pass@10 flatters by exactly the amount you cannot exploit.
  - It also ignores everything that matters in the product: latency, diff size, style conformance, whether it broke an unrelated test, and reviewability.
  - Report pass@1 for the user-facing number, and use pass@k only when describing a system that genuinely samples and filters.
- *Follow-ups:* When is pass@k the *right* metric? (agentic loops with a real test harness.) What is pass^k and why does it matter for reliability claims?
- *Red flag answer:* Quoting a headline pass@k as the product's accuracy.

**Q.** How do you evaluate a summarisation feature where there is no single correct output?
- *Level:* senior
- *What they are really testing:* handling open-ended generation.
- *Answer skeleton:*
  - Decompose the quality into checkable dimensions: factual consistency with the source (entailment-checkable), coverage of key points (against a human-extracted key-point list), absence of fabricated entities, length/format conformance.
  - ROUGE/BLEU are n-gram overlap metrics with poor correlation to human judgement for abstractive summaries — report them only if you must, never gate on them.
  - Pairwise preference (new vs current, both orders, validated judge) is the most reliable comparative signal.
  - Build a small human-rated gold set and use it to validate the automated judge, then let the judge scale.
  - Add an adversarial slice: sources with contradictions, numbers, dates and negations, where summarisers fail specifically.
- *Follow-ups:* Why does factual consistency matter more than fluency now? How would you catch a summariser that drops the single most important sentence?
- *Red flag answer:* "ROUGE score."

**Q.** What would you monitor on an LLM feature in production, day one?
- *Level:* senior
- *What they are really testing:* observability instincts applied to a stochastic system.
- *Answer skeleton:*
  - Systems: TTFT, p50/p95/p99 ITL and end-to-end latency, error and timeout rate by provider/model, token throughput, GPU/KV-pool utilisation, queue depth.
  - Economics: tokens in/out per request, cost per request and per user, cache hit rate — with per-tenant attribution from day one or you will never get it later.
  - Quality proxies: refusal rate, empty/truncated output rate, schema-validation failure rate, finish-reason distribution, output-length distribution (drifts are the earliest signal of a model change).
  - Sampled deep checks: groundedness/judge scores on a traffic sample, with alerting on drift rather than absolute value.
  - User signals: thumbs, regenerate, edit, abandonment, escalation.
  - Log full prompts and responses (with PII handling) — you cannot debug an LLM incident from metrics alone.
- *Follow-ups:* Which of these would you page on at 3am? How do you detect that a provider silently changed the model behind an endpoint?
- *Red flag answer:* Latency and error rate only.

**Q.** How do you compare two models for a product decision in a way that survives scrutiny?
- *Level:* staff
- *What they are really testing:* rigour in the most common real decision.
- *Answer skeleton:*
  - Evaluate on *your* distribution with your prompts — a model tuned for your prompt format will beat one that isn't, regardless of leaderboard rank; re-tune the prompt per model before comparing (otherwise you are measuring prompt transfer).
  - Fix everything else: same retrieval, same temperature policy, same output constraints, same eval set, same judge, both orders.
  - Report quality, p95 latency and cost per request together; a 2-point quality gain at 4x cost is usually a no.
  - Include operational risk: rate limits, region availability, data-handling terms, deprecation history, and self-host feasibility.
  - Run the shortlist in a live A/B before committing; and keep the abstraction layer thin enough to switch back.
- *Follow-ups:* How do you handle the fact that prompt-per-model tuning makes the comparison unfair in the other direction? What is your migration plan when the winner is deprecated?
- *Red flag answer:* Swapping the model name in one config and comparing benchmark scores.

---

## 11. Security

**Q.** What is indirect prompt injection, and why is it architecturally harder than direct injection?
- *Level:* staff
- *What they are really testing:* the defining security problem of LLM systems.
- *Answer skeleton:*
  - Direct: the user types a malicious instruction. Indirect: the payload is in *retrieved content* — a web page, an email, a PDF, a code comment, a calendar invite — which the model reads as part of its context.
  - The core problem: there is no separation between the instruction channel and the data channel. Everything is tokens in one stream, so "data" can always be interpreted as instructions.
  - Harder because the attacker never touches your interface, the payload arrives through a trusted-looking path, and the victim is an agent with credentials and tools.
  - There is **no known prompt-level fix**. Delimiters, "ignore instructions in documents", and instruction-hierarchy training all reduce the rate; none of them are a boundary you can rely on.
  - Therefore the defence must be architectural: treat every model output as untrusted input to the next stage.
- *Follow-ups:* Sketch a concrete exfiltration chain through a RAG corpus. Why does instruction-hierarchy training help without solving it?
- *Red flag answer:* "We put the user content in XML tags and tell the model to ignore instructions inside them."

**Q.** Design the defence for an agent that reads untrusted web content and can send emails.
- *Level:* staff
- *What they are really testing:* whether you can design containment rather than filtering.
- *Answer skeleton:*
  - Least privilege: scope credentials per task, per tenant; the email tool can send only to a pre-approved recipient set, or requires human confirmation for new recipients.
  - Taint tracking: mark content that came from untrusted sources; any action derived from tainted context is downgraded to read-only or routed to confirmation.
  - Privilege separation (dual-LLM / quarantine pattern): a privileged planner never sees raw untrusted text; an unprivileged, tool-less quarantined model processes it and returns structured, schema-validated data only.
  - Deterministic authorisation outside the model — policy code decides what is permitted, using the *user's* identity, never the model's assertion.
  - Egress control: no arbitrary URLs, no markdown image rendering to attacker-controlled domains (a classic silent exfiltration channel), allowlisted network destinations.
  - Detection and response: log every tool call with its provenance, alert on anomalies, rate-limit, and support fast revocation.
- *Follow-ups:* How does a markdown image tag exfiltrate data, and what exactly do you block? Where do you put the human in the loop so it is not trivially fatigued into clicking approve?
- *Red flag answer:* A classifier that scans retrieved content for injection strings, as the primary control.

**Q.** What is the "lethal trifecta" and why does naming it help you design?
- *Level:* staff
- *What they are really testing:* a compact risk model you can apply on the spot.
- *Answer skeleton:*
  - The dangerous combination is: (1) access to private data, (2) exposure to untrusted content, (3) ability to externally communicate/act. Any two are manageable; all three enables exfiltration.
  - Design move: break one leg for any given flow. Strip the egress capability from the component that reads untrusted content; or quarantine untrusted content away from the component with data access.
  - It gives you a fast audit question for any proposed feature: which of the three does this add, and to which component?
  - It also explains why connecting a corporate RAG system to a browsing agent with a webhook tool is a serious change, not a convenience feature.
- *Follow-ups:* Apply it to an MCP deployment with three servers. Which leg is cheapest to break in practice, and why is it usually egress?
- *Red flag answer:* Treating each capability as independently safe because each was reviewed on its own.

**Q.** What are the security risks specific to MCP / third-party tool servers?
- *Level:* staff
- *What they are really testing:* currency with the current attack surface.
- *Answer skeleton:*
  - Tool descriptions are prompt content: a malicious server can embed instructions in its tool description or schema that the model reads and follows ("tool poisoning").
  - Rug-pull: a server changes its tool definition after approval, so what you reviewed is not what runs. Pin and hash tool definitions; re-approve on change.
  - Cross-server shadowing: one server's tool description can manipulate the model's use of another server's tools, including redirecting arguments containing secrets.
  - Over-broad OAuth scopes and long-lived tokens held by the client; a compromised server inherits them.
  - Controls: allowlist servers, pin versions, review tool definitions as code, scope credentials narrowly, sandbox execution, log every call, and require confirmation for side-effecting tools.
- *Follow-ups:* How would you implement tool-definition pinning in practice? What is the supply-chain review you would require before adding a third-party server?
- *Red flag answer:* Treating an MCP server as trusted infrastructure because it is "just an API".

**Q.** How do you prevent an LLM from leaking training data or system-prompt contents?
- *Level:* senior
- *What they are really testing:* realism about what is and is not preventable.
- *Answer skeleton:*
  - System prompts are **not** secrets — assume extraction. Never put credentials, keys, internal URLs or business rules you must keep private into a prompt.
  - Training-data memorisation is real for duplicated content; defend at the data layer: deduplication, PII scrubbing before training, and (where warranted) differential privacy — accepting a quality cost.
  - Output-side: PII detection/redaction on responses, and blocking verbatim reproduction of long spans from the training corpus where that is the risk.
  - RAG changes the shape: the leak vector is retrieval permissions, not weights — enforce ACLs at retrieval (see the RAG section).
  - Separate multi-tenant data by index/namespace and by cache scope; a shared prompt or embedding cache is a cross-tenant channel.
- *Follow-ups:* Why does deduplication reduce memorisation so effectively? What is the tenant-isolation argument against a shared prefix cache?
- *Red flag answer:* "We instruct the model never to reveal the system prompt."

**Q.** How would you red-team an LLM feature before launch?
- *Level:* staff
- *What they are really testing:* proactive security process, not a checklist recital.
- *Answer skeleton:*
  - Threat-model first: who attacks, what do they want, which capability grants it. Enumerate against OWASP LLM Top 10 as a coverage check, not as the plan.
  - Automated adversarial generation: injection payloads in every untrusted field, jailbreak templates, encoding tricks (base64, unicode confusables, low-resource languages), multi-turn escalation where the first turns are benign.
  - Agent-specific: can the model be induced to call a forbidden tool, exceed a spend limit, or act on another user's behalf?
  - Manual expert testing on the highest-value paths — automation misses creative chains.
  - Turn every finding into a permanent regression test; re-run on every model and prompt change, because upgrading the base model can silently reopen a fixed hole.
- *Follow-ups:* How do you measure red-team coverage rather than just counting findings? Why must red-team suites be re-run on model upgrade?
- *Red flag answer:* A one-time pre-launch pentest with no regression suite.

**Q.** What are the risks of letting an LLM generate and execute code, and how do you contain them?
- *Level:* senior
- *What they are really testing:* sandboxing fundamentals — straightforward for you, and frequently asked.
- *Answer skeleton:*
  - Assume the generated code is attacker-controlled (indirect injection reaches it). Contain, do not review-by-model.
  - Isolation: container/microVM (gVisor, Firecracker), non-root, read-only root filesystem, no host mounts, ephemeral and destroyed per execution.
  - Resource bounds: CPU, memory, wall-clock, process and file-descriptor limits — an infinite loop must not be an outage.
  - Network: deny by default; allowlist only what the task needs. This is the control that blocks exfiltration.
  - Secrets: none in the sandbox environment; pass data explicitly, and never mount cloud-instance credentials (metadata-service access is a standard escalation path — block it).
  - Output handling: treat the execution result as untrusted input to the next LLM call.
- *Follow-ups:* Why is static analysis of generated code an insufficient control? What does the instance metadata endpoint have to do with this?
- *Red flag answer:* "We check the generated code for dangerous imports before running it."

---

## 12. System design

These are whole-interview prompts. The skeleton is the *structure of a good answer*, not the answer.

**Q.** Design a customer-support assistant over 500K documents for 2M users, in English and three Indian languages.
- *Level:* staff
- *What they are really testing:* whether you scope, quantify and sequence rather than list technologies.
- *Answer skeleton:*
  - Clarify first: QPS and peak, latency SLO, accuracy bar and cost of being wrong, data freshness, languages and their traffic split, on-prem/data-residency constraints, escalation path to humans.
  - Architecture: ingest/CDC → chunk (+contextual augmentation) → hybrid index (per-language analysers) → retrieve → rerank → generate with citations → groundedness check → escalate on low confidence.
  - Size it out loud: 500K docs x ~10 chunks = 5M vectors; at 1024-dim FP16 ≈ 10 GB plus HNSW graph → fits in RAM on one large node, replicate for HA. Model choice driven by the latency SLO and the multilingual requirement.
  - Serving: vLLM-style pool with continuous batching, prefix caching on the shared system prompt (huge here), separate pools for interactive and batch.
  - Reliability and cost: caching layers (exact-match answer cache for FAQ head traffic — often 20-40% of support volume), fallback model, graceful degradation to search-only, per-tenant budgets.
  - Evaluation and ops: groundedness sampling, deflection rate as the business metric, human escalation loop feeding the eval set.
- *Follow-ups:* Where is the biggest cost, and what is the first thing you would cut? What happens when a policy document changes at 2am? How do you prevent answering from a document the user may not see?
- *Red flag answer:* Naming a stack (LangChain + Pinecone + GPT) with no numbers, no failure modes, and no eval plan.

**Q.** Design the serving infrastructure for an internal LLM platform serving 20 teams.
- *Level:* staff
- *What they are really testing:* platform thinking — closest to your existing seniority.
- *Answer skeleton:*
  - Multi-tenancy: quotas in **tokens/s and GPU-seconds**, not QPS; per-tenant rate limiting, priority classes (interactive vs batch), and fair-share scheduling so one team's batch job cannot starve another's chat product.
  - Pooling: shared pools per model, separate pools for interactive vs batch SLOs, multi-LoRA for team-specific fine-tunes on a shared base.
  - Gateway: auth, routing, retries with jitter, circuit breaking, fallback to a smaller model or an external API on capacity loss, request/response logging with PII controls.
  - Capacity: autoscale on KV-pool utilisation and queue depth rather than CPU or QPS; keep warm headroom because model load takes minutes; plan for GPU scarcity with a queue rather than a 503.
  - Chargeback and observability: per-team token accounting from day one, dashboards for cost per team per feature.
  - Model lifecycle: versioned endpoints, canary, deprecation windows, and a shared eval harness so teams can compare before migrating.
- *Follow-ups:* How do you stop one team's 100K-token prompts from destroying everyone's p99? How do you price internally so incentives are right?
- *Red flag answer:* A single shared endpoint with no isolation, no quotas and no priority classes.

**Q.** Design a code-review assistant for a monorepo with 10K commits/day.
- *Level:* staff
- *What they are really testing:* scoping against a hard cost/noise constraint.
- *Answer skeleton:*
  - Scope hard: do not review everything. Route by risk — file paths, diff size, test coverage change, author history, security-sensitive directories. This is a cost and a signal-to-noise decision, both.
  - Context assembly is the hard problem: the diff plus the *right* surrounding code (callers, tests, related definitions) via repo-aware retrieval; naive diff-only review produces confident nonsense about code it cannot see.
  - Precision over recall as the explicit objective: a noisy reviewer gets muted within a week, and the mute is irreversible socially.
  - Verify before posting: run static analysis/tests to confirm a claimed bug where possible; suppress unverifiable low-confidence comments.
  - Cost: 10K commits x ~5K tokens is manageable, but only with routing and prefix caching on the shared instruction block.
  - Metrics: comment acceptance/resolution rate per reviewer and per rule, false-positive rate, and (the real one) defects caught before merge.
- *Follow-ups:* How do you measure whether it actually prevents bugs? How do you handle a 2000-file refactor diff?
- *Red flag answer:* Sending the whole diff to a model on every commit and posting whatever comes back.

**Q.** A product needs sub-200ms end-to-end responses from an LLM. What are your options, in order?
- *Level:* staff
- *What they are really testing:* the latency-budget decomposition.
- *Answer skeleton:*
  - Budget it: network + auth + retrieval + prefill + N x TPOT + post-processing. Write the numbers down before proposing anything.
  - Sub-200ms end-to-end means a small model, a short output, or streaming (TTFT < 200ms while total is longer) — say which interpretation you are solving.
  - Levers in order: shorten the output (the dominant term — constrain format, don't generate prose); smaller/distilled model; quantize weights for decode; prefix caching to kill prefill; speculative decoding; TP for the latency floor; co-locate retrieval; cache whole answers for head traffic.
  - Check the floor first: model bytes ÷ bandwidth × output tokens. If the floor exceeds the budget, the answer is a smaller model, and you should say so immediately.
  - Sometimes the right answer is not an LLM: a classifier or a cache for the 80% head, LLM for the tail.
- *Follow-ups:* Which lever has the best latency-per-engineering-hour? What quality do you lose at each step, and how would you detect it?
- *Red flag answer:* Going straight to "use a faster GPU."

**Q.** Design a document-processing pipeline for 10M PDFs (extraction to structured fields).
- *Level:* staff
- *What they are really testing:* batch-scale thinking, where the tradeoffs are different from serving.
- *Answer skeleton:*
  - This is throughput-only: no latency SLO, so maximise batch size, use the cheapest sufficient model, and run on spot/preemptible capacity with checkpointing.
  - Pipeline: classify document type → route (native text extraction where possible; OCR/VLM only for scans — a large cost difference) → extract with constrained decoding against a schema → validate → route low-confidence to human review.
  - Estimate the bill out loud: 10M x ~3K tokens = 3e10 input tokens; at a self-hosted ~$1/M that is ~$30K, so the model/route choice is a five-figure decision and deserves an experiment.
  - Idempotency and checkpointing keyed by document hash; partial failure must not restart the corpus. Dead-letter queue for permanent failures.
  - Quality: sample-based human audit with a measured error rate per document type, not a global accuracy claim; confidence thresholds tuned per field by cost of error.
- *Follow-ups:* How do you decide the human-review threshold per field? How do you handle a schema change halfway through the run?
- *Red flag answer:* One model for all document types, no routing, no confidence-based human loop.

**Q.** How would you migrate a production feature from an API provider to self-hosted open weights?
- *Level:* staff
- *What they are really testing:* migration risk management.
- *Answer skeleton:*
  - Establish the baseline first: capture real traffic, build the eval set from it, and measure the incumbent's quality, latency and cost. Without this the migration is unfalsifiable.
  - Prompts do not transfer: re-tune per model before comparing, or you will measure prompt transfer and conclude the open model is worse.
  - Shadow mode: run the candidate on live traffic without serving it, compare offline, and find the disagreement cases — that set is where the real regressions hide.
  - Capacity and ops readiness: GPU procurement lead time, autoscaling, on-call, model-load time, and a rollback path that is a config flip.
  - Canary by segment with guardrails on quality, latency and cost; keep the API as a fallback for overflow and for the hard tail.
  - Be explicit about what you are buying: cost at scale, data control, customisability — and what you are paying: headcount and frontier-quality gap.
- *Follow-ups:* What traffic slice do you migrate last, and why? What is your plan if the open model is 3% worse but 5x cheaper?
- *Red flag answer:* A big-bang cutover with a benchmark comparison as the only evidence.

**Q.** Design an evaluation and release process for a team shipping prompt and model changes weekly.
- *Level:* staff
- *What they are really testing:* turning ML into an engineering discipline — the most staff-shaped question here.
- *Answer skeleton:*
  - Prompts are code: version-controlled, reviewed, tagged with an ID that is logged on every response so any output can be traced to its exact prompt version.
  - CI gate: frozen regression eval on every change, with thresholds; deterministic checks first (fast, cheap), judge-based checks second, on a sampled subset for cost.
  - Release: flag-gated, canary by percentage, automatic rollback on guardrail breach, fixed soak time. Same discipline as a backend deploy.
  - Feedback loop: production failures triaged weekly into the eval set; the suite's coverage is the team's real asset.
  - Provider risk: pin model versions where the provider allows, and run the regression suite on any provider-side model update — silent upgrades are a real outage class.
  - Track eval-suite health itself: flag examples everything passes (no signal) and examples nothing passes (broken or mislabelled).
- *Follow-ups:* What is your policy when the eval suite and the A/B disagree? How do you stop the eval suite from rotting?
- *Red flag answer:* Prompts edited in a dashboard by whoever is on duty, with no versioning or rollback.

**Q.** You inherit an LLM feature costing $180K/month. Cut it by 70% without a quality regression. Where do you start?
- *Level:* staff
- *What they are really testing:* cost engineering, prioritised by measurement.
- *Answer skeleton:*
  - Measure before cutting: break spend down by feature, tenant, prompt version, and input vs output tokens. Cost is almost always concentrated — find the head.
  - Usually-large, usually-easy wins: prompt-prefix caching (restructure so static content is first); trimming bloated system prompts and few-shot blocks; capping `max_tokens`; deduplicating redundant calls; an exact/semantic cache for repeated head queries.
  - Routing: send the easy majority to a small model, escalate on a confidence or complexity signal. Typically the single biggest lever, and it needs an eval to prove the quality floor holds.
  - Output length is the dominant per-request cost driver — structured/terse outputs cut spend directly.
  - Then infrastructure: quantization, better batching, self-hosting the high-volume path, reserved/committed capacity instead of on-demand.
  - Gate every change on the regression suite and an online guardrail; "without a quality regression" is a measurable claim, so measure it.
- *Follow-ups:* What would you *not* cut? How do you avoid a slow quality erosion that no single change caused?
- *Red flag answer:* Starting with quantization before looking at the token bill.

---

## 13. Questions the candidate should ask

Asked at the end, these signal that you have run systems like this. Pick 3-4 matched to the role; asking all of them is an interrogation, not a conversation.

1. **What is your cost per million tokens today, and do you know it broken down by feature?** — Teams that cannot answer have no cost attribution, which tells you the maturity level immediately.
2. **What is your eval suite, how big is it, and who owns it?** — The single strongest predictor of whether the team ships improvements or vibes.
3. **When was the last time an eval caught a regression before users did?** — Separates a real suite from a decorative one.
4. **Self-hosted or API, and what drove that decision?** — Reveals whether decisions are made with arithmetic or by default.
5. **What is your p99 time-between-tokens, and do you alert on it?** — Only teams that have actually tuned serving track this.
6. **What fraction of your GPU fleet is doing useful work at p50?** — Utilisation honesty; the answer is often uncomfortable and the reaction is informative.
7. **How do you handle prompt versioning and rollback?** — Tells you whether prompts are engineering artifacts or dashboard text.
8. **What is your most common production failure mode right now?** — A specific answer means they instrument; a vague one means they don't.
9. **How do you decide between fine-tuning and retrieval for a new requirement?** — Probes whether there is a framework or a fashion.
10. **What does the model-upgrade process look like when a provider deprecates a version?** — Migration pain is where LLM teams actually spend their quarters.
11. **Where does this team sit between research and platform — am I optimising kernels, or building the system around the model?** — Sets your expectations, and gets you a straight answer about the actual job.
12. **What would you want me to have shipped six months in for this to be a clear success?** — The single best calibration question at staff level.
13. **How much of the engineering time goes to the model versus the data and evaluation pipeline around it?** — A healthy answer is heavily weighted to the latter; anyone who says "mostly the model" is early.
14. **What is the hardest technical constraint you are living with that you would fix if you could?** — Invites the honest answer and gives you the real scope of the role.

---

## Milestone

The phase is done when you can, **unprepared and out loud in under 10 minutes**, do this end-to-end:

> Given a config (`n_layers`, `d_model`, `n_heads`, `n_kv_heads`, `d_ff`, `vocab`), a
> traffic spec (QPS, prompt-length and output-length distribution) and an SLO, derive the
> parameter count, the KV-cache bytes per token, the decode-latency floor from HBM
> bandwidth, the GPU count for prefill and for decode separately, and the cost per million
> output tokens — then name the first three optimisations you would apply, the failure
> mode of each, and the metric you would gate on.

Artifact that proves it: `interview/sizing.md` — your own worked version of that derivation
for **two** models (one ~8B, one ~70B) at two context lengths, with every assumption
stated, checked against a real `config.json` you downloaded and against a measured
benchmark from your own `vLLM`-on-Colab run. If a number in it came from this file rather
than from your own arithmetic, it does not count.
