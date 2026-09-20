# Phases 6–7 — Efficiency/optimization, and post-training + distributed

Lesson-level plan. This is a syllabus, not the lessons. Claude teaches each entry
live using the 10-step loop in `CLAUDE.md`; the entries below fix *what* is taught,
in *what order*, with *what experiment*.

Lesson directories are created at the repo root in teaching order (`<NN>_topic/`),
continuing the numbering that Phase 5 ended on. Do not hard-code directory numbers
from this file — read `curriculum/PROGRESS.md` for the next free number.

**Hard prereqs carried in from earlier phases:** autograd and the backward pass;
multi-head attention implemented from scratch; KV cache; RoPE; the full decoder block
(RMSNorm, SwiGLU, residual stream); a working tiny training loop with AdamW; and the
prefill/decode distinction from the inference phase. Several Phase 6 lessons are
meaningless without RoPE (6.20) and without KV cache (6.5, 6.16), so if those are not
genuinely solid, go back before starting.

---

## What you can claim after this phase

- I can read a model card that says "4-bit AWQ, 128-group, 32k context via YaRN" and
  say exactly what was quantized, what was not, what quality risk each choice carries,
  and what it does to the memory bandwidth bill at decode time.
- I can derive arithmetic intensity for a matmul and for attention, and predict from
  hardware specs alone whether a given inference workload is compute-bound or
  bandwidth-bound — and therefore which optimizations are pointless.
- I can explain FlashAttention from the online-softmax recurrence up, including why
  the backward pass recomputes instead of stores, and I have a working implementation
  of the recurrence I wrote myself.
- I can implement LoRA from scratch (not `peft.get_peft_model`), justify rank/alpha,
  merge adapters back into base weights, and do the optimizer-memory arithmetic that
  motivates PEFT in the first place.
- I can derive DPO from the RLHF objective, implement its loss, explain what GRPO drops
  and why that matters for memory, and describe concretely how each of them gets hacked.
- I can take "13B model, 4×A100-40GB, 8k sequence" and work out on paper what fits,
  which ZeRO stage or parallelism axis I need, what the per-step communication volume
  is, and where the failure domain boundaries sit.

---

# PHASE 6 — Efficiency and optimization

## 6A. Number formats

### 6.1 — Bit layouts: FP32, FP16, BF16
- **Question it answers:** What is actually stored in a floating-point number, and what does "dynamic range vs precision" mean in bits?
- **Prereqs:** binary representation; nothing DL-specific.
- **Experiment:** Take 12 scalars spanning 1e-8 to 1e6. Round-trip each through `torch.float16` and `torch.bfloat16` and print the exact stored value plus relative error. Decode one of them by hand: pull sign/exponent/mantissa out with `np.float32(x).view(np.uint32)` and bit-shifts. Find FP16's largest finite value (65504) and the smallest normal, empirically, by doubling/halving until inf/0.
- **Shapes to nail:** none — this is scalar-level. Nail instead: FP32 = 1/8/23, FP16 = 1/5/10, BF16 = 1/8/7. BF16 has FP32's exponent field and a truncated mantissa.
- **Predict-first prompt:** "BF16 has 7 mantissa bits vs FP16's 10. Which one loses a 1e-8 gradient to zero, and which one loses precision on a value near 1.0? Answer before running."
- **Runs on:** CPU (MPS also, but do it on CPU so bit-views work cleanly).
- **Interview hooks:** Why does BF16→FP32 conversion require no arithmetic, just a mantissa pad? What is the relative precision (epsilon) of BF16 and why does that not matter for gradients? Why does FP16 need loss scaling but BF16 usually does not?
- **Common misconception:** "BF16 is more accurate than FP16 because it's newer." It is strictly *less* precise (7 vs 10 mantissa bits) and wins purely on exponent range.

### 6.2 — Why BF16 beat FP16 for training
- **Question it answers:** What actually broke in FP16 training, and what does mixed precision keep in FP32?
- **Prereqs:** 6.1; backward pass; AdamW state.
- **Experiment:** Train the tiny model from the earlier training-loop lesson for ~50 steps, casting gradients to FP16 before the step. Count how many gradient elements flush to zero (`(g.half()==0) & (g!=0)`). Then repeat with a loss scale of 2**12 and count again. Then repeat in BF16 with no scaling.
- **Shapes to nail:** parameter, gradient, Adam `exp_avg`, `exp_avg_sq` are all the same shape as the parameter — that is the 4× memory fact 7.6 will use. FP32 master copy = a 5th.
- **Predict-first prompt:** "Loss scaling multiplies the loss by 2^k before backward. Why does that rescue small gradients, and where must the scale be undone so the optimizer step is unchanged?"
- **Runs on:** MPS or CPU (BF16 is supported on MPS in torch 2.10; verify with a one-line `torch.ones(2, device='mps', dtype=torch.bfloat16)` smoke test before building the lesson on it).
- **Interview hooks:** Why does mixed precision keep FP32 master weights even when compute is BF16? What is dynamic loss scaling and how does the scaler detect overflow? Where does BF16's low precision actually bite — and why is the accumulator inside a tensor-core matmul still FP32?
- **Common misconception:** "Mixed precision halves memory." Weights+grads shrink, but FP32 master weights and FP32 Adam state usually dominate, so the real win is activation memory and tensor-core throughput.

### 6.3 — FP8, INT8, INT4, NF4
- **Question it answers:** How do sub-16-bit formats differ structurally — floating (E4M3/E5M2), integer-affine, and quantile-based (NF4)?
- **Prereqs:** 6.1.
- **Experiment:** Build three 4-bit grids in numpy over the same weight tensor sampled from `N(0,1)`: (a) uniform INT4 symmetric, (b) uniform INT4 affine, (c) NF4's 16 quantile levels of a normal distribution. Round the tensor to each and compare mean squared error. Print `torch.finfo(torch.float8_e4m3fn)` and `torch.float8_e5m2` to see range vs precision tradeoff inside 8 bits.
- **Shapes to nail:** a `[out, in]` weight quantized per-tensor stores one scale; per-channel stores `[out]` scales; group-size-128 stores `[out, in/128]` scales. Write those three numbers down for a real `[4096, 4096]` matrix.
- **Predict-first prompt:** "NF4 assumes weights are roughly normally distributed. Which of the three grids should win the MSE comparison on `N(0,1)` data, and what distribution would make it lose?"
- **Runs on:** CPU/MPS for the numpy grids and for `torch.float8_*` as a *storage* dtype. FP8 *matmul* needs Hopper/Ada tensor cores — **needs CUDA** (H100 on runpod/modal, or skip; the format lesson is what matters here, not the kernel).
- **Interview hooks:** Why two FP8 formats — where is E4M3 used and where E5M2? Why is NF4 "information-theoretically optimal for normally distributed data" and what is the assumption hiding in that claim? Why is INT4 fine for weights and hostile to activations?
- **Common misconception:** "4-bit means 4 bits per weight." With group scales in FP16 it is ~4.5 bits; QLoRA's double quantization exists precisely to claw back ~0.37 of those extra bits per parameter.

## 6B. The performance model

### 6.4 — Roofline and arithmetic intensity, applied to matmul
- **Question it answers:** For a given op, is the GPU waiting on FLOPs or on bytes?
- **Prereqs:** matmul FLOP counting (2·M·N·K).
- **Experiment:** By hand, for `[M,K] @ [K,N]`: FLOPs = 2MNK, bytes moved = 2(MK + KN + MN) in BF16. Compute intensity = FLOPs/bytes for (M,N,K) = (1,4096,4096) and (4096,4096,4096). Then measure: time both on MPS with proper sync (forward-reference 6.25's harness, or use a crude `torch.mps.synchronize()` + repeat loop) and compute achieved FLOP/s and GB/s.
- **Shapes to nail:** M=1 is the decode case (one token), M=4096 the prefill case. Same weights, same K and N, wildly different intensity.
- **Predict-first prompt:** "Both matmuls read the same 4096×4096 weight matrix. One does 4096× more FLOPs. Predict the ratio of their wall-clock times *before* measuring — and say what your prediction implies about which regime each sits in."
- **Runs on:** MPS (M3 unified memory has very different bandwidth/FLOPs ratio than an H100 — that difference is itself the lesson; look up both and compute the ridge point for each).
- **Interview hooks:** What is the ridge point of an H100 in BF16, and what intensity must a kernel exceed to be compute-bound? Why does batching help decode but not prefill? Why do quantized weights speed up decode even when the matmul runs in FP16?
- **Common misconception:** "Bigger matmul = slower." At fixed weight size, adding rows to the activation is nearly free until you cross the ridge point.

### 6.5 — Attention's arithmetic intensity, and why decode is bandwidth-bound
- **Question it answers:** In autoregressive decoding, what exactly is the bottleneck, and how do you compute it in GB/token?
- **Prereqs:** 6.4; KV cache; multi-head attention shapes.
- **Experiment:** For a 7B-class config (L=32, d_model=4096, n_heads=32, GQA n_kv=8, BF16), compute on paper: (a) bytes of weights read per decoded token, (b) bytes of KV cache read per decoded token at context 1k / 8k / 32k, (c) FLOPs per decoded token. Divide by a real device bandwidth to get a theoretical tokens/sec ceiling. Then measure actual decode tokens/sec for a small local model on MPS and compare against your ceiling.
- **Shapes to nail:** KV cache per layer = `[B, n_kv_heads, T, head_dim]` ×2 (K and V). Total bytes = `2 · L · B · n_kv · head_dim · T · dtype_size`. Decode attention: `Q[B,H,1,d] @ K^T[B,H,d,T] -> [B,H,1,T]`.
- **Predict-first prompt:** "At what context length does KV-cache traffic exceed weight traffic for this config? Compute it before you look at the measurement."
- **Runs on:** MPS.
- **Interview hooks:** Why does GQA/MQA help decode far more than it helps prefill? Why is decode latency roughly independent of batch size until you run out of memory? Given a target of 50 tok/s/user at 32k context, what does that imply about the hardware you must buy?
- **Common misconception:** "Attention is the expensive part of a decoder-only LLM." At short context the MLP weights dominate the bytes; attention dominates only as T grows, and it is KV *reads*, not attention FLOPs, that hurt.

## 6C. FlashAttention

### 6.6 — The online softmax recurrence, from scratch
- **Question it answers:** How do you compute a numerically stable softmax over a stream of values you never hold in memory at once?
- **Prereqs:** softmax with max-subtraction; nothing else.
- **Experiment:** In numpy, a `online_softmax(scores)` function that walks a length-16 vector in chunks of 4, carrying only `(m, l)` — running max and running sum of `exp(x - m)`. On each chunk: new max `m'`, rescale the old sum by `exp(m - m')`, add the chunk's contribution. Assert it matches `scipy`/manual softmax to 1e-6. Then extend to carrying a running *output* accumulator `o` rescaled the same way, weighted by V chunks.
- **Shapes to nail:** state is `m: scalar`, `l: scalar`, `o: [d_head]` per query row. For a real tile: `m: [B,H,Tq]`, `l: [B,H,Tq]`, `o: [B,H,Tq,d]`.
- **Predict-first prompt:** "You have processed chunk 1 and hold `(m, l)`. Chunk 2 contains a value larger than `m`. Write down, before coding, the exact correction factor that must be applied to the stored `l` and `o`."
- **Runs on:** CPU (numpy). This is deliberately CPU — the idea is the point, not the kernel.
- **Interview hooks:** Why is the rescaling factor `exp(m_old - m_new)` and never the reverse? What breaks if you skip the max entirely? Why does this make attention's memory O(T) instead of O(T²) without changing the math at all?
- **Common misconception:** "FlashAttention approximates attention." The output is exact up to floating-point associativity; only the memory schedule changed.

### 6.7 — Tiling and IO-awareness: FlashAttention forward
- **Question it answers:** What does FlashAttention actually avoid writing to HBM, and why does that make it faster despite doing *more* arithmetic?
- **Prereqs:** 6.6, 6.4, GPU memory hierarchy (HBM vs SRAM/shared memory — teach the 3-level model here if it has not appeared).
- **Experiment:** Torch-CPU implementation: loop over K/V tiles of size `Bc=4` for query tiles of size `Br=4` with `T=16, d=8`, carrying `(m, l, o)` from 6.6. Assert `allclose` against the naive `softmax(QK^T/√d) @ V`. Add a counter for bytes that a naive implementation would write for the `[T,T]` score matrix vs what the tiled version writes.
- **Shapes to nail:** naive materializes `S: [B,H,T,T]`. Tiled materializes only `S_tile: [Br,Bc]` plus `(m,l): [Br]` and `O: [T,d]`. Write out the memory of each for B=1,H=32,T=8192 in BF16 — this is the "why" in one number.
- **Predict-first prompt:** "For B=1, H=32, T=8192, BF16 — how many GB is the materialized attention-score tensor? Guess before computing. Then say why a 80GB GPU still OOMs on it."
- **Runs on:** CPU/MPS for the reference implementation. The *real* CUDA kernel (`flash-attn`) — **needs CUDA**; on MPS use `torch.nn.functional.scaled_dot_product_attention`, which dispatches to a fused Metal kernel with the same O(T) memory property but different kernel internals. Do not claim you ran FlashAttention on the Mac.
- **Interview hooks:** Why is FlashAttention faster while performing more FLOPs than the naive version? Why does causal masking let the kernel skip whole tiles, and what does that do to the load balance? What limits the tile size — and what happens on a GPU with smaller SRAM per SM?
- **Common misconception:** "It's faster because it's written in CUDA." It is faster because HBM round-trips of the `T×T` matrix dominate, and tiling removes them — the IO count, not the FLOP count, is the objective it minimizes.

### 6.8 — The backward pass: recomputation instead of storage
- **Question it answers:** If the attention matrix was never stored, how does the backward pass get its gradients?
- **Prereqs:** 6.7; backprop through softmax and matmul.
- **Experiment:** Extend the CPU tiled implementation to a manual backward for one tile: save only `O` and the log-sum-exp stats `L = m + log(l)`, recompute `S_tile` on the fly, and verify `dQ, dK, dV` against `torch.autograd.grad` on the naive version to 1e-5.
- **Shapes to nail:** saved for backward = `Q,K,V: [B,H,T,d]`, `O: [B,H,T,d]`, `L: [B,H,T]`. Not saved: anything `[B,H,T,T]`.
- **Predict-first prompt:** "Recomputing `S` in the backward pass costs extra FLOPs. Under what arithmetic-intensity condition is that trade a win? Answer using 6.4's framework."
- **Runs on:** CPU.
- **Interview hooks:** Why store log-sum-exp rather than `m` and `l` separately? How does this relate to generic activation checkpointing (7.28) — what's the same, what's different? Why is `dS = P ∘ (dP - rowsum(dP ∘ P))` and where does that rowsum come from?
- **Common misconception:** "Recomputation is a memory-saving hack that costs speed." Here it saves both, because the recompute is compute-bound and the storage it replaces was bandwidth-bound.

### 6.9 — FlashAttention-2 and -3: what changed
- **Question it answers:** What were the actual deltas, and which are algorithmic vs hardware-generation-specific?
- **Prereqs:** 6.7, 6.8.
- **Experiment:** Reading + whiteboard lesson, no new implementation. Deliverable is a one-page comparison the learner writes: FA-2 = fewer non-matmul FLOPs (defer the rescaling division to the end), parallelize over the *sequence* dimension as well as batch/heads so occupancy holds at small batch, better warp partitioning to cut shared-memory traffic. FA-3 = Hopper-specific: warp specialization into producer/consumer warpgroups using TMA + WGMMA asynchrony, ping-pong scheduling to overlap softmax with matmul, and FP8 support; ~1.5–2.0× over FA-2 on H100 (≈75–85% BF16 utilization vs FA-2's ~35% there), and ~1.3 PFLOP/s in FP8.
- **Shapes to nail:** none new. Instead nail the parallelization axes: (batch, heads) for FA-1 vs (batch, heads, query-blocks) for FA-2.
- **Predict-first prompt:** "FA-1 parallelizes over batch×heads. At batch=1, T=32k inference, how many thread blocks does that give on a 132-SM H100 — and what does that tell you about why FA-2 added an axis?"
- **Runs on:** **Needs CUDA** to benchmark at all, and FA-3 needs Hopper specifically. Treat this lesson as conceptual; if you want a number, rent one H100 hour on runpod/modal and run the upstream benchmark script once.
- **Interview hooks:** Why does non-matmul FLOP count matter so much when tensor cores are ~16× faster than the FP32 pipeline? What is warp specialization buying that a plain loop cannot? Why is FP8 attention harder than FP8 matmul (hint: what is the dynamic range of `exp`)?
- **Common misconception:** "FlashAttention-3 is just FA-2 with FP8." The asynchrony/warp-specialization work is the bulk of the gain and is orthogonal to precision.

## 6D. Quantization

### 6.10 — Quantization theory: affine vs symmetric, and granularity
- **Question it answers:** Given a float tensor, what are the exact formulas for quantize/dequantize, and what does choosing a granularity cost in storage?
- **Prereqs:** 6.3.
- **Experiment:** Implement `quantize(x, n_bits, symmetric, group_size)` and `dequantize` in ~15 lines of torch. Apply to a real `[512, 512]` weight slice pulled from a small local model. Report reconstruction MSE for: per-tensor symmetric, per-channel symmetric, group-128 affine. Then inject one outlier (set `x[0,0] = 50·std`) and rerun all three.
- **Shapes to nail:** `x:[512,512]` → per-tensor: `scale: []`; per-channel (output dim): `scale:[512,1]`; group-128 along input dim: `scale:[512,4]`, `zero:[512,4]`. Overhead bits/weight = `(scale_bits + zero_bits) / group_size`.
- **Predict-first prompt:** "You inject one 50-sigma outlier into a 512×512 tensor. Rank the three granularities by how much their reconstruction error degrades. Justify before running."
- **Runs on:** CPU/MPS.
- **Interview hooks:** When is a zero-point strictly necessary rather than an optimization? Why quantize weights per *output* channel and not per input channel? What does group size trade off, in both bits and kernel efficiency?
- **Common misconception:** "Smaller MSE on the weights means better model quality." Layer-wise reconstruction error is a proxy that fails; what matters is the output error after the whole stack, which is why GPTQ/AWQ optimize different objectives (6.12).

### 6.11 — Activation outliers, LLM.int8() and SmoothQuant
- **Question it answers:** Why is quantizing activations qualitatively harder than quantizing weights in an LLM?
- **Prereqs:** 6.10; forward hooks.
- **Experiment:** Register a forward hook on the input of a few `Linear` layers of a small local model (TinyLlama/Qwen-0.5B class). Run 8 prompts. For each layer, plot/print per-channel max-abs of the activation. Find the handful of channels that are 10–100× the median — the same channel indices across tokens and prompts. Then implement SmoothQuant's migration by hand: pick `s_j = max|X_j|^α / max|W_j|^(1-α)` with α=0.5, divide activations by `s`, multiply the corresponding weight rows by `s`, and show the activation's per-channel range collapses while the product is unchanged.
- **Shapes to nail:** activation `[B,T,d_in]` → per-channel stat over `d_in` (reduce over B,T). Smoothing vector `s: [d_in]`, folded into `W: [d_out, d_in]` by scaling columns.
- **Predict-first prompt:** "SmoothQuant multiplies weights by exactly what it divides activations by. Why is that free at inference time, and in which layer does the division get absorbed so it costs nothing?"
- **Runs on:** MPS (hooks and statistics are plain torch). The fused INT8 kernels — **needs CUDA**.
- **Interview hooks:** Why does per-token activation quantization plus per-channel weight quantization sidestep most of this, and what kernel cost does it add? What does LLM.int8() do differently (decompose the outlier columns into a separate FP16 matmul) and why is that slow? Which architectural choices make outliers worse?
- **Common misconception:** "Outliers are noise / bad training." They are systematic, live in specific channels, and appear to carry real function — removing them destroys the model.

### 6.12 — GPTQ and AWQ
- **Question it answers:** Two weight-only PTQ methods with different objectives — what is each actually minimizing, and what does each need from you?
- **Prereqs:** 6.10, 6.11.
- **Experiment:** Minimal GPTQ intuition on a single small layer, CPU: for `W:[64,64]` and calibration activations `X:[256,64]`, compute `H = 2 X^T X`, quantize columns left-to-right, and after each column push the induced error into the *not-yet-quantized* columns using the Hessian-derived update. Compare final `||XW - XŴ||²` against naive round-to-nearest. Separately, AWQ intuition: find the 1% of channels with the largest activation magnitude, scale those weight channels up before rounding (and scale the activation down), measure the same output error.
- **Shapes to nail:** `X:[N,d_in]`, `W:[d_in,d_out]`, `H:[d_in,d_in]`. Calibration set is typically 128 sequences of 512–2048 tokens — note how small that is.
- **Predict-first prompt:** "GPTQ minimizes layer *output* error given calibration data; round-to-nearest minimizes *weight* error. Which objective is correct, and construct a case where they disagree."
- **Runs on:** CPU/MPS for the toy. Running real `autogptq`/`autoawq` on a 7B — **needs CUDA** (their kernels are CUDA-only); the honest local path is to *consume* a pre-quantized AWQ/GPTQ checkpoint via a CPU/Metal runtime, or do the quantization pass on a rented L4/A10 for ~$0.5.
- **Interview hooks:** What is the failure mode of calibration-set overfitting, and how would you detect it? Why is AWQ's scaling search cheaper than GPTQ's Hessian pass? Which of the two would you pick for a model you must re-quantize weekly, and why?
- **Common misconception:** "GPTQ and AWQ are both 4-bit so they're interchangeable." They differ in calibration cost, robustness to calibration-domain shift, and kernel availability — pick on those axes, not bit width.

### 6.13 — GGUF and llama.cpp k-quants
- **Question it answers:** What does `Q4_K_M` mean, byte for byte, and why is this format the one that runs on your Mac?
- **Prereqs:** 6.10.
- **Experiment:** Actually run it: `llama.cpp` builds with Metal on M3. Download one small model in two quants (e.g. `Q8_0` and `Q4_K_M`), measure tokens/sec and RSS for both, and diff the outputs on 5 fixed prompts at temperature 0. Then read the block struct: K-quants use 256-element super-blocks split into 32-element sub-blocks, each sub-block carrying its own scale, and those scales are themselves quantized (6-bit for Q4_K) — a second-level quantization exactly like QLoRA's double quantization. `_S/_M/_L` are mixes that keep chosen tensors (attention, `output`) at higher precision.
- **Shapes to nail:** bits-per-weight accounting for Q4_K: 4 bits of payload + quantized sub-block scales/mins amortized over 32, ≈4.5 bpw. Compute the file size that predicts for a 7B model and compare with the actual download.
- **Predict-first prompt:** "Q4_K_M is ~4.5 bits/weight and Q8_0 is ~8.5. Predict the ratio of their decode tokens/sec on your M3 *from bandwidth alone* (6.5), then measure."
- **Runs on:** MPS/Metal — this is the best local-hardware lesson in Phase 6. Use it.
- **Interview hooks:** What is an importance matrix (imatrix) and how does it change which weights get the error? Why does llama.cpp keep some tensors at higher precision, and which ones? Why is GGUF weight-only and what would it take to do activations too?
- **Common misconception:** "Q4 is 2× faster than Q8." Decode speed tracks bytes read, so it is closer to the bpw ratio — and prompt processing (compute-bound) may not speed up at all.

### 6.14 — Weight-only vs weight+activation; PTQ vs QAT
- **Question it answers:** When does a quantized model actually get *faster* rather than just smaller, and when do you have to retrain?
- **Prereqs:** 6.4, 6.10–6.13.
- **Experiment:** Decision-table lesson plus one tiny QAT run: take the small model from the earlier training phase, insert fake-quant (quantize-dequantize in the forward, straight-through estimator in the backward — implement the STE by hand as `x + (q(x) - x).detach()`), fine-tune 200 steps at 4-bit weights, and compare held-out loss against PTQ round-to-nearest at the same bit width.
- **Shapes to nail:** none new; nail instead which matmul input is low-precision in each scheme. Weight-only: dequantize to BF16 before the matmul → saves bytes, not FLOPs. W8A8: both operands INT8 → uses INT8 tensor cores → saves both.
- **Predict-first prompt:** "A weight-only 4-bit model dequantizes to BF16 before every matmul, so the arithmetic is identical to the unquantized model. Why is it still faster at decode — and why is it *not* faster at prefill?"
- **Runs on:** MPS/CPU for the STE experiment. INT8 tensor-core speedups — **needs CUDA**.
- **Interview hooks:** What does the straight-through estimator actually assume, and where is it wrong? Why is QAT rarely used for 7B+ LLMs but standard on-device for small models? What is "quantization-aware distillation" and where does it sit between the two?
- **Common misconception:** "Quantization always speeds things up." In a compute-bound regime (large-batch prefill, training) weight-only quantization can be *slower* because of dequantization overhead.

### 6.15 — Measuring quality loss honestly
- **Question it answers:** How do you prove a quantized model is or is not degraded, and why does perplexity lie?
- **Prereqs:** 6.13; perplexity from the training phase.
- **Experiment:** For the Q8_0 vs Q4_K_M pair from 6.13: (1) WikiText-2 perplexity on both; (2) mean KL divergence between full-precision and quantized next-token distributions over a few hundred positions — the sharper signal; (3) a small task eval (50 hand-picked prompts, exact-match or a rubric) with greedy decoding and a fixed seed; (4) report the *variance* across 3 runs. Note explicitly which of the four would have caught a real regression.
- **Shapes to nail:** logits `[B,T,V]`; KL is a reduction over `V`, averaged over `B·T` positions. Compare like-for-like: same tokenizer, same positions, same prompt formatting.
- **Predict-first prompt:** "A quantized model has perplexity 0.02 higher than the original but fails 30% more of your task evals. Give two concrete mechanisms that produce that pattern."
- **Runs on:** MPS/CPU.
- **Interview hooks:** Why is KL-to-the-original a better quantization metric than absolute perplexity? Why do long-generation and tool-calling tasks degrade before perplexity does? How would you build a quantization regression gate for CI?
- **Common misconception:** "ΔPPL < 0.1 means lossless." Perplexity averages over easy tokens; failures concentrate in the rare, decisive ones.

### 6.16 — KV-cache quantization
- **Question it answers:** Given 6.5 showed KV traffic dominates at long context, what happens if you store it in 8 or 4 bits?
- **Prereqs:** 6.5, 6.10, KV cache implementation.
- **Experiment:** In your own from-scratch KV cache, add per-token per-head quantization of K and V to INT8 (and INT4), dequantizing on read. Measure: cache bytes, and output divergence from the FP16 path over a 200-token generation. Quantize K only, then V only, then both, and report which hurts more.
- **Shapes to nail:** cache `[B, n_kv, T, d_head]`; quantize along `d_head` per (token, head) → scales `[B, n_kv, T, 1]`. Overhead = 2 bytes per (token, head) — compute what fraction that is for `d_head=128`.
- **Predict-first prompt:** "K feeds a softmax; V is a linear average. Predict which one is more sensitive to 4-bit quantization, and give the reason before measuring."
- **Runs on:** MPS.
- **Interview hooks:** Why does RoPE make K harder to quantize per-channel? How does KV quantization interact with paged attention and block allocation? At what context length does KV quantization beat just using a smaller model?
- **Common misconception:** "KV quantization and weight quantization are the same problem." KV is written once and read T times, is per-request, and cannot be calibrated offline.

## 6E. Mixture of Experts

### 6.17 — An MoE layer from scratch
- **Question it answers:** What literally replaces the dense MLP, and what does the router compute?
- **Prereqs:** the decoder block's MLP; softmax.
- **Experiment:** Build a top-2 MoE FFN: `B=2, T=4, d_model=8, d_ff=16, n_experts=4, top_k=2`. Router = `nn.Linear(8, 4)`. Softmax over experts, take top-2, renormalize the two gate values, run each token through its two experts, combine weighted. Print which expert each of the 8 tokens picked. Verify parameter count vs a dense FFN and FLOPs vs a dense FFN — and note they move in opposite directions.
- **Shapes to nail:** `x:[B,T,d]` → flatten to `[B·T, d]`; `router_logits:[B·T, E]`; `topk_idx:[B·T, k]`; `gates:[B·T, k]`; each expert sees a ragged `[n_tokens_e, d]`. Output back to `[B,T,d]`.
- **Predict-first prompt:** "8 tokens, 4 experts, top-2. If routing were uniform, how many tokens does each expert see? Now predict what the untrained random router actually does — and why that is the load-balancing problem."
- **Runs on:** CPU/MPS.
- **Interview hooks:** Why top-k and not a soft mixture of all experts? Why is the gate value multiplied into the output rather than just used for selection (what would the router's gradient be otherwise)? Why are experts placed in the FFN and not in attention?
- **Common misconception:** "MoE means the model is smaller." It has far *more* parameters; only the FLOPs per token are smaller.

### 6.18 — Load balancing, capacity, and expert parallelism
- **Question it answers:** What breaks at scale in 6.17, and what does MoE cost you in a distributed system?
- **Prereqs:** 6.17; 6.5 (memory bandwidth). Cross-reference forward to 7.21 for collectives.
- **Experiment:** Add Switch-Transformer-style auxiliary loss to 6.17: `aux = E · Σ_e f_e · P_e` where `f_e` is the fraction of tokens routed to expert `e` and `P_e` the mean router probability for `e`. Train the tiny router for 200 steps on random data with and without the aux term; plot the routing histogram in both cases. Then add a capacity factor and count dropped tokens.
- **Shapes to nail:** `f: [E]`, `P: [E]`, aux is a scalar. Capacity per expert = `capacity_factor · (B·T·k / E)`. Under expert parallelism, the dispatch is an all-to-all of `[B·T·k, d]` activations, twice per MoE layer.
- **Predict-first prompt:** "A well-balanced 8-expert top-2 router still needs capacity_factor > 1.0. Why — what is the source of the imbalance that balancing cannot remove?"
- **Runs on:** CPU/MPS for the aux-loss experiment. Real expert parallelism needs multiple GPUs — **needs CUDA (multi-GPU)**. Note: `gloo` does not implement `all_to_all`, so you cannot even emulate the dispatch collective on CPU without an `all_gather`-plus-slice stand-in; write that stand-in, it teaches the communication volume.
- **Interview hooks:** Why is MoE inference memory-hungry but FLOP-cheap, and what does that do to your cost-per-token at batch size 1 vs 256? What is the failure mode when one expert becomes dominant? Why does expert parallelism interact badly with sequence-length variance across a batch?
- **Common misconception:** "MoE is free capacity." You pay in HBM (all experts resident), in all-to-all bandwidth, in load-balance fragility, and in training instability.

## 6F. Long context

### 6.19 — Sliding window, attention sinks, StreamingLLM
- **Question it answers:** If you truncate attention to a window, why does the model collapse — and what one-line fix rescues it?
- **Prereqs:** causal attention; softmax; 6.16.
- **Experiment:** With a small local model, generate 512 tokens three ways: (a) full attention, (b) evict the oldest KV entries to keep a 128-token window, (c) same window but *always* keep the first 4 tokens. Compare perplexity/coherence. Then inspect an attention map and confirm the large mass sitting on the first tokens regardless of content.
- **Shapes to nail:** rolling cache `[B, n_kv, W, d]` with a write pointer; sink+window cache = `[B, n_kv, 4+W, d]`. Critically, position ids in the cache must be *relative to cache position*, not original token index.
- **Predict-first prompt:** "Softmax weights must sum to 1. If a head has nothing relevant to attend to, where does its probability mass go — and what does that imply about evicting token 0?"
- **Runs on:** MPS.
- **Interview hooks:** Why does StreamingLLM give infinite-length *streaming* but not infinite *context*? Why does the position-id remapping matter more than the eviction policy? How does this relate to Mistral-style sliding-window attention at *training* time, which is a different thing?
- **Common misconception:** "Attention sinks mean the first tokens are semantically important." They are a softmax normalization artifact; a learned dummy token works just as well.

### 6.20 — RoPE scaling: position interpolation, NTK-aware, YaRN
- **Question it answers:** How do you take a model trained at 4k to 32k without retraining from scratch, and why do naive approaches fail?
- **Prereqs:** RoPE implemented from scratch (frequencies `θ_i = base^(-2i/d)`, wavelengths, rotation by position).
- **Experiment:** In numpy, for `d_head=64` and base=10000, compute each dimension's wavelength `λ_i = 2π/θ_i`. Sort them: some dimensions complete many rotations inside 4k tokens, some never complete one. Then implement three transforms and plot the resulting `θ` per dimension: (a) linear position interpolation (`pos/s`), (b) NTK-aware base scaling (`base → base·s^(d/(d-2))`), (c) NTK-by-parts / YaRN, which interpolates only the *high-frequency* (short-wavelength) dims via a ramp and leaves the low-frequency dims extrapolating, plus an attention temperature `√t = 0.1·ln(s) + 1` applied to the logits.
- **Shapes to nail:** `θ: [d_head/2]`; `cos/sin: [T, d_head/2]` broadcast to `[B,H,T,d_head]`. The ramp in YaRN is a per-dimension `[d_head/2]` mask in [0,1].
- **Predict-first prompt:** "Linear interpolation divides every position by `s`. Which dimensions does that hurt most — the ones with short wavelengths or long ones — and what does the model lose as a result?"
- **Runs on:** CPU/MPS for the math and plots; applying a YaRN config to a real model and evaluating is MPS-feasible on a ≤1B model.
- **Interview hooks:** Why does NTK-aware scaling work without fine-tuning while linear PI needs it? What is the attention-temperature term correcting for, and why does it become necessary as `s` grows? What does changing `rope_theta` at pretraining time (10k → 500k) buy versus post-hoc scaling?
- **Common misconception:** "Context extension is just a config change." Every method still needs some continued pretraining at the target length to be genuinely good; the config change only avoids catastrophic failure.

### 6.21 — "Long context" vs "uses long context well"
- **Question it answers:** What is the difference between a model that accepts 128k tokens and one that can use them?
- **Prereqs:** 6.19, 6.20.
- **Experiment:** Build a 3-tier probe on a small long-context model: (1) needle-in-a-haystack — a single fact at varying depths; (2) *multi*-needle — 4 facts, all required in the answer; (3) aggregation — "how many times does X appear", which cannot be solved by retrieval. Plot accuracy vs context length for all three. Report where they diverge.
- **Shapes to nail:** none new — this is an evaluation-design lesson. Nail the axes instead: depth (%) × context length × task type.
- **Predict-first prompt:** "A model scores 100% on single-needle at 128k. Predict its multi-needle and aggregation scores, and say which failure you expect to appear first."
- **Runs on:** MPS for a ≤1B long-context model; larger models via an API or a rented GPU. The cheap honest version is a 1B model at 32k.
- **Interview hooks:** Why does single-needle NIAH saturate and stop discriminating? What does "effective context length" mean operationally, and how would you measure it for a model you are about to deploy? When is RAG still correct even with a 1M-token window?
- **Common misconception:** "Bigger context window replaces retrieval." Cost is O(T) in KV bytes per token and attention quality degrades with distance; retrieval is often both cheaper and more accurate.

## 6G. Model compression and measurement

### 6.22 — Knowledge distillation
- **Question it answers:** How does a small model learn from a large one's *distribution* rather than from labels?
- **Prereqs:** softmax, cross-entropy, KL divergence.
- **Experiment:** Distill a 4-layer student from an 8-layer teacher (both tiny, from the earlier training phase) on the same data. Loss = `α·CE(student, labels) + (1-α)·T²·KL(student/T ‖ teacher/T)`. Sweep `T ∈ {1, 2, 5}` and `α ∈ {0, 0.5, 1}` for 300 steps each; compare held-out loss against training the student alone.
- **Shapes to nail:** teacher and student logits both `[B,T,V]`; KL reduces over `V`. If vocabularies differ, this does not work — say why out loud.
- **Predict-first prompt:** "Why is the `T²` factor there? Work out what happens to the magnitude of the gradient of the softened KL as `T` grows, before running."
- **Runs on:** MPS/CPU.
- **Interview hooks:** What are "dark knowledge" and why do wrong-class probabilities carry signal? When is on-policy/sequence-level distillation (student generates, teacher scores) better than logit matching, and what does it cost? How do modern small models actually get distilled when teacher logits are unavailable?
- **Common misconception:** "Distillation = training on the teacher's outputs." Training on sampled text is a much weaker signal than matching the full distribution; they are different algorithms with different sample efficiency.

### 6.23 — Pruning: structured vs unstructured
- **Question it answers:** Why does removing 50% of weights usually not make anything faster?
- **Prereqs:** 6.4; matmul kernel intuition.
- **Experiment:** On one `[1024,1024]` linear layer: (a) magnitude-prune 50% of individual weights, store as dense-with-zeros, benchmark — unchanged; (b) store as `torch.sparse_csr`, benchmark — likely *slower*; (c) structured: drop 50% of the *rows* (and the matching columns of the next layer), benchmark — genuinely ~2× and shape-visible. Measure quality loss on all three with a small model.
- **Shapes to nail:** structured pruning changes shapes: `[d_ff, d] → [d_ff/2, d]` and the down-projection `[d, d_ff] → [d, d_ff/2]`. Unstructured does not change any shape — that is exactly the problem.
- **Predict-first prompt:** "A 50%-sparse dense tensor has the same shape and the same byte count. Predict what the matmul time does, and say what would have to be true of the hardware for sparsity to pay."
- **Runs on:** MPS/CPU. 2:4 semi-structured sparsity needs Ampere+ sparse tensor cores — **needs CUDA** (A100/L4 on Colab or runpod); locally you can still build the 2:4 mask and verify the pattern constraint.
- **Interview hooks:** What is 2:4 sparsity and why did hardware vendors pick that specific pattern? Why is depth pruning (dropping whole layers) surprisingly competitive for LLMs? Why has pruning lost to quantization in practice for LLM serving?
- **Common misconception:** "90% sparsity = 10× speedup." Without hardware or kernel support for the sparsity *pattern*, it is a 1× speedup and a quality loss.

### 6.24 — torch.compile and graph capture
- **Question it answers:** What does a compiler do to a model that eager PyTorch cannot, and what is a graph break?
- **Prereqs:** 6.4 (fusion is a bandwidth argument); Python basics.
- **Experiment:** Define a function with several elementwise ops chained (`x.silu() * y + z`, layernorm-ish). Time eager vs compiled on the largest tensor MPS will hold. Then add a data-dependent branch (`if x.sum() > 0:`) and rerun with `TORCH_LOGS=graph_breaks` to see the break. Count the kernels launched in each case conceptually: N elementwise ops = N HBM round-trips eagerly, 1 when fused.
- **Shapes to nail:** none new. Nail instead: for an elementwise chain, bytes moved eagerly ≈ `2 · N · numel · dtype_size`; fused ≈ `2 · numel · dtype_size`.
- **Predict-first prompt:** "Three chained elementwise ops on a 100M-element tensor. Compute the HBM traffic eager vs fused, and predict the speedup from bandwidth alone."
- **Runs on:** CPU with the inductor backend definitely; **MPS support for `torch.compile`/inductor has been landing incrementally** — smoke-test it on torch 2.10 first (`torch.compile(f)(x)` on a `device='mps'` tensor) and if it errors, use `backend="aot_eager"` on MPS, or run the fusion measurement on CPU where the argument is identical. Do not claim an MPS inductor speedup you did not measure.
- **Interview hooks:** What is the difference between Dynamo (capture), AOTAutograd (backward) and Inductor (codegen)? Why does a graph break cost more than just the un-fused region? What is CUDA-graph capture and why does it matter specifically for decode?
- **Common misconception:** "`torch.compile` speeds up matmuls." It mostly removes launch overhead and fuses memory-bound elementwise/reduction ops; the big matmuls already call cuBLAS.

### 6.25 — Benchmarking honestly
- **Question it answers:** What is the minimum protocol for a latency/throughput/memory number you would defend in a design review?
- **Prereqs:** 6.4, 6.5; all of 6D.
- **Experiment:** Write `common/bench.py`: warmup iterations discarded, explicit `torch.mps.synchronize()` (or `torch.cuda.synchronize()`), N≥20 timed reps, report median + p90 + stddev (never the mean of a bimodal sample), peak memory via `torch.mps.current_allocated_memory()`, and a fixed seed. Then use it to *redo* one earlier measurement without sync and show the wrong (asynchronous) number you would have reported.
- **Shapes to nail:** none. Nail definitions instead: TTFT (prefill latency), ITL/TPOT (per-output-token latency), throughput (tokens/s aggregated across concurrent requests) — and the fact that optimizing throughput usually *worsens* TTFT.
- **Predict-first prompt:** "You time a GPU op without synchronizing and get 0.05 ms. What did you actually measure, and would adding more work to the op change the number?"
- **Runs on:** MPS.
- **Interview hooks:** Why report p99 latency and not mean for a serving system? Why does batch size have to be reported with every throughput number? What is a fair comparison between two quantization schemes — same hardware, same batch, same context, same decode length, same seed?
- **Common misconception:** "Faster average = better." A serving SLO is a tail statistic; and a throughput win bought by batching can violate the latency SLO it was meant to serve.

---

# PHASE 7 — Post-training: SFT, PEFT, alignment, distributed

## 7A. The pipeline

### 7.1 — The map: pretraining → SFT → alignment
- **Question it answers:** What does each stage actually change about the model, and what can it not fix?
- **Prereqs:** pretraining objective (next-token prediction); sampling.
- **Experiment:** Load a base model and its instruct sibling (e.g. a ≤1B pair). Give both the identical raw prompt "Explain what a mutex is." with no chat template. Then give both the same prompt *with* the instruct model's chat template. Four outputs. Explain each one's behaviour in terms of what the training distribution contained.
- **Shapes to nail:** none new. Nail the data shapes per stage instead: pretraining = unlabeled text `[N_docs]`; SFT = (prompt, response) pairs; RM = (prompt, chosen, rejected) triples; RL = (prompt, sampled completions, scalar rewards).
- **Predict-first prompt:** "The base model is given a question with no chat template. Predict its output form — before running. Why is that the *correct* behaviour for its objective?"
- **Runs on:** MPS.
- **Interview hooks:** What capability is added by SFT versus merely *surfaced* by it? Why does alignment sometimes reduce benchmark scores ("alignment tax")? If your fine-tune fails, how do you decide whether the problem is data, stage, or base model?
- **Common misconception:** "SFT teaches the model new knowledge." It mostly teaches format and behaviour; new facts are learned poorly and hallucinated confidently — this drives the data decisions in 7.18–7.19.

### 7.2 — Instruction data and chat templates
- **Question it answers:** How does a list of message dicts become one token sequence, and what breaks when you get it wrong?
- **Prereqs:** tokenization; special tokens.
- **Experiment:** Take `messages = [{"role":"system",...},{"role":"user",...},{"role":"assistant",...}]`. Call `tokenizer.apply_chat_template(messages, tokenize=False)` for three different models and diff the raw strings. Then tokenize and print the token IDs around every role boundary. Deliberately break it: hand-concatenate "User: ... Assistant: ..." without the special tokens and compare the model's output.
- **Shapes to nail:** `input_ids: [1, T]`; the boundary indices of the assistant span (you will need them in 7.3). Note `add_generation_prompt=True` appends the assistant header and nothing else.
- **Predict-first prompt:** "Two models were trained with different chat templates. You apply model A's template to model B. Predict what fails, and whether it fails loudly or quietly."
- **Runs on:** CPU.
- **Interview hooks:** Why must special tokens be single tokens rather than parseable text? What is the security consequence of a template that lets user content forge a role boundary? Why does a tokenizer mismatch between training and serving produce subtle rather than obvious degradation?
- **Common misconception:** "The chat template is cosmetic." It *is* the interface the model was trained against; mismatches are the single most common cause of "my fine-tune got worse".

### 7.3 — SFT mechanics and prompt loss masking
- **Question it answers:** Which tokens contribute to the SFT loss, and what happens when you get that wrong?
- **Prereqs:** 7.2; cross-entropy; label shifting.
- **Experiment:** Hand-build one training example: tokenize a templated (prompt, response) pair, construct `labels` as a copy of `input_ids` with every prompt-position set to `-100`, and print `input_ids`, `labels` and the shift alignment side by side for a `T=24` example. Then train the same 200 steps twice — masked and unmasked — on 100 examples, and compare generations. Also verify the `-100` convention by checking `F.cross_entropy(..., ignore_index=-100)`.
- **Shapes to nail:** `input_ids:[B,T]`, `attention_mask:[B,T]`, `labels:[B,T]`. Loss compares `logits[:, :-1, :]` with `labels[:, 1:]` — write out one index by hand. Padding positions get `-100` too.
- **Predict-first prompt:** "If you do *not* mask the prompt, the model is trained to predict the user's question as well as the answer. Predict the concrete failure you will see at generation time."
- **Runs on:** MPS (a ≤1B model, LoRA or even full fine-tune of a 0.5B in BF16 fits M3 comfortably).
- **Interview hooks:** When is training on the prompt actually *correct*? Why does per-example loss normalization differ from per-batch-token normalization, and which does your trainer do? What does a multi-turn conversation's mask look like?
- **Common misconception:** "The library handles masking." Several do not by default, or do it only if you pass the right `dataset_text_field`/formatting function — verify by printing labels, every time.

### 7.4 — Packing, padding and collation
- **Question it answers:** How do you batch variable-length examples without wasting compute or leaking across examples?
- **Prereqs:** 7.3; attention masks.
- **Experiment:** Take 8 examples of lengths 7..61. Batch them three ways: (a) pad to max-in-batch, (b) sort-by-length bucketing, (c) pack into fixed `T=128` blocks with EOS separators. For each, report padding waste (%) and the number of blocks. Then demonstrate cross-contamination: in the packed batch *without* a block-diagonal mask, show that a position in example 2 can attend to example 1.
- **Shapes to nail:** padded `[8, 61]` with a `[8,61]` mask; packed `[N,128]`. Correct packing needs a block-diagonal attention mask (or `position_ids` reset + a FlashAttention varlen `cu_seqlens` layout) — `[B,1,T,T]` for the naive mask version.
- **Predict-first prompt:** "Lengths 7..61 padded to max: what fraction of your FLOPs is spent on pad tokens? Compute it before measuring."
- **Runs on:** MPS/CPU.
- **Interview hooks:** Why does naive packing still often work despite contamination, and when does it stop working? What is `cu_seqlens` and why does the varlen kernel need it? How does packing interact with loss normalization (7.3)?
- **Common misconception:** "Padding is free because of the attention mask." The mask fixes correctness, not cost — you still run the FLOPs.

### 7.5 — Catastrophic forgetting
- **Question it answers:** What does a narrow fine-tune destroy, and how would you have detected it?
- **Prereqs:** 7.3.
- **Experiment:** Before fine-tuning, record the model's output on a 20-item "capability canary" set (arithmetic, a code snippet, a factual question, a refusal case, a different language). Fine-tune hard on a narrow domain (300 steps, lr 2e-4, full fine-tune of a small model). Re-run the canaries. Then repeat with (a) a 10× lower LR, (b) LoRA instead (forward-reference 7.7), (c) 20% general-instruction data mixed in.
- **Shapes to nail:** none new. Nail the protocol: canary set is fixed, greedy, seeded, and recorded *before* training.
- **Predict-first prompt:** "Which canary do you expect to break first — arithmetic, the refusal, or the other language? Commit to an order and a reason."
- **Runs on:** MPS.
- **Interview hooks:** Why does LoRA forget less — is it the parameter count or the effective learning rate? What is replay/rehearsal mixing and what ratio is typical? How does forgetting relate to the alignment tax in 7.1?
- **Common misconception:** "Held-out loss on my fine-tune data tells me it went well." It measures only the new distribution; forgetting is invisible there by construction. This is the seed of 7.20.

## 7B. PEFT

### 7.6 — The memory math that motivates PEFT
- **Question it answers:** Exactly where does training memory go, and what fraction is the optimizer?
- **Prereqs:** 6.2; AdamW.
- **Experiment:** Paper first, then verify. For a 7B model in BF16 with AdamW: weights 14GB, gradients 14GB, FP32 master weights 28GB, Adam `m`+`v` in FP32 28GB ≈ 84GB before a single activation. Then measure the real thing at small scale: for a 0.5B model on MPS, print allocated memory after (a) load, (b) forward, (c) backward, (d) `optimizer.step()` — four numbers — and check the ratios against your prediction.
- **Shapes to nail:** every optimizer state tensor has the parameter's shape. Activation memory ≈ `B·T·d_model·L·(a small constant)` — derive the constant for your block by counting saved tensors.
- **Predict-first prompt:** "Predict the four memory numbers (in MB) for the 0.5B model before running, using the 16-bytes-per-parameter rule. Which step causes the largest single jump?"
- **Runs on:** MPS (use `torch.mps.current_allocated_memory()`; note it reports differently from CUDA's `max_memory_allocated`).
- **Interview hooks:** Why is SGD-with-momentum rarely used for LLMs despite halving optimizer state? Why does 8-bit Adam work at all? Which of the four buckets does ZeRO-1/2/3 shard, respectively (forward-reference to 7.23)?
- **Common misconception:** "Model size ≈ training memory." The rule of thumb is ~16 bytes/parameter for mixed-precision AdamW, i.e. ~12× the BF16 weight size, before activations.

### 7.7 — LoRA from scratch
- **Question it answers:** What is the low-rank hypothesis, and what exactly do `r`, `alpha` and the scaling do?
- **Prereqs:** 7.6; matmul; the notion of matrix rank (teach rank here if it has not appeared: rank as the number of independent directions, and `BA` having rank ≤ r).
- **Experiment:** Write `class LoRALinear(nn.Module)` yourself — frozen base `W:[out,in]`, `A:[r,in]` init `N(0, σ)`, `B:[out,r]` init **zeros**, forward = `base(x) + (alpha/r) · (x @ A.T @ B.T)`. On `d=8, r=2`: verify the output equals the base model exactly at step 0, then verify `ΔW = B@A` has rank ≤ 2 via `torch.linalg.matrix_rank`. Count trainable parameters vs total and compare with the 7.6 memory math.
- **Shapes to nail:** `x:[B,T,in] → xA^T:[B,T,r] → (·)B^T:[B,T,out]`. Params: `r·(in+out)` vs `in·out`. For `in=out=4096, r=8`: 65,536 vs 16,777,216.
- **Predict-first prompt:** "`B` is initialized to zeros and `A` randomly. What is the model's output at step 0, and what would go wrong if *both* were zero? Answer both before coding."
- **Runs on:** MPS.
- **Interview hooks:** Why does scaling by `alpha/r` let you change rank without retuning the learning rate — and why does rsLoRA argue `alpha/√r` is the right scaling instead? Why does LoRA cut optimizer memory by ~1000× but activation memory hardly at all? Is the low-rank hypothesis a claim about weights or about weight *updates*?
- **Common misconception:** "Higher rank is always better." Beyond a task-dependent point it buys nothing and starts to forget more; rank interacts with which modules you target (7.8) far more than with raw capacity.

### 7.8 — Target modules and merging adapters
- **Question it answers:** Which linear layers should get adapters, and how do you fold them back into the base weights for deployment?
- **Prereqs:** 7.7; the decoder block's linear layers.
- **Experiment:** Take a trained LoRA from 7.7's class and merge: `W_merged = W + (alpha/r)·B@A`. Assert the merged dense model's logits match the adapter model's to ~1e-4 on a fixed input (and explain the residual difference as float associativity, not a bug). Then ablate targets on a small task: `q,v` only vs all four attention projections vs attention+MLP — compare trainable params, final loss, and wall-clock.
- **Shapes to nail:** `B@A: [out, in]` — same shape as `W`, which is why merging is possible at all. Per-module param counts for `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`.
- **Predict-first prompt:** "The original LoRA paper targeted `q` and `v`. Predict what adding the MLP projections does to (a) parameter count, (b) quality, (c) inference latency after merging. One of those three does not change — which?"
- **Runs on:** MPS.
- **Interview hooks:** Why is merged LoRA zero-overhead at inference while unmerged is not? How do you serve 50 different adapters on one base model (and what is the batching problem that creates)? Why do people target the MLP for knowledge-ish tasks and attention for style-ish ones?
- **Common misconception:** "You can merge a LoRA trained on a quantized base back into the FP16 base losslessly." You cannot cleanly — the adapter learned to correct quantization error; this is exactly the QLoRA merge caveat in 7.9.

### 7.9 — QLoRA: NF4, double quantization, paged optimizers
- **Question it answers:** How do you fine-tune a 7B on one 24GB card, and which pieces of that are genuinely CUDA-only?
- **Prereqs:** 6.3, 6.10, 7.7, 7.8.
- **Experiment:** Two halves. (1) **Local, MPS/CPU:** implement NF4 yourself — the 16 quantile levels, blockwise absmax with block size 64, and double quantization (quantize the FP32 block scales to 8-bit with a second-level scale per 256 blocks; verify the ~0.37 bits/param saving by counting bytes). Quantize a real weight matrix, dequantize, measure error vs uniform INT4. (2) **Rented GPU:** an actual QLoRA fine-tune with `bitsandbytes`, whose 4-bit kernels are CUDA-only.
- **Shapes to nail:** `W:[4096,4096]` → 16.7M weights → 262,144 blocks of 64 → 262,144 FP32 absmax scales (1MB) → after double quantization, 262,144 INT8 + 1024 FP32. Do that arithmetic explicitly.
- **Predict-first prompt:** "Block size 64 with an FP32 scale per block: how many bits per weight does the *scale* alone cost? Compute it, then say why 0.37 bits/param is worth an entire engineering trick."
- **Runs on:** Part (1) **CPU/MPS**. Part (2) **needs CUDA** — `bitsandbytes` NF4 kernels and paged optimizers both depend on CUDA (paged optimizers specifically use NVIDIA unified memory to survive the memory spikes that gradient checkpointing creates). Cheap path: Colab free T4 (16GB) handles QLoRA on a 7B at short sequence length; an A10/L4 on runpod/modal is ~$0.3–0.6/hr if you need longer sequences. Budget one 2-hour session.
- **Interview hooks:** Why is the base model dequantized to BF16 for the matmul anyway — where is the memory actually saved? Why does QLoRA train more slowly per step than LoRA despite using less memory? Why is merging a QLoRA adapter into an FP16 base lossy, and what do people do instead?
- **Common misconception:** "QLoRA is LoRA but faster." It is LoRA but *smaller*; it is meaningfully slower per step because of on-the-fly dequantization.

### 7.10 — LoRA variants: DoRA, rsLoRA, LoRA+, PiSSA
- **Question it answers:** What specific deficiency of LoRA does each variant target?
- **Prereqs:** 7.7, 7.8.
- **Experiment:** Implement DoRA's decomposition only, small scale: split the weight into magnitude `m:[out]` and direction `V/||V||`, adapt the direction with LoRA and train `m` separately; check that it reduces to LoRA when `m` is frozen at `||W||`. Run one A/B on the 7.8 task at `r=4` where LoRA is known to underperform.
- **Shapes to nail:** `W:[out,in]`; column-wise norm `||W||_c : [1,in]` (per-column, matching DoRA's formulation); `m:[1,in]` trainable.
- **Predict-first prompt:** "DoRA separates magnitude from direction. Why would that help most at *low* rank specifically?"
- **Runs on:** MPS.
- **Interview hooks:** What did the DoRA authors observe about how full fine-tuning changes magnitude vs direction, compared to LoRA? What does LoRA+ change (different LRs for `A` and `B`) and why is that asymmetric? What does PiSSA initialize from, and what does that buy?
- **Common misconception:** "More variants = strictly better." Most variants buy a point or two at low rank and cost inference-time complexity; at `r≥16` on a decent dataset they mostly converge.

## 7C. Alignment

### 7.11 — Preference data and reward modelling
- **Question it answers:** How do you turn "A is better than B" into a differentiable scalar reward?
- **Prereqs:** 7.3; logistic regression / sigmoid; cross-entropy.
- **Experiment:** Attach a scalar head to a small model (`nn.Linear(d_model, 1)` on the last token's hidden state). Train on 200 synthetic preference pairs with the Bradley-Terry loss `-log σ(r_chosen - r_rejected)`. Confirm rewards are only meaningful *relatively* by adding a constant to all rewards and checking the loss is unchanged.
- **Shapes to nail:** `hidden:[B,T,d]` → last non-pad position → `[B,d]` → `[B,1]` → squeeze `[B]`. A pair batch is `[2B]` forward passes or a `[B,2,T]` reshaped batch.
- **Predict-first prompt:** "The Bradley-Terry loss depends only on `r_chosen - r_rejected`. What does that imply about the absolute scale of the learned reward, and why does that matter for the KL coefficient in 7.12?"
- **Runs on:** MPS.
- **Interview hooks:** Why is the reward taken from the last token's hidden state and not pooled? What is the distribution-shift problem — the RM is trained on SFT-policy samples and then used to score a drifting policy? How do you evaluate a reward model, given it has no ground truth?
- **Common misconception:** "The reward model learns what is good." It learns what *this* annotator population preferred on *this* prompt distribution, including length and formatting biases.

### 7.12 — RLHF with PPO, and why the KL penalty exists
- **Question it answers:** What are the four models in a PPO run, what does each do, and what is the KL term protecting?
- **Prereqs:** 7.11; policy-gradient basics (teach in one paragraph: `∇ E[R] = E[R ∇log π]`); advantage.
- **Experiment:** Toy bandit, no language model: 5 arms, a "policy" as a softmax over 5 logits, a hand-written reward function with a deliberately exploitable arm (huge reward, bad in reality). Run vanilla policy gradient — watch it collapse onto the exploit. Add `-β·KL(π ‖ π_ref)` and sweep `β ∈ {0, 0.01, 0.1, 1}`; plot the reward vs KL frontier.
- **Shapes to nail:** policy logits `[5]`; in the LM case, per-token `logprobs:[B,T]`, `values:[B,T]`, `advantages:[B,T]`, token-level KL `[B,T]`. Four models resident: policy, reference (frozen), reward, value.
- **Predict-first prompt:** "Reward is unbounded above; KL is a penalty. Predict what the optimal policy looks like for β=0 and for β→∞, and sketch the curve between them before you plot it."
- **Runs on:** CPU for the bandit — do this one on CPU, the point is the objective. A real PPO run on a ≤1B model is technically MPS-possible but painful (four models resident + sampling); **use a rented GPU** for any real run (A10/L4, ~$0.4/hr, a few hours).
- **Interview hooks:** Why is the KL reference the SFT model and not the pretrained base? Why is PPO's ratio clipping needed given you already have a KL term — what is each one preventing? What is the value model for, and how much memory does it cost?
- **Common misconception:** "The KL penalty prevents the model from getting worse." It bounds *drift* from the reference, which bounds both damage and improvement; it is a leash, not a quality metric.

### 7.13 — DPO, derived and implemented
- **Question it answers:** How do you get RLHF's optimum without sampling, a reward model, or a value model?
- **Prereqs:** 7.11, 7.12.
- **Experiment:** Derive on paper: the KL-constrained optimum is `π*(y|x) ∝ π_ref(y|x)·exp(r(x,y)/β)`; invert to get `r(x,y) = β log(π(y|x)/π_ref(y|x)) + β log Z(x)`; substitute into Bradley-Terry — `Z(x)` cancels because it is shared between chosen and rejected. Then implement the loss in ~10 lines and train on the same 200 pairs from 7.11: `-log σ(β·[(logπ_c - logπ_ref_c) - (logπ_r - logπ_ref_r)])`. Log the implicit reward margin and the accuracy (fraction of pairs where chosen > rejected).
- **Shapes to nail:** for each of chosen/rejected: `logits:[B,T,V]` → gather at labels → `[B,T]` → mask prompt positions → **sum** (not mean) over the response → `[B]`. Four sequence-logprob numbers per pair. Reference logprobs can be precomputed once.
- **Predict-first prompt:** "Where exactly does the partition function `Z(x)` cancel, and what property of the preference pair makes that cancellation legal? Say it before you see the derivation's last line."
- **Runs on:** MPS — a real DPO run on a ≤1B model with LoRA fits on M3. This is the phase's best local alignment experiment; do it properly.
- **Interview hooks:** Why sum rather than mean the token logprobs, and what bias does the alternative introduce? Why does DPO tend to *decrease* the probability of chosen responses too, and what do IPO/KTO/SimPO change? DPO is off-policy — when does that matter, and what is "online DPO"?
- **Common misconception:** "DPO has no reward model." It has an *implicit* one — the log-ratio to the reference — which is why the reference model is still resident and still matters.

### 7.14 — GRPO and dropping the value model
- **Question it answers:** What does group-relative advantage replace, and why does that change what you can afford to train?
- **Prereqs:** 7.12, 7.13.
- **Experiment:** Extend the 7.12 bandit: sample a group of `G=8` actions per state, compute advantage as `(r_i - mean(r)) / std(r)` across the group, and do the policy-gradient update with no value network. Compare gradient variance and convergence against the PPO-with-critic version at equal sample budget. Then sweep `G ∈ {2, 8, 64}` and observe the variance/compute tradeoff.
- **Shapes to nail:** group rewards `[B, G]`; advantages `[B, G]` broadcast to every token of each completion `[B, G, T]`. Memory saved = one full model (the critic).
- **Predict-first prompt:** "PPO's critic gives a *per-state* baseline. GRPO's group mean gives a per-prompt baseline shared by all `G` samples. Predict which has lower-variance advantages at small `G`, and what happens as `G` grows."
- **Runs on:** CPU for the bandit. Real GRPO on an LLM requires heavy sampling — **needs a rented GPU** (it is sampling-bound; a single A10 with a ≤1.5B model and a verifiable reward is a real, affordable experiment at ~$0.4/hr).
- **Interview hooks:** Why does GRPO fit reasoning tasks specifically (hint: what kind of reward is available)? What does dropping the critic cost you in credit assignment across tokens? Why does GRPO need a large `G` and what does that do to the inference cost of a training step?
- **Common misconception:** "GRPO is just PPO without the critic." The advantage estimator is structurally different — normalized within a group of completions for the *same* prompt, which is only valid because they share a prompt.

### 7.15 — Reward hacking
- **Question it answers:** What does optimizing a proxy reward actually produce, and how would you catch it?
- **Prereqs:** 7.11–7.14.
- **Experiment:** Make it happen on purpose. In the 7.11 setup, use a reward model you know is length-biased (train it on pairs where longer is labelled better). Run DPO against it and plot mean response length and reward vs steps. Then add a length-normalized reward and rerun. Document the exact step at which reward rises while quality falls.
- **Shapes to nail:** none new. Nail the diagnostic instead: always plot reward *and* KL-to-reference *and* a proxy-independent quality metric on the same x-axis.
- **Predict-first prompt:** "Your reward curve is going up and to the right and KL is climbing steadily. Name three things that could be happening, only one of which you want."
- **Runs on:** MPS.
- **Interview hooks:** What is Goodhart's law in the RLHF setting concretely? Why is KL-to-reference the standard early-warning signal, and how is it defeated? How do you build an eval that the optimizer cannot see (and why does using your eval set as a reward destroy it)?
- **Common misconception:** "A better reward model fixes reward hacking." It moves the exploit; any fixed proxy optimized hard enough decouples from the thing it proxies.

### 7.16 — Reasoning models and test-time compute
- **Question it answers:** What changed when RL moved from human preferences to verifiable rewards?
- **Prereqs:** 7.14, 7.15.
- **Experiment:** Two parts. (1) Test-time compute without training: on a small model and 30 grade-school math problems, measure accuracy for greedy, then self-consistency with `k ∈ {1,4,16}` samples and majority vote. Plot accuracy vs total tokens generated — this is the scaling curve that matters. (2) Measure output-length distribution for a reasoning-tuned small model vs its non-reasoning sibling on the same problems.
- **Shapes to nail:** none new. Nail the cost model: `k` samples × `T_thinking` tokens, all of which must be generated at decode-time cost from 6.5 — test-time compute is bought in KV bandwidth.
- **Predict-first prompt:** "Majority-vote-over-k at k=16 costs 16× the tokens. For a task where greedy is 40% correct, predict roughly where the accuracy curve saturates and why it cannot reach 100%."
- **Runs on:** MPS.
- **Interview hooks:** Why does RL on *verifiable* rewards sidestep the reward-hacking problem from 7.15 — and where does it reintroduce it? What is the serving consequence of a model that generates 5,000 thinking tokens before answering? Why did long-CoT RL change the compute allocation between pretraining and post-training?
- **Common misconception:** "Reasoning models are just prompted to think step by step." The behaviour is trained in with RL against checkable answers; prompting a base model does not reproduce the length or self-correction behaviour.

## 7D. Data and evaluation (where the quality actually comes from)

### 7.17 — Model merging: task arithmetic, TIES, SLERP
- **Question it answers:** Can you combine two fine-tunes without retraining, and what does that say about where capabilities live?
- **Prereqs:** 7.8 (merging a LoRA is the special case).
- **Experiment:** Train two small LoRAs on two different toy tasks from the same base. Compute task vectors `τ = θ_ft - θ_base`. Merge three ways: (a) simple average `θ_base + 0.5τ₁ + 0.5τ₂`; (b) TIES — trim to the top-20% magnitude entries per vector, elect a sign per parameter by summed magnitude, average only the sign-agreeing entries; (c) SLERP between the two on the unit sphere. Evaluate each merged model on both tasks and on a control task.
- **Shapes to nail:** `τ` has exactly the model's parameter shapes. TIES's sign election is an elementwise `[numel]` operation. SLERP needs the angle `Ω = arccos(cos_sim(θ₁,θ₂))` computed per-tensor, not globally.
- **Predict-first prompt:** "Naive averaging of two task vectors degrades both tasks. TIES claims the cause is sign conflicts. Predict roughly what fraction of parameters have opposite signs between two unrelated fine-tunes."
- **Runs on:** MPS/CPU.
- **Interview hooks:** Why does merging work at all — what does it imply about the loss landscape near a pretrained initialization? When does merging beat multi-task fine-tuning, and when does it lose badly? Can you merge models with different architectures or tokenizers, and why not?
- **Common misconception:** "Merging is an ensemble." It averages *parameters*, not outputs — one model at inference, and no ensemble variance reduction.

### 7.18 — Synthetic data generation
- **Question it answers:** How do you produce SFT/preference data at scale without collapsing onto the generator's biases?
- **Prereqs:** 7.2, 7.3, 7.11.
- **Experiment:** Build a 200-example synthetic SFT set end-to-end: seed with 20 human-written prompts, expand with an evolution step (make each prompt harder/more specific along one axis), generate responses, then filter — by a verifier where one exists (code that must run, math that must check), by a rubric-scored judge where it does not, and by rejection sampling (generate `n=4`, keep the best). Report the yield at each stage and inspect 10 rejected items by hand.
- **Shapes to nail:** none. Nail the pipeline's accounting instead: prompts in → completions generated → completions surviving each filter → final examples, with the cost per surviving example.
- **Predict-first prompt:** "You generate training data with model X and fine-tune model Y on it. What is the ceiling on Y's capability, and name a mechanism by which Y could nonetheless exceed X on a specific task."
- **Runs on:** MPS for a small local generator; an API or rented GPU for a strong one. The *pipeline design* is the lesson; the generator's size is not.
- **Interview hooks:** What is model collapse and under what conditions does iterating on synthetic data actually degrade a model? Why does rejection sampling with a verifier behave differently from judge-based filtering? What licence/provenance issues does distilling from a commercial API create?
- **Common misconception:** "Synthetic data is a cheap substitute for real data." It is a cheap way to *amplify* a small amount of high-quality real data and to cover verifiable domains; unfiltered it amplifies the generator's failure modes too.

### 7.19 — Curation, deduplication and decontamination
- **Question it answers:** How do you find near-duplicates and benchmark leakage in a corpus, at a scale where you cannot compare every pair?
- **Prereqs:** hashing. (Your distributed-systems background carries this lesson — MinHash/LSH is the same machinery as a sketch-based join.)
- **Experiment:** Implement MinHash + LSH banding yourself on 2,000 short documents with planted near-duplicates: shingle into 5-grams, `k=128` hash permutations, band into `b=16` bands of `r=8`, bucket by band, and compare pairs only within buckets. Plot the S-curve `1-(1-s^r)^b` and verify your recall of the planted duplicates against it. Then decontaminate: take 50 benchmark items and find n-gram (13-gram) overlaps against the corpus.
- **Shapes to nail:** signature matrix `[n_docs, 128]`; `b·r = 128`; candidate pairs compared vs `C(2000,2)` = 1,999,000 — report the reduction factor you achieved.
- **Predict-first prompt:** "With `b=16, r=8`, at what Jaccard similarity does the LSH detection probability cross 50%? Compute it from the S-curve formula before plotting."
- **Runs on:** CPU.
- **Interview hooks:** Why does exact dedup miss most of the duplication that matters? How would you decontaminate against a benchmark you are not allowed to see? What is the difference between removing duplicates and *downweighting* them, and which is right for pretraining vs SFT?
- **Common misconception:** "Dedup is a data-cleaning chore." It is one of the highest-leverage quality interventions known, and contamination is the single most common reason a reported benchmark number is meaningless.

### 7.20 — Evaluating a fine-tune
- **Question it answers:** What is the minimum eval suite that would have caught the failures from 7.5, 7.15 and 7.19?
- **Prereqs:** 7.5, 7.15, 7.19, 6.15.
- **Experiment:** Assemble a five-part harness for your own fine-tune and run it: (1) held-out loss on the target distribution; (2) the capability canary set from 7.5; (3) a task-specific metric with a fixed rubric; (4) pairwise win-rate against the base model using an LLM judge, with **position-swapped duplicate comparisons** to measure and correct position bias; (5) a contamination check of the eval set against the training set using 7.19's machinery. Report all five plus run-to-run variance.
- **Shapes to nail:** none. Nail statistics instead: with `n` pairwise comparisons, the standard error on win rate is `√(p(1-p)/n)` — compute how many comparisons you need to distinguish 55% from 50%.
- **Predict-first prompt:** "Your fine-tune beats the base model 60% of the time on 20 judged comparisons. Is that significant? Compute the standard error before answering."
- **Runs on:** MPS (a small local judge is fine for the mechanics; note that judge *quality* is a separate axis from judge *protocol*).
- **Interview hooks:** What biases does an LLM judge have (position, length, self-preference, formatting) and how do you control each? Why is held-out loss on the fine-tune distribution nearly useless as a quality signal? How do you build a regression suite that survives model upgrades?
- **Common misconception:** "Lower eval loss means a better model." It means a better fit to that distribution; it says nothing about forgetting, about the alignment tax, or about behaviour outside the training distribution.

## 7E. Distributed training

### 7.21 — Collectives and the communication model
- **Question it answers:** What primitives exist, what do they cost in bytes moved, and which one is every parallelism strategy built from?
- **Prereqs:** none DL-specific — this is your home turf; connect it explicitly to the fan-out/fan-in and quorum patterns you already know.
- **Experiment:** Run it for real on the Mac: `torch.distributed` with the `gloo` backend and 4 processes on CPU (`torchrun --nproc_per_node=4`). Implement and time `all_reduce`, `all_gather`, `reduce_scatter`, `broadcast`, `barrier` on a 10M-element tensor. Then derive and verify: ring all-reduce moves `2(N-1)/N · S` bytes per rank, and `all_reduce = reduce_scatter + all_gather` — check that the timings are consistent with that identity.
- **Shapes to nail:** for a tensor of `S` bytes on `N` ranks — all-reduce: output `S` per rank, `2(N-1)/N·S` moved; all-gather: input `S/N`, output `S`; reduce-scatter: input `S`, output `S/N`.
- **Predict-first prompt:** "All-reduce on `N` ranks. Predict the bytes-per-rank as `N→∞` — does it grow without bound, or converge? Answer before you look at the formula."
- **Runs on:** **CPU with the `gloo` backend, multi-process on your M3** — this is fully local and genuinely teaches the collectives. NCCL and real interconnect bandwidth need multi-GPU.
- **Interview hooks:** Why is ring all-reduce bandwidth-optimal but latency-poor, and when does a tree beat it? What is the difference between intra-node NVLink and inter-node InfiniBand bandwidth, and where does that discontinuity show up in a parallelism plan? What happens to a collective when one rank is slow, and what is the failure domain of a synchronous training job?
- **Common misconception:** "All-reduce sends everything to everyone." Ring all-reduce moves ~2S bytes per rank regardless of `N`, which is exactly why data parallelism scales.

### 7.22 — DDP and gradient all-reduce
- **Question it answers:** What does DDP synchronize, when, and how does it hide the cost?
- **Prereqs:** 7.21; backward pass.
- **Experiment:** Wrap the tiny model in `DistributedDataParallel` with 4 gloo CPU processes. Verify parameters are identical across ranks after a step. Then break it deliberately: skip the all-reduce on one rank (use `no_sync()`) and watch the ranks diverge. Print the gradient bucket sizes and reason about why the *last* layer's gradients are ready first.
- **Shapes to nail:** every rank holds full `[params]`, full optimizer state, and a `1/N` slice of the batch. Communication per step = gradient bytes, independent of batch size. Effective batch = `per_rank_batch · N · grad_accum_steps`.
- **Predict-first prompt:** "Backward computes gradients from the last layer to the first. DDP buckets gradients and all-reduces each bucket as it becomes ready. Predict how much of the communication can be hidden — and what determines the part that cannot."
- **Runs on:** **CPU/gloo, 4 processes, locally.**
- **Interview hooks:** Why must the LR schedule account for the effective batch size when you change world size? What does `find_unused_parameters=True` cost and why does it exist? Why is DDP useless for a model that does not fit on one device — and what exactly is the limiting resource (7.6's four buckets)?
- **Common misconception:** "More GPUs = faster training, linearly." Communication is fixed per step while compute per step shrinks, so scaling efficiency falls; and larger effective batches need LR and warmup changes to converge the same.

### 7.23 — ZeRO 1/2/3 and FSDP
- **Question it answers:** Which of the memory buckets from 7.6 does each ZeRO stage shard, and what extra communication does each buy that with?
- **Prereqs:** 7.6, 7.21, 7.22.
- **Experiment:** Table first — for a 7B model on `N=8`, compute per-rank memory under DDP, ZeRO-1 (optimizer state sharded), ZeRO-2 (+gradients), ZeRO-3 (+parameters). Then run FSDP (`fully_shard`/`FullyShardedDataParallel`) with 4 gloo CPU processes on the tiny model; print per-rank parameter storage to confirm each rank holds `1/N`, and observe the all-gather before each layer's forward.
- **Shapes to nail:** ZeRO-3 per-rank params = `[numel/N]` flat shards; the all-gather materializes a full layer `[numel_layer]` transiently, which sets the *peak* memory, not the average. Comms per step: DDP = 1 all-reduce of grads; ZeRO-3 ≈ all-gather params (fwd) + all-gather params (bwd) + reduce-scatter grads ≈ 1.5× DDP's volume.
- **Predict-first prompt:** "ZeRO-3 shards parameters, but the forward pass needs whole layers. Where does the full layer come from, when is it freed, and what does that do to peak memory relative to average memory?"
- **Runs on:** **CPU/gloo locally** for the mechanics (FSDP works on gloo for small models — smoke-test on your torch build first). Real throughput numbers need multi-GPU; take the concepts locally and the numbers from published scaling tables.
- **Interview hooks:** When is ZeRO-3 slower than ZeRO-2 despite fitting more, and what does prefetching recover? What is CPU offloading and what does it do to your step time? Why do people run ZeRO-3 *within* a node and data-parallel across nodes?
- **Common misconception:** "FSDP and ZeRO-3 are different techniques." They are the same idea, differently implemented (PyTorch-native vs DeepSpeed).

### 7.24 — Tensor parallelism
- **Question it answers:** How do you split a single matmul across devices, and why is the collective placement not negotiable?
- **Prereqs:** 7.21; the decoder block's exact linear layers.
- **Experiment:** Do it on paper then in numpy on one machine. For the MLP `Y = down(act(up(X)))`: shard `up` **column-wise** (`W_up:[d,4d] → [d,4d/N]`) so each rank computes an independent slice of the activation with no communication; shard `down` **row-wise** (`W_down:[4d,d] → [4d/N,d]`) so each rank produces a partial sum, then one all-reduce gives the result. Verify numerically for `N=2, d=8` that the sharded result equals the dense one. Do the same for attention, sharding by heads.
- **Shapes to nail:** column-parallel: `X:[B,T,d] @ W:[d,4d/N] → [B,T,4d/N]`. Row-parallel: `[B,T,4d/N] @ [4d/N,d] → [B,T,d]` *partial*, then all-reduce. Two all-reduces per transformer block in the forward (one for attention, one for MLP), two more in the backward.
- **Predict-first prompt:** "Why column-parallel first and row-parallel second? Work out what communication the *other* ordering would require, and how much."
- **Runs on:** **CPU/numpy locally** for the math (this is the honest local version and it teaches the whole idea). Real TP **needs multi-GPU with fast interconnect** — TP all-reduces per block are why it is confined inside a node.
- **Interview hooks:** Why is TP restricted to a single node in practice, and what bandwidth number justifies that? Where does the non-linearity force a communication boundary? How does TP interact with GQA when `n_kv_heads < N`?
- **Common misconception:** "TP splits the model so each GPU does less work." Each GPU does `1/N` of the FLOPs but participates in a synchronous all-reduce twice per layer — TP buys memory and latency, and spends bandwidth to do it.

### 7.25 — Pipeline parallelism and the bubble
- **Question it answers:** Where does the idle time come from, and what is the exact formula for it?
- **Prereqs:** 7.21, 7.22.
- **Experiment:** Simulate — no GPUs needed. Write a scheduler simulator: `P` stages, `M` microbatches, unit time per stage-microbatch. Produce an ASCII Gantt chart for (a) naive sequential, (b) GPipe all-forward-then-all-backward, (c) 1F1B interleaved. Compute bubble fraction `(P-1)/(M+P-1)` and confirm your simulation matches it. Then compute peak activation memory per stage for GPipe vs 1F1B and see why 1F1B wins on memory too.
- **Shapes to nail:** the inter-stage transfer is one activation tensor `[B_micro, T, d_model]` per microbatch boundary — compare that byte count against TP's all-reduce of the same tensor, since that comparison is the whole reason PP goes across nodes and TP does not.
- **Predict-first prompt:** "P=4 stages. How many microbatches do you need for the bubble to be under 10%? Derive it from the formula before simulating."
- **Runs on:** **CPU** (a simulator, which is the right tool — it makes the schedule visible in a way a real run never does).
- **Interview hooks:** Why does PP tolerate slow interconnects where TP cannot? What is the memory asymmetry across pipeline stages and how do interleaved schedules address it? How do you place layers when stages have unequal cost (embedding and LM head)?
- **Common misconception:** "Pipeline parallelism is model parallelism." Both split the model, but PP splits by *layer* with point-to-point transfers, while TP splits *within* a layer with collectives — completely different communication profiles.

### 7.26 — Sequence and context parallelism
- **Question it answers:** When the *activations* for one sequence do not fit, what do you shard?
- **Prereqs:** 7.24; 6.6–6.7 (ring attention is the online-softmax recurrence, distributed).
- **Experiment:** Implement ring attention's core in numpy for `N=2` "ranks": each rank owns half the sequence's K/V. Rotate K/V blocks around the ring, and at each step update `(m, l, o)` using **exactly the 6.6 recurrence**. Verify the result matches full attention. This is the payoff for having built 6.6 by hand.
- **Shapes to nail:** rank `i` holds `Q:[B,H,T/N,d]`, `K,V:[B,H,T/N,d]` and the running `(m,l):[B,H,T/N]`, `o:[B,H,T/N,d]`. Communication = `N-1` rounds of `[B,H,T/N,d]×2` bytes, overlappable with compute.
- **Predict-first prompt:** "Ring attention passes K/V around `N` ranks. Does it recompute anything, and is the result exact or approximate? Justify from 6.6."
- **Runs on:** **CPU/numpy locally** — the whole idea is expressible in one file.
- **Interview hooks:** Why does causal masking make the ring load-imbalanced, and how is that fixed (zigzag/striped assignment)? What is "sequence parallelism" in the Megatron sense — sharding the LayerNorm/dropout regions that TP leaves replicated — and how does it differ from context parallelism? Which axis do you shard first at 1M context?
- **Common misconception:** "Sequence parallelism and context parallelism are the same thing." Megatron-SP shards the TP-replicated activation regions; context/ring parallelism shards the attention computation itself. Different problems.

### 7.27 — 3D parallelism, activation checkpointing, gradient accumulation
- **Question it answers:** How do the axes compose, and which two knobs trade memory for compute without any communication at all?
- **Prereqs:** 7.22–7.26; 6.8 (recomputation).
- **Experiment:** Two halves. (1) Local and real: add `torch.utils.checkpoint.checkpoint` to your from-scratch block; measure peak memory and step time with and without, and count how many activations are saved in each case. Add gradient accumulation with `accum=4` and verify the loss curve matches `batch=4·b` (watch the loss-scaling-by-`1/accum` detail — get it wrong first, then fix it). (2) Paper: lay out a 3D mapping for 64 GPUs across 8 nodes — TP=8 within a node, PP=4, DP=2 — and justify every placement by the bandwidth available on that axis.
- **Shapes to nail:** checkpointing saves only the block's *input* `[B,T,d]` instead of every intermediate; recompute cost ≈ one extra forward ≈ +33% step time for ~√L memory with optimal placement. World size = `TP · PP · DP` — check the arithmetic.
- **Predict-first prompt:** "You have 64 GPUs in 8 nodes of 8, NVLink inside a node and slower Ethernet between. Assign TP, PP and DP to axes *before* reading the answer, and say what each choice costs if you get it backwards."
- **Runs on:** **MPS/CPU** for part (1), which is the part with a measurement. Part (2) is a design exercise — treat it as a `/sysdesign` drill.
- **Interview hooks:** Why does TP go innermost and DP outermost? What is the failure domain of a 3D-parallel job — which single node loss kills the run, and what does checkpoint/restart cost at that scale? Why is gradient accumulation not equivalent to a larger batch under BatchNorm (and why is that irrelevant for transformers)?
- **Common misconception:** "Gradient accumulation is the same as a bigger batch." For LayerNorm transformers it is numerically near-equivalent, but only if you scale the loss correctly and only if there is no cross-example normalization — and it does nothing for the memory that optimizer state occupies.

### 7.28 — The memory-budget exercise
- **Question it answers:** Given a model and a GPU, what fits — and what must you shard?
- **Prereqs:** all of 7E, plus 7.6.
- **Experiment:** Work three budgets on paper, then verify the scaled-down version locally. (a) Full fine-tune a 7B on 1×A100-80GB, `B=4, T=2048` — show it does not fit and say exactly which bucket overflows. (b) Same with ZeRO-3 on 8×A100 — show what fits and give the per-step communication volume. (c) QLoRA on 1×T4-16GB — show what fits and what sequence length is the binding constraint. Then verify your model of activation memory empirically on M3 by fitting the measured peak against `B·T·d·L` for three different `(B,T)` settings.
- **Shapes to nail:** the full budget equation — `params·2 + grads·2 + master·4 + adam·8 + activations(B,T,d,L) + KV(if generating) + fragmentation`. Produce it from memory, in bytes, on demand.
- **Predict-first prompt:** "7B full fine-tune, AdamW mixed precision. Before any activations: how many GB? Now, how many A100-80GBs is that, and is the answer 1?"
- **Runs on:** MPS/CPU for the empirical activation fit; the budgets are paper.
- **Interview hooks:** Your job OOMs at step 900 of 1000 with everything constant — what are the three most likely causes? Why does peak memory exceed the sum of the buckets, and what does allocator fragmentation do about it? Given a fixed budget, where do you spend the marginal dollar — bigger model, longer context, or more data?
- **Common misconception:** "Memory scales with model size." Activation memory scales with `B·T·d·L` and can dominate at long sequence length; the model may be the smaller half of your budget.

### 7.29 — The cheap GPU path
- **Question it answers:** For the CUDA-only lessons above, what is the actual minimum-cost way to run them without wasting rental time?
- **Prereqs:** the CUDA-marked lessons you have queued up (6.3 FP8, 6.9 FA-3, 6.12 GPTQ/AWQ, 6.23 2:4, 7.9 QLoRA, 7.12 PPO, 7.14 GRPO).
- **Experiment:** Build the discipline before spending money. Write one `run_remote.sh` that: syncs the repo, installs pinned deps, runs a chosen script headless, writes results as JSON + logs back to the repo, and shuts down. Test it end-to-end on a Colab T4 (free) with a trivial job *before* you rent anything. Then batch every CUDA-marked lesson into one queued session.
- **Shapes to nail:** none. Nail the cost table instead: Colab T4 16GB (free, preemptible, fine for QLoRA-7B at short `T`); L4 24GB ≈ $0.3–0.5/hr; A10 24GB ≈ $0.4/hr; A100-40GB ≈ $1.2–2/hr; H100 ≈ $2–4/hr (only needed for FP8 and FA-3). Verify current prices — they move.
- **Predict-first prompt:** "You have $20. Which of your CUDA-marked experiments do you run, in what order, and which do you drop? Justify by what each one teaches that the local version cannot."
- **Runs on:** local script authoring; the runs themselves **need CUDA** by construction.
- **Interview hooks:** How do you make a training job survive a spot preemption? What do you checkpoint, how often, and what is the restart cost? Why is the dominant cost of GPU work usually idle rented time rather than compute?
- **Common misconception:** "I'll figure out the environment once I'm on the GPU." Interactive debugging on a rented GPU is the most expensive way to write code; every CUDA-marked experiment should be debugged to completion on CPU/MPS with tiny shapes first.

---

## Milestone

**One artifact: `projects/quantize-and-align/` — a small model taken end-to-end, with numbers you can defend.**

Pick one base model ≤1.5B. Ship a repo directory containing:

1. **`sft/`** — a LoRA SFT you implemented with your own `LoRALinear` (not `peft`), on your own curated dataset, with prompt-loss masking verified by a printed label dump. Include the MinHash dedup + decontamination report for the dataset (7.19).
2. **`dpo/`** — a DPO run on top of it using your own loss implementation, with the reward-margin and KL curves, run locally on MPS.
3. **`quant/`** — the merged model exported to GGUF at two quantization levels, with your own NF4 implementation checked against `bitsandbytes` output for one tensor, and the k-quant bit-accounting worked out.
4. **`bench/`** — `common/bench.py` used to produce a table of: tokens/sec, TTFT, peak memory and KV-cache bytes at 3 context lengths for base / merged / Q8_0 / Q4_K_M, with p50/p90 and stddev over ≥20 reps, plus the roofline prediction for each row next to the measurement.
5. **`eval/`** — the five-part harness from 7.20 run on every variant: held-out loss, capability canaries, task metric, position-swapped judge win-rate with standard errors, contamination check.
6. **`WRITEUP.md`** — ≤1500 words. What you predicted, what you measured, where the prediction was wrong and why. One section must be "what I would need a real GPU for, and what it would cost" (7.29). Include the memory-budget worksheet from 7.28 for scaling this to 7B.

The phase is done when every number in `bench/` and `eval/` was produced by your own harness, you can explain any row in either table without looking it up, and the writeup is honest about the gaps.
