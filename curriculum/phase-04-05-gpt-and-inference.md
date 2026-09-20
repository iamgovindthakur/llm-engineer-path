# Phase 4 & 5 — Build a GPT, then serve it

Phase 4 turns the Phase 3 transformer block into a trainable language model and
trains it. Phase 5 is the serving stack: everything that happens between
"weights exist" and "a p99 latency SLO is met".

Hardware note up front: everything in Phase 4 runs on M3/MPS at toy scale. The
only Phase 4 item that wants a real GPU is the *final* milestone run if you want
a GPT-2-124M-class result in hours rather than days. In Phase 5, roughly two
thirds runs on MPS/CPU as a toy; the explicitly-marked lessons need CUDA because
the artifact being studied (vLLM, FlashAttention-backed paged kernels, NCCL/NIXL
KV transfer) has no MPS build at all.

---

# PHASE 4 — Build a GPT from scratch and train it

## What you can claim after this phase

- I implemented a decoder-only transformer end to end — embeddings, blocks, LM
  head, weight tying — and trained it to convergence without copying a training
  loop.
- I can explain exactly how the next-token objective is constructed at the
  tensor level, and I can derive what the loss should be at step 0 before
  running anything.
- I can debug a training run: I know what a diverging loss, a flat loss, a
  gradient-norm spike and a silently-broken causal mask each look like.
- I can convert between nats/token, perplexity and bits/byte, and I can say
  whether a given number is good for a given model and dataset.
- I can size a training run from first principles: given a parameter count and a
  token budget, estimate FLOPs, GPU-hours and dollars, and justify the
  parameter/token split with Chinchilla rather than vibes.
- I can open an unfamiliar modelling file (nanoGPT, a HuggingFace
  `modeling_*.py`) and map every line to a component I have implemented myself.

---

## 4A — Assembling the model

### 4.1 — From one block to the decoder-only stack
- **Question it answers:** What is a "GPT" other than N copies of the block I already built?
- **Prereqs:** Phase 3 transformer block (attention + MLP + residual + norm), `nn.Embedding`, positional embeddings.
- **Experiment:** `GPT` module: token emb + learned position emb -> `nn.ModuleList` of 2 blocks -> final LayerNorm. `vocab=32, block_size=8, d_model=16, n_heads=2, n_layers=2`. Forward a batch of IDs and print the shape after every stage.
- **Shapes to nail:** idx `(B,T)` -> tok_emb `(B,T,16)`; pos ids `(T,)` -> pos_emb `(T,16)` broadcast to `(B,T,16)`; every block preserves `(B,T,16)`; final norm `(B,T,16)`.
- **Predict-first prompt:** The position embedding is `(T, d_model)` and the token embedding is `(B, T, d_model)`. Which broadcasting rule makes `tok + pos` legal, and what would break if you stored positions as `(B, T, d_model)` instead?
- **Runs on:** MPS
- **Interview hooks:** Why is there a final LayerNorm *after* the last block in GPT-2 but not in the original 2017 encoder? Why does a pre-norm stack train more stably at depth than post-norm? What exactly does the residual stream carry, and why do people call it a "read/write bus"?
- **Common misconception:** That depth means "the layers each do a different named job". The layers all read from and write to the same residual stream; there is no per-layer semantic contract.

### 4.2 — The language-modelling head and weight tying
- **Question it answers:** How do `d_model`-dimensional vectors become a distribution over the vocabulary, and why do many models reuse the embedding matrix to do it?
- **Prereqs:** 4.1, matmul shapes, `nn.Linear` weight layout `(out_features, in_features)`.
- **Experiment:** Add `lm_head = nn.Linear(d_model, vocab, bias=False)`. Print `lm_head.weight.shape` and `wte.weight.shape`. Then tie: `self.lm_head.weight = self.wte.weight`. Verify `id()` equality and that `sum(p.numel() for p in model.parameters())` drops by `vocab*d_model`.
- **Shapes to nail:** `(B,T,d_model) @ (d_model, V) -> (B,T,V)`; `lm_head.weight` is `(V, d_model)`, identical in shape to `wte.weight`, which is why tying is even possible.
- **Predict-first prompt:** With `vocab=50257, d_model=768`, how many parameters does the LM head hold, and what fraction of GPT-2-124M is that? Predict before counting.
- **Runs on:** MPS
- **Interview hooks:** Why does tying work at all — what does it assume about input and output embedding spaces? When does tying hurt? For a 1B model with a 128k vocab, what fraction of params and of the backward pass is the head, and what does that imply for tensor-parallel sharding of the vocab dimension?
- **Common misconception:** That tying is a parameter-saving hack only. It also acts as a regulariser and changes gradient flow into the embedding table — the table now receives gradient from every output position, not just from the tokens that appear in the input.

### 4.3 — The shift-by-one objective: constructing labels exactly
- **Question it answers:** What precisely is the model predicting, and how do `x` and `y` differ by one index?
- **Prereqs:** 4.1, causal masking from Phase 3, Python slicing.
- **Experiment:** From a 1-D token stream `data` of length 100 and `block_size=8`, build `x = data[i:i+8]`, `y = data[i+1:i+9]`. Print both side by side as a two-row table for one sample. Then assert `torch.equal(x[1:], y[:-1])`.
- **Shapes to nail:** `x (B,T)`, `y (B,T)`, logits `(B,T,V)`. Position `t` of the logits predicts `y[:, t]`, i.e. `x[:, t+1]`. `T` predictions come out of one forward pass, not one.
- **Predict-first prompt:** If the causal mask were removed, what would the training loss do in the first 50 steps, and why is that a *worse* failure than the loss going up?
- **Runs on:** CPU
- **Interview hooks:** Why is a decoder-only LM "T training signals per sequence" while a classifier is one? What breaks if you shift by two? Why can you not use this objective as-is for a bidirectional encoder?
- **Common misconception:** That the model is trained to predict only the last token. Every position is supervised simultaneously; the causal mask is the only thing preventing that from being cheating.

### 4.4 — Cross-entropy over `(B,T,V)` and the loss you should see at step 0
- **Question it answers:** How do I compute the loss, and what value proves the model is correctly initialised?
- **Prereqs:** 4.2, 4.3, softmax, log, negative log-likelihood.
- **Experiment:** `F.cross_entropy(logits.view(B*T, V), y.view(B*T))` on a freshly-initialised model with `V=65`. Predict the value first. Then compute `-log(1/65)` by hand and compare. Repeat with `V=50257`.
- **Shapes to nail:** CE wants `(N, V)` and `(N,)`. `N = B*T`. `.view` requires contiguity — know when to use `.reshape`. Output is a scalar `()` unless `reduction='none'`, which gives `(N,)`.
- **Predict-first prompt:** For `V=50257`, what loss should a correctly-initialised untrained model report? What does it mean if you see 13.0 instead?
- **Runs on:** CPU
- **Interview hooks:** Why does `cross_entropy` take logits rather than probabilities? What is the log-sum-exp trick and what overflows without it? What does `ignore_index=-100` do and when do you need it (hint: padding, and instruction-tuning prompt masking in Phase 6)?
- **Common misconception:** That you should softmax before calling `cross_entropy`. Doing so applies softmax twice and silently flattens your gradients.

---

## 4B — The data pipeline

### 4.5 — A corpus to a token stream to blocks
- **Question it answers:** How does a text file become a tensor the model can be trained on?
- **Prereqs:** tokenisation (Phase 2), `torch.tensor`, dtypes.
- **Experiment:** Take ~1MB of TinyShakespeare. Encode with a char-level vocab first (so you can read the IDs), store as `torch.uint16` or `np.uint16`, print `len(data)`, `data.dtype`, `data[:20]`, and decode it back. Then repeat with a BPE tokenizer and compare token counts.
- **Shapes to nail:** The whole corpus is one flat `(N,)` tensor. Documents are *concatenated*, not padded. `N_blocks = N // block_size`, and blocks straddle document boundaries unless you insert an EOS/BOS token.
- **Predict-first prompt:** 1MB of English text: how many characters, and roughly how many BPE tokens? What is the bytes-per-token ratio you expect, and why does that number matter for the bits/byte metric later?
- **Runs on:** CPU
- **Interview hooks:** Why store token IDs as uint16 and what breaks at vocab > 65535? What is the cost of concatenating documents without a separator? Why does pretraining use a flat stream while SFT uses padded per-example batches?
- **Common misconception:** That each training example is one document. In pretraining, an example is an arbitrary `block_size` window of a document-concatenated stream.

### 4.6 — The batch sampler, the train/val split, and why it is not a DataLoader
- **Question it answers:** How do I get `(x, y)` batches, and where does the split go so validation loss means something?
- **Prereqs:** 4.3, 4.5, `torch.randint`, `torch.stack`.
- **Experiment:** `get_batch(split)`: draw `ix = torch.randint(len(data) - block_size, (B,))`, stack `data[i:i+T]` and `data[i+1:i+T+1]`. `B=4, T=8`. Print `x.shape`, `y.shape`, `x.device`. Split 90/10 by slicing the stream, not by shuffling windows.
- **Shapes to nail:** `ix (B,)`; `x (B,T)` int64 after `.long()`; `y (B,T)`. One epoch is not a meaningful unit here — you sample with replacement and count *steps*, not epochs.
- **Predict-first prompt:** If you split by randomly assigning *windows* rather than by slicing the stream, what leaks, and in which direction does the val loss move?
- **Runs on:** MPS (add `pin_memory` + `non_blocking` only when you move to CUDA; MPS shares memory with the CPU so it is a no-op)
- **Interview hooks:** Why does random-offset sampling with replacement work for pretraining while SFT needs a real shuffled epoch? How would you shard this sampler across 8 data-parallel ranks without duplicating samples? Where would you put tokenisation — online or offline — and why does that decision flip at scale?
- **Common misconception:** That you need `Dataset`/`DataLoader`. For a flat token stream, a `randint` + slice is faster, simpler and has no worker-process failure modes.

---

## 4C — Training

### 4.7 — The minimal training loop and the optimizer
- **Question it answers:** What are the five lines that actually constitute training, and how should AdamW parameter groups be set up?
- **Prereqs:** 4.4, 4.6, autograd basics (Phase 2), `.backward()`, `.step()`, `zero_grad()`.
- **Experiment:** 200 steps on TinyShakespeare char-level, `B=8, T=16, d_model=64, n_layers=2, n_heads=4`. Two AdamW param groups: weight-decay on all 2-D params (matmul weights, embeddings), no decay on 1-D params (LayerNorm gains, biases). Print the count in each group.
- **Shapes to nail:** every `p.grad` has `p.shape`; `loss` is `()`; the optimizer state for AdamW holds two extra tensors per param (exp_avg, exp_avg_sq) — so optimizer memory is roughly 2x params in fp32.
- **Predict-first prompt:** A 124M-param model trained in fp32 with AdamW: how many bytes for params, grads and optimizer state before any activations? Predict the multiplier on parameter count.
- **Runs on:** MPS
- **Interview hooks:** Why `set_to_none=True` in `zero_grad`? Why is weight decay excluded from LayerNorm and biases? What is the difference between Adam's L2 penalty and AdamW's decoupled decay, and which one interacts badly with adaptive scaling? What does `betas=(0.9, 0.95)` (rather than 0.999) buy for LLM training?
- **Common misconception:** That `loss.backward()` overwrites gradients. It *accumulates* — which is exactly why gradient accumulation works, and exactly why forgetting `zero_grad` silently trains on a running sum.

### 4.8 — Overfit 200 tokens: the correctness proof
- **Question it answers:** How do I *prove* the model, labels and loop are correct before spending any real compute?
- **Prereqs:** 4.7.
- **Experiment:** Take a single batch of ~200 tokens, loop on that same batch for 500 steps with no dropout, no weight decay, LR 1e-3. Target: loss -> ~0. Then deliberately break the causal mask and rerun; then deliberately shift labels by 0 instead of 1 and rerun. Record all three loss curves.
- **Shapes to nail:** unchanged — this lesson is about the loss scalar over time, not shapes.
- **Predict-first prompt:** With the causal mask removed, what does the loss on a single memorised batch converge to, and is it higher or lower than with the mask? Predict, then run.
- **Runs on:** MPS
- **Interview hooks:** Why is "can it overfit one batch" the first test in every serious training codebase? Name three bugs this test catches and two it cannot. If the loss goes to zero but generation is gibberish, what is the most likely cause?
- **Common misconception:** That a falling loss means the pipeline is correct. A no-mask model also shows a falling loss — it falls *faster*, which is the tell.

### 4.9 — Gradient clipping and reading the gradient norm
- **Question it answers:** Why does every LLM training loop clip, and what is the norm being clipped?
- **Prereqs:** 4.7, vector norms.
- **Experiment:** Log `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` — it *returns* the pre-clip total norm. Plot it for 300 steps. Then set LR to 1e-1 and watch the norm spike before the loss does.
- **Shapes to nail:** the total norm is a scalar computed over the concatenation of all `p.grad` flattened — not per-tensor. Clipping rescales *every* gradient by the same factor `max_norm / total_norm`, preserving direction.
- **Predict-first prompt:** If clipping rescales all gradients by one shared factor, does it change the update *direction*? Does it change the direction of the AdamW *step*? These have different answers — say why.
- **Runs on:** MPS
- **Interview hooks:** Why clip by global norm rather than per-parameter value? Why does clipping interact with gradient accumulation (when do you clip — before or after unscaling)? Grad norm spikes at step 3000 of a long run: what is your debugging order?
- **Common misconception:** That clipping fixes instability. It bounds the damage of a bad batch; a persistently spiking norm is a symptom (bad data, too-high LR, fp16 overflow) that clipping hides rather than cures.

### 4.10 — Warmup + cosine decay, implemented by hand
- **Question it answers:** Why is the learning rate a function of step, and what shape should that function be?
- **Prereqs:** 4.7.
- **Experiment:** Write `get_lr(step)` with linear warmup over `warmup_steps`, cosine decay to `min_lr = 0.1 * max_lr` over `max_steps`, constant `min_lr` after. Plot it for 5000 steps *without training anything*. Then wire it in by setting `param_group['lr']` at the top of each step.
- **Shapes to nail:** none — but note the LR is set per *optimizer step*, not per micro-batch, which matters once gradient accumulation exists.
- **Predict-first prompt:** Sketch the loss curve difference between (a) constant LR, (b) cosine with no warmup, (c) warmup + cosine. Where specifically does the no-warmup run get hurt, and why is it the first few hundred steps?
- **Runs on:** CPU (plotting) then MPS
- **Interview hooks:** Why does Adam specifically need warmup (what is wrong with its second-moment estimate at step 1)? Why cosine and not step decay? What goes wrong if you set `max_steps` in the cosine schedule to a value you then don't train to — and why does that make mid-run checkpoints of a cosine schedule hard to compare? What is WSD/trapezoidal scheduling trying to fix about cosine?
- **Common misconception:** That `torch.optim.lr_scheduler` is required. Writing `get_lr(step)` by hand is 6 lines and removes an entire class of "did the scheduler step before or after the optimizer" bugs.

### 4.11 — Mixed precision: what actually works on MPS
- **Question it answers:** What do fp16, bf16 and fp32 each do to memory, speed and numerics — and which of that is real on an M3?
- **Prereqs:** 4.7, floating-point representation (exponent vs mantissa bits).
- **Experiment:** Print `torch.finfo(torch.float16).max` vs `torch.finfo(torch.bfloat16).max` vs fp32. Then run the same 100 training steps under `torch.autocast(device_type="mps", dtype=torch.bfloat16)` and without, comparing wall time, peak memory and the loss curve. Verified on this machine (torch 2.10): MPS autocast accepts both fp16 and bf16, and matmuls do come back in the autocast dtype. `torch.amp.GradScaler(device="mps")` constructs and reports enabled.
- **Shapes to nail:** shapes are unchanged; *dtypes* are what move. Master weights stay fp32, activations become 16-bit, the loss is accumulated in fp32. Check `.dtype` after a matmul inside vs outside the autocast context.
- **Predict-first prompt:** bf16 has the same exponent range as fp32 but 8 fewer mantissa bits; fp16 has more mantissa but a max of 65504. Which one needs a GradScaler and why? Predict before reading the answer.
- **Runs on:** MPS — with a caveat. Apple Silicon's unified memory means the memory *saving* from 16-bit is real, but the throughput win is far smaller than on a CUDA tensor core, where bf16 matmul is a distinct hardware path. Do not expect the 2-3x you read about in CUDA blog posts. Run the same A/B on a Colab T4 (fp16 only — T4 is Turing and has no bf16) or an A10/L4 (bf16) to see the real speedup, and to see a live GradScaler actually skipping steps on inf.
- **Interview hooks:** Why did bf16 largely replace fp16 for LLM training? What exactly does GradScaler do, and what is `found_inf` doing to your effective batch count? Why are LayerNorm and softmax usually kept in fp32 even under autocast? What is fp8 training buying and what does it cost?
- **Common misconception:** That autocast converts the model to half precision. It does not — parameters stay fp32; autocast only chooses a dtype per *op* via an allow/deny list. `model.half()` is a different, more dangerous thing.

### 4.12 — Reproducibility: seeds, determinism and what you cannot control
- **Question it answers:** What do I have to seed to get a bit-identical rerun, and where is that impossible?
- **Prereqs:** 4.6, 4.7.
- **Experiment:** Run 20 steps twice with `torch.manual_seed(1337)` at the top; diff the loss lists. Then move the seed call to *after* model construction and diff again. Then run the same matmul on CPU and MPS and print `(a - b).abs().max()`.
- **Shapes to nail:** none — but note that the seed consumption order is a hidden state machine: init, dropout and `get_batch` all draw from the same generator unless you give each its own `torch.Generator`.
- **Predict-first prompt:** Same seed, same code, same machine, but you changed `B` from 4 to 8. Will step 0's loss be identical? Will it be *similar*? Why?
- **Runs on:** MPS/CPU
- **Interview hooks:** Why is fp addition non-associative and what does that do to multi-GPU reduction determinism? What does `torch.use_deterministic_algorithms(True)` cost? In a distributed run, which seeds must differ across ranks and which must match? (This lesson is the direct prereq for 5.9, determinism at *serving* time.)
- **Common misconception:** That "set the seed" means one call. Reproducibility is data order + init + dropout + cuDNN/MPS kernel selection + reduction order, and on different hardware the last one is simply not available to you.

### 4.13 — Checkpointing and resuming without silently corrupting the run
- **Question it answers:** What goes into a checkpoint besides the weights, and what breaks if each piece is missing?
- **Prereqs:** 4.7, 4.10, 4.12.
- **Experiment:** Save `{model_state_dict, optimizer_state_dict, step, config, best_val_loss, rng_state}`. Then run 100 steps straight; separately run 50, save, kill the process, resume, run 50 more. Diff the two loss curves. Then deliberately omit `optimizer_state_dict` on resume and diff again.
- **Shapes to nail:** the state dict is `{str: Tensor}` with the param shapes; the optimizer state has two same-shaped tensors per param plus scalar `step` counters. File size ≈ 4 bytes/param (weights) + 8 bytes/param (Adam moments) in fp32.
- **Predict-first prompt:** You resume with the weights but not the optimizer state. Predict the shape of the loss curve for the next 200 steps. Does it spike, plateau or continue smoothly — and why is this the most common silent bug in long runs?
- **Runs on:** MPS/CPU
- **Interview hooks:** Why does checkpoint frequency trade against wasted compute on preemption, and how do you pick the interval? What is the atomic-write pattern (write to tmp, fsync, rename) and why does a partial checkpoint kill a week-long run? How does sharded/distributed checkpointing (FSDP `state_dict_type`) differ, and why can you not just `torch.save` the rank-0 model? Why does `torch.load(weights_only=True)` matter for anything you did not produce yourself?
- **Common misconception:** That the weights are the checkpoint. The optimizer state is 2x the size of the weights and dropping it costs you hundreds of steps of re-warming Adam's moments.

---

## 4D — Measuring and generating

### 4.14 — Perplexity, bits/byte, and what counts as a good number
- **Question it answers:** My loss is 1.8 — is that good? Good compared to what?
- **Prereqs:** 4.4, 4.5, logs and log-base change.
- **Experiment:** Write three conversions and check them against each other on your own val loss: `ppl = exp(loss_nats)`; `bits_per_token = loss_nats / ln(2)`; `bits_per_byte = bits_per_token * (n_tokens / n_bytes)`. Compute all three for your char-level model and for a BPE model on the *same* text, and confirm that BPB is comparable across the two while perplexity is not.
- **Shapes to nail:** the loss must be a *token-weighted* mean over the whole val set, not a mean of per-batch means, whenever batches have different valid-token counts. Accumulate `sum_nll` and `sum_tokens` separately.
- **Predict-first prompt:** Two models report perplexity 12 and 18 on the same corpus, but the first uses a 32k vocab and the second a 128k vocab. Which is the better model? Predict before doing the BPB conversion.
- **Runs on:** MPS
- **Interview hooks:** Why is perplexity not comparable across tokenizers, and why is bits/byte? What is the relationship between cross-entropy in nats and compression ratio? Why is a val loss of ~2.85 nats/token (ppl ≈ 17) the GPT-2-124M reference point on OpenWebText, while ~1.48 nats/char is the reference for char-level TinyShakespeare? Why do benchmark scores and perplexity decouple after instruction tuning?
- **Common misconception:** That lower perplexity always means a better model. It means better *on that tokenizer, that corpus, that context length* — change any of the three and the number is meaningless.

### 4.15 — Generating text from your own model
- **Question it answers:** How do I turn `(B,T,V)` logits into text, and what is the minimal correct loop?
- **Prereqs:** 4.1-4.4, 4.8 (a model that has actually learned something).
- **Experiment:** `generate(idx, max_new_tokens)`: crop `idx` to the last `block_size` tokens, forward, take `logits[:, -1, :]`, softmax, `torch.multinomial`, `torch.cat`. Start from a single BOS/newline token. `B=1, max_new_tokens=100`. Wrap in `@torch.no_grad()` and `model.eval()`.
- **Shapes to nail:** input `(B,T)` grows to `(B,T+1)` each iteration; `logits (B,T,V)` but you slice `[:, -1, :] -> (B,V)`; `multinomial` returns `(B,1)`. The `(B,V)` slice is where all the sampling in Phase 5 will happen.
- **Predict-first prompt:** Each step you forward the whole sequence but use only the last position's logits. For a 200-token generation with `block_size=256`, how many times is position 0's key/value recomputed? This number is the entire motivation for Phase 5.
- **Runs on:** MPS
- **Interview hooks:** Why `model.eval()` — name every module whose behaviour changes. Why is generation memory-light but latency-heavy compared to training? What does `no_grad` save, and would `inference_mode` save more?
- **Common misconception:** That you must crop to `block_size`. You must, *here*, because the learned position embedding table only has `block_size` rows — index past it and you get an IndexError, not extrapolation. RoPE changes this constraint (and its own failure mode is different).

### 4.16 — The real run: throughput, tokens/sec, and reading a loss curve
- **Question it answers:** How long will my run take, and how do I tell "converging slowly" from "broken"?
- **Prereqs:** 4.7-4.15.
- **Experiment:** Train the milestone model (~10-30M params, `block_size=256`, BPE) for a fixed wall-clock budget. Log every 50 steps: step, train loss, val loss, LR, grad norm, tokens/sec, seconds/step. Plot train vs val on a log-x axis. Add gradient accumulation to hit a target *effective* batch size in tokens and confirm the loss curve is unchanged when you halve micro-batch and double accum steps.
- **Shapes to nail:** effective tokens per optimizer step `= B * T * accum_steps * world_size`. Know this number — it is the unit every scaling-law and LR heuristic is quoted in.
- **Predict-first prompt:** You halve the micro-batch and double the accumulation steps. Predict what happens to (a) the loss curve, (b) peak memory, (c) wall-clock per optimizer step. One of the three should be unchanged — which?
- **Runs on:** MPS for the full lesson at 10-30M params (expect hours, not minutes). **MFU cannot be computed meaningfully on MPS** — Apple does not publish a comparable peak FLOPs figure and there is no `nvidia-smi` equivalent for utilisation. Do the MFU half on a Colab T4 (65 TFLOPS fp16 peak) or a rented A10/L4/A100, where `model_flops / (peak_flops * seconds)` is a defensible number.
- **Interview hooks:** Derive tokens/sec from `6 * N * tokens` FLOPs and your measured MFU. Why is 40-50% MFU good and 15% a red flag? Train loss flat, val loss rising — diagnosis? Both flat after a good start — diagnosis? Loss to NaN at step 4000 — what do you check, in order?
- **Common misconception:** That a train/val gap means overfitting. At pretraining scale on a large corpus you will usually see *no* gap at all; a gap means your dataset is small relative to the model, which is a scaling-law problem, not a regularisation one.

---

## 4E — Situating what you built

### 4.17 — Scaling laws: Kaplan vs Chinchilla, and costing a real run
- **Question it answers:** Given a compute budget, how big a model and how many tokens — and where does `C ≈ 6ND` come from?
- **Prereqs:** 4.16, parameter counting, FLOPs per matmul.
- **Experiment:** No training. Build a spreadsheet/notebook calculator: inputs `N` (non-embedding params) and `D` (tokens), outputs FLOPs `C = 6ND`, GPU-hours at a given peak-FLOPs and MFU, and dollars at a given $/hour. Derive the 6 yourself: 2 FLOPs per MAC, forward ≈ `2N` per token, backward ≈ `4N` per token. Then verify the Chinchilla ratio on three real models by looking up their N and D.
- **Shapes to nail:** not tensors — units. FLOPs (not FLOPS), tokens, params, GPU-hours. Keep `N` as *non-embedding* params when applying `6ND`, and say why the embedding table is excluded.
- **Predict-first prompt:** A 124M-param model, Chinchilla-optimal at 20 tokens/param, so D ≈ 2.5B tokens. Compute C = 6ND yourself. Then, at 30% MFU on a T4 (≈65 TFLOPS fp16 peak, so ≈2e13 effective FLOP/s), predict the wall-clock hours before you divide. (Answer to check against: C ≈ 1.9e18 FLOPs, ≈26 hours.)
- **Runs on:** CPU
- **Interview hooks:** Kaplan (2020) implied model size should grow much faster than data — roughly 1.7 tokens/param at GPT-3 scale; Chinchilla (Hoffmann et al., 2022) found ≈20 tokens/param compute-optimal and trained 70B on 1.4T tokens to beat 175B on 300B. What did Kaplan get wrong (LR schedule not tuned per run, embedding params counted, limited small-model range)? Why do production models like Llama-3 train far *past* Chinchilla-optimal — what objective is being optimised when you do that? At what token count do you stop, if inference cost dominates training cost by 100x?
- **Common misconception:** That Chinchilla says "20 tokens per parameter" as a law of nature. It is the *compute-optimal* point for a fixed training budget, which is the wrong objective if you will serve the model a trillion times. It is also fit on a specific architecture/data regime and the constants move.

### 4.18 — Read nanoGPT and a real `modeling_*.py`, and map every line
- **Question it answers:** Can I open unfamiliar production model code and recognise every component as something I built?
- **Prereqs:** all of Phase 4.
- **Experiment:** Two passes. (1) `nanoGPT/model.py` (~300 lines): annotate every class and every non-obvious line against your own implementation, and list every place nanoGPT does something you did not — `_init_weights` scaled by `1/sqrt(2*n_layer)` on residual projections, the `configure_optimizers` param-group split, `F.scaled_dot_product_attention` vs the manual path, `crop_block_size`. (2) A HuggingFace `modeling_llama.py`: find RoPE, RMSNorm, SwiGLU, GQA, the KV cache object, and the `past_key_values` plumbing. Write a two-column diff: "same as mine" / "different, and why".
- **Shapes to nail:** confirm by reading, not running: what shape `past_key_values` holds per layer, and where `q_len == 1` is special-cased in the decode path. This is the bridge into 5.10.
- **Predict-first prompt:** Before opening it: name five things you expect a production modelling file to contain that your toy does not. Then check how many you got.
- **Runs on:** CPU (reading); instantiate a tiny config on MPS to step through with a debugger.
- **Interview hooks:** Why is the residual projection initialised with a depth-scaled std? What does `scaled_dot_product_attention` dispatch to, and how do you check which backend was chosen? Why does HF separate `LlamaModel` from `LlamaForCausalLM`? What is `attention_mask` vs `causal_mask` vs `position_ids` actually doing, and why are there three?
- **Common misconception:** That production code is doing something conceptually deeper. It is mostly your code plus config plumbing, dtype handling, cache objects, and three years of accumulated compatibility flags — and being able to say that with confidence is the skill.

---

## Milestone (Phase 4)

`projects/nano-gpt/` containing: your from-scratch GPT (no copied modelling code), a
training script with clipping + warmup/cosine + checkpoint/resume + seeding, a
trained checkpoint, a `RESULTS.md` with the loss curve, final val loss reported in
nats/token **and** perplexity **and** bits/byte, tokens/sec, total FLOPs via `6ND`,
and 200 tokens of generated sample text. Plus a `tests/test_gpt.py` asserting:
causal masking (changing token `t` cannot change logits at positions `< t`),
loss-at-init ≈ `ln(V)`, weight tying is a shared tensor, and resume-from-checkpoint
reproduces the uninterrupted loss curve.

---
---

# PHASE 5 — Inference and serving

## What you can claim after this phase

- I can explain, with arithmetic intensity numbers rather than adjectives, why
  prefill is compute-bound and decode is memory-bandwidth-bound — and therefore
  why every serving optimisation targets one phase or the other.
- I implemented every sampling strategy by hand (greedy, temperature, top-k,
  top-p, min-p, repetition/frequency penalties) and can say what each does to
  the tail of the distribution and why that matters for a given workload.
- I derived and implemented a KV cache, and I can state the memory formula from
  memory, compute it for a real model, and use it to size a GPU fleet.
- I can explain continuous batching, PagedAttention, prefix caching, chunked
  prefill, disaggregated prefill/decode and speculative decoding at the level of
  "what problem, what mechanism, what it costs" — and map each onto a scheduling
  or memory-management concept I already know from backend systems.
- I can build a latency budget: TTFT vs ITL vs end-to-end, p50 vs p99, and
  defend a max-batch-size / admission-control choice against an SLO.

**Backend bridge, stated once and referenced throughout:** an LLM server is a
queueing system with an unusual service-time model. A request's service time is
`prefill_time(prompt_len) + n_output_tokens * ITL(batch_size)`, where the output
length is *unknown at admission time* and ITL depends on what else is in the
batch. Every Phase 5 topic is one of: admission control (5.13, 5.14, 5.23),
memory allocation (5.11, 5.12, 5.15, 5.16), work-conserving scheduling (5.14,
5.17), resource isolation across workload classes (5.18), or latency hiding
(5.19). You already have the instincts; this phase supplies the LLM-specific
service-time model.

---

## 5A — The generation loop and its cost structure

### 5.1 — Autoregressive generation as a loop, and the quadratic waste
- **Question it answers:** What exactly is recomputed on every decode step, and how much of it is redundant?
- **Prereqs:** 4.15.
- **Experiment:** Instrument the naive `generate` from 4.15: count forward-pass FLOPs (or just token-positions processed) per step, and plot cumulative work vs tokens generated. `prompt_len=16, max_new_tokens=64`. Then time each step individually and plot step time vs step index.
- **Shapes to nail:** step `i` forwards `(1, prompt_len + i)` and discards all but `[:, -1, :]`. Total positions processed over `n` steps is `O(n^2)`, of which `O(n)` is new.
- **Predict-first prompt:** Plot the per-step latency of naive generation against step index. Is it flat, linear or quadratic? Predict the shape, then run — and explain the gap between your prediction and what you measure.
- **Runs on:** MPS
- **Interview hooks:** What is the asymptotic FLOP cost of generating `n` tokens with and without a cache? Why does the *wall clock* per step not grow as fast as the FLOP count at small `n`? What does this tell you about which resource is actually saturated?
- **Common misconception:** That the fix is "make attention faster". The fix is to not recompute keys and values at all — and noticing that requires seeing that K and V for past positions are a pure function of past tokens.

### 5.2 — Prefill vs decode: compute-bound vs memory-bandwidth-bound
- **Question it answers:** Why do the two phases of a request have completely different performance characteristics, and which resource limits each?
- **Prereqs:** 5.1, FLOPs counting from 4.17, the idea of memory bandwidth.
- **Experiment:** For one linear layer `(d_model=4096 -> 4096)` in fp16: compute FLOPs and bytes-moved for a `(1, 1, 4096)` input (decode) and a `(1, 2048, 4096)` input (prefill). Arithmetic intensity = FLOPs / bytes. Do it on paper first, then confirm with a timing sweep of batch-seq sizes 1, 8, 64, 512, 2048 on MPS and plot achieved throughput vs input tokens — the knee is the roofline turning point.
- **Shapes to nail:** decode: weights `(4096,4096)` = 33.5MB moved to do `2*4096*4096 ≈ 33.5 MFLOP` — intensity ≈ 1 FLOP/byte. Prefill with 2048 tokens: same 33.5MB moved, 2048x the FLOPs — intensity ≈ 2000 FLOP/byte. Modern accelerators need ~100-300 FLOP/byte to saturate compute.
- **Predict-first prompt:** In decode, batch size 1 vs batch size 32: the FLOPs go up 32x. Does the step latency go up 32x? Predict, then reason about which bytes are shared across the batch and which are not.
- **Runs on:** MPS for the sweep (the *shape* of the roofline is visible on M3 unified memory; the absolute numbers differ from HBM). Do the same sweep on a T4/A100 to see a sharper knee.
- **Interview hooks:** Give the arithmetic intensity of decode at batch size B and explain why increasing B is nearly free until it isn't. Why does a bigger model help decode throughput *per parameter*? Why is quantisation a *latency* win in decode but often not in prefill? Which phase does FlashAttention help more, and why?
- **Common misconception:** That decode is slow because attention is quadratic. At batch 1, decode is slow because you stream the entire weight matrix from memory to do one token's worth of arithmetic. Attention is a minority of decode time until context gets long.

---

## 5B — Turning logits into tokens

### 5.3 — Logits to probabilities: softmax and temperature in logit space
- **Question it answers:** What is a logit, and what does temperature do to the distribution — exactly?
- **Prereqs:** 4.4, softmax.
- **Experiment:** Fixed logit vector of length 8. Apply `softmax(logits / T)` for `T` in `{0.1, 0.5, 1.0, 1.5, 2.0}` and print all five distributions plus their entropy. Then show `T -> 0` approaches argmax and `T -> inf` approaches uniform.
- **Shapes to nail:** last-step logits `(B, V)`; probabilities `(B, V)` summing to 1 along dim=-1. Temperature is applied to logits *before* softmax and before any truncation.
- **Predict-first prompt:** Adding a constant `c` to every logit — does it change the probabilities? Multiplying every logit by 2 — does it? Answer both before running.
- **Runs on:** CPU
- **Interview hooks:** Why is softmax shift-invariant but not scale-invariant, and how is the first fact used for numerical stability? What is temperature 0 in practice (no implementation actually divides by zero)? Why do serving APIs apply temperature before top-p rather than after?
- **Common misconception:** That temperature "makes the model more creative". It only rescales an already-fixed logit vector; it cannot introduce information the model did not already assign mass to.

### 5.4 — Greedy decoding, and why the most likely token is not the most likely sequence
- **Question it answers:** What is greedy, when is it right, and what does it systematically fail at?
- **Prereqs:** 5.3.
- **Experiment:** `argmax(logits[:, -1, :])` on your Phase 4 model, generate 100 tokens from a fixed prompt, three times. Confirm byte-identical output. Then generate 200 tokens and find where it starts looping.
- **Shapes to nail:** `(B,V) -> argmax(dim=-1) -> (B,)` -> `unsqueeze(-1) -> (B,1)` for the `cat`.
- **Predict-first prompt:** Greedy picks the highest-probability token at each step. Does that produce the highest-probability *sequence*? Construct a 2-step, 2-token counterexample on paper before you answer.
- **Runs on:** MPS
- **Interview hooks:** When *should* you use greedy in production (structured extraction, evals, caching)? What is beam search doing differently, and why is it largely absent from LLM serving stacks despite dominating NMT? Why does greedy degenerate into repetition — what property of the learned distribution causes it?
- **Common misconception:** That greedy is "the deterministic option" and everything else is noise. Greedy is a specific, biased search that reliably overproduces high-frequency text.

### 5.5 — Top-k sampling
- **Question it answers:** How do I cut off the tail with a fixed budget, and what does that cost?
- **Prereqs:** 5.3, `torch.topk`.
- **Experiment:** Implement in 4 lines: `v, _ = torch.topk(logits, k)`; `logits[logits < v[:, [-1]]] = -inf`; softmax; multinomial. `k=5` on a `V=65` char model. Print the surviving token set for a confident position and an uncertain one.
- **Shapes to nail:** `topk` returns values `(B,k)` and indices `(B,k)`. `v[:, [-1]]` keeps `(B,1)` so it broadcasts against `(B,V)` — using `v[:, -1]` gives `(B,)` and broadcasts along the wrong axis. Masking with `-inf` before softmax (not zeroing after) is what keeps renormalisation correct.
- **Predict-first prompt:** The model is 99% sure the next token is `,`. With `k=50`, how many of those 50 candidates have non-negligible probability after renormalisation? What does that tell you about top-k's weakness?
- **Runs on:** CPU
- **Interview hooks:** Why mask logits with `-inf` rather than zero probabilities and renormalise — are they equivalent, and what about gradients/numerics? What is the cost of `topk` over a 128k vocab per token per sequence, and does it show up in your ITL budget? How does top-k interact with temperature ordering?
- **Common misconception:** That top-k adapts to model confidence. It keeps exactly `k` regardless — too permissive when the model is certain, too restrictive when it is genuinely uncertain. That is precisely the gap top-p and min-p address.

### 5.6 — Top-p (nucleus) sampling
- **Question it answers:** How do I make the truncation adapt to the shape of the distribution?
- **Prereqs:** 5.5, `torch.sort`, `torch.cumsum`, `scatter`.
- **Experiment:** Sort descending, cumsum the softmax, mask everything after cumulative probability first exceeds `p`, un-sort via `scatter`, renormalise. `p=0.9`. Print the nucleus size for a confident vs uncertain position — the whole point is that these two numbers differ.
- **Shapes to nail:** `sort` gives `(B,V)` values and `(B,V)` indices; cumsum along `dim=-1`; the shift-by-one on the mask (`mask[..., 1:] = mask[..., :-1].clone()`) is what guarantees at least one token survives. `scatter` maps the sorted mask back to vocab order — get this wrong and you mask random tokens with no error raised.
- **Predict-first prompt:** With the off-by-one wrong, what happens when the top token alone has probability > p? Predict the failure, then introduce it deliberately and watch.
- **Runs on:** CPU
- **Interview hooks:** Top-p is `O(V log V)` per token from the sort; top-k is `O(V log k)`. At `V=128k` and batch 256 does that matter? What happens when you compose top-k and top-p — which applies first and why does the order matter? Why do most APIs default to `top_p=1.0` and expose temperature instead?
- **Common misconception:** That top-p and top-k are alternatives that do the same thing. Top-p's nucleus size varies by orders of magnitude across positions; that variance is the feature, and it is also why top-p output length/quality is harder to predict.

### 5.7 — Min-p sampling
- **Question it answers:** What does min-p fix about top-p, and why does it hold up better at high temperature?
- **Prereqs:** 5.6.
- **Experiment:** Implement: threshold `= min_p * p_max` where `p_max` is the top token's probability; keep tokens with `p >= threshold`. `min_p=0.05`. On the same two positions from 5.6, compare the surviving-set sizes for top-p vs min-p. Then sweep temperature 0.7 -> 2.0 and plot surviving-set size for both.
- **Shapes to nail:** `p_max = probs.max(dim=-1, keepdim=True).values -> (B,1)`; threshold `(B,1)` broadcasts against `(B,V)`. No sort required — this is `O(V)`.
- **Predict-first prompt:** Model is confident (`p_max = 0.9`) vs uncertain (`p_max = 0.15`). With `min_p = 0.05`, what is the absolute probability floor in each case? Which regime lets more tokens through, and is that the behaviour you want?
- **Runs on:** CPU
- **Interview hooks:** Min-p is a *relative* floor and top-p is a *cumulative* budget — construct a distribution where they disagree sharply. Why does min-p degrade more gracefully as temperature rises (the claim in Nguyen et al., 2024, "Turning Up the Heat")? Why is min-p cheaper to compute than top-p at serving scale?
- **Common misconception:** That min-p is a fixed probability floor. The floor is `min_p * p_max` — it moves with the model's confidence, which is the entire mechanism.

### 5.8 — Repetition, frequency and presence penalties
- **Question it answers:** How do you suppress loops, and what are the three different penalty formulations actually doing?
- **Prereqs:** 5.3, 5.4 (you have seen greedy loop).
- **Experiment:** Three implementations on the same logits: (a) repetition penalty — divide positive logits by `r`, multiply negative ones by `r` (the CTRL formulation; note the sign asymmetry); (b) frequency penalty — subtract `alpha * count(token)`; (c) presence penalty — subtract `beta * [count > 0]`. Build the count tensor with `torch.bincount` or `scatter_add`. Generate from a looping prompt with each.
- **Shapes to nail:** counts `(B, V)` built from the generated-so-far `(B, t)` ids; penalties applied to `(B,V)` logits before temperature. Decide and document whether the prompt tokens count toward the penalty — this is a real, commonly-mismatched API semantic.
- **Predict-first prompt:** Why does the repetition penalty divide *positive* logits but multiply *negative* ones? What would a single unconditional division do to a token whose logit is -8?
- **Runs on:** CPU
- **Interview hooks:** Why do penalties hurt code and structured output specifically? Frequency vs presence — when does each fit the workload? Why is repetition a symptom of the *model* and penalties only a serving-side patch? What state must the server keep per request to apply these, and how does that interact with prefix caching (5.16)?
- **Common misconception:** That repetition penalty is monotonic and safe. Applied over the whole context it will eventually penalise required tokens — `def`, `{`, `the` — and the failure looks like a subtly incoherent model, not an obvious bug.

### 5.9 — Determinism at serving time: what "seed" actually buys you
- **Question it answers:** Same seed, same prompt, same model — will I get the same output on a production server?
- **Prereqs:** 4.12, 5.5-5.7.
- **Experiment:** Sample with a fixed `torch.Generator` twice: identical. Then run the same single prompt (a) alone and (b) padded into a batch of 8 with other sequences, and diff the logits with `(a-b).abs().max()`. Then the same forward on CPU vs MPS and diff. Expect small non-zero differences in both cases — that is the lesson.
- **Shapes to nail:** none new; what matters is that a `(1,V)` logit row is *not* bitwise identical when the same sequence is computed inside a `(8,V)` batch, because reduction order and kernel selection change with batch shape.
- **Predict-first prompt:** A logit differs by 1e-6 between two runs. What is the probability that the sampled token changes? Under what condition is it near-zero, and under what condition is it near-50%?
- **Runs on:** MPS/CPU
- **Interview hooks:** What is batch-invariance in inference and why is it hard (fp non-associativity + shape-dependent kernel/split selection)? Why can a provider promise "seed" as best-effort but not as a guarantee? How would you build a genuinely reproducible eval harness on top of a non-deterministic server? Why does temperature 0 not fully solve it?
- **Common misconception:** That greedy decoding is deterministic end to end. Greedy is deterministic *given logits*; the logits are not batch-invariant, so two near-tied tokens can swap depending on who else is in your batch.

---

## 5C — The KV cache

### 5.10 — Deriving the KV cache, and why Q is not cached
- **Question it answers:** Which tensors in attention are reusable across decode steps, and which is structurally not?
- **Prereqs:** 5.1, Phase 3 attention (Q/K/V projections), 4.15.
- **Experiment:** Run attention twice — once on `(1,4,d)` and once on `(1,5,d)` where the first 4 tokens are identical. Print `K` and `V` for both and assert the first 4 rows are equal. Then print `Q` for both and observe the same thing — then ask *why that fact is useless*. Then implement the cache: a per-layer `(B, n_kv_heads, T, d_head)` pair, appended along `dim=2`, with the decode forward taking `q_len=1`.
- **Shapes to nail:** decode step: `q (B, n_heads, 1, d_head)`; cached `k, v (B, n_kv_heads, T_past+1, d_head)`; scores `q @ k.transpose(-2,-1) -> (B, n_heads, 1, T_past+1)`; output `(B, n_heads, 1, d_head) -> (B, 1, d_model)`. The causal mask disappears in pure decode — a single query attending to all cached keys is already causal.
- **Predict-first prompt:** K and V for past tokens are reused. Q for past tokens is *also* identical across steps. So why is there no Q cache? Answer in terms of which rows of the attention output you actually consume at each step.
- **Runs on:** MPS
- **Interview hooks:** Why does the causal mask become unnecessary during decode but essential during prefill? What is the FLOP reduction from caching, exactly? Why do cached K/V for a RoPE model store *post*-rotation keys, and what breaks if you cache pre-rotation? What has to happen to the cache when a request is preempted?
- **Common misconception:** That Q is not cached because it changes. It doesn't change — but you only ever need the *last* row of the attention output, so only the last row's query is needed. The cache exists to avoid recomputing things you *do* need, not things that merely happen to be stable.

### 5.11 — `n_kv_heads`: MHA vs MQA vs GQA as a KV-cache decision
- **Question it answers:** Why do modern models have fewer K/V heads than Q heads?
- **Prereqs:** 5.10, multi-head attention from Phase 3. (If Phase 3 already covered GQA, this is a 10-minute recap focused only on the cache consequence.)
- **Experiment:** Same model config three ways: `n_heads=8` with `n_kv_heads` = 8 (MHA), 2 (GQA), 1 (MQA). Implement the K/V head expansion (`repeat_interleave` on the head dim, or an equivalent broadcast) and confirm output shapes are identical in all three. Print the cache size per token for each.
- **Shapes to nail:** `q (B, 8, T, d_head)`; `k (B, n_kv_heads, T, d_head)` expanded to `(B, 8, T, d_head)` with each K/V head repeated `8/n_kv_heads` times. The expansion is a view/broadcast where possible — materialising it defeats the memory saving at the attention kernel boundary.
- **Predict-first prompt:** GQA with 8 Q heads and 2 KV heads: what fraction of the MHA cache remains? Does the *parameter* count drop by the same fraction? Does the *FLOP* count of attention drop at all?
- **Runs on:** MPS
- **Interview hooks:** What does MQA give up in quality and what does GQA recover? Why did MQA/GQA matter more for serving than for training? How does GQA interact with tensor parallelism when `n_kv_heads < tp_size`? Where does MLA (DeepSeek) sit on this axis?
- **Common misconception:** That GQA reduces attention compute. It reduces KV *memory* and *bandwidth*; after expansion the attention math is the same shape as MHA.

### 5.12 — The KV cache memory formula, with worked examples
- **Question it answers:** Exactly how many bytes does one request cost, and how many concurrent requests fit on one GPU?
- **Prereqs:** 5.10, 5.11.
- **Experiment:** Write the formula as code and check it against a measured cache allocation for your own toy model:
  `bytes = 2 * n_layers * n_kv_heads * d_head * dtype_bytes * seqlen * batch`
  (the leading 2 is K and V). Then compute, by hand, for three real configs and check against a lookup. Reference answers to verify against:
  - **Llama-3-8B** (32 layers, 8 KV heads, d_head 128, fp16): `2*32*8*128*2 = 131072` bytes = **128 KiB/token**; 8192 tokens = **1 GiB per sequence**.
  - **Llama-2-7B, MHA** (32 layers, 32 heads, d_head 128, fp16): **512 KiB/token**; 4096 tokens = **2 GiB per sequence**.
  - Then: on an 80GB A100 holding 16GB of fp16 Llama-3-8B weights, how many 8k-context sequences fit in the remaining ~60GB?
- **Shapes to nail:** the per-layer cache pair is `(B, n_kv_heads, T, d_head)` x2. Know each factor's origin: 2 = K and V, `n_layers` = an independent cache per layer, `n_kv_heads * d_head` = the KV width (not `d_model`, once GQA exists).
- **Predict-first prompt:** Before computing: is the Llama-3-8B KV cache for a single 128k-context request larger or smaller than the model weights? Predict, then compute.
- **Runs on:** CPU (arithmetic); measure real allocation on MPS with your toy model.
- **Interview hooks:** Derive max concurrent requests on a given GPU given weights, activations and a context length. Why does fp8/int8 KV quantisation give a nearly proportional throughput win? Why does long context break serving economics even when attention itself is fast? Where does this formula change for MLA, or for sliding-window attention?
- **Common misconception:** That the KV cache is a minor overhead next to the weights. At production context lengths and batch sizes it is routinely the *majority* of GPU memory, and it is the actual binding constraint on throughput.

---

## 5D — Batching and memory management

### 5.13 — Static batching and why it wastes most of the GPU
- **Question it answers:** What goes wrong if you batch LLM requests the way you would batch a classifier?
- **Prereqs:** 5.1, 5.2.
- **Experiment:** Simulate (no serving framework needed): 8 requests with output lengths `[10, 300, 15, 20, 250, 12, 18, 30]`. Under static batching the batch runs until the longest finishes. Compute total GPU-steps, the useful fraction, and the TTFT of a request that arrives one step after the batch launched.
- **Shapes to nail:** the batch tensor is `(8, T_max)` with padding; the fraction of the `(8, T_max)` grid that is real work is your utilisation number. Make the padded grid visible as an ASCII plot.
- **Predict-first prompt:** With that length distribution, what fraction of computed positions is useful work? Predict before computing.
- **Runs on:** CPU (simulation)
- **Interview hooks:** This is head-of-line blocking with an unknown service time — where have you solved that before, and what did you do? Why is the classifier batching intuition (pad to max, one shot) actively harmful here? Why is the output length not known at admission, and what does that break about classical scheduling?
- **Common misconception:** That you can fix it by bucketing requests by length. You cannot bucket by *output* length, because nobody knows it — including the model — until the EOS token appears.

### 5.14 — Continuous / in-flight batching
- **Question it answers:** What does "batch at the iteration level instead of the request level" actually mean in code?
- **Prereqs:** 5.13, 5.10.
- **Experiment:** Build a toy scheduler around your Phase 4 model: a `running` list and a `waiting` queue; each iteration, run one decode step for all running sequences, evict any that emitted EOS, admit new ones from the queue if a memory budget allows, and log the batch composition per step. Replay the same 8-request workload from 5.13 and compare total steps and per-request completion times.
- **Shapes to nail:** the decode batch is `(B_t, 1)` where `B_t` *changes every iteration*; the KV cache must support per-sequence append and removal, which is why a single dense `(B, H, T, D)` tensor is awkward — this is the pain that motivates 5.15.
- **Predict-first prompt:** Continuous batching improves throughput. What does it do to the p99 ITL of a request that was already running when 20 new requests were admitted? Predict the direction and the mechanism.
- **Runs on:** MPS (the toy scheduler is the lesson; real throughput numbers need CUDA)
- **Interview hooks:** What is the analogue in your backend experience (iteration-level preemption, work-conserving scheduling)? Why does admitting a prefill into a decode batch spike ITL for everyone (setup for 5.17)? What is the eviction/preemption policy when memory runs out mid-generation — recompute or swap, and how do you choose? How does Orca's iteration-level scheduling differ from what vLLM v1 does today?
- **Common misconception:** That continuous batching means "dynamic batch size". It means the *scheduling unit is one forward pass*, so a finished sequence's slot is reclaimed immediately rather than at batch end.

### 5.15 — PagedAttention and the fragmentation problem
- **Question it answers:** Why does the KV cache need a virtual-memory-style allocator?
- **Prereqs:** 5.12, 5.14.
- **Experiment:** Two-part. (a) Simulate contiguous preallocation: reserve `max_seq_len` per request, run the 5.13 workload, and compute internal fragmentation (reserved but never used) and external fragmentation (free-but-unusable gaps). (b) Implement a toy block allocator in pure Python: `block_size=16` tokens, a free-list, a per-sequence block table `List[int]`, and `allocate`/`append`/`free`. Re-run and compare waste. Then implement copy-on-write block sharing for two sequences from the same prompt (parallel sampling / beam) and show the physical block count does not double.
- **Shapes to nail:** the physical cache is `(n_blocks, block_size, n_kv_heads, d_head)` per layer per K/V; a sequence is a *list of block indices*, not a contiguous slice. Attention must gather via the block table — that gather is the part that needs a custom kernel.
- **Predict-first prompt:** With `block_size=16`, what is the worst-case internal waste per sequence? Compare that to reserving 2048 tokens for a request that emits 30. Predict both numbers.
- **Runs on:** **Toy allocator: CPU/MPS** — the block table, free list, fragmentation accounting and CoW sharing are all pure bookkeeping and teach the entire mechanism. **Real PagedAttention: needs CUDA** — the production version is a custom CUDA kernel that reads K/V through a block table inside the attention computation; there is no MPS implementation and writing one is not a learning exercise. Run vLLM on a Colab T4 or a rented L4/A10 to see it end to end.
- **Interview hooks:** Map every PagedAttention concept to OS virtual memory: block = page, block table = page table, and what is the TLB analogue? Why is there no swapping to disk in the common path (bandwidth), and when *is* there? What is the cost of the indirection — why is a paged attention kernel slower per-FLOP than a contiguous one, and why is that trade obviously worth it? What is the right block size and what does it trade?
- **Common misconception:** That PagedAttention makes attention faster. It makes it slightly slower per token and enables 2-4x higher *batch size* by eliminating reservation waste — a memory-management win, not a kernel win.

### 5.16 — Prefix / prompt caching and its economics
- **Question it answers:** When two requests share a prefix, what can be reused, and what is that worth in dollars?
- **Prereqs:** 5.15, 5.2.
- **Experiment:** Extend the toy block allocator with a hash-of-block-contents -> physical-block map, so identical prefix blocks are shared with a refcount. Feed it 20 requests sharing a 500-token system prompt plus a distinct 50-token suffix, and report the block-reuse rate and the prefill tokens actually computed. Then compute the cost delta: prefill FLOPs saved = `6N * tokens_reused` (forward-only: `2N * tokens`).
- **Shapes to nail:** cache hits are aligned to block boundaries — a prefix match of 507 tokens with `block_size=16` reuses 31 blocks (496 tokens) and recomputes the remainder. This alignment rule is why "put the variable part last" is real advice.
- **Predict-first prompt:** A 2000-token system prompt and a 50-token user turn. What fraction of prefill FLOPs does prefix caching remove? What does it do to TTFT? What does it do to *decode* throughput — anything at all?
- **Runs on:** CPU/MPS for the toy (the hashing, refcounting and reuse accounting are the lesson). Real numbers need CUDA + vLLM (`--enable-prefix-caching`) or SGLang.
- **Interview hooks:** SGLang's RadixAttention keeps prefixes in a radix tree rather than a flat hash map — what does the tree buy over hashed blocks? Why do providers discount cached input tokens heavily, and what does that reveal about their cost structure? What is the eviction policy for a prefix cache, and why is LRU not obviously right for a tree? What is the security concern with cross-tenant prefix sharing, and what is the timing side channel?
- **Common misconception:** That prefix caching speeds up generation. It removes prefill work only — TTFT improves, ITL does not.

### 5.17 — Chunked prefill
- **Question it answers:** How do you stop a long prompt from stalling everyone else's decode?
- **Prereqs:** 5.2, 5.14.
- **Experiment:** Simulate a token budget per iteration (say 2048 tokens). A 8000-token prefill is split into 4 chunks, each co-scheduled with the decode tokens of running requests. Plot per-iteration ITL for running requests with and without chunking, and plot TTFT for the long request in both regimes.
- **Shapes to nail:** the mixed batch's flattened token dimension is `n_decode_seqs * 1 + chunk_size`; attention must handle heterogeneous per-sequence query lengths in one call (varlen/cu_seqlens layout), which is why this needs a kernel that supports it.
- **Predict-first prompt:** Chunked prefill improves ITL for running requests. What does it do to the TTFT of the long request itself, and why is that trade usually correct?
- **Runs on:** CPU (simulation of the scheduler and the latency trade). The real implementation needs a varlen attention kernel — **CUDA in practice** (FlashAttention varlen); the token-budget scheduling logic itself is device-independent and is the transferable part.
- **Interview hooks:** How do you pick the chunk token budget, and what SLO does it encode? Why does chunking hurt prefill efficiency slightly (arithmetic intensity per chunk, plus re-reading the KV of earlier chunks)? Which of your backend patterns is this — and what is the analogue of a long-running query being broken into pages?
- **Common misconception:** That chunked prefill is a throughput optimisation. It is a *tail-latency* optimisation that costs a little throughput; it exists because one 30k-token prefill otherwise adds hundreds of milliseconds to every concurrent request's ITL.

### 5.18 — Disaggregated prefill and decode
- **Question it answers:** If the two phases are bound by different resources, why run them on the same machine?
- **Prereqs:** 5.2, 5.12, 5.17.
- **Experiment:** Paper + spreadsheet. Given prefill-bound and decode-bound throughput numbers for a model, size two pools independently to meet a TTFT SLO and an ITL SLO, and compute the KV transfer cost: bytes to move = the per-token cache size from 5.12 times the prompt length. For Llama-3-8B at 128 KiB/token and a 4k prompt, that is 512 MiB per request — over 400Gb/s RDMA, ~10ms. Decide whether that is acceptable against the TTFT budget.
- **Shapes to nail:** the transferred object is, per layer, `(1, n_kv_heads, prompt_len, d_head)` x2 — and it is transferred *per layer*, which enables layer-wise overlap with the remaining prefill compute.
- **Predict-first prompt:** Disaggregation adds a network hop to every request. Under what workload (prompt length, output length, arrival rate) does it still win? State the condition before reading about DistServe or Splitwise.
- **Runs on:** **needs CUDA (and multiple GPUs)** to run for real — the KV transfer path is RDMA/NVLink via NIXL or equivalent, and both DistServe/Splitwise-style research systems and NVIDIA Dynamo assume it. The sizing exercise, the transfer arithmetic and the SLO argument are the parts you own, and they are device-independent.
- **Interview hooks:** This is a read/write split with an interconnect cost — what is the analogue in a database or microservice architecture you have built? Why does disaggregation let you use *different* parallelism strategies per pool? What happens to prefix caching when prefill and decode are on different machines? When is colocation with chunked prefill simply better?
- **Common misconception:** That disaggregation is strictly better. It adds a network dependency and a failure mode to every request; below a certain scale, chunked prefill on one pool is cheaper and simpler.

---

## 5E — Making decode faster

### 5.19 — Speculative decoding and the acceptance-rate math
- **Question it answers:** How do you generate more than one token per forward pass of the big model without changing the output distribution — and when does it actually pay?
- **Prereqs:** 5.2 (decode is bandwidth-bound, so the big model's forward is nearly free at small batch), 5.3, 5.10.
- **Experiment:** Two parts. (a) Implement the *math* first, on paper and in a 20-line simulation: draft `gamma` tokens, accept token `i` with probability `min(1, p_target(x)/q_draft(x))`, and on rejection sample from the residual `norm(max(0, p - q))` plus one bonus token. Verify empirically that the output distribution matches plain sampling from `p`. (b) Plot expected tokens per step `E = (1 - alpha^(gamma+1)) / (1 - alpha)` for `alpha` in 0.5..0.9 and `gamma` in 1..8, then overlay the *actual* speedup `E / (1 + gamma * c)` where `c` is the draft/target cost ratio, and find where it goes below 1.
- **Shapes to nail:** the verification pass is a single target forward on `(B, gamma+1)` — that is the whole trick: `gamma+1` positions cost roughly the same wall-clock as 1 in a bandwidth-bound decode. Draft logits `(B, gamma, V)`, target logits `(B, gamma+1, V)`.
- **Predict-first prompt:** `alpha = 0.8`, `gamma = 4`, draft costs 10% of target per call. Compute `E` and the net speedup before plotting. Then: at what `alpha` does `gamma=4` become *slower* than no speculation?
- **Runs on:** MPS for the correctness/distribution simulation and the acceptance math — use a tiny Phase 4 model as target and an even tinier one as draft. Measuring a *real* wall-clock speedup **needs CUDA**: the win depends on a bandwidth-bound decode with tree/batched verification kernels, and the MPS timing will not reproduce the published numbers. Use a Colab T4 with vLLM's `speculative_config` (n-gram or draft-model) for real numbers.
- **Interview hooks:** Prove that rejection sampling with the residual preserves the target distribution exactly — why is this "lossless"? Why does speculative decoding's benefit *shrink* as batch size grows? Medusa (extra heads, ~60-80% acceptance) vs EAGLE-3 (feature-level autoregression, ~70-85%+) vs n-gram/prompt-lookup — what does each trade? When is a draft *model* better than extra heads?
- **Common misconception:** That speculative decoding trades quality for speed. With the standard acceptance/residual rule the output distribution is *identical* to sampling from the target; what you trade is extra FLOPs and memory for the draft, which is only free while you are bandwidth-bound.

### 5.20 — Structured and constrained decoding
- **Question it answers:** How do you guarantee valid JSON, and what does that guarantee cost per token?
- **Prereqs:** 5.5 (logit masking), 5.6.
- **Experiment:** Build the mechanism by hand, small: a 3-state FSM accepting `{"a": <digits>}` over a char vocab, producing a boolean mask `(V,)` per state; apply `logits[~mask] = -inf` before sampling; generate and confirm every output parses. Then time the mask construction per token and compare naive per-token regex re-evaluation vs a precompiled state -> mask table.
- **Shapes to nail:** mask `(B, V)` boolean, one row per sequence because each sequence is in its own FSM state; applied to logits *after* penalties and *before* temperature/truncation. The precomputed table is `(n_states, V)` bits — note its size at `V=128k`, which is exactly why compilation time and memory are the real engineering problem.
- **Predict-first prompt:** The mask is a `(B, V)` boolean applied per token. Is the cost of constrained decoding dominated by applying the mask or by computing it? Predict, then measure both in your toy.
- **Runs on:** CPU/MPS — the FSM, the mask and the logit application are all device-independent and fully teachable here. Production libraries (XGrammar, the default structured-output backend in vLLM/SGLang/TensorRT-LLM; Outlines; llguidance) are worth reading; running them *in a server* needs CUDA but their mask-compilation step does not.
- **Interview hooks:** JSON-schema -> regex -> FSM -> per-state token mask: where does each translation lose expressiveness? Why is a context-free grammar harder than a regex here (pushdown state, so the mask depends on a stack, not a finite state)? XGrammar reports sub-40us per-token mask computation and ~0.1ms end-to-end overhead by splitting context-independent from context-dependent tokens — why is that split the key idea? What does constrained decoding do to *model quality*, not just latency — why can forcing a token the model assigned low mass to derail the rest of the generation? How does constrained decoding interact with speculative decoding and with prefix caching?
- **Common misconception:** That constrained decoding makes the model *understand* the schema. It only makes invalid tokens unsamplable; the model can still emit schema-valid nonsense, and a badly-chosen grammar can actively degrade output.

---

## 5F — Real systems and the latency budget

### 5.21 — What vLLM and SGLang actually do: an architecture read
- **Question it answers:** Can I read a production inference engine and name which of my lessons each subsystem implements?
- **Prereqs:** 5.14-5.20.
- **Experiment:** Reading, with a written map as the deliverable. In vLLM: the scheduler (running/waiting queues, token budget, preemption), the block manager (5.15), the prefix cache (5.16), the model runner and the attention backend dispatch, and the V1 engine-core/API-server process split. In SGLang: RadixAttention's prefix tree (5.16), the prefill-first vs vLLM-v1's decode-first scheduling bias, and the structured-output integration (5.20). Produce a one-page table: subsystem -> lesson -> the one design decision it encodes.
- **Shapes to nail:** find in the code where the flattened variable-length batch is constructed (the `cu_seqlens`/slot-mapping layout), and where the block table is passed into the attention kernel. Those two data structures are the whole serving architecture in concrete form.
- **Predict-first prompt:** Before reading: list the components you expect a serving engine to have, given lessons 5.14-5.20. Then check your list against the real directory structure and note what you missed.
- **Runs on:** CPU for the code read. **Actually running vLLM or SGLang needs CUDA** — both target NVIDIA (with AMD/TPU variants); there is no usable MPS backend, and vLLM's CPU backend is a compatibility path, not a way to observe the behaviours above. Use a Colab T4 (fits a 1-3B model in fp16), or rent an L4/A10 on modal/runpod for an hour, and drive it with the benchmark scripts.
- **Interview hooks:** Where does vLLM decide to preempt, and recompute-vs-swap on what basis? Why does the API server run in a separate process from the engine core in vLLM V1? Compare the two schedulers' bias (SGLang prefill-first, vLLM V1 decode-first) — what SLO does each favour? What would you change to serve a 90%-shared-prefix agentic workload?
- **Common misconception:** That these engines are "fast because of CUDA kernels". The kernels matter, but the 3-5x over a naive loop comes from the *scheduler and memory manager* — which is exactly the part your backend background makes you good at.

### 5.22 — The latency budget: TTFT, ITL, TPOT, end-to-end, p50 vs p99
- **Question it answers:** Given an SLO, what batch size and what max context are you allowed to run?
- **Prereqs:** 5.2, 5.12, 5.14.
- **Experiment:** Build the budget as a spreadsheet/notebook model, then validate the *shape* against measurements from your own toy server. Define: `TTFT = queue_wait + prefill_time`; `ITL` = per-output-token gap in decode; `TPOT` = mean ITL; `E2E = TTFT + (n_out - 1) * TPOT`. Sweep batch size 1..64 and plot (a) throughput tok/s and (b) p50/p99 ITL on the same x-axis. Find the batch size where throughput saturates and note what p99 is doing there.
- **Shapes to nail:** not tensors — the four numbers and their composition. Be able to write E2E for a 2000-token prompt and 500-token output given a TTFT and a TPOT, without looking it up.
- **Predict-first prompt:** Sketch throughput and p99 ITL against batch size before you plot. Where do the two curves stop agreeing, and what is the name of that region?
- **Runs on:** MPS for the sweep shape (the throughput/latency trade is visible on M3); representative absolute numbers need CUDA + a real engine (5.21).
- **Interview hooks:** A chat SLO of "TTFT < 500ms p95, ITL < 50ms p95" — what does each constrain (TTFT -> prefill/queue/chunking, ITL -> batch size and context length)? Why is p99 ITL dominated by prefill interference rather than by decode itself? Why does a streaming UI care about ITL while a batch summarisation job cares only about throughput, and how do you serve both from one fleet? What does `n_out` uncertainty do to your E2E percentile estimate?
- **Common misconception:** That you optimise "latency". You optimise a *specific* percentile of a *specific* phase; TTFT and ITL respond to opposite knobs, and improving one usually costs the other.

### 5.23 — Queueing, admission control, backpressure and SLOs
- **Question it answers:** How do I run this as a service that degrades predictably instead of collapsing?
- **Prereqs:** 5.22, 5.14, 5.15.
- **Experiment:** Extend the toy scheduler from 5.14 into a small simulator: Poisson arrivals, sampled prompt/output lengths, a KV-memory admission check using the 5.12 formula, a bounded queue with a reject path, and a per-class (interactive vs batch) priority. Sweep arrival rate through saturation and plot p50/p99 TTFT and E2E, throughput, and rejection rate. Add a deadline-based drop and show the p99 cliff flatten.
- **Shapes to nail:** the admission predicate — `free_blocks >= blocks_needed_for(prompt_len) + headroom_for_growth` — and the fact that `blocks_needed` *grows* during generation, so admission is a bet, not a reservation. That is the whole difference from classical admission control.
- **Predict-first prompt:** Arrival rate crosses service rate. With an unbounded queue, what happens to p99 TTFT over 10 minutes? What does the *throughput* curve do at the same time? Answer both before simulating.
- **Runs on:** CPU (simulation). The policies transfer directly to a CUDA deployment.
- **Interview hooks:** Little's Law applied to an LLM server — what is `L`, what is `W`, and what is unusual about the service-time distribution? Why is "just increase max batch size" a latency bug disguised as a throughput fix? What is the LLM equivalent of load shedding, and what do you shed first? How do you preempt a half-generated request, and what is the recompute-vs-swap cost of doing so (tie back to 5.15)? How do you set a per-tenant rate limit when the cost of a request is unknown until it finishes?
- **Common misconception:** That an LLM server is a special case needing new theory. It is a queueing system with (a) an unknown, heavy-tailed service time and (b) a hard memory constraint that grows during service. Your existing instincts apply; only those two properties are new.

---

## Milestone (Phase 5)

`projects/mini-serve/` — a single-GPU-shaped inference server built on your own
Phase 4 model, containing:

1. A KV-cached generation path, verified against the naive path for identical
   greedy output, with a measured speedup curve vs sequence length.
2. All six sampling strategies implemented from scratch, with a test that each
   reduces to the expected degenerate case (`temperature -> 0` = greedy,
   `top_k = V` and `top_p = 1.0` = plain sampling).
3. A continuous-batching scheduler with a paged (block-table) KV allocator,
   prefix-cache block sharing, and a chunked-prefill token budget.
4. `BENCHMARK.md`: a throughput-vs-p99-ITL curve over batch size, a KV-memory
   budget table computed from the 5.12 formula for one real model, an
   acceptance-rate/speedup analysis for speculative decoding, and an explicit
   SLO statement with the max batch size it justifies.
5. `CUDA-GAP.md`: the four things you could not measure on M3 (real
   PagedAttention kernel, vLLM/SGLang end-to-end, speculative-decoding wall
   clock, disaggregated KV transfer), what you did instead, and the numbers you
   collected on a rented GPU to close each gap.
