# Phase 0 + Phase 1 — Foundations

Phase 0: workspace, the minimum math, and why floats lie.
Phase 1: tensors, autograd, and a neural net built from nothing.

This is the bridge from "I know Python" to "I can read PyTorch model code."
Everything in Phases 2–9 stands on this. Do not speed-run it.

Prereq from prior work: `01_embeddings/` (token IDs, `nn.Embedding`, dot product
vs matmul). Phase 0 deliberately does **not** re-teach those.

Lesson dirs created by this phase: `00_setup_math/`, `01b_tensors/`,
`02b_autograd/`, `03_nn_from_scratch/`. (`01_embeddings/` and `02_attention/`
already exist and keep their numbers; Phase 1 dirs are suffixed rather than
renumbering work already done.)

---

## What you can claim after this phase

- I can read an arbitrary PyTorch `forward()` and state the shape of every
  intermediate tensor without running it, including where broadcasting silently
  changes rank.
- I can derive the backward pass of a 2-layer MLP on paper, implement it with no
  autograd, and show it matches `.backward()` to within float32 tolerance.
- I can explain what `.backward()` actually computes (a vector-Jacobian product,
  not "the derivative"), why `.grad` accumulates, and what `torch.no_grad()`
  turns off.
- I can derive cross-entropy from softmax + negative log-likelihood, explain why
  models emit logits rather than probabilities, and show the log-sum-exp trick
  preventing an overflow I produced myself.
- I can state the optimizer-state memory cost of SGD vs momentum vs Adam in
  bytes-per-parameter, and use that to size a training run.
- I can diagnose a non-learning model with a fixed checklist: overfit one batch,
  check gradient norms, check init scale, check the device, check the shapes.

---

## Phase 0 — Workspace and math warm-up

Goal: remove every excuse for being lost later. Each math item is motivated by a
specific shape or gradient you will actually meet in Phase 2+.

### 00.1 — Environment and kernel sanity check
- **Question it answers:** Am I certain the Python running my notebook cells is
  the same interpreter as `myenv/bin/python`, with a working GPU backend?
- **Prereqs:** none.
- **Experiment:** in a fresh notebook print `sys.executable`,
  `torch.__version__`, `torch.backends.mps.is_available()`,
  `torch.backends.mps.is_built()`. Then move a `(2, 3)` tensor to `mps` and back,
  and confirm `sys.executable` equals the venv path exactly. Register the kernel
  under a named spec if the listed kernels do not point at `myenv`.
- **Shapes to nail:** `(2, 3)` — trivial on purpose; this lesson is about the
  interpreter, not the tensor.
- **Predict-first prompt:** If `pip install` in a terminal and `import` in a
  notebook disagree about whether a package exists, which one is wrong, and what
  single value would you print to settle it?
- **Runs on:** CPU + MPS.
- **Interview hooks:** What is a Jupyter kernelspec and how does it resolve an
  interpreter? Why does `!pip install` inside a notebook frequently install into
  the wrong environment? How would you make an ML repo's environment
  reproducible for a teammate?
- **Common misconception:** That "the venv is activated" means the notebook uses
  it. The kernelspec's `argv[0]` decides, and it is set at registration time.

### 00.2 — Devices: what MPS is, and how it differs from CUDA
- **Question it answers:** What actually happens when I write `.to("mps")`, and
  which CUDA-world assumptions do not transfer?
- **Prereqs:** 00.1.
- **Experiment:** write `common/device.py` with a `get_device()` helper
  (mps → cuda → cpu). Time a `(2048, 2048) @ (2048, 2048)` matmul on CPU vs MPS,
  calling `torch.mps.synchronize()` before reading the clock. Then attempt
  `torch.ones(3, dtype=torch.float64, device="mps")` and read the error.
- **Shapes to nail:** `(2048, 2048) @ (2048, 2048) -> (2048, 2048)`; ~2·2048³ ≈
  1.7e10 FLOPs — the number you divide wall time by to get FLOP/s.
- **Predict-first prompt:** Without `synchronize()`, will the MPS timing look
  faster or slower than reality, and why? (Hint: who is allowed to return first,
  the Python call or the GPU?)
- **Runs on:** MPS (that is the point).
- **Interview hooks:** What is a unified memory architecture and what does it
  eliminate versus a discrete GPU? Why is kernel launch asynchronous, and what
  breaks in a benchmark that ignores it? Name three things that require NVIDIA
  hardware specifically and say why (FlashAttention kernels, bitsandbytes 4-bit,
  Triton/vLLM, FP8, NCCL multi-GPU).
- **Common misconception:** "MPS is just a slower CUDA." It is a different
  backend with a different op coverage surface: unsupported ops fall back to CPU
  (silently, unless you set `PYTORCH_ENABLE_MPS_FALLBACK=1` and watch for it),
  `float64` is unsupported outright, and numerics differ from CUDA in the last
  bits.

### 00.3 — Vectors: norm, projection, cosine
- **Question it answers:** What does a dot product mean *geometrically*, and why
  is that the operation attention uses to score relevance?
- **Prereqs:** dot product mechanics (already covered in `01_embeddings/`).
- **Experiment:** three 4-d vectors by hand. Compute `‖v‖` via `torch.linalg.norm`
  and by hand as `sqrt(v @ v)`. Compute cosine similarity between a vector, its
  own 3× scaling, and a near-orthogonal vector. Confirm scaling changes the dot
  product but not the cosine.
- **Shapes to nail:** `(4,) · (4,) -> ()` — a 0-d tensor, not a Python float.
  `torch.Size([])` is a shape you must be comfortable reading.
- **Predict-first prompt:** If I double the length of one vector, what happens to
  the dot product? To the cosine similarity? Which of those two does raw
  attention scoring use, and what problem does that create?
- **Runs on:** CPU.
- **Interview hooks:** Why does attention score with a dot product rather than
  cosine similarity? Why divide attention scores by `sqrt(d_k)` — what is the
  variance argument? Why are embedding vectors usually compared by cosine in a
  retrieval index but by dot product inside the model?
- **Common misconception:** That "dot product = similarity." It is similarity
  *scaled by both magnitudes*, which is exactly why unnormalised scores grow with
  dimension and need the `1/sqrt(d_k)` correction.

### 00.4 — Matmul as composed dot products, and shape algebra
- **Question it answers:** Given `(a, b) @ (c, d)`, when is it legal and what
  comes out — answered in one second, without running it?
- **Prereqs:** 00.3.
- **Experiment:** build `A` shape `(2, 3)` and `B` shape `(3, 4)` with known
  integer entries. Compute `A @ B` two ways: `torch.matmul`, and a double Python
  loop filling `C[i, j] = A[i] @ B[:, j]`. Assert they match. Then deliberately
  do `B @ A` and read the error.
- **Shapes to nail:** `(2, 3) @ (3, 4) -> (2, 4)`. The inner dims must agree and
  they *vanish*; the outer dims survive. Cost: `2 * 2 * 3 * 4` FLOPs.
- **Predict-first prompt:** In `(B=2, T=4, d_model=8) @ (8, 16)`, what is the
  output shape, and which dimension disappeared?
- **Runs on:** CPU.
- **Interview hooks:** What is the FLOP count of an `(M, K) @ (K, N)` matmul, and
  why is it 2·M·K·N rather than M·K·N? For a Transformer layer with `d_model=d`,
  which matmuls dominate at small sequence length vs long sequence length? Why is
  matmul the operation hardware vendors optimise above all others?
- **Common misconception:** That the last dim of the output "comes from" the input.
  It comes from the *weight*. `x @ W` where `W` is `(d_in, d_out)` maps feature
  count `d_in -> d_out` and leaves every leading (batch/time) dim untouched.

### 00.5 — Transpose, and why `W^T` is everywhere
- **Question it answers:** Why does the backward pass of a linear layer contain a
  transposed weight, and why does attention compute `Q @ K^T`?
- **Prereqs:** 00.4.
- **Experiment:** `A` shape `(2, 3)`. Verify `(A @ B)^T == B^T @ A^T`
  numerically. Then with `Q` and `K` both `(T=4, d_k=8)`, compute `Q @ K.T` and
  point at the entry that is "token 2 querying token 0".
- **Shapes to nail:** `(4, 8) @ (8, 4) -> (4, 4)` — the `T × T` score matrix, the
  single most important shape in the whole curriculum. Memorise that its size is
  `O(T²)` and independent of `d_k`.
- **Predict-first prompt:** `Q` is `(4, 8)`, `K` is `(4, 8)`. `Q @ K` errors.
  What is the error, and what does the transpose fix mean *semantically* — what
  is row `i`, column `j` of the result?
- **Runs on:** CPU.
- **Interview hooks:** Why is the attention score matrix the memory bottleneck at
  long context, and what is its size in bytes for `B=8, n_heads=32, T=8192,
  fp16`? What does FlashAttention change about that? Why does `(AB)^T = B^T A^T`
  show up in every backward-pass derivation?
- **Common misconception:** That transpose copies data. `.T` / `.transpose()`
  returns a *view* with permuted strides and no data movement — which is exactly
  why it can then make `.view()` fail (see 01.5).

### 00.6 — Derivatives and partial derivatives, rebuilt from scratch
- **Question it answers:** What number is `∂L/∂w` actually telling me about my
  weight, operationally?
- **Prereqs:** none beyond high-school algebra; deliberately assumes you have not
  done this in years.
- **Experiment:** `f(x) = x²`. Compute the finite-difference slope
  `(f(x+h) - f(x)) / h` at `x = 3.0` for `h = 1e-1, 1e-3, 1e-5, 1e-7, 1e-9` in
  float32 and watch it approach 6 and then get *worse*. Then
  `g(w, b) = (w*2 + b - 5)²` and compute `∂g/∂w` and `∂g/∂b` by finite difference,
  holding the other fixed.
- **Shapes to nail:** scalar in, scalar out. `∂g/∂w` is a scalar per parameter —
  so a parameter tensor of shape `(3, 4)` has a gradient of shape `(3, 4)`.
  Gradients always have the shape of the thing they differentiate.
- **Predict-first prompt:** As `h` shrinks toward 1e-9, does the finite-difference
  estimate get monotonically more accurate? Predict yes or no before running, and
  commit to a reason.
- **Runs on:** CPU.
- **Interview hooks:** Why is finite-difference gradient checking done in float64
  and not float32? What is the optimal `h` and why does it scale like
  `sqrt(machine_eps)`? If a gradient check fails by 1e-3 relative error, is that
  a bug or a tolerance problem — how would you decide?
- **Common misconception:** That smaller `h` is always better. Truncation error
  falls with `h` while floating-point cancellation error grows like `eps/h`; the
  total is U-shaped. This is the first payoff of 00.8, deliberately previewed here.

### 00.7 — Chain rule and the gradient vector
- **Question it answers:** How does an error measured at the output become an
  update to a weight five layers back?
- **Prereqs:** 00.6.
- **Experiment:** `L = (y - t)²` where `y = w*x + b`, `x = 2, t = 5`. Derive
  `dL/dw = 2(y - t) * x` on paper, then confirm by finite difference. Then chain
  one more level: `y = w2 * h + b2`, `h = w1 * x + b1`, and derive `dL/dw1` by
  hand as a product of three local derivatives.
- **Shapes to nail:** for `L` scalar and `W` of shape `(d_in, d_out)`,
  `∂L/∂W` has shape `(d_in, d_out)`. Scalar loss is what makes this a *gradient*
  and not a full Jacobian — hold onto that; it is why `.backward()` needs no
  argument on a scalar.
- **Predict-first prompt:** In the two-level chain, `dL/dw1` is a product of how
  many factors, and which single factor would make the whole gradient zero if it
  were zero? (This is the vanishing-gradient story arriving early.)
- **Runs on:** CPU.
- **Interview hooks:** Why does backprop cost roughly 2× the forward pass rather
  than N× the number of parameters? Why is the gradient the direction of steepest
  *ascent*, and what does that imply about the sign in the update rule? What is
  the difference between a gradient and a Jacobian, and which one does a scalar
  loss give you?
- **Common misconception:** That backprop is "the chain rule applied
  symbolically." It is the chain rule applied *numerically, in reverse order,
  reusing cached forward values*. The reuse is the entire algorithmic
  contribution; the calculus is high-school.

### 00.8 — Floats are not reals: float32 layout and catastrophic cancellation
- **Question it answers:** Why did my mathematically-correct implementation
  produce `nan`, `inf`, or `0.0`?
- **Prereqs:** 00.6.
- **Experiment:** print `torch.finfo(torch.float32)` and `torch.finfo(torch.bfloat16)`
  side by side (note `eps` 1.19e-7 vs 7.8e-3, same exponent range). Then three
  shocks: (a) `torch.tensor(1e8) + torch.tensor(1.0) == torch.tensor(1e8)` is
  `True`; (b) `torch.tensor([1e8, 1.0, -1e8]).sum()` gives `0.0`, losing the 1.0
  entirely; (c) `torch.exp(torch.tensor([1000., 1001., 1002.]))` is all `inf`
  while `torch.softmax` on the same input returns the correct
  `[0.0900, 0.2447, 0.6652]`.
- **Shapes to nail:** none — this lesson is about the 32 bits inside one cell:
  1 sign, 8 exponent, 23 mantissa. bfloat16 keeps the 8 exponent bits and cuts
  the mantissa to 7; fp16 uses 5 exponent bits and 10 mantissa.
- **Predict-first prompt:** `sum([1e8, 1.0, -1e8])` — predict the float32 result
  before running. Then predict `sum([-1e8, 1e8, 1.0])`. If they differ, what does
  that tell you about float addition as an algebraic operation?
- **Runs on:** CPU (also do it on MPS to see the numerics are not bit-identical).
- **Interview hooks:** Why is bfloat16 preferred to float16 for training LLMs
  despite having *fewer* mantissa bits? What is loss scaling and which of the two
  formats needs it? Why must layernorm and softmax accumulate in fp32 even in a
  mixed-precision model? Why is float addition non-associative, and what does
  that do to reproducibility across batch sizes or GPU counts?
- **Common misconception:** That fp16 and bf16 differ mainly in precision. The
  decisive difference is *dynamic range* — bf16 shares fp32's exponent, so
  activations and gradients that overflow in fp16 are fine in bf16. Precision is
  the price paid for that.

### 00.9 — Mean, variance, and why random sums grow like `sqrt(n)`
- **Question it answers:** A dot product of two *random* vectors gets bigger as
  the vectors get longer. By exactly how much — and why is the answer `sqrt(n)`
  rather than `n`?
- **Prereqs:** 00.3. No statistics assumed; this lesson supplies it.
- **Why it exists:** `03.2` (the `sqrt(d_k)` scaling), `03.15` (RMSNorm) and
  `01.20` (initialization) all list "variance of a sum of independent variables"
  as a prerequisite, and nothing in the curriculum taught it. This is that lesson.
  Schedule it immediately before resuming `02_attention/`, so it is fresh.
- **Experiment:** (a) sample 10000 pairs `x, y ~ N(0,1)`; measure `mean(x*y)`
  and `var(x*y)` — find ≈0 and ≈1. (b) Now sum `n` such products for
  `n ∈ {4, 64, 1024}`; measure the variance of the sum and find it ≈ `n`, so the
  standard deviation is ≈ `sqrt(n)`. Print a table of `n`, measured variance,
  measured std, `sqrt(n)`. (c) The contrast that makes it click: sum `n` copies
  of the *same* number instead of `n` independent draws — that grows like `n`.
  Cancellation between mixed signs is the entire difference.
- **Shapes to nail:** samples `(N, n)`; rowwise sum → `(N,)`; `var ≈ n`;
  `std ≈ sqrt(n)`. Dividing the sum by `sqrt(n)` returns the variance to ≈1.
- **Predict-first prompt:** `Var(X) = 1`, `Var(Y) = 1`, independent. What is
  `Var(X + Y)`? Then `Var` of a sum of `n` of them? Then the *standard
  deviation* of that sum? Say all three before running anything.
- **Runs on:** CPU.
- **Interview hooks:** Derive `Var(q·k) = d_k` from this. Why does the spread
  grow like `sqrt(n)` and not `n`? Where else does `sqrt(n)` appear for exactly
  this reason — initialization scaling, standard error, random walks? What
  breaks in the argument if the components are *correlated* rather than
  independent, and is that realistic inside a trained Transformer?
- **Common misconception:** That adding more terms makes a sum grow in
  proportion to `n`. For independent mixed-sign terms the *mean* stays put and
  only the *spread* grows — and it grows like `sqrt(n)`. This is the single
  fact the whole `sqrt(d_k)` argument rests on.

---

## Phase 1 — Tensors, autograd, and a neural network from scratch

### 01.1 — Tensor creation, dtype, device
- **Question it answers:** What are the three pieces of metadata I must check
  before trusting any tensor?
- **Prereqs:** 00.1, 00.2, 00.8.
- **Experiment:** create the same logical data five ways — `torch.tensor`,
  `torch.zeros`, `torch.arange`, `torch.randn`, `torch.from_numpy` — and print
  `.shape`, `.dtype`, `.device` for each. Find the two that give `int64` and the
  one that shares memory with numpy (mutate the numpy array and watch the tensor
  change). Write `common/shapes.py` with a `describe(t, name)` helper printing
  all three plus `.requires_grad`.
- **Shapes to nail:** `torch.Size([])` (0-d) vs `torch.Size([1])` (1-d, one
  element) vs `torch.Size([1, 1])`. These are three different things and the
  difference will bite you in broadcasting.
- **Predict-first prompt:** `torch.tensor([1, 2, 3])` and
  `torch.tensor([1.0, 2, 3])` — same dtype or different? What does that do to a
  later division?
- **Runs on:** CPU + MPS.
- **Interview hooks:** Why do embedding indices have to be `int64`? What is the
  cost of a host-to-device copy and why does calling `.item()` in a training loop
  destroy throughput? When does `torch.from_numpy` share memory and when does it
  copy?
- **Common misconception:** That dtype promotion "just works." Integer tensors
  are real, `torch.tensor([1,2,3]) / 2` promotes but `//` does not, and an
  accidental `int64` parameter tensor will refuse to take a gradient.

### 01.2 — Indexing and slicing
- **Question it answers:** Which indexing operations return a view of the same
  storage and which return a copy?
- **Prereqs:** 01.1.
- **Experiment:** `x = torch.arange(24).reshape(2, 3, 4)`. Do basic slicing
  `x[0]`, `x[:, 1]`, `x[..., -1]`, `x[:, :, 1:3]`; then integer-array indexing
  `x[[0, 1], [0, 2]]`; then boolean masking `x[x > 10]`. For each, mutate the
  result in place and check whether `x` changed. Then the one that matters most:
  `x[:, -1, :]` vs `x[:, -1:, :]`.
- **Shapes to nail:** `(2, 3, 4)`: `x[0] -> (3, 4)`, `x[:, 1] -> (2, 4)`,
  `x[:, -1, :] -> (2, 4)` (rank drops), `x[:, -1:, :] -> (2, 1, 4)` (rank kept),
  `x[x > 10] -> (13,)` (always flattened).
- **Predict-first prompt:** During generation you want the last time step's
  logits from a `(B=2, T=4, V=50)` tensor to feed into softmax. Write the index
  expression and predict the shape. Which of the two last-step forms do you want,
  and what breaks if you pick the other?
- **Runs on:** CPU.
- **Interview hooks:** Why does boolean-mask indexing force a synchronisation and
  a copy on GPU? What is the difference between `index_select`, `gather`, and
  fancy indexing, and when does each win? Why does in-place mutation of a view
  sometimes raise an autograd error?
- **Common misconception:** That slicing copies, as `list[:]` does in Python.
  Basic slicing gives a *view*; advanced (integer-array / boolean) indexing gives
  a *copy*. In-place writes to the first silently change the parent.

### 01.3 — Broadcasting rules
- **Question it answers:** Given two shapes, what is the output shape — derived
  mechanically, right-to-left, without guessing?
- **Prereqs:** 01.1.
- **Experiment:** state the rule first (align trailing dims; each pair must be
  equal, or one of them 1; a missing leading dim is treated as 1). Then work
  eight pairs on paper *before* running: `(3,) + (3,)`; `(2,3) + (3,)`;
  `(2,3) + (2,1)`; `(2,3) + (3,1)`; `(2,1,4) + (3,4)`; `(2,3) + (2,)`;
  `(B=2,T=4,1) + (T=4,)`; `(4,1) + (1,4)`. Mark each legal/illegal and give the
  result shape. Score yourself. Anything below 8/8 means repeat the drill.
- **Shapes to nail:** `(4, 1) * (1, 4) -> (4, 4)` — the outer-product accident.
  `(2, 1, 4) + (3, 4) -> (2, 3, 4)` — rank grew. `(2, 3) + (3, 1)` and
  `(2, 3) + (2,)` both **error**, while `(2, 3) + (2, 1)` works: the fix for a
  per-row vector is always an explicit trailing `1`.
- **Predict-first prompt:** You have per-token scores `(B=2, T=4)` and a causal
  mask `(T=4, T=4)`. You write `scores + mask`. What shape comes out, does it
  error, and is it what you meant?
- **Runs on:** CPU.
- **Interview hooks:** Does broadcasting allocate the expanded tensor? What is the
  memory cost of `x.expand(...)` vs `x.repeat(...)`? Why does adding a
  `(T,)` positional-bias to a `(B, T)` tensor work while `(B,)` does not?
- **Common misconception:** That "compatible" means "same number of dimensions."
  Rank is padded on the *left* with 1s automatically; it is the *trailing*
  alignment that decides, which is why a `(B,)` per-example vector almost never
  broadcasts the way people expect and needs `(B, 1)`.

### 01.4 — The broadcasting bug drill
- **Question it answers:** How do I catch a broadcasting bug that does not raise
  an exception?
- **Prereqs:** 01.3.
- **Experiment:** deliberately write three bugs and measure the damage.
  (a) MSE with `pred` `(B=8, 1)` and `target` `(B=8,)` — produces an `(8, 8)`
  loss surface and a plausible-looking scalar that is wrong.
  (b) A per-sample weight `(B=8,)` multiplying a `(B=8, d=4)` activation.
  (c) Subtracting a mean without `keepdim=True`. For each, first confirm no
  exception, then find it two ways: an explicit `assert a.shape == b.shape`, and
  a rewrite in `einsum` where the bug becomes a shape error.
- **Shapes to nail:** `(8, 1) - (8,) -> (8, 8)`. Burn this in. It is the single
  most common silent bug in hand-written training code.
- **Predict-first prompt:** `((pred - target) ** 2).mean()` with `pred (8,1)` and
  `target (8,)` — will it raise? If not, is the number it returns larger or
  smaller than the correct MSE, and why?
- **Runs on:** CPU.
- **Interview hooks:** How would you defend a training pipeline against silent
  shape bugs — asserts, named tensors, `einops`, typed shape annotations? Which
  do you actually use in production and what is the runtime cost? Why does a
  broadcasting bug often show as "loss decreases but the model is useless"
  rather than as a crash?
- **Common misconception:** That an exception-free run means correct shapes.
  Broadcasting's job is to *avoid* raising. Silence is not evidence.

### 01.5 — reshape vs view vs permute vs transpose, and contiguity
- **Question it answers:** Why did `.view()` throw "view size is not compatible
  with input tensor's size and stride" right after a `.transpose()`?
- **Prereqs:** 01.2, 01.3.
- **Experiment:** `x = torch.randn(2, 3, 4)`. Print `x.stride()` → `(12, 4, 1)`.
  Take `t = x.transpose(1, 2)`; print `t.shape` `(2, 4, 3)` and `t.stride()`
  `(12, 1, 4)`; confirm `t.is_contiguous()` is `False`. Now `t.view(2, 12)`
  raises, `t.reshape(2, 12)` succeeds (by copying), and
  `t.contiguous().view(2, 12)` succeeds explicitly. Then the real use case: the
  multi-head split `(B=2, T=4, d_model=8) -> (2, 4, n_heads=2, d_head=4)
  -> transpose(1,2) -> (2, 2, 4, 4)`, and the reverse merge.
- **Shapes to nail:** `(B, T, d_model) -> view -> (B, T, H, d_head)
  -> transpose(1, 2) -> (B, H, T, d_head)`, and back:
  `-> transpose(1, 2) -> .contiguous() -> .view(B, T, d_model)`. You will write
  this exact sequence in Phase 2. Memorise why the `.contiguous()` is mandatory
  on the way back and forbidden-by-being-pointless on the way out.
- **Predict-first prompt:** After `transpose(1, 2)` on a `(2, 3, 4)` tensor, did
  any float move in memory? What changed, then? Predict the new stride tuple
  before printing it.
- **Runs on:** CPU.
- **Interview hooks:** What is a stride and how does PyTorch compute a flat offset
  from an index tuple? Why is `reshape` "view if possible, copy otherwise" and
  when is that a hidden allocation in a hot loop? Why does the head-merge in
  attention require a contiguous copy while the head-split does not?
- **Common misconception:** That `permute` or `transpose` reorders data. They
  reorder *strides*. The tensor is "wrong" in memory afterwards, which is exactly
  why `view` — which only reinterprets an existing contiguous layout — refuses.

### 01.6 — matmul and batched matmul semantics
- **Question it answers:** What does `torch.matmul` do for 1-D, 2-D, and >2-D
  inputs — the four separate rules?
- **Prereqs:** 00.4, 01.3, 01.5.
- **Experiment:** run all four cases and check the printed shape against your
  prediction: `(3,) @ (3, 4) -> (4,)`; `(3, 4) @ (4,) -> (3,)`;
  `(2, 3, 4) @ (4, 5) -> (2, 3, 5)`; and the broadcasting case
  `(2, 1, 3, 4) @ (5, 4, 6) -> (2, 5, 3, 6)`. Then the one you actually need:
  `Q (B=2, H=2, T=4, d_head=8) @ K.transpose(-2, -1) (2, 2, 8, 4) -> (2, 2, 4, 4)`.
  Compare `@`, `torch.matmul`, `torch.bmm`, and `torch.mm` and note which accept
  batch dims.
- **Shapes to nail:** `(B, H, T, d_head) @ (B, H, d_head, T) -> (B, H, T, T)`.
  FLOPs: `2 * B * H * T * T * d_head`. Note the `T²` — this is why context length
  is expensive.
- **Predict-first prompt:** For `(2, 1, 3, 4) @ (5, 4, 6)`, which dims are
  "batch" and which are "matrix"? Predict the output shape and name the rule that
  produced the `5`.
- **Runs on:** CPU + MPS.
- **Interview hooks:** Why is `torch.bmm` restricted to exactly 3-D while
  `matmul` is not? Given `d_model=4096`, `T=2048`, `B=1`, compute the FLOPs of
  one attention score matmul vs one MLP matmul — which dominates, and at what `T`
  does the crossover happen? Why can a batched matmul be more efficient than a
  loop of matmuls even at identical FLOP count?
- **Common misconception:** That `torch.matmul` is "just `@` for matrices." The
  leading dims are batch dims that *broadcast*, and the 1-D cases silently
  prepend/append a dimension and then remove it again — which is where a rank
  quietly changes under you.

### 01.7 — einsum as the shape-explicit alternative
- **Question it answers:** How do I write a contraction so that its intent is
  visible and a wrong shape becomes an error instead of a broadcast?
- **Prereqs:** 01.6.
- **Experiment:** rewrite four things you already wrote: dot product `"i,i->"`,
  matmul `"ij,jk->ik"`, batched attention scores
  `"bhtd,bhsd->bhts"`, and a weighted sum over values
  `"bhts,bhsd->bhtd"`. Verify each matches the `@` version with
  `torch.allclose`. Then break one index letter on purpose and read the error.
- **Shapes to nail:** `einsum("bhtd,bhsd->bhts", Q, K)` with
  `Q (2, 2, 4, 8)`, `K (2, 2, 4, 8)` → `(2, 2, 4, 4)`. The repeated `d` is the
  contracted (summed) axis; every letter in the output survives.
- **Predict-first prompt:** In `"bhtd,bhsd->bhts"`, which letter is summed over
  and how do you know from the notation alone? What would `"bhtd,bhsd->bhtsd"`
  compute, and what would it cost in memory?
- **Runs on:** CPU + MPS.
- **Interview hooks:** Is `einsum` slower than `matmul`, and under what
  conditions does `opt_einsum`-style contraction ordering matter? Why does
  `einops.rearrange` reduce bugs relative to `permute` + `view`? When is a
  handwritten `matmul` still preferable to `einsum` in production?
- **Common misconception:** That einsum is a performance tool. Its value here is
  that the shape contract is written down in the call; a typo'd index becomes a
  loud error rather than a silent broadcast.

### 01.8 — Reductions and the meaning of `dim=`
- **Question it answers:** When I pass `dim=1`, am I reducing *along* that axis or
  *keeping* it?
- **Prereqs:** 01.3.
- **Experiment:** `x = torch.arange(24).float().reshape(2, 3, 4)`. Compute
  `x.sum()`, `x.sum(dim=0)`, `x.sum(dim=1)`, `x.sum(dim=-1)`,
  `x.sum(dim=(0, 2))`, and each with `keepdim=True`. Predict every shape first.
  Then apply it: `softmax` over `dim=-1` of a `(B=2, H=2, T=4, T=4)` score matrix
  and verify each row sums to 1 — then do `dim=-2` and see the rows no longer sum
  to 1, which is the classic attention bug.
- **Shapes to nail:** `(2, 3, 4)`: `sum(dim=1) -> (2, 4)`;
  `sum(dim=1, keepdim=True) -> (2, 1, 4)`. Softmax on attention scores is always
  `dim=-1` — over *keys*, for a fixed query.
- **Predict-first prompt:** For an attention score matrix `(B, H, T_q, T_k)`,
  should softmax be over `dim=-1` or `dim=-2`? Justify it in terms of what must
  sum to 1, then verify numerically.
- **Runs on:** CPU.
- **Interview hooks:** Why does `keepdim=True` exist — give the layernorm example
  in shapes. What is the numerically stable way to compute a mean over a masked
  axis? Why do padding tokens corrupt a `mean(dim=1)` and how do you fix it?
- **Common misconception:** That `dim=1` means "operate on dimension 1 and keep
  it." The named dim is the one that is *consumed*. `keepdim=True` leaves a `1`
  in its place specifically so the result still broadcasts against the input.

### 01.9 — Backprop by hand on a 2-layer net (no autograd)
- **Question it answers:** Can I produce every gradient in a small network with
  nothing but the chain rule and matmuls?
- **Prereqs:** 00.6, 00.7, 01.6, 01.8.
- **Experiment:** `x (B=4, d_in=3) -> W1 (3, 5) -> h_pre -> tanh -> h (4, 5)
  -> W2 (5, 1) -> y_hat (4, 1)`, loss `MSE`. Derive on paper:
  `dL/dy_hat`, then `dL/dW2 = h^T @ dL/dy_hat`, `dL/dh = dL/dy_hat @ W2^T`,
  `dL/dh_pre = dL/dh * (1 - h²)`, `dL/dW1 = x^T @ dL/dh_pre`. Implement all of it
  with plain tensor ops and `requires_grad=False`. Validate each against a
  finite-difference estimate on a few entries (in float64, on CPU).
- **Shapes to nail:** every gradient matches its parameter: `dL/dW1` is `(3, 5)`,
  `dL/dW2` is `(5, 1)`, `dL/dh` is `(4, 5)`. Note the `x^T @ delta` pattern —
  `(3, 4) @ (4, 5) -> (3, 5)` — where the batch dim is what gets summed over.
- **Predict-first prompt:** `dL/dW1` must have shape `(3, 5)`. You have
  `x (4, 3)` and `dL/dh_pre (4, 5)`. There is exactly one way to combine them
  into `(3, 5)` — write it, and say which dimension the sum runs over and why
  summing over it is correct.
- **Runs on:** CPU (float64 finite-difference check cannot run on MPS —
  MPS has no `float64`; keep the gradient check on CPU).
- **Interview hooks:** Why does the batch dimension get summed in the weight
  gradient but averaged in the loss — and what does that do to the effective
  learning rate when you change batch size? Why does backprop need the forward
  activations retained, and what is the resulting activation-memory cost of a
  deep net? What is gradient checkpointing trading away?
- **Common misconception:** That backprop computes "the derivative of the network."
  It computes `∂(scalar loss)/∂(each parameter)` — one number per parameter — by
  propagating a single vector backwards. There is never a full Jacobian in memory.

### 01.10 — `requires_grad` and the computation graph
- **Question it answers:** What does PyTorch build while my forward pass runs, and
  how do I see it?
- **Prereqs:** 01.9.
- **Experiment:** `a = torch.tensor(2.0, requires_grad=True)`,
  `b = torch.tensor(3.0, requires_grad=True)`, `c = a * b`, `d = c + a`,
  `L = d ** 2`. Print `.requires_grad`, `.is_leaf`, and `.grad_fn` for every node.
  Walk `L.grad_fn.next_functions` by hand and draw the DAG. Then set
  `b.requires_grad = False` and see which `grad_fn`s disappear.
- **Shapes to nail:** all scalars here — `torch.Size([])`. The point is topology,
  not shape.
- **Predict-first prompt:** Which of `a, b, c, d, L` are leaves? Which will have a
  non-`None` `.grad` after `L.backward()`? Predict both lists before printing.
- **Runs on:** CPU.
- **Interview hooks:** Why is the autograd graph built dynamically per forward
  pass rather than compiled once, and what does `torch.compile` change? What is
  the memory held by the graph, and why does keeping a loss tensor in a Python
  list leak an entire graph? What does `retain_graph=True` actually retain?
- **Common misconception:** That the graph is the model. The graph is rebuilt on
  every forward pass and freed on every `.backward()`; `nn.Module` just holds the
  parameters.

### 01.11 — What `.backward()` actually computes
- **Question it answers:** Why can I call `.backward()` with no argument on a
  scalar but not on a vector?
- **Prereqs:** 01.9, 01.10.
- **Experiment:** call `.backward()` on the 01.9 network's loss and
  `torch.allclose` every autograd gradient against your hand-derived one — this
  is the payoff moment of the phase. Then: call `.backward()` twice without
  `zero_grad()` and watch `.grad` double. Then call `.backward()` on a
  non-scalar `(3,)` output and read the "grad can be implicitly created only for
  scalar outputs" error; fix it by passing `torch.ones(3)` and explain what that
  vector *is*.
- **Shapes to nail:** `param.grad.shape == param.shape`, always. The `v` in
  `y.backward(v)` has `y`'s shape — it is the left vector in `vᵀJ`.
- **Predict-first prompt:** After two consecutive `.backward()` calls with no
  `zero_grad()` in between, what is in `W1.grad` — the second gradient, the sum,
  or the mean? Why would the framework choose that default?
- **Runs on:** CPU + MPS (do the hand-vs-autograd comparison on CPU for clean
  float32 tolerances; `atol=1e-6` is reasonable).
- **Interview hooks:** What is a vector-Jacobian product and why is reverse-mode
  AD the right choice when `n_params >> n_outputs`? When would forward-mode AD
  win instead? Why does gradient accumulation across micro-batches work "for
  free" given this design, and what must you divide by?
- **Common misconception:** That `.grad` is overwritten each call. It *accumulates*
  — a deliberate design choice that makes gradient accumulation and multi-loss
  training trivial, and that makes a forgotten `zero_grad()` a silent bug where
  the model trains but badly.

### 01.12 — `no_grad`, `detach`, and leaf vs non-leaf
- **Question it answers:** How do I run a forward pass that costs no graph memory,
  and how do I cut a tensor out of the graph mid-computation?
- **Prereqs:** 01.10, 01.11.
- **Experiment:** measure the difference. Run a `(512, 512)` chain of 10 matmuls
  with and without `torch.no_grad()`, comparing `torch.mps.current_allocated_memory()`
  before/after. Then: `y = x * 2; z = y.detach() * 3` — call `.backward()` and
  see `x.grad` is `None`-or-unchanged. Then try to read `.grad` on a non-leaf and
  read the warning; fix with `.retain_grad()`.
- **Shapes to nail:** unchanged by any of this — `no_grad` and `detach` change
  graph membership, never shape or value.
- **Predict-first prompt:** Inside `torch.no_grad()`, does `requires_grad=True` on
  an input tensor still hold? Does the *output* have `requires_grad=True`?
  Predict both.
- **Runs on:** MPS (the memory delta is the lesson) + CPU.
- **Interview hooks:** What is the difference between `torch.no_grad()`,
  `torch.inference_mode()`, and `model.eval()` — name one thing each does that
  the others do not. Why is `model.eval()` alone insufficient for inference
  memory? Where does `.detach()` appear in a real training loop (logging, KV
  cache, target networks)?
- **Common misconception:** That `model.eval()` disables gradients. It only
  switches dropout and batchnorm to inference behaviour. Gradients keep being
  tracked and the graph keeps being built until you wrap it in `no_grad` /
  `inference_mode`.

### 01.13 — MSE, and what a loss function is for
- **Question it answers:** What property must a loss have for gradient descent to
  work at all?
- **Prereqs:** 01.11.
- **Experiment:** implement MSE three ways — by hand
  `((p - t) ** 2).mean()`, via `F.mse_loss`, and with the 01.4 broadcasting bug
  deliberately present. Compare all three numbers. Then plot `L(w)` for a
  one-parameter model over `w ∈ [-3, 3]` and mark the gradient's sign on either
  side of the minimum.
- **Shapes to nail:** `pred (B=8, 1)` and `target (B=8, 1)` → loss
  `torch.Size([])`. Check `reduction="none"` gives `(8, 1)`, `"sum"` gives `()`,
  `"mean"` gives `()` — and know which one `F.mse_loss` defaults to.
- **Predict-first prompt:** `reduction="sum"` vs `"mean"` on a batch of 8 — by
  what factor do the gradients differ, and what would you have to change about
  the learning rate to compensate?
- **Runs on:** CPU.
- **Interview hooks:** Why is MSE wrong for classification — give the gradient
  argument, not the "it's for regression" answer. What does MSE assume about the
  noise distribution? Why does `reduction="mean"` make the learning rate
  batch-size-independent, and why is that not quite true in practice?
- **Common misconception:** That the loss value is the thing that matters. The
  *gradient* of the loss is what trains the model; two losses with identical
  minima can have wildly different optimisation behaviour.

### 01.14 — Softmax → NLL → cross-entropy, derived
- **Question it answers:** Where does `-log(p_correct)` come from, and why is its
  gradient so clean?
- **Prereqs:** 01.8, 01.13.
- **Experiment:** logits `(B=4, V=5)`, targets `(4,)` of class indices. Build it in
  stages: `softmax -> probs (4, 5)`; gather the true-class probability with
  `probs.gather(1, targets.unsqueeze(1))` → `(4, 1)`; `-log` it; `.mean()`.
  Compare to `F.cross_entropy(logits, targets)` and to
  `F.nll_loss(F.log_softmax(logits, -1), targets)`. Then derive on paper that
  `∂L/∂logits = (softmax(logits) - onehot(target)) / B` and verify it against
  `logits.grad`.
- **Shapes to nail:** `F.cross_entropy` takes logits `(N, C)` and targets `(N,)`
  of **int64 class indices** — not one-hot, not probabilities. For an LM you must
  flatten: `logits (B, T, V) -> (B*T, V)` and `targets (B, T) -> (B*T,)`.
  Write that flatten out; you will use it in Phase 3.
- **Predict-first prompt:** `∂L/∂logits` for a single example with
  `softmax = [0.1, 0.7, 0.2]` and true class `0`. Write all three numbers before
  running.
- **Runs on:** CPU.
- **Interview hooks:** Why does the softmax+CE gradient simplify to `p - y`, and
  what would go wrong if you implemented softmax and NLL as separate autograd
  nodes in fp16? What is the cross-entropy of a uniform distribution over a
  50257-token vocab, and why is that the number you sanity-check step 0 against?
  How does label smoothing change the gradient?
- **Common misconception:** That cross-entropy needs probabilities as input.
  `F.cross_entropy` expects **raw logits** and applies `log_softmax` internally;
  passing it softmax output double-applies the nonlinearity and trains a subtly
  wrong model that still looks like it is learning.

### 01.15 — The log-sum-exp trick, and why logits not probabilities
- **Question it answers:** Why is `log(sum(exp(x)))` computed by subtracting the
  max first, and what exactly does that save?
- **Prereqs:** 00.8, 01.14.
- **Experiment:** `z = torch.tensor([1000., 1001., 1002.])`. Naive:
  `torch.log(torch.exp(z).sum())` → `inf`. Stable: subtract `z.max()`, exponentiate,
  sum, log, add the max back → `1002.4076`. Confirm `torch.logsumexp(z, -1)`
  agrees. Prove the identity algebraically first. Then the underflow direction:
  `z = torch.tensor([-1000., -1001.])`, where naive gives `-inf` and stable gives
  the right answer.
- **Shapes to nail:** `logsumexp(x, dim=-1)` on `(B, T, V)` → `(B, T)`; with
  `keepdim=True` → `(B, T, 1)`, which is the form that broadcasts back for
  `log_softmax = x - logsumexp(x, -1, keepdim=True)`.
- **Predict-first prompt:** Subtracting the max changes every input. Why does it
  not change the output? State the algebraic identity before you run anything.
- **Runs on:** CPU + MPS.
- **Interview hooks:** Where else does log-sum-exp appear (attention softmax,
  FlashAttention's online rescaling, beam-search scoring, mixture models)? Why
  does FlashAttention need a *running* max and a rescaling factor rather than a
  single pass? What precision must the softmax accumulator use under bf16
  training, and why?
- **Common misconception:** That the max-subtraction is an approximation or a
  numerical hack with a cost. It is exact: `logsumexp(z) = m + logsumexp(z - m)`
  for any `m`. Choosing `m = max(z)` merely guarantees the largest exponent is
  `exp(0) = 1`.

### 01.16 — A complete training loop, with manual updates
- **Question it answers:** What are the exact five steps of a training iteration,
  and what breaks if I reorder them?
- **Prereqs:** 01.11, 01.13.
- **Experiment:** synthetic data: `y = 3*x1 - 2*x2 + 0.5 + noise`, `N=256`,
  `d_in=2`. Model is the 01.9 two-layer net. Loop, with **no `torch.optim`**:
  forward → loss → `loss.backward()` → under `torch.no_grad()` do
  `p -= lr * p.grad` for each param → `p.grad.zero_()`. Run 200 steps, print loss
  every 20, recover the true coefficients. Then break it on purpose: remove
  `zero_()`; remove `no_grad()` around the update; move `backward()` after the
  update. Record the symptom of each.
- **Shapes to nail:** `x (256, 2)`, batch `x_b (32, 2)`, `y_hat (32, 1)`,
  `loss ()`, `W1.grad (2, 5)`. Confirm every `p.grad.shape == p.shape` before the
  first update.
- **Predict-first prompt:** What happens if you omit the `no_grad()` around
  `p -= lr * p.grad`? Predict the specific error or the specific wrong behaviour
  before running it.
- **Runs on:** CPU + MPS.
- **Interview hooks:** Why must `zero_grad()` come before `backward()` and not
  after the update — or does it matter? What does `set_to_none=True` change and
  why is it the default now? Where does a gradient-accumulation loop insert
  itself into these five steps?
- **Common misconception:** That `torch.optim` does something you could not
  write. SGD is literally `p -= lr * p.grad`. Writing it by hand once makes every
  later optimizer legible as "the same line with extra state."

### 01.17 — Overfit a single batch
- **Question it answers:** Before I debug anything else, how do I prove the
  model + loss + loop are capable of learning at all?
- **Prereqs:** 01.16.
- **Experiment:** take **one** batch of 8 examples, train on only that batch for
  500 steps, and drive the loss to ~0 (for CE, below 0.01). Then inject four
  bugs, one at a time, and confirm the test catches each: (a) a detached
  activation; (b) `lr = 0` on one parameter group; (c) targets shuffled relative
  to inputs; (d) the 01.4 broadcasting bug in the loss. Record which bugs the
  test catches and which it does not.
- **Shapes to nail:** `x (8, d_in)`, `y (8,)` held fixed across all 500 steps.
  No shuffling, no augmentation, no dropout.
- **Predict-first prompt:** If a model *cannot* drive a single batch to near-zero
  loss, list three possible causes ranked by how often they are the real one.
- **Runs on:** CPU + MPS.
- **Interview hooks:** What does a model that overfits one batch but not a small
  dataset tell you? What does one that cannot even overfit one batch tell you?
  Why is this the first test Karpathy's recipe prescribes, and what is the second?
  Which bug classes does this test *not* catch (data leakage, eval-mode bugs,
  tokenisation mismatches)?
- **Common misconception:** That overfitting is a failure state to avoid. Here it
  is the *goal*: it is a capacity-and-plumbing test. A model that cannot memorise
  8 examples has a bug, not a regularisation problem.

### 01.18 — SGD and momentum
- **Question it answers:** What state does an optimizer hold, and what problem
  does momentum solve that a larger learning rate does not?
- **Prereqs:** 01.16.
- **Experiment:** swap your manual loop for `torch.optim.SGD` and confirm
  identical numbers with `momentum=0`. Then implement momentum by hand
  (`v = mu*v + g; p -= lr*v`) and check it against `torch.optim.SGD(momentum=0.9)`
  — note PyTorch's default is *not* Nesterov and its formulation differs subtly
  from the classic one; read the docs' formula and match it exactly. Optimise a
  badly-conditioned quadratic (`L = 10*a² + 0.1*b²`) and plot the trajectory with
  and without momentum.
- **Shapes to nail:** momentum buffer has the same shape as the parameter — so
  momentum costs `+1x` parameter memory. For a 1B-param fp32 model: 4 GB params
  + 4 GB grads + 4 GB momentum.
- **Predict-first prompt:** On `L = 10a² + 0.1b²`, plain SGD with a single global
  learning rate must compromise. Which coordinate oscillates and which crawls,
  and what does that say about the largest usable learning rate?
- **Runs on:** CPU + MPS.
- **Interview hooks:** What is the condition number of a loss surface and how does
  it bound SGD's convergence rate? Why does momentum behave like an exponential
  moving average of gradients — what is the effective averaging window at
  `mu=0.9`? Why does `weight_decay` in `torch.optim.SGD` behave differently from
  L2 in the loss once momentum is on?
- **Common misconception:** That momentum is "just a bigger step." It is a
  low-pass filter on the gradient sequence: it cancels oscillation across the
  steep axis while accumulating along the consistent one.

### 01.19 — Adam, AdamW, and optimizer-state memory
- **Question it answers:** Why is AdamW the default for LLM training, and what
  does it cost me in GPU memory per parameter?
- **Prereqs:** 01.18.
- **Experiment:** implement Adam by hand on the same badly-conditioned quadratic:
  first moment `m`, second moment `v`, bias correction
  `m_hat = m/(1-b1^t)`, `v_hat = v/(1-b2^t)`, update
  `p -= lr * m_hat / (sqrt(v_hat) + eps)`. Match `torch.optim.Adam` to
  `1e-6`. Then show the AdamW difference concretely: `torch.optim.Adam` defaults
  to `weight_decay=0` and couples it into the gradient when set;
  `torch.optim.AdamW` defaults to `weight_decay=0.01` and applies it *decoupled*,
  directly to the parameter. Construct a case where the two give different
  updates. Finally: compute optimizer state in bytes for a 7B model and write it
  in `common/memory.py` as a function.
- **Shapes to nail:** `m` and `v` each have the parameter's shape. Per parameter
  in fp32: 4 B params + 4 B grad + 4 B `m` + 4 B `v` = **16 B/param**, so a 7B
  model needs ~112 GB before a single activation — the number that forces ZeRO /
  8-bit optimizers / LoRA in Phase 7.
- **Predict-first prompt:** Adam's second moment divides by `sqrt(v)`. If a
  parameter's gradient has been near-zero for many steps and then spikes, is the
  resulting step large or small? What does `eps` protect against?
- **Runs on:** CPU + MPS (the 7B memory figure is arithmetic, not a run).
- **Interview hooks:** Why is bias correction needed, and what specifically goes
  wrong in the first ~10 steps without it? Why is decoupled weight decay the
  correct formulation — what does L2-in-the-loss do once Adam's per-parameter
  scaling is applied? Which parameters do you exclude from weight decay in a
  Transformer and why (biases, LayerNorm gains)? How does 8-bit Adam cut the
  16 B/param figure and what does it trade?
- **Common misconception:** That Adam and AdamW differ by a hyperparameter
  default. They differ in *where* decay is applied: Adam adds `wd*p` to the
  gradient (so it is then divided by `sqrt(v)`, making the effective decay
  parameter-dependent), AdamW subtracts `lr*wd*p` from the parameter directly.

### 01.20 — Initialization: why scale decides whether training starts
- **Question it answers:** Why does a net initialised with `randn * 1.0` fail
  while `randn * 0.02` trains?
- **Prereqs:** 01.16, 01.17.
- **Experiment:** a 10-layer MLP, `d=128`, tanh activations. Initialise weights
  three ways — `randn * 1.0`, `randn * 0.01`, and Xavier
  (`std = sqrt(1/fan_in)` for tanh) — and for each print the **std of the
  activations at every layer**. Watch one saturate to ±1 and one collapse to 0.
  Repeat with ReLU and Kaiming (`std = sqrt(2/fan_in)`) and show why the factor
  is 2. Then inspect what `nn.Linear` actually does by default and be surprised.
- **Shapes to nail:** `fan_in` is `W.shape[1]` for an `nn.Linear` weight stored
  as `(out_features, in_features)` — note PyTorch stores it transposed relative to
  the `x @ W` convention, and computes `x @ W.T`. Getting `fan_in` backwards is a
  real and common bug.
- **Predict-first prompt:** With `W ~ randn * 1.0` and `d=128`, what is the
  approximate std of the pre-activation at layer 1 if the input has std 1?
  Compute it from the variance-sum argument, then predict layer 10.
- **Runs on:** CPU + MPS.
- **Interview hooks:** Derive the Kaiming factor of 2 from ReLU zeroing half the
  inputs. Why do GPT-style models initialise residual-branch output projections
  with an extra `1/sqrt(2*n_layers)` factor? Why does LayerNorm make a network
  less init-sensitive but not init-independent?
- **Common misconception:** That `nn.Linear`'s default is Kaiming-for-ReLU.
  It calls `kaiming_uniform_(weight, a=sqrt(5))`, which is equivalent to
  `U(-1/sqrt(fan_in), +1/sqrt(fan_in))` — a legacy choice, not a
  principled ReLU init. Real Transformer code overrides it explicitly.

### 01.21 — Vanishing and exploding gradients
- **Question it answers:** How do I measure whether gradients are actually
  reaching the early layers?
- **Prereqs:** 01.9, 01.20.
- **Experiment:** the same 10-layer tanh MLP. After one `backward()`, print
  `p.grad.norm()` per layer and plot it against depth on a log scale — for the
  bad inits you will see it fall by orders of magnitude toward layer 1. Then
  compute the global norm
  `torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)` (note it *returns* the
  pre-clip total norm — log that value, it is a primary training diagnostic).
  Then show the fix preview: add a residual connection `h = h + f(h)` around each
  block and re-measure the per-layer gradient norms.
- **Shapes to nail:** `p.grad.norm()` is `()` per parameter tensor; the global
  norm is the L2 norm over all parameters concatenated —
  `sqrt(sum(g.norm()**2 for g in grads))`.
- **Predict-first prompt:** In a 10-layer chain, the gradient at layer 1 is a
  product of ~10 Jacobian factors. If each has typical magnitude 0.8, what is the
  ratio of layer-1 to layer-10 gradient magnitude? At 1.2?
- **Runs on:** CPU + MPS.
- **Interview hooks:** Why does a residual connection give the gradient an
  identity path, and what does that do to the product of Jacobians? Why is
  `clip_grad_norm_` (global) preferred to `clip_grad_value_` (elementwise)? What
  does a sudden spike in grad norm during LLM pretraining usually indicate, and
  what is the standard mitigation (skip the batch / lower LR / check the data
  shard)? Where does `clip_grad_norm_` have to sit relative to
  `scaler.unscale_()` under mixed precision?
- **Common misconception:** That vanishing gradients are about the activation
  function alone. They are about the *product of Jacobians over depth*;
  activation choice, weight scale, and normalisation all shift the same product,
  and residual connections attack it structurally rather than by tuning.

---

## Milestone

A single file `03_nn_from_scratch/mlp_from_scratch.py` plus a matching
`tests/test_mlp_from_scratch.py` that, in one `pytest` run, proves:

1. A 2-layer MLP (`d_in=3 -> 5 -> 1`, tanh) implemented with **hand-written
   forward and hand-written backward**, no autograd anywhere.
2. Every hand-computed gradient matches `.backward()` on the equivalent autograd
   model to `torch.allclose(..., atol=1e-6)` — for `W1`, `b1`, `W2`, `b2`.
3. The hand-written `cross_entropy` (softmax + NLL with the log-sum-exp trick)
   matches `F.cross_entropy` on logits containing `+1000`, where a naive
   implementation returns `nan`.
4. A hand-written Adam step matches `torch.optim.Adam` for 5 consecutive steps.
5. `test_overfit_single_batch` drives a fixed 8-example batch below `1e-3` loss
   within 500 steps, seeded and deterministic.

Plus `00_setup_math/NOTES.md` and `03_nn_from_scratch/NOTES.md` written **in your
own words** — including your answers to the predict-first prompts you got wrong,
which are the ones worth writing down.

Exit criterion beyond the tests: hand someone an unfamiliar `forward()` from a
real repo and annotate every line with its output shape, correctly, without
running it.
