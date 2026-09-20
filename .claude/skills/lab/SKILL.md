---
name: lab
description: Build one small from-scratch PyTorch experiment with explicit tensor shapes, on the M3/MPS box. Use when the learner says "/lab", "let's experiment with X", "show me X in code", or wants to poke at a mechanism without a full lesson. Shapes and predictions come before code.
---

# Lab — one experiment, one idea

A lab is smaller than a lesson. One mechanism, one runnable file, one insight.
No curriculum bookkeeping, no ten steps.

## Rules

- **One idea.** If it needs two, it is two labs.
- **Tiny dimensions.** `B=2, T=4, d_model=8, n_heads=2`. Every intermediate
  tensor should be small enough to print in full and read with your eyes.
- **Shapes first.** Write the shape table before the code:

  | Tensor | Shape | Meaning |
  | --- | --- | --- |

- **Predict first.** Ask the learner for the output shape (or a specific value)
  before running. Wait for their answer.
- **Raw ops.** `@`, `einsum`, explicit `reshape`/`transpose`. No high-level
  module that hides the operation being studied.
- **Seed everything.** `torch.manual_seed(0)` at the top.
- **Print intermediates.** The point of a lab is seeing the middle, not the end.
  `torch.set_printoptions(precision=3, sci_mode=False)`.

## Device

```python
import torch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

Or `from common.device import get_device`.

If the experiment needs CUDA, say so up front, explain *why* the hardware
matters, and offer either the Colab path or a CPU toy that still shows the
mechanism. Never pretend something ran locally.

## Sanity check

Where a PyTorch reference exists, assert against it:

```python
torch.testing.assert_close(mine, reference, rtol=1e-4, atol=1e-5)
```

On MPS, float32 accumulation differs from CPU — if a close-comparison fails
marginally, check the device before suspecting the math.

## Output

A single file under the relevant `NN_topic/` directory, or `scratch/` if it is
throwaway. End with one sentence: what this experiment proved.
