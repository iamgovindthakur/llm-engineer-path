# 03_tensor_shapes — my notes (write these yourself, in your own words)

## 01.2 Indexing and slicing
- Which index expressions give a view and which give a copy, and how I test it:
- What an integer index does to the rank vs what a slice does:
- `logits[:, -1, :]` vs `logits[:, -1:, :]` — which one I want in a generation loop, and why:
- What surprised me:

## 01.3 Broadcasting rules
- The rule in my own words (alignment direction, and what makes a pair compatible):
- What happens to a shape that has fewer dimensions than the other:
- Why broadcasting does not copy, and what stride 0 means:
- What surprised me:

## 01.4 The bug drill
- Why `(3,)` and `(3,1)` behave differently against a `(3,3)`, in my words:
- What `keepdim=True` is actually protecting me from:
- Mask orientation: what `(1,T)` masks vs what `(T,1)` masks, and which produces `nan`:
- The habit I am going to use to stop this class of bug:
- What surprised me:

## 01.5 reshape vs view vs transpose
- Why `view` failed right after `transpose`, in my words:
- The difference between `view`, `reshape` and `.contiguous().view()`, and which I prefer while learning:
- The multi-head split as two operations, with the shape after each:
- Why the WRONG single-view version is dangerous rather than merely incorrect:
- Why `.contiguous()` is mandatory on the merge back but not on the split out:

## 01.6 matmul semantics
- What matmul does with a 1-D operand:
- How batch dimensions broadcast, and when I would use `bmm` instead:

## 01.7 einsum
- How I know, from the notation alone, which axis is summed:
- What einsum buys me that `@` does not:

## 01.8 reductions
- What `dim=` actually names:
- What `keepdim=True` is protecting, connected back to the 01.4 bug:
- What surprised me:
