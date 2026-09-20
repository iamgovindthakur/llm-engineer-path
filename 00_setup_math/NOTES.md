# 00_setup_math — my notes (write these yourself, in your own words)

## 00.2 Devices
- What I now understand about async GPU calls:
- Why the no-sync timing was wrong:
- What MPS lacks vs CUDA:
- What surprised me:

## 00.3 Vectors: norm, dot product, cosine
- What the norm of a vector is, and why `|a| = sqrt(a . a)`:
- What a dot product actually measures (the two reasons it can be big):
- The difference between the algebraic form and the geometric form, in my words:
- Why cosine similarity divides by the lengths, and what range it lives in:
- How matrix multiplication relates to dot products:
- What surprised me:

## 00.4 Matmul as composed dot products, and shape algebra
- The rule for when `(a,b) @ (c,d)` is legal, in my own words:
- Which dimension vanishes and why:
- Why square matrices are a bad way to learn this:
- What a leading (batch) dimension means, and why it is not "the same work twice":
- Where the output's last dimension actually comes from:
- The exact operation count, and why it is `2k-1` per cell not `k`:
- What surprised me:
