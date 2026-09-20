# Revision index

Quick path through everything hands-on, in learning order. Open the notebook,
read the markdown cells and "Interview one-liners" first; outputs are saved.

| Lesson | Notebook | Core takeaway |
| --- | --- | --- |
| 02 Embeddings | [01_embeddings/embeddings.ipynb](01_embeddings/embeddings.ipynb) | ID = row index; `E[b,t,d] = W[T[b,t],d]` |
| 03 Attention (partial) | [02_attention/02_attention.ipynb](02_attention/02_attention.ipynb) | Q/K/V, `(T,T)` scores, causal mask |
| 00.2 Devices + FLOPs | [00_setup_math/00_devices.ipynb](00_setup_math/00_devices.ipynb) | Always `synchronize()` when timing; no float64 on MPS |
| 00.3 Vectors | [00_setup_math/01_vectors.ipynb](00_setup_math/01_vectors.ipynb) | `\|a\| = sqrt(a·a)`; a dot product is big for two reasons (alignment **or** length); cosine divides length out |
| 00.4 Matmul + shapes | [00_setup_math/02_matmul_shapes.ipynb](00_setup_math/02_matmul_shapes.ipynb) | Inner dims match then vanish, outer survive; leading dims are batches; the output's last dim comes from the **weight** |
| 00.5 Transpose | [00_setup_math/03_transpose.ipynb](00_setup_math/03_transpose.ipynb) | `.T` swaps strides, copies nothing; `Q @ K^T` makes `d_k` the inner dim so it vanishes; `(T,T)` is `O(T^2)` and independent of `d_k` — 32 GiB at B=8,H=32,T=8192,fp16 |
| 01.2-01.4 Indexing, broadcasting, bugs | [03_tensor_shapes/00_indexing_broadcasting.ipynb](03_tensor_shapes/00_indexing_broadcasting.ipynb) | Basic slicing = view, advanced = copy; broadcasting aligns from the right; `(3,)` vs `(3,1)` both work and differ — the `keepdim` bug |
| 01.5-01.8 Reshape, matmul, einsum, reductions | [03_tensor_shapes/01_reshape_matmul_einsum.ipynb](03_tensor_shapes/01_reshape_matmul_einsum.ipynb) | The multi-head split is view **then** transpose — one `view` gives the right shape and scrambled heads; `dim=` names the axis that disappears |

Add a row here whenever a lesson is finished. Your own summaries live in each
folder's `NOTES.md`.
