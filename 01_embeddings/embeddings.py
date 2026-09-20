# %% [markdown]
# # Embeddings — experiment 1: token IDs are addresses
#
# ## Goal
#
# See the smallest possible embedding lookup. A token ID is just an integer
# label; an embedding is a short row of numbers associated with that label.
#
# We use a pretend vocabulary of five tokens. This is not tokenization yet—we
# choose the IDs ourselves so we can focus only on the lookup.
#
# | Token | ID |
# | --- | ---: |
# | `cat` | 0 |
# | `dog` | 1 |
# | `runs` | 2 |
# | `sleeps` | 3 |
# | `.` | 4 |
#
# Each embedding has 3 numbers, so `embedding_dim = 3`.

# %%
import torch

# Use the Apple GPU (MPS) when this notebook kernel can access it.
# Otherwise use the CPU, so the lesson remains runnable everywhere.
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# A tiny input sequence: cat, sleeps, dog.
# Shape: (3,) because this sequence contains 3 token IDs.
token_ids = torch.tensor([0, 3, 1], device=device)

# One row per possible token ID, and 3 numbers per row.
# Shape: (vocab_size, embedding_dim) = (5, 3).
embedding_table = torch.tensor(
    [
        [0.1, 0.2, 0.3],  # ID 0: cat
        [0.4, 0.5, 0.6],  # ID 1: dog
        [0.7, 0.8, 0.9],  # ID 2: runs
        [1.0, 1.1, 1.2],  # ID 3: sleeps
        [1.3, 1.4, 1.5],  # ID 4: .
    ],
    device=device,
)

# Use each ID as a row address in the table.
# Shape: (3, 3): one 3-number embedding for each of 3 input tokens.
input_embeddings = embedding_table[token_ids]

print("token IDs:", token_ids)
print("embedding table shape:", embedding_table.shape)
print("input embeddings shape:", input_embeddings.shape)
print("input embeddings:\n", input_embeddings)

# %% [markdown]
# ## What happened?
#
# `token_ids` is `[0, 3, 1]`. Indexing `embedding_table[token_ids]` selects
# rows 0, 3, and 1—in that exact order. It does not do arithmetic with the IDs:
# ID 3 simply means “give me row 3.”
#
# The table has shape `(5, 3)`: 5 vocabulary entries, each represented by 3
# numbers. The result has shape `(3, 3)`: 3 input tokens, each represented by
# 3 numbers. The two tensors must live on the same device; here, both are on
# `device`. In a trained LLM, the numbers are learned; we chose readable values.

# %% [markdown]
# # Embeddings — experiment 2: PyTorch's embedding layer
#
# ## Problem
#
# We can index an ordinary tensor ourselves, but a neural network needs to know
# which tensors are its learnable parts. `torch.nn.Embedding` is PyTorch's layer
# for an embedding table.
#
# ## Intuition and mathematics
#
# `nn.Embedding(5, 3)` owns a weight table `W` with shape `(5, 3)`. Its lookup
# rule is the same as before: for an input ID `t`, its output is row `W[t]`.
# For a sequence shaped `(3,)`, it produces `(3, 3)`. There is still **no matrix
# multiplication** here—only row selection.
#
# The layer starts with random values. Training will adjust them in a later
# lesson so useful tokens receive useful vectors.

# %%
# A seed makes this layer's initial random numbers reproducible.
torch.manual_seed(7)

# 5 possible token IDs (0 through 4); 3 numbers in every embedding.
# The layer owns a learnable weight table with shape (5, 3).
embedding_layer = torch.nn.Embedding(
    num_embeddings=5,
    embedding_dim=3,
    device=device,
)

# Call the layer with the same IDs as experiment 1.
# Input shape: (3,); output shape: (3, 3).
layer_embeddings = embedding_layer(token_ids)

print("weight table shape:", embedding_layer.weight.shape)
print("input token IDs shape:", token_ids.shape)
print("layer output shape:", layer_embeddings.shape)
print("layer output:\n", layer_embeddings)

# %% [markdown]
# ## What happened?
#
# `embedding_layer.weight` is the table PyTorch will eventually learn. Calling
# `embedding_layer(token_ids)` selected one row per ID, just as
# `embedding_table[token_ids]` did in experiment 1. The numbers differ because
# this is a newly initialized random table; the shape rule is the important
# result.
#
# ## Pause and predict
#
# A batch contains two token-ID sequences, each four tokens long. Its input
# shape is `(2, 4)`. With this layer's `embedding_dim = 3`, what output shape do
# you predict? Explain what each of the three dimensions represents.
