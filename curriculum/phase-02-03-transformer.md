# Phase 2 — Tokenization & Positional Information
# Phase 3 — Attention & the Transformer Block

Lesson-level plan. This is a syllabus, not the lessons. Claude teaches each entry live
using the 10-step loop in `CLAUDE.md`; this file fixes the ordering, the experiment, and
the shapes so no lesson drifts.

Directory mapping: Phase 2 → `03_tokenization/`, `04_positional/`.
Phase 3 → `02_attention/` (continued), `05_multihead/`, `06_rope/`, `07_norm_ffn/`, `08_block/`.

Assumed already known (do NOT re-teach): token IDs, embedding lookup, `nn.Embedding`,
batched lookup, toy `dict` vocab, dot product vs matmul, and a partly-finished
single-head causal self-attention notebook at `02_attention/02_attention.ipynb`
(unbatched, hand-written weight matrices, `(T,T)` scores, causal mask, softmax).

Environment facts verified on this machine: torch 2.10.0 (MPS available), transformers
5.3.0, tokenizers 0.22.2, datasets 4.6.1. **`sentencepiece` is not installed** — the
Unigram lesson uses the `tokenizers` library's own Unigram trainer instead, so nothing
in these two phases requires a new native dependency.

---

## What you can claim after this phase

- I can implement byte-level BPE end to end from scratch — train the merge table, encode
  with merge ranks, decode back to the exact original bytes — and explain where WordPiece
  and Unigram diverge from it and why.
- I can defend vocabulary size as an architecture decision, quantified: embedding and
  output-projection parameters, softmax cost per step, sequence length / fertility, and
  what weight tying buys and costs.
- I can derive the `sqrt(d_k)` scaling from a variance argument, explain the softmax
  max-subtraction trick, and say exactly why the causal mask uses `-inf`.
- I can write multi-head attention by hand — every reshape and transpose, with the shape
  at each step — and explain why `(B,T,H,d_head) -> (B,H,T,d_head)` is a transpose and not
  a reshape.
- I can motivate MQA / GQA / MLA from KV-cache bytes rather than from quality, and derive
  the cache size for a real config.
- I can implement RoPE, show algebraically that it makes attention scores depend only on
  relative position, and explain how it interacts with a KV cache.
- I can assemble one complete Transformer block, count its parameters by hand, and get
  within a rounding error of a published model's reported parameter count.

---

# PHASE 2 — Tokenization and positional information

## 2A. Why subwords exist

### 02.1 — The two failure modes: word-level and char-level vocabularies
- **Question it answers:** Why is neither "one token per word" nor "one token per character" usable at scale?
- **Prereqs:** token IDs, `nn.Embedding`, toy dict vocab.
- **Experiment:** Take ~200KB of plain text. Build (a) a whitespace word vocab, (b) a character vocab. Report for each: vocab size, sequence length in tokens, and OOV rate when you then tokenise a *held-out* slice containing an unseen word (e.g. `tokenizer`, `antidisestablishmentarianism`, `covid`).
- **Shapes to nail:** embedding matrix `(V, d_model)` — compute its parameter count for V=50k and V=256 at `d_model=768`; token-count ratio between the two schemes on the same text.
- **Predict-first prompt:** Word-level on this corpus gives maybe 25k types. Char-level gives ~100. Before running: which one produces more *parameters*, and which one produces more *compute per document*? Are those the same axis?
- **Runs on:** CPU (pure Python, no torch needed except for the param count).
- **Interview hooks:** Why does a word-level model still fail even if you add an `<unk>` token? What is Zipf's law doing to your vocab coverage curve? If char-level has no OOV problem at all, why did nobody ship it?
- **Common misconception:** That OOV is the only problem with word-level. The deeper problem is that a long tail of rare words gets embeddings that are almost never updated — they are parameters that never learn.

### 02.2 — Bytes and UTF-8: the representation that cannot OOV
- **Question it answers:** Why do modern tokenizers operate on bytes rather than on Unicode characters?
- **Prereqs:** 02.1.
- **Experiment:** `"héllo 世界 🙂".encode("utf-8")` — inspect the byte list, its length vs `len(str)`. Show that a 4-byte emoji is 4 IDs in a 256-symbol byte vocab. Then show the GPT-2 byte↔unicode remapping trick: a reversible map from the 256 byte values onto printable Unicode codepoints so that the tokenizer's internal strings are always printable and whitespace is visible.
- **Shapes to nail:** base vocab is exactly 256 entries; `len(text)` (chars) vs `len(text.encode())` (bytes) vs number of tokens — three different numbers, keep them distinct.
- **Predict-first prompt:** If the base alphabet is 256 bytes, can this tokenizer ever emit `<unk>`? What is the worst-case token count for a 1000-character Devanagari string?
- **Runs on:** CPU.
- **Interview hooks:** Why is byte-level fair to every language in *coverage* but not in *cost*? What breaks if you split a multi-byte codepoint across two tokens during streaming decode? Why does GPT-2 map bytes to printable codepoints instead of just using `bytes` directly?
- **Common misconception:** "Byte-level means no unknown tokens, therefore no problem." Coverage is solved; **fertility** (tokens per word) is not, and that is the real multilingual tax — revisited in 02.11.

## 2B. BPE from scratch — the core interview asset

> Give these three lessons real weight. "Implement BPE" is one of the most common
> take-home / whiteboard asks for LLM roles. The learner must be able to write the
> training loop and the encoder from a blank file.

### 02.3 — BPE training: learning the merge table
- **Question it answers:** How does a BPE tokenizer *learn* its vocabulary from a corpus?
- **Prereqs:** 02.2, Python dicts and `collections.Counter`.
- **Experiment:** Corpus of ~5 short sentences, target vocab 256 + 20 merges. Loop: count adjacent symbol-pair frequencies across all word sequences → take the argmax pair → append it to `merges` with its rank → replace every occurrence of that pair in place → repeat. Print the pair chosen and the new vocab entry at *every* iteration so the merge order is visible.
- **Shapes to nail:** `merges: list[tuple[bytes,bytes]]` of length `n_merges`, index = rank; `vocab: dict[int, bytes]` of size `256 + n_merges`; the per-word symbol list shrinking by one on each applied merge.
- **Predict-first prompt:** On a corpus where `"the "` is the most frequent 4-gram, what will merges 1, 2 and 3 be, in order? Write them down before running.
- **Runs on:** CPU.
- **Interview hooks:** What is the time complexity of the naive training loop (recount all pairs every merge) and how would you get it down? Why must merges be stored as an *ordered* list rather than a set? What does the pre-tokenizer (splitting on whitespace/punctuation first) prevent a merge from doing, and why is that desirable?
- **Common misconception:** That BPE merges the most frequent *token*. It merges the most frequent adjacent *pair*, and only pairs that occur within a pre-token — merges never cross a whitespace boundary if the pre-tokenizer says so.

### 02.4 — BPE encoding: applying merges by rank
- **Question it answers:** Given a trained merge table, how do you turn new text into IDs — and why is it not simply "longest match"?
- **Prereqs:** 02.3.
- **Experiment:** Implement `encode(text) -> list[int]`. Split to pre-tokens → to bytes → repeatedly find the pair in the current symbol list with the **lowest merge rank** and apply it, until no pair is in the table. Encode a word the trainer never saw and trace the merge sequence.
- **Shapes to nail:** `list[int]` of length `T`; trace the symbol list length going `n_bytes -> ... -> T` one merge at a time.
- **Predict-first prompt:** The word is `lower`, and the table contains `lo` (rank 3), `we` (rank 1), `er` (rank 0). Which merge fires first, and what is the final token sequence? Now swap the ranks of `lo` and `er` — does the answer change?
- **Runs on:** CPU.
- **Interview hooks:** Why lowest-rank-first rather than greedy-longest-match — what invariant does it preserve between training and inference? Is BPE encoding deterministic? Is it the *optimal* (fewest-token) segmentation given the vocab? What would BPE-dropout change here?
- **Common misconception:** That encoding is a dictionary lookup of the longest substring in the vocab. It is a *replay of the training merge order*; that is why the ranks are stored, not just the vocab.

### 02.5 — BPE decoding, round-tripping, and where it breaks
- **Question it answers:** How do you go from IDs back to the exact original bytes, and when can you not?
- **Prereqs:** 02.4.
- **Experiment:** `decode(ids) -> str`: concatenate the byte strings for each ID, then `.decode("utf-8", errors="replace")`. Assert `decode(encode(s)) == s` over a list of adversarial strings: trailing spaces, `\t\n`, `"  "` double space, emoji, mixed-script. Then decode a *prefix* of an emoji's tokens to see the replacement character — the streaming-decode bug.
- **Shapes to nail:** `str -> list[int] (T) -> bytes -> str`; confirm the byte string is identical, not just visually similar.
- **Predict-first prompt:** You decode token-by-token for a streaming UI. The next token completes the second byte of a 3-byte codepoint. What does the user see on screen this frame?
- **Runs on:** CPU.
- **Interview hooks:** How do you implement correct streaming detokenisation in a serving stack? Why is `decode(encode(x)) == x` a test you should actually write? When is a tokenizer legitimately lossy (lowercasing, NFKC normalisation, accent stripping) and what does that cost you?
- **Common misconception:** That you can decode each token independently and concatenate the strings. You must concatenate the *bytes* and decode once, or buffer incomplete sequences.

## 2C. The other two algorithms

### 02.6 — WordPiece: score-based merges and the `##` convention
- **Question it answers:** WordPiece and BPE both merge pairs — what is actually different?
- **Prereqs:** 02.3.
- **Experiment:** Reuse the 02.3 training loop, swap the selection criterion from raw pair count to the likelihood-style score `count(ab) / (count(a) * count(b))`. Train on the same tiny corpus and diff the resulting merge order against BPE's. Then load `bert-base-uncased`'s tokenizer and inspect `##` continuation pieces.
- **Shapes to nail:** same `vocab` / `merges` structures; the diff is in *which* pair is chosen at each step, not in the data structures.
- **Predict-first prompt:** A pair where both halves are individually very common (e.g. `t` + `h`) has a high raw count. Under the WordPiece score, does it go up or down relative to a pair of two rare-but-always-together symbols?
- **Runs on:** CPU.
- **Interview hooks:** Why does the WordPiece score prefer "surprising" pairs? What does the `##` marker buy you that byte-level BPE gets from the `Ġ`/space-prefix convention instead? Why is WordPiece's *inference* greedy longest-match-first while BPE's is rank-replay?
- **Common misconception:** That `##` is cosmetic. It makes word-initial and word-internal pieces *different tokens*, which changes the vocab budget and makes detokenisation rule-based rather than byte-exact.

### 02.7 — Unigram / SentencePiece: prune down instead of merge up
- **Question it answers:** What is a fundamentally different way to get a subword vocabulary, and what does it give you that BPE cannot?
- **Prereqs:** 02.6; basic idea of a probability over segmentations.
- **Experiment:** Train a Unigram model on the same tiny corpus with `tokenizers.models.Unigram` + `UnigramTrainer` (`sentencepiece` is *not* installed on this machine — use the `tokenizers` library, already present at 0.22.2). Inspect the resulting vocab *with its log-probabilities*. Show that BPE gives one segmentation of a word and Unigram can score several.
- **Shapes to nail:** vocab as `list[(piece, log_prob)]` — note that BPE's vocab has no scores at all; that difference is the whole point.
- **Predict-first prompt:** Unigram starts from a large candidate set and *removes* pieces. What does it optimise when deciding which piece to drop, and does the vocab size go up or down during training?
- **Runs on:** CPU.
- **Interview hooks:** How does subword regularisation / sampled segmentation work and why is it a training-time regulariser? Why is SentencePiece's "treat the input as a raw stream, encode spaces as `▁`" design important for languages without spaces? Which of BPE/WordPiece/Unigram can tell you the probability of a segmentation, and why does that matter?
- **Common misconception:** That Unigram is just "BPE done backwards". The vocab is a probabilistic model over segmentations; multiple segmentations are legal and scored, which is what makes sampling possible.

## 2D. From your implementation to the production API

### 02.8 — Mapping the HuggingFace tokenizer API back onto your code
- **Question it answers:** For every object in my from-scratch BPE, where does it live in a real `tokenizers` / `transformers` tokenizer?
- **Prereqs:** 02.4, 02.5, 02.7.
- **Experiment:** Load `gpt2`'s tokenizer. Walk the four-stage pipeline explicitly: `normalizer -> pre_tokenizer -> model -> post_processor` (+ `decoder`). Call `tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(s)` and compare its output to your own pre-tokenizer. Dump `tokenizer.get_vocab()` and the `merges` and confirm your `encode` reproduces HF's `input_ids` on 20 sample strings.
- **Shapes to nail:** the call returns a `BatchEncoding`, a dict-like with `input_ids` and `attention_mask`, each `(B, T)` when `return_tensors="pt"`. `BatchEncoding.tokens()` and `.word_ids()` for alignment.
- **Predict-first prompt:** Your from-scratch encoder and GPT-2's will disagree on at least one of these 20 strings. Which category will it be — and is the bug in your merge loop or in your pre-tokenizer regex?
- **Runs on:** CPU (tokenizers are Rust, not GPU work; nothing here touches MPS).
- **Interview hooks:** Which pipeline stage is responsible for a tokenizer being lossy? Why were the "slow" Python tokenizers removed in transformers v5, and what did the fast/Rust path buy? What is `offset_mapping` for, and which NLP tasks break without it?
- **Common misconception:** That "the tokenizer" is one algorithm. It is a 4-5 stage pipeline, and most surprising behaviour comes from the normalizer or pre-tokenizer, not the BPE model.
- **Version note (verified):** on transformers 5.x, `AutoTokenizer.from_pretrained` no longer takes `use_fast` (there is only one implementation per model), `additional_special_tokens` was renamed to `extra_special_tokens`, and `apply_chat_template` now returns a full `BatchEncoding` rather than a bare list of ids.

### 02.9 — Special tokens, BOS/EOS/PAD, and chat templates
- **Question it answers:** Where do `<|begin_of_text|>` and the chat role markers actually come from, and who is responsible for inserting them?
- **Prereqs:** 02.8.
- **Experiment:** Print `tokenizer.special_tokens_map` and the special IDs for a base model vs an instruct model. Then `tokenizer.apply_chat_template(messages, tokenize=False)` on a 3-turn conversation to see the literal rendered string, and again with `add_generation_prompt=True` to see the trailing assistant header. Diff the two strings character by character.
- **Shapes to nail:** `apply_chat_template(..., tokenize=True)` → `BatchEncoding` with `input_ids (B,T)`; the same conversation rendered by two different models' templates gives two different `T`.
- **Predict-first prompt:** You fine-tune on data formatted with template A and serve behind an engine that applies template B. The model still produces text. What specifically degrades, and would your eval catch it?
- **Runs on:** CPU.
- **Interview hooks:** Why is the chat template stored in the tokenizer config rather than in the serving code? What is `add_generation_prompt` and what happens at inference if you forget it? Why must PAD be masked out *and* excluded from the loss — are those the same mechanism? What goes wrong if PAD and EOS are the same ID?
- **Common misconception:** That chat formatting is a serving-layer string concatenation. It is part of the model's learned distribution; a template mismatch is a silent train/serve skew bug.

### 02.10 — Padding, truncation, and the attention mask
- **Question it answers:** How do variable-length sequences become one rectangular tensor without corrupting the model's view?
- **Prereqs:** 02.9; the causal mask from `02_attention/`.
- **Experiment:** Tokenise 3 strings of clearly different lengths with `padding=True, truncation=True, max_length=8, return_tensors="pt"`. Print `input_ids` and `attention_mask` side by side. Then flip `padding_side` between `"right"` and `"left"` and explain which one generation needs. Manually zero out the mask for one real token and observe the attention row change.
- **Shapes to nail:** `input_ids (3, 8)`, `attention_mask (3, 8)` of 0/1 → broadcast to `(B, 1, 1, T)` to combine with the causal `(1, 1, T, T)` mask → additive mask `(B, 1, T, T)`. Nail why the padding mask is a *key* mask (a column) while the causal mask is a triangle.
- **Predict-first prompt:** `B=3, T=8`, one sequence has 3 real tokens. Before running: how many of the 64 entries in its `(T,T)` attention matrix should be `-inf` after combining both masks?
- **Runs on:** MPS (tiny; also fine on CPU).
- **Interview hooks:** Why does decoder generation with right-padding produce garbage, while training with right-padding is fine? What is the throughput cost of padding to the batch max, and what do length-bucketing and sequence packing do about it? If a row is *entirely* masked, what does softmax return — and how do you defend against it?
- **Common misconception:** That the attention mask and the causal mask are the same object. One is per-sequence data-dependent (padding), one is a fixed architectural triangle; they are combined, and confusing them is a classic bug.

## 2E. Vocabulary as an architecture decision

### 02.11 — Why vocab size is a real tradeoff
- **Question it answers:** What do I actually pay for, on each axis, when I move V from 32k to 128k to 256k?
- **Prereqs:** 02.10, 02.2 (fertility).
- **Experiment:** Build a 4-column table by direct computation, not by looking anything up. For `d_model=4096`, V ∈ {32000, 128256, 256000}: (1) embedding params `V*d`, (2) untied output-projection params `V*d`, (3) final-logits compute per generated token `2*V*d` FLOPs and the logits tensor `(B, T, V)` in bytes, (4) measured fertility — tokens per word — for the *same* paragraph in English, Hindi and Python source, using three real tokenizers (`gpt2` V=50257, `bert-base-uncased` V=30522, a Llama-3 tokenizer V=128256).
- **Shapes to nail:** embedding `(V, d_model)`; logits `(B, T, V)` — compute its size in MB for `B=8, T=2048, V=128256` in fp32 vs bf16 and notice it can exceed the activations of several layers.
- **Predict-first prompt:** Doubling V doubles the embedding parameters. What does it do to the *number of tokens* a Hindi paragraph costs, and therefore to the attention cost at fixed wall-clock budget? Which effect wins?
- **Runs on:** CPU for the fertility measurement; arithmetic by hand.
- **Interview hooks:** Why did Llama 3 move from 32k to 128k and what did it buy on non-English throughput? Why is the final softmax an outsized share of compute for *small* models but not large ones? What is the KL/quality cost of a vocab so large that rare tokens are undertrained? How does a large V interact with the loss's effective class imbalance?
- **Common misconception:** That a bigger vocab is purely a parameter cost. It simultaneously *shortens* sequences (cheaper attention, more content per context window) — it is a genuine two-sided tradeoff, and the fertility side is why multilingual models pay for a big vocab.

### 02.12 — Tokenizer pathologies: numbers, code, whitespace, non-English
- **Question it answers:** Which model failures are actually tokenizer failures?
- **Prereqs:** 02.11.
- **Experiment:** Tokenise and print the pieces for: `1234567`, `1,234,567`, `3.14159`; the same integer with and without a leading space; a Python snippet with 4-space and 8-space indentation; `strawberry`, `raspberry`; the same sentence in English / Hindi / Devanagari-transliterated; a long URL; a UUID. Then the classic: ask a model to count the `r`s in `strawberry` and show the token split alongside.
- **Shapes to nail:** token count per string; the *identity* of the pieces — `" 1234"` vs `"1234"` are different IDs and that is the whole lesson.
- **Predict-first prompt:** How many tokens is `1234567`? Is it split left-to-right in 3-digit groups, or somewhere else entirely? What does that imply for a model doing column-aligned addition?
- **Runs on:** CPU.
- **Interview hooks:** Why does digit-by-digit tokenisation (or right-to-left 3-digit grouping) improve arithmetic, and which models adopted it? "Count the r's in strawberry" — explain the failure in terms of what the model can and cannot see. What is the SolidGoldMagikarp / glitch-token phenomenon and what does it tell you about vocab training data vs model training data? Why do code models need whitespace-aware tokenizers?
- **Common misconception:** That these are reasoning failures. The model never sees characters; it sees 2-4 opaque IDs. Fixing the prompt cannot fix it; fixing the tokenizer can.

### 02.13 — Embedding weight tying with the output projection
- **Question it answers:** What is `lm_head.weight is embed_tokens.weight` doing, and when should you not do it?
- **Prereqs:** 02.11; that the output head is a `(d_model, V)` projection producing logits.
- **Experiment:** Tiny model: `V=100, d_model=16`. Build untied (two matrices) and tied (`lm_head.weight = embed.weight`) variants; count parameters in both. Do one backward pass on the tied version and inspect `embed.weight.grad` — show it receives gradient from *both* the lookup path and the logits path. Then check `tie_word_embeddings` in the real configs for a small and a large open model and note that they differ.
- **Shapes to nail:** embedding `(V, d_model)`; logits computed as `h @ W_E.T` where `h` is `(B,T,d_model)` and `W_E.T` is `(d_model, V)` → `(B, T, V)`. The transpose is the entire mechanism.
- **Predict-first prompt:** `V=128256, d_model=4096`. How many parameters does tying save, and what fraction is that of a 1B-parameter model? Of an 8B one? Which of the two would you expect to tie?
- **Runs on:** MPS.
- **Interview hooks:** Why is tying defensible at all — what does it assert about input and output token representations? Why do small models tie and large ones often don't? How does tying interact with the softmax temperature / logit scale, and why do some models add a learned output scale? What breaks if you tie but use different tokenizers for input and output (e.g. an encoder-decoder)?
- **Common misconception:** That tying is purely a memory optimisation. It also couples two gradient paths and constrains the geometry of the embedding space, which is a modelling decision, not just a plumbing one.

## 2F. Positional information

### 02.14 — Self-attention is permutation-equivariant: prove it
- **Question it answers:** Why does attention need position information injected at all?
- **Prereqs:** the single-head attention from `02_attention/`; 02.13.
- **Experiment:** `T=4, d_model=8, d_head=4`. Run attention **without** the causal mask on `X`, then on `X[perm]` for a permutation `perm`. Show `out(X[perm]) == out(X)[perm]` to within float tolerance. Then repeat *with* the causal mask and show the identity breaks.
- **Shapes to nail:** `X (4,8)`, `Q/K/V (4,4)`, scores `(4,4)`, output `(4,4)`; the permuted output is a row-permutation of the original.
- **Predict-first prompt:** Before running: if I shuffle the rows of `X`, do the attention *weights* change, or just their arrangement? Is the output for a given token identical, or merely similar?
- **Runs on:** MPS (watch float tolerance — use `torch.allclose` with an explicit `atol`, and check on CPU too if MPS numerics look off).
- **Interview hooks:** The precise term is permutation-*equivariant*, not invariant — state the difference. Given that the causal mask already breaks symmetry, why is a causal LM still not positionally aware enough without explicit encodings? Is an MLP-only model permutation-equivariant across the sequence axis?
- **Common misconception:** That the causal mask supplies position. It supplies *ordering constraints* (who may see whom) but not *distance*; a token cannot tell whether a permitted neighbour is 1 or 400 positions back.

### 02.15 — Learned absolute positional embeddings
- **Question it answers:** What is the simplest thing that works, and what is its hard ceiling?
- **Prereqs:** 02.14.
- **Experiment:** `nn.Embedding(max_T, d_model)` indexed by `arange(T)`, added to the token embeddings. `B=2, T=4, d_model=8, max_T=16`. Verify the sum is elementwise and that the same token at positions 0 and 3 now has different vectors. Then feed `T=20 > max_T` and watch it raise.
- **Shapes to nail:** token emb `(B,T,d)` + pos emb `(T,d)` broadcast over `B` → `(B,T,d)`. Say out loud why the broadcast works.
- **Predict-first prompt:** `pos_emb` is `(T, d)` and `tok_emb` is `(B, T, d)`. Which axis does broadcasting insert, and what would go wrong if `pos_emb` were `(B, d)` instead?
- **Runs on:** MPS.
- **Interview hooks:** Why is *addition* (rather than concatenation) the standard, and what does that cost? How many parameters does `max_T=4096, d=4096` add? What exactly happens if you fine-tune to a longer context — can you just resize the table? Why is GPT-2 hard-capped at 1024 tokens?
- **Common misconception:** That adding position to content "pollutes" the embedding. The residual stream has enough dimensions for the model to keep sub-spaces largely separate; the empirical objection to learned-absolute is extrapolation, not interference.

### 02.16 — Sinusoidal positional encodings
- **Question it answers:** Can position be a fixed function instead of learned parameters, and what does the frequency structure buy?
- **Prereqs:** 02.15.
- **Experiment:** Implement the original `sin/cos` formula for `d_model=16, max_T=64`. Heatmap the `(64,16)` matrix. Then compute the cosine similarity between `pos_enc[i]` and `pos_enc[j]` for several offsets and show similarity decays with `|i-j|`. Show that `PE(pos+k)` is a fixed linear map of `PE(pos)` for fixed `k` — this is the seed for RoPE.
- **Shapes to nail:** `pos_enc (max_T, d_model)`; even indices sin, odd indices cos; wavelengths going from `2π` to `10000·2π` across the dimension axis.
- **Predict-first prompt:** The low-index dimensions oscillate fast, the high-index ones slowly. If you only had the *slowest* dimension, what could you determine about a position? If only the fastest?
- **Runs on:** MPS/CPU (matplotlib for the heatmap).
- **Interview hooks:** Zero parameters and unbounded `T` — so why did nobody keep using it? What property does the paper claim about relative positions, and does it survive the Q/K projections? Why does the base `10000` matter, and what happens if you change it?
- **Common misconception:** That sinusoidal encodings extrapolate to longer contexts in practice. The *encoding* is defined for any position; the trained attention patterns are not, and quality still collapses beyond the trained length.

### 02.17 — Why absolute position is the wrong abstraction (setup only)
- **Question it answers:** What is the argument that attention should depend on `i-j` rather than on `i` and `j` separately? (Setup for Phase 3 — do NOT implement RoPE here.)
- **Prereqs:** 02.16.
- **Experiment:** No new implementation. One derivation on paper: write the attention score `q_i · k_j` with absolute embeddings added in, expand the four cross terms (content-content, content-position, position-content, position-position), and mark which terms depend on `i-j` and which do not. Then a 3-line literature map — relative bias (T5), ALiBi, RoPE — with one sentence each on where the position information is injected (added to scores vs rotated into q/k).
- **Shapes to nail:** the expansion of `(x_i + p_i)W_Q · (x_j + p_j)W_K` into four terms; note that only one of them is pure content.
- **Predict-first prompt:** "The cat sat on the mat" appearing at position 5 and at position 500 in a long document. Which attention terms change, and should they?
- **Runs on:** CPU (paper exercise).
- **Interview hooks:** Why would a relative scheme be expected to extrapolate better? What does ALiBi do that requires no parameters and no rotation at all? Which schemes are compatible with a KV cache without re-encoding the whole prefix? (Answer in 03.13.)
- **Common misconception:** That relative position means "subtract the position indices". It means the *score function* is a function of the offset; different families achieve that in structurally different places.

---

# PHASE 3 — Attention and the Transformer block

## 3A. Finish and solidify single-head causal attention

> `02_attention/02_attention.ipynb` currently has hand-written `W_Q/W_K/W_V` tensors,
> no batch dimension, and an unexplained `sqrt(d_head)` division. These lessons close it out.

### 03.1 — Q, K, V as learned projections, batched
- **Question it answers:** What do the three projections actually *mean*, and why three rather than one or two?
- **Prereqs:** the existing notebook; matmul shapes.
- **Experiment:** Replace the hand-written matrices with `nn.Linear(d_model, d_head, bias=False)` ×3 and add the batch dimension. `B=2, T=4, d_model=8, d_head=4`, seeded. Confirm the outputs match the manual version when you copy the weights across (mind that `nn.Linear` stores `W` transposed — `(out,in)`). Then ablate: set `W_V = I` and describe what the output becomes.
- **Shapes to nail:** `X (2,4,8)`; `W_Q/W_K/W_V` as `nn.Linear` weight `(4,8)`; `Q/K/V (2,4,4)`; `Q @ K.transpose(-2,-1)` → `(2,4,4)`; `attn @ V` → `(2,4,4)`. Make the learner say why `.transpose(-2,-1)` and not `.T`.
- **Predict-first prompt:** `nn.Linear(8, 4)` — is its `.weight` shape `(8,4)` or `(4,8)`? Predict, then print. Why did the library choose that layout?
- **Runs on:** MPS.
- **Interview hooks:** Why are Q and K separate matrices when only the product `W_Q W_K^T` appears in the score — what is the effective rank and why does the factorisation matter? What does V represent that K does not? Why is `bias=False` standard in these projections? What is the query/key/value database analogy and where does it break down?
- **Common misconception:** That Q, K, V are three "views" of the token with fixed semantic meanings. They are learned; their interpretability is emergent (induction heads etc.), not designed.

### 03.2 — The `sqrt(d_k)` scaling: derive the variance argument
- **Question it answers:** Why divide by `sqrt(d_k)` and not by `d_k`, or by nothing?
- **Prereqs:** 03.1; variance of a sum of independent variables.
- **Experiment:** Sample `q, k ~ N(0,1)` with `d_k ∈ {4, 64, 512, 4096}`, 10000 pairs each. Empirically measure `Var(q·k)` and confirm it equals `d_k`. Then push unscaled scores through softmax and measure the max probability and the entropy of the resulting distribution as `d_k` grows. Finally, backprop through both and compare gradient norms.
- **Shapes to nail:** `q,k (N, d_k)`; rowwise dot → `(N,)`; measured variance ≈ `d_k`; after dividing by `sqrt(d_k)`, variance ≈ 1.
- **Predict-first prompt:** If each component has mean 0 and variance 1 and they are independent, what is `Var(sum of d_k products)`? Derive it before you measure it. Then: what does the standard deviation do as `d_k` goes 64 → 4096?
- **Runs on:** MPS.
- **Interview hooks:** Derive `Var(q·k) = d_k` from scratch on a whiteboard. Why does a large-variance input make softmax saturate, and what does a saturated softmax do to the *gradient*? Why `sqrt(d_k)` and not `d_k` — what would over-scaling cost you? The assumption is unit-variance iid components; what in the real architecture (LayerNorm, initialisation) makes that roughly true, and when does it fail? What is QK-norm and what problem is it solving beyond this one?
- **Common misconception:** That the scaling exists "to keep numbers small for numerical stability". Softmax is already shift-invariant and numerically stabilised (03.3). The real reason is gradient flow: unscaled scores saturate softmax into a near-one-hot distribution with vanishing gradients.

### 03.3 — Softmax numerics and the max-subtraction trick
- **Question it answers:** Why does every real softmax subtract the row max first, and why is that free?
- **Prereqs:** 03.2.
- **Experiment:** Write `softmax_naive` and `softmax_stable`. Feed `[1000., 1001., 1002.]` — naive gives `nan`/`inf`, stable is correct. Feed `[-1000., -1001.]` — naive underflows to `0/0`. Prove shift-invariance algebraically, then verify numerically. Compare both against `torch.softmax` on the same inputs.
- **Shapes to nail:** softmax over `dim=-1` of a `(B,H,T,T)` tensor; the max is taken with `keepdim=True` giving `(B,H,T,1)` so it broadcasts back.
- **Predict-first prompt:** `exp(1000)` in fp32 — what do you get? Now predict what `exp(1000)/(exp(1000)+exp(1001))` *should* be mathematically, and what the naive code will actually return.
- **Runs on:** MPS and CPU — run both, compare. MPS numerics differ from CPU; this is a good place to learn that habit.
- **Interview hooks:** Prove `softmax(x + c) = softmax(x)` for scalar `c`. What is log-sum-exp and where else does it show up? What is the fp16 dynamic range and why is attention often computed in fp32 even in a bf16 model? How does online/streaming softmax (the running-max rescaling in Flash Attention) work, and why is it required to avoid materialising the `(T,T)` matrix?
- **Common misconception:** That subtracting the max is an approximation. It is *exactly* equal by shift-invariance — zero accuracy cost.

### 03.4 — Causal masking: `-inf` vs a large negative number
- **Question it answers:** Why fill masked positions with `float('-inf')` rather than `-1e9`, and when does the choice bite?
- **Prereqs:** 03.3.
- **Experiment:** `T=4`. Build the mask with `torch.tril(torch.ones(T,T))` and `masked_fill(mask==0, float('-inf'))`. Show softmax puts exact 0.0 in the masked entries and the rows sum to exactly 1. Then redo with `-1e9` in fp32 (fine) and in fp16 (`-1e9` overflows fp16's ~65504 max → `-inf` anyway) and with `-1e4` in fp16 (leakage: masked entries get small but nonzero probability). Finally, construct an all-masked row and show `-inf` gives `nan` while a finite fill does not.
- **Shapes to nail:** mask `(T,T)` → broadcast to `(B,H,T,T)`; `torch.tril` gives lower-triangular *including* the diagonal — verify that the diagonal is 1, i.e. a token attends to itself.
- **Predict-first prompt:** After masking and softmax, what is the exact value at `attn[0,1]` (row 0, column 1)? And what is `attn[0].sum()`? Say the numbers before running.
- **Runs on:** MPS. Note MPS's fp16 behaviour may differ from CUDA — check the dtype before blaming the math.
- **Interview hooks:** Why is masking done *additively* on scores instead of multiplicatively on the post-softmax weights? What happens if you multiply the weights by 0 instead — does the row still sum to 1? Why do some implementations still prefer a large finite negative number? How does a padding mask interact with a causal mask when a whole row is masked (03.10 padding, 02.10)?
- **Common misconception:** That masking after softmax and renormalising is equivalent. It is numerically different and it silently changes what the model was trained with; the mask must go *before* the softmax.

### 03.5 — The `(T,T)` matrix: O(T²) time and memory
- **Question it answers:** Where exactly does quadratic cost come from, and is it time, memory, or both?
- **Prereqs:** 03.4.
- **Experiment:** For `T ∈ {128, 256, 512, 1024, 2048}` at `B=1, H=8, d_head=64`, compute (a) the analytic bytes of the `(B,H,T,T)` score tensor in fp32 and bf16, (b) measured peak memory and wall-clock (`torch.mps.synchronize()` before timing — MPS is async, a naive timer lies). Plot both against `T` on log-log and read off the slope.
- **Shapes to nail:** scores `(B, H, T, T)`; bytes = `B*H*T*T*dtype_size`. Compute it for `B=8, H=32, T=8192, bf16` and state the number in GB out loud.
- **Predict-first prompt:** Going from `T=1024` to `T=2048`: predict the factor increase in the score tensor's memory, and separately in the QK^T FLOPs. Are they the same factor? Now predict the factor for the *FFN* over the same change.
- **Runs on:** MPS (use `torch.mps.current_allocated_memory()`; on CUDA the equivalent is `torch.cuda.max_memory_allocated()`). Real Flash-Attention kernels are CUDA-only — measure the naive version here and reason about the fused one.
- **Interview hooks:** At long context, is attention compute-bound or memory-bandwidth-bound? What does Flash Attention change — the FLOP count, or the HBM traffic? Why does it still cost O(T²) time but O(T) memory? Rank the components of a forward pass by FLOPs at T=512 vs T=32768.
- **Common misconception:** That attention dominates transformer cost. At typical training lengths the FFN dominates FLOPs; attention dominates *memory* and only takes over FLOPs at long context — quantify the crossover in 03.21.

### 03.6 — Dropout on attention weights
- **Question it answers:** Where exactly is dropout applied in an attention block, and what does dropping a *weight* mean semantically?
- **Prereqs:** 03.5.
- **Experiment:** Apply `nn.Dropout(p=0.5)` to the post-softmax weights with `T=4`. Print a row in `train()` mode and in `eval()` mode. Verify the `1/(1-p)` rescaling at train time and that rows no longer sum to 1 after dropout.
- **Shapes to nail:** dropout applied to `(B,H,T,T)` after softmax, before `@ V`; a second dropout after the output projection.
- **Predict-first prompt:** After dropout on a softmax row that summed to 1, does the row still sum to 1? Should it? What does that do to the magnitude of the attended output?
- **Runs on:** MPS.
- **Interview hooks:** Why is inverted dropout (scale at train, identity at eval) the standard? Why do modern large-scale pretraining runs use `dropout=0.0` while fine-tuning runs do not? What does dropping an attention weight prevent the model from relying on? Where else does dropout appear in a GPT-2 block — name all the sites.
- **Common misconception:** That `model.eval()` is only about BatchNorm. Forgetting it makes attention stochastic at inference; this is a very common and very quiet bug.

## 3B. Multi-head attention

### 03.7 — Head splitting: the reshape/transpose gymnastics
- **Question it answers:** How do you get `H` independent attention computations out of one `(B,T,d_model)` tensor without a Python loop?
- **Prereqs:** 03.6; `view`/`reshape` vs `transpose`, and contiguity.
- **Experiment:** `B=2, T=4, d_model=8, n_heads=2` → `d_head=4`. One fused `nn.Linear(d_model, 3*d_model)` → split into Q,K,V → `.view(B, T, H, d_head)` → `.transpose(1, 2)` → `(B, H, T, d_head)`. **Verify the split is correct** by also computing head 0 with an explicit slice `Q[..., :4]` and asserting equality — this is the step everyone gets wrong silently.
- **Shapes to nail:** the full chain, said out loud in order: `(B,T,d_model) -> (B,T,3*d_model) -> 3×(B,T,d_model) -> (B,T,H,d_head) -> (B,H,T,d_head) -> scores (B,H,T,T) -> (B,H,T,d_head)`. And why `d_model = H * d_head` must hold exactly.
- **Predict-first prompt:** Why is `.view(B, H, T, d_head)` directly from `(B,T,d_model)` **wrong**, while `.view(B,T,H,d_head).transpose(1,2)` is right? Predict which elements end up in head 0 under each, then verify with an `arange` tensor.
- **Runs on:** MPS.
- **Interview hooks:** Why does `transpose` not copy memory, and what does `.contiguous()` cost? Why is a fused QKV projection faster than three separate `nn.Linear`s? Does splitting into heads change the total parameter count versus one head of size `d_model`? Why do heads keep `d_head` fixed (typically 64/128) and scale `H` instead?
- **Common misconception:** That `reshape` and `transpose` are interchangeable because both "rearrange dimensions". `transpose` changes the *logical* order with the same storage; `view` reinterprets the *existing* memory layout. Getting this backwards silently mixes data across heads and the model still trains — just worse.

### 03.8 — Merging heads and the output projection `W_O`
- **Question it answers:** After per-head attention, how do the heads recombine, and what is `W_O` for?
- **Prereqs:** 03.7.
- **Experiment:** `(B,H,T,d_head) -> .transpose(1,2) -> .contiguous() -> .view(B,T,d_model) -> nn.Linear(d_model, d_model)`. Drop the `.contiguous()` and observe the error. Then ablate `W_O = I` and describe what the block loses. Assemble the whole MHA module and assert it matches `torch.nn.functional.scaled_dot_product_attention(..., is_causal=True)` head-for-head.
- **Shapes to nail:** `(2,2,4,4) -> (2,4,2,4) -> (2,4,8) -> (2,4,8)`; `W_O` is `(d_model, d_model)`, i.e. `H*d_head -> d_model`.
- **Predict-first prompt:** Without `W_O`, the concatenated heads are just written into disjoint slices of the residual stream. What can `W_O` do that concatenation alone cannot?
- **Runs on:** MPS. `F.scaled_dot_product_attention` runs on MPS but selects a *different* backend than CUDA's Flash kernel — compare outputs with a loose `atol`, and do not read timings from it as if they were CUDA numbers.
- **Interview hooks:** Count MHA parameters: `4 * d_model^2` (+biases) — derive it. Why is `W_O` sometimes described as "the only place heads talk to each other"? What does the `W_V W_O` product mean as a single low-rank map per head (the OV circuit)?
- **Common misconception:** That concatenation is the combination step. Concatenation just stacks; `W_O` is the learned mixing, and it is a quarter of the attention parameters.

### 03.9 — The KV cache: where inference memory actually goes
- **Question it answers:** During generation, what do you store per token, and how big does it get?
- **Prereqs:** 03.8.
- **Experiment:** Implement naive incremental decoding twice: (a) re-run the full prefix each step, (b) cache K and V and pass only the new token's query. `B=1, T grows 1→16, d_model=8, H=2`. Assert the two produce identical logits. Time both. Then compute cache bytes analytically: `2 * n_layers * T * n_kv_heads * d_head * dtype_bytes * B`.
- **Shapes to nail:** cached `K,V` each `(B, H, T_so_far, d_head)`, growing by 1 along `dim=2` per step; the new query is `(B, H, 1, d_head)`; scores are `(B, H, 1, T_so_far)` — a *row*, not a matrix.
- **Predict-first prompt:** With a KV cache, what is the per-step attention cost — O(T) or O(T²)? And what is the *total* cost of generating T tokens? Compute the cache size for Llama-3-8B (`L=32, n_kv_heads=8, d_head=128, bf16`) at `T=8192, B=1` before you run the formula.
- **Runs on:** MPS.
- **Interview hooks:** Why is decoding memory-bandwidth-bound while prefill is compute-bound? At what batch size does the KV cache exceed the model weights? What are paged attention and prefix caching solving? Why can't you cache Q?
- **Common misconception:** That the KV cache is an optimisation you add later. It changes the shape of every tensor in the decode path and dictates the attention *architecture* — which is the next lesson.

### 03.10 — MHA vs MQA vs GQA vs MLA, motivated by cache bytes
- **Question it answers:** Why did the industry move off plain MHA, and what is each variant actually trading?
- **Prereqs:** 03.9.
- **Experiment:** One `Attention` module parameterised by `n_kv_heads`. `n_kv_heads = H` → MHA; `= 1` → MQA; `= H/g` → GQA. `B=1, T=4, d_model=8, H=4`, so `n_kv_heads ∈ {4, 1, 2}`. Implement the K/V expansion with `repeat_interleave` (or `expand` + reshape) and *state which one copies memory*. Tabulate for each variant: attention parameter count, KV-cache bytes at `T=8192`, and FLOPs. Then read the MLA idea on paper only: cache a low-rank latent `c_KV` per token and up-project per head, with a separate small decoupled-RoPE key carried alongside — implement a toy low-rank KV compression to feel the shape change, but do not build full MLA.
- **Shapes to nail:** `Q (B,H,T,d_head)` but `K,V (B, n_kv_heads, T, d_head)`; the expansion from `n_kv_heads` to `H` before the matmul; for the MLA toy, latent `c (B, T, d_c)` with `d_c << H*d_head`.
- **Predict-first prompt:** MQA cuts the cache by a factor of `H`. Does it cut the attention *FLOPs* by the same factor? Predict before you count. What is the parameter change?
- **Runs on:** MPS.
- **Interview hooks:** Why does GQA exist if MQA already minimises the cache — what is MQA's measured quality cost? Why is GQA cheap to obtain by up-training an existing MHA checkpoint? What does MLA's low-rank compression break about RoPE, and what is the decoupled-RoPE fix? At what serving regime (batch size, context length) does the variant choice stop mattering?
- **Common misconception:** That MQA/GQA are about saving parameters or training compute. They are almost entirely about *decode-time memory bandwidth*: fewer KV bytes to read per generated token.

## 3C. RoPE

### 03.11 — RoPE derived: rotating pairs of dimensions
- **Question it answers:** How can position be injected multiplicatively so the score depends only on `i-j`?
- **Prereqs:** 02.16, 02.17, 03.8; 2×2 rotation matrices.
- **Experiment:** By hand, `d_head=4` → 2 rotation pairs. Build `theta_k = base^(-2k/d_head)` with `base=10000`. Rotate `q` at position `m` and `k` at position `n`, and verify numerically that `rot(q,m) · rot(k,n)` depends only on `m-n`: hold `m-n=2` fixed while sliding both `m` and `n` upward, and show the score is constant.
- **Shapes to nail:** `q (…, d_head)` split into `d_head/2` pairs; `cos`/`sin` tables of shape `(T, d_head/2)` (or `(T, d_head)` if duplicated); the rotation is applied *per pair*, not across the whole vector.
- **Predict-first prompt:** A 2D rotation by angle `mθ` applied to `q` and by `nθ` applied to `k`. What is the dot product of the two rotated vectors in terms of the original dot product and the angles? Write the identity before running.
- **Runs on:** MPS. **Caution:** MPS float64 support is incomplete — keep the angle tables in float32 and verify with a generous `atol`, or compute the check on CPU.
- **Interview hooks:** Prove the relative property: `⟨R_m q, R_n k⟩ = ⟨R_{m-n} q, k⟩`. Why rotate pairs instead of scaling? What role does the frequency `base` play, and what does each pair "see" (fast pairs = local, slow pairs = long-range)? Why is RoPE applied to Q and K but not V?
- **Common misconception:** That RoPE is "just another vector added to the embedding". It is a norm-preserving rotation applied to Q and K *inside every layer*, not a one-time addition at the input.

### 03.12 — RoPE implemented and wired into MHA
- **Question it answers:** How does RoPE fit into the actual forward pass, and which of the two common implementations am I looking at?
- **Prereqs:** 03.11.
- **Experiment:** Precompute `cos/sin` for `max_T`, implement `apply_rope(x, cos, sin)` using the `rotate_half` formulation, apply to Q and K after projection and head-splitting. `B=2, T=4, d_model=8, H=2, d_head=4`. Then show the two pairing conventions — interleaved `(x0,x1),(x2,x3)` vs split-half `(x0,x_{d/2}),(x1,x_{d/2+1})` — produce different numbers, and note that HF checkpoints assume the split-half layout.
- **Shapes to nail:** `cos/sin (T, d_head)` → broadcast to `(1, 1, T, d_head)` to match `(B, H, T, d_head)`. The broadcast axes are the whole trick.
- **Predict-first prompt:** Does RoPE add any learnable parameters? Does it change the shape of Q or K? If neither, where is the position information physically stored in the tensor?
- **Runs on:** MPS.
- **Interview hooks:** Why does RoPE go after the Q/K projection rather than on the input embedding? Two checkpoints trained with different pairing conventions — what happens if you load one into the other's code, and how would you detect it? Why does RoPE need no separate positional embedding table at all?
- **Common misconception:** That RoPE is applied once at the embedding layer like sinusoidal encodings. It is re-applied in every layer's attention, to Q and K only.

### 03.13 — RoPE with the KV cache, and a preview of RoPE scaling
- **Question it answers:** Do you cache K before or after rotation, and how do models extend context beyond their trained length?
- **Prereqs:** 03.12, 03.9.
- **Experiment:** Extend the 03.9 cache to be RoPE-aware: at decode step `t`, the new key must be rotated at absolute position `t`, so cache the *rotated* K and track a position offset. Deliberately introduce the classic bug — rotate every cached key at position 0 each step — and show the generation degrade. Then, conceptually only, compute what `position_ids` beyond `max_T` do to the angles and list the scaling families: PI (linear position interpolation), NTK-aware base scaling, YaRN, and Llama-3.1's frequency-dependent scaling. No full implementation.
- **Shapes to nail:** cached rotated `K (B, n_kv_heads, T_so_far, d_head)`; new key `(B, n_kv_heads, 1, d_head)` rotated at position `T_so_far`; `position_ids (B, T)` as the explicit offset carrier.
- **Predict-first prompt:** Because RoPE's score depends only on `m-n`, could you instead cache *unrotated* K and rotate at read time? Would the result be identical? Which is cheaper?
- **Runs on:** MPS.
- **Interview hooks:** Why does the attention score degrade past the trained context length if RoPE is mathematically defined for all positions? What does PI trade away at short context, and why is NTK-aware scaling described as "interpolate the high frequencies, extrapolate the low ones"? How does context extension interact with the KV-cache budget from 03.9?
- **Common misconception:** That RoPE gives free length extrapolation. The rotation is defined everywhere, but the model has only ever seen a bounded range of relative angles; beyond that it is out of distribution.

## 3D. Normalisation and the residual stream

### 03.14 — LayerNorm from scratch
- **Question it answers:** What exactly does LayerNorm normalise over, and why not over the batch?
- **Prereqs:** 03.8; mean and variance.
- **Experiment:** Implement `(x - mean) / sqrt(var + eps) * gamma + beta` over the last dim. `B=2, T=4, d_model=8`. Assert equality with `nn.LayerNorm(8)` to `atol=1e-6`. Show the output row has mean ≈ 0 and std ≈ 1 *per token*. Then vary `eps` and show what happens when the input row is constant.
- **Shapes to nail:** input `(2,4,8)`; mean and var `(2,4,1)` with `keepdim=True`; `gamma, beta` each `(8,)` — `2*d_model` parameters, and notice how small that is versus the rest of the block.
- **Predict-first prompt:** `x` is `(2,4,8)`. Over which axis is the mean taken, and what is the shape of the mean tensor? How many independent normalisations happen in this forward pass?
- **Runs on:** MPS. Use the *biased* variance (`unbiased=False`) — this is the classic off-by-one when matching `nn.LayerNorm`.
- **Interview hooks:** Why is BatchNorm inappropriate for sequence models with variable lengths and batch-size-1 inference? Does LayerNorm depend on other tokens in the sequence? Other examples in the batch? What is `eps` protecting against numerically? Why does normalisation help optimisation at all?
- **Common misconception:** That LayerNorm normalises across the sequence. It normalises each token's feature vector independently — no cross-token and no cross-batch coupling, which is exactly why it suits autoregressive decoding.

### 03.15 — RMSNorm: drop the mean subtraction
- **Question it answers:** Why did Llama, Mistral, Qwen and friends all drop the centring step?
- **Prereqs:** 03.14.
- **Experiment:** Implement `x / sqrt(mean(x^2) + eps) * gamma`. Compare to LayerNorm on the same input: measure output difference, parameter count difference, and op count. Empirically check the claim that the hidden-state mean is already near zero at initialisation and stays small, by plotting the per-token mean through a few random layers.
- **Shapes to nail:** `mean(x^2)` is `(B,T,1)`; parameters are `gamma (d_model,)` only — `d_model` not `2*d_model`, and no `beta`.
- **Predict-first prompt:** If the input row already has mean ≈ 0, how different will RMSNorm's output be from LayerNorm's? Predict the relative error magnitude before measuring.
- **Runs on:** MPS.
- **Interview hooks:** What exactly does RMSNorm save — FLOPs, parameters, or memory passes? Which of those actually matters on a GPU? Why is re-centring argued to be unnecessary for the invariance property that makes normalisation work? Where does the norm's `gamma` end up in a quantised model, and why is it kept in higher precision?
- **Common misconception:** That RMSNorm is a quality improvement. It is essentially quality-neutral and adopted for cost and simplicity — say that plainly in an interview rather than claiming it is better.

### 03.16 — Pre-norm vs post-norm and the residual stream
- **Question it answers:** Why does every modern model normalise *before* the sublayer, and what is the "residual stream" view?
- **Prereqs:** 03.15.
- **Experiment:** Build both wirings for a 12-layer stack of toy sublayers: post-norm `x = Norm(x + Sublayer(x))` and pre-norm `x = x + Sublayer(Norm(x))`. Run a random input through both at init and plot the activation norm per layer. Backprop a scalar loss and plot gradient norm per layer. Show post-norm's gradient decay without warmup and pre-norm's growing residual norm.
- **Shapes to nail:** `x (B,T,d_model)` unchanged by every sublayer — that invariance *is* the residual stream. Note the final `Norm` needed before the LM head in a pre-norm model.
- **Predict-first prompt:** In pre-norm, there is an unbroken identity path from the input to the output with no normalisation on it. What does that do to the *magnitude* of `x` as depth grows, and what does it do to the relative contribution of layer 12 versus layer 1?
- **Runs on:** MPS.
- **Interview hooks:** Why did the original Transformer need learning-rate warmup and pre-norm models need it less? What is the residual stream as a "shared communication channel" and what do read/write operations mean in that framing? What is the growing-residual-norm problem and what do fixes like scaled residuals or a final norm address? What does sandwich/QK norm add?
- **Common misconception:** That pre-norm vs post-norm is a minor code-ordering detail. It changes the gradient path fundamentally; deep post-norm stacks are genuinely hard to train without careful warmup and initialisation.

## 3E. The feed-forward block

### 03.17 — The FFN and GELU
- **Question it answers:** What is the MLP doing, and why is it 4× wider than `d_model`?
- **Prereqs:** 03.16.
- **Experiment:** `nn.Linear(d, 4d) -> GELU -> nn.Linear(4d, d)` with `d_model=8`. Verify the shape round-trip. Plot GELU vs ReLU vs the `tanh` GELU approximation and compare near zero and in the negative tail. Count FFN parameters (`8*d^2`) against MHA's (`4*d^2`) and state the ratio.
- **Shapes to nail:** `(B,T,d) -> (B,T,4d) -> (B,T,d)`; per-token and position-independent — the FFN has no sequence axis interaction at all.
- **Predict-first prompt:** The FFN processes each token independently. Given that, which of the two sublayers is trivially parallel across the sequence, and which one is the reason you need a KV cache?
- **Runs on:** MPS.
- **Interview hooks:** Why is the FFN usually the FLOP majority of a transformer at moderate context? Why is 4× the convention, and what happens at 2× or 8×? What is the key-value memory interpretation of the FFN? Why is GELU smooth-at-zero preferred over ReLU here?
- **Common misconception:** That attention is where "the thinking" happens and the FFN is plumbing. The FFN holds roughly two-thirds of the parameters and is where most factual knowledge is stored.

### 03.18 — GLU variants, SwiGLU, and the 2/3 adjustment
- **Question it answers:** Why do modern FFNs have *three* weight matrices, and where does the `8/3 d_model` come from?
- **Prereqs:** 03.17.
- **Experiment:** Implement `SwiGLU(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down`. `d_model=8`. Show the elementwise gate. Then solve on paper: matching a 2-matrix `4d` FFN's `8d²` parameters with 3 matrices of hidden size `h` gives `3*d*h = 8d²` → `h = 8d/3`. Then check the real Llama-3-8B config: `hidden_size=4096` → `8/3 * 4096 = 10923`, but the config says `intermediate_size=14336`. Have the learner find the multiplier and the rounding rule that reconciles them (a ~1.3 FFN multiplier, rounded to a multiple of 1024).
- **Shapes to nail:** `W_gate, W_up` each `(d_model, h)`; `W_down` `(h, d_model)`; the gate product is elementwise at `(B,T,h)`.
- **Predict-first prompt:** Three matrices instead of two at the same hidden size `h` — what happens to parameters and FLOPs? By what factor must `h` shrink to break even? Derive it before looking at any config.
- **Runs on:** MPS.
- **Interview hooks:** What does the multiplicative gate let the FFN express that a fixed nonlinearity cannot? Why is `intermediate_size` always rounded to a multiple of 128/256/1024 — what hardware fact is that serving? Why does the GLU paper say the improvement comes with no principled explanation, and how do you talk about that honestly? Which is the correct parameter count for a SwiGLU FFN: `3*d*h` — now compute it for Llama-3-8B.
- **Common misconception:** That SwiGLU is "just a different activation function". It changes the *structure* — two parallel projections and an elementwise gate — and forces the hidden-size adjustment to stay parameter-neutral.

## 3F. Assembly and accounting

### 03.19 — One complete Transformer block
- **Question it answers:** Can I write a modern decoder block from a blank file, correctly, in one sitting?
- **Prereqs:** every lesson above.
- **Experiment:** Assemble: `x = x + Attn(RMSNorm(x))`, `x = x + SwiGLU_FFN(RMSNorm(x))`, with GQA + RoPE + causal mask + optional KV cache. `B=2, T=8, d_model=64, H=8, n_kv_heads=2, d_head=8`. Assert `out.shape == x.shape`. Stack 4 blocks and confirm shape invariance. Write a pytest in `tests/` covering: shape invariance, causality (perturb token `t` and assert outputs at positions `< t` are bit-identical), and cache/no-cache equivalence.
- **Shapes to nail:** `(B,T,d_model)` in and out, unchanged. Every intermediate shape recited from memory without looking.
- **Predict-first prompt:** The causality test perturbs `x[:, 3, :]` and checks outputs at positions 0-2. If you accidentally used a bidirectional mask, which assertion fails first — the shape test or the causality test?
- **Runs on:** MPS.
- **Interview hooks:** In what order do the sublayers, norms and residuals go, and what breaks in each wrong permutation? Which components have no parameters at all? Which are position-mixing and which are channel-mixing? How would you shard this block across 2 GPUs (tensor parallel) and where do the collectives land?
- **Common misconception:** That the residual add happens after the norm in a pre-norm block. The norm is applied to the sublayer's *input only*; the residual branch stays untouched.

### 03.20 — Counting parameters by hand
- **Question it answers:** Given a config, can I derive the parameter count without instantiating the model?
- **Prereqs:** 03.19.
- **Experiment:** Derive the per-block formula symbolically: attention `d*(H*d_head) + 2*d*(n_kv_heads*d_head) + (H*d_head)*d`; FFN `3*d*h`; norms `2*d`. Plus embedding `V*d` and an untied head `V*d`. Then verify against your own module with `sum(p.numel() for p in block.parameters())` for three different configs. Then do Llama-3-8B by hand: `d=4096, L=32, H=32, n_kv_heads=8, d_head=128, h=14336, V=128256, untied` — it should land at ≈8.03B. Also check `tie_word_embeddings` in the real configs of a ~1B model and an ~8B model and explain why they differ.
- **Shapes to nail:** every weight matrix in the block, listed with its shape, adding to the per-layer total (≈218M for Llama-3-8B) and the ×32 + embeddings total.
- **Predict-first prompt:** Before computing: of Llama-3-8B's 8B parameters, what fraction is attention, what fraction is FFN, and what fraction is the two embedding matrices? Guess to the nearest 10%, then compute.
- **Runs on:** CPU (arithmetic). Do **not** download an 8B checkpoint — this is a config exercise; `AutoConfig.from_pretrained` is enough if you want the numbers verified.
- **Interview hooks:** What fraction of an 8B model is embeddings, and how does that fraction move at 1B and at 70B? Why is GQA nearly free in parameters but decisive in cache? Given a 24GB card, what is the largest model you can hold in bf16 — and what is left for the KV cache?
- **Common misconception:** That "8B parameters" is mostly attention. FFN is roughly 2× attention per layer, and at small scale the embeddings dominate everything.

### 03.21 — FLOP counting for a real config (phase capstone)
- **Question it answers:** Given a model and a workload, can I predict its compute and say what is actually expensive?
- **Prereqs:** 03.20, 03.5, 03.9.
- **Experiment:** For Llama-3-8B, derive from first principles (a matmul `(m,k)@(k,n)` is `2*m*k*n` FLOPs): (1) forward FLOPs per token from the parameter matmuls, and check it against the `≈2N` rule of thumb; (2) the attention `QK^T` + `AV` term, which is *not* proportional to N — derive its per-token cost as a function of `L`, `T` and `d_model`; (3) the context length at which the attention term equals the parameter term (it lands in the tens of thousands of tokens — compute it exactly and sanity-check that at T=8192 attention is a minority but not a rounding error); (4) training cost via `≈6N` per token and the total for a 15T-token run; (5) arithmetic intensity at decode with batch size 1 and the resulting bandwidth bound. Write the whole thing up in `08_block/NOTES.md` **in your own words**.
- **Shapes to nail:** for each matmul in the block, its `(m,k,n)` and its `2mkn`; the per-layer FLOP table summing to the per-token total.
- **Predict-first prompt:** Where does the `6` in `6N` come from? Decompose it into forward and backward, and explain why backward is roughly 2× forward.
- **Runs on:** CPU (arithmetic). The *verification* against a real GPU's achieved FLOP/s needs an A100/H100 — do the derivation here and, if you want a measured MFU number, run one short benchmark on a Colab T4 or a rented instance; M3/MPS has no comparable published peak-FLOPs figure to compute MFU against honestly.
- **Interview hooks:** Derive `6N` from scratch. Why is MFU the metric and not FLOP/s? At batch size 1 decode, what is the arithmetic intensity, and why does that make the GPU idle on compute? Estimate the cost in GPU-hours of training an 8B model on 15T tokens at 40% MFU on H100s. Where does the 2× of activation checkpointing enter this accounting?
- **Common misconception:** That the FLOP count predicts the wall-clock. Decode is bandwidth-bound and the FLOP count is nearly irrelevant there — this is the single most important distinction between training and serving economics, and it is home turf for a distributed-systems engineer.

---

## Milestone

**`08_block/transformer_block.py`** — a single dependency-free (torch only) `TransformerBlock`
implementing pre-norm + RMSNorm + GQA + RoPE + causal masking + optional KV cache +
SwiGLU FFN, plus `03_tokenization/bpe.py`, a from-scratch byte-level BPE with
`train`, `encode` and `decode`.

It is done when all of the following hold:

1. `tests/test_bpe.py` passes: `decode(encode(s)) == s` for an adversarial string set
   (whitespace, emoji, multi-script, code indentation), and your `encode` matches
   HuggingFace's GPT-2 `input_ids` exactly on 20 sample strings.
2. `tests/test_block.py` passes: shape invariance `(B,T,d) -> (B,T,d)`; causality
   (perturbing token `t` leaves outputs at positions `< t` bit-identical); cached and
   uncached decoding produce identical logits; and your attention matches
   `F.scaled_dot_product_attention(..., is_causal=True)` within tolerance.
3. `08_block/NOTES.md` contains, in your own words, the parameter-count and FLOP-count
   derivation for Llama-3-8B with the numbers matching the published ~8.03B, plus the
   context length at which attention FLOPs overtake parameter FLOPs and the bandwidth
   argument for why decode is not compute-bound.

Written by you. Not pasted.
