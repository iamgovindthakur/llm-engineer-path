# Learner profile

Personal to one learner. Claude reads this at the start of every session, together with
`curriculum/PROGRESS.md`. New learner? Run `scripts/start_fresh.sh` and edit this file.

## Who I am, and what I am actually trying to do

I am an SDE-3 backend engineer (Java, distributed systems). I have real
production depth in backend/systems engineering and effectively **zero** depth
in deep learning internals.

My goal is to become an LLM / AI engineer who can:

- implement the important components from scratch
- explain the mathematics behind them
- reason about tensor shapes and computational + memory complexity
- reason about GPU cost, latency, throughput and memory bandwidth
- understand production tradeoffs and failure modes
- debug real implementations
- design and defend LLM systems in a staff-level interview

I am **not** trying to learn how to call APIs or glue high-level libraries.

> UNDERSTANDING > CODE COMPLETION

### Career context (read this before planning any lesson)

I am targeting senior/staff LLM-engineering roles at the top of the Indian
market. Be honest with me about that band rather than encouraging: roles at
that level are won on **demonstrated systems impact plus genuine model depth**,
not on course completion. Concretely that means this workspace has to produce
three things, not one:

1. **Depth** — I can derive and implement the internals.
2. **Evidence** — shippable projects with benchmarks, writeups, and a repo.
3. **Positioning** — my existing Java/distributed-systems depth is the
   differentiator, not a thing to hide. Serving, throughput, reliability and
   cost are my home turf; connect new concepts back to it whenever it is
   genuinely apt, and do not force the analogy when it is not.

See `career/POSITIONING.md` and `projects/PORTFOLIO.md`.

---

## My current level — do not overestimate it

Assume I am starting from approximately zero in deep learning and Transformer
internals.

I know Python basics. Do **not** assume I understand:

- advanced Python
- PyTorch internals
- tensor broadcasting
- matrix multiplication / linear algebra beyond the basics
- neural-network mathematics, autograd, backpropagation
- Transformer architecture

When introducing something unfamiliar, explain the prerequisite first.
Do not skip fundamentals because a topic is considered "basic".

**Already covered** (do not re-teach unless I ask):
token IDs, embedding lookup, `nn.Embedding`, batched lookup, a toy
string-to-ID vocabulary, and the distinction between dot product and matrix
multiplication. See `01_embeddings/`.

**In progress:** single-head causal self-attention. See `02_attention/`.
