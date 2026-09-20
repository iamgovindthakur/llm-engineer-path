# career/POSITIONING.md — repositioning from SDE-3 (Java) to LLM engineer

This is the market-facing companion to `projects/PORTFOLIO.md`. That file says
what to build; this one says how to be read correctly once you have built it.

Nothing here is motivational. Where a number is uncertain, it is marked as such.

---

## What you can claim after this phase

- You can state, in one sentence, what kind of LLM engineer you are — and it is
  not "generalist".
- You can name which of your existing skills transfer directly, and back each
  one with a specific artifact rather than an assertion.
- You know what senior and staff LLM screens actually test, and where your
  current gaps sit against them.
- You have resume bullets written as measured outcomes, not responsibilities.
- You have an accurate map of the employer archetypes, what each one screens
  for, and which of them fit your lane.

---

## 1. The repositioning, stated plainly

You are not becoming an ML researcher. The realistic and far stronger
destination is: **an infrastructure/serving-leaning LLM engineer who also
understands model internals.**

That is a real and under-supplied archetype. The supply glut is at the other end:
people who can build a RAG demo and call an API. The scarcity is at:

- people who can make an LLM system hold a latency and cost SLO under load
- people who can debug *why* quality regressed, with evidence
- people who can own a GPU budget

All three are distributed-systems problems wearing new vocabulary, and you
already do two of them for a living.

**Your one-sentence positioning, to be used verbatim:**

> "I'm a backend/distributed-systems engineer who went deep on LLM internals —
> I build the serving and evaluation layer for LLM systems, and I implement the
> model internals from scratch so I can reason about what the serving layer is
> actually scheduling."

The second clause is what stops you being filtered as "a Java guy learning AI".
It is only credible because `projects/P3` exists.

**The trap to avoid:** claiming to be an ML scientist. You will be asked about
optimisation theory, loss landscapes and paper reproduction, you will lose, and
the loss is unnecessary because that is not the job you want.

---

## 2. What transfers directly (and how to say it)

These are not analogies. They are the same skill with different nouns.

| What you already do | The LLM-world equivalent | How to say it in an interview |
| --- | --- | --- |
| Throughput vs latency tuning, thread pools, queue sizing | Continuous batching, admission control, KV-block budgets | "Batch size is a queueing decision. Raising it raises throughput and p99 together — it's the same curve as a thread pool, except the resource is KV-cache memory instead of threads." |
| p50/p95/p99, SLOs, error budgets | TTFT / TPOT / goodput SLOs | "Most LLM benchmarks report means. I report TTFT and TPOT at p95, because a mean hides exactly the requests that make users leave." |
| Backpressure, load shedding, circuit breakers | Overload behaviour of an inference server | "An LLM request holds GPU memory for its whole lifetime, so queueing under overload is worse than shedding — a queued request is a held KV allocation." |
| Caching, cache keys, invalidation, hit rate | KV cache, prefix caching, semantic caching, cache-aware routing | "Prefix caching is a cache-key design problem: the key is the longest shared token prefix, so system-prompt layout determines your hit rate." |
| Cost-per-request and capacity planning | Cost-per-million-tokens, GPU utilisation, MFU | "I can build the cost model: tokens → FLOPs → GPU-seconds → rupees, and tell you which term dominates." |
| Idempotency, retries, timeouts, cancellation | Streaming cancellation, retry-with-nondeterminism, tool-call safety | "Cancellation is a correctness issue in streaming inference. If the client disconnects and you don't abort, you burn GPU for nobody." |
| Observability, tracing, SLIs | LLM tracing, eval-in-CI, drift detection | "I treat eval as an SRE problem: measure suite variance first, because that sets the smallest regression you can actually gate on." |
| Data pipelines, schema evolution, idempotent ingest | Corpus ingestion, chunking, re-indexing | "Chunking is a schema decision, not a hyperparameter." |
| Threat modelling, least privilege, input validation | Prompt injection, excessive agency, tool sandboxing | "There's no reliable separation of instructions and data in one context window, so the defence has to be architectural — least privilege on the tool credentials, not a better system prompt." |

**How to deploy this:** never open with the analogy. Answer the LLM question
first, correctly and technically; *then* add the transfer as a closing sentence.
Leading with "this is just like Kafka consumer groups" reads as deflection from
someone who doesn't know the LLM answer. Following with it reads as depth.

---

## 3. What senior vs staff LLM roles actually screen for

The titles are not consistent across companies. Screen on the *job*, not the
title. Three distinct archetypes exist, and they interview completely differently.

**Applied AI / LLM product engineer** (the largest pool, and the most crowded)
- Screens: RAG architecture, eval design, structured outputs and tool calling,
  cost/latency control, prompt-vs-finetune judgement, shipping velocity.
- Typically *does not* deeply screen: CUDA, training, architecture internals.
- Your P8/P9/P10 cover this. Your Java background is neutral-to-mildly-positive.

**Inference / serving / LLM infra engineer** (smaller, hungrier, pays better per
unit of difficulty — **this is your lane**)
- Screens: KV cache mechanics and memory math, batching and scheduling,
  quantization tradeoffs, profiling, GPU memory hierarchy and bandwidth,
  vLLM/SGLang/TensorRT-LLM operational knowledge, cost-per-token ownership.
- Your P4/P5/P6 are exactly this, and your existing career is a genuine
  advantage rather than a thing to explain away.

**Research / training engineer** (smallest, hardest to enter laterally)
- Screens: training dynamics, distributed training, data curation, paper
  fluency, often a publication record.
- Realistic for you only after several years inside the field. Do not target it
  now; do not claim it.

**The senior → staff line**, in every archetype, is not technical depth. It is:

- **Scope:** you owned a system that other teams depended on, not a feature.
- **Ambiguity:** you chose the problem, not just the solution.
- **Leverage:** your work made other engineers faster or cheaper.
- **Judgement under cost:** you said no to something expensive and were right.
- **Written artifacts:** design docs and postmortems others acted on.

You almost certainly already clear this bar in your Java work. **The failure mode
for you is being downlevelled to mid-level because your LLM evidence is all
small projects and none of it shows scope.** The fix is to talk about scope from
your current job and depth from the portfolio, in the same answer — not to
pretend the portfolio is production scope.

---

## 4. Honest gap analysis and timeline

Where you are as of this file: embeddings done, single-head attention in
progress, no autograd/backprop, no Transformer, no training experience.

| Gap | Severity for your target lane | Closed by |
| --- | --- | --- |
| Backprop / autograd intuition | High — it is the standard "is this person real" probe | P1 (~2 weeks) |
| Transformer internals, shapes, complexity | Critical | Phases 2-3 + P3 (~3 months) |
| Inference mechanics: KV cache, batching, paging | Critical, and it is your lane | Phase 5 + P4 (~2 months) |
| Quantization and numerics | High | Phase 6 + P6 (~1.5 months) |
| Training/fine-tuning practice | Medium — you need to have done it once, credibly | Phase 7 + P7 (~2 months) |
| GPU/CUDA fluency | Medium-high for the serving lane | Partly unavoidable on an M3; buy rented-GPU hours (see below) |
| Kernel writing (Triton/CUDA) | Low for your target, high for a kernels role | Deliberately out of scope for now |
| Production LLM ownership | High, and **unfixable by side projects** | Only fixable at work — see §8 |
| Python depth | Medium | Absorbed along the way; do not study it separately |

**Realistic timeline at 8-10 h/week alongside a full-time SDE-3 job:**

- **Months 0-4:** Phases 0-3 done. P1 and P2 shipped. P3 in progress.
  You are interview-viable for junior-to-mid LLM roles, which you do not want.
- **Months 4-9:** Phases 4-6. P3, P4, P5 shipped. **This is the first point at
  which you are credible for a senior LLM-infra role**, because P5 combined with
  your actual job history is a coherent, non-junior story.
- **Months 9-15:** Phases 8-11. P6, P7, P9 shipped. Now credible for senior
  roles across all three archetypes, and competitive for staff *if* your current
  job supplies the scope evidence.
- **Months 15-24:** the staff-level move, which is won mostly on what you did at
  work in months 9-18, not on more projects.

Two honest caveats.

1. **At 4-5 h/week, double every number.** Consistency beats intensity here;
   two 90-minute sessions a week that actually happen beat a planned weekend
   that doesn't.
2. **The single largest accelerator is not this curriculum.** It is getting one
   LLM system into production *at your current job*. Six months of side projects
   is worth less than one quarter of "I own the inference path for X". Spend
   political capital on that. It is the difference between a lateral move at
   senior and a level-up to staff.

---

## 5. GitHub and READMEs as evidence

Hiring managers spend 60-120 seconds on your GitHub. Optimise for that, not for
a thorough reader.

**Profile README** — the only thing you can rely on being seen:
- One paragraph of positioning (§1's sentence, expanded slightly).
- Three pinned repos, each with the headline *number* in the description, not a
  description of the feature. "goodput flat at 3x offered load" beats "an LLM
  gateway with rate limiting".
- One line naming your background as the reason the serving work exists.
- No badge walls, no skill icons, no "currently learning" list.

**Repo README template** — same structure every time:
1. **What this is** — two sentences.
2. **The headline result** — one table or one plot, above the fold.
3. **Reproduce it** — exact commands, exact device, exact seed.
4. **The tradeoff** — what this design costs, stated as a table.
5. **What I got wrong** — the negative result, the config that lost, the bug
   that took three days. This is the section senior reviewers actually read,
   and the one that signals you are not narrating a tutorial.
6. **Limitations** — what these numbers do not support.

**Rules:**
- Every number states its device. An MPS number and an A100 number never share
  a column.
- Commit the raw result JSON and the plotting script, not just the PNG.
- Commit history should show incremental work over weeks. A single "initial
  commit" of a finished project reads as copied.
- Write a short writeup per flagship project and publish it somewhere with a
  URL. The writeup is what gets forwarded internally; the repo is what gets
  checked afterwards.
- Do not pad the profile with forked tutorials or half-finished repos. Three
  strong repos beats eleven; archive the rest.

---

## 6. Resume bullets: before and after

The rule: **a bullet is a measured outcome with a mechanism.** Responsibilities
are invisible; numbers without mechanism read as inherited; mechanism without
numbers reads as a tutorial. You need both.

**1. Reframing existing Java work so it reads as AI-adjacent infrastructure**
- *Before:* "Developed and maintained microservices in Java and Spring Boot for
  the payments platform."
- *After:* "Owned a Java/Spring service at ~12k req/s; cut p99 from 840 ms to
  310 ms by replacing synchronous fan-out with a bounded-concurrency pipeline and
  right-sizing pools against measured service-time distribution — the same
  queueing analysis I now apply to LLM batch-size and KV-budget tuning."

**2. The serving project**
- *Before:* "Built an LLM API gateway using Spring Boot and Spring AI."
- *After:* "Built a JVM inference gateway (Spring Boot + SSE) over a vLLM
  backend: token-denominated rate limiting with an output-length estimator
  (p95 error 18%), deadline-aware admission control that held goodput flat to 3x
  offered load where the unprotected baseline collapsed, and cancellation
  propagation that cut wasted backend generation after client disconnect by ~95%."

**3. The from-scratch model**
- *Before:* "Implemented a GPT model from scratch to learn Transformer
  architecture."
- *After:* "Implemented a 15M-parameter decoder-only LM from scratch (RoPE,
  RMSNorm, SwiGLU, GQA) and trained it on Apple M3/MPS; ran a 4-way architecture
  ablation at fixed token budget and reconciled a predicted 6·N·D FLOPs model and
  a 16 bytes/param AdamW memory model against measured throughput and peak
  memory."

**4. The quantization work — note that the finding is the bullet**
- *Before:* "Experimented with model quantization techniques including GPTQ and
  AWQ."
- *After:* "Benchmarked 6 quantization configs of a 3B model across perplexity,
  a multiple-choice benchmark and a generative eval; showed the three metrics
  rank configurations differently (INT4 moved perplexity <1% but the generative
  score 8%), and published a memory/latency/quality decision table with per-seed
  variance so sub-noise 'wins' are visible as noise."

**5. The RAG work — separate the metrics, that is the whole signal**
- *Before:* "Built a RAG pipeline with vector search and reranking that improved
  answer quality."
- *After:* "Built a hybrid-retrieval RAG system over a 40k-chunk corpus with a
  100-query hand-labelled golden set; measured retrieval (recall@20 0.84,
  nDCG@10 0.71) separately from generation (groundedness 0.91, citation accuracy
  0.86), which localised 20% of end-to-end failures to retrieval recall rather
  than prompting, and showed the cross-encoder reranker bought +0.09 nDCG for
  +80 ms p95."

**6. The eval/ops work**
- *Before:* "Set up evaluation and monitoring for LLM applications."
- *After:* "Built an eval-in-CI gate for two LLM services: measured suite
  run-to-run variance to derive a minimum detectable effect, calibrated an
  LLM judge against 180 hand-labelled examples (kappa 0.71 after correcting a
  position bias worth 6 points), and blocked a prompt change that regressed
  groundedness 11%."

**Anti-patterns to delete from any draft:** "passionate about AI", "hands-on
experience with LangChain/LlamaIndex/OpenAI API" as a skills line,
"worked on GenAI POCs", tool lists without outcomes, and any bullet whose verb
is "utilised".

---

## 7. Where these roles are, and what each archetype wants

| Archetype | Examples | What they want | Notes |
| --- | --- | --- | --- |
| **Global product companies / top GCCs** | Google, Microsoft, Adobe, Atlassian, Uber, Salesforce, Nvidia India | Strong general engineering + one deep ML area; leetcode + system design still present | Largest volume of genuinely senior roles. Your Java background is a *positive* here. These companies sort on level, not on the AI label, so the lever is level. |
| **AI-first Indian product companies** | Sarvam AI, Krutrim, and the next tier of well-funded startups | Depth + ownership + willingness to be scrappy | Best place to get real production LLM ownership fast — which is the thing your resume most lacks. |
| **Frontier labs with India presence** | OpenAI, Anthropic (Bengaluru office opened early 2026), Google DeepMind India | Exceptional depth, or exceptional applied/forward-deployed skill | Very few seats, extremely high bar, and many roles are applied/forward-deployed rather than research. Treat as a 2-3 year target, not a month-6 target. |
| **Infra / serving startups** | Inference providers, GPU clouds, eval and observability startups (including US companies hiring remote from India) | Exactly your lane: serving, throughput, cost | Highest fit for P4/P5/P6. US-remote roles screen on demonstrable depth rather than pedigree, which favours you. |
| **Java-adjacent enterprise AI** | Banks, insurers, large SaaS, consultancies building Spring AI / LangChain4j systems | Someone who can put an LLM into a regulated JVM stack safely | The *easiest* door for you and the fastest route to production ownership. Lower ceiling. Use it as a bridge role if you take it, not a destination. |
| **Services / IT companies** | Large Indian IT services "GenAI CoE" roles | Broad, shallow, delivery-led | Avoid unless the specific team genuinely owns a product. The title inflation here is severe. |

**What actually decides where you land**, in order:

- **Level.** Staff/principal at a global product company or top GCC. This tracks
  the company's level ladder rather than the AI label.
- **Scarce scope.** You own a system the business measurably depends on, and a
  small number of employers are bidding for the specific person who has done it.
- **A frontier-lab seat**, where the bar is exceptional and the seats are few.
- **US-linked roles**, via a US-remote position.

What does *not* decide it: a strong portfolio, a certificate, or knowing more
about attention than the next candidate. Depth gets you into the room and gets
you levelled correctly. **Scope and demonstrated impact decide the rest.** You
already have scope in Java; the whole game is transferring it, not rebuilding it.

Verify any market claim you read against people one level above you in your own
network. Published figures for this market are mostly SEO content with no
methodology, and they skew high.

---

## 8. The first 90 days

Two tracks, run in parallel. The work track matters more than the study track
and is the one most people skip.

**Days 1-30 — depth and a visible start**
- Finish Phase 1 and get through causal self-attention, multi-head attention and
  RoPE in Phase 3. Non-negotiable: shapes written out by hand before any code.
- Ship **P1 (autodiff)**. It is small and it closes the biggest credibility gap.
- Set up the GitHub profile README now, with P1 pinned. Publishing early forces
  the README discipline while the stakes are low.
- **At work:** find the LLM-shaped problem that already exists — an internal
  search, a support-ticket triage, a doc Q&A, anything. Identify who owns it.
  Do not ask for a transfer; ask what is blocking it.

**Days 31-60 — the first real artifact and the first external signal**
- Finish Phase 3. Start **P3 (micro-gpt)**; ship **P2 (bpe-lab)** alongside it —
  P2 is a weekend and its fertility numbers are an unusually good conversation
  opener in the Indian market.
- Write one public technical post from P2. Not a tutorial — the measurement and
  its cost implication. One good post outperforms twenty applications.
- **At work:** propose a scoped, two-week LLM piece of work with a measurable
  outcome. Frame it in reliability/cost terms, which is language your management
  already buys. Get it on a roadmap.
- Start a loose conversation with 3-5 people already doing this in Bengaluru or
  Hyderabad. Ask what their team screens for. Do not ask for referrals yet —
  you have nothing to refer.

**Days 61-90 — the lane becomes visible**
- Finish P3 with the ablation table. This is the single most important artifact
  in the first 90 days.
- Start Phase 5 and begin **P5 (jvm-llm-gateway)** early — you can build most of
  it with your existing skills while Phase 5 teaches you what it is scheduling,
  and it is the project with the highest return per hour for you specifically.
- Rewrite the resume using §6's format. Have someone who hires for these roles
  read it and tell you what they think you do. If the answer is not
  "LLM serving/infra with real systems depth", the positioning has failed and
  the fix is the resume, not more projects.
- **At work:** ship the two-week piece and write it up internally. This becomes
  a resume bullet with production scope, which no side project can give you.
- Rent GPU hours once, deliberately — a few hours on an A100 (community-tier
  rates were around **$1.19-1.49/GPU-hr**, with H100s from roughly $1.99/hr) so
  that "I have run on CUDA" is true and so the M3-only gap stops being a
  blocker in your head. Budget for the entire portfolio is roughly **$100-200**.

**What success at day 90 looks like:** Phase 3 complete, three repos public, one
public writeup, one LLM thing shipped at work however small, and a resume that a
stranger reads correctly in 30 seconds. Not a job offer — it is too early, and
interviewing before P3 exists spends your network on a weaker version of you.

---

## 9. Things to say, and things not to say

**Say:**
- "I don't know" — immediately, then reason out loud toward the answer.
- "I measured it" — with the device, the seed and the variance.
- "That configuration lost, and here's why I shipped the other one."
- "The retrieval was the bottleneck, not the prompt, and here's the number."
- "Here's the cost model for that: tokens → FLOPs → GPU-seconds → rupees."

**Don't say:**
- "I'm transitioning into AI." You are an engineer who builds LLM systems.
  The word *transitioning* invites downlevelling.
- "I've been learning AI for N months." Nobody asks how long; saying it makes
  the number the subject.
- "I used LangChain." Say what you built and what it cost.
- Anything about a course, a certificate, or a number of tutorials completed.
- An analogy to your Java work *before* you have answered the actual question.

---

## Milestone

A single-page positioning artifact you can send cold: a resume rewritten in §6's
format, a GitHub profile README pinning three projects with numbers in their
descriptions, and one published technical writeup. The phase is done when a
senior engineer who does not know you reads all three in under five minutes and
describes you as "an infra person who understands the models" — unprompted.
