---
name: recap
description: Spaced-repetition review of previously learned LLM topics to fight forgetting. Use when the learner says "/recap", "review", "what have I forgotten", or at the start of a session after a gap of several days.
---

# Recap — fight the forgetting curve

Learning this material over months means the early phases decay while the later
ones are being built on top of them. Recap is the maintenance pass.

## Selecting what to review

Read `curriculum/PROGRESS.md` (completion dates) and `interview/WEAK_SPOTS.md`.
Pick 3-5 items, weighted:

- anything in `WEAK_SPOTS.md` — highest priority
- topics completed >2 weeks ago and not revisited
- prereqs of whatever they are about to learn next
- one item they have consistently got right, to confirm it has actually stuck

## Format — retrieval, not re-reading

Re-reading feels like learning and is not. Make them produce the answer.

For each item, pick one:

- **Derive it.** "Write the scaled dot-product attention equation and say why
  the scaling term is there."
- **Shape it.** "Input `(2, 128, 4096)`, 32 heads. Shape after head-splitting?"
- **Compute it.** A number, from memory, with the formula stated.
- **Debug it.** Show a 5-line snippet with one planted bug. Find it.
- **Explain it.** In two sentences, to a backend engineer who has not done ML.

Then grade: got it / shaky / gone.

## After

- Update `interview/WEAK_SPOTS.md`: remove what is solid, add what decayed.
- Anything marked **gone** gets a re-teach, not another recap. Offer `/teach`.
- Note the pattern if there is one — "you lose the derivations but keep the
  systems reasoning" is useful information for how they should study.

Keep it under 15 minutes. Recap is a warm-up, not the session.
