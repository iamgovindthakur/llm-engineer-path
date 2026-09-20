---
name: progress
description: Update the curriculum tracker and report status and next steps. Use when the learner says "/progress", "where am I", "what's next", "mark X done", or at the end of a lesson.
---

# Progress

`curriculum/PROGRESS.md` is the source of truth for what has actually been
learned. It is not a to-do list — it is evidence.

## Marking something done

A lesson is done when **all** of these hold. Do not mark it done otherwise,
and say which one is missing if they ask you to.

- [ ] The experiment ran and they read the output
- [ ] They answered the predict-first question (right or wrong — they attempted it)
- [ ] They can state the shapes involved without looking
- [ ] They attempted the interview hooks
- [ ] `NOTES.md` exists in the lesson directory, written by them

Record: the date, the lesson directory, a one-line "what I can now do", and any
question they left open.

## Reporting status

When asked where they are:

1. Current phase and position within it.
2. What was finished most recently, and when.
3. **The next lesson**, with its prereqs confirmed met.
4. Anything in `interview/WEAK_SPOTS.md` that should be recapped first.
5. Which portfolio project (`projects/PORTFOLIO.md`) the current phase unlocks.

Be accurate rather than encouraging. If three weeks have passed with two
lessons finished, say that — a tracker that flatters them is worthless. If they
has skipped prereqs to chase a later topic, name it.

## Open questions

Keep a `## Open questions` section in `PROGRESS.md` for things they asked that
were deferred. Surface any that the current lesson now makes answerable — that
is how a deferred question becomes a satisfying moment rather than a loose end.
