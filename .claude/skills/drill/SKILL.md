---
name: drill
description: Run a timed from-scratch implementation drill from interview/CODING_DRILLS.md — blank file, no notes, shapes stated before code, one test that catches the likely bug. Use when the learner says "/drill", "timed drill", "let me try implementing X cold", or at the end of a phase as step 1 of the cadence in ROADMAP.md.
---

# Drill — timed from-scratch implementation

The cadence in `curriculum/ROADMAP.md` puts a timed drill first after every
phase. This is that step. Understanding a lesson and producing it in 25 minutes
under pressure are different skills; this trains the second.

## Picking the drill

Read `interview/CODING_DRILLS.md`. Only offer drills whose **"After"** lesson is
already marked done in `curriculum/PROGRESS.md` — never drill untaught material.

Prefer, in order:
1. A drill they have never run.
2. A drill they ran more than a week ago (the second run is the real score).
3. A drill tied to something in `interview/WEAK_SPOTS.md`.

State which drill, its target time, and the test it must pass. Then start.

## The rules, stated to them once

- Blank file. No notes, no earlier notebooks, no copy-paste.
- **Shapes first.** Before any code they say the shape of every input,
  intermediate and output. Do not let them skip this — it is the whole point.
- They write one test that would catch the most likely bug, not a test that
  passes trivially.
- You write **no code** while the clock runs. Not a skeleton, not a signature,
  not a "here's the shape of it". Nothing.

## While the clock runs

- Note the start time. Report elapsed at the halfway mark and at the target.
- If they stall past the target, give **one** nudge pointing at the stuck line —
  a question, not the fix.
- If they ask "is this right?", answer only "run your test" until the target
  time has passed.
- Say the clock out loud the way an interviewer would: "twelve minutes in,
  no output yet — commit to something runnable."

## After

Run their code. Then:

- **Did the test catch anything?** If their test passed but the implementation
  is wrong, that is the more important finding — say so, and show the input
  that breaks it without fixing the code.
- **Review like `/check`:** find the bug, point at the line, do not rewrite.
- **Push once:** can it be done in less memory? What breaks at batch size 1,
  or sequence length 1, or an empty input?

Then append a row to the score log at the bottom of `interview/CODING_DRILLS.md`:
date, drill number, minutes taken, whether the test passed, and one line on what
went wrong. Add anything shaky to `interview/WEAK_SPOTS.md`.

Close with one sentence: would this have cleared a senior bar, honestly.
