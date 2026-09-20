# LLM engineering — a from-scratch learning workspace

A structured path from "I can code" to "I can implement, explain and serve LLMs", taught
interactively by Claude Code. It is a teaching harness, not a notes dump:

- `CLAUDE.md` defines *how* Claude must teach (tiny experiments, predict-before-run, shapes first).
- `.claude/skills/` are slash commands that enforce it.
- `curriculum/` says *what* is taught, in order.
- `interview/`, `projects/`, `career/` turn the learning into hiring evidence.

Built for engineers targeting senior LLM roles. It assumes you can program; it does **not**
assume any deep-learning background.

## Start here (new learner)

You need [Claude Code](https://claude.com/claude-code) and Python 3.10+.

```bash
git clone https://github.com/iamgovindthakur/llm-engineer-path.git && cd llm-engineer-path
scripts/setup.sh          # creates .venv, installs deps, runs the tests
scripts/start_fresh.sh    # resets progress to zero; moves the author's notebooks to reference/
```

Then edit [`LEARNER.md`](LEARNER.md) (be honest about your level), open the folder in Claude Code and run:

```
/progress          where am I, what is next
/teach <topic>     a full lesson
```

That is the whole loop.

## Skills

| Skill | Use it when |
| --- | --- |
| `/teach <topic>` | Learning something new — full 10-step lesson |
| `/lab <idea>` | One quick experiment, shapes first |
| `/quiz [topic]` | Testing whether you can actually defend it |
| `/check` | You wrote code and it is wrong (or you think it is) |
| `/sysdesign [prompt]` | Practising the system-design interview |
| `/recap` | Coming back after a gap |
| `/progress` | Status, and marking a lesson done |

## Layout

```
CLAUDE.md                  the teaching contract — read this first
LEARNER.md                 who you are and your level (per person)
INDEX.md                   one-line revision table of finished lessons (per person)
curriculum/
  ROADMAP.md               12 phases, the ordering argument, the timeline
  PROGRESS.md              what is actually done — source of truth (per person)
  phase-*.md               lesson-level plans
NN_topic/                  one directory per lesson, numbered in teaching order
  NN_topic.ipynb           the experiments
  NOTES.md                 your own words — you write this, Claude does not
common/                    device selection, shape printing, honest timing
tests/                     pytest checks that from-scratch code matches PyTorch
interview/                 question bank, system-design drills, weak spots
projects/                  the portfolio track
career/                    positioning notes
templates/                 blank versions of the per-person files
scripts/                   setup.sh, start_fresh.sh
reference/                 (after start_fresh) the author's worked notebooks — peek only after trying
```

## The three rules that make it work

1. **Predict before running.** Claude stops and asks for a shape or value before executing. Answer first.
2. **Write `NOTES.md` yourself.** Your own explanation is what survives to the interview.
3. **Build the project a phase unlocks** (`projects/PORTFOLIO.md`) before moving on.

## Hardware

Written on an Apple M3 (PyTorch MPS, no CUDA). Most lessons run on CPU/MPS/CUDA unchanged via
`common.get_device()`. Lessons that genuinely need NVIDIA hardware are marked in each phase file.

## Sharing your progress

Fork it and commit your own `curriculum/PROGRESS.md`, `LEARNER.md` and `NN_topic/NOTES.md`. Pull
requests that fix errors in the curriculum are welcome — the plan files were drafted quickly and
have not all been fact-checked, so treat technical claims skeptically and verify them.
