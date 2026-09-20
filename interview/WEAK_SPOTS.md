# Weak spots

Maintained by `/quiz` and `/recap`. This is the targeting list: what gets
reviewed next, and what needs re-teaching rather than reviewing.

Be honest here. A weak spot you delete because it is embarrassing is one you
will discover live in an interview instead.

## Status key

- **shaky** — got there eventually, or with a hint. Recap it.
- **gone** — could not produce it. Re-teach with `/teach`, do not recap.
- **watch** — got it right, but it was recently learned and has not been tested cold.

---

| Topic | Status | First flagged | Last tested | Note |
| --- | --- | --- | --- | --- |
| Broadcasting rules | watch | 2026-09-20 | — | Never formally covered — Phase 1 backfill |
| Autograd / backprop | gone | 2026-09-20 | — | Never covered. Blocks Phase 4. |
| Cross-entropy from softmax + NLL | gone | 2026-09-20 | — | Never covered |
| Why the `sqrt(d_k)` scaling | watch | 2026-09-20 | — | Appears in `02_attention/`, derivation not attempted |

---

## Patterns

Notes on *how* things are being forgotten, which is more useful than the list
itself. Filled in by `/recap` once there is enough history.

_(nothing yet — needs a few recap sessions)_

## Added 2026-09-20 (prerequisite audit)

| Topic | Status | First flagged | Last tested | Note |
| --- | --- | --- | --- | --- |
| Variance of a sum of independent variables | gone | 2026-09-20 | — | Never taught anywhere; required by 03.2, 03.15, 01.20. New lesson 00.9 written to cover it. |
| Transpose as a view with permuted strides | gone | 2026-09-20 | — | 00.5 was skipped; it is the prerequisite for both `Q @ K^T` and 01.5 contiguity. Un-skipped. |
| View vs copy in indexing | gone | 2026-09-20 | — | 01.2 was not in the trimmed list; 01.5 assumes it. Added back. |
