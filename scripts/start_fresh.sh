#!/usr/bin/env bash
# For a NEW learner who cloned this repo: reset the personal files to blank templates and set the
# author's worked notebooks aside so you can do every lesson yourself.
#
# Safe to run once. It refuses to run twice so it can never wipe your own progress.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f .fresh_started ]; then
  echo "Already started fresh on $(cat .fresh_started). Refusing to overwrite your progress."
  exit 1
fi

mkdir -p reference
for d in [0-9][0-9]_*/; do
  [ -d "$d" ] && mv "$d" "reference/" && echo "moved $d -> reference/ (author's worked example)"
done

cp templates/LEARNER.md LEARNER.md
cp templates/PROGRESS.md curriculum/PROGRESS.md
cp templates/WEAK_SPOTS.md interview/WEAK_SPOTS.md
cp templates/INDEX.md INDEX.md
date +%F > .fresh_started

echo
echo "Fresh start ready. Now:"
echo "  1. Fill in LEARNER.md (be honest about your level)."
echo "  2. Open the folder in Claude Code and run /progress, then /teach."
