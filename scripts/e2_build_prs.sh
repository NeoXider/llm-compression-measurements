#!/usr/bin/env bash
# E2: build llama-bench for llama.cpp PRs #30792 and #30202 and their base commits (CPU-heavy; run when no GPU timing runs).
# Separate worktrees under <workdir>; the main fork checkout and branches are not modified.
set -u
R=<llama.cpp-checkout>
L=<release-root>/artifacts/e2_builds.log
declare -A REF=( [pr30792]=cq-pr-30792 [base30792]=99db719bdf07f7d537431ca556a0512bf3d995b6 [pr30202]=cq-pr-30202 [base30202]=dac308739429d60ccf99c46ecc1fbb1dff20ca77 )
for name in base30792 pr30792 base30202 pr30202; do
  wt=<llama.cpp-checkout>-e2-$name
  [ -d $wt ] || git -C $R worktree add --detach $wt ${REF[$name]} >> $L 2>&1
  cp <llama.cpp-checkout>-kvc/cq_build.bat $wt/cq_build_bench.bat
  sed -i 's/--target llama-perplexity llama-server llama-bench llama-cli/--target llama-bench/; s/build-kvc/build-e2/g' $wt/cq_build_bench.bat
  echo "build $name $(git -C $wt log -1 --format=%H) start $(date -Is)" >> $L
  timeout 5400 cmd //c "$(cygpath -w $wt/cq_build_bench.bat)" > $wt/cq_build_bench.log 2>&1
  echo "build $name exit=$? $(date -Is)" >> $L
done
