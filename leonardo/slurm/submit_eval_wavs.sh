#!/usr/bin/env bash
# Submit one independent eval-wav SLURM job per experiment, so they run in
# PARALLEL (one GPU each) instead of looping serially inside a single job.
#
# Usage:
#   bash leonardo/slurm/submit_eval_wavs.sh                       # all six v2
#   bash leonardo/slurm/submit_eval_wavs.sh multi_v2_llm          # one
#   bash leonardo/slurm/submit_eval_wavs.sh greek_male_det greek_male_llm
#
# Forward extra python args + tune the per-job wall time via env:
#   GEN_ARGS="--modes greedy" bash leonardo/slurm/submit_eval_wavs.sh
#   EVAL_TIME=02:00:00        bash leonardo/slurm/submit_eval_wavs.sh
set -euo pipefail

HERE_REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$HERE_REPO"
# shellcheck disable=SC1091
source leonardo/env.sh

# Default to the v2 set defined in env.sh.
if [ "$#" -gt 0 ]; then
  TARGETS=("$@")
else
  # shellcheck disable=SC2206
  TARGETS=($EXTRA_EXPERIMENTS)
fi

# One model at bs=1 is ~1h; 3h gives margin (model load + dataset decode + both
# decoding modes) and a tighter request schedules sooner than the 8h default.
EVAL_TIME="${EVAL_TIME:-03:00:00}"
GEN_ARGS="${GEN_ARGS:-}"

mkdir -p leonardo/logs

for exp in "${TARGETS[@]}"; do
  config="$(config_for "$exp")"
  if [ ! -f "$config" ]; then
    echo "[submit-eval] unknown/missing config for EXP=$exp ($config); skipping" >&2
    continue
  fi
  # EXPS/GEN_ARGS are set for this sbatch invocation; --export=ALL carries them
  # into the job (handles spaces in GEN_ARGS, unlike --export=ALL,VAR=...).
  jid=$(EXPS="$exp" GEN_ARGS="$GEN_ARGS" sbatch --parsable \
    --job-name="ptts-eval-$exp" \
    --time="$EVAL_TIME" \
    --export=ALL \
    leonardo/slurm/generate_eval_wavs.slurm)
  echo "[submit-eval] $exp -> job $jid  (--time=$EVAL_TIME)"
done

echo
echo "Submitted. Watch with: squeue -u \$USER"
