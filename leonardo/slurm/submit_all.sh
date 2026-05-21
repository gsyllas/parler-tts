#!/usr/bin/env bash
# Submit one independent SLURM job per finetuning experiment.
#
# Usage:
#   bash leonardo/slurm/submit_all.sh                  # all six
#   bash leonardo/slurm/submit_all.sh multi_llm        # one
#   bash leonardo/slurm/submit_all.sh female_det male_det   # a subset
#
# Each job runs on a single A100 and is queued independently (no dependencies).
set -euo pipefail

HERE_REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$HERE_REPO"
# shellcheck disable=SC1091
source leonardo/env.sh

# Default to the full set defined in env.sh.
if [ "$#" -gt 0 ]; then
  TARGETS=("$@")
else
  # shellcheck disable=SC2206
  TARGETS=($EXPERIMENTS)
fi

mkdir -p leonardo/logs

for exp in "${TARGETS[@]}"; do
  config="$(config_for "$exp")"
  if [ ! -f "$config" ]; then
    echo "[submit] unknown/missing config for EXP=$exp ($config); skipping" >&2
    continue
  fi
  jid=$(sbatch --parsable \
    --job-name="ptts-$exp" \
    --export=ALL,EXP="$exp" \
    leonardo/slurm/train.slurm)
  echo "[submit] $exp -> job $jid"
done

echo
echo "Submitted. Watch with: squeue -u \$USER"
