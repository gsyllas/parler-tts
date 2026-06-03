#!/bin/bash
#SBATCH --job-name=ptts-eval-wavs
#SBATCH --account=EUHPC_D29_081
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=leonardo/logs/%x_%j.out
#SBATCH --error=leonardo/logs/%x_%j.err
#
# Generate WAVs for post-training listening/eval.
# By default, each experiment writes ground_truth/, greedy/, and sample/ folders.
#
# Defaults to the v2 experiment set. Override with:
#
#   EXPS="multi_v2_llm greek_female_llm" sbatch --export=ALL leonardo/slurm/generate_eval_wavs.slurm
#
# Extra Python args can be passed with GEN_ARGS, e.g.
#
#   GEN_ARGS="--limit 16 --modes greedy --overwrite-manifest" \
#       sbatch --export=ALL leonardo/slurm/generate_eval_wavs.slurm
set -euo pipefail
module purge
module load cuda/12.2

cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source leonardo/env.sh

activate_conda_env
set_offline_mode
mkdir -p leonardo/logs

EXPS="${EXPS:-$EXTRA_EXPERIMENTS}"
GEN_ARGS="${GEN_ARGS:-}"

echo "[eval-wavs] EXPS       = $EXPS"
echo "[eval-wavs] GEN_ARGS   = $GEN_ARGS"
echo "[eval-wavs] HF_HOME    = $HF_HOME"
nvidia-smi -L || true

# shellcheck disable=SC2086
python leonardo/generate_eval_wavs.py --experiments $EXPS $GEN_ARGS

echo "[eval-wavs] DONE"
