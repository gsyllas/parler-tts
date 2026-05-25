#!/bin/bash
#SBATCH --job-name=ptts-train
#SBATCH --account=EUHPC_D29_081
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=leonardo/logs/%x_%j.out
#SBATCH --error=leonardo/logs/%x_%j.err
#
# Generic parler-tts finetuning job. One A100, one experiment.
#
#   EXP=multi_llm sbatch --export=ALL,EXP=multi_llm leonardo/slurm/train.slurm
#
# EXP is one of: multi_det multi_llm female_det female_llm male_det male_llm
#
# QOS / wall-time defaults below are the normal 24h QOS, sized for the
# female/male runs (~30 epochs). The multi runs (~100 epochs) need longer:
# submit them on the long-production QOS, which allows up to 4 days:
#
#   EXP=multi_llm sbatch --qos=boost_qos_lprod --time=72:00:00 \
#       --export=ALL,EXP=multi_llm leonardo/slurm/train.slurm
#
# submit_all.sh applies these per-experiment flags automatically.
set -euo pipefail
module purge
module load cuda/12.2

cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source leonardo/env.sh

: "${EXP:?set EXP=multi_det|multi_llm|female_det|female_llm|male_det|male_llm}"
CONFIG="$(config_for "$EXP")"
if [ ! -f "$CONFIG" ]; then
  echo "[train] missing config $CONFIG" >&2
  exit 1
fi

activate_conda_env
set_offline_mode
mkdir -p "$TRAIN_ROOT/$EXP" leonardo/logs

echo "[train] EXP        = $EXP"
echo "[train] config     = $CONFIG"
echo "[train] base ckpt  = $(base_ckpt_for "$EXP")"
echo "[train] HF_HOME    = $HF_HOME"
echo "[train] WANDB_MODE = ${WANDB_MODE:-online}"
nvidia-smi -L || true
python - <<'PY'
import torch

print(f"[train] torch      = {torch.__version__}")
print(f"[train] cuda avail = {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("[train] ERROR: PyTorch cannot see CUDA; refusing to train on CPU.")
print(f"[train] cuda count = {torch.cuda.device_count()}")
print(f"[train] cuda dev 0 = {torch.cuda.get_device_name(0)}")
PY

accelerate launch --num_processes=1 \
  training/run_parler_tts_training.py "$CONFIG"

echo "[train] DONE: $EXP -> $TRAIN_ROOT/$EXP/output"
