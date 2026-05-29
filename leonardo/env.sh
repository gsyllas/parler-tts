#!/usr/bin/env bash
# Shared config for all parler-tts leonardo scripts.
# Sourced by login-node helpers and by SLURM jobs.
#
# Mirrors the conventions proven in ~/dataspeech/leonardo/env.sh:
#   - in-repo conda env (no $HOME quota)
#   - in-repo HF / torch caches
#   - single source of truth for paths + the per-experiment (EXP) resolver
#
# Override any of these by exporting them before sourcing, or by editing this
# file. Per-job runtime overrides (e.g. EXP=multi_llm sbatch ...) are honored
# via the `${VAR:-default}` pattern.

set -u

# ---- account / partition --------------------------------------------------
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-EUHPC_D29_081}"
export SLURM_PARTITION="${SLURM_PARTITION:-boost_usr_prod}"

# ---- repo / cache / data --------------------------------------------------
# REPO_ROOT is the absolute path to the cloned parler-tts repo on /leonardo_work.
# It must NOT be in $HOME (no quota). Default: parent dir of this file's parent
# (i.e. the repo root, since this file lives in <repo>/leonardo/env.sh).
# Ignore inherited REPO_ROOT values from sibling projects such as dataspeech.
_parler_script_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -n "${REPO_ROOT:-}" ] && [ "$REPO_ROOT" != "$_parler_script_repo_root" ]; then
  echo "[env.sh] ignoring inherited REPO_ROOT=$REPO_ROOT; using $_parler_script_repo_root" >&2
fi
REPO_ROOT="$_parler_script_repo_root"
export REPO_ROOT
unset _parler_script_repo_root

# Conda env lives INSIDE the repo (per user requirement: no $HOME space).
# Ignore inherited CONDA_ENV_PREFIX values from sibling projects; otherwise a
# shell that starts inside the dataspeech env will keep re-activating dataspeech.
_parler_default_conda_env="$REPO_ROOT/.conda/parler-tts"
if [ -n "${CONDA_ENV_PREFIX:-}" ] && [ "$CONDA_ENV_PREFIX" != "$_parler_default_conda_env" ]; then
  case "$CONDA_ENV_PREFIX" in
    "$REPO_ROOT"/*) ;;
    *)
      echo "[env.sh] ignoring inherited CONDA_ENV_PREFIX=$CONDA_ENV_PREFIX; using $_parler_default_conda_env" >&2
      CONDA_ENV_PREFIX="$_parler_default_conda_env"
      ;;
  esac
fi
export CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-$_parler_default_conda_env}"
unset _parler_default_conda_env

# All HF / torch caches live next to the repo on /leonardo_work. Some login-node
# shells already export HF_HOME / TORCH_HOME from old projects; ignore those
# inherited values so this pipeline does not fill the wrong quota. To
# intentionally move every cache, set CACHE_ROOT before sourcing env.sh.
export CACHE_ROOT="${CACHE_ROOT:-$REPO_ROOT/cache}"
_parler_note_ignored_cache_var() {
  local name="$1"
  local value="${!name:-}"
  local expected="$2"
  if [ -n "$value" ] && [ "$value" != "$expected" ]; then
    echo "[env.sh] ignoring inherited $name=$value; using $expected" >&2
  fi
}
_parler_note_ignored_cache_var HF_HOME "$CACHE_ROOT/hf"
_parler_note_ignored_cache_var TORCH_HOME "$CACHE_ROOT/torch"
_parler_note_ignored_cache_var PIP_CACHE_DIR "$CACHE_ROOT/pip"
export HF_HOME="$CACHE_ROOT/hf"
export HF_HUB_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$CACHE_ROOT/torch"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
unset -f _parler_note_ignored_cache_var
# Keep conda's package + env metadata caches in the repo too (don't pollute $HOME).
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$REPO_ROOT/.conda/pkgs}"
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-$REPO_ROOT/.conda/envs}"

# wandb runs offline on compute nodes; sync from a login node afterwards.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-parler-tts-greek-finetune}"
export WANDB_DIR="${WANDB_DIR:-$REPO_ROOT/leonardo/logs}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$CACHE_ROOT/wandb}"

# ---- data -----------------------------------------------------------------
# Root of the dataspeech outputs produced by the sibling dataspeech pipeline.
export DATA_ROOT="${DATA_ROOT:-/leonardo_work/EUHPC_D29_081/gsyllas0/data/tts}"
# Named-multi outputs (deterministic + LLM prompts, with speaker names).
export NAMED_OUT_ROOT="${NAMED_OUT_ROOT:-$DATA_ROOT/dataspeech_out_named}"
# Single-speaker female/male outputs.
export OUT_ROOT="${OUT_ROOT:-$DATA_ROOT/dataspeech_out}"

# Where trained checkpoints + DAC token buffers are written (on /leonardo_work).
export TRAIN_ROOT="${TRAIN_ROOT:-$REPO_ROOT/leonardo/runs}"

# ---- base checkpoints -----------------------------------------------------
export BASE_MULTILINGUAL="${BASE_MULTILINGUAL:-parler-tts/parler-tts-mini-multilingual-v1.1}"
export BASE_GREEK_DET="${BASE_GREEK_DET:-gsyllas/parler-tts-mini-multilingual-to-greek-v1.1_deterministic_60_epochs}"
export BASE_GREEK_LLM="${BASE_GREEK_LLM:-gsyllas/parler-tts-mini-multilingual-to-greek-v1.1_52_epochs_speakers}"

# Shared tokenizers / feature extractor. The two text inputs need DIFFERENT
# tokenizers (set explicitly per-config in leonardo/configs/*.json):
#   - PROMPT (spoken transcript): parler-tts-mini-multilingual-v1.1. flan-t5-large
#     cannot encode Greek (every Greek word -> <unk>) and silently empties it.
#   - DESCRIPTION (style text): google/flan-t5-large, to match the model's flan-t5
#     text encoder (vocab 32128). The multilingual tokenizer emits higher token
#     ids that overflow that encoder's embedding -> CUDA device-side assert.
# TEXT_TOKENIZER below is the DESCRIPTION tokenizer (the prompt one is the model).
export FEATURE_EXTRACTOR="${FEATURE_EXTRACTOR:-parler-tts/dac_44khZ_8kbps}"
export TEXT_TOKENIZER="${TEXT_TOKENIZER:-google/flan-t5-large}"
# Greek WER ASR model (distil-large-v2 is English-only).
export ASR_MODEL="${ASR_MODEL:-openai/whisper-large-v3}"
export CLAP_MODEL="${CLAP_MODEL:-laion/larger_clap_music_and_speech}"

# ---- experiment ids -------------------------------------------------------
# The original six finetuning experiments. Each maps to a config JSON + base checkpoint.
export EXPERIMENTS="${EXPERIMENTS:-multi_det multi_llm female_det female_llm male_det male_llm}"
export EXTRA_EXPERIMENTS="${EXTRA_EXPERIMENTS:-multi_v2_det multi_v2_llm greek_female_det greek_female_llm greek_male_det greek_male_llm}"

# Resolve the base checkpoint for an experiment id. Pass id as $1.
base_ckpt_for() {
  case "${1:-}" in
    multi_det|multi_llm|multi_v2_det|multi_v2_llm) echo "$BASE_MULTILINGUAL" ;;
    female_det|male_det|greek_female_det|greek_male_det) echo "$BASE_GREEK_DET" ;;
    female_llm|male_llm|greek_female_llm|greek_male_llm) echo "$BASE_GREEK_LLM" ;;
    *) echo "[env.sh] unknown EXP='${1:-}'; expected one of: $EXPERIMENTS $EXTRA_EXPERIMENTS" >&2; return 1 ;;
  esac
}

# Resolve the config JSON for an experiment id. Pass id as $1.
config_for() { echo "$REPO_ROOT/leonardo/configs/${1:?EXP required}.json"; }

# ---- dataset path helpers -------------------------------------------------
# multi: native train/eval splits; audio in 01_hf_dataset, descriptions in 04x.
multi_audio_dir()      { echo "$NAMED_OUT_ROOT/multi/01_hf_dataset"; }
multi_det_meta_dir()   { echo "$NAMED_OUT_ROOT/multi/04a_prompts_deterministic"; }
multi_llm_meta_dir()   { echo "$NAMED_OUT_ROOT/multi/04b_prompts_llm"; }

# female/male: no eval split; audio in 01_hf_dataset, descriptions in 04x.
# 03_make_eval_splits.py merges them into 05_merged_{det,llm} (train+eval).
sm_audio_dir()    { echo "$OUT_ROOT/${1:?dataset}/01_hf_dataset"; }
sm_det_meta_dir() { echo "$OUT_ROOT/${1:?dataset}/04a_prompts_deterministic"; }
sm_llm_meta_dir() { echo "$OUT_ROOT/${1:?dataset}/04b_prompts_llm"; }
sm_merged_dir()   { echo "$OUT_ROOT/${1:?dataset}/05_merged_${2:?prompt-type}"; }

# ---- helpers --------------------------------------------------------------
# Activate the in-repo conda env on Leonardo. Looks for conda in common spots.
activate_conda_env() {
  local conda_sh=""
  for c in \
    "$REPO_ROOT/.conda/miniforge/etc/profile.d/conda.sh" \
    "$HOME/miniforge3/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "/leonardo/prod/opt/tools/miniconda3/2024.06/none/etc/profile.d/conda.sh"; do
    if [ -f "$c" ]; then conda_sh="$c"; break; fi
  done
  if [ -z "$conda_sh" ]; then
    echo "[env.sh] could not find conda.sh; install miniforge first (see leonardo/login/00_setup_conda_env.sh)" >&2
    return 1
  fi
  # shellcheck disable=SC1090
  source "$conda_sh"
  conda activate "$CONDA_ENV_PREFIX"

  # TorchCodec/datasets dlopen FFmpeg at runtime. On Leonardo batch nodes the
  # dynamic linker can otherwise pick the system libstdc++ before conda's newer
  # runtime, which breaks conda-forge FFmpeg with GLIBCXX_* errors.
  local conda_lib="${CONDA_PREFIX:-$CONDA_ENV_PREFIX}/lib"
  if [ -d "$conda_lib" ]; then
    case ":${LD_LIBRARY_PATH:-}:" in
      *":$conda_lib:"*) ;;
      *) export LD_LIBRARY_PATH="$conda_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
  fi
}

# Force offline mode for compute nodes (no internet).
set_offline_mode() {
  export HF_DATASETS_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_OFFLINE=1
}

set +u
