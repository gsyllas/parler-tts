# Parler-TTS finetuning on Leonardo — Plan

End-to-end plan to finetune `parler-tts/parler-tts-mini-multilingual-v1.1`
(and two Greek-finetuned derivatives) on the dataspeech-annotated Greek TTS
datasets, on the Leonardo Booster partition (A100 64GB), running fully offline
on the compute nodes.

This document mirrors the conventions already proven in
`~/dataspeech/leonardo/` (in-repo conda env, in-repo caches, `env.sh` single
source of truth, login-node staging + offline SLURM jobs).

---

## 1. Goals

1. **Separate, self-contained environment.** A dedicated conda env for
   parler-tts living *inside* the repo on `/leonardo_work` (never `$HOME` — home
   quota is tiny). All HF / torch caches redirected into the repo too.

2. **Six independent finetuning experiments**, one SLURM job each, on a single
   A100 per job:

   | # | id | dataset | base checkpoint |
   |---|----|---------|-----------------|
   | 1 | `multi_det`  | named multi, deterministic prompts | `parler-tts/parler-tts-mini-multilingual-v1.1` |
   | 2 | `multi_llm`  | named multi, Qwen-LLM prompts      | `parler-tts/parler-tts-mini-multilingual-v1.1` |
   | 3 | `female_det` | female, deterministic prompts      | `gsyllas/parler-tts-mini-multilingual-to-greek-v1.1_deterministic_60_epochs` |
   | 4 | `female_llm` | female, LLM prompts                | `gsyllas/parler-tts-mini-multilingual-to-greek-v1.1_52_epochs_speakers` |
   | 5 | `male_det`   | male, deterministic prompts        | `gsyllas/parler-tts-mini-multilingual-to-greek-v1.1_deterministic_60_epochs` |
   | 6 | `male_llm`   | male, LLM prompts                  | `gsyllas/parler-tts-mini-multilingual-to-greek-v1.1_52_epochs_speakers` |

   The det→deterministic-base and llm→speakers-base pairing for female/male is
   exactly as specified. multi always finetunes from the multilingual base.

3. **Fully offline compute.** Compute nodes have no internet. Everything
   (checkpoints, tokenizers, DAC, eval models, metrics) is pre-staged on a login
   node; jobs run with `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`.

4. **Reuse the released recipe** (`helpers/training_configs/starting_point_v1.json`)
   with finetuning-appropriate hyper-parameters and conservative batch sizing for
   the 64GB A100.

### Locked decisions

- **1 GPU per job** (`--gres=gpu:1`); all six run as independent queue jobs.
- **Conservative (v1-like) batch**: `per_device_train_batch_size=8`,
  `gradient_accumulation_steps=4` (effective 32), `max_duration_in_seconds=30`.
- **Held-out eval for female/male** via a login-node merge+split (multi already
  has a native `eval` split).
- **No Hub push** — trained checkpoints stay on `/leonardo_work`.
- **Tokenizers / feature extractor**: `google/flan-t5-large` (description +
  prompt) and `parler-tts/dac_44khZ_8kbps` — confirmed for v1.1.

---

## 2. Findings that drive the design

Verified on a Leonardo login node with `load_from_disk`:

- **Audio is not in `04a/04b`.** The dataspeech prompt stages keep audio in
  `01_hf_dataset/` and write only metadata + `text_description` into
  `04a_prompts_deterministic/` and `04b_prompts_llm/`.
- **No `id` column anywhere.** Shared columns:
  - multi: `gender, speaker_id, speaker_name, text`
  - female/male: `filename, gender, origin_dataset, speaker_id, text, transcription_original`
  - `filename` is a usable unique key for female/male; multi has no unique key.
- **Split sizes match exactly** on the audio side (`01_hf_dataset`) and the
  metadata side (`04x`):
  - multi: `train=36432`, `eval=1316`
  - female: `train=10111` (train-only)
  - male: `train=13960` (train-only)
- dataspeech maps rows in place (no reorder, no row drops — counts are equal), so
  a **row-order join is safe**.
- Column roles: prompt/transcript = `text`, description = `text_description`,
  audio = `audio` (`Audio(...)` feature, embedded waveform).

### How the training script joins audio + descriptions

`training/run_parler_tts_training.py` → `training/data.py::load_multiple_datasets`
is *designed* to merge a dataset-with-audio (`train_dataset_name`) and a
dataset-with-descriptions (`train_metadata_dataset_name`) — exactly the released
LibriTTS recipe pattern:

- `data.py:271` → `concatenate_datasets([dataset, metadata_dataset], axis=1)`
- If `id_column_name` is set, it renames the metadata id to `metadata_<id>`,
  drops overlapping columns, concatenates, then asserts `id == metadata_id`
  row-by-row (`data.py:277-288`).
- If `id_column_name` is unset, it just drops overlapping columns and
  concatenates **by row order**.

Since there is no `id` column, we rely on row-order alignment (guaranteed by the
equal split sizes and in-place maps). No `data.py` patch is required.

---

## 3. Join / dataset strategy per experiment

### multi (runs 1–2) — built-in join, no audio rewrite

```
train_dataset_name          = <root>/dataspeech_out_named/multi/01_hf_dataset       (split: train)
train_metadata_dataset_name = <root>/dataspeech_out_named/multi/04{a,b}_prompts_…    (split: train)
eval_dataset_name           = <root>/dataspeech_out_named/multi/01_hf_dataset        (split: eval)
eval_metadata_dataset_name  = <root>/dataspeech_out_named/multi/04{a,b}_prompts_…    (split: eval)
# id_column_name omitted  → row-order concat
```

### female/male (runs 3–6) — pre-merge + holdout split (login node)

They have no eval split, and the join cannot cleanly carve one out. So
`03_make_eval_splits.py` does, once per (dataset × prompt-type):

1. `load_from_disk(01_hf_dataset)` (audio) and `load_from_disk(04x)` (metadata).
2. Verify alignment via the shared unique `filename` (assert equal, row-by-row).
3. Drop metadata columns that duplicate the audio side; `concatenate_datasets(axis=1)`.
4. `train_test_split(test_size=200, seed=…)` on the merged dataset.
5. `save_to_disk` a combined `DatasetDict` (`train`+`eval`) carrying audio +
   `text_description` together.

Output path convention:

```
<root>/dataspeech_out/female/05_merged_det
<root>/dataspeech_out/female/05_merged_llm
<root>/dataspeech_out/male/05_merged_det
<root>/dataspeech_out/male/05_merged_llm
```

Training then points at the single merged dir, **no metadata join**:

```
train_dataset_name = <root>/dataspeech_out/female/05_merged_llm   (split: train)
eval_dataset_name  = <root>/dataspeech_out/female/05_merged_llm   (split: eval)
```

Cost: rewrites female+male audio once (~24k clips, a few GB). multi (37k clips)
is never rewritten.

`<root>` = `/leonardo_work/EUHPC_D29_081/gsyllas0/data/tts`.

---

## 4. Repository layout

Clone parler-tts onto `/leonardo_work` and add a `leonardo/` subtree:

```
/leonardo_work/EUHPC_D29_081/gsyllas0/parler-tts/
├── leonardo/
│   ├── PLAN.md                     # this document
│   ├── env.sh                      # paths, cache redirection, conda + offline helpers, EXP resolver
│   ├── login/                      # run on a LOGIN node (has internet)
│   │   ├── 00_setup_conda_env.sh   # build .conda/parler-tts; pip install -e .[train]
│   │   ├── 01_cache_models.py      # pre-fetch checkpoints + tokenizers + DAC + whisper + clap + wer
│   │   ├── 02_check_datasets.py    # load_from_disk all paths; print columns + split sizes
│   │   └── 03_make_eval_splits.py  # female/male: merge audio+desc, train_test_split, save_to_disk
│   ├── configs/                    # one JSON per experiment
│   │   ├── multi_det.json   multi_llm.json
│   │   ├── female_det.json  female_llm.json
│   │   └── male_det.json    male_llm.json
│   ├── slurm/
│   │   ├── train.slurm             # generic: EXP=multi_llm sbatch leonardo/slurm/train.slurm
│   │   └── submit_all.sh           # submit all six (or a named subset)
│   └── logs/                       # SLURM stdout/stderr
├── .conda/parler-tts/              # in-repo conda env (no $HOME quota)
└── cache/{hf,torch}                # in-repo HF / torch caches
```

---

## 5. Implementation steps

### Step 0 — clone + env.sh

Clone the repo to `/leonardo_work/EUHPC_D29_081/gsyllas0/parler-tts`. Author
`leonardo/env.sh` (adapted from the dataspeech one):

- `SLURM_ACCOUNT=EUHPC_D29_081`, `SLURM_PARTITION=boost_usr_prod`
- `REPO_ROOT` = repo dir; `CONDA_ENV_PREFIX=$REPO_ROOT/.conda/parler-tts`
- `CACHE_ROOT=$REPO_ROOT/cache`; export `HF_HOME`, `HF_HUB_CACHE`,
  `TRANSFORMERS_CACHE`, `HF_DATASETS_CACHE`, `TORCH_HOME`, `PIP_CACHE_DIR`,
  `CONDA_PKGS_DIRS`, `CONDA_ENVS_PATH` — all inside the repo.
- `DATA_ROOT=/leonardo_work/EUHPC_D29_081/gsyllas0/data/tts`
- `activate_conda_env()` and `set_offline_mode()` (sets `HF_HUB_OFFLINE=1`,
  `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`).
- An `EXP` resolver mapping each id → `{config json, base checkpoint,
  train/eval dataset paths}` so SLURM and login scripts share one source of truth.

### Step 1 — conda env (login node)

`leonardo/login/00_setup_conda_env.sh`:

```bash
conda create -p $CONDA_ENV_PREFIX python=3.10 pip git ffmpeg libsndfile
conda activate $CONDA_ENV_PREFIX
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e .[train]      # transformers==4.46.1, datasets[audio], accelerate, wandb, jiwer, evaluate, DAC
```

cu121 wheels are forward-compatible with `module load cuda/12.2` on compute
nodes. Finish with a sanity import (`parler_tts`, `transformers`, `datasets`,
`accelerate`, torch CUDA flag).

### Step 2 — pre-cache everything offline (login node)

`leonardo/login/01_cache_models.py` downloads into the in-repo HF/torch cache:

- Checkpoints: `parler-tts/parler-tts-mini-multilingual-v1.1`,
  `gsyllas/…_deterministic_60_epochs`, `gsyllas/…_52_epochs_speakers`.
- Tokenizers + feature extractor: `google/flan-t5-large`,
  `parler-tts/dac_44khZ_8kbps`.
- Eval: `openai/whisper-large-v3` (Greek WER — the default
  `distil-whisper/distil-large-v2` is English-only), CLAP, torchaudio SQUIM.
- `evaluate.load("wer")` metric.

Also configure `wandb` offline (`WANDB_MODE=offline`); sync from a login node
after jobs finish.

### Step 3 — verify datasets (login node)

`leonardo/login/02_check_datasets.py`: `load_from_disk` all six dataset dirs +
the three `01_hf_dataset` audio dirs; print per-split sizes and columns; assert
audio side has the `Audio(...)` feature and metadata side has `text_description`.
(Already run manually — confirmed; this script makes it repeatable.)

### Step 4 — female/male eval holdout (login node)

`leonardo/login/03_make_eval_splits.py` — as in §3: merge audio + metadata by
row order (verify with `filename`), `train_test_split(test_size=200)`, write the
four `05_merged_{det,llm}` DatasetDicts. multi is skipped (native eval split).

### Step 5 — configs

Six JSON files under `leonardo/configs/`, based on `starting_point_v1.json`.
Shared block (see §6); per-experiment fields = `model_name_or_path`,
`train_dataset_name` (+ `train_metadata_dataset_name` for multi only),
`eval_dataset_name` (+ `eval_metadata_dataset_name` for multi only),
`train_split_name` / `eval_split_name`, `output_dir`, `wandb_run_name`,
`num_train_epochs`.

### Step 6 — SLURM

`leonardo/slurm/train.slurm` (generic):

```bash
#SBATCH --account=EUHPC_D29_081
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --gres=gpu:1
#SBATCH --time=24:00:00
module purge && module load cuda/12.2
cd "$SLURM_SUBMIT_DIR"
source leonardo/env.sh
: "${EXP:?set EXP=multi_llm|multi_det|female_llm|…}"
activate_conda_env
set_offline_mode
accelerate launch --num_processes=1 \
  training/run_parler_tts_training.py "leonardo/configs/$EXP.json"
```

`leonardo/slurm/submit_all.sh`: loop over the six ids (or a passed subset),
`EXP=<id> sbatch --job-name=ptts-<id> --export=ALL,EXP=<id> leonardo/slurm/train.slurm`.

---

## 6. Configuration spec

Shared finetuning hyper-parameters (all six configs):

| field | value | rationale |
|-------|-------|-----------|
| `feature_extractor_name` | `parler-tts/dac_44khZ_8kbps` | matches v1.1 |
| `description_tokenizer_name` / `prompt_tokenizer_name` | `google/flan-t5-large` | matches v1.1 |
| `target_audio_column_name` | `audio` | |
| `prompt_column_name` | `text` | transcript |
| `description_column_name` | `text_description` | dataspeech output |
| `freeze_text_encoder` | `true` | save compute; standard for finetune |
| `dtype` / `attn_implementation` | `bfloat16` / `sdpa` | A100 |
| `per_device_train_batch_size` | `8` | conservative for 64GB |
| `gradient_accumulation_steps` | `4` | effective batch 32 |
| `gradient_checkpointing` | `false` | |
| `audio_encoder_per_device_batch_size` | `24` | DAC pre-tokenisation |
| `per_device_eval_batch_size` | `4` | |
| `max_duration_in_seconds` / `min_duration_in_seconds` | `30` / `2.0` | |
| `max_text_length` | `600` | |
| `learning_rate` | `1e-4` | finetune (vs 9.5e-4 from scratch) |
| `lr_scheduler_type` | `cosine` | longer finetune runs |
| `warmup_steps` | `~500` | short relative to dataset |
| `group_by_length` | `true` | |
| `do_train` / `do_eval` | `true` / `true` | |
| `predict_with_generate` / `include_inputs_for_metrics` | `true` / `true` | WER + CLAP |
| `add_audio_samples_to_wandb` | `true` | |
| `asr_model_name_or_path` | `openai/whisper-large-v3` | Greek WER |
| `evaluation_strategy` / `eval_steps` / `save_steps` | `steps` / tuned | |
| `max_eval_samples` | `96` | |
| `save_to_disk` / `temporary_save_to_disk` | per-experiment buffer dirs | precompute DAC tokens once, reuse on re-run |
| `report_to` | `wandb` (offline) | |
| `preprocessing_num_workers` / `dataloader_num_workers` | `8` / `8` | 32-core node |

Starting epoch counts (tune from wandb): **multi ≈ 30** (from multilingual base),
**female/male ≈ 15** (from already-Greek bases). `save_steps`/`eval_steps`
chosen so each run gets several checkpoints.

`save_to_disk` / `temporary_save_to_disk` get **distinct per-experiment paths**
so the one-time DAC audio-token precompute is cached and reused across re-runs of
the same experiment.

---

## 7. Run order on Leonardo

```bash
# --- login node (internet) ---
cd /leonardo_work/EUHPC_D29_081/gsyllas0/parler-tts
source leonardo/env.sh
bash   leonardo/login/00_setup_conda_env.sh
activate_conda_env
python leonardo/login/01_cache_models.py       # pre-stage all weights/metrics
python leonardo/login/02_check_datasets.py      # re-verify columns/sizes
python leonardo/login/03_make_eval_splits.py    # female/male eval holdouts

# --- submit (login node) ---
bash leonardo/slurm/submit_all.sh               # all six
# or one:
EXP=multi_llm sbatch --export=ALL,EXP=multi_llm leonardo/slurm/train.slurm
squeue -u $USER

# --- after jobs (login node) ---
wandb sync leonardo/logs/wandb/offline-run-*    # push metrics/audio to wandb
```

---

## 8. Risks / things to watch

1. **Row-order join (multi).** Relies on `01_hf_dataset` and `04x` being aligned.
   Guaranteed by equal split sizes + in-place dataspeech maps; re-checked by
   `02_check_datasets.py`. If a future re-annotation filters rows, sizes would
   diverge and the check would catch it.
2. **VRAM.** Conservative batch should fit a 64GB A100 at 30s max duration. If it
   OOMs, lower `max_duration_in_seconds` to 25 before touching batch size.
3. **First run is slow.** DAC audio-token precompute runs once per experiment
   before training; budget for it in the 24h wall-time (multi is the largest).
   Subsequent re-runs reuse the `save_to_disk` buffer.
4. **CLAP for Greek.** CLAP text encoder is English-centric, so CLAP-similarity is
   a weak signal for Greek descriptions; rely primarily on WER (whisper-large-v3)
   and listening to wandb audio samples.
5. **Offline gating.** Any model not cached in Step 2 will fail on the compute
   node. Re-run `01_cache_models.py` if a checkpoint/tokenizer id changes.
```
