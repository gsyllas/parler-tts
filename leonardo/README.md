# Parler-TTS finetuning on Leonardo

Six independent finetuning experiments on the dataspeech-annotated Greek TTS
datasets, one SLURM job each on a single A100 (64 GB), running fully offline on
the Booster compute nodes. See [PLAN.md](PLAN.md) for the full design rationale.

Conventions mirror `~/dataspeech/leonardo/`: in-repo conda env, in-repo HF/torch
caches, [`env.sh`](env.sh) as the single source of truth, login-node staging +
offline SLURM jobs.

## Experiments

| EXP          | dataset                              | prompts        | base checkpoint |
|--------------|--------------------------------------|----------------|-----------------|
| `multi_det`  | named multi (`dataspeech_out_named`) | deterministic  | `parler-tts/parler-tts-mini-multilingual-v1.1` |
| `multi_llm`  | named multi                          | Qwen-LLM       | `parler-tts/parler-tts-mini-multilingual-v1.1` |
| `female_det` | female                               | deterministic  | `gsyllas/…_deterministic_60_epochs` |
| `female_llm` | female                               | LLM            | `gsyllas/…_52_epochs_speakers` |
| `male_det`   | male                                 | deterministic  | `gsyllas/…_deterministic_60_epochs` |
| `male_llm`   | male                                 | LLM            | `gsyllas/…_52_epochs_speakers` |

## Layout

```
leonardo/
├── env.sh                      paths, cache redirection, conda + offline helpers, EXP resolver
├── generate_eval_wavs.py       generate per-row eval WAVs from trained checkpoints
├── login/                      run on a LOGIN node (has internet)
│   ├── 00_setup_conda_env.sh   build .conda/parler-tts; pip install -e .[train]
│   ├── 01_cache_models.py      pre-fetch checkpoints + tokenizers + DAC + whisper + clap + wer
│   ├── 02_check_datasets.py    load_from_disk all paths; verify columns + split sizes
│   ├── 03_make_eval_splits.py  female/male: merge audio+desc, train_test_split, save_to_disk
│   └── 04_verify_loader.py     confirm the (patched) training loader reads save_to_disk dirs
├── configs/                    one JSON per experiment (based on starting_point_v1.json)
├── slurm/
│   ├── train.slurm             generic job: EXP=<id> sbatch …
│   ├── submit_all.sh           submit all six (or a named subset)
│   └── generate_eval_wavs.slurm generate eval WAVs from trained checkpoints
└── runs/<exp>/                 output checkpoints + DAC token buffers (created at runtime)
```

The conda env (`.conda/parler-tts`) and caches (`cache/{hf,torch}`) live inside
the repo on `/leonardo_work` — never `$HOME` (tiny quota).

## Run order

> **Source `env.sh` from bash, not zsh.** It uses `BASH_SOURCE` and bash
> indirect expansion (`${!var}`), so run `bash -lc '…'` (or start a `bash`
> shell) if your login shell is zsh. The SLURM scripts are `#!/bin/bash`, so
> jobs are unaffected.

```bash
# --- login node (internet), bash shell ---
cd /leonardo_work/EUHPC_D29_081/gsyllas0/parler-tts
source leonardo/env.sh

bash leonardo/login/00_setup_conda_env.sh     # one-time: env + deps
activate_conda_env

huggingface-cli login                          # the gsyllas/* bases may be gated
python leonardo/login/01_cache_models.py       # pre-stage all weights/metrics offline
python leonardo/login/02_check_datasets.py     # verify columns / split sizes
python leonardo/login/03_make_eval_splits.py   # female/male eval holdouts (multi skipped)
python leonardo/login/04_verify_loader.py      # confirm the patched loader reads the dirs (all OK)

# --- submit (login node) ---
bash leonardo/slurm/submit_all.sh              # all six (per-exp QOS/time auto-applied)
# or a subset:
bash leonardo/slurm/submit_all.sh female_det male_det
# manual single job — note the multi runs need the long QOS + longer wall:
EXP=multi_llm  sbatch --qos=boost_qos_lprod --time=72:00:00 \
    --export=ALL,EXP=multi_llm leonardo/slurm/train.slurm
EXP=female_llm sbatch --export=ALL,EXP=female_llm leonardo/slurm/train.slurm
squeue -u $USER

# --- after jobs (login node) ---
wandb sync leonardo/logs/wandb/offline-run-*   # push metrics + audio to wandb

# --- listening eval WAVs (GPU node) ---
# parallel: one job per model, each on its own GPU (preferred — ~1h/model):
bash leonardo/slurm/submit_eval_wavs.sh          # all six v2; or pass a subset
# serial: all models in a single 8h job (loops internally):
sbatch leonardo/slurm/generate_eval_wavs.slurm   # defaults to the v2 experiment set
# subset / smoke test:
EXPS="multi_v2_llm" GEN_ARGS="--limit 16 --overwrite-manifest" \
    sbatch --export=ALL leonardo/slurm/generate_eval_wavs.slurm
```

## Dataset join strategy

> `training/data.py` was patched (backward-compatible) so the loader reads local
> `save_to_disk` dirs via `load_from_disk` and tolerates omitted config/metadata.
> Without it the stock `load_dataset()` silently misreads these dirs into
> positional columns. `04_verify_loader.py` checks this; see PLAN.md §2.

- **multi** (`multi_det`, `multi_llm`) — `data.py::load_multiple_datasets`
  concatenates the audio dir (`01_hf_dataset`) and the metadata dir (`04a`/`04b`)
  by row order, per split (native `train` + `eval`). No `id_column_name` is set,
  so it joins positionally; equal split sizes + in-place dataspeech maps make
  this safe. No audio rewrite.
- **female/male** (`*_det`, `*_llm`) — no native eval split, so
  `03_make_eval_splits.py` merges audio + descriptions once (verified via the
  shared `filename` key), holds out 200 rows, and writes `05_merged_{det,llm}`
  as a `train`+`eval` `DatasetDict`. Training points straight at that dir with no
  metadata join.

## Tuning notes

- Conservative batch for the 64 GB A100: `per_device_train_batch_size=8`,
  `gradient_accumulation_steps=4` (effective 32), `max_duration_in_seconds=30`.
  If a job OOMs, drop `max_duration_in_seconds` to 25 before touching batch size.
- Epochs: multi = 100 (from the multilingual base; needs the long-production
  QOS `boost_qos_lprod`, ≤4 days, submitted with `--time=72:00:00`),
  female/male = 30 (from the already-Greek bases, fit the normal 24h QOS).
  Tune from wandb.
- Eval metric: **WER only**, via `openai/whisper-large-v3` (Greek-capable; the
  default `distil-whisper/distil-large-v2` is English-only). CLAP
  (`compute_clap_similarity_metric`) and SQUIM SI-SDR (`compute_noise_level_metric`)
  are **disabled** in every config — CLAP is English-centric (weak for Greek) and
  SQUIM adds an offline weight dependency for little signal. Rely on WER +
  listening to the wandb audio samples. To re-enable either, flip the flag in the
  config *and* confirm its weights cached on the login node (`01_cache_models.py`),
  or the offline compute job will fail mid-eval.
- First run per experiment precomputes DAC audio tokens into
  `runs/<exp>/dataset_audio`; subsequent re-runs reuse that buffer. Budget for it
  in the 24 h wall-time (multi is largest).
- `leonardo/generate_eval_wavs.py` defaults to `--source raw`: generation and
  ground-truth are re-derived from the *same* raw dataset, so the `greedy/` /
  `sample/` WAVs stay row-aligned with `ground_truth/` by construction. (The
  cached `save_to_disk` split stores audio only as DAC tokens and keeps no row id
  back to the source, so taking generation from cache while reading ground-truth
  from raw — the old default — could silently mispair the two. Use
  `GEN_ARGS="--source cache"` only if you need the exact training subset and
  accept that pairing then relies on every filter replaying identically.)
  It writes `ground_truth/`, `greedy/` (`do_sample=False`), and
  `sample/` (`do_sample=True`) folders under `eval_wavs/<checkpoint>/`, always
  passing both text attention masks. Each folder includes `transcriptions.tsv`
  and `manifest.jsonl`. Use `GEN_ARGS="--modes greedy"` or
  `GEN_ARGS="--modes sample"` for a single decoding mode.
- Batch size defaults to `1` so the listening eval is faithful: a single row has
  no padding, so the left-padded prompt / right-padded description interactions
  (and the train-time encoder masking that plain `generate()` does not replicate)
  cannot color the audio. Raise it with `GEN_ARGS="--batch-size N"` only to speed
  up large sweeps where exact fidelity matters less.
- No Hub push — checkpoints stay under `leonardo/runs/<exp>/output`.
