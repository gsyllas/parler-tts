"""Carve held-out eval splits for the single-speaker female/male datasets.

multi has a native eval split, but female/male are train-only. The metadata
join in training/data.py can't cleanly hold one out, so we do it here, once per
(dataset x prompt-type), on a login node:

    source leonardo/env.sh && activate_conda_env
    python leonardo/login/03_make_eval_splits.py            # all four
    python leonardo/login/03_make_eval_splits.py --only female_llm

For each case it:
  1. load_from_disk the audio dir (01_hf_dataset) and the metadata dir (04x)
  2. verify alignment via the shared unique `filename` (row-by-row), with a
     row-order fallback if `filename` is absent on either side
  3. drop metadata columns that duplicate the audio side; concatenate axis=1
     (audio kept lazy with decode=False so we don't rewrite waveforms twice)
  4. train_test_split(test_size=TEST_SIZE, seed=SEED) on the merged dataset
  5. save_to_disk a DatasetDict {train, eval} carrying audio + text_description

Output: <OUT_ROOT>/<dataset>/05_merged_{det,llm}
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from datasets import Audio, Dataset, DatasetDict, concatenate_datasets, load_from_disk


DATA_ROOT = os.environ.get("DATA_ROOT", "/leonardo_work/EUHPC_D29_081/gsyllas0/data/tts")
OUT = os.environ.get("OUT_ROOT", f"{DATA_ROOT}/dataspeech_out")
TEST_SIZE = int(os.environ.get("EVAL_HOLDOUT_SIZE", "200"))
SEED = int(os.environ.get("EVAL_SPLIT_SEED", "456"))

# case id -> (dataset, prompt stage dir, output suffix)
CASES = {
    "female_det": ("female", "04a_prompts_deterministic", "det"),
    "female_llm": ("female", "04b_prompts_llm", "llm"),
    "male_det":   ("male",   "04a_prompts_deterministic", "det"),
    "male_llm":   ("male",   "04b_prompts_llm", "llm"),
    "greek_female_det": ("greek_female_tts", "04a_prompts_deterministic", "det"),
    "greek_female_llm": ("greek_female_tts", "04b_prompts_llm", "llm"),
    "greek_male_det":   ("greek_male_tts",   "04a_prompts_deterministic", "det"),
    "greek_male_llm":   ("greek_male_tts",   "04b_prompts_llm", "llm"),
}

KEY = "filename"  # unique key shared by audio + metadata for female/male


def _single(ds) -> Dataset:
    """Return the one underlying Dataset, whether it's a DatasetDict or not."""
    if isinstance(ds, DatasetDict):
        if "train" in ds:
            return ds["train"]
        return ds[next(iter(ds))]
    return ds


def _verify_alignment(audio: Dataset, meta: Dataset, label: str) -> None:
    if len(audio) != len(meta):
        raise SystemExit(
            f"{label}: length mismatch audio={len(audio)} metadata={len(meta)}; "
            "cannot join by row order."
        )
    if KEY in audio.column_names and KEY in meta.column_names:
        a = audio[KEY]
        m = meta[KEY]
        mism = [i for i in range(len(a)) if a[i] != m[i]]
        if mism:
            raise SystemExit(
                f"{label}: {len(mism)} rows have mismatched {KEY!r} "
                f"(first at index {mism[0]}: {a[mism[0]]!r} vs {m[mism[0]]!r}); "
                "audio/metadata are not row-aligned."
            )
        print(f"  alignment OK via {KEY!r} across {len(a)} rows")
    else:
        print(f"  WARNING: {KEY!r} absent on one side; trusting row order (equal lengths)")


def make_case(label: str, overwrite: bool) -> None:
    dataset, stage, suffix = CASES[label]
    audio_dir = f"{OUT}/{dataset}/01_hf_dataset"
    meta_dir = f"{OUT}/{dataset}/{stage}"
    out_dir = f"{OUT}/{dataset}/05_merged_{suffix}"

    print(f"\n=== {label} ===")
    print(f"audio:    {audio_dir}")
    print(f"metadata: {meta_dir}")
    print(f"output:   {out_dir}")

    if Path(out_dir).exists():
        if not overwrite:
            print("  exists, skipping (use --overwrite to rebuild)")
            return
        shutil.rmtree(out_dir)

    for tag, path in (("audio", audio_dir), ("metadata", meta_dir)):
        if not Path(path).is_dir():
            raise SystemExit(f"{label}: missing {tag} dir {path}")

    audio = _single(load_from_disk(audio_dir))
    meta = _single(load_from_disk(meta_dir))

    if "audio" not in audio.column_names:
        raise SystemExit(f"{label}: audio dir has no 'audio' column ({audio.column_names})")
    if "text_description" not in meta.column_names:
        raise SystemExit(f"{label}: metadata has no 'text_description' column ({meta.column_names})")

    _verify_alignment(audio, meta, label)

    # Keep the full audio side; from metadata take only columns the audio side
    # lacks (text_description, and text if missing on the audio side). Keep audio
    # lazy so the join doesn't decode waveforms.
    audio = audio.cast_column("audio", Audio(decode=False))
    dup = set(meta.column_names).intersection(set(audio.column_names))
    meta_extra = meta.remove_columns(list(dup))
    merged = concatenate_datasets([audio, meta_extra], axis=1)
    print(f"  merged columns: {merged.column_names}")

    split = merged.train_test_split(test_size=TEST_SIZE, seed=SEED)
    out = DatasetDict({"train": split["train"], "eval": split["test"]})
    print(f"  train={len(out['train'])}  eval={len(out['eval'])}")

    Path(out_dir).parent.mkdir(parents=True, exist_ok=True)
    out.save_to_disk(out_dir)
    print(f"  wrote {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=sorted(CASES), action="append",
                        help="Only build these case(s). Repeatable. Default: all four.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Rebuild even if the 05_merged_* dir already exists.")
    args = parser.parse_args()

    targets = args.only or sorted(CASES)
    print(f"holdout size = {TEST_SIZE}, seed = {SEED}")
    for label in targets:
        make_case(label, overwrite=args.overwrite)
    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
