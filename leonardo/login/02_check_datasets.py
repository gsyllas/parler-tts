"""Verify the dataspeech outputs are shaped the way training expects.

Run on a login node (or any node with the datasets on disk):

    source leonardo/env.sh && activate_conda_env
    python leonardo/login/02_check_datasets.py

For each experiment it loads the audio dir (01_hf_dataset) and the metadata dir
(04a/04b) with `load_from_disk`, prints per-split sizes + columns, and asserts:

  - the audio side carries an `Audio(...)` feature in column `audio`
  - the metadata side carries a `text_description` column
  - matching splits have equal length (required for the row-order join)

multi has native train+eval splits; female/male are train-only here (their eval
holdout is carved later by 03_make_eval_splits.py).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from datasets import Audio, DatasetDict, load_from_disk


# (label, audio_dir_env_fn, metadata_dir) tuples filled from env paths below.
def _root(name: str, default: str) -> str:
    return os.environ.get(name, default)


DATA_ROOT = _root("DATA_ROOT", "/leonardo_work/EUHPC_D29_081/gsyllas0/data/tts")
NAMED_OUT = _root("NAMED_OUT_ROOT", f"{DATA_ROOT}/dataspeech_out_named")
OUT = _root("OUT_ROOT", f"{DATA_ROOT}/dataspeech_out")

CASES = [
    # label,        audio dir,                          metadata dir
    ("multi_det",   f"{NAMED_OUT}/multi/01_hf_dataset",  f"{NAMED_OUT}/multi/04a_prompts_deterministic"),
    ("multi_llm",   f"{NAMED_OUT}/multi/01_hf_dataset",  f"{NAMED_OUT}/multi/04b_prompts_llm"),
    ("female_det",  f"{OUT}/female/01_hf_dataset",       f"{OUT}/female/04a_prompts_deterministic"),
    ("female_llm",  f"{OUT}/female/01_hf_dataset",       f"{OUT}/female/04b_prompts_llm"),
    ("male_det",    f"{OUT}/male/01_hf_dataset",         f"{OUT}/male/04a_prompts_deterministic"),
    ("male_llm",    f"{OUT}/male/01_hf_dataset",         f"{OUT}/male/04b_prompts_llm"),
]


def _as_dict(ds) -> DatasetDict:
    if isinstance(ds, DatasetDict):
        return ds
    # A bare Dataset -> wrap so the split loop is uniform.
    return DatasetDict({"train": ds})


def check_case(label: str, audio_dir: str, meta_dir: str) -> list[str]:
    problems: list[str] = []
    print(f"\n=== {label} ===")
    print(f"audio:    {audio_dir}")
    print(f"metadata: {meta_dir}")

    for tag, path in (("audio", audio_dir), ("metadata", meta_dir)):
        if not Path(path).is_dir():
            problems.append(f"{label}: missing {tag} dir {path}")
    if problems:
        for p in problems:
            print(f"  MISSING: {p}")
        return problems

    audio = _as_dict(load_from_disk(audio_dir))
    meta = _as_dict(load_from_disk(meta_dir))

    print("audio splits:    ", {s: len(audio[s]) for s in audio})
    print("metadata splits: ", {s: len(meta[s]) for s in meta})

    # audio feature present?
    a_split = next(iter(audio))
    a_feats = audio[a_split].features
    if "audio" not in a_feats:
        problems.append(f"{label}: audio side has no 'audio' column ({list(a_feats)})")
    elif not isinstance(a_feats["audio"], Audio):
        problems.append(f"{label}: 'audio' column is {a_feats['audio']}, not an Audio(...) feature")
    else:
        print(f"audio dtype:     {a_feats['audio']}")

    # text_description present on metadata side?
    m_split = next(iter(meta))
    m_cols = meta[m_split].column_names
    print("metadata cols:   ", m_cols)
    if "text_description" not in m_cols:
        problems.append(f"{label}: metadata has no 'text_description' column")
    if "text" not in m_cols:
        problems.append(f"{label}: metadata has no 'text' (prompt/transcript) column")

    # split-length alignment for shared splits (row-order join precondition).
    for split in meta:
        if split not in audio:
            problems.append(f"{label}: split {split!r} in metadata but not audio")
            continue
        if len(audio[split]) != len(meta[split]):
            problems.append(
                f"{label}: split {split!r} length mismatch audio={len(audio[split])} "
                f"metadata={len(meta[split])}"
            )

    if not problems:
        print("  OK")
    else:
        for p in problems:
            print(f"  PROBLEM: {p}")
    return problems


def main() -> int:
    all_problems: list[str] = []
    for label, audio_dir, meta_dir in CASES:
        try:
            all_problems += check_case(label, audio_dir, meta_dir)
        except Exception as e:  # noqa: BLE001
            all_problems.append(f"{label}: raised {type(e).__name__}: {e}")
            print(f"  ERROR: {type(e).__name__}: {e}")

    print("\n=== summary ===")
    if all_problems:
        print(f"{len(all_problems)} problem(s):")
        for p in all_problems:
            print(f"  - {p}")
        return 1
    print("all six dataset pairs look good for training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
