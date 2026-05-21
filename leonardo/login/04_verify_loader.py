"""Smoke-test whether the training data loader can actually consume the
dataspeech `save_to_disk` outputs, BEFORE committing to any data.py change.

Run on a login node (internet not required; only the local dataset dirs):

    source leonardo/env.sh && activate_conda_env
    python leonardo/login/04_verify_loader.py

It checks the three high-severity concerns raised in review:

  1. config-name split crash  -- convert_dataset_str_to_list(None config)
  2. load_dataset vs save_to_disk -- does load_dataset() read these dirs at all?
  3. None-metadata indexing    -- convert_dataset_str_to_list(metadata=None)

Nothing here mutates data. Exit code 0 means "loader is compatible as-is";
non-zero means at least one concern is confirmed and a fix is needed.
"""

from __future__ import annotations

import os
import traceback
from pathlib import Path

import datasets
from datasets import load_dataset, load_from_disk


DATA_ROOT = os.environ.get("DATA_ROOT", "/leonardo_work/EUHPC_D29_081/gsyllas0/data/tts")
NAMED_OUT = os.environ.get("NAMED_OUT_ROOT", f"{DATA_ROOT}/dataspeech_out_named")
OUT = os.environ.get("OUT_ROOT", f"{DATA_ROOT}/dataspeech_out")


def hr(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def first_existing(*paths: str) -> str | None:
    for p in paths:
        if Path(p).is_dir():
            return p
    return None


def check_convert_none_config() -> bool:
    """Concern #1: passing a None config should NOT crash if the loader is
    compatible with our configs (which omit train_dataset_config_name)."""
    hr("concern #1: None dataset_config_name")
    try:
        from training.data import convert_dataset_str_to_list
    except Exception as e:  # noqa: BLE001
        print(f"  could not import training.data: {type(e).__name__}: {e}")
        print("  run from the repo root with the env active.")
        return False
    try:
        convert_dataset_str_to_list(
            dataset_names="/some/local/dir",
            dataset_config_names=None,   # <- what our configs produce
            metadata_dataset_names=None,
            splits="train",
        )
        print("  OK: None config tolerated (no split crash).")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  CONFIRMED crash: {type(e).__name__}: {e}")
        return False


def check_convert_none_metadata() -> bool:
    """Concern #3: female/male omit metadata; None must be tolerated."""
    hr("concern #3: None metadata_dataset_names")
    try:
        from training.data import convert_dataset_str_to_list
    except Exception as e:  # noqa: BLE001
        print(f"  could not import training.data: {type(e).__name__}: {e}")
        return False
    try:
        # Give a valid config so we isolate the metadata-indexing path from #1.
        out = convert_dataset_str_to_list(
            dataset_names="/some/local/dir",
            dataset_config_names="default",
            metadata_dataset_names=None,
            splits="train",
        )
        print(f"  OK: None metadata tolerated -> {out}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  CONFIRMED crash: {type(e).__name__}: {e}")
        return False


def check_load_dataset_on_save_to_disk() -> bool:
    """Concern #2: the loader uses load_dataset(); our dirs are save_to_disk.
    Confirm load_from_disk works AND whether load_dataset works on the same dir."""
    hr("concern #2: load_dataset() vs save_to_disk dir")
    # Pick a real dir that should have a 'train' split.
    target = first_existing(
        f"{NAMED_OUT}/multi/04b_prompts_llm",
        f"{NAMED_OUT}/multi/01_hf_dataset",
        f"{OUT}/female/04a_prompts_deterministic",
        f"{OUT}/female/01_hf_dataset",
    )
    if target is None:
        print("  no dataset dir found under DATA_ROOT; set DATA_ROOT and retry.")
        return False
    print(f"  target dir: {target}")
    print(f"  datasets version: {datasets.__version__}")

    # Baseline: load_from_disk must work (that's how it was written).
    try:
        ds = load_from_disk(target)
        splits = list(ds) if hasattr(ds, "keys") else ["<single>"]
        print(f"  load_from_disk OK -> splits={splits}")
    except Exception as e:  # noqa: BLE001
        print(f"  load_from_disk FAILED ({type(e).__name__}: {e}); unexpected.")
        return False

    # The real question: does load_dataset() read it the way training does?
    try:
        got = load_dataset(target, split="train")
        print(f"  load_dataset(split='train') OK -> {len(got)} rows, cols={got.column_names[:6]}")
        print("  => loader is compatible as-is for this dir.")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  load_dataset FAILED: {type(e).__name__}: {e}")
        print("  (this is the expected outcome if save_to_disk dirs are incompatible)")
        traceback.print_exc()
        return False


def main() -> int:
    print(f"DATA_ROOT  = {DATA_ROOT}")
    results = {
        "none_config_ok": check_convert_none_config(),
        "none_metadata_ok": check_convert_none_metadata(),
        "load_dataset_ok": check_load_dataset_on_save_to_disk(),
    }
    hr("summary")
    for k, v in results.items():
        print(f"  {k:18s}: {'OK' if v else 'NEEDS FIX'}")
    all_ok = all(results.values())
    if all_ok:
        print("\nLoader is compatible with the dataspeech outputs as-is. No data.py patch needed.")
        return 0
    print("\nAt least one concern confirmed -> a loader fix (or data re-export) is required.")
    print("Share this output and I'll apply the targeted fix.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
