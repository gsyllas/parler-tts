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
import sys
import traceback
from pathlib import Path

# Allow `import training.*` when run as `python leonardo/login/04_verify_loader.py`
# (Python puts this file's dir on sys.path, not the repo root).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
        print("  importing training.data (pulls torch+transformers; ~1-3 min cold on Leonardo FS)...", flush=True)
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

    # Flag the save_to_disk markers up front so we know what we're poking at.
    markers = [m for m in ("dataset_dict.json", "state.json", "dataset_info.json")
               if (Path(target) / m).exists() or list(Path(target).glob(f"*/{m}"))]
    if markers:
        print(f"  note: dir carries save_to_disk markers {markers} (written by save_to_disk).")

    # The stock loader uses load_dataset(); use streaming so we resolve the dir +
    # read ONE row instead of all 36k. Crucially, check the *schema*, not just
    # that a row came back: load_dataset misreads save_to_disk shards into
    # positional columns ('0','1',...), which is a silent corruption, not success.
    try:
        stream = load_dataset(target, split="train", streaming=True)
        first = next(iter(stream))
        cols = list(first)
        positional = all(c.isdigit() for c in cols)
        print(f"  load_dataset(streaming) returned cols={cols[:8]}")
        if positional or "text_description" not in cols:
            print("  => MISREAD: columns are positional/garbage, not the real schema.")
            print("     load_dataset cannot read save_to_disk dirs -> loader patch required.")
            return False
        print("  => load_dataset read real columns; compatible as-is.")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  load_dataset FAILED: {type(e).__name__}: {e}")
        print("  (expected if save_to_disk dirs are incompatible with load_dataset)")
        traceback.print_exc()
        return False


def check_patched_loader() -> bool:
    """After the data.py patch: does training's own _load_split() read the dir
    with the correct schema via load_from_disk?"""
    hr("post-patch: training._load_split reads real schema")
    target = first_existing(
        f"{NAMED_OUT}/multi/04b_prompts_llm",
        f"{NAMED_OUT}/multi/01_hf_dataset",
        f"{OUT}/female/04a_prompts_deterministic",
        f"{OUT}/female/01_hf_dataset",
    )
    if target is None:
        print("  no dataset dir found; skipping.")
        return False
    try:
        from training.data import _load_split
    except Exception as e:  # noqa: BLE001
        print(f"  could not import training.data._load_split: {type(e).__name__}: {e}")
        print("  (older data.py without the patch?)")
        return False
    try:
        ds = _load_split(target, None, "train", False)
        cols = ds.column_names
        ok = "text" in cols and ("text_description" in cols or "audio" in cols)
        print(f"  _load_split OK -> {len(ds)} rows, cols={cols[:8]}")
        if not ok:
            print("  => unexpected schema; investigate.")
        return ok
    except Exception as e:  # noqa: BLE001
        print(f"  _load_split FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def main() -> int:
    print(f"DATA_ROOT  = {DATA_ROOT}")
    # Gating checks: these must pass for training to work.
    gates = {
        "none_config_ok": check_convert_none_config(),
        "none_metadata_ok": check_convert_none_metadata(),
        "patched_loader_ok": check_patched_loader(),
    }
    # Informational: stock load_dataset is EXPECTED to misread save_to_disk dirs.
    # This is why _load_split() bypasses it; the result does not gate success.
    stock_load_dataset_works = check_load_dataset_on_save_to_disk()

    hr("summary")
    for k, v in gates.items():
        print(f"  {k:18s}: {'OK' if v else 'NEEDS FIX'}")
    print(f"  {'stock load_dataset':18s}: "
          f"{'reads dir' if stock_load_dataset_works else 'misreads (expected; patch bypasses it)'}")

    if all(gates.values()):
        print("\nLoader is compatible with the dataspeech outputs. Ready to train.")
        return 0
    print("\nA gating check failed -> share this output for the next fix.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
