"""Preflight-check new Greek TTS training datasets before submitting SLURM jobs.

Run on a Leonardo login node after pulling this repo:

    cd /leonardo_work/EUHPC_D29_081/gsyllas0/parler-tts
    source leonardo/env.sh
    activate_conda_env
    python leonardo/login/05_preflight_new_datasets.py

The check is intentionally read-only. It verifies that each config can load its
train data, that prompt/description columns exist, and that the old temptation
of using the multilingual prompt tokenizer for descriptions is incompatible
with the flan-T5 text encoder vocabulary.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoConfig, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIGS = [
    "multi_v2_det",
    "multi_v2_llm",
    "greek_female_det",
    "greek_female_llm",
    "greek_male_det",
    "greek_male_llm",
]


def _config_path(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.suffix == ".json" or path.exists():
        return path
    return REPO_ROOT / "leonardo" / "configs" / f"{name_or_path}.json"


def _load_config(name_or_path: str) -> tuple[Path, dict[str, Any]]:
    path = _config_path(name_or_path)
    with path.open("r", encoding="utf-8") as f:
        return path, json.load(f)


def _load_split(path: str, split: str) -> Dataset:
    loaded = load_from_disk(path)
    if isinstance(loaded, DatasetDict):
        if split not in loaded:
            raise SystemExit(f"{path}: missing split {split!r}; available: {list(loaded)}")
        return loaded[split]
    return loaded


def _sample_column(ds: Dataset, column: str, n: int, rng: random.Random) -> tuple[list[int], list[str]]:
    if column not in ds.column_names:
        raise SystemExit(f"column {column!r} missing; columns={ds.column_names}")
    count = min(n, len(ds))
    indices = rng.sample(range(len(ds)), count) if count else []
    rows = ds.select(indices)[column] if indices else []
    return indices, ["" if x is None else str(x) for x in rows]


def _max_token_id(tokenizer, texts: list[str]) -> int:
    if not texts:
        return -1
    encoded = tokenizer(texts, add_special_tokens=True, truncation=False)
    flat = [token_id for row in encoded["input_ids"] for token_id in row]
    return max(flat) if flat else -1


def _unk_count(tokenizer, texts: list[str]) -> tuple[int, int]:
    unk_id = tokenizer.unk_token_id
    if unk_id is None or not texts:
        return 0, 0
    encoded = tokenizer(texts, add_special_tokens=True, truncation=False)
    flat = [token_id for row in encoded["input_ids"] for token_id in row]
    return sum(1 for token_id in flat if token_id == unk_id), len(flat)


def _vocab_size_for_text_encoder(config_name: str, tokenizer) -> int:
    try:
        return int(AutoConfig.from_pretrained(config_name).vocab_size)
    except Exception:  # noqa: BLE001
        return len(tokenizer)


def check_config(config_name: str, samples: int, seed: int) -> list[str]:
    path, cfg = _load_config(config_name)
    label = path.stem
    rng = random.Random(seed + sum(ord(c) for c in label))
    problems: list[str] = []

    print(f"\n=== {label} ===")
    train_path = cfg["train_dataset_name"]
    train_split = cfg.get("train_split_name", "train")
    prompt_col = cfg.get("prompt_column_name", "text")
    desc_col = cfg.get("description_column_name", "text_description")
    meta_path = cfg.get("train_metadata_dataset_name")

    prompt_ds = _load_split(train_path, train_split)
    if meta_path:
        desc_ds = _load_split(meta_path, train_split)
        if len(prompt_ds) != len(desc_ds):
            problems.append(f"train length mismatch audio={len(prompt_ds)} metadata={len(desc_ds)}")
    else:
        desc_ds = prompt_ds

    print(f"config:       {path.relative_to(REPO_ROOT)}")
    print(f"train data:   {train_path} [{train_split}] rows={len(prompt_ds)}")
    if meta_path:
        print(f"metadata:     {meta_path} [{train_split}] rows={len(desc_ds)}")
    print(f"columns:      prompt={prompt_col!r} description={desc_col!r}")

    prompt_indices, prompt_texts = _sample_column(prompt_ds, prompt_col, samples, rng)
    desc_indices, desc_texts = _sample_column(desc_ds, desc_col, samples, rng)

    desc_tok_name = cfg["description_tokenizer_name"]
    prompt_tok_name = cfg["prompt_tokenizer_name"]
    desc_tok = AutoTokenizer.from_pretrained(desc_tok_name)
    prompt_tok = AutoTokenizer.from_pretrained(prompt_tok_name)
    text_encoder_vocab = _vocab_size_for_text_encoder(desc_tok_name, desc_tok)
    prompt_tokenizer_vocab = len(prompt_tok)

    desc_max_id = _max_token_id(desc_tok, desc_texts)
    desc_unk, desc_total = _unk_count(desc_tok, desc_texts)
    multilingual_desc_max_id = _max_token_id(prompt_tok, desc_texts)

    print(f"description tokenizer: {desc_tok_name}")
    print(f"prompt tokenizer:      {prompt_tok_name}")
    print(f"text encoder vocab:    {text_encoder_vocab}")
    print(f"prompt tokenizer vocab:{prompt_tokenizer_vocab}")
    print(f"desc max id with configured tokenizer: {desc_max_id}")
    if desc_total:
        print(f"desc unk tokens with configured tokenizer: {desc_unk}/{desc_total}")

    if prompt_tokenizer_vocab != text_encoder_vocab:
        print(
            "multilingual-as-description: NO "
            f"(tokenizer vocab {prompt_tokenizer_vocab} != text encoder vocab {text_encoder_vocab}; "
            "token ids would address the wrong embedding table)"
        )
        if multilingual_desc_max_id >= text_encoder_vocab:
            print(
                "  sample also overflows: "
                f"max token id {multilingual_desc_max_id} >= text encoder vocab {text_encoder_vocab}"
            )
    else:
        print(
            "multilingual-as-description: vocab-size compatible on paper; "
            "still verify the text encoder was trained with that exact tokenizer"
        )

    print("sample prompts:")
    for idx, text in zip(prompt_indices[:3], prompt_texts[:3]):
        print(f"  [{idx}] {text[:220]}")
    print("sample descriptions:")
    for idx, text in zip(desc_indices[:5], desc_texts[:5]):
        print(f"  [{idx}] {text[:260]}")

    for p in problems:
        print(f"  PROBLEM: {p}")
    if not problems:
        print("  OK")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        help="Config id or path. Repeatable. Default: the six new Greek/multi_v2 configs.",
    )
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    configs = args.configs or DEFAULT_CONFIGS
    all_problems: list[str] = []
    for config in configs:
        try:
            all_problems.extend(f"{config}: {p}" for p in check_config(config, args.samples, args.seed))
        except Exception as e:  # noqa: BLE001
            all_problems.append(f"{config}: {type(e).__name__}: {e}")
            print(f"  ERROR: {type(e).__name__}: {e}")

    print("\n=== summary ===")
    if all_problems:
        print(f"{len(all_problems)} problem(s):")
        for p in all_problems:
            print(f"  - {p}")
        return 1
    print("all checked configs look ready to submit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
