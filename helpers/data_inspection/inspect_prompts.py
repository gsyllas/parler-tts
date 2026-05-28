"""Inspect Parler-TTS Greek training prompts as the model sees them.

Three subcommands:
  overview      - dataset shape, columns, top speakers, char histogram, flagged chars
  per-speaker   - dump N samples per speaker with raw text + tokenizer round-trip
  eval-dump     - dump every prompt in the eval split (small, full audit)

Run on Leonardo against the save_to_disk dirs referenced by the configs, e.g.
  /leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/dataspeech_out/female/05_merged_det

Usage examples:
  python inspect_prompts.py overview     --dataset <path> --split train
  python inspect_prompts.py per-speaker  --dataset <path> --split train --samples 25
  python inspect_prompts.py eval-dump    --dataset <path> --split eval
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import unicodedata
from collections import Counter, defaultdict

from datasets import DatasetDict, load_from_disk
from transformers import AutoTokenizer

DEFAULT_TOKENIZER = "parler-tts/parler-tts-mini-multilingual-v1.1"

# Allowed / expected character classes for clean Greek TTS prompts.
GREEK_RANGES = (
    (0x0370, 0x03FF),  # Greek and Coptic
    (0x1F00, 0x1FFF),  # Greek Extended
)
SAFE_PUNCT = set(".,;:!? '\"()-")  # standard sentence punctuation
SAFE_ASCII_EXTRA = set("0123456789")  # digits sometimes appear


def _in_ranges(cp, ranges):
    return any(lo <= cp <= hi for lo, hi in ranges)


def is_safe_char(ch: str) -> bool:
    if ch.isspace():
        return True
    cp = ord(ch)
    if _in_ranges(cp, GREEK_RANGES):
        return True
    if "a" <= ch.lower() <= "z":
        return True
    if ch in SAFE_PUNCT or ch in SAFE_ASCII_EXTRA:
        return True
    return False


SUSPICIOUS_SUBSTRINGS = [
    "---", "--", "...", "….", "..",
    ">>", "<<", ">>>", "<<<",
    "***", "**",
    "((", "))",
    "[", "]", "{", "}",
    "\\", "|", "~", "^", "_", "=", "+", "*",
    "@", "#", "$", "%", "&",
    "…", "—", "–",
    "«", "»", "“", "”", "‘", "’", "„", "‹", "›",
    "​", "‌", "‍", "﻿",  # zero-width / BOM
]


def load_split(dataset_path: str, split: str):
    ds = load_from_disk(dataset_path)
    if isinstance(ds, DatasetDict):
        if split not in ds:
            raise SystemExit(
                f"split {split!r} not in dataset; available: {list(ds)}"
            )
        return ds[split]
    return ds


def find_column(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def char_label(ch: str) -> str:
    cp = ord(ch)
    try:
        name = unicodedata.name(ch)
    except ValueError:
        name = "<unnamed>"
    return f"U+{cp:04X} {name!r}"


def cmd_overview(args):
    ds = load_split(args.dataset, args.split)
    cols = ds.column_names
    prompt_col = args.prompt_column or find_column(cols, ["text", "transcript", "prompt"])
    if prompt_col is None:
        raise SystemExit(f"could not find a prompt column in {cols}")
    speaker_col = args.speaker_column or find_column(
        cols, ["speaker_id", "speaker", "speaker_name", "client_id"]
    )

    print(f"== Dataset: {args.dataset}")
    print(f"== Split:   {args.split}   rows={len(ds)}")
    print(f"== Columns: {cols}")
    print(f"== Prompt column:  {prompt_col}")
    print(f"== Speaker column: {speaker_col}")
    print()

    if speaker_col is not None:
        spk_counts = Counter(ds[speaker_col])
        print(f"== Top {min(args.top_speakers, len(spk_counts))} speakers by row count:")
        for spk, n in spk_counts.most_common(args.top_speakers):
            print(f"   {n:>7d}   {spk}")
        print()

    char_counts = Counter()
    suspect_examples = defaultdict(list)
    flagged_char_examples = defaultdict(list)
    end_char_counts = Counter()

    texts = ds[prompt_col]
    for i, t in enumerate(texts):
        if not isinstance(t, str):
            continue
        s = t.strip()
        if not s:
            continue
        char_counts.update(s)
        end_char_counts[s[-1]] += 1
        for ch in set(s):
            if not is_safe_char(ch):
                if len(flagged_char_examples[ch]) < args.examples_per_finding:
                    flagged_char_examples[ch].append((i, s))
        for needle in SUSPICIOUS_SUBSTRINGS:
            if needle in s:
                if len(suspect_examples[needle]) < args.examples_per_finding:
                    suspect_examples[needle].append((i, s))

    print(f"== Total distinct characters: {len(char_counts)}")
    print("== Flagged (non-Greek / non-standard) characters:")
    if not flagged_char_examples:
        print("   (none)")
    else:
        for ch in sorted(flagged_char_examples, key=lambda c: -char_counts[c]):
            print(f"   {char_counts[ch]:>7d}   {char_label(ch)}")
            for idx, sample in flagged_char_examples[ch]:
                print(f"           row {idx}: {sample[:160]!r}")
    print()

    print("== Suspicious substrings:")
    any_hit = False
    for needle in SUSPICIOUS_SUBSTRINGS:
        hits = suspect_examples.get(needle)
        if not hits:
            continue
        any_hit = True
        print(f"   {needle!r}  ({len(hits)} sampled, may be more)")
        for idx, sample in hits:
            print(f"           row {idx}: {sample[:160]!r}")
    if not any_hit:
        print("   (none)")
    print()

    print("== Top sentence-ending characters (last char of stripped text):")
    for ch, n in end_char_counts.most_common(15):
        print(f"   {n:>7d}   {char_label(ch)}")


def cmd_per_speaker(args):
    ds = load_split(args.dataset, args.split)
    cols = ds.column_names
    prompt_col = args.prompt_column or find_column(cols, ["text", "transcript", "prompt"])
    speaker_col = args.speaker_column or find_column(
        cols, ["speaker_id", "speaker", "speaker_name", "client_id"]
    )
    if prompt_col is None:
        raise SystemExit(f"could not find a prompt column in {cols}")
    if speaker_col is None:
        raise SystemExit(
            f"could not find a speaker column in {cols}; pass --speaker-column"
        )

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    spk_counts = Counter(ds[speaker_col])
    if args.speakers:
        wanted = args.speakers
    else:
        wanted = [s for s, _ in spk_counts.most_common(args.top_speakers)]
    print(f"== Inspecting speakers: {wanted}")
    print(f"== Tokenizer: {args.tokenizer}")
    print()

    rng = random.Random(args.seed)
    by_speaker = defaultdict(list)
    for idx, spk in enumerate(ds[speaker_col]):
        if spk in wanted:
            by_speaker[spk].append(idx)

    for spk in wanted:
        idxs = by_speaker.get(spk, [])
        print(f"\n#### Speaker {spk!r}  total rows: {spk_counts[spk]}  sampling {min(args.samples, len(idxs))}")
        if not idxs:
            print("   (no rows)")
            continue
        sampled = idxs if len(idxs) <= args.samples else rng.sample(idxs, args.samples)
        sampled.sort()
        for i, idx in enumerate(sampled, 1):
            raw = ds[prompt_col][idx]
            s = (raw or "").strip()
            ids = tok(s)["input_ids"]
            roundtrip = tok.decode(ids, skip_special_tokens=True)
            flagged = sorted({ch for ch in s if not is_safe_char(ch)})
            suspect = [n for n in SUSPICIOUS_SUBSTRINGS if n in s]
            print(f"\n[{i:>2}] row={idx}")
            print(f"   raw         : {s!r}")
            print(f"   ends_with   : {char_label(s[-1]) if s else '(empty)'}")
            print(f"   roundtrip   : {roundtrip!r}")
            if s != roundtrip:
                print(f"   *** tokenizer changed the string ***")
            if flagged:
                print(f"   flagged_chars: {[char_label(c) for c in flagged]}")
            if suspect:
                print(f"   suspect_subs: {suspect}")
            if args.show_tokens:
                pieces = tok.convert_ids_to_tokens(ids)
                print(f"   tokens      : {pieces}")


def cmd_eval_dump(args):
    ds = load_split(args.dataset, args.split)
    cols = ds.column_names
    prompt_col = args.prompt_column or find_column(cols, ["text", "transcript", "prompt"])
    speaker_col = args.speaker_column or find_column(
        cols, ["speaker_id", "speaker", "speaker_name", "client_id"]
    )
    if prompt_col is None:
        raise SystemExit(f"could not find a prompt column in {cols}")

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    n = len(ds)
    limit = n if args.limit is None else min(args.limit, n)
    print(f"== Dumping {limit} of {n} rows from split {args.split!r}")
    print(f"== Tokenizer: {args.tokenizer}")
    print()

    for idx in range(limit):
        s = (ds[prompt_col][idx] or "").strip()
        ids = tok(s)["input_ids"]
        roundtrip = tok.decode(ids, skip_special_tokens=True)
        flagged = sorted({ch for ch in s if not is_safe_char(ch)})
        suspect = [needle for needle in SUSPICIOUS_SUBSTRINGS if needle in s]
        spk = ds[speaker_col][idx] if speaker_col else "?"
        print(f"[{idx:>3}] speaker={spk}")
        print(f"     raw      : {s!r}")
        print(f"     ends_with: {char_label(s[-1]) if s else '(empty)'}")
        if s != roundtrip:
            print(f"     roundtrip: {roundtrip!r}   *** mismatch ***")
        if flagged:
            print(f"     flagged  : {[char_label(c) for c in flagged]}")
        if suspect:
            print(f"     suspect  : {suspect}")
        print()


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--dataset", required=True, help="path to a save_to_disk dataset dir")
        sp.add_argument("--split", default="train")
        sp.add_argument("--prompt-column", default=None)
        sp.add_argument("--speaker-column", default=None)
        sp.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)

    sp = sub.add_parser("overview")
    add_common(sp)
    sp.add_argument("--top-speakers", type=int, default=20)
    sp.add_argument("--examples-per-finding", type=int, default=3)
    sp.set_defaults(func=cmd_overview)

    sp = sub.add_parser("per-speaker")
    add_common(sp)
    sp.add_argument("--speakers", nargs="+", default=None,
                    help="speaker ids to inspect; default = top-N by count")
    sp.add_argument("--top-speakers", type=int, default=2,
                    help="if --speakers not given, take the top N speakers by count")
    sp.add_argument("--samples", type=int, default=25)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--show-tokens", action="store_true",
                    help="also print BPE tokens from the tokenizer")
    sp.set_defaults(func=cmd_per_speaker)

    sp = sub.add_parser("eval-dump")
    add_common(sp)
    sp.add_argument("--limit", type=int, default=None,
                    help="dump at most N rows; default = all")
    sp.set_defaults(func=cmd_eval_dump)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
