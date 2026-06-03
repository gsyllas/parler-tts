#!/usr/bin/env python
"""Generate WAV files for the Leonardo v2 eval sets.

This is a post-training listening pass: it loads each experiment's trained
model, runs generation on the same cached eval dataset used during training
when available, and writes generated WAVs, ground-truth WAVs, JSONL manifests,
and tab-separated transcription sheets.

Default experiments are the v2 runs:

    python leonardo/generate_eval_wavs.py

On Leonardo, run it on a GPU node through ``leonardo/slurm/generate_eval_wavs.slurm``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import wave
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V2_EXPERIMENTS = [
    "multi_v2_det",
    "multi_v2_llm",
    "greek_female_det",
    "greek_female_llm",
    "greek_male_det",
    "greek_male_llm",
]

np = None
torch = None
Audio = None
DatasetDict = None
concatenate_datasets = None
load_from_disk = None
tqdm = None
AutoTokenizer = None
ParlerTTSForConditionalGeneration = None


def _load_runtime_deps() -> None:
    global AutoTokenizer
    global Audio
    global DatasetDict
    global ParlerTTSForConditionalGeneration
    global concatenate_datasets
    global load_from_disk
    global np
    global torch
    global tqdm

    import numpy as _np
    import torch as _torch
    from datasets import Audio as _Audio
    from datasets import DatasetDict as _DatasetDict
    from datasets import concatenate_datasets as _concatenate_datasets
    from datasets import load_from_disk as _load_from_disk
    from tqdm import tqdm as _tqdm
    from transformers import AutoTokenizer as _AutoTokenizer

    from parler_tts import ParlerTTSForConditionalGeneration as _ParlerTTSForConditionalGeneration

    np = _np
    torch = _torch
    Audio = _Audio
    DatasetDict = _DatasetDict
    concatenate_datasets = _concatenate_datasets
    load_from_disk = _load_from_disk
    tqdm = _tqdm
    AutoTokenizer = _AutoTokenizer
    ParlerTTSForConditionalGeneration = _ParlerTTSForConditionalGeneration


def _config_path(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.exists() or path.suffix == ".json":
        return path
    return REPO_ROOT / "leonardo" / "configs" / f"{name_or_path}.json"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_split(path: str, split: str) -> Dataset:
    loaded = load_from_disk(path)
    if isinstance(loaded, DatasetDict):
        if split not in loaded:
            raise ValueError(f"{path}: missing split {split!r}; available={list(loaded)}")
        return loaded[split]
    return loaded


def _load_cached_eval_dataset(cfg: dict[str, Any]) -> Dataset | None:
    cache_path = cfg.get("save_to_disk")
    if not cache_path or not Path(cache_path).is_dir():
        return None
    loaded = load_from_disk(cache_path)
    if isinstance(loaded, DatasetDict):
        if "eval" not in loaded:
            return None
        return loaded["eval"]
    return None


def _load_raw_eval_dataset(cfg: dict[str, Any]) -> Dataset:
    split = cfg.get("eval_split_name") or cfg.get("train_split_name", "train")
    audio_path = cfg.get("eval_dataset_name") or cfg["train_dataset_name"]
    metadata_path = cfg.get("eval_metadata_dataset_name") or cfg.get("train_metadata_dataset_name")
    prompt_col = cfg.get("prompt_column_name", "text")
    desc_col = cfg.get("description_column_name", "text_description")
    audio_col = cfg.get("target_audio_column_name", "audio")

    audio = _load_split(audio_path, split)

    if metadata_path:
        metadata = _load_split(metadata_path, split)
        if len(audio) != len(metadata):
            raise ValueError(f"eval length mismatch: audio={len(audio)} metadata={len(metadata)}")

        # Metadata descriptions/prompts should win when columns overlap.
        if prompt_col in audio.column_names and prompt_col in metadata.column_names:
            audio = audio.remove_columns(prompt_col)
        duplicate_cols = set(audio.column_names).intersection(metadata.column_names)
        metadata = metadata.remove_columns(list(duplicate_cols))
        dataset = concatenate_datasets([audio, metadata], axis=1)
    else:
        dataset = audio

    missing = [col for col in (prompt_col, desc_col) if col not in dataset.column_names]
    if missing:
        raise ValueError(f"eval dataset missing columns {missing}; columns={dataset.column_names}")

    if cfg.get("max_eval_samples") is not None:
        max_eval_samples = min(int(cfg["max_eval_samples"]), len(dataset))
        dataset = dataset.shuffle(seed=int(cfg.get("seed", 0))).select(range(max_eval_samples))

    if cfg.get("max_text_length") is not None:
        dataset = dataset.filter(lambda x: len(str(x)) < int(cfg["max_text_length"]), input_columns=[desc_col])

    if audio_col in dataset.column_names:
        min_duration_s = float(cfg.get("min_duration_in_seconds", 0.0))
        max_duration_s = float(cfg.get("max_duration_in_seconds", float("inf")))

        def is_audio_in_length_range(audio: dict[str, Any]) -> bool:
            sampling_rate = audio.get("sampling_rate") or 1
            duration_s = len(audio["array"]) / sampling_rate
            return duration_s > min_duration_s and duration_s < max_duration_s

        dataset = dataset.filter(is_audio_in_length_range, input_columns=[audio_col])

    return dataset


def _tokenize_raw_eval_dataset(
    dataset: Dataset,
    cfg: dict[str, Any],
    description_tokenizer,
    prompt_tokenizer,
    num_proc: int | None,
) -> Dataset:
    prompt_col = cfg.get("prompt_column_name", "text")
    desc_col = cfg.get("description_column_name", "text_description")

    def tokenize(description: str, prompt: str) -> dict[str, list[int]]:
        return {
            "input_ids": description_tokenizer(str(description).strip())["input_ids"],
            "prompt_input_ids": prompt_tokenizer(str(prompt).strip())["input_ids"],
        }

    return dataset.map(
        tokenize,
        input_columns=[desc_col, prompt_col],
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc="tokenize raw eval",
    )


def _load_eval_dataset(
    cfg: dict[str, Any],
    source: str,
    description_tokenizer,
    prompt_tokenizer,
    num_proc: int | None,
) -> tuple[Dataset, str]:
    if source in {"cache", "auto"}:
        cached = _load_cached_eval_dataset(cfg)
        if cached is not None:
            return cached, "cache"
        if source == "cache":
            raise FileNotFoundError(f"no cached eval dataset found at {cfg.get('save_to_disk')!r}")

    raw = _load_raw_eval_dataset(cfg)
    return _tokenize_raw_eval_dataset(raw, cfg, description_tokenizer, prompt_tokenizer, num_proc), "raw"


def _apply_eval_window(dataset, args: argparse.Namespace):
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    if args.start_index:
        dataset = dataset.select(range(args.start_index, len(dataset)))
    return dataset


def _load_ground_truth_dataset(cfg: dict[str, Any], args: argparse.Namespace, sampling_rate: int):
    dataset = _load_raw_eval_dataset(cfg)
    audio_col = cfg.get("target_audio_column_name", "audio")
    if audio_col not in dataset.column_names:
        raise ValueError(f"ground truth dataset has no {audio_col!r} column; columns={dataset.column_names}")
    dataset = dataset.cast_column(audio_col, Audio(sampling_rate=sampling_rate))
    return _apply_eval_window(dataset, args)


def _resolve_model_path(cfg: dict[str, Any], checkpoint: str) -> tuple[Path, str]:
    output_dir = Path(cfg["output_dir"])
    if checkpoint == "final":
        return output_dir, "final"
    if checkpoint == "base":
        return Path(cfg["model_name_or_path"]), "base"
    path = Path(checkpoint)
    if path.exists():
        return path, path.name
    return output_dir / checkpoint, checkpoint


def _default_output_dir(cfg: dict[str, Any], exp_name: str, checkpoint_label: str) -> Path:
    run_dir = Path(cfg.get("output_dir", "")).parent
    if str(run_dir) in {"", "."}:
        run_dir = REPO_ROOT / "leonardo" / "runs" / exp_name
    return run_dir / "eval_wavs" / checkpoint_label


def _torch_dtype(name: str, device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
    lookup = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    if name not in lookup:
        raise ValueError(f"unsupported dtype {name!r}; choose one of {sorted(lookup)}")
    return lookup[name]


def _pad_batch(
    rows: list[dict[str, Any]],
    description_tokenizer,
    prompt_tokenizer,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    descriptions = [{"input_ids": row["input_ids"]} for row in rows]
    prompts = [{"input_ids": row["prompt_input_ids"]} for row in rows]

    desc_batch = description_tokenizer.pad(descriptions, return_tensors="pt", padding=True)
    prompt_batch = prompt_tokenizer.pad(prompts, return_tensors="pt", padding=True)

    batch = {
        "input_ids": desc_batch["input_ids"],
        "attention_mask": desc_batch.get("attention_mask"),
        "prompt_input_ids": prompt_batch["input_ids"],
        "prompt_attention_mask": prompt_batch.get("attention_mask"),
    }
    return {key: value.to(device) for key, value in batch.items() if value is not None}


def _write_wav(path: Path, audio: Any, sampling_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(audio, "detach"):
        audio_np = audio.detach().float().cpu().numpy()
    else:
        audio_np = np.asarray(audio, dtype=np.float32)
    audio_np = np.squeeze(audio_np)
    if audio_np.ndim != 1:
        audio_np = audio_np.reshape(-1)
    audio_np = np.nan_to_num(audio_np, nan=0.0, posinf=0.0, neginf=0.0)
    audio_i16 = np.clip(audio_np, -1.0, 1.0)
    audio_i16 = (audio_i16 * 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sampling_rate)
        wav.writeframes(audio_i16.tobytes())


def _decode_texts(rows: list[dict[str, Any]], description_tokenizer, prompt_tokenizer) -> tuple[list[str], list[str]]:
    descriptions = [row["input_ids"] for row in rows]
    prompts = [row["prompt_input_ids"] for row in rows]
    return (
        description_tokenizer.batch_decode(descriptions, skip_special_tokens=True),
        prompt_tokenizer.batch_decode(prompts, skip_special_tokens=True),
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tsv_field(value: Any) -> str:
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def _mode_generation_kwargs(
    mode: str,
    args: argparse.Namespace,
    max_length: int,
    min_new_tokens: int,
) -> dict[str, Any]:
    generation_kwargs: dict[str, Any] = {
        "do_sample": mode == "sample",
        "max_length": max_length,
        "min_new_tokens": min_new_tokens,
        "return_dict_in_generate": True,
    }
    if mode == "sample":
        generation_kwargs["temperature"] = args.temperature
        if args.top_p is not None:
            generation_kwargs["top_p"] = args.top_p
        if args.top_k is not None:
            generation_kwargs["top_k"] = args.top_k
    return generation_kwargs


def _open_metadata_files(out_dir: Path, args: argparse.Namespace):
    manifest_path = out_dir / "manifest.jsonl"
    transcriptions_path = out_dir / "transcriptions.tsv"
    manifest_mode = "w" if args.overwrite_manifest else "a"
    write_transcription_header = (
        manifest_mode == "w" or not transcriptions_path.exists() or transcriptions_path.stat().st_size == 0
    )
    return manifest_path, transcriptions_path, manifest_mode, write_transcription_header


def _write_transcription_row(
    transcriptions,
    wav_name: str,
    duration_s: float,
    prompt: str,
    description: str,
) -> None:
    transcriptions.write(
        "\t".join(
            [
                _tsv_field(wav_name),
                _tsv_field(round(duration_s, 6)),
                _tsv_field(prompt),
                _tsv_field(description),
            ]
        )
        + "\n"
    )
    transcriptions.flush()


def _write_ground_truth(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    exp_name: str,
    checkpoint_label: str,
    out_dir: Path,
    ground_truth_dataset,
    sampling_rate: int,
) -> None:
    gt_out_dir = out_dir / "ground_truth"
    audio_col = cfg.get("target_audio_column_name", "audio")
    prompt_col = cfg.get("prompt_column_name", "text")
    desc_col = cfg.get("description_column_name", "text_description")
    manifest_path, transcriptions_path, manifest_mode, write_header = _open_metadata_files(gt_out_dir, args)

    print(f"[ground_truth] writing {len(ground_truth_dataset)} wavs")
    gt_out_dir.mkdir(parents=True, exist_ok=True)

    with (
        manifest_path.open(manifest_mode, encoding="utf-8") as manifest,
        transcriptions_path.open(manifest_mode, encoding="utf-8") as transcriptions,
    ):
        if write_header:
            transcriptions.write("wav\tduration_s\ttranscription\tdescription\n")
        for dataset_i, row in enumerate(tqdm(ground_truth_dataset, desc=f"{exp_name} ground_truth")):
            wav_path = gt_out_dir / f"{dataset_i:06d}.wav"
            audio = row[audio_col]
            audio_array = audio["array"]
            if not wav_path.exists() or args.overwrite:
                _write_wav(wav_path, audio_array, sampling_rate)

            duration_s = len(audio_array) / sampling_rate if sampling_rate else math.nan
            prompt = str(row[prompt_col])
            description = str(row[desc_col])
            record = {
                "experiment": exp_name,
                "checkpoint": checkpoint_label,
                "mode": "ground_truth",
                "dataset_index": dataset_i,
                "wav": str(wav_path),
                "duration_s": round(duration_s, 6),
                "sampling_rate": sampling_rate,
                "prompt": prompt,
                "description": description,
            }
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            manifest.flush()
            _write_transcription_row(transcriptions, wav_path.name, duration_s, prompt, description)

    print(f"[ground_truth] wrote:     {gt_out_dir}")
    print(f"[ground_truth] manifest: {manifest_path}")
    print(f"[ground_truth] text:     {transcriptions_path}")


def _generate_mode(
    mode: str,
    args: argparse.Namespace,
    exp_name: str,
    checkpoint_label: str,
    source: str,
    out_dir: Path,
    eval_dataset,
    model,
    description_tokenizer,
    prompt_tokenizer,
    device,
    sampling_rate: int,
    frame_rate: int,
    max_length: int,
    min_new_tokens: int,
) -> None:
    mode_out_dir = out_dir / mode
    generation_kwargs = _mode_generation_kwargs(mode, args, max_length, min_new_tokens)

    print(
        f"[{mode}] generation: "
        f"do_sample={generation_kwargs['do_sample']} max_length={max_length} min_new_tokens={min_new_tokens}"
    )
    mode_out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path, transcriptions_path, manifest_mode, write_transcription_header = _open_metadata_files(
        mode_out_dir, args
    )
    with (
        manifest_path.open(manifest_mode, encoding="utf-8") as manifest,
        transcriptions_path.open(manifest_mode, encoding="utf-8") as transcriptions,
    ):
        if write_transcription_header:
            transcriptions.write("wav\tduration_s\ttranscription\tdescription\n")
        for batch_start in tqdm(range(0, len(eval_dataset), args.batch_size), desc=f"{exp_name} {mode}"):
            indices = list(range(batch_start, min(batch_start + args.batch_size, len(eval_dataset))))
            rows = [eval_dataset[int(i)] for i in indices]
            batch = _pad_batch(rows, description_tokenizer, prompt_tokenizer, device)

            with torch.no_grad():
                generated = model.generate(**batch, **generation_kwargs)

            audios = generated.sequences
            audio_lengths = list(getattr(generated, "audios_length", []))
            descriptions, prompts = _decode_texts(rows, description_tokenizer, prompt_tokenizer)

            for local_i, dataset_i in enumerate(indices):
                wav_path = mode_out_dir / f"{dataset_i:06d}.wav"

                length = int(audio_lengths[local_i]) if audio_lengths else int(audios[local_i].shape[0])
                audio = audios[local_i, :length]
                if not wav_path.exists() or args.overwrite:
                    _write_wav(wav_path, audio, sampling_rate)

                duration_s = length / sampling_rate if sampling_rate else math.nan
                code_steps = duration_s * frame_rate if not math.isnan(duration_s) else math.nan
                record = {
                    "experiment": exp_name,
                    "checkpoint": checkpoint_label,
                    "mode": mode,
                    "dataset_index": dataset_i,
                    "wav": str(wav_path),
                    "duration_s": round(duration_s, 6),
                    "approx_code_steps": round(code_steps, 2) if not math.isnan(code_steps) else None,
                    "sampling_rate": sampling_rate,
                    "source": source,
                    "do_sample": bool(generation_kwargs["do_sample"]),
                    "temperature": generation_kwargs.get("temperature"),
                    "top_p": generation_kwargs.get("top_p"),
                    "top_k": generation_kwargs.get("top_k"),
                    "prompt": prompts[local_i],
                    "description": descriptions[local_i],
                }
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifest.flush()
                _write_transcription_row(
                    transcriptions,
                    wav_path.name,
                    duration_s,
                    prompts[local_i],
                    descriptions[local_i],
                )

    print(f"[{mode}] wrote:     {mode_out_dir}")
    print(f"[{mode}] manifest: {manifest_path}")
    print(f"[{mode}] text:     {transcriptions_path}")


def generate_for_config(args: argparse.Namespace, config_path: Path) -> None:
    cfg = _load_json(config_path)
    exp_name = config_path.stem
    model_path, checkpoint_label = _resolve_model_path(cfg, args.checkpoint)
    out_dir = (
        Path(args.output_root) / exp_name / checkpoint_label
        if args.output_root
        else _default_output_dir(cfg, exp_name, checkpoint_label)
    )

    if not model_path.exists() and args.checkpoint != "base":
        raise FileNotFoundError(f"{exp_name}: model path does not exist: {model_path}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = _torch_dtype(args.dtype or cfg.get("dtype", "float32"), device)
    attn_implementation = args.attn_implementation or cfg.get("attn_implementation", "sdpa")

    print(f"\n=== {exp_name} ===")
    print(f"config:     {config_path}")
    print(f"model:      {model_path}")
    print(f"output:     {out_dir}")
    print(f"device:     {device} ({dtype})")
    print(f"attention:  {attn_implementation}")

    description_tokenizer = AutoTokenizer.from_pretrained(cfg["description_tokenizer_name"])
    prompt_tokenizer = AutoTokenizer.from_pretrained(cfg["prompt_tokenizer_name"], padding_side="left")

    eval_dataset, source = _load_eval_dataset(
        cfg,
        args.source,
        description_tokenizer,
        prompt_tokenizer,
        args.preprocessing_num_workers,
    )
    eval_dataset = _apply_eval_window(eval_dataset, args)
    print(f"eval rows:  {len(eval_dataset)} ({source})")

    if len(eval_dataset) == 0:
        print("nothing to generate")
        return

    _set_seed(int(args.seed if args.seed is not None else cfg.get("seed", 0)))

    model = ParlerTTSForConditionalGeneration.from_pretrained(
        str(model_path),
        attn_implementation={"decoder": attn_implementation, "text_encoder": "eager"},
    ).to(device, dtype=dtype)
    model.eval()
    model.generation_config.cache_implementation = None

    sampling_rate = int(model.audio_encoder.config.sampling_rate)
    frame_rate = int(getattr(model.audio_encoder.config, "frame_rate", 86))
    min_new_tokens = args.min_new_tokens if args.min_new_tokens is not None else model.decoder.config.num_codebooks + 1
    max_length = args.max_length if args.max_length is not None else int(cfg.get("max_length", 2580))

    if not args.skip_ground_truth:
        ground_truth_dataset = _load_ground_truth_dataset(cfg, args, sampling_rate)
        if len(ground_truth_dataset) != len(eval_dataset):
            raise ValueError(
                f"{exp_name}: ground-truth row count ({len(ground_truth_dataset)}) does not match "
                f"generation eval row count ({len(eval_dataset)}). Use --skip-ground-truth if you only "
                "want generated WAVs."
            )
        _write_ground_truth(args, cfg, exp_name, checkpoint_label, out_dir, ground_truth_dataset, sampling_rate)

    for mode in args.modes:
        mode_seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
        if mode == "sample":
            _set_seed(mode_seed)
        _generate_mode(
            mode=mode,
            args=args,
            exp_name=exp_name,
            checkpoint_label=checkpoint_label,
            source=source,
            out_dir=out_dir,
            eval_dataset=eval_dataset,
            model=model,
            description_tokenizer=description_tokenizer,
            prompt_tokenizer=prompt_tokenizer,
            device=device,
            sampling_rate=sampling_rate,
            frame_rate=frame_rate,
            max_length=max_length,
            min_new_tokens=min_new_tokens,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=DEFAULT_V2_EXPERIMENTS,
        help="Experiment ids under leonardo/configs/. Defaults to the six v2 runs.",
    )
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        help="Explicit config path or id. Repeatable. If set, --experiments is ignored.",
    )
    parser.add_argument(
        "--checkpoint",
        default="final",
        help="Model checkpoint to load: 'final' for config output_dir, 'base', a path, or a name under output_dir.",
    )
    parser.add_argument("--output-root", help="Optional root. Outputs go to <root>/<exp>/<checkpoint>/")
    parser.add_argument(
        "--source",
        choices=["auto", "cache", "raw"],
        default="raw",
        help=(
            "Eval source. Defaults to 'raw': generation and ground-truth are both re-derived from the "
            "same raw dataset, so the two WAV folders are row-aligned by construction. 'cache' matches "
            "the exact training eval subset but takes ground-truth from a separate raw reconstruction, "
            "so pairing is only correct if every filter replays identically (see README). 'auto' prefers "
            "cache and falls back to raw."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Eval batch size. Defaults to 1 for a faithful listening eval: with a single row there "
            "is no padding, so the left-padded prompt / right-padded description interactions and the "
            "train-time encoder masking that plain generate() does not replicate cannot color the audio. "
            "Raise it only to speed up large sweeps where exact fidelity matters less."
        ),
    )
    parser.add_argument("--limit", type=int, help="Generate only the first N eval rows.")
    parser.add_argument("--start-index", type=int, default=0, help="Skip eval rows before this index.")
    parser.add_argument("--device", help="Torch device. Defaults to cuda when available.")
    parser.add_argument("--dtype", help="float32, bfloat16, or float16. Defaults to the config dtype.")
    parser.add_argument(
        "--attn-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Override config attention implementation.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        help="Generation max_length in codec steps. Defaults to config max_length or 2580.",
    )
    parser.add_argument("--min-new-tokens", type=int, help="Defaults to num_codebooks + 1.")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["greedy", "sample"],
        default=["greedy", "sample"],
        help="Generation modes to write. Defaults to both: greedy/no-sample and sampled.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Shorthand for --modes sample.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--seed", type=int, help="Generation seed. Defaults to config seed.")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite existing WAV files.")
    parser.add_argument("--skip-ground-truth", action="store_true", help="Do not write ground_truth WAVs.")
    parser.add_argument(
        "--overwrite-manifest",
        action="store_true",
        help="Rewrite manifest.jsonl instead of appending.",
    )
    parser.add_argument("--preprocessing-num-workers", type=int, help="Workers for raw-source tokenization.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample:
        args.modes = ["sample"]
    else:
        args.modes = list(dict.fromkeys(args.modes))
    _load_runtime_deps()
    config_inputs = args.configs if args.configs else args.experiments
    config_paths = [_config_path(item) for item in config_inputs]
    for config_path in config_paths:
        if not config_path.exists():
            raise FileNotFoundError(f"missing config: {config_path}")
        generate_for_config(args, config_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
