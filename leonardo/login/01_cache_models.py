"""Pre-fetch every model parler-tts finetuning needs onto local cache.

Compute nodes on Leonardo have no internet, so HF / torch must find everything
offline. Run this once per repo clone on a LOGIN node (internet):

    source leonardo/env.sh && activate_conda_env
    huggingface-cli login    # needed: the gsyllas/* bases may be gated/private
    python leonardo/login/01_cache_models.py

Idempotent. Re-run any time a checkpoint / tokenizer id changes in env.sh.

What it stages (all into the in-repo HF / torch cache):
  - base checkpoints: multilingual v1.1 + the two Greek-finetuned derivatives
  - tokenizers + feature extractor: google/flan-t5-large, DAC 44kHz 8kbps
  - eval models: openai/whisper-large-v3 (Greek WER), CLAP, torchaudio SQUIM
  - the `wer` metric from `evaluate`
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _info(msg: str) -> None:
    print(f"[cache] {msg}", flush=True)


def _warn(msg: str) -> None:
    print(f"[cache][WARN] {msg}", file=sys.stderr, flush=True)


def normalize_cache_env() -> None:
    """Keep cache downloads inside this repo unless CACHE_ROOT is explicit."""
    cache_root = Path(os.environ.get("CACHE_ROOT", REPO_ROOT / "cache")).resolve()
    expected = {
        "HF_HOME": cache_root / "hf",
        "HF_HUB_CACHE": cache_root / "hf",
        "HUGGINGFACE_HUB_CACHE": cache_root / "hf",
        "TRANSFORMERS_CACHE": cache_root / "hf",
        "HF_DATASETS_CACHE": cache_root / "hf" / "datasets",
        "TORCH_HOME": cache_root / "torch",
    }
    for name, path in expected.items():
        old = os.environ.get(name)
        new = str(path)
        if old and Path(old).resolve() != path:
            _info(f"ignoring inherited {name}={old}; using {new}")
        os.environ[name] = new


def ids_from_env() -> dict[str, list[str]]:
    """Collect every repo id to cache, deduped, grouped by purpose."""
    checkpoints = [
        os.environ.get("BASE_MULTILINGUAL", "parler-tts/parler-tts-mini-multilingual-v1.1"),
        os.environ.get("BASE_GREEK_DET", "gsyllas/parler-tts-mini-multilingual-to-greek-v1.1_deterministic_60_epochs"),
        os.environ.get("BASE_GREEK_LLM", "gsyllas/parler-tts-mini-multilingual-to-greek-v1.1_52_epochs_speakers"),
    ]
    tokenizers = [
        os.environ.get("TEXT_TOKENIZER", "google/flan-t5-large"),
        os.environ.get("FEATURE_EXTRACTOR", "parler-tts/dac_44khZ_8kbps"),
    ]
    eval_models = [
        os.environ.get("ASR_MODEL", "openai/whisper-large-v3"),
        os.environ.get("CLAP_MODEL", "laion/larger_clap_music_and_speech"),
    ]
    return {"checkpoints": checkpoints, "tokenizers": tokenizers, "eval_models": eval_models}


def cache_repo(repo_id: str) -> None:
    """Full snapshot download of a model/tokenizer repo (config + weights)."""
    from huggingface_hub import snapshot_download

    _info(f"snapshot_download {repo_id}")
    try:
        path = snapshot_download(
            repo_id=repo_id,
            allow_patterns=[
                "*.json",
                "*.model",
                "*.safetensors",
                "*.bin",
                "*.txt",
                "*.py",
                "tokenizer*",
                "spiece.model",
                "special_tokens_map.json",
                "generation_config.json",
                "preprocessor_config.json",
            ],
        )
        _info(f"  -> {path}")
    except Exception as e:  # noqa: BLE001
        _warn(f"failed to snapshot {repo_id}: {type(e).__name__}: {e}")
        _warn("  if this is a gated/private repo, run `huggingface-cli login` first.")
        raise


def cache_squim() -> None:
    _info("downloading torchaudio SQUIM_OBJECTIVE (SI-SDR / PESQ / STOI)")
    try:
        from torchaudio.pipelines import SQUIM_OBJECTIVE

        model = SQUIM_OBJECTIVE.get_model()
        _info(f"  loaded {type(model).__name__}")
    except Exception as e:  # noqa: BLE001
        _warn(f"SQUIM download failed ({type(e).__name__}: {e}); eval still works without it")


def cache_wer_metric() -> None:
    _info("loading `wer` metric from evaluate (caches the metric script)")
    try:
        import evaluate

        evaluate.load("wer")
        _info("  wer metric cached")
    except Exception as e:  # noqa: BLE001
        _warn(f"evaluate.load('wer') failed ({type(e).__name__}: {e})")


def main() -> int:
    # Caches must be set BEFORE importing torch / hf_hub.
    normalize_cache_env()
    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)

    _info(f"HF_HOME    = {os.environ['HF_HOME']}")
    _info(f"TORCH_HOME = {os.environ['TORCH_HOME']}")

    groups = ids_from_env()
    for group, ids in groups.items():
        _info(f"--- {group} ---")
        for repo_id in dict.fromkeys(ids):  # dedupe, keep order
            cache_repo(repo_id)

    cache_squim()
    cache_wer_metric()

    _info("all model caches populated. Compute jobs can now run offline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
