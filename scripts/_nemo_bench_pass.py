#!/usr/bin/env python3
"""NeMo benchmark pass — run as a SUBPROCESS (under /usr/bin/time -v for RSS).

Loads a NeMo ASR model on CPU, pins the thread count, selects the decoder head
to match what our ggml engine decodes (transducer for rnnt/tdt/hybrid models,
ctc for standalone CTC models), then transcribes each manifest entry with
``batch_size=1``, timing ONLY each ``transcribe`` call (``perf_counter``).

Emits a single JSON document to ``--out``:

    {"model","head","threads","nemo_version","load_s",
     "files":[{"path","audio_sec","proc_s","text"}, ...]}

This script is intentionally separate from benchmark.py: it must run in its own
process so ``/usr/bin/time -v`` measures the peak RSS of the NeMo/PyTorch pass
alone (not the orchestrator).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


def _read_manifest(path: str) -> list[str]:
    """Return audio paths from a TSV manifest, skipping ``#``/blank lines."""
    paths: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            wav = line.split("\t", 1)[0].strip()
            if wav:
                paths.append(wav)
    return paths


def _audio_sec(path: str) -> float:
    import soundfile as sf  # local import: only needed in this process

    info = sf.info(path)
    return info.frames / info.samplerate


def _transcript_text(out) -> str:
    """Pull plain text out of NeMo's transcribe() return (handles variants)."""
    hyps = out[0] if isinstance(out, tuple) else out
    first = hyps[0]
    return first.text if hasattr(first, "text") else str(first)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HF id or local .nemo")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--head", choices=["ctc", "rnnt"], required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--device", default="cpu",
                    help="torch device for NeMo (cpu | cuda). Default cpu.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch

    torch.set_num_threads(args.threads)

    try:
        import nemo
        from nemo.collections.asr.models import ASRModel
    except ImportError as e:  # pragma: no cover - env guard
        print(f"nemo-bench: NeMo import failed: {e}", file=sys.stderr)
        return 77

    nemo_version = getattr(nemo, "__version__", "unknown")

    is_local = Path(args.model).exists()
    t0 = time.perf_counter()
    try:
        if is_local:
            m = ASRModel.restore_from(args.model, map_location=args.device)
        else:
            m = ASRModel.from_pretrained(args.model, map_location=args.device)
    except Exception as e:  # pragma: no cover - network/cache guard
        print(f"nemo-bench: model load failed: {e}", file=sys.stderr)
        return 77
    m = m.to(args.device)
    m.eval()
    load_s = time.perf_counter() - t0

    # Select the decoder head. change_decoding_strategy(decoder_type=...) is only
    # meaningful for hybrid models; for a standalone CTC or transducer-only model
    # it can raise — that's fine, the model already uses its only head.
    decoder_type = "ctc" if args.head == "ctc" else "rnnt"
    if hasattr(m, "change_decoding_strategy") and hasattr(m, "cur_decoder"):
        try:
            m.change_decoding_strategy(decoder_type=decoder_type)
        except Exception as e:  # pragma: no cover - single-head model
            print(
                f"nemo-bench: change_decoding_strategy(decoder_type={decoder_type})"
                f" failed ({e}); using the model's default head",
                file=sys.stderr,
            )

    paths = _read_manifest(args.manifest)

    # Warm up once (untimed) so the first timed transcribe doesn't pay GPU JIT /
    # cuDNN autotune / lazy-init costs. Essential for fair GPU RTFx (the first
    # call is ~100x slower than steady state on CUDA).
    if paths:
        try:
            m.transcribe([paths[0]], batch_size=1)
        except Exception:
            pass

    files = []
    for p in paths:
        asec = _audio_sec(p)
        t = time.perf_counter()
        out = m.transcribe([p], batch_size=1)
        proc_s = time.perf_counter() - t
        files.append(
            {
                "path": p,
                "audio_sec": asec,
                "proc_s": proc_s,
                "text": _transcript_text(out),
            }
        )

    doc = {
        "model": args.model,
        "head": args.head,
        "threads": args.threads,
        "nemo_version": nemo_version,
        "load_s": load_s,
        "files": files,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(doc, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
