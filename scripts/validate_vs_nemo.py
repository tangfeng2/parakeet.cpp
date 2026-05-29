#!/usr/bin/env python3
"""Validate a converted parakeet.cpp GGUF end-to-end against NeMo (word-level WER).

Loads the NeMo reference model, selects a decoder head, gets NeMo's transcript via
``m.transcribe([audio])``, shells out to the C++ ``parakeet-cli transcribe`` on the
same GGUF + audio, computes a word-level WER (Levenshtein over whitespace tokens),
and prints a one-line summary plus both transcripts followed by ``PASS``/``FAIL``.

Head selection (``--head``):

* ``auto`` (default): a standalone CTC model (``EncDecCTCModelBPE`` / ``arch=ctc``)
  -> ``ctc``; an RNNT/TDT model (transducer, no aux CTC) -> ``rnnt``; a hybrid
  (``aux_ctc`` present) -> the transducer head (``rnnt``), which is NeMo's default
  ``cur_decoder``.
* ``ctc``  : force the CTC head.
* ``rnnt`` : force the transducer (RNNT/TDT) head.

The C++ CLI exposes only ``--decoder ctc|tdt``; both the ``rnnt`` and ``tdt`` heads
map to the CLI's ``tdt`` selector (the CLI picks the transducer head and routes by
the presence of TDT durations).

Exit codes: 0 = PASS (WER==0), 1 = FAIL (WER>0 or error), 77 = NeMo import failed
(skip-friendly, mirrors the ctest convention).
"""
import argparse
import pathlib
import subprocess
import sys
import warnings

# Single source of truth for the WER metric (scripts/asr_metrics.py).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from asr_metrics import wer as _wer  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

try:
    from nemo.collections.asr.models import ASRModel
except ImportError as e:  # pragma: no cover - env guard
    print(f"validate: NeMo import failed: {e}", file=sys.stderr)
    print("PARAKEET_NEMO_UNAVAILABLE", file=sys.stderr)
    sys.exit(77)


def _get(cfg, key, default=None):
    """Read ``key`` from an OmegaConf node or plain object, tolerating both."""
    try:
        return cfg[key]
    except Exception:
        return getattr(cfg, key, default)


def detect_arch(m):
    """Map a NeMo model to one of ctc/rnnt/tdt/hybrid_rnnt_ctc/hybrid_tdt_ctc.

    Mirrors scripts/convert_parakeet_to_gguf.py:detect_arch so the harness and the
    converter agree on the family/head, with no extra dependency.
    """
    cfg = m.cfg
    if _get(cfg, "aux_ctc") is not None:
        loss = _get(_get(cfg, "loss", {}) or {}, "loss_name", "")
        durs = _get(_get(cfg, "decoding", {}) or {}, "durations")
        return "hybrid_tdt_ctc" if (loss == "tdt" or durs) else "hybrid_rnnt_ctc"
    if _get(cfg, "joint") is not None:
        durs = _get(_get(cfg, "decoding", {}) or {}, "durations")
        nxo = _get(_get(cfg, "joint", {}) or {}, "num_extra_outputs", 0)
        return "tdt" if (durs or (nxo and nxo > 0)) else "rnnt"
    return "ctc"


def resolve_head(requested, arch):
    """Resolve the effective head ('ctc' or 'rnnt') from --head + detected arch."""
    if requested in ("ctc", "rnnt"):
        return requested
    # auto
    if arch == "ctc":
        return "ctc"
    # rnnt / tdt / hybrid_* -> the transducer head (NeMo default cur_decoder)
    return "rnnt"


def wer(ref, hyp):
    """Word-level error rate: Levenshtein distance over whitespace tokens / |ref|.

    Delegates to scripts/asr_metrics.wer. This harness has always compared the
    raw (unnormalized) whitespace tokens — an exact byte-for-byte parity check —
    so we keep that behavior by disabling the shared normalizer here.
    """
    return _wer(ref, hyp, normalize_text=False)


def nemo_text(m, head):
    """NeMo reference transcript for the selected head, as a plain string."""
    # A standalone CTC model has no transducer head; a transducer-only model has no
    # CTC head. change_decoding_strategy is only meaningful for hybrids (and is a
    # no-op-friendly switch otherwise), so only call it when the model can switch.
    if hasattr(m, "change_decoding_strategy") and hasattr(m, "cur_decoder"):
        try:
            decoder_type = "ctc" if head == "ctc" else "rnnt"
            m.change_decoding_strategy(decoder_type=decoder_type)
        except Exception as e:  # pragma: no cover - model can't switch to that head
            print(
                f"validate: change_decoding_strategy(decoder_type={decoder_type}) "
                f"failed ({e}); using the model's default head",
                file=sys.stderr,
            )
    out = m.transcribe([str(AUDIO)], batch_size=1)
    hyps = out[0] if isinstance(out, tuple) else out
    first = hyps[0]
    return first.text if hasattr(first, "text") else str(first)


def cpp_text(cli, gguf, audio, head):
    """C++ parakeet-cli transcript for the selected head."""
    cli_decoder = "ctc" if head == "ctc" else "tdt"  # rnnt/tdt -> CLI 'tdt'
    cmd = [
        str(cli), "transcribe",
        "--model", str(gguf),
        "--input", str(audio),
        "--decoder", cli_decoder,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"parakeet-cli exited {res.returncode}\ncmd: {' '.join(cmd)}\n"
            f"stderr:\n{res.stderr}"
        )
    return res.stdout.strip()


AUDIO = None  # set in main(); used by nemo_text


def main():
    global AUDIO
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HF id or local .nemo")
    ap.add_argument("--gguf", required=True, help="converted GGUF for the C++ port")
    ap.add_argument("--audio", required=True, help="16k mono wav clip")
    ap.add_argument("--head", choices=["ctc", "rnnt", "auto"], default="auto")
    ap.add_argument(
        "--cli",
        default="./build/examples/cli/parakeet-cli",
        help="path to the parakeet-cli binary",
    )
    args = ap.parse_args()
    AUDIO = pathlib.Path(args.audio)

    if not AUDIO.exists():
        print(f"validate: audio not found: {AUDIO}", file=sys.stderr)
        sys.exit(1)
    if not pathlib.Path(args.gguf).exists():
        print(f"validate: gguf not found: {args.gguf}", file=sys.stderr)
        sys.exit(1)
    if not pathlib.Path(args.cli).exists():
        print(f"validate: parakeet-cli not found: {args.cli}", file=sys.stderr)
        sys.exit(1)

    is_local = pathlib.Path(args.model).exists()
    try:
        if is_local:
            m = ASRModel.restore_from(args.model, map_location="cpu")
        else:
            m = ASRModel.from_pretrained(args.model, map_location="cpu")
    except Exception as e:  # pragma: no cover - network/cache guard
        print(f"PARAKEET_MODEL_UNAVAILABLE: {e}", file=sys.stderr)
        sys.exit(77)
    m.eval()

    arch = detect_arch(m)
    xscaling = bool(_get(_get(m.cfg, "encoder", {}) or {}, "xscaling", True))
    head = resolve_head(args.head, arch)

    ref = nemo_text(m, head)
    hyp = cpp_text(args.cli, args.gguf, args.audio, head)
    w = wer(ref, hyp)

    print(
        f"MODEL {args.model} HEAD {head} arch={arch} xscaling={str(xscaling).lower()} "
        f"WER {w:.4f}"
    )
    print(f"  NeMo: {ref!r}")
    print(f"  CPP:  {hyp!r}")
    if w == 0.0:
        print("PASS")
        sys.exit(0)
    print(f"FAIL WER>0 ({w:.4f})")
    print(f"  NeMo={ref!r}")
    print(f"  CPP={hyp!r}")
    sys.exit(1)


if __name__ == "__main__":
    main()
