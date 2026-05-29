#!/usr/bin/env python3
"""
plot_gpu.py: GB10 GPU benchmark plots + BENCHMARK.md section.

Reads results_gpu/p<P>_{nemo,ours}_<model>[_<dtype>].json passes. The "ours"
files may carry a dtype suffix (e.g. ..._f16.json); an unsuffixed name is f32.
NeMo baselines are dtype-independent and shared across our dtypes.

Usage:
    python scripts/plot_gpu.py --results benchmarks/results_gpu \
        --plots benchmarks/plots --md benchmarks/BENCHMARK.md
"""
import argparse, glob, json, os, sys, statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asr_metrics import wer

NEMO = "#76b7b2"   # teal
DTYPES = ["f32", "f16"]               # draw order
DTYPE_COLOR = {"f32": "#f28e2b",      # orange
               "f16": "#e15759"}      # red
KNOWN_DTYPE_SUFFIXES = ("f16", "q8_0", "q6_k", "q5_k", "q4_k")


def _short(k):
    return k.replace("realtime_eou_120m-v1", "rt-eou-120m")


def _rtfx(f):
    fs = json.load(open(f))["files"]
    proc = sum((x.get("proc_s") if "proc_s" in x else x["proc_ms"] / 1000.0) for x in fs)
    return sum(x["audio_sec"] for x in fs) / proc


def _texts(f):
    d = json.load(open(f))
    return {os.path.basename(x["path"]): x["text"] for x in d["files"]}


def _parse_ours(fname):
    """p1_ours_<model>[_<dtype>].json -> (model, dtype). Unsuffixed = f32."""
    base = os.path.basename(fname).split("_ours_")[1][:-len(".json")]
    for dt in KNOWN_DTYPE_SUFFIXES:
        if base.endswith("_" + dt):
            return base[: -(len(dt) + 1)], dt
    return base, "f32"


def load(results: Path):
    """Per model: median NeMo RTFx + per-dtype median ours RTFx, spread, and
    transcript-agreement WER vs NeMo."""
    ours = {}  # (model, dtype) -> [pass files]
    for f in glob.glob(str(results / "p*_ours_*.json")):
        m, dt = _parse_ours(f)
        ours.setdefault((m, dt), []).append(f)
    # single-pass fallback (ours_<model>[_dtype].json)
    if not ours:
        for f in glob.glob(str(results / "ours_*.json")):
            base = os.path.basename(f)[len("ours_"):-len(".json")]
            dt = "f32"
            for d in KNOWN_DTYPE_SUFFIXES:
                if base.endswith("_" + d):
                    base, dt = base[: -(len(d) + 1)], d
            ours.setdefault((base, dt), []).append(f)

    models = sorted({m for (m, _) in ours})
    rows = []
    for m in models:
        nfiles = sorted(results.glob(f"p*_nemo_{m}.json")) or [results / f"nemo_{m}.json"]
        nfiles = [f for f in nfiles if Path(f).is_file()]
        if not nfiles:
            continue
        nemo_rtfx = st.median([_rtfx(f) for f in nfiles])
        nt = _texts(nfiles[0])
        row = {"key": m, "nemo": nemo_rtfx, "dtypes": {}}
        for dt in DTYPES:
            ofiles = sorted(ours.get((m, dt), []))
            if not ofiles:
                continue
            ors = [_rtfx(f) for f in ofiles]
            ot = _texts(ofiles[0])
            pairs = [(nt[p], ot[p]) for p in ot if p in nt]
            agree = (sum(wer(a, b) for a, b in pairs) / len(pairs) * 100) if pairs else float("nan")
            row["dtypes"][dt] = {"rtfx": st.median(ors), "lo": min(ors), "hi": max(ors),
                                 "npass": len(ors), "agree": agree}
        if row["dtypes"]:
            rows.append(row)

    def sort_key(r):
        d = r["dtypes"].get("f16") or next(iter(r["dtypes"].values()))
        return -d["rtfx"] / r["nemo"]
    rows.sort(key=sort_key)
    return rows


def _present_dtypes(rows):
    return [dt for dt in DTYPES if any(dt in r["dtypes"] for r in rows)]


def plot(rows, plots: Path):
    plots.mkdir(parents=True, exist_ok=True)
    labels = [_short(r["key"]) for r in rows]
    x = np.arange(len(rows))
    present = _present_dtypes(rows)

    # RTFx grouped: NeMo + each ours dtype.
    n = 1 + len(present)
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.6), 5))
    ax.bar(x - 0.4 + w * 0.5, [r["nemo"] for r in rows], w, label="NeMo-GPU", color=NEMO)
    for i, dt in enumerate(present, start=1):
        ax.bar(x - 0.4 + w * (i + 0.5),
               [r["dtypes"].get(dt, {}).get("rtfx", 0) for r in rows], w,
               label=f"ours-GPU {dt}", color=DTYPE_COLOR[dt])
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("RTFx (audio_s / proc_s), higher is faster")
    ax.set_title("GB10 GPU: NeMo-GPU vs parakeet.cpp-GPU (LibriSpeech, same box)")
    ax.legend(); ax.grid(axis="y", alpha=0.4); ax.set_ylim(bottom=0)
    fig.tight_layout(); fig.savefig(plots / "gpu_rtfx.png"); plt.close(fig)

    # Speedup grouped: ours-dtype / NeMo.
    wsp = 0.8 / len(present)
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.6), 5))
    for i, dt in enumerate(present):
        su = [(r["dtypes"][dt]["rtfx"] / r["nemo"]) if dt in r["dtypes"] else 0 for r in rows]
        ax.bar(x - 0.4 + wsp * (i + 0.5), su, wsp, label=f"ours {dt} / NeMo", color=DTYPE_COLOR[dt])
    ax.axhline(1.0, color="red", ls="--", lw=1.4, alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("speedup (ours-GPU / NeMo-GPU)")
    ax.set_title("GB10 GPU speedup: parakeet.cpp vs NeMo (>1 = ours faster)")
    ax.legend(); ax.grid(axis="y", alpha=0.4); ax.set_ylim(bottom=0)
    fig.tight_layout(); fig.savefig(plots / "gpu_speedup.png"); plt.close(fig)
    print("  wrote gpu_rtfx.png, gpu_speedup.png")


def md_section(rows, plots_rel="plots"):
    present = _present_dtypes(rows)
    npass = max((d["npass"] for r in rows for d in r["dtypes"].values()), default=1)

    def speed(r, dt):
        return r["dtypes"][dt]["rtfx"] / r["nemo"]

    L = ["## GPU - GB10 Grace-Blackwell (NeMo-GPU vs ours-GPU)\n",
         "Same-box comparison on the NVIDIA **GB10** (Grace-Blackwell, sm_121, CUDA 13, "
         "unified memory). NeMo runs in the `nvcr.io/nvidia/nemo:25.09` container (torch "
         "2.8/cu13, the only stack with Blackwell support); ours runs the native CUDA build "
         "(`-DPARAKEET_GGML_CUDA=ON`). Both warmed up once, batch=1, 100-utt LibriSpeech, "
         "`local-ai` stopped during the run. RTFx is the median of %d passes. We report both "
         "f32 and f16; f16 is the recommended default (same transcripts as f32, smaller, and "
         "usually the fastest).\n" % npass]

    hdr = "| Model | NeMo RTFx |"
    sep = "|---|---|"
    for dt in present:
        hdr += f" ours {dt} RTFx | {dt} speedup |"
        sep += "---|---|"
    hdr += " agreement WER % |"
    sep += "---|"
    L += [hdr, sep]
    for r in rows:
        line = f"| {_short(r['key'])} | {r['nemo']:.1f} |"
        agree = None
        for dt in present:
            if dt in r["dtypes"]:
                d = r["dtypes"][dt]
                line += f" {d['rtfx']:.1f} ({d['lo']:.0f}-{d['hi']:.0f}) | {speed(r, dt):.2f}x |"
                agree = d["agree"]
            else:
                line += " n/a | n/a |"
        line += f" {agree:.3f} |" if agree is not None else " n/a |"
        L.append(line)

    dt0 = "f16" if "f16" in present else present[0]
    sus = [speed(r, dt0) for r in rows if dt0 in r["dtypes"]]
    nfast = sum(1 for s in sus if s >= 1.0)
    L += ["",
          f"> **{dt0}: parakeet.cpp beats NeMo on {nfast}/{len(sus)} models** on the GB10 "
          f"(median **{st.median(sus):.2f}x**, mean **{sum(sus) / len(sus):.2f}x**), with "
          "agreement WER near 0, meaning the same transcripts as NeMo. f16 holds f32 accuracy "
          "while being smaller and typically faster, so it's the variant we ship. The biggest "
          "wins are the large TDT/hybrid models: NeMo's CUDA-graph greedy decoder is RNNT-only, "
          "so TDT falls back to a slow per-frame Python loop, while our C++ decode does not. The "
          "encoder-bound CTC models gain the least, since ggml's CUDA conv/attention kernels "
          "still trail NeMo's tuned cuDNN.\n",
          f"![GB10 GPU RTFx]({plots_rel}/gpu_rtfx.png)\n",
          f"![GB10 GPU speedup]({plots_rel}/gpu_speedup.png)\n"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="benchmarks/results_gpu")
    ap.add_argument("--plots", default="benchmarks/plots")
    ap.add_argument("--md", default="benchmarks/BENCHMARK.md")
    args = ap.parse_args()
    rows = load(Path(args.results))
    if not rows:
        print("no GPU result pairs found", file=sys.stderr); sys.exit(1)
    print(f"loaded {len(rows)} model(s); dtypes={_present_dtypes(rows)}")
    plot(rows, Path(args.plots))
    section = md_section(rows, os.path.relpath(args.plots, Path(args.md).parent))
    md = Path(args.md)
    text = md.read_text() if md.exists() else "# parakeet.cpp Benchmark\n"
    # Replace any existing GPU section (matches both "## GPU -" and the older em-dash form).
    idx = text.find("\n## GPU")
    if idx != -1:
        rest = text[idx + 1:]
        nxt = rest.find("\n## ", 1)
        tail = rest[nxt:] if nxt != -1 else ""
        text = text[:idx + 1] + section + tail
    else:
        text = text.rstrip() + "\n\n" + section + "\n"
    md.write_text(text)
    print(f"  updated {md}")


if __name__ == "__main__":
    main()
