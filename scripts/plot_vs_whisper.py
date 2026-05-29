#!/usr/bin/env python3
"""
plot_vs_whisper.py: head-to-head of parakeet.cpp vs whisper.cpp on the same
100-utt LibriSpeech set. Reads the bench result JSONs (same schema for both
engines: files[].audio_sec / proc_s / text), computes RTFx (CPU and GPU) and
WER vs ground truth, then writes a plot + a BENCHMARK.md section.

Inputs (under benchmarks/):
  whisper CPU : results_whisper/cpu_<model>.json
  whisper GPU : results_whisper/gpu_<model>.json
  parakeet CPU: results_whisper/cpu_parakeet_<slug>.json
  parakeet GPU: results_gpu/p*_ours_<slug>_f16.json  (median of passes)
"""
import argparse, glob, json, os, statistics as st, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asr_metrics import wer, normalize

WHISPER = "#4e79a7"   # blue
PARAKEET = "#e15759"  # red


def _rtfx(files):
    proc = sum((f.get("proc_s") if "proc_s" in f else f["proc_ms"] / 1000.0) for f in files)
    return sum(f["audio_sec"] for f in files) / proc if proc else float("nan")


def _wer_vs_truth(files, refs):
    pairs = [(refs[os.path.basename(f["path"])], f["text"]) for f in files
             if os.path.basename(f["path"]) in refs]
    if not pairs:
        return float("nan")
    return sum(wer(normalize(r), normalize(h)) for r, h in pairs) / len(pairs) * 100


def _load(path):
    p = Path(path)
    return json.load(open(p))["files"] if p.is_file() else None


def _load_gpu_median(glob_pat):
    """Median RTFx over multi-pass GPU files; transcripts from the first pass."""
    fs = sorted(glob.glob(glob_pat))
    if not fs:
        return None, None
    rtfxs = [_rtfx(json.load(open(f))["files"]) for f in fs]
    return st.median(rtfxs), json.load(open(fs[0]))["files"]


def load_refs(manifest):
    refs = {}
    for line in open(manifest):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            refs[os.path.basename(parts[0].strip())] = parts[1].strip()
    return refs


# (display, engine, params, cpu_json, gpu_path_or_glob)
ENTRIES = [
    ("whisper base.en", "whisper", "74M",
     "results_whisper/cpu_base.en.json", "results_whisper/gpu_base.en.json"),
    ("whisper large-v3-turbo", "whisper", "809M",
     "results_whisper/cpu_large-v3-turbo.json", "results_whisper/gpu_large-v3-turbo.json"),
    ("parakeet tdt_ctc-110m", "parakeet", "110M",
     "results_whisper/cpu_parakeet_tdt_ctc-110m.json", "results_gpu/p*_ours_tdt_ctc-110m_f16.json"),
    ("parakeet tdt-0.6b-v2", "parakeet", "0.6B",
     "results_whisper/cpu_parakeet_tdt-0.6b-v2.json", "results_gpu/p*_ours_tdt-0.6b-v2_f16.json"),
    ("parakeet tdt-1.1b", "parakeet", "1.1B",
     "results_whisper/cpu_parakeet_tdt-1.1b.json", "results_gpu/p*_ours_tdt-1.1b_f16.json"),
]


def collect(bench_dir: Path, refs):
    rows = []
    for disp, eng, params, cpu_rel, gpu_rel in ENTRIES:
        cpu_files = _load(bench_dir / cpu_rel)
        if "*" in gpu_rel:
            gpu_rtfx, gpu_files = _load_gpu_median(str(bench_dir / gpu_rel))
        else:
            gpu_files = _load(bench_dir / gpu_rel)
            gpu_rtfx = _rtfx(gpu_files) if gpu_files else None
        # WER: prefer CPU transcripts, fall back to GPU
        tfiles = cpu_files or gpu_files
        rows.append({
            "name": disp, "engine": eng, "params": params,
            "cpu_rtfx": _rtfx(cpu_files) if cpu_files else None,
            "gpu_rtfx": gpu_rtfx,
            "wer": _wer_vs_truth(tfiles, refs) if tfiles else None,
        })
    return rows


def plot(rows, out: Path):
    have = [r for r in rows if r["cpu_rtfx"] or r["gpu_rtfx"]]
    labels = [f"{r['name']}\n({r['params']})" for r in have]
    x = np.arange(len(have)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(9, len(have) * 1.7), 5.2))
    cpu = [r["cpu_rtfx"] or 0 for r in have]
    gpu = [r["gpu_rtfx"] or 0 for r in have]
    colors = [PARAKEET if r["engine"] == "parakeet" else WHISPER for r in have]
    b1 = ax.bar(x - w / 2, cpu, w, label="CPU RTFx", color=colors, alpha=0.55, edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + w / 2, gpu, w, label="GPU RTFx (GB10)", color=colors, edgecolor="black", linewidth=0.5)
    for r, xi in zip(have, x):
        if r["wer"] is not None:
            ax.text(xi, max(r["cpu_rtfx"] or 0, r["gpu_rtfx"] or 0) + 1, f"WER {r['wer']:.1f}%",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("RTFx (audio_s / proc_s), higher is faster")
    ax.set_title("parakeet.cpp (red) vs whisper.cpp (blue): RTFx on LibriSpeech (CPU light / GPU solid)")
    ax.legend(); ax.grid(axis="y", alpha=0.4)
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out / "vs_whisper.png"); plt.close(fig)
    print("  wrote vs_whisper.png")


def md_section(rows, plots_rel="plots"):
    L = ["## parakeet.cpp vs whisper.cpp\n",
         "Same 100-utt LibriSpeech set, same box. CPU is 20-core x86 (threads=8); "
         "GPU is the NVIDIA GB10. whisper.cpp runs `whisper-cli` (greedy, ggml models); "
         "parakeet runs the f16 GGUF. RTFx is audio-seconds over processing-seconds "
         "(higher is faster); WER is normalized word error rate vs the LibriSpeech "
         "ground truth (lower is better). proc time excludes model load on both sides.\n",
         "| Model | params | CPU RTFx | GPU RTFx | WER % |",
         "|---|---|---:|---:|---:|"]
    def fmt(v, s="{:.1f}"):
        return s.format(v) if isinstance(v, (int, float)) and v == v else "n/a"
    for r in rows:
        L.append(f"| {r['name']} | {r['params']} | {fmt(r['cpu_rtfx'])} | "
                 f"{fmt(r['gpu_rtfx'])} | {fmt(r['wer'])} |")
    L += ["", f"![parakeet vs whisper]({plots_rel}/vs_whisper.png)\n"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="benchmarks")
    ap.add_argument("--manifest", default="benchmarks/librispeech_manifest.tsv")
    ap.add_argument("--plots", default="benchmarks/plots")
    ap.add_argument("--md", default="benchmarks/BENCHMARK.md")
    a = ap.parse_args()
    refs = load_refs(a.manifest)
    rows = collect(Path(a.bench), refs)
    for r in rows:
        print(f"  {r['name']:26s} cpu={r['cpu_rtfx']} gpu={r['gpu_rtfx']} wer={r['wer']}")
    plot(rows, Path(a.plots))
    section = md_section(rows, os.path.relpath(a.plots, Path(a.md).parent))
    md = Path(a.md)
    text = md.read_text() if md.exists() else "# parakeet.cpp Benchmark\n"
    marker = "\n## parakeet.cpp vs whisper.cpp"
    idx = text.find(marker)
    if idx != -1:
        rest = text[idx + 1:]
        nxt = rest.find("\n## ", 1)
        text = text[:idx + 1] + section + (rest[nxt:] if nxt != -1 else "")
    else:
        text = text.rstrip() + "\n\n" + section + "\n"
    md.write_text(text)
    print(f"  updated {md}")


if __name__ == "__main__":
    main()
