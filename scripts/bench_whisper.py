#!/usr/bin/env python3
"""
bench_whisper.py: run whisper.cpp (whisper-cli) over a LibriSpeech-style
manifest and emit a JSON result in the same shape as the parakeet bench
(`{"model","threads","files":[{"path","audio_sec","proc_s","text"}]}`), so the
RTFx / WER plumbing can compare parakeet.cpp vs whisper.cpp on the same audio.

proc_s excludes model load (whisper-cli reloads per file; we subtract its
reported load time), matching the parakeet bench which loads once.

Usage:
    python scripts/bench_whisper.py --cli <whisper-cli> --model <ggml.bin> \
        --manifest benchmarks/librispeech_manifest.tsv --out out.json [--threads 8] [--lang en]
"""
import argparse, json, re, subprocess, sys, wave
from pathlib import Path


def audio_sec(p):
    with wave.open(p, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def run_one(cli, model, wav, threads, lang):
    cmd = [cli, "-m", model, "-f", wav, "-nt", "-t", str(threads)]
    if lang:
        cmd += ["-l", lang]
    r = subprocess.run(cmd, capture_output=True, text=True)
    text = " ".join(r.stdout.split()).strip()
    load = total = None
    for line in r.stderr.splitlines():
        m = re.search(r"load time\s*=\s*([\d.]+)\s*ms", line)
        if m:
            load = float(m.group(1))
        m = re.search(r"total time\s*=\s*([\d.]+)\s*ms", line)
        if m:
            total = float(m.group(1))
    if total is None:
        raise RuntimeError(f"no timing parsed for {wav}; stderr tail: {r.stderr[-300:]}")
    proc = (total - (load or 0.0)) / 1000.0
    return text, proc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--lang", default="en")
    a = ap.parse_args()

    files = []
    for line in open(a.manifest):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        wav = line.split("\t")[0].strip()
        if not wav:
            continue
        text, proc = run_one(a.cli, a.model, wav, a.threads, a.lang)
        files.append({"path": wav, "audio_sec": audio_sec(wav), "proc_s": proc, "text": text})
        print(f"  {Path(wav).name}: {proc:.2f}s", file=sys.stderr)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": a.model, "threads": a.threads, "files": files}, open(a.out, "w"))
    ta = sum(f["audio_sec"] for f in files)
    tp = sum(f["proc_s"] for f in files)
    print(f"RTFx={ta / tp:.2f} over {len(files)} files -> {a.out}")


if __name__ == "__main__":
    main()
