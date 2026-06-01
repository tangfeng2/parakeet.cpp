#!/usr/bin/env bash
# Usage: bench_metal_dw.sh <model.gguf> <input.wav> [reps] [decoder]
#
# Measures steady-state inference speed (RTFx = audio_sec / proc_sec, higher is
# faster) and prints one markdown table row:
#   Metal (GPU) | CPU-only
#
# It drives `parakeet-cli bench`, which loads the model once, WARMS UP once
# (untimed, so the one-time Metal pipeline JIT + weight upload is excluded), then
# times ONLY transcription per manifest entry. We list the clip `reps` times and
# take the best (min proc_ms) to get a stable steady-state number, isolating
# inference from process-startup noise.
set -euo pipefail
MODEL="$1"; WAV="$2"; REPS="${3:-6}"; DEC="${4:-}"
CLI=./build/examples/cli/parakeet-cli

MAN=$(mktemp)
trap 'rm -f "$MAN"' EXIT
for _ in $(seq "$REPS"); do echo "$WAV" >> "$MAN"; done
decarg=(); [ -n "$DEC" ] && decarg=(--decoder "$DEC")

run() {  # echo best RTFx for the given env prefix; empty string on failure
  local prefix="$1" json
  # ${arr[@]+"${arr[@]}"} expands safely for an empty array under `set -u` on
  # bash 3.2 (macOS default), where a bare "${arr[@]}" is an unbound-variable error.
  json=$(env $prefix "$CLI" bench --model "$MODEL" --manifest "$MAN" ${decarg[@]+"${decarg[@]}"} 2>/dev/null) || { echo ""; return; }
  printf '%s' "$json" | python3 -c '
import sys, json
d = json.load(sys.stdin)
ms  = [f["proc_ms"] for f in d["files"]]
sec = d["files"][0]["audio_sec"]
print(f"{sec/(min(ms)/1000.0):.1f}")
' 2>/dev/null || echo ""
}

metal=$(run "")
cpu=$(run "PARAKEET_DEVICE=cpu")
printf "| %s | %s | %s |\n" "$(basename "$MODEL")" "${metal:-ERR}" "${cpu:-ERR}"
