#!/usr/bin/env python3
"""Shared ASR text metrics for the parakeet.cpp benchmark.

Two helpers, deliberately dependency-free so any script (and ctest) can import
them:

* ``normalize(text)`` — lowercase, strip punctuation, collapse whitespace. This
  is the canonical text normalization used before computing WER, so that
  casing/punctuation differences (e.g. NeMo's cased output vs LibriSpeech's
  UPPERCASE refs) don't inflate the error rate.
* ``wer(ref, hyp)`` — word-level error rate, a Levenshtein (edit) distance over
  whitespace-separated tokens divided by ``len(ref)``. Returns a float.

``scripts/validate_vs_nemo.py`` imports ``wer`` from here so there is a single
WER implementation in the tree.
"""
from __future__ import annotations

import re
import unicodedata

# Match any Unicode punctuation/symbol so the normalizer is multilingual-safe
# (the Italian diverse clip exercises this). We replace punctuation with a space
# rather than deleting it so "do for you, ask" -> "do for you ask" tokenizes the
# same as a transcript without the comma.
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    Unicode-normalized (NFKC) first so accented characters compare consistently.
    Returns a single-space-joined, stripped, lowercase string.
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def _edit_distance(r: list[str], h: list[str]) -> int:
    """Classic DP Levenshtein (substitution/insertion/deletion) over tokens."""
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cost = 0 if rw == hw else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[len(h)]


def wer(ref: str, hyp: str, *, normalize_text: bool = True) -> float:
    """Word-level error rate = edit_distance(ref, hyp) / len(ref).

    By default both inputs are passed through ``normalize`` first. Pass
    ``normalize_text=False`` to compare raw (already-tokenized) strings.

    Edge cases: empty ref + empty hyp -> 0.0; empty ref + non-empty hyp -> 1.0.
    """
    if normalize_text:
        ref = normalize(ref)
        hyp = normalize(hyp)
    r = ref.split()
    h = hyp.split()
    if not r:
        return 0.0 if not h else 1.0
    return _edit_distance(r, h) / len(r)
