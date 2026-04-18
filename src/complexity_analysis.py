"""
Sequence complexity analysis module.

This module provides a sliding-window complexity/degeneracy score and related metrics
designed to reward balanced base composition and low repetition, while penalizing
homopolymers and periodic repeats.

Even though input is cleaned by the calling GCEH module, 'N' nucleotide handling is
included for future versatility. Defensive programming (like sequence ambiguity handling)
for breakout use added.

_normalize_sequence(seq): uppercases input, converts U→T, and replaces non-ACGT with N.
_shannon_entropy_from_counts(counts): mono-nucleotide Shannon entropy (bits) over A/C/G/T.
_gc_balance(counts): GC balance score peaking at 0.5 GC, 0 at extremes.
_longest_run(seq): length of the longest homopolymer run (A/C/G/T only).
_kmer_richness(seq, k): distinct k-mer count normalized by min(4^k, valid windows).
_periodicity_penalty(seq, max_shift): maximum fixed-offset self-similarity; higher means more periodic.
window_metrics(seq, k, max_n_frac, gcbal_alpha): per-window composite complexity score and components
(entropy, GC balance, k-mer richness, homopolymer penalty, periodicity penalty, N fraction).
compute_complexity_track(sequence, window_size, step, k, max_n_frac, gcbal_alpha, return_components, smooth)
slides across a sequence to produce arrays of window start/end/mid coordinates and complexity scores
(optionally smoothed) plus component tracks for plotting or downstream analysis.

Primary entry point for notebooks:

    from complexity_analysis import compute_complexity_track
    result = compute_complexity_track(sequence, window_size=150, step=10, k=3)
    # result contains numpy arrays: start, end, mid, score, and components
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple
import math
import numpy as np

# -------------------------
# Helpers and basic metrics
# -------------------------

_BASES = ("A", "C", "G", "T")
_MAX_PERIODICITY_SHIFT = 32


def _normalize_sequence(seq: str) -> str:
    """
    Uppercases input, converts U→T, maps non-ACGT to 'N'.
    """

    s = seq.upper().replace("U", "T")
    if set(s) <= set("ACGTN"):
        return s

    return "".join(ch if ch in "ACGT" else "N" for ch in s)


def _shannon_entropy_from_counts(counts: Dict[str, int]) -> float:
    """
    Shannon entropy (bits) for counts over A,C,G,T only.

    Returns value in [0, 2] for four-symbol alphabet.
    """

    total = sum(counts.get(b, 0) for b in _BASES)
    if total == 0:
        return 0.0
    H = 0.0
    for b in _BASES:
        c = counts.get(b, 0)
        if c:
            p = c / total
            H -= p * math.log2(p)

    return H


def _gc_balance(counts: Dict[str, int]) -> float:
    """GC balance score in [0,1], peaking at GC=0.5 and decreasing linearly to 0 at 0 or 1."""

    total_called = sum(counts.get(b, 0) for b in _BASES)
    if total_called == 0:
        return 0.0
    gc = (counts.get("G", 0) + counts.get("C", 0)) / total_called
    return 1.0 - 2.0 * abs(gc - 0.5)


def _longest_run(seq: str) -> int:
    """Length of the longest homopolymer run (A/C/G/T only). 'N' breaks runs."""

    best = 0
    cur = 0
    last = None
    for ch in seq:
        if ch not in _BASES:
            cur = 0
            last = None
            continue
        if ch == last:
            cur += 1
        else:
            cur = 1
            last = ch
        if cur > best:
            best = cur

    return best


def _kmer_richness(seq: str, k: int = 3) -> float:
    """
    Distinct k-mers divided by the maximum possible (min(4^k, valid_windows)).
    Windows containing 'N' are skipped for counting. Returns value in [0,1].
    """

    n = len(seq)
    if n < k or k <= 0:
        return 0.0
    seen: set[str] = set()
    valid = 0
    # Slide and collect k-mers avoiding N
    for i in range(n - k + 1):
        kmer = seq[i : i + k]
        if "N" in kmer:
            continue
        valid += 1
        seen.add(kmer)
    if valid == 0:
        return 0.0
    # denom prevents score deflation in large windows.
    # maximum distinct k-mer count for a {4}-alphabert is 4^k.
    denom = min(4**k, valid)
    return len(seen) / denom


def _periodicity_penalty(seq: str, max_shift: Optional[int] = None) -> float:
    """
    Single-base periodicity penalty based on maximum self-similarity at fixed offsets.

    For shifts t=1..max_shift, compute fraction of matches between seq[i] and
    seq[i+t]. Returns the maximum fraction observed as a penalty in [0,1].
    Non-ACGT characters never match. Shift scans are capped to keep large
    GUI-selected windows from making the plot unexpectedly expensive.
    """

    n = len(seq)

    if max_shift is None:
        max_shift = min(_MAX_PERIODICITY_SHIFT, max(1, n // 6))
    else:
        max_shift = min(max_shift, _MAX_PERIODICITY_SHIFT)
    best = 0.0
    for t in range(1, min(max_shift, n - 1) + 1):
        matches = 0
        denom = n - t
        for i in range(denom):
            a, b = seq[i], seq[i + t]
            if a in _BASES and b in _BASES and a == b:
                matches += 1
        best = max(best, matches / denom if denom > 0 else 0.0)

    return best


def _center_crop_to_length(values: np.ndarray, target_len: int) -> np.ndarray:
    """Center-crop a convolution result so it matches the window track length."""
    if values.size == target_len:
        return values
    if values.size < target_len:
        raise ValueError(
            f"Cannot expand convolution result from {values.size} to {target_len}."
        )
    start = (values.size - target_len) // 2
    return values[start:start + target_len]


# -------------------------
# Window-level metric suite
# -------------------------

def window_metrics(
    seq: str,
    *,
    k: int = 3,
    max_n_frac: float = 0.2,
    gcbal_alpha: float = 1.0,
) -> Dict[str, float]:
    """
    Compute metrics and composite complexity score for a window.

    Parameters
    - seq: normalized window (ACGTN)
    - k: k-mer richness setting
    - max_n_frac: cut-off level for invalid characters
    - gcbal_alpha: exponent applied to GC-balance term (0.0 disables)

    Returns a dict with keys:
    - score: final composite score in [0,1]
    - H1_norm: normalized mono-nucleotide entropy in [0,1]
    - gcbal: GC balance in [0,1]
    - kmer_richness: distinct k-mers normalized to [0,1]
    - run_penalty: longest run length / window size in [0,1]
    - repeat_penalty: periodicity penalty in [0,1]
    - n_frac: fraction of 'N' in window
    """
    n = len(seq)
    if n == 0:
        return {
            "score": float("nan"),
            "H1_norm": float("nan"),
            "gcbal": float("nan"),
            "kmer_richness": float("nan"),
            "run_penalty": float("nan"),
            "repeat_penalty": float("nan"),
            "n_frac": float("nan"),
        }

    counts = Counter(ch for ch in seq if ch in _BASES)
    total_called = sum(counts.values())
    n_frac = 1.0 - (total_called / n)
    if n_frac > max_n_frac:
        nan = float("nan")
        return {
            "score": nan,
            "H1_norm": nan,
            "gcbal": nan,
            "kmer_richness": nan,
            "run_penalty": nan,
            "repeat_penalty": nan,
            "n_frac": n_frac,
        }

    H1 = _shannon_entropy_from_counts(counts)
    H1_norm = H1 / 2.0  # 2 bits max for 4 letters
    gcbal = _gc_balance(counts)
    run_pen = _longest_run(seq) / n
    rep_pen = _periodicity_penalty(seq)
    kmr = _kmer_richness(seq, k=k)

    # Composite score: composition × anti-repeat × (GC-balance^alpha)
    comp = 0.5 * H1_norm + 0.5 * kmr
    # Exponent allows down/up-weighting GC balance influence; alpha=0 -> ignore GC-balance
    gcbal_term = gcbal ** float(gcbal_alpha)
    score = comp * gcbal_term * (1.0 - max(run_pen, rep_pen))
    score = max(0.0, min(1.0, score))

    return {
        "score": score,
        "H1_norm": H1_norm,
        "gcbal": gcbal,
        "kmer_richness": kmr,
        "run_penalty": run_pen,
        "repeat_penalty": rep_pen,
        "n_frac": n_frac,
    }


# -------------------------
# Wrapper
# -------------------------

def compute_complexity_track(
    sequence: str,
    *,
    window_size: int = 150,
    step: int = 10,
    k: int = 3,
    max_n_frac: float = 0.2,
    gcbal_alpha: float = 1.0,
    return_components: bool = True,
    smooth: Optional[int] = None,
) -> Dict[str, np.ndarray | Dict[str, np.ndarray]]:
    """
    Compute sliding-window complexity track for plotting.

    Parameters
    - sequence: input DNA string; non-ACGT treated as 'N'.
    - window_size: window length W.
    - step: stride between windows.
    - k: k for k-mer richness metric.
    - max_n_frac: windows with > this fraction of N are NaN.
    - gcbal_alpha: exponent applied to GC-balance term; 0.0 removes GC-balance influence; >0.0 emphasizes it.
    - return_components: include per-metric arrays alongside score.
    - smooth: optional smoothing window size (in windows) for score; uses
        simple centered moving average (odd size recommended). Components are not smoothed.

    Returns a dict with numpy arrays:
    - start, end, mid: window coordinates (0-based, end-exclusive); mid is integer midpoint.
    - score: composite score in [0,1] (NaN where invalid).
    - components (if requested): dict of arrays for H1_norm, gcbal, kmer_richness, run_penalty,
        repeat_penalty, n_frac.
    """
    if window_size <= 0:
        raise ValueError("Window_size must be > 0")
    if step <= 0:
        raise ValueError("Step must be > 0")
    if k <= 0:
        raise ValueError("k must be > 0")

    seq = _normalize_sequence(sequence)
    n = len(seq)
    if n < window_size:
        # Nothing to compute; return empty arrays
        empty = np.array([], dtype=int)
        out_empty: Dict[str, np.ndarray | Dict[str, np.ndarray]] = {
            "start": empty,
            "end": empty,
            "mid": empty,
            "score": np.array([], dtype=float),
        }
        if return_components:
            out_empty["components"] = {
                "H1_norm": np.array([], dtype=float),
                "gcbal": np.array([], dtype=float),
                "kmer_richness": np.array([], dtype=float),
                "run_penalty": np.array([], dtype=float),
                "repeat_penalty": np.array([], dtype=float),
                "n_frac": np.array([], dtype=float),
            }
        return out_empty

    starts: List[int] = []
    ends: List[int] = []
    mids: List[int] = []
    scores: List[float] = []
    H1s: List[float] = []
    gcbals: List[float] = []
    kmrs: List[float] = []
    runps: List[float] = []
    repps: List[float] = []
    nfrs: List[float] = []

    for i in range(0, n - window_size + 1, step):
        j = i + window_size
        wseq = seq[i:j]
        m = window_metrics(wseq, k=k, max_n_frac=max_n_frac, gcbal_alpha=gcbal_alpha)
        starts.append(i)
        ends.append(j)
        mids.append(i + window_size // 2)
        scores.append(m["score"])
        H1s.append(m["H1_norm"])  # type: ignore[index]
        gcbals.append(m["gcbal"])  # type: ignore[index]
        kmrs.append(m["kmer_richness"])  # type: ignore[index]
        runps.append(m["run_penalty"])  # type: ignore[index]
        repps.append(m["repeat_penalty"])  # type: ignore[index]
        nfrs.append(m["n_frac"])  # type: ignore[index]

    start_arr = np.asarray(starts, dtype=int)
    end_arr = np.asarray(ends, dtype=int)
    mid_arr = np.asarray(mids, dtype=int)
    score_arr = np.asarray(scores, dtype=float)

    if smooth is not None and smooth > 1 and score_arr.size > 0:
        w = int(smooth)
        if w % 2 == 0:
            w += 1  # enforce odd for centered window
        kernel = np.ones(w, dtype=float) / w
        # NaN-safe centered moving average over N consecutive window scores: convolve scores
        # (NaN→0) and a validity mask separately, then divide — skips NaN scores without biasing the mean.
        mask = ~np.isnan(score_arr)
        sum_conv = np.convolve(np.where(mask, score_arr, 0.0), kernel, mode="same")
        cnt_conv = np.convolve(mask.astype(float), kernel, mode="same")
        if sum_conv.size != score_arr.size:
            sum_conv = _center_crop_to_length(sum_conv, score_arr.size)
            cnt_conv = _center_crop_to_length(cnt_conv, score_arr.size)
        with np.errstate(invalid="ignore"):
            score_arr = sum_conv / cnt_conv
        score_arr[cnt_conv == 0] = np.nan

    if score_arr.size != mid_arr.size:
        raise AssertionError(
            f"Complexity-track length mismatch: {score_arr.size} scores vs {mid_arr.size} positions."
        )

    out_computed: Dict[str, np.ndarray | Dict[str, np.ndarray]] = {
        "start": start_arr,
        "end": end_arr,
        "mid": mid_arr,
        "score": score_arr,
    }
    if return_components:
        out_computed["components"] = {
            "H1_norm": np.asarray(H1s, dtype=float),
            "gcbal": np.asarray(gcbals, dtype=float),
            "kmer_richness": np.asarray(kmrs, dtype=float),
            "run_penalty": np.asarray(runps, dtype=float),
            "repeat_penalty": np.asarray(repps, dtype=float),
            "n_frac": np.asarray(nfrs, dtype=float),
        }
    return out_computed
