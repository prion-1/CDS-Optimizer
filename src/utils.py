"""
Core utility functions and variables for pre-processing, optimization and analysis.

Referenced by: main.ipynb, pre_optimization.py, optimization.py, local_repair.py

Includes:
- Codon tables loading (LRU-cached)
- Precomputed CAI score vectors per host (cached)
- GC / GC3 content
- CAI calculation (numba-accelerated, cache-aware)
- tRNA Adaptation Index (tAI) — opt-in, requires per-host weight files
- Unwanted-motif counting (deterministic, overlap-counting on both paths)
- mRNA folding energy (ViennaRNA when available, deterministic fallback)
- Accessibility score
- Repetitive sequence and windowed repeat penalty
- Cryptic splice site count
- Internal start codon check
- DNA / protein translation helpers
- Codon-pair bias score (CPS) — empirical tables when bundled
- Windowed profile-correlation metric (CHARMING-style readout)
- Dynamic host registry loaded from data/codon_tables/
- Numba-accelerated helper functions
"""

import os
import re
import math
import warnings
import functools
import pandas as pd
import numpy as np
from typing import Callable, Dict, List, Tuple, Optional
from collections import defaultdict

# Try to import numba for acceleration
try:
    import numba as nb
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: Numba not installed. Using slower Python implementations.")

# Try to import ViennaRNA, but make it optional
try:
    import RNA
    VIENNA_AVAILABLE = True
except ImportError:
    VIENNA_AVAILABLE = False
    print("Warning: ViennaRNA not installed. Using simplified folding energy calculations.")


# ---------------------------------------------------------------------------
# Genetic code & data paths
# ---------------------------------------------------------------------------

# Standard genetic code
GENETIC_CODE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
}

# Reverse mapping: amino acid to codons
AA_TO_CODONS: Dict[str, List[str]] = defaultdict(list)
for codon, aa in GENETIC_CODE.items():
    AA_TO_CODONS[aa].append(codon)

# Single letter to three letter amino acid code
AA_SINGLE_TO_THREE = {
    'A': 'Ala', 'C': 'Cys', 'D': 'Asp', 'E': 'Glu', 'F': 'Phe',
    'G': 'Gly', 'H': 'His', 'I': 'Ile', 'K': 'Lys', 'L': 'Leu',
    'M': 'Met', 'N': 'Asn', 'P': 'Pro', 'Q': 'Gln', 'R': 'Arg',
    'S': 'Ser', 'T': 'Thr', 'V': 'Val', 'W': 'Trp', 'Y': 'Tyr',
    '*': 'Stop'
}

AA_THREE_TO_SINGLE = {v: k for k, v in AA_SINGLE_TO_THREE.items()}

# Resolve project root for data paths (this file lives in <root>/src/)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
CODON_TABLE_DIR = os.path.join(_PROJECT_ROOT, 'data', 'codon_tables')
TRNA_WEIGHTS_DIR = os.path.join(_PROJECT_ROOT, 'data', 'trna_weights')
CODON_PAIR_TABLE_DIR = os.path.join(_PROJECT_ROOT, 'data', 'codon_pair_tables')

# Stable 64-codon ordering and integer index map for vectorized scoring.
ALL_CODONS: List[str] = [
    c1 + c2 + c3
    for c1 in 'ACGT'
    for c2 in 'ACGT'
    for c3 in 'ACGT'
]
CODON_TO_INDEX: Dict[str, int] = {codon: i for i, codon in enumerate(ALL_CODONS)}
INDEX_TO_CODON: Dict[int, str] = {i: codon for i, codon in enumerate(ALL_CODONS)}
SENSE_CODONS: List[str] = [codon for codon in ALL_CODONS if GENETIC_CODE[codon] != '*']
SENSE_CODON_INDEX_MASK: np.ndarray = np.array(
    [GENETIC_CODE[codon] != '*' for codon in ALL_CODONS],
    dtype=bool,
)

# Per-host GC fall-backs derived from organism wide CDS GC% (Kazusa).
# Hosts without an explicit value fall back to whatever load_codon_table()
# computes from their codon-usage frequencies (see get_target_gc()).
HOST_TARGET_GC: Dict[str, float] = {
    'hsapiens':    52.3,
    'mmusculus':   52.0,
    'ecoli':       52.4,
    'scerevisiae': 39.8,
    'spombe':      39.8,
}

# Per-host GC3 targets used by the optional GC3 deviation fitness term.
# Hosts without an explicit value fall back to a codon-table-derived estimate
# from the host's raw third-position codon frequencies.
HOST_TARGET_GC3: Dict[str, float] = {
    'hsapiens':    60.0,
    'mmusculus':   60.0,
    'ecoli':       60.0,
    'scerevisiae': 25.0,
    'spombe':      25.0,
}

# Eukaryote / prokaryote tagging for the bundled hosts. Hosts not listed here
# fall back to "unknown" and the caller decides what to do.
HOST_KINGDOM: Dict[str, str] = {
    'hsapiens':    'eukaryote',
    'mmusculus':   'eukaryote',
    'scerevisiae': 'eukaryote',
    'spombe':      'eukaryote',
    'ecoli':       'prokaryote',
}


# ---------------------------------------------------------------------------
# Host registry — dynamic discovery from data/codon_tables/
# ---------------------------------------------------------------------------

def list_available_hosts() -> List[str]:
    """
    Return all hosts that currently have a codon-usage CSV available.

    Discovery is filesystem-driven, so dropping a new <host>.csv into
    data/codon_tables/ makes it immediately selectable from the rest of the
    pipeline (no constants to update).
    """
    if not os.path.isdir(CODON_TABLE_DIR):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(CODON_TABLE_DIR)
        if f.endswith('.csv') and not f.startswith('.')
    )


def is_eukaryote_host(host: str) -> Optional[bool]:
    """
    Return True/False/None for eukaryote/prokaryote/unknown.

    Used by the GUI host-class indicator and by the splice-site penalty
    (which only fires for eukaryotic hosts).
    """
    kind = HOST_KINGDOM.get(host.lower())
    if kind == 'eukaryote':
        return True
    if kind == 'prokaryote':
        return False
    return None


# ---------------------------------------------------------------------------
# Codon usage table loading (LRU-cached)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=64)
def load_codon_table(host: str) -> Dict[str, float]:
    """
    Load codon usage table for the specified host (relative adaptiveness per AA).

    Cached; subsequent calls for the same host are free. Use
    load_codon_table.cache_clear() if a CSV is edited at runtime.

    Returns:
        Dictionary mapping codons (DNA, uppercase) to their relative adaptiveness
        (0..1, normalised against the most-used codon for each amino acid).
    """
    host_lower = host.lower()
    available = list_available_hosts()
    if host_lower not in available:
        raise ValueError(
            f'Invalid host {host!r}. Available hosts: {", ".join(available) or "(none)"}'
        )

    filepath = os.path.join(CODON_TABLE_DIR, f'{host_lower}.csv')
    df = pd.read_csv(filepath)

    codon_table: Dict[str, float] = {}
    for aa in df['amino_acid'].unique():
        aa_df = df[df['amino_acid'] == aa]
        max_freq = aa_df['frequency'].max()
        for _, row in aa_df.iterrows():
            codon_table[row['codon']] = (
                row['frequency'] / max_freq if max_freq > 0 else 0.0
            )
    return codon_table


@functools.lru_cache(maxsize=64)
def load_codon_frequencies(host: str) -> Dict[str, float]:
    """
    Load *raw* per-amino-acid frequencies (sum to 1 within each AA) without
    rescaling against the most-used codon. Used for codon-pair bias and
    profile-correlation calculations where the absolute frequencies matter.
    """
    host_lower = host.lower()
    available = list_available_hosts()
    if host_lower not in available:
        raise ValueError(
            f'Invalid host {host!r}. Available hosts: {", ".join(available) or "(none)"}'
        )
    filepath = os.path.join(CODON_TABLE_DIR, f'{host_lower}.csv')
    df = pd.read_csv(filepath)
    return {row['codon']: float(row['frequency']) for _, row in df.iterrows()}


@functools.lru_cache(maxsize=64)
def get_codon_score_vector(host: str) -> np.ndarray:
    """
    Precomputed CAI score vector for a host: shape (64,), float32.

    Index `i` corresponds to ALL_CODONS[i] (CODON_TO_INDEX). Cached so the GA
    can grab it once per generation instead of rebuilding it per evaluation.
    """
    table = load_codon_table(host)
    vec = np.zeros(64, dtype=np.float32)
    for codon, idx in CODON_TO_INDEX.items():
        vec[idx] = float(table.get(codon, 0.0))
    return vec


@functools.lru_cache(maxsize=64)
def get_target_gc(host: str) -> float:
    """
    Target GC% for a host. Falls back to a derived value if the host is not
    in the bundled HOST_TARGET_GC table — computed by aggregating GC content
    over all codons weighted by their per-AA frequency. This makes new hosts
    usable end-to-end without manual constants.
    """
    host_lower = host.lower()
    if host_lower in HOST_TARGET_GC:
        return HOST_TARGET_GC[host_lower]

    freqs = load_codon_frequencies(host_lower)
    # Weighted average GC of (codon-fraction * gc-fraction-of-codon)
    weighted_gc_sum = 0.0
    weighted_total = 0.0
    for codon, f in freqs.items():
        gc = sum(1 for c in codon if c in 'GC') / 3.0
        weighted_gc_sum += f * gc
        weighted_total += f
    if weighted_total == 0:
        return 50.0
    return (weighted_gc_sum / weighted_total) * 100.0


@functools.lru_cache(maxsize=64)
def get_target_gc3(host: str) -> float:
    """
    Target GC3% for a host. Uses explicit bundled defaults when available,
    otherwise derives a target from raw codon frequencies in the host table.
    """
    host_lower = host.lower()
    if host_lower in HOST_TARGET_GC3:
        return HOST_TARGET_GC3[host_lower]

    freqs = load_codon_frequencies(host_lower)
    weighted_gc3_sum = 0.0
    weighted_total = 0.0
    for codon, f in freqs.items():
        weighted_gc3_sum += f * (1.0 if codon[2] in 'GC' else 0.0)
        weighted_total += f
    if weighted_total == 0:
        return 50.0
    return (weighted_gc3_sum / weighted_total) * 100.0


# ---------------------------------------------------------------------------
# Sequence helpers
# ---------------------------------------------------------------------------

def _seq_to_codon_indices(sequence: str) -> np.ndarray:
    """
    Convert a DNA sequence into an int32 array of codon indices into ALL_CODONS.
    Codons that contain unknown bases (anything outside ACGT) are dropped.
    """
    n_codons = len(sequence) // 3
    out = np.empty(n_codons, dtype=np.int32)
    write = 0
    for i in range(n_codons):
        codon = sequence[i * 3:i * 3 + 3]
        idx = CODON_TO_INDEX.get(codon)
        if idx is not None:
            out[write] = idx
            write += 1
    return out[:write]


# ---------------------------------------------------------------------------
# Composition metrics
# ---------------------------------------------------------------------------

def calculate_gc_content(sequence: str) -> float:
    """Return total GC content as a percentage (0-100)."""
    if not sequence:
        return 0.0

    if NUMBA_AVAILABLE:
        seq_array = np.array(list(sequence.encode('ascii')), dtype=np.uint8)
        gc_count = _calculate_gc_content_numba(seq_array)
        return (gc_count / len(sequence)) * 100
    else:
        gc_count = sequence.count('G') + sequence.count('C')
        return (gc_count / len(sequence)) * 100


def calculate_gc3_content(sequence: str) -> float:
    """Return third-position GC content as a percentage."""
    frame_aligned_len = len(sequence) - (len(sequence) % 3)
    if frame_aligned_len < 3:
        return 0.0
    third_positions = sequence[2:frame_aligned_len:3]
    if not third_positions:
        return 0.0
    gc3_count = sum(1 for nt in third_positions if nt in 'GC')
    return (gc3_count / len(third_positions)) * 100


# ---------------------------------------------------------------------------
# CAI
# ---------------------------------------------------------------------------

def calculate_cai(sequence: str, codon_table: Dict[str, float]) -> float:
    """
    Calculate Codon Adaptation Index (CAI) for a sequence.

    Returns CAI in [0, 1]. Codons with weight zero (or unknown) are excluded
    from the geometric mean — same convention as Sharp & Li (1987).
    """
    if NUMBA_AVAILABLE:
        # Build a 64-vector once from the supplied table
        codon_scores = np.zeros(64, dtype=np.float32)
        for codon, idx in CODON_TO_INDEX.items():
            codon_scores[idx] = float(codon_table.get(codon, 0.0))
        codon_indices = _seq_to_codon_indices(sequence)
        if codon_indices.size == 0:
            return 0.0
        return float(_calculate_cai_numba(codon_indices, codon_scores))

    # Pure-python fallback
    scores = []
    for i in range(0, len(sequence), 3):
        codon = sequence[i:i + 3]
        if codon in codon_table:
            score = codon_table[codon]
            if score > 0:
                scores.append(score)
    if not scores:
        return 0.0
    log_sum = sum(math.log(s) for s in scores)
    return math.exp(log_sum / len(scores))


def calculate_cai_fast(codon_indices: np.ndarray, score_vector: np.ndarray) -> float:
    """
    Fast CAI for callers that already hold a precomputed score vector and
    a codon-index array (e.g. the GA inner loop). Skips dictionary work.
    """
    if codon_indices.size == 0:
        return 0.0
    if NUMBA_AVAILABLE:
        return float(_calculate_cai_numba(codon_indices, score_vector))
    log_sum = 0.0
    n = 0
    for idx in codon_indices:
        s = float(score_vector[idx])
        if s > 0:
            log_sum += math.log(s)
            n += 1
    return math.exp(log_sum / n) if n else 0.0


# ---------------------------------------------------------------------------
# tRNA Adaptation Index (tAI) — opt-in, requires per-host weight file
# ---------------------------------------------------------------------------
#
# Format expected at data/trna_weights/<host>.csv:
#   codon,weight
#   AAA,0.31
#   AAC,0.85
#   ...
#
# Weights should be relative adaptiveness values in (0, 1] following the
# dos Reis et al. (2004) convention (geometric mean over a sequence's
# per-codon w-values is the tAI). Compute them from GtRNAdb anticodon copy
# numbers via the standard wobble-pairing weights, then normalize each
# w-value against the most-used w-value across all amino acids.
#
# When no file is present for a host, tAI calculations raise FileNotFoundError
# with an explanatory message; the GUI offers tAI as opt-in and the GA's
# default fitness weight for tAI is 0.0, so existing pipelines keep working.

@functools.lru_cache(maxsize=64)
def load_tai_weights(host: str) -> Dict[str, float]:
    """
    Load per-codon tAI weights for a host. Cached.

    Raises:
        FileNotFoundError: if no tAI weight file is present for the host.
    """
    host_lower = host.lower()
    filepath = os.path.join(TRNA_WEIGHTS_DIR, f'{host_lower}.csv')
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"No tAI weights for host {host_lower!r}. "
            f"Expected file: {filepath}\n"
            "tAI is opt-in. Provide a CSV with columns 'codon,weight' to enable it."
        )
    df = pd.read_csv(filepath)
    if not {'codon', 'weight'}.issubset(df.columns):
        raise ValueError(
            f"Invalid tAI weights format for {host_lower!r}. "
            "Expected columns 'codon,weight'."
        )

    weights = {
        str(row['codon']).upper(): float(row['weight'])
        for _, row in df.iterrows()
    }

    missing = sorted(codon for codon in SENSE_CODONS if codon not in weights)
    if missing:
        raise ValueError(
            f"Invalid tAI weights for {host_lower!r}: missing sense codons "
            f'{", ".join(missing)}'
        )

    invalid = [
        codon for codon in SENSE_CODONS
        if not math.isfinite(weights[codon]) or weights[codon] <= 0.0 or weights[codon] > 1.0
    ]
    if invalid:
        raise ValueError(
            f"Invalid tAI weights for {host_lower!r}: weights must be finite "
            f'and in (0, 1] for every sense codon. Invalid codons: {", ".join(sorted(invalid))}'
        )

    return {codon: weights[codon] for codon in SENSE_CODONS}


@functools.lru_cache(maxsize=64)
def get_tai_score_vector(host: str) -> np.ndarray:
    """Precomputed tAI w-vector indexed by ALL_CODONS for fast scoring."""
    weights = load_tai_weights(host)
    vec = np.zeros(64, dtype=np.float32)
    for codon, idx in CODON_TO_INDEX.items():
        vec[idx] = float(weights.get(codon, 0.0))
    return vec


def calculate_tai(sequence: str, host: str) -> float:
    """
    tRNA Adaptation Index (dos Reis et al. 2004) — geometric mean of per-codon
    w-values. Returns tAI in [0, 1]. Raises FileNotFoundError if the host has
    no bundled tAI weights.
    """
    score_vec = get_tai_score_vector(host)
    indices = _seq_to_codon_indices(sequence)
    return calculate_cai_fast(indices, score_vec)


def tai_available_for(host: str) -> bool:
    """True if tAI weights are bundled for this host."""
    return os.path.exists(os.path.join(TRNA_WEIGHTS_DIR, f'{host.lower()}.csv'))


# ---------------------------------------------------------------------------
# Codon-pair bias score (CPS) — empirical tables
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=64)
def load_codon_pair_table(host: str) -> pd.DataFrame:
    """
    Load an empirical codon-pair score table for a host.

    Expected CSV formats:
      - codon1,codon2,cps
      - pair,cps   where pair is six DNA bases (e.g. AAAACG)
    """
    host_lower = host.lower()
    filepath = os.path.join(CODON_PAIR_TABLE_DIR, f'{host_lower}.csv')
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"No codon-pair table for host {host_lower!r}. "
            f"Expected file: {filepath}\n"
            "Bundle an empirical CPS table to enable codon-pair scoring."
        )

    df = pd.read_csv(filepath)
    if {'codon1', 'codon2', 'cps'}.issubset(df.columns):
        out = df[['codon1', 'codon2', 'cps']].copy()
    elif {'pair', 'cps'}.issubset(df.columns):
        out = pd.DataFrame({
            'codon1': df['pair'].astype(str).str.upper().str[:3],
            'codon2': df['pair'].astype(str).str.upper().str[3:6],
            'cps': df['cps'],
        })
    else:
        raise ValueError(
            f"Invalid codon-pair table format for {host_lower!r}. "
            "Expected columns 'codon1,codon2,cps' or 'pair,cps'."
        )

    out['codon1'] = out['codon1'].astype(str).str.upper()
    out['codon2'] = out['codon2'].astype(str).str.upper()
    out['cps'] = out['cps'].astype(float)
    return out


def codon_pair_available_for(host: str) -> bool:
    """True if an empirical codon-pair table is bundled for this host."""
    return os.path.exists(os.path.join(CODON_PAIR_TABLE_DIR, f'{host.lower()}.csv'))


@functools.lru_cache(maxsize=64)
def get_cps_score_matrix(host: str) -> np.ndarray:
    """
    Load and cache a 64x64 empirical CPS matrix for the host.

    Rows/columns are indexed by ALL_CODONS. Pairs that are absent from the
    bundled table default to 0.0 so partially populated tables remain usable.
    """
    mat = np.zeros((64, 64), dtype=np.float32)
    df = load_codon_pair_table(host)
    for row in df.itertuples(index=False):
        idx1 = CODON_TO_INDEX.get(str(row.codon1).upper())
        idx2 = CODON_TO_INDEX.get(str(row.codon2).upper())
        if idx1 is None or idx2 is None:
            continue
        mat[idx1, idx2] = float(row.cps)
    return mat


def calculate_codon_pair_score(sequence: str, host: str) -> float:
    """
    Mean codon-pair score for a sequence under the host's CPS matrix.

    Higher = better. Returns 0 for sequences shorter than two sense codons.
    Stop-containing pairs are skipped because bundled CPS tables only cover
    61 x 61 sense-codon pairs.
    """
    indices = _seq_to_codon_indices(sequence)
    if indices.size < 2:
        return 0.0
    try:
        mat = get_cps_score_matrix(host)
    except FileNotFoundError as exc:
        warnings.warn(str(exc), RuntimeWarning)
        return 0.0
    left = indices[:-1]
    right = indices[1:]
    valid_pairs = SENSE_CODON_INDEX_MASK[left] & SENSE_CODON_INDEX_MASK[right]
    if not np.any(valid_pairs):
        return 0.0
    pair_scores = mat[left[valid_pairs], right[valid_pairs]]
    return float(np.mean(pair_scores))


# ---------------------------------------------------------------------------
# Profile-correlation metric (CHARMING-style readout)
# ---------------------------------------------------------------------------

def windowed_codon_track(
    sequence: str,
    codon_table: Dict[str, float],
    window_codons: int = 17,
    step_codons: int = 1,
) -> np.ndarray:
    """
    Windowed CAI track over a sequence.

    Returns a 1D array of geometric-mean codon adaptiveness across overlapping
    codon windows. The default window of 17 codons matches CHARMING's
    canonical harmonization window.
    """
    n_codons = len(sequence) // 3
    if n_codons == 0:
        return np.zeros(0, dtype=np.float32)

    weights = np.zeros(n_codons, dtype=np.float32)
    for i in range(n_codons):
        codon = sequence[i * 3:i * 3 + 3]
        weights[i] = float(codon_table.get(codon, 0.0))

    positive = weights > 0
    log_weights = np.zeros(n_codons, dtype=np.float64)
    log_weights[positive] = np.log(weights[positive])
    prefix_logs = np.concatenate(([0.0], np.cumsum(log_weights)))
    prefix_counts = np.concatenate(([0], np.cumsum(positive.astype(np.int32))))

    effective_window = min(window_codons, n_codons)
    if n_codons <= effective_window:
        starts = [0]
    else:
        starts = range(0, n_codons - effective_window + 1, step_codons)

    out = np.empty(
        (max(1, n_codons - effective_window + 1) + step_codons - 1) // step_codons,
        dtype=np.float32,
    )
    write = 0
    for start in starts:
        end = start + effective_window
        count = int(prefix_counts[end] - prefix_counts[start])
        if count == 0:
            out[write] = 0.0
        else:
            mean_log = (prefix_logs[end] - prefix_logs[start]) / count
            out[write] = float(np.exp(mean_log))
        write += 1
    return out[:write]


def windowed_minmax_track(
    sequence: str,
    host: str,
    window_codons: int = 17,
    step_codons: int = 1,
) -> np.ndarray:
    """
    Windowed %MinMax track for a sequence under a target host's codon usage.

    The implementation follows the CHARMING %MinMax definition but returns
    only the valid sliding-window values (no edge-padding zeros).
    """
    codons = [sequence[i:i + 3] for i in range(0, len(sequence) - 2, 3)]
    n_codons = len(codons)
    if n_codons == 0:
        return np.zeros(0, dtype=np.float32)

    freq_dict = load_codon_frequencies(host)
    effective_window = min(window_codons, n_codons)
    if n_codons <= effective_window:
        starts = [0]
    else:
        starts = range(0, n_codons - effective_window + 1, step_codons)

    out = np.empty(
        (max(1, n_codons - effective_window + 1) + step_codons - 1) // step_codons,
        dtype=np.float32,
    )
    write = 0
    for start in starts:
        window = codons[start:start + effective_window]
        actual = 0.0
        max_val = 0.0
        min_val = 0.0
        avg_val = 0.0
        used = 0

        for codon in window:
            aa = GENETIC_CODE.get(codon)
            if aa in (None, '*'):
                continue
            aa_codons = [c for c in AA_TO_CODONS.get(aa, []) if GENETIC_CODE.get(c) != '*']
            if not aa_codons:
                continue
            aa_freqs = [float(freq_dict.get(c, 0.0)) for c in aa_codons]
            actual += float(freq_dict.get(codon, 0.0))
            max_val += max(aa_freqs)
            min_val += min(aa_freqs)
            avg_val += sum(aa_freqs) / len(aa_freqs)
            used += 1

        if used == 0:
            out[write] = 0.0
            write += 1
            continue

        actual /= used
        max_val /= used
        min_val /= used
        avg_val /= used

        if actual >= avg_val:
            denom = max_val - avg_val
            value = ((actual - avg_val) / denom) * 100.0 if denom > 0 else 0.0
        else:
            denom = avg_val - min_val
            value = -((avg_val - actual) / denom) * 100.0 if denom > 0 else 0.0
        out[write] = float(value)
        write += 1

    return out[:write]


def _normalize_profile_measure(measure: str) -> str:
    normalized = measure.lower().replace('%', '').replace('-', '_')
    if normalized in ('cai', 'adaptiveness'):
        return 'cai'
    if normalized in ('minmax', 'percent_minmax', 'percentminmax'):
        return 'percent_minmax'
    raise ValueError(
        f"Unsupported profile measure {measure!r}. Expected 'cai' or '%MinMax'."
    )


def windowed_profile_track(
    sequence: str,
    target_host: str,
    window_codons: int = 17,
    measure: str = 'cai',
) -> np.ndarray:
    """Windowed harmonization profile track using CAI or %MinMax."""
    measure_key = _normalize_profile_measure(measure)
    if measure_key == 'cai':
        return windowed_codon_track(
            sequence,
            load_codon_table(target_host),
            window_codons=window_codons,
        )
    return windowed_minmax_track(
        sequence,
        target_host,
        window_codons=window_codons,
    )


def _paired_profile_tracks(
    source_sequence: str,
    target_sequence: str,
    target_host: str,
    window_codons: int = 17,
    measure: str = 'cai',
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Return aligned source/target profile tracks under the target host."""
    if len(source_sequence) != len(target_sequence):
        raise ValueError(
            "Source and target sequences must be the same length for "
            "profile correlation. Use the original CDS and the optimized CDS."
        )

    measure_key = _normalize_profile_measure(measure)
    src_track = windowed_profile_track(
        source_sequence,
        target_host,
        window_codons=window_codons,
        measure=measure_key,
    )
    tgt_track = windowed_profile_track(
        target_sequence,
        target_host,
        window_codons=window_codons,
        measure=measure_key,
    )
    n = min(src_track.size, tgt_track.size)
    return src_track[:n], tgt_track[:n], measure_key


def compute_profile_deviation(
    source_sequence: str,
    target_sequence: str,
    target_host: str,
    window_codons: int = 17,
    measure: str = 'cai',
) -> Dict[str, float]:
    """
    CHARMING-style net deviation between source and optimized window tracks.

    Returns the sum of absolute per-window differences (`net_abs_dev`) plus
    a convenience mean (`mean_abs_dev`) over the same windows.
    """
    src_track, tgt_track, measure_key = _paired_profile_tracks(
        source_sequence,
        target_sequence,
        target_host,
        window_codons=window_codons,
        measure=measure,
    )
    n = src_track.size
    if n == 0:
        return {
            'net_abs_dev': float('nan'),
            'mean_abs_dev': float('nan'),
            'n_windows': 0,
            'measure': measure_key,
        }
    abs_diff = np.abs(src_track - tgt_track)
    return {
        'net_abs_dev': float(np.sum(abs_diff)),
        'mean_abs_dev': float(np.mean(abs_diff)),
        'n_windows': int(n),
        'measure': measure_key,
    }


def compute_profile_correlation(
    source_sequence: str,
    target_sequence: str,
    target_host: str,
    window_codons: int = 17,
    measure: str = 'cai',
) -> Dict[str, float]:
    """Pearson correlation between source and optimized window tracks."""
    src_track, tgt_track, measure_key = _paired_profile_tracks(
        source_sequence,
        target_sequence,
        target_host,
        window_codons=window_codons,
        measure=measure,
    )
    n = src_track.size
    if n == 0:
        return {'pearson_r': float('nan'), 'n_windows': 0, 'measure': measure_key}
    if np.std(src_track) == 0 or np.std(tgt_track) == 0:
        r = float('nan')
    else:
        r = float(np.corrcoef(src_track, tgt_track)[0, 1])
    return {'pearson_r': r, 'n_windows': int(n), 'measure': measure_key}


def harmonization_correlation(
    source_sequence: str,
    target_sequence: str,
    target_host: str,
    window_codons: int = 17,
    measure: str = 'cai',
) -> Dict[str, float]:
    """
    Quantify harmonization quality with CHARMING-style diagnostics.

    Returns:
        dict with keys
            'pearson_r'         — Pearson correlation between the two tracks
            'net_abs_dev'       — sum of absolute per-window differences
            'mean_abs_dev'      — mean absolute deviation per window
            'n_windows'         — number of windows used
        Pearson r is NaN if either track has zero variance.
    """
    deviation = compute_profile_deviation(
        source_sequence,
        target_sequence,
        target_host,
        window_codons=window_codons,
        measure=measure,
    )
    correlation = compute_profile_correlation(
        source_sequence,
        target_sequence,
        target_host,
        window_codons=window_codons,
        measure=measure,
    )
    return {
        'pearson_r': correlation['pearson_r'],
        'net_abs_dev': deviation['net_abs_dev'],
        'mean_abs_dev': deviation['mean_abs_dev'],
        'n_windows': deviation['n_windows'],
        'measure': deviation['measure'],
    }


# ---------------------------------------------------------------------------
# Motif / repeat / splice / start codon penalties
# ---------------------------------------------------------------------------

def count_unwanted_motifs(sequence: str, motifs: List[str]) -> int:
    """
    Count occurrences of unwanted motifs in a sequence.

    OVERLAP-COUNTING is used on both the numba and pure-python paths so the
    metric is identical regardless of accelerator availability. (Previously
    the non-numba path used str.count() which is non-overlapping; this caused
    different optimization outcomes between environments.)

    Args:
        sequence: Input DNA
        motifs: List of motif sequences to avoid (default polyA signals are
                always added on top).

    Returns:
        Total count of motif hits, counting overlapping matches once each.
    """
    default_polya = ['AATAAA', 'ATTAAA', 'AAAAAA']
    all_motifs = list(motifs) + default_polya

    if NUMBA_AVAILABLE:
        seq_array = np.array(list(sequence.encode('ascii')), dtype=np.uint8)
        total = 0
        for motif in all_motifs:
            motif_array = np.array(list(motif.encode('ascii')), dtype=np.uint8)
            total += int(_count_motif_numba(seq_array, motif_array))
        return total

    # Deterministic pure-python overlap count
    total = 0
    for motif in all_motifs:
        if not motif:
            continue
        start = 0
        while True:
            idx = sequence.find(motif, start)
            if idx == -1:
                break
            total += 1
            start = idx + 1
    return total


def calculate_folding_energy_windowed(
    sequence: str, window_size: int = 150
) -> Tuple[float, float]:
    """
    Calculate mRNA folding energy focusing on the 5' region.

    Returns (worst_mfe, fraction_stable_windows). When ViennaRNA is not
    installed, a deterministic GC + palindrome heuristic is used.
    """
    rna_seq = sequence.replace('T', 'U')
    analysis_region = rna_seq[:min(window_size, len(rna_seq))]

    if VIENNA_AVAILABLE:
        try:
            _, mfe = RNA.fold(analysis_region)

            window_mfes = []
            step = 10
            sub_window = 30
            for i in range(0, len(analysis_region) - sub_window + 1, step):
                sub_seq = analysis_region[i:i + sub_window]
                _, sub_mfe = RNA.fold(sub_seq)
                window_mfes.append(sub_mfe)

            worst_mfe = min(window_mfes) if window_mfes else mfe
            stable_threshold = -10.0
            stable_count = sum(1 for m in window_mfes if m < stable_threshold)
            fraction_stable = stable_count / len(window_mfes) if window_mfes else 0.0
            return worst_mfe, fraction_stable
        except Exception as e:
            print(f"ViennaRNA error: {e}")

    # Deterministic fallback heuristic
    gc_content = calculate_gc_content(analysis_region)
    estimated_mfe = -0.5 * gc_content - 5.0
    hairpin_penalty = 0.0
    for i in range(len(analysis_region) - 10):
        segment = analysis_region[i:i + 10]
        rev_comp = segment[::-1].translate(str.maketrans('AUGC', 'UACG'))
        if segment == rev_comp:
            hairpin_penalty -= 2
    worst_mfe = estimated_mfe + hairpin_penalty
    fraction_stable = 0.3 if gc_content > 60 else 0.1
    return worst_mfe, fraction_stable


def calculate_window_mfe(rna_window: str) -> float:
    """
    Best-effort minimum free energy for a small RNA window.

    Used by the structure-aware preoptimization strategy. Returns a single float;
    when ViennaRNA is missing, falls back to a GC-based estimate so the
    beam search still has a usable signal (just less accurate).
    """
    if VIENNA_AVAILABLE:
        try:
            _, mfe = RNA.fold(rna_window.replace('T', 'U'))
            return float(mfe)
        except Exception:
            pass
    gc = calculate_gc_content(rna_window)
    return -0.45 * gc - 2.0


def calculate_accessibility_score(sequence: str, start_region_size: int = 30) -> float:
    """
    Approximate ribosome-binding accessibility from 5' folding energy.
    Higher score = more accessible.
    """
    start_region = sequence[:min(start_region_size, len(sequence))]

    if VIENNA_AVAILABLE:
        try:
            rna_seq = start_region.replace('T', 'U')
            _, mfe = RNA.fold(rna_seq)
            return 1.0 / (1.0 + math.exp(-mfe / 5.0))
        except Exception:
            pass

    gc_content = calculate_gc_content(start_region)
    accessibility = 1.0 - (gc_content / 100.0) * 0.5
    return max(0.0, min(1.0, accessibility))


def count_repetitive_sequences(
    sequence: str,
    homopolymer_threshold: int = 4,
    dinuc_threshold: int = 6,
    per_bp_scale: int = 1000,
) -> float:
    """
    Smooth repetitive-sequence penalty (homopolymers + dinucleotide repeats),
    normalised to a per-`per_bp_scale`-bp scale so scores are length-comparable.
    """
    penalty = 0.0

    # Homopolymers
    for nucleotide in 'ATGC':
        for match in re.finditer(rf'{nucleotide}+', sequence):
            length = len(match.group())
            if length >= homopolymer_threshold:
                penalty += (length - homopolymer_threshold + 1) * 0.5

    # Dinucleotide repeats (XY)+ with X != Y
    i = 0
    n = len(sequence)
    while i <= n - 2:
        a = sequence[i]
        b = sequence[i + 1]
        if a == b:
            i += 1
            continue
        j = i + 2
        ab = sequence[i:i + 2]
        while j <= n - 2 and sequence[j:j + 2] == ab:
            j += 2
        run_len = j - i
        if run_len >= dinuc_threshold:
            excess_units = (run_len - dinuc_threshold) // 2
            penalty += 0.3 * (1 + excess_units)
        i = j - 1

    if len(sequence) == 0:
        return 0.0
    return penalty * (per_bp_scale / len(sequence))


REPEAT_WINDOW_MIN_SEQUENCE_NT = 100
REPEAT_WINDOW_MAX_SEQUENCE_NT = 2000
REPEAT_WINDOW_MIN_NT = 30
REPEAT_WINDOW_MAX_NT = 500
REPEAT_WINDOW_INCREMENT_NT = 10


def repeat_penalty_window_params(
    sequence_length: int,
    *,
    min_sequence_length: int = REPEAT_WINDOW_MIN_SEQUENCE_NT,
    max_sequence_length: int = REPEAT_WINDOW_MAX_SEQUENCE_NT,
    min_window: int = REPEAT_WINDOW_MIN_NT,
    max_window: int = REPEAT_WINDOW_MAX_NT,
    window_increment: int = REPEAT_WINDOW_INCREMENT_NT,
) -> Tuple[int, int]:
    """
    Resolve length-aware repeat scan window/step settings.

    The caller should pass the validated sequence length that is actually
    scored by the optimizer. Windows scale linearly from 30/15 nt at 100 nt
    to 500/250 nt at 2000 nt, rounded down to stable increments.
    """
    if sequence_length < 0:
        raise ValueError('sequence_length must be non-negative')
    if min_sequence_length <= 0 or max_sequence_length <= min_sequence_length:
        raise ValueError('invalid sequence-length bounds')
    if min_window <= 0 or max_window < min_window:
        raise ValueError('invalid repeat-window bounds')
    if window_increment <= 0:
        raise ValueError('window_increment must be positive')

    clipped_length = min(
        max(sequence_length, min_sequence_length),
        max_sequence_length,
    )
    span_fraction = (
        (clipped_length - min_sequence_length)
        / (max_sequence_length - min_sequence_length)
    )
    raw_window = min_window + span_fraction * (max_window - min_window)
    rounded_window = int(math.floor(raw_window / window_increment) * window_increment)
    window = min(max(rounded_window, min_window), max_window)
    step = max(1, window // 2)
    return window, step


def repeat_penalty_windowed(
    sequence: str,
    window: Optional[int] = None,
    step: Optional[int] = None,
    threshold: float = 0.5,
) -> Tuple[float, float, float]:
    """
    Compute repeat penalties in sliding windows. Returns
    (mean_per_window, max_per_window, frac_windows_above_threshold).
    """
    n = len(sequence)
    auto_window, auto_step = repeat_penalty_window_params(n)
    if window is None:
        window = auto_window
    if step is None:
        step = auto_step if window == auto_window else max(1, int(window) // 2)
    if window <= 0 or step <= 0:
        raise ValueError('repeat penalty window and step must be positive')

    vals: List[float] = []
    if n <= window:
        vals.append(
            count_repetitive_sequences(sequence, per_bp_scale=max(1, n))
        )
    else:
        for i in range(0, n - window + 1, step):
            sub = sequence[i:i + window]
            vals.append(count_repetitive_sequences(sub, per_bp_scale=window))

    assert vals, 'No repeat penalty windows evaluated.'
    mean_pen = float(sum(vals) / len(vals))
    max_pen = float(max(vals))
    frac_bad = float(sum(1 for v in vals if v > threshold) / len(vals))
    return mean_pen, max_pen, frac_bad


def _count_cryptic_splice_sites_heuristic(sequence: str) -> int:
    """Simple motif scan for putative donor and acceptor-like splice signals."""
    count = 0
    count += len(re.findall(r'GT[AG]AG', sequence))
    count += len(re.findall(r'[CT]{5,}[ATGC]{0,10}AG', sequence))
    return count


def count_cryptic_splice_sites(
    sequence: str,
    is_eukaryote: bool,
    donor_threshold: Optional[float] = None,
    acceptor_threshold: Optional[float] = None,
) -> int:
    """
    Count putative cryptic splice sites with a lightweight motif heuristic.

    This is intentionally dependency-free. It counts donor-like `GT[AG]AG`
    motifs and acceptor-like polypyrimidine tracts ending in `AG`.
    Prokaryotic hosts always return 0. The threshold arguments are retained
    for backward API compatibility and are ignored by the heuristic.
    """
    if not is_eukaryote:
        return 0

    seq = sequence.upper().replace('U', 'T')
    return _count_cryptic_splice_sites_heuristic(seq)


def check_internal_start_codons(sequence: str, region_size: int = 90) -> int:
    """Count in-frame ATG codons in the 5' region (excluding the start codon)."""
    check_region = sequence[3:min(region_size, len(sequence))]
    count = 0
    for i in range(0, len(check_region) - 2, 3):
        if check_region[i:i + 3] == 'ATG':
            count += 1
    return count


# ---------------------------------------------------------------------------
# Translation helpers
# ---------------------------------------------------------------------------

def translate_dna_to_protein(sequence: str) -> str:
    """Translate DNA -> protein, terminating at the first stop codon."""
    protein = []
    for i in range(0, len(sequence), 3):
        codon = sequence[i:i + 3]
        aa = GENETIC_CODE.get(codon, 'X')
        if aa == '*':
            break
        protein.append(aa)
    return ''.join(protein)


def back_translate_protein(protein_seq: str, host: str = 'hsapiens') -> str:
    """
    Deterministic back-translation: pick the highest-weight host codon for
    each amino acid, with lexicographic tie-breaking.
    """
    codon_table = load_codon_table(host)
    dna_sequence: List[str] = []
    for aa in protein_seq:
        if aa == '*':
            break
        if aa not in AA_TO_CODONS:
            raise ValueError(f"Unknown amino acid code: {aa!r}")
        possible_codons = AA_TO_CODONS[aa]
        best_codon = max(possible_codons, key=lambda c: (codon_table.get(c, 0.0), c))
        dna_sequence.append(best_codon)
    return ''.join(dna_sequence)


def get_synonymous_codons(codon: str) -> List[str]:
    """All codons (including the input) coding for the same amino acid."""
    aa = GENETIC_CODE.get(codon)
    if not aa:
        return [codon]
    return AA_TO_CODONS[aa]


# ---------------------------------------------------------------------------
# Numba-accelerated helper functions
# ---------------------------------------------------------------------------

if NUMBA_AVAILABLE:
    @nb.jit(nopython=True)
    def _calculate_gc_content_numba(seq_array: np.ndarray) -> int:
        gc_count = 0
        for base in seq_array:
            if base == ord('G') or base == ord('C'):
                gc_count += 1
        return gc_count

    @nb.jit(nopython=True)
    def _calculate_cai_numba(codon_indices: np.ndarray, codon_scores: np.ndarray) -> float:
        log_sum = 0.0
        count = 0
        for idx in codon_indices:
            score = codon_scores[idx]
            if score > 0:
                log_sum += math.log(score)
                count += 1
        if count == 0:
            return 0.0
        return math.exp(log_sum / count)

    @nb.jit(nopython=True)
    def _count_motif_numba(seq_array: np.ndarray, motif_array: np.ndarray) -> int:
        count = 0
        seq_len = len(seq_array)
        motif_len = len(motif_array)
        if motif_len == 0 or motif_len > seq_len:
            return 0
        for i in range(seq_len - motif_len + 1):
            match = True
            for j in range(motif_len):
                if seq_array[i + j] != motif_array[j]:
                    match = False
                    break
            if match:
                count += 1
        return count
