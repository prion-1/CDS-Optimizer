"""
Shared degeneracy metrics for CDS scoring and candidate selection.

This module contains sequence-shape measurements that are independent from
host-specific expression terms such as CAI/tAI. The hybrid selector and future
parameter sweeps should use these functions so the same repeat/complexity
signals are compared across pipelines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from .complexity_analysis import compute_complexity_track
from .utils import (
    count_repetitive_sequences,
    repeat_penalty_window_params,
    repeat_penalty_windowed,
)


STRICT_GC_RUN_THRESHOLD = 8
STRICT_TANDEM_REPEAT_NT = 8
LOW_COMPLEXITY_CUTOFF = 0.56
REPEAT_BAD_WINDOW_THRESHOLD = 0.5
LONG_REPEAT_MIN_SEQUENCE_NT = 2000
LONG_REPEAT_CORE_NT = 2000
LONG_REPEAT_LOCAL_WINDOW_NT = 500
LONG_REPEAT_LOCAL_STEP_NT = 250
BROAD_REPEAT_FRAC_BAD_THRESHOLD = 0.45
BROAD_REPEAT_SPAN_FRACTION_THRESHOLD = 0.60

GA_REPEAT_PENALTY_MODE_LEGACY = 'legacy'
GA_REPEAT_PENALTY_MODE_BLEND = 'blend'
GA_REPEAT_PENALTY_MODE_ALIGNED = 'aligned'
GA_REPEAT_PENALTY_MODES = (
    GA_REPEAT_PENALTY_MODE_LEGACY,
    GA_REPEAT_PENALTY_MODE_BLEND,
    GA_REPEAT_PENALTY_MODE_ALIGNED,
)
DEFAULT_GA_REPEAT_PENALTY_MODE = GA_REPEAT_PENALTY_MODE_LEGACY
DEFAULT_GA_STRICT_DEGENERACY_WEIGHT = 1.0
DEFAULT_GA_REPEAT_PENALTY_SCALE = 20.0


@dataclass(frozen=True)
class RepeatBurdenWindow:
    start: int
    end: int
    score: float

    @property
    def center(self) -> float:
        return (self.start + self.end) / 2.0

    @property
    def is_bad(self) -> bool:
        return self.score > REPEAT_BAD_WINDOW_THRESHOLD


def homopolymer_runs(sequence: str) -> List[Tuple[str, int, int]]:
    runs: List[Tuple[str, int, int]] = []
    if not sequence:
        return runs

    start = 0
    current = sequence[0]
    for idx, base in enumerate(sequence[1:], start=1):
        if base == current:
            continue
        runs.append((current, start, idx - start))
        start = idx
        current = base
    runs.append((current, start, len(sequence) - start))
    return runs


def longest_binary_run(sequence: str, allowed: set[str]) -> int:
    best = 0
    current = 0
    for base in sequence:
        if base in allowed:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def binary_runs(sequence: str, allowed: set[str]) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    start = None

    for idx, base in enumerate(sequence):
        if base in allowed:
            if start is None:
                start = idx
        elif start is not None:
            runs.append((start, idx - start))
            start = None

    if start is not None:
        runs.append((start, len(sequence) - start))
    return runs


def tandem_repeat_runs(
    sequence: str,
    unit_size: int,
    min_bases: int = STRICT_TANDEM_REPEAT_NT,
) -> List[Tuple[int, int, str]]:
    runs: List[Tuple[int, int, str]] = []
    n = len(sequence)
    i = 0

    while i <= n - (2 * unit_size):
        unit = sequence[i:i + unit_size]
        j = i + unit_size
        while j <= n - unit_size and sequence[j:j + unit_size] == unit:
            j += unit_size

        run_len = j - i
        repeat_units = run_len // unit_size
        if repeat_units >= 2 and run_len >= min_bases:
            runs.append((i, run_len, unit))
            i = max(i + 1, j - unit_size + 1)
        else:
            i += 1

    return runs


def nan_stats(values: np.ndarray) -> Dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {'mean': math.nan, 'min': math.nan, 'max': math.nan, 'std': math.nan}

    return {
        'mean': float(np.mean(finite)),
        'min': float(np.min(finite)),
        'max': float(np.max(finite)),
        'std': float(np.std(finite)),
    }


def fraction_below(values: np.ndarray, cutoff: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    return float(np.mean(finite < cutoff))


def mean_or_zero(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def percentile_or_zero(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[int(position)])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def finite_or(value: Any, fallback: float) -> float:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return fallback


def normalize_ga_repeat_penalty_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in GA_REPEAT_PENALTY_MODES:
        valid = ', '.join(GA_REPEAT_PENALTY_MODES)
        raise ValueError(f'Invalid repeat_penalty_mode {mode!r}. Expected one of: {valid}.')
    return normalized


def _short_fragment(sequence: str, start: int, length: int, max_len: int = 32) -> str:
    fragment = sequence[start:start + length]
    if len(fragment) <= max_len:
        return fragment
    return fragment[:max_len] + '...'


def strict_degenerate_examples(
    sequence: str,
    *,
    min_gc_run: int = STRICT_GC_RUN_THRESHOLD,
    min_tandem_bases: int = STRICT_TANDEM_REPEAT_NT,
    max_examples: int = 5,
) -> Tuple[str, ...]:
    examples: List[Tuple[int, int, str, str]] = []

    for start, length in binary_runs(sequence, {'G', 'C'}):
        if length >= min_gc_run:
            examples.append((
                length,
                start,
                'GC-run',
                _short_fragment(sequence, start, length),
            ))

    for unit_size, label in ((2, 'dinuc'), (3, 'trinuc')):
        for start, length, unit in tandem_repeat_runs(
            sequence,
            unit_size,
            min_tandem_bases,
        ):
            examples.append((
                length,
                start,
                f'{label}:{unit}',
                _short_fragment(sequence, start, length),
            ))

    examples.sort(key=lambda item: (-item[0], item[1], item[2]))
    return tuple(
        f'{label}@{start + 1}:{fragment}'
        for _length, start, label, fragment in examples[:max_examples]
    )


def _repeat_burden_starts(
    sequence_length: int,
    window: int,
    step: int,
    *,
    include_terminal: bool,
) -> List[int]:
    if sequence_length <= 0:
        return [0]
    if sequence_length <= window:
        return [0]
    starts = list(range(0, sequence_length - window + 1, step))
    terminal = sequence_length - window
    if include_terminal and starts[-1] != terminal:
        starts.append(terminal)
    return starts


def repeat_burden_windows(
    sequence: str,
    *,
    window: int,
    step: int,
    include_terminal: bool = False,
) -> Tuple[RepeatBurdenWindow, ...]:
    """Return scored repeat windows with optional terminal coverage."""

    if window <= 0 or step <= 0:
        raise ValueError('repeat burden window and step must be positive')

    n = len(sequence)
    if n <= window:
        return (RepeatBurdenWindow(
            0,
            n,
            count_repetitive_sequences(sequence, per_bp_scale=max(1, n)),
        ),)

    windows: List[RepeatBurdenWindow] = []
    for start in _repeat_burden_starts(
        n,
        window,
        step,
        include_terminal=include_terminal,
    ):
        end = min(start + window, n)
        windows.append(RepeatBurdenWindow(
            start,
            end,
            count_repetitive_sequences(sequence[start:end], per_bp_scale=window),
        ))
    return tuple(windows)


def summarize_repeat_burden_windows(
    windows: Tuple[RepeatBurdenWindow, ...],
) -> Dict[str, float]:
    scores = [window.score for window in windows]
    bad_count = sum(1 for window in windows if window.is_bad)
    return {
        'window_count': float(len(windows)),
        'repeat_mean': mean_or_zero(scores),
        'repeat_max': max(scores) if scores else 0.0,
        'repeat_frac_bad': bad_count / len(windows) if windows else 0.0,
        'bad_window_count': float(bad_count),
    }


def merge_bad_repeat_regions(
    windows: Tuple[RepeatBurdenWindow, ...],
    *,
    sequence_length: int,
    max_gap: int,
) -> Tuple[Tuple[int, int, float], ...]:
    """Merge nearby bad repeat windows into broad sequence spans."""

    bad_windows = sorted(
        (window for window in windows if window.is_bad),
        key=lambda window: (window.start, window.end),
    )
    if not bad_windows:
        return ()

    merged: List[List[float]] = []
    for window in bad_windows:
        if not merged or window.start > merged[-1][1] + max_gap:
            merged.append([float(window.start), float(window.end), float(window.score)])
            continue
        merged[-1][1] = max(merged[-1][1], float(window.end))
        merged[-1][2] = max(merged[-1][2], float(window.score))

    return tuple(
        (max(0, int(start)), min(sequence_length, int(end)), float(score))
        for start, end, score in merged
    )


def repeat_burden_junction_windows(
    sequence: str,
    boundaries: List[int],
    *,
    window: int = LONG_REPEAT_LOCAL_WINDOW_NT,
    step: int = LONG_REPEAT_LOCAL_STEP_NT,
) -> Tuple[RepeatBurdenWindow, ...]:
    """Sample repeat burden around long-CDS segment boundaries."""

    n = len(sequence)
    if not boundaries:
        return ()

    starts = set()
    half_window = window // 2
    for boundary in boundaries:
        for offset in (-step, 0, step):
            center = boundary + offset
            start = int(round(center - half_window))
            start = max(0, min(start, max(0, n - window)))
            starts.add(start)

    scored: List[RepeatBurdenWindow] = []
    for start in sorted(starts):
        end = min(start + window, n)
        scored.append(RepeatBurdenWindow(
            start,
            end,
            count_repetitive_sequences(sequence[start:end], per_bp_scale=window),
        ))
    return tuple(scored)


def compute_long_repeat_burden_metrics(
    sequence: str,
    *,
    current_repeat_frac_bad: float | None = None,
) -> Dict[str, Any]:
    """
    Report capped local repeat-burden metrics for long CDS post-selection.

    For sequences longer than 2 kb this keeps the biologically capped 500/250 nt
    repeat scan, assigns windows to 2 kb cores, and reports the span occupied by
    merged bad local windows. Shorter sequences use the ordinary scaled window
    settings and are marked with `long_repeat_mode = False`.
    """

    n = len(sequence)
    long_mode = n > LONG_REPEAT_MIN_SEQUENCE_NT
    if long_mode:
        window = LONG_REPEAT_LOCAL_WINDOW_NT
        step = LONG_REPEAT_LOCAL_STEP_NT
        windows = repeat_burden_windows(
            sequence,
            window=window,
            step=step,
            include_terminal=True,
        )
        boundaries = list(range(LONG_REPEAT_CORE_NT, n, LONG_REPEAT_CORE_NT))
        junction = repeat_burden_junction_windows(
            sequence,
            boundaries,
            window=window,
            step=step,
        )
        segment_fracs: List[float] = []
        for core_start in range(0, n, LONG_REPEAT_CORE_NT):
            core_end = min(core_start + LONG_REPEAT_CORE_NT, n)
            assigned = tuple(
                window_obj
                for window_obj in windows
                if core_start <= window_obj.center < core_end
            )
            segment_fracs.append(
                summarize_repeat_burden_windows(assigned)['repeat_frac_bad']
            )
    else:
        window, step = repeat_penalty_window_params(n)
        windows = repeat_burden_windows(
            sequence,
            window=window,
            step=step,
            include_terminal=False,
        )
        junction = ()
        segment_fracs = [summarize_repeat_burden_windows(windows)['repeat_frac_bad']]

    local_summary = summarize_repeat_burden_windows(windows)
    junction_summary = summarize_repeat_burden_windows(junction)
    bad_regions = merge_bad_repeat_regions(
        windows,
        sequence_length=n,
        max_gap=step,
    )
    bad_span_nt = sum(end - start for start, end, _score in bad_regions)
    max_bad_span_nt = max((end - start for start, end, _score in bad_regions), default=0)
    if current_repeat_frac_bad is None:
        _, _, current_repeat_frac_bad = repeat_penalty_windowed(sequence)

    bad_region_span_fraction = bad_span_nt / n if n else 0.0
    broad_burden = (
        long_mode
        and current_repeat_frac_bad > BROAD_REPEAT_FRAC_BAD_THRESHOLD
        and bad_region_span_fraction > BROAD_REPEAT_SPAN_FRACTION_THRESHOLD
    )

    return {
        'long_repeat_mode': long_mode,
        'long_repeat_core_nt': LONG_REPEAT_CORE_NT,
        'long_repeat_window_nt': window,
        'long_repeat_step_nt': step,
        'long_repeat_window_count': int(local_summary['window_count']),
        'long_repeat_mean': local_summary['repeat_mean'],
        'long_repeat_max': local_summary['repeat_max'],
        'long_repeat_frac_bad': local_summary['repeat_frac_bad'],
        'long_repeat_segment_count': len(segment_fracs),
        'long_repeat_segment_frac_bad_mean': mean_or_zero(segment_fracs),
        'long_repeat_segment_frac_bad_max': max(segment_fracs) if segment_fracs else 0.0,
        'long_repeat_segment_frac_bad_p90': percentile_or_zero(segment_fracs, 0.90),
        'long_repeat_junction_window_count': int(junction_summary['window_count']),
        'long_repeat_junction_bad_count': int(junction_summary['bad_window_count']),
        'long_repeat_junction_repeat_max': junction_summary['repeat_max'],
        'long_repeat_junction_repeat_frac_bad': junction_summary['repeat_frac_bad'],
        'long_repeat_bad_region_count': len(bad_regions),
        'long_repeat_bad_region_span_nt': bad_span_nt,
        'long_repeat_bad_region_span_fraction': bad_region_span_fraction,
        'long_repeat_max_bad_region_span_nt': max_bad_span_nt,
        'long_repeat_bad_region_worst_score': max(
            (score for _start, _end, score in bad_regions),
            default=0.0,
        ),
        'long_repeat_bad_region_ranges': tuple(
            f'{start + 1}-{end}:{score:.3f}'
            for start, end, score in bad_regions[:8]
        ),
        'long_repeat_broad_burden': broad_burden,
    }


def calculate_degeneracy_score(metrics: Dict[str, Any]) -> float:
    complexity_min = finite_or(metrics.get('complexity_min'), LOW_COMPLEXITY_CUTOFF)
    complexity_shortfall = max(0.0, LOW_COMPLEXITY_CUTOFF - complexity_min)

    return (
        1.20 * finite_or(metrics.get('repeat_mean'), 0.0)
        + 1.80 * finite_or(metrics.get('repeat_max'), 0.0)
        + 2.50 * finite_or(metrics.get('repeat_frac_bad'), 0.0)
        + 0.45 * max(
            0.0,
            finite_or(metrics.get('longest_gc_only_run'), 0.0)
            - STRICT_GC_RUN_THRESHOLD
            + 1.0,
        )
        + 0.30 * finite_or(metrics.get('gc_run_excess'), 0.0)
        + 0.55 * max(0.0, finite_or(metrics.get('longest_homopolymer'), 0.0) - 4.0)
        + 0.70 * finite_or(metrics.get('homopolymer_runs_ge5'), 0.0)
        + 0.20 * max(
            0.0,
            finite_or(metrics.get('max_dinuc_tandem_bases'), 0.0)
            - STRICT_TANDEM_REPEAT_NT
            + 1.0,
        )
        + 0.15 * max(
            0.0,
            finite_or(metrics.get('max_trinuc_tandem_bases'), 0.0)
            - STRICT_TANDEM_REPEAT_NT
            + 1.0,
        )
        + 5.00 * complexity_shortfall
        + 1.50 * finite_or(metrics.get('low_complexity_frac'), 0.0)
    )


def calculate_legacy_repeat_score(metrics: Dict[str, Any]) -> float:
    """Current GA repeat score based only on windowed repeat burden."""

    repeat_score = (
        0.60 * finite_or(metrics.get('repeat_mean'), 0.0)
        + 0.40 * finite_or(metrics.get('repeat_max'), 0.0)
    )
    if finite_or(metrics.get('repeat_frac_bad'), 0.0) > 0.30:
        repeat_score *= 1.10
    return repeat_score


def calculate_strict_degeneracy_component(metrics: Dict[str, Any]) -> float:
    """Strict non-legacy component for opt-in GA repeat alignment modes."""

    complexity_min = finite_or(metrics.get('complexity_min'), LOW_COMPLEXITY_CUTOFF)
    complexity_shortfall = max(0.0, LOW_COMPLEXITY_CUTOFF - complexity_min)

    return (
        0.45 * max(
            0.0,
            finite_or(metrics.get('longest_gc_only_run'), 0.0)
            - STRICT_GC_RUN_THRESHOLD
            + 1.0,
        )
        + 0.30 * finite_or(metrics.get('gc_run_excess'), 0.0)
        + 0.55 * max(0.0, finite_or(metrics.get('longest_homopolymer'), 0.0) - 4.0)
        + 0.70 * finite_or(metrics.get('homopolymer_runs_ge5'), 0.0)
        + 0.20 * max(
            0.0,
            finite_or(metrics.get('max_dinuc_tandem_bases'), 0.0)
            - STRICT_TANDEM_REPEAT_NT
            + 1.0,
        )
        + 0.15 * max(
            0.0,
            finite_or(metrics.get('max_trinuc_tandem_bases'), 0.0)
            - STRICT_TANDEM_REPEAT_NT
            + 1.0,
        )
        + 5.00 * complexity_shortfall
        + 1.50 * finite_or(metrics.get('low_complexity_frac'), 0.0)
    )


def calculate_ga_repeat_score(
    metrics: Dict[str, Any],
    *,
    mode: str = DEFAULT_GA_REPEAT_PENALTY_MODE,
    strict_degeneracy_weight: float = DEFAULT_GA_STRICT_DEGENERACY_WEIGHT,
) -> float:
    """
    Score repeat burden for GA fitness.

    `legacy` preserves the existing objective. `blend` adds strict
    degeneracy pressure to that score. `aligned` uses the full shared
    degeneracy score used by the hybrid selector.
    """

    normalized_mode = normalize_ga_repeat_penalty_mode(mode)
    if strict_degeneracy_weight < 0:
        raise ValueError('strict_degeneracy_weight must be non-negative')

    legacy_score = calculate_legacy_repeat_score(metrics)
    if normalized_mode == GA_REPEAT_PENALTY_MODE_LEGACY:
        return legacy_score
    if normalized_mode == GA_REPEAT_PENALTY_MODE_BLEND:
        strict_score = calculate_strict_degeneracy_component(metrics)
        return legacy_score + strict_degeneracy_weight * strict_score
    return calculate_degeneracy_score(metrics)


def normalize_ga_repeat_penalty(
    repeat_score: float,
    *,
    repeat_penalty_scale: float = DEFAULT_GA_REPEAT_PENALTY_SCALE,
) -> float:
    if repeat_penalty_scale <= 0:
        raise ValueError('repeat_penalty_scale must be positive')
    return max(0.0, min(float(repeat_score) / repeat_penalty_scale, 1.0))


def compute_degeneracy_metrics(
    sequence: str,
    *,
    complexity_window: int = 150,
    complexity_step: int = 30,
    complexity_k: int = 3,
    complexity_alpha: float = 1.0,
) -> Dict[str, Any]:
    """Compute shared repeat, run, tandem-repeat, and complexity metrics."""

    complexity = compute_complexity_track(
        sequence,
        window_size=complexity_window,
        step=complexity_step,
        k=complexity_k,
        gcbal_alpha=complexity_alpha,
        smooth=None,
    )
    complexity_stats = nan_stats(complexity['score'])
    low_complexity_frac = fraction_below(complexity['score'], LOW_COMPLEXITY_CUTOFF)
    repeat_mean, repeat_max, repeat_frac_bad = repeat_penalty_windowed(sequence)
    repeat_window, repeat_step = repeat_penalty_window_params(len(sequence))
    runs = homopolymer_runs(sequence)
    gc_runs = binary_runs(sequence, {'G', 'C'})
    dinuc_runs = tandem_repeat_runs(sequence, 2)
    trinuc_runs = tandem_repeat_runs(sequence, 3)
    strict_examples = strict_degenerate_examples(sequence)

    max_by_base = {
        base: max((length for run_base, _, length in runs if run_base == base), default=0)
        for base in 'ACGT'
    }

    metrics: Dict[str, Any] = {
        'length_bp': len(sequence),
        'complexity_mean': complexity_stats['mean'],
        'complexity_min': complexity_stats['min'],
        'complexity_max': complexity_stats['max'],
        'complexity_std': complexity_stats['std'],
        'low_complexity_frac': low_complexity_frac,
        'repeat_mean': repeat_mean,
        'repeat_max': repeat_max,
        'repeat_frac_bad': repeat_frac_bad,
        'repeat_window_nt': repeat_window,
        'repeat_step_nt': repeat_step,
        'longest_homopolymer': max((length for _, _, length in runs), default=0),
        'homopolymer_runs_ge3': sum(1 for _, _, length in runs if length >= 3),
        'homopolymer_runs_ge4': sum(1 for _, _, length in runs if length >= 4),
        'homopolymer_runs_ge5': sum(1 for _, _, length in runs if length >= 5),
        'max_A_run': max_by_base['A'],
        'max_C_run': max_by_base['C'],
        'max_G_run': max_by_base['G'],
        'max_T_run': max_by_base['T'],
        'longest_gc_only_run': max((length for _, length in gc_runs), default=0),
        'gc_runs_ge_threshold': sum(
            1 for _, length in gc_runs if length >= STRICT_GC_RUN_THRESHOLD
        ),
        'gc_run_excess': sum(
            max(0, length - STRICT_GC_RUN_THRESHOLD + 1)
            for _, length in gc_runs
        ),
        'longest_at_only_run': longest_binary_run(sequence, {'A', 'T'}),
        'max_dinuc_tandem_bases': max(
            (length for _, length, _ in dinuc_runs),
            default=0,
        ),
        'dinuc_tandem_runs_ge_threshold': len(dinuc_runs),
        'max_trinuc_tandem_bases': max(
            (length for _, length, _ in trinuc_runs),
            default=0,
        ),
        'trinuc_tandem_runs_ge_threshold': len(trinuc_runs),
        'strict_degenerate_examples': strict_examples,
        'strict_degenerate_example_count': len(strict_examples),
    }
    metrics.update(compute_long_repeat_burden_metrics(
        sequence,
        current_repeat_frac_bad=repeat_frac_bad,
    ))
    metrics['degeneracy_score'] = calculate_degeneracy_score(metrics)
    return metrics


def is_degeneracy_clean_candidate(metrics: Dict[str, Any]) -> bool:
    """True when strict GC-only and tandem-repeat checks pass."""

    return (
        finite_or(metrics.get('longest_gc_only_run'), 0.0) < STRICT_GC_RUN_THRESHOLD
        and finite_or(metrics.get('max_dinuc_tandem_bases'), 0.0) < STRICT_TANDEM_REPEAT_NT
        and finite_or(metrics.get('max_trinuc_tandem_bases'), 0.0) < STRICT_TANDEM_REPEAT_NT
        and finite_or(metrics.get('strict_degenerate_example_count'), 0.0) == 0.0
    )


__all__ = [
    'STRICT_GC_RUN_THRESHOLD',
    'STRICT_TANDEM_REPEAT_NT',
    'LOW_COMPLEXITY_CUTOFF',
    'REPEAT_BAD_WINDOW_THRESHOLD',
    'LONG_REPEAT_MIN_SEQUENCE_NT',
    'LONG_REPEAT_CORE_NT',
    'LONG_REPEAT_LOCAL_WINDOW_NT',
    'LONG_REPEAT_LOCAL_STEP_NT',
    'BROAD_REPEAT_FRAC_BAD_THRESHOLD',
    'BROAD_REPEAT_SPAN_FRACTION_THRESHOLD',
    'GA_REPEAT_PENALTY_MODE_LEGACY',
    'GA_REPEAT_PENALTY_MODE_BLEND',
    'GA_REPEAT_PENALTY_MODE_ALIGNED',
    'GA_REPEAT_PENALTY_MODES',
    'DEFAULT_GA_REPEAT_PENALTY_MODE',
    'DEFAULT_GA_STRICT_DEGENERACY_WEIGHT',
    'DEFAULT_GA_REPEAT_PENALTY_SCALE',
    'homopolymer_runs',
    'RepeatBurdenWindow',
    'longest_binary_run',
    'binary_runs',
    'tandem_repeat_runs',
    'finite_or',
    'repeat_burden_windows',
    'summarize_repeat_burden_windows',
    'merge_bad_repeat_regions',
    'repeat_burden_junction_windows',
    'compute_long_repeat_burden_metrics',
    'normalize_ga_repeat_penalty_mode',
    'strict_degenerate_examples',
    'calculate_degeneracy_score',
    'calculate_legacy_repeat_score',
    'calculate_strict_degeneracy_component',
    'calculate_ga_repeat_score',
    'normalize_ga_repeat_penalty',
    'compute_degeneracy_metrics',
    'is_degeneracy_clean_candidate',
]
