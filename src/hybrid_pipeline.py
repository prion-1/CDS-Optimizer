"""
Hybrid pipeline defaults and optional multi-seed GA polish.

The defaults in this module reflect the PINK1 parameter sweeps in reports/
from 2026-04-19: local repair is the production default, and GA polish is
treated as an optional multi-seed search rather than a single stochastic run.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .complexity_analysis import compute_complexity_track
from .optimization import genetic_algorithm
from .utils import (
    calculate_cai,
    calculate_gc3_content,
    calculate_gc_content,
    calculate_tai,
    get_target_gc,
    load_codon_table,
    repeat_penalty_windowed,
    translate_dna_to_protein,
)


DEFAULT_HYBRID_REPAIR_WINDOW_NT = 36
DEFAULT_HYBRID_REPAIR_MAX_SUBS = 3
DEFAULT_HYBRID_REPAIR_GC_TOLERANCE = 5.0

DEFAULT_POLISH_POP_SIZE = 20
DEFAULT_POLISH_GENERATIONS = 8
DEFAULT_POLISH_MUTATION_RATE = 0.008
DEFAULT_POLISH_SEEDS = (1701, 2701, 3701, 4701, 5701)

STRICT_GC_RUN_THRESHOLD = 8
STRICT_TANDEM_REPEAT_NT = 8
LOW_COMPLEXITY_CUTOFF = 0.56

REPEAT_GUARD_POLISH_WEIGHTS: Dict[str, float] = {
    'cai': 0.30,
    'gc_deviation': 0.20,
    'folding_energy': 0.10,
    'unwanted_motifs': 0.10,
    'repeats': 0.18,
    'cryptic_splice': 0.07,
    'gc3_deviation': 0.0,
    'internal_atg': 0.0,
    'accessibility': 0.0,
    'tai': 0.05,
    'codon_pair_bias': 0.0,
}

DEGENERACY_GUARD_POLISH_WEIGHTS: Dict[str, float] = {
    'cai': 0.24,
    'gc_deviation': 0.16,
    'folding_energy': 0.08,
    'unwanted_motifs': 0.08,
    'repeats': 0.32,
    'cryptic_splice': 0.05,
    'gc3_deviation': 0.0,
    'internal_atg': 0.0,
    'accessibility': 0.0,
    'tai': 0.04,
    'codon_pair_bias': 0.03,
}

EXTREME_REPEAT_GUARD_POLISH_WEIGHTS: Dict[str, float] = {
    'cai': 0.18,
    'gc_deviation': 0.16,
    'folding_energy': 0.08,
    'unwanted_motifs': 0.08,
    'repeats': 0.42,
    'cryptic_splice': 0.04,
    'gc3_deviation': 0.0,
    'internal_atg': 0.0,
    'accessibility': 0.0,
    'tai': 0.02,
    'codon_pair_bias': 0.02,
}

DEFAULT_POLISH_WEIGHTS = EXTREME_REPEAT_GUARD_POLISH_WEIGHTS


@dataclass(frozen=True)
class PolishRunResult:
    seed: int
    sequence: str
    fitness: float
    fitness_metrics: Dict[str, float]
    quality_metrics: Dict[str, Any]


@dataclass(frozen=True)
class MultiSeedPolishResult:
    best_sequence: str
    best_seed: int
    best_fitness: float
    best_fitness_metrics: Dict[str, float]
    best_quality_metrics: Dict[str, Any]
    runs: Tuple[PolishRunResult, ...]


def _homopolymer_runs(sequence: str) -> List[Tuple[str, int, int]]:
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


def _longest_binary_run(sequence: str, allowed: set[str]) -> int:
    best = 0
    current = 0
    for base in sequence:
        if base in allowed:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _binary_runs(sequence: str, allowed: set[str]) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None

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


def _tandem_repeat_runs(
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


def _nan_stats(values: np.ndarray) -> Dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {'mean': math.nan, 'min': math.nan, 'max': math.nan, 'std': math.nan}

    return {
        'mean': float(np.mean(finite)),
        'min': float(np.min(finite)),
        'max': float(np.max(finite)),
        'std': float(np.std(finite)),
    }


def _fraction_below(values: np.ndarray, cutoff: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    return float(np.mean(finite < cutoff))


def _short_fragment(sequence: str, start: int, length: int, max_len: int = 32) -> str:
    fragment = sequence[start:start + length]
    if len(fragment) <= max_len:
        return fragment
    return fragment[:max_len] + '...'


def _strict_degenerate_examples(
    sequence: str,
    *,
    min_gc_run: int = STRICT_GC_RUN_THRESHOLD,
    min_tandem_bases: int = STRICT_TANDEM_REPEAT_NT,
    max_examples: int = 5,
) -> Tuple[str, ...]:
    examples: List[Tuple[int, int, str, str]] = []

    for start, length in _binary_runs(sequence, {'G', 'C'}):
        if length >= min_gc_run:
            examples.append((
                length,
                start,
                'GC-run',
                _short_fragment(sequence, start, length),
            ))

    for unit_size, label in ((2, 'dinuc'), (3, 'trinuc')):
        for start, length, unit in _tandem_repeat_runs(
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


def _finite_or(value: Any, fallback: float) -> float:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return fallback


def _degeneracy_score(metrics: Dict[str, Any]) -> float:
    complexity_min = _finite_or(metrics.get('complexity_min'), LOW_COMPLEXITY_CUTOFF)
    complexity_shortfall = max(0.0, LOW_COMPLEXITY_CUTOFF - complexity_min)

    return (
        1.20 * _finite_or(metrics.get('repeat_mean'), 0.0)
        + 1.80 * _finite_or(metrics.get('repeat_max'), 0.0)
        + 2.50 * _finite_or(metrics.get('repeat_frac_bad'), 0.0)
        + 0.45 * max(
            0.0,
            _finite_or(metrics.get('longest_gc_only_run'), 0.0)
            - STRICT_GC_RUN_THRESHOLD
            + 1.0,
        )
        + 0.30 * _finite_or(metrics.get('gc_run_excess'), 0.0)
        + 0.55 * max(0.0, _finite_or(metrics.get('longest_homopolymer'), 0.0) - 4.0)
        + 0.70 * _finite_or(metrics.get('homopolymer_runs_ge5'), 0.0)
        + 0.20 * max(
            0.0,
            _finite_or(metrics.get('max_dinuc_tandem_bases'), 0.0)
            - STRICT_TANDEM_REPEAT_NT
            + 1.0,
        )
        + 0.15 * max(
            0.0,
            _finite_or(metrics.get('max_trinuc_tandem_bases'), 0.0)
            - STRICT_TANDEM_REPEAT_NT
            + 1.0,
        )
        + 5.00 * complexity_shortfall
        + 1.50 * _finite_or(metrics.get('low_complexity_frac'), 0.0)
    )


def parse_seed_list(seed_text: str) -> Tuple[int, ...]:
    """Parse comma-separated integer seeds for notebook/user input."""

    seeds: List[int] = []
    for part in seed_text.split(','):
        part = part.strip()
        if not part:
            continue
        seeds.append(int(part))
    if not seeds:
        raise ValueError('At least one GA polish seed is required.')
    return tuple(seeds)


def sequence_quality_metrics(
    sequence: str,
    *,
    host: str = 'hsapiens',
    input_protein: Optional[str] = None,
    codon_table: Optional[Dict[str, float]] = None,
    complexity_window: int = 150,
    complexity_step: int = 30,
    complexity_k: int = 3,
    complexity_alpha: float = 1.0,
) -> Dict[str, Any]:
    """Compute the lightweight quality metrics used for hybrid-polish ranking."""

    if codon_table is None:
        codon_table = load_codon_table(host)
    if input_protein is None:
        input_protein = translate_dna_to_protein(sequence)

    complexity = compute_complexity_track(
        sequence,
        window_size=complexity_window,
        step=complexity_step,
        k=complexity_k,
        gcbal_alpha=complexity_alpha,
        smooth=None,
    )
    complexity_stats = _nan_stats(complexity['score'])
    low_complexity_frac = _fraction_below(complexity['score'], LOW_COMPLEXITY_CUTOFF)
    repeat_mean, repeat_max, repeat_frac_bad = repeat_penalty_windowed(sequence)
    runs = _homopolymer_runs(sequence)
    gc_runs = _binary_runs(sequence, {'G', 'C'})
    dinuc_runs = _tandem_repeat_runs(sequence, 2)
    trinuc_runs = _tandem_repeat_runs(sequence, 3)
    strict_examples = _strict_degenerate_examples(sequence)
    try:
        tai = calculate_tai(sequence, host)
    except FileNotFoundError:
        tai = 0.0

    max_by_base = {
        base: max((length for run_base, _, length in runs if run_base == base), default=0)
        for base in 'ACGT'
    }

    metrics: Dict[str, Any] = {
        'length_bp': len(sequence),
        'gc': calculate_gc_content(sequence),
        'gc3': calculate_gc3_content(sequence),
        'cai': calculate_cai(sequence, codon_table),
        'tai': tai,
        'complexity_mean': complexity_stats['mean'],
        'complexity_min': complexity_stats['min'],
        'complexity_max': complexity_stats['max'],
        'complexity_std': complexity_stats['std'],
        'low_complexity_frac': low_complexity_frac,
        'repeat_mean': repeat_mean,
        'repeat_max': repeat_max,
        'repeat_frac_bad': repeat_frac_bad,
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
        'longest_at_only_run': _longest_binary_run(sequence, {'A', 'T'}),
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
        'protein_identity_ok': translate_dna_to_protein(sequence) == input_protein,
        'sequence_sha256': hashlib.sha256(sequence.encode('ascii')).hexdigest()[:16],
    }
    metrics['degeneracy_score'] = _degeneracy_score(metrics)
    return metrics


def is_degeneracy_clean_candidate(metrics: Dict[str, Any]) -> bool:
    """True when strict GC-only and tandem-repeat checks pass."""

    return (
        _finite_or(metrics.get('longest_gc_only_run'), 0.0) < STRICT_GC_RUN_THRESHOLD
        and _finite_or(metrics.get('max_dinuc_tandem_bases'), 0.0) < STRICT_TANDEM_REPEAT_NT
        and _finite_or(metrics.get('max_trinuc_tandem_bases'), 0.0) < STRICT_TANDEM_REPEAT_NT
        and _finite_or(metrics.get('strict_degenerate_example_count'), 0.0) == 0.0
    )


def is_clean_hybrid_candidate(metrics: Dict[str, Any]) -> bool:
    """Clean-pass criterion from the multi-seed PINK1 sweep."""

    return (
        is_degeneracy_clean_candidate(metrics)
        and _finite_or(metrics.get('repeat_max'), math.inf) <= 1.0
        and _finite_or(metrics.get('repeat_frac_bad'), math.inf) <= 0.20
        and _finite_or(metrics.get('longest_homopolymer'), math.inf) <= 4
        and _finite_or(metrics.get('homopolymer_runs_ge5'), math.inf) == 0
        and _finite_or(metrics.get('complexity_mean'), -math.inf) >= LOW_COMPLEXITY_CUTOFF
    )


def is_strict_hybrid_candidate(metrics: Dict[str, Any]) -> bool:
    """Stricter clean-pass criterion used to break ties."""

    return (
        is_clean_hybrid_candidate(metrics)
        and _finite_or(metrics.get('repeat_mean'), math.inf) <= 0.30
        and _finite_or(metrics.get('cai'), -math.inf) >= 0.70
    )


def hybrid_candidate_rank(metrics: Dict[str, Any], fitness: float = 0.0) -> Tuple[float, ...]:
    """
    Rank candidates by explicit sequence-quality criteria.

    The tuple is suitable for max(...). It prefers clean and strict candidates,
    then lower repeat burden, then higher complexity, CAI/tAI, and finally GA
    fitness as a last tie-breaker.
    """

    return (
        1.0 if metrics.get('protein_identity_ok') else 0.0,
        1.0 if is_degeneracy_clean_candidate(metrics) else 0.0,
        1.0 if is_clean_hybrid_candidate(metrics) else 0.0,
        1.0 if is_strict_hybrid_candidate(metrics) else 0.0,
        -_finite_or(metrics.get('degeneracy_score'), math.inf),
        -_finite_or(metrics.get('strict_degenerate_example_count'), math.inf),
        -_finite_or(metrics.get('longest_gc_only_run'), math.inf),
        -_finite_or(metrics.get('max_dinuc_tandem_bases'), math.inf),
        -_finite_or(metrics.get('max_trinuc_tandem_bases'), math.inf),
        -_finite_or(metrics.get('repeat_mean'), math.inf),
        -_finite_or(metrics.get('repeat_max'), math.inf),
        -_finite_or(metrics.get('repeat_frac_bad'), math.inf),
        -_finite_or(metrics.get('homopolymer_runs_ge5'), math.inf),
        -_finite_or(metrics.get('longest_homopolymer'), math.inf),
        _finite_or(metrics.get('complexity_mean'), -math.inf),
        _finite_or(metrics.get('complexity_min'), -math.inf),
        _finite_or(metrics.get('cai'), -math.inf),
        _finite_or(metrics.get('tai'), -math.inf),
        _finite_or(fitness, -math.inf),
    )


def multi_seed_ga_polish(
    initial_cds: str,
    *,
    host: str = 'hsapiens',
    is_eukaryote: bool = True,
    target_gc: Optional[float] = None,
    pop_size: int = DEFAULT_POLISH_POP_SIZE,
    generations: int = DEFAULT_POLISH_GENERATIONS,
    mutation_rate: float = DEFAULT_POLISH_MUTATION_RATE,
    weights: Optional[Dict[str, float]] = None,
    avoid_sequences: str = '',
    seeds: Sequence[int] = DEFAULT_POLISH_SEEDS,
    input_protein: Optional[str] = None,
    verbose: bool = False,
    progress_callback: Optional[Callable[[int, int, int, int, int], None]] = None,
    complexity_window: int = 150,
    complexity_step: int = 30,
    complexity_k: int = 3,
    complexity_alpha: float = 1.0,
) -> MultiSeedPolishResult:
    """
    Run GA polish across seeds and select by hybrid sequence-quality metrics.

    progress_callback, when provided, is called as:

        callback(seed, seed_index, total_seeds, generation, generations)
    """

    if not seeds:
        raise ValueError('At least one GA polish seed is required.')
    if target_gc is None:
        target_gc = get_target_gc(host)
    if weights is None:
        weights = DEFAULT_POLISH_WEIGHTS
    if input_protein is None:
        input_protein = translate_dna_to_protein(initial_cds)

    codon_table = load_codon_table(host)
    runs: List[PolishRunResult] = []
    total = len(seeds)

    for seed_index, seed in enumerate(seeds, start=1):
        def seed_progress(generation: int, total_generations: int) -> None:
            if progress_callback is not None:
                progress_callback(seed, seed_index, total, generation, total_generations)

        sequence, fitness, fitness_metrics = genetic_algorithm(
            initial_cds=initial_cds,
            host=host,
            is_eukaryote=is_eukaryote,
            target_gc=target_gc,
            pop_size=pop_size,
            generations=generations,
            mutation_rate=mutation_rate,
            weights=weights,
            avoid_sequences=avoid_sequences,
            random_seed=int(seed),
            verbose=verbose,
            progress_callback=seed_progress,
        )

        quality_metrics = sequence_quality_metrics(
            sequence,
            host=host,
            input_protein=input_protein,
            codon_table=codon_table,
            complexity_window=complexity_window,
            complexity_step=complexity_step,
            complexity_k=complexity_k,
            complexity_alpha=complexity_alpha,
        )
        if not quality_metrics['protein_identity_ok']:
            raise AssertionError(f'GA polish changed protein identity for seed {seed}.')

        runs.append(PolishRunResult(
            seed=int(seed),
            sequence=sequence,
            fitness=fitness,
            fitness_metrics=fitness_metrics,
            quality_metrics=quality_metrics,
        ))

    best = max(runs, key=lambda run: hybrid_candidate_rank(run.quality_metrics, run.fitness))
    return MultiSeedPolishResult(
        best_sequence=best.sequence,
        best_seed=best.seed,
        best_fitness=best.fitness,
        best_fitness_metrics=best.fitness_metrics,
        best_quality_metrics=best.quality_metrics,
        runs=tuple(runs),
    )
