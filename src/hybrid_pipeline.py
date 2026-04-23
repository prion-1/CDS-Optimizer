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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .degeneracy import (
    BROAD_REPEAT_FRAC_BAD_THRESHOLD,
    BROAD_REPEAT_SPAN_FRACTION_THRESHOLD,
    DEFAULT_GA_REPEAT_PENALTY_MODE,
    DEFAULT_GA_REPEAT_PENALTY_SCALE,
    DEFAULT_GA_STRICT_DEGENERACY_WEIGHT,
    LONG_REPEAT_MIN_SEQUENCE_NT,
    LOW_COMPLEXITY_CUTOFF,
    compute_degeneracy_metrics,
    finite_or as _finite_or,
    is_degeneracy_clean_candidate,
)
from .optimization import genetic_algorithm
from .utils import (
    calculate_cai,
    calculate_gc3_content,
    calculate_gc_content,
    calculate_tai,
    check_internal_start_codons,
    count_cryptic_splice_sites,
    get_target_gc,
    get_target_gc3,
    load_codon_table,
    translate_dna_to_protein,
)


DEFAULT_HYBRID_REPAIR_WINDOW_NT = 36
DEFAULT_HYBRID_REPAIR_MAX_SUBS = 3
DEFAULT_HYBRID_REPAIR_GC_TOLERANCE = 5.0

DEFAULT_POLISH_POP_SIZE = 20
DEFAULT_POLISH_GENERATIONS = 8
DEFAULT_POLISH_MUTATION_RATE = 0.008
DEFAULT_POLISH_SEEDS = (1701, 2701, 3701, 4701, 5701)

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

REGULATORY_MOTIF_HIGH_POLISH_WEIGHTS: Dict[str, float] = {
    'cai': 0.14,
    'gc_deviation': 0.08,
    'folding_energy': 0.04,
    'unwanted_motifs': 0.35,
    'repeats': 0.18,
    'cryptic_splice': 0.15,
    'gc3_deviation': 0.0,
    'internal_atg': 0.04,
    'accessibility': 0.0,
    'tai': 0.01,
    'codon_pair_bias': 0.01,
}

DEFAULT_REGULATORY_POLISH_WEIGHTS = REGULATORY_MOTIF_HIGH_POLISH_WEIGHTS

DEFAULT_POLYA_MOTIFS: Tuple[str, ...] = ('AATAAA', 'ATTAAA', 'AAAAAA')


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


@dataclass(frozen=True)
class RegulatorySelectionPolicy:
    """Relaxed constructability and report-score settings for regulatory ranking."""

    min_cai: float = 0.55
    max_gc_deviation_pct: float = 12.0
    max_gc3_deviation_pct: float = 30.0
    max_longest_gc_run: int = 14
    max_longest_homopolymer: int = 8
    max_tandem_repeat_bases: int = 18
    max_repeat_frac_bad: float = 0.45
    long_repeat_min_length_nt: int = LONG_REPEAT_MIN_SEQUENCE_NT
    max_broad_repeat_frac_bad: float = BROAD_REPEAT_FRAC_BAD_THRESHOLD
    max_broad_repeat_span_fraction: float = BROAD_REPEAT_SPAN_FRACTION_THRESHOLD
    degeneracy_score_cap: float = 10.0
    default_motif_weight: float = 0.22
    custom_motif_weight: float = 0.22
    splice_weight: float = 0.26
    internal_atg_weight: float = 0.08
    cai_weight: float = 0.12
    tai_weight: float = 0.04
    residual_default_motif_weight: float = 0.25
    residual_custom_motif_weight: float = 0.25
    residual_splice_weight: float = 0.10
    residual_internal_atg_weight: float = 0.20
    degeneracy_penalty_weight: float = 0.04
    repeat_frac_bad_weight: float = 0.25
    constructability_failure_weight: float = 0.25


DEFAULT_REGULATORY_SELECTION_POLICY = RegulatorySelectionPolicy()


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


def parse_avoid_sequence_list(avoid_sequences: Any) -> Tuple[str, ...]:
    """Normalize comma-separated or iterable avoid motifs to uppercase DNA strings."""

    if avoid_sequences is None:
        return ()
    if isinstance(avoid_sequences, str):
        raw_parts = avoid_sequences.split(',')
    else:
        raw_parts = list(avoid_sequences)

    motifs: List[str] = []
    for part in raw_parts:
        motif = str(part).strip().upper().replace('U', 'T')
        if motif:
            motifs.append(motif)
    return tuple(motifs)


def _overlap_motif_count(sequence: str, motifs: Sequence[str]) -> int:
    """Count motif hits with overlap, matching the GA motif-count semantics."""

    seq = sequence.upper().replace('U', 'T')
    total = 0
    for motif in motifs:
        normalized = motif.upper().replace('U', 'T')
        if not normalized:
            continue
        start = 0
        while True:
            index = seq.find(normalized, start)
            if index == -1:
                break
            total += 1
            start = index + 1
    return total


def sequence_regulatory_metrics(
    sequence: str,
    *,
    is_eukaryote: bool = True,
    avoid_sequences: Any = '',
) -> Dict[str, Any]:
    """Compute raw, non-saturating regulatory counts for post-selection."""

    avoid_motifs = parse_avoid_sequence_list(avoid_sequences)
    default_motifs = _overlap_motif_count(sequence, DEFAULT_POLYA_MOTIFS)
    custom_motifs = _overlap_motif_count(sequence, avoid_motifs)
    splice_sites = count_cryptic_splice_sites(sequence, is_eukaryote)
    internal_atg = check_internal_start_codons(sequence)
    length_kb = max(1.0, len(sequence) / 1000.0)

    return {
        'regulatory_default_motifs': default_motifs,
        'regulatory_custom_motifs': custom_motifs,
        'regulatory_total_motifs': default_motifs + custom_motifs,
        'regulatory_splice_sites': splice_sites,
        'regulatory_internal_atg': internal_atg,
        'regulatory_default_motifs_per_kb': default_motifs / length_kb,
        'regulatory_custom_motifs_per_kb': custom_motifs / length_kb,
        'regulatory_splice_sites_per_kb': splice_sites / length_kb,
    }


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

    degeneracy_metrics = compute_degeneracy_metrics(
        sequence,
        complexity_window=complexity_window,
        complexity_step=complexity_step,
        complexity_k=complexity_k,
        complexity_alpha=complexity_alpha,
    )
    try:
        tai = calculate_tai(sequence, host)
    except FileNotFoundError:
        tai = 0.0

    metrics: Dict[str, Any] = {
        'length_bp': len(sequence),
        'gc': calculate_gc_content(sequence),
        'gc3': calculate_gc3_content(sequence),
        'cai': calculate_cai(sequence, codon_table),
        'tai': tai,
        'protein_identity_ok': translate_dna_to_protein(sequence) == input_protein,
        'sequence_sha256': hashlib.sha256(sequence.encode('ascii')).hexdigest()[:16],
    }
    metrics.update(degeneracy_metrics)
    return metrics


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


def regulatory_constructability_failures(
    metrics: Dict[str, Any],
    *,
    target_gc: Optional[float] = None,
    target_gc3: Optional[float] = None,
    policy: RegulatorySelectionPolicy = DEFAULT_REGULATORY_SELECTION_POLICY,
) -> Tuple[str, ...]:
    """
    Return relaxed constructability failures for regulatory post-selection.

    These thresholds intentionally differ from repeat-guard selection. They
    reject severe constructability problems without making degeneracy the
    dominant objective for regulatory presets.
    """

    failures: List[str] = []
    if not metrics.get('protein_identity_ok'):
        failures.append('protein_identity')
    if _finite_or(metrics.get('cai'), 0.0) < policy.min_cai:
        failures.append('min_cai')
    if (
        target_gc is not None
        and abs(_finite_or(metrics.get('gc'), 0.0) - target_gc)
        > policy.max_gc_deviation_pct
    ):
        failures.append('gc_deviation')
    if (
        target_gc3 is not None
        and abs(_finite_or(metrics.get('gc3'), 0.0) - target_gc3)
        > policy.max_gc3_deviation_pct
    ):
        failures.append('gc3_deviation')
    if _finite_or(metrics.get('longest_gc_only_run'), math.inf) > policy.max_longest_gc_run:
        failures.append('severe_gc_run')
    if (
        _finite_or(metrics.get('longest_homopolymer'), math.inf)
        > policy.max_longest_homopolymer
    ):
        failures.append('severe_homopolymer')
    if (
        _finite_or(metrics.get('max_dinuc_tandem_bases'), math.inf)
        > policy.max_tandem_repeat_bases
    ):
        failures.append('severe_dinuc_tandem')
    if (
        _finite_or(metrics.get('max_trinuc_tandem_bases'), math.inf)
        > policy.max_tandem_repeat_bases
    ):
        failures.append('severe_trinuc_tandem')
    repeat_frac_bad = _finite_or(metrics.get('repeat_frac_bad'), math.inf)
    length_bp = _finite_or(metrics.get('length_bp'), 0.0)
    long_repeat_mode = bool(metrics.get('long_repeat_mode')) or (
        length_bp > policy.long_repeat_min_length_nt
    )
    span_present = metrics.get('long_repeat_bad_region_span_fraction') is not None
    if long_repeat_mode and span_present:
        bad_region_span_fraction = _finite_or(
            metrics.get('long_repeat_bad_region_span_fraction'),
            0.0,
        )
        if (
            repeat_frac_bad > policy.max_broad_repeat_frac_bad
            and bad_region_span_fraction > policy.max_broad_repeat_span_fraction
        ):
            failures.append('broad_repeat_burden')
    elif repeat_frac_bad > policy.max_repeat_frac_bad:
        failures.append('repeat_frac_bad')
    return tuple(failures)


def _proportional_reduction(before: float, after: float) -> float:
    if before <= 0:
        return 0.0 if after <= 0 else -after
    return (before - after) / before


def regulatory_post_selection_score(
    metrics: Dict[str, Any],
    *,
    baseline_metrics: Optional[Dict[str, Any]] = None,
    constructability_failures_count: int = 0,
    policy: RegulatorySelectionPolicy = DEFAULT_REGULATORY_SELECTION_POLICY,
) -> float:
    """Score regulatory candidates with raw non-saturating report metrics."""

    baseline = baseline_metrics or {}
    default_motifs = _finite_or(metrics.get('regulatory_default_motifs'), 0.0)
    custom_motifs = _finite_or(metrics.get('regulatory_custom_motifs'), 0.0)
    splice_sites = _finite_or(metrics.get('regulatory_splice_sites'), 0.0)
    internal_atg = _finite_or(metrics.get('regulatory_internal_atg'), 0.0)

    baseline_default = _finite_or(
        baseline.get('regulatory_default_motifs'),
        _finite_or(metrics.get('baseline_regulatory_default_motifs'), 0.0),
    )
    baseline_custom = _finite_or(
        baseline.get('regulatory_custom_motifs'),
        _finite_or(metrics.get('baseline_regulatory_custom_motifs'), 0.0),
    )
    baseline_splice = _finite_or(
        baseline.get('regulatory_splice_sites'),
        _finite_or(metrics.get('baseline_regulatory_splice_sites'), 0.0),
    )
    baseline_atg = _finite_or(
        baseline.get('regulatory_internal_atg'),
        _finite_or(metrics.get('baseline_regulatory_internal_atg'), 0.0),
    )

    length_kb = max(1.0, _finite_or(metrics.get('length_bp'), 0.0) / 1000.0)
    residual_regulatory_load = (
        policy.residual_default_motif_weight * (default_motifs / length_kb)
        + policy.residual_custom_motif_weight * (custom_motifs / length_kb)
        + policy.residual_splice_weight * (splice_sites / length_kb)
        + policy.residual_internal_atg_weight * internal_atg
    )
    constructability_penalty = (
        policy.degeneracy_penalty_weight
        * min(_finite_or(metrics.get('degeneracy_score'), 0.0), policy.degeneracy_score_cap)
        + policy.repeat_frac_bad_weight * _finite_or(metrics.get('repeat_frac_bad'), 0.0)
        + policy.constructability_failure_weight * constructability_failures_count
    )

    return (
        policy.default_motif_weight
        * _proportional_reduction(baseline_default, default_motifs)
        + policy.custom_motif_weight
        * _proportional_reduction(baseline_custom, custom_motifs)
        + policy.splice_weight
        * _proportional_reduction(baseline_splice, splice_sites)
        + policy.internal_atg_weight
        * _proportional_reduction(baseline_atg, internal_atg)
        + policy.cai_weight * _finite_or(metrics.get('cai'), 0.0)
        + policy.tai_weight * _finite_or(metrics.get('tai'), 0.0)
        - 0.08 * residual_regulatory_load
        - constructability_penalty
    )


def annotate_regulatory_post_selection_metrics(
    metrics: Dict[str, Any],
    sequence: str,
    *,
    is_eukaryote: bool = True,
    avoid_sequences: Any = '',
    baseline_metrics: Optional[Dict[str, Any]] = None,
    target_gc: Optional[float] = None,
    target_gc3: Optional[float] = None,
    policy: RegulatorySelectionPolicy = DEFAULT_REGULATORY_SELECTION_POLICY,
) -> Dict[str, Any]:
    """Return quality metrics augmented with regulatory selector fields."""

    annotated = dict(metrics)
    annotated.update(
        sequence_regulatory_metrics(
            sequence,
            is_eukaryote=is_eukaryote,
            avoid_sequences=avoid_sequences,
        )
    )

    baseline = baseline_metrics or {}
    for key in (
        'regulatory_default_motifs',
        'regulatory_custom_motifs',
        'regulatory_total_motifs',
        'regulatory_splice_sites',
        'regulatory_internal_atg',
    ):
        baseline_value = _finite_or(baseline.get(key), 0.0)
        current_value = _finite_or(annotated.get(key), 0.0)
        annotated[f'baseline_{key}'] = baseline_value
        annotated[f'{key}_delta'] = current_value - baseline_value

    failures = regulatory_constructability_failures(
        annotated,
        target_gc=target_gc,
        target_gc3=target_gc3,
        policy=policy,
    )
    annotated['regulatory_constructability_failures'] = failures
    annotated['regulatory_constructability_failure_count'] = len(failures)
    annotated['regulatory_constructability_pass'] = not failures
    annotated['regulatory_post_selection_score'] = regulatory_post_selection_score(
        annotated,
        baseline_metrics=baseline,
        constructability_failures_count=len(failures),
        policy=policy,
    )
    return annotated


def regulatory_candidate_rank(
    metrics: Dict[str, Any],
    fitness: float = 0.0,
    *,
    target_gc: Optional[float] = None,
    target_gc3: Optional[float] = None,
    baseline_metrics: Optional[Dict[str, Any]] = None,
    policy: RegulatorySelectionPolicy = DEFAULT_REGULATORY_SELECTION_POLICY,
) -> Tuple[float, ...]:
    """
    Rank regulatory candidates by raw motif/splice cleanup plus relaxed gates.

    The tuple is suitable for max(...). It first preserves protein identity,
    then prefers candidates passing the relaxed constructability policy, then
    ranks by the non-saturating regulatory score and residual regulatory load.
    """

    failures = metrics.get('regulatory_constructability_failures')
    if failures is None:
        failures = regulatory_constructability_failures(
            metrics,
            target_gc=target_gc,
            target_gc3=target_gc3,
            policy=policy,
        )
    failure_count = len(tuple(failures))
    score = _finite_or(
        metrics.get('regulatory_post_selection_score'),
        regulatory_post_selection_score(
            metrics,
            baseline_metrics=baseline_metrics,
            constructability_failures_count=failure_count,
            policy=policy,
        ),
    )

    return (
        1.0 if metrics.get('protein_identity_ok') else 0.0,
        1.0 if failure_count == 0 else 0.0,
        -float(failure_count),
        score,
        -_finite_or(metrics.get('regulatory_total_motifs'), math.inf),
        -_finite_or(metrics.get('regulatory_default_motifs'), math.inf),
        -_finite_or(metrics.get('regulatory_custom_motifs'), math.inf),
        -_finite_or(metrics.get('regulatory_splice_sites'), math.inf),
        -_finite_or(metrics.get('regulatory_internal_atg'), math.inf),
        -_finite_or(metrics.get('repeat_frac_bad'), math.inf),
        -_finite_or(metrics.get('long_repeat_bad_region_span_fraction'), 0.0),
        -_finite_or(metrics.get('long_repeat_segment_frac_bad_max'), 0.0),
        -_finite_or(metrics.get('long_repeat_junction_repeat_max'), 0.0),
        -_finite_or(metrics.get('degeneracy_score'), math.inf),
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
    repeat_penalty_mode: str = DEFAULT_GA_REPEAT_PENALTY_MODE,
    strict_degeneracy_weight: float = DEFAULT_GA_STRICT_DEGENERACY_WEIGHT,
    repeat_penalty_scale: float = DEFAULT_GA_REPEAT_PENALTY_SCALE,
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
            repeat_penalty_mode=repeat_penalty_mode,
            strict_degeneracy_weight=strict_degeneracy_weight,
            repeat_penalty_scale=repeat_penalty_scale,
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


def multi_seed_regulatory_ga_polish(
    initial_cds: str,
    *,
    host: str = 'hsapiens',
    is_eukaryote: bool = True,
    target_gc: Optional[float] = None,
    target_gc3: Optional[float] = None,
    pop_size: int = DEFAULT_POLISH_POP_SIZE,
    generations: int = DEFAULT_POLISH_GENERATIONS,
    mutation_rate: float = DEFAULT_POLISH_MUTATION_RATE,
    weights: Optional[Dict[str, float]] = None,
    avoid_sequences: str = '',
    seeds: Sequence[int] = DEFAULT_POLISH_SEEDS,
    input_protein: Optional[str] = None,
    baseline_cds: Optional[str] = None,
    policy: RegulatorySelectionPolicy = DEFAULT_REGULATORY_SELECTION_POLICY,
    verbose: bool = False,
    progress_callback: Optional[Callable[[int, int, int, int, int], None]] = None,
    complexity_window: int = 150,
    complexity_step: int = 30,
    complexity_k: int = 3,
    complexity_alpha: float = 1.0,
    repeat_penalty_mode: str = DEFAULT_GA_REPEAT_PENALTY_MODE,
    strict_degeneracy_weight: float = DEFAULT_GA_STRICT_DEGENERACY_WEIGHT,
    repeat_penalty_scale: float = DEFAULT_GA_REPEAT_PENALTY_SCALE,
) -> MultiSeedPolishResult:
    """
    Run GA polish across seeds and select by regulatory post-selection metrics.

    This helper is intended for regulatory presets. It keeps GA execution the
    same, then selects the seed using raw non-saturating motif/splice/internal
    ATG counts plus relaxed constructability gates. The global repeat-guard
    defaults are not changed by this helper.
    """

    if not seeds:
        raise ValueError('At least one GA polish seed is required.')
    if target_gc is None:
        target_gc = get_target_gc(host)
    if target_gc3 is None:
        target_gc3 = get_target_gc3(host)
    if weights is None:
        weights = DEFAULT_REGULATORY_POLISH_WEIGHTS
    if input_protein is None:
        input_protein = translate_dna_to_protein(initial_cds)
    if baseline_cds is None:
        baseline_cds = initial_cds

    baseline_regulatory_metrics = sequence_regulatory_metrics(
        baseline_cds,
        is_eukaryote=is_eukaryote,
        avoid_sequences=avoid_sequences,
    )
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
            repeat_penalty_mode=repeat_penalty_mode,
            strict_degeneracy_weight=strict_degeneracy_weight,
            repeat_penalty_scale=repeat_penalty_scale,
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

        quality_metrics = annotate_regulatory_post_selection_metrics(
            quality_metrics,
            sequence,
            is_eukaryote=is_eukaryote,
            avoid_sequences=avoid_sequences,
            baseline_metrics=baseline_regulatory_metrics,
            target_gc=target_gc,
            target_gc3=target_gc3,
            policy=policy,
        )

        runs.append(PolishRunResult(
            seed=int(seed),
            sequence=sequence,
            fitness=fitness,
            fitness_metrics=fitness_metrics,
            quality_metrics=quality_metrics,
        ))

    best = max(
        runs,
        key=lambda run: regulatory_candidate_rank(
            run.quality_metrics,
            run.fitness,
            target_gc=target_gc,
            target_gc3=target_gc3,
            baseline_metrics=baseline_regulatory_metrics,
            policy=policy,
        ),
    )
    return MultiSeedPolishResult(
        best_sequence=best.sequence,
        best_seed=best.seed,
        best_fitness=best.fitness,
        best_fitness_metrics=best.fitness_metrics,
        best_quality_metrics=best.quality_metrics,
        runs=tuple(runs),
    )
