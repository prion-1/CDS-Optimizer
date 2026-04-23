"""
Multi-objective fitness function and synonymous-codon genetic algorithm.

Performance notes
-----------------
This module is the hot path for any non-trivial optimization. Two changes
make it materially faster than the v1 implementation:

  1. The host codon table is loaded *once* when the GA is constructed and
     cached on the instance. fitness_function() takes the cached table as
     an argument so individual evaluations don't re-parse the CSV from disk.
  2. CAI is scored against a precomputed 64-element score vector
     (utils.get_codon_score_vector(host)) so the inner loop only does an
     array lookup per codon instead of dictionary work.

Both caches are LRU at the utils level — multiple GA runs with the same host
share state. Local repair benefits from the same caches without extra wiring.
"""

import random
from dataclasses import dataclass
from typing import Any, List, Tuple, Dict, Optional, Callable
import numpy as np

from .degeneracy import (
    DEFAULT_GA_REPEAT_PENALTY_MODE,
    DEFAULT_GA_REPEAT_PENALTY_SCALE,
    DEFAULT_GA_STRICT_DEGENERACY_WEIGHT,
    GA_REPEAT_PENALTY_MODE_LEGACY,
    calculate_ga_repeat_score,
    calculate_strict_degeneracy_component,
    compute_degeneracy_metrics,
    normalize_ga_repeat_penalty,
    normalize_ga_repeat_penalty_mode,
)
from .utils import (
    load_codon_table,
    calculate_cai,
    calculate_cai_fast,
    calculate_gc_content,
    calculate_gc3_content,
    count_unwanted_motifs,
    calculate_folding_energy_windowed,
    calculate_accessibility_score,
    repeat_penalty_windowed,
    repeat_penalty_window_params,
    count_cryptic_splice_sites,
    check_internal_start_codons,
    translate_dna_to_protein,
    get_synonymous_codons,
    get_codon_score_vector,
    get_target_gc,
    get_target_gc3,
    get_cps_score_matrix,
    calculate_codon_pair_score,
    get_tai_score_vector,
    tai_available_for,
    _seq_to_codon_indices,
    SENSE_CODON_INDEX_MASK,
    HOST_TARGET_GC,
    GENETIC_CODE,
)


# ---------------------------------------------------------------------------
# Fitness configuration
# ---------------------------------------------------------------------------

@dataclass
class FitnessWeights:
    """
    Container for fitness function weights.

    All terms are exposed so users can compose the trade-off explicitly.
    The defaults sum to 1.0 across the historically active terms; the new
    optional axes (tAI, codon-pair bias) default to zero so existing
    pipelines keep producing identical scores until users opt in.
    """
    cai: float = 0.40
    gc_deviation: float = 0.20
    folding_energy: float = 0.15
    unwanted_motifs: float = 0.10
    repeats: float = 0.07
    cryptic_splice: float = 0.08
    gc3_deviation: float = 0.0
    internal_atg: float = 0.0
    accessibility: float = 0.0
    tai: float = 0.0
    codon_pair_bias: float = 0.0

    def __post_init__(self) -> None:
        total = sum([
            self.cai, self.gc_deviation, self.folding_energy,
            self.unwanted_motifs, self.repeats, self.cryptic_splice,
            self.gc3_deviation, self.internal_atg, self.accessibility,
            self.tai, self.codon_pair_bias,
        ])
        if abs(total - 1.0) > 0.01:
            print(f"Warning: Weights sum to {total:.2f}, not 1.0")


# ---------------------------------------------------------------------------
# Fitness function
# ---------------------------------------------------------------------------

def fitness_function(
    sequence: str,
    host: str,
    is_eukaryote: bool,
    target_gc: Optional[float] = None,
    weights: Optional[FitnessWeights] = None,
    avoid_motifs: Optional[List[str]] = None,
    *,
    codon_table: Optional[Dict[str, float]] = None,
    cai_score_vector: Optional[np.ndarray] = None,
    cps_matrix: Optional[np.ndarray] = None,
    tai_score_vector: Optional[np.ndarray] = None,
    repeat_penalty_mode: str = DEFAULT_GA_REPEAT_PENALTY_MODE,
    strict_degeneracy_weight: float = DEFAULT_GA_STRICT_DEGENERACY_WEIGHT,
    repeat_penalty_scale: float = DEFAULT_GA_REPEAT_PENALTY_SCALE,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute the multi-objective fitness score for a sequence.

    The keyword-only `codon_table`, `cai_score_vector`, `cps_matrix`, and
    `tai_score_vector` arguments let the GA pre-load these per-host objects
    once and reuse them across thousands of evaluations. Standalone callers
    can omit them and pay a one-time load cost on first invocation.
    """
    if weights is None:
        weights = FitnessWeights()
    if target_gc is None:
        target_gc = get_target_gc(host)
    if avoid_motifs is None:
        avoid_motifs = []

    if codon_table is None:
        codon_table = load_codon_table(host)
    if cai_score_vector is None:
        cai_score_vector = get_codon_score_vector(host)

    normalized_repeat_mode = normalize_ga_repeat_penalty_mode(repeat_penalty_mode)
    metrics: Dict[str, Any] = {}

    codon_indices = _seq_to_codon_indices(sequence)

    # 1. CAI (higher is better)
    metrics['cai'] = calculate_cai_fast(codon_indices, cai_score_vector)

    # 2. GC content deviation
    gc_content = calculate_gc_content(sequence)
    metrics['gc_content'] = gc_content
    metrics['gc_deviation'] = abs(gc_content - target_gc) / 100.0

    # 3. GC3 content deviation
    gc3_content = calculate_gc3_content(sequence)
    metrics['gc3_content'] = gc3_content
    target_gc3 = get_target_gc3(host)
    metrics['gc3_deviation'] = abs(gc3_content - target_gc3) / 100.0

    # 4. Folding energy (5' region)
    worst_mfe, fraction_stable = calculate_folding_energy_windowed(sequence, window_size=150)
    metrics['worst_mfe'] = worst_mfe
    metrics['fraction_stable'] = fraction_stable
    folding_penalty = (abs(worst_mfe) / 50.0) * 0.5 + fraction_stable * 0.5
    metrics['folding_penalty'] = folding_penalty

    # 5. Accessibility (positive term)
    metrics['accessibility'] = calculate_accessibility_score(sequence, start_region_size=30)

    # 6. Unwanted motifs
    motif_count = count_unwanted_motifs(sequence, avoid_motifs)
    metrics['unwanted_motifs'] = motif_count
    metrics['motif_penalty'] = min(motif_count / 10.0, 1.0)

    # 7. Repetitive sequences
    if normalized_repeat_mode == GA_REPEAT_PENALTY_MODE_LEGACY:
        mean_pen, max_pen, frac_bad = repeat_penalty_windowed(sequence)
        repeat_window, repeat_step = repeat_penalty_window_params(len(sequence))
        repeat_metrics: Dict[str, Any] = {
            'repeat_mean': mean_pen,
            'repeat_max': max_pen,
            'repeat_frac_bad': frac_bad,
            'repeat_window_nt': repeat_window,
            'repeat_step_nt': repeat_step,
        }
    else:
        repeat_metrics = compute_degeneracy_metrics(sequence)
        numeric_repeat_metrics = {
            key: value
            for key, value in repeat_metrics.items()
            if isinstance(value, (int, float))
        }
        metrics.update(numeric_repeat_metrics)

    metrics.update({
        'repeat_mean': repeat_metrics['repeat_mean'],
        'repeat_max': repeat_metrics['repeat_max'],
        'repeat_frac_bad': repeat_metrics['repeat_frac_bad'],
        'repeat_window_nt': repeat_metrics['repeat_window_nt'],
        'repeat_step_nt': repeat_metrics['repeat_step_nt'],
    })
    repeat_score = calculate_ga_repeat_score(
        repeat_metrics,
        mode=normalized_repeat_mode,
        strict_degeneracy_weight=strict_degeneracy_weight,
    )
    metrics['legacy_repeat_score'] = calculate_ga_repeat_score(
        repeat_metrics,
        mode=GA_REPEAT_PENALTY_MODE_LEGACY,
    )
    metrics['strict_degeneracy_component'] = calculate_strict_degeneracy_component(
        repeat_metrics
    )
    metrics['repetitive_sequences'] = repeat_score
    metrics['repeat_penalty'] = normalize_ga_repeat_penalty(
        repeat_score,
        repeat_penalty_scale=repeat_penalty_scale,
    )
    metrics['repeat_penalty_scale'] = repeat_penalty_scale
    metrics['strict_degeneracy_weight'] = strict_degeneracy_weight

    # 8. Cryptic splice sites
    splice_count = count_cryptic_splice_sites(sequence, is_eukaryote)
    metrics['cryptic_splice_sites'] = splice_count
    metrics['splice_penalty'] = min(splice_count / 5.0, 1.0)

    # 9. Internal ATG
    internal_atg = check_internal_start_codons(sequence, region_size=90)
    metrics['internal_atg'] = internal_atg
    metrics['atg_penalty'] = min(internal_atg / 3.0, 1.0)

    # 10. tAI (positive term, opt-in)
    if weights.tai > 0:
        if tai_score_vector is None:
            try:
                tai_score_vector = get_tai_score_vector(host)
            except FileNotFoundError:
                tai_score_vector = None
        if tai_score_vector is not None:
            metrics['tai'] = calculate_cai_fast(codon_indices, tai_score_vector)
        else:
            # tAI was requested but no weights file is present; report 0
            # and let the user fix the inputs.
            metrics['tai'] = 0.0
    else:
        metrics['tai'] = 0.0

    # 11. Codon-pair bias score (positive term, opt-in)
    if weights.codon_pair_bias > 0:
        if cps_matrix is None:
            try:
                cps_matrix = get_cps_score_matrix(host)
            except FileNotFoundError:
                cps_matrix = None
        if cps_matrix is not None and codon_indices.size >= 2:
            left = codon_indices[:-1]
            right = codon_indices[1:]
            valid_pairs = SENSE_CODON_INDEX_MASK[left] & SENSE_CODON_INDEX_MASK[right]
            if np.any(valid_pairs):
                pair_scores = cps_matrix[left[valid_pairs], right[valid_pairs]]
                cps_value = float(np.mean(pair_scores))
                cps_normalized = 1.0 / (1.0 + np.exp(-cps_value))
            else:
                cps_value = 0.0
                cps_normalized = 0.0
        else:
            cps_value = 0.0
            cps_normalized = 0.0
        metrics['codon_pair_score'] = cps_value
        metrics['cps_normalized'] = cps_normalized
    else:
        metrics['codon_pair_score'] = 0.0
        metrics['cps_normalized'] = 0.0

    fitness = (
        weights.cai * metrics['cai']
        - weights.gc_deviation * metrics['gc_deviation']
        - weights.folding_energy * metrics['folding_penalty']
        - weights.unwanted_motifs * metrics['motif_penalty']
        - weights.repeats * metrics['repeat_penalty']
        - weights.cryptic_splice * metrics['splice_penalty']
        - weights.gc3_deviation * metrics['gc3_deviation']
        - weights.internal_atg * metrics['atg_penalty']
        + weights.accessibility * metrics['accessibility']
        + weights.tai * metrics['tai']
        + weights.codon_pair_bias * metrics['cps_normalized']
    )

    metrics['fitness'] = fitness
    return fitness, metrics


# ---------------------------------------------------------------------------
# Genetic algorithm
# ---------------------------------------------------------------------------

class GeneticAlgorithm:
    """Genetic algorithm for synonymous-codon sequence optimization."""

    def __init__(
        self,
        initial_sequence: str,
        host: str,
        is_eukaryote: bool,
        target_gc: Optional[float] = None,
        pop_size: int = 80,
        generations: int = 200,
        mutation_rate: float = 0.02,
        crossover_rate: float = 0.7,
        tournament_size: int = 5,
        elitism: int = 2,
        weights: Optional[FitnessWeights] = None,
        avoid_motifs: Optional[List[str]] = None,
        random_seed: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
        repeat_penalty_mode: str = DEFAULT_GA_REPEAT_PENALTY_MODE,
        strict_degeneracy_weight: float = DEFAULT_GA_STRICT_DEGENERACY_WEIGHT,
        repeat_penalty_scale: float = DEFAULT_GA_REPEAT_PENALTY_SCALE,
    ):
        self.initial_sequence = initial_sequence.upper()
        self.start_codon = self.initial_sequence[:3]
        self.host = host
        self.is_eukaryote = is_eukaryote
        self.target_gc = target_gc if target_gc is not None else get_target_gc(host)
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.weights = weights or FitnessWeights()
        self.avoid_motifs = avoid_motifs or []
        self.progress_callback = progress_callback
        self.repeat_penalty_mode = normalize_ga_repeat_penalty_mode(repeat_penalty_mode)
        self.strict_degeneracy_weight = strict_degeneracy_weight
        self.repeat_penalty_scale = repeat_penalty_scale

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        # Cached per-host artefacts. Loaded once here, reused on every
        # fitness call. Big perf win versus the previous implementation.
        self.codon_table = load_codon_table(host)
        self.cai_score_vector = get_codon_score_vector(host)
        self.cps_matrix = None
        if self.weights.codon_pair_bias > 0:
            try:
                self.cps_matrix = get_cps_score_matrix(host)
            except FileNotFoundError as exc:
                print(f"Warning: codon-pair bias requested but {exc}")
                self.cps_matrix = None
        self.tai_score_vector = None
        if self.weights.tai > 0:
            try:
                self.tai_score_vector = get_tai_score_vector(host)
            except FileNotFoundError as exc:
                print(f"Warning: tAI weight requested but {exc}")
                self.tai_score_vector = None

        self.original_protein = translate_dna_to_protein(self.initial_sequence)

        self.population: List[str] = []
        self.fitness_history: List[float] = []
        self.best_sequence: Optional[str] = None
        self.best_fitness: float = float('-inf')
        self.best_metrics: Optional[Dict[str, Any]] = None

    # ----- population management -----------------------------------------

    def initialize_population(self) -> None:
        """Seed the population with the preoptimization sequence plus variants."""
        self.population = [self.initial_sequence]
        mutation_rates = [0.1, 0.2, 0.3, 0.4]

        while len(self.population) < self.pop_size:
            rate = mutation_rates[len(self.population) % len(mutation_rates)]
            variant = self.mutate_for_initialization(self.initial_sequence, rate)
            if self.validate_offspring(variant):
                self.population.append(variant)

    def mutate_for_initialization(self, sequence: str, variation_rate: float) -> str:
        mutated = [self.start_codon]
        for i in range(3, len(sequence) - 3, 3):
            codon = sequence[i:i + 3]
            if random.random() < variation_rate:
                synonymous = get_synonymous_codons(codon)
                if len(synonymous) > 1:
                    weights = [self.codon_table.get(c, 0.1) for c in synonymous]
                    total = sum(weights)
                    if total > 0:
                        probs = [w / total for w in weights]
                        new_codon = np.random.choice(synonymous, p=probs)
                    else:
                        new_codon = random.choice(synonymous)
                    mutated.append(new_codon)
                else:
                    mutated.append(codon)
            else:
                mutated.append(codon)
        mutated.append(sequence[-3:])
        return ''.join(mutated)

    # ----- evaluation ----------------------------------------------------

    def evaluate_population(self) -> List[Tuple[str, float, Dict[str, Any]]]:
        evaluated = []
        for seq in self.population:
            fitness, metrics = fitness_function(
                seq,
                self.host,
                self.is_eukaryote,
                self.target_gc,
                self.weights,
                self.avoid_motifs,
                codon_table=self.codon_table,
                cai_score_vector=self.cai_score_vector,
                cps_matrix=self.cps_matrix,
                tai_score_vector=self.tai_score_vector,
                repeat_penalty_mode=self.repeat_penalty_mode,
                strict_degeneracy_weight=self.strict_degeneracy_weight,
                repeat_penalty_scale=self.repeat_penalty_scale,
            )
            evaluated.append((seq, fitness, metrics))
        evaluated.sort(key=lambda x: x[1], reverse=True)
        return evaluated

    # ----- variation operators -------------------------------------------

    def tournament_selection(
        self,
        evaluated: List[Tuple[str, float, Dict[str, Any]]],
    ) -> str:
        tournament = random.sample(evaluated, min(self.tournament_size, len(evaluated)))
        return max(tournament, key=lambda x: x[1])[0]

    def crossover(self, parent1: str, parent2: str) -> Tuple[str, str]:
        if random.random() > self.crossover_rate:
            return parent1, parent2

        offspring1 = [self.start_codon]
        offspring2 = [self.start_codon]
        for i in range(3, len(parent1) - 3, 3):
            codon1 = parent1[i:i + 3]
            codon2 = parent2[i:i + 3]
            if GENETIC_CODE.get(codon1) == GENETIC_CODE.get(codon2):
                if random.random() < 0.5:
                    offspring1.append(codon1)
                    offspring2.append(codon2)
                else:
                    offspring1.append(codon2)
                    offspring2.append(codon1)
            else:
                offspring1.append(codon1)
                offspring2.append(codon2)
        offspring1.append(parent1[-3:])
        offspring2.append(parent2[-3:])
        return ''.join(offspring1), ''.join(offspring2)

    def mutate(self, sequence: str) -> str:
        mutated = [self.start_codon]
        for i in range(3, len(sequence) - 3, 3):
            codon = sequence[i:i + 3]
            if random.random() < self.mutation_rate:
                synonymous = get_synonymous_codons(codon)
                if len(synonymous) > 1:
                    alternatives = [c for c in synonymous if c != codon]
                    if alternatives:
                        weights = [self.codon_table.get(c, 0.1) for c in alternatives]
                        total = sum(weights)
                        if total > 0:
                            probs = [w / total for w in weights]
                            new_codon = np.random.choice(alternatives, p=probs)
                        else:
                            new_codon = random.choice(alternatives)
                        mutated.append(new_codon)
                    else:
                        mutated.append(codon)
                else:
                    mutated.append(codon)
            else:
                mutated.append(codon)
        mutated.append(sequence[-3:])
        return ''.join(mutated)

    def validate_offspring(self, sequence: str) -> bool:
        try:
            if self.start_codon and sequence[:3] != self.start_codon:
                return False
            return translate_dna_to_protein(sequence) == self.original_protein
        except Exception:
            return False

    # ----- main loop -----------------------------------------------------

    def run(self, verbose: bool = True) -> Tuple[str, float, Dict[str, Any]]:
        self.initialize_population()

        for generation in range(self.generations):
            if self.progress_callback:
                self.progress_callback(generation, self.generations)

            evaluated = self.evaluate_population()

            best_in_gen = evaluated[0]
            if best_in_gen[1] > self.best_fitness:
                self.best_sequence = best_in_gen[0]
                self.best_fitness = best_in_gen[1]
                self.best_metrics = best_in_gen[2]

            self.fitness_history.append(self.best_fitness)

            if verbose and generation % 10 == 0:
                avg_fitness = np.mean([f for _, f, _ in evaluated])
                print(
                    f"Generation {generation:3d}: Best = {self.best_fitness:.4f}, "
                    f"Avg = {avg_fitness:.4f}"
                )

            if len(self.fitness_history) > 20:
                recent = self.fitness_history[-20:]
                if max(recent) - min(recent) < 0.0001:
                    if verbose:
                        print(f"Converged at generation {generation}")
                    if self.progress_callback:
                        self.progress_callback(self.generations, self.generations)
                    break

            new_population: List[str] = []
            for i in range(min(self.elitism, len(evaluated))):
                new_population.append(evaluated[i][0])

            while len(new_population) < self.pop_size:
                parent1 = self.tournament_selection(evaluated)
                parent2 = self.tournament_selection(evaluated)
                offspring1, offspring2 = self.crossover(parent1, parent2)
                offspring1 = self.mutate(offspring1)
                offspring2 = self.mutate(offspring2)
                if self.validate_offspring(offspring1):
                    new_population.append(offspring1)
                if len(new_population) < self.pop_size and self.validate_offspring(offspring2):
                    new_population.append(offspring2)

            self.population = new_population[:self.pop_size]

        if self.progress_callback:
            self.progress_callback(self.generations, self.generations)

        if verbose:
            print("\nOptimization complete!")
            print(f"Final fitness: {self.best_fitness:.4f}")

        return self.best_sequence, self.best_fitness, self.best_metrics


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def genetic_algorithm(
    initial_cds: str,
    host: str = 'hsapiens',
    is_eukaryote: bool = True,
    target_gc: Optional[float] = None,
    pop_size: int = 80,
    generations: int = 200,
    mutation_rate: float = 0.02,
    weights: Optional[Dict[str, float]] = None,
    avoid_sequences: str = "",
    random_seed: Optional[int] = None,
    verbose: bool = True,
    progress_callback: Optional[Callable] = None,
    repeat_penalty_mode: str = DEFAULT_GA_REPEAT_PENALTY_MODE,
    strict_degeneracy_weight: float = DEFAULT_GA_STRICT_DEGENERACY_WEIGHT,
    repeat_penalty_scale: float = DEFAULT_GA_REPEAT_PENALTY_SCALE,
) -> Tuple[str, float, Dict[str, Any]]:
    """
    Convenience wrapper that constructs a GeneticAlgorithm with weights
    parsed from a plain dict.
    """
    fitness_weights = FitnessWeights(**weights) if weights else FitnessWeights()

    avoid_motifs: List[str] = []
    if avoid_sequences:
        avoid_motifs = [seq.strip().upper() for seq in avoid_sequences.split(',') if seq.strip()]

    ga = GeneticAlgorithm(
        initial_sequence=initial_cds,
        host=host,
        is_eukaryote=is_eukaryote,
        target_gc=target_gc,
        pop_size=pop_size,
        generations=generations,
        mutation_rate=mutation_rate,
        weights=fitness_weights,
        avoid_motifs=avoid_motifs,
        random_seed=random_seed,
        progress_callback=progress_callback,
        repeat_penalty_mode=repeat_penalty_mode,
        strict_degeneracy_weight=strict_degeneracy_weight,
        repeat_penalty_scale=repeat_penalty_scale,
    )
    return ga.run(verbose=verbose)
