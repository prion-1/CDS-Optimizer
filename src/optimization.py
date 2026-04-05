"""
Genetic algorithm and fitness function for mRNA sequence optimization.
"""

import random
import copy
from typing import List, Tuple, Dict, Optional, Callable
import numpy as np

from .utils import (
    load_codon_table, calculate_cai, calculate_gc_content, calculate_gc3_content,
    count_unwanted_motifs, calculate_folding_energy_windowed, calculate_accessibility_score,
    repeat_penalty_windowed, count_cryptic_splice_sites, check_internal_start_codons,
    translate_dna_to_protein, get_synonymous_codons,
    HOST_TARGET_GC, GENETIC_CODE
)


class FitnessWeights:
    """Container for fitness function weights."""
    def __init__(self,
                 cai: float = 0.40,
                 gc_deviation: float = 0.20,
                 folding_energy: float = 0.15,
                 unwanted_motifs: float = 0.10,
                 repeats: float = 0.07,
                 cryptic_splice: float = 0.08,
                 gc3_deviation: float = 0.0,
                 internal_atg: float = 0.0,
                 accessibility: float = 0.0):
        """
        Initialize fitness weights.
        
        Args:
            cai: Weight for Codon Adaptation Index
            gc_deviation: Weight for GC content deviation penalty
            folding_energy: Weight for folding energy penalty
            unwanted_motifs: Weight for unwanted motifs penalty
            repeats: Weight for repetitive sequences penalty
            cryptic_splice: Weight for cryptic splice sites penalty
            gc3_deviation: Weight for GC3 content deviation (optional)
            internal_atg: Weight for internal ATG penalty (optional)
            accessibility: Weight for 5' accessibility bonus (optional)
        """
        self.cai = cai
        self.gc_deviation = gc_deviation
        self.folding_energy = folding_energy
        self.unwanted_motifs = unwanted_motifs
        self.repeats = repeats
        self.cryptic_splice = cryptic_splice
        self.gc3_deviation = gc3_deviation
        self.internal_atg = internal_atg
        self.accessibility = accessibility
        
        # Normalize if requested
        total = sum([cai, gc_deviation, folding_energy, unwanted_motifs,
                    repeats, cryptic_splice, gc3_deviation, internal_atg, accessibility])
        if abs(total - 1.0) > 0.01:
            print(f"Warning: Weights sum to {total:.2f}, not 1.0")


def fitness_function(sequence: str, 
                    host: str,
                    is_eukaryote: bool,
                    target_gc: Optional[float] = None,
                    weights: Optional[FitnessWeights] = None,
                    avoid_motifs: Optional[List[str]] = None) -> Tuple[float, Dict[str, float]]:
    """
    Calculate fitness score for a sequence.
    
    Args:
        sequence: DNA sequence to evaluate
        host: Host organism
        target_gc: Target GC content (uses host default if None)
        weights: FitnessWeights object (uses defaults if None)
        avoid_motifs: List of motifs to avoid
    
    Returns:
        Tuple of (fitness_score, metrics_dict)
    """
    if weights is None:
        weights = FitnessWeights()
    
    if target_gc is None:
        target_gc = HOST_TARGET_GC.get(host.lower(), 50.0)
    
    if avoid_motifs is None:
        avoid_motifs = []
    
    # Load codon table for host
    codon_table = load_codon_table(host)
    
    # Calculate individual metrics
    metrics = {}
    
    # 1. Codon Adaptation Index (higher is better)
    metrics['cai'] = calculate_cai(sequence, codon_table)
    
    # 2. GC content deviation (lower is better)
    gc_content = calculate_gc_content(sequence)
    metrics['gc_content'] = gc_content
    metrics['gc_deviation'] = abs(gc_content - target_gc) / 100.0
    
    # 3. GC3 content (for hosts like yeast)
    gc3_content = calculate_gc3_content(sequence)
    metrics['gc3_content'] = gc3_content
    
    # Calculate target GC3 based on host
    if host.lower() in ['scerevisiae', 's.cerevisiae', 'yeast', 'spombe', 's.pombe']:
        target_gc3 = 25.0  # Yeast prefer AT at position 3
    else:
        target_gc3 = 60.0  # Mammals prefer GC at position 3
    
    metrics['gc3_deviation'] = abs(gc3_content - target_gc3) / 100.0
    
    # 4. Folding energy (5' region analysis)
    worst_mfe, fraction_stable = calculate_folding_energy_windowed(sequence, window_size=150)
    metrics['worst_mfe'] = worst_mfe
    metrics['fraction_stable'] = fraction_stable
    
    # Normalize folding penalty (less negative is better)
    # MFE typically ranges from 0 to -50 kcal/mol
    folding_penalty = (abs(worst_mfe) / 50.0) * 0.5 + fraction_stable * 0.5
    metrics['folding_penalty'] = folding_penalty
    
    # 5. Accessibility score for ribosome binding
    metrics['accessibility'] = calculate_accessibility_score(sequence, start_region_size=30)
    
    # 6. Unwanted motifs (lower is better)
    motif_count = count_unwanted_motifs(sequence, avoid_motifs)
    metrics['unwanted_motifs'] = motif_count
    # Normalize (assume max 10 motifs for full penalty)
    metrics['motif_penalty'] = min(motif_count / 10.0, 1.0)
    
    # 7. Repetitive sequences (lower is better)
    mean_pen, max_pen, frac_bad = repeat_penalty_windowed(sequence)
    repeat_score = 0.6 * mean_pen + 0.4 * max_pen
    # Apply prevalence penalty if repeats are widespread (arbitrary: 10% boost if > 30% windows repetitive)
    if frac_bad > 0.3:
        repeat_score *= 1.1

    metrics['repetitive_sequences'] = repeat_score
    # Normalize (assume max score of 20 for full penalty)
    metrics['repeat_penalty'] = min(repeat_score / 20.0, 1.0)
    
    # 8. Cryptic splice sites (lower is better)
    splice_count = count_cryptic_splice_sites(sequence, is_eukaryote)
    metrics['cryptic_splice_sites'] = splice_count
    # Normalize (assume max 5 sites for full penalty)
    metrics['splice_penalty'] = min(splice_count / 5.0, 1.0)
    
    # 9. Internal ATG codons in 5' region (lower is better)
    internal_atg = check_internal_start_codons(sequence, region_size=90)
    metrics['internal_atg'] = internal_atg
    # Normalize (assume max 3 for full penalty)
    metrics['atg_penalty'] = min(internal_atg / 3.0, 1.0)
    
    # Calculate composite fitness score
    fitness = (
        weights.cai * metrics['cai'] -
        weights.gc_deviation * metrics['gc_deviation'] -
        weights.folding_energy * metrics['folding_penalty'] -
        weights.unwanted_motifs * metrics['motif_penalty'] -
        weights.repeats * metrics['repeat_penalty'] -
        weights.cryptic_splice * metrics['splice_penalty'] -
        weights.gc3_deviation * metrics['gc3_deviation'] -
        weights.internal_atg * metrics['atg_penalty'] +
        weights.accessibility * metrics['accessibility']
    )
    
    metrics['fitness'] = fitness
    
    return fitness, metrics


class GeneticAlgorithm:
    """Genetic algorithm for sequence optimization."""
    
    def __init__(self,
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
                 progress_callback: Optional[Callable] = None):
        """
        Initialize genetic algorithm.
        
        Args:
            initial_sequence: Starting DNA sequence
            host: Host organism
            target_gc: Target GC content
            pop_size: Population size
            generations: Number of generations
            mutation_rate: Probability of mutation per codon
            crossover_rate: Probability of crossover
            tournament_size: Size of tournament selection
            elitism: Number of elite individuals to preserve
            weights: Fitness function weights
            avoid_motifs: Motifs to avoid
            random_seed: Random seed for reproducibility
            progress_callback: Optional callback function(current_gen, total_gens) for progress updates
        """
        #####REMOVE
        ''''
        is_valid, error_msg = validate_sequence(initial_sequence)
        if not is_valid:
            raise ValueError(f"Invalid input sequence: {error_msg}")
        '''
        
        self.initial_sequence = initial_sequence.upper()
        self.host = host
        self.is_eukaryote = is_eukaryote
        self.target_gc = target_gc or HOST_TARGET_GC.get(host.lower(), 50.0)
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.weights = weights or FitnessWeights()
        self.avoid_motifs = avoid_motifs or []
        self.progress_callback = progress_callback
        
        # Set random seed if provided
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Load codon table
        self.codon_table = load_codon_table(host)
        
        # Store original protein for validation
        self.original_protein = translate_dna_to_protein(self.initial_sequence)
        
        # Initialize population
        self.population = []
        self.fitness_history = []
        self.best_sequence = None
        self.best_fitness = float('-inf')
        self.best_metrics = None
    
    def initialize_population(self):
        """Create initial population with sequence variants."""
        self.population = []
        
        # Add pre-optimized sequence (from Phase 1)
        self.population.append(self.initial_sequence)
        
        # Create variants with different mutation levels
        mutation_rates = [0.1, 0.2, 0.3, 0.4]  # Different variation levels
        
        while len(self.population) < self.pop_size:
            # Use different mutation rates for diversity
            rate = mutation_rates[len(self.population) % len(mutation_rates)]
            variant = self.mutate_for_initialization(self.initial_sequence, rate)
            if self.validate_offspring(variant):
                self.population.append(variant)
    
    def mutate_for_initialization(self, sequence: str, variation_rate: float) -> str:
        """
        Create a variant by mutating with higher rate for initialization.
        
        Args:
            sequence: Original sequence
            variation_rate: Rate of mutation for this variant
        
        Returns:
            Variant sequence
        """
        mutated = []
        
        # Preserve start codon
        mutated.append('ATG')
        
        # Mutate middle codons with higher rate for diversity
        for i in range(3, len(sequence) - 3, 3):
            codon = sequence[i:i+3]
            
            if random.random() < variation_rate:
                # Get synonymous codons
                synonymous = get_synonymous_codons(codon)
                
                if len(synonymous) > 1:
                    # Choose based on codon table
                    weights = [self.codon_table.get(c, 0.1) for c in synonymous]
                    total = sum(weights)
                    if total > 0:
                        weights = [w/total for w in weights]
                        new_codon = np.random.choice(synonymous, p=weights)
                    else:
                        new_codon = random.choice(synonymous)
                    mutated.append(new_codon)
                else:
                    mutated.append(codon)
            else:
                mutated.append(codon)
        
        # Preserve stop codon
        mutated.append(sequence[-3:])
        
        return ''.join(mutated)
    
    def evaluate_population(self) -> List[Tuple[str, float, Dict]]:
        """
        Evaluate fitness of all individuals in population.
        
        Returns:
            List of (sequence, fitness, metrics) tuples
        """
        evaluated = []
        
        for seq in self.population:
            fitness, metrics = fitness_function(seq, self.host, self.is_eukaryote, self.target_gc,
                                               self.weights, self.avoid_motifs)
            evaluated.append((seq, fitness, metrics))
        
        # Sort by fitness (descending)
        evaluated.sort(key=lambda x: x[1], reverse=True)
        
        return evaluated
    
    def tournament_selection(self, evaluated: List[Tuple[str, float, Dict]]) -> str:
        """
        Select individual using tournament selection.
        
        Args:
            evaluated: List of evaluated individuals
        
        Returns:
            Selected sequence
        """
        tournament = random.sample(evaluated, min(self.tournament_size, len(evaluated)))
        winner = max(tournament, key=lambda x: x[1])
        return winner[0]
    
    def crossover(self, parent1: str, parent2: str) -> Tuple[str, str]:
        """
        Perform uniform crossover between two parents.
        
        Args:
            parent1: First parent sequence
            parent2: Second parent sequence
        
        Returns:
            Two offspring sequences
        """
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        offspring1 = []
        offspring2 = []
        
        # Preserve start codon
        offspring1.append('ATG')
        offspring2.append('ATG')
        
        # Uniform crossover for middle codons
        for i in range(3, len(parent1) - 3, 3):
            codon1 = parent1[i:i+3]
            codon2 = parent2[i:i+3]
            
            # Ensure both codons encode the same amino acid
            if GENETIC_CODE.get(codon1) == GENETIC_CODE.get(codon2):
                if random.random() < 0.5:
                    offspring1.append(codon1)
                    offspring2.append(codon2)
                else:
                    offspring1.append(codon2)
                    offspring2.append(codon1)
            else:
                # Keep original if amino acids don't match
                offspring1.append(codon1)
                offspring2.append(codon2)
        
        # Preserve stop codon
        offspring1.append(parent1[-3:])
        offspring2.append(parent2[-3:])
        
        return ''.join(offspring1), ''.join(offspring2)
    
    def mutate(self, sequence: str) -> str:
        """
        Apply mutation to a sequence.
        
        Args:
            sequence: Input sequence
        
        Returns:
            Mutated sequence
        """
        mutated = []
        
        # Preserve start codon
        mutated.append('ATG')
        
        # Potentially mutate middle codons
        for i in range(3, len(sequence) - 3, 3):
            codon = sequence[i:i+3]
            
            if random.random() < self.mutation_rate:
                # Get synonymous codons
                synonymous = get_synonymous_codons(codon)
                
                if len(synonymous) > 1:
                    # Remove current codon from options
                    alternatives = [c for c in synonymous if c != codon]
                    if alternatives:
                        # Choose based on codon table
                        weights = [self.codon_table.get(c, 0.1) for c in alternatives]
                        total = sum(weights)
                        if total > 0:
                            weights = [w/total for w in weights]
                            new_codon = np.random.choice(alternatives, p=weights)
                        else:
                            new_codon = random.choice(alternatives)
                        mutated.append(new_codon)
                    else:
                        mutated.append(codon)
                else:
                    mutated.append(codon)
            else:
                mutated.append(codon)
        
        # Preserve stop codon
        mutated.append(sequence[-3:])
        
        return ''.join(mutated)
    
    def validate_offspring(self, sequence: str) -> bool:
        """
        Validate that offspring preserves the original protein.
        
        Args:
            sequence: Sequence to validate
        
        Returns:
            True if valid, False otherwise
        """
        try:
            protein = translate_dna_to_protein(sequence)
            return protein == self.original_protein
        except:
            return False
    
    def run(self, verbose: bool = True) -> Tuple[str, float, Dict]:
        """
        Run the genetic algorithm.
        
        Args:
            verbose: Whether to print progress
        
        Returns:
            Tuple of (best_sequence, best_fitness, best_metrics)
        """
        # Initialize population
        self.initialize_population()
        
        # Evolution loop
        for generation in range(self.generations):
            # Call progress callback if provided
            if self.progress_callback:
                self.progress_callback(generation, self.generations)
            
            # Evaluate current population
            evaluated = self.evaluate_population()
            
            # Track best individual
            best_in_gen = evaluated[0]
            if best_in_gen[1] > self.best_fitness:
                self.best_sequence = best_in_gen[0]
                self.best_fitness = best_in_gen[1]
                self.best_metrics = best_in_gen[2]
            
            self.fitness_history.append(self.best_fitness)
            
            # Print progress
            if verbose and generation % 10 == 0:
                avg_fitness = np.mean([f for _, f, _ in evaluated])
                print(f"Generation {generation:3d}: Best = {self.best_fitness:.4f}, "
                      f"Avg = {avg_fitness:.4f}")
            
            # Check for convergence
            if len(self.fitness_history) > 20:
                recent = self.fitness_history[-20:]
                if max(recent) - min(recent) < 0.0001:
                    if verbose:
                        print(f"Converged at generation {generation}")
                    # Final progress update
                    if self.progress_callback:
                        self.progress_callback(self.generations, self.generations)
                    break
            
            # Create next generation
            new_population = []
            
            # Elitism - preserve best individuals
            for i in range(min(self.elitism, len(evaluated))):
                new_population.append(evaluated[i][0])
            
            # Generate offspring
            while len(new_population) < self.pop_size:
                # Selection
                parent1 = self.tournament_selection(evaluated)
                parent2 = self.tournament_selection(evaluated)
                
                # Crossover
                offspring1, offspring2 = self.crossover(parent1, parent2)
                
                # Mutation
                offspring1 = self.mutate(offspring1)
                offspring2 = self.mutate(offspring2)
                
                # Validate and add to population
                if self.validate_offspring(offspring1):
                    new_population.append(offspring1)
                if len(new_population) < self.pop_size and self.validate_offspring(offspring2):
                    new_population.append(offspring2)
            
            self.population = new_population[:self.pop_size]
        
        # Final progress update
        if self.progress_callback:
            self.progress_callback(self.generations, self.generations)
        
        if verbose:
            print(f"\nOptimization complete!")
            print(f"Final fitness: {self.best_fitness:.4f}")
        
        return self.best_sequence, self.best_fitness, self.best_metrics


def genetic_algorithm(initial_cds: str,
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
                     progress_callback: Optional[Callable] = None) -> Tuple[str, float, Dict]:
    """
    Main entry point for genetic algorithm optimization.
    
    Args:
        initial_cds: Initial CDS sequence
        host: Host organism
        target_gc: Target GC content (uses host default if None)
        pop_size: Population size
        generations: Number of generations
        mutation_rate: Mutation rate per codon
        weights: Dictionary of weight values
        avoid_sequences: Comma-separated sequences to avoid
        random_seed: Random seed for reproducibility
        verbose: Whether to print progress
        progress_callback: Optional callback function(current_gen, total_gens) for progress updates
    
    Returns:
        Tuple of (optimized_sequence, fitness_score, metrics)
    """
    # Parse weights if provided
    if weights:
        fitness_weights = FitnessWeights(**weights)
    else:
        fitness_weights = FitnessWeights()
    
    # Parse sequences to avoid
    avoid_motifs = []
    if avoid_sequences:
        avoid_motifs = [seq.strip().upper() for seq in avoid_sequences.split(',') if seq.strip()]
    
    # Create and run genetic algorithm
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
        progress_callback=progress_callback
    )
    
    return ga.run(verbose=verbose)