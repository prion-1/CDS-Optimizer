"""
Strategies for initial codon optimization. The output of this module is what is then fed into the genetic algorithm.
Three strategies to choose from: maximum CAI (MC), percentile matching (PM), 

and distribution matching (DM).


MC uses the most frequent host codon for each AA.
PM tries to match the input codon usage pattern to the host codon usage pattern by percentile, preserving the codon
bias of the input while converting it into host codons of corresponding frequencies.


CHARMING!!!

^^^^^^^^^^^^^^^^^^^^
^^^^^^^^^^^^^^^^^^^^
"""

import numpy as np
from typing import Dict, Tuple, Optional
from collections import defaultdict
from scipy.stats import wasserstein_distance
from itertools import permutations

from .utils import (
    load_codon_table, GENETIC_CODE, AA_TO_CODONS,
    translate_dna_to_protein,
    # Import numba functions from utils instead of duplicating
    NUMBA_AVAILABLE
)

def simple_best_codon_optimization(sequence: str, host: str) -> str:
    """
    Simple optimization using the most preferred host codon for each AA.
    
    Args:
        sequence: Input DNA sequence
        host: Host organism
    
    Returns:
        Optimized DNA sequence using best codons
    """
    codon_table = load_codon_table(host)
    optimized = []
    
    # Process each codon
    for i in range(0, len(sequence), 3):
        codon = sequence[i:i+3]
        
        # Get amino acid
        aa = GENETIC_CODE.get(codon, 'X')
        if aa == 'X':
            optimized.append(codon)  # Keep unknown codon as is
            continue
        
        # Get all synonymous codons
        synonymous = AA_TO_CODONS.get(aa, [codon])  # Fallback makes sure synonymous is always iterable
        
        # Choose the most preferred codon for this host
        best_codon = max(synonymous, key = lambda c: codon_table.get(c, 0))
        optimized.append(best_codon)
    
    return ''.join(optimized)


def percentile_matching_optimization(sequence: str, host: str) -> str:
    """
    Optimize codons while preserving usage pattern using percentile matching. This is a simple, balanced harmonization.
    Maintain pausing patterns/translational rhythm.
    
    Args:
        sequence: Input DNA sequence
        host: Host organism
    
    Returns:
        Optimized DNA sequence preserving usage patterns
    """
    codon_table = load_codon_table(host)
    
    # Analyze codon usage in original sequence
    aa_codon_counts = defaultdict(lambda: defaultdict(int))
    codon_positions = defaultdict(list)  # Track positions for each codon
    
    for i in range(0, len(sequence), 3):
        codon = sequence[i:i+3]
        aa = GENETIC_CODE.get(codon, 'X')
        if aa != 'X' and aa != '*':  # Skip stop codons
            aa_codon_counts[aa][codon] += 1
            codon_positions[codon].append(i)
    
    # Create percentile-based mapping
    aa_codon_mapping = {}
    
    for aa, codon_counts in aa_codon_counts.items():
        # Sort CDS codons by frequency with consistent tiebreaking
        cds_codons_sorted = sorted(
            codon_counts.items(),
            key=lambda x: (-x[1], min(codon_positions[x[0]]))  # Count desc, then first position
        )
        
        # Calculate percentiles for CDS codons
        total_count = sum(codon_counts.values())
        cds_percentiles = []
        cumulative = 0
        
        for codon, count in cds_codons_sorted:
            # Use midpoint percentile
            percentile = (cumulative + count/2) / total_count if total_count > 0 else 0.5
            cds_percentiles.append((codon, percentile, count))
            cumulative += count
        
        # Get host codons and their frequencies
        host_codons = AA_TO_CODONS.get(aa, [])
        host_codon_freqs = [(c, codon_table.get(c, 0)) for c in host_codons]
        
        # Handle edge case: no host codons
        if not host_codon_freqs:
            for codon, _, _ in cds_percentiles:
                aa_codon_mapping[(aa, codon)] = codon
            continue
        
        # Sort host codons by frequency with consistent tiebreaking
        host_codon_freqs.sort(key=lambda x: (-x[1], x[0]))  # Freq desc, then alphabetical
        
        # Calculate percentiles for host codons
        total_freq = sum(f for _, f in host_codon_freqs)
        if total_freq == 0:  # All frequencies are 0
            # Distribute evenly
            for i, (codon, _, _) in enumerate(cds_percentiles):
                host_idx = min(i, len(host_codon_freqs) - 1)
                aa_codon_mapping[(aa, codon)] = host_codon_freqs[host_idx][0]
        else:
            host_percentiles = []
            cumulative = 0
            
            for codon, freq in host_codon_freqs:
                percentile = (cumulative + freq/2) / total_freq
                host_percentiles.append((codon, percentile))
                cumulative += freq
            
            # Map CDS codons to nearest percentile host codon
            for cds_codon, cds_percentile, _ in cds_percentiles:
                # Find host codon with nearest percentile
                best_host = min(
                    host_percentiles,
                    key=lambda x: abs(x[1] - cds_percentile)
                )
                aa_codon_mapping[(aa, cds_codon)] = best_host[0]
    
    # Apply mapping to sequence
    optimized = []
    for i in range(0, len(sequence), 3):
        codon = sequence[i:i+3]
        aa = GENETIC_CODE.get(codon, 'X')
        mapped_codon = aa_codon_mapping.get((aa, codon), codon)
        optimized.append(mapped_codon)
    
    return ''.join(optimized)


'''
######## PROBABLY REPLACE BY CHARMING
def distribution_matching_optimization(sequence: str, host: str) -> str:
    """
    Optimize codons to match distribution shape as closely as possible.
    Uses Earth Mover's Distance for optimal mapping.
    
    Args:
        sequence: Input DNA sequence
        host: Host organism
    
    Returns:
        Optimized DNA sequence matching distribution
    """
    codon_table = load_codon_table(host)
    
    # Analyze codon usage in original sequence
    aa_codon_counts = defaultdict(lambda: defaultdict(int))
    codon_positions = defaultdict(list)
    
    for i in range(0, len(sequence), 3):
        codon = sequence[i:i+3]
        aa = GENETIC_CODE.get(codon, 'X')
        if aa != 'X' and aa != '*':  # Skip stop codons
            aa_codon_counts[aa][codon] += 1
            codon_positions[codon].append(i)
    
    # Create distribution-based mapping
    aa_codon_mapping = {}
    
    for aa, codon_counts in aa_codon_counts.items():
        # Get CDS codons and their counts
        cds_codons = list(codon_counts.keys())
        cds_counts = list(codon_counts.values())
        
        # Normalize to create distribution
        total_count = sum(cds_counts)
        cds_distribution = [c/total_count for c in cds_counts] if total_count > 0 else [1.0/len(cds_counts)] * len(cds_counts)
        
        # Get host codons and their frequencies
        host_codons = AA_TO_CODONS.get(aa, [])
        host_freqs = [codon_table.get(c, 0) for c in host_codons]
        
        # Handle edge cases
        if not host_codons:
            for codon in cds_codons:
                aa_codon_mapping[(aa, codon)] = codon
            continue
        
        if len(cds_codons) == 1:
            # Single codon - use best host codon
            best_host = max(host_codons, key=lambda c: codon_table.get(c, 0))
            aa_codon_mapping[(aa, cds_codons[0])] = best_host
            continue
        
        # For small numbers of codons (≤6), try all permutations
        if len(cds_codons) <= 6 and len(cds_codons) <= len(host_codons):
            best_distance = float('inf')
            best_mapping = {}
            
            # Try all possible mappings
            for perm in permutations(host_codons, len(cds_codons)):
                # Calculate distribution of this mapping
                mapped_freqs = [codon_table.get(h, 0) for h in perm]
                
                # Normalize host frequencies
                total_freq = sum(mapped_freqs)
                if total_freq > 0:
                    host_distribution = [f/total_freq for f in mapped_freqs]
                else:
                    host_distribution = [1.0/len(mapped_freqs)] * len(mapped_freqs)
                
                # Calculate Earth Mover's Distance
                try:
                    distance = wasserstein_distance(cds_distribution, host_distribution)
                except:
                    # Fallback to simple difference if wasserstein fails
                    distance = sum(abs(a - b) for a, b in zip(cds_distribution, host_distribution))
                
                if distance < best_distance:
                    best_distance = distance
                    best_mapping = dict(zip(cds_codons, perm))
            
            # Apply best mapping
            for cds_codon, host_codon in best_mapping.items():
                aa_codon_mapping[(aa, cds_codon)] = host_codon
        else:
            # Too many codons - use greedy matching based on rank
            # Sort both by frequency/count
            cds_sorted = sorted(zip(cds_codons, cds_counts), key=lambda x: -x[1])
            host_sorted = sorted(host_codons, key=lambda c: -codon_table.get(c, 0))
            
            # Map by rank
            for i, (cds_codon, _) in enumerate(cds_sorted):
                host_idx = min(i, len(host_sorted) - 1)
                aa_codon_mapping[(aa, cds_codon)] = host_sorted[host_idx]
    
    # Apply mapping to sequence
    optimized = []
    for i in range(0, len(sequence), 3):
        
        # remove
        # Check for terminal codons
        #if _handle_terminal_codons(sequence, codon_table, optimized, i):
        #    continue
        
        codon = sequence[i:i+3]
        aa = GENETIC_CODE.get(codon, 'X')
        mapped_codon = aa_codon_mapping.get((aa, codon), codon)
        optimized.append(mapped_codon)
    
    return ''.join(optimized)
'''


def optimize_codons(sequence: str, host: str, method: str = 'simple') -> str:
    """
    Main entry point for codon optimization.
    
    Args:
        sequence: Input DNA sequence
        host: Host organism
        method: Optimization method ('simple', 'percentile', 'charming')
    
    Returns:
        Optimized DNA sequence
    """
    # Validate that protein is preserved
    original_protein = translate_dna_to_protein(sequence)
    
    # Apply selected optimization method
    method_lower = method.lower()
    if 'simple' in method_lower:
        optimized = simple_best_codon_optimization(sequence, host)
    elif 'percentile' in method_lower:
        optimized = percentile_matching_optimization(sequence, host)
    else:
        raise ValueError(f"Unknown optimization method: {method}")

    '''
    elif 'distribution' in method_lower:
        optimized = distribution_matching_optimization(sequence, host)
    '''
    
    # Verify protein is preserved
    optimized_protein = translate_dna_to_protein(optimized)
    if original_protein != optimized_protein:
        raise ValueError("Protein sequence changed during optimization!")
    
    return optimized