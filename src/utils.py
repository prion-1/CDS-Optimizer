"""
Miscellaneous utility functions for pre-processing and analysis.
Utility functions for optimization algorigthms/fitness function.
"""

import os
import re
import math
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
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
AA_TO_CODONS = defaultdict(list)
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

# Host-specific target GC contents (Kazusa database, coding seq. GC values)
HOST_TARGET_GC = {
    'hsapiens': 52.3,
    'mmusculus': 52.0,
    'ecoli': 52.4,
    'scerevisiae': 39.8,
    'spombe': 39.8
}
# Future work: load dynamically from CSV in data/

# Valid host organisms
VALID_HOSTS = {'hsapiens', 'mmusculus', 'ecoli', 'scerevisiae', 'spombe'}
# Future work: load dynamically from CSV


def load_codon_table(host: str) -> Dict[str, float]:
    """
    Load codon usage table for specified host organism (relative freqs).
    
    Returns:
        Dictionary mapping codons to their frequencies normalized against the most-used codon for each AA in a host.
    """
    host_lower = host.lower()
    
    # Validate host
    if host_lower not in VALID_HOSTS:
        raise ValueError(f"Invalid host '{host}'. Must be one of: {', '.join(VALID_HOSTS)}")
    
    filepath = os.path.join('data', 'codon_tables', f'{host_lower}.csv')
    
    df = pd.read_csv(filepath)
    codon_table = {}
    
    # Group by amino acid to normalize frequencies
    for aa in df['amino_acid'].unique():
        aa_df = df[df['amino_acid'] == aa]
        max_freq = aa_df['frequency'].max()
        
        for _, row in aa_df.iterrows():
            # Calculate relative adaptiveness value per codon
            codon_table[row['codon']] = row['frequency'] / max_freq if max_freq > 0 else 0
    
    return codon_table
    

def calculate_gc_content(sequence: str) -> float:
    """
    Return total GC content.
    """
    if not sequence:
        return 0.0
    
    # Use numba acceleration if available
    if NUMBA_AVAILABLE:
        seq_array = np.array(list(sequence.encode('ascii')), dtype = np.uint8)
        gc_count = _calculate_gc_content_numba(seq_array)
        return (gc_count / len(sequence)) * 100
    else:
        gc_count = sequence.count('G') + sequence.count('C')
        return (gc_count / len(sequence)) * 100


def calculate_gc3_content(sequence: str) -> float:
    """
    Return GC3 value.
    """ 
    third_positions = [sequence[i+2] for i in range(0, len(sequence), 3)]
    gc3_count = sum(1 for nt in third_positions if nt in 'GC')
    return (gc3_count / len(third_positions)) * 100


def calculate_cai(sequence: str, codon_table: Dict[str, float]) -> float:
    """
    Calculate Codon Adaptation Index (CAI) for a sequence.
    
    Returns:
        CAI score between 0 and 1.
    """
    
    if NUMBA_AVAILABLE:
        # Convert to arrays for numba
        codon_indices = []
        codon_scores = np.zeros(64, dtype = np.float32)
        codon_to_idx = {}
        
        # Build codon index mapping
        idx = 0
        for c1 in 'ACGT':
            for c2 in 'ACGT':
                for c3 in 'ACGT':
                    codon = c1 + c2 + c3
                    codon_to_idx[codon] = idx
                    codon_scores[idx] = codon_table.get(codon, 0.0) # 0.0 if codon not in CSV table
                    idx += 1
        
        # Convert sequence to indices
        for i in range(0, len(sequence), 3):
            codon = sequence[i:i+3]
            if codon in codon_to_idx:
                codon_indices.append(codon_to_idx[codon])
        
        if codon_indices:
            codon_indices_array = np.array(codon_indices, dtype = np.int32)
            return _calculate_cai_numba(codon_indices_array, codon_scores)
        else:
            return 0.0
    else:
        # Slow implementation
        scores = []
        for i in range(0, len(sequence), 3):
            codon = sequence[i:i+3]
            if codon in codon_table:
                score = codon_table[codon]
                if score > 0:
                    scores.append(score)
        
        if not scores:
            return 0.0  
        
        # Geometric mean
        log_sum = sum(math.log(s) for s in scores)
        cai = math.exp(log_sum / len(scores))
        
        return cai


def count_unwanted_motifs(sequence: str, motifs: List[str]) -> int:
    """
    Count occurrences of unwanted motifs in sequence.
    
    Args:
        sequence: Input DNA
        motifs: List of motif sequences to avoid (polyA avoided automatically)
    
    Returns:
        Total count of unwanted motifs
    """
    count = 0
    
    # Default polyA blacklist
    default_polya = ['AATAAA', 'ATTAAA', 'AAAAAA']
    motifs = list(motifs) + default_polya
    
    if NUMBA_AVAILABLE:
        # Use numba for faster searching, counts overlapping occurrences
        seq_array = np.array(list(sequence.encode('ascii')), dtype = np.uint8)
        for motif in motifs:
            motif_array = np.array(list(motif.encode('ascii')), dtype = np.uint8)
            count += _count_motif_numba(seq_array, motif_array)
    else:
        # Counts only non-overlapping occurrences
        for motif in motifs:
            count += sequence.count(motif)
    
    return count


def calculate_folding_energy_windowed(sequence: str, window_size: int = 150) -> Tuple[float, float]:
    """
    Calculate mRNA folding energy focusing on 5' region.
    
    Args:
        sequence: DNA sequence
        window_size: 5' analysis window (default 150 nt)
    
    Returns:
        Tuple of (worst_mfe, fraction_stable_windows)
    """
    # Convert DNA to RNA
    rna_seq = sequence.replace('T', 'U')
    
    # Analyze first window_size bases
    analysis_region = rna_seq[:min(window_size, len(rna_seq))]
    
    if VIENNA_AVAILABLE:
        try:
            # Calculate MFE for the whole analysis region
            structure, mfe = RNA.fold(analysis_region)
            
            # Sliding window analysis with 30nt windows
            window_mfes = []
            step = 10
            sub_window = 30
            
            for i in range(0, len(analysis_region) - sub_window + 1, step):
                sub_seq = analysis_region[i:i+sub_window]
                sub_struct, sub_mfe = RNA.fold(sub_seq)
                window_mfes.append(sub_mfe)
            
            # Find worst (most negative) MFE
            worst_mfe = min(window_mfes) if window_mfes else mfe
            
            # Calculate fraction of overly stable windows (< -10 kcal/mol)
            stable_threshold = -10.0
            stable_count = sum(1 for m in window_mfes if m < stable_threshold)
            fraction_stable = stable_count / len(window_mfes) if window_mfes else 0
            
            return worst_mfe, fraction_stable
            
        except Exception as e:
            print(f"ViennaRNA error: {e}")
    
    # Simplified folding energy calculation (when ViennaRNA not available)
    gc_content = calculate_gc_content(analysis_region)
    
    # More GC = more stable (more negative energy)
    estimated_mfe = -0.5 * gc_content - 5.0
    
    # Check for potential hairpins (simple palindrome detection)
    # Very crude, sliding window search for palindromes of length 10, penalized by -2 kcal/mol each
    hairpin_penalty = 0
    for i in range(len(analysis_region) - 10):
        segment = analysis_region[i:i+10]
        rev_comp = segment[::-1].translate(str.maketrans('AUGC', 'UACG'))
        if segment == rev_comp:
            hairpin_penalty -= 2
    
    worst_mfe = estimated_mfe + hairpin_penalty
    fraction_stable = 0.3 if gc_content > 60 else 0.1
    
    return worst_mfe, fraction_stable


def calculate_accessibility_score(sequence: str, start_region_size: int = 30) -> float:
    """
    Calculate accessibility score for ribosome binding (simplified) based on 5' folding energy.
    
    Args:
        sequence: DNA sequence
        start_region_size: 5' window size
    
    Returns:
        Accessibility score (0-1, higher is better)
    """
    start_region = sequence[:min(start_region_size, len(sequence))]
    
    if VIENNA_AVAILABLE:
        try:
            rna_seq = start_region.replace('T', 'U')
            structure, mfe = RNA.fold(rna_seq)
            
            # Less stable structure = more accessible
            # Convert MFE to accessibility score
            # Logistic sigmoid transformation onto (0,1). Temperature factor of 5 to scale decision window around 0.
            accessibility = 1.0 / (1.0 + math.exp(-mfe / 5.0))
            return accessibility
            
        except:
            pass
    
    # Simplified calculation
    # Crude, fallback never delivers a value below 0.5 - as opposed to ViennaRNA
    gc_content = calculate_gc_content(start_region)
    # Lower GC in start region = more accessible
    accessibility = 1.0 - (gc_content / 100.0) * 0.5
    return max(0.0, min(1.0, accessibility))


def count_repetitive_sequences(sequence: str, homopolymer_threshold: int = 5,
                              dinuc_threshold: int = 6,
                              per_bp_scale: int = 1000) -> float:
    """
    Count repetitive sequences with smooth penalty.
    
    Args:
        sequence: DNA sequence
        homopolymer_threshold: Minimum length for homopolymer penalty
        dinuc_threshold: Minimum length for dinucleotide repeat penalty
        per_bp_scale: Scale penalties to argument (default → per-kb in case of call w/o specification). Normalization is always applied.
    
    Returns:
        Smooth penalty score (higher is worse)
    """
    penalty = 0.0
    
    # Check for homopolymers
    for nucleotide in 'ATGC':
        for match in re.finditer(rf'{nucleotide}+', sequence):
            length = len(match.group())
            if length >= homopolymer_threshold:
                # Smooth penalty that increases with length
                penalty += (length - homopolymer_threshold + 1) * 0.5
    
    # Check for dinucleotide repeats (maximal runs of any XY repeated as (XY)+, X != Y)
    i = 0
    n = len(sequence)
    while i <= n - 2:
        a = sequence[i]
        b = sequence[i + 1]
        # Skip homopolymer-like pairs; homopolymers are handled above
        if a == b:
            i += 1
            continue

        # Extend maximal (ab)(ab)(ab)... run
        j = i + 2
        ab = sequence[i:i+2]
        while j <= n - 2 and sequence[j:j+2] == ab:
            j += 2

        run_len = j - i  # dinuc run length in bases
        if run_len >= dinuc_threshold:
            # Base penalty at threshold per dinucleotide + 0.3 per additional dinucleotide unit
            excess_units = (run_len - dinuc_threshold) // 2
            penalty += 0.3 * (1 + excess_units)

        # Step to the boundary (j-1) so we don't skip a new dinuc that could start at j-1:j+1.
        # (j-2,j-1) is ab, (j,j+1) is not. But (j-1,j) could start a new dinuc.
        i = j - 1
    
    # Normalize to a per_bp_scale so scores are length-comparable
    penalty = penalty * (per_bp_scale / len(sequence))
    return penalty


# Sliding window repeat penalty function
def repeat_penalty_windowed(
    sequence: str,
    window: int = 300,
    step: int = 150,
    threshold: float = 0.5,
) -> Tuple[float, float, float]:
    """
    Compute repeat penalties in sliding windows to capture local hotspots and provide
    length-independent metrics. This avoids dilution of local hotspots and weight bias for long sequences.

    Args:
        sequence: DNA sequence
        window: Window size (default 300 nt)
        step: Sliding window step (default 150 nt; overlapping windows)
        threshold: Threshold on the per-window penalty used to compute the fraction of "bad" windows
        (Per-window penalties are normalized to the window length; scores are comparable across windows.)

    Returns:
        Tuple of (mean_per_window, max_per_window, frac_windows_above_threshold)
        where each per-window penalty is calculated by count_repetitive_sequences().
    """
    n = len(sequence)

    # If sequence is shorter than the window, evaluate once on the whole sequence
    vals: List[float] = []
    if n <= window:
        vals.append(
            count_repetitive_sequences(
                sequence,
                per_bp_scale = n,   # Scale to seq length if shorter than window to avoid penalty inflation
            )
        )
    else:
        for i in range(0, n - window + 1, step):
            sub = sequence[i:i + window]
            vals.append(
                count_repetitive_sequences(
                    sub,
                    per_bp_scale = window,
                )
            )

    assert len(vals) > 0, 'No repeat penalty windows evaluated, this should not happen. Good job.'
    mean_pen = float(sum(vals) / len(vals))
    max_pen = float(max(vals))
    frac_bad = float(sum(1 for v in vals if v > threshold) / len(vals))

    return mean_pen, max_pen, frac_bad


def count_cryptic_splice_sites(sequence: str, is_eukaryote: bool) -> int:
    """
    Count potential cryptic splice sites (lightweight heuristic, very rudimentary).
    
    Args:
        sequence: DNA sequence
        is_eukaryote: Eukaryotic host flag determined from host seletion dropdown
    
    Returns:
        Count of potential splice sites
    """
    # No splicing in prokaryotes
    if not is_eukaryote:
        return 0
    
    count = 0
    
    # Donor-like, GT then [A or G] then AG
    count += len(re.findall(r'GT[AG]AG', sequence))
    
    # Acceptor-like
    # Polypyrimidine tract (C/T), at least 5 bases long
	# Spacer up to 10 nt
	# AG = the canonical 3′ acceptor dinucleotide at intron end
    count += len(re.findall(r'[CT]{5,}[ATGC]{0,10}AG', sequence))
    
    return count


def check_internal_start_codons(sequence: str, region_size: int = 90) -> int:
    """
    Check for internal ATG codons in the 5' region.
    
    Args:
        sequence: DNA sequence
        region_size: Size of 5' region to check
    
    Returns:
        Count of internal ATG codons
    """
    check_region = sequence[3:min(region_size, len(sequence))]
    
    count = 0
    for i in range(0, len(check_region) - 2, 3):
        if check_region[i:i+3] == 'ATG':
            count += 1
    
    return count


def translate_dna_to_protein(sequence: str) -> str:
    """
    Translate DNA sequence to protein sequence, used for validating that the optimized sequence is synonymous with the input.
    
    Args:
        sequence: DNA sequence
    
    Returns:
        Protein sequence in single letter code
    """
    protein = []
    for i in range(0, len(sequence), 3):
        codon = sequence[i:i+3]
        aa = GENETIC_CODE.get(codon, 'X')
        if aa == '*':
            break  # Stop at first stop codon
        protein.append(aa)
    
    return ''.join(protein)


def back_translate_protein(protein_seq: str, host: str = 'hsapiens') -> str:
    """
    Back-translate the input protein sequence to DNA using host codon preferences.

    This deterministic version picks the highest-weight codon for each AA according to the
    host's codon usage table (relative adaptiveness).

    Args:
        protein_seq: Protein sequence in single-letter code. Translation stops at '*' if present.
        host: Host organism for codon selection (used to load the codon usage table).

    Returns:
        DNA sequence composed of maximum-weight codons for each AA.

    Raises:
        ValueError: If protein_seq contains an unknown amino-acid code.
    """
    codon_table = load_codon_table(host)

    dna_sequence: List[str] = []

    for aa in protein_seq:
        # Stop codon → terminate CDS construction
        if aa == '*':
            break
        # Validate amino acid code
        if aa not in AA_TO_CODONS:
            raise ValueError(f"Unknown amino acid code: {aa!r}")
        possible_codons = AA_TO_CODONS[aa]
        # Pick the best codon deterministically. Tie-break lexicographically for stability.
        best_codon = max(possible_codons, key = lambda c: (codon_table.get(c, 0.0), c))
        dna_sequence.append(best_codon)

    return ''.join(dna_sequence)


def get_synonymous_codons(codon: str) -> List[str]:
    """
    Get all synonymous codons for a given codon.
    
    Args:
        codon: Input codon
    
    Returns:
        List of synonymous codons (including input)
    """
    aa = GENETIC_CODE.get(codon)
    if not aa:
        return [codon]
    
    return AA_TO_CODONS[aa]


# Numba-accelerated helper functions
if NUMBA_AVAILABLE:
    @nb.jit(nopython = True)
    def _calculate_gc_content_numba(seq_array: np.ndarray) -> int:
        """Numba-accelerated GC counting."""
        gc_count = 0
        for base in seq_array:
            if base == ord('G') or base == ord('C'):
                gc_count += 1
        return gc_count
    
    @nb.jit(nopython = True)
    def _calculate_cai_numba(codon_indices: np.ndarray, codon_scores: np.ndarray) -> float:
        """Numba-accelerated CAI calculation."""
        log_sum = 0.0
        count = 0
        
        for idx in codon_indices:
            score = codon_scores[idx]
            if score > 0:
                log_sum += np.log(score)
                count += 1
        
        if count == 0:
            return 0.0
        
        cai = np.exp(log_sum / count)
        return cai
    
    @nb.jit(nopython = True)
    def _count_motif_numba(seq_array: np.ndarray, motif_array: np.ndarray) -> int:
        """Numba-accelerated motif counting."""
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