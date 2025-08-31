"""
Strategies for initial codon optimization. The output of this module is
then fed into the genetic algorithm.

Available strategies:
- Maximum CAI (MC): most frequent host codon per AA.
- Percentile matching (PM): map input codon usage percentiles to host.
- FlowRamp (FR): gentle 5' translational ramp + simple repeat avoidance.
"""

from typing import Dict, Tuple, Optional
from collections import defaultdict

# FlowRamp defaults (tunable)
FR_PAIR_WEIGHT = 1.5
FR_RANK_PENALTY = 2.0
FR_WOBBLE_BONUS = 0.5
FR_WOBBLE_PENALTY = 0.3
FR_WOBBLE_EARLY_LEN = 12

FR_SEARCH_WINDOW = 3           # ranks around target to consider
FR_RAMP_MIN = 8
FR_RAMP_MAX = 30
FR_RAMP_SCALE_DIV = 4          # ~ first 25% + offset
FR_RAMP_OFFSET = 6

FR_HOMOPOLYMER_THRESHOLD = 5   # e.g., AAAAA
FR_ALT_DINUC_REPEATS = 3       # e.g., ATATAT/CGCGCG

# FlowRamp: junction 4-mer heuristics (last two of c1 + first two of c2)
# - Strongly problematic (hard homopolymers at the junction)
FR_JUNCTION_STRONG_BAD_4MERS = {
    "GGGG", "CCCC",
}

# - Moderately problematic: slippery seeds, short homopolymers, alternating runs, palindromes
FR_JUNCTION_MOD_BAD_4MERS = {
    # Slippery-like seeds (include mirrors of existing ones)
    "GAAA", "TTTC", "AAAG", "CTTT",
    # Short homopolymers and simple repeats
    "AAAA", "TTTT", "ATAT", "TATA", "CGCG", "GCGC",
    # Palindromic 4-mers (hairpin seeds, also includes CpG variants)
    "AGCT", "TCGA", "TGCA", "ACGT", "GTAC",
}

# Host-specific junction patterns (light penalties to avoid over-filtering)
FR_JUNCTION_EUK_BAD_4MERS = {
    # Donor-like seeds across junctions
    "AGGT", "CAGG", "GTAG",
}

FR_JUNCTION_PROK_BAD_4MERS = {
    # Shine–Dalgarno-like seeds within CDS
    "AGGA", "GGAG",
}

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
        if aa != 'X' and aa != '*':
            aa_codon_counts[aa][codon] += 1
            codon_positions[codon].append(i)
    
    # Create percentile-based mapping
    aa_codon_mapping = {}
    
    for aa, codon_counts in aa_codon_counts.items():
        # Sort CDS codons by frequency with consistent tiebreaking
        cds_codons_sorted = sorted(
            codon_counts.items(),
            key = lambda x: (-x[1], min(codon_positions[x[0]]))  # Count desc, then first position
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
        host_codon_freqs.sort(key = lambda x: (-x[1], x[0]))  # Freq desc, then alphabetical
        
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
                    key = lambda x: abs(x[1] - cds_percentile)
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


def flowramp_optimization(
    sequence: str,
    host: str,
    is_eukaryote: bool,
    *,
    pair_weight: float = FR_PAIR_WEIGHT,
    rank_penalty: float = FR_RANK_PENALTY,
    wobble_bonus: float = FR_WOBBLE_BONUS,
    wobble_penalty: float = FR_WOBBLE_PENALTY,
    wobble_early_len: int = FR_WOBBLE_EARLY_LEN,
    search_window: int = FR_SEARCH_WINDOW,
    ramp_min: int = FR_RAMP_MIN,
    ramp_max: int = FR_RAMP_MAX,
    ramp_scale_div: int = FR_RAMP_SCALE_DIV,
    ramp_offset: int = FR_RAMP_OFFSET,
    homopolymer_threshold: int = FR_HOMOPOLYMER_THRESHOLD,
    alt_dinuc_repeats: int = FR_ALT_DINUC_REPEATS,
    # Feature toggles (auto-inferred when None)
    avoid_splice: Optional[bool] = None,
    avoid_sd: Optional[bool] = None,
    avoid_cpg: Optional[bool] = None,
    dam_dcm_sensitive: Optional[bool] = None,
) -> str:
    """
    FlowRamp: Codon optimization with a gentle 5' translational ramp and minimal interference.

    - First N codons form a ramp: choose lower→mid adaptiveness codons to reduce early ribosome pile-ups.
    - After the ramp, prefer the host's best codons for throughput (higher CAI).
    - While building, avoid simple pathologies: long homopolymers and alternating AT/CG dinucleotide runs.

    Returns:
        Harmonized DNA sequence
    """
    codon_table = load_codon_table(host)

    # Infer broad host features and allow overrides
    h = host.lower()

    def _is_vertebrate(host_lower: str) -> bool:
        tokens = (
            'hsapiens', 'mmusculus', 'rnorvegicus', 'drerio', 'danio', 'ggallus',
            'mmulatta', 'ptroglodytes', 'btaurus', 'cfamiliaris', 'xlaevis', 'xtropicalis'
        )
        return any(t in host_lower for t in tokens)

    if avoid_splice is None:
        avoid_splice = bool(is_eukaryote)
    if avoid_sd is None:
        avoid_sd = not bool(is_eukaryote)
    if avoid_cpg is None:
        avoid_cpg = bool(is_eukaryote) and _is_vertebrate(h)
    if dam_dcm_sensitive is None:
        dam_dcm_sensitive = (h == 'ecoli')

    def get_codon_pair_score(codon1: str, codon2: str) -> float:
        """Heuristic codon-pair preference score (DNA alphabet)."""
        cai1 = codon_table.get(codon1, 0.1)
        cai2 = codon_table.get(codon2, 0.1)

        # Base: rare-rare discouraged, common-common encouraged, transitions slightly encouraged
        if cai1 < 0.3 and cai2 < 0.3:
            base = -2.0
        elif cai1 > 0.7 and cai2 > 0.7:
            base = 1.0
        elif (cai1 > 0.7 and cai2 < 0.3) or (cai1 < 0.3 and cai2 > 0.7):
            base = 0.5
        else:
            base = 0.0

        # Discourage exact codon repeats (frameshift/slippage concerns)
        if codon1 == codon2:
            base -= 0.5

        # Avoid problematic junctions (last 2 + first 2)
        junction = codon1[1:] + codon2[:2]
        if junction in FR_JUNCTION_STRONG_BAD_4MERS:
            base -= 1.0
        elif junction in FR_JUNCTION_MOD_BAD_4MERS:
            base -= 0.5

        # Host-group tweaks (feature-flag driven)
        if avoid_cpg:
            # CpG at the boundary (vertebrate-focused)
            if codon1[2] == 'C' and codon2[0] == 'G':
                base -= 0.8
        if avoid_splice:
            # Eukaryote splice donor-like seeds across junctions
            if junction in FR_JUNCTION_EUK_BAD_4MERS:
                base -= 0.4
        if avoid_sd:
            # Prokaryotic SD-like seeds inside coding regions
            if junction in FR_JUNCTION_PROK_BAD_4MERS:
                base -= 0.4
        # Host-specific nudges remain precise where we have data
        if h == "ecoli":
            if (codon1, codon2) in (("GAA", "GAA"), ("CTG", "CTG")):
                base += 0.5

        return base

    def sorted_synonyms_by_host(aa: str):
        syns = AA_TO_CODONS.get(aa, [])
        return sorted(syns, key=lambda c: (-codon_table.get(c, 0.0), c))  # stable: by score desc, then lexicographic

    def would_create_homopolymer(prefix: str, candidate: str, threshold: int = FR_HOMOPOLYMER_THRESHOLD) -> bool:
        s = prefix + candidate
        if not s:
            return False
        tail = s[-1]
        run = 0
        for ch in reversed(s):
            if ch == tail:
                run += 1
                if run >= threshold:
                    return True
            else:
                break
        return False

    def is_alt_dinuc_run(prefix: str, candidate: str, repeats: int = FR_ALT_DINUC_REPEATS) -> bool:
        s = prefix + candidate
        if len(s) < 2 * repeats:
            return False
        tail = s[-2 * repeats:]
        kmer = tail[:2]
        return kmer * repeats == tail and kmer[0] != kmer[1]

    n_codons = len(sequence) // 3
    ramp_len = max(ramp_min, min(ramp_max, n_codons // max(1, ramp_scale_div) + ramp_offset))

    optimized = []

    for i in range(0, len(sequence), 3):
        codon = sequence[i:i + 3]
        aa = GENETIC_CODE.get(codon, 'X')

        # Preserve unknowns and stops verbatim
        if aa == 'X' or aa == '*':
            optimized.append(codon)
            continue

        syns = sorted_synonyms_by_host(aa)
        if not syns:
            optimized.append(codon)
            continue

        pos = i // 3
        if pos < ramp_len and len(syns) > 1:
            worst = len(syns) - 1
            late = 1 if len(syns) > 1 else 0
            t = pos / max(1, ramp_len - 1)
            target_rank = int(round(worst * (1 - t) + late * t))
        else:
            target_rank = 0

        # Build and score candidates around target rank (±search_window), with pair awareness
        search = min(search_window, len(syns))
        prefix = ''.join(optimized)
        prefer_at_wobble = pos < min(wobble_early_len, ramp_len)
        candidates = []

        for r in range(max(0, target_rank - search), min(len(syns), target_rank + search + 1)):
            cand = syns[r]

            # Early wobble bias: prefer A/T wobble for first ~12 codons of the ramp
            wobble_term = 0.0
            if prefer_at_wobble:
                wobble_term = (wobble_bonus if cand[2] in 'AT' else -wobble_penalty)

            # Skip if it immediately creates obvious pathologies
            if would_create_homopolymer(prefix, cand) or is_alt_dinuc_run(prefix, cand):
                continue

            # Score: closeness to target rank + pair score with previous codon
            score = 0.0
            score -= abs(r - target_rank) * rank_penalty
            if optimized:
                score += pair_weight * get_codon_pair_score(optimized[-1], cand)
            score += wobble_term

            candidates.append((score, cand))

        # Pick best-scoring candidate or fall back to target rank
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            chosen = candidates[0][1]
        else:
            chosen = syns[target_rank]

        optimized.append(chosen)

    return ''.join(optimized)


def optimize_codons(sequence: str, host: str, is_eukaryote: bool, method: str = 'simple') -> str:
    """
    Main entry point for codon optimization.
    
    Args:
        sequence: Input DNA sequence
        host: Host organism
        is_eukaryote: Whether the selected host is eukaryotic
        method: Optimization method ('simple', 'percentile', 'flowramp')
    
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
    elif 'flowramp' in method_lower:
        optimized = flowramp_optimization(sequence, host, is_eukaryote)
    else:
        raise ValueError(f"Unknown optimization method: {method}")
    
    # Verify protein is preserved
    optimized_protein = translate_dna_to_protein(optimized)
    if original_protein != optimized_protein:
        raise ValueError("Protein sequence changed during optimization!")
    
    return optimized
