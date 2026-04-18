"""
Preoptimization strategies for initial deterministic codon redesign.
The output of this module is fed into the genetic algorithm or local repair.

Available strategies:
- Maximum CAI ("simple"): most-frequent host codon per amino acid.
- Percentile matching ("percentile"): map source codon-usage percentiles
  onto the target host (with optional source-host-informed mode).
- Structure-aware ("structure"): beam search over synonymous codons scoring
  host adaptiveness and local mRNA folding heuristics jointly. It gives the
  downstream GA a deterministic structure-aware seed without claiming to solve
  global MFE optimization.
"""

from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from .utils import (
    load_codon_table,
    GENETIC_CODE,
    AA_TO_CODONS,
    translate_dna_to_protein,
    calculate_window_mfe,
    VIENNA_AVAILABLE,
)


# ---------------------------------------------------------------------------
# Strategy A: Maximum CAI
# ---------------------------------------------------------------------------

def simple_best_codon_optimization(sequence: str, host: str) -> str:
    """
    Replace each codon with the most host-preferred synonymous codon.

    This is the "max-CAI" baseline. It's what most commercial vendors default
    to and what reviewers will expect as a comparison point. On its own it
    depletes codon diversity and can disrupt co-translational folding, so
    prefer the GA or local-repair pipelines on top of it for real designs.
    """
    codon_table = load_codon_table(host)
    optimized = []

    for i in range(0, len(sequence), 3):
        codon = sequence[i:i + 3]
        if i == 0:
            # Preserve the literal start codon: GTG/TTG/CTG carry initiation
            # semantics in bacteria that translation-based validation misses.
            optimized.append(codon)
            continue
        aa = GENETIC_CODE.get(codon, 'X')
        if aa == 'X':
            optimized.append(codon)
            continue
        synonymous = AA_TO_CODONS.get(aa, [codon])
        best_codon = max(synonymous, key=lambda c: codon_table.get(c, 0.0))
        optimized.append(best_codon)

    return ''.join(optimized)


# ---------------------------------------------------------------------------
# Strategy B: Percentile-matched harmonization
# ---------------------------------------------------------------------------

def percentile_matching_optimization(
    sequence: str,
    host: str,
    source_host: Optional[str] = None,
) -> str:
    """
    Preserve the source codon-usage pattern while switching to the target host.

    Modes:
      - Source-agnostic (source_host=None): infer codon ranks from the input
        CDS itself. Works for natively expressed genes. Circular for short
        or synthetic inputs.
      - Source-host-informed (source_host set): rank codons by their
        frequency in the source organism's codon table. More reliable — this
        is the recommended default whenever the source organism is known.
    """
    codon_table = load_codon_table(host)

    # --- Source host-informed percentile matching ---
    if source_host is not None:
        source_table = load_codon_table(source_host)

        codon_mapping: Dict[str, str] = {}
        for aa, codons in AA_TO_CODONS.items():
            if aa == '*':
                continue
            if len(codons) == 1:
                codon_mapping[codons[0]] = codons[0]
                continue

            source_freqs = sorted(
                [(c, source_table.get(c, 0.0)) for c in codons],
                key=lambda x: (-x[1], x[0]),
            )
            total_source = sum(f for _, f in source_freqs)

            source_percentiles: Dict[str, float] = {}
            if total_source > 0:
                cumulative = 0.0
                for c, f in source_freqs:
                    source_percentiles[c] = (cumulative + f / 2) / total_source
                    cumulative += f
            else:
                for i, (c, _) in enumerate(source_freqs):
                    source_percentiles[c] = (i + 0.5) / len(source_freqs)

            target_freqs = sorted(
                [(c, codon_table.get(c, 0.0)) for c in codons],
                key=lambda x: (-x[1], x[0]),
            )
            total_target = sum(f for _, f in target_freqs)

            target_percentiles: List[Tuple[str, float]] = []
            if total_target > 0:
                cumulative = 0.0
                for c, f in target_freqs:
                    target_percentiles.append((c, (cumulative + f / 2) / total_target))
                    cumulative += f
            else:
                for i, (c, _) in enumerate(target_freqs):
                    target_percentiles.append((c, (i + 0.5) / len(target_freqs)))

            for src_codon in codons:
                src_pct = source_percentiles[src_codon]
                best_target = min(target_percentiles, key=lambda x: abs(x[1] - src_pct))
                codon_mapping[src_codon] = best_target[0]

        optimized = []
        for i in range(0, len(sequence), 3):
            codon = sequence[i:i + 3]
            if i == 0:
                optimized.append(codon)
                continue
            optimized.append(codon_mapping.get(codon, codon))
        return ''.join(optimized)

    # --- Source-agnostic percentile matching (CDS-internal ranking) ---
    aa_codon_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    codon_positions: Dict[str, List[int]] = defaultdict(list)

    for i in range(0, len(sequence), 3):
        codon = sequence[i:i + 3]
        aa = GENETIC_CODE.get(codon, 'X')
        if aa != 'X' and aa != '*':
            aa_codon_counts[aa][codon] += 1
            codon_positions[codon].append(i)

    aa_codon_mapping: Dict[Tuple[str, str], str] = {}

    for aa, codon_counts in aa_codon_counts.items():
        cds_codons_sorted = sorted(
            codon_counts.items(),
            key=lambda x: (-x[1], min(codon_positions[x[0]])),
        )

        total_count = sum(codon_counts.values())
        cds_percentiles: List[Tuple[str, float, int]] = []
        cumulative = 0
        for codon, count in cds_codons_sorted:
            percentile = (cumulative + count / 2) / total_count if total_count > 0 else 0.5
            cds_percentiles.append((codon, percentile, count))
            cumulative += count

        host_codons = AA_TO_CODONS.get(aa, [])
        host_codon_freqs = [(c, codon_table.get(c, 0.0)) for c in host_codons]

        if not host_codon_freqs:
            for codon, _, _ in cds_percentiles:
                aa_codon_mapping[(aa, codon)] = codon
            continue

        host_codon_freqs.sort(key=lambda x: (-x[1], x[0]))
        total_freq = sum(f for _, f in host_codon_freqs)
        if total_freq == 0:
            for i, (codon, _, _) in enumerate(cds_percentiles):
                host_idx = min(i, len(host_codon_freqs) - 1)
                aa_codon_mapping[(aa, codon)] = host_codon_freqs[host_idx][0]
        else:
            host_percentiles: List[Tuple[str, float]] = []
            cumulative_f = 0.0
            for codon, freq in host_codon_freqs:
                percentile = (cumulative_f + freq / 2) / total_freq
                host_percentiles.append((codon, percentile))
                cumulative_f += freq

            for cds_codon, cds_percentile, _ in cds_percentiles:
                best_host = min(host_percentiles, key=lambda x: abs(x[1] - cds_percentile))
                aa_codon_mapping[(aa, cds_codon)] = best_host[0]

    optimized = []
    for i in range(0, len(sequence), 3):
        codon = sequence[i:i + 3]
        if i == 0:
            optimized.append(codon)
            continue
        aa = GENETIC_CODE.get(codon, 'X')
        mapped_codon = aa_codon_mapping.get((aa, codon), codon)
        optimized.append(mapped_codon)
    return ''.join(optimized)


# ---------------------------------------------------------------------------
# Strategy C: Structure-aware heuristic (beam search)
# ---------------------------------------------------------------------------

def structure_optimized_beam_search(
    sequence: str,
    host: str,
    *,
    beam_width: int = 8,
    window_nt: int = 30,
    cai_weight: float = 1.0,
    mfe_weight: float = 0.15,
) -> str:
    """
    Beam-search over synonymous codons, jointly scoring host adaptiveness and
    local mRNA folding heuristics.

    At each codon position we extend the current beam with every synonymous
    codon for that amino acid, score each partial sequence by
        score = cai_weight * mean_log_weight
              - mfe_weight * (-local_mfe)
    where local_mfe is the minimum free energy of the last `window_nt` of the
    partial sequence (computed with ViennaRNA when available, or a GC-based
    heuristic fall-back). We keep the top `beam_width` partial sequences and
    recurse. The result is a structure-aware deterministic seed which the GA
    or local repair can refine further.

    Notes:
      - This is not LinearDesign: it doesn't parse the full Pareto lattice.
        It's a pragmatic approximation that fits inside the existing pipeline
        without introducing a large new dependency surface.
      - When ViennaRNA is unavailable, the MFE term uses a deterministic GC
        approximation so the search still runs — less accurate, but stable.

    Args:
        sequence:     Input DNA. Length must be a multiple of 3.
        host:         Target host for CAI scoring.
        beam_width:   Number of partial sequences kept between steps.
        window_nt:    Length of the 3' sliding window scored for MFE.
        cai_weight:   Weight on the log-CAI contribution to the score.
        mfe_weight:   Weight on the -MFE (avoid-stable-structure) term.

    Returns:
        Optimized DNA sequence (one of the top-scoring beams).
    """
    import math

    codon_table = load_codon_table(host)

    # Seed beam with an empty prefix
    # Each entry: (cumulative_score, partial_seq, log_weight_sum, codon_count)
    beam: List[Tuple[float, str, float, int]] = [(0.0, '', 0.0, 0)]

    def score_partial(partial: str, log_weight_sum: float, codon_count: int) -> float:
        # Geometric-mean log CAI (equivalent to log-CAI normalized by codons)
        mean_log_w = (log_weight_sum / codon_count) if codon_count else 0.0
        cai_term = cai_weight * mean_log_w
        # MFE term: penalise very negative MFE in the 3' sliding window
        if len(partial) >= window_nt:
            window = partial[-window_nt:]
            mfe = calculate_window_mfe(window)
        elif len(partial) >= 9:
            mfe = calculate_window_mfe(partial)
        else:
            mfe = 0.0
        structure_term = -mfe_weight * (-mfe)  # == mfe_weight * mfe (penalty on stability)
        return cai_term + structure_term

    for i in range(0, len(sequence), 3):
        codon = sequence[i:i + 3]
        aa = GENETIC_CODE.get(codon, 'X')

        if aa == 'X' or aa == '*':
            # Pass through untouched; every beam extends with the same codon.
            new_beam = []
            for _score, partial, lws, n in beam:
                extended = partial + codon
                new_beam.append((score_partial(extended, lws, n), extended, lws, n))
            beam = sorted(new_beam, key=lambda x: x[0], reverse=True)[:beam_width]
            continue

        candidates = [codon] if i == 0 else AA_TO_CODONS.get(aa, [codon])
        new_beam: List[Tuple[float, str, float, int]] = []

        for _score, partial, lws, n in beam:
            for cand in candidates:
                w = codon_table.get(cand, 0.0)
                if w <= 0.0:
                    # Skip zero-weight codons in CAI term; they'd corrupt the
                    # log sum. If every candidate is zero, we fall through to
                    # a neutral add below.
                    continue
                new_lws = lws + math.log(w)
                new_n = n + 1
                extended = partial + cand
                new_beam.append(
                    (score_partial(extended, new_lws, new_n), extended, new_lws, new_n)
                )

        if not new_beam:
            # Defensive fallback: keep the highest-weight synonym verbatim
            chosen = max(candidates, key=lambda c: codon_table.get(c, 0.0))
            for _score, partial, lws, n in beam:
                extended = partial + chosen
                new_beam.append((score_partial(extended, lws, n), extended, lws, n))

        beam = sorted(new_beam, key=lambda x: x[0], reverse=True)[:beam_width]

    return beam[0][1]


# ---------------------------------------------------------------------------
# Dispatch entry point
# ---------------------------------------------------------------------------

def optimize_codons(
    sequence: str,
    host: str,
    is_eukaryote: bool,
    method: str = 'simple',
    source_host: Optional[str] = None,
) -> str:
    """
    Main entry point for preoptimization codon redesign.

    Args:
        sequence:    Input DNA.
        host:        Target host.
        is_eukaryote: Whether the host is eukaryotic. Kept in the signature
                     for forward compatibility; current strategies don't use
                     it directly, but splice-aware post-processors do.
        method:      One of {'simple', 'percentile', 'structure'}.
        source_host: Optional source organism for percentile mode.

    Returns:
        Synonymously-optimized DNA.
    """
    # Validate that protein is preserved
    original_protein = translate_dna_to_protein(sequence)

    method_lower = method.lower()
    if 'simple' in method_lower or 'max_cai' in method_lower:
        optimized = simple_best_codon_optimization(sequence, host)
    elif 'percentile' in method_lower or 'harmon' in method_lower:
        optimized = percentile_matching_optimization(sequence, host, source_host=source_host)
    elif 'structure' in method_lower or 'mfe' in method_lower or 'beam' in method_lower:
        optimized = structure_optimized_beam_search(sequence, host)
    else:
        raise ValueError(
            f"Unknown optimization method: {method!r}. "
            "Expected one of: 'simple', 'percentile', 'structure'."
        )

    optimized_protein = translate_dna_to_protein(optimized)
    if original_protein != optimized_protein:
        raise ValueError("Protein sequence changed during optimization!")
    if sequence[:3] and optimized[:3] != sequence[:3]:
        raise ValueError("Start codon changed during optimization!")

    return optimized
