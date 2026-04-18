"""
Local repair: windowed scan-and-fix for sequence pathologies.

Slides a window across the preoptimization output, detects problems (homopolymers,
dinucleotide repeats, unwanted motifs, cryptic splice sites, local GC drift),
and fixes them via synonymous codon substitutions that minimally disrupt the
preoptimization design intent.

Called by: main.ipynb (hybrid pipeline)
Depends on: utils.py
"""

from typing import Dict, List, Tuple, Optional

from .utils import (
    load_codon_table, calculate_gc_content,
    count_repetitive_sequences, count_unwanted_motifs,
    count_cryptic_splice_sites,
    translate_dna_to_protein,
    get_target_gc,
    GENETIC_CODE, AA_TO_CODONS,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _problem_score(
    window_seq: str,
    is_eukaryote: bool,
    target_gc: float,
    gc_tolerance: float,
    motifs: List[str],
) -> float:
    """Total problem score for a window subsequence.  Higher = worse."""
    score = 0.0

    # Homopolymer + dinucleotide repeats (length-normalised)
    score += count_repetitive_sequences(window_seq, per_bp_scale = len(window_seq))

    # Unwanted motifs (polyA signals checked automatically inside)
    score += count_unwanted_motifs(window_seq, motifs) * 2.0

    # Cryptic splice sites (0 for prokaryotes)
    score += count_cryptic_splice_sites(window_seq, is_eukaryote) * 1.5

    # Local GC deviation beyond tolerance
    gc = calculate_gc_content(window_seq)
    gc_dev = abs(gc - target_gc)
    if gc_dev > gc_tolerance:
        score += (gc_dev - gc_tolerance) * 0.1

    return score


def _disruption_cost(
    preoptimization_codon: str,
    candidate: str,
    codon_table: Dict[str, float],
) -> float:
    """
    Penalty for replacing the preoptimization codon with *candidate*.
    Zero when they are the same codon; grows with speed-tier distance.
    """
    if preoptimization_codon == candidate:
        return 0.0

    orig_w = codon_table.get(preoptimization_codon, 0.0)
    cand_w = codon_table.get(candidate, 0.0)

    # Base cost for any change
    penalty = 0.3
    # Proportional to adaptiveness distance
    penalty += abs(orig_w - cand_w) * 0.7
    # Same wobble base is less disruptive
    if preoptimization_codon[2] == candidate[2]:
        penalty *= 0.5

    return penalty


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def local_repair(
    sequence: str,
    host: str,
    is_eukaryote: bool,
    preoptimization_sequence: str,
    target_gc: Optional[float] = None,
    window_nt: int = 42,
    max_subs_per_window: int = 2,
    motifs: Optional[List[str]] = None,
    gc_tolerance: float = 5.0,
) -> Tuple[str, List[dict]]:
    """
    Windowed local repair of sequence pathologies.

    Scans the sequence in overlapping windows.  At each window that contains
    a problem, tries every single-codon synonymous substitution, picks the
    one that yields the best net improvement (problem reduction minus
    preoptimization disruption), and applies it.  Repeats up to *max_subs_per_window* times
    per window position before advancing.

    After the pathology pass, a GC-balancing pass adjusts codons (within the
    same speed tier) to nudge overall GC toward the target.

    Args:
        sequence:            Sequence to repair (normally == preoptimization_sequence).
        host:                Target host organism key.
        is_eukaryote:        Eukaryote flag.
        preoptimization_sequence: Original preoptimization output used as disruption reference.
        target_gc:           Target GC% (host default when *None*).
        window_nt:           Window width in nt (rounded down to codon boundary).
        max_subs_per_window: Max substitutions per window before advancing.
        motifs:              Extra motifs to avoid (on top of default polyA signals).
        gc_tolerance:        GC deviation (%) tolerance before triggering fixes.

    Returns:
        (repaired_sequence, changelog)
        *changelog* is a list of dicts recording every substitution.
    """
    # load_codon_table is LRU-cached, so this is free on repeated calls.
    codon_table = load_codon_table(host)
    if target_gc is None:
        target_gc = get_target_gc(host)
    if motifs is None:
        motifs = []

    # Work on a mutable list of codons
    codons = [sequence[i:i + 3] for i in range(0, len(sequence), 3)]
    preoptimization_codons = [
        preoptimization_sequence[i:i + 3]
        for i in range(0, len(preoptimization_sequence), 3)
    ]
    n_codons = len(codons)

    # Codon-align window; step ~1/3 of window for heavy overlap
    win_codons = max(3, window_nt // 3)
    win_codons = min(win_codons, n_codons)
    step = max(1, win_codons // 3)

    changelog: List[dict] = []
    PROBLEM_THRESHOLD = 0.3

    # ------------------------------------------------------------------
    # Pass 1: pathology fixes
    # ------------------------------------------------------------------
    for win_start in range(0, max(1, n_codons - win_codons + 1), step):
        win_end = min(win_start + win_codons, n_codons)

        for _sub_round in range(max_subs_per_window):
            win_seq = ''.join(codons[win_start:win_end])
            current_score = _problem_score(
                win_seq, is_eukaryote, target_gc, gc_tolerance, motifs,
            )
            if current_score < PROBLEM_THRESHOLD:
                break

            # Find the single best substitution in this window
            best_net = 0.0
            best_ci: Optional[int] = None
            best_new: Optional[str] = None

            for ci in range(win_start, win_end):
                if ci == 0:
                    continue  # preserve start codon
                codon = codons[ci]
                aa = GENETIC_CODE.get(codon)
                if aa is None or aa == '*':
                    continue
                syns = AA_TO_CODONS[aa]
                if len(syns) <= 1:
                    continue

                for alt in syns:
                    if alt == codon:
                        continue
                    # Tentative substitution
                    codons[ci] = alt
                    trial_seq = ''.join(codons[win_start:win_end])
                    trial_score = _problem_score(
                        trial_seq, is_eukaryote, target_gc, gc_tolerance, motifs,
                    )
                    codons[ci] = codon  # revert

                    improvement = current_score - trial_score
                    disruption = _disruption_cost(
                        preoptimization_codons[ci], alt, codon_table,
                    )
                    net = improvement - disruption

                    if net > best_net:
                        best_net = net
                        best_ci = ci
                        best_new = alt

            if best_ci is None:
                break  # no beneficial substitution

            # Apply and log
            old_codon = codons[best_ci]
            codons[best_ci] = best_new

            # Determine dominant reason by comparing sub-scores
            win_before = ''.join(
                codons[win_start:best_ci]
                + [old_codon]
                + codons[best_ci + 1:win_end]
            )
            win_after = ''.join(codons[win_start:win_end])
            reasons: List[str] = []
            if count_repetitive_sequences(win_after, per_bp_scale = len(win_after)) < \
               count_repetitive_sequences(win_before, per_bp_scale = len(win_before)):
                reasons.append('repeat')
            if count_unwanted_motifs(win_after, motifs) < \
               count_unwanted_motifs(win_before, motifs):
                reasons.append('motif')
            if count_cryptic_splice_sites(win_after, is_eukaryote) < \
               count_cryptic_splice_sites(win_before, is_eukaryote):
                reasons.append('splice')
            gc_before = abs(calculate_gc_content(win_before) - target_gc)
            gc_after = abs(calculate_gc_content(win_after) - target_gc)
            if gc_after < gc_before - 0.5:
                reasons.append('GC')

            changelog.append({
                'position': best_ci + 1,       # 1-indexed codon
                'nt_position': best_ci * 3 + 1, # 1-indexed nucleotide
                'old_codon': old_codon,
                'new_codon': best_new,
                'amino_acid': GENETIC_CODE.get(old_codon, '?'),
                'reason': '+'.join(reasons) if reasons else 'repair',
            })

    # ------------------------------------------------------------------
    # Pass 2: GC balancing
    # ------------------------------------------------------------------
    repaired_seq = ''.join(codons)
    gc = calculate_gc_content(repaired_seq)

    if abs(gc - target_gc) > gc_tolerance:
        need_more_gc = gc < target_gc

        # Collect candidates: (disruption, codon_idx, candidate, current)
        gc_candidates: List[Tuple[float, int, str, str]] = []

        for ci in range(1, n_codons):  # skip start codon
            codon = codons[ci]
            aa = GENETIC_CODE.get(codon)
            if aa is None or aa == '*':
                continue

            for alt in AA_TO_CODONS[aa]:
                if alt == codon:
                    continue

                old_gc_count = sum(1 for nt in codon if nt in 'GC')
                new_gc_count = sum(1 for nt in alt if nt in 'GC')

                if need_more_gc and new_gc_count <= old_gc_count:
                    continue
                if not need_more_gc and new_gc_count >= old_gc_count:
                    continue

                # Same speed tier: within 0.15 of the preoptimization codon's adaptiveness
                preopt_w = codon_table.get(preoptimization_codons[ci], 0.0)
                alt_w = codon_table.get(alt, 0.0)
                if abs(preopt_w - alt_w) > 0.15:
                    continue

                disruption = _disruption_cost(preoptimization_codons[ci], alt, codon_table)
                gc_candidates.append((disruption, ci, alt, codon))

        # Apply least-disruptive swaps until GC is within tolerance
        gc_candidates.sort()
        for disruption, ci, alt, expected_current in gc_candidates:
            if codons[ci] != expected_current:
                continue  # stale (changed in pathology pass)

            old_codon = codons[ci]
            codons[ci] = alt
            changelog.append({
                'position': ci + 1,
                'nt_position': ci * 3 + 1,
                'old_codon': old_codon,
                'new_codon': alt,
                'amino_acid': GENETIC_CODE.get(old_codon, '?'),
                'reason': 'GC balance',
            })

            gc = calculate_gc_content(''.join(codons))
            if abs(gc - target_gc) <= gc_tolerance:
                break

    repaired = ''.join(codons)

    # Safety: verify protein identity
    original_protein = translate_dna_to_protein(preoptimization_sequence)
    repaired_protein = translate_dna_to_protein(repaired)
    assert repaired[:3] == preoptimization_sequence[:3], \
        'Local repair changed the start codon — this is a bug.'
    assert repaired_protein == original_protein, \
        'Local repair changed the protein sequence — this is a bug.'

    return repaired, changelog
