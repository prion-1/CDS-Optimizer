"""
Diversification masks for repeated protein-coding regions.

The mask text is user-facing: comma-separated amino-acid or nucleotide
sequences. Once validated, the optimizer uses coordinate ranges rather than
string matching so synonymous downstream edits cannot move or lose masks.
"""

from __future__ import annotations

import html
import itertools
import math
import re
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .utils import (
    AA_TO_CODONS,
    GENETIC_CODE,
    calculate_cai,
    load_codon_table,
    translate_dna_to_protein,
)


VALID_NUCS = set("ACGTU")
VALID_AAS = set("ACDEFGHIKLMNPQRSTVWY")

DEFAULT_DIVERSIFICATION_STRENGTH = 0.75
DEFAULT_MAX_EXACT_SHARED_NT = 12
DEFAULT_MIN_REGION_CAI_FRACTION = 0.40
DEFAULT_MIN_MASK_AA = 6

MASK_COLORS = (
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
)


@dataclass(frozen=True)
class MaskOccurrence:
    """One resolved occurrence of a diversification mask."""

    start_nt: int
    end_nt: int
    aa_start: int
    aa_end: int

    @property
    def length_nt(self) -> int:
        return self.end_nt - self.start_nt


@dataclass(frozen=True)
class DiversificationGroup:
    """All occurrences belonging to one comma-separated mask entry."""

    label: str
    source_pattern: str
    aa_sequence: str
    occurrences: Tuple[MaskOccurrence, ...]
    color: str
    min_possible_pairwise_identity: float
    max_pairwise_identity: float
    max_exact_shared_nt: int = DEFAULT_MAX_EXACT_SHARED_NT
    min_region_cai_fraction: float = DEFAULT_MIN_REGION_CAI_FRACTION
    baseline_region_cai: Tuple[float, ...] = ()

    @property
    def length_nt(self) -> int:
        return len(self.aa_sequence) * 3


@dataclass(frozen=True)
class DiversificationPlan:
    """Validated diversification mask set."""

    mask_type: str
    groups: Tuple[DiversificationGroup, ...]
    sequence_length_nt: int
    warnings: Tuple[str, ...] = ()

    @property
    def active(self) -> bool:
        return bool(self.groups)


@dataclass(frozen=True)
class DiversificationResult:
    """Sequence returned by the diversification pass."""

    sequence: str
    plan: DiversificationPlan
    metrics: Dict[str, Any]
    changes: Tuple[Dict[str, Any], ...]


def _clean_cds(sequence: str) -> str:
    return re.sub(r"[\s\d]+", "", sequence.upper()).replace("U", "T")


def _mask_entries(mask_text: str) -> Tuple[str, ...]:
    entries: List[str] = []
    seen: set[str] = set()
    for part in mask_text.split(","):
        entry = re.sub(r"\s+", "", part.upper())
        if not entry:
            continue
        if entry in seen:
            raise ValueError(f"Duplicate diversification mask entry: {entry}")
        seen.add(entry)
        entries.append(entry)
    return tuple(entries)


def _find_all(haystack: str, needle: str) -> Tuple[int, ...]:
    starts: List[int] = []
    start = 0
    while True:
        index = haystack.find(needle, start)
        if index == -1:
            break
        starts.append(index)
        start = index + 1
    return tuple(starts)


def _assert_non_overlapping(groups: Sequence[DiversificationGroup]) -> None:
    ranges: List[Tuple[int, int, str]] = []
    for group in groups:
        for occ in group.occurrences:
            ranges.append((occ.start_nt, occ.end_nt, group.label))
    ranges.sort()

    previous: Optional[Tuple[int, int, str]] = None
    for current in ranges:
        if previous is not None and current[0] < previous[1]:
            raise ValueError(
                "Diversification masks overlap: "
                f"{previous[2]} at nt {previous[0] + 1}-{previous[1]} and "
                f"{current[2]} at nt {current[0] + 1}-{current[1]}."
            )
        previous = current


def _codon_match_count(left: str, right: str) -> int:
    return sum(1 for a, b in zip(left, right) if a == b)


def mathematically_min_pairwise_identity(aa_sequence: str) -> float:
    """
    Lower bound for pairwise DNA identity imposed by codon degeneracy.

    This is the best possible two-copy aligned identity if expression and
    regulatory constraints are ignored. Single-codon amino acids force identity
    at their codon positions.
    """

    total_matches = 0
    total_nt = len(aa_sequence) * 3
    if total_nt == 0:
        return 1.0

    for aa in aa_sequence:
        codons = tuple(c for c in AA_TO_CODONS.get(aa, ()) if GENETIC_CODE.get(c) == aa)
        if not codons:
            raise ValueError(f"Cannot diversify unsupported amino acid {aa!r}.")
        if len(codons) == 1:
            total_matches += 3
            continue
        total_matches += min(
            _codon_match_count(left, right)
            for left in codons
            for right in codons
        )
    return total_matches / total_nt


def adaptive_max_pairwise_identity(
    aa_sequence: str,
    *,
    strength: float = DEFAULT_DIVERSIFICATION_STRENGTH,
) -> float:
    """
    Return the identity gate for a repeated coding region.

    `strength` is the fraction of mathematically possible diversification the
    optimizer should attempt. The returned threshold is never below the
    degeneracy-imposed minimum.
    """

    if not 0.0 <= strength <= 1.0:
        raise ValueError("Diversification strength must be in [0, 1].")
    lower_bound = mathematically_min_pairwise_identity(aa_sequence)
    return lower_bound + (1.0 - lower_bound) * (1.0 - strength)


def _build_group(
    *,
    label: str,
    source_pattern: str,
    aa_sequence: str,
    occurrences: Sequence[MaskOccurrence],
    color: str,
    strength: float,
    max_exact_shared_nt: int,
    min_region_cai_fraction: float,
) -> DiversificationGroup:
    min_identity = mathematically_min_pairwise_identity(aa_sequence)
    max_identity = adaptive_max_pairwise_identity(aa_sequence, strength=strength)
    return DiversificationGroup(
        label=label,
        source_pattern=source_pattern,
        aa_sequence=aa_sequence,
        occurrences=tuple(occurrences),
        color=color,
        min_possible_pairwise_identity=min_identity,
        max_pairwise_identity=max_identity,
        max_exact_shared_nt=max_exact_shared_nt,
        min_region_cai_fraction=min_region_cai_fraction,
    )


def _resolve_nucleotide_masks(
    entries: Sequence[str],
    cds_sequence: str,
    *,
    strength: float,
    max_exact_shared_nt: int,
    min_region_cai_fraction: float,
) -> DiversificationPlan:
    groups: List[DiversificationGroup] = []
    protein = translate_dna_to_protein(cds_sequence)

    for index, raw_entry in enumerate(entries):
        pattern = raw_entry.replace("U", "T")
        invalid = set(pattern) - set("ACGT")
        if invalid:
            raise ValueError(
                f"Nucleotide mask {raw_entry!r} contains invalid bases: "
                f"{', '.join(sorted(invalid))}."
            )
        if len(pattern) % 3 != 0:
            raise ValueError(
                f"Nucleotide mask {raw_entry!r} length is {len(pattern)} nt; "
                "nucleotide masks must be divisible by 3."
            )
        if len(pattern) // 3 < DEFAULT_MIN_MASK_AA:
            raise ValueError(
                f"Nucleotide mask {raw_entry!r} is too short; use at least "
                f"{DEFAULT_MIN_MASK_AA * 3} nt."
            )

        starts = _find_all(cds_sequence, pattern)
        if not starts:
            raise ValueError(f"Nucleotide mask {raw_entry!r} does not occur in the input CDS.")

        out_of_frame = [start + 1 for start in starts if start % 3 != 0]
        if out_of_frame:
            joined = ", ".join(str(pos) for pos in out_of_frame[:8])
            raise ValueError(
                f"Nucleotide mask {raw_entry!r} occurs out of frame at nt {joined}."
            )

        if len(starts) < 2:
            raise ValueError(
                f"Nucleotide mask {raw_entry!r} occurs {len(starts)} time(s); "
                "diversification masks must occur at least twice."
            )

        occurrences = tuple(
            MaskOccurrence(
                start_nt=start,
                end_nt=start + len(pattern),
                aa_start=start // 3,
                aa_end=(start + len(pattern)) // 3,
            )
            for start in starts
        )
        aa_start = occurrences[0].aa_start
        aa_end = occurrences[0].aa_end
        aa_sequence = protein[aa_start:aa_end]

        groups.append(_build_group(
            label=f"mask {index + 1}",
            source_pattern=pattern,
            aa_sequence=aa_sequence,
            occurrences=occurrences,
            color=MASK_COLORS[index % len(MASK_COLORS)],
            strength=strength,
            max_exact_shared_nt=max_exact_shared_nt,
            min_region_cai_fraction=min_region_cai_fraction,
        ))

    _assert_non_overlapping(groups)
    return DiversificationPlan(
        mask_type="nucleotide",
        groups=tuple(groups),
        sequence_length_nt=len(cds_sequence),
    )


def _resolve_aa_masks(
    entries: Sequence[str],
    cds_sequence: str,
    *,
    strength: float,
    max_exact_shared_nt: int,
    min_region_cai_fraction: float,
) -> DiversificationPlan:
    groups: List[DiversificationGroup] = []
    protein = translate_dna_to_protein(cds_sequence)

    for index, pattern in enumerate(entries):
        invalid = set(pattern) - VALID_AAS
        if invalid:
            raise ValueError(
                f"Amino-acid mask {pattern!r} contains invalid residues: "
                f"{', '.join(sorted(invalid))}."
            )
        if len(pattern) < DEFAULT_MIN_MASK_AA:
            raise ValueError(
                f"Amino-acid mask {pattern!r} is too short; use at least "
                f"{DEFAULT_MIN_MASK_AA} amino acids."
            )

        starts_aa = _find_all(protein, pattern)
        if not starts_aa:
            raise ValueError(f"Amino-acid mask {pattern!r} does not occur in the input protein.")
        if len(starts_aa) < 2:
            raise ValueError(
                f"Amino-acid mask {pattern!r} occurs {len(starts_aa)} time(s); "
                "diversification masks must occur at least twice."
            )

        occurrences = tuple(
            MaskOccurrence(
                start_nt=start * 3,
                end_nt=(start + len(pattern)) * 3,
                aa_start=start,
                aa_end=start + len(pattern),
            )
            for start in starts_aa
        )
        groups.append(_build_group(
            label=f"mask {index + 1}",
            source_pattern=pattern,
            aa_sequence=pattern,
            occurrences=occurrences,
            color=MASK_COLORS[index % len(MASK_COLORS)],
            strength=strength,
            max_exact_shared_nt=max_exact_shared_nt,
            min_region_cai_fraction=min_region_cai_fraction,
        ))

    _assert_non_overlapping(groups)
    return DiversificationPlan(
        mask_type="aa",
        groups=tuple(groups),
        sequence_length_nt=len(cds_sequence),
    )


def resolve_diversification_masks(
    mask_text: str,
    cds_sequence: str,
    *,
    strength: float = DEFAULT_DIVERSIFICATION_STRENGTH,
    max_exact_shared_nt: int = DEFAULT_MAX_EXACT_SHARED_NT,
    min_region_cai_fraction: float = DEFAULT_MIN_REGION_CAI_FRACTION,
) -> Optional[DiversificationPlan]:
    """
    Parse and validate user diversification masks against a forward CDS.

    A mask set is interpreted as nucleotide when every entry is a valid
    in-frame nucleotide mask. Otherwise it is interpreted as amino-acid masks.
    Mixed nucleotide/amino-acid mask classes are rejected by construction.
    """

    entries = _mask_entries(mask_text)
    if not entries:
        return None

    cds = _clean_cds(cds_sequence)
    if len(cds) % 3 != 0:
        raise ValueError("Input CDS must be frame-aligned before resolving masks.")

    all_nucleotide_like = all(set(entry) <= VALID_NUCS for entry in entries)
    all_aa_like = all(set(entry) <= VALID_AAS for entry in entries)
    warnings: List[str] = []
    errors: List[str] = []

    if all_nucleotide_like:
        try:
            plan = _resolve_nucleotide_masks(
                entries,
                cds,
                strength=strength,
                max_exact_shared_nt=max_exact_shared_nt,
                min_region_cai_fraction=min_region_cai_fraction,
            )
            if all_aa_like:
                warnings.append(
                    "Masks contain only A/C/G/T/U letters and were interpreted as nucleotide masks."
                )
            return replace(plan, warnings=tuple(warnings))
        except ValueError as exc:
            errors.append(str(exc))

    if all_aa_like:
        try:
            return _resolve_aa_masks(
                entries,
                cds,
                strength=strength,
                max_exact_shared_nt=max_exact_shared_nt,
                min_region_cai_fraction=min_region_cai_fraction,
            )
        except ValueError as exc:
            errors.append(str(exc))

    if not all_nucleotide_like and not all_aa_like:
        invalid = sorted(set("".join(entries)) - (VALID_AAS | VALID_NUCS))
        raise ValueError(
            "Diversification masks must be all amino-acid sequences or all "
            f"nucleotide sequences. Invalid characters: {', '.join(invalid)}."
        )

    raise ValueError(" ".join(errors) if errors else "Invalid diversification masks.")


def with_baseline_region_cai(
    plan: DiversificationPlan,
    sequence: str,
    host: str,
    *,
    codon_table: Optional[Dict[str, float]] = None,
) -> DiversificationPlan:
    """Attach per-occurrence CAI baselines from the current optimization seed."""

    if codon_table is None:
        codon_table = load_codon_table(host)

    groups: List[DiversificationGroup] = []
    for group in plan.groups:
        baselines = tuple(
            calculate_cai(sequence[occ.start_nt:occ.end_nt], codon_table)
            for occ in group.occurrences
        )
        groups.append(replace(group, baseline_region_cai=baselines))
    return replace(plan, groups=tuple(groups))


def _pairwise_identity(left: str, right: str) -> float:
    if len(left) != len(right):
        raise ValueError("Cannot compare mask occurrences with different lengths.")
    if not left:
        return 1.0
    return sum(1 for a, b in zip(left, right) if a == b) / len(left)


def longest_common_substring_length(left: str, right: str) -> int:
    """Length of the longest exact consecutive nucleotide string shared by two sequences."""

    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    best = 0
    for i, left_ch in enumerate(left, start=1):
        current = [0] * (len(right) + 1)
        for j, right_ch in enumerate(right, start=1):
            if left_ch == right_ch:
                current[j] = previous[j - 1] + 1
                if current[j] > best:
                    best = current[j]
        previous = current
    return best


def _group_metrics(
    sequence: str,
    group: DiversificationGroup,
    codon_table: Dict[str, float],
) -> Dict[str, Any]:
    regions = [sequence[occ.start_nt:occ.end_nt] for occ in group.occurrences]
    pair_identities: List[float] = []
    longest_exact_values: List[int] = []

    for left, right in itertools.combinations(regions, 2):
        pair_identities.append(_pairwise_identity(left, right))
        longest_exact_values.append(longest_common_substring_length(left, right))

    region_cais = tuple(calculate_cai(region, codon_table) for region in regions)
    if group.baseline_region_cai and len(group.baseline_region_cai) == len(region_cais):
        fractions = tuple(
            1.0 if baseline <= 0 else current / baseline
            for current, baseline in zip(region_cais, group.baseline_region_cai)
        )
    else:
        fractions = tuple(1.0 for _ in region_cais)

    max_identity = max(pair_identities) if pair_identities else 0.0
    longest_exact = max(longest_exact_values) if longest_exact_values else 0
    min_cai_fraction = min(fractions) if fractions else 1.0

    identity_excess = (
        0.0
        if group.max_pairwise_identity >= 1.0
        else max(0.0, (max_identity - group.max_pairwise_identity) / (1.0 - group.max_pairwise_identity))
    )
    exact_excess = max(
        0.0,
        (longest_exact - group.max_exact_shared_nt) / max(1, group.max_exact_shared_nt),
    )
    cai_deficit = max(
        0.0,
        (group.min_region_cai_fraction - min_cai_fraction)
        / max(group.min_region_cai_fraction, 1e-9),
    )
    penalty = min(1.0, identity_excess + exact_excess + cai_deficit)

    return {
        "label": group.label,
        "source_pattern": group.source_pattern,
        "occurrence_count": len(group.occurrences),
        "length_nt": group.length_nt,
        "min_possible_pairwise_identity": group.min_possible_pairwise_identity,
        "max_pairwise_identity_limit": group.max_pairwise_identity,
        "max_pairwise_identity": max_identity,
        "longest_exact_shared_nt": longest_exact,
        "max_exact_shared_nt_limit": group.max_exact_shared_nt,
        "region_cai": region_cais,
        "baseline_region_cai": group.baseline_region_cai,
        "region_cai_fraction": fractions,
        "min_region_cai_fraction": min_cai_fraction,
        "min_region_cai_fraction_limit": group.min_region_cai_fraction,
        "identity_excess": identity_excess,
        "exact_excess": exact_excess,
        "cai_deficit": cai_deficit,
        "diversification_penalty": penalty,
        "diversification_pass": (
            max_identity <= group.max_pairwise_identity + 1e-9
            and longest_exact <= group.max_exact_shared_nt
            and min_cai_fraction + 1e-9 >= group.min_region_cai_fraction
        ),
    }


def calculate_diversification_metrics(
    sequence: str,
    plan: Optional[DiversificationPlan],
    host: str = "hsapiens",
    *,
    codon_table: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Summarize diversification quality for a sequence."""

    if plan is None or not plan.active:
        return {
            "diversification_active": False,
            "diversification_pass": True,
            "diversification_penalty": 0.0,
        }
    if codon_table is None:
        codon_table = load_codon_table(host)

    group_metrics = tuple(
        _group_metrics(sequence, group, codon_table)
        for group in plan.groups
    )
    penalty = max((m["diversification_penalty"] for m in group_metrics), default=0.0)
    return {
        "diversification_active": True,
        "diversification_mask_type": plan.mask_type,
        "diversification_group_count": len(plan.groups),
        "diversification_pass": all(m["diversification_pass"] for m in group_metrics),
        "diversification_penalty": penalty,
        "diversification_max_pairwise_identity": max(
            (m["max_pairwise_identity"] for m in group_metrics),
            default=0.0,
        ),
        "diversification_longest_exact_shared_nt": max(
            (m["longest_exact_shared_nt"] for m in group_metrics),
            default=0,
        ),
        "diversification_min_region_cai_fraction": min(
            (m["min_region_cai_fraction"] for m in group_metrics),
            default=1.0,
        ),
        "diversification_group_metrics": group_metrics,
    }


def _codon_weight(codon: str, codon_table: Dict[str, float]) -> float:
    return max(float(codon_table.get(codon, 0.0)), 0.0)


def _candidate_codons(aa: str, codon_table: Dict[str, float]) -> Tuple[str, ...]:
    codons = tuple(c for c in AA_TO_CODONS.get(aa, ()) if GENETIC_CODE.get(c) == aa)
    positive = tuple(c for c in codons if _codon_weight(c, codon_table) > 0.0)
    return positive or codons


def _assignment_score(
    assignment: Sequence[str],
    codon_table: Dict[str, float],
) -> Tuple[float, float, Tuple[str, ...]]:
    if len(assignment) <= 1:
        pair_score = 0.0
    else:
        pair_score = max(
            _codon_match_count(left, right) / 3.0
            for left, right in itertools.combinations(assignment, 2)
        )
    best_weight = max((_codon_weight(c, codon_table) for c in assignment), default=1.0)
    if best_weight <= 0:
        cai_loss = 0.0
    else:
        cai_loss = sum(
            max(0.0, best_weight - _codon_weight(c, codon_table)) / best_weight
            for c in assignment
        ) / len(assignment)
    return pair_score, 0.20 * cai_loss, tuple(assignment)


def _choose_position_assignment(
    aa: str,
    current_codons: Sequence[str],
    codon_indices: Sequence[int],
    codon_table: Dict[str, float],
) -> Tuple[str, ...]:
    choices = _candidate_codons(aa, codon_table)
    if not choices:
        return tuple(current_codons)

    per_occurrence_choices: List[Tuple[str, ...]] = []
    for codon, codon_index in zip(current_codons, codon_indices):
        if codon_index == 0:
            per_occurrence_choices.append((codon,))
        else:
            per_occurrence_choices.append(choices)

    if len(per_occurrence_choices) <= 5:
        best = min(
            itertools.product(*per_occurrence_choices),
            key=lambda assignment: _assignment_score(assignment, codon_table),
        )
        return tuple(best)

    assigned: List[str] = []
    for options in per_occurrence_choices:
        if len(options) == 1:
            assigned.append(options[0])
            continue
        best = min(
            options,
            key=lambda candidate: (
                max(
                    (_codon_match_count(candidate, prev) / 3.0 for prev in assigned),
                    default=0.0,
                ),
                -_codon_weight(candidate, codon_table),
                candidate,
            ),
        )
        assigned.append(best)
    return tuple(assigned)


def diversify_sequence(
    sequence: str,
    plan: DiversificationPlan,
    host: str = "hsapiens",
    *,
    codon_table: Optional[Dict[str, float]] = None,
) -> DiversificationResult:
    """
    Rewrite masked regions synonymously to reduce DNA homology between copies.

    The pass is deterministic. It prioritizes reducing per-position codon
    identity between occurrences while avoiding zero-weight codons when host
    data offers alternatives. CAI, exact-match, and pairwise-identity gates are
    reported in the returned metrics.
    """

    if codon_table is None:
        codon_table = load_codon_table(host)
    plan = with_baseline_region_cai(plan, sequence, host, codon_table=codon_table)
    codons = [sequence[i:i + 3] for i in range(0, len(sequence), 3)]
    changes: List[Dict[str, Any]] = []

    for group in plan.groups:
        aa_length = len(group.aa_sequence)
        for aa_offset in range(aa_length):
            aa = group.aa_sequence[aa_offset]
            codon_indices = [
                occ.aa_start + aa_offset
                for occ in group.occurrences
            ]
            current = [codons[index] for index in codon_indices]
            assignment = _choose_position_assignment(aa, current, codon_indices, codon_table)
            for occ_index, (codon_index, old, new) in enumerate(zip(codon_indices, current, assignment)):
                if old == new:
                    continue
                codons[codon_index] = new
                changes.append({
                    "group": group.label,
                    "occurrence": occ_index + 1,
                    "codon_position": codon_index + 1,
                    "nt_position": codon_index * 3 + 1,
                    "amino_acid": aa,
                    "old_codon": old,
                    "new_codon": new,
                })

    diversified = "".join(codons)
    if translate_dna_to_protein(diversified) != translate_dna_to_protein(sequence):
        raise AssertionError("Diversification changed protein identity.")
    if diversified[:3] != sequence[:3]:
        raise AssertionError("Diversification changed the start codon.")

    metrics = calculate_diversification_metrics(
        diversified,
        plan,
        host,
        codon_table=codon_table,
    )
    return DiversificationResult(
        sequence=diversified,
        plan=plan,
        metrics=metrics,
        changes=tuple(changes),
    )


def render_diversification_map_html(
    plan: Optional[DiversificationPlan],
    *,
    sequence_length_nt: Optional[int] = None,
    title: str = "Diversification masks",
) -> str:
    """Return an HTML/SVG nucleotide-coordinate map for notebook output."""

    if plan is None or not plan.active:
        return ""

    total_nt = sequence_length_nt or plan.sequence_length_nt
    total_nt = max(1, int(total_nt))
    height = 42 + 24 * len(plan.groups)
    rows: List[str] = []
    legend: List[str] = []

    for row_index, group in enumerate(plan.groups):
        y = 30 + row_index * 24
        for occ_index, occ in enumerate(group.occurrences, start=1):
            x = occ.start_nt / total_nt * 100.0
            width = max(0.5, occ.length_nt / total_nt * 100.0)
            tooltip = html.escape(
                f"{group.label} occurrence {occ_index}: "
                f"nt {occ.start_nt + 1}-{occ.end_nt}, "
                f"aa {occ.aa_start + 1}-{occ.aa_end}"
            )
            rows.append(
                f'<rect x="{x:.4f}%" y="{y}" width="{width:.4f}%" height="14" '
                f'rx="2" fill="{group.color}">'
                f"<title>{tooltip}</title></rect>"
            )
        legend.append(
            '<span style="display:inline-flex;align-items:center;margin-right:14px;">'
            f'<span style="display:inline-block;width:12px;height:12px;background:{group.color};'
            'margin-right:5px;border-radius:2px;"></span>'
            f'{html.escape(group.label)} ({len(group.occurrences)}x, {group.length_nt} nt)'
            '</span>'
        )

    return f"""
    <div style="max-width: 760px;">
      <div style="font-weight: 600; margin: 0 0 4px 0;">{html.escape(title)}</div>
      <svg viewBox="0 0 1000 {height}" width="100%" height="{height}"
           role="img" aria-label="{html.escape(title)}">
        <rect x="0" y="14" width="1000" height="12" rx="3"
              fill="#eeeeee" stroke="#bbbbbb" />
        <text x="0" y="10" font-size="10" fill="#555">nt 1</text>
        <text x="1000" y="10" font-size="10" text-anchor="end" fill="#555">nt {total_nt}</text>
        {''.join(rows)}
      </svg>
      <div style="font-size: 12px; color: #333;">{''.join(legend)}</div>
    </div>
    """
