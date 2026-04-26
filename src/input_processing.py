"""
Shared input validation and cleanup for notebook and future CLI entry points.

The optimizer expects a frame-aligned coding sequence without stop codons.
This module owns the ingress cleanup so UI and terminal paths use the same
contract before calling downstream optimization/scoring code.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from .utils import back_translate_protein


STOP_CODONS = ('TAA', 'TAG', 'TGA')
VALID_AAS = set('ACDEFGHIKLMNPQRSTVWY')
VALID_NUCS = set('ATGC')


def _format_invalid(values: set[str]) -> str:
    return ', '.join(sorted(values))


def validate_protein_sequence(sequence: str) -> Tuple[Optional[str], Optional[str]]:
    """Validate protein sequence and return (cleaned_sequence, error)."""

    cleaned = re.sub(r'\s+', '', sequence.upper())

    if not cleaned:
        return None, 'Sequence is empty'

    invalid_chars = set(cleaned) - VALID_AAS
    if invalid_chars:
        return None, f'Invalid amino acids: {_format_invalid(invalid_chars)}'

    return cleaned, None


def clean_nucleotide_sequence(
    sequence: str,
) -> Tuple[Optional[str], Optional[str], List[str]]:
    """
    Clean and validate nucleotide input.

    Returns (cleaned_cds, error, messages). The returned CDS is uppercase DNA,
    frame-aligned, and truncated before the first in-frame stop codon. The stop
    codon itself is not included in the output.
    """

    messages: List[str] = []
    cleaned = re.sub(r'[\s\d]+', '', sequence.upper())

    if not cleaned:
        return None, 'Sequence is empty', messages
    if len(cleaned) < 9:
        return None, 'What are you trying to optimize, you great optimizer?', messages

    cleaned = cleaned.replace('U', 'T')
    invalid_chars = set(cleaned) - VALID_NUCS
    if invalid_chars:
        return None, f'Invalid nucleotides: {_format_invalid(invalid_chars)}', messages

    if len(cleaned) % 3 != 0:
        rest = len(cleaned) % 3
        cleaned = cleaned[:-rest]
        messages.append(f'Disrupted sequence frame! Truncating last {rest} nucleotides.')

    for i in range(0, len(cleaned) - 2, 3):
        codon = cleaned[i:i + 3]
        if codon not in STOP_CODONS:
            continue

        if i + 3 == len(cleaned):
            messages.append(f'Terminal stop codon found at position {i + 1}; removing it.')
        else:
            discarded = len(cleaned) - i
            discarded_pct = discarded / len(cleaned) * 100
            messages.append(
                f'Non-terminal stop codon found at position {i + 1}; '
                f'truncating before stop and discarding last {discarded_pct:.0f}%.'
            )
        cleaned = cleaned[:i]
        break

    if not cleaned:
        return None, 'No coding sequence remains after stop-codon trimming', messages

    return cleaned, None, messages


def prepare_input_sequence(
    sequence: str,
    *,
    input_type: str,
    host: str = 'hsapiens',
) -> Tuple[Optional[str], Optional[str], List[str]]:
    """
    Prepare user input for downstream optimization.

    input_type is either 'dna'/'rna' for nucleotide input or 'protein' for
    amino-acid input that should be back-translated for the selected host.
    """

    kind = input_type.strip().lower()
    if kind in {'dna', 'rna', 'nucleotide', 'nucleotides', 'cds'}:
        return clean_nucleotide_sequence(sequence)

    if kind in {'protein', 'aa', 'amino_acid', 'amino-acid'}:
        protein, error = validate_protein_sequence(sequence)
        if error:
            return None, error, []
        return back_translate_protein(protein, host), None, []

    return None, "input_type must be 'dna' or 'protein'", []
