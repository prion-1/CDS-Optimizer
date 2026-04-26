from src.input_processing import (
    clean_nucleotide_sequence,
    prepare_input_sequence,
    validate_protein_sequence,
)


def test_clean_nucleotide_sequence_removes_terminal_stop_codon():
    cleaned, error, messages = clean_nucleotide_sequence('ATG GCT TAA')

    assert error is None
    assert cleaned == 'ATGGCT'
    assert messages == ['Terminal stop codon found at position 7; removing it.']


def test_clean_nucleotide_sequence_truncates_before_internal_stop_and_drops_stop():
    cleaned, error, messages = clean_nucleotide_sequence('ATGGCTTAAATGCGT')

    assert error is None
    assert cleaned == 'ATGGCT'
    assert messages == [
        'Non-terminal stop codon found at position 7; '
        'truncating before stop and discarding last 60%.'
    ]


def test_clean_nucleotide_sequence_converts_rna_and_trims_partial_frame():
    cleaned, error, messages = clean_nucleotide_sequence('aug gcu uaa cc')

    assert error is None
    assert cleaned == 'ATGGCT'
    assert messages == [
        'Disrupted sequence frame! Truncating last 2 nucleotides.',
        'Terminal stop codon found at position 7; removing it.',
    ]


def test_clean_nucleotide_sequence_rejects_invalid_nucleotides_deterministically():
    cleaned, error, messages = clean_nucleotide_sequence('ATGBXTAACC')

    assert cleaned is None
    assert error == 'Invalid nucleotides: B, X'
    assert messages == []


def test_clean_nucleotide_sequence_rejects_stop_only_after_trimming():
    cleaned, error, messages = clean_nucleotide_sequence('TAAGCTGCC')

    assert cleaned is None
    assert error == 'No coding sequence remains after stop-codon trimming'
    assert messages == [
        'Non-terminal stop codon found at position 1; '
        'truncating before stop and discarding last 100%.'
    ]


def test_validate_protein_sequence_and_prepare_protein_input():
    protein, error = validate_protein_sequence(' mkt ')

    assert error is None
    assert protein == 'MKT'

    cds, error, messages = prepare_input_sequence('MKT', input_type='protein', host='ecoli')

    assert error is None
    assert cds is not None
    assert len(cds) == 9
    assert messages == []


def test_validate_protein_sequence_reports_sorted_invalid_amino_acids():
    protein, error = validate_protein_sequence('MAZ*')

    assert protein is None
    assert error == 'Invalid amino acids: *, Z'


def test_prepare_input_sequence_rejects_unknown_input_type():
    cleaned, error, messages = prepare_input_sequence('ATGGCT', input_type='plasmid')

    assert cleaned is None
    assert error == "input_type must be 'dna' or 'protein'"
    assert messages == []
