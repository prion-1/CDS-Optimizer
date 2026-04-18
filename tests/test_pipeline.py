import pytest

from src.local_repair import local_repair
from src.optimization import genetic_algorithm
from src.pre_optimization import optimize_codons
from src.utils import get_target_gc, translate_dna_to_protein


PREOPTIMIZATION_METHODS = ('simple', 'percentile', 'structure')
GA_WEIGHTS = {
    'cai': 1.0,
    'gc_deviation': 0.0,
    'folding_energy': 0.0,
    'unwanted_motifs': 0.0,
    'repeats': 0.0,
    'cryptic_splice': 0.0,
    'gc3_deviation': 0.0,
    'internal_atg': 0.0,
    'accessibility': 0.0,
    'tai': 0.0,
    'codon_pair_bias': 0.0,
}


def _make_test_cds(start_codon: str = 'ATG') -> str:
    codons = [
        start_codon,
        'GCT', 'GAC', 'GAA', 'CTG', 'CCG', 'AAG', 'TTC', 'AAC', 'GGT',
        'TCC', 'GTC', 'GAT', 'CTG', 'ATC', 'GAC', 'GTT', 'GAG', 'GCC',
        'AAC', 'TAC', 'CGT', 'CTG', 'ACC', 'GGT', 'GCG', 'GAC', 'TTC',
        'CAG', 'GCT', 'AAC', 'GTC', 'TAA',
    ]
    return ''.join(codons)


@pytest.mark.parametrize('method', PREOPTIMIZATION_METHODS)
@pytest.mark.parametrize('start_codon', ('ATG', 'GTG'))
def test_full_pipeline_preserves_translation_and_start_codon(method, start_codon):
    input_seq = _make_test_cds(start_codon)
    input_protein = translate_dna_to_protein(input_seq)

    preoptimized = optimize_codons(input_seq, 'ecoli', False, method=method)
    assert len(preoptimized) == len(input_seq)
    assert preoptimized[:3] == input_seq[:3]
    assert translate_dna_to_protein(preoptimized) == input_protein

    ga_seq, _, _ = genetic_algorithm(
        preoptimized,
        host='ecoli',
        is_eukaryote=False,
        pop_size=10,
        generations=5,
        mutation_rate=0.05,
        weights=GA_WEIGHTS,
        random_seed=1,
        verbose=False,
    )
    assert len(ga_seq) == len(input_seq)
    assert ga_seq[:3] == input_seq[:3]
    assert translate_dna_to_protein(ga_seq) == input_protein

    repaired, _ = local_repair(
        ga_seq,
        'ecoli',
        False,
        preoptimized,
        target_gc=get_target_gc('ecoli'),
        window_nt=42,
        max_subs_per_window=2,
    )
    assert len(repaired) == len(input_seq)
    assert repaired[:3] == input_seq[:3]
    assert translate_dna_to_protein(repaired) == input_protein
