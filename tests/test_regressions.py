import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.complexity_analysis import compute_complexity_track
from src.local_repair import local_repair
from src.optimization import FitnessWeights, GeneticAlgorithm, fitness_function
from src.pre_optimization import optimize_codons
from src import utils


def _weights_with(**overrides):
    base = {
        'cai': 0.0,
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
    base.update(overrides)
    return FitnessWeights(**base)


def _make_test_cds(start_codon: str = 'ATG') -> str:
    codons = [
        start_codon,
        'GCT', 'GAC', 'GAA', 'CTG', 'CCG', 'AAG', 'TTC', 'AAC', 'GGT',
        'TCC', 'GTC', 'GAT', 'CTG', 'ATC', 'GAC', 'GTT', 'GAG', 'GCC',
        'AAC', 'TAC', 'CGT', 'CTG', 'ACC', 'GGT', 'GCG', 'GAC', 'TTC',
        'CAG', 'GCT', 'AAC', 'GTC', 'TAA',
    ]
    return ''.join(codons)


def test_preoptimization_and_local_repair_preserve_gtg_start_codon():
    sequence = _make_test_cds('GTG')

    for method in ('simple', 'percentile', 'structure'):
        preoptimized = optimize_codons(sequence, 'ecoli', False, method=method)
        repaired, _ = local_repair(
            preoptimized,
            'ecoli',
            False,
            preoptimized,
            target_gc=utils.get_target_gc('ecoli'),
            window_nt=42,
            max_subs_per_window=2,
        )

        assert preoptimized.startswith('GTG')
        assert repaired.startswith('GTG')


def test_ga_preserves_gtg_start_codon():
    sequence = 'GTGGCTGCCGCTTAA'
    ga = GeneticAlgorithm(
        initial_sequence=sequence,
        host='ecoli',
        is_eukaryote=False,
        pop_size=8,
        generations=3,
        mutation_rate=0.2,
        weights=_weights_with(cai=1.0),
        random_seed=1,
    )

    ga.initialize_population()

    assert len(ga.population) == ga.pop_size
    assert all(candidate.startswith('GTG') for candidate in ga.population)
    assert ga.validate_offspring(sequence)
    assert not ga.validate_offspring('ATG' + sequence[3:])

    best_sequence, best_fitness, metrics = ga.run(verbose=False)

    assert best_sequence.startswith('GTG')
    assert best_fitness == pytest.approx(metrics['fitness'])


def test_calculate_gc3_content_handles_partial_codons():
    assert utils.calculate_gc3_content('ATGG') == pytest.approx(100.0)
    assert utils.calculate_gc3_content('AT') == pytest.approx(0.0)


def test_complexity_track_smoothing_preserves_coordinate_length():
    sequence = ('ATGC' * 45)[:180]
    result = compute_complexity_track(
        sequence,
        window_size=150,
        step=10,
        smooth=99,
    )

    assert len(result['start']) == 4
    assert len(result['start']) == len(result['end']) == len(result['mid']) == len(result['score'])


def test_count_unwanted_motifs_matches_between_numba_and_python(monkeypatch):
    sequence = 'AATAAATAAA'

    python_count = utils.count_unwanted_motifs(sequence, [])

    def fake_numba_counter(seq_array, motif_array):
        seq = ''.join(chr(int(base)) for base in seq_array)
        motif = ''.join(chr(int(base)) for base in motif_array)
        total = 0
        for i in range(len(seq) - len(motif) + 1):
            if seq[i:i + len(motif)] == motif:
                total += 1
        return total

    monkeypatch.setattr(utils, 'NUMBA_AVAILABLE', True)
    monkeypatch.setattr(utils, '_count_motif_numba', fake_numba_counter, raising=False)
    numba_count = utils.count_unwanted_motifs(sequence, [])

    assert python_count == numba_count == 2


def test_harmonization_correlation_reports_net_and_mean_deviation():
    sequence = 'ATGGCTGAGGCCGTTTTCGACTAA'

    metrics = utils.harmonization_correlation(
        sequence,
        sequence,
        'hsapiens',
        window_codons=4,
        measure='cai',
    )

    assert metrics['measure'] == 'cai'
    assert metrics['n_windows'] > 0
    assert metrics['net_abs_dev'] == pytest.approx(0.0)
    assert metrics['mean_abs_dev'] == pytest.approx(0.0)


def test_calculate_codon_pair_score_uses_empirical_table(tmp_path, monkeypatch):
    table_dir = tmp_path / 'codon_pair_tables'
    table_dir.mkdir()
    table_path = table_dir / 'testhost.csv'

    with table_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['codon1', 'codon2', 'cps'])
        writer.writeheader()
        writer.writerow({'codon1': 'ATG', 'codon2': 'GCT', 'cps': 1.5})
        writer.writerow({'codon1': 'GCT', 'codon2': 'GCC', 'cps': -0.5})

    monkeypatch.setattr(utils, 'CODON_PAIR_TABLE_DIR', str(table_dir))
    utils.load_codon_pair_table.cache_clear()
    utils.get_cps_score_matrix.cache_clear()

    score = utils.calculate_codon_pair_score('ATGGCTGCC', 'testhost')

    assert score == pytest.approx(0.5)
    assert utils.calculate_codon_pair_score('ATGGCTGCCTAA', 'testhost') == pytest.approx(0.5)


def test_missing_codon_pair_data_is_fitness_neutral(tmp_path, monkeypatch):
    table_dir = tmp_path / 'codon_pair_tables'
    table_dir.mkdir()
    monkeypatch.setattr(utils, 'CODON_PAIR_TABLE_DIR', str(table_dir))
    utils.load_codon_pair_table.cache_clear()
    utils.get_cps_score_matrix.cache_clear()

    baseline_fitness, _ = fitness_function(
        sequence='ATGGCTGCC',
        host='hsapiens',
        is_eukaryote=False,
        weights=_weights_with(cai=1.0, codon_pair_bias=0.0),
    )

    fitness_with_missing_cps, metrics_with_missing_cps = fitness_function(
        sequence='ATGGCTGCC',
        host='hsapiens',
        is_eukaryote=False,
        weights=_weights_with(cai=1.0, codon_pair_bias=1.0),
        cps_matrix=None,
    )

    assert metrics_with_missing_cps['codon_pair_score'] == pytest.approx(0.0)
    assert metrics_with_missing_cps['cps_normalized'] == pytest.approx(0.0)
    assert fitness_with_missing_cps == pytest.approx(baseline_fitness)


def test_fitness_codon_pair_bias_uses_matrix_and_skips_stop_pairs():
    cps_matrix = np.zeros((64, 64), dtype=np.float32)
    cps_matrix[utils.CODON_TO_INDEX['ATG'], utils.CODON_TO_INDEX['GCT']] = 2.0
    cps_matrix[utils.CODON_TO_INDEX['GCT'], utils.CODON_TO_INDEX['GCC']] = 2.0
    cps_matrix[utils.CODON_TO_INDEX['GCC'], utils.CODON_TO_INDEX['TAA']] = -5.0

    fitness, metrics = fitness_function(
        sequence='ATGGCTGCCTAA',
        host='ecoli',
        is_eukaryote=False,
        weights=_weights_with(codon_pair_bias=1.0),
        cps_matrix=cps_matrix,
    )

    expected = 1.0 / (1.0 + math.exp(-2.0))
    assert metrics['codon_pair_score'] == pytest.approx(2.0)
    assert metrics['cps_normalized'] == pytest.approx(expected)
    assert fitness == pytest.approx(expected)


def test_fitness_tai_uses_vector_and_skips_stop_codon():
    tai_vector = np.zeros(64, dtype=np.float32)
    tai_vector[utils.CODON_TO_INDEX['ATG']] = 0.50
    tai_vector[utils.CODON_TO_INDEX['GCT']] = 0.25
    tai_vector[utils.CODON_TO_INDEX['GCC']] = 1.00

    fitness, metrics = fitness_function(
        sequence='ATGGCTGCCTAA',
        host='ecoli',
        is_eukaryote=False,
        weights=_weights_with(tai=1.0),
        tai_score_vector=tai_vector,
    )

    expected = (0.50 * 0.25 * 1.00) ** (1.0 / 3.0)
    assert metrics['tai'] == pytest.approx(expected)
    assert fitness == pytest.approx(expected)


def test_bundled_codon_pair_tables_are_complete_and_finite():
    utils.load_codon_pair_table.cache_clear()
    utils.get_cps_score_matrix.cache_clear()

    expected_pairs = {
        (codon1, codon2)
        for codon1 in utils.SENSE_CODONS
        for codon2 in utils.SENSE_CODONS
    }

    for host in ('ecoli', 'hsapiens', 'mmusculus', 'scerevisiae', 'spombe'):
        assert utils.codon_pair_available_for(host)
        df = utils.load_codon_pair_table(host)
        observed_pairs = list(zip(df['codon1'], df['codon2']))

        assert len(df) == len(expected_pairs) == 3721
        assert len(set(observed_pairs)) == len(expected_pairs)
        assert set(observed_pairs) == expected_pairs
        assert all(math.isfinite(value) for value in df['cps'])
        assert df['cps'].max() > df['cps'].min()
        assert df['cps'].mean() == pytest.approx(0.0, abs=1e-5)
        assert math.isfinite(utils.calculate_codon_pair_score('ATGGCTGCC', host))


def test_bundled_tai_weights_are_complete_positive_and_finite():
    utils.load_tai_weights.cache_clear()
    utils.get_tai_score_vector.cache_clear()

    for host in ('ecoli', 'hsapiens', 'mmusculus', 'scerevisiae', 'spombe'):
        assert utils.tai_available_for(host)
        weights = utils.load_tai_weights(host)

        assert set(weights) == set(utils.SENSE_CODONS)
        assert all(math.isfinite(value) for value in weights.values())
        assert all(0.0 < value <= 1.0 for value in weights.values())
        assert max(weights.values()) == pytest.approx(1.0)
        assert math.isfinite(utils.calculate_tai('ATGGCTGCC', host))
        assert utils.calculate_tai('ATGGCTGCCTAA', host) == pytest.approx(
            utils.calculate_tai('ATGGCTGCC', host)
        )


def test_tai_loader_rejects_incomplete_weight_tables(tmp_path, monkeypatch):
    table_dir = tmp_path / 'trna_weights'
    table_dir.mkdir()
    table_path = table_dir / 'testhost.csv'
    with table_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['codon', 'weight'])
        writer.writeheader()
        writer.writerow({'codon': 'ATG', 'weight': 1.0})

    monkeypatch.setattr(utils, 'TRNA_WEIGHTS_DIR', str(table_dir))
    utils.load_tai_weights.cache_clear()
    utils.get_tai_score_vector.cache_clear()

    with pytest.raises(ValueError, match='missing sense codons'):
        utils.load_tai_weights('testhost')


def test_count_cryptic_splice_sites_uses_dependency_free_heuristic():
    sequence = 'CAGGTAAGTAAAAAATTTTTTCAGAAAA'

    assert utils.count_cryptic_splice_sites(sequence, True) == 2
    assert utils.count_cryptic_splice_sites(sequence, False) == 0

    # Threshold arguments are retained for API compatibility but the current
    # heuristic is intentionally unthresholded.
    assert utils.count_cryptic_splice_sites(
        sequence,
        True,
        donor_threshold=999.0,
        acceptor_threshold=999.0,
    ) == 2


def test_get_target_gc3_uses_host_defaults_and_table_fallback(tmp_path, monkeypatch):
    assert utils.get_target_gc3('ecoli') == pytest.approx(60.0)
    assert utils.get_target_gc3('scerevisiae') == pytest.approx(25.0)

    table_dir = tmp_path / 'codon_tables'
    table_dir.mkdir()
    table_path = table_dir / 'testhost.csv'
    with table_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['amino_acid', 'codon', 'frequency'])
        writer.writeheader()
        writer.writerow({'amino_acid': 'Ala', 'codon': 'GCT', 'frequency': 0.25})
        writer.writerow({'amino_acid': 'Ala', 'codon': 'GCC', 'frequency': 0.75})

    monkeypatch.setattr(utils, 'CODON_TABLE_DIR', str(table_dir))
    utils.load_codon_frequencies.cache_clear()
    utils.get_target_gc3.cache_clear()

    assert utils.get_target_gc3('testhost') == pytest.approx(75.0)


def test_main_notebook_code_cells_compile_and_keep_host_aware_backtranslation():
    notebook_path = Path(__file__).resolve().parents[1] / 'main.ipynb'
    notebook = json.loads(notebook_path.read_text())

    code_text = []
    for i, cell in enumerate(notebook['cells']):
        if cell.get('cell_type') != 'code':
            continue
        source = cell.get('source', '')
        source = ''.join(source) if isinstance(source, list) else source
        compile(source, f'main.ipynb cell {i}', 'exec')
        code_text.append(source)

    joined = '\n'.join(code_text)
    assert "warnings.filterwarnings('ignore')" not in joined
    assert "{', '.join" not in joined
    assert 'f\'Invalid amino acids: {", ".join(invalid_chars)}\'' in joined
    assert 'f\'Invalid nucleotides: {", ".join(invalid_chars)}\'' in joined
    assert "back_translate_protein(clean_protein, host_dropdown.value)" in joined
    assert "PREOPTIMIZATION: Optimization complete" in joined
    assert "if pipeline != 'preoptimization_only':" in joined
    assert "print('\\nFinal Optimized Sequence:')" in joined
    assert joined.count("'codon_pair_bias': weight_sliders['cps'].value") == 2
    assert joined.count("'tai': weight_sliders['tai'].value") == 2
    assert "calculate_codon_pair_score" in joined
    assert "calculate_tai" in joined
    assert "Final codon-pair score:" in joined
    assert "Final tAI:" in joined
