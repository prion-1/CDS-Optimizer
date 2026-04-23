import csv
import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.complexity_analysis import compute_complexity_track
from src.local_repair import local_repair
from src.optimization import FitnessWeights, GeneticAlgorithm, fitness_function
from src.pre_optimization import optimize_codons
from src import degeneracy
from src import presets
from src import utils
from src import hybrid_pipeline


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


def test_local_repair_api_defaults_match_hybrid_sweep_default():
    signature = inspect.signature(local_repair)

    assert (
        signature.parameters['window_nt'].default
        == hybrid_pipeline.DEFAULT_HYBRID_REPAIR_WINDOW_NT
        == 36
    )
    assert (
        signature.parameters['max_subs_per_window'].default
        == hybrid_pipeline.DEFAULT_HYBRID_REPAIR_MAX_SUBS
        == 3
    )


def test_default_polish_weights_keep_sweep_derived_envelope():
    assert hybrid_pipeline.DEFAULT_POLISH_POP_SIZE == 20
    assert hybrid_pipeline.DEFAULT_POLISH_GENERATIONS == 8
    assert hybrid_pipeline.DEFAULT_POLISH_MUTATION_RATE == pytest.approx(0.008)
    assert hybrid_pipeline.DEFAULT_POLISH_SEEDS == (1701, 2701, 3701, 4701, 5701)
    assert hybrid_pipeline.DEFAULT_POLISH_WEIGHTS is (
        hybrid_pipeline.EXTREME_REPEAT_GUARD_POLISH_WEIGHTS
    )
    assert hybrid_pipeline.DEFAULT_POLISH_WEIGHTS['repeats'] == pytest.approx(0.42)
    assert hybrid_pipeline.DEFAULT_POLISH_WEIGHTS['cai'] == pytest.approx(0.18)


def test_named_optimization_presets_are_normalized_and_exported():
    preset_rows = presets.list_optimization_presets()
    names = [preset.name for preset in preset_rows]

    assert names == [
        presets.PRESET_REPEAT_GUARD,
        presets.PRESET_EXPRESSION_CONSERVATIVE,
        presets.PRESET_SECONDARY_STRUCTURE,
        presets.PRESET_SECONDARY_ACCESSIBILITY,
        presets.PRESET_REGULATORY_MOTIF_HIGH,
        presets.PRESET_REGULATORY_CONSTRUCT_SAFE,
    ]
    for preset in preset_rows:
        assert set(preset.weights) == set(presets.FITNESS_WEIGHT_KEYS)
        assert sum(preset.weights.values()) == pytest.approx(1.0)
        assert presets.get_optimization_preset(preset.name) is preset

    assert (
        presets.get_optimization_preset(presets.PRESET_REPEAT_GUARD).weights['repeats']
        == pytest.approx(0.42)
    )
    assert (
        presets.get_optimization_preset(
            presets.PRESET_EXPRESSION_CONSERVATIVE
        ).weights['cai']
        == pytest.approx(0.30)
    )
    assert (
        presets.get_optimization_preset(
            presets.PRESET_SECONDARY_STRUCTURE
        ).weights['folding_energy']
        == pytest.approx(0.34)
    )
    assert (
        presets.get_optimization_preset(
            presets.PRESET_REGULATORY_CONSTRUCT_SAFE
        ).selector
        == presets.PRESET_SELECTOR_REGULATORY
    )


def test_unknown_optimization_preset_is_rejected():
    with pytest.raises(ValueError, match='Unknown optimization preset'):
        presets.get_optimization_preset('not-a-preset')


def test_preset_helpers_filter_roundtrip_and_host_adapt(monkeypatch):
    generic_names = [
        preset.name
        for preset in presets.list_optimization_presets(
            selector=presets.PRESET_SELECTOR_GENERIC
        )
    ]
    regulatory_names = [
        preset.name
        for preset in presets.list_optimization_presets(
            selector=presets.PRESET_SELECTOR_REGULATORY
        )
    ]

    assert presets.DEFAULT_OPTIMIZATION_PRESET == presets.PRESET_REPEAT_GUARD
    assert generic_names == [
        presets.PRESET_REPEAT_GUARD,
        presets.PRESET_EXPRESSION_CONSERVATIVE,
        presets.PRESET_SECONDARY_STRUCTURE,
        presets.PRESET_SECONDARY_ACCESSIBILITY,
    ]
    assert regulatory_names == [
        presets.PRESET_REGULATORY_MOTIF_HIGH,
        presets.PRESET_REGULATORY_CONSTRUCT_SAFE,
    ]

    original = presets.get_optimization_preset(
        presets.PRESET_EXPRESSION_CONSERVATIVE
    ).weights
    notebook_weights = presets.fitness_weights_to_notebook_weights(original)
    roundtrip = presets.notebook_weights_to_fitness_weights(notebook_weights)

    assert sum(notebook_weights.values()) == pytest.approx(1.0)
    for notebook_key, fitness_key in presets.NOTEBOOK_WEIGHT_KEY_MAP.items():
        assert roundtrip[fitness_key] == pytest.approx(notebook_weights[notebook_key])

    for preset_name in (
        presets.PRESET_SECONDARY_ACCESSIBILITY,
        presets.PRESET_REGULATORY_MOTIF_HIGH,
        presets.PRESET_REGULATORY_CONSTRUCT_SAFE,
    ):
        notebook_total = sum(
            presets.fitness_weights_to_notebook_weights(
                presets.get_optimization_preset(preset_name).weights
            ).values()
        )
        assert notebook_total == pytest.approx(1.0)

    monkeypatch.setattr(presets, 'tai_available_for', lambda host: False)
    monkeypatch.setattr(presets, 'codon_pair_available_for', lambda host: False)
    adapted = presets.preset_weights_for_host(
        presets.PRESET_EXPRESSION_CONSERVATIVE,
        'host_without_optional_tables',
    )

    assert adapted['tai'] == pytest.approx(0.0)
    assert adapted['codon_pair_bias'] == pytest.approx(0.0)
    assert sum(adapted.values()) == pytest.approx(1.0)
    assert adapted['repeats'] > original['repeats']


def test_repeat_penalty_window_params_scale_with_sequence_length():
    assert utils.repeat_penalty_window_params(0) == (30, 15)
    assert utils.repeat_penalty_window_params(100) == (30, 15)
    assert utils.repeat_penalty_window_params(1999)[0] < 500
    assert utils.repeat_penalty_window_params(2000) == (500, 250)
    assert utils.repeat_penalty_window_params(5000) == (500, 250)

    window, step = utils.repeat_penalty_window_params(1050)

    assert 30 < window < 500
    assert window % 10 == 0
    assert step == window // 2


def test_fitness_default_repeat_penalty_mode_is_legacy():
    sequence = 'ATG' + 'GCGCCGCG' + 'GCT' * 5 + 'TAA'
    weights = _weights_with(repeats=1.0)

    default_fitness, default_metrics = fitness_function(
        sequence=sequence,
        host='ecoli',
        is_eukaryote=False,
        weights=weights,
    )
    legacy_fitness, legacy_metrics = fitness_function(
        sequence=sequence,
        host='ecoli',
        is_eukaryote=False,
        weights=weights,
        repeat_penalty_mode=degeneracy.GA_REPEAT_PENALTY_MODE_LEGACY,
    )
    repeat_mean, repeat_max, repeat_frac_bad = utils.repeat_penalty_windowed(sequence)
    expected_score = degeneracy.calculate_legacy_repeat_score({
        'repeat_mean': repeat_mean,
        'repeat_max': repeat_max,
        'repeat_frac_bad': repeat_frac_bad,
    })

    assert default_metrics['repetitive_sequences'] == pytest.approx(expected_score)
    assert default_metrics['repeat_penalty'] == pytest.approx(
        degeneracy.normalize_ga_repeat_penalty(expected_score)
    )
    assert default_metrics['repetitive_sequences'] == pytest.approx(
        legacy_metrics['repetitive_sequences']
    )
    assert default_fitness == pytest.approx(legacy_fitness)
    assert 'degeneracy_score' not in default_metrics


def test_fitness_repeat_penalty_modes_enable_degeneracy_alignment():
    sequence = 'ATG' + 'GCGCCGCG' + 'GCT' * 5 + 'TAA'
    weights = _weights_with(repeats=1.0)

    legacy_fitness, legacy_metrics = fitness_function(
        sequence=sequence,
        host='ecoli',
        is_eukaryote=False,
        weights=weights,
        repeat_penalty_mode='legacy',
    )
    blend_fitness, blend_metrics = fitness_function(
        sequence=sequence,
        host='ecoli',
        is_eukaryote=False,
        weights=weights,
        repeat_penalty_mode='blend',
    )
    aligned_fitness, aligned_metrics = fitness_function(
        sequence=sequence,
        host='ecoli',
        is_eukaryote=False,
        weights=weights,
        repeat_penalty_mode='aligned',
    )

    assert blend_metrics['strict_degeneracy_component'] > 0
    assert blend_metrics['repeat_penalty'] > legacy_metrics['repeat_penalty']
    assert blend_fitness < legacy_fitness
    assert aligned_metrics['repetitive_sequences'] == pytest.approx(
        aligned_metrics['degeneracy_score']
    )
    assert aligned_fitness < legacy_fitness


def test_fitness_repeat_penalty_mode_validation():
    sequence = 'ATGGCTGCCTAA'
    weights = _weights_with(repeats=1.0)

    with pytest.raises(ValueError, match='Invalid repeat_penalty_mode'):
        fitness_function(
            sequence=sequence,
            host='ecoli',
            is_eukaryote=False,
            weights=weights,
            repeat_penalty_mode='unknown',
        )

    with pytest.raises(ValueError, match='strict_degeneracy_weight'):
        fitness_function(
            sequence=sequence,
            host='ecoli',
            is_eukaryote=False,
            weights=weights,
            strict_degeneracy_weight=-1.0,
        )

    with pytest.raises(ValueError, match='repeat_penalty_scale'):
        fitness_function(
            sequence=sequence,
            host='ecoli',
            is_eukaryote=False,
            weights=weights,
            repeat_penalty_scale=0.0,
        )


def test_multi_seed_ga_polish_selects_by_quality_not_fitness(monkeypatch):
    bad_sequence = 'ATGGCTGCTGCTTAA'
    clean_sequence = 'ATGGCCGCCGCCTAA'

    def fake_genetic_algorithm(*, random_seed, **kwargs):
        if random_seed == 1:
            return bad_sequence, 10.0, {'fitness': 10.0}
        return clean_sequence, 1.0, {'fitness': 1.0}

    def fake_sequence_quality_metrics(sequence, **kwargs):
        common = {
            'protein_identity_ok': True,
            'complexity_min': 0.45,
            'cai': 0.71,
            'tai': 0.32,
        }
        if sequence == clean_sequence:
            return {
                **common,
                'degeneracy_score': 0.50,
                'strict_degenerate_example_count': 0,
                'longest_gc_only_run': 6,
                'max_dinuc_tandem_bases': 0,
                'max_trinuc_tandem_bases': 0,
                'repeat_mean': 0.10,
                'repeat_max': 0.50,
                'repeat_frac_bad': 0.00,
                'longest_homopolymer': 4,
                'homopolymer_runs_ge5': 0,
                'complexity_mean': 0.57,
            }
        return {
            **common,
            'degeneracy_score': 8.00,
            'strict_degenerate_example_count': 1,
            'longest_gc_only_run': 8,
            'max_dinuc_tandem_bases': 0,
            'max_trinuc_tandem_bases': 0,
            'repeat_mean': 1.00,
            'repeat_max': 2.00,
            'repeat_frac_bad': 0.70,
            'longest_homopolymer': 5,
            'homopolymer_runs_ge5': 1,
            'complexity_mean': 0.58,
        }

    monkeypatch.setattr(hybrid_pipeline, 'genetic_algorithm', fake_genetic_algorithm)
    monkeypatch.setattr(hybrid_pipeline, 'sequence_quality_metrics', fake_sequence_quality_metrics)

    result = hybrid_pipeline.multi_seed_ga_polish(
        'ATGGCTGCTGCTTAA',
        host='ecoli',
        is_eukaryote=False,
        seeds=(1, 2),
    )

    assert result.best_seed == 2
    assert result.best_sequence == clean_sequence
    assert len(result.runs) == 2


def test_multi_seed_regulatory_ga_polish_selects_raw_regulatory_cleanup(monkeypatch):
    bad_sequence = 'ATGAATAAAGTAAGGTAA'
    clean_sequence = 'ATGGCCGCCGCCGCCTAA'

    def fake_genetic_algorithm(*, random_seed, **kwargs):
        if random_seed == 1:
            return bad_sequence, 100.0, {'fitness': 100.0}
        return clean_sequence, 1.0, {'fitness': 1.0}

    def fake_sequence_quality_metrics(sequence, **kwargs):
        return {
            'length_bp': len(sequence),
            'protein_identity_ok': True,
            'sequence_sha256': f'seed-{len(sequence)}',
            'gc': 52.0,
            'gc3': 50.0,
            'cai': 0.75,
            'tai': 0.32,
            'degeneracy_score': 1.0,
            'strict_degenerate_example_count': 0,
            'longest_gc_only_run': 7,
            'max_dinuc_tandem_bases': 0,
            'max_trinuc_tandem_bases': 0,
            'repeat_mean': 0.10,
            'repeat_max': 0.50,
            'repeat_frac_bad': 0.00,
            'longest_homopolymer': 4,
            'homopolymer_runs_ge5': 0,
            'complexity_mean': 0.60,
            'complexity_min': 0.58,
        }

    monkeypatch.setattr(hybrid_pipeline, 'genetic_algorithm', fake_genetic_algorithm)
    monkeypatch.setattr(hybrid_pipeline, 'sequence_quality_metrics', fake_sequence_quality_metrics)

    result = hybrid_pipeline.multi_seed_regulatory_ga_polish(
        clean_sequence,
        host='hsapiens',
        is_eukaryote=True,
        seeds=(1, 2),
    )

    assert result.best_seed == 2
    assert result.best_sequence == clean_sequence
    assert result.best_quality_metrics['regulatory_total_motifs'] == 0
    assert result.best_quality_metrics['regulatory_splice_sites'] == 0
    assert result.best_quality_metrics['regulatory_constructability_pass']

    bad_run = next(run for run in result.runs if run.seed == 1)
    assert bad_run.quality_metrics['regulatory_default_motifs'] == 1
    assert bad_run.quality_metrics['regulatory_splice_sites'] == 1
    assert (
        result.best_quality_metrics['regulatory_post_selection_score']
        > bad_run.quality_metrics['regulatory_post_selection_score']
    )


def test_regulatory_constructability_gate_is_relaxed_vs_repeat_guard():
    target_gc = utils.get_target_gc('hsapiens')
    target_gc3 = utils.get_target_gc3('hsapiens')
    metrics = {
        'protein_identity_ok': True,
        'cai': 0.62,
        'gc': target_gc,
        'gc3': target_gc3,
        'longest_gc_only_run': 10,
        'longest_homopolymer': 5,
        'max_dinuc_tandem_bases': 10,
        'max_trinuc_tandem_bases': 10,
        'strict_degenerate_example_count': 1,
        'repeat_frac_bad': 0.30,
    }

    assert not hybrid_pipeline.is_degeneracy_clean_candidate(metrics)
    assert hybrid_pipeline.regulatory_constructability_failures(
        metrics,
        target_gc=target_gc,
        target_gc3=target_gc3,
    ) == ()

    severe = {**metrics, 'longest_gc_only_run': 15}

    assert hybrid_pipeline.regulatory_constructability_failures(
        severe,
        target_gc=target_gc,
        target_gc3=target_gc3,
    ) == ('severe_gc_run',)


def test_long_repeat_burden_metrics_detect_broad_distributed_repeats():
    sequence = 'ATGAAAAA' * 300
    metrics = degeneracy.compute_degeneracy_metrics(sequence)

    assert metrics['long_repeat_mode']
    assert metrics['repeat_frac_bad'] > 0.45
    assert metrics['long_repeat_bad_region_span_fraction'] > 0.60
    assert metrics['long_repeat_broad_burden']
    assert metrics['long_repeat_segment_count'] == 2
    assert metrics['long_repeat_junction_window_count'] > 0


def test_regulatory_constructability_uses_long_broad_repeat_gate():
    common = {
        'protein_identity_ok': True,
        'cai': 0.70,
        'longest_gc_only_run': 10,
        'longest_homopolymer': 5,
        'max_dinuc_tandem_bases': 10,
        'max_trinuc_tandem_bases': 10,
        'repeat_frac_bad': 0.46,
        'length_bp': 5600,
        'long_repeat_mode': True,
    }

    broad = {
        **common,
        'long_repeat_bad_region_span_fraction': 0.61,
    }
    local = {
        **common,
        'long_repeat_bad_region_span_fraction': 0.55,
        'long_repeat_segment_frac_bad_max': 0.80,
    }
    short = {
        **common,
        'length_bp': 1500,
        'long_repeat_mode': False,
        'long_repeat_bad_region_span_fraction': 0.90,
    }

    assert hybrid_pipeline.regulatory_constructability_failures(broad) == (
        'broad_repeat_burden',
    )
    assert hybrid_pipeline.regulatory_constructability_failures(local) == ()
    assert hybrid_pipeline.regulatory_constructability_failures(short) == (
        'repeat_frac_bad',
    )


def test_regulatory_constructability_keeps_local_tandem_separate_from_broad_burden():
    cacna1f_like = {
        'protein_identity_ok': True,
        'cai': 0.70,
        'longest_gc_only_run': 10,
        'longest_homopolymer': 5,
        'max_dinuc_tandem_bases': 0,
        'max_trinuc_tandem_bases': 51,
        'repeat_frac_bad': 0.27,
        'length_bp': 5901,
        'long_repeat_mode': True,
        'long_repeat_bad_region_span_fraction': 0.38,
        'long_repeat_segment_frac_bad_max': 0.71,
    }

    assert hybrid_pipeline.regulatory_constructability_failures(cacna1f_like) == (
        'severe_trinuc_tandem',
    )


def test_regulatory_post_selection_score_is_non_saturating_for_splice():
    baseline = {
        'regulatory_default_motifs': 0,
        'regulatory_custom_motifs': 0,
        'regulatory_splice_sites': 20,
        'regulatory_internal_atg': 0,
    }
    common = {
        'length_bp': 1000,
        'protein_identity_ok': True,
        'regulatory_default_motifs': 0,
        'regulatory_custom_motifs': 0,
        'regulatory_total_motifs': 0,
        'regulatory_internal_atg': 0,
        'cai': 0.75,
        'tai': 0.32,
        'degeneracy_score': 1.0,
        'repeat_frac_bad': 0.0,
    }

    score_five_splice = hybrid_pipeline.regulatory_post_selection_score(
        {**common, 'regulatory_splice_sites': 5},
        baseline_metrics=baseline,
    )
    score_ten_splice = hybrid_pipeline.regulatory_post_selection_score(
        {**common, 'regulatory_splice_sites': 10},
        baseline_metrics=baseline,
    )

    assert score_five_splice > score_ten_splice


def test_hybrid_candidate_rank_prefers_degeneracy_clean_sequence():
    high_objective_with_gc_run = {
        'protein_identity_ok': True,
        'degeneracy_score': 2.0,
        'strict_degenerate_example_count': 1,
        'longest_gc_only_run': 8,
        'max_dinuc_tandem_bases': 0,
        'max_trinuc_tandem_bases': 0,
        'repeat_mean': 0.10,
        'repeat_max': 0.50,
        'repeat_frac_bad': 0.00,
        'longest_homopolymer': 4,
        'homopolymer_runs_ge5': 0,
        'complexity_mean': 0.60,
        'complexity_min': 0.50,
        'cai': 0.75,
        'tai': 0.35,
    }
    lower_objective_clean = {
        **high_objective_with_gc_run,
        'degeneracy_score': 3.0,
        'strict_degenerate_example_count': 0,
        'longest_gc_only_run': 7,
        'cai': 0.70,
        'tai': 0.30,
    }

    assert hybrid_pipeline.is_degeneracy_clean_candidate(lower_objective_clean)
    assert not hybrid_pipeline.is_degeneracy_clean_candidate(high_objective_with_gc_run)
    assert hybrid_pipeline.hybrid_candidate_rank(
        lower_objective_clean,
        fitness=0.1,
    ) > hybrid_pipeline.hybrid_candidate_rank(
        high_objective_with_gc_run,
        fitness=10.0,
    )


def test_sequence_quality_metrics_reports_strict_degeneracy_fields():
    sequence = 'ATG' + 'GCGCCGCG' + 'GCT' * 5 + 'TAA'
    metrics = hybrid_pipeline.sequence_quality_metrics(sequence, host='ecoli')
    shared_metrics = degeneracy.compute_degeneracy_metrics(sequence)

    repeat_window, repeat_step = utils.repeat_penalty_window_params(len(sequence))

    assert metrics['repeat_window_nt'] == repeat_window
    assert metrics['repeat_step_nt'] == repeat_step
    assert metrics['degeneracy_score'] == pytest.approx(shared_metrics['degeneracy_score'])
    assert metrics['longest_gc_only_run'] >= 8
    assert metrics['gc_runs_ge_threshold'] >= 1
    assert metrics['strict_degenerate_example_count'] >= 1
    assert metrics['degeneracy_score'] > 0
    assert not hybrid_pipeline.is_degeneracy_clean_candidate(metrics)


def test_sequence_quality_metrics_reports_long_repeat_burden_fields():
    sequence = 'ATGAAAAA' * 300
    metrics = hybrid_pipeline.sequence_quality_metrics(sequence, host='ecoli')

    assert metrics['long_repeat_mode']
    assert metrics['long_repeat_window_nt'] == degeneracy.LONG_REPEAT_LOCAL_WINDOW_NT
    assert metrics['long_repeat_step_nt'] == degeneracy.LONG_REPEAT_LOCAL_STEP_NT
    assert metrics['long_repeat_bad_region_span_fraction'] > 0.60
    assert metrics['long_repeat_broad_burden']


def test_parse_seed_list_rejects_empty_input():
    with pytest.raises(ValueError):
        hybrid_pipeline.parse_seed_list(' , ')


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
    assert "DEFAULT_WEIGHTS = fitness_weights_to_notebook_weights(" in joined
    assert "preset_weights_for_host(DEFAULT_OPTIMIZATION_PRESET, 'hsapiens')" in joined
    assert "preset_dropdown = Dropdown(" in joined
    assert "preset_section = VBox([" in joined
    assert "preset_dropdown.observe(on_preset_change, names = 'value')" in joined
    assert "selected_optimization_preset()" in joined
    assert "active_fitness_weights(host)" in joined
    assert "multi_seed_regulatory_ga_polish(" in joined
    assert "regulatory_candidate_rank(r.quality_metrics, r.fitness)" in joined
    assert "preset_weights_for_host(preset.name, host)" in joined
    assert "preset_dropdown.value == CUSTOM_PRESET_NAME" in joined
    assert "calculate_codon_pair_score" in joined
    assert "calculate_tai" in joined
    assert "Final codon-pair score:" in joined
    assert "Final tAI:" in joined
    assert "value = 'structure'," in joined
    assert "value = 'hybrid'," in joined
    assert "on_pipeline_change({'new': pipeline_dropdown.value})" in joined
    assert "on_codon_method_change({'new': codon_method_dropdown.value})" in joined
    assert "refresh_preset_dropdown_options()" in joined
