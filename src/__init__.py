"""
mRNA Optimization Tool — source package.

Exposes the public API used by main.ipynb and external scripts. Internal
modules (e.g. complexity_analysis, gceh_module) are imported on demand.
"""

from .optimization import genetic_algorithm, FitnessWeights, GeneticAlgorithm
from .pre_optimization import (
    optimize_codons,
    simple_best_codon_optimization,
    percentile_matching_optimization,
    structure_optimized_beam_search,
)
from .local_repair import local_repair
from .utils import (
    translate_dna_to_protein,
    back_translate_protein,
    calculate_cai,
    calculate_gc_content,
    calculate_gc3_content,
    load_codon_table,
    list_available_hosts,
    is_eukaryote_host,
    get_target_gc,
    get_target_gc3,
    harmonization_correlation,
    compute_profile_deviation,
    compute_profile_correlation,
    calculate_codon_pair_score,
    calculate_tai,
    tai_available_for,
    codon_pair_available_for,
)

__version__ = "2.0.0"
__all__ = [
    'genetic_algorithm',
    'FitnessWeights',
    'GeneticAlgorithm',
    'optimize_codons',
    'simple_best_codon_optimization',
    'percentile_matching_optimization',
    'structure_optimized_beam_search',
    'local_repair',
    'translate_dna_to_protein',
    'back_translate_protein',
    'calculate_cai',
    'calculate_gc_content',
    'calculate_gc3_content',
    'load_codon_table',
    'list_available_hosts',
    'is_eukaryote_host',
    'get_target_gc',
    'get_target_gc3',
    'harmonization_correlation',
    'compute_profile_deviation',
    'compute_profile_correlation',
    'calculate_codon_pair_score',
    'calculate_tai',
    'tai_available_for',
    'codon_pair_available_for',
]
