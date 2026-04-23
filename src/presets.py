"""
Named optimization preset definitions.

These helpers provide the code-level preset layer that a future UI dropdown can
call directly. They intentionally do not run optimization and do not change the
current default pipeline behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

from .hybrid_pipeline import (
    EXTREME_REPEAT_GUARD_POLISH_WEIGHTS,
    REGULATORY_MOTIF_HIGH_POLISH_WEIGHTS,
)
from .utils import codon_pair_available_for, tai_available_for


FITNESS_WEIGHT_KEYS = (
    'cai',
    'gc_deviation',
    'folding_energy',
    'unwanted_motifs',
    'repeats',
    'cryptic_splice',
    'gc3_deviation',
    'internal_atg',
    'accessibility',
    'tai',
    'codon_pair_bias',
)

PRESET_SELECTOR_GENERIC = 'generic_ga'
PRESET_SELECTOR_REGULATORY = 'regulatory_post_selection'

PRESET_REPEAT_GUARD = 'repeat_guard'
PRESET_EXPRESSION_CONSERVATIVE = 'expression_conservative'
PRESET_SECONDARY_STRUCTURE = 'secondary_structure'
PRESET_SECONDARY_ACCESSIBILITY = 'secondary_accessibility'
PRESET_REGULATORY_MOTIF_HIGH = 'regulatory_motif_high'
PRESET_REGULATORY_CONSTRUCT_SAFE = 'regulatory_construct_safe'
DEFAULT_OPTIMIZATION_PRESET = PRESET_REPEAT_GUARD

NOTEBOOK_WEIGHT_KEY_MAP = {
    'cai': 'cai',
    'gc': 'gc_deviation',
    'gc3': 'gc3_deviation',
    'folding': 'folding_energy',
    'motifs': 'unwanted_motifs',
    'repeats': 'repeats',
    'splice': 'cryptic_splice',
    'atg': 'internal_atg',
    'accessibility': 'accessibility',
    'cps': 'codon_pair_bias',
    'tai': 'tai',
}
FITNESS_TO_NOTEBOOK_WEIGHT_KEY_MAP = {
    fitness_key: notebook_key
    for notebook_key, fitness_key in NOTEBOOK_WEIGHT_KEY_MAP.items()
}


@dataclass(frozen=True)
class OptimizationPreset:
    """Code-level preset metadata and GA fitness weights."""

    name: str
    family: str
    label: str
    ui_label: str
    selector: str
    weights: Dict[str, float]
    description: str

    def weights_dict(self) -> Dict[str, float]:
        """Return a mutable copy suitable for optimization calls."""

        return dict(self.weights)


def normalize_preset_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Return a full normalized fitness-weight dictionary."""

    full = {key: float(weights.get(key, 0.0)) for key in FITNESS_WEIGHT_KEYS}
    total = sum(full.values())
    if total <= 0:
        raise ValueError('Preset weights must have a positive total.')
    return {key: value / total for key, value in full.items()}


EXPRESSION_CONSERVATIVE_POLISH_WEIGHTS = normalize_preset_weights({
    'cai': 0.30,
    'gc_deviation': 0.14,
    'folding_energy': 0.06,
    'unwanted_motifs': 0.05,
    'repeats': 0.30,
    'cryptic_splice': 0.03,
    'tai': 0.06,
    'codon_pair_bias': 0.06,
})

SECONDARY_STRUCTURE_POLISH_WEIGHTS = normalize_preset_weights({
    'cai': 0.09,
    'gc_deviation': 0.15,
    'folding_energy': 0.34,
    'unwanted_motifs': 0.04,
    'repeats': 0.30,
    'cryptic_splice': 0.04,
    'tai': 0.02,
    'codon_pair_bias': 0.02,
})

SECONDARY_ACCESSIBILITY_POLISH_WEIGHTS = normalize_preset_weights({
    'cai': 0.13,
    'gc_deviation': 0.18,
    'folding_energy': 0.28,
    'unwanted_motifs': 0.04,
    'repeats': 0.21,
    'cryptic_splice': 0.04,
    'accessibility': 0.08,
    'tai': 0.02,
    'codon_pair_bias': 0.02,
})

REGULATORY_CONSTRUCT_SAFE_POLISH_WEIGHTS = normalize_preset_weights({
    'cai': 0.10,
    'gc_deviation': 0.08,
    'folding_energy': 0.04,
    'unwanted_motifs': 0.22,
    'repeats': 0.32,
    'cryptic_splice': 0.20,
    'internal_atg': 0.04,
})


OPTIMIZATION_PRESETS: Tuple[OptimizationPreset, ...] = (
    OptimizationPreset(
        name=PRESET_REPEAT_GUARD,
        family='repeat_guard',
        label='current_extreme_repeat_guard',
        ui_label='Repeat guard',
        selector=PRESET_SELECTOR_GENERIC,
        weights=normalize_preset_weights(EXTREME_REPEAT_GUARD_POLISH_WEIGHTS),
        description='Default repeat/degen-safety preset from the PINK1 sweeps.',
    ),
    OptimizationPreset(
        name=PRESET_EXPRESSION_CONSERVATIVE,
        family='cai_tai_cps',
        label='expression_conservative',
        ui_label='Expression conservative',
        selector=PRESET_SELECTOR_GENERIC,
        weights=EXPRESSION_CONSERVATIVE_POLISH_WEIGHTS,
        description='Conservative CAI/tAI/codon-pair emphasis with repeat pressure retained.',
    ),
    OptimizationPreset(
        name=PRESET_SECONDARY_STRUCTURE,
        family='secondary',
        label='secondary_fold34_repeat30',
        ui_label='Secondary structure',
        selector=PRESET_SELECTOR_GENERIC,
        weights=SECONDARY_STRUCTURE_POLISH_WEIGHTS,
        description='Stable folding-biased preset with stronger repeat guard.',
    ),
    OptimizationPreset(
        name=PRESET_SECONDARY_ACCESSIBILITY,
        family='secondary',
        label='secondary_accessibility_anchor',
        ui_label='Secondary accessibility',
        selector=PRESET_SELECTOR_GENERIC,
        weights=SECONDARY_ACCESSIBILITY_POLISH_WEIGHTS,
        description='Accessibility-biased secondary preset for stronger 5 prime access pressure.',
    ),
    OptimizationPreset(
        name=PRESET_REGULATORY_MOTIF_HIGH,
        family='regulatory',
        label='regulatory_motif_high',
        ui_label='Regulatory motif high',
        selector=PRESET_SELECTOR_REGULATORY,
        weights=normalize_preset_weights(REGULATORY_MOTIF_HIGH_POLISH_WEIGHTS),
        description='Primary regulatory preset for motif/splice cleanup.',
    ),
    OptimizationPreset(
        name=PRESET_REGULATORY_CONSTRUCT_SAFE,
        family='regulatory',
        label='regulatory_construct_safe',
        ui_label='Regulatory construct safe',
        selector=PRESET_SELECTOR_REGULATORY,
        weights=REGULATORY_CONSTRUCT_SAFE_POLISH_WEIGHTS,
        description='Regulatory backup with stronger repeat/constructability pressure.',
    ),
)


def list_optimization_presets(
    *,
    selector: Optional[str] = None,
) -> Tuple[OptimizationPreset, ...]:
    """Return named optimization presets, optionally filtered by selector."""

    if selector is None:
        return OPTIMIZATION_PRESETS

    normalized = selector.strip().lower()
    if normalized not in {PRESET_SELECTOR_GENERIC, PRESET_SELECTOR_REGULATORY}:
        raise ValueError(
            f'Unknown preset selector {selector!r}. '
            f'Expected {PRESET_SELECTOR_GENERIC!r} or {PRESET_SELECTOR_REGULATORY!r}.'
        )
    return tuple(
        preset
        for preset in OPTIMIZATION_PRESETS
        if preset.selector == normalized
    )


def get_optimization_preset(name: str) -> OptimizationPreset:
    """Return a named optimization preset."""

    normalized = name.strip().lower()
    for preset in OPTIMIZATION_PRESETS:
        if preset.name == normalized:
            return preset
    valid = ', '.join(preset.name for preset in OPTIMIZATION_PRESETS)
    raise ValueError(f'Unknown optimization preset {name!r}. Expected one of: {valid}.')


def preset_weight_profiles() -> Dict[str, Dict[str, float]]:
    """Return preset weights keyed by preset name."""

    return {preset.name: preset.weights_dict() for preset in OPTIMIZATION_PRESETS}


def notebook_weights_to_fitness_weights(
    weights: Mapping[str, float],
) -> Dict[str, float]:
    """Translate notebook slider weights to fitness-function weight keys."""

    translated = {key: 0.0 for key in FITNESS_WEIGHT_KEYS}
    for notebook_key, fitness_key in NOTEBOOK_WEIGHT_KEY_MAP.items():
        translated[fitness_key] = float(weights.get(notebook_key, 0.0))
    return translated


def fitness_weights_to_notebook_weights(
    weights: Mapping[str, float],
) -> Dict[str, float]:
    """Translate fitness-function weight keys to notebook slider keys."""

    translated = {key: 0.0 for key in NOTEBOOK_WEIGHT_KEY_MAP}
    for fitness_key, notebook_key in FITNESS_TO_NOTEBOOK_WEIGHT_KEY_MAP.items():
        translated[notebook_key] = float(weights.get(fitness_key, 0.0))
    return translated


def adapt_fitness_weights_for_capabilities(
    weights: Mapping[str, float],
    *,
    tai_available: bool = True,
    codon_pair_available: bool = True,
) -> Dict[str, float]:
    """
    Zero unsupported optional axes and renormalize the remaining envelope.

    This keeps named presets internally consistent on hosts that do not ship
    tAI or codon-pair support tables.
    """

    adapted = {key: float(weights.get(key, 0.0)) for key in FITNESS_WEIGHT_KEYS}
    if not tai_available:
        adapted['tai'] = 0.0
    if not codon_pair_available:
        adapted['codon_pair_bias'] = 0.0
    return normalize_preset_weights(adapted)


def preset_weights_for_host(name: str, host: str) -> Dict[str, float]:
    """Return host-aware preset weights with unsupported optional axes removed."""

    preset = get_optimization_preset(name)
    return adapt_fitness_weights_for_capabilities(
        preset.weights,
        tai_available=tai_available_for(host),
        codon_pair_available=codon_pair_available_for(host),
    )


__all__ = [
    'FITNESS_WEIGHT_KEYS',
    'PRESET_SELECTOR_GENERIC',
    'PRESET_SELECTOR_REGULATORY',
    'PRESET_REPEAT_GUARD',
    'PRESET_EXPRESSION_CONSERVATIVE',
    'PRESET_SECONDARY_STRUCTURE',
    'PRESET_SECONDARY_ACCESSIBILITY',
    'PRESET_REGULATORY_MOTIF_HIGH',
    'PRESET_REGULATORY_CONSTRUCT_SAFE',
    'DEFAULT_OPTIMIZATION_PRESET',
    'NOTEBOOK_WEIGHT_KEY_MAP',
    'FITNESS_TO_NOTEBOOK_WEIGHT_KEY_MAP',
    'OptimizationPreset',
    'normalize_preset_weights',
    'EXPRESSION_CONSERVATIVE_POLISH_WEIGHTS',
    'SECONDARY_STRUCTURE_POLISH_WEIGHTS',
    'SECONDARY_ACCESSIBILITY_POLISH_WEIGHTS',
    'REGULATORY_CONSTRUCT_SAFE_POLISH_WEIGHTS',
    'OPTIMIZATION_PRESETS',
    'list_optimization_presets',
    'get_optimization_preset',
    'preset_weight_profiles',
    'notebook_weights_to_fitness_weights',
    'fitness_weights_to_notebook_weights',
    'adapt_fitness_weights_for_capabilities',
    'preset_weights_for_host',
]
