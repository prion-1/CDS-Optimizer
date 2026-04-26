# CDS Optimizer

CDS Optimizer is a Python toolkit and notebook interface for synonymous coding-sequence redesign for heterologous expression. It combines deterministic codon redesign with either a multi-objective genetic algorithm **(GA)** or a targeted local-repair pass, then visualizes codon usage, GC/GC3, sequence complexity, and harmonization diagnostics.

## What It Does

Given a DNA/RNA CDS or a protein sequence, the repository can:

- clean and validate nucleotide input, assuming the first input base is codon position 1, converting RNA to DNA, trimming incomplete trailing bases, and removing the first in-frame stop codon plus anything after it.
- back-translate protein to DNA with host-preferred codons through the core API.
- redesign synonymous codons for a target host.
- run deterministic preoptimization alone, preoptimization plus a genetic algorithm (GA), or a hybrid local-repair workflow with optional light GA polish.
- apply GA preset profiles for repeat control, expression-oriented scoring, secondary-structure handling, and regulatory cleanup
- optimize or score CAI, GC balance, GC3 balance, 5' folding heuristics, 5' accessibility proxies, motif avoidance, repeat avoidance, cryptic splice-site avoidance, internal ATG count, tAI, and empirical codon-pair bias where host data are bundled
- preserve protein identity and the exact input start codon across preoptimization, GA, and local repair, which matters for alternative initiators such as bacterial `GTG`
- compare input vs. optimized sequences with codon-usage, GC, GC3, complexity, CHARMING-style harmonization, and regulatory-cleanup readouts

The bundled host set is discovered dynamically from `data/codon_tables/*.csv`:

| Key | Organism | Empirical CPS data | tAI data |
|-----|----------|--------------------|----------|
| `ecoli` | *Escherichia coli* | yes | yes |
| `hsapiens` | *Homo sapiens* | yes | yes |
| `mmusculus` | *Mus musculus* | yes | yes |
| `scerevisiae` | *Saccharomyces cerevisiae* | yes | yes |
| `spombe` | *Schizosaccharomyces pombe* | yes | yes |

Dropping a new `<host>.csv` file into `data/codon_tables/` makes the host available to the loader. For notebook use, also add the host to `HOST_KINGDOM` in `src/utils.py` so the GUI knows whether to apply eukaryotic splice-site scoring.


> [!tip]
> Launch the workbench via the binder below!

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/prion-1/CDS-Optimizer/main?urlpath=%2Fdoc%2Ftree%2Fmain.ipynb)

## Pipelines

The notebook exposes three pipelines:

1. `preoptimization_only`
   Deterministic synonymous redesign only.
2. `preoptimization_ga`
   Preoptimization seed followed by a genetic algorithm.
3. `hybrid`
   Preoptimization seed followed by local repair, with optional light GA polish.

In practice:

- `preoptimization_only` preserves the deterministic preoptimization design exactly.
- `preoptimization_ga` runs the GA after preoptimization and exposes generic GA presets plus fitness-weight sliders.
- `hybrid` is the conservative path when you want to keep most preoptimization decisions while still cleaning local pathologies; if GA polish is enabled, it exposes GA presets and weights for the polish step.

The notebook GUI exposes protein input, direct DNA/RNA input, target host selection, preoptimization method selection, percentile-matching source-host options, avoid-motif input, GA parameters, local-repair parameters, optional GA polish settings, named GA presets, fitness-weight sliders, complexity-plot settings, and progress reporting.

## GA Presets and Selection Modes

`src/presets.py` defines the current named GA weight profiles:

| Preset | Selector | Current role |
|--------|----------|--------------|
| `repeat_guard` | generic GA | Default repeat/degeneracy-safe profile |
| `expression_conservative` | generic GA | CAI/tAI/CPS-oriented profile with repeat pressure retained |
| `secondary_structure` | generic GA | Stronger 5' folding penalty with repeat guard |
| `secondary_accessibility` | generic GA | Secondary-structure profile with explicit 5' accessibility pressure near the start region |
| `regulatory_motif_high` | regulatory post-selection | Aggressive motif/splice/internal-ATG cleanup profile |
| `regulatory_construct_safe` | regulatory post-selection | Regulatory cleanup with stronger repeat/GC-run/tandem-repeat constructability pressure |

Generic presets are available in `preoptimization_ga` and in hybrid GA polish. Regulatory presets are only surfaced when the `hybrid` pipeline and GA polish are both enabled, because they rely on multi-seed regulatory post-selection rather than plain GA fitness ranking.

`preset_weights_for_host()` makes these presets host-aware by zeroing unsupported optional axes (`tai`, `codon_pair_bias`) and renormalizing the remaining weights.

### How The Presets Were Developed

The current preset layer was selected empirically around candidate quality, not
derived from a closed-form model. The final profiles represent useful operating
points:

- `repeat_guard` is the default conservative profile. It deliberately gives
  strong weight to repeat and degeneracy control so GC-only runs, tandem
  repeats, homopolymers, and low-complexity windows are penalized even when that
  means backing off raw CAI.
- `expression_conservative` restores more expression-oriented pressure through
  CAI, tAI, and codon-pair bias while retaining repeat pressure.
- `secondary_structure` emphasizes the 5' folding penalty while keeping repeat
  control active.
- `secondary_accessibility` is the start-region accessibility variant. It still
  cares about folding, but also rewards a more accessible 5' region through the
  explicit accessibility term.
- `regulatory_motif_high` is the more aggressive regulatory cleanup profile for
  unwanted motifs, cryptic-splice-like motifs, and internal ATGs.
- `regulatory_construct_safe` is the less motif-maximal regulatory backup. It
  keeps regulatory cleanup active but shifts more pressure toward
  constructability: repeat burden, GC-only runs, tandem repeats, and severe
  local sequence pathologies.

A GA seed is the integer that initializes the GA's random choices. Multi-seed
polish runs several reproducible GA attempts from the same starting sequence
and then selects the best result by sequence-quality criteria.

In other words, the preset layer is a set of operational envelopes built around
observed failure modes in long CDS optimization: a repeat-safe default, a few
controlled generic variants, and a separate regulatory branch with its own
post-selection logic.

## Preoptimization Methods

Preoptimization methods live in `src/pre_optimization.py`.

### `simple`

Replace each codon with the most host-preferred synonymous codon. This is the max-CAI baseline: strong host adaptation, low codon diversity, and a tendency to create GC or repeat artifacts that later stages may need to clean up.

### `percentile`

Percentile-matched harmonization. The method preserves relative codon-usage rank rather than maximizing speed everywhere.

Two modes are implemented:

- source-agnostic: infer ranks from the input CDS itself
- source-informed: rank codons against a specified source-host table

This is the best choice when preserving relative translational rhythm matters more than raw CAI. The notebook exposes both modes.

### `structure`

Structure-aware heuristic beam search. At each codon position, the method explores synonymous codons and scores partial sequences by a weighted combination of codon adaptiveness and local folding-energy heuristics. It uses ViennaRNA when available and falls back to a deterministic GC-based MFE approximation when it is not.

Supported method names are:

- `simple`
- `percentile`
- `structure`

## Optimization and Repair

### Genetic algorithm

The GA in `src/optimization.py` optimizes a weighted objective. The fitness function supports terms for:

- CAI
- GC deviation
- GC3 deviation
- 5' folding penalty
- 5' accessibility proxy
- unwanted motif count
- repeat penalty
- cryptic splice-site penalty
- internal ATG count
- tAI
- empirical codon-pair bias

The notebook exposes sliders for CAI, GC content, folding, motifs, repeats, splice sites, codon-pair bias, and tAI wherever GA weights are active: `preoptimization_ga`, and `hybrid` when optional GA polish is enabled. The codon-pair and tAI sliders are hidden and reset to zero when the selected host has no backing data. GC3, accessibility, and internal-ATG terms are available in the Python API and default to zero. GC3 targets are configured per host in `HOST_TARGET_GC3`; hosts without an explicit entry derive their GC3 target from the raw codon-frequency table.
Codon-pair bias is a GA fitness term, so it is active in `preoptimization_ga` and in
the optional GA polish step of `hybrid`; pure preoptimization and pure local
repair do not optimize CPS directly.

For repeat handling, the GA API also exposes `repeat_penalty_mode`. The current default is `legacy`, with `blend` and `aligned` available for stricter degeneracy-aware repeat scoring through the Python API.

Protein identity is enforced, and the exact input start codon is pinned so optimization does not silently rewrite alternative initiators such as `GTG` while preserving the same translated amino acid. Per-host codon tables and score vectors are cached for repeated fitness evaluation.

### Local repair

`src/local_repair.py` scans overlapping windows and applies synonymous substitutions only where a local problem is detected. It looks for homopolymers, dinucleotide repeats, unwanted motifs, cryptic splice sites, and local GC drift. It is intentionally conservative: each candidate change is penalized for deviating from the preoptimization codon choice, the start codon is preserved verbatim, and the final sequence is checked for protein identity.

The current default local-repair setting is `window_nt=36`,
`max_subs_per_window=3`, and `gc_tolerance=5.0`, tuned for short local edits
that remove GC-rich degenerate stretches while preserving CAI and host GC
balance. Optional hybrid GA polish is treated as a multi-seed search:
`src.hybrid_pipeline.multi_seed_ga_polish()` runs the same repaired sequence
through the default seeds `(1701, 2701, 3701, 4701, 5701)`, computes
degeneracy-aware repeat/complexity quality metrics, and selects the best seed
by those quality criteria rather than by GA fitness alone. The default polish
weights use the repeat-guard profile.

For regulatory cleanup, `src.hybrid_pipeline.multi_seed_regulatory_ga_polish()`
uses the same GA machinery but ranks seeds with raw motif, splice-site, and
internal-ATG cleanup metrics plus relaxed constructability gates instead of the
generic repeat-guard selector.

## Analysis Layer

`src/gceh_module.py`, `src/complexity_analysis.py`, and analysis helpers in `src/utils.py` provide:

- codon-usage tables and plots
- input-vs-optimized codon-usage comparison plots
- cumulative GC3 plots
- sliding-window GC and GC3 plots
- sliding-window complexity tracks with entropy, GC balance, k-mer richness, homopolymer, periodicity, and invalid-base components
- degeneracy-aware repeat-burden and low-complexity quality metrics for hybrid seed selection
- programmatic CHARMING-style windowed CAI and `%MinMax` profile calculations. `%MinMax` compares each local codon-usage window to the fastest and slowest synonymous alternatives for the same amino-acid sequence, producing a relative speed profile for harmonization analysis
- notebook CAI harmonization diagnostics: Pearson correlation, net absolute deviation, and mean absolute deviation between input and optimized profiles


## Optional Features and Data Requirements

Some scoring axes are optional and remain disabled unless their backing dependency or data is present.

| Feature | Requirement | Behavior if absent |
|---------|-------------|--------------------|
| tAI scoring | Bundled for all current hosts; add `data/trna_weights/<host>.csv` with complete 61-sense-codon `codon,weight` rows for more hosts | Notebook hides the tAI control for hosts without data; GA fitness uses `0.0` for tAI if requested without data |
| Codon-pair bias | Bundled for all current hosts; add `data/codon_pair_tables/<host>.csv` with `codon1,codon2,cps` or `pair,cps` columns for more hosts | Notebook hides the codon-pair control for hosts without data; GA fitness uses `0.0` if requested without data |
| ViennaRNA folding | `RNA` Python module / ViennaRNA installation | Falls back to deterministic GC/palindrome folding heuristics |
| Numba acceleration | `numba` | Falls back to pure Python implementations |

Add host-specific tAI CSV files or additional codon-pair CSV files in the documented folders to enable those scoring axes for more hosts.
The bundled tAI weights are gene-copy-number proxies derived from GtRNAdb
tRNAscan-SE predictions, not measured tRNA abundance or tissue-specific tRNA
activity.

Cryptic splice-site counting is dependency-free and heuristic-only. For
eukaryotic hosts it counts donor-like `GT[AG]AG` motifs and acceptor-like
polypyrimidine tracts ending in `AG`; prokaryotic hosts return `0`.

## Installation

Python 3.12+ is recommended for the current notebook. The core modules are regular Python modules, but the notebook is the primary user interface.

```bash
git clone https://github.com/prion-1/CDS-Optimizer.git
cd CDS-Optimizer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` includes the core runtime and notebook widget dependencies, but does not install JupyterLab itself. If your environment does not already provide a notebook server, install one:

```bash
pip install jupyterlab
jupyter lab main.ipynb
```

You can also open `main.ipynb` directly in VS Code with a Python/Jupyter kernel that has the requirements installed.

For tests:

```bash
pip install -r requirements-dev.txt
pytest
```

Optional accelerators and scientific backends live in `requirements-optional.txt`:

```bash
pip install -r requirements-optional.txt
```

## Programmatic Use

The notebook is the easiest entry point, but the same pipeline pieces can be called directly:

```python
from src.pre_optimization import optimize_codons
from src.local_repair import local_repair
from src.hybrid_pipeline import multi_seed_ga_polish
from src.input_processing import clean_nucleotide_sequence
from src.presets import PRESET_REPEAT_GUARD, preset_weights_for_host
from src.utils import get_target_gc, is_eukaryote_host

raw_cds = "ATGGCTGACTAA"
input_cds, _, messages = clean_nucleotide_sequence(raw_cds)

host = "ecoli"
is_eukaryote = is_eukaryote_host(host)

preoptimized = optimize_codons(input_cds, host, is_eukaryote, method="percentile")
repaired, changelog = local_repair(
    preoptimized,
    host,
    is_eukaryote,
    preoptimization_sequence=preoptimized,
    target_gc=get_target_gc(host),
)
polish_result = multi_seed_ga_polish(
    repaired,
    host=host,
    is_eukaryote=is_eukaryote,
    target_gc=get_target_gc(host),
    weights=preset_weights_for_host(PRESET_REPEAT_GUARD, host),
)
optimized = polish_result.best_sequence
```

For regulatory presets, use `multi_seed_regulatory_ga_polish()` with one of the
regulatory preset names from `src.presets` instead of the generic
`multi_seed_ga_polish()` selector.

## Testing

The active tests cover pipeline preservation of translation/start codons, local repair regressions, named preset behavior, generic and regulatory multi-seed selection, degeneracy metrics, complexity-track sizing, optional codon-pair behavior, host-specific GC3 targets, notebook code-cell compilation, harmonization diagnostics, and heuristic splice-site handling.
They also validate that bundled empirical codon-pair tables are complete,
finite 61 x 61 sense-codon matrices and that bundled tAI weights are complete,
positive, finite 61-sense-codon vectors.

```bash
pip install -r requirements-dev.txt
pytest
```

## Project Structure

```text
CDS Optimizer/
├── README.md
├── LICENSE
├── .gitignore
├── main.ipynb
├── scripts/
│   ├── fetch_cocoputs_codon_pair_tables.py
│   └── fetch_gtrnadb_trna_weights.py
├── src/
│   ├── __init__.py
│   ├── complexity_analysis.py
│   ├── degeneracy.py
│   ├── gceh_module.py
│   ├── hybrid_pipeline.py
│   ├── input_processing.py
│   ├── local_repair.py
│   ├── optimization.py
│   ├── pre_optimization.py
│   ├── presets.py
│   └── utils.py
├── data/
│   ├── codon_pair_tables/
│   │   ├── ecoli.csv
│   │   ├── hsapiens.csv
│   │   ├── metadata.json
│   │   ├── mmusculus.csv
│   │   ├── scerevisiae.csv
│   │   ├── spombe.csv
│   │   └── README.md
│   ├── codon_tables/
│   │   ├── ecoli.csv
│   │   ├── hsapiens.csv
│   │   ├── mmusculus.csv
│   │   ├── scerevisiae.csv
│   │   └── spombe.csv
│   └── trna_weights/
│       ├── ecoli.csv
│       ├── hsapiens.csv
│       ├── metadata.json
│       ├── mmusculus.csv
│       ├── scerevisiae.csv
│       ├── spombe.csv
│       └── README.md
├── tests/
│   ├── test_input_processing.py
│   ├── test_pipeline.py
│   └── test_regressions.py
├── requirements.txt
├── requirements-dev.txt
└── requirements-optional.txt
```

## Codon Table Format

Files in `data/codon_tables/` use rounded within-amino-acid codon frequencies:

```csv
amino_acid,codon,frequency
Ala,GCT,0.26
Ala,GCC,0.40
...
```

At runtime:

- `load_codon_frequencies()` uses these per-AA codon frequency fractions directly
- `load_codon_table()` converts them to CAI-style relative adaptiveness within each amino-acid family

Analysis, percentile matching, `%MinMax`, and host GC estimation need raw frequencies, while CAI scoring needs normalized adaptiveness.

Optional tAI files use:

```csv
codon,weight
AAA,0.31
AAC,0.85
...
```

The bundled tAI tables were derived from GtRNAdb tRNAscan-SE gene predictions:

```text
https://gtrnadb.ucsc.edu/
```

They are generated with `scripts/fetch_gtrnadb_trna_weights.py`, which fetches
host-specific GtRNAdb `*-tRNAs.tar.gz` archives, parses tRNA type and
anticodon copy counts, excludes pseudogenes and noncanonical tRNAs, and
computes dos Reis-style gene-copy tAI weights. See
`data/trna_weights/metadata.json` for exact per-host source URLs, archive
members, anticodon counts, and scoring details.

Optional codon-pair tables use either:

```csv
codon1,codon2,cps
AAA,AAC,0.42
```

or:

```csv
pair,cps
AAAAAC,0.42
```

The bundled codon-pair tables were derived from RefSeq genomic CoCoPUTs /
HIVE-CUTs codon and bicodon counts on FDA DNA HIVE, service `537`:

```text
https://dnahive.fda.gov/dna.cgi?cmd=codon_usage&id=537&mode=cocoputs
```

They are generated with `scripts/fetch_cocoputs_codon_pair_tables.py`, which
fetches `Refseq_species.tsv` codon counts and `Refseq_Bicod.tsv` bicodon
counts through the `ionTaxidCollapseExt` endpoint with `Organelle=genomic`.
See `data/codon_pair_tables/metadata.json` for the exact per-host source URLs,
taxids, source counts, and scoring details. Runtime CPS scoring skips
stop-containing codon pairs because the bundled tables cover only 61 x 61
sense-codon pairs.
