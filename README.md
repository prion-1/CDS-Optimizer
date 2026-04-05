# CDS Optimizer
#TODO inspect readme!

A Python toolkit for optimizing coding sequences (CDS) for heterologous expression. Given an input DNA or protein sequence, CDS Optimizer redesigns synonymous codon usage to optimize expression in a chosen host organism and make the sequence easier
to work with, reducing issues during molecular cloning.

---

## Overview

Recombinant protein expression often suffers from suboptimal codon usage: the source organism's preferred codons may be rare in the expression host, causing ribosome stalling, frameshifts, or low yield. This tool addresses that by applying a two-phase optimization pipeline:

1. **Pre-optimization** — deterministic seeding via one of three strategies (see below)
2. **Genetic algorithm refinement** — stochastic search guided by a multi-objective fitness function

The result is a synonymous CDS tuned for the target host's codon preferences, GC content, mRNA stability, repeat avoidance, and other expression-relevant properties.

---

## Supported Host Organisms

| Key | Organism |
|-----|----------|
| `hsapiens` | *Homo sapiens* |
| `mmusculus` | *Mus musculus* |
| `ecoli` | *Escherichia coli* |
| `scerevisiae` | *Saccharomyces cerevisiae* |
| `spombe` | *Schizosaccharomyces pombe* |

Host-specific codon usage tables are stored as CSV files in [data/codon_tables/](data/codon_tables/).

#TODO expand the supported organisms by whatever is in the folder

---

## Project Structure

```
CDS Optimizer/
├── main.ipynb                 # Interactive Jupyter notebook interface
├── src/
│   ├── utils.py               # Core sequence utilities and metrics
│   ├── pre_optimization.py    # Phase 1: deterministic seeding strategies
│   ├── optimization.py        # Phase 2: genetic algorithm
│   ├── gceh_module.py         # Sequence analysis and visualization (GCEH)
│   └── complexity_analysis.py # Sliding-window sequence complexity scoring
├── data/
│   └── codon_tables/
│       ├── hsapiens.csv
│       ├── mmusculus.csv
│       ├── ecoli.csv
│       ├── scerevisiae.csv
│       └── spombe.csv
└── requirements.txt
```

---

## Installation

#TODO include wrapper on github

**Prerequisites:** Python 3.10+

```bash
git clone https://github.com/prion-1/CDS-Optimizer.git
cd "CDS Optimizer"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Optional dependencies

| Package | Effect if absent |
|---------|-----------------|
| `ViennaRNA` | Falls back to a GC-content heuristic for mRNA folding energy |
| `numba` | Falls back to pure Python (slower for long sequences) |

ViennaRNA must be installed separately.

---

## Usage

### Interactive notebook

Open `main.ipynb` in JupyterLab, VS Code, or any compatible environment. The notebook provides widget-based controls for sequence input, host selection, optimization strategy, and parameter tuning, and renders comparative plots of the input vs. optimized sequence.

---

## Optimization Pipeline

### Phase 1 — Pre-optimization (seeding)

Three deterministic strategies are available via `src/pre_optimization.py`. The output is used to seed the genetic algorithm's initial population.

#### Maximum CAI (`simple`)

Replaces every codon with the single most-frequent synonymous codon in the host. This produces the highest possible Codon Adaptation Index — a measure of how closely the codon usage matches highly expressed host genes.

The result is a sequence where every amino acid is encoded by the one codon the host translates fastest. This is the simplest strategy and gives a strong CAI baseline, but it has trade-offs: the output has no codon diversity within each amino acid family (e.g. every Leucine becomes CTG in human), which can introduce long homopolymer runs or GC-biased stretches that the GA must clean up in Phase 2. It also flattens any translational pausing that may be needed for co-translational folding.

Best suited for: short, structurally simple proteins where maximum translation speed is desired and folding fidelity is not a concern.

#### Percentile Matching (`percentile`)

Many proteins fold co-translationally: the ribosome pauses at rare codons, giving emerging domains time to adopt their native structure before the next segment is synthesized. Replacing every codon with the host's fastest synonym (Maximum CAI) destroys these pausing patterns, which can cause misfolding. Percentile Matching preserves the *relative* speed profile by mapping each codon's frequency rank in the source to an equivalent rank in the target host.

Two modes are available, selectable via the notebook UI:

**Simple percentile matching (source host-agnostic)**

For each amino acid, the algorithm counts how often each synonymous codon appears *within the input CDS*, sorts by frequency, and assigns midpoint percentiles. It does the same for the target host's codon table, then maps each CDS codon to the host codon with the nearest percentile. The most-used CDS codon maps to the most-used host codon; the rarest to the rarest.

This works well for naturally expressed genes, where CDS-internal frequencies correlate with the source organism's genome-wide preferences. It can misfire for short genes (noisy counts), synthetic constructs, or sequences already optimized for a different host.

**Source host-informed percentile matching**

When the source organism is known, this mode ranks codons against the *source organism's codon usage table* instead of the CDS-internal counts. For each amino acid, both the source and target host codons are sorted by their table frequencies and assigned percentiles; each input codon is then mapped from its source percentile to the nearest target percentile. This answers the biological question directly — "how fast was this position relative to what the source organism considers fast?" — and is more reliable for short, synthetic, or previously optimized sequences.

In the notebook, select "Source host-informed percentile matching" from the percentile mode dropdown to reveal a source organism selector (populated from `data/codon_tables/`).

Best suited for: genes encoding proteins with complex domain architecture, known co-translational folding requirements, or where Maximum CAI produces insoluble/aggregated protein. Use the source-informed mode whenever the source organism is known.

#### FlowRamp (`flowramp`)

A composite strategy that combines a translational ramp, codon-pair optimization, and local pathology avoidance in a single greedy pass. Unlike Maximum CAI (which maximizes speed everywhere) or Percentile Matching (which preserves the source rhythm), FlowRamp constructs a new speed profile from scratch, designed around the biophysics of translation initiation.

- **Translational ramp:** The first ~8–30 codons (scaled to gene length) use progressively faster codons, starting from the least-preferred synonym at position 1 and ramping up to near-optimal by the end. This reduces ribosome pile-ups at the start site: early ribosomes translate slowly and clear the initiation region before the next ribosome loads, preventing collisions that stall translation globally.
- **Codon-pair scoring:** Each candidate codon is scored for compatibility with its predecessor. Pairs of rare codons are penalized (ribosome stalling cascades); common-common pairs are rewarded. The 4-mer junction between adjacent codons (last 2 nt of codon *i* + first 2 nt of codon *i+1*) is checked against host-specific blacklists: CpG suppression for vertebrates, Shine-Dalgarno-like seeds for prokaryotes, splice-donor motifs for eukaryotes. Exact codon repeats (e.g. CTG-CTG) are mildly penalized to reduce frameshift risk.
- **Local pathology avoidance:** Candidate codons that would create homopolymer runs (>=4 nt, e.g. AAAA) or alternating dinucleotide repeats (>=3 units, e.g. ATATAT) are skipped during construction.
- **Early wobble bias:** The first ~12 codons prefer A/T at the wobble (third) position, reducing local GC content and secondary structure potential near the 5' end to keep the ribosome binding site accessible.

Best suited for: general-purpose optimization where no source organism information is available and the goal is a well-rounded sequence with both good CAI and favorable translational initiation properties.

### How Phase 1 and Phase 2 relate

Phase 1 and Phase 2 optimize different things and their strengths are complementary:

**Phase 1 controls codon-level strategy.** Each strategy encodes a specific design philosophy — maximum speed (MC), rhythm preservation (PM), or ramp-based initiation engineering (FlowRamp).

**Phase 2 optimizes global sequence properties.** The GA's fitness function evaluates aggregate metrics: overall CAI, GC content, folding energy, repeat density, motif counts. It has no explicit pressure to preserve the codons chosen by Phase 1. This means the GA treats the Phase 1 output as a starting point, not a constraint.

**What survives the GA, and what doesn't.** Global properties that Phase 1 contributes — reasonable GC content, moderate codon diversity, absence of gross repeats — tend to persist because they overlap with what the fitness function rewards. Position-specific features — PM's translational rhythm, FlowRamp's ramp gradient — erode over generations as the GA mutates slow codons toward faster ones (improving CAI). With the default GA settings (modest population and generation counts), the erosion is partial; with aggressive settings it can be substantial.

**Phase 1 as convergence accelerator.** The GA's initialization creates variants of the Phase 1 sequence at 10–40% mutation rates, with replacement codons weighted by host codon frequencies. Even from a poor seed, the most-mutated variants are already biased toward decent host codons. A good Phase 1 seed doesn't change *where* the GA converges — the fitness function determines that — but it gets there faster, which matters when the GA budget is limited. A bad seed costs a few percent of final fitness, not a catastrophic failure.

**When to use Phase 1 output directly.** The notebook displays the Phase 1 (pre-optimized) sequence alongside the final GA output. If position-specific codon choices matter for your use case — co-translational folding fidelity (PM) or a controlled initiation ramp (FlowRamp) — consider using the Phase 1 sequence directly and skipping or limiting the GA. The Phase 1 output will have lower composite fitness (more repeats, less optimal GC) but preserves the design intent that the GA would otherwise erode.

### Phase 2 — Genetic algorithm refinement

The `GeneticAlgorithm` class in `src/optimization.py` uses the Phase 1 output as seed for evolutionary search.


**Parameters (key defaults):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pop_size` | 80 | Population size |
| `generations` | 200 | Maximum generations |
| `mutation_rate` | 0.02 | Per-codon mutation probability |
| `crossover_rate` | 0.7 | Crossover probability |
| `tournament_size` | 5 | Tournament selection pool |
| `elitism` | 2 | Elite individuals carried forward each generation |

The algorithm converges automatically if the best fitness does not improve by more than 0.0001 over 20 consecutive generations.

**Genetic operators:**
- **Initialization:** Population seeded from the Phase 1 sequence with additional variants generated at mutation rates of 10–40%.
- **Selection:** Tournament selection.
- **Crossover:** Uniform crossover (codon-level); only synonymous swaps are accepted.
- **Mutation:** Synonymous codon substitution weighted by host codon table frequencies.
- **Validation:** Every offspring is translated and rejected if the protein sequence changes.

#TODO hybrid "phase 3" implementation???????

---

## Fitness Function

The composite fitness score drives the genetic algorithm. It is a weighted sum of normalized metrics:

```
fitness = w_cai × CAI
        − w_gc  × |GC − target_GC| / 100
        − w_fold × folding_penalty
        − w_motif × motif_penalty
        − w_rep  × repeat_penalty
        − w_splice × splice_penalty
        − w_gc3  × |GC3 − target_GC3| / 100
        − w_atg  × atg_penalty
        + w_acc  × accessibility
```

**Default weights:**

| Metric | Weight | Notes |
|--------|--------|-------|
| CAI | 0.40 | Codon Adaptation Index (geometric mean of relative adaptiveness) |
| GC deviation | 0.20 | Deviation from host target GC (Kazusa coding-seq values) |
| Folding energy | 0.15 | 5′ mRNA stability; penalizes overly stable 150 nt window |
| Unwanted motifs | 0.10 | User-specified sequences + default polyA signals (AATAAA, ATTAAA, AAAAAA) |
| Repeats | 0.07 | Windowed homopolymer + dinucleotide repeat scoring |
| Cryptic splice sites | 0.08 | GT[AG]AG donor-like + polypyrimidine-AG acceptor-like patterns (eukaryotes only) |
| GC3 deviation | 0.00 | Optional; useful for yeast (prefers AT at 3rd position) |
| Internal ATG | 0.00 | Optional; penalizes upstream ORFs in the 5′ 90 nt |
| 5′ accessibility | 0.00 | Optional; rewards open secondary structure near start codon |

Weights must sum to 1.0 (a warning is printed otherwise). All non-CAI components are normalized to [0, 1] before weighting.

### Repeat scoring

`count_repetitive_sequences()` applies a smooth length-dependent penalty to homopolymers (≥4 nt) and alternating dinucleotide runs (≥6 nt). `repeat_penalty_windowed()` runs this over overlapping 300 nt windows (step 150 nt) and returns mean, max, and the fraction of windows exceeding a threshold — preventing long-sequence dilution of local repeat hotspots.

---

## Sequence Analysis (GCEH)

`src/gceh_module.py` provides standalone analysis independent of the optimizer:

- Per-codon usage frequency bar charts for input vs. reference host
- GC content track (sliding window)
- Sequence complexity track (Shannon entropy, GC balance, k-mer richness, homopolymer penalty, periodicity — see `src/complexity_analysis.py`)
- Side-by-side comparison plots for input vs. optimized sequences

These are rendered automatically by `main.ipynb` after optimization.

---

## Codon Table Format

CSV files in `data/codon_tables/` follow this schema:

#TODO maybe include also species name and eukaryote/prokaryote in csv for GUI population. also think about how to deal with target GC. MAYBE DIRECTLY INCLUDE IN CSV??????????????? <----------------------

```
amino_acid,codon,frequency
Ala,GCT,0.26
Ala,GCC,0.40
...
```

Frequencies are absolute usage frequencies per amino acid (not normalized). The loader normalizes them internally to relative adaptiveness values (0–1) for use in CAI calculation.

#TODO ^^^^^^^^^^^not anymore, correct this^^^^^^^^^^^^

---

## Reproducibility

Pass `random_seed` to `genetic_algorithm()` to get deterministic results:

```python
optimized, fitness, metrics = genetic_algorithm(
    initial_cds=cds,
    host="hsapiens",
    random_seed=42,
)
```

---


## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations, numba interop |
| `pandas` | Codon table loading |
| `matplotlib`, `seaborn` | Plotting |
| `scipy` | Statistical utilities |
| `ipywidgets` | Notebook UI widgets |
| `numba` | JIT-accelerated GC/CAI/motif functions |
| `biopython` | Sequence I/O |
| `tabulate` | Tabular output formatting |
| `ViennaRNA` | RNA secondary structure (optional) |
