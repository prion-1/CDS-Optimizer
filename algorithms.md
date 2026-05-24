# Algorithms and Bioinformatic Principles in CDS Optimizer

A summary of the algorithmic and bioinformatic methods present in
the project. It covers the runtime modules in `src/`, the notebook-facing
pipeline behavior, and the data-builder scripts in `scripts/`.

## Main Optimization Algorithms

### Deterministic synonymous codon redesign

- **Maximum-CAI / best-codon replacement** (`src/pre_optimization.py`, `src/utils.py`)
  - Replaces each non-start codon with the highest host-adaptiveness synonymous codon.
  - Uses host codon-usage tables normalized within each amino-acid family.
  - Preserves the literal start codon and validates that the translated protein is unchanged.

- **Deterministic protein back-translation** (`src/utils.py`, `src/input_processing.py`)
  - Converts amino-acid input into DNA by selecting the best host codon for each amino acid.
  - Uses deterministic tie-breaking so repeated runs produce the same sequence.

### Percentile-matched codon harmonization

- **Source-host-informed percentile matching** (`src/pre_optimization.py`)
  - Sorts synonymous codons by source-host frequency and target-host frequency.
  - Converts codon usage into cumulative mid-percentiles.
  - Maps each source codon to the target codon with the closest percentile.
  - Bioinformatic principle: preserves relative codon-usage rank / translational rhythm instead of simply maximizing expression.

- **Source-agnostic percentile matching** (`src/pre_optimization.py`)
  - Infers codon ranks from the input CDS itself.
  - Counts codon usage within each amino-acid family, computes cumulative codon percentiles, then maps to nearest target-host percentiles.
  - Uses first observed codon position as a tie-breaker for equal internal counts.

### Structure-aware beam search

- **Heuristic beam search over synonymous codons** (`src/pre_optimization.py`)
  - At each codon position, expands each current partial sequence with synonymous candidates.
  - Scores partial sequences with a weighted combination of mean log codon adaptiveness and local folding energy.
  - Keeps only the top `beam_width` partial sequences before continuing.
  - Uses ViennaRNA MFE (`RNA.fold`) when available; otherwise uses a deterministic GC-based MFE approximation.
  - This is a heuristic search, not a full global RNA-folding optimization algorithm.

### Genetic algorithm

- **Synonymous-codon genetic algorithm** (`src/optimization.py`)
  - Population initialization: starts from the preoptimized sequence and creates synonymous variants.
  - Selection: tournament selection.
  - Recombination: codon-position crossover between two parents, preserving the start codon.
  - Mutation: synonymous codon substitution, weighted by host codon adaptiveness when possible.
  - Elitism: carries the best candidates directly into the next generation.
  - Convergence: stops when recent best fitness values are nearly flat.
  - Safety checks: validates protein identity and exact start-codon preservation.

- **Multi-objective scalar fitness** (`src/optimization.py`)
  - Combines multiple biological objectives into one weighted score:
    - CAI
    - GC deviation
    - GC3 deviation
    - 5-prime folding penalty
    - 5-prime accessibility
    - unwanted motif count
    - repeat penalty
    - cryptic splice-site count
    - internal ATG count
    - tAI
    - codon-pair bias
  - Uses positive terms for desirable metrics and subtracts normalized penalties.

### Local repair / greedy local search

- **Declared repeated-domain diversification** (`src/diversification.py`)
  - Accepts one comma-separated mask input, interpreted as either amino-acid masks or nucleotide masks.
  - Resolves valid masks to CDS nucleotide coordinates before synonymous optimization changes the literal DNA.
  - Requires each mask entry to occur at least twice, rejects overlapping resolved ranges, and requires nucleotide masks to be in-frame and divisible by 3.
  - Diversifies each mask entry only against other occurrences of the same entry.
  - Runs after preoptimization and before local repair so downstream repair can clean collateral motif, GC, splice, or repeat issues.
  - Later GA fitness and hybrid/regulatory seed selection receive the same mask plan so polishing does not silently collapse diversified copies back to identical high-CAI codons.

- **Overlapping-window local repair** (`src/local_repair.py`)
  - Scans codon-aligned overlapping windows.
  - Computes a local problem score from repeats, unwanted motifs, cryptic splice sites, and GC drift.
  - Enumerates all single-codon synonymous substitutions in the window.
  - Applies the change with the best net improvement:
    - `net = problem_score_reduction - preoptimization_disruption_cost`
  - `problem_score_reduction` is the drop in the local pathology score after a proposed synonymous swap.
  - `preoptimization_disruption_cost` is the penalty for moving away from the preoptimized codon choice, including codon-adaptiveness loss.
  - A substitution is useful only when its net value is positive and is the best available improvement in that window.
  - Repeats up to `max_subs_per_window` per window.
  - Preserves the start codon and protein identity.

- **GC-balancing repair pass** (`src/local_repair.py`)
  - After local pathology repair, selects synonymous swaps that move total GC toward the host target.
  - Restricts swaps to similar codon-adaptiveness tiers.
  - Sorts candidate swaps by disruption cost and applies the least disruptive ones first.

### Hybrid and multi-seed post-selection

- **Multi-seed GA polish** (`src/hybrid_pipeline.py`)
  - Runs the same repaired sequence through several reproducible GA seeds.
  - Does not simply choose the highest GA fitness.
  - Ranks candidates by protein identity, degeneracy cleanliness, repeat burden, complexity, CAI/tAI, and finally GA fitness.

- **Regulatory multi-seed post-selection** (`src/hybrid_pipeline.py`)
  - Runs GA polish, then ranks by raw regulatory cleanup metrics.
  - Uses motif, splice-site, internal-ATG, CAI/tAI, repeat, GC, and constructability gates.
  - Scores proportional reduction from a baseline plus residual regulatory load penalties.

## Established Bioinformatic Algorithms and Principles

- **Standard genetic code translation**
  - DNA codons are translated using a fixed standard genetic code table.
  - Optimization is constrained to synonymous codon substitutions.

- **Codon Adaptation Index (CAI)**
  - Implemented as the geometric mean of per-codon relative adaptiveness values.
  - Uses log-space summation for numerical stability.
  - Host codon frequencies are normalized against the most frequent synonymous codon for each amino acid.

- **tRNA Adaptation Index (tAI)**
  - Runtime scoring uses bundled per-codon `w_i` weights and the same geometric-mean structure as CAI.
  - Data-builder script derives gene-copy tAI weights from GtRNAdb tRNAscan-SE anticodon copy counts using dos Reis-style wobble coefficients.

- **Codon-pair score / codon-pair bias**
  - Runtime scoring averages empirical codon-pair scores over adjacent sense-codon pairs.
  - Data-builder script derives CPS from CoCoPUTs/HIVE-CUTs codon and bicodon counts:
    - observed/expected log-ratio
    - pseudocount smoothing
    - expected count conditioned on amino-acid pair
    - mean-centering per host

- **CHARMING-style harmonization diagnostics**
  - Windowed CAI tracks compare source and optimized sequences.
  - `%MinMax` profiles compare each local window to fastest, slowest, and average synonymous alternatives for the same amino-acid sequence.
  - Reports Pearson correlation, net absolute deviation, and mean absolute deviation.

- **mRNA secondary-structure proxy**
  - Uses ViennaRNA MFE when installed.
  - Falls back to deterministic GC and palindrome heuristics.
  - Also computes a 5-prime accessibility proxy from folding energy or GC content.

- **Motif avoidance**
  - Counts default polyA-like motifs plus user-specified avoid motifs.
  - Counts overlapping motif hits.

- **Cryptic splice-site heuristic**
  - For eukaryotic hosts, scans donor-like `GT[AG]AG` motifs and acceptor-like polypyrimidine tracts ending in `AG`.
  - Disabled for prokaryotic hosts.

- **Internal start-codon check**
  - Counts in-frame `ATG` codons in the early coding region, excluding the true start codon.

- **GC and GC3 host matching**
  - Scores total GC and third-position GC deviation from host targets.
  - Host targets are either fixed constants or derived from codon-frequency tables.

- **Repeat and low-complexity avoidance**
  - Penalizes homopolymers, dinucleotide/trinucleotide tandem repeats, GC-only runs, broad repeat burden, and low complexity windows.

## Sequence Analysis Algorithms

- **Codon usage counting**
  - Splits sequences into codons.
  - Counts codons within amino-acid families.
  - Normalizes counts to within-amino-acid frequencies for plotting and comparison.

- **Sliding-window GC and GC3**
  - Uses moving-window scans for local GC and GC3 percentages.
  - The plotting helpers update window counts incrementally instead of recounting every window from scratch.

- **Cumulative GC3**
  - Tracks the running percentage of codons whose third base is G or C.

- **Sliding-window complexity score** (`src/complexity_analysis.py`)
  - Per-window components:
    - mononucleotide Shannon entropy
    - GC balance
    - k-mer richness
    - longest homopolymer run penalty
    - fixed-offset periodicity penalty
    - invalid-base fraction
  - Composite score combines entropy, k-mer richness, GC balance, and anti-repeat penalties.
  - Optional centered moving-average smoothing uses convolution with a validity mask.

- **Degeneracy metrics** (`src/degeneracy.py`)
  - Homopolymer run detection.
  - Binary GC-only and AT-only run detection.
  - Dinucleotide and trinucleotide tandem-repeat detection.
  - Sliding-window repeat burden.
  - Merged bad repeat regions.
  - Long-CDS repeat sampling with local windows, 2 kb core segmentation, and junction-window checks.
  - Weighted degeneracy score used for GA repeat modes and hybrid seed selection.

## Recurrent Sorting and Ranking Patterns

The project does not implement a custom sorting algorithm. It repeatedly uses
Python `sorted()`, list `.sort()`, tuple ranking, `min()`, and `max()` for
deterministic ordering and candidate selection. Under CPython, the concrete
sort implementation is Timsort, but the code relies on ranking semantics, not
on Timsort-specific behavior.

Recurring patterns:

- **Descending fitness ranking**
  - GA population evaluations are sorted by fitness in descending order.
  - Elitism and best-candidate tracking use the top of this sorted list.

- **Codon frequency ranking**
  - Percentile matching sorts codons by descending frequency.
  - Tie-breakers include codon lexical order or first observed position.

- **Nearest-percentile matching**
  - After frequency sorting, codons are selected by the minimum absolute distance between source and target percentile.

- **Lexicographic tuple ranking**
  - Hybrid and regulatory post-selection build rank tuples and choose `max(...)`.
  - This creates a priority order: identity/quality gates first, then penalties, then expression metrics, then fitness.

- **Least-disruptive repair ordering**
  - GC repair candidates are sorted by disruption cost before application.

- **Bad-window / bad-region ordering**
  - Repeat burden windows and strict-degeneracy examples are sorted by coordinates or severity before merging and reporting.

- **Deterministic registry ordering**
  - Host lists, metadata keys, invalid-character reports, and table output use sorted order for stable UI and test behavior.

## Mathematical Methods Used

Markdown can render mathematical notation with LaTeX delimiters. Inline
formulas use `$...$`, and display formulas use `$$...$$`. The formulas below
use that syntax.

Notation used below:

- Formula-specific symbols are defined near the formula that introduces them.
- $s$ is the coding sequence being scored or optimized.
- $|s|$ is the sequence length in nucleotides.
- $c$ is a codon, and $c_i$ is codon $i$ in a coding sequence.
- $w_i$ is the adaptiveness weight assigned to codon $c_i$.
- $n$ is the number of scored codons, adjacent codon pairs, or windows, depending on context.
- Indices such as $i$, $j$, $k$, and $t$ are local counters defined by each formula.
- $I(\cdot)$ is an indicator function: it is `1` when the condition is true and `0` otherwise.

### Weighted objective functions

- **Linear scalarization of multiple objectives** (`src/optimization.py`)
  - The GA converts several biological goals into one scalar fitness value.
  - Positive terms are added, and penalty terms are subtracted.
  - In compact form:

    $$
    F(s)
      = \sum_{r \in R} \lambda_r R_r(s)
        - \sum_{p \in P} \lambda_p P_p(s)
    $$

  - $F(s)$ is the final GA fitness for sequence $s$.
  - $R$ is the set of reward metrics, and $R_r(s)$ is the value of reward metric $r$ for sequence $s$.
  - $P$ is the set of penalty metrics, and $P_p(s)$ is the value of penalty metric $p$ for sequence $s$.
  - $\lambda_r$ and $\lambda_p$ are the normalized user or preset weights for the corresponding reward or penalty term.
  - In this project, reward terms include CAI, tAI, accessibility, and codon-pair bias. Penalty terms include GC deviation, GC3 deviation, folding penalty, motif penalty, repeat penalty, splice penalty, and internal-ATG penalty.
  - This is a weighted-sum multi-objective optimization method. It does not compute a Pareto frontier; the user-selected or preset weights define the trade-off.

- **Preset weight normalization** (`src/presets.py`)
  - Preset dictionaries are normalized so active weights sum to `1.0`.
  - For raw weights $a_i$, normalized weights are:

    $$
    \lambda_i = \frac{a_i}{\sum_j a_j}
    $$

  - $a_i$ is the raw, unnormalized weight for objective axis $i$.
  - $\lambda_i$ is the normalized weight used by the optimizer for that axis.
  - The denominator sums over the active objective axes $j$ after unsupported axes have been removed.
  - Optional unsupported axes, such as tAI or codon-pair bias for hosts without data, are zeroed and the remaining terms are renormalized.

- **Saturating penalty normalization**
  - Several raw counts are converted to capped penalty values:

    $$
    P(x; c) = \min\left(\frac{x}{c}, 1\right)
    $$

  - $x$ is a raw count or unbounded score that should become a normalized penalty.
  - $c$ is the cap or scale value at which the normalized penalty reaches `1.0`.
  - $P(x; c)$ is therefore bounded between `0.0` and `1.0`.
  - Examples in the code:
    - motif penalty: $P(m; 10)$, where $m$ is the number of unwanted motif hits.
    - splice penalty: $P(q; 5)$, where $q$ is the number of cryptic splice-site hits.
    - internal-ATG penalty: $P(a; 3)$, where $a$ is the number of in-frame internal `ATG` codons in the early coding region.
    - repeat penalty: $P(r; \mathrm{repeat\_penalty\_scale})$, where $r$ is the repeat score and `repeat_penalty_scale` is the configured repeat cap.
  - This prevents one metric from growing without bound and completely dominating the fitness.

### Geometric means and log-space scoring

- **CAI** (`src/utils.py`, `src/optimization.py`)
  - CAI is calculated as the geometric mean of per-codon relative adaptiveness values:

    $$
    \mathrm{CAI}(s)
      = \left(\prod_{i=1}^{n} w_i\right)^{1/n}
      = \exp\left(\frac{1}{n}\sum_{i=1}^{n}\ln w_i\right)
    $$

  - $w_i$ is the host relative adaptiveness weight for codon $c_i$, usually in the range `(0, 1]`.
  - $n$ is the number of codons with positive available weights.
  - Codons with zero or missing weights are skipped.
  - Log-space summation avoids numerical underflow and turns multiplication of many weights into addition.

- **tAI** (`src/utils.py`, `scripts/fetch_gtrnadb_trna_weights.py`)
  - Runtime tAI uses the same geometric-mean form as CAI, but with tRNA-derived `w_i` values.
  - The tRNA data builder computes absolute codon availability:

    $$
    W_c = \sum_j N_j p_{j,c}
    $$

    where $W_c$ is the absolute tRNA availability for codon $c$, $j$ indexes tRNA/anticodon types, $N_j$ is the copy number of type $j$, and $p_{j,c}$ is the wobble coefficient for decoding codon $c$ with tRNA type $j$.

  - It then normalizes by the maximum nonzero `W_i`:

    $$
    w_c = \frac{W_c}{\max_k W_k}
    $$

  - $w_c$ is the normalized tAI adaptiveness weight for codon $c$.
  - $k$ indexes codons in the nonzero availability set used to find the maximum.
  - Zero normalized values are replaced with the geometric mean of nonzero normalized weights.

- **Windowed CAI prefix sums** (`src/utils.py`)
  - CHARMING-style windowed CAI tracks use prefix sums of log codon weights.
  - For each window, the code subtracts cumulative log sums rather than recomputing the whole window:

    $$
    L_k = \sum_{i=1}^{k}\ln w_i
    $$

  - $L_k$ is the prefix sum of log weights through codon index $k$.
  - $w_i$ is the positive CAI weight for codon $i$.

    $$
    \mathrm{CAI}_{a:b}
      = \exp\left(\frac{L_b - L_a}{C_b - C_a}\right)
    $$

  - $a$ and $b$ are prefix indices delimiting the window, with codons in `(a, b]` contributing to the window score.
  - $C_k$ is the cumulative count of positive codon weights through prefix index $k$, so $C_b - C_a$ is the number of scored codons in the window.

  - This is an efficient sliding-window geometric mean.

### Composition and deviation metrics

- **Total GC percentage** (`src/utils.py`)
  - For a sequence $s$:

    $$
    \mathrm{GC}(s)
      = 100 \cdot
        \frac{N_G(s) + N_C(s)}{|s|}
    $$

  - $N_G(s)$ and $N_C(s)$ are the counts of `G` and `C` bases in sequence $s$.
  - $|s|$ is the nucleotide length of the sequence.
  - Used in scoring, local repair, plotting, and target-GC matching.

- **GC3 percentage** (`src/utils.py`, `src/gceh_module.py`)
  - For codons $c_1,\ldots,c_n$, with $c_{i,3}$ denoting the third base of codon $i$:

    $$
    \mathrm{GC3}(s)
      = 100 \cdot
        \frac{\sum_{i=1}^{n} I(c_{i,3} \in \{G,C\})}{n}
    $$

  - $n$ is the number of complete codons scored.
  - $c_{i,3}$ is the third nucleotide of codon $c_i$.
  - Used as a separate third-position composition constraint because synonymous codon choice mostly changes wobble bases.

- **Absolute deviation from target**
  - GC and GC3 penalties use absolute distance from host target values:

    $$
    D_{\mathrm{GC}}(s)
      = \frac{|\mathrm{GC}(s) - \mathrm{GC}_{\mathrm{target}}|}{100}
    $$

    $$
    D_{\mathrm{GC3}}(s)
      = \frac{|\mathrm{GC3}(s) - \mathrm{GC3}_{\mathrm{target}}|}{100}
    $$

  - $\mathrm{GC}_{\mathrm{target}}$ and $\mathrm{GC3}_{\mathrm{target}}$ are host target percentages.
  - $D_{\mathrm{GC}}(s)$ and $D_{\mathrm{GC3}}(s)$ are fractional deviations, so a 10 percentage-point mismatch is `0.10`.
  - Local repair also uses absolute GC drift, but only penalizes drift beyond a tolerance.

- **Host-derived target composition**
  - For hosts without explicit constants, target GC and GC3 are derived as weighted averages over codon-frequency tables.

    $$
    \widehat{\mathrm{GC}}_{\mathrm{host}}
      = 100 \cdot
        \frac{\sum_c f_c \cdot \mathrm{GCfrac}(c)}{\sum_c f_c}
    $$

    $$
    \widehat{\mathrm{GC3}}_{\mathrm{host}}
      = 100 \cdot
        \frac{\sum_c f_c \cdot I(c_3 \in \{G,C\})}{\sum_c f_c}
    $$

  - $\widehat{\mathrm{GC}}_{\mathrm{host}}$ and $\widehat{\mathrm{GC3}}_{\mathrm{host}}$ are derived host target percentages.
  - $f_c$ is the host codon frequency for codon $c$.
  - $\mathrm{GCfrac}(c)$ is the fraction of the three codon positions that are `G` or `C`.
  - $c_3$ is the third nucleotide of codon $c$.

### Sliding-window calculations

- **Moving GC and GC3 windows** (`src/gceh_module.py`)
  - The plotting functions update counts incrementally:

    $$
    g_{t+1}
      = g_t
        - I(s_t \in \{G,C\})
        + I(s_{t+W} \in \{G,C\})
    $$

  - $g_t$ is the number of `G` or `C` bases in the window starting at position $t$.
  - $W$ is the nucleotide window length.
  - $s_t$ is the base leaving the window, and $s_{t+W}$ is the base entering the next window.
  - The same recurrence is used conceptually for GC3 windows, replacing nucleotides with codon third positions.
  - This avoids recounting every window from scratch.

- **Sliding-window repeat burden** (`src/utils.py`, `src/degeneracy.py`)
  - Repeat scores are computed across windows and summarized as:

    $$
    \overline{r}
      = \frac{1}{n}\sum_{i=1}^{n} r_i
    $$

    $$
    r_{\max} = \max_i r_i
    $$

    $$
    f_{\mathrm{bad}}
      = \frac{1}{n}\sum_{i=1}^{n} I(r_i > \tau)
    $$

  - $r_i$ is the repeat penalty score for window $i$.
  - $\overline{r}$ is the mean repeat score across windows.
  - $r_{\max}$ is the worst window repeat score.
  - $f_{\mathrm{bad}}$ is the fraction of windows whose repeat score exceeds threshold $\tau$.
  - $n$ is the number of windows evaluated.
  - Window sizes scale with sequence length, from small windows for short CDSs to larger windows for long CDSs.

- **Complexity-track smoothing** (`src/complexity_analysis.py`)
  - Optional smoothing uses centered moving averages implemented with convolution.
  - A separate validity-mask convolution makes the smoothing NaN-aware:

    $$
    \widetilde{x}_t
      = \frac{(K * (M \odot X))_t}{(K * M)_t}
    $$

  - $\widetilde{x}_t$ is the smoothed complexity score at window index $t$.
  - $X$ is the unsmoothed score vector.
  - $M$ is a binary valid-value mask: `1` for finite scores and `0` for NaN scores.
  - $K$ is the moving-average kernel, `*` is convolution, and $\odot$ is elementwise multiplication.

### Harmonization profile statistics

- **Pearson correlation** (`src/utils.py`)
  - Input and optimized window tracks are compared with Pearson correlation:

    $$
    \rho_{x,y}
      = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}
             {\sqrt{\sum_i (x_i - \bar{x})^2}
              \sqrt{\sum_i (y_i - \bar{y})^2}}
    $$

  - $x_i$ and $y_i$ are aligned window values from the two profile tracks.
  - $\bar{x}$ and $\bar{y}$ are the means of those aligned track values.
  - $\rho_{x,y}$ is the Pearson correlation coefficient between tracks $x$ and $y$.
  - If either track has zero variance, the correlation is reported as NaN.
  - Used for CHARMING-style similarity of translational profiles.

- **Net absolute deviation**
  - The project reports the total absolute profile mismatch:

    $$
    \mathrm{NAD}(x,y)
      = \sum_{i=1}^{n} |x_i - y_i|
    $$

  - $x_i$ and $y_i$ are aligned window values, and $n$ is the number of aligned windows.
  - Measures total profile mismatch across all windows.

- **Mean absolute deviation**
  - The same mismatch normalized by the number of windows:

    $$
    \mathrm{MAD}(x,y)
      = \frac{1}{n}\sum_{i=1}^{n} |x_i - y_i|
    $$

  - $n$ is the number of aligned windows, so MAD is NAD divided by the profile length.
  - Normalizes the same profile mismatch by the number of windows.

- **%MinMax scaling** (`src/utils.py`)
  - Each codon window is compared to synonymous fastest, slowest, and average alternatives for the same amino-acid sequence.
  - If the actual value $A$ is above the average $\bar{A}$, it is scaled toward the maximum $A_{\max}$:

    $$
    \% \mathrm{MinMax}
      = 100 \cdot
        \frac{A - \bar{A}}{A_{\max} - \bar{A}}
    $$

  - If $A$ is below $\bar{A}$, it is scaled toward the minimum $A_{\min}$:

    $$
    \% \mathrm{MinMax}
      = -100 \cdot
        \frac{\bar{A} - A}{\bar{A} - A_{\min}}
    $$

  - $A$ is the actual local codon-usage value for the observed window.
  - $\bar{A}$ is the average value over synonymous alternatives for the same amino-acid window.
  - $A_{\max}$ and $A_{\min}$ are the fastest and slowest synonymous alternatives for that amino-acid window.
  - This produces a relative local codon-speed profile rather than an absolute usage score.

### Entropy and complexity scoring

- **Shannon entropy** (`src/complexity_analysis.py`)
  - Per-window nucleotide entropy is:

    $$
    H
      = -\sum_{b \in \{A,C,G,T\}} p_b \log_2 p_b
    $$

  - $b$ is one nucleotide base in `{A, C, G, T}`.
  - $p_b$ is the fraction of valid bases in the window that are base $b$.
  - Terms with $p_b = 0$ contribute `0` in the implementation.
  - For A/C/G/T, the maximum is 2 bits, so the normalized value is:

    $$
    H_{\mathrm{norm}} = \frac{H}{2}
    $$

  - $H_{\mathrm{norm}}$ is the entropy scaled to `[0, 1]` for a four-base alphabet.
  - Higher entropy means more balanced mononucleotide composition.

- **GC balance**
  - GC balance peaks at 50% GC and decreases linearly toward AT-only or GC-only extremes:

    $$
    B_{\mathrm{GC}}
      = 1 - 2|g - 0.5|
    $$

    where $B_{\mathrm{GC}}$ is the GC-balance score and $g$ is the GC fraction in the window.

  - The complexity score can raise this term to an exponent `alpha`:

    $$
    B_{\mathrm{GC},\alpha}
      = B_{\mathrm{GC}}^{\alpha}
    $$

  - $\alpha$ is the configured `gcbal_alpha` exponent. `alpha = 0` removes GC-balance influence; larger values make GC imbalance more punitive.

- **k-mer richness**
  - Counts distinct valid k-mers in a window.
  - Normalizes by the smaller of the theoretical k-mer space and the number of valid k-mer windows:

    $$
    R_k
      = \frac{|\mathrm{unique\_kmers}_k|}
             {\min(4^k, N_{\mathrm{valid}})}
    $$

  - $k$ is the k-mer length.
  - $|\mathrm{unique\_kmers}_k|$ is the number of distinct valid k-mers observed in the window.
  - $4^k$ is the maximum possible number of k-mers over A/C/G/T.
  - $N_{\mathrm{valid}}$ is the number of window positions that contain a valid k-mer with no `N`.
  - This rewards local sequence diversity.

- **Run and periodicity penalties**
  - Homopolymer penalty is based on the longest same-base run divided by window length.
  - Periodicity penalty scans fixed shifts and measures maximum self-similarity:

    $$
    Q
      = \max_{1 \le t \le T}
        \frac{1}{L-t}
        \sum_{i=1}^{L-t} I(s_i = s_{i+t})
    $$

  - $Q$ is the periodicity penalty.
  - $t$ is a fixed offset being tested, and $T$ is the maximum offset scanned.
  - $L$ is the window length.
  - $s_i$ and $s_{i+t}$ are bases separated by offset $t$.
  - The composite complexity score multiplies composition diversity by an anti-repeat term:

    $$
    C
      = \left(0.5H_{\mathrm{norm}} + 0.5R_k\right)
        B_{\mathrm{GC},\alpha}
        \left(1 - \max(P_{\mathrm{run}}, P_{\mathrm{periodic}})\right)
    $$

  - $C$ is the final per-window complexity score.
  - $P_{\mathrm{run}}$ is the longest-homopolymer run length divided by window length.
  - $P_{\mathrm{periodic}}$ is the periodicity penalty $Q$ above.
  - The `0.5` coefficients give equal weight to entropy and k-mer richness before GC balance and repeat penalties are applied.

### Repeat and degeneracy scoring

- **Adaptive repeated-domain identity gate** (`src/diversification.py`)
  - For a repeated amino-acid mask, the lowest possible pairwise DNA identity is dictated by codon degeneracy:

    $$
    I_{\min}(a_1,\ldots,a_n)
      = \frac{1}{3n}
        \sum_{i=1}^{n}
        \min_{c,d \in \mathrm{syn}(a_i)}
        \mathrm{matches}(c,d)
    $$

  - $a_i$ is amino acid $i$ in the masked domain.
  - $\mathrm{syn}(a_i)$ is the synonymous codon set for amino acid $a_i$.
  - $\mathrm{matches}(c,d)$ is the number of identical nucleotide positions between codons $c$ and $d$.
  - The default allowed pairwise identity is:

    $$
    I_{\max}
      = I_{\min} + (1 - I_{\min})(1 - \eta)
    $$

  - $\eta$ is the diversification strength. The default is `0.75`, meaning the pass targets 75% of the mathematically possible identity reduction.
  - This ensures the identity gate is never lower than what the amino-acid sequence can physically achieve.
  - The same metric is complemented by a longest exact shared consecutive DNA threshold, defaulting to `12 nt`, and a masked-region CAI floor of `40%` of the pre-diversification masked-region CAI.

- **Homopolymer excess scoring** (`src/utils.py`, `src/degeneracy.py`)
  - Runs above thresholds add increasing penalty.
  - A simplified threshold-excess form is:

    $$
    E_{\mathrm{run}}
      = \max(0, L_{\mathrm{run}} - L_{\mathrm{threshold}} + 1)
    $$

  - $L_{\mathrm{run}}$ is the observed homopolymer run length.
  - $L_{\mathrm{threshold}}$ is the minimum run length that begins accruing penalty.
  - $E_{\mathrm{run}}$ is the number of threshold-excess bases counted by this simplified form.

- **Tandem-repeat detection**
  - Dinucleotide and trinucleotide tandem repeats are found by extending repeated units.
  - For a repeated unit of length $u$ appearing $k$ times contiguously:

    $$
    L_{\mathrm{tandem}} = uk
    $$

  - $u$ is the repeat-unit length, such as `2` for dinucleotide repeats or `3` for trinucleotide repeats.
  - $k$ is the number of contiguous copies of that unit.
  - $L_{\mathrm{tandem}}$ is the total tandem-repeat span in nucleotides.
  - Runs are scored by total repeated bases and threshold exceedance.

- **GC-only run scoring**
  - Binary runs over the set `{G, C}` detect long GC-only stretches.
  - These are treated separately from ordinary total GC because they can create local synthesis or sequencing problems even when global GC is acceptable.

- **Weighted degeneracy score** (`src/degeneracy.py`)
  - Degeneracy metrics are combined into a weighted score using repeat burden, GC-only run excess, homopolymer burden, tandem-repeat burden, and low-complexity shortfall.
  - In compact form:

    $$
    D(s)
      = \sum_i \alpha_i M_i(s)
    $$

  - $D(s)$ is the combined degeneracy score for sequence $s$.
  - $M_i(s)$ is degeneracy metric $i$ measured on sequence $s$.
  - $\alpha_i$ is the fixed implementation weight for metric $M_i$.
  - Metrics include repeat maximum, repeat fraction bad, GC-run excess, homopolymer burden, tandem-repeat burden, and low-complexity shortfall.

  - GA repeat modes use either:
    - legacy repeat-window score
    - blended legacy plus strict degeneracy components
    - aligned full degeneracy score

- **Long-CDS repeat burden**
  - Long sequences use capped local windows and 2 kb core segmentation.
  - Nearby bad windows are merged into broader bad regions.
  - The project reports bad-region span fraction and worst-region scores to detect broad repeat burden.

### Log-ratio, sigmoid, and empirical bias scores

- **Codon-pair score construction** (`scripts/fetch_cocoputs_codon_pair_tables.py`)
  - Empirical CPS uses a smoothed observed/expected log-ratio:

    $$
    \mathrm{CPS}(c_1,c_2)
      = \ln\left(\frac{O(c_1,c_2)+0.5}{E(c_1,c_2)+0.5}\right)
    $$

  - $c_1$ and $c_2$ are adjacent sense codons.
  - $O(c_1,c_2)$ is the observed bicodon count for that ordered codon pair.
  - $E(c_1,c_2)$ is the expected bicodon count under the amino-acid-pair-conditioned model.
  - `0.5` is the pseudocount used to keep rare or absent pairs finite.
  - Expected counts are conditioned on the amino-acid pair:

    $$
    E(c_1,c_2)
      = N(a_1,a_2) \cdot P(c_1 \mid a_1) \cdot P(c_2 \mid a_2)
    $$

  - $a_1$ and $a_2$ are the amino acids encoded by codons $c_1$ and $c_2$.
  - $N(a_1,a_2)$ is the observed total count of bicodons encoding amino-acid pair $(a_1,a_2)$.
  - $P(c_1 \mid a_1)$ and $P(c_2 \mid a_2)$ are synonymous codon probabilities within the corresponding amino-acid families.
  - Host CPS tables are mean-centered after scoring.

- **Runtime codon-pair scoring** (`src/utils.py`, `src/optimization.py`)
  - Runtime CPS is the mean of adjacent codon-pair scores over valid sense-codon pairs.

    $$
    \overline{\mathrm{CPS}}(s)
      = \frac{1}{n-1}
        \sum_{i=1}^{n-1}\mathrm{CPS}(c_i,c_{i+1})
    $$

  - $n$ is the number of valid sense codons in sequence $s$.
  - $\mathrm{CPS}(c_i,c_{i+1})$ is the empirical codon-pair score for adjacent codons $i$ and $i+1$.
  - $\overline{\mathrm{CPS}}(s)$ is the mean adjacent-pair score used as the raw runtime CPS metric.
  - In GA fitness, the mean CPS is passed through a logistic sigmoid:

    $$
    \sigma(x)
      = \frac{1}{1 + e^{-x}}
    $$

  - $x$ is the real-valued raw score being mapped to `[0, 1]`.
  - $e$ is Euler's number.

    $$
    \mathrm{CPS}_{\mathrm{norm}}(s)
      = \sigma(\overline{\mathrm{CPS}}(s))
    $$

  - $\mathrm{CPS}_{\mathrm{norm}}(s)$ is the bounded codon-pair contribution used by the GA fitness.
  - This maps arbitrary log-ratio scores onto a bounded positive contribution.

- **5-prime accessibility sigmoid** (`src/utils.py`)
  - When ViennaRNA is available, start-region MFE is transformed into an accessibility proxy:

    $$
    A_{\mathrm{5prime}}
      = \frac{1}{1 + e^{-\mathrm{MFE}/5}}
    $$

  - $A_{\mathrm{5prime}}$ is the accessibility score for the 5-prime start region.
  - $\mathrm{MFE}$ is the predicted minimum free energy for that region; more negative values indicate more stable structure.
  - The divisor `5` is the fixed scale used by the implementation before applying the sigmoid.
  - Very stable negative MFE values produce lower accessibility scores.

### Percentiles and interpolation

- **Mid-percentile codon mapping** (`src/pre_optimization.py`)
  - Percentile matching assigns each codon a midpoint in its cumulative frequency bin:

    $$
    q_c
      = \frac{C_{\mathrm{before}} + f_c/2}{\sum_k f_k}
    $$

  - $q_c$ is the midpoint percentile assigned to codon $c$ within a synonymous codon family.
  - $C_{\mathrm{before}}$ is the cumulative frequency of higher-ranked codons before $c$.
  - $f_c$ is the frequency or count for codon $c$ in the source table, target table, or input CDS count table.
  - The denominator sums frequencies or counts over codons $k$ in the same amino-acid family.
  - Target codons are chosen by nearest absolute percentile distance.

    $$
    c_{\mathrm{target}}
      = \arg\min_{d \in \mathrm{syn}(a)}
        |q_c - q_d|
    $$

  - $a$ is the amino acid encoded by the source codon.
  - $\mathrm{syn}(a)$ is the set of synonymous target-host codons for amino acid $a$.
  - $d$ is one candidate target codon from that synonymous set.
  - $q_d$ is the target-host percentile assigned to candidate codon $d$.

- **Linear percentile interpolation** (`src/degeneracy.py`)
  - Summary percentiles, such as p90 long-repeat segment burden, are computed by sorting values and linearly interpolating between neighboring ranks.
  - For ordered values $x_0 \le \cdots \le x_{n-1}$ and percentile $p$:

    $$
    h = (n-1)p
    $$

  - $n$ is the number of ordered values.
  - $p$ is the requested percentile as a fraction in `[0, 1]`, such as `0.90` for p90.
  - $h$ is the fractional rank position in the sorted vector.

    $$
    Q(p)
      = (1-\gamma)x_{\lfloor h \rfloor}
        + \gamma x_{\lceil h \rceil}
    $$

  - $Q(p)$ is the interpolated percentile value.
  - $x_0 \le \cdots \le x_{n-1}$ are the sorted observed values.
  - $\lfloor h \rfloor$ and $\lceil h \rceil$ are the lower and upper neighboring rank indices.
  - $\gamma = h - \lfloor h \rfloor$ is the fractional interpolation weight between those neighbors.

  - This avoids jumps when the number of segment values is small.

### Threshold gates and ranking tuples

- **Boolean threshold gates**
  - Hybrid and regulatory selection use hard thresholds for protein identity, repeat burden, GC deviation, CAI, homopolymer length, tandem repeats, and broad repeat burden.
  - A generic gate has the form:

    $$
    G_m(s;\tau) = I(M_m(s) \le \tau)
    $$

  - $G_m(s;\tau)$ is `1` when sequence $s$ passes gate $m$ and `0` otherwise.
  - $M_m(s)$ is the metric checked by gate $m$.
  - $\tau$ is the maximum allowed value for that metric.
  - These gates encode "must pass" constructability constraints before softer scores are considered.

- **Lexicographic rank tuples**
  - Candidate ranking builds tuples such as:
    - identity pass
    - cleanliness pass
    - negative penalties
    - positive expression metrics
    - fallback GA fitness
  - Python compares tuples lexicographically, so earlier criteria dominate later ones.

- **Proportional reduction scoring**
  - Regulatory post-selection scores improvement relative to a baseline:

    $$
    R_{\mathrm{reduction}}
      = \frac{x_{\mathrm{before}} - x_{\mathrm{after}}}
             {x_{\mathrm{before}}}
    $$

  - $x_{\mathrm{before}}$ is the baseline count or burden before regulatory cleanup.
  - $x_{\mathrm{after}}$ is the count or burden for the candidate sequence after cleanup.
  - $R_{\mathrm{reduction}}$ is the proportional improvement, where `1.0` means complete removal and `0.0` means no reduction.
  - If the baseline count is zero, residual counts are penalized instead of producing a division by zero.

### Computational numerical methods

- **Vectorized codon indexing** (`src/utils.py`, `src/optimization.py`)
  - Codons are mapped to integer indices in a fixed 64-codon ordering.
  - CAI/tAI use 64-element score vectors.
  - Codon-pair bias uses a 64 x 64 score matrix.
  - This turns repeated dictionary lookups into array indexing in the GA hot path.

- **Numba-accelerated reductions**
  - When available, Numba compiles GC counting, CAI log-sum scoring, and motif counting loops.
  - Pure-Python fallbacks preserve the same scoring semantics.

- **LRU caching**
  - Host tables, codon-frequency dictionaries, score vectors, tAI vectors, and CPS matrices are cached.
  - This is not a mathematical scoring method, but it is important to the computational form of repeated optimization.

- **Compact hash identifiers** (`src/hybrid_pipeline.py`)
  - Sequence quality reports include a short SHA-256 digest prefix.
  - This is used as a compact deterministic identifier, not as a biological metric.

## Data Processing Algorithms in Scripts

### `scripts/fetch_gtrnadb_trna_weights.py`

- Fetches or reads GtRNAdb tRNAscan-SE archives.
- Parses tRNA type and anticodon copy counts.
- Filters pseudogenes, undetermined anticodons, noncanonical tRNAs, and initiator Met/fMet.
- Computes reverse complements for anticodon/codon compatibility checks.
- Applies wobble-pairing coefficients, including bacterial lysidine Ile2 handling.
- Computes absolute codon availability `W_i` as a sum of copy number times wobble coefficient.
- Normalizes weights to the maximum nonzero `W_i`.
- Replaces zero normalized weights with the geometric mean of nonzero weights.

### `scripts/fetch_cocoputs_codon_pair_tables.py`

- Fetches or reads CoCoPUTs/HIVE-CUTs codon and bicodon count tables.
- Validates complete 61-sense-codon and 61 x 61 sense-codon-pair coverage.
- Computes conditional codon probabilities within amino-acid families.
- Computes expected codon-pair counts conditioned on amino-acid pair.
- Computes smoothed log observed/expected codon-pair scores.
- Mean-centers all codon-pair scores for each host.

## Not Observed in This Project

I did not find implementations of these common bioinformatics algorithms:

- BLAST or seed-and-extend alignment.
- Needleman-Wunsch or Smith-Waterman alignment.
- HMMs or profile-HMMs.
- de Bruijn graph assembly.
- Phylogenetic tree inference.
- Machine-learning model training.
- Codon-aware multiple sequence alignment.
- Full dynamic-programming RNA design such as LinearDesign; the project uses a beam-search heuristic plus optional ViennaRNA folding calls.
