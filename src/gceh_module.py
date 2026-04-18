"""
gceh_module.py — Genetic Code Exploration Helper

Standalone module mirroring the full functionality of gceh.ipynb.

Provides:
- Reference codon frequency tables loaded from data/codon_tables/
- Input validation and cleaning
- analyze_sequence(): full standalone analysis with all plots (matches notebook)
- Two-sequence comparison plotting (used by main.ipynb optimizer)
- Individual plot helpers for flexible use
"""

import functools
from dataclasses import dataclass
from typing import Dict, List
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from .complexity_analysis import compute_complexity_track
from .utils import AA_TO_CODONS, GENETIC_CODE, load_codon_frequencies, list_available_hosts

import matplotlib.colors as mcolors
import numpy as np

try:
    from tabulate import tabulate
except ImportError:
    def tabulate(rows, headers, tablefmt=None):
        table = [list(headers)] + [[str(cell) for cell in row] for row in rows]
        widths = [max(len(str(row[i])) for row in table) for i in range(len(headers))]

        def fmt(row):
            return " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))

        separator = "-+-".join("-" * width for width in widths)
        return "\n".join([fmt(table[0]), separator] + [fmt(row) for row in table[1:]])


# ================================================================
# Reference Codon Frequency Tables (loaded from CSV source of truth)
# ================================================================

STOP_CODONS = {codon for codon, aa in GENETIC_CODE.items() if aa == '*'}

HOST_DISPLAY_NAMES = {
    'ecoli': 'E. coli',
    'hsapiens': 'H. sapiens',
    'mmusculus': 'M. musculus',
    'scerevisiae': 'S. cerevisiae',
    'spombe': 'S. pombe',
}

NO_COMP_VALUE = "__NO_COMPARISON__"


def _format_host_label(host: str) -> str:
    """Readable display label for a host key."""
    return HOST_DISPLAY_NAMES.get(host.lower(), host)


@functools.lru_cache(maxsize=64)
def load_reference_table(host: str) -> Dict[str, Dict[str, List[float]]]:
    """
    Load a plotting-ready reference table from the same codon-frequency CSVs
    used by the optimizer.

    Frequencies here are the raw within-amino-acid codon frequencies, not the
    CAI-style relative adaptiveness values returned by load_codon_table().
    """
    freqs = load_codon_frequencies(host)
    table: Dict[str, Dict[str, List[float]]] = {}
    for aa in sorted(aa for aa in AA_TO_CODONS if aa != '*'):
        codons = [codon for codon in AA_TO_CODONS[aa] if GENETIC_CODE.get(codon) != '*']
        table[aa] = {
            'codons': codons,
            'frequencies': [float(freqs.get(codon, 0.0)) for codon in codons],
        }
    return table


def get_reference_tables() -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """Return all currently available reference tables from data/codon_tables/."""
    return {
        _format_host_label(host): load_reference_table(host)
        for host in list_available_hosts()
    }


REFERENCE_TABLES = get_reference_tables()


# ================================================================
# Data classes
# ================================================================

@dataclass
class SeqAnalysis:
    name: str
    query_dna: str                    # verbatim DNA string
    seq_cods: List[str]               # codon list
    aa_to_freqs: Dict[str, Dict[str, float]]  # {AA: {codon: rel_freq_in_this_AA}}
    aa_order: List[str]               # AA order appearing in this sequence
    table_str: str                    # printable table
    gc3_ratio: List[float]            # cumulative GC3 (%)
    max_label_size: int               # for optional labeling in plots


# ================================================================
# Core helpers (two-sequence API)
# ================================================================

def _split_codons(q: str) -> List[str]:
    return [q[i:i+3] for i in range(0, len(q), 3)]


def _compute_gc3_ratio(seq_cods: List[str]) -> List[float]:
    ratio = []
    gc_count = 0
    for i, c in enumerate(seq_cods, 1):
        if c[2] in ("G", "C"):
            gc_count += 1
        ratio.append(gc_count / i * 100.0)
    return ratio


def _aa_codons_from_ref() -> Dict[str, List[str]]:
    return {
        aa: [codon for codon in codons if GENETIC_CODE.get(codon) != '*']
        for aa, codons in AA_TO_CODONS.items()
        if aa != '*'
    }


def _count_codon_usage(seq_cods: List[str]) -> Dict[str, Dict[str, int]]:
    aa_to_codons = _aa_codons_from_ref()
    counts: Dict[str, Dict[str, int]] = {aa: {c: 0 for c in cods} for aa, cods in aa_to_codons.items()}
    for c in seq_cods:
        aa = GENETIC_CODE.get(c)
        if aa in counts:
            counts[aa][c] += 1
    return counts


def _normalize_counts_to_freqs(counts: Dict[str, Dict[str, int]]):
    freqs: Dict[str, Dict[str, float]] = {}
    aa_order: List[str] = []
    for aa in sorted(counts.keys()):
        total = sum(counts[aa].values())
        if total > 0:
            aa_order.append(aa)
            freqs[aa] = {codon: cnt / total for codon, cnt in counts[aa].items() if cnt > 0}
    rows = []
    for aa in aa_order:
        cods = list(freqs[aa].keys())
        frs = [freqs[aa][c] for c in cods]
        rows.append([aa, ", ".join(cods), ", ".join(f"{x:.2f}" for x in frs)])
    table_str = tabulate(rows, headers=["AA", "Codons Found", "Observed Frequencies"], tablefmt="grid")
    return freqs, aa_order, table_str


def gceh_anal(name: str, raw_seq: str, max_label_size: int = 12) -> SeqAnalysis:
    q = raw_seq
    seq_cods = _split_codons(q)
    counts = _count_codon_usage(seq_cods)
    freqs, aa_order, table_str = _normalize_counts_to_freqs(counts)
    gc3_ratio = _compute_gc3_ratio(seq_cods)
    return SeqAnalysis(
        name=name,
        query_dna=q,
        seq_cods=seq_cods,
        aa_to_freqs=freqs,
        aa_order=aa_order,
        table_str=table_str,
        gc3_ratio=gc3_ratio,
        max_label_size=max_label_size,
    )


# ================================================================
# Standalone analysis (mirrors gceh.ipynb analyze_sequence)
# ================================================================

def analyze_sequence(
        project, input_seq,
        max_font_size=12, ref_table=NO_COMP_VALUE,
        ref_table_name="No Comparison", gc_winsize_1=50,
        gc_winsize_2=0, gc3_winsize_1=15,
        gc3_winsize_2=0, comp_window=50,
        comp_step=5, comp_k=3,
        comp_alpha=0.5, comp_smooth=4
        ):
    """
    Full standalone analysis matching gceh.ipynb behavior.

    Analyzes codon usage of the input sequence, optionally comparing against
    a reference table. Also analyzes GC, GC3 content, and sequence complexity.

    Args:
        project: Project name / plot title.
        input_seq: Raw nucleic acid sequence (DNA or RNA).
        max_font_size: Font size for codon labels on bar chart (0 = off).
        ref_table: Reference codon-usage dict from load_reference_table() or
                   REFERENCE_TABLES, or NO_COMP_VALUE for no comparison.
        ref_table_name: Display name for reference (e.g. "H. sapiens").
        gc_winsize_1: Primary GC sliding window size (bases).
        gc_winsize_2: Secondary GC sliding window size (0 = off).
        gc3_winsize_1: Primary GC3 sliding window size (codons).
        gc3_winsize_2: Secondary GC3 sliding window size (0 = off).
        comp_window: Complexity track window size.
        comp_step: Complexity track step size.
        comp_k: Complexity track k-mer k.
        comp_alpha: Complexity track GC balance alpha.
        comp_smooth: Complexity track smoothing window.
    """
    # --- Input Processing and Validation ---
    print("--- Processing input sequence ---")
    if not input_seq.strip():
        raise ValueError('empty')

    input_seq = input_seq.strip().upper().replace(" ", "").replace("\n", "").replace("\r", "")
    input_seq = input_seq.replace("U", "T")

    valid_chars = set("ACGT")
    if not set(input_seq).issubset(valid_chars):
        invalid_found = set(input_seq) - valid_chars
        print(f'   Invalid characters found: {", ".join(sorted(list(invalid_found)))}')
        raise ValueError('illegal')

    lengthmod = len(input_seq) % 3
    if lengthmod != 0:
        input_seq = input_seq[:-lengthmod]
        print(f"WARNING: Trimmed incomplete 3' codon ({lengthmod} bases removed).")

    clean_seq = input_seq

    seq_cods = []
    stop_flag = False
    i = 0
    while i < len(clean_seq):
        codon = clean_seq[i:i+3]
        if codon in STOP_CODONS:
            print(f'WARNING: Stop codon "{codon}" found at base {i+1}/{len(clean_seq)}. Sequence will be truncated.')
            clean_seq = clean_seq[:i]
            stop_flag = True
            break
        seq_cods.append(codon)
        i += 3

    num_codons_analyzed = len(seq_cods)
    print(f"Sequence length (after trimming): {len(clean_seq)} bases")
    print(f"Stop codons found: {stop_flag}")
    print(f"Codons entering analysis: {num_codons_analyzed}")

    # --- GC ---
    gc_count = clean_seq.count('G') + clean_seq.count('C')
    print(f'GC content: {gc_count / len(clean_seq) * 100:.2f}%')

    # --- GC3 ---
    gc3_flag = []
    gc3_counter = 0
    gc3_ratio = []
    for idx, cod in enumerate(seq_cods):
        if (cod[2] == 'G') or (cod[2] == 'C'):
            gc3_flag.append('1')
            gc3_counter += 1
        else:
            gc3_flag.append('0')
        gc3_ratio.append(gc3_counter / (idx + 1) * 100)

    print(f"GC3 content: {gc3_counter / len(seq_cods) * 100:.2f}%")

    # --- Codon/AA Counting ---
    codon_map = {
        codon: [0, aa]
        for codon, aa in GENETIC_CODE.items()
        if aa != '*'
    }

    for codon in seq_cods:
        if codon in codon_map:
            codon_map[codon][0] += 1

    present = [[codon, data[0], data[1]] for codon, data in codon_map.items() if data[0] > 0]

    if not present:
        raise ValueError('no_codons')

    aa_ord = sorted(set(entry[2] for entry in present))
    pasta = []
    for aa in aa_ord:
        codons_for_aa = [[entry[0], entry[1]] for entry in present if entry[2] == aa]
        pasta.append({'aa': aa, 'count': len(codons_for_aa), 'codons': codons_for_aa})

    # --- Frequency Calculation and Formatting ---
    anal_raw = []
    anal_pretty = []
    for dic in pasta:
        aa = dic['aa']
        codon_entries = dic['codons']
        codons_list = [entry[0] for entry in codon_entries]
        counts_list = [entry[1] for entry in codon_entries]
        cumulative = sum(counts_list)
        frequencies = [count / cumulative for count in counts_list]
        anal_pretty.append([
            aa,
            ", ".join(codons_list),
            ", ".join([f"{r:.2f}" for r in frequencies])
        ])
        anal_raw.append([aa, codons_list, frequencies])

    # --- Codon Usage Table Output ---
    print("\n--- Codon Usage Frequency in Input Sequence ---")
    col_names = ["AA", "Codons Found", "Observed Frequencies"]
    print(tabulate(anal_pretty, headers=col_names, tablefmt="grid"))

    # --- Codon Usage Plot ---
    print("\n--- Generating codon usage plot ---")
    plot_codon_usage_single(anal_raw, ref_table, ref_table_name, max_font_size, project)

    # --- Complexity Track ---
    plot_complexity_track(clean_seq, comp_window, comp_step, comp_k, comp_alpha, comp_smooth, project)

    # --- Sliding Window GC ---
    print('\n--- GC Calculations ---')
    plot_gc_sliding_single(clean_seq, gc_winsize_1, gc_winsize_2, project)

    # --- Sliding Window GC3 ---
    plot_gc3_sliding_single(seq_cods, gc3_winsize_1, gc3_winsize_2, project)

    # --- Global GC3 Ratio ---
    plot_gc3_ratio(gc3_ratio, project)


# ================================================================
# Standalone plot functions (matching notebook visuals)
# ================================================================

def plot_codon_usage_single(anal_raw, ref_table=NO_COMP_VALUE, ref_table_name="No Comparison",
                            max_font_size=12, project="Codon Usage Analysis"):
    """Stacked bar chart of codon usage, optionally compared to a reference table."""
    amino_acids = [item[0] for item in anal_raw]

    colors_list = ['#8986e5', '#f6786c', '#36b600', '#00bfc3', '#9690fe', '#e66bf3']
    colors_list_ref = ['#b8b6ef', '#faada7', '#88d366', '#66c5e8', '#bfbcff', '#f1a6f8']

    plt.figure(figsize=(17, 8))
    index = np.arange(len(amino_acids))
    sub_size = round(0.8 * max_font_size)
    bar_width = 0.35

    is_comparing = ref_table != NO_COMP_VALUE

    for i, plot_data in enumerate(anal_raw):
        aa = plot_data[0]
        codons = plot_data[1]
        frequencies = plot_data[2]

        if is_comparing:
            input_pos = index[i] - bar_width / 2
            ref_pos = index[i] + bar_width / 2
            current_bar_width = bar_width
        else:
            input_pos = index[i]
            ref_pos = None
            current_bar_width = bar_width * 1.5

        # Plot: Analyzed Sequence
        bottom = 0
        for j, (codon, freq) in enumerate(zip(codons, frequencies)):
            color_index = j % len(colors_list)
            color = colors_list[color_index]

            plt.bar(input_pos, freq, current_bar_width, bottom=bottom, color=color,
                    edgecolor='grey', linewidth=0.5)

            y_position = bottom + freq / 2
            is_max_freq = (freq == max(frequencies))
            fontsize = max_font_size if is_max_freq else sub_size
            fontweight = 'bold' if is_max_freq else 'normal'
            rotation = 90 if is_max_freq else 0
            if max_font_size > 0:
                plt.text(input_pos, y_position, codon, ha='center', va='center',
                         fontsize=fontsize, color='white', fontweight=fontweight, rotation=rotation)
            bottom += freq

        # Plot: Reference Table
        if is_comparing:
            ref_codons = ref_table[aa]['codons']
            ref_frequencies = ref_table[aa]['frequencies']
            ref_bottom = 0
            for j, (codon, freq) in enumerate(zip(ref_codons, ref_frequencies)):
                color_index = j % len(colors_list_ref)
                color = colors_list_ref[color_index]

                plt.bar(ref_pos, freq, current_bar_width, bottom=ref_bottom, color=color,
                        hatch='///', edgecolor='grey', linewidth=0.5)

                y_position = ref_bottom + freq / 2
                is_max_freq = (freq == max(ref_frequencies))
                fontsize = max_font_size if is_max_freq else sub_size
                fontweight = 'bold' if is_max_freq else 'normal'
                rotation = 90 if is_max_freq else 0
                if max_font_size > 0:
                    plt.text(ref_pos, y_position, codon, ha='center', va='center',
                             fontsize=fontsize, color='black', fontweight=fontweight, rotation=rotation)
                ref_bottom += freq

    # Legend
    legend_handles = []
    analyzed_patch = Patch(facecolor='white', edgecolor='black', label='Analyzed Sequence')
    legend_handles.append(analyzed_patch)

    if is_comparing:
        reference_patch = Patch(facecolor='white', edgecolor='black', hatch='///',
                                label=f'Reference: {ref_table_name}')
        legend_handles.append(reference_patch)

    plt.xlabel('Amino Acids')
    plt.ylabel('Relative Codon Frequency')
    plt.title(f'Codon Map - {project}')
    plt.xticks(index, amino_acids, rotation=45, ha='center')
    plt.ylim(0, 1.05)
    plt.legend(handles=legend_handles)
    plt.tight_layout()
    plt.show()


def plot_complexity_track(clean_seq, comp_window=50, comp_step=5, comp_k=3,
                          comp_alpha=0.5, comp_smooth=4, project="Codon Usage Analysis"):
    """Sliding-window complexity track plot using complexity_analysis module."""
    comp = compute_complexity_track(
        clean_seq,
        window_size=comp_window,
        step=comp_step,
        k=comp_k,
        gcbal_alpha=comp_alpha,
        smooth=comp_smooth)

    mid = comp['mid']
    score = comp['score']

    x = mid + 1
    y = score

    if len(y) > 0:
        plt.figure(figsize=(17, 8))
        plt.plot(x, y, lw=1)
        plt.xlim(left=1, right=max(len(clean_seq), x.max() if hasattr(x, 'max') else x[-1]))
        plt.ylim(0, 1)
        plt.xlabel('Position (bp)')
        plt.ylabel('Complexity')
        plt.title(f'Sliding-window Complexity (W = {comp_window}, step = {comp_step}, k = {comp_k}, alpha = {comp_alpha}, smooth = {comp_smooth}) - {project}')
        plt.tight_layout()
        # Force ticks to start at 1
        default_ticks = plt.gca().get_xticks()
        xstep = int(max(1, default_ticks[1] - default_ticks[0])) if len(default_ticks) > 1 else 1
        xticks = np.arange(1, max(len(clean_seq), x[-1]) + 1, xstep)
        plt.xticks(xticks, [str(int(t)) for t in xticks])
        plt.axhline(y=0.5, color='green', linestyle='--', linewidth=0.8)
        plt.axhspan(0, 0.2, color="red", alpha=0.12)
        plt.axhspan(0.5, 1, color="green", alpha=0.12)
        plt.axhline(y=y.max(), color='grey', linestyle='--', linewidth=0.8)
        plt.axhline(y=y.min(), color='grey', linestyle='--', linewidth=0.8)
        plt.show()
    else:
        print(f'Sequence too short for complexity track (window_size={comp_window}).')


def _gc_sliding_W(seq, winsize):
    """Sliding window GC% over a nucleotide sequence."""
    if winsize > len(seq):
        print('ERROR: GC window size must be smaller than query length')
        return [], []

    window = seq[:winsize].count('G') + seq[:winsize].count('C')
    results = [window / winsize * 100]

    for i in range(1, len(seq) - winsize + 1):
        if seq[i - 1] in 'GC':
            window -= 1
        if seq[i + winsize - 1] in 'GC':
            window += 1
        results.append(window / winsize * 100)

    starts = np.arange(len(seq) - winsize + 1, dtype=float)
    centers = starts + winsize / 2.0
    print(f'Max.: {max(results):.2f}%')
    print(f'Min.: {min(results):.2f}%')
    print(f'Mean: {np.mean(results):.2f}%')
    return centers, np.asarray(results, dtype=float)


def _gc3_sliding_W(seq_cods, winsize):
    """Sliding window GC3% over a codon list."""
    if winsize > len(seq_cods):
        print('ERROR: GC3 window size must be smaller than codons in query.')
        return [], []

    window = sum((codon[2] in 'GC') for codon in seq_cods[:winsize])
    results = [window / winsize * 100]

    for i in range(1, len(seq_cods) - winsize + 1):
        if seq_cods[i - 1][2] in 'GC':
            window -= 1
        if seq_cods[i + winsize - 1][2] in 'GC':
            window += 1
        results.append(window / winsize * 100)

    starts = np.arange(len(seq_cods) - winsize + 1, dtype=float)
    centers = starts + winsize / 2.0
    print(f'Max.: {max(results):.2f}%')
    print(f'Min.: {min(results):.2f}%')
    print(f'Mean: {np.mean(results):.2f}%')
    return centers, np.asarray(results, dtype=float)


def plot_gc_sliding_single(clean_seq, gc_winsize_1=50, gc_winsize_2=0, project="Codon Usage Analysis"):
    """Sliding window GC% plot with optional second overlay window."""
    print('GC window 1:')
    gc_x1, gc_y1 = _gc_sliding_W(clean_seq, gc_winsize_1)

    if len(gc_x1) == 0:
        return

    plt.figure(figsize=(17, 8))
    gc_win_1, = plt.plot(gc_x1, gc_y1)
    gc_handles = [gc_win_1]
    gc_labels = [f'Window Size: {gc_winsize_1}']

    gc_second_window = False
    if gc_winsize_2 > 0:
        gc_second_window = True
        print('\nGC window 2:')
        gc_x2, gc_y2 = _gc_sliding_W(clean_seq, gc_winsize_2)
        if len(gc_x2) > 0:
            gc_win_2, = plt.plot(gc_x2, gc_y2)
            gc_handles.append(gc_win_2)
            gc_labels.append(f'Window Size: {gc_winsize_2}')

    plt.title(f'GC % Moving Average - {project}')
    plt.xlabel('Base Position (window center)')
    plt.ylabel('GC Percentage')

    default_ticks = plt.gca().get_xticks()
    xstep = int(max(1, default_ticks[1] - default_ticks[0]))
    xticks = np.arange(0, gc_x1[-1] + 1, xstep)
    xlabels = [str(int(t + 1)) for t in xticks]
    plt.xticks(xticks, xlabels)
    plt.ylim(0, 105)
    plt.legend(handles=gc_handles, labels=gc_labels)
    plt.axhline(y=50, color='green', linestyle='--', linewidth=0.8)

    if not gc_second_window:
        y_max = max(gc_y1)
        y_min = min(gc_y1)
        plt.axhline(y=y_max, color='grey', linestyle='--', linewidth=0.8)
        plt.axhline(y=y_min, color='grey', linestyle='--', linewidth=0.8)
        plt.annotate(f"{y_max:.2f}", xy=(0, y_max), xytext=(0, y_max + 1.5))
        plt.annotate(f"{y_min:.2f}", xy=(0, y_min), xytext=(0, y_min - 3))

    plt.show()


def plot_gc3_sliding_single(seq_cods, gc3_winsize_1=15, gc3_winsize_2=0, project="Codon Usage Analysis"):
    """Sliding window GC3% plot with optional second overlay window."""
    print('GC3 window 1:')
    gc3_x1, gc3_y1 = _gc3_sliding_W(seq_cods, gc3_winsize_1)

    if len(gc3_x1) == 0:
        return

    plt.figure(figsize=(17, 8))
    gc3_win_1, = plt.plot(gc3_x1, gc3_y1)
    gc3_handles = [gc3_win_1]
    gc3_labels = [f'Window Size: {gc3_winsize_1}']

    if gc3_winsize_2 > 0:
        print('\nGC3 window 2:')
        gc3_x2, gc3_y2 = _gc3_sliding_W(seq_cods, gc3_winsize_2)
        if len(gc3_x2) > 0:
            gc3_win_2, = plt.plot(gc3_x2, gc3_y2)
            gc3_handles.append(gc3_win_2)
            gc3_labels.append(f'Window Size: {gc3_winsize_2}')

    plt.title(f'GC3 % Moving Average - {project}')
    plt.xlabel('Codon Position (window center)')
    plt.ylabel('GC3 Percentage')

    default_ticks = plt.gca().get_xticks()
    xstep = int(max(1, default_ticks[1] - default_ticks[0]))
    xticks = np.arange(0, gc3_x1[-1] + 1, xstep)
    xlabels = [str(int(t + 1)) for t in xticks]
    plt.xticks(xticks, xlabels)
    plt.ylim(0, 105)
    plt.legend(handles=gc3_handles, labels=gc3_labels)

    plt.show()


def plot_gc3_ratio(gc3_ratio, project="Codon Usage Analysis"):
    """Cumulative GC3 ratio along the sequence."""
    plt.figure(figsize=(17, 8))
    plt.plot(gc3_ratio)
    plt.xlabel('Codon Position')
    plt.ylabel('GC3 Percentage')
    plt.title(f'GC3 Ratio Along Sequence - {project}')

    default_ticks = plt.gca().get_xticks()
    xstep = int(max(1, default_ticks[1] - default_ticks[0]))
    xticks = range(0, len(gc3_ratio), xstep)
    xlabels = [str(i + 1) for i in xticks]
    plt.xticks(xticks, xlabels)
    plt.ylim(0, 105)
    plt.show()


# ================================================================
# Two-sequence comparison plotting (used by main.ipynb)
# ================================================================

def _make_shades(base_color, n):
    """Return n shades of the given base color (light -> dark)."""
    cmap = mcolors.LinearSegmentedColormap.from_list(
        None, ["white", base_color], N=n+2
    )
    return [cmap(i/(n+1)) for i in range(1, n+1)]


def _ensure_same_aa_chain(data1: SeqAnalysis, data2: SeqAnalysis):
    if data1.aa_order != data2.aa_order:
        raise ValueError(
            "Amino-acid orders differ between sequences. "
            "Ensure both sequences encode the same protein (same AA chain)."
        )


def plot_codon_usage_compare(data1: SeqAnalysis, data2: SeqAnalysis, title: str = "Codon usage — comparison"):
    _ensure_same_aa_chain(data1, data2)
    aa_order = data1.aa_order

    seq1_color = "#1b9e77"
    seq2_color = "#7570b3"

    idx = np.arange(len(aa_order))
    bar_width = 0.45

    plt.figure(figsize=(17, 8))

    for i, aa in enumerate(aa_order):
        codon_list = sorted(set(data1.aa_to_freqs[aa].keys()).union(data2.aa_to_freqs[aa].keys()))

        shades1 = _make_shades(seq1_color, len(codon_list))
        shades2 = _make_shades(seq2_color, len(codon_list))

        bottom = 0.0
        freqs1 = [data1.aa_to_freqs[aa].get(c, 0.0) for c in codon_list]
        max1 = max(freqs1) if freqs1 else 0.0
        for c, f, col in zip(codon_list, freqs1, shades1):
            plt.bar(i - bar_width/2, f, bar_width, bottom=bottom, color=col, edgecolor="grey", linewidth=0.5)
            if f == max1 and f > 0 and data1.max_label_size > 0:
                plt.text(i - bar_width/2, bottom + f/2, c, ha="center", va="center",
                         fontsize=data1.max_label_size, fontweight="bold", rotation=90)
            bottom += f

        bottom = 0.0
        freqs2 = [data2.aa_to_freqs[aa].get(c, 0.0) for c in codon_list]
        max2 = max(freqs2) if freqs2 else 0.0
        for c, f, col in zip(codon_list, freqs2, shades2):
            plt.bar(i + bar_width/2, f, bar_width, bottom=bottom, color=col, edgecolor="grey", linewidth=0.5)
            if f == max2 and f > 0 and data2.max_label_size > 0:
                plt.text(i + bar_width/2, bottom + f/2, c, ha="center", va="center",
                         fontsize=data2.max_label_size, fontweight="bold", rotation=90)
            bottom += f

    plt.title(title)
    plt.xlabel("Amino acids")
    plt.ylabel("Relative codon frequency")
    plt.xticks(idx, aa_order, rotation=45, ha="center")
    plt.ylim(0, 1.05)

    legend_patches = [
        Patch(facecolor=seq1_color, label=data1.name),
        Patch(facecolor=seq2_color, label=data2.name)
    ]
    plt.legend(handles=legend_patches, loc="best")
    plt.tight_layout()
    plt.grid(False)
    plt.show()


def plot_gc3_compare(data1: SeqAnalysis, data2: SeqAnalysis, title: str = "GC3 along sequence — comparison"):
    plt.figure(figsize=(17, 8))
    plt.plot(data1.gc3_ratio, label=f"{data1.name}")
    plt.plot(data2.gc3_ratio, label=f"{data2.name}")
    plt.title(title)
    plt.xlabel("Codon position")
    plt.ylabel("GC3 percentage")
    ax = plt.gca()
    default = ax.get_xticks()
    step = int(max(1, default[1] - default[0])) if len(default) > 1 else 1
    idxs = range(0, max(len(data1.gc3_ratio), len(data2.gc3_ratio)), step)
    labels = [str(i+1) for i in idxs]
    plt.xticks(idxs, labels)
    plt.ylim(0, 105)
    plt.legend()
    plt.tight_layout()
    plt.grid(False)
    plt.show()


def _gc3_sliding(seq_cods: List[str], winsize: int) -> List[float]:
    if winsize <= 0 or winsize > len(seq_cods):
        return []
    res = []
    window = sum(1 for i in range(winsize) if seq_cods[i][2] in ("G", "C"))
    res.append(window / winsize * 100.0)
    for i in range(1, len(seq_cods) - winsize + 1):
        if seq_cods[i-1][2] in ("G", "C"):
            window -= 1
        if seq_cods[i + winsize - 1][2] in ("G", "C"):
            window += 1
        res.append(window / winsize * 100.0)
    return res


def plot_gc3_sliding_compare(data1: SeqAnalysis, data2: SeqAnalysis, win_size: int = 10, title: str = "GC3 moving average — comparison"):
    plt.figure(figsize=(17, 8))
    s1 = _gc3_sliding(data1.seq_cods, win_size)
    s2 = _gc3_sliding(data2.seq_cods, win_size)
    plt.plot(s1, label=f"{data1.name} (win {win_size})")
    plt.plot(s2, label=f"{data2.name} (win {win_size})")

    plt.title(title)
    plt.xlabel("Codon position at window start")
    plt.ylabel("GC3 percentage")
    ax = plt.gca()
    default = ax.get_xticks()
    step = int(max(1, default[1] - default[0])) if len(default) > 1 else 1
    longest = 0
    for line in ax.get_lines():
        longest = max(longest, len(line.get_xdata()))
    idxs = range(0, longest, step)
    labels = [str(i+1) for i in idxs]
    plt.xticks(idxs, labels)
    plt.legend()
    plt.tight_layout()
    plt.grid(False)
    plt.show()


# ---------- Sliding-window overall GC% (two-sequence) ----------

def _gc_sliding_nt(seq: str, winsize_nt: int, step_nt: int = 1):
    if winsize_nt <= 0 or winsize_nt > len(seq) or step_nt <= 0:
        return []

    gc = 0
    for i in range(winsize_nt):
        if seq[i] in ("G", "C"):
            gc += 1
    values = [gc / winsize_nt * 100.0]

    for start in range(1, len(seq) - winsize_nt + 1, step_nt):
        out_nt = seq[start - 1]
        if out_nt in ("G", "C"):
            gc -= 1
        in_nt = seq[start + winsize_nt - 1]
        if in_nt in ("G", "C"):
            gc += 1
        values.append(gc / winsize_nt * 100.0)

    return values


def plot_gc_sliding_compare(data1: SeqAnalysis, data2: SeqAnalysis, win_nt: int = 51, step_nt: int = 1, title: str = "Sliding GC% — nucleotide window"):
    s1 = _gc_sliding_nt(data1.query_dna, win_nt, step_nt)
    s2 = _gc_sliding_nt(data2.query_dna, win_nt, step_nt)

    plt.figure(figsize=(17, 8))
    if s1:
        plt.plot(s1, label=f"{data1.name} (win {win_nt}, step {step_nt})")
    if s2:
        plt.plot(s2, label=f"{data2.name} (win {win_nt}, step {step_nt})")

    plt.title(title)
    plt.xlabel("Window start (nt)")
    plt.ylabel("GC percentage")
    ax = plt.gca()
    default = ax.get_xticks()
    step = int(max(1, default[1] - default[0])) if len(default) > 1 else 1
    longest = max(len(s1), len(s2)) if (s1 or s2) else 0
    idxs = range(0, longest, step)
    labels = [str(i+1) for i in idxs]
    plt.xticks(idxs, labels)
    plt.ylim(0, 105)
    plt.legend()
    plt.tight_layout()
    plt.grid(False)
    plt.axhspan(40, 60, facecolor="#cacaca", alpha=0.2, label="Target band")
    plt.axhline(y=50, color="red", linestyle="--", linewidth=1)
    plt.show()


# ---------- Sliding-window complexity (two-sequence) ----------

def plot_complexity_compare(
    data1: SeqAnalysis, data2: SeqAnalysis,
    comp_window: int = 50, comp_step: int = 5, comp_k: int = 3,
    comp_alpha: float = 0.5, comp_smooth: int = 4,
    title: str = "Sliding-window Complexity — comparison",
):
    """Overlay complexity tracks for two sequences."""
    seq1_color = "#1b9e77"
    seq2_color = "#7570b3"

    def _get_track(seq):
        comp = compute_complexity_track(
            seq, window_size=comp_window, step=comp_step,
            k=comp_k, gcbal_alpha=comp_alpha, smooth=comp_smooth)
        return comp['mid'], comp['score']

    mid1, score1 = _get_track(data1.query_dna)
    mid2, score2 = _get_track(data2.query_dna)

    if len(score1) == 0 and len(score2) == 0:
        print(f'Both sequences too short for complexity track (window_size={comp_window}).')
        return

    plt.figure(figsize=(17, 8))
    if len(score1) > 0:
        plt.plot(mid1 + 1, score1, lw=1, color=seq1_color, label=data1.name)
    if len(score2) > 0:
        plt.plot(mid2 + 1, score2, lw=1, color=seq2_color, label=data2.name)

    seq_len = max(len(data1.query_dna), len(data2.query_dna))
    plt.xlim(left=1, right=seq_len)
    plt.ylim(0, 1)
    plt.xlabel('Position (bp)')
    plt.ylabel('Complexity')
    plt.title(f'{title}  (W={comp_window}, step={comp_step}, k={comp_k}, α={comp_alpha}, smooth={comp_smooth})')
    plt.axhline(y=0.5, color='green', linestyle='--', linewidth=0.8)
    plt.axhspan(0, 0.2, color="red", alpha=0.12)
    plt.axhspan(0.5, 1, color="green", alpha=0.12)
    plt.legend()
    plt.tight_layout()
    plt.grid(False)
    plt.show()
