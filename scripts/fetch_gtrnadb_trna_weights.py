#!/usr/bin/env python3
"""
Fetch GtRNAdb tRNAscan-SE results and derive gene-copy tAI weights.

The output format matches src.utils.load_tai_weights():

    codon,weight
    AAA,0.123456

Weights are relative adaptiveness values for the 61 sense codons. They are
derived from high-confidence/non-pseudogene tRNA anticodon copy counts using
the dos Reis et al. (2004) tAI wobble penalties.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import tarfile
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, Tuple
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "trna_weights"
BASE_URL = "https://gtrnadb.ucsc.edu"
PSEUDOCOUNT_REPLACEMENT = "zero W_i values are replaced by the geometric mean of nonzero normalized w_i values"

GENETIC_CODE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

AA_THREE_TO_SINGLE = {
    "Ala": "A", "Cys": "C", "Asp": "D", "Glu": "E", "Phe": "F",
    "Gly": "G", "His": "H", "Ile": "I", "Ile2": "I", "Lys": "K",
    "Leu": "L", "Met": "M", "Asn": "N", "Pro": "P", "Gln": "Q",
    "Arg": "R", "Ser": "S", "Thr": "T", "Val": "V", "Trp": "W",
    "Tyr": "Y",
}

COMPLEMENT = str.maketrans("ACGT", "TGCA")
SENSE_CODONS = [
    first + second + third
    for first in "ACGT"
    for second in "ACGT"
    for third in "ACGT"
    if GENETIC_CODE[first + second + third] != "*"
]

# dos Reis et al. (2004), Table 2 optimized s-values. Coefficients below are
# p = 1 - s. Anticodon A in tRNA genes is treated as inosine at wobble.
WOBBLE_COEFFICIENTS = {
    ("A", "T"): 1.0,      # I:U
    ("A", "C"): 0.72,     # I:C, 1 - 0.28
    ("A", "A"): 0.0001,   # I:A, 1 - 0.9999
    ("G", "C"): 1.0,      # G:C
    ("G", "T"): 0.59,     # G:U, 1 - 0.41
    ("T", "A"): 1.0,      # U:A
    ("T", "G"): 0.32,     # U:G, 1 - 0.68
    ("C", "G"): 1.0,      # C:G
}
LYSIDINE_ILE2_COEFFICIENT = 0.11  # L:A, 1 - 0.89


@dataclass(frozen=True)
class HostSpec:
    key: str
    organism: str
    genome: str
    domain: str
    tarball_url: str
    tRNAscan_member: str
    superkingdom: str


HOSTS = {
    "ecoli": HostSpec(
        key="ecoli",
        organism="Escherichia coli str. K-12 substr. MG1655",
        genome="Esch_coli_K_12_MG1655",
        domain="bacteria",
        tarball_url=f"{BASE_URL}/genomes/bacteria/Esch_coli_K_12_MG1655/eschColi_K_12_MG1655-tRNAs.tar.gz",
        tRNAscan_member="eschColi_K_12_MG1655-tRNAs.out",
        superkingdom="bacteria",
    ),
    "hsapiens": HostSpec(
        key="hsapiens",
        organism="Homo sapiens GRCh38/hg38",
        genome="Hsapi38",
        domain="eukaryota",
        tarball_url=f"{BASE_URL}/genomes/eukaryota/Hsapi38/hg38-tRNAs.tar.gz",
        tRNAscan_member="hg38-tRNAs-confidence-set.out",
        superkingdom="eukaryota",
    ),
    "mmusculus": HostSpec(
        key="mmusculus",
        organism="Mus musculus GRCm39/mm39",
        genome="Mmusc39",
        domain="eukaryota",
        tarball_url=f"{BASE_URL}/genomes/eukaryota/Mmusc39/mm39-tRNAs.tar.gz",
        tRNAscan_member="mm39-tRNAs-confidence-set.out",
        superkingdom="eukaryota",
    ),
    "scerevisiae": HostSpec(
        key="scerevisiae",
        organism="Saccharomyces cerevisiae S288C",
        genome="Scere3",
        domain="eukaryota",
        tarball_url=f"{BASE_URL}/genomes/eukaryota/Scere3/sacCer3-tRNAs.tar.gz",
        tRNAscan_member="sacCer3-tRNAs.out-noChrM",
        superkingdom="eukaryota",
    ),
    "spombe": HostSpec(
        key="spombe",
        organism="Schizosaccharomyces pombe 972h-",
        genome="Schi_pomb_972h",
        domain="eukaryota",
        tarball_url=f"{BASE_URL}/genomes/eukaryota/Schi_pomb_972h/schiPomb_972H-tRNAs.tar.gz",
        tRNAscan_member="schiPomb_972H-tRNAs.out-noChrM",
        superkingdom="eukaryota",
    ),
}


def reverse_complement(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def fetch_bytes(url: str, timeout: int) -> bytes:
    request = Request(url, headers={"User-Agent": "CDS-Optimizer data builder"})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def read_or_fetch_tarball(host: HostSpec, args: argparse.Namespace) -> tuple[bytes, str]:
    cache_path = None
    if args.raw_dir:
        cache_path = Path(args.raw_dir) / f"{host.key}-tRNAs.tar.gz"

    if args.no_fetch:
        if cache_path is None:
            raise ValueError("--no-fetch requires --raw-dir")
        return cache_path.read_bytes(), str(cache_path)

    data = fetch_bytes(host.tarball_url, args.timeout)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(data)
    return data, host.tarball_url


def extract_member_text(tarball_data: bytes, member_name: str) -> str:
    with tarfile.open(fileobj=io.BytesIO(tarball_data), mode="r:gz") as archive:
        member = archive.extractfile(member_name)
        if member is None:
            raise ValueError(f"Archive does not contain {member_name!r}")
        return member.read().decode("utf-8")


def parse_trnascan_counts(text: str) -> tuple[Counter[tuple[str, str]], dict]:
    counts: Counter[tuple[str, str]] = Counter()
    stats = {
        "records_seen": 0,
        "records_included": 0,
        "records_excluded": 0,
        "excluded_types": Counter(),
    }

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("Sequence") or stripped.startswith("Name") or stripped.startswith("-"):
            continue

        fields = stripped.split()
        if len(fields) < 6:
            continue

        stats["records_seen"] += 1
        trna_type = fields[4]
        anticodon = fields[5].upper()
        note = " ".join(fields[6:]).lower()

        if "pseudo" in note or anticodon == "NNN" or trna_type not in AA_THREE_TO_SINGLE:
            stats["records_excluded"] += 1
            stats["excluded_types"][trna_type] += 1
            continue

        if trna_type in {"iMet", "fMet"}:
            stats["records_excluded"] += 1
            stats["excluded_types"][trna_type] += 1
            continue

        counts[(trna_type, anticodon)] += 1
        stats["records_included"] += 1

    stats["excluded_types"] = dict(sorted(stats["excluded_types"].items()))
    return counts, stats


def wobble_coefficient(trna_type: str, anticodon: str, codon: str, superkingdom: str) -> float:
    if trna_type == "Ile2":
        if superkingdom == "bacteria" and anticodon == "CAT" and codon == "ATA":
            return LYSIDINE_ILE2_COEFFICIENT
        return 0.0

    codon_aa = GENETIC_CODE[codon]
    trna_aa = AA_THREE_TO_SINGLE[trna_type]
    if trna_aa != codon_aa:
        return 0.0

    if len(anticodon) != 3 or set(anticodon) - {"A", "C", "G", "T"}:
        return 0.0

    if anticodon[2] != reverse_complement(codon[0])[0]:
        return 0.0
    if anticodon[1] != reverse_complement(codon[1])[0]:
        return 0.0

    return WOBBLE_COEFFICIENTS.get((anticodon[0], codon[2]), 0.0)


def derive_weights(
    counts: Counter[tuple[str, str]],
    superkingdom: str,
) -> tuple[dict[str, float], dict[str, float]]:
    absolute: dict[str, float] = {}
    for codon in SENSE_CODONS:
        total = 0.0
        for (trna_type, anticodon), copy_number in counts.items():
            total += copy_number * wobble_coefficient(trna_type, anticodon, codon, superkingdom)
        absolute[codon] = total

    nonzero = [value for value in absolute.values() if value > 0]
    if not nonzero:
        raise ValueError("No nonzero tAI W_i values could be derived")

    max_abs = max(nonzero)
    relative = {
        codon: absolute_value / max_abs
        for codon, absolute_value in absolute.items()
    }

    nonzero_relative = [value for value in relative.values() if value > 0]
    geometric_mean = math.exp(sum(math.log(value) for value in nonzero_relative) / len(nonzero_relative))
    for codon, value in list(relative.items()):
        if value <= 0:
            relative[codon] = geometric_mean

    return relative, absolute


def write_weights(path: Path, weights: dict[str, float]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["codon", "weight"])
        for codon in SENSE_CODONS:
            writer.writerow([codon, f"{weights[codon]:.6f}"])


def serializable_counts(counts: Counter[tuple[str, str]]) -> dict[str, int]:
    return {
        f"{trna_type}:{anticodon}": count
        for (trna_type, anticodon), count in sorted(counts.items())
    }


def build_host(host: HostSpec, args: argparse.Namespace) -> dict:
    tarball_data, _ = read_or_fetch_tarball(host, args)
    trnascan_text = extract_member_text(tarball_data, host.tRNAscan_member)
    counts, stats = parse_trnascan_counts(trnascan_text)
    weights, absolute = derive_weights(counts, host.superkingdom)

    output_path = OUTPUT_DIR / f"{host.key}.csv"
    write_weights(output_path, weights)

    return {
        "organism": host.organism,
        "genome": host.genome,
        "domain": host.domain,
        "superkingdom": host.superkingdom,
        "source_url": host.tarball_url,
        "tRNAscan_member": host.tRNAscan_member,
        "records_seen": stats["records_seen"],
        "records_included": stats["records_included"],
        "records_excluded": stats["records_excluded"],
        "excluded_types": stats["excluded_types"],
        "anticodon_copy_counts": serializable_counts(counts),
        "rows_written": len(weights),
        "min_weight": round(min(weights.values()), 6),
        "max_weight": round(max(weights.values()), 6),
        "zero_absolute_weights_replaced": sum(1 for value in absolute.values() if value <= 0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hosts",
        nargs="+",
        choices=sorted(HOSTS),
        default=sorted(HOSTS),
        help="Host keys to build.",
    )
    parser.add_argument(
        "--raw-dir",
        help="Optional directory for raw GtRNAdb tarball caching.",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Read raw tarballs from --raw-dir instead of making network requests.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Network timeout in seconds for each GtRNAdb request.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    host_metadata = {}
    for host_key in args.hosts:
        host_metadata[host_key] = build_host(HOSTS[host_key], args)

    metadata = {
        "generated_at": date.today().isoformat(),
        "source_database": "GtRNAdb",
        "source_database_url": BASE_URL,
        "source_release": "GtRNAdb Data Release 22 (September 2024)",
        "method": (
            "Gene-copy tAI weights derived from GtRNAdb tRNAscan-SE anticodon "
            "copy counts. W_i is the sum over matching tRNAs of copy_number * "
            "(1 - s_ij), using dos Reis et al. (2004) optimized wobble "
            "penalties: s_IU=0, s_GC=0, s_UA=0, s_CG=0, s_GU=0.41, "
            "s_IC=0.28, s_IA=0.9999, s_UG=0.68, s_LA=0.89. Anticodon A at "
            "wobble is treated as inosine; bacterial Ile2-CAT is treated as "
            "lysidine-modified and decodes ATA. Weights are normalized by the "
            f"maximum W_i; {PSEUDOCOUNT_REPLACEMENT}."
        ),
        "hosts": host_metadata,
    }
    with (OUTPUT_DIR / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
