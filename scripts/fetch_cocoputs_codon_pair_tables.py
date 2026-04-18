#!/usr/bin/env python3
"""
Fetch CoCoPUTs/HIVE-CUTs codon and bicodon counts and derive CPS tables.

The output format matches src.utils.load_codon_pair_table():

    codon1,codon2,cps
    AAA,AAC,0.123456

Scores are log observed/expected codon-pair scores over sense-codon pairs.
Expected counts are conditioned on the encoded amino-acid pair so the score
captures synonymous codon-pair preference rather than amino-acid composition.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "codon_pair_tables"

BASE_URL = "https://dnahive.fda.gov/dna.cgi"
COCOPUTS_SERVICE_ID = "537"
COCOPUTS_SERVICE_TITLE = (
    "CoCoPUTs, TissueCoCoPUTs, and CancerCoCoPUTs updated September, 2021"
)
COCOPUTS_SERVICE_MODIFIED = "2021-10-01 14:56:16-04:00"
PSEUDOCOUNT = 0.5

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

ALL_CODONS = [a + b + c for a in "TCAG" for b in "TCAG" for c in "TCAG"]
SENSE_CODONS = [codon for codon in ALL_CODONS if GENETIC_CODE[codon] != "*"]
CODONS_BY_AA: Dict[str, list[str]] = defaultdict(list)
for _codon in SENSE_CODONS:
    CODONS_BY_AA[GENETIC_CODE[_codon]].append(_codon)


@dataclass(frozen=True)
class HostSpec:
    key: str
    organism: str
    taxid: str
    source_prefix: str = "Refseq_"
    source_name: str = "RefSeq"
    organelle: str = "genomic"


HOSTS = {
    "ecoli": HostSpec("ecoli", "Escherichia coli", "562"),
    "hsapiens": HostSpec("hsapiens", "Homo sapiens", "9606"),
    "mmusculus": HostSpec("mmusculus", "Mus musculus", "10090"),
    "scerevisiae": HostSpec("scerevisiae", "Saccharomyces cerevisiae", "4932"),
    "spombe": HostSpec("spombe", "Schizosaccharomyces pombe", "4896"),
}


def build_url(host: HostSpec, datatype: str) -> str:
    if datatype == "codon":
        file_source = f"{host.source_prefix}species.tsv"
        plen = "3"
    elif datatype == "bicod":
        file_source = f"{host.source_prefix}Bicod.tsv"
        plen = "6"
    else:
        raise ValueError(f"Unsupported datatype: {datatype}")

    params = {
        "cmd": "ionTaxidCollapseExt",
        "svcType": "svc-codon-usage",
        "objId": COCOPUTS_SERVICE_ID,
        "fileSource": file_source,
        "plen": plen,
        "taxid": host.taxid,
        "filterInColName": json.dumps(["Organelle"], separators=(",", ":")),
        "filterIn": json.dumps([host.organelle], separators=(",", ":")),
        "searchDeep": "true",
        "raw": "1",
    }
    return f"{BASE_URL}?{urlencode(params)}"


def fetch_text(url: str, timeout: int) -> str:
    request = Request(url, headers={"User-Agent": "CDS-Optimizer data builder"})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def parse_key_value_table(text: str, expected_key_len: int) -> Tuple[Dict[str, int], Dict[str, str]]:
    counts: Dict[str, int] = {}
    metadata: Dict[str, str] = {}

    rows = csv.reader(text.splitlines())
    header = next(rows, None)
    if header != ["id", "value"]:
        raise ValueError("Unexpected CoCoPUTs response format; expected 'id,value' header")

    for row in rows:
        if len(row) < 2:
            continue
        key = row[0].strip().strip('"')
        value = row[1].strip()
        key_upper = key.upper()
        if len(key_upper) == expected_key_len and set(key_upper) <= {"A", "C", "G", "T"}:
            counts[key_upper] = int(value)
        else:
            metadata[key] = value

    return counts, metadata


def derive_cps(codon_counts: Dict[str, int], pair_counts: Dict[str, int]) -> list[tuple[str, str, float]]:
    aa_totals: Dict[str, int] = defaultdict(int)
    aa_pair_totals: Dict[tuple[str, str], int] = defaultdict(int)

    for codon in SENSE_CODONS:
        aa_totals[GENETIC_CODE[codon]] += codon_counts.get(codon, 0)

    for codon1 in SENSE_CODONS:
        aa1 = GENETIC_CODE[codon1]
        for codon2 in SENSE_CODONS:
            aa2 = GENETIC_CODE[codon2]
            aa_pair_totals[(aa1, aa2)] += pair_counts.get(codon1 + codon2, 0)

    rows: list[tuple[str, str, float]] = []
    for codon1 in SENSE_CODONS:
        aa1 = GENETIC_CODE[codon1]
        p1 = (
            (codon_counts.get(codon1, 0) + PSEUDOCOUNT)
            / (aa_totals[aa1] + PSEUDOCOUNT * len(CODONS_BY_AA[aa1]))
        )
        for codon2 in SENSE_CODONS:
            aa2 = GENETIC_CODE[codon2]
            p2 = (
                (codon_counts.get(codon2, 0) + PSEUDOCOUNT)
                / (aa_totals[aa2] + PSEUDOCOUNT * len(CODONS_BY_AA[aa2]))
            )
            observed = pair_counts.get(codon1 + codon2, 0)
            expected = aa_pair_totals[(aa1, aa2)] * p1 * p2
            cps = math.log((observed + PSEUDOCOUNT) / (expected + PSEUDOCOUNT))
            rows.append((codon1, codon2, cps))

    mean_cps = sum(row[2] for row in rows) / len(rows)
    return [(codon1, codon2, cps - mean_cps) for codon1, codon2, cps in rows]


def write_cps_table(path: Path, rows: Iterable[tuple[str, str, float]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["codon1", "codon2", "cps"])
        for codon1, codon2, cps in rows:
            writer.writerow([codon1, codon2, f"{cps:.6f}"])


def read_or_fetch(host: HostSpec, datatype: str, args: argparse.Namespace) -> tuple[str, str]:
    url = build_url(host, datatype)
    cache_path = None
    if args.raw_dir:
        cache_path = Path(args.raw_dir) / f"{host.key}_{datatype}.csv"

    if args.no_fetch:
        if cache_path is None:
            raise ValueError("--no-fetch requires --raw-dir")
        return cache_path.read_text(), url

    text = fetch_text(url, args.timeout)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text)
    return text, url


def int_metadata_value(metadata: Dict[str, str], key: str) -> int | None:
    value = metadata.get(key)
    return int(value) if value is not None and value.isdigit() else None


def build_host(host: HostSpec, args: argparse.Namespace) -> dict:
    codon_text, codon_url = read_or_fetch(host, "codon", args)
    bicod_text, bicod_url = read_or_fetch(host, "bicod", args)

    codon_counts, codon_meta = parse_key_value_table(codon_text, 3)
    pair_counts, bicod_meta = parse_key_value_table(bicod_text, 6)

    missing_codons = [codon for codon in SENSE_CODONS if codon not in codon_counts]
    missing_pairs = [
        codon1 + codon2
        for codon1 in SENSE_CODONS
        for codon2 in SENSE_CODONS
        if codon1 + codon2 not in pair_counts
    ]
    if missing_codons or missing_pairs:
        raise ValueError(
            f"{host.key}: source data incomplete "
            f"({len(missing_codons)} missing codons, {len(missing_pairs)} missing pairs)"
        )

    rows = derive_cps(codon_counts, pair_counts)
    output_path = OUTPUT_DIR / f"{host.key}.csv"
    write_cps_table(output_path, rows)

    values = [row[2] for row in rows]
    return {
        "organism": host.organism,
        "taxid": host.taxid,
        "source": host.source_name,
        "organelle": host.organelle,
        "codon_url": codon_url,
        "bicodon_url": bicod_url,
        "rows_written": len(rows),
        "source_codon_count": int_metadata_value(codon_meta, "#codon"),
        "source_bicodon_count": int_metadata_value(bicod_meta, "#codon"),
        "source_cds_count": int_metadata_value(bicod_meta, "#CDS"),
        "taxa_collapsed": int_metadata_value(bicod_meta, "collapse"),
        "mean_cps_after_centering": round(sum(values) / len(values), 12),
        "min_cps": round(min(values), 6),
        "max_cps": round(max(values), 6),
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
        help="Optional directory for raw CoCoPUTs response caching.",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Read raw responses from --raw-dir instead of making network requests.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Network timeout in seconds for each CoCoPUTs request.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    host_metadata = {}
    for host_key in args.hosts:
        host = HOSTS[host_key]
        host_metadata[host_key] = build_host(host, args)

    metadata = {
        "generated_at": date.today().isoformat(),
        "source_database": "CoCoPUTs / HIVE-CUTs",
        "source_service_id": COCOPUTS_SERVICE_ID,
        "source_service_title": COCOPUTS_SERVICE_TITLE,
        "source_service_modified": COCOPUTS_SERVICE_MODIFIED,
        "method": (
            "For each sense-codon pair, CPS = ln((observed + 0.5) / "
            "(expected + 0.5)). Expected count = amino-acid-pair count * "
            "P(codon1 | amino_acid1) * P(codon2 | amino_acid2). Scores are "
            "centered per host by subtracting the mean over all 61x61 "
            "sense-codon pairs."
        ),
        "hosts": host_metadata,
    }
    with (OUTPUT_DIR / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
