This directory contains per-host empirical codon-pair score tables named
`<host>.csv`.

Bundled tables:

- `ecoli.csv` — *Escherichia coli*, NCBI taxid 562
- `hsapiens.csv` — *Homo sapiens*, NCBI taxid 9606
- `mmusculus.csv` — *Mus musculus*, NCBI taxid 10090
- `scerevisiae.csv` — *Saccharomyces cerevisiae*, NCBI taxid 4932
- `spombe.csv` — *Schizosaccharomyces pombe*, NCBI taxid 4896

These bundled tables were generated from RefSeq genomic CoCoPUTs/HIVE-CUTs
codon and bicodon counts served by FDA DNA HIVE:

```text
https://dnahive.fda.gov/dna.cgi?cmd=codon_usage&id=537&mode=cocoputs
```

The builder fetches CoCoPUTs service `537` through the
`ionTaxidCollapseExt` endpoint using:

- `fileSource=Refseq_species.tsv`, `plen=3` for codon counts
- `fileSource=Refseq_Bicod.tsv`, `plen=6` for bicodon counts
- `filterInColName=["Organelle"]`, `filterIn=["genomic"]`
- `searchDeep=true`

Per-host source URLs, taxids, source counts, and score ranges are recorded in
`metadata.json`.

Supported formats:

```text
codon1,codon2,cps
AAA,AAC,0.42
```

or

```text
pair,cps
AAAAAC,0.42
```

`cps` should be the empirical codon-pair score for the non-stop codon pair.

The bundled files use the `codon1,codon2,cps` format and contain all 61 x 61
sense-codon pairs. Stop-containing pairs are excluded, and runtime CPS
scoring skips stop-containing pairs such as the final sense-to-stop context
in a complete CDS.

Score construction:

1. Fetch host-level RefSeq genomic codon counts and bicodon counts from
   CoCoPUTs/HIVE-CUTs service `537`.
2. For each sense-codon pair, compute
   `ln((observed + 0.5) / (expected + 0.5))`.
3. Compute `expected` as
   `amino_acid_pair_count * P(codon1 | amino_acid1) * P(codon2 | amino_acid2)`.
4. Center each host table by subtracting the mean score over all 61 x 61
   sense-codon pairs.

Regenerate the bundled tables with:

```bash
python3 scripts/fetch_cocoputs_codon_pair_tables.py
```

When a host file is absent, the notebook hides the codon-pair weight control
for that host and the GA uses a neutral `0.0` codon-pair contribution.

Recommended data path:

1. Build or obtain host-specific codon-pair scores from a curated coding
   sequence corpus.
2. Exclude stop-containing codon pairs unless you intentionally want to score
   terminal contexts in downstream code.
3. Use log-ratio or another documented empirical codon-pair score consistently
   across all pairs in a host table.
4. Name the file with the same host key used in `data/codon_tables/`, for
   example `ecoli.csv` or `hsapiens.csv`.
