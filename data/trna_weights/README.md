This directory contains per-host tRNA Adaptation Index (tAI) weight tables
named `<host>.csv`.

Bundled tables:

- `ecoli.csv` — *Escherichia coli* str. K-12 substr. MG1655
- `hsapiens.csv` — *Homo sapiens* GRCh38/hg38
- `mmusculus.csv` — *Mus musculus* GRCm39/mm39
- `scerevisiae.csv` — *Saccharomyces cerevisiae* S288C
- `spombe.csv` — *Schizosaccharomyces pombe* 972h-

These bundled tables were generated from GtRNAdb tRNAscan-SE gene predictions:

```text
https://gtrnadb.ucsc.edu/
```

Per-host source URLs, tRNAscan-SE archive members, anticodon copy counts, and
score ranges are recorded in `metadata.json`.

Expected columns:

```text
codon,weight
AAA,0.31
AAC,0.85
...
```

`weight` is the relative adaptiveness `w_i` value for each sense codon.
Bundled files contain all 61 sense codons; stop codons are excluded.

Score construction:

1. Fetch host-specific GtRNAdb `*-tRNAs.tar.gz` archives.
2. Parse high-confidence or non-pseudogene tRNAscan-SE records for tRNA type
   and anticodon copy counts.
3. Exclude pseudogenes, undetermined predictions, selenocysteine tRNAs,
   suppressor tRNAs, and initiator Met/fMet tRNAs.
4. Compute absolute codon adaptiveness `W_i` from anticodon copy counts using
   dos Reis et al. (2004) optimized wobble penalties.
5. Treat anticodon `A` at wobble as inosine. For bacteria, treat `Ile2-CAT`
   as lysidine-modified and decoding `ATA`.
6. Normalize by the maximum `W_i`. If a codon has zero absolute availability,
   replace it with the geometric mean of nonzero normalized weights.

Regenerate the bundled tables with:

```bash
python3 scripts/fetch_gtrnadb_trna_weights.py
```

These gene-copy weights are tRNA availability proxies. They are not measured
tRNA expression or tissue-specific tRNA activity.

When a host file is absent, the notebook hides the tAI weight control for that
host and the GA uses a neutral `0.0` tAI contribution.

Recommended data path:

1. Collect host-specific tRNA gene copy or expression data from a curated
   source such as GtRNAdb or a host-specific annotation source.
2. Convert anticodon availability into per-codon `w` values using the
   tAI model you want to benchmark against.
3. Normalize weights into `(0, 1]` before writing the CSV.
4. Name the file with the same host key used in `data/codon_tables/`, for
   example `ecoli.csv` or `hsapiens.csv`.
