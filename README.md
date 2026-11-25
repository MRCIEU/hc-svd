# hc-svd

`hcsvd` computes **haplotype components (HCs)** from a combined chunk-length matrix: the output of [`combine`](https://github.com/MRCIEU/combine-pbwt-sparsepaint), which merges `pbwt -paintSparse` paintings across chromosomes.

It is the HC step of [pipeline-nf-haplotype-components](https://github.com/MRCIEU/pipeline-nf-haplotype-components), which runs the whole calculation from phased VCFs.

HCs are the top singular vectors of log10(matrix + 1), scaled by the square roots of their singular values: **HCs = U·√D**. That's the definition in [SparsePainter's HC pipeline](https://github.com/YaolingYang/SparsePainter/tree/main/painting-pipeline/Compute%20haplotype%20components%20(HCs)), whose SVD step is written in R (`Matrix` + `sparsesvd`). `hcsvd` gives the same HCs, adds options for relatives and sample subsets, and uses a parallel randomised SVD with 64-bit indexing, so it isn't limited to R's 2³¹ − 1 matrix entries and scales to biobank-sized matrices.

## Build

```
cargo build --release
./target/release/hcsvd --help
```

## Use

```
hcsvd --matrix combined.rows.bin --samples samples.txt --nhcs 150 --out HCs.csv \
    [--sums combined.sums] [--diagnostics HC_diagnostics.tsv] \
    [--remove_pairs pairs.txt] [--fit_samples unrelated.txt] [--project] [--reweight] [--threads N]
```

### Inputs

- **`--matrix`:** `combine`'s matrix, in either format (detected automatically):
  - the binary row stream from `combine --binary` (`<prefix>.rows.bin`), recommended for large matrices;
  - the text `index1 index2 value` (1-based), gzipped or not (`<prefix>.txt.gz`).
- **`--samples`:** sample IDs, one per line, in the order of the painted files. Matrix index *i* is line *i*.
- **`--sums`:** `combine`'s row sums (`<prefix>.sums`, `index1 sum`), each row's total before `combine --restrict` dropped entries. Needed for `--reweight`; also used in the diagnostics.

### Options

| Option | |
|---|---|
| `--nhcs` | number of HCs |
| `--remove_pairs FILE` | sample-ID pairs (e.g. relatives) whose entries are removed, in both directions. The first two whitespace-separated columns are used; pairs with an ID not in `--samples` (or a header line) are skipped. |
| `--fit_samples FILE` | only these samples' rows and columns are used for the SVD, and only they get HCs, unless `--project` is given |
| `--project` | with `--fit_samples`: the other samples get HCs too, by projecting their rows (entries to fitted samples only) onto the SVD: row × V·D^(−1/2) |
| `--reweight` | rescale each row to its total in `--sums` after entries were removed (by `combine --restrict`, `--remove_pairs` or `--fit_samples`) |
| `--diagnostics FILE` | per-sample diagnostics (below) |
| `--threads` | threads (default: all) |
| `--oversample`, `--power_iterations`, `--tol`, `--seed` | randomised SVD settings (below) |

To fit HCs on unrelated individuals and give their relatives HCs by projection, use `--fit_samples`, `--project`, `--remove_pairs` and `--reweight` together (see the pipeline's README, "Handling relatives").

### Outputs

- **`--out`:** CSV with columns `sample, HC1, …, HCk`. One row per fitted (and, with `--project`, projected) sample, in `--samples` order. Fewer than `--nhcs` columns if the matrix has fewer non-zero singular values.
- **`--diagnostics`:** tab-separated, one row per sample, with columns `sample, n_entries, total_precap, total_kept, frac_kept, largest_share, frac_removed, frac_outside, n_entries_used, reweight_factor, status` (`fitted`, `projected` or `not fitted`). Missing values are written `NA`.

## Method

The order of operations on each row is:
1. remove listed pairs;
2. keep entries to fitted samples;
3. reweight;
4. log10(x + 1).

Rows are processed as the matrix is read, and stored in compressed sparse row form: 32-bit column indices and values, 64-bit row offsets, plus a transposed copy.

**The SVD** is randomised (Halko, Martinsson & Tropp, *SIAM Review* 2011):
1. multiply by a random block of `--nhcs` + `--oversample` columns;
2. repeat power iterations (products with Aᵀ and A, re-orthonormalised by QR) until the top `--nhcs` singular-value estimates change by less than `--tol` (relative) between iterations, or `--power_iterations` is reached;
3. take the SVD of the small projected matrix.

Signs are fixed so that each component's largest entry in U is positive. Matrix products run in parallel (`rayon`); dense factorisations use [`faer`](https://github.com/sarah-quinones/faer-rs).

**Defaults:** oversampling = the larger of 50 and `--nhcs`; `--tol` 1e-12; at most 100 power iterations. With these, HCs agree with the R implementation (`sparsesvd`) to within about 10⁻⁵ of each HC's standard deviation on painted 1000 Genomes data (20 and 150 HCs). In the flattest part of the spectrum, the remaining difference is mostly `sparsesvd`'s own convergence.

## Testing

```
cargo test
```

- **Unit tests** cover the sparse products and the randomised SVD against an exact SVD.
- **`tests/r_reference.rs`** checks agreement with an R implementation of the same calculation (R `Matrix` + `sparsesvd`, with the same options).
  - It uses painted 1000 Genomes data (`tests/data`: 904 samples, 20 HCs; made by `testdata/make_hc_testdata.sh` in pipeline-nf-haplotype-components v0.1.0, at full HapMap 3 density with the production painting settings) and four sets of options: plain; relatives' pairs removed and reweighted; fit on a subset; and fit on the unrelated samples with relatives projected.
  - HCs must agree up to sign (|correlation| ≥ 0.999999, largest difference ≤ 10⁻⁴ of each HC's SD), and the diagnostics must match.
  - The R outputs are stored in `tests/data/r`; `tests/make_r_reference.sh <calculate_haplotype_components.R>` regenerates them (needs docker).
- **`tests/hc_step.rs`** checks what the options do, on the same data:
  - HCs separate the superpopulations, and relatives' HCs are shrunk without any handling;
  - `--remove_pairs` with `--reweight` restores them, and `--reweight` alone changes nothing;
  - `--fit_samples` with a group, or every sample;
  - projecting relatives onto the unrelated: they keep their size and land nearest their own superpopulation;
  - an exact check of the projection formula, using a projected copy of a fitted sample.
- **`tests/cli.rs`** covers the binary and text matrix formats (same output), a sample with no entries, the licence flags, and errors on invalid input.

CI (GitHub Actions) runs formatting, lints and the tests on pushes and pull requests to `main`.

## Licence

MIT (see [LICENSE](LICENSE)). `hcsvd` also incorporates third-party Rust crates under permissive licences, whose notices are in [NOTICE](NOTICE). `hcsvd -l` prints a summary and `hcsvd --licences` the full text.

After changing dependencies, regenerate NOTICE with [cargo-about](https://github.com/EmbarkStudios/cargo-about): `cargo about generate notice.hbs -o NOTICE`.
