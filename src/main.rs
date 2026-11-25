//! hcsvd: haplotype components (HCs) from combine's chunk-length matrix.
//!
//! HCs are the top singular vectors of log10(matrix + 1), scaled by the square roots of the singular
//! values (U * sqrt(D)). Optionally: entries between listed pairs of samples (e.g. relatives) are
//! removed, the SVD uses only some samples (`--fit_samples`), the others are projected onto it
//! (`--project`: row * V * D^(-1/2)), and rows are rescaled to their totals before entries were
//! removed (`--reweight`). The SVD is randomised (see rsvd.rs).

mod input;
mod rsvd;
mod sparse;

use anyhow::{Result, bail, ensure};
use clap::Parser;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};

use sparse::Csr;

#[derive(Parser, Debug)]
#[command(
    version,
    about = "Haplotype components (HCs) from combine's chunk-length matrix",
    override_usage = "hcsvd --matrix <MATRIX> --samples <SAMPLES> --nhcs <NHCS> --out <OUT> [OPTIONS]"
)]
struct Args {
    /// combine's matrix: <prefix>.rows.bin (combine --binary) or <prefix>.txt.gz ("index1 index2 value", 1-based)
    #[arg(long, required_unless_present_any = ["licences", "licences_summary"], hide_default_value = true, default_value = "")]
    matrix: String,

    /// Sample IDs, one per line, in the order of the painted BCF/VCFs
    #[arg(long, required_unless_present_any = ["licences", "licences_summary"], hide_default_value = true, default_value = "")]
    samples: String,

    /// Number of HCs to compute
    #[arg(long, required_unless_present_any = ["licences", "licences_summary"], hide_default_value = true, default_value_t = 0)]
    nhcs: usize,

    /// Output file for the HCs (CSV: sample, HC1, HC2, ...)
    #[arg(long, required_unless_present_any = ["licences", "licences_summary"], hide_default_value = true, default_value = "")]
    out: String,

    /// combine's row sums (<prefix>.sums: "index1 sum"), before combine --restrict dropped entries. Needed for --reweight; also used in the diagnostics
    #[arg(long)]
    sums: Option<String>,

    /// File of sample ID pairs (e.g. relatives) whose entries are removed, in both directions: the first two whitespace-separated columns, other columns ignored. Pairs with an ID not in --samples (or a header line) are skipped
    #[arg(long = "remove_pairs", alias = "remove-pairs")]
    remove_pairs: Option<String>,

    /// File of sample IDs, one per line: only these samples' rows and columns are used for the SVD (their entries to other samples are removed), and only they get HCs. IDs not in --samples are skipped
    #[arg(long = "fit_samples", alias = "fit-samples")]
    fit_samples: Option<String>,

    /// With --fit_samples: also give the other samples HCs, by projecting their rows (entries to fitted samples only, after --remove_pairs and --reweight) onto the SVD of the fitted samples
    #[arg(long)]
    project: bool,

    /// Rescale each row's values to add up to its sum in --sums (its total before combine --restrict and --remove_pairs removed entries) before the SVD
    #[arg(long)]
    reweight: bool,

    /// Output file for per-sample diagnostics (tab-separated)
    #[arg(long)]
    diagnostics: Option<String>,

    /// Number of threads [default: all]
    #[arg(long)]
    threads: Option<usize>,

    /// Randomised SVD: extra random directions beyond --nhcs [default: the larger of 50 and --nhcs]. More means fewer power iterations; the cost per iteration grows with --nhcs + --oversample
    #[arg(long)]
    oversample: Option<usize>,

    /// Randomised SVD: maximum number of power iterations
    #[arg(
        long = "power_iterations",
        alias = "power-iterations",
        default_value_t = 100
    )]
    power_iterations: usize,

    /// Randomised SVD: stop the power iterations once the top --nhcs singular values change by less than this (relative) between iterations; 0 = always run --power_iterations
    #[arg(long, default_value_t = 1e-12)]
    tol: f64,

    /// Randomised SVD: random seed
    #[arg(long, default_value_t = 1)]
    seed: u64,

    /// Print a summary of licence information
    #[arg(short = 'l')]
    licences_summary: bool,

    /// Print full licence information, including all third-party notices
    #[arg(long, alias = "licenses")]
    licences: bool,
}

/// The NOTICE file: this program's licence, and the notices of its third-party dependencies
const NOTICE: &str = include_str!("../NOTICE");

/// The opening statement of NOTICE, i.e. everything before the first third-party notice
fn get_notice_summary() -> &'static str {
    NOTICE
        .split_once("\n---")
        .map_or(NOTICE, |(summary, _)| summary)
}

/// Per-sample bookkeeping, as in the R implementation's diagnostics.
#[derive(Default, Clone)]
struct Row {
    n_entries: usize,
    total_precap: Option<f64>,
    total_kept: f64,
    largest: Option<f64>,
    total_removed: f64,
    total_outside: f64,
    n_entries_used: usize,
    total_used: f64,
    reweight_factor: f64,
    fitted: bool,
    projected: bool,
}

fn main() {
    if let Err(e) = run(Args::parse()) {
        eprintln!("Error: {e:#}");
        std::process::exit(1);
    }
}

fn run(a: Args) -> Result<()> {
    if a.licences {
        print!("{NOTICE}");
        return Ok(());
    }
    if a.licences_summary {
        println!("{}", get_notice_summary());
        println!(
            "Use \"hcsvd --licences\" to print the full text, including all third-party notices."
        );
        return Ok(());
    }
    if a.reweight && a.sums.is_none() {
        bail!("--reweight needs --sums");
    }
    if a.project && a.fit_samples.is_none() {
        bail!("--project needs --fit_samples");
    }
    if a.remove_pairs.is_some() && !a.reweight {
        eprintln!(
            "Warning: --remove_pairs without --reweight. Rows that lose entries (e.g. a relative's) keep only the rest of their total, \
             which can shrink their HCs towards 0; --reweight restores their totals."
        );
    }
    let threads = a
        .threads
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .ok();
    faer::set_global_parallelism(if threads > 1 {
        faer::Par::rayon(threads)
    } else {
        faer::Par::Seq
    });

    // --- inputs ---
    eprintln!("Reading samples");
    let samples = input::read_samples(&a.samples)?;
    let nind = samples.len();
    let index: HashMap<&str, usize> = samples
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();
    eprintln!("  {nind} samples");
    let sums = match &a.sums {
        Some(p) => input::read_sums(p, nind)?,
        None => vec![None; nind],
    };
    let mut rows: Vec<Row> = sums
        .iter()
        .map(|&s| Row {
            total_precap: s,
            reweight_factor: 1.0,
            fitted: true,
            ..Default::default()
        })
        .collect();

    let mut removed_pairs: HashSet<(u32, u32)> = HashSet::new();
    if let Some(p) = &a.remove_pairs {
        let (pairs, skipped) = input::read_pairs(p, &index)?;
        for &(i, j) in &pairs {
            removed_pairs.insert((i as u32, j as u32));
            removed_pairs.insert((j as u32, i as u32));
        }
        eprintln!(
            "Removing pairs: {} pairs ({skipped} skipped: an ID is not in {})",
            pairs.len(),
            a.samples
        );
    }
    if let Some(p) = &a.fit_samples {
        let (fit, skipped) = input::read_id_set(p, &index)?;
        ensure!(
            !fit.is_empty(),
            "none of the samples in {p} are in {}",
            a.samples
        );
        let fitset: HashSet<usize> = fit.iter().copied().collect();
        for (i, r) in rows.iter_mut().enumerate() {
            r.fitted = fitset.contains(&i);
            r.projected = a.project && !r.fitted;
        }
        eprintln!(
            "Keeping only the fit samples: {} of {nind} samples kept ({skipped} IDs in {p} skipped: not in {}){}",
            fit.len(),
            a.samples,
            if a.project {
                format!("; the other {} will be projected", nind - fit.len())
            } else {
                String::new()
            }
        );
    }
    let fit_pos: Vec<Option<u32>> = {
        let mut k = 0u32;
        rows.iter()
            .map(|r| {
                r.fitted.then(|| {
                    k += 1;
                    k - 1
                })
            })
            .collect()
    };
    let nfit = rows.iter().filter(|r| r.fitted).count();
    let proj_idx: Vec<usize> = (0..nind).filter(|&i| rows[i].projected).collect();
    ensure!(a.nhcs >= 1, "--nhcs must be at least 1");
    ensure!(
        a.nhcs < nfit,
        "--nhcs ({}) must be less than the number of samples fitted ({nfit})",
        a.nhcs
    );

    // --- matrix: row operations as rows stream in (remove pairs, keep fitted columns, reweight, log) ---
    eprintln!("Reading matrix");
    let mut fit_m = Csr::new(nfit);
    let mut proj_m = Csr::new(nfit);
    let (mut ncol, mut nval) = (Vec::new(), Vec::new());
    let (mut n_removed, mut rows_removed) = (0usize, 0usize);
    let mut missing_sums = false;
    let total = input::for_each_row(&a.matrix, nind, |i, cols, vals| {
        let r = &mut rows[i];
        r.n_entries = cols.len();
        r.total_kept = vals.iter().map(|&v| v as f64).sum();
        r.largest = vals.iter().map(|&v| v as f64).reduce(f64::max);
        ncol.clear();
        nval.clear();
        let mut lost = false;
        for (&j, &v) in cols.iter().zip(vals) {
            let v = v as f64;
            if !removed_pairs.is_empty() && removed_pairs.contains(&(i as u32, j)) {
                r.total_removed += v;
                n_removed += 1;
                lost = true;
                continue;
            }
            match fit_pos[j as usize] {
                None => r.total_outside += v,
                Some(c) if r.fitted || r.projected => {
                    ncol.push(c);
                    nval.push(v);
                }
                Some(_) => {}
            }
        }
        rows_removed += lost as usize;
        r.n_entries_used = ncol.len();
        r.total_used = nval.iter().sum();
        if a.reweight && r.n_entries_used > 0 {
            match r.total_precap {
                Some(t) => r.reweight_factor = t / r.total_used,
                None => missing_sums = true,
            }
        }
        if r.fitted || r.projected {
            let f = r.reweight_factor;
            let logv: Vec<f32> = nval.iter().map(|&v| (v * f + 1.0).log10() as f32).collect();
            if r.fitted {
                fit_m.push_row(&ncol, &logv)
            } else {
                proj_m.push_row(&ncol, &logv)
            }
        }
        Ok(())
    })?;
    if missing_sums {
        bail!(
            "rows in the matrix are missing from {}",
            a.sums.as_deref().unwrap_or("the sums file")
        );
    }
    eprintln!("  {total} entries");
    if a.remove_pairs.is_some() {
        eprintln!("  removed {n_removed} entries from {rows_removed} rows");
    }
    if a.reweight {
        let f: Vec<f64> = rows
            .iter()
            .filter(|r| r.n_entries_used > 0)
            .map(|r| r.reweight_factor)
            .collect();
        if !f.is_empty() {
            eprintln!(
                "  reweighting: scale factors range from {:.6} to {:.6}",
                f.iter().copied().fold(f64::INFINITY, f64::min),
                f.iter().copied().fold(0.0, f64::max)
            );
        }
    }
    let nempty = rows
        .iter()
        .filter(|r| (r.fitted || r.projected) && r.n_entries_used == 0)
        .count();
    if nempty > 0 {
        eprintln!("  {nempty} samples have no entries in the matrix; their HCs will be 0");
    }

    // --- SVD of the fitted samples ---
    eprintln!(
        "Calculating SVD ({nfit} x {nfit}, {} entries; {} HCs)",
        fit_m.nnz(),
        a.nhcs
    );
    let fit_t = fit_m.transpose();
    let svd = rsvd::randomised_svd(
        &fit_m,
        &fit_t,
        a.nhcs,
        rsvd::Settings {
            oversample: a.oversample.unwrap_or(a.nhcs.max(50)),
            power_iterations: a.power_iterations,
            tol: a.tol,
            seed: a.seed,
        },
    )?;
    drop(fit_t);
    let k = svd.k;
    eprintln!("  {} power iterations", svd.iterations);
    if k < a.nhcs {
        eprintln!("  Warning: only {k} non-zero singular values; writing {k} HCs");
    }

    // --- HCs: U sqrt(D) for fitted samples; row * V * D^(-1/2) for projected ones ---
    eprintln!("Calculating HCs");
    let mut hcs = vec![0f64; nind * k];
    for i in 0..nind {
        if let Some(p) = fit_pos[i] {
            for c in 0..k {
                hcs[i * k + c] = svd.u[p as usize * k + c] * svd.d[c].sqrt();
            }
        }
    }
    if !proj_idx.is_empty() {
        eprintln!("Projecting {} samples", proj_idx.len());
        let pv = proj_m.mul_dense(&svd.v, k);
        for (t, &i) in proj_idx.iter().enumerate() {
            for c in 0..k {
                hcs[i * k + c] = pv[t * k + c] / svd.d[c].sqrt();
            }
        }
    }

    eprintln!("Writing HCs");
    let mut w = BufWriter::new(File::create(&a.out)?);
    write!(w, "sample")?;
    for c in 1..=k {
        write!(w, ",HC{c}")?;
    }
    writeln!(w)?;
    for i in 0..nind {
        if rows[i].fitted || rows[i].projected {
            write!(w, "{}", samples[i])?;
            for c in 0..k {
                write!(w, ",{}", hcs[i * k + c])?;
            }
            writeln!(w)?;
        }
    }
    w.flush()?;

    if let Some(p) = &a.diagnostics {
        eprintln!("Writing diagnostics");
        write_diagnostics(p, &samples, &rows)?;
    }
    Ok(())
}

/// R-style numbers: NA for missing, NaN and Inf as R writes them.
fn num(x: Option<f64>) -> String {
    match x {
        None => "NA".into(),
        Some(v) if v.is_nan() => "NaN".into(),
        Some(v) if v.is_infinite() => {
            if v > 0.0 {
                "Inf".into()
            } else {
                "-Inf".into()
            }
        }
        Some(v) => format!("{v}"),
    }
}

fn write_diagnostics(path: &str, samples: &[String], rows: &[Row]) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    writeln!(
        w,
        "sample\tn_entries\ttotal_precap\ttotal_kept\tfrac_kept\tlargest_share\tfrac_removed\tfrac_outside\tn_entries_used\treweight_factor\tstatus"
    )?;
    for (s, r) in samples.iter().zip(rows) {
        let status = if r.fitted {
            "fitted"
        } else if r.projected {
            "projected"
        } else {
            "not fitted"
        };
        writeln!(
            w,
            "{s}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{status}",
            r.n_entries,
            num(r.total_precap),
            num(Some(r.total_kept)),
            num(r.total_precap.map(|t| r.total_kept / t)),
            num(r.largest.map(|l| l / r.total_kept)),
            num(Some(r.total_removed / r.total_kept)),
            num((r.fitted || r.projected).then(|| r.total_outside / r.total_kept)),
            r.n_entries_used,
            num(Some(r.reweight_factor)),
        )?;
    }
    w.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notice_summary_stops_before_first_third_party_notice() {
        let summary = get_notice_summary();
        assert!(summary.starts_with("hc-svd is licensed under the MIT License"));
        assert!(!summary.contains("---"));
        assert!(summary.len() < NOTICE.len());
    }
}
