//! What hcsvd's options do to HCs, on painted 1000 Genomes data (tests/data: 904 samples in five
//! superpopulations, 154 of them in related pairs; 20 HCs): population structure, relatives' HCs with
//! and without --remove_pairs and --reweight, fit subsets, and projection (including an exact test of
//! the projection formula).

mod common;
use common::*;

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::sync::OnceLock;

/// The test data's samples, superpopulations and related pairs.
struct Data {
    samples: Vec<String>,
    superpop: HashMap<String, String>,
    pairs: Vec<(String, String)>,
    related: HashSet<String>,
}

fn d() -> &'static Data {
    static D: OnceLock<Data> = OnceLock::new();
    D.get_or_init(|| {
        let samples: Vec<String> = read_lines(&data("samples.txt"));
        let superpop = read_lines(&data("samples_superpop.txt"))
            .iter()
            .map(|l| {
                let f: Vec<&str> = l.split_whitespace().collect();
                (f[0].to_string(), f[1].to_string())
            })
            .collect();
        let pairs: Vec<(String, String)> = read_lines(&data("related_pairs.txt"))
            .iter()
            .map(|l| {
                let f: Vec<&str> = l.split_whitespace().collect();
                (f[0].to_string(), f[1].to_string())
            })
            .collect();
        let related = pairs
            .iter()
            .flat_map(|(a, b)| [a.clone(), b.clone()])
            .collect();
        Data {
            samples,
            superpop,
            pairs,
            related,
        }
    })
}

fn dir() -> PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| out_dir("hc_step")).clone()
}

/// A file of lines in the test output directory.
fn write_list(name: &str, lines: &[String]) -> String {
    let p = dir().join(name);
    fs::write(&p, lines.join("\n") + "\n").unwrap();
    p.to_str().unwrap().to_string()
}

fn path(name: &str) -> String {
    data(name).to_str().unwrap().to_string()
}

fn base() -> Vec<String> {
    [
        "--matrix",
        &path("combined.chunklengths.txt.gz"),
        "--samples",
        &path("samples.txt"),
        "--sums",
        &path("combined.chunklengths.sums"),
        "--nhcs",
        "20",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// HCs by sample, in file order.
struct Hcs {
    samples: Vec<String>,
    rows: Vec<Vec<f64>>,
    header: Vec<String>,
}

impl Hcs {
    fn get(&self, s: &str) -> &[f64] {
        let i = self.samples.iter().position(|x| x == s).unwrap();
        &self.rows[i]
    }
}

/// Diagnostics: one map of column -> value per sample.
type Diag = Vec<HashMap<String, String>>;

struct Run {
    log: String,
    hcs: Hcs,
    diag: Diag,
}

/// Runs hcsvd, which must succeed, with --out and --diagnostics in the test directory.
fn run(name: &str, args: &[String]) -> Run {
    let (out, tsv) = (
        dir().join(format!("{name}.csv")),
        dir().join(format!("{name}.tsv")),
    );
    let mut a = args.to_vec();
    a.extend([
        "--out".into(),
        out.to_str().unwrap().into(),
        "--diagnostics".into(),
        tsv.to_str().unwrap().into(),
    ]);
    let o = hcsvd(&a);
    let log = String::from_utf8_lossy(&o.stderr).to_string();
    assert!(o.status.success(), "{name}: hcsvd failed:\n{log}");
    let (header, rows) = read_table(&out, ',');
    let hcs = Hcs {
        samples: rows.iter().map(|r| r[0].clone()).collect(),
        rows: rows
            .iter()
            .map(|r| r[1..].iter().map(|v| v.parse().unwrap()).collect())
            .collect(),
        header,
    };
    let (dh, drows) = read_table(&tsv, '\t');
    let diag = drows
        .into_iter()
        .map(|r| dh.iter().cloned().zip(r).collect())
        .collect();
    Run { log, hcs, diag }
}

fn default_run() -> &'static Run {
    static R: OnceLock<Run> = OnceLock::new();
    R.get_or_init(|| run("default", &base()))
}

fn reweight_run() -> &'static Run {
    static R: OnceLock<Run> = OnceLock::new();
    R.get_or_init(|| {
        let mut a = base();
        a.push("--reweight".into());
        run("reweight", &a)
    })
}

fn with(extra: &[&str]) -> Vec<String> {
    let mut a = base();
    a.extend(extra.iter().map(|s| s.to_string()));
    a
}

fn unrelated_list() -> String {
    let d = d();
    let u: Vec<String> = d
        .samples
        .iter()
        .filter(|s| !d.related.contains(*s))
        .cloned()
        .collect();
    write_list("unrelated.txt", &u)
}

fn median(mut x: Vec<f64>) -> f64 {
    x.sort_by(f64::total_cmp);
    let n = x.len();
    if n % 2 == 1 {
        x[n / 2]
    } else {
        (x[n / 2 - 1] + x[n / 2]) / 2.0
    }
}

/// The largest absolute difference between the HCs of the samples in `a`, and the same samples in `b`.
fn maxdiff(a: &Hcs, b: &Hcs) -> f64 {
    a.samples
        .iter()
        .zip(&a.rows)
        .flat_map(|(s, r)| r.iter().zip(b.get(s)).map(|(x, y)| (x - y).abs()))
        .fold(0.0, f64::max)
}

/// The size of related samples' HC2-HC10 relative to unrelated samples of the same superpopulation
/// (median over superpopulations of the ratio of medians).
fn size_ratio(h: &Hcs) -> f64 {
    let d = d();
    let mut by_sp: HashMap<&str, (Vec<f64>, Vec<f64>)> = HashMap::new();
    for s in &d.samples {
        let n = h.get(s)[1..10].iter().map(|v| v * v).sum::<f64>().sqrt();
        let e = by_sp.entry(d.superpop[s].as_str()).or_default();
        if d.related.contains(s) {
            e.0.push(n)
        } else {
            e.1.push(n)
        }
    }
    median(
        by_sp
            .into_values()
            .map(|(r, u)| median(r) / median(u))
            .collect(),
    )
}

/// R² of one HC (0-based column) on superpopulation: between-group over total sum of squares.
fn superpop_r2(h: &Hcs, hc: usize) -> f64 {
    let d = d();
    let x: Vec<(f64, &str)> = d
        .samples
        .iter()
        .map(|s| (h.get(s)[hc], d.superpop[s].as_str()))
        .collect();
    let mean = x.iter().map(|v| v.0).sum::<f64>() / x.len() as f64;
    let mut groups: HashMap<&str, (f64, usize)> = HashMap::new();
    for (v, g) in &x {
        let e = groups.entry(g).or_default();
        e.0 += v;
        e.1 += 1;
    }
    let total: f64 = x.iter().map(|v| (v.0 - mean).powi(2)).sum();
    let between: f64 = groups
        .values()
        .map(|(s, n)| *n as f64 * (s / *n as f64 - mean).powi(2))
        .sum();
    between / total
}

#[test]
fn default_run_shows_structure_and_relatives_are_shrunk() {
    let (d, r) = (d(), default_run());
    let expected: Vec<String> = std::iter::once("sample".to_string())
        .chain((1..=20).map(|i| format!("HC{i}")))
        .collect();
    assert_eq!(r.hcs.header, expected);
    assert_eq!(
        r.hcs.samples, d.samples,
        "one row per sample, in sample order"
    );
    assert!(r.hcs.rows.iter().flatten().all(|v| v.is_finite()));
    let r2 = superpop_r2(&r.hcs, 1);
    assert!(r2 > 0.8, "HC2 separates superpopulations (R² {r2:.2})");
    let ratio = size_ratio(&r.hcs);
    assert!(
        ratio < 0.6,
        "relatives' HCs are shrunk without any handling (size ratio {ratio:.2})"
    );
    assert!(r.diag.iter().map(|g| &g["sample"]).eq(d.samples.iter()));
    assert!(r.diag.iter().all(|g| g["status"] == "fitted"));
    let share = |rel: bool| {
        median(
            r.diag
                .iter()
                .filter(|g| d.related.contains(&g["sample"]) == rel)
                .map(|g| g["largest_share"].parse().unwrap())
                .collect(),
        )
    };
    assert!(
        share(true) > 0.5 && share(false) < 0.1,
        "relatives' rows are dominated by their largest entry"
    );
}

#[test]
fn reweight_with_nothing_removed_changes_nothing() {
    let r = reweight_run();
    assert!(r.diag.iter().all(|g| {
        let f: f64 = g["reweight_factor"].parse().unwrap();
        (f - 1.0).abs() < 1e-4
    }));
    assert!(maxdiff(&r.hcs, &default_run().hcs) < 1e-3);
}

#[test]
fn fitting_every_sample_is_the_default() {
    let d = d();
    let mut rev = d.samples.clone();
    rev.reverse();
    for (name, list) in [("all", d.samples.clone()), ("all_reversed", rev)] {
        let f = write_list(&format!("{name}.txt"), &list);
        let r = run(&format!("fit_{name}"), &with(&["--fit_samples", &f]));
        assert_eq!(r.hcs.rows, default_run().hcs.rows, "{name}");
        assert_eq!(r.hcs.samples, default_run().hcs.samples, "{name}");
    }
}

#[test]
fn removing_relatives_pairs_and_reweighting_restores_their_hcs() {
    let d = d();
    let pairs = path("related_pairs.txt");
    let r = run("pairs", &with(&["--remove_pairs", &pairs]));
    assert!(r.log.contains("Warning: --remove_pairs without --reweight"));

    let rw = run("pairs_rw", &with(&["--remove_pairs", &pairs, "--reweight"]));
    let n = d.pairs.len();
    assert!(
        rw.log.contains(&format!("{n} pairs (0 skipped")),
        "all pairs found:\n{}",
        rw.log
    );
    assert!(rw.diag.iter().all(|g| {
        let f: f64 = g["frac_removed"].parse().unwrap();
        if d.related.contains(&g["sample"]) {
            f > 0.0
        } else {
            f == 0.0
        }
    }));
    let ratio = size_ratio(&rw.hcs);
    assert!(
        ratio > 0.75,
        "relatives' HCs are restored (size ratio {ratio:.2})"
    );

    // a kinship-style file: a header, a third column, and a pair not in the samples
    let mut k = vec!["ID1 ID2 Kinship".to_string()];
    k.extend(d.pairs.iter().map(|(a, b)| format!("{a} {b} 0.25")));
    k.push("NOTREAL1 NOTREAL2 0.25".into());
    let kin = write_list("kinship.txt", &k);
    let kr = run("kinship", &with(&["--remove_pairs", &kin, "--reweight"]));
    assert!(
        kr.log.contains(&format!("{n} pairs (2 skipped")),
        "header and unknown pair skipped:\n{}",
        kr.log
    );
    assert_eq!(
        kr.hcs.rows, rw.hcs.rows,
        "kinship-style file gives the same HCs"
    );
}

#[test]
fn fitting_a_group_gives_only_its_samples_hcs() {
    let d = d();
    let mut afr: Vec<String> = d
        .samples
        .iter()
        .filter(|s| d.superpop[*s] == "AFR")
        .cloned()
        .collect();
    let expected = afr.clone();
    afr.push("NOTREAL1".into());
    let f = write_list("afr.txt", &afr);
    let r = run("afr", &with(&["--fit_samples", &f, "--reweight"]));
    assert!(
        r.log.contains("1 IDs in") && r.log.contains("skipped"),
        "unknown ID skipped:\n{}",
        r.log
    );
    assert_eq!(
        r.hcs.samples, expected,
        "only the group's samples, in sample order"
    );
    for g in &r.diag {
        let fitted = expected.contains(&g["sample"]);
        assert_eq!(g["status"], if fitted { "fitted" } else { "not fitted" });
    }
}

#[test]
fn projecting_relatives_onto_the_unrelated() {
    let d = d();
    let u = unrelated_list();
    let unrel = run("unrel", &with(&["--fit_samples", &u, "--reweight"]));
    let p = run(
        "proj",
        &with(&["--fit_samples", &u, "--project", "--reweight"]),
    );
    assert_eq!(
        p.hcs.samples, d.samples,
        "every sample gets HCs, in sample order"
    );
    assert!(
        maxdiff(&unrel.hcs, &p.hcs) < 1e-10,
        "--project doesn't change the fitted samples' HCs"
    );
    for g in &p.diag {
        let rel = d.related.contains(&g["sample"]);
        assert_eq!(g["status"], if rel { "projected" } else { "fitted" });
    }
    let ratio = size_ratio(&p.hcs);
    assert!(
        ratio > 0.75,
        "projected relatives' HCs are not shrunk (size ratio {ratio:.2})"
    );

    // projected relatives are nearest the centroid (HC1-HC10) of their own superpopulation's unrelated samples
    let mut cent: HashMap<&str, (Vec<f64>, usize)> = HashMap::new();
    for s in d.samples.iter().filter(|s| !d.related.contains(*s)) {
        let e = cent
            .entry(d.superpop[s].as_str())
            .or_insert((vec![0.0; 10], 0));
        e.0.iter_mut()
            .zip(&p.hcs.get(s)[..10])
            .for_each(|(c, v)| *c += v);
        e.1 += 1;
    }
    let related: Vec<&String> = d
        .samples
        .iter()
        .filter(|s| d.related.contains(*s))
        .collect();
    let right = related
        .iter()
        .filter(|s| {
            let h = &p.hcs.get(s)[..10];
            let nearest = cent
                .iter()
                .map(|(sp, (sum, n))| {
                    let dist: f64 = sum
                        .iter()
                        .zip(h)
                        .map(|(c, v)| (c / *n as f64 - v).powi(2))
                        .sum();
                    (dist, *sp)
                })
                .min_by(|a, b| a.0.total_cmp(&b.0))
                .unwrap()
                .1;
            nearest == d.superpop[**s]
        })
        .count();
    assert!(
        right as f64 >= 0.95 * related.len() as f64,
        "projected relatives nearest their own superpopulation: {right}/{}",
        related.len()
    );

    let all = write_list("all_proj.txt", &d.samples);
    let pa = run("proj_all", &with(&["--fit_samples", &all, "--project"]));
    assert_eq!(
        pa.hcs.rows,
        default_run().hcs.rows,
        "--project with every sample fitted is the default"
    );
}

/// A test of the projection formula: a "clone" sample whose row is a copy of a fitted sample's row,
/// and which no fitted sample has an entry to, must be projected to that sample's fitted HCs, without
/// changing anyone else's. Fitted HCs (U·√d) and projected ones (row·V/√d) agree only as closely as
/// the randomised SVD has converged (about 1e-8 here), so the tolerance is 1e-6; a wrong formula
/// (e.g. a missing √d) gives differences of order 1.
#[test]
fn a_projected_copy_of_a_fitted_row_gets_its_hcs() {
    let d = d();
    let (orig, clone) = (1usize, d.samples.len() + 1);
    let mut m = read_lines(&data("combined.chunklengths.txt.gz"));
    let copies: Vec<String> = m
        .iter()
        .filter_map(|l| {
            let (i, rest) = l.split_once(' ').unwrap();
            (i.parse::<usize>().unwrap() == orig).then(|| format!("{clone} {rest}"))
        })
        .collect();
    m.extend(copies);
    let mut sums = read_lines(&data("combined.chunklengths.sums"));
    let s = sums
        .iter()
        .find_map(|l| {
            let (i, rest) = l.split_once(' ').unwrap();
            (i.parse::<usize>().unwrap() == orig).then(|| format!("{clone} {rest}"))
        })
        .unwrap();
    sums.push(s);
    let mut samples = d.samples.clone();
    samples.push("CLONE".into());
    let all = write_list("clone_fit.txt", &d.samples);
    let args: Vec<String> = [
        "--matrix",
        &write_list("clone.txt", &m),
        "--samples",
        &write_list("clone_samples.txt", &samples),
        "--sums",
        &write_list("clone.sums", &sums),
        "--nhcs",
        "20",
        "--fit_samples",
        &all,
        "--project",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    for rw in [false, true] {
        let mut a = args.clone();
        if rw {
            a.push("--reweight".into());
        }
        let x = run(if rw { "clone_rw" } else { "clone" }, &a);
        let (c, o) = (x.hcs.get("CLONE"), x.hcs.get(&d.samples[orig - 1]));
        let diff = c
            .iter()
            .zip(o)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            diff < 1e-6,
            "reweight {rw}: the clone's HCs differ by {diff:e}"
        );
        let reference = if rw {
            &reweight_run().hcs
        } else {
            &default_run().hcs
        };
        assert!(
            maxdiff(reference, &x.hcs) < 1e-8,
            "reweight {rw}: the projected clone changed the fitted HCs"
        );
    }
}
