//! Helpers shared by the integration tests: paths, running hcsvd, and comparing outputs.
#![allow(dead_code)]

use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// A file in tests/data.
pub fn data(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

/// A path in a fresh directory for one test's outputs.
pub fn out_dir(test: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join(test);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Runs hcsvd with these arguments.
pub fn hcsvd<I, S>(args: I) -> Output
where
    I: IntoIterator<Item = S>,
    S: AsRef<std::ffi::OsStr>,
{
    Command::new(env!("CARGO_BIN_EXE_hcsvd"))
        .args(args)
        .output()
        .expect("could not run hcsvd")
}

/// Runs hcsvd and panics, showing its messages, if it fails.
pub fn hcsvd_ok<I, S>(args: I)
where
    I: IntoIterator<Item = S>,
    S: AsRef<std::ffi::OsStr>,
{
    let o = hcsvd(args);
    assert!(
        o.status.success(),
        "hcsvd failed:\n{}",
        String::from_utf8_lossy(&o.stderr)
    );
}

/// Runs hcsvd, which must fail with a message containing `msg`.
pub fn hcsvd_fails_with<I, S>(msg: &str, args: I)
where
    I: IntoIterator<Item = S>,
    S: AsRef<std::ffi::OsStr>,
{
    let o = hcsvd(args);
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(!o.status.success(), "hcsvd succeeded, expected: {msg}");
    assert!(err.contains(msg), "expected '{msg}' in:\n{err}");
}

/// The test matrix, its sums and samples, with these HC and diagnostics outputs.
pub fn base_args(out: &Path, diagnostics: &Path) -> Vec<String> {
    let p = |x: &Path| x.to_str().unwrap().to_string();
    vec![
        "--matrix".into(),
        p(&data("combined.chunklengths.txt.gz")),
        "--samples".into(),
        p(&data("samples.txt")),
        "--sums".into(),
        p(&data("combined.chunklengths.sums")),
        "--nhcs".into(),
        "20".into(),
        "--out".into(),
        p(out),
        "--diagnostics".into(),
        p(diagnostics),
    ]
}

/// A text file (gzipped if its name ends .gz) as lines.
pub fn read_lines(path: &Path) -> Vec<String> {
    let f = File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let r: Box<dyn Read> = if path.extension().is_some_and(|e| e == "gz") {
        Box::new(flate2::read::MultiGzDecoder::new(f))
    } else {
        Box::new(f)
    };
    BufReader::new(r).lines().map(|l| l.unwrap()).collect()
}

/// A delimited table: header, then rows.
pub fn read_table(path: &Path, sep: char) -> (Vec<String>, Vec<Vec<String>>) {
    let lines = read_lines(path);
    let split = |l: &String| l.split(sep).map(str::to_string).collect::<Vec<_>>();
    (split(&lines[0]), lines[1..].iter().map(split).collect())
}

/// HCs from an --out file: sample IDs and columns.
pub fn read_hcs(path: &Path) -> (Vec<String>, Vec<Vec<f64>>) {
    let (header, rows) = read_table(path, ',');
    assert_eq!(header[0], "sample");
    let samples = rows.iter().map(|r| r[0].clone()).collect();
    let cols = (1..header.len())
        .map(|c| rows.iter().map(|r| r[c].parse().unwrap()).collect())
        .collect();
    (samples, cols)
}

fn sd(x: &[f64]) -> f64 {
    let m = x.iter().sum::<f64>() / x.len() as f64;
    (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (x.len() - 1) as f64).sqrt()
}

fn cor(x: &[f64], y: &[f64]) -> f64 {
    let (mx, my) = (
        x.iter().sum::<f64>() / x.len() as f64,
        y.iter().sum::<f64>() / y.len() as f64,
    );
    let sxy: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
    let sxx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
    let syy: f64 = y.iter().map(|b| (b - my).powi(2)).sum();
    sxy / (sxx * syy).sqrt()
}

/// HCs in `test` agree with `reference` up to sign: same samples and number of HCs, and for each HC
/// |correlation| >= 0.999999 and the largest difference (signs aligned) at most 1e-4 of its SD.
pub fn assert_hcs_agree(reference: &Path, test: &Path) {
    let (rs, rc) = read_hcs(reference);
    let (ts, tc) = read_hcs(test);
    assert_eq!(rs, ts, "the samples differ");
    assert_eq!(rc.len(), tc.len(), "the number of HCs differs");
    for (c, (r, t)) in rc.iter().zip(&tc).enumerate() {
        let k = cor(r, t);
        let diff = r
            .iter()
            .zip(t)
            .map(|(a, b)| (a - k.signum() * b).abs())
            .fold(0.0, f64::max)
            / sd(r);
        assert!(
            k.abs() >= 0.999999 && diff <= 1e-4,
            "HC{}: |cor| {:.8}, max |diff| / SD {diff:.1e}",
            c + 1,
            k.abs()
        );
    }
}

/// Diagnostics in `test` agree with `reference`: same columns, samples and status; missing and
/// non-finite values (NA, NaN, Inf) in the same places; numbers within 1e-6 (relative).
pub fn assert_diagnostics_agree(reference: &Path, test: &Path) {
    let (rh, rr) = read_table(reference, '\t');
    let (th, tr) = read_table(test, '\t');
    assert_eq!(rh, th, "the columns differ");
    assert_eq!(rr.len(), tr.len(), "the number of rows differs");
    for (r, t) in rr.iter().zip(&tr) {
        for (c, name) in rh.iter().enumerate() {
            let (x, y) = (&r[c], &t[c]);
            if x == y {
                continue;
            }
            let close = match (x.parse::<f64>(), y.parse::<f64>()) {
                (Ok(a), Ok(b)) if a.is_finite() && b.is_finite() => {
                    (a - b).abs() <= 1e-6 * a.abs().max(1e-12)
                }
                _ => false,
            };
            assert!(close, "{}, {name}: {x} (reference) vs {y}", r[0]);
        }
    }
}
