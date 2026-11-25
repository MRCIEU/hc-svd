//! hcsvd's command line: matrix formats, edge inputs and errors.

mod common;
use common::*;

use std::fs;
use std::io::Write;
use std::path::Path;

fn s(p: &Path) -> String {
    p.to_str().unwrap().to_string()
}

/// Writes the text matrix as combine's binary row stream (`combine --binary`): magic, u64 number of
/// rows, then per row u32 n, n u32 0-based columns, n f32 values, little-endian.
fn write_binary(text: &Path, binary: &Path) {
    let mut entries: Vec<(u32, u32, f32)> = read_lines(text)
        .iter()
        .map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            (
                f[0].parse::<u32>().unwrap() - 1,
                f[1].parse::<u32>().unwrap() - 1,
                f[2].parse::<f64>().unwrap() as f32,
            )
        })
        .collect();
    entries.sort_unstable_by_key(|&(i, j, _)| (i, j));
    let n_rows = entries.iter().map(|&(i, j, _)| i.max(j) + 1).max().unwrap();
    let mut out = Vec::new();
    out.extend_from_slice(b"CMBROWS1");
    out.extend_from_slice(&(n_rows as u64).to_le_bytes());
    let mut k = 0;
    for row in 0..n_rows {
        let start = k;
        while k < entries.len() && entries[k].0 == row {
            k += 1;
        }
        let e = &entries[start..k];
        out.extend_from_slice(&(e.len() as u32).to_le_bytes());
        e.iter()
            .for_each(|x| out.extend_from_slice(&x.1.to_le_bytes()));
        e.iter()
            .for_each(|x| out.extend_from_slice(&x.2.to_le_bytes()));
    }
    fs::File::create(binary).unwrap().write_all(&out).unwrap();
}

#[test]
fn binary_and_text_matrices_give_the_same_output() {
    let dir = out_dir("binary_vs_text");
    let bin = dir.join("matrix.rows.bin");
    write_binary(&data("combined.chunklengths.txt.gz"), &bin);
    let (t_out, t_diag) = (dir.join("text.csv"), dir.join("text.tsv"));
    let (b_out, b_diag) = (dir.join("bin.csv"), dir.join("bin.tsv"));
    hcsvd_ok(base_args(&t_out, &t_diag));
    let mut args = base_args(&b_out, &b_diag);
    args[1] = s(&bin);
    hcsvd_ok(&args);
    assert_eq!(fs::read(&t_out).unwrap(), fs::read(&b_out).unwrap());
    assert_eq!(fs::read(&t_diag).unwrap(), fs::read(&b_diag).unwrap());
}

#[test]
fn a_sample_with_no_entries_gets_zero_hcs_and_leaves_the_others_unchanged() {
    let dir = out_dir("empty_row");
    let samples = dir.join("samples.txt");
    let mut ids = fs::read_to_string(data("samples.txt")).unwrap();
    ids.push_str("EMPTY\n");
    fs::write(&samples, ids).unwrap();
    let (out, diag) = (dir.join("HCs.csv"), dir.join("diagnostics.tsv"));
    let o = s(&out);
    hcsvd_ok([
        "--matrix",
        &s(&data("combined.chunklengths.txt.gz")),
        "--samples",
        &s(&samples),
        "--nhcs",
        "20",
        "--out",
        &o,
        "--diagnostics",
        &s(&diag),
    ]);
    let (ids, cols) = read_hcs(&out);
    assert_eq!(ids.last().unwrap(), "EMPTY");
    assert!(cols.iter().all(|c| *c.last().unwrap() == 0.0));
    // without the empty sample, the HCs are R's
    let lines = read_lines(&out);
    let trimmed = dir.join("trimmed.csv");
    fs::write(&trimmed, lines[..lines.len() - 1].join("\n") + "\n").unwrap();
    assert_hcs_agree(&data("r/plain.csv.gz"), &trimmed);
}

#[test]
fn project_without_fit_samples_fails() {
    let dir = out_dir("err_project");
    let mut args = base_args(&dir.join("HCs.csv"), &dir.join("d.tsv"));
    args.push("--project".into());
    hcsvd_fails_with("--project needs --fit_samples", &args);
}

#[test]
fn reweight_without_sums_fails() {
    let dir = out_dir("err_reweight");
    hcsvd_fails_with(
        "--reweight needs --sums",
        [
            "--matrix",
            &s(&data("combined.chunklengths.txt.gz")),
            "--samples",
            &s(&data("samples.txt")),
            "--nhcs",
            "5",
            "--out",
            &s(&dir.join("HCs.csv")),
            "--reweight",
        ],
    );
}

#[test]
fn too_many_hcs_fails() {
    let dir = out_dir("err_nhcs");
    let mut args = base_args(&dir.join("HCs.csv"), &dir.join("d.tsv"));
    args[7] = "904".into();
    hcsvd_fails_with("must be less than the number of samples fitted", &args);
    args[7] = "301".into();
    args.extend(["--fit_samples".into(), s(&data("fit_third.txt"))]);
    hcsvd_fails_with(
        "must be less than the number of samples fitted (301)",
        &args,
    );
}

#[test]
fn a_matrix_index_beyond_the_samples_fails() {
    let dir = out_dir("err_index");
    let samples = dir.join("samples.txt");
    let ids = fs::read_to_string(data("samples.txt")).unwrap();
    fs::write(
        &samples,
        ids.lines().take(100).collect::<Vec<_>>().join("\n") + "\n",
    )
    .unwrap();
    hcsvd_fails_with(
        "there are only 100 samples",
        [
            "--matrix",
            &s(&data("combined.chunklengths.txt.gz")),
            "--samples",
            &s(&samples),
            "--nhcs",
            "5",
            "--out",
            &s(&dir.join("HCs.csv")),
        ],
    );
}

#[test]
fn duplicate_sample_ids_fail() {
    let dir = out_dir("err_duplicates");
    let samples = dir.join("samples.txt");
    let ids = fs::read_to_string(data("samples.txt")).unwrap();
    let first = ids.lines().next().unwrap().to_string();
    fs::write(&samples, ids + &first + "\n").unwrap();
    hcsvd_fails_with(
        "duplicate sample IDs",
        [
            "--matrix",
            &s(&data("combined.chunklengths.txt.gz")),
            "--samples",
            &s(&samples),
            "--nhcs",
            "5",
            "--out",
            &s(&dir.join("HCs.csv")),
        ],
    );
}
