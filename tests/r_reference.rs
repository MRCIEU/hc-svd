//! hcsvd against the R step it replaces (R Matrix + sparsesvd): on painted 1000 Genomes data (904
//! samples, 20 HCs), HCs must agree up to sign and diagnostics must match the reference outputs in
//! tests/data/r, which tests/make_r_reference.sh makes. The case names and options must match that script's.

mod common;
use common::*;

fn check(case: &str, extra: &[&str]) {
    let dir = out_dir(&format!("r_reference_{case}"));
    let (out, diag) = (dir.join("HCs.csv"), dir.join("diagnostics.tsv"));
    let mut args = base_args(&out, &diag);
    args.extend(extra.iter().map(|a| match data(a) {
        p if p.is_file() => p.to_str().unwrap().to_string(),
        _ => a.to_string(),
    }));
    hcsvd_ok(&args);
    assert_hcs_agree(&data(&format!("r/{case}.csv.gz")), &out);
    assert_diagnostics_agree(&data(&format!("r/{case}.tsv.gz")), &diag);
}

#[test]
fn plain() {
    check("plain", &[]);
}

#[test]
fn pairs_removed_and_reweighted() {
    check(
        "pairs_reweight",
        &["--remove_pairs", "related_pairs.txt", "--reweight"],
    );
}

#[test]
fn fit_a_subset() {
    check("fit_third", &["--fit_samples", "fit_third.txt"]);
}

#[test]
fn fit_unrelated_and_project_relatives() {
    check(
        "fit_unrelated_project",
        &[
            "--fit_samples",
            "fit_unrelated.txt",
            "--project",
            "--remove_pairs",
            "related_pairs.txt",
            "--reweight",
        ],
    );
}
