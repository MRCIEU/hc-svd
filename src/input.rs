//! Reading the inputs: sample IDs, combine's row sums, ID lists and pairs, and the combined matrix
//! (combine's gzipped text `index1 index2 value`, or its binary row stream from `--binary`).

use anyhow::{Context, Result, bail, ensure};
use flate2::read::MultiGzDecoder;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};

/// The first 8 bytes of combine's binary row stream (`combine --binary`).
pub const BINARY_MAGIC: &[u8; 8] = b"CMBROWS1";

/// Opens a file for reading lines, decompressing it if it starts with the gzip magic bytes.
fn open_text(path: &str) -> Result<Box<dyn BufRead>> {
    let mut f = File::open(path).with_context(|| format!("could not open {path}"))?;
    let mut magic = [0u8; 2];
    let n = f.read(&mut magic)?;
    let f = File::open(path)?;
    Ok(if n == 2 && magic == [0x1f, 0x8b] {
        Box::new(BufReader::new(MultiGzDecoder::new(f)))
    } else {
        Box::new(BufReader::new(f))
    })
}

/// Non-empty lines of a file, trimmed.
fn lines(path: &str) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for line in open_text(path)?.lines() {
        let line = line.with_context(|| format!("reading {path}"))?;
        let t = line.trim();
        if !t.is_empty() {
            out.push(t.to_string());
        }
    }
    Ok(out)
}

/// Sample IDs, one per line, in the order of the painted files. Fails on duplicates.
pub fn read_samples(path: &str) -> Result<Vec<String>> {
    let s = lines(path)?;
    ensure!(!s.is_empty(), "no samples in {path}");
    let mut seen = HashMap::with_capacity(s.len());
    for (i, id) in s.iter().enumerate() {
        if seen.insert(id.as_str(), i).is_some() {
            bail!("duplicate sample IDs in {path} ({id})");
        }
    }
    Ok(s)
}

/// combine's `.sums` file ("index1 sum", 1-based): the total of each row before `--restrict`.
/// Rows not in the file are None.
pub fn read_sums(path: &str, nind: usize) -> Result<Vec<Option<f64>>> {
    let mut sums = vec![None; nind];
    for line in lines(path)? {
        let mut f = line.split_whitespace();
        let (i, s) = (f.next(), f.next());
        let (Some(i), Some(s)) = (i, s) else {
            bail!("bad line in {path}: {line}")
        };
        let i: usize = i
            .parse()
            .with_context(|| format!("bad index in {path}: {line}"))?;
        ensure!(
            i >= 1 && i <= nind,
            "the sums file has rows that are not in the samples file ({i} > {nind})"
        );
        sums[i - 1] = Some(
            s.parse()
                .with_context(|| format!("bad sum in {path}: {line}"))?,
        );
    }
    Ok(sums)
}

/// Sample IDs in a file, one per line, as indices into `samples` (unknown IDs are skipped and counted).
pub fn read_id_set(path: &str, index: &HashMap<&str, usize>) -> Result<(Vec<usize>, usize)> {
    let (mut ids, mut skipped) = (Vec::new(), 0);
    for id in lines(path)? {
        match index.get(id.as_str()) {
            Some(&i) => ids.push(i),
            None => skipped += 1,
        }
    }
    ids.sort_unstable();
    ids.dedup();
    Ok((ids, skipped))
}

/// Pairs of sample IDs: the first two whitespace-separated columns of each line (other columns are
/// ignored). Pairs with an ID that isn't a sample (or a header line) are skipped and counted, as are
/// self-pairs. Returns unique pairs as (i, j) indices with i < j.
pub fn read_pairs(
    path: &str,
    index: &HashMap<&str, usize>,
) -> Result<(Vec<(usize, usize)>, usize)> {
    let (mut pairs, mut skipped) = (Vec::new(), 0);
    for line in lines(path)? {
        let mut f = line.split_whitespace();
        match (
            f.next().and_then(|a| index.get(a)),
            f.next().and_then(|b| index.get(b)),
        ) {
            (Some(&i), Some(&j)) if i != j => pairs.push((i.min(j), i.max(j))),
            (Some(&i), Some(&j)) if i == j => {}
            _ => skipped += 1,
        }
    }
    pairs.sort_unstable();
    pairs.dedup();
    Ok((pairs, skipped))
}

/// Calls `f(row, cols, vals)` for every row 0..nind in order (0-based columns, ascending, and
/// values), including empty rows, from either input format. Fails if the matrix refers to a sample
/// index beyond nind.
pub fn for_each_row(
    path: &str,
    nind: usize,
    mut f: impl FnMut(usize, &[u32], &[f32]) -> Result<()>,
) -> Result<u64> {
    let mut file = File::open(path).with_context(|| format!("could not open {path}"))?;
    let mut magic = [0u8; 8];
    let n = file.read(&mut magic)?;
    if n == 8 && &magic == BINARY_MAGIC {
        read_binary(path, nind, &mut f)
    } else {
        read_text(path, nind, &mut f)
    }
}

fn read_binary(
    path: &str,
    nind: usize,
    f: &mut impl FnMut(usize, &[u32], &[f32]) -> Result<()>,
) -> Result<u64> {
    let mut r = BufReader::with_capacity(1 << 22, File::open(path)?);
    let mut buf8 = [0u8; 8];
    r.read_exact(&mut buf8)?; // magic
    r.read_exact(&mut buf8)?;
    let n_rows = u64::from_le_bytes(buf8) as usize;
    ensure!(
        n_rows <= nind,
        "the matrix has sample index {n_rows}, but there are only {nind} samples"
    );
    let (mut cols, mut vals, mut bytes) = (Vec::new(), Vec::new(), Vec::new());
    let mut total = 0u64;
    for row in 0..nind {
        if row >= n_rows {
            f(row, &[], &[])?;
            continue;
        }
        let mut b4 = [0u8; 4];
        r.read_exact(&mut b4)
            .with_context(|| format!("{path}: truncated at row {row}"))?;
        let n = u32::from_le_bytes(b4) as usize;
        bytes.resize(8 * n, 0);
        r.read_exact(&mut bytes)
            .with_context(|| format!("{path}: truncated at row {row}"))?;
        cols.clear();
        vals.clear();
        cols.extend(
            bytes[..4 * n]
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap())),
        );
        vals.extend(
            bytes[4 * n..]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap())),
        );
        if let Some(&c) = cols.last() {
            ensure!(
                (c as usize) < nind,
                "the matrix has sample index {}, but there are only {nind} samples",
                c as usize + 1
            );
        }
        ensure!(
            cols.windows(2).all(|w| w[0] < w[1]),
            "{path}: row {row}'s columns are not ascending"
        );
        total += n as u64;
        f(row, &cols, &vals)?;
    }
    let mut rest = [0u8; 1];
    ensure!(
        r.read(&mut rest)? == 0,
        "{path}: trailing data after the last row"
    );
    Ok(total)
}

fn read_text(
    path: &str,
    nind: usize,
    f: &mut impl FnMut(usize, &[u32], &[f32]) -> Result<()>,
) -> Result<u64> {
    // the text can be in any order: collect, then sort by row and column
    let mut entries: Vec<(u32, u32, f32)> = Vec::new();
    for line in open_text(path)?.lines() {
        let line = line.with_context(|| format!("reading {path}"))?;
        let mut it = line.split_whitespace();
        let (Some(i), Some(j), Some(v)) = (it.next(), it.next(), it.next()) else {
            if line.trim().is_empty() {
                continue;
            }
            bail!("bad line in {path}: {line}");
        };
        let (i, j): (usize, usize) = (i.parse()?, j.parse()?);
        ensure!(i >= 1 && j >= 1, "bad index in {path}: {line}");
        ensure!(
            i <= nind && j <= nind,
            "the matrix has sample index {}, but there are only {nind} samples",
            i.max(j)
        );
        entries.push(((i - 1) as u32, (j - 1) as u32, v.parse::<f64>()? as f32));
    }
    entries.sort_unstable_by_key(|&(i, j, _)| (i, j));
    let total = entries.len() as u64;
    let (mut cols, mut vals) = (Vec::new(), Vec::new());
    let mut k = 0;
    for row in 0..nind {
        cols.clear();
        vals.clear();
        while k < entries.len() && entries[k].0 as usize == row {
            ensure!(
                cols.last().is_none_or(|&c| c < entries[k].1),
                "duplicate entry ({}, {}) in {path}",
                row + 1,
                entries[k].1 + 1
            );
            cols.push(entries[k].1);
            vals.push(entries[k].2);
            k += 1;
        }
        f(row, &cols, &vals)?;
    }
    Ok(total)
}
