//! Randomised truncated SVD of a sparse matrix (Halko, Martinsson & Tropp 2011, algorithms 4.4 and
//! 5.1): a random range finder with power iterations, then the SVD of the small projected matrix.

use faer::Mat;

use crate::sparse::Csr;

/// The top k singular triplets: u (nrows x k) and v (ncols x k), both row-major, and d (descending);
/// and the number of power iterations run.
#[derive(Debug)]
pub struct Svd {
    pub k: usize,
    pub u: Vec<f64>,
    pub d: Vec<f64>,
    pub v: Vec<f64>,
    pub iterations: usize,
}

/// Settings for the randomised SVD.
#[derive(Debug, Clone, Copy)]
pub struct Settings {
    /// Extra random directions beyond k (oversampling).
    pub oversample: usize,
    /// Maximum number of power iterations (each is one product with A' and one with A).
    pub power_iterations: usize,
    /// Stop early once the estimates of the top k singular values change by less than this
    /// (relative) between iterations. 0 = always run power_iterations.
    pub tol: f64,
    pub seed: u64,
}

/// A small, fast, reproducible generator of standard normal numbers (SplitMix64 + Box-Muller).
struct Normal {
    state: u64,
    spare: Option<f64>,
}

impl Normal {
    fn new(seed: u64) -> Self {
        Normal {
            state: seed,
            spare: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform on (0, 1).
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn sample(&mut self) -> f64 {
        if let Some(s) = self.spare.take() {
            return s;
        }
        let (u1, u2) = (self.uniform(), self.uniform());
        let r = (-2.0 * u1.ln()).sqrt();
        let t = 2.0 * std::f64::consts::PI * u2;
        self.spare = Some(r * t.sin());
        r * t.cos()
    }
}

fn to_mat(x: &[f64], n: usize, l: usize) -> Mat<f64> {
    Mat::from_fn(n, l, |i, j| x[i * l + j])
}

fn to_rows(m: &Mat<f64>) -> Vec<f64> {
    let (n, l) = (m.nrows(), m.ncols());
    let mut x = vec![0f64; n * l];
    for j in 0..l {
        for i in 0..n {
            x[i * l + j] = m[(i, j)];
        }
    }
    x
}

/// An orthonormal basis for the columns of x (n x l, row-major), row-major.
fn orthonormalise(x: &[f64], n: usize, l: usize) -> Vec<f64> {
    let q = to_mat(x, n, l).qr().compute_thin_Q();
    to_rows(&q)
}

/// The top k singular triplets of a (nrows x ncols). `at` must be a's transpose. Components
/// whose singular value is zero (relative to the largest) are dropped, so fewer than k may be
/// returned. Signs are fixed so that each column of u has its largest absolute entry positive.
pub fn randomised_svd(a: &Csr, at: &Csr, k: usize, s: Settings) -> anyhow::Result<Svd> {
    let (n, m) = (a.nrows, a.ncols);
    anyhow::ensure!(
        at.nrows == m && at.ncols == n,
        "at is not the transpose of a"
    );
    let l = (k + s.oversample).min(n).min(m);
    anyhow::ensure!(
        k <= l,
        "k ({k}) must be at most min(nrows, ncols) ({})",
        n.min(m)
    );

    // range finder: Q ~ an orthonormal basis for the range of A
    let mut rng = Normal::new(s.seed);
    let omega: Vec<f64> = (0..m * l).map(|_| rng.sample()).collect();
    let mut q = orthonormalise(&a.mul_dense(&omega, l), n, l);
    let mut prev: Option<Vec<f64>> = None;
    let mut iterations = 0;
    for _ in 0..s.power_iterations {
        // A'Q = Z R: R's singular values are the current estimates of A's top singular values
        let zqr = to_mat(&at.mul_dense(&q, l), m, l).qr();
        let z = to_rows(&zqr.compute_thin_Q());
        let mut est = zqr
            .thin_R()
            .singular_values()
            .map_err(|e| anyhow::anyhow!("SVD of R failed: {e:?}"))?;
        est.sort_by(|x, y| y.total_cmp(x));
        est.truncate(k);
        q = orthonormalise(&a.mul_dense(&z, l), n, l);
        iterations += 1;
        if let Some(p) = &prev {
            let change = est
                .iter()
                .zip(p)
                .map(|(x, y)| (x - y).abs() / x.abs().max(f64::MIN_POSITIVE))
                .fold(0.0, f64::max);
            if s.tol > 0.0 && change < s.tol {
                break;
            }
        }
        prev = Some(est);
    }

    // B = Q' A (l x m); work with B' = A' Q (m x l), whose thin SVD is B' = Ub S Vb', so that
    // A ~ Q B = (Q Vb) S Ub'
    let bt = to_mat(&at.mul_dense(&q, l), m, l);
    let svd = bt
        .thin_svd()
        .map_err(|e| anyhow::anyhow!("SVD of the projected matrix failed: {e:?}"))?;
    let (ub, vb) = (svd.U(), svd.V());
    let sv: Vec<f64> = (0..l).map(|i| svd.S()[i]).collect();

    // order by singular value (descending), drop zero ones, keep k
    let mut order: Vec<usize> = (0..l).collect();
    order.sort_by(|&x, &y| sv[y].total_cmp(&sv[x]));
    let smax = sv[order[0]].max(0.0);
    let keep: Vec<usize> = order
        .into_iter()
        .filter(|&c| sv[c] > smax * 1e-12)
        .take(k)
        .collect();
    let kk = keep.len();

    // U = Q * (selected columns of Vb), as one (parallel) matrix product
    let vb_sel = Mat::from_fn(l, kk, |t, c| vb[(t, keep[c])]);
    let um = &to_mat(&q, n, l) * &vb_sel;
    let mut u = vec![0f64; n * kk];
    let mut v = vec![0f64; m * kk];
    let mut d = vec![0f64; kk];
    for (out, &c) in keep.iter().enumerate() {
        d[out] = sv[c];
        // sign: largest |u| entry positive
        let imax = (0..n)
            .max_by(|&x, &y| um[(x, out)].abs().total_cmp(&um[(y, out)].abs()))
            .unwrap_or(0);
        let sign = if n > 0 && um[(imax, out)] < 0.0 {
            -1.0
        } else {
            1.0
        };
        for i in 0..n {
            u[i * kk + out] = sign * um[(i, out)];
        }
        for j in 0..m {
            v[j * kk + out] = sign * ub[(j, c)];
        }
    }
    Ok(Svd {
        k: kk,
        u,
        d,
        v,
        iterations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A random sparse matrix with a decaying spectrum: a few strong rank-one components plus noise.
    fn test_matrix(n: usize, m: usize, seed: u64, density: f64) -> Csr {
        let mut rng = Normal::new(seed);
        let comps: Vec<(Vec<f64>, Vec<f64>, f64)> = (0..6)
            .map(|c| {
                (
                    (0..n).map(|_| rng.sample()).collect(),
                    (0..m).map(|_| rng.sample()).collect(),
                    10.0 / (c + 1) as f64,
                )
            })
            .collect();
        let mut a = Csr::new(m);
        for i in 0..n {
            let (mut cols, mut vals) = (Vec::new(), Vec::new());
            for j in 0..m {
                if rng.uniform() < density {
                    // strong low-rank signal plus small noise
                    let x: f64 = comps.iter().map(|(u, v, s)| s * u[i] * v[j]).sum::<f64>()
                        + 0.001 * rng.sample();
                    cols.push(j as u32);
                    vals.push(x as f32);
                }
            }
            a.push_row(&cols, &vals);
        }
        a
    }

    fn dense(a: &Csr) -> Mat<f64> {
        let mut x = Mat::<f64>::zeros(a.nrows, a.ncols);
        for r in 0..a.nrows {
            for k in a.indptr[r] as usize..a.indptr[r + 1] as usize {
                x[(r, a.indices[k] as usize)] = a.data[k] as f64;
            }
        }
        x
    }

    #[test]
    fn matches_exact_svd() {
        let a = test_matrix(120, 90, 1, 1.0); // dense: low rank + small noise
        let at = a.transpose();
        let k = 5;
        let r = randomised_svd(
            &a,
            &at,
            k,
            Settings {
                oversample: 20,
                power_iterations: 4,
                tol: 0.0,
                seed: 7,
            },
        )
        .unwrap();
        let exact = dense(&a).thin_svd().unwrap();
        let mut ex: Vec<f64> = (0..exact.S().dim()).map(|i| exact.S()[i]).collect();
        ex.sort_by(|x, y| y.total_cmp(x));
        for c in 0..k {
            assert!(
                (r.d[c] - ex[c]).abs() < 1e-6 * ex[0],
                "singular value {c}: {} vs {}",
                r.d[c],
                ex[c]
            );
        }
        // A v = d u for each component
        for c in 0..k {
            let vcol: Vec<f64> = (0..a.ncols).map(|j| r.v[j * k + c]).collect();
            let av = a.mul_dense(&vcol, 1);
            for (i, avi) in av.iter().enumerate() {
                assert!((avi - r.d[c] * r.u[i * k + c]).abs() < 1e-6 * ex[0]);
            }
        }
    }

    #[test]
    fn reproducible_with_a_seed() {
        let a = test_matrix(60, 60, 3, 0.3);
        let at = a.transpose();
        let s = Settings {
            oversample: 10,
            power_iterations: 2,
            tol: 0.0,
            seed: 11,
        };
        let (r1, r2) = (
            randomised_svd(&a, &at, 4, s).unwrap(),
            randomised_svd(&a, &at, 4, s).unwrap(),
        );
        assert_eq!(r1.u, r2.u);
        assert_eq!(r1.d, r2.d);
    }

    #[test]
    fn drops_zero_singular_values() {
        // rank 2 matrix, ask for 4
        let mut a = Csr::new(4);
        a.push_row(&[0, 1], &[1.0, 1.0]);
        a.push_row(&[2], &[2.0]);
        a.push_row(&[0, 1], &[1.0, 1.0]);
        a.push_row(&[], &[]);
        let r = randomised_svd(
            &a,
            &a.transpose(),
            3,
            Settings {
                oversample: 1,
                power_iterations: 2,
                tol: 0.0,
                seed: 1,
            },
        )
        .unwrap();
        assert_eq!(r.k, 2);
    }
}
