//! Truncated SVD of a sparse matrix by the augmented implicitly restarted Lanczos bidiagonalization
//! (IRLBA; Baglama & Reichel, SIAM J. Sci. Comput. 2005), the method of R's irlba package.
//!
//! A Lanczos bidiagonalization A V = W B, A' W = V B' + f e' is built up to `work` columns; the SVD
//! of the small matrix B gives approximate singular triplets of A, whose residuals are
//! |f| |last row of B's left singular vectors|. If the top k haven't converged, the method restarts
//! from those k approximations plus the residual direction, keeping the Lanczos relation (B then has
//! an "arrow" column), and extends again to `work` columns. Both bases are fully reorthogonalised
//! (twice, classical Gram-Schmidt): that costs O(n work^2) per restart, small next to the products
//! with A, and keeps the method numerically sound. Each step needs one product with A and one with
//! A', so far fewer products than a randomised SVD with power iterations.

use faer::{Col, Mat};

use crate::rsvd::{Normal, Svd};
use crate::sparse::Csr;

/// Settings for the Lanczos SVD.
#[derive(Debug, Clone, Copy)]
pub struct Settings {
    /// Size of the Lanczos basis (columns of V and W), more than k.
    pub work: usize,
    /// Stop when every one of the top k residuals is below tol times the largest singular value.
    pub tol: f64,
    /// Maximum number of restarts.
    pub max_restarts: usize,
    pub seed: u64,
}

fn to_col(x: &[f64]) -> Col<f64> {
    Col::from_fn(x.len(), |i| x[i])
}

fn to_vec(c: &Col<f64>) -> Vec<f64> {
    (0..c.nrows()).map(|i| c[i]).collect()
}

/// Orthogonalises x against the first `j` columns of q, twice.
fn reorth(x: &mut Col<f64>, q: &Mat<f64>, j: usize) {
    if j == 0 {
        return;
    }
    let qs = q.get(.., 0..j);
    for _ in 0..2 {
        let c = qs.transpose() * &*x;
        *x -= qs * &c;
    }
}

/// A random unit vector orthogonal to the first `j` columns of q (for an invariant subspace).
fn random_orth(rng: &mut Normal, n: usize, q: &Mat<f64>, j: usize) -> Col<f64> {
    loop {
        let mut x = Col::from_fn(n, |_| rng.sample());
        reorth(&mut x, q, j);
        let nx = x.norm_l2();
        if nx > 1e-8 {
            return x / nx;
        }
    }
}

/// The top k singular triplets of a (nrows x ncols); `at` must be a's transpose. Returns the same
/// `Svd` as the randomised method (u, v row-major; zero singular values dropped; each column of u
/// with its largest absolute entry positive); `iterations` is the number of restarts.
pub fn lanczos_svd(a: &Csr, at: &Csr, k: usize, s: Settings) -> anyhow::Result<Svd> {
    let (n, m) = (a.nrows, a.ncols);
    anyhow::ensure!(
        at.nrows == m && at.ncols == n,
        "at is not the transpose of a"
    );
    let p = s.work.min(n.min(m));
    anyhow::ensure!(
        k >= 1 && k <= p,
        "k ({k}) must be between 1 and min(nrows, ncols) ({})",
        n.min(m)
    );
    let eps = f64::EPSILON;
    let mut rng = Normal::new(s.seed);
    let mut v = Mat::<f64>::zeros(m, p); // right basis
    let mut w = Mat::<f64>::zeros(n, p); // left basis
    let mut b = Mat::<f64>::zeros(p, p);

    let v0 = random_orth(&mut rng, m, &v, 0);
    v.col_mut(0).copy_from(&v0);
    let mut start = 0;
    let mut smax = 0f64;
    let mut restarts = 0;
    let mut f = Col::<f64>::zeros(m);
    let mut beta = 0f64;
    // first column of W
    {
        let mut wj = to_col(&a.mul_vec(&to_vec(&v.col(0).to_owned())));
        let alpha = wj.norm_l2();
        anyhow::ensure!(alpha > 0.0, "the matrix is zero");
        wj /= alpha;
        w.col_mut(0).copy_from(&wj);
        b[(0, 0)] = alpha;
    }
    let (ub, sv, vb) = loop {
        // extend the bidiagonalization from column `start` to p
        for j in start..p {
            let mut fj = to_col(&at.mul_vec(&to_vec(&w.col(j).to_owned())));
            fj -= b[(j, j)] * v.col(j);
            reorth(&mut fj, &v, j + 1);
            beta = fj.norm_l2();
            if j + 1 < p {
                let vn = if beta > eps * smax.max(1.0) * 10.0 {
                    &fj / beta
                } else {
                    beta = 0.0; // invariant subspace: continue with a fresh direction
                    random_orth(&mut rng, m, &v, j + 1)
                };
                v.col_mut(j + 1).copy_from(&vn);
                b[(j, j + 1)] = beta;
                let mut wn = to_col(&a.mul_vec(&to_vec(&vn)));
                wn -= beta * w.col(j);
                reorth(&mut wn, &w, j + 1);
                let mut alpha = wn.norm_l2();
                if alpha > eps * smax.max(1.0) * 10.0 {
                    wn /= alpha;
                } else {
                    alpha = 0.0;
                    wn = random_orth(&mut rng, n, &w, j + 1);
                }
                w.col_mut(j + 1).copy_from(&wn);
                b[(j + 1, j + 1)] = alpha;
            } else {
                f = fj;
            }
        }
        // SVD of B, singular values in descending order
        let svd = b
            .thin_svd()
            .map_err(|e| anyhow::anyhow!("SVD of the bidiagonal matrix failed: {e:?}"))?;
        let mut order: Vec<usize> = (0..p).collect();
        order.sort_by(|&x, &y| svd.S()[y].total_cmp(&svd.S()[x]));
        let ub = Mat::from_fn(p, p, |i, c| svd.U()[(i, order[c])]);
        let vb = Mat::from_fn(p, p, |i, c| svd.V()[(i, order[c])]);
        let sv: Vec<f64> = order.iter().map(|&c| svd.S()[c]).collect();
        smax = smax.max(sv[0]);
        let resid: Vec<f64> = (0..p).map(|c| beta * ub[(p - 1, c)]).collect();
        let converged = (0..k).all(|c| resid[c].abs() <= s.tol * smax);
        restarts += 1;
        if converged || restarts > s.max_restarts || p == n.min(m) {
            break (ub, sv, vb);
        }
        // restart from the top k approximations and the residual direction
        let vk = &v * vb.get(.., 0..k);
        let wk = &w * ub.get(.., 0..k);
        v.get_mut(.., 0..k).copy_from(&vk);
        w.get_mut(.., 0..k).copy_from(&wk);
        let vnext = if beta > eps * smax * 10.0 {
            &f / beta
        } else {
            random_orth(&mut rng, m, &v, k)
        };
        v.col_mut(k).copy_from(&vnext);
        b = Mat::zeros(p, p);
        for c in 0..k {
            b[(c, c)] = sv[c];
            b[(c, k)] = resid[c];
        }
        let mut wn = to_col(&a.mul_vec(&to_vec(&vnext)));
        reorth(&mut wn, &w, k);
        let alpha = wn.norm_l2();
        if alpha > eps * smax * 10.0 {
            wn /= alpha;
            b[(k, k)] = alpha;
        } else {
            wn = random_orth(&mut rng, n, &w, k);
        }
        w.col_mut(k).copy_from(&wn);
        start = k;
    };
    if restarts > s.max_restarts {
        eprintln!(
            "  Warning: Lanczos didn't converge in {} restarts; increase --max_restarts or --work",
            s.max_restarts
        );
    }

    // the top k triplets, dropping zero singular values
    let keep: Vec<usize> = (0..k).filter(|&c| sv[c] > sv[0] * 1e-12).collect();
    let kk = keep.len();
    let um = &w * Mat::from_fn(p, kk, |i, c| ub[(i, keep[c])]);
    let vm = &v * Mat::from_fn(p, kk, |i, c| vb[(i, keep[c])]);
    let mut u = vec![0f64; n * kk];
    let mut vv = vec![0f64; m * kk];
    let mut d = vec![0f64; kk];
    for (out, &c) in keep.iter().enumerate() {
        d[out] = sv[c];
        let imax = (0..n)
            .max_by(|&x, &y| um[(x, out)].abs().total_cmp(&um[(y, out)].abs()))
            .unwrap_or(0);
        let sign = if um[(imax, out)] < 0.0 { -1.0 } else { 1.0 };
        for i in 0..n {
            u[i * kk + out] = sign * um[(i, out)];
        }
        for j in 0..m {
            vv[j * kk + out] = sign * vm[(j, out)];
        }
    }
    Ok(Svd {
        k: kk,
        u,
        d,
        v: vv,
        iterations: restarts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dense matrix with a decaying spectrum (ncomp components with strengths 10 / (c + 1)^decay,
    /// plus noise), as a Csr.
    fn test_matrix_with(n: usize, m: usize, seed: u64, ncomp: usize, decay: f64) -> Csr {
        let mut rng = Normal::new(seed);
        let comps: Vec<(Vec<f64>, Vec<f64>, f64)> = (0..ncomp)
            .map(|c| {
                (
                    (0..n).map(|_| rng.sample()).collect(),
                    (0..m).map(|_| rng.sample()).collect(),
                    10.0 / ((c + 1) as f64).powf(decay),
                )
            })
            .collect();
        let mut a = Csr::new(m);
        for i in 0..n {
            let cols: Vec<u32> = (0..m as u32).collect();
            let vals: Vec<f32> = (0..m)
                .map(|j| {
                    (comps.iter().map(|(u, v, s)| s * u[i] * v[j]).sum::<f64>()
                        + 0.01 * rng.sample()) as f32
                })
                .collect();
            a.push_row(&cols, &vals);
        }
        a
    }

    fn test_matrix(n: usize, m: usize, seed: u64) -> Csr {
        test_matrix_with(n, m, seed, 8, 1.0)
    }

    fn exact(a: &Csr) -> Vec<f64> {
        let x = Mat::from_fn(a.nrows, a.ncols, |i, j| {
            let r = a.indptr[i] as usize..a.indptr[i + 1] as usize;
            a.indices[r.clone()]
                .iter()
                .position(|&c| c as usize == j)
                .map_or(0.0, |k| a.data[r.start + k] as f64)
        });
        let s = x.thin_svd().unwrap();
        let mut d: Vec<f64> = (0..s.S().dim()).map(|i| s.S()[i]).collect();
        d.sort_by(|x, y| y.total_cmp(x));
        d
    }

    fn check(a: &Csr, r: &Svd, k: usize) {
        let ex = exact(a);
        for c in 0..k {
            assert!(
                (r.d[c] - ex[c]).abs() < 1e-6 * ex[0],
                "singular value {c}: {} vs {}",
                r.d[c],
                ex[c]
            );
            // A v = d u
            let vcol: Vec<f64> = (0..a.ncols).map(|j| r.v[j * r.k + c]).collect();
            let av = a.mul_vec(&vcol);
            for (i, x) in av.iter().enumerate() {
                assert!((x - r.d[c] * r.u[i * r.k + c]).abs() < 1e-6 * ex[0]);
            }
        }
    }

    #[test]
    fn matches_exact_svd() {
        let a = test_matrix(120, 90, 1);
        let s = Settings {
            work: 30,
            tol: 1e-10,
            max_restarts: 200,
            seed: 3,
        };
        let r = lanczos_svd(&a, &a.transpose(), 6, s).unwrap();
        check(&a, &r, 6);
    }

    #[test]
    fn converges_with_many_restarts() {
        // a slowly decaying spectrum and a small basis force restarts
        let a = test_matrix_with(150, 140, 2, 40, 0.3);
        let s = Settings {
            work: 12,
            tol: 1e-10,
            max_restarts: 1000,
            seed: 5,
        };
        let r = lanczos_svd(&a, &a.transpose(), 8, s).unwrap();
        assert!(r.iterations > 1, "expected restarts, got {}", r.iterations);
        check(&a, &r, 8);
    }

    #[test]
    fn rank_deficient_matrix() {
        // rank 2: the basis runs out of directions (invariant subspace); zero singular values are dropped
        let mut a = Csr::new(5);
        for i in 0..6 {
            let x = (i + 1) as f32;
            a.push_row(&[0, 1, 2, 3, 4], &[x, 2.0 * x, 0.5, 1.0, x - 3.0]);
        }
        let s = Settings {
            work: 5,
            tol: 1e-10,
            max_restarts: 50,
            seed: 1,
        };
        let r = lanczos_svd(&a, &a.transpose(), 4, s).unwrap();
        assert_eq!(r.k, 2);
        check(&a, &r, 2);
    }

    #[test]
    fn reproducible_with_a_seed() {
        let a = test_matrix(60, 60, 4);
        let s = Settings {
            work: 20,
            tol: 1e-10,
            max_restarts: 100,
            seed: 9,
        };
        let (r1, r2) = (
            lanczos_svd(&a, &a.transpose(), 5, s).unwrap(),
            lanczos_svd(&a, &a.transpose(), 5, s).unwrap(),
        );
        assert_eq!(r1.u, r2.u);
        assert_eq!(r1.d, r2.d);
    }
}
