//! Compressed sparse row (CSR) matrices with 32-bit column indices and values, 64-bit row offsets
//! (so the number of entries isn't limited to 2^31), and parallel products with dense blocks.

use rayon::prelude::*;

/// A sparse matrix in compressed sparse row form.
#[derive(Debug, Clone, PartialEq)]
pub struct Csr {
    pub nrows: usize,
    pub ncols: usize,
    /// Row r's entries are indptr[r]..indptr[r + 1].
    pub indptr: Vec<u64>,
    pub indices: Vec<u32>,
    pub data: Vec<f32>,
}

impl Csr {
    /// An empty matrix with ncols columns, to which rows are appended with `push_row`.
    pub fn new(ncols: usize) -> Self {
        Csr {
            nrows: 0,
            ncols,
            indptr: vec![0],
            indices: Vec::new(),
            data: Vec::new(),
        }
    }

    /// Appends a row; columns must be < ncols.
    pub fn push_row(&mut self, cols: &[u32], vals: &[f32]) {
        debug_assert_eq!(cols.len(), vals.len());
        debug_assert!(cols.iter().all(|&c| (c as usize) < self.ncols));
        self.indices.extend_from_slice(cols);
        self.data.extend_from_slice(vals);
        self.indptr.push(self.indices.len() as u64);
        self.nrows += 1;
    }

    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// The transpose, also in CSR form (i.e. this matrix in compressed sparse column form).
    pub fn transpose(&self) -> Csr {
        let mut counts = vec![0u64; self.ncols + 1];
        for &c in &self.indices {
            counts[c as usize + 1] += 1;
        }
        for c in 0..self.ncols {
            counts[c + 1] += counts[c];
        }
        let indptr = counts.clone();
        let mut next = counts;
        let mut indices = vec![0u32; self.nnz()];
        let mut data = vec![0f32; self.nnz()];
        for r in 0..self.nrows {
            for k in self.indptr[r] as usize..self.indptr[r + 1] as usize {
                let c = self.indices[k] as usize;
                let dst = next[c] as usize;
                indices[dst] = r as u32;
                data[dst] = self.data[k];
                next[c] += 1;
            }
        }
        Csr {
            nrows: self.ncols,
            ncols: self.nrows,
            indptr,
            indices,
            data,
        }
    }

    /// self (nrows x ncols) times x (ncols x l, row-major) -> nrows x l, row-major. Rows are
    /// computed in parallel.
    pub fn mul_dense(&self, x: &[f64], l: usize) -> Vec<f64> {
        assert_eq!(x.len(), self.ncols * l);
        let mut y = vec![0f64; self.nrows * l];
        y.par_chunks_mut(l).enumerate().for_each(|(r, yr)| {
            for k in self.indptr[r] as usize..self.indptr[r + 1] as usize {
                let v = self.data[k] as f64;
                let xc = &x[self.indices[k] as usize * l..(self.indices[k] as usize + 1) * l];
                for (a, b) in yr.iter_mut().zip(xc) {
                    *a += v * b;
                }
            }
        });
        y
    }

    /// y = self * x for a single vector x (length ncols), in parallel over chunks of rows.
    pub fn mul_vec(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.ncols);
        let mut y = vec![0f64; self.nrows];
        y.par_iter_mut()
            .with_min_len(256)
            .enumerate()
            .for_each(|(r, yr)| {
                let mut acc = 0f64;
                for k in self.indptr[r] as usize..self.indptr[r + 1] as usize {
                    acc += self.data[k] as f64 * x[self.indices[k] as usize];
                }
                *yr = acc;
            });
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example() -> Csr {
        // [[1, 0, 2],
        //  [0, 0, 0],
        //  [3, 4, 0]]
        let mut a = Csr::new(3);
        a.push_row(&[0, 2], &[1.0, 2.0]);
        a.push_row(&[], &[]);
        a.push_row(&[0, 1], &[3.0, 4.0]);
        a
    }

    #[test]
    fn mul_vec_matches_mul_dense() {
        let a = example();
        let x = [0.5, -1.0, 2.0];
        assert_eq!(a.mul_vec(&x), a.mul_dense(&x, 1));
    }

    #[test]
    fn transpose_twice_is_identity() {
        let a = example();
        assert_eq!(a.transpose().transpose(), a);
        let t = a.transpose();
        assert_eq!(t.indptr, vec![0, 2, 3, 4]);
        assert_eq!(t.indices, vec![0, 2, 2, 0]);
        assert_eq!(t.data, vec![1.0, 3.0, 4.0, 2.0]);
    }

    #[test]
    fn mul_dense_matches_dense_product() {
        let a = example();
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 x 2, row-major
        let y = a.mul_dense(&x, 2);
        assert_eq!(
            y,
            vec![1.0 + 10.0, 2.0 + 12.0, 0.0, 0.0, 3.0 + 12.0, 6.0 + 16.0]
        );
    }
}
