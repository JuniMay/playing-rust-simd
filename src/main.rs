#![feature(portable_simd)]

use std::simd::{f32x64, Mask};

use rand::Rng;
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

fn matmul_naive(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            for l in 0..k {
                c[i * n + j] += a[i * k + l] * b[l * n + j];
            }
        }
    }
}

fn matmul_loop_interchange(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for l in 0..k {
            for j in 0..n {
                c[i * n + j] += a[i * k + l] * b[l * n + j];
            }
        }
    }
}

fn matmul_iterator(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        // c[i, j] += a[i, l] * b[l, j] for all l
        // --> c[i, 0..n] += a[i, l] * b[l, 0..n] for all l
        // --> auto-vectorize
        for l in 0..k {
            let a_il = a[i * k + l];
            let b_ln = &b[l * n..(l + 1) * n];
            let c_in = &mut c[i * n..(i + 1) * n];
            for (c_ij, b_lj) in c_in.iter_mut().zip(b_ln.iter()) {
                *c_ij += a_il * b_lj;
            }
        }
    }
}

fn matmul_simd(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    const SIMD_WIDTH: usize = 64;

    for i in 0..m {
        for l in 0..k {
            let a_il = a[i * k + l];
            let b_ln = &b[l * n..(l + 1) * n];
            let c_row = &mut c[i * n..(i + 1) * n];

            let mut j = 0;
            while j < n {
                let c_ij = f32x64::load_or_default(&c_row[j..]);
                let b_lj = f32x64::load_or_default(&b_ln[j..]);
                let result = c_ij + b_lj * f32x64::splat(a_il);
                // if the element is out of bounds, store_select will ignore it
                result.store_select(&mut c_row[j..], Mask::splat(true));
                j += SIMD_WIDTH;
            }
        }
    }
}

fn matmul_parallelized_autovectorize(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    c.par_chunks_mut(n)
        .take(m)
        .enumerate()
        .for_each(|(i, row)| {
            for l in 0..k {
                let a_il = a[i * k + l];
                let b_ln = &b[l * n..(l + 1) * n];

                for (c_ij, b_lj) in row.iter_mut().zip(b_ln.iter()) {
                    *c_ij += a_il * b_lj;
                }
            }
        })
}

fn matmul_parallelized(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    c.par_chunks_mut(n)
        .take(m)
        .enumerate()
        .for_each(|(i, row)| {
            for l in 0..k {
                let a_il = a[i * k + l];
                let b_ln = &b[l * n..(l + 1) * n];

                let simd_width = 64;
                let mut j = 0;

                let num_simd = n / simd_width;
                let num_scalar = n % simd_width;

                for _ in 0..num_simd {
                    // if using `load_or_default` and remove the upper boundary,
                    // rayon will not parallelize the loop, not sure why
                    let c_ij = f32x64::from_slice(&row[j..j + simd_width]);
                    let b_lj = f32x64::from_slice(&b_ln[j..j + simd_width]);
                    let result = c_ij + b_lj * f32x64::splat(a_il);
                    result.store_select(&mut row[j..j + simd_width], Mask::splat(true));
                    j += simd_width;
                }

                for _ in 0..num_scalar {
                    row[j] += a_il * b_ln[j];
                    j += 1;
                }
            }
        })
}

fn benchmark<F>(
    f: F,
    num_iterations: usize,
    warmup_iterations: usize,
    m: usize,
    n: usize,
    k: usize,
) -> (f64, f64)
where
    F: Fn(&[f32], &[f32], &mut [f32], usize, usize, usize),
{
    let mut rng = rand::thread_rng();
    let a: Vec<f32> = (0..m * k).map(|_| rng.gen()).collect();
    let b: Vec<f32> = (0..k * n).map(|_| rng.gen()).collect();
    let mut c: Vec<f32> = vec![0.0; m * n];
    for _ in 0..warmup_iterations {
        f(&a, &b, &mut c, m, n, k);
    }
    let start = std::time::Instant::now();
    for _ in 0..num_iterations {
        f(&a, &b, &mut c, m, n, k);
    }
    let elapsed = start.elapsed();
    let time = elapsed.as_secs_f64() / num_iterations as f64;
    let gflops = 2.0 * m as f64 * n as f64 * k as f64 / time / 1e9;

    (time, gflops)
}

fn main() {
    let num_iterations = 50;
    let warmup_iterations = 2;
    let m = 512;
    let n = 4096;
    let k = 512;
    let perf_naive = benchmark(matmul_naive, num_iterations, warmup_iterations, m, n, k);
    let perf_loop_interchange = benchmark(
        matmul_loop_interchange,
        num_iterations,
        warmup_iterations,
        m,
        n,
        k,
    );
    let perf_iterator = benchmark(matmul_iterator, num_iterations, warmup_iterations, m, n, k);
    let perf_simd = benchmark(matmul_simd, num_iterations, warmup_iterations, m, n, k);
    let perf_parallelized_autovectorize = benchmark(
        matmul_parallelized_autovectorize,
        num_iterations,
        warmup_iterations,
        m,
        n,
        k,
    );
    let perf_parallelized = benchmark(
        matmul_parallelized,
        num_iterations,
        warmup_iterations,
        m,
        n,
        k,
    );

    println!("Naive: {:.6} s, {:.6} GFLOPS", perf_naive.0, perf_naive.1);
    println!(
        "Loop interchange: {:.6} s, {:.6} GFLOPS",
        perf_loop_interchange.0, perf_loop_interchange.1
    );
    println!(
        "Iterator: {:.6} s, {:.6} GFLOPS",
        perf_iterator.0, perf_iterator.1
    );
    println!("SIMD: {:.6} s, {:.6} GFLOPS", perf_simd.0, perf_simd.1);
    println!("Rayon threads: {}", rayon::current_num_threads());
    println!(
        "Parallelized autovectorize: {:.6} s, {:.6} GFLOPS",
        perf_parallelized_autovectorize.0, perf_parallelized_autovectorize.1
    );
    println!(
        "Parallelized: {:.6} s, {:.6} GFLOPS",
        perf_parallelized.0, perf_parallelized.1
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul() {
        // random matrices, regard naive as reference
        let m = 123;
        let n = 247;
        let k = 119;
        let mut rng = rand::thread_rng();
        let a: Vec<f32> = (0..m * k).map(|_| rng.gen()).collect();
        let b: Vec<f32> = (0..k * n).map(|_| rng.gen()).collect();
        let mut c_naive = vec![0.0; m * n];
        let mut c_loop_interchange = vec![0.0; m * n];
        let mut c_iterator = vec![0.0; m * n];
        let mut c_simd = vec![0.0; m * n];
        let mut c_parallelized_autovectorize = vec![0.0; m * n];
        let mut c_parallelized = vec![0.0; m * n];

        matmul_naive(&a, &b, &mut c_naive, m, n, k);
        matmul_loop_interchange(&a, &b, &mut c_loop_interchange, m, n, k);
        matmul_iterator(&a, &b, &mut c_iterator, m, n, k);
        matmul_simd(&a, &b, &mut c_simd, m, n, k);
        matmul_parallelized_autovectorize(&a, &b, &mut c_parallelized_autovectorize, m, n, k);
        matmul_parallelized(&a, &b, &mut c_parallelized, m, n, k);

        for i in 0..m {
            for j in 0..n {
                assert_eq!(c_naive[i * n + j], c_loop_interchange[i * n + j]);
                assert_eq!(c_naive[i * n + j], c_iterator[i * n + j]);
                assert_eq!(c_naive[i * n + j], c_simd[i * n + j]);
                assert_eq!(c_naive[i * n + j], c_parallelized_autovectorize[i * n + j]);
                assert_eq!(c_naive[i * n + j], c_parallelized[i * n + j]);
            }
        }
    }
}
