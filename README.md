# Playing with Rust Portable SIMD

This repo contains some simplr benchmarks on the portable SIMD in Rust.

## Performance

Running in release mode, the output can be similar as follows on a M2 Max MacBook Pro:

```text
Loop interchange: 0.079338 s, 27.067447 GFLOPS
Iterator: 0.078275 s, 27.435095 GFLOPS
SIMD: 0.077029 s, 27.878780 GFLOPS
Rayon threads: 12
Parallelized autovectorize: 0.010069 s, 213.266677 GFLOPS
Parallelized: 0.010252 s, 209.468485 GFLOPS
```
