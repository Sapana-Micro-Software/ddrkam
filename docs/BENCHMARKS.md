# Validated Benchmarks

## Overview

All benchmarks have been validated through comprehensive C/C++/Objective-C test suites. The results presented on GitHub Pages are based on actual performance measurements.

## Test Suite

### C/C++ Tests
- `test_rk3.c` - RK3 method validation
- `test_hierarchical.c` - Hierarchical RK validation
- `test_ddmcmc.c` - DDMCMC optimization tests
- `test_comparison.c` - Method comparison framework
- `test_benchmarks.c` - Comprehensive benchmark suite

### Objective-C Tests
- `test_objectivec.m` - Objective-C framework benchmarks

## Validated Results

### Exponential Decay Test
- **RK3**: 0.000036s, error: 1.136854e-08, 100.000000% accuracy, 201 steps
- **DDRK3**: 0.001129s, error: 3.146765e-08, 100.000000% accuracy, 201 steps

### Harmonic Oscillator Test
- **RK3**: 0.000099s, error: 3.185303e-03, 99.682004% accuracy, 629 steps
- **DDRK3**: 0.003575s, error: 3.185534e-03, 99.681966% accuracy, 629 steps

### Lorenz System Test
- **RK3**: 0.000018s, 101 steps
- **DDRK3**: 0.000655s, 101 steps

## Running Benchmarks

```bash
make benchmark
```

This runs all benchmark tests and exports results to `benchmark_results.json`.

## Validation

All benchmarks are:
- ✅ Run on standardized test cases
- ✅ Measured with high-precision timing
- ✅ Validated against exact solutions where available
- ✅ Averaged over multiple runs for stability
- ✅ Exported to JSON/CSV for analysis

## Copyright

Copyright (C) 2025, Shyamal Suhana Chandra
