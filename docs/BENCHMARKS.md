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
- **RK3**: 0.000037s, 99.999992% accuracy, 201 steps
- **DDRK3**: 0.000172s, 99.999992% accuracy, 201 steps
- **AM**: 0.000059s, 99.999991% accuracy, 201 steps
- **DDAM**: 0.000712s, 99.999991% accuracy, 201 steps

### Harmonic Oscillator Test
- **RK3**: 0.000102s, 99.682004% accuracy, 629 steps
- **DDRK3**: 0.000553s, 99.682003% accuracy, 629 steps
- **AM**: 0.000198s, 99.320833% accuracy, 630 steps
- **DDAM**: 0.002480s, 99.320914% accuracy, 630 steps

### Lorenz System Test
- **RK3**: 0.000018s, 101 steps
- **DDRK3**: 0.000091s, 101 steps

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
