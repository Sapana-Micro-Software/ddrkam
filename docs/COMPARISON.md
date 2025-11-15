# Method Comparison: RK3 vs DDRK3 vs AM vs DDAM

## Overview

This document provides a comprehensive comparison of four numerical methods implemented in DDRKAM:

- **RK3**: Standard Runge-Kutta 3rd order method
- **DDRK3**: Data-Driven Runge-Kutta 3rd order (hierarchical/transformer-inspired)
- **AM**: Standard Adams Methods (Bashforth & Moulton)
- **DDAM**: Data-Driven Adams Methods (hierarchical)

## Usage

### C/C++ API

```c
#include "comparison.h"

void my_ode(double t, const double* y, double* dydt, void* params) {
    dydt[0] = -y[0];
}

int main() {
    double t0 = 0.0, t_end = 1.0;
    double y0[1] = {1.0};
    double exact[1] = {exp(-t_end)};
    double h = 0.01;
    
    ComparisonResults results;
    compare_methods(my_ode, t0, t_end, y0, 1, h, NULL, exact, &results);
    
    print_comparison_results(&results);
    export_comparison_csv("results.csv", &results);
    
    return 0;
}
```

### Objective-C API

```objc
#import <DDRKAM/DDRKAMComparison.h>

DDRKAMComparisonResults* results = [DDRKAMComparison compareMethodsWithFunction:^(double t, 
                                                                                  const double* y, 
                                                                                  double* dydt, 
                                                                                  void* params) {
    dydt[0] = -y[0];
} startTime:0.0 endTime:1.0 
initialState:@[@1.0] stepSize:0.01 
exactSolution:@[@(exp(-1.0))] params:NULL];

NSLog(@"%@", results);
[DDRKAMComparison exportResults:results toCSV:@"/path/to/results.csv"];
```

## Comparison Metrics

The comparison framework evaluates:

1. **Execution Time**: Wall-clock time for integration
2. **Steps**: Number of integration steps taken
3. **Error**: L2 norm error compared to exact solution
4. **Accuracy**: Percentage accuracy (1 - relative error)

## Expected Results

### Performance Characteristics

- **RK3**: Fast, good accuracy, standard method
- **DDRK3**: Slightly slower due to hierarchical processing, potentially better accuracy
- **AM**: Multi-step method, good for smooth solutions
- **DDAM**: Hierarchical Adams, adaptive refinement

### When to Use

- **RK3**: General purpose, well-tested, fast
- **DDRK3**: Complex systems, adaptive refinement needed
- **AM**: Smooth solutions, multi-step efficiency
- **DDAM**: High-dimensional, complex optimization landscapes

## Running Comparisons

```bash
make test
# Runs comparison tests and generates CSV files
```

Output files:
- `exponential_comparison.csv`
- `oscillator_comparison.csv`

## Copyright

Copyright (C) 2025, Shyamal Suhana Chandra
