# Method Comparison: RK3 vs DDRK3 vs AM vs DDAM

## Overview

This document provides a comprehensive comparison of numerical methods implemented in DDRKAM:

**Standard Methods:**
- **RK3**: Standard Runge-Kutta 3rd order method
- **DDRK3**: Data-Driven Runge-Kutta 3rd order (hierarchical/transformer-inspired)
- **AM**: Standard Adams Methods (Bashforth & Moulton)
- **DDAM**: Data-Driven Adams Methods (hierarchical)

**Parallel Methods:**
- **Parallel RK3**: Multi-threaded RK3 with OpenMP/pthreads
- **Parallel AM**: Multi-threaded Adams Methods
- **Stacked RK3**: Hierarchical stacked layers with attention

**Real-Time & Online Methods:**
- **Real-Time RK3**: Streaming data processing with minimal latency
- **Online RK3**: Adaptive online learning with step size adjustment
- **Dynamic RK3**: Dynamic step size adaptation

**Advanced Solvers:**
- **Nonlinear ODE Solver**: Gradient descent-based nonlinear programming
- **Distributed Data-Driven**: Distributed computing with data-driven methods
- **Online Data-Driven**: Online learning combined with data-driven architecture
- **Real-Time Data-Driven**: Real-time processing with data-driven enhancement

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

**Standard Methods:**
- **RK3**: Fast, good accuracy, standard method
- **DDRK3**: Slightly slower due to hierarchical processing, potentially better accuracy
- **AM**: Multi-step method, good for smooth solutions
- **DDAM**: Hierarchical Adams, adaptive refinement

**Parallel Methods:**
- **Parallel RK3**: Faster execution on multi-core systems, maintains accuracy
- **Stacked RK3**: Enhanced accuracy through hierarchical refinement, moderate overhead

**Real-Time & Online:**
- **Real-Time RK3**: Optimized for streaming data, minimal latency
- **Online RK3**: Adaptive step sizing, good for varying dynamics
- **Dynamic RK3**: Automatic step size adjustment based on error estimates

**Advanced Solvers:**
- **Nonlinear ODE**: Handles nonlinear constraints and optimization objectives
- **Distributed Data-Driven**: Scalable to large systems, combines distributed computing with ML
- **Online Data-Driven**: Continuous learning from streaming data
- **Real-Time Data-Driven**: Low-latency processing with adaptive learning

### When to Use

- **RK3**: General purpose, well-tested, fast
- **DDRK3**: Complex systems, adaptive refinement needed
- **AM**: Smooth solutions, multi-step efficiency
- **DDAM**: High-dimensional, complex optimization landscapes
- **Parallel RK3**: Multi-core systems, large-scale problems
- **Real-Time RK3**: Streaming data, live monitoring applications
- **Online RK3**: Systems with varying dynamics, adaptive control
- **Nonlinear ODE**: Constrained optimization problems, nonlinear programming
- **Distributed Data-Driven**: Large-scale distributed systems, cloud computing

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
