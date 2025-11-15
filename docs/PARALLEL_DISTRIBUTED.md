# Parallel, Distributed, Concurrent, Hierarchical, and Stacked Methods

## Overview

This document describes the comprehensive parallel, distributed, concurrent, hierarchical, and stacked implementations of all numerical methods in DDRKAM.

## Methods Implemented

### 1. Runge-Kutta 3rd Order (RK3)
- **Parallel RK3**: Multi-threaded execution using OpenMP or pthreads
- **Distributed RK3**: MPI-based distributed computing
- **Stacked RK3**: Hierarchical/stacked architecture with multiple layers
- **Concurrent RK3**: Execute multiple RK3 instances simultaneously

### 2. Adams Methods (AM)
- **Parallel AM**: Parallel predictor-corrector execution
- **Distributed AM**: MPI-based distributed Adams methods
- **Stacked AM**: Hierarchical refinement of Adams steps
- **Concurrent AM**: Execute multiple AM instances simultaneously

### 3. Euler's Method
- **Parallel Euler**: Multi-threaded Euler steps
- **Distributed Euler**: MPI-based distributed Euler
- **Stacked Euler**: Hierarchical Euler with transformer layers
- **Concurrent Euler**: Execute multiple Euler instances simultaneously

### 4. Data-Driven Variants (DD)
All DD variants (DDRK3, DDAM, DDEuler) support:
- Parallel execution
- Distributed computing
- Stacked/hierarchical architectures
- Concurrent execution

## Parallel Execution Modes

### OpenMP (Shared Memory)
```c
ParallelRKSolver solver;
parallel_rk_init(&solver, state_dim, num_workers, PARALLEL_OPENMP, NULL);
```

### POSIX Threads (pthreads)
```c
ParallelRKSolver solver;
parallel_rk_init(&solver, state_dim, num_workers, PARALLEL_PTHREAD, NULL);
```

### MPI (Distributed)
```c
ParallelRKSolver solver;
parallel_rk_init(&solver, state_dim, num_workers, PARALLEL_MPI, NULL);
```

### Hybrid (MPI + OpenMP)
```c
ParallelRKSolver solver;
parallel_rk_init(&solver, state_dim, num_workers, PARALLEL_HYBRID, NULL);
```

## Stacked/Hierarchical Architecture

Stacked configurations allow multiple hierarchical layers with attention mechanisms:

```c
StackedConfig stacked = {
    .num_layers = 4,
    .layer_dims = {16, 32, 64, 32},
    .hidden_dim = 64,
    .learning_rate = 0.01,
    .use_attention = 1,
    .use_residual = 1
};

ParallelRKSolver solver;
parallel_rk_init(&solver, state_dim, num_workers, PARALLEL_OPENMP, &stacked);
```

## Concurrent Execution

Execute multiple methods simultaneously:

```c
ParallelRKSolver* solvers[3];
double** results;

// Initialize solvers
solvers[0] = &rk3_solver;
solvers[1] = &adams_solver;
solvers[2] = &euler_solver;

// Execute concurrently
concurrent_rk_execute(solvers, 3, f, t, y, h, params, results);
```

## Performance Characteristics

- **Parallel Speedup**: Up to N× speedup with N workers (for large systems)
- **Distributed Scaling**: Linear scaling across multiple nodes
- **Stacked Accuracy**: Enhanced accuracy through hierarchical refinement
- **Concurrent Efficiency**: Compare multiple methods in parallel

## API Reference

See `include/parallel_rk.h`, `include/parallel_adams.h`, and `include/parallel_euler.h` for complete API documentation.

## Examples

### Example 1: Parallel RK3
```c
#include "parallel_rk.h"

void my_ode(double t, const double* y, double* dydt, void* params) {
    dydt[0] = -y[0];
}

int main() {
    ParallelRKSolver solver;
    parallel_rk_init(&solver, 1, 4, PARALLEL_OPENMP, NULL);
    
    double y0[1] = {1.0};
    double t_out[100];
    double y_out[100];
    
    size_t steps = parallel_rk_solve(&solver, my_ode, 0.0, 1.0, y0, 
                                     0.01, NULL, t_out, y_out);
    
    parallel_rk_free(&solver);
    return 0;
}
```

### Example 2: Stacked RK3
```c
StackedConfig stacked = {
    .num_layers = 3,
    .hidden_dim = 32,
    .learning_rate = 0.01,
    .use_attention = 1
};

ParallelRKSolver solver;
parallel_rk_init(&solver, state_dim, 8, PARALLEL_OPENMP, &stacked);

// Use stacked_rk_step for hierarchical execution
double t_new = stacked_rk_step(&solver, f, t, y, h, params);
```

## Copyright

Copyright (C) 2025, Shyamal Suhana Chandra
