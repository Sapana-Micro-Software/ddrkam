# DDRKAM - Data-Driven Runge-Kutta and Adams Methods

Copyright (C) 2025, Shyamal Suhana Chandra

## Overview

DDRKAM is a comprehensive framework for solving differential equations (both **ODEs** and **PDEs**) using:
- Runge-Kutta 3rd order method (ODEs)
- Adams-Bashforth and Adams-Moulton methods (ODEs)
- Hierarchical data-driven architecture (Transformer-inspired)
- **PDE Solver**: Heat, Wave, Advection, Burgers, Laplace, Poisson equations
- **Real-Time Solvers**: Streaming data processing for RK3 and Adams methods
- **Stochastic Solvers**: Noise injection and Brownian motion support
- **Data-Driven Control**: Adaptive step size and method selection

## Features

- High-performance C/C++ core implementation
- Objective-C framework for Apple platforms (macOS, iOS, visionOS)
- Visualization capabilities
- Hierarchical/Transformer-like ODE solver
- **DDMCMC**: Data-Driven MCMC for multinomial optimization
- Efficient search algorithms for learning optimization functions
- Comprehensive documentation

## Building

### C/C++ Library

```bash
make
```

This builds both static (`lib/libddrkam.a`) and shared (`lib/libddrkam.dylib`) libraries.

### Running Tests

```bash
make test
```

## Usage

### C API

```c
#include "rk3.h"

void my_ode(double t, const double* y, double* dydt, void* params) {
    dydt[0] = -y[0];
}

double y0[1] = {1.0};
double t_out[100];
double y_out[100];
size_t steps = rk3_solve(my_ode, 0.0, 1.0, y0, 1, 0.01, 
                         NULL, t_out, y_out);
```

### Objective-C API

```objc
#import <DDRKAM/DDRKAM.h>

DDRKAMSolver* solver = [[DDRKAMSolver alloc] initWithDimension:2];
NSDictionary* result = [solver solveWithFunction:^(double t, 
                                                    const double* y, 
                                                    double* dydt, 
                                                    void* params) {
    dydt[0] = -y[0];
    dydt[1] = -2.0 * y[1];
} startTime:0.0 endTime:1.0 
initialState:@[@1.0, @1.0] stepSize:0.01 params:NULL];
```

## Method Comparison

Compare all four methods (RK3, DDRK3, AM, DDAM):

```c
#include "comparison.h"

ComparisonResults results;
compare_methods(my_ode, t0, t_end, y0, n, h, params, exact, &results);
print_comparison_results(&results);
```

See `docs/COMPARISON.md` for detailed comparison guide.

## PDE Solving

DDRKAM now supports solving **Partial Differential Equations (PDEs)**:

```c
#include "pde_solver.h"

// Solve 1D Heat Equation
PDEProblem problem;
pde_problem_init(&problem, PDE_HEAT, DIM_1D, 100, 1, 1, 0.01, 1.0, 1.0, 0.0001);
problem.alpha = 0.1;

PDESolution solution;
pde_solve_heat_1d(&problem, 0.1, &solution);
```

See `docs/PDE_GUIDE.md` for complete PDE documentation.

## Documentation

- Paper: `docs/paper.tex`
- Presentation: `docs/presentation.tex`
- Reference Manual: `docs/reference_manual.tex`
- DDMCMC Guide: `docs/DDMCMC_README.md`
- Comparison Guide: `docs/COMPARISON.md`
- PDE Guide: `docs/PDE_GUIDE.md`
- Real-Time & Stochastic Guide: `docs/REALTIME_STOCHASTIC.md`

## License

Copyright (C) 2025, Shyamal Suhana Chandra. All rights reserved.

For licensing information, please contact:
**sapanamicrosoftware@duck.com**

## Repository

https://github.com/Sapana-Micro-Software/ddrkam
