# PDE Solver Guide

## Overview

DDRKAM now supports solving both **Ordinary Differential Equations (ODEs)** and **Partial Differential Equations (PDEs)**. This guide covers the PDE solving capabilities.

## Supported PDE Types

### 1. Heat/Diffusion Equation
- **1D**: `u_t = α * u_xx`
- **2D**: `u_t = α * (u_xx + u_yy)`
- Applications: Heat conduction, diffusion processes

### 2. Wave Equation
- **1D**: `u_tt = c² * u_xx`
- Applications: Vibrating strings, acoustic waves

### 3. Advection Equation
- **1D**: `u_t + a * u_x = 0`
- Applications: Transport phenomena, fluid flow

### 4. Burgers Equation
- **1D**: `u_t + u * u_x = ν * u_xx`
- Applications: Nonlinear wave propagation, shock waves

### 5. Laplace Equation
- **2D**: `u_xx + u_yy = 0`
- Applications: Steady-state heat distribution, potential fields

### 6. Poisson Equation
- **2D**: `u_xx + u_yy = f(x,y)`
- Applications: Electrostatics, gravitational fields

## C/C++ API Usage

### 1D Heat Equation

```c
#include "pde_solver.h"

PDEProblem problem;
pde_problem_init(&problem, PDE_HEAT, DIM_1D, 100, 1, 1, 0.01, 1.0, 1.0, 0.0001);
problem.alpha = 0.1;

// Set initial condition
for (size_t i = 0; i < problem.nx; i++) {
    double x = i * problem.dx;
    problem.initial_condition[i] = exp(-(x - 0.5) * (x - 0.5) / 0.01);
}

PDESolution solution;
pde_solve_heat_1d(&problem, 0.1, &solution);

// Export solution
pde_export_solution(&solution, &problem, "heat_solution.csv");

pde_solution_free(&solution);
pde_problem_free(&problem);
```

### 2D Heat Equation

```c
PDEProblem problem;
pde_problem_init(&problem, PDE_HEAT, DIM_2D, 50, 50, 1, 0.02, 0.02, 1.0, 0.0001);
problem.alpha = 0.1;

// Set 2D initial condition
for (size_t j = 0; j < problem.ny; j++) {
    for (size_t i = 0; i < problem.nx; i++) {
        double x = i * problem.dx;
        double y = j * problem.dy;
        problem.initial_condition[j * problem.nx + i] = 
            exp(-((x-0.5)*(x-0.5) + (y-0.5)*(y-0.5)) / 0.01);
    }
}

PDESolution solution;
pde_solve_heat_2d(&problem, 0.05, &solution);
```

### Wave Equation

```c
PDEProblem problem;
pde_problem_init(&problem, PDE_WAVE, DIM_1D, 100, 1, 1, 0.01, 1.0, 1.0, 0.001);
problem.c = 1.0;  // Wave speed

// Initial condition: sine wave
for (size_t i = 0; i < problem.nx; i++) {
    double x = i * problem.dx;
    problem.initial_condition[i] = sin(M_PI * x);
}

PDESolution solution;
pde_solve_wave_1d(&problem, 0.5, &solution);
```

## Objective-C API Usage

```objc
#import <DDRKAM/DDRKAMPDESolver.h>

// Solve 1D Heat Equation
NSMutableArray<NSNumber*>* u0 = [NSMutableArray arrayWithCapacity:100];
for (NSUInteger i = 0; i < 100; i++) {
    double x = i * 0.01;
    [u0 addObject:@(exp(-(x - 0.5) * (x - 0.5) / 0.01))];
}

DDRKAMPDESolution* solution = [DDRKAMPDESolver solveHeat1DWithGridPoints:100
                                                              spatialStep:0.01
                                                                timeStep:0.0001
                                                           diffusionCoeff:0.1
                                                         initialCondition:u0
                                                                finalTime:0.1];

NSLog(@"Solution: %@", solution);
[DDRKAMPDESolver exportSolution:solution toFile:@"/path/to/output.csv" 
                       dimension:DDRKAMSpatialDimension1D gridSizeX:100 gridSizeY:1];
```

## Stability Conditions

- **Heat Equation**: `r = α * dt / dx² ≤ 0.5`
- **Wave Equation**: `r = c * dt / dx ≤ 1.0` (CFL condition)
- **Advection Equation**: `|r| = |a * dt / dx| ≤ 1.0`

## Running Tests

```bash
make test
# Includes PDE solver tests
```

## Copyright

Copyright (C) 2025, Shyamal Suhana Chandra
