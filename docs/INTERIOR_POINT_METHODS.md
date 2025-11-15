# Interior Point Methods for Non-Convex, Nonlinear, and Online Algorithms

## Overview

This document describes the implementation of Interior Point Methods (IPM) for solving differential equations using non-convex, nonlinear, and online optimization algorithms.

## Features

### 1. **Interior Point Method Core Algorithm**
- Barrier function method for constrained optimization
- Slack variables for inequality constraints
- Dual (Lagrange multiplier) variables
- Barrier-augmented gradient computation
- Line search with Armijo condition for non-convex problems

### 2. **Non-Convex Optimization Support**
- Perturbation-based escape from local minima
- Multiple local minima handling
- Adaptive barrier parameter adjustment
- Conservative line search for non-convex landscapes

### 3. **Online/Adaptive Algorithms**
- Adaptive barrier parameter based on constraint violation
- Real-time barrier parameter adjustment
- Incremental learning from streaming data
- Dynamic step size adaptation

### 4. **Nonlinear Differential Equations**
- ODE solving as optimization problem
- Constraint handling (inequality constraints)
- Barrier function integration
- Feasibility and optimality tolerance control

## API Reference

### Initialization

```c
NonlinearODESolver solver;
nonlinear_ode_init(&solver, state_dim, NLP_INTERIOR_POINT,
                   objective_function, constraint_function, params);
```

### Setting Constraints

```c
// Set number of constraints
nonlinear_ode_set_constraints(&solver, num_constraints);

// Configure interior point parameters
InteriorPointParams ip_params = {
    .barrier_parameter = 1.0,
    .barrier_reduction = 0.1,
    .centering_parameter = 0.1,
    .feasibility_tolerance = 1e-6,
    .optimality_tolerance = 1e-6,
    .max_barrier_iterations = 100,
    .handle_nonconvex = 0,
    .perturbation_radius = 0.01
};
nonlinear_ode_set_interior_point_params(&solver, &ip_params);
```

### Enabling Non-Convex Handling

```c
// Enable non-convex optimization with perturbation
nonlinear_ode_enable_nonconvex(&solver, 1, 0.01);
```

### Solving

```c
double y0[1] = {1.0};
double y_out[1];
nonlinear_ode_solve(&solver, ode_function, t0, t_end, y0, y_out);
```

## Online Interior Point Method

The online version adapts the barrier parameter based on constraint violations:

```c
OnlineNonlinearSolver online_solver;
online_nonlinear_init(&online_solver, &base_solver, learning_rate);
online_nonlinear_solve(&online_solver, ode_function, t0, t_end, y0, y_out);
```

The online method:
- Monitors constraint violations in real-time
- Increases barrier parameter when violations are high
- Decreases barrier parameter when feasible
- Adapts step size based on error estimates

## Algorithm Details

### Barrier Function

For inequality constraints `g(x) <= 0`, the barrier function is:

```
φ(x, μ) = f(x) - μ * Σ log(-g_i(x))
```

where:
- `f(x)` is the objective function
- `μ` is the barrier parameter
- `g_i(x)` are the constraint functions

### Barrier-Augmented Gradient

The gradient includes both objective and barrier terms:

```
∇φ(x, μ) = ∇f(x) - μ * Σ (1/g_i(x)) * ∇g_i(x)
```

### Interior Point Step

1. Compute barrier-augmented gradient
2. Perform line search with backtracking
3. Update slack variables: `s = -g(x) + perturbation`
4. Update dual variables: `λ = μ / s`
5. Check convergence (gradient norm < tolerance)

### Non-Convex Handling

When `handle_nonconvex` is enabled:
- Perturbation is added to slack variables to escape local minima
- Conservative line search (Armijo condition)
- Multiple barrier parameter reductions

### Online Adaptation

The barrier parameter adapts as:
- If `max_violation > feasibility_tolerance`: `μ *= 1.5`
- If `max_violation <= feasibility_tolerance`: `μ *= barrier_reduction`
- Clamped to range `[1e-10, 1e6]`

## Parameters

### InteriorPointParams

- `barrier_parameter`: Initial barrier parameter (default: 1.0)
- `barrier_reduction`: Reduction factor per iteration (default: 0.1)
- `centering_parameter`: Centering parameter (default: 0.1)
- `feasibility_tolerance`: Constraint violation tolerance (default: 1e-6)
- `optimality_tolerance`: Gradient norm tolerance (default: 1e-6)
- `max_barrier_iterations`: Max iterations per barrier step (default: 100)
- `handle_nonconvex`: Enable non-convex handling (default: 0)
- `perturbation_radius`: Perturbation for non-convex escape (default: 0.01)

## Example Usage

```c
#include "nonlinear_solver.h"

// Objective: minimize ||dy/dt - f(t,y)||^2
double objective(const double* x, size_t n, void* params) {
    // Compute residual
    return residual;
}

// Constraints: y >= 0
void constraints(const double* x, size_t n, double* c, void* params) {
    for (size_t i = 0; i < n; i++) {
        c[i] = -x[i]; // -x <= 0 means x >= 0
    }
}

int main() {
    NonlinearODESolver solver;
    double y0[1] = {1.0};
    double y_out[1];
    
    // Initialize
    nonlinear_ode_init(&solver, 1, NLP_INTERIOR_POINT,
                       objective, constraints, NULL);
    
    // Set constraints
    nonlinear_ode_set_constraints(&solver, 1);
    
    // Enable non-convex
    nonlinear_ode_enable_nonconvex(&solver, 1, 0.01);
    
    // Solve
    nonlinear_ode_solve(&solver, ode_function, 0.0, 1.0, y0, y_out);
    
    // Cleanup
    nonlinear_ode_free(&solver);
    return 0;
}
```

## Applications

1. **Constrained ODE Solving**: Solve ODEs with inequality constraints (e.g., non-negativity)
2. **Non-Convex Optimization**: Handle problems with multiple local minima
3. **Online Learning**: Adapt to streaming data with constraint violations
4. **Real-Time Systems**: Process data with minimal latency
5. **Nonlinear Programming**: Solve general nonlinear optimization problems

## Performance Considerations

- **Barrier Parameter**: Start large, reduce gradually
- **Line Search**: Conservative for non-convex, aggressive for convex
- **Constraint Evaluation**: Minimize constraint function calls
- **Gradient Computation**: Use automatic differentiation when available

## References

- Nocedal, J., & Wright, S. (2006). *Numerical Optimization*. Springer.
- Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.
- Karmarkar, N. (1984). A new polynomial-time algorithm for linear programming.

## Copyright

Copyright (C) 2025, Shyamal Suhana Chandra
