# Randomized Dynamic Programming for ODE Solving

## Overview

This document describes **randomized dynamic programming algorithms** for solving differential equations. These methods combine the optimal substructure of dynamic programming with randomization to handle high-dimensional state spaces efficiently, providing both probabilistic guarantees and practical performance.

## Theoretical Foundation

### Randomized Dynamic Programming

Traditional dynamic programming for ODEs requires exploring the entire state space, leading to exponential complexity in high dimensions. **Randomized DP** uses:

1. **Random Sampling**: Sample states instead of enumerating all
2. **Monte Carlo Estimation**: Estimate values via sampling
3. **Importance Sampling**: Focus on high-probability regions
4. **Adaptive Refinement**: Dynamically refine sampled regions

### Algorithm Variants

#### 1. Monte Carlo Dynamic Programming (MCDP)

**Concept**: Sample random paths through state space, estimate value function via Monte Carlo.

```
V(s) ≈ (1/N) Σᵢ R(pathᵢ) where pathᵢ starts at s
```

**Complexity**: O(N × T) where N = samples (fixed), T = time steps
**O(1) per step**: If N is fixed constant

#### 2. Randomized Value Iteration (RVI)

**Concept**: Randomly sample next states in value iteration.

```
V(s) ← max_{a} E[R(s,a) + γ·V(s')] where s' ~ sample from transition
```

**Complexity**: O(N × |A|) where N = samples per action (fixed)
**O(1) per step**: With fixed sampling budget

#### 3. UCB-based Randomized DP (UCB-RDP)

**Concept**: Use Upper Confidence Bound to balance exploration/exploitation.

```
V(s) ← max_{a} [Q(s,a) + c·√(log(t)/N(s,a))]
```

**Complexity**: O(N × |A|) with adaptive sampling
**O(1) per step**: With fixed exploration budget

#### 4. Thompson Sampling DP (TS-DP)

**Concept**: Sample from posterior over value functions.

```
V(s) ~ p(V | observations)
Action: a* = argmax E[V(s') | V ~ posterior]
```

**Complexity**: O(N × |A|) where N = posterior samples
**O(1) per step**: With fixed posterior sample size

## Randomized DP for ODEs

### Problem Formulation

Given ODE: `dy/dt = f(t, y, θ)`

Formulate as **stochastic optimal control**:

```
minimize: J = ∫ L(y(t), u(t), t) dt
subject to: dy/dt = f(t, y, u(t), θ)
           y(0) = y₀
```

Where:
- **State**: y(t)
- **Control**: u(t) (can be step size, method selection, etc.)
- **Cost**: L(y, u, t) (error, computation time, etc.)

### Randomized Value Function

**Value function**: V(t, y) = minimum cost from (t, y) to final time

**Randomized estimation**:
```
V(t, y) ≈ (1/N) Σᵢ min_{u} [L(y, u, t) + V(t+Δt, y + f·Δt + noise_i)]
```

Where `noise_i` are random samples representing uncertainty.

### Algorithm: Randomized DP ODE Solver

```c
typedef struct {
    // State space sampling
    double** sampled_states;      // [N][state_dim] - random samples
    double* state_weights;         // [N] - importance weights
    
    // Value function estimates
    double* value_estimates;       // [N] - V(t, y_i) for each sample
    
    // Control/action space
    double* control_candidates;    // [M] - candidate step sizes/methods
    size_t num_controls;
    
    // Sampling parameters
    size_t num_samples;            // N - fixed constant for O(1)
    double exploration_rate;       // ε for ε-greedy
    double ucb_constant;           // c for UCB
    
    // ODE parameters
    void (*ode_func)(double t, const double* y, double* dydt, void* params);
    void* ode_params;
    
    // Cost function
    double (*cost_function)(double t, const double* y, const double* u, void* params);
    void* cost_params;
    
    // Statistics
    double* best_control;          // Optimal control at each step
    double* expected_value;        // Expected value estimate
    double* value_variance;        // Variance of value estimate
} RandomizedDPSolver;
```

### Implementation Strategy

#### Step 1: Random State Sampling

```c
// Sample N states from neighborhood of current state
void sample_states(RandomizedDPSolver* solver,
                  const double* current_state,
                  double radius,
                  double** sampled_states) {
    for (size_t i = 0; i < solver->num_samples; i++) {
        // Sample from Gaussian around current state
        for (size_t j = 0; j < solver->state_dim; j++) {
            double noise = gaussian_random(0.0, radius);
            sampled_states[i][j] = current_state[j] + noise;
        }
    }
}
```

#### Step 2: Value Function Estimation

```c
// Estimate V(t, y) via Monte Carlo
double estimate_value(RandomizedDPSolver* solver,
                     double t,
                     const double* y,
                     double* optimal_control) {
    double min_value = INFINITY;
    double best_control = 0.0;
    
    // Try each control candidate
    for (size_t m = 0; m < solver->num_controls; m++) {
        double u = solver->control_candidates[m];
        
        // Sample next states
        double** next_states = allocate_samples(solver->num_samples, solver->state_dim);
        sample_next_states(solver, t, y, u, next_states);
        
        // Estimate expected value
        double expected = 0.0;
        for (size_t i = 0; i < solver->num_samples; i++) {
            double cost = solver->cost_function(t, y, &u, solver->cost_params);
            double next_value = lookup_value(solver, t + u, next_states[i]);
            expected += (cost + next_value) / solver->num_samples;
        }
        
        // UCB: add exploration bonus
        double ucb_value = expected + solver->ucb_constant * 
                          sqrt(log(solver->step_count) / solver->control_counts[m]);
        
        if (ucb_value < min_value) {
            min_value = ucb_value;
            best_control = u;
        }
        
        free_samples(next_states, solver->num_samples);
    }
    
    *optimal_control = best_control;
    return min_value;
}
```

#### Step 3: Backward Induction

```c
// Solve ODE using randomized DP (backward from final time)
int randomized_dp_solve(RandomizedDPSolver* solver,
                      double t0,
                      double t_end,
                      const double* y0,
                      double* solution_path) {
    // Discretize time
    size_t num_steps = (size_t)((t_end - t0) / solver->min_step_size);
    double* time_points = allocate_time_points(num_steps);
    
    // Initialize value function at final time
    for (size_t i = 0; i < solver->num_samples; i++) {
        solver->value_estimates[i] = final_cost(solver, solver->sampled_states[i]);
    }
    
    // Backward induction
    for (int step = num_steps - 2; step >= 0; step--) {
        double t = time_points[step];
        
        // Sample states at this time
        sample_states(solver, solution_path + step * solver->state_dim,
                     solver->sampling_radius, solver->sampled_states);
        
        // Estimate value function
        for (size_t i = 0; i < solver->num_samples; i++) {
            double optimal_u;
            solver->value_estimates[i] = estimate_value(solver, t,
                                                       solver->sampled_states[i],
                                                       &optimal_u);
            solver->best_control[i] = optimal_u;
        }
    }
    
    // Forward pass: execute optimal policy
    double* y_current = copy_state(y0, solver->state_dim);
    double t = t0;
    size_t step = 0;
    
    while (t < t_end) {
        // Find nearest sampled state
        size_t nearest = find_nearest_sample(solver, y_current);
        double optimal_u = solver->best_control[nearest];
        
        // Apply control and step forward
        step_forward(solver, t, y_current, optimal_u, y_current);
        t += optimal_u;
        
        // Store solution
        copy_state(y_current, solution_path + step * solver->state_dim);
        step++;
    }
    
    free(y_current);
    free(time_points);
    return 0;
}
```

## O(1) Real-Time Implementation

### Key Optimizations

1. **Fixed Sample Size**: N = constant → O(1) sampling
2. **Pre-computed Controls**: Fixed control candidates → O(1) evaluation
3. **Hash-based Lookup**: O(1) value function lookup
4. **Incremental Updates**: Only update changed regions

### Real-Time Variant

```c
// O(1) per-step randomized DP
int realtime_randomized_dp_step(RandomizedDPSolver* solver,
                                double t,
                                const double* y_current,
                                double* y_next,
                                double* optimal_control) {
    // Sample states: O(N) but N fixed → O(1)
    sample_states(solver, y_current, solver->sampling_radius,
                 solver->sampled_states);
    
    // Estimate value: O(N × M) but N, M fixed → O(1)
    double min_value = INFINITY;
    double best_u = solver->control_candidates[0];
    
    for (size_t m = 0; m < solver->num_controls; m++) {
        double u = solver->control_candidates[m];
        double value = estimate_value_fast(solver, t, y_current, u);
        
        if (value < min_value) {
            min_value = value;
            best_u = u;
        }
    }
    
    // Apply optimal control: O(1)
    step_forward(solver, t, y_current, best_u, y_next);
    *optimal_control = best_u;
    
    return 0;
}
```

## Probabilistic Guarantees

### Concentration Bounds

With N samples, value estimate satisfies:

```
P(|V̂(s) - V(s)| > ε) ≤ 2·exp(-2N·ε²/R²)
```

Where R is the range of rewards.

### Convergence

**Theorem**: Randomized DP converges to optimal value function as N → ∞.

**Rate**: O(1/√N) convergence in probability.

## Applications

1. **Adaptive Step Size Selection**: Choose optimal step size at each step
2. **Method Selection**: Choose best numerical method (RK3 vs AM) dynamically
3. **Uncertainty Propagation**: Handle uncertain parameters via sampling
4. **Multi-Objective Optimization**: Balance accuracy vs speed
5. **Real-Time Control**: O(1) decisions for hard deadlines

## Comparison with Other Methods

| Method | Complexity | Output | Guarantees |
|--------|-----------|--------|------------|
| Standard DP | O(|S|²) | Optimal | Exact |
| Randomized DP | O(N) ≈ O(1) | Near-optimal | Probabilistic |
| Greedy | O(1) | Suboptimal | None |
| Randomized DP | O(N) ≈ O(1) | Near-optimal | High probability |

## Implementation Notes

- **Sample Size**: N = 100-1000 typically sufficient
- **Exploration**: UCB constant c = √2 for good exploration
- **State Sampling**: Gaussian with radius = 0.1 × state range
- **Control Candidates**: Logarithmic spacing (h, h/2, h/4, ...)

## References

- **Monte Carlo Methods**: Robert & Casella (2004). "Monte Carlo Statistical Methods"
- **UCB Algorithm**: Auer et al. (2002). "Finite-time Analysis of the Multiarmed Bandit Problem"
- **Thompson Sampling**: Thompson (1933). "On the Likelihood that One Unknown Probability Exceeds Another"
- **Randomized Algorithms**: Motwani & Raghavan (1995). "Randomized Algorithms"
