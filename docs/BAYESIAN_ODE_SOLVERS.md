# Real-Time Bayesian ODE Solvers with Dynamic Programming

## Overview

This document describes **O(1) real-time Bayesian solvers** for differential equations that use **dynamic programming** to provide both **probabilistic (approximate)** and **exact (MAP)** solutions. These solvers treat ODE solving as a Bayesian state estimation problem, enabling uncertainty quantification and optimal path finding.

## Theoretical Foundation

### Bayesian ODE Formulation

Given an ODE:
```
dy/dt = f(t, y, θ)
```

We formulate it as a **Bayesian state estimation problem**:

1. **State Space Model**:
   - **Hidden States**: y(t) - the true solution
   - **Observations**: Noisy measurements or predictions
   - **Parameters**: θ - ODE parameters (can be uncertain)

2. **Probabilistic Model**:
   ```
   p(y(t+Δt) | y(t), θ) ~ Transition model
   p(observation | y(t)) ~ Observation model
   p(θ) ~ Prior on parameters
   ```

3. **Inference Goals**:
   - **Probabilistic**: Full posterior p(y(t) | observations)
   - **Exact (MAP)**: Most likely path argmax p(y(t) | observations)

### Dynamic Programming Approaches

#### 1. Forward-Backward Algorithm (Probabilistic)

Computes **marginal posteriors** p(y(t) | all observations):

```
Forward Pass: α(t) = p(y(t) | observations[0:t])
Backward Pass: β(t) = p(observations[t+1:T] | y(t))
Posterior: p(y(t) | all) ∝ α(t) × β(t)
```

**Complexity**: O(T × |S|²) where T = time steps, |S| = state space size
**O(1) per step**: With pre-computed transition matrices and fixed state space

#### 2. Viterbi Algorithm (Exact/MAP)

Finds **most likely path** (MAP estimate):

```
V(t, s) = max_{path} p(path ending at state s at time t | observations)
```

**Complexity**: O(T × |S|²)
**O(1) per step**: With pre-computed transitions

#### 3. Particle Filter (Approximate Probabilistic)

Uses **Monte Carlo sampling** for nonlinear/non-Gaussian cases:

```
Particles: {y₁(t), y₂(t), ..., yₙ(t)} with weights {w₁, w₂, ..., wₙ}
Posterior: p(y(t)) ≈ Σᵢ wᵢ δ(y - yᵢ(t))
```

**Complexity**: O(N) per step where N = number of particles (fixed)
**O(1)**: If N is fixed constant

## O(1) Implementation Strategies

### Strategy 1: Pre-computed Transition Matrices

**Offline**: Pre-compute transition probabilities for discretized state space
**Online**: O(1) lookup and matrix-vector multiplication

```c
// Pre-compute (offline)
double** transition_matrix;  // [state_i][state_j] = p(y(t+1)=j | y(t)=i)

// Online: O(1) per step
void forward_step(double* alpha_prev, double* alpha_next) {
    // Matrix-vector multiply: O(|S|²) but |S| is fixed → O(1)
    for (size_t i = 0; i < state_space_size; i++) {
        alpha_next[i] = 0.0;
        for (size_t j = 0; i < state_space_size; j++) {
            alpha_next[i] += transition_matrix[j][i] * alpha_prev[j];
        }
    }
}
```

### Strategy 2: Fixed-Size Particle Filter

**Fixed number of particles** → O(1) per step

```c
#define NUM_PARTICLES 100  // Fixed constant

typedef struct {
    double* particles[NUM_PARTICLES];  // Fixed size
    double weights[NUM_PARTICLES];     // Fixed size
} ParticleFilter;

// O(1) step: fixed number of particles
void particle_filter_step(ParticleFilter* pf, double observation) {
    // Resample, propagate, update: all O(NUM_PARTICLES) = O(1)
    for (size_t i = 0; i < NUM_PARTICLES; i++) {
        propagate_particle(&pf->particles[i]);
        update_weight(&pf->weights[i], observation);
    }
    normalize_weights(pf->weights, NUM_PARTICLES);
}
```

### Strategy 3: Discretized State Space with Hash Lookup

**Discretize state space** → O(1) hash lookup

```c
// Discretize: y ∈ [y_min, y_max] → {0, 1, 2, ..., N-1}
size_t discretize_state(double y) {
    return (size_t)((y - y_min) / (y_max - y_min) * (N - 1));
}

// O(1) hash lookup for transition probabilities
double get_transition_prob(size_t from, size_t to) {
    size_t hash = hash_pair(from, to);
    return transition_hash_table[hash];  // O(1) lookup
}
```

## Implementation Modes

### Mode 1: Probabilistic (Full Posterior)

**Output**: Full probability distribution p(y(t) | observations)

```c
typedef struct {
    double* posterior;        // p(y(t) | observations) for each state
    size_t state_space_size;
    double* state_values;    // Discretized state values
} ProbabilisticSolver;

// Returns: mean, variance, full distribution
int solve_probabilistic(ProbabilisticSolver* solver,
                       double t,
                       double* y_mean,
                       double* y_variance,
                       double* full_posterior);
```

**Use Cases**:
- Uncertainty quantification
- Risk assessment
- Robust control
- Confidence intervals

### Mode 2: Exact (MAP Estimate)

**Output**: Most likely solution path (Viterbi)

```c
typedef struct {
    double* map_path;        // Most likely y(t) for each time
    double map_probability;   // Probability of MAP path
    size_t* backpointers;    // For path reconstruction
} ExactSolver;

// Returns: MAP estimate
int solve_exact(ExactSolver* solver,
               double t,
               double* y_map);
```

**Use Cases**:
- Deterministic control
- Optimal path planning
- Maximum likelihood estimation
- When exact solution is required

### Mode 3: Hybrid (Both)

**Output**: Both MAP and full posterior

```c
typedef struct {
    ProbabilisticSolver* prob;
    ExactSolver* exact;
    int mode;  // 0=probabilistic, 1=exact, 2=both
} HybridBayesianSolver;
```

## Dynamic Programming Algorithms

### Forward-Backward for Probabilistic Inference

```c
// Forward pass: α(t) = p(y(t) | observations[0:t])
void forward_pass(double** alpha,           // [time][state]
                  double** transition,       // [state][state]
                  double** observation,     // [time][state]
                  size_t T, size_t S) {
    // Initialize: α(0) = prior
    for (size_t s = 0; s < S; s++) {
        alpha[0][s] = prior[s] * observation[0][s];
    }
    normalize(alpha[0], S);
    
    // Forward recursion: O(T × S²) but S fixed → O(T) ≈ O(1) per step
    for (size_t t = 1; t < T; t++) {
        for (size_t s = 0; s < S; s++) {
            alpha[t][s] = 0.0;
            for (size_t s_prev = 0; s_prev < S; s_prev++) {
                alpha[t][s] += transition[s_prev][s] * alpha[t-1][s_prev];
            }
            alpha[t][s] *= observation[t][s];
        }
        normalize(alpha[t], S);
    }
}

// Backward pass: β(t) = p(observations[t+1:T] | y(t))
void backward_pass(double** beta,
                   double** transition,
                   double** observation,
                   size_t T, size_t S) {
    // Initialize: β(T-1) = 1
    for (size_t s = 0; s < S; s++) {
        beta[T-1][s] = 1.0;
    }
    
    // Backward recursion
    for (int t = T-2; t >= 0; t--) {
        for (size_t s = 0; s < S; s++) {
            beta[t][s] = 0.0;
            for (size_t s_next = 0; s_next < S; s_next++) {
                beta[t][s] += transition[s][s_next] * 
                              observation[t+1][s_next] * 
                              beta[t+1][s_next];
            }
        }
        normalize(beta[t], S);
    }
}

// Posterior: p(y(t) | all observations) ∝ α(t) × β(t)
void compute_posterior(double** alpha, double** beta,
                      double** posterior, size_t T, size_t S) {
    for (size_t t = 0; t < T; t++) {
        for (size_t s = 0; s < S; s++) {
            posterior[t][s] = alpha[t][s] * beta[t][s];
        }
        normalize(posterior[t], S);
    }
}
```

### Viterbi for Exact (MAP) Inference

```c
// Viterbi algorithm: find most likely path
void viterbi(double** viterbi_table,      // V(t, s) = max prob ending at s at t
             size_t** backpointers,        // For path reconstruction
             double** transition,
             double** observation,
             double* prior,
             size_t T, size_t S) {
    // Initialize: V(0, s) = prior[s] × observation[0][s]
    for (size_t s = 0; s < S; s++) {
        viterbi_table[0][s] = prior[s] * observation[0][s];
        backpointers[0][s] = 0;  // No previous state
    }
    normalize(viterbi_table[0], S);
    
    // Forward recursion: O(T × S²)
    for (size_t t = 1; t < T; t++) {
        for (size_t s = 0; s < S; s++) {
            double max_prob = -INFINITY;
            size_t best_prev = 0;
            
            for (size_t s_prev = 0; s_prev < S; s_prev++) {
                double prob = viterbi_table[t-1][s_prev] * 
                             transition[s_prev][s] * 
                             observation[t][s];
                if (prob > max_prob) {
                    max_prob = prob;
                    best_prev = s_prev;
                }
            }
            
            viterbi_table[t][s] = max_prob;
            backpointers[t][s] = best_prev;
        }
        normalize(viterbi_table[t], S);
    }
}

// Reconstruct MAP path
void reconstruct_map_path(size_t** backpointers,
                         double** viterbi_table,
                         size_t* map_path,
                         size_t T, size_t S) {
    // Find best final state
    double max_prob = -INFINITY;
    size_t best_final = 0;
    for (size_t s = 0; s < S; s++) {
        if (viterbi_table[T-1][s] > max_prob) {
            max_prob = viterbi_table[T-1][s];
            best_final = s;
        }
    }
    
    // Backtrack
    map_path[T-1] = best_final;
    for (int t = T-2; t >= 0; t--) {
        map_path[t] = backpointers[t+1][map_path[t+1]];
    }
}
```

## Real-Time O(1) Implementation

### Key Optimizations

1. **Pre-computed Transitions**: Compute transition matrices offline
2. **Fixed State Space**: Discretize to fixed size |S| → O(1) operations
3. **Incremental Updates**: Only update changed states
4. **Parallel Evaluation**: Evaluate multiple states in parallel
5. **Caching**: Cache frequently used probabilities

### Example: O(1) Real-Time Bayesian Solver

```c
typedef struct {
    // Pre-computed (offline)
    double** transition_matrix;  // [S][S] - fixed size
    double* state_values;        // [S] - discretized states
    size_t state_space_size;     // S - fixed constant
    
    // Online state
    double* alpha;                // Forward probabilities [S]
    double* beta;                 // Backward probabilities [S]
    double* posterior;           // Current posterior [S]
    
    // Mode
    int mode;  // 0=probabilistic, 1=exact, 2=both
    
    // Exact mode
    double* viterbi;              // Viterbi table [S]
    size_t* backpointers;         // For path reconstruction
    
    // Statistics
    double* y_mean;               // Mean estimate
    double* y_variance;           // Variance estimate
    double* y_map;                 // MAP estimate
} RealTimeBayesianSolver;

// O(1) per-step update
int realtime_bayesian_step(RealTimeBayesianSolver* solver,
                          double t,
                          double observation,
                          double* y_out) {
    // Forward update: O(S²) but S is fixed → O(1)
    forward_update(solver->alpha, solver->transition_matrix,
                   observation, solver->state_space_size);
    
    if (solver->mode == 0 || solver->mode == 2) {
        // Probabilistic: compute posterior
        compute_posterior_step(solver->alpha, solver->beta,
                              solver->posterior, solver->state_space_size);
        
        // Compute statistics: O(S) but S fixed → O(1)
        compute_mean_variance(solver->posterior, solver->state_values,
                             solver->state_space_size,
                             solver->y_mean, solver->y_variance);
        
        if (y_out) {
            *y_out = *solver->y_mean;  // Return mean
        }
    }
    
    if (solver->mode == 1 || solver->mode == 2) {
        // Exact: Viterbi update
        viterbi_update(solver->viterbi, solver->backpointers,
                      solver->transition_matrix, observation,
                      solver->state_space_size);
        
        // Find MAP: O(S) but S fixed → O(1)
        size_t map_state = find_map_state(solver->viterbi,
                                         solver->state_space_size);
        *solver->y_map = solver->state_values[map_state];
        
        if (y_out && solver->mode == 1) {
            *y_out = *solver->y_map;  // Return MAP
        }
    }
    
    return 0;
}
```

## Applications

### 1. Uncertain Parameter ODEs

When ODE parameters θ are uncertain:

```c
// Prior: p(θ)
// Posterior: p(θ | observations)
// Solution: p(y(t) | θ, observations) marginalized over θ

double solve_with_uncertain_params(double t, 
                                   double* param_posterior,
                                   size_t num_param_samples) {
    double result = 0.0;
    for (size_t i = 0; i < num_param_samples; i++) {
        double theta = sample_from_posterior(param_posterior, i);
        double y_given_theta = solve_ode(t, theta);
        result += param_posterior[i] * y_given_theta;
    }
    return result;
}
```

### 2. Noisy Observations

When observations are noisy:

```c
// Observation model: p(observation | y(t))
// Likelihood: p(observation | y(t)) = N(observation; y(t), σ²)

double observation_likelihood(double observation,
                             double y_predicted,
                             double noise_variance) {
    double diff = observation - y_predicted;
    return exp(-0.5 * diff * diff / noise_variance);
}
```

### 3. Robust Control

Use full posterior for robust control:

```c
// Control action based on uncertainty
double compute_robust_control(RealTimeBayesianSolver* solver) {
    double mean = *solver->y_mean;
    double std = sqrt(*solver->y_variance);
    
    // Conservative control: account for uncertainty
    double safety_margin = 2.0 * std;  // 2-sigma
    return compute_control(mean - safety_margin);
}
```

## Performance Characteristics

| Method | Time per Step | Memory | Accuracy | Output |
|--------|---------------|--------|----------|--------|
| Forward-Backward | O(S²) ≈ O(1) | O(S) | High | Full posterior |
| Viterbi | O(S²) ≈ O(1) | O(S) | Exact (MAP) | Most likely path |
| Particle Filter | O(N) ≈ O(1) | O(N) | Approximate | Sample-based posterior |
| Hybrid | O(S²) ≈ O(1) | O(S) | Both | MAP + posterior |

Where:
- S = state space size (fixed constant)
- N = number of particles (fixed constant)

## Comparison with Traditional Methods

| Aspect | Traditional (RK3) | Bayesian DP |
|--------|------------------|-------------|
| **Output** | Single solution | Distribution + MAP |
| **Uncertainty** | None | Full quantification |
| **Parameters** | Fixed | Probabilistic |
| **Observations** | Not used | Integrated via likelihood |
| **Complexity** | O(n) per step | O(1) with pre-computation |
| **Real-time** | Yes | Yes (with pre-computation) |

## Future Extensions

1. **Nonlinear Systems**: Extended Kalman Filter (EKF) or Unscented Kalman Filter (UKF)
2. **Non-Gaussian**: Particle filters with adaptive resampling
3. **High Dimensions**: Reduced-order Bayesian models
4. **Online Learning**: Update transition models from data
5. **Multi-Modal**: Handle multiple solution modes

## References

- **Kalman Filtering**: Kalman (1960). "A New Approach to Linear Filtering and Prediction Problems"
- **Particle Filters**: Doucet et al. (2001). "Sequential Monte Carlo Methods in Practice"
- **Dynamic Programming**: Bellman (1957). "Dynamic Programming"
- **Bayesian State Estimation**: Särkkä (2013). "Bayesian Filtering and Smoothing"

## Conclusion

Real-time Bayesian ODE solvers with dynamic programming provide:
- ✅ **O(1) complexity** with pre-computation
- ✅ **Uncertainty quantification** via full posterior
- ✅ **Exact solutions** via MAP estimation
- ✅ **Robustness** to parameter uncertainty
- ✅ **Real-time capability** for hard deadlines

The choice between probabilistic and exact modes depends on the application:
- **Probabilistic**: When uncertainty matters (risk, robustness)
- **Exact**: When deterministic solution is required (control, optimization)
