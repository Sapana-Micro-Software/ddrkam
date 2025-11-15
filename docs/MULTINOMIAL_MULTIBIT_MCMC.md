# Multinomial Multi-Bit-Flipping MCMC

## Overview

This document describes the implementation of Multinomial Multi-Bit-Flipping MCMC (Markov Chain Monte Carlo) methods for efficient sampling and optimization. These methods extend the standard MCMC approach by allowing multiple bits/categories to be flipped simultaneously, enabling faster exploration of the state space.

## Features

### 1. **Multi-Bit-Flipping MCMC**
- Simultaneous flipping of multiple bits/categories
- Configurable maximum number of simultaneous flips
- Adaptive proposal distribution
- Temperature annealing for better convergence

### 2. **Multinomial Multi-Bit-Flipping MCMC**
- Multi-dimensional multinomial sampling
- Independent proposal distributions per dimension
- Simultaneous flips across multiple dimensions
- Efficient high-dimensional optimization

### 3. **Optimization Integration**
- Direct optimization function interface
- Automatic discretization of parameter space
- Target distribution construction from objective function
- Optimal parameter extraction from samples

## API Reference

### Multi-Bit-Flipping MCMC

```c
// Initialize sampler
MultiBitFlipMCMC sampler;
MultinomialDist target_dist;
multinomial_init(&target_dist, n_categories, probabilities);

multibit_flip_mcmc_init(&sampler, &target_dist, 
                        temperature, learning_rate,
                        max_flips, flip_probability);

// Run sampling
size_t n_samples = multibit_flip_mcmc_sample(&sampler, 
                                             n_iterations, burn_in);

// Cleanup
multibit_flip_mcmc_free(&sampler);
multinomial_free(&target_dist);
```

### Multinomial Multi-Bit-Flipping MCMC

```c
// Initialize sampler
MultinomialMultiBitMCMC sampler;
multinomial_multibit_mcmc_init(&sampler,
                               n_dimensions, n_categories_per_dim,
                               temperature, learning_rate,
                               max_flips, flip_probability);

// Set target distribution
MultinomialDist target_dist;
multinomial_init(&target_dist, n_categories, probabilities);
multinomial_multibit_mcmc_set_target(&sampler, &target_dist);

// Run sampling
size_t n_samples = multinomial_multibit_mcmc_sample(&sampler,
                                                    n_iterations, burn_in);

// Access samples
for (size_t i = 0; i < n_samples; i++) {
    double* sample = sampler.samples[i];
    // Use sample[i] for dimension i
}

// Cleanup
multinomial_multibit_mcmc_free(&sampler);
multinomial_free(&target_dist);
```

### Optimization Interface

```c
// Optimization function
double my_function(const double* params, size_t n_params, void* context) {
    double sum = 0.0;
    for (size_t i = 0; i < n_params; i++) {
        sum += params[i] * params[i];
    }
    return sum;
}

// Optimize
double initial[2] = {0.0, 0.0};
double optimal[2];
double optimal_value;

multinomial_multibit_mcmc_optimize(my_function, initial, 2, 10,
                                  1000, 2, NULL, optimal, &optimal_value);
```

## Algorithm Details

### Multi-Bit-Flipping Proposal

1. **Select Number of Flips**: Randomly choose 1 to `max_flips` bits to flip
2. **Select Indices**: Randomly select distinct indices to flip
3. **Flip Bits**: For each selected index, sample new category from proposal distribution
4. **Accept/Reject**: Use Metropolis-Hastings acceptance probability

### Acceptance Probability

For multi-bit flip from state `x` to `x'`:

```
α = min(1, exp((log π(x') - log π(x)) / T))
```

where:
- `π(x)` is the target distribution
- `T` is the temperature parameter

### Temperature Annealing

Temperature decreases over iterations:
```
T_{t+1} = T_t * 0.9999
```

Minimum temperature clamped to 0.1 to maintain exploration.

### Multinomial Multi-Bit-Flipping

For multi-dimensional problems:
- Each dimension has its own proposal distribution
- Multiple dimensions can be flipped simultaneously
- Acceptance probability computed across all dimensions
- Independent sampling per dimension

## Parameters

### MultiBitFlipMCMC

- `max_flips`: Maximum number of bits to flip simultaneously (default: 3)
- `flip_probability`: Probability of flipping each bit (default: 0.1)
- `temperature`: Initial temperature for annealing (default: 1.0)
- `learning_rate`: Learning rate for proposal adaptation (default: 0.01)

### MultinomialMultiBitMCMC

- `n_dimensions`: Number of dimensions
- `n_categories_per_dim`: Number of categories per dimension
- `max_flips`: Maximum simultaneous dimension flips (default: 2)
- `flip_probability`: Probability of flipping each dimension (default: 0.1)
- `temperature`: Initial temperature (default: 1.0)
- `learning_rate`: Learning rate (default: 0.01)

## Advantages

1. **Faster Exploration**: Multiple simultaneous flips explore state space faster
2. **Better Mixing**: Reduces correlation between samples
3. **High-Dimensional**: Efficient for multi-dimensional problems
4. **Adaptive**: Proposal distributions adapt based on accepted samples
5. **Flexible**: Configurable flip counts and probabilities

## Use Cases

1. **Discrete Optimization**: Binary/discrete parameter optimization
2. **Sampling**: Efficient sampling from multinomial distributions
3. **Bayesian Inference**: Posterior sampling for discrete models
4. **Combinatorial Optimization**: Traveling salesman, graph problems
5. **Feature Selection**: Selecting subsets of features

## Example: Optimization

```c
#include "ddmcmc.h"

double sphere_function(const double* params, size_t n_params, void* context) {
    double sum = 0.0;
    for (size_t i = 0; i < n_params; i++) {
        sum += params[i] * params[i];
    }
    return sum;
}

int main() {
    double initial[3] = {0.0, 0.0, 0.0};
    double optimal[3];
    double optimal_value;
    
    // Optimize with multi-bit-flipping MCMC
    multinomial_multibit_mcmc_optimize(sphere_function, initial, 3, 20,
                                      5000, 3, NULL, optimal, &optimal_value);
    
    printf("Optimal: [%.4f, %.4f, %.4f], value: %.6f\n",
           optimal[0], optimal[1], optimal[2], optimal_value);
    
    return 0;
}
```

## Performance Considerations

- **Max Flips**: Higher values explore faster but may reduce acceptance rate
- **Temperature**: Start high, anneal gradually
- **Burn-in**: Use 10-20% of iterations for burn-in
- **Iterations**: More iterations = better convergence
- **Categories**: More categories = finer discretization but slower

## Comparison with Standard MCMC

| Feature | Standard MCMC | Multi-Bit-Flipping MCMC |
|---------|---------------|------------------------|
| Flips per step | 1 | 1 to max_flips |
| Exploration speed | Slower | Faster |
| Acceptance rate | Higher | Lower (but compensated by speed) |
| Mixing | Good | Better |
| High-dimensional | Moderate | Excellent |

## References

- Metropolis, N., et al. (1953). "Equation of State Calculations by Fast Computing Machines"
- Hastings, W. K. (1970). "Monte Carlo Sampling Methods Using Markov Chains"
- Geyer, C. J. (2011). "Introduction to MCMC"

## Copyright

Copyright (C) 2025, Shyamal Suhana Chandra
