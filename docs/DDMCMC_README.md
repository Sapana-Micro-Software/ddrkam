# DDMCMC - Data-Driven Markov Chain Monte Carlo

## Overview

DDMCMC (Data-Driven Markov Chain Monte Carlo) is an extension to DDRKAM that provides efficient search algorithms for learning optimization functions using multinomial variables. This implementation enables adaptive MCMC sampling with hierarchical refinement for high-dimensional optimization problems.

## Features

- **Multinomial Distribution Support**: Efficient sampling from multinomial distributions
- **Adaptive MCMC**: Proposal distribution adapts based on accepted samples
- **Temperature Annealing**: Automatic temperature scheduling for better convergence
- **Hierarchical Optimization**: Multi-layer refinement for complex optimization landscapes
- **Efficient Search**: Optimized for learning optimization functions

## API Usage

### Basic Optimization

```c
#include "ddmcmc.h"

double my_function(const double* params, size_t n_params, void* context) {
    // Your optimization function
    double sum = 0.0;
    for (size_t i = 0; i < n_params; i++) {
        sum += params[i] * params[i];
    }
    return sum;
}

int main() {
    double initial[2] = {0.0, 0.0};
    double optimal[2];
    double optimal_value;
    
    ddmcmc_optimize(my_function, initial, 2, 10, 1000, 
                    NULL, optimal, &optimal_value);
    
    printf("Optimal value: %f\n", optimal_value);
    return 0;
}
```

### Hierarchical Optimization

For high-dimensional or complex optimization problems:

```c
ddmcmc_hierarchical_optimize(my_function, initial, n_params, 
                             n_categories, n_layers, n_iterations,
                             context, optimal, &optimal_value);
```

## Algorithm

1. **Discretization**: Parameter space is discretized into multinomial categories
2. **Target Distribution**: Optimization function values mapped to probabilities
3. **MCMC Sampling**: Metropolis-Hastings algorithm with adaptive proposals
4. **Refinement**: Hierarchical layers progressively refine the search space

## Performance

- Efficient for discrete and discretized continuous optimization
- Adaptive proposals reduce burn-in time
- Hierarchical approach scales to high dimensions
- Temperature annealing improves convergence

## Copyright

Copyright (C) 2025, Shyamal Suhana Chandra
