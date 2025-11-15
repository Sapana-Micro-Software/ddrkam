/*
 * Data-Driven Markov Chain Monte Carlo (DDMCMC) for Multinomial Variables
 * Efficient Search Algorithms for Learning Optimization Functions
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#ifndef DDMCMC_H
#define DDMCMC_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/**
 * Multinomial distribution structure
 */
typedef struct {
    size_t n_categories;      // Number of categories
    double* probabilities;     // Probability vector (sums to 1.0)
    double* log_probabilities; // Log probabilities for numerical stability
} MultinomialDist;

/**
 * DDMCMC sampler structure
 */
typedef struct {
    MultinomialDist* target_dist;  // Target multinomial distribution
    MultinomialDist* proposal_dist; // Proposal distribution
    double temperature;            // Temperature for annealing
    double learning_rate;           // Learning rate for adaptation
    size_t n_samples;               // Number of samples generated
    size_t n_accepted;              // Number of accepted proposals
    double* samples;                 // Generated samples
    double* log_likelihoods;        // Log likelihoods of samples
} DDMCMCSampler;

/**
 * Optimization function pointer type
 * @param params: Parameters to optimize
 * @param n_params: Number of parameters
 * @return: Objective function value (to minimize)
 */
typedef double (*OptimizationFunction)(const double* params, size_t n_params, void* context);

/**
 * Initialize multinomial distribution
 * 
 * @param dist: Distribution structure to initialize
 * @param n_categories: Number of categories
 * @param probabilities: Probability vector (will be normalized)
 * @return: 0 on success, -1 on failure
 */
int multinomial_init(MultinomialDist* dist, size_t n_categories, const double* probabilities);

/**
 * Free multinomial distribution resources
 */
void multinomial_free(MultinomialDist* dist);

/**
 * Sample from multinomial distribution
 * 
 * @param dist: Multinomial distribution
 * @param rng_state: Random number generator state (pass NULL for default)
 * @return: Sampled category index (0 to n_categories-1)
 */
size_t multinomial_sample(const MultinomialDist* dist, uint32_t* rng_state);

/**
 * Initialize DDMCMC sampler
 * 
 * @param sampler: Sampler structure to initialize
 * @param target_dist: Target multinomial distribution
 * @param temperature: Initial temperature (higher = more exploration)
 * @param learning_rate: Learning rate for adaptive proposals
 * @return: 0 on success, -1 on failure
 */
int ddmcmc_init(DDMCMCSampler* sampler, MultinomialDist* target_dist, 
                double temperature, double learning_rate);

/**
 * Free DDMCMC sampler resources
 */
void ddmcmc_free(DDMCMCSampler* sampler);

/**
 * Run MCMC sampling with adaptive proposals
 * 
 * @param sampler: DDMCMC sampler
 * @param n_iterations: Number of MCMC iterations
 * @param burn_in: Number of burn-in iterations
 * @return: Number of samples generated
 */
size_t ddmcmc_sample(DDMCMCSampler* sampler, size_t n_iterations, size_t burn_in);

/**
 * Optimize function using DDMCMC search
 * 
 * @param func: Optimization function to minimize
 * @param initial_params: Initial parameter values
 * @param n_params: Number of parameters
 * @param n_categories: Number of categories per parameter (for discretization)
 * @param n_iterations: Number of optimization iterations
 * @param context: User-defined context for the function
 * @param optimal_params: Output optimal parameters
 * @param optimal_value: Output optimal function value
 * @return: 0 on success, -1 on failure
 */
int ddmcmc_optimize(OptimizationFunction func, const double* initial_params,
                   size_t n_params, size_t n_categories, size_t n_iterations,
                   void* context, double* optimal_params, double* optimal_value);

/**
 * Hierarchical DDMCMC for high-dimensional optimization
 * 
 * @param func: Optimization function
 * @param initial_params: Initial parameters
 * @param n_params: Number of parameters
 * @param n_categories: Categories per parameter
 * @param n_layers: Number of hierarchical layers
 * @param n_iterations: Iterations per layer
 * @param context: Function context
 * @param optimal_params: Output optimal parameters
 * @param optimal_value: Output optimal value
 * @return: 0 on success, -1 on failure
 */
int ddmcmc_hierarchical_optimize(OptimizationFunction func, const double* initial_params,
                                 size_t n_params, size_t n_categories, size_t n_layers,
                                 size_t n_iterations, void* context,
                                 double* optimal_params, double* optimal_value);

/**
 * Multi-Bit-Flipping MCMC Sampler
 * Flips multiple bits/categories simultaneously for faster exploration
 */
typedef struct {
    MultinomialDist* target_dist;  // Target distribution
    MultinomialDist* proposal_dist; // Proposal distribution
    double temperature;            // Temperature for annealing
    double learning_rate;           // Learning rate for adaptation
    size_t n_bits;                 // Number of bits/categories to flip
    size_t max_flips;              // Maximum number of simultaneous flips
    double flip_probability;       // Probability of flipping each bit
    size_t n_samples;              // Number of samples generated
    size_t n_accepted;             // Number of accepted proposals
    double* samples;                // Generated samples
    double* log_likelihoods;        // Log likelihoods of samples
    size_t* current_state;         // Current state (array of category indices)
    size_t* proposed_state;        // Proposed state
} MultiBitFlipMCMC;

/**
 * Multinomial Multi-Bit-Flipping MCMC
 * Combines multinomial sampling with multi-bit-flipping
 */
typedef struct {
    MultinomialDist* target_dist;
    MultinomialDist** proposal_dists; // One per dimension
    double temperature;
    double learning_rate;
    size_t n_dimensions;           // Number of dimensions
    size_t n_categories_per_dim;   // Categories per dimension
    size_t max_flips;              // Max simultaneous flips
    double flip_probability;
    size_t n_samples;
    size_t n_accepted;
    double** samples;              // 2D array: samples[iteration][dimension]
    double* log_likelihoods;
    size_t* current_state;          // Current state per dimension
    size_t* proposed_state;         // Proposed state per dimension
} MultinomialMultiBitMCMC;

/**
 * Initialize multi-bit-flipping MCMC sampler
 * 
 * @param sampler: Sampler structure to initialize
 * @param target_dist: Target multinomial distribution
 * @param temperature: Initial temperature
 * @param learning_rate: Learning rate for adaptation
 * @param max_flips: Maximum number of bits to flip simultaneously
 * @param flip_probability: Probability of flipping each bit
 * @return: 0 on success, -1 on failure
 */
int multibit_flip_mcmc_init(MultiBitFlipMCMC* sampler, MultinomialDist* target_dist,
                            double temperature, double learning_rate,
                            size_t max_flips, double flip_probability);

/**
 * Free multi-bit-flipping MCMC sampler resources
 */
void multibit_flip_mcmc_free(MultiBitFlipMCMC* sampler);

/**
 * Run multi-bit-flipping MCMC sampling
 * 
 * @param sampler: Multi-bit-flipping MCMC sampler
 * @param n_iterations: Number of MCMC iterations
 * @param burn_in: Number of burn-in iterations
 * @return: Number of samples generated
 */
size_t multibit_flip_mcmc_sample(MultiBitFlipMCMC* sampler, size_t n_iterations, size_t burn_in);

/**
 * Initialize multinomial multi-bit-flipping MCMC
 * 
 * @param sampler: Sampler structure to initialize
 * @param n_dimensions: Number of dimensions
 * @param n_categories_per_dim: Number of categories per dimension
 * @param temperature: Initial temperature
 * @param learning_rate: Learning rate
 * @param max_flips: Maximum simultaneous flips
 * @param flip_probability: Probability of flipping each dimension
 * @return: 0 on success, -1 on failure
 */
int multinomial_multibit_mcmc_init(MultinomialMultiBitMCMC* sampler,
                                   size_t n_dimensions, size_t n_categories_per_dim,
                                   double temperature, double learning_rate,
                                   size_t max_flips, double flip_probability);

/**
 * Free multinomial multi-bit-flipping MCMC resources
 */
void multinomial_multibit_mcmc_free(MultinomialMultiBitMCMC* sampler);

/**
 * Set target distribution for multinomial multi-bit MCMC
 * 
 * @param sampler: Sampler
 * @param target_dist: Target distribution (will be copied per dimension)
 * @return: 0 on success, -1 on failure
 */
int multinomial_multibit_mcmc_set_target(MultinomialMultiBitMCMC* sampler,
                                         const MultinomialDist* target_dist);

/**
 * Run multinomial multi-bit-flipping MCMC sampling
 * 
 * @param sampler: Multinomial multi-bit MCMC sampler
 * @param n_iterations: Number of iterations
 * @param burn_in: Burn-in period
 * @return: Number of samples generated
 */
size_t multinomial_multibit_mcmc_sample(MultinomialMultiBitMCMC* sampler,
                                       size_t n_iterations, size_t burn_in);

/**
 * Optimize using multinomial multi-bit-flipping MCMC
 * 
 * @param func: Optimization function
 * @param initial_params: Initial parameters
 * @param n_params: Number of parameters
 * @param n_categories: Categories per parameter
 * @param n_iterations: Number of iterations
 * @param max_flips: Maximum simultaneous flips
 * @param context: Function context
 * @param optimal_params: Output optimal parameters
 * @param optimal_value: Output optimal value
 * @return: 0 on success, -1 on failure
 */
int multinomial_multibit_mcmc_optimize(OptimizationFunction func,
                                      const double* initial_params,
                                      size_t n_params, size_t n_categories,
                                      size_t n_iterations, size_t max_flips,
                                      void* context,
                                      double* optimal_params, double* optimal_value);

#ifdef __cplusplus
}
#endif

#endif /* DDMCMC_H */
