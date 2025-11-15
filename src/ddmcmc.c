/*
 * Data-Driven MCMC Implementation for Multinomial Variables
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#include "ddmcmc.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

// Simple RNG (Xorshift)
static uint32_t xorshift32(uint32_t* state) {
    if (state == NULL) {
        static uint32_t default_state = 12345;
        state = &default_state;
    }
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static double uniform_random(uint32_t* state) {
    return (double)xorshift32(state) / UINT32_MAX;
}

int multinomial_init(MultinomialDist* dist, size_t n_categories, const double* probabilities) {
    if (!dist || n_categories == 0 || !probabilities) {
        return -1;
    }
    
    dist->n_categories = n_categories;
    dist->probabilities = (double*)malloc(n_categories * sizeof(double));
    dist->log_probabilities = (double*)malloc(n_categories * sizeof(double));
    
    if (!dist->probabilities || !dist->log_probabilities) {
        multinomial_free(dist);
        return -1;
    }
    
    // Normalize probabilities
    double sum = 0.0;
    for (size_t i = 0; i < n_categories; i++) {
        if (probabilities[i] < 0.0) {
            multinomial_free(dist);
            return -1;
        }
        sum += probabilities[i];
    }
    
    if (sum == 0.0) {
        // Uniform distribution
        double uniform = 1.0 / n_categories;
        for (size_t i = 0; i < n_categories; i++) {
            dist->probabilities[i] = uniform;
            dist->log_probabilities[i] = log(uniform);
        }
    } else {
        for (size_t i = 0; i < n_categories; i++) {
            dist->probabilities[i] = probabilities[i] / sum;
            dist->log_probabilities[i] = log(dist->probabilities[i]);
        }
    }
    
    return 0;
}

void multinomial_free(MultinomialDist* dist) {
    if (!dist) return;
    
    if (dist->probabilities) {
        free(dist->probabilities);
        dist->probabilities = NULL;
    }
    if (dist->log_probabilities) {
        free(dist->log_probabilities);
        dist->log_probabilities = NULL;
    }
}

size_t multinomial_sample(const MultinomialDist* dist, uint32_t* rng_state) {
    if (!dist || !dist->probabilities) {
        return 0;
    }
    
    double u = uniform_random(rng_state);
    double cumulative = 0.0;
    
    for (size_t i = 0; i < dist->n_categories; i++) {
        cumulative += dist->probabilities[i];
        if (u <= cumulative) {
            return i;
        }
    }
    
    return dist->n_categories - 1; // Fallback
}

int ddmcmc_init(DDMCMCSampler* sampler, MultinomialDist* target_dist, 
                double temperature, double learning_rate) {
    if (!sampler || !target_dist) {
        return -1;
    }
    
    sampler->target_dist = target_dist;
    sampler->temperature = temperature > 0.0 ? temperature : 1.0;
    sampler->learning_rate = learning_rate > 0.0 ? learning_rate : 0.01;
    sampler->n_samples = 0;
    sampler->n_accepted = 0;
    sampler->samples = NULL;
    sampler->log_likelihoods = NULL;
    
    // Initialize proposal distribution as uniform
    sampler->proposal_dist = (MultinomialDist*)malloc(sizeof(MultinomialDist));
    if (!sampler->proposal_dist) {
        return -1;
    }
    
    double* uniform_probs = (double*)malloc(target_dist->n_categories * sizeof(double));
    if (!uniform_probs) {
        free(sampler->proposal_dist);
        return -1;
    }
    
    double uniform = 1.0 / target_dist->n_categories;
    for (size_t i = 0; i < target_dist->n_categories; i++) {
        uniform_probs[i] = uniform;
    }
    
    if (multinomial_init(sampler->proposal_dist, target_dist->n_categories, uniform_probs) != 0) {
        free(uniform_probs);
        free(sampler->proposal_dist);
        return -1;
    }
    
    free(uniform_probs);
    return 0;
}

void ddmcmc_free(DDMCMCSampler* sampler) {
    if (!sampler) return;
    
    if (sampler->proposal_dist) {
        multinomial_free(sampler->proposal_dist);
        free(sampler->proposal_dist);
        sampler->proposal_dist = NULL;
    }
    
    if (sampler->samples) {
        free(sampler->samples);
        sampler->samples = NULL;
    }
    
    if (sampler->log_likelihoods) {
        free(sampler->log_likelihoods);
        sampler->log_likelihoods = NULL;
    }
}

static double compute_acceptance_probability(const DDMCMCSampler* sampler, 
                                            size_t current, size_t proposed,
                                            uint32_t* rng_state) {
    (void)rng_state; // Suppress unused parameter warning
    double log_target_current = sampler->target_dist->log_probabilities[current];
    double log_target_proposed = sampler->target_dist->log_probabilities[proposed];
    
    double log_proposal_current = sampler->proposal_dist->log_probabilities[current];
    double log_proposal_proposed = sampler->proposal_dist->log_probabilities[proposed];
    
    // Metropolis-Hastings acceptance probability
    double log_alpha = (log_target_proposed - log_target_current) / sampler->temperature +
                       (log_proposal_current - log_proposal_proposed);
    
    return log_alpha > 0.0 ? 1.0 : exp(log_alpha);
}

static void adapt_proposal(DDMCMCSampler* sampler, size_t accepted_category) {
    // Adaptive proposal: increase probability of accepted category
    double lr = sampler->learning_rate;
    
    for (size_t i = 0; i < sampler->proposal_dist->n_categories; i++) {
        if (i == accepted_category) {
            sampler->proposal_dist->probabilities[i] += lr * (1.0 - sampler->proposal_dist->probabilities[i]);
        } else {
            sampler->proposal_dist->probabilities[i] *= (1.0 - lr);
        }
    }
    
    // Renormalize
    double sum = 0.0;
    for (size_t i = 0; i < sampler->proposal_dist->n_categories; i++) {
        sum += sampler->proposal_dist->probabilities[i];
    }
    
    for (size_t i = 0; i < sampler->proposal_dist->n_categories; i++) {
        sampler->proposal_dist->probabilities[i] /= sum;
        sampler->proposal_dist->log_probabilities[i] = log(sampler->proposal_dist->probabilities[i]);
    }
}

size_t ddmcmc_sample(DDMCMCSampler* sampler, size_t n_iterations, size_t burn_in) {
    if (!sampler || n_iterations == 0) {
        return 0;
    }
    
    size_t n_samples = n_iterations - burn_in;
    if (n_samples == 0) {
        return 0;
    }
    
    sampler->samples = (double*)realloc(sampler->samples, n_samples * sizeof(double));
    sampler->log_likelihoods = (double*)realloc(sampler->log_likelihoods, n_samples * sizeof(double));
    
    if (!sampler->samples || !sampler->log_likelihoods) {
        return 0;
    }
    
    uint32_t rng_state = 12345;
    size_t current = multinomial_sample(sampler->target_dist, &rng_state);
    size_t sample_idx = 0;
    
    for (size_t iter = 0; iter < n_iterations; iter++) {
        // Propose new state
        size_t proposed = multinomial_sample(sampler->proposal_dist, &rng_state);
        
        // Compute acceptance probability
        double alpha = compute_acceptance_probability(sampler, current, proposed, &rng_state);
        
        // Accept or reject
        if (uniform_random(&rng_state) < alpha) {
            current = proposed;
            sampler->n_accepted++;
            
            // Adapt proposal distribution
            adapt_proposal(sampler, current);
        }
        
        // Store sample after burn-in
        if (iter >= burn_in) {
            sampler->samples[sample_idx] = (double)current;
            sampler->log_likelihoods[sample_idx] = sampler->target_dist->log_probabilities[current];
            sample_idx++;
        }
        
        // Temperature annealing
        sampler->temperature *= 0.9999;
        if (sampler->temperature < 0.1) {
            sampler->temperature = 0.1;
        }
    }
    
    sampler->n_samples = n_samples;
    return n_samples;
}

int ddmcmc_optimize(OptimizationFunction func, const double* initial_params,
                   size_t n_params, size_t n_categories, size_t n_iterations,
                   void* context, double* optimal_params, double* optimal_value) {
    if (!func || !initial_params || !optimal_params || !optimal_value || 
        n_params == 0 || n_categories == 0) {
        return -1;
    }
    
    // Discretize parameter space
    size_t total_categories = 1;
    for (size_t i = 0; i < n_params; i++) {
        total_categories *= n_categories;
    }
    
    // Initialize uniform target distribution
    double* uniform_probs = (double*)calloc(total_categories, sizeof(double));
    if (!uniform_probs) {
        return -1;
    }
    
    double uniform = 1.0 / total_categories;
    for (size_t i = 0; i < total_categories; i++) {
        uniform_probs[i] = uniform;
    }
    
    MultinomialDist target_dist;
    if (multinomial_init(&target_dist, total_categories, uniform_probs) != 0) {
        free(uniform_probs);
        return -1;
    }
    
    // Evaluate function at all discretized points and update target distribution
    double min_value = DBL_MAX;
    
    for (size_t idx = 0; idx < total_categories; idx++) {
        // Decode index to parameter values
        double* params = (double*)malloc(n_params * sizeof(double));
        if (!params) {
            multinomial_free(&target_dist);
            free(uniform_probs);
            return -1;
        }
        
        size_t temp_idx = idx;
        for (size_t i = 0; i < n_params; i++) {
            size_t cat = temp_idx % n_categories;
            temp_idx /= n_categories;
            // Map category to parameter value (simplified: assume range [0, 1])
            params[i] = (double)cat / (n_categories - 1);
        }
        
        double value = func(params, n_params, context);
        
        // Use negative value as log probability (minimize function = maximize probability)
        target_dist.log_probabilities[idx] = -value;
        target_dist.probabilities[idx] = exp(-value);
        
        if (value < min_value) {
            min_value = value;
        }
        
        free(params);
    }
    
    // Normalize target distribution
    double sum = 0.0;
    for (size_t i = 0; i < total_categories; i++) {
        sum += target_dist.probabilities[i];
    }
    for (size_t i = 0; i < total_categories; i++) {
        target_dist.probabilities[i] /= sum;
        target_dist.log_probabilities[i] = log(target_dist.probabilities[i]);
    }
    
    // Run MCMC
    DDMCMCSampler sampler;
    if (ddmcmc_init(&sampler, &target_dist, 1.0, 0.01) != 0) {
        multinomial_free(&target_dist);
        free(uniform_probs);
        return -1;
    }
    
    size_t burn_in = n_iterations / 10;
    ddmcmc_sample(&sampler, n_iterations, burn_in);
    
    // Find optimal from samples
    double best_value = DBL_MAX;
    
    for (size_t i = 0; i < sampler.n_samples; i++) {
        size_t idx = (size_t)sampler.samples[i];
        if (idx < total_categories) {
            // Decode and evaluate
            double* params = (double*)malloc(n_params * sizeof(double));
            if (params) {
                size_t temp_idx = idx;
                for (size_t j = 0; j < n_params; j++) {
                    size_t cat = temp_idx % n_categories;
                    temp_idx /= n_categories;
                    params[j] = (double)cat / (n_categories - 1);
                }
                
                double value = func(params, n_params, context);
                if (value < best_value) {
                    best_value = value;
                    memcpy(optimal_params, params, n_params * sizeof(double));
                }
                free(params);
            }
        }
    }
    
    *optimal_value = best_value;
    
    ddmcmc_free(&sampler);
    multinomial_free(&target_dist);
    free(uniform_probs);
    
    return 0;
}

int ddmcmc_hierarchical_optimize(OptimizationFunction func, const double* initial_params,
                                 size_t n_params, size_t n_categories, size_t n_layers,
                                 size_t n_iterations, void* context,
                                 double* optimal_params, double* optimal_value) {
    if (!func || !initial_params || !optimal_params || !optimal_value ||
        n_params == 0 || n_categories == 0 || n_layers == 0) {
        return -1;
    }
    
    double* current_params = (double*)malloc(n_params * sizeof(double));
    if (!current_params) {
        return -1;
    }
    memcpy(current_params, initial_params, n_params * sizeof(double));
    
    double current_value = func(current_params, n_params, context);
    
    // Hierarchical refinement: start coarse, refine at each layer
    for (size_t layer = 0; layer < n_layers; layer++) {
        // Reduce search space around current best
        double* layer_optimal = (double*)malloc(n_params * sizeof(double));
        double layer_optimal_value;
        
        // Create local optimization function
        OptimizationFunction local_func = func; // Simplified: use same function
        
        if (ddmcmc_optimize(local_func, current_params, n_params, n_categories,
                           n_iterations, context, layer_optimal, &layer_optimal_value) == 0) {
            if (layer_optimal_value < current_value) {
                current_value = layer_optimal_value;
                memcpy(current_params, layer_optimal, n_params * sizeof(double));
            }
        }
        
        free(layer_optimal);
    }
    
    memcpy(optimal_params, current_params, n_params * sizeof(double));
    *optimal_value = current_value;
    
    free(current_params);
    return 0;
}
