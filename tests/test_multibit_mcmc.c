/*
 * Test Suite for Multinomial Multi-Bit-Flipping MCMC
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/ddmcmc.h"

// Test optimization function: minimize sum of squares
double test_function(const double* params, size_t n_params, void* context) {
    (void)context;
    double sum = 0.0;
    for (size_t i = 0; i < n_params; i++) {
        double diff = params[i] - 0.5; // Optimal at 0.5
        sum += diff * diff;
    }
    return sum;
}

int test_multibit_flip_mcmc() {
    printf("=== Test 1: Multi-Bit-Flipping MCMC ===\n");
    
    // Create target distribution
    MultinomialDist target_dist;
    size_t n_categories = 10;
    double* probs = (double*)malloc(n_categories * sizeof(double));
    if (!probs) {
        printf("  FAIL: Memory allocation failed\n");
        return 1;
    }
    
    // Create a distribution with peak at category 5
    for (size_t i = 0; i < n_categories; i++) {
        double diff = (double)i - 5.0;
        probs[i] = exp(-diff * diff / 2.0); // Gaussian-like
    }
    
    if (multinomial_init(&target_dist, n_categories, probs) != 0) {
        free(probs);
        printf("  FAIL: Distribution initialization failed\n");
        return 1;
    }
    
    // Initialize multi-bit-flipping MCMC
    MultiBitFlipMCMC sampler;
    if (multibit_flip_mcmc_init(&sampler, &target_dist, 1.0, 0.01, 3, 0.1) != 0) {
        multinomial_free(&target_dist);
        free(probs);
        printf("  FAIL: Sampler initialization failed\n");
        return 1;
    }
    
    // Run sampling
    size_t n_iterations = 1000;
    size_t burn_in = 100;
    size_t n_samples = multibit_flip_mcmc_sample(&sampler, n_iterations, burn_in);
    
    printf("  Samples generated: %zu\n", n_samples);
    printf("  Acceptance rate: %.2f%%\n", 
           (double)sampler.n_accepted / n_iterations * 100.0);
    
    if (n_samples > 0) {
        printf("  First sample: %.2f\n", sampler.samples[0]);
        printf("  Last sample: %.2f\n", sampler.samples[n_samples - 1]);
        printf("  %s\n\n", (n_samples > 0) ? "PASS" : "FAIL");
    } else {
        printf("  FAIL: No samples generated\n\n");
    }
    
    multibit_flip_mcmc_free(&sampler);
    multinomial_free(&target_dist);
    free(probs);
    
    return (n_samples > 0) ? 0 : 1;
}

int test_multinomial_multibit_mcmc() {
    printf("=== Test 2: Multinomial Multi-Bit-Flipping MCMC ===\n");
    
    MultinomialMultiBitMCMC sampler;
    size_t n_dimensions = 2;
    size_t n_categories_per_dim = 10;
    
    if (multinomial_multibit_mcmc_init(&sampler, n_dimensions, n_categories_per_dim,
                                      1.0, 0.01, 2, 0.1) != 0) {
        printf("  FAIL: Initialization failed\n");
        return 1;
    }
    
    // Create target distribution
    MultinomialDist target_dist;
    double* probs = (double*)malloc(n_categories_per_dim * sizeof(double));
    if (!probs) {
        multinomial_multibit_mcmc_free(&sampler);
        printf("  FAIL: Memory allocation failed\n");
        return 1;
    }
    
    // Gaussian-like distribution
    for (size_t i = 0; i < n_categories_per_dim; i++) {
        double diff = (double)i - 5.0;
        probs[i] = exp(-diff * diff / 2.0);
    }
    
    if (multinomial_init(&target_dist, n_categories_per_dim, probs) != 0) {
        free(probs);
        multinomial_multibit_mcmc_free(&sampler);
        printf("  FAIL: Target distribution initialization failed\n");
        return 1;
    }
    
    if (multinomial_multibit_mcmc_set_target(&sampler, &target_dist) != 0) {
        multinomial_free(&target_dist);
        free(probs);
        multinomial_multibit_mcmc_free(&sampler);
        printf("  FAIL: Setting target distribution failed\n");
        return 1;
    }
    
    // Run sampling
    size_t n_iterations = 1000;
    size_t burn_in = 100;
    size_t n_samples = multinomial_multibit_mcmc_sample(&sampler, n_iterations, burn_in);
    
    printf("  Samples generated: %zu\n", n_samples);
    printf("  Acceptance rate: %.2f%%\n",
           (double)sampler.n_accepted / n_iterations * 100.0);
    
    if (n_samples > 0) {
        printf("  First sample: [%.2f, %.2f]\n",
               sampler.samples[0][0], sampler.samples[0][1]);
        printf("  Last sample: [%.2f, %.2f]\n",
               sampler.samples[n_samples - 1][0],
               sampler.samples[n_samples - 1][1]);
        printf("  %s\n\n", (n_samples > 0) ? "PASS" : "FAIL");
    } else {
        printf("  FAIL: No samples generated\n\n");
    }
    
    multinomial_free(&target_dist);
    free(probs);
    multinomial_multibit_mcmc_free(&sampler);
    
    return (n_samples > 0) ? 0 : 1;
}

int test_multinomial_multibit_optimize() {
    printf("=== Test 3: Multinomial Multi-Bit-Flipping Optimization ===\n");
    
    double initial[2] = {0.0, 0.0};
    double optimal[2];
    double optimal_value;
    
    if (multinomial_multibit_mcmc_optimize(test_function, initial, 2, 10,
                                          1000, 2, NULL, optimal, &optimal_value) != 0) {
        printf("  FAIL: Optimization failed\n");
        return 1;
    }
    
    printf("  Optimal params: [%.6f, %.6f]\n", optimal[0], optimal[1]);
    printf("  Optimal value: %.6f\n", optimal_value);
    printf("  Expected: [0.5, 0.5], value ~0.0\n");
    
    double error = fabs(optimal[0] - 0.5) + fabs(optimal[1] - 0.5);
    printf("  Error: %.6f\n", error);
    printf("  %s\n\n", (error < 0.5) ? "PASS" : "FAIL");
    
    return (error < 0.5) ? 0 : 1;
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║     Multinomial Multi-Bit-Flipping MCMC Test Suite         ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    int failures = 0;
    failures += test_multibit_flip_mcmc();
    failures += test_multinomial_multibit_mcmc();
    failures += test_multinomial_multibit_optimize();
    
    printf("=== Test Summary ===\n");
    printf("Failures: %d\n", failures);
    
    if (failures == 0) {
        printf("✅ All Multinomial Multi-Bit-Flipping MCMC tests PASSED\n");
    } else {
        printf("❌ Some tests FAILED\n");
    }
    
    return (failures == 0) ? 0 : 1;
}
