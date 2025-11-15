/*
 * Test suite for DDMCMC implementation
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/ddmcmc.h"

// Test function: Rosenbrock function (common optimization benchmark)
double rosenbrock(const double* params, size_t n_params, void* context) {
    (void)context;
    if (n_params < 2) return 0.0;
    
    double sum = 0.0;
    for (size_t i = 0; i < n_params - 1; i++) {
        double a = params[i + 1] - params[i] * params[i];
        double b = 1.0 - params[i];
        sum += 100.0 * a * a + b * b;
    }
    return sum;
}

// Test function: Simple quadratic
double quadratic(const double* params, size_t n_params, void* context) {
    (void)context;
    double sum = 0.0;
    for (size_t i = 0; i < n_params; i++) {
        double diff = params[i] - 0.5;
        sum += diff * diff;
    }
    return sum;
}

int test_multinomial() {
    printf("Testing multinomial distribution...\n");
    
    double probs[3] = {0.5, 0.3, 0.2};
    MultinomialDist dist;
    
    if (multinomial_init(&dist, 3, probs) != 0) {
        printf("  FAIL: Initialization failed\n");
        return 1;
    }
    
    // Test sampling
    uint32_t rng_state = 12345;
    size_t counts[3] = {0, 0, 0};
    size_t n_samples = 10000;
    
    for (size_t i = 0; i < n_samples; i++) {
        size_t sample = multinomial_sample(&dist, &rng_state);
        if (sample < 3) {
            counts[sample]++;
        }
    }
    
    // Check approximate probabilities
    double tolerance = 0.05;
    int success = 1;
    for (size_t i = 0; i < 3; i++) {
        double empirical = (double)counts[i] / n_samples;
        double expected = dist.probabilities[i];
        if (fabs(empirical - expected) > tolerance) {
            printf("  Category %zu: expected %.3f, got %.3f\n", i, expected, empirical);
            success = 0;
        }
    }
    
    multinomial_free(&dist);
    printf("  %s\n\n", success ? "PASS" : "FAIL");
    return success ? 0 : 1;
}

int test_ddmcmc_optimization() {
    printf("Testing DDMCMC optimization...\n");
    
    double initial[2] = {0.0, 0.0};
    double optimal[2];
    double optimal_value;
    
    if (ddmcmc_optimize(quadratic, initial, 2, 10, 1000, NULL, optimal, &optimal_value) != 0) {
        printf("  FAIL: Optimization failed\n");
        return 1;
    }
    
    printf("  Optimal value: %.6f\n", optimal_value);
    printf("  Optimal params: [%.6f, %.6f]\n", optimal[0], optimal[1]);
    printf("  Expected: [0.5, 0.5], value ~0.0\n");
    
    // Check if we're close to the minimum
    double error = fabs(optimal[0] - 0.5) + fabs(optimal[1] - 0.5);
    int success = (error < 0.2 && optimal_value < 0.1);
    
    printf("  %s\n\n", success ? "PASS" : "FAIL");
    return success ? 0 : 1;
}

int test_hierarchical_optimization() {
    printf("Testing hierarchical DDMCMC optimization...\n");
    
    double initial[2] = {0.0, 0.0};
    double optimal[2];
    double optimal_value;
    
    if (ddmcmc_hierarchical_optimize(quadratic, initial, 2, 8, 3, 500, 
                                      NULL, optimal, &optimal_value) != 0) {
        printf("  FAIL: Hierarchical optimization failed\n");
        return 1;
    }
    
    printf("  Optimal value: %.6f\n", optimal_value);
    printf("  Optimal params: [%.6f, %.6f]\n", optimal[0], optimal[1]);
    
    double error = fabs(optimal[0] - 0.5) + fabs(optimal[1] - 0.5);
    int success = (error < 0.15 && optimal_value < 0.1);
    
    printf("  %s\n\n", success ? "PASS" : "FAIL");
    return success ? 0 : 1;
}

int main() {
    printf("=== DDMCMC Test Suite ===\n\n");
    
    int failures = 0;
    failures += test_multinomial();
    failures += test_ddmcmc_optimization();
    failures += test_hierarchical_optimization();
    
    printf("=== Test Summary ===\n");
    printf("Failures: %d\n", failures);
    
    return (failures == 0) ? 0 : 1;
}
