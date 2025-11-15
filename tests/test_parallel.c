/*
 * Test suite for Parallel, Distributed, Concurrent, Hierarchical, and Stacked Methods
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "../include/parallel_rk.h"
#include "../include/parallel_adams.h"
#include "../include/parallel_euler.h"

// Test ODE: dy/dt = -y
void test_exponential(double t, const double* y, double* dydt, void* params) {
    (void)t;
    (void)params;
    dydt[0] = -y[0];
}

int test_parallel_rk3() {
    printf("=== Parallel RK3 Test ===\n");
    
    ParallelRKSolver solver;
    if (parallel_rk_init(&solver, 1, 4, PARALLEL_OPENMP, NULL) != 0) {
        printf("  FAIL: Initialization failed\n");
        return 1;
    }
    
    double t0 = 0.0;
    double t_end = 2.0;
    double y0[1] = {1.0};
    double h = 0.01;
    
    size_t max_steps = (size_t)((t_end - t0) / h) + 1;
    double* t_out = (double*)malloc(max_steps * sizeof(double));
    double* y_out = (double*)malloc(max_steps * sizeof(double));
    
    clock_t start = clock();
    size_t num_steps = parallel_rk_solve(&solver, test_exponential, t0, t_end, y0,
                                       h, NULL, t_out, y_out);
    clock_t end = clock();
    double elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    double y_final = y_out[num_steps - 1];
    double y_exact = exp(-t_end);
    double error = fabs(y_final - y_exact);
    double accuracy = (1.0 - error / fabs(y_exact)) * 100.0;
    
    printf("  Steps: %zu\n", num_steps);
    printf("  Time: %.6f seconds\n", elapsed);
    printf("  Final value: %.10f\n", y_final);
    printf("  Exact value: %.10f\n", y_exact);
    printf("  Error: %.6e\n", error);
    printf("  Accuracy: %.6f%%\n", accuracy);
    printf("  %s\n\n", (error < 1e-3) ? "PASS" : "FAIL");
    
    parallel_rk_free(&solver);
    free(t_out);
    free(y_out);
    return (error < 1e-3) ? 0 : 1;
}

int test_stacked_rk3() {
    printf("=== Stacked RK3 Test ===\n");
    
    StackedConfig stacked = {
        .num_layers = 3,
        .layer_dims = NULL,
        .hidden_dim = 32,
        .learning_rate = 0.01,
        .use_attention = 1,
        .use_residual = 1
    };
    
    ParallelRKSolver solver;
    if (parallel_rk_init(&solver, 1, 4, PARALLEL_OPENMP, &stacked) != 0) {
        printf("  FAIL: Initialization failed\n");
        return 1;
    }
    
    double t0 = 0.0;
    double t_end = 2.0;
    double y0[1] = {1.0};
    double h = 0.01;
    
    size_t max_steps = (size_t)((t_end - t0) / h) + 1;
    double* t_out = (double*)malloc(max_steps * sizeof(double));
    double* y_out = (double*)malloc(max_steps * sizeof(double));
    
    clock_t start = clock();
    size_t num_steps = parallel_rk_solve(&solver, test_exponential, t0, t_end, y0,
                                       h, NULL, t_out, y_out);
    clock_t end = clock();
    double elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    double y_final = y_out[num_steps - 1];
    double y_exact = exp(-t_end);
    double error = fabs(y_final - y_exact);
    double accuracy = (1.0 - error / fabs(y_exact)) * 100.0;
    
    printf("  Steps: %zu\n", num_steps);
    printf("  Time: %.6f seconds\n", elapsed);
    printf("  Final value: %.10f\n", y_final);
    printf("  Exact value: %.10f\n", y_exact);
    printf("  Error: %.6e\n", error);
    printf("  Accuracy: %.6f%%\n", accuracy);
    printf("  %s\n\n", (error < 1e-3) ? "PASS" : "FAIL");
    
    parallel_rk_free(&solver);
    free(t_out);
    free(y_out);
    return (error < 1e-3) ? 0 : 1;
}

int test_parallel_euler() {
    printf("=== Parallel Euler Test ===\n");
    
    ParallelEulerSolver solver;
    if (parallel_euler_init(&solver, 1, 4, PARALLEL_OPENMP, NULL) != 0) {
        printf("  FAIL: Initialization failed\n");
        return 1;
    }
    
    double t0 = 0.0;
    double t_end = 2.0;
    double y0[1] = {1.0};
    double h = 0.01;
    
    size_t max_steps = (size_t)((t_end - t0) / h) + 1;
    double* t_out = (double*)malloc(max_steps * sizeof(double));
    double* y_out = (double*)malloc(max_steps * sizeof(double));
    
    clock_t start = clock();
    size_t num_steps = parallel_euler_solve(&solver, test_exponential, t0, t_end, y0,
                                           h, NULL, t_out, y_out);
    clock_t end = clock();
    double elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    double y_final = y_out[num_steps - 1];
    double y_exact = exp(-t_end);
    double error = fabs(y_final - y_exact);
    double accuracy = (1.0 - error / fabs(y_exact)) * 100.0;
    
    printf("  Steps: %zu\n", num_steps);
    printf("  Time: %.6f seconds\n", elapsed);
    printf("  Final value: %.10f\n", y_final);
    printf("  Exact value: %.10f\n", y_exact);
    printf("  Error: %.6e\n", error);
    printf("  Accuracy: %.6f%%\n", accuracy);
    printf("  %s\n\n", (error < 1e-2) ? "PASS" : "FAIL");
    
    parallel_euler_free(&solver);
    free(t_out);
    free(y_out);
    return (error < 1e-2) ? 0 : 1;
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  Parallel, Distributed, Concurrent, Hierarchical, Stacked    ║\n");
    printf("║                    Methods Test Suite                          ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    int failures = 0;
    failures += test_parallel_rk3();
    failures += test_stacked_rk3();
    failures += test_parallel_euler();
    
    printf("=== Test Summary ===\n");
    printf("Failures: %d\n", failures);
    
    if (failures == 0) {
        printf("✅ All parallel method tests PASSED\n");
    } else {
        printf("❌ Some tests FAILED\n");
    }
    
    return (failures == 0) ? 0 : 1;
}
