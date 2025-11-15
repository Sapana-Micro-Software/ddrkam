/*
 * Test suite for Euler's Method and Data-Driven Euler's Method
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "../include/euler.h"
#include "../include/hierarchical_euler.h"

// Test ODE: dy/dt = -y, solution: y(t) = y0 * exp(-t)
void test_exponential(double t, const double* y, double* dydt, void* params) {
    (void)t;
    (void)params;
    dydt[0] = -y[0];
}

// Test ODE: Simple harmonic oscillator
void test_oscillator(double t, const double* y, double* dydt, void* params) {
    (void)t;
    (void)params;
    dydt[0] = y[1];
    dydt[1] = -y[0];
}

int test_euler_exponential() {
    printf("=== Euler's Method: Exponential Decay ===\n");
    
    double t0 = 0.0;
    double t_end = 2.0;
    double y0[1] = {1.0};
    double h = 0.01;
    
    size_t max_steps = (size_t)((t_end - t0) / h) + 1;
    double* t_out = (double*)malloc(max_steps * sizeof(double));
    double* y_out = (double*)malloc(max_steps * sizeof(double));
    
    clock_t start = clock();
    size_t num_steps = euler_solve(test_exponential, t0, t_end, y0, 1, h, NULL, t_out, y_out);
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
    
    free(t_out);
    free(y_out);
    return (error < 1e-2) ? 0 : 1;
}

int test_ddeuler_exponential() {
    printf("=== DDEuler Method: Exponential Decay ===\n");
    
    HierarchicalEulerSolver solver;
    if (hierarchical_euler_init(&solver, 3, 1, 16) != 0) {
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
    size_t num_steps = hierarchical_euler_solve(&solver, test_exponential, t0, t_end, y0, h, NULL, t_out, y_out);
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
    
    hierarchical_euler_free(&solver);
    free(t_out);
    free(y_out);
    return (error < 1e-2) ? 0 : 1;
}

int test_euler_oscillator() {
    printf("=== Euler's Method: Harmonic Oscillator ===\n");
    
    double t0 = 0.0;
    double t_end = 2.0 * M_PI;
    double y0[2] = {1.0, 0.0};
    double h = 0.01;
    
    size_t max_steps = (size_t)((t_end - t0) / h) + 1;
    double* t_out = (double*)malloc(max_steps * sizeof(double));
    double* y_out = (double*)malloc(max_steps * 2 * sizeof(double));
    
    clock_t start = clock();
    size_t num_steps = euler_solve(test_oscillator, t0, t_end, y0, 2, h, NULL, t_out, y_out);
    clock_t end = clock();
    double elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    double x_final = y_out[(num_steps - 1) * 2];
    double v_final = y_out[(num_steps - 1) * 2 + 1];
    double x_exact = cos(t_end);
    double v_exact = -sin(t_end);
    
    double error_x = fabs(x_final - x_exact);
    double error_v = fabs(v_final - v_exact);
    double error_total = sqrt(error_x * error_x + error_v * error_v);
    double accuracy = (1.0 - error_total / sqrt(x_exact * x_exact + v_exact * v_exact)) * 100.0;
    
    printf("  Steps: %zu\n", num_steps);
    printf("  Time: %.6f seconds\n", elapsed);
    printf("  Final position: %.10f (exact: %.10f, error: %.6e)\n", x_final, x_exact, error_x);
    printf("  Final velocity: %.10f (exact: %.10f, error: %.6e)\n", v_final, v_exact, error_v);
    printf("  Total error: %.6e\n", error_total);
    printf("  Accuracy: %.6f%%\n", accuracy);
    printf("  %s\n\n", (error_total < 0.1) ? "PASS" : "FAIL");
    
    free(t_out);
    free(y_out);
    return (error_total < 0.1) ? 0 : 1;
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║     Euler's Method & DDEuler Test Suite                      ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    int failures = 0;
    failures += test_euler_exponential();
    failures += test_ddeuler_exponential();
    failures += test_euler_oscillator();
    
    printf("=== Test Summary ===\n");
    printf("Failures: %d\n", failures);
    
    if (failures == 0) {
        printf("✅ All Euler method tests PASSED\n");
    } else {
        printf("❌ Some tests FAILED\n");
    }
    
    return (failures == 0) ? 0 : 1;
}
