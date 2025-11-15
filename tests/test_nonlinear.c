/*
 * Test suite for Nonlinear Programming Solvers
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "../include/nonlinear_solver.h"
#include "../include/distributed_solvers.h"

// Test ODE: dy/dt = -y
void test_exponential(double t, const double* y, double* dydt, void* params) {
    (void)t;
    (void)params;
    dydt[0] = -y[0];
}

// Objective function for NLP
double objective_func(const double* x, size_t n, void* params) {
    (void)params;
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += x[i] * x[i];
    }
    return sum;
}

int test_nonlinear_ode() {
    printf("=== Nonlinear ODE Solver Test ===\n");
    
    NonlinearODESolver solver;
    if (nonlinear_ode_init(&solver, 1, NLP_GRADIENT_DESCENT, objective_func, NULL, NULL) != 0) {
        printf("  FAIL: Initialization failed\n");
        return 1;
    }
    
    double t0 = 0.0;
    double t_end = 2.0;
    double y0[1] = {1.0};
    double y_out[1];
    
    clock_t start = clock();
    int result = nonlinear_ode_solve(&solver, test_exponential, t0, t_end, y0, y_out);
    clock_t end = clock();
    double elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    if (result != 0) {
        printf("  FAIL: Solve failed\n");
        nonlinear_ode_free(&solver);
        return 1;
    }
    
    double y_exact = exp(-t_end);
    double error = fabs(y_out[0] - y_exact);
    double accuracy = (1.0 - error / fabs(y_exact)) * 100.0;
    
    printf("  Time: %.6f seconds\n", elapsed);
    printf("  Final value: %.10f\n", y_out[0]);
    printf("  Exact value: %.10f\n", y_exact);
    printf("  Error: %.6e\n", error);
    printf("  Accuracy: %.6f%%\n", accuracy);
    printf("  %s\n\n", (error < 1e-2) ? "PASS" : "FAIL");
    
    nonlinear_ode_free(&solver);
    return (error < 1e-2) ? 0 : 1;
}

int test_distributed_datadriven() {
    printf("=== Distributed Data-Driven Solver Test ===\n");
    
    DistributedDataDrivenSolver solver;
    if (distributed_datadriven_init(&solver, 1, 4, 3) != 0) {
        printf("  FAIL: Initialization failed\n");
        return 1;
    }
    
    double t0 = 0.0;
    double t_end = 2.0;
    double y0[1] = {1.0};
    double h = 0.01;
    double y_out[1];
    
    clock_t start = clock();
    int result = distributed_datadriven_solve(&solver, test_exponential, t0, t_end, y0, h, NULL, y_out);
    clock_t end = clock();
    double elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    if (result != 0) {
        printf("  FAIL: Solve failed\n");
        distributed_datadriven_free(&solver);
        return 1;
    }
    
    double y_exact = exp(-t_end);
    double error = fabs(y_out[0] - y_exact);
    double accuracy = (1.0 - error / fabs(y_exact)) * 100.0;
    
    printf("  Time: %.6f seconds\n", elapsed);
    printf("  Final value: %.10f\n", y_out[0]);
    printf("  Exact value: %.10f\n", y_exact);
    printf("  Error: %.6e\n", error);
    printf("  Accuracy: %.6f%%\n", accuracy);
    printf("  %s\n\n", (error < 1e-3) ? "PASS" : "FAIL");
    
    distributed_datadriven_free(&solver);
    return (error < 1e-3) ? 0 : 1;
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  Nonlinear Programming & Additional Solvers Test Suite      ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    int failures = 0;
    failures += test_nonlinear_ode();
    failures += test_distributed_datadriven();
    
    printf("=== Test Summary ===\n");
    printf("Failures: %d\n", failures);
    
    if (failures == 0) {
        printf("✅ All nonlinear and additional solver tests PASSED\n");
    } else {
        printf("❌ Some tests FAILED\n");
    }
    
    return (failures == 0) ? 0 : 1;
}
