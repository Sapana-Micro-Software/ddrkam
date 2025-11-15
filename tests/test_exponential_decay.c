/*
 * Comprehensive Exponential Decay Test Suite
 * Copyright (C) 2025, Shyamal Suhana Chandra
 * 
 * Tests all methods (RK3, DDRK3, AM, DDAM) on exponential decay ODE
 * ODE: dy/dt = -y
 * Exact solution: y(t) = y0 * exp(-t)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "../include/rk3.h"
#include "../include/adams.h"
#include "../include/hierarchical_rk.h"
#include "../include/comparison.h"

// Exponential decay ODE: dy/dt = -y
void exponential_ode(double t, const double* y, double* dydt, void* params) {
    (void)t;
    (void)params;
    dydt[0] = -y[0];
}

// Exact solution: y(t) = y0 * exp(-t)
double exact_exponential(double t, double y0) {
    return y0 * exp(-t);
}

int test_rk3_exponential() {
    printf("=== RK3 Method: Exponential Decay ===\n");
    
    double t0 = 0.0;
    double t_end = 2.0;
    double y0[1] = {1.0};
    double h = 0.01;
    
    size_t max_steps = (size_t)((t_end - t0) / h) + 1;
    double* t_out = (double*)malloc(max_steps * sizeof(double));
    double* y_out = (double*)malloc(max_steps * sizeof(double));
    
    clock_t start = clock();
    size_t num_steps = rk3_solve(exponential_ode, t0, t_end, y0, 1, h, NULL, t_out, y_out);
    clock_t end = clock();
    double elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    double y_final = y_out[num_steps - 1];
    double y_exact = exact_exponential(t_end, 1.0);
    double error = fabs(y_final - y_exact);
    double accuracy = (1.0 - error / fabs(y_exact)) * 100.0;
    
    printf("  Steps: %zu\n", num_steps);
    printf("  Time: %.6f seconds\n", elapsed);
    printf("  Final value: %.10f\n", y_final);
    printf("  Exact value: %.10f\n", y_exact);
    printf("  Error: %.6e\n", error);
    printf("  Accuracy: %.6f%%\n", accuracy);
    printf("  %s\n\n", (error < 1e-5) ? "PASS" : "FAIL");
    
    free(t_out);
    free(y_out);
    return (error < 1e-5) ? 0 : 1;
}

int test_ddrk3_exponential() {
    printf("=== DDRK3 Method: Exponential Decay ===\n");
    
    HierarchicalRKSolver solver;
    if (hierarchical_rk_init(&solver, 3, 1, 16) != 0) {
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
    size_t num_steps = hierarchical_rk_solve(&solver, exponential_ode, t0, t_end, y0, h, NULL, t_out, y_out);
    clock_t end = clock();
    double elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    double y_final = y_out[num_steps - 1];
    double y_exact = exact_exponential(t_end, 1.0);
    double error = fabs(y_final - y_exact);
    double accuracy = (1.0 - error / fabs(y_exact)) * 100.0;
    
    printf("  Steps: %zu\n", num_steps);
    printf("  Time: %.6f seconds\n", elapsed);
    printf("  Final value: %.10f\n", y_final);
    printf("  Exact value: %.10f\n", y_exact);
    printf("  Error: %.6e\n", error);
    printf("  Accuracy: %.6f%%\n", accuracy);
    printf("  %s\n\n", (error < 1e-5) ? "PASS" : "FAIL");
    
    hierarchical_rk_free(&solver);
    free(t_out);
    free(y_out);
    return (error < 1e-5) ? 0 : 1;
}

int test_am_exponential() {
    printf("=== Adams Methods (AM): Exponential Decay ===\n");
    
    double t0 = 0.0;
    double t_end = 2.0;
    double y0[1] = {1.0};
    double h = 0.01;
    
    size_t max_steps = (size_t)((t_end - t0) / h) + 1;
    double* t_out = (double*)malloc(max_steps * sizeof(double));
    double* y_out = (double*)malloc(max_steps * sizeof(double));
    
    // Use Adams-Bashforth 3rd order with Adams-Moulton corrector
    double y_current[1] = {y0[0]};
    double t_current = t0;
    size_t step = 0;
    
    // Store initial condition
    t_out[step] = t_current;
    y_out[step] = y_current[0];
    step++;
    
    // Need history for multi-step methods - use RK3 for first steps
    double y_prev[1], y_prev2[1];
    double f_prev[1], f_prev2[1];
    
    clock_t start = clock();
    while (t_current < t_end && step < max_steps) {
        double h_actual = (t_current + h > t_end) ? (t_end - t_current) : h;
        
        if (step == 1) {
            // First step: use RK3
            rk3_step(exponential_ode, t_current, y_current, 1, h_actual, NULL);
            t_current += h_actual;
        } else if (step == 2) {
            // Second step: use RK3, save history
            y_prev[0] = y_current[0];
            double f_current[1];
            exponential_ode(t_current, y_current, f_current, NULL);
            f_prev[0] = f_current[0];
            
            rk3_step(exponential_ode, t_current, y_current, 1, h_actual, NULL);
            t_current += h_actual;
        } else {
            // Multi-step: use Adams-Bashforth predictor
            double f_current[1];
            exponential_ode(t_current, y_current, f_current, NULL);
            
            // Adams-Bashforth 3rd order: y_{n+1} = y_n + h*(23f_n - 16f_{n-1} + 5f_{n-2})/12
            double y_pred = y_current[0] + h_actual * (23.0 * f_current[0] - 16.0 * f_prev[0] + 5.0 * f_prev2[0]) / 12.0;
            
            // Adams-Moulton corrector: y_{n+1} = y_n + h*(5f_{n+1} + 8f_n - f_{n-1})/12
            double f_pred[1];
            exponential_ode(t_current + h_actual, &y_pred, f_pred, NULL);
            y_current[0] = y_current[0] + h_actual * (5.0 * f_pred[0] + 8.0 * f_current[0] - f_prev[0]) / 12.0;
            
            // Update history
            y_prev2[0] = y_prev[0];
            y_prev[0] = y_current[0];
            f_prev2[0] = f_prev[0];
            f_prev[0] = f_current[0];
            
            t_current += h_actual;
        }
        
        t_out[step] = t_current;
        y_out[step] = y_current[0];
        step++;
    }
    clock_t end = clock();
    double elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    double y_final = y_out[step - 1];
    double y_exact = exact_exponential(t_end, 1.0);
    double error = fabs(y_final - y_exact);
    double accuracy = (1.0 - error / fabs(y_exact)) * 100.0;
    
    printf("  Steps: %zu\n", step);
    printf("  Time: %.6f seconds\n", elapsed);
    printf("  Final value: %.10f\n", y_final);
    printf("  Exact value: %.10f\n", y_exact);
    printf("  Error: %.6e\n", error);
    printf("  Accuracy: %.6f%%\n", accuracy);
    printf("  %s\n\n", (error < 1e-5) ? "PASS" : "FAIL");
    
    free(t_out);
    free(y_out);
    return (error < 1e-5) ? 0 : 1;
}

int test_ddam_exponential() {
    printf("=== DDAM Method: Exponential Decay ===\n");
    
    // Use comparison framework which includes DDAM
    double t0 = 0.0;
    double t_end = 2.0;
    double y0[1] = {1.0};
    double h = 0.01;
    double exact[1] = {exact_exponential(t_end, 1.0)};
    
    ComparisonResults results;
    
    clock_t start = clock();
    if (compare_methods(exponential_ode, t0, t_end, y0, 1, h, NULL, exact, &results) != 0) {
        printf("  FAIL: Comparison failed\n");
        return 1;
    }
    clock_t end = clock();
    double elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("  Time: %.6f seconds\n", elapsed);
    printf("  DDAM Error: %.6e\n", results.ddam_error);
    printf("  DDAM Accuracy: %.6f%%\n", results.ddam_accuracy * 100.0);
    printf("  %s\n\n", (results.ddam_error < 1e-5) ? "PASS" : "FAIL");
    
    return (results.ddam_error < 1e-5) ? 0 : 1;
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║     Exponential Decay Test Suite - All Methods              ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Test Case: dy/dt = -y, y(0) = 1.0, t ∈ [0, 2.0]\n");
    printf("Exact Solution: y(t) = exp(-t)\n");
    printf("Expected Final Value: y(2.0) = %.10f\n\n", exp(-2.0));
    
    int failures = 0;
    failures += test_rk3_exponential();
    failures += test_ddrk3_exponential();
    failures += test_am_exponential();
    failures += test_ddam_exponential();
    
    printf("=== Test Summary ===\n");
    printf("Failures: %d\n", failures);
    
    if (failures == 0) {
        printf("✅ All Exponential Decay tests PASSED\n");
    } else {
        printf("❌ Some tests FAILED\n");
    }
    
    return (failures == 0) ? 0 : 1;
}
