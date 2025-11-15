/*
 * Comprehensive Harmonic Oscillator Test Suite
 * Copyright (C) 2025, Shyamal Suhana Chandra
 * 
 * Tests all methods (RK3, DDRK3, AM, DDAM) on harmonic oscillator ODE
 * ODE: d²x/dt² = -x, or dx/dt = v, dv/dt = -x
 * Exact solution: x(t) = cos(t), v(t) = -sin(t)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "../include/rk3.h"
#include "../include/adams.h"
#include "../include/hierarchical_rk.h"
#include "../include/comparison.h"

// Harmonic oscillator ODE: dx/dt = v, dv/dt = -x
void oscillator_ode(double t, const double* y, double* dydt, void* params) {
    (void)t;
    (void)params;
    dydt[0] = y[1];   // dx/dt = v
    dydt[1] = -y[0];  // dv/dt = -x
}

// Exact solution: x(t) = cos(t), v(t) = -sin(t)
void exact_oscillator(double t, double x0, double v0, double* x, double* v) {
    *x = x0 * cos(t) - v0 * sin(t);
    *v = -x0 * sin(t) - v0 * cos(t);
}

int test_rk3_oscillator() {
    printf("=== RK3 Method: Harmonic Oscillator ===\n");
    
    double t0 = 0.0;
    double t_end = 2.0 * M_PI;
    double y0[2] = {1.0, 0.0};  // x(0) = 1.0, v(0) = 0.0
    double h = 0.01;
    
    size_t max_steps = (size_t)((t_end - t0) / h) + 1;
    double* t_out = (double*)malloc(max_steps * sizeof(double));
    double* y_out = (double*)malloc(max_steps * 2 * sizeof(double));
    
    clock_t start = clock();
    size_t num_steps = rk3_solve(oscillator_ode, t0, t_end, y0, 2, h, NULL, t_out, y_out);
    clock_t end = clock();
    double elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    double x_final = y_out[(num_steps - 1) * 2];
    double v_final = y_out[(num_steps - 1) * 2 + 1];
    double x_exact, v_exact;
    exact_oscillator(t_end, 1.0, 0.0, &x_exact, &v_exact);
    
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
    printf("  %s\n\n", (error_total < 0.01) ? "PASS" : "FAIL");
    
    free(t_out);
    free(y_out);
    return (error_total < 0.01) ? 0 : 1;
}

int test_ddrk3_oscillator() {
    printf("=== DDRK3 Method: Harmonic Oscillator ===\n");
    
    HierarchicalRKSolver solver;
    if (hierarchical_rk_init(&solver, 3, 2, 16) != 0) {
        printf("  FAIL: Initialization failed\n");
        return 1;
    }
    
    double t0 = 0.0;
    double t_end = 2.0 * M_PI;
    double y0[2] = {1.0, 0.0};
    double h = 0.01;
    
    size_t max_steps = (size_t)((t_end - t0) / h) + 1;
    double* t_out = (double*)malloc(max_steps * sizeof(double));
    double* y_out = (double*)malloc(max_steps * 2 * sizeof(double));
    
    clock_t start = clock();
    size_t num_steps = hierarchical_rk_solve(&solver, oscillator_ode, t0, t_end, y0, h, NULL, t_out, y_out);
    clock_t end = clock();
    double elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    double x_final = y_out[(num_steps - 1) * 2];
    double v_final = y_out[(num_steps - 1) * 2 + 1];
    double x_exact, v_exact;
    exact_oscillator(t_end, 1.0, 0.0, &x_exact, &v_exact);
    
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
    printf("  %s\n\n", (error_total < 0.01) ? "PASS" : "FAIL");
    
    hierarchical_rk_free(&solver);
    free(t_out);
    free(y_out);
    return (error_total < 0.01) ? 0 : 1;
}

int test_am_oscillator() {
    printf("=== Adams Methods (AM): Harmonic Oscillator ===\n");
    
    double t0 = 0.0;
    double t_end = 2.0 * M_PI;
    double y0[2] = {1.0, 0.0};
    double h = 0.01;
    
    size_t max_steps = (size_t)((t_end - t0) / h) + 1;
    double* t_out = (double*)malloc(max_steps * sizeof(double));
    double* y_out = (double*)malloc(max_steps * 2 * sizeof(double));
    
    double y_current[2] = {y0[0], y0[1]};
    double t_current = t0;
    size_t step = 0;
    
    t_out[step] = t_current;
    y_out[step * 2] = y_current[0];
    y_out[step * 2 + 1] = y_current[1];
    step++;
    
    double y_prev[2], y_prev2[2];
    double f_prev[2], f_prev2[2];
    
    clock_t start = clock();
    while (t_current < t_end && step < max_steps) {
        double h_actual = (t_current + h > t_end) ? (t_end - t_current) : h;
        
        if (step == 1) {
            rk3_step(oscillator_ode, t_current, y_current, 2, h_actual, NULL);
            t_current += h_actual;
        } else if (step == 2) {
            y_prev[0] = y_current[0];
            y_prev[1] = y_current[1];
            double f_current[2];
            oscillator_ode(t_current, y_current, f_current, NULL);
            f_prev[0] = f_current[0];
            f_prev[1] = f_current[1];
            
            rk3_step(oscillator_ode, t_current, y_current, 2, h_actual, NULL);
            t_current += h_actual;
        } else {
            double f_current[2];
            oscillator_ode(t_current, y_current, f_current, NULL);
            
            // Adams-Bashforth 3rd order
            double y_pred[2];
            y_pred[0] = y_current[0] + h_actual * (23.0 * f_current[0] - 16.0 * f_prev[0] + 5.0 * f_prev2[0]) / 12.0;
            y_pred[1] = y_current[1] + h_actual * (23.0 * f_current[1] - 16.0 * f_prev[1] + 5.0 * f_prev2[1]) / 12.0;
            
            // Adams-Moulton corrector
            double f_pred[2];
            oscillator_ode(t_current + h_actual, y_pred, f_pred, NULL);
            y_current[0] = y_current[0] + h_actual * (5.0 * f_pred[0] + 8.0 * f_current[0] - f_prev[0]) / 12.0;
            y_current[1] = y_current[1] + h_actual * (5.0 * f_pred[1] + 8.0 * f_current[1] - f_prev[1]) / 12.0;
            
            y_prev2[0] = y_prev[0];
            y_prev2[1] = y_prev[1];
            y_prev[0] = y_current[0];
            y_prev[1] = y_current[1];
            f_prev2[0] = f_prev[0];
            f_prev2[1] = f_prev[1];
            f_prev[0] = f_current[0];
            f_prev[1] = f_current[1];
            
            t_current += h_actual;
        }
        
        t_out[step] = t_current;
        y_out[step * 2] = y_current[0];
        y_out[step * 2 + 1] = y_current[1];
        step++;
    }
    clock_t end = clock();
    double elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    double x_final = y_out[(step - 1) * 2];
    double v_final = y_out[(step - 1) * 2 + 1];
    double x_exact, v_exact;
    exact_oscillator(t_end, 1.0, 0.0, &x_exact, &v_exact);
    
    double error_x = fabs(x_final - x_exact);
    double error_v = fabs(v_final - v_exact);
    double error_total = sqrt(error_x * error_x + error_v * error_v);
    double accuracy = (1.0 - error_total / sqrt(x_exact * x_exact + v_exact * v_exact)) * 100.0;
    
    printf("  Steps: %zu\n", step);
    printf("  Time: %.6f seconds\n", elapsed);
    printf("  Final position: %.10f (exact: %.10f, error: %.6e)\n", x_final, x_exact, error_x);
    printf("  Final velocity: %.10f (exact: %.10f, error: %.6e)\n", v_final, v_exact, error_v);
    printf("  Total error: %.6e\n", error_total);
    printf("  Accuracy: %.6f%%\n", accuracy);
    printf("  %s\n\n", (error_total < 0.01) ? "PASS" : "FAIL");
    
    free(t_out);
    free(y_out);
    return (error_total < 0.01) ? 0 : 1;
}

int test_ddam_oscillator() {
    printf("=== DDAM Method: Harmonic Oscillator ===\n");
    
    double t0 = 0.0;
    double t_end = 2.0 * M_PI;
    double y0[2] = {1.0, 0.0};
    double h = 0.01;
    double x_exact, v_exact;
    exact_oscillator(t_end, 1.0, 0.0, &x_exact, &v_exact);
    double exact[2] = {x_exact, v_exact};
    
    ComparisonResults results;
    
    clock_t start = clock();
    if (compare_methods(oscillator_ode, t0, t_end, y0, 2, h, NULL, exact, &results) != 0) {
        printf("  FAIL: Comparison failed\n");
        return 1;
    }
    clock_t end = clock();
    double elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("  Time: %.6f seconds\n", elapsed);
    printf("  DDAM Error: %.6e\n", results.ddam_error);
    printf("  DDAM Accuracy: %.6f%%\n", results.ddam_accuracy * 100.0);
    printf("  %s\n\n", (results.ddam_error < 0.01) ? "PASS" : "FAIL");
    
    return (results.ddam_error < 0.01) ? 0 : 1;
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║     Harmonic Oscillator Test Suite - All Methods             ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Test Case: d²x/dt² = -x, x(0) = 1.0, v(0) = 0.0, t ∈ [0, 2π]\n");
    printf("Exact Solution: x(t) = cos(t), v(t) = -sin(t)\n");
    double x_exact, v_exact;
    exact_oscillator(2.0 * M_PI, 1.0, 0.0, &x_exact, &v_exact);
    printf("Expected Final Values: x(2π) = %.10f, v(2π) = %.10f\n\n", x_exact, v_exact);
    
    int failures = 0;
    failures += test_rk3_oscillator();
    failures += test_ddrk3_oscillator();
    failures += test_am_oscillator();
    failures += test_ddam_oscillator();
    
    printf("=== Test Summary ===\n");
    printf("Failures: %d\n", failures);
    
    if (failures == 0) {
        printf("✅ All Harmonic Oscillator tests PASSED\n");
    } else {
        printf("❌ Some tests FAILED\n");
    }
    
    return (failures == 0) ? 0 : 1;
}
